"""
PostgreSQL + pgvector — connection management and schema.

Why pgvector over FAISS
-----------------------
FAISS limitations that pgvector solves:
  1. Metadata filtering in FAISS requires building a separate in-memory subset
     index for every filtered query — O(|candidates|) index rebuild per request.
     pgvector handles this with a native SQL WHERE clause evaluated before ANN.

  2. FAISS stores vectors in .faiss files and metadata in .pkl files.
     pgvector stores everything in one ACID-compliant PostgreSQL table —
     no desync between vector index and metadata.

  3. FAISS IVFFlat requires retraining when nlist changes or corpus shifts.
     pgvector HNSW: just CREATE INDEX CONCURRENTLY — zero downtime, no retrain.

  4. n_probe in FAISS is set on a shared index object (not thread-safe without locks).
     pgvector: SET ivfflat.probes = N is a per-session setting — safe under concurrency.

  5. FAISS has no transactions — concurrent ingestion can corrupt the index.
     pgvector inherits PostgreSQL MVCC — safe parallel ingestion.

Index types in pgvector
-----------------------
IVFFlat (current):
  • SQL:  CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists=N)
  • lists ≈ sqrt(total_rows).  Tune with SET ivfflat.probes = K at query time.
  • n_probe equivalent: ivfflat.probes  (exact same semantics as FAISS nprobe)
  • Requires table to be populated before index creation (training step).

HNSW (migrate at ~100K rows):
  • SQL:  CREATE INDEX CONCURRENTLY ... USING hnsw (embedding vector_cosine_ops)
           WITH (m=32, ef_construction=200)
  • No training. O(log N) search. Zero-downtime migration via CONCURRENTLY.
  • Tune with SET hnsw.ef_search = K (default 40; higher = better recall, slower).
  • Migration steps:
      1. CREATE INDEX CONCURRENTLY hnsw_idx ON recipe_chunks
             USING hnsw (embedding vector_cosine_ops) WITH (m=32, ef_construction=200);
      2. DROP INDEX ivfflat_idx;   -- old index, no downtime
      3. Run recall benchmark to confirm >= baseline.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

# ── connection config (override via env vars) ─────────────────────────────────
import getpass as _getpass

DB_CONFIG: dict = {
    "host":     os.environ.get("PGHOST",     "localhost"),
    "port":     int(os.environ.get("PGPORT", "5432")),
    "dbname":   os.environ.get("PGDATABASE", "recipes"),
    # Default to the OS user (Homebrew Postgres uses peer/trust for local connections).
    # Override via PGUSER / PGPASSWORD env vars for Docker or remote servers.
    "user":     os.environ.get("PGUSER",     _getpass.getuser()),
    "password": os.environ.get("PGPASSWORD", None),
}

# Number of IVFFlat partition lists.
# Rule of thumb: sqrt(expected_row_count).
# For 150 rows this should be ~12, but we use 64 to match the FAISS config
# and demonstrate behaviour with nlist > sqrt(N) (same warning FAISS issues).
IVFFLAT_LISTS = 64

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"


# ── connection ────────────────────────────────────────────────────────────────
def get_connection(retries: int = 5, delay: float = 2.0) -> psycopg2.extensions.connection:
    """
    Open a PostgreSQL connection with pgvector type adapters registered.
    Retries up to `retries` times with `delay` seconds between attempts,
    so the caller does not need to wait for Docker to be ready.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            # The vector type must exist before register_vector() can look it up —
            # on a brand-new database this extension hasn't been created yet.
            with conn.cursor() as cur:
                cur.execute(CREATE_EXTENSION)
            conn.commit()
            register_vector(conn)
            return conn
        except psycopg2.OperationalError as e:
            last_error = e
            if attempt < retries:
                print(f"[DB] Connection attempt {attempt}/{retries} failed — retrying in {delay}s…")
                time.sleep(delay)
    raise RuntimeError(
        f"Cannot connect to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}.\n"
        f"Start the database with:  docker compose up -d\n"
        f"Original error: {last_error}"
    )


# ── schema ────────────────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS recipe_chunks (
    id              SERIAL PRIMARY KEY,
    recipe_id       TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    section         TEXT    NOT NULL,   -- 'ingredients' | 'method' | 'tips'
    text            TEXT    NOT NULL,
    cuisine         TEXT,
    tags            TEXT[], 
    dietary         TEXT[], 
    prep_time_mins  INTEGER DEFAULT 0,
    cook_time_mins  INTEGER DEFAULT 0,
    total_time_mins INTEGER DEFAULT 0,
    servings        INTEGER DEFAULT 0,
    calories_kcal   INTEGER DEFAULT 0,
    chunk_index     INTEGER DEFAULT 0,
    char_count      INTEGER DEFAULT 0,
    embedding       vector(384),
    UNIQUE (recipe_id, section, chunk_index)
);
"""

CREATE_IVFFLAT_INDEX = f"""
CREATE INDEX IF NOT EXISTS recipe_chunks_embedding_ivfflat_idx
ON recipe_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = {IVFFLAT_LISTS});
"""

CREATE_TAG_INDEX = """
CREATE INDEX IF NOT EXISTS recipe_chunks_tags_gin_idx
ON recipe_chunks USING GIN (tags);
"""

CREATE_TIME_INDEX = """
CREATE INDEX IF NOT EXISTS recipe_chunks_total_time_idx
ON recipe_chunks (total_time_mins);
"""


def setup_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Create the pgvector extension, recipe_chunks table, and all indexes.
    Safe to call on every startup — all DDL uses IF NOT EXISTS.
    """
    with conn.cursor() as cur:
        cur.execute(CREATE_EXTENSION)
        cur.execute(CREATE_TABLE)
        # GIN index for fast tag array containment queries
        cur.execute(CREATE_TAG_INDEX)
        cur.execute(CREATE_TIME_INDEX)
        # IVFFlat ANN index — requires at least one row to exist when first created.
        # We create it here but only if the table has rows; otherwise it's created
        # after ingestion in ingest_to_pgvector().
        cur.execute("SELECT COUNT(*) FROM recipe_chunks")
        (row_count,) = cur.fetchone()
        if row_count > 0:
            cur.execute(CREATE_IVFFLAT_INDEX)
    conn.commit()
    print(f"[DB] Schema ready — recipe_chunks table exists")


def row_count(conn: psycopg2.extensions.connection) -> int:
    """Return current number of rows in recipe_chunks."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recipe_chunks")
        (n,) = cur.fetchone()
    return n


def truncate_chunks(conn: psycopg2.extensions.connection) -> None:
    """Remove all rows and reset sequences (for force_reingest)."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE recipe_chunks RESTART IDENTITY CASCADE")
    conn.commit()
    print("[DB] recipe_chunks truncated")
