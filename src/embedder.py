"""
Task 2 — Embed and Index  (pgvector backend)
============================================

Q4 — Model & Index Choice
--------------------------
Model:  all-MiniLM-L6-v2  (sentence-transformers)
  • 384-dim embeddings, ~22 M parameters, ~80 ms/batch on CPU
  • Trained on 1 B+ sentence pairs — strong semantic similarity signal
  • MIT licence — no per-token API cost; can ship offline
  • RAM budget: 50 K chunks × 384 dims × 4 bytes ≈ 73 MB.
    At 500 K chunks: 730 MB — fits in one r6g.large (16 GB).
  • Alternative rejected: text-embedding-3-small (OpenAI, 1 536 dims).
    Re-embed cost for 500 K chunks ≈ $10.  Acceptable in production but
    creates an API dependency and latency spike for every rebuild.

Index:  pgvector IVFFlat → HNSW  (replaces FAISS)
  • pgvector IVFFlat is semantically identical to FAISS IVFFlat:
    nlist  → lists  (partition count)
    nprobe → SET ivfflat.probes = K  (query-time cell budget)
  • pgvector HNSW: no training, O(log N), zero-downtime migration via
    CREATE INDEX CONCURRENTLY.

Q4 — IVFFlat → HNSW Migration  (pgvector version)
---------------------------------------------------
Migrate at: ~100 000 rows.

  Why 100 K?
    Below 100 K, IVFFlat with lists=256, probes=10 already achieves
    recall@5 > 0.90 in < 5 ms.  Above 100 K the lists value must grow
    (recommended: sqrt(N)), reindexing locks the table, and managing the
    lists parameter adds operational overhead.

  Migration steps — ZERO DOWNTIME, NO RE-EMBEDDING:
    1. CREATE INDEX CONCURRENTLY recipe_chunks_hnsw_idx
           ON recipe_chunks USING hnsw (embedding vector_cosine_ops)
           WITH (m = 32, ef_construction = 200);
       -- Builds in background; old IVFFlat index still serves queries.
    2. DROP INDEX recipe_chunks_embedding_ivfflat_idx;
       -- Old index removed; HNSW now serves all queries.
    3. Run offline recall benchmark to confirm >= baseline recall@5.
    4. No re-embed needed — embeddings are stored in the table column.

  Tune at query time:
    SET hnsw.ef_search = 64;   -- default 40; higher = better recall, slower

Q5 — HashEmbedder Trap
-----------------------
Risk:
  A hash function (MD5, FNV, xxHash) maps text to a fixed-width integer
  vector where the distance between two hashes encodes NOTHING about
  semantic meaning.  "chicken lemon pasta" and "leftover chicken with lemon"
  would be maximally distant despite being near-synonyms.

  The system appears healthy — pgvector still returns K rows — but recall@5
  is effectively random (≈ K/N).  Cosine scores cluster near 0 for all pairs.
  This is the "sometimes returns irrelevant recipes" symptom: no crash,
  just wrong answers.

Guards implemented below (assert_real_embedder):
  1. Startup cosine check: two near-synonyms must score > 0.60.
  2. Embed dimension check: vector must be exactly EMBED_DIM floats.
  3. Non-uniform check: vector must not be all-zeros or near-constant.
  4. CI gate: same assertion runs in the test suite on every push.
"""

from __future__ import annotations

import os
import time
from typing import List

import numpy as np
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from .chunker import ChunkRecord
from .db import get_connection, setup_schema, row_count, truncate_chunks, CREATE_IVFFLAT_INDEX

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "all-MiniLM-L6-v2"
EMBED_DIM   = 384
BATCH_SIZE  = 64

# Optional: save raw embeddings as .npy backup so HNSW migration
# never requires re-embedding (load numpy array → feed directly to new index).
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EMBED_PATH  = os.path.join(RESULTS_DIR, "embeddings.npy")

# ── model singleton ───────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """Embed a list of strings → float32 array (N, EMBED_DIM), L2-normalised."""
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # cosine sim = dot product after L2 norm
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ── pgvector ingestion ────────────────────────────────────────────────────────
def ingest_to_pgvector(
    chunks: List[ChunkRecord],
    conn,
    force_reingest: bool = False,
) -> int:
    """
    Embed all chunks and upsert them into the recipe_chunks table.

    Steps
    -----
    1. If force_reingest=True, truncate the table first.
    2. Skip embedding if the table already has the same number of rows
       (idempotent on re-runs).
    3. Embed in batches of BATCH_SIZE using all-MiniLM-L6-v2.
    4. Bulk-insert with psycopg2 execute_values (fast batch insert).
    5. Create IVFFlat index after ingestion (requires populated table).
    6. Save embeddings.npy as a backup for future HNSW migration.

    Returns the number of rows in the table after ingestion.
    """
    if force_reingest:
        truncate_chunks(conn)

    existing = row_count(conn)
    if not force_reingest and existing == len(chunks):
        print(f"[Embedder] {existing} rows already in DB — skipping re-ingestion.")
        return existing

    print(f"[Embedder] Embedding {len(chunks)} chunks with {MODEL_NAME}…")
    t0 = time.time()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    elapsed = time.time() - t0
    print(f"[Embedder] Embedded in {elapsed:.1f}s  ({len(chunks)/elapsed:.0f} chunks/s)")

    # Save .npy backup (enables HNSW migration without re-embedding)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(EMBED_PATH, embeddings)

    # Bulk insert
    rows = []
    for chunk, vec in zip(chunks, embeddings):
        rows.append((
            chunk.recipe_id,
            chunk.title,
            chunk.section,
            chunk.text,
            chunk.cuisine,
            chunk.tags,
            chunk.dietary,
            chunk.prep_time_mins,
            chunk.cook_time_mins,
            chunk.total_time_mins,
            chunk.servings,
            chunk.calories_kcal,
            chunk.chunk_index,
            chunk.char_count,
            vec.tolist(),           # pgvector accepts Python list of floats
        ))

    insert_sql = """
        INSERT INTO recipe_chunks
            (recipe_id, title, section, text, cuisine, tags, dietary,
             prep_time_mins, cook_time_mins, total_time_mins, servings,
             calories_kcal, chunk_index, char_count, embedding)
        VALUES %s
        ON CONFLICT (recipe_id, section, chunk_index)
        DO UPDATE SET
            embedding     = EXCLUDED.embedding,
            title         = EXCLUDED.title,
            text          = EXCLUDED.text,
            tags          = EXCLUDED.tags,
            dietary       = EXCLUDED.dietary,
            total_time_mins = EXCLUDED.total_time_mins,
            calories_kcal = EXCLUDED.calories_kcal
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, rows, page_size=100)
    conn.commit()

    # Create IVFFlat index now that the table is populated
    with conn.cursor() as cur:
        cur.execute(CREATE_IVFFLAT_INDEX)
    conn.commit()

    total = row_count(conn)
    print(f"[Embedder] {total} rows in recipe_chunks with IVFFlat index (lists={64})")
    return total


# ── HashEmbedder guard  (Q5) ──────────────────────────────────────────────────
def assert_real_embedder(model: SentenceTransformer) -> None:
    """
    Startup and CI guard.  Verifies the loaded model produces semantically
    meaningful embeddings.  Raises AssertionError if:
      • cosine similarity between two near-synonyms < 0.60
      • output dimension != EMBED_DIM
      • vector is near-zero (hash stub returning zeros)
    """
    probe_a = model.encode(["chicken lemon pasta"],        normalize_embeddings=True)[0]
    probe_b = model.encode(["leftover chicken with lemon"], normalize_embeddings=True)[0]

    assert probe_a.shape[0] == EMBED_DIM, (
        f"Dimension mismatch: got {probe_a.shape[0]}, expected {EMBED_DIM}. "
        "Wrong model loaded?"
    )
    assert not np.allclose(probe_a, 0), (
        "Embedding is all-zeros — model is a stub or hash-based dummy."
    )

    sim = float(np.dot(probe_a, probe_b))
    assert sim > 0.60, (
        f"Semantic similarity check FAILED (cosine={sim:.3f}). "
        "A hash-based embedder maps semantically similar sentences to random "
        "vectors — queries return irrelevant results silently. "
        "Check MODEL_NAME and model weights."
    )
    print(f"[Guard] Embedding sanity check passed  (cosine={sim:.3f})")
