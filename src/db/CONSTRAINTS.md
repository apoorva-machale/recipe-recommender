# src/db/CONSTRAINTS.md — Database Hard Constraints

This file documents every hard constraint on database operations.
**An AI agent or engineer must read this before writing any SQL or modifying `src/db.py`.**

> The database module lives at `src/db.py`.
> This `src/db/` directory exists solely for constraint documentation.

---

## Table: `recipe_chunks`

### Schema (authoritative copy)

```sql
CREATE TABLE IF NOT EXISTS recipe_chunks (
    id              SERIAL PRIMARY KEY,
    recipe_id       TEXT NOT NULL,
    title           TEXT NOT NULL,
    section         TEXT NOT NULL,       -- 'ingredients' | 'method' | 'tips'
    text            TEXT NOT NULL,
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
    UNIQUE (recipe_id, section, chunk_index)   -- ← upsert key
);
```

### Indexes

| Index name | Type | Column(s) | Purpose |
|---|---|---|---|
| `recipe_chunks_embedding_ivfflat_idx` | IVFFlat cosine | `embedding` | ANN vector search |
| `recipe_chunks_tags_gin_idx` | GIN | `tags` | `tags @> ARRAY[...]` containment filter |
| `recipe_chunks_total_time_idx` | btree | `total_time_mins` | `total_time_mins <= N` filter |

---

## Hard Constraints

### C1 — Never DROP the `recipe_chunks` table in production

Dropping the table destroys all 30,000 embeddings. Re-embedding takes ~30 seconds on
CPU for 30k chunks; at 100k+ chunks it would take significantly longer.

**Allowed operations:**
- `TRUNCATE recipe_chunks` — only via `truncate_chunks()` in `src/db.py`, only when
  called with explicit `force_reingest=True` from the pipeline.
- `ALTER TABLE recipe_chunks ADD COLUMN ...` — safe, always provide a `DEFAULT`.
- `ALTER TABLE recipe_chunks DROP COLUMN ...` — safe for non-embedding columns only.

**Forbidden without explicit approval:**
- `DROP TABLE recipe_chunks`
- Changing `embedding` column type or dimension (vector dimension cannot be altered
  in-place; requires full re-embed)

### C2 — Never ALTER the embedding dimension

The `embedding` column is `vector(384)`. The `all-MiniLM-L6-v2` model produces 384-dim
vectors. If you switch models (e.g. to `all-mpnet-base-v2` at 768-dim) you must:

1. Drop and recreate the `embedding` column: `ALTER TABLE … DROP COLUMN embedding; ALTER TABLE … ADD COLUMN embedding vector(768);`
2. Re-embed all chunks: set `force_reingest=True` in `ingest_to_pgvector()`.
3. Recreate the IVFFlat index with the new dimension.

**Never mix embedding dimensions in the same column.**

### C3 — Always use the upsert pattern for inserts

All inserts must use:
```sql
INSERT INTO recipe_chunks (...)
VALUES %s
ON CONFLICT (recipe_id, section, chunk_index)
DO UPDATE SET embedding = EXCLUDED.embedding, ...
```

A plain `INSERT` without `ON CONFLICT` will raise a `UniqueViolation` on re-runs.

### C4 — Never run raw DDL outside `src/db.py::setup_schema()`

All schema changes (CREATE TABLE, CREATE INDEX, ALTER TABLE) must go through
`setup_schema()`. This ensures the schema is always applied in the correct order and
idempotently (using `IF NOT EXISTS` / `IF NOT EXISTS` guards).

### C5 — IVFFlat index requires a populated table

The IVFFlat index cannot be created on an empty table — pgvector needs training vectors.
The index is created **after** bulk insert, inside `src/embedder.py::ingest_to_pgvector()`,
not in `setup_schema()`.

`setup_schema()` creates the index only if rows already exist (checked via `row_count()`).
This is intentional — do not move index creation to `setup_schema()`.

### C6 — Connection configuration is environment-variable driven

Connection defaults in `src/db.py`:

| Variable | Default | Override |
|---|---|---|
| `PGHOST` | `localhost` | Export env var |
| `PGPORT` | `5432` | Export env var |
| `PGDATABASE` | `recipes` | Export env var |
| `PGUSER` | `os.getlogin()` | Export env var |
| `PGPASSWORD` | `""` | Export env var |

**Never hard-code credentials in source code.**

For Docker: `PGUSER=recipes PGPASSWORD=recipes`  
For Homebrew local: no `PGPASSWORD` needed (peer auth).

### C7 — GIN index must exist before filtered queries

`src/filters.py` uses `tags @> ARRAY[...]` containment, which requires a GIN index
on `tags` for acceptable performance. The index is created in `setup_schema()`:

```sql
CREATE INDEX IF NOT EXISTS recipe_chunks_tags_gin_idx
    ON recipe_chunks USING GIN (tags);
```

If you drop this index, filtered queries degrade to a sequential scan: O(N) per query
instead of O(log N). At 30k rows this adds ~20ms per query; at 300k rows it becomes
unacceptable.

### C8 — Probe count is a session-level setting

`ivfflat.probes` (the ANN search breadth) is set per query via:
```sql
SET LOCAL ivfflat.probes = 10;
```

`SET LOCAL` scopes it to the current transaction. Do not use `SET` (session-level)
in a pooled connection environment, as it would leak the setting across requests.

**Default probes = 10** balances recall (~90%) and latency (~30ms on 30k rows).
Increase to 50+ if recall matters more than latency.

---

## Safe Operations Reference

```python
# Safe: read row count
from src.db import row_count
count = row_count(conn)

# Safe: full re-ingest (clears and re-embeds everything)
from src.embedder import ingest_to_pgvector
ingest_to_pgvector(chunks, conn, force_reingest=True)

# Safe: partial upsert (only new/changed chunks)
ingest_to_pgvector(chunks, conn, force_reingest=False)

# Safe: retrieve with probe tuning
from src.retrieval import retrieve
results, latency = retrieve("easy vegan dinner", conn, top_k=5, probes=10)

# Safe: metadata-filtered retrieve
from src.filters import filtered_retrieve
results, latency = filtered_retrieve("keto dinner under 30 minutes", conn, top_k=5)
```

---

## Migrations Runbook

### Add a column

```sql
-- In a new migration, never in setup_schema() unless you add IF NOT EXISTS
ALTER TABLE recipe_chunks ADD COLUMN IF NOT EXISTS source_url TEXT;
```

### HNSW migration (at ~100k rows, zero downtime)

```sql
-- Step 1: build HNSW concurrently (old IVFFlat still serves queries)
CREATE INDEX CONCURRENTLY recipe_chunks_hnsw_idx
    ON recipe_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 200);

-- Step 2: drop the old IVFFlat index
DROP INDEX recipe_chunks_embedding_ivfflat_idx;

-- Step 3: tune at query time (default ef_search = 40)
SET LOCAL hnsw.ef_search = 64;
```

No re-embedding required — the `embedding` column values are unchanged.
