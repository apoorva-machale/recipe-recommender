# src/ARCHITECTURE.md — System Architecture Decisions

This document records **why** the system is built the way it is.
Read this before modifying any `src/` module.

---

## System Overview

```
                        ┌──────────────────────────────────────┐
                        │            run.py (entry)            │
                        └──────────────────┬───────────────────┘
                                           │
                        ┌──────────────────▼───────────────────┐
                        │         src/pipeline.py              │
                        │  Phase 1: chunk  Phase 2: embed+ingest│
                        │  Phase 3: n_probe bench              │
                        │  Phase 4: filter bench               │
                        │  Phase 5: chatbot REPL               │
                        └──┬──────────┬──────────┬────────────┘
                           │          │          │
             ┌─────────────▼──┐  ┌───▼──────┐  ┌▼────────────────┐
             │  src/chunker.py│  │src/db.py │  │src/embedder.py  │
             │  PDF/TXT parse │  │DDL+conn  │  │MiniLM-L6-v2     │
             │  ChunkRecord   │  │          │  │IVFFlat index    │
             └────────────────┘  └────┬─────┘  └────────────────┘
                                      │
                        ┌─────────────▼────────────┐
                        │   PostgreSQL + pgvector  │
                        │   recipe_chunks table    │
                        │   vector(384) column     │
                        └─────────────┬────────────┘
                                      │
                   ┌──────────────────▼───────────────────┐
                   │  src/retrieval.py    src/filters.py   │
                   │  cosine ANN search   SQL pre-filter   │
                   │  n_probe tuning      NL → WHERE       │
                   └──────────────────────────────────────┘
```

---

## Decision Log

### D1 — Chunk Unit: Section-Level, Not Whole-Recipe or Step-Level

**Decision:** One chunk = one recipe section (ingredients | method | tips).

**Alternatives considered:**
- *Whole-recipe chunk*: centroid drifts to the average of all three sections.
  A query for "chicken and lemon" would compete with the cooking method signal.
- *One-step-per-chunk*: "Flip carefully and cook another 2 minutes" is meaningless
  without ingredient context. Recall degrades.

**Section-level wins** because user queries hit two axes simultaneously — ingredient
axis ("chicken, lemon") and constraint axis ("under 30 minutes, keto"). Section-level
chunks let the retriever serve each axis from the most relevant chunk type.

---

### D2 — Overlap: 150 Chars on Long Method Sections Only

**Decision:** Apply 150-char trailing overlap only to method sections >1,200 chars
when splitting at step boundaries.

**Rationale:**
- Overlap prevents orphaned references (e.g. "add the mixture from step 3" landing
  in a chunk without the context of step 3).
- 150 chars ≈ 1 medium sentence — enough context, not so much that it inflates
  embeddings with redundant signal.
- Ingredients are an enumeration (order-independent) — overlap adds no value.
- Tips are self-contained bullets — overlap adds no value.

---

### D3 — Embedding Model: all-MiniLM-L6-v2

**Decision:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dim.

**Alternatives considered:**
- `text-embedding-3-small` (OpenAI, 1,536-dim): re-embed cost for 30k chunks ≈ $0.05;
  acceptable in production but creates API dependency and per-request latency.
- `all-mpnet-base-v2` (768-dim): 2× RAM, 2× index size, marginal recall improvement
  at this scale. Not worth the trade-off on CPU-only machines.

**MiniLM-L6-v2 wins** because it is MIT-licensed, CPU-efficient (~28s for 30k chunks),
ships offline, and achieves cosine ≥ 0.76 for near-synonym pairs (well above the 0.60
guard threshold).

---

### D4 — Vector Store: pgvector over FAISS

**Decision:** PostgreSQL + pgvector instead of FAISS.

**FAISS limitations that drove migration:**
1. **No structured metadata filtering** — FAISS cannot filter by `tags`, `total_time_mins`,
   or `calories_kcal` without a parallel data store and post-fetch joins.
2. **No concurrent writes** — FAISS index is in-process memory; multiple writer processes
   would corrupt it.
3. **Training requirement** — IVFFlat needs `nlist × 39` training vectors minimum.
   With only 150 chunks the warning fires: "WARNING clustering 150 points to 64 centroids".
4. **No durability** — index lives on disk as a flat file; no transactions, no WAL.

**pgvector wins** because:
- SQL `WHERE` clauses on any column serve as native metadata pre-filters.
- PostgreSQL handles concurrency, transactions, and durability.
- `SET ivfflat.probes = K` is a session-level parameter — identical semantics to
  FAISS `index.nprobe`.
- HNSW migration path is zero-downtime: `CREATE INDEX CONCURRENTLY … USING hnsw`.

---

### D5 — IVFFlat Parameters: lists=64

**Decision:** `lists=64` for the initial 30k-row corpus.

**Rule of thumb:** `lists ≈ sqrt(N)` for N rows.
- At 30,000 rows: `sqrt(30000) ≈ 173`. We use 64 because:
  - The corpus is synthetic and query distribution is narrow — fewer lists is fine.
  - `lists=64` requires only 64 × 39 = 2,496 training vectors; well within 30k.
  - Probes=10 with 64 lists still covers 15% of the index per query.

**When to switch to HNSW:** at ~100,000 rows. See migration steps in `src/embedder.py`.

---

### D6 — Metadata Schema: 14 Fields per Chunk

**Decision:** Each `ChunkRecord` carries full recipe metadata (not just a foreign key
to a separate recipe table).

**Rationale:** Denormalised schema eliminates joins at query time. A retrieval query
returns everything needed to render a chatbot response in a single SQL query.
At 30k rows the storage overhead (~2KB/row × 30k = 60MB) is negligible.

Fields: `recipe_id`, `title`, `section`, `text`, `cuisine`, `tags[]`, `dietary[]`,
`prep_time_mins`, `cook_time_mins`, `total_time_mins`, `servings`, `calories_kcal`,
`chunk_index`, `char_count`, `embedding`.

---

### D7 — SQL Metadata Filtering Strategy

**Decision:** Parse natural-language constraints from the query string, convert to
SQL `WHERE` clauses, run ANN search on the candidate pool, fall back to unfiltered
if pool < 10 candidates.

**Implemented filters** (`src/filters.py`):
- Dietary tags: `tags @> ARRAY['vegan']` (GIN index — O(1) containment check)
- Time constraint: `total_time_mins <= N` (btree index)
- Calorie constraint: `calories_kcal <= N` (btree index)

**Fallback threshold:** If filtered pool has < 10 rows, the filter is probably too
restrictive — fall back to unfiltered to avoid returning zero results.

---

### D8 — PDF Format: Structured Plain Text Layout

**Decision:** PDFs follow a rigid structured-text format so the existing regex parser
works unchanged — `pypdf` extracts plain text, the same `_extract_meta()` and
`_split_sections()` functions process it.

**Format:**
```
Title: ...
Cuisine: ...
Prep Time: X mins | Cook Time: Y mins | Servings: Z | Calories: N kcal
Tags: ...

Ingredients
-----------
- ...

Method
------
1. ...

Tips
----
• ...
```

This means the PDF and TXT pipelines are identical above the file-read level.

---

## Module Dependency Graph

```
pipeline.py
  ├── chunker.py      (no deps on other src/ modules)
  ├── db.py           (no deps on other src/ modules)
  ├── embedder.py     → chunker.py, db.py
  ├── retrieval.py    → embedder.py (model singleton)
  └── filters.py      → retrieval.py
```

**No circular imports.** `chunker.py` and `db.py` are leaves.

---

## Performance Benchmarks (10k PDF Corpus)

| Phase | Time | Notes |
|---|---|---|
| PDF generation | ~39s | 10,000 PDFs, fpdf2 |
| Chunking | ~10s | 30,000 chunks, pypdf extraction |
| Embedding | ~28s | 1,065 chunks/sec, batch_size=64 |
| DB ingest | ~5s | bulk execute_values, page_size=100 |
| IVFFlat index build | ~2s | lists=64 |
| Query latency (cold) | ~100-350ms | model cold-start included |
| Query latency (warm) | ~35-80ms | model cached in memory |

---

## Future Work

- [ ] **HNSW migration** at 100k rows — zero-downtime, no re-embedding needed.
- [ ] **Benchmark ground truth fix** — current recall@5 is misleading because expected
      titles come from the 50 TXT recipes, not the 10k PDF corpus. Generate matched
      ground truth from the PDF corpus.
- [ ] **Test suite** — `assert_real_embedder()` is documented as a CI gate but no
      test files (`tests/`) exist yet.
- [ ] **Reranking** — add a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`)
      for top-K refinement before chatbot response generation.
- [ ] **LLM response generation** — currently the chatbot REPL returns raw chunk text.
      Integrate an LLM (e.g. GPT-4o-mini via OpenAI) for fluent natural-language answers.
