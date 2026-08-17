# 🍳 Recipe RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** ingestion and retrieval system for a
recipe chatbot — built around a real-world prompt: *"A food startup wants a
chatbot that answers user questions from a library of 10,000 recipe PDFs.
Build the ingestion pipeline and a working retrieval loop, then diagnose why
it sometimes returns irrelevant recipes."*

It ingests recipe documents, chunks them at the section level (ingredients /
method / tips), embeds each chunk, stores them in **PostgreSQL + pgvector**,
and exposes a query loop that combines semantic search with metadata
filtering (dietary tags, cook time, calories).

---

## Features

- 📄 **Section-aware chunking** — recipes are split by ingredients / method /
  tips rather than as one blob, keeping embeddings sharp.
- 🔍 **Semantic + metadata search** — vector similarity combined with SQL
  pre-filtering on tags, time, and calories.
- 🧠 **Embedding sanity guard** — a startup check that catches the classic RAG
  failure mode of a broken/hash-based embedder silently degrading every
  result.
- ⚡ **Tunable recall/latency** — benchmarking across `ivfflat.probes` values
  to balance speed against retrieval quality.
- 🧰 **One-command pipeline** — `python3 run.py` generates the corpus, embeds
  it, benchmarks it, and drops you into a chatbot REPL.

## Tech Stack

| Layer | Choice |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) (IVFFlat) |
| Corpus | 10,000 synthetic recipe PDFs → 30,000 section-level chunks |
| Language | Python 3 |

## Quick Start

**Requirements:** Python 3.10+, Docker (for PostgreSQL) or a local PostgreSQL 16+ install.

```bash
# 1. Clone and enter the project
git clone https://github.com/apoorva-machale/Grind.git
cd Grind/recipe_recommender

# 2. Set up the environment and start the database
make setup        # create venv + install dependencies
make db-up         # start PostgreSQL + pgvector via docker compose

# 3. Run the full pipeline: generate corpus → chunk → embed → benchmark → chatbot
make run
```

Or run it manually:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker compose up -d
export PGHOST=localhost PGPORT=5432 PGDATABASE=recipes PGUSER=recipes PGPASSWORD=recipes
python3 run.py
```

`run.py` is idempotent — it skips PDF generation and re-ingestion if they've
already run.

### Option C — Fully Dockerized

No local Python or PostgreSQL install needed — everything runs in containers:

```bash
docker compose up --build
```

This starts PostgreSQL + pgvector, waits for it to be healthy, then builds and
runs the app container, which executes `run.py` end-to-end. Since the last
step is an interactive chatbot REPL, attach a terminal to it:

```bash
docker compose run --rm app
```

Generated PDFs and benchmark results are persisted to `data/recipe_pdfs/` and
`results/` on the host via bind mounts, so re-running `docker compose up`
skips regenerating/re-ingesting the corpus, same as the local flow.

## How It Works

1. **Chunk** — each recipe is split into ingredients, method, and tips
   sections (one chunk per section), with overlap only on long method text.
2. **Embed** — chunks are embedded with `all-MiniLM-L6-v2` and stored in
   Postgres as `vector(384)` columns.
3. **Filter** — natural-language constraints ("vegan", "under 30 minutes")
   are parsed into SQL `WHERE` clauses and applied before the vector search.
4. **Retrieve** — cosine similarity search via pgvector's IVFFlat index,
   tuned via `ivfflat.probes`.
5. **Guard** — a startup assertion checks the embedder produces real
   semantic vectors (not degenerate hash output), which is the direct fix
   for the "sometimes returns irrelevant recipes" symptom.

## Roadmap

- Hybrid dense + BM25 retrieval with cross-encoder reranking
- LLM-based answer synthesis with inline citations
- Swap the synthetic corpus for a real recipe dataset
- Wrap retrieval in a FastAPI service
- CI pipeline (lint, tests, embedding sanity check)

## License

[MIT](LICENSE)
