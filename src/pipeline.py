"""
End-to-end orchestration pipeline  (pgvector backend)
Runs: chunking → pgvector ingest → n_probe benchmark → filter benchmark → chatbot
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunker   import chunk_directory, chunk_pdf_directory, ChunkRecord
from src.db        import get_connection, setup_schema, row_count
from src.embedder  import ingest_to_pgvector, assert_real_embedder, get_model
from src.retrieval import run_nprobe_benchmark, chatbot_loop
from src.filters   import run_filter_benchmark

PDF_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "recipe_pdfs")
TXT_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "recipes")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _detect_corpus() -> tuple[str, str]:
    """
    Returns (directory, mode) where mode is "pdf" or "txt".
    Prefers the PDF directory if it contains at least 1 PDF.
    Falls back to the legacy .txt directory.
    """
    if os.path.isdir(PDF_DIR):
        pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        if pdfs:
            return PDF_DIR, "pdf"
    return TXT_DIR, "txt"


def _bar(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c]    for c in cols))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


def main(interactive: bool = False) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Phase 1: Chunk ────────────────────────────────────────────────────────
    _bar("Phase 1 · Ingestion — Chunking")
    corpus_dir, corpus_mode = _detect_corpus()
    print(f"  Corpus mode : {corpus_mode.upper()}")
    print(f"  Corpus dir  : {corpus_dir}")

    if corpus_mode == "pdf":
        file_count = len([f for f in os.listdir(corpus_dir) if f.endswith(".pdf")])
        print(f"  PDF files   : {file_count}")
        print("  Extracting text from PDFs and chunking…")
        chunks: list[ChunkRecord] = chunk_pdf_directory(corpus_dir)
    else:
        file_count = len([f for f in os.listdir(corpus_dir) if f.endswith(".txt")])
        print(f"  TXT files   : {file_count}")
        chunks = chunk_directory(corpus_dir)

    section_counts: dict[str, int] = {}
    for c in chunks:
        section_counts[c.section] = section_counts.get(c.section, 0) + 1

    print(f"  Recipes   : {len({c.recipe_id for c in chunks})}")
    print(f"  Chunks    : {len(chunks)}")
    for sec, cnt in sorted(section_counts.items()):
        print(f"    {sec:<15}: {cnt}")

    sample = chunks[0]
    print(f"\n  Sample chunk — {sample.recipe_id}")
    print(f"    title          : {sample.title}")
    print(f"    section        : {sample.section}")
    print(f"    tags           : {sample.tags}")
    print(f"    dietary        : {sample.dietary}")
    print(f"    total_time_mins: {sample.total_time_mins}")
    print(f"    calories_kcal  : {sample.calories_kcal}")
    print(f"    char_count     : {sample.char_count}")
    print(f"    text preview   : {sample.text[:120]}…")

    # ── Phase 2: pgvector ingest ──────────────────────────────────────────────
    _bar("Phase 2 · Embedding + pgvector Index")
    print("  Connecting to PostgreSQL + pgvector…")
    conn = get_connection()
    setup_schema(conn)
    ingest_to_pgvector(chunks, conn, force_reingest=False)

    # Q5 guard: assert the model is NOT a hash-based dummy
    assert_real_embedder(get_model())

    total_rows = row_count(conn)
    print(f"  Rows in recipe_chunks : {total_rows}")
    print(f"  Embedding model       : all-MiniLM-L6-v2  (384 dims)")
    print(f"  Index type            : IVFFlat (lists=64, probes=10 default)")
    print(f"  n_probe equivalent    : SET ivfflat.probes = K")

    # ── Phase 3: n_probe benchmark (Q6) ──────────────────────────────────────
    _bar("Phase 3 · Retrieval — ivfflat.probes Benchmark  (Q6)")
    nprobe_rows = run_nprobe_benchmark(conn, n_probe_values=[1, 5, 10, 50])

    display = [{
        "probes":     r["probes"],
        "query":      r["query"][:45] + "…",
        "recall@5":   r["recall@5"],
        "latency_ms": r["latency_ms"],
    } for r in nprobe_rows]
    _table(display, ["probes", "query", "recall@5", "latency_ms"])

    print("\n  Aggregated (mean across 3 queries):")
    for probes in [1, 5, 10, 50]:
        subset       = [r for r in nprobe_rows if r["probes"] == probes]
        mean_recall  = sum(r["recall@5"]   for r in subset) / len(subset)
        mean_latency = sum(r["latency_ms"] for r in subset) / len(subset)
        print(f"    probes={probes:>2}  recall@5={mean_recall:.2f}  latency={mean_latency:.1f} ms")

    # ── Phase 4: filter benchmark (Stretch Goal A) ────────────────────────────
    _bar("Phase 4 · Stretch Goal A — SQL Metadata Pre-filtering")
    filter_rows = run_filter_benchmark(conn)

    _table(filter_rows, [
        "query", "candidate_pool",
        "recall_unfiltered", "recall_filtered",
        "latency_unfiltered_ms", "latency_filtered_ms",
    ])

    print("\n  Interpretation:")
    for row in filter_rows:
        dr = row["recall_filtered"] - row["recall_unfiltered"]
        dl = row["latency_filtered_ms"] - row["latency_unfiltered_ms"]
        print(f"    {row['query']}")
        print(f"      pool={row['candidate_pool']}  "
              f"recall {'↑' if dr>0 else ('↓' if dr<0 else '=')}{abs(dr):.2f}  "
              f"latency {'↑' if dl>0 else '↓'}{abs(dl):.1f}ms")

    # ── Persist results ───────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(out_path, "w") as fh:
        json.dump({
            "nprobe_benchmark": nprobe_rows,
            "filter_benchmark": filter_rows,
            "corpus_stats": {
                "recipes":        len({c.recipe_id for c in chunks}),
                "total_chunks":   len(chunks),
                "section_counts": section_counts,
                "backend":        "pgvector",
            },
        }, fh, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Interactive chatbot ───────────────────────────────────────────────────
    if interactive:
        chatbot_loop(conn)

    conn.close()
    _bar("Pipeline complete")


if __name__ == "__main__":
    main(interactive=True)
