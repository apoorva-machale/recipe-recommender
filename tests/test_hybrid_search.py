"""
test_hybrid_search.py — end-to-end test: hybrid search + reranker + citations
================================================================================
Verifies:
  1. DB tsvector column exists and is populated
  2. hybrid_retrieve() returns results and is faster than 3 s
  3. rerank() changes the order (or at least doesn't crash)
  4. build_cited_answer() produces inline [N:recipe_id] tags
  5. Spot-checks two queries that expose BM25 vs dense differences

Usage:
    source venv/bin/activate
    python3 tests/test_hybrid_search.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db        import get_connection, setup_schema
from src.retrieval import (
    retrieve, hybrid_retrieve, rerank,
    build_cited_answer, _fmt_sources_list,
)

SEP  = "─" * 66
PASS = "  ✓"
FAIL = "  ✗"

def check(label: str, ok: bool) -> bool:
    print(f"{PASS if ok else FAIL}  {label}")
    return ok

def run():
    print("\n" + "=" * 66)
    print("  Hybrid Search + Reranker — End-to-End Test")
    print("=" * 66)

    conn = get_connection()
    # ensure tsvector column exists (idempotent migration)
    setup_schema(conn)

    all_ok = True

    # ── Check 1: tsvector column populated ───────────────────────────────────
    print(f"\n{SEP}\n  Check 1: tsvector column\n{SEP}")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM recipe_chunks
            WHERE text_search_vec IS NOT NULL
        """)
        (populated,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM recipe_chunks")
        (total,) = cur.fetchone()

    all_ok &= check(f"text_search_vec populated: {populated:,} / {total:,} rows",
                    populated > 0)

    # ── Queries to test ───────────────────────────────────────────────────────
    # Q1: exact recipe name — BM25 should boost it; dense might miss
    # Q2: conceptual query — dense should dominate; BM25 helps with keywords
    QUERIES = [
        ("Chicken Tikka Masala",                  "exact name — BM25 decisive"),
        ("easy vegan dinner with vegetables",      "conceptual — dense + BM25"),
        ("spicy chicken marinade for grilling",    "keyword + semantic blend"),
    ]

    for query, note in QUERIES:
        print(f"\n{SEP}\n  Query: \"{query}\"  ({note})\n{SEP}")

        # Stage 1: hybrid
        h_results, t_hybrid = hybrid_retrieve(query, conn, top_k=20, probes=10)
        all_ok &= check(f"hybrid_retrieve got {len(h_results)} results", len(h_results) > 0)
        all_ok &= check(f"hybrid latency {t_hybrid:.0f} ms < 3000", t_hybrid < 3000)

        # Stage 2: rerank
        reranked, t_rerank = rerank(query, h_results)
        top5 = reranked[:5]
        all_ok &= check(f"rerank returned {len(reranked)} results", len(reranked) > 0)
        all_ok &= check(f"rerank latency {t_rerank:.0f} ms < 5000", t_rerank < 5000)

        # Check scores are floats
        all_ok &= check("reranker scores are numeric",
                        all(isinstance(r.score, float) for r in top5))

        # Stage 3: cited answer rendering
        answer = build_cited_answer(top5)
        all_ok &= check("build_cited_answer non-empty", len(answer) > 50)
        all_ok &= check("[rank:recipe_id] tags present",
                        any(f"[{r.rank}:{r.recipe_id}]" in answer for r in top5))

        # ── Show full output ──────────────────────────────────────────────────
        print(f"\n  Pipeline: hybrid {t_hybrid:.0f} ms  +  rerank {t_rerank:.0f} ms\n")
        print(build_cited_answer(top5))
        print(_fmt_sources_list(top5))

    conn.close()

    print(f"\n{'=' * 66}")
    print(f"  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED — see ✗ above'}")
    print("=" * 66 + "\n")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    run()
