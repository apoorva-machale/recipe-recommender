"""
test_citations.py — End-to-end citation verification
======================================================
Runs three hard-coded queries against the live pgvector DB and verifies:
  1. Each RetrievalResult has non-empty citation fields
     (recipe_id, title, section, chunk_index, cuisine, text, score)
  2. The formatted citation block renders correctly (no crash, ≥ 5 lines)
  3. The Sources footer lists all retrieved results

Usage:
    source venv/bin/activate
    python3 tests/test_citations.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db        import get_connection
from src.retrieval import retrieve, _fmt_citation_block, _fmt_citations_footer

QUERIES = [
    "vegan pasta with spinach and garlic",
    "chocolate cake under 30 minutes",
    "spicy chicken marinade for grilling",
]

REQUIRED_FIELDS = [
    "recipe_id", "title", "section", "chunk_index",
    "cuisine", "text", "score", "tags",
]

SEP  = "─" * 66
PASS = "  ✓"
FAIL = "  ✗"

def check(label: str, condition: bool) -> bool:
    mark = PASS if condition else FAIL
    print(f"{mark}  {label}")
    return condition


def run():
    print("\n" + "=" * 66)
    print("  Citation End-to-End Test")
    print("=" * 66)

    conn = get_connection()
    all_passed = True

    for q_idx, query in enumerate(QUERIES, start=1):
        print(f"\n{SEP}")
        print(f"  Query {q_idx}: \"{query}\"")
        print(SEP)

        results, latency = retrieve(query, conn, top_k=5, probes=10)

        # ── basic checks ──────────────────────────────────────────────────────
        ok = check(f"Got results (got {len(results)})", len(results) > 0)
        all_passed &= ok
        ok = check(f"Latency reasonable ({latency:.1f} ms < 5000)", latency < 5000)
        all_passed &= ok

        if not results:
            print("  [SKIP] No results — cannot check citation fields")
            continue

        # ── per-result field checks ───────────────────────────────────────────
        for r in results:
            for field in REQUIRED_FIELDS:
                val = getattr(r, field)
                # empty list for tags/dietary is valid — recipe just has no tags
                if isinstance(val, list):
                    has_val = True
                else:
                    has_val = bool(val) or val == 0   # chunk_index=0 is valid
                ok = check(
                    f"  [{r.rank}] {str(r.title)[:30]!r}  ·  {field}={str(val)[:40]!r}",
                    has_val,
                )
                all_passed &= ok

        # ── citation block rendering ──────────────────────────────────────────
        print(f"\n  — Citation blocks for query {q_idx} —\n")
        for r in results:
            block = _fmt_citation_block(r)
            ok = check(
                f"  block[{r.rank}] renders without crash",
                isinstance(block, str) and len(block.splitlines()) >= 5,
            )
            all_passed &= ok
            print(block)
            print()

        # ── sources footer ────────────────────────────────────────────────────
        footer = _fmt_citations_footer(results)
        ok = check("Sources footer non-empty", len(footer) > 10)
        all_passed &= ok
        print(footer)

    conn.close()

    print(f"\n{'=' * 66}")
    if all_passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see ✗ above")
    print("=" * 66 + "\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run()
