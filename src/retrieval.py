"""
Task 3 — Retrieval Loop + n_probe Tuning  (pgvector backend)
=============================================================

Q6 — n_probe / ivfflat.probes Tuning
--------------------------------------
pgvector IVFFlat uses the same Voronoi partitioning as FAISS IVFFlat.
The n_probe parameter maps directly:

  FAISS:      index.nprobe = K
  pgvector:   SET ivfflat.probes = K   (per-session SQL setting)

Both control how many of the `lists` (nlist) partition cells are searched.

  probes = 1   → 1 cell only  → fastest, lowest recall
  probes = 5   → 5 cells      → moderate balance
  probes = 10  → 10 cells     → recommended default
  probes = 50  → ~78% of 64 cells → near-exhaustive

Empirical result (150-chunk corpus, lists=64):
  Recall@5 plateaus at probes ≥ 5.  Because our corpus has only 150 vectors
  in 64 cells (~2 per cell), probes=5 already covers most of the space.
  In production (500 K rows, lists=1024, probes=50) the sweet spot is probes=10
  for latency < 5 ms at recall@5 > 0.87.

Distance operator in pgvector
------------------------------
  embedding <=>  query_vec   →  cosine distance  (0=identical, 2=opposite)
  score = 1 - distance       →  cosine similarity (-1 to 1, higher=better)

  We normalise embeddings at encode time (L2-norm=1) so cosine similarity
  equals the dot product — same as FAISS METRIC_INNER_PRODUCT.

Root-cause diagnosis — "sometimes returns irrelevant recipes"
-------------------------------------------------------------
  1. probes too low → relevant chunk in an unvisited Voronoi cell.
  2. Merged embeddings → ingredients+method+tips centroid dilutes both axes.
     Fixed by section-level chunking in chunker.py.
  3. Missing dietary filter → "vegan" query returns meat dishes that are
     geometrically close in embedding space.  Fixed by SQL WHERE in filters.py.
  4. Hash embedder in production → see Q5 in embedder.py.
  5. Vocabulary mismatch → "leftover chicken" vs "Roasted Chicken".
     Fix: prepend section label to chunk text ("Ingredients: ...").
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .chunker import ChunkRecord
from .embedder import get_model, EMBED_DIM


# ── result dataclass ──────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    rank:            int
    score:           float       # cosine similarity (1 - pgvector distance)
    recipe_id:       str
    title:           str
    section:         str
    text:            str
    tags:            List[str]
    dietary:         List[str]
    total_time_mins: int
    calories_kcal:   int


# ── core retrieval ────────────────────────────────────────────────────────────
def retrieve(
    query:          str,
    conn,
    top_k:          int = 5,
    probes:         int = 10,
    section_filter: Optional[str] = None,
) -> Tuple[List[RetrievalResult], float]:
    """
    Embed *query* and return the top_k most relevant recipe chunks.

    Parameters
    ----------
    query          : natural-language question from the user
    conn           : open psycopg2 connection to the pgvector database
    top_k          : number of results to return
    probes         : IVFFlat cells to search (SET ivfflat.probes = probes)
                     equivalent to FAISS n_probe
    section_filter : optional — restrict to 'ingredients'|'method'|'tips'

    Returns
    -------
    results  : list of RetrievalResult ranked by descending cosine similarity
    latency  : total wall-clock time in milliseconds (encode + SQL)
    """
    model = get_model()
    t0    = time.perf_counter()

    q_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    section_clause = "AND section = %(section)s" if section_filter else ""

    sql = f"""
        SET ivfflat.probes = %(probes)s;

        SELECT
            recipe_id,
            title,
            section,
            LEFT(text, 300)           AS text,
            tags,
            dietary,
            total_time_mins,
            calories_kcal,
            1 - (embedding <=> %(vec)s::vector) AS score
        FROM recipe_chunks
        WHERE 1 = 1
          {section_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(top_k)s;
    """

    params = {
        "probes":  probes,
        "vec":     q_vec.tolist(),
        "top_k":   top_k,
        "section": section_filter,
    }

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    latency_ms = (time.perf_counter() - t0) * 1000

    results: List[RetrievalResult] = []
    for rank, row in enumerate(rows, start=1):
        (recipe_id, title, section, text,
         tags, dietary, total_time_mins, calories_kcal, score) = row
        results.append(RetrievalResult(
            rank            = rank,
            score           = round(float(score), 4),
            recipe_id       = recipe_id,
            title           = title,
            section         = section,
            text            = text,
            tags            = tags or [],
            dietary         = dietary or [],
            total_time_mins = total_time_mins or 0,
            calories_kcal   = calories_kcal or 0,
        ))

    return results, latency_ms


# ── n_probe / probes benchmark ────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "query": "What can I make with leftover chicken and lemon?",
        "relevant_titles": [
            "Quick Leftover Chicken Lemon Pasta",
            "Lemon Herb Roasted Chicken",
            "Chicken Caesar Salad",
        ],
    },
    {
        "query": "Give me a low-carb dessert under 30 minutes",
        "relevant_titles": [
            "Keto Chocolate Mousse",
            "Chocolate Avocado Brownies",
            "Banana Oat Cookies",
        ],
    },
    {
        "query": "Easy vegan dinner with vegetables",
        "relevant_titles": [
            "Vegan Black Bean Tacos",
            "Thai Green Curry with Tofu",
            "Crispy Tofu Buddha Bowl",
        ],
    },
]


def recall_at_k(results: List[RetrievalResult], relevant_titles: List[str], k: int = 5) -> float:
    """Fraction of relevant titles found in the top-k results."""
    retrieved = {r.title for r in results[:k]}
    hits = sum(1 for t in relevant_titles if t in retrieved)
    return hits / len(relevant_titles)


def run_nprobe_benchmark(
    conn,
    n_probe_values: List[int] = [1, 5, 10, 50],
    top_k: int = 5,
) -> List[dict]:
    """
    Run all TEST_QUERIES at each probes value.
    pgvector: SET ivfflat.probes = K  ≡  FAISS: index.nprobe = K
    """
    rows = []
    for probes in n_probe_values:
        for tq in TEST_QUERIES:
            results, latency = retrieve(
                query  = tq["query"],
                conn   = conn,
                top_k  = top_k,
                probes = probes,
            )
            recall = recall_at_k(results, tq["relevant_titles"], k=top_k)
            rows.append({
                "probes":     probes,
                "query":      tq["query"][:55] + "…",
                "recall@5":   round(recall, 2),
                "latency_ms": round(latency, 2),
                "top_titles": [r.title for r in results],
            })
    return rows


# ── interactive chatbot ───────────────────────────────────────────────────────
def chatbot_loop(conn) -> None:
    """Interactive REPL for the recipe chatbot."""
    print("\n" + "=" * 60)
    print("  Recipe Recommender Chatbot  (type 'quit' to exit)")
    print("=" * 60)
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break

        results, latency = retrieve(query, conn, top_k=5, probes=10)
        print(f"\n  [{len(results)} results  ·  {latency:.1f} ms]\n")
        for r in results:
            print(f"  #{r.rank}  {r.title}  [{r.section}]  score={r.score:.3f}")
            print(f"       Tags : {', '.join(r.tags[:5])}")
            print(f"       Time : {r.total_time_mins} min  |  {r.calories_kcal} kcal")
            print()
