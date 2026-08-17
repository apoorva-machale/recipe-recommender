"""
Stretch Goal A — Metadata Pre-filtering  (pgvector backend)
============================================================

pgvector vs FAISS for filtered search
--------------------------------------
FAISS approach (old):
  Build a temporary in-memory IndexFlatIP subset from matching chunk IDs,
  then run ANN on it.  Cost: O(|candidates|) index rebuild on every query.
  Bug-prone: id_map must stay in sync with the subset index.

pgvector approach (new):
  Add a WHERE clause to the ANN query.  PostgreSQL evaluates the filter
  BEFORE or ALONGSIDE the vector scan — no separate index rebuild needed.

  With an IVFFlat index:
    PostgreSQL uses the filter to prune candidate rows from each Voronoi
    cell.  A GIN index on the tags[] column makes the array containment
    check (&>) near-instant.

  With an HNSW index:
    PostgreSQL applies the filter as a post-scan predicate.  For high-
    selectivity filters (small candidate pool), consider partial indexes:
      CREATE INDEX ... WHERE 'vegan' = ANY(tags);

SQL operators used
------------------
  tags @> ARRAY['vegan']::text[]         -- tags contains ALL of these
  total_time_mins <= %(max_time)s         -- time constraint
  calories_kcal   <= %(max_cal)s          -- calorie constraint

Fallback
--------
If the filtered pool would return fewer than min_candidates rows, we
skip the filter rather than returning an empty result set.  The caller
can inspect filter_stats['used_filter'] to see which path was taken.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import get_model
from .retrieval import RetrievalResult, recall_at_k, TEST_QUERIES, retrieve

# ── keyword dictionaries ──────────────────────────────────────────────────────
DIETARY_KEYWORDS: Dict[str, List[str]] = {
    "vegan":       ["vegan", "plant-based"],
    "vegetarian":  ["vegetarian", "veggie"],
    "keto":        ["keto", "ketogenic"],
    "gluten-free": ["gluten-free", "gluten free", "gf"],
    "paleo":       ["paleo"],
    "dairy-free":  ["dairy-free", "dairy free"],
    "low-carb":    ["low-carb", "low carb"],
}

TIME_PATTERN    = re.compile(r"under\s+(\d+)\s*(?:min(?:ute)?s?|mins?)", re.IGNORECASE)
CALORIE_PATTERN = re.compile(r"under\s+(\d+)\s*(?:kcal|calories?|cal)",  re.IGNORECASE)


# ── query parser ──────────────────────────────────────────────────────────────
def parse_filters(query: str) -> dict:
    """
    Extract structured filters from a natural-language query.

    Returns
    -------
    {
      "dietary":   ["vegan", "keto", ...],
      "max_time":  int | None,
      "max_cal":   int | None,
    }
    """
    q = query.lower()
    dietary: List[str] = [
        label
        for label, synonyms in DIETARY_KEYWORDS.items()
        if any(s in q for s in synonyms)
    ]
    max_time = int(m.group(1)) if (m := TIME_PATTERN.search(q)) else None
    max_cal  = int(m.group(1)) if (m := CALORIE_PATTERN.search(q)) else None
    return {"dietary": dietary, "max_time": max_time, "max_cal": max_cal}


# ── candidate count helper ────────────────────────────────────────────────────
def _count_candidates(conn, filters: dict) -> int:
    """Count rows that satisfy the metadata filters (no vector scan)."""
    clauses = []
    params: dict = {}

    if filters["dietary"]:
        clauses.append("tags @> %(dietary)s::text[]")
        params["dietary"] = filters["dietary"]
    if filters["max_time"] is not None:
        clauses.append("total_time_mins <= %(max_time)s")
        params["max_time"] = filters["max_time"]
    if filters["max_cal"] is not None:
        clauses.append("calories_kcal <= %(max_cal)s")
        params["max_cal"] = filters["max_cal"]

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT COUNT(*) FROM recipe_chunks {where}"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        (n,) = cur.fetchone()
    return n


# ── filtered retrieval ────────────────────────────────────────────────────────
def filtered_retrieve(
    query:          str,
    conn,
    top_k:          int = 5,
    probes:         int = 10,
    min_candidates: int = 10,
) -> Tuple[List[RetrievalResult], float, dict]:
    """
    1. Parse filters from query text.
    2. Count matching rows (fast — GIN + btree indexes, no vector scan).
    3a. If pool >= min_candidates: run ANN with WHERE clause.
    3b. Otherwise: fall back to unfiltered ANN search.

    Returns
    -------
    results       : ranked RetrievalResults
    latency_ms    : total wall-clock time in ms
    filter_stats  : dict describing what was applied and pool size
    """
    t0      = time.perf_counter()
    filters = parse_filters(query)
    pool    = _count_candidates(conn, filters) if any(
        v for v in filters.values() if v
    ) else None

    filter_stats = {
        "filters_applied": filters,
        "candidate_pool":  pool,
        "used_filter":     pool is not None and pool >= min_candidates,
    }

    model = get_model()
    q_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    if filter_stats["used_filter"]:
        # Build WHERE clause — all filters are additive (AND)
        clauses = []
        params: dict = {"vec": q_vec.tolist(), "probes": probes, "top_k": top_k}

        if filters["dietary"]:
            clauses.append("tags @> %(dietary)s::text[]")
            params["dietary"] = filters["dietary"]
        if filters["max_time"] is not None:
            clauses.append("total_time_mins <= %(max_time)s")
            params["max_time"] = filters["max_time"]
        if filters["max_cal"] is not None:
            clauses.append("calories_kcal <= %(max_cal)s")
            params["max_cal"] = filters["max_cal"]

        where = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SET ivfflat.probes = %(probes)s;
            SELECT
                recipe_id,
                title,
                section,
                LEFT(text, 300)                           AS text,
                tags,
                dietary,
                total_time_mins,
                calories_kcal,
                1 - (embedding <=> %(vec)s::vector)       AS score
            FROM recipe_chunks
            {where}
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT %(top_k)s;
        """
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
        return results, latency_ms, filter_stats

    else:
        # Fallback: unfiltered search
        filter_stats["used_filter"] = False
        results, latency_ms = retrieve(query, conn, top_k=top_k, probes=probes)
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms, filter_stats


# ── benchmark: filtered vs unfiltered ────────────────────────────────────────
FILTERED_TEST_QUERIES = [
    {
        "query": "Give me a low-carb dessert under 30 minutes",
        "relevant_titles": ["Keto Chocolate Mousse", "Banana Oat Cookies"],
    },
    {
        "query": "Easy vegan dinner with vegetables",
        "relevant_titles": ["Vegan Black Bean Tacos", "Thai Green Curry with Tofu"],
    },
    {
        "query": "Quick gluten-free high-protein meal under 30 minutes",
        "relevant_titles": ["Salmon with Miso Glazed Bok Choy", "Shrimp Stir Fry with Vegetables"],
    },
]


def run_filter_benchmark(conn, top_k: int = 5) -> List[dict]:
    """Compare filtered vs unfiltered search on recall@5 and latency."""
    rows = []
    for tq in FILTERED_TEST_QUERIES:
        res_unf, lat_unf = retrieve(tq["query"], conn, top_k=top_k, probes=10)
        recall_unf       = recall_at_k(res_unf, tq["relevant_titles"], k=top_k)

        res_filt, lat_filt, stats = filtered_retrieve(
            tq["query"], conn, top_k=top_k
        )
        recall_filt = recall_at_k(res_filt, tq["relevant_titles"], k=top_k)

        rows.append({
            "query":                  tq["query"][:55] + "…",
            "filters":                stats["filters_applied"],
            "candidate_pool":         stats["candidate_pool"],
            "used_filter":            stats["used_filter"],
            "recall_unfiltered":      round(recall_unf, 2),
            "recall_filtered":        round(recall_filt, 2),
            "latency_unfiltered_ms":  round(lat_unf, 2),
            "latency_filtered_ms":    round(lat_filt, 2),
        })
    return rows
