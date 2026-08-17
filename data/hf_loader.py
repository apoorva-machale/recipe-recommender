"""
data/hf_loader.py — Load real recipe datasets from HuggingFace
==============================================================

Supported datasets
------------------
PRIMARY (default):
  Shengtao/recipe          — 32,722 AllRecipes.com recipes, clean structured data
  https://huggingface.co/datasets/Shengtao/recipe

FALLBACK:
  AkashPS11/recipes_data_food.com — Food.com dataset (only ~1,228 valid rows
                                    despite 1M+ total; rest are null-padded)

Field mapping — Shengtao/recipe
--------------------------------
HF field             → ChunkRecord field
-----------            ----------------------
title (str)          → title
category (str)       → cuisine
ingredients (str)    → ingredients chunk  ("; " separated list)
instructions_list    → method chunk       (Python list repr string)
directions (str)     → method chunk fallback
description (str)    → tips chunk
prep_time (str)      → prep_time_mins     "10 mins" → 10
cook_time (str)      → cook_time_mins
total_time (str)     → total_time_mins
servings (str)       → servings           "4 servings" → 4
calories (str)       → calories_kcal      "630.2" → 630
recipe_id            → derived: "recipe_<index>"

Three chunks per recipe
-----------------------
section="ingredients"  text = formatted ingredient list
section="method"       text = numbered instruction steps
section="tips"         text = description + nutrition + rating summary
"""

from __future__ import annotations

import ast
import re
import sys
import os
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunker import ChunkRecord

# ── Supported dataset adapters ─────────────────────────────────────────────────

DATASETS = {
    "shengtao":  "Shengtao/recipe",
    "foodcom":   "AkashPS11/recipes_data_food.com",
}
DEFAULT_DATASET = "shengtao"

# ── Tag / dietary config ───────────────────────────────────────────────────────

DIETARY_KEYWORDS = {"vegan", "vegetarian", "keto", "gluten-free",
                    "dairy-free", "paleo", "halal", "kosher"}

_CATEGORY_TAG_MAP: dict[str, str] = {
    "main-dish":    "main-dish",
    "side-dish":    "side-dish",
    "appetizers":   "appetizer",
    "desserts":     "dessert",
    "breakfast":    "breakfast",
    "beverages":    "beverage",
    "soups":        "soup",
    "salads":       "salad",
    "bread":        "bread",
    "condiments":   "condiment",
}

_DIETARY_TEXT_MAP: dict[str, str] = {
    "vegan":       "vegan",
    "vegetarian":  "vegetarian",
    "keto":        "keto",
    "gluten-free": "gluten-free",
    "gluten free": "gluten-free",
    "dairy-free":  "dairy-free",
    "dairy free":  "dairy-free",
    "paleo":       "paleo",
    "low-carb":    "low-carb",
    "low carb":    "low-carb",
    "healthy":     "healthy",
    "quick":       "quick",
    "easy":        "easy",
}


# ── Parser utilities ──────────────────────────────────────────────────────────

def _parse_minutes(time_str) -> int:
    """
    Parse human-readable or ISO-8601 time strings to total minutes.

    Examples:
      "30 mins"   → 30
      "1 hr 20 min" → 80
      "PT4H25M"   → 265
      "2 hours"   → 120
    """
    if not time_str or str(time_str).strip() in ("None", "nan", "NA", ""):
        return 0
    s = str(time_str).strip()

    # ISO-8601: PT4H25M
    iso = re.match(
        r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s, re.IGNORECASE
    )
    if s.startswith("P") and iso:
        days  = int(iso.group(1) or 0)
        hours = int(iso.group(2) or 0)
        mins  = int(iso.group(3) or 0)
        return min(days * 1440 + hours * 60 + mins, 2880)

    # Human-readable: "1 hr 20 min", "2 hours", "30 mins"
    total = 0
    for val, unit in re.findall(r"(\d+)\s*(hour|hr|h|minute|min|m)s?", s, re.IGNORECASE):
        if unit.lower().startswith("h"):
            total += int(val) * 60
        else:
            total += int(val)

    # Plain number: "30" → 30
    if total == 0:
        m = re.search(r"(\d+)", s)
        if m:
            total = int(m.group(1))

    return min(total, 2880)


def _parse_int(val) -> int:
    """Parse a possibly-string number to int, ignoring text."""
    if val is None:
        return 0
    s = str(val).strip()
    if s in ("None", "nan", "NA", ""):
        return 0
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return int(float(m.group(1))) if m else 0


def _parse_r_vector(val) -> List[str]:
    """Parse R-style c("a", "b") or plain string into a list."""
    if val is None:
        return []
    s = str(val).strip()
    if not s or s in ("NA", "None", "nan", "NULL"):
        return []
    if s.startswith("c("):
        return [item for item in re.findall(r'"([^"]*)"', s) if item.strip()]
    if s.startswith('"') and s.endswith('"'):
        return [s[1:-1]] if len(s) > 2 else []
    return [s]


def _parse_python_list(val) -> List[str]:
    """
    Parse a Python list repr string like "['step1', 'step2']" into a list.
    Falls back to splitting by newline or period.
    """
    if not val or str(val).strip() in ("None", "nan", "[]", ""):
        return []
    s = str(val).strip()
    if s.startswith("["):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, list):
                return [str(i).strip() for i in items if str(i).strip()]
        except (ValueError, SyntaxError):
            pass
    # Fallback: split by newline
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    return lines if len(lines) > 1 else [s]


def _derive_tags(
    category: str,
    description: str,
    total_time: int,
) -> tuple[list[str], list[str]]:
    """Build tags and dietary lists from category, description, and time."""
    tags: list[str] = []
    text = f"{category} {description}".lower()

    # Category tag
    for key, canonical in _CATEGORY_TAG_MAP.items():
        if key in category.lower() and canonical not in tags:
            tags.append(canonical)

    # Dietary / attribute tags from text
    for key, canonical in _DIETARY_TEXT_MAP.items():
        if key in text and canonical not in tags:
            tags.append(canonical)

    # Time-based tags
    if total_time and total_time <= 15:
        tags.append("under-15-minutes")
    if total_time and total_time <= 30:
        tags.append("under-30-minutes")
    if total_time and total_time <= 60:
        tags.append("under-60-minutes")

    dietary = [t for t in tags if t in DIETARY_KEYWORDS]
    return tags, dietary


# ── Text builders ─────────────────────────────────────────────────────────────

def _ingredients_text(ingredients_raw: str) -> str:
    """Format "; " separated ingredient list into a readable chunk."""
    if not ingredients_raw or str(ingredients_raw).strip() in ("None", "nan"):
        return ""
    items = [i.strip() for i in str(ingredients_raw).split(";") if i.strip()]
    if not items:
        return ""
    lines = ["Ingredients:"] + [f"- {item}" for item in items]
    return "\n".join(lines)


def _method_text(instructions_list_raw, directions_raw) -> str:
    """Build numbered step text from instructions_list (preferred) or directions."""
    steps = _parse_python_list(instructions_list_raw)
    if not steps and directions_raw:
        # Fall back to directions as a single block
        text = str(directions_raw).strip()
        if text and text not in ("None", "nan"):
            return f"Instructions:\n{text}"
        return ""
    if not steps:
        return ""
    lines = ["Instructions:"] + [f"{i}. {step}" for i, step in enumerate(steps, 1)]
    return "\n".join(lines)


def _tips_text(
    description,
    category,
    rating,
    rating_count,
    protein_g,
    fat_g,
    carbs_g,
) -> str:
    """Build description/nutrition summary as the tips chunk."""
    parts = []
    desc = str(description or "").strip()
    if desc and desc not in ("None", "nan"):
        parts.append(desc)

    cat = str(category or "").strip()
    if cat and cat not in ("None", "nan"):
        parts.append(f"Category: {cat}.")

    try:
        r = float(rating or 0)
        rc = int(float(rating_count or 0))
        if r > 0:
            parts.append(f"Rating: {r:.1f}/5 ({rc} reviews).")
    except (ValueError, TypeError):
        pass

    nutrition = []
    for label, val in [("Protein", protein_g), ("Fat", fat_g), ("Carbs", carbs_g)]:
        try:
            v = float(val or 0)
            if v > 0:
                nutrition.append(f"{label} {v:.1f}g")
        except (ValueError, TypeError):
            pass
    if nutrition:
        parts.append("Nutrition per serving: " + ", ".join(nutrition) + ".")

    return "  ".join(parts)


# ── Adapters: one per dataset ─────────────────────────────────────────────────

def _row_shengtao(row: dict, index: int) -> List[ChunkRecord]:
    """Convert one Shengtao/recipe row to ChunkRecords."""
    title = str(row.get("title") or "").strip()
    if not title or title in ("None", "nan"):
        return []

    recipe_id = f"recipe_{index}"
    cuisine   = str(row.get("category") or "unknown").strip()

    prep  = _parse_minutes(row.get("prep_time"))
    cook  = _parse_minutes(row.get("cook_time"))
    total = _parse_minutes(row.get("total_time")) or (prep + cook)

    servings = _parse_int(row.get("servings"))
    calories = _parse_int(row.get("calories"))

    tags, dietary = _derive_tags(
        cuisine,
        str(row.get("description") or ""),
        total,
    )

    ing_text  = _ingredients_text(row.get("ingredients", ""))
    meth_text = _method_text(
        row.get("instructions_list"), row.get("directions")
    )
    tips_text = _tips_text(
        description  = row.get("description"),
        category     = cuisine,
        rating       = row.get("rating"),
        rating_count = row.get("rating_count"),
        protein_g    = row.get("protein_g"),
        fat_g        = row.get("fat_g"),
        carbs_g      = row.get("carbohydrates_g"),
    )

    if not any([ing_text, meth_text, tips_text]):
        return []

    records: List[ChunkRecord] = []
    chunk_idx = 0
    for section, text in [
        ("ingredients", ing_text),
        ("method",      meth_text),
        ("tips",        tips_text),
    ]:
        if not text:
            continue
        records.append(ChunkRecord(
            recipe_id       = recipe_id,
            title           = title,
            section         = section,
            text            = text,
            cuisine         = cuisine,
            tags            = tags,
            dietary         = dietary,
            prep_time_mins  = prep,
            cook_time_mins  = cook,
            total_time_mins = total,
            servings        = servings,
            calories_kcal   = calories,
            chunk_index     = chunk_idx,
            char_count      = len(text),
        ))
        chunk_idx += 1
    return records


def _row_foodcom(row: dict, index: int) -> List[ChunkRecord]:
    """Convert one AkashPS11/recipes_data_food.com row to ChunkRecords."""
    if row.get("RecipeId") is None:
        return []
    title = str(row.get("Name") or "").strip()
    if not title:
        return []

    recipe_id = f"recipe_{int(row['RecipeId'])}"
    cuisine   = str(row.get("RecipeCategory") or "unknown").strip()

    prep  = _parse_minutes(row.get("PrepTime"))
    cook  = _parse_minutes(row.get("CookTime"))
    total = _parse_minutes(row.get("TotalTime")) or (prep + cook)

    servings = _parse_int(row.get("RecipeServings"))
    calories = _parse_int(row.get("Calories"))

    # Parse R-style keywords
    raw_kw = _parse_r_vector(row.get("Keywords", ""))
    tags: list[str] = []
    for kw in raw_kw:
        lower = kw.strip().lower()
        for key, canonical in _DIETARY_TEXT_MAP.items():
            if key in lower and canonical not in tags:
                tags.append(canonical)
        if "< 30 mins" in lower and "under-30-minutes" not in tags:
            tags.append("under-30-minutes")
        if "< 60 mins" in lower and "under-60-minutes" not in tags:
            tags.append("under-60-minutes")
        if "freezer" in lower and "freezer-friendly" not in tags:
            tags.append("freezer-friendly")

    if total and total <= 30 and "under-30-minutes" not in tags:
        tags.append("under-30-minutes")
    if total and total <= 60 and "under-60-minutes" not in tags:
        tags.append("under-60-minutes")
    dietary = [t for t in tags if t in DIETARY_KEYWORDS]

    qty_list  = _parse_r_vector(row.get("RecipeIngredientQuantities", ""))
    part_list = _parse_r_vector(row.get("RecipeIngredientParts", ""))
    if part_list:
        lines = ["Ingredients:"]
        for i, part in enumerate(part_list):
            qty = qty_list[i] if i < len(qty_list) else ""
            lines.append(f"- {qty} {part}".strip("- ").rjust(2, "- "))
        ing_text = "\n".join(lines) if len(lines) > 1 else ""
    else:
        ing_text = ""

    steps = _parse_r_vector(row.get("RecipeInstructions", ""))
    meth_text = ("Instructions:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))) if steps else ""

    tips_text = _tips_text(
        description  = row.get("Description"),
        category     = cuisine,
        rating       = row.get("AggregatedRating"),
        rating_count = row.get("ReviewCount"),
        protein_g    = row.get("ProteinContent"),
        fat_g        = row.get("FatContent"),
        carbs_g      = row.get("CarbohydrateContent"),
    )

    if not any([ing_text, meth_text, tips_text]):
        return []

    records: List[ChunkRecord] = []
    chunk_idx = 0
    for section, text in [("ingredients", ing_text), ("method", meth_text), ("tips", tips_text)]:
        if not text:
            continue
        records.append(ChunkRecord(
            recipe_id       = recipe_id,
            title           = title,
            section         = section,
            text            = text,
            cuisine         = cuisine,
            tags            = tags,
            dietary         = dietary,
            prep_time_mins  = prep,
            cook_time_mins  = cook,
            total_time_mins = total,
            servings        = servings,
            calories_kcal   = calories,
            chunk_index     = chunk_idx,
            char_count      = len(text),
        ))
        chunk_idx += 1
    return records


_ADAPTERS = {
    "shengtao": (_row_shengtao, "Shengtao/recipe"),
    "foodcom":  (_row_foodcom,  "AkashPS11/recipes_data_food.com"),
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_hf_chunks(
    max_recipes: int = 10_000,
    dataset_key: str = DEFAULT_DATASET,
    verbose: bool = True,
) -> List[ChunkRecord]:
    """
    Load up to `max_recipes` rows from a HuggingFace recipe dataset
    and convert each row to ChunkRecord objects.

    Parameters
    ----------
    max_recipes  : recipes to load. 0 or None = load entire dataset.
                   Shengtao/recipe has 32,722 rows total.
    dataset_key  : "shengtao" (default) or "foodcom"
    verbose      : show progress.

    Returns
    -------
    List[ChunkRecord]
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install with:  pip install datasets"
        )

    if dataset_key not in _ADAPTERS:
        raise ValueError(f"Unknown dataset key {dataset_key!r}. Choose from: {list(_ADAPTERS)}")

    row_fn, hf_name = _ADAPTERS[dataset_key]
    slice_str = f"train[:{max_recipes}]" if max_recipes else "train"

    if verbose:
        limit_str = f"{max_recipes:,}" if max_recipes else "all"
        print(f"[HF Loader] Dataset : {hf_name}")
        print(f"[HF Loader] Loading : up to {limit_str} recipes …")

    ds = load_dataset(hf_name, split=slice_str)

    if verbose:
        print(f"[HF Loader] Rows    : {len(ds):,} — converting to chunks …")

    all_chunks: List[ChunkRecord] = []
    errors = 0
    batch_report = max(len(ds) // 20, 1)   # report ~20 times

    for i, row in enumerate(ds):
        try:
            chunks = row_fn(row, i)
            all_chunks.extend(chunks)
        except Exception as e:
            errors += 1
            if verbose and errors <= 5:
                print(f"\n[WARN] Row {i} skipped: {e}")

        if verbose and (i + 1) % batch_report == 0:
            pct = (i + 1) / len(ds) * 100
            filled = (i + 1) * 30 // len(ds)
            bar = "█" * filled + "░" * (30 - filled)
            unique = len({c.recipe_id for c in all_chunks})
            print(f"\r  [{bar}] {i+1:>6}/{len(ds)}  ({pct:.0f}%)  "
                  f"{unique} recipes  {len(all_chunks)} chunks",
                  end="", flush=True)

    if verbose:
        bar = "█" * 30
        n = len(ds)
        unique = len({c.recipe_id for c in all_chunks})
        print(f"\r  [{bar}] {n:>6}/{n}  (100%)  "
              f"{unique} recipes  {len(all_chunks)} chunks", flush=True)
        if errors:
            print(f"  [WARN] {errors} rows skipped.")
        print(f"[HF Loader] Done — {len(all_chunks)} chunks "
              f"from {unique} recipes.")

    return all_chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",     type=int, default=5)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(_ADAPTERS))
    args = parser.parse_args()

    chunks = load_hf_chunks(max_recipes=args.max, dataset_key=args.dataset)
    for c in chunks[:6]:
        print(f"\n[{c.section:<12}] {c.title!r}")
        print(f"  cuisine={c.cuisine}  tags={c.tags[:3]}  "
              f"cal={c.calories_kcal}  time={c.total_time_mins}min")
        print(f"  text: {c.text[:120]!r}")
