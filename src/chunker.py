"""
Task 1 — Chunking Strategy
==========================

Q1 — Chunk Design
-----------------
Unit of a chunk: ONE SECTION of ONE RECIPE (ingredients block, method block, or tips block).

Rationale:
  A cosine similarity search works by comparing the semantic centroid of a
  chunk's embedding against the query embedding.  If we merged all three
  sections into one embedding, the centroid would drift toward the average of
  ingredients + steps + tips — diluting the signal for any single axis.

  User queries hit two distinct semantic axes simultaneously:
    • Ingredient axis  → "leftover chicken and lemon"
    • Constraint axis  → "low-carb", "under 30 minutes"

  By keeping sections separate we can retrieve the *ingredients chunk* for
  ingredient-match queries and the *metadata-filtered index* for constraint
  queries, then fuse the ranked lists.  A merged embedding would force a
  trade-off between the two signals rather than serving both.

  One-step-per-chunk (sub-chunking the method) was considered but rejected:
  individual cooking steps lack the ingredient context needed for similarity
  to work — "Flip carefully and cook another 2 minutes" is meaningless alone.

Q2 — Metadata Schema
--------------------
Each chunk carries:
  recipe_id     str   — unique identifier (filename stem)
  title         str   — human-readable recipe name
  section       str   — "ingredients" | "method" | "tips"
  cuisine       str   — e.g. "Italian", "Thai"
  tags          list  — ["keto", "gluten-free", "under-30-minutes", ...]
  dietary       list  — subset: ["vegan", "vegetarian", "keto"]
  prep_time_mins int  — prep time only
  cook_time_mins int  — cook time only
  total_time_mins int — prep + cook
  servings      int
  calories_kcal int
  chunk_index   int   — position within recipe (0=ingredients, 1=method, 2=tips)
  char_count    int   — length in characters of this chunk's text

Q3 — Overlap Decision
---------------------
For the METHOD section of long recipes (>600 chars) we add a 150-character
overlap at section boundaries.  Rationale:
  • Overlap is useful when a sentence straddles two logical segments (e.g.
    "After the sauce thickens [←end of step 4], add the pasta [→step 5]").
  • 150 chars ≈ 1 medium sentence — enough context to prevent orphaned
    references without doubling embedding count.
  • We do NOT apply overlap to ingredients or tips: ingredients are an
    enumeration (order-independent), and tips are self-contained bullets.
  • We do NOT further split the method section into one-step-per-chunk (see
    Q1 above) — the entire method stays one chunk unless it exceeds
    MAX_METHOD_CHARS (1 200 chars), in which case we split at step boundaries
    with 150-char overlap.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    from pypdf import PdfReader as _PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

# ── constants ────────────────────────────────────────────────────────────────
MAX_METHOD_CHARS = 1_200   # split method sections longer than this
OVERLAP_CHARS    = 150     # overlap window for long method splits

SECTION_HEADERS = {
    "ingredients": re.compile(r"^Ingredients\s*$", re.IGNORECASE | re.MULTILINE),
    "method":      re.compile(r"^Method\s*$",      re.IGNORECASE | re.MULTILINE),
    "tips":        re.compile(r"^Tips\s*$",        re.IGNORECASE | re.MULTILINE),
}

META_LINE = re.compile(
    r"Prep Time:\s*(\d+)\s*mins\s*\|\s*Cook Time:\s*(\d+)\s*mins\s*\|\s*Servings:\s*(\d+)\s*\|\s*Calories:\s*(\d+)"
)
TAGS_LINE = re.compile(r"^Tags:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
TITLE_LINE = re.compile(r"^Title:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
CUISINE_LINE = re.compile(r"^Cuisine:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

DIETARY_KEYWORDS = {"vegan", "vegetarian", "keto", "gluten-free", "paleo", "halal", "kosher"}


# ── data model ───────────────────────────────────────────────────────────────
@dataclass
class ChunkRecord:
    """
    One embeddable unit — a single section of a single recipe.
    The 'text' field is what gets embedded; everything else is metadata.
    """
    recipe_id:       str
    title:           str
    section:         str          # "ingredients" | "method" | "tips"
    text:            str          # raw text sent to the encoder
    cuisine:         str
    tags:            List[str]    = field(default_factory=list)
    dietary:         List[str]    = field(default_factory=list)
    prep_time_mins:  int          = 0
    cook_time_mins:  int          = 0
    total_time_mins: int          = 0
    servings:        int          = 0
    calories_kcal:   int          = 0
    chunk_index:     int          = 0   # 0=ingredients, 1=method[0], 2=method[1]…
    char_count:      int          = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── helpers ───────────────────────────────────────────────────────────────────
def _extract_meta(raw: str) -> dict:
    """Parse the header lines of a recipe file into a metadata dict."""
    meta: dict = {
        "title": "", "cuisine": "", "tags": [], "dietary": [],
        "prep_time_mins": 0, "cook_time_mins": 0, "servings": 0, "calories_kcal": 0,
    }
    if m := TITLE_LINE.search(raw):
        meta["title"] = m.group(1).strip()
    if m := CUISINE_LINE.search(raw):
        meta["cuisine"] = m.group(1).strip()
    if m := META_LINE.search(raw):
        meta["prep_time_mins"] = int(m.group(1))
        meta["cook_time_mins"] = int(m.group(2))
        meta["servings"]       = int(m.group(3))
        meta["calories_kcal"]  = int(m.group(4))
    if m := TAGS_LINE.search(raw):
        tags = [t.strip().lower() for t in m.group(1).split(",")]
        meta["tags"] = tags
        meta["dietary"] = [t for t in tags if t in DIETARY_KEYWORDS]
    return meta


def _split_sections(raw: str) -> dict[str, str]:
    """
    Locate each section header and extract the text block beneath it.
    Returns a dict keyed by section name.
    """
    positions: dict[str, int] = {}
    for name, pattern in SECTION_HEADERS.items():
        if m := pattern.search(raw):
            positions[name] = m.end()

    if not positions:
        return {}

    sorted_sections = sorted(positions.items(), key=lambda x: x[1])
    sections: dict[str, str] = {}

    for i, (name, start) in enumerate(sorted_sections):
        end = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(raw)
        # Strip the dashed separator line that follows headers
        block = raw[start:end].lstrip("\n").lstrip("-").strip()
        sections[name] = block

    return sections


def _split_long_method(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Split a method block that exceeds max_chars at numbered-step boundaries.
    Each split carries `overlap` characters of trailing context from the
    previous split to avoid orphaned references (Q3).
    """
    # Split at numbered steps like "1. " or "Step 1:"
    step_pattern = re.compile(r"(?=\d+\.\s|\bStep\s+\d+)", re.IGNORECASE)
    steps = [s.strip() for s in step_pattern.split(text) if s.strip()]

    chunks: List[str] = []
    current = ""
    for step in steps:
        if len(current) + len(step) <= max_chars:
            current = (current + " " + step).strip()
        else:
            if current:
                chunks.append(current)
            # Carry overlap from end of previous chunk
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + " " + step).strip()
    if current:
        chunks.append(current)

    return chunks if chunks else [text]


# ── public API ────────────────────────────────────────────────────────────────
def chunk_recipe_file(filepath: str) -> List[ChunkRecord]:
    """
    Parse one recipe text file and return a list of ChunkRecord objects,
    one per section (with long method blocks split further).
    """
    recipe_id = os.path.splitext(os.path.basename(filepath))[0]
    with open(filepath, "r", encoding="utf-8") as fh:
        raw = fh.read()

    meta = _extract_meta(raw)
    sections = _split_sections(raw)

    records: List[ChunkRecord] = []
    chunk_idx = 0

    section_order = ["ingredients", "method", "tips"]
    for section_name in section_order:
        if section_name not in sections:
            continue

        section_text = sections[section_name]

        if section_name == "method" and len(section_text) > MAX_METHOD_CHARS:
            sub_chunks = _split_long_method(section_text, MAX_METHOD_CHARS, OVERLAP_CHARS)
        else:
            sub_chunks = [section_text]

        for sub in sub_chunks:
            rec = ChunkRecord(
                recipe_id       = recipe_id,
                title           = meta["title"],
                section         = section_name,
                text            = sub,
                cuisine         = meta["cuisine"],
                tags            = meta["tags"],
                dietary         = meta["dietary"],
                prep_time_mins  = meta["prep_time_mins"],
                cook_time_mins  = meta["cook_time_mins"],
                total_time_mins = meta["prep_time_mins"] + meta["cook_time_mins"],
                servings        = meta["servings"],
                calories_kcal   = meta["calories_kcal"],
                chunk_index     = chunk_idx,
                char_count      = len(sub),
            )
            records.append(rec)
            chunk_idx += 1

    return records


def chunk_directory(directory: str) -> List[ChunkRecord]:
    """Chunk all .txt recipe files in a directory."""
    all_chunks: List[ChunkRecord] = []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))
    for fname in files:
        path = os.path.join(directory, fname)
        try:
            chunks = chunk_recipe_file(path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
    return all_chunks


# ── PDF support ───────────────────────────────────────────────────────────────

def _extract_text_from_pdf(filepath: str) -> str:
    """Extract plain text from a PDF using pypdf."""
    if not _PYPDF_AVAILABLE:
        raise ImportError(
            "pypdf is required for PDF support.  "
            "Install it with:  pip install pypdf"
        )
    reader = _PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def chunk_pdf_file(filepath: str) -> List[ChunkRecord]:
    """
    Parse one recipe PDF and return ChunkRecord objects.
    The PDF must follow the same structured text format as the .txt recipes
    (Title / Cuisine / Prep-Cook-Servings-Calories / Tags /
     Ingredients / Method / Tips sections).
    """
    recipe_id = os.path.splitext(os.path.basename(filepath))[0]
    raw = _extract_text_from_pdf(filepath)

    meta = _extract_meta(raw)
    sections = _split_sections(raw)

    records: List[ChunkRecord] = []
    chunk_idx = 0

    section_order = ["ingredients", "method", "tips"]
    for section_name in section_order:
        if section_name not in sections:
            continue

        section_text = sections[section_name]

        if section_name == "method" and len(section_text) > MAX_METHOD_CHARS:
            sub_chunks = _split_long_method(section_text, MAX_METHOD_CHARS, OVERLAP_CHARS)
        else:
            sub_chunks = [section_text]

        for sub in sub_chunks:
            rec = ChunkRecord(
                recipe_id       = recipe_id,
                title           = meta["title"],
                section         = section_name,
                text            = sub,
                cuisine         = meta["cuisine"],
                tags            = meta["tags"],
                dietary         = meta["dietary"],
                prep_time_mins  = meta["prep_time_mins"],
                cook_time_mins  = meta["cook_time_mins"],
                total_time_mins = meta["prep_time_mins"] + meta["cook_time_mins"],
                servings        = meta["servings"],
                calories_kcal   = meta["calories_kcal"],
                chunk_index     = chunk_idx,
                char_count      = len(sub),
            )
            records.append(rec)
            chunk_idx += 1

    return records


def chunk_pdf_directory(directory: str, verbose: bool = True) -> List[ChunkRecord]:
    """
    Chunk all .pdf recipe files in a directory.
    Prints a progress bar for large corpora.
    """
    all_chunks: List[ChunkRecord] = []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".pdf"))
    total = len(files)
    errors = 0

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        try:
            chunks = chunk_pdf_file(path)
            all_chunks.extend(chunks)
        except Exception as e:
            errors += 1
            if verbose and errors <= 5:
                print(f"\n[WARN] Skipping {fname}: {e}")

        if verbose and (i + 1) % 500 == 0:
            pct = (i + 1) / total * 100
            bar = "█" * ((i + 1) * 30 // total) + "░" * (30 - (i + 1) * 30 // total)
            print(f"\r  [{bar}] {i+1:>6}/{total}  ({pct:.0f}%)", end="", flush=True)

    if verbose:
        print(f"\r  [{'█'*30}] {total:>6}/{total}  (100%)", flush=True)
        if errors:
            print(f"  [WARN] {errors} PDFs skipped due to parse errors.")

    return all_chunks
