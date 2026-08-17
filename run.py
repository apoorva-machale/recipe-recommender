"""
Single entry point — does everything in one command:
  1. Generate 10,000 recipe PDFs  (skipped if already present)
  2. Chunk all PDFs
  3. Connect to pgvector, set up schema
  4. Embed + ingest chunks (skipped if already ingested)
  5. Run n_probe benchmark
  6. Run metadata filter benchmark
  7. Drop into interactive chatbot

Usage:
    python3 run.py

Requirements:
    • venv activated  (source venv/bin/activate)
    • PostgreSQL + pgvector running  (brew services start postgresql@17)
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PDF_DIR = os.path.join(ROOT, "data", "recipe_pdfs")


def _step(msg: str) -> None:
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


# ── Step 1: generate 10,000 recipe PDFs if missing ────────────────────────────
existing_pdfs = (
    [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if os.path.isdir(PDF_DIR) else []
)

if len(existing_pdfs) < 10_000:
    _step(f"Generating 10,000 recipe PDFs in data/recipe_pdfs/ …")
    from data.generate_pdfs import generate_pdfs
    generate_pdfs(PDF_DIR, count=10_000)
else:
    print(f"[Setup] {len(existing_pdfs)} recipe PDFs found — skipping generation.")

# ── Step 2–7: run full pipeline + interactive chatbot ─────────────────────────
from src.pipeline import main
main(interactive=True)
