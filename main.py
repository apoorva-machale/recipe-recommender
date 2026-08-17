"""
Single entry point — run this file to do everything:
  1. Generate recipe text files (skipped if already present)
  2. Chunk all recipes
  3. Embed + build FAISS index (skipped if cached)
  4. Run n_probe and filter benchmarks
  5. Drop into the interactive chatbot

Usage:
    python3 main.py
"""

import os
import sys

# Make sure imports resolve from the project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RECIPES_DIR = os.path.join(ROOT, "data", "recipes")


def step(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ── Step 1: Generate recipe files if missing ──────────────────
recipe_files = [f for f in os.listdir(RECIPES_DIR) if f.endswith(".txt")] if os.path.isdir(RECIPES_DIR) else []

if not recipe_files:
    step("Generating recipe text files...")
    from data.generate_recipes import generate
    generate()
else:
    print(f"[Setup] {len(recipe_files)} recipe files found — skipping generation.")

# ── Step 2–5: Run the full pipeline + interactive chatbot ─────
from src.pipeline import main
main(interactive=True)
