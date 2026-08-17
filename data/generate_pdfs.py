"""
Generates 10,000 recipe PDF files in data/recipe_pdfs/.

Each PDF follows the exact same structured format the chunker parses:
  Title / Cuisine / Prep-Cook-Servings-Calories line / Tags
  Ingredients section
  Method section
  Tips section

Strategy for 10,000 unique recipes
------------------------------------
  • 200 base dish archetypes × 50 variations each = 10,000 recipes
  • Variations randomise: proteins/vegetables, cuisine region, cooking verb,
    dietary tags, prep/cook times, calorie count, and tip text.
  • Every recipe gets a unique recipe_id so the pipeline deduplicates correctly.

Usage
-----
    python3 data/generate_pdfs.py          # generate all 10,000
    python3 data/generate_pdfs.py --count 100   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from fpdf import FPDF, XPos, YPos

random.seed(42)

# ── vocabulary pools ──────────────────────────────────────────────────────────

PROTEINS = [
    "chicken breast", "chicken thighs", "ground beef", "beef sirloin",
    "salmon fillet", "tuna steak", "shrimp", "prawns", "tofu", "tempeh",
    "chickpeas", "black beans", "lentils", "lamb shoulder", "pork loin",
    "turkey breast", "duck breast", "cod fillet", "eggs", "halloumi",
]

VEGETABLES = [
    "spinach", "kale", "broccoli", "zucchini", "bell peppers", "mushrooms",
    "sweet potato", "butternut squash", "cauliflower", "asparagus",
    "green beans", "bok choy", "eggplant", "cherry tomatoes", "leek",
    "corn", "peas", "artichoke hearts", "beets", "fennel",
]

GRAINS = [
    "basmati rice", "jasmine rice", "quinoa", "pasta", "couscous",
    "farro", "bulgur", "noodles", "polenta", "orzo",
]

AROMATICS = [
    "garlic and ginger", "shallots and thyme", "onion and cumin",
    "lemongrass and lime", "rosemary and lemon", "chilli and coriander",
    "smoked paprika and oregano", "soy and sesame", "miso and mirin",
    "tahini and lemon",
]

CUISINE_REGIONS = [
    "Italian", "Thai", "Mexican", "Indian", "Japanese", "Greek",
    "Middle Eastern", "Chinese", "Korean", "French", "American",
    "Vietnamese", "Ethiopian", "Moroccan", "Spanish", "Brazilian",
    "Lebanese", "Turkish", "Peruvian", "British",
]

DISH_TYPES = [
    ("Stir Fry",       ["quick", "under-30-minutes"]),
    ("Curry",          ["comfort-food", "meal-prep"]),
    ("Salad",          ["light", "under-30-minutes"]),
    ("Soup",           ["meal-prep", "freezer-friendly"]),
    ("Roast",          ["comfort-food", "high-protein"]),
    ("Bowl",           ["meal-prep", "balanced"]),
    ("Pasta",          ["comfort-food"]),
    ("Tacos",          ["quick", "under-30-minutes"]),
    ("Burger",         ["comfort-food"]),
    ("Skillet",        ["quick", "under-30-minutes"]),
    ("Casserole",      ["meal-prep", "freezer-friendly"]),
    ("Frittata",       ["breakfast", "high-protein"]),
    ("Wrap",           ["quick", "meal-prep"]),
    ("Stew",           ["comfort-food", "freezer-friendly"]),
    ("Noodle Dish",    ["quick", "under-30-minutes"]),
    ("Stuffed Pepper", ["meal-prep"]),
    ("Grain Bowl",     ["meal-prep", "high-fiber"]),
    ("Flatbread",      ["quick"]),
    ("Bake",           ["meal-prep"]),
    ("One-Pan Meal",   ["quick", "under-30-minutes"]),
]

DIETARY_PROFILES = [
    (["vegan", "plant-based", "high-fiber"],     ["vegan", "vegetarian"]),
    (["vegetarian", "gluten-free"],               ["vegetarian"]),
    (["keto", "low-carb", "gluten-free"],         ["keto"]),
    (["high-protein", "gluten-free"],             []),
    (["meal-prep", "freezer-friendly"],           []),
    (["vegan", "gluten-free", "high-fiber"],      ["vegan", "vegetarian"]),
    (["vegetarian", "high-protein"],              ["vegetarian"]),
    (["gluten-free", "dairy-free"],               []),
    (["under-30-minutes", "quick"],               []),
    (["keto", "high-protein", "low-carb"],        ["keto"]),
]

COOKING_METHODS = [
    "Heat olive oil in a large skillet over medium-high heat",
    "Preheat oven to 200C (400F) and line a baking tray",
    "Bring a large pot of salted water to the boil",
    "Heat a wok over high heat until smoking",
    "Warm coconut oil in a heavy-bottomed saucepan",
    "Heat a griddle pan until very hot",
    "Place a large Dutch oven over medium heat",
]

TIP_TEMPLATES = [
    "Leftovers keep for up to {days} days refrigerated. Reheat gently.",
    "For extra flavour, marinate the protein for {hours} hours before cooking.",
    "This dish freezes well for up to 3 months in airtight containers.",
    "Add a squeeze of lemon or lime just before serving to brighten the flavour.",
    "Swap {protein} for {alt_protein} for a different flavour profile.",
    "Serve over {grain} or with crusty bread to soak up the sauce.",
    "Make a double batch - the flavour improves overnight.",
    "For a spicier version, add {amount} tsp chilli flakes with the aromatics.",
]


# ── recipe builder ────────────────────────────────────────────────────────────

def _build_recipe(recipe_id: str, index: int) -> dict:
    rng = random.Random(index)

    protein   = rng.choice(PROTEINS)
    vegetable = rng.choice(VEGETABLES)
    grain     = rng.choice(GRAINS)
    aromatic  = rng.choice(AROMATICS)
    cuisine   = rng.choice(CUISINE_REGIONS)
    dish_type, base_tags = rng.choice(DISH_TYPES)
    diet_tags, dietary   = rng.choice(DIETARY_PROFILES)

    tags = list(dict.fromkeys(base_tags + diet_tags))  # deduplicated, ordered

    prep  = rng.choice([5, 10, 15, 20, 25, 30])
    cook  = rng.choice([0, 10, 15, 20, 25, 30, 40, 45, 60, 75, 90])
    servings = rng.choice([2, 2, 4, 4, 4, 6])
    cal   = rng.randint(180, 620) // 10 * 10

    title = f"{cuisine} {protein.title()} {dish_type}"

    # Ingredients
    qty_units = ["g", "g", "tbsp", "tsp", "ml", "cups", "pieces"]
    ingredients = [
        f"{rng.randint(1, 4) * 50 if 'g' in u or 'ml' in u else rng.randint(1, 4)} "
        f"{rng.choice(qty_units)} {item.strip()}"
        for item, u in [
            (protein,   "g"),
            (vegetable, "g"),
            (grain,     "g"),
            (f"{aromatic} paste", "tbsp"),
            ("olive oil or coconut oil", "tbsp"),
            ("salt and black pepper", "tsp"),
        ]
    ]
    # Add 2–4 extra random ingredients
    extras = rng.sample([
        "1 lemon, zested and juiced",
        "2 tbsp soy sauce or tamari",
        "1 tsp smoked paprika",
        "1 can (400 g) chopped tomatoes",
        "200 ml coconut milk",
        "1 tbsp fish sauce",
        "2 tsp curry powder",
        "1 tbsp honey or maple syrup",
        "Fresh herbs to garnish",
        "1 tsp ground cumin",
        "½ cup vegetable or chicken stock",
        "1 tbsp tomato paste",
        "Sesame seeds to garnish",
        "Chilli flakes to taste",
    ], k=rng.randint(2, 4))
    ingredients += extras

    # Method (5–7 steps)
    step_1 = rng.choice(COOKING_METHODS)
    method_steps = [
        f"{step_1}.",
        f"Add {aromatic} and cook, stirring, for 2 minutes until fragrant.",
        f"Add {protein} and cook for {rng.randint(3, 6)} minutes until golden.",
        f"Stir in {vegetable} and cook for a further {rng.randint(2, 5)} minutes.",
        f"Add seasoning and any sauce ingredients. Simmer for {rng.randint(5, 15)} minutes.",
        f"Meanwhile, prepare {grain} according to package directions.",
        f"Serve hot over {grain}, garnished with fresh herbs.",
    ]
    num_steps = rng.randint(5, 7)
    method = method_steps[:num_steps]

    # Tips
    alt_protein = rng.choice([p for p in PROTEINS if p != protein])
    tips = [
        rng.choice(TIP_TEMPLATES).format(
            days=rng.randint(3, 5), hours=rng.randint(1, 24),
            protein=protein, alt_protein=alt_protein,
            grain=grain, amount=rng.randint(1, 2)
        ),
        f"Prep ahead: chop {vegetable} and measure spices the night before to cut cook time.",
        f"Adjust {aromatic} to taste — start with half and add more if needed.",
    ]

    return {
        "recipe_id": recipe_id,
        "title":     title,
        "cuisine":   cuisine,
        "prep":      prep,
        "cook":      cook,
        "servings":  servings,
        "calories":  cal,
        "tags":      tags,
        "dietary":   dietary,
        "ingredients": ingredients,
        "method":    method,
        "tips":      tips,
    }


# ── PDF writer ────────────────────────────────────────────────────────────────

def _write_pdf(recipe: dict, path: str) -> None:
    def _safe(s: str) -> str:
        """Replace non-latin-1 characters so fpdf's core fonts don't choke."""
        return s.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_margins(left=18, top=18, right=18)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def h(text: str, size: int = 12, bold: bool = False) -> None:
        pdf.set_font("Helvetica", style="B" if bold else "", size=size)
        pdf.multi_cell(
            0, 6, text=_safe(text),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    def body(text: str) -> None:
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0, 5, text=_safe(text),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    def blank() -> None:
        pdf.ln(3)

    r = recipe
    h(f"Title: {r['title']}", size=13, bold=True)
    h(f"Cuisine: {r['cuisine']}", size=10)
    h(
        f"Prep Time: {r['prep']} mins | Cook Time: {r['cook']} mins | "
        f"Servings: {r['servings']} | Calories: {r['calories']} kcal",
        size=10,
    )
    h(f"Tags: {', '.join(r['tags'])}", size=10)
    blank()

    h("Ingredients", size=11, bold=True)
    h("-----------", size=10)
    for ing in r["ingredients"]:
        body(f"- {ing}")
    blank()

    h("Method", size=11, bold=True)
    h("------", size=10)
    for i, step in enumerate(r["method"], 1):
        body(f"{i}. {step}")
    blank()

    h("Tips", size=11, bold=True)
    h("----", size=10)
    for tip in r["tips"]:
        body(f"* {tip}")

    pdf.output(path)


# ── main ──────────────────────────────────────────────────────────────────────

def generate_pdfs(out_dir: str, count: int = 10_000) -> None:
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".pdf")])
    if existing >= count:
        print(f"[generate_pdfs] {existing} PDFs already exist in {out_dir}/ — skipping.")
        return

    print(f"[generate_pdfs] Generating {count} recipe PDFs in {out_dir}/")
    print(f"[generate_pdfs] Each PDF has Title / Ingredients / Method / Tips sections.")

    batch_size = 500
    for i in range(count):
        recipe_id = f"recipe_{i:05d}"
        fname     = os.path.join(out_dir, f"{recipe_id}.pdf")

        if os.path.exists(fname):          # resume-safe
            continue

        recipe = _build_recipe(recipe_id, i)
        _write_pdf(recipe, fname)

        if (i + 1) % batch_size == 0 or (i + 1) == count:
            pct = (i + 1) / count * 100
            bar = "█" * ((i + 1) * 30 // count) + "░" * (30 - (i + 1) * 30 // count)
            print(f"\r  [{bar}] {i+1:>6}/{count}  ({pct:.0f}%)", end="", flush=True)

    print(f"\n[generate_pdfs] Done — {count} PDFs written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recipe PDFs")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--out",   type=str,
                        default=os.path.join(os.path.dirname(__file__), "recipe_pdfs"))
    args = parser.parse_args()
    generate_pdfs(args.out, args.count)
