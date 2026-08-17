"""
Generates 50 recipe TXT files in data/recipes/ — the legacy benchmark corpus.

Same structured format the chunker parses (see data/generate_pdfs.py):
  Title / Cuisine / Prep-Cook-Servings-Calories line / Tags
  Ingredients section
  Method section
  Tips section

11 recipes are hand-crafted with fixed titles matching the ground-truth
titles hardcoded in src/retrieval.py (TEST_QUERIES) and src/filters.py
(FILTERED_TEST_QUERIES) — without them the legacy recall@k benchmark has
nothing real to retrieve. The remaining recipes are synthetic filler built
with the same recipe builder as data/generate_pdfs.py, padding the corpus
to 50 recipes.

Usage
-----
    python3 data/generate_recipes.py
"""

from __future__ import annotations

import os

from data.generate_pdfs import _build_recipe

OUT_DIR = os.path.join(os.path.dirname(__file__), "recipes")
TOTAL_RECIPES = 50

# ── hand-crafted ground-truth recipes ─────────────────────────────────────────
# Titles match TEST_QUERIES (src/retrieval.py) and FILTERED_TEST_QUERIES
# (src/filters.py) so recall@k has real recipes to find.
REQUIRED_RECIPES = [
    {
        "recipe_id": "recipe_gt_00",
        "title":     "Quick Leftover Chicken Lemon Pasta",
        "cuisine":   "Italian-American",
        "prep": 10, "cook": 15, "servings": 2, "calories": 480,
        "tags":    ["quick", "under-30-minutes"],
        "dietary": [],
        "ingredients": [
            "250 g cooked leftover chicken, shredded",
            "200 g spaghetti or linguine",
            "2 tbsp olive oil",
            "3 cloves garlic, thinly sliced",
            "1 lemon, zested and juiced",
            "30 g parmesan, grated",
            "Salt and black pepper to taste",
            "Fresh parsley, chopped",
        ],
        "method": [
            "Bring a large pot of salted water to the boil and cook pasta until al dente.",
            "Meanwhile, heat olive oil in a large skillet over medium heat.",
            "Add garlic and cook, stirring, for 1 minute until fragrant.",
            "Add shredded chicken and warm through, about 3 minutes.",
            "Stir in lemon zest and juice, then toss in the drained pasta.",
            "Finish with parmesan and parsley, season to taste, and serve immediately.",
        ],
        "tips": [
            "Save a cup of pasta water to loosen the sauce if it looks dry.",
            "Rotisserie chicken works great here for an even quicker meal.",
            "Add a pinch of chilli flakes for extra warmth.",
        ],
    },
    {
        "recipe_id": "recipe_gt_01",
        "title":     "Lemon Herb Roasted Chicken",
        "cuisine":   "American",
        "prep": 15, "cook": 45, "servings": 4, "calories": 520,
        "tags":    ["comfort-food", "high-protein"],
        "dietary": [],
        "ingredients": [
            "1.5 kg whole chicken",
            "1 lemon, halved",
            "4 cloves garlic",
            "2 tbsp olive oil",
            "1 tbsp fresh rosemary, chopped",
            "1 tbsp fresh thyme, chopped",
            "Salt and black pepper to taste",
        ],
        "method": [
            "Preheat oven to 200C (400F).",
            "Pat the chicken dry and rub all over with olive oil, salt, and pepper.",
            "Stuff the cavity with the lemon halves and garlic cloves.",
            "Sprinkle rosemary and thyme over the skin.",
            "Roast for 45-60 minutes until the internal temperature reaches 75C.",
            "Rest for 10 minutes before carving.",
        ],
        "tips": [
            "Basting halfway through keeps the skin extra crisp.",
            "Save the pan juices for a quick gravy.",
            "Leftovers keep for up to 4 days refrigerated.",
        ],
    },
    {
        "recipe_id": "recipe_gt_02",
        "title":     "Chicken Caesar Salad",
        "cuisine":   "American",
        "prep": 15, "cook": 10, "servings": 2, "calories": 430,
        "tags":    ["light", "under-30-minutes"],
        "dietary": [],
        "ingredients": [
            "2 chicken breasts",
            "1 large romaine lettuce, chopped",
            "40 g parmesan, shaved",
            "60 g croutons",
            "3 tbsp Caesar dressing",
            "1 tbsp olive oil",
            "Salt and black pepper to taste",
        ],
        "method": [
            "Heat olive oil in a skillet over medium-high heat.",
            "Season chicken breasts and cook 4-5 minutes per side until golden and cooked through.",
            "Rest for 5 minutes, then slice.",
            "Toss romaine with Caesar dressing in a large bowl.",
            "Top with sliced chicken, croutons, and shaved parmesan.",
        ],
        "tips": [
            "Grill the chicken instead for a smokier flavour.",
            "Add anchovies for a more traditional Caesar dressing.",
            "Swap croutons for toasted chickpeas for extra crunch.",
        ],
    },
    {
        "recipe_id": "recipe_gt_03",
        "title":     "Keto Chocolate Mousse",
        "cuisine":   "American",
        "prep": 15, "cook": 0, "servings": 4, "calories": 220,
        "tags":    ["keto", "low-carb", "gluten-free"],
        "dietary": ["keto"],
        "ingredients": [
            "250 ml heavy cream, chilled",
            "80 g dark chocolate (85%+), melted",
            "2 tbsp powdered erythritol",
            "1 tsp vanilla extract",
            "Pinch of salt",
        ],
        "method": [
            "Whip the heavy cream to soft peaks.",
            "Fold in the melted chocolate, erythritol, vanilla, and salt.",
            "Continue folding until fully combined and airy.",
            "Divide into 4 ramekins and chill for at least 1 hour before serving.",
        ],
        "tips": [
            "Use chocolate with 85% cocoa or higher to keep net carbs low.",
            "Top with a few raspberries for a touch of tartness.",
            "Keeps refrigerated for up to 3 days.",
        ],
    },
    {
        "recipe_id": "recipe_gt_04",
        "title":     "Chocolate Avocado Brownies",
        "cuisine":   "American",
        "prep": 15, "cook": 25, "servings": 9, "calories": 280,
        "tags":    ["vegetarian", "gluten-free"],
        "dietary": ["vegetarian"],
        "ingredients": [
            "2 ripe avocados, mashed",
            "2 eggs",
            "150 g sugar",
            "60 g cocoa powder",
            "80 g almond flour",
            "1 tsp baking powder",
            "1 tsp vanilla extract",
            "60 g dark chocolate chips",
        ],
        "method": [
            "Preheat oven to 175C (350F) and line a baking tin.",
            "Blend mashed avocado, eggs, sugar, and vanilla until smooth.",
            "Whisk in cocoa powder, almond flour, and baking powder.",
            "Fold in chocolate chips and pour batter into the tin.",
            "Bake for 22-25 minutes until a toothpick comes out mostly clean.",
            "Cool completely before slicing into squares.",
        ],
        "tips": [
            "You won't taste the avocado — it just adds fudgy moisture.",
            "Slightly underbaking keeps the centre gooey.",
            "Freezes well for up to 2 months.",
        ],
    },
    {
        "recipe_id": "recipe_gt_05",
        "title":     "Banana Oat Cookies",
        "cuisine":   "American",
        "prep": 10, "cook": 15, "servings": 12, "calories": 180,
        "tags":    ["vegetarian", "low-carb", "high-fiber", "under-30-minutes"],
        "dietary": ["vegetarian"],
        "ingredients": [
            "2 ripe bananas, mashed",
            "180 g rolled oats",
            "40 g raisins or chopped dates",
            "1 tsp cinnamon",
            "1 tsp vanilla extract",
            "Pinch of salt",
        ],
        "method": [
            "Preheat oven to 175C (350F) and line a baking tray.",
            "Mix mashed bananas with oats, cinnamon, vanilla, and salt.",
            "Fold in raisins.",
            "Drop spoonfuls onto the tray and flatten slightly.",
            "Bake for 12-15 minutes until lightly golden.",
            "Cool on the tray for 5 minutes before transferring to a rack.",
        ],
        "tips": [
            "Use very ripe, spotty bananas for the most natural sweetness.",
            "Add a handful of walnuts for extra crunch.",
            "Store in an airtight container for up to 5 days.",
        ],
    },
    {
        "recipe_id": "recipe_gt_06",
        "title":     "Vegan Black Bean Tacos",
        "cuisine":   "Mexican",
        "prep": 10, "cook": 15, "servings": 4, "calories": 380,
        "tags":    ["vegan", "quick", "under-30-minutes"],
        "dietary": ["vegan", "vegetarian"],
        "ingredients": [
            "2 cans (400 g each) black beans, drained and rinsed",
            "8 small corn tortillas",
            "1 tsp ground cumin",
            "1 tsp smoked paprika",
            "1 avocado, sliced",
            "1 lime, cut into wedges",
            "Fresh coriander, chopped",
            "Pickled red onion, to serve",
        ],
        "method": [
            "Heat olive oil in a skillet over medium heat.",
            "Add black beans, cumin, and smoked paprika. Cook, mashing lightly, for 8-10 minutes.",
            "Warm the tortillas in a dry pan or over an open flame.",
            "Fill tortillas with the black bean mixture.",
            "Top with avocado, coriander, and pickled onion.",
            "Serve with lime wedges.",
        ],
        "tips": [
            "Mash half the beans for a creamier filling that holds together.",
            "Add hot sauce or jalapeños for extra heat.",
            "Leftovers keep refrigerated for up to 3 days.",
        ],
    },
    {
        "recipe_id": "recipe_gt_07",
        "title":     "Thai Green Curry with Tofu",
        "cuisine":   "Thai",
        "prep": 15, "cook": 20, "servings": 4, "calories": 420,
        "tags":    ["vegan", "comfort-food", "meal-prep"],
        "dietary": ["vegan", "vegetarian"],
        "ingredients": [
            "400 g firm tofu, cubed",
            "1 can (400 ml) coconut milk",
            "3 tbsp Thai green curry paste",
            "150 g green beans, trimmed",
            "1 red bell pepper, sliced",
            "1 tbsp soy sauce or tamari",
            "1 tsp brown sugar",
            "Fresh basil leaves",
            "Jasmine rice, to serve",
        ],
        "method": [
            "Heat a wok over high heat until smoking.",
            "Fry tofu cubes until golden on all sides, then set aside.",
            "Add curry paste and cook, stirring, for 1-2 minutes until fragrant.",
            "Pour in coconut milk and bring to a gentle simmer.",
            "Add green beans and bell pepper, simmer for 8 minutes.",
            "Stir in soy sauce, sugar, and tofu. Simmer 3 more minutes.",
            "Finish with basil and serve over jasmine rice.",
        ],
        "tips": [
            "Press the tofu for 15 minutes before frying for a firmer texture.",
            "Adjust curry paste to taste — start with less if unsure of heat level.",
            "Freezes well for up to 3 months.",
        ],
    },
    {
        "recipe_id": "recipe_gt_08",
        "title":     "Crispy Tofu Buddha Bowl",
        "cuisine":   "Vietnamese",
        "prep": 15, "cook": 15, "servings": 2, "calories": 450,
        "tags":    ["vegan", "meal-prep", "balanced", "high-fiber"],
        "dietary": ["vegan", "vegetarian"],
        "ingredients": [
            "300 g firm tofu, cubed",
            "1 tbsp cornstarch",
            "150 g quinoa, cooked",
            "1 cup shredded purple cabbage",
            "1 carrot, julienned",
            "1 avocado, sliced",
            "2 tbsp tahini",
            "1 tbsp soy sauce",
            "1 tsp sesame seeds",
        ],
        "method": [
            "Toss tofu cubes in cornstarch and pan-fry until crispy on all sides.",
            "Cook quinoa according to package directions.",
            "Whisk tahini with soy sauce and a splash of water to make the dressing.",
            "Arrange quinoa, cabbage, carrot, and avocado in bowls.",
            "Top with crispy tofu, drizzle with tahini dressing, and sprinkle sesame seeds.",
        ],
        "tips": [
            "Prep the vegetables and dressing ahead for a fast weeknight assembly.",
            "Swap quinoa for brown rice if that's what you have on hand.",
            "Keeps well in the fridge for meal prep, dressing stored separately.",
        ],
    },
    {
        "recipe_id": "recipe_gt_09",
        "title":     "Salmon with Miso Glazed Bok Choy",
        "cuisine":   "Japanese",
        "prep": 10, "cook": 20, "servings": 2, "calories": 480,
        "tags":    ["gluten-free", "high-protein", "quick"],
        "dietary": [],
        "ingredients": [
            "2 salmon fillets",
            "4 baby bok choy, halved",
            "2 tbsp white miso paste",
            "1 tbsp mirin",
            "1 tbsp soy sauce or tamari",
            "1 tsp sesame oil",
            "1 tsp sesame seeds",
        ],
        "method": [
            "Preheat oven to 200C (400F) and line a baking tray.",
            "Whisk miso, mirin, soy sauce, and sesame oil together.",
            "Brush half the glaze over the salmon fillets, reserve the rest.",
            "Place salmon and bok choy on the tray and roast for 12-15 minutes.",
            "Brush bok choy with the remaining glaze halfway through roasting.",
            "Sprinkle with sesame seeds before serving.",
        ],
        "tips": [
            "Salmon is done when it flakes easily with a fork.",
            "Swap bok choy for broccolini if that's what's available.",
            "White miso is milder than red — use less red miso if substituting.",
        ],
    },
    {
        "recipe_id": "recipe_gt_10",
        "title":     "Shrimp Stir Fry with Vegetables",
        "cuisine":   "Chinese",
        "prep": 10, "cook": 15, "servings": 2, "calories": 350,
        "tags":    ["gluten-free", "high-protein", "quick", "under-30-minutes"],
        "dietary": [],
        "ingredients": [
            "300 g shrimp, peeled and deveined",
            "1 red bell pepper, sliced",
            "1 cup broccoli florets",
            "1 carrot, sliced thin",
            "2 cloves garlic, minced",
            "1 tbsp fresh ginger, grated",
            "2 tbsp tamari or soy sauce",
            "1 tbsp sesame oil",
            "1 tsp cornstarch mixed with 2 tbsp water",
        ],
        "method": [
            "Heat a wok over high heat until smoking.",
            "Add sesame oil, then garlic and ginger, stir for 30 seconds.",
            "Add shrimp and cook until just pink, about 2 minutes, then remove.",
            "Add bell pepper, broccoli, and carrot, stir-fry for 4-5 minutes.",
            "Return shrimp to the wok, add tamari and cornstarch slurry.",
            "Toss until the sauce thickens and everything is glazed, about 1 minute.",
        ],
        "tips": [
            "Don't overcook the shrimp — they turn rubbery quickly.",
            "Prep all vegetables before starting since stir-frying moves fast.",
            "Serve over rice or noodles for a complete meal.",
        ],
    },
]


# ── writer ────────────────────────────────────────────────────────────────────
def _write_txt(recipe: dict, path: str) -> None:
    r = recipe
    lines = [
        f"Title: {r['title']}",
        f"Cuisine: {r['cuisine']}",
        f"Prep Time: {r['prep']} mins | Cook Time: {r['cook']} mins | "
        f"Servings: {r['servings']} | Calories: {r['calories']} kcal",
        f"Tags: {', '.join(r['tags'])}",
        "",
        "Ingredients",
        "-----------",
    ]
    lines += [f"- {ing}" for ing in r["ingredients"]]
    lines += ["", "Method", "------"]
    lines += [f"{i}. {step}" for i, step in enumerate(r["method"], 1)]
    lines += ["", "Tips", "----"]
    lines += [f"* {tip}" for tip in r["tips"]]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────
def generate(out_dir: str = OUT_DIR, total: int = TOTAL_RECIPES) -> None:
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".txt")])
    if existing >= total:
        print(f"[generate_recipes] {existing} recipes already exist in {out_dir}/ — skipping.")
        return

    print(f"[generate_recipes] Generating {total} recipe TXT files in {out_dir}/")

    for recipe in REQUIRED_RECIPES:
        _write_txt(recipe, os.path.join(out_dir, f"{recipe['recipe_id']}.txt"))

    used_titles = {r["title"] for r in REQUIRED_RECIPES}
    filler_needed = total - len(REQUIRED_RECIPES)
    index = 500_000  # offset well clear of the 10k PDF corpus's 0..9999 range

    written = 0
    while written < filler_needed:
        recipe = _build_recipe(f"recipe_filler_{written:03d}", index)
        index += 1
        if recipe["title"] in used_titles:
            continue
        used_titles.add(recipe["title"])
        _write_txt(recipe, os.path.join(out_dir, f"{recipe['recipe_id']}.txt"))
        written += 1

    print(f"[generate_recipes] Done — {total} recipes written to {out_dir}/ "
          f"({len(REQUIRED_RECIPES)} ground-truth + {filler_needed} filler)")


if __name__ == "__main__":
    generate()
