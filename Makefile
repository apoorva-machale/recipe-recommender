# Recipe Recommender — Makefile
#
# Usage:
#   make setup         — create venv + install all dependencies
#   make db-up         — start PostgreSQL + pgvector via docker compose
#   make db-down       — stop and remove the PostgreSQL container
#   make generate-pdfs — generate all 10,000 recipe PDFs
#   make run           — full pipeline + interactive chatbot (10k PDFs)
#   make run-legacy    — legacy 50-recipe TXT pipeline
#   make run-pipeline  — pipeline only (no interactive chatbot)
#   make lint          — run ruff linter on src/ and data/
#   make check         — run embedding sanity guard
#   make clean         — remove __pycache__ and .pyc files
#   make clean-results — remove benchmark results and embedding backup
#   make help          — show this help

.PHONY: setup db-up db-down generate-pdfs run run-legacy run-pipeline \
        lint check clean clean-results help

PYTHON     := python3
VENV       := venv
PIP        := $(VENV)/bin/pip
VENV_PY    := $(VENV)/bin/python
RUFF       := $(VENV)/bin/ruff
SRC_DIRS   := src data

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	@echo "── Creating virtual environment ──"
	$(PYTHON) -m venv $(VENV)
	@echo "── Installing dependencies ──"
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@echo "── Setup complete. Activate with: source venv/bin/activate ──"

# ── Database ──────────────────────────────────────────────────────────────────

db-up:
	@echo "── Starting PostgreSQL + pgvector (docker compose) ──"
	docker compose up -d
	@echo "── Waiting for PostgreSQL to be ready ──"
	@sleep 2
	@docker compose exec db psql -U recipes -c "SELECT version();" recipes 2>/dev/null \
		&& echo "── DB ready ──" \
		|| echo "── DB not ready yet — run: docker compose logs db ──"

db-down:
	@echo "── Stopping PostgreSQL container ──"
	docker compose down

# ── Data generation ───────────────────────────────────────────────────────────

generate-pdfs:
	@echo "── Generating 10,000 recipe PDFs in data/recipe_pdfs/ ──"
	$(VENV_PY) data/generate_pdfs.py --count 10000

generate-recipes:
	@echo "── Generating 50 TXT recipes in data/recipes/ ──"
	$(VENV_PY) data/generate_recipes.py

# ── Pipeline ──────────────────────────────────────────────────────────────────

run:
	@echo "── Running full pipeline + interactive chatbot (10k PDFs) ──"
	$(VENV_PY) run.py

run-legacy:
	@echo "── Running legacy 50-recipe TXT pipeline ──"
	$(VENV_PY) main.py

run-pipeline:
	@echo "── Running pipeline phases only (no chatbot) ──"
	$(VENV_PY) -c "import sys; sys.path.insert(0,''); from src.pipeline import main; main(interactive=False)"

# ── Quality checks ────────────────────────────────────────────────────────────

lint:
	@echo "── Running ruff linter ──"
	@$(VENV)/bin/ruff check $(SRC_DIRS) run.py main.py || \
		(echo "── ruff not installed — run: make setup ──" && exit 1)

check:
	@echo "── Running embedding sanity guard ──"
	$(VENV_PY) -c "\
from src.embedder import get_model, assert_real_embedder; \
model = get_model(); \
assert_real_embedder(model); \
print('All checks passed.')"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	@echo "── Removing __pycache__ and .pyc files ──"
	find . -type d -name __pycache__ -not -path "./${VENV}/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"            -not -path "./${VENV}/*" -delete 2>/dev/null || true
	@echo "── Clean done ──"

clean-results:
	@echo "── Removing results/benchmark_results.json and results/embeddings.npy ──"
	rm -f results/benchmark_results.json results/embeddings.npy
	@echo "── Results cleaned ──"

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | \
		grep -v '^\.' | \
		awk -F: '{print "  make " $$1}' | \
		sort
	@echo ""
	@echo "See README.md for full documentation."
