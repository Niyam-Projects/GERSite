# justfile — GERSite & OpenPOIs task runner
# Install just: https://just.systems/man/en/  (winget install Casey.Just)
# Run a recipe:  just <recipe>
# List recipes:  just --list

# Use PowerShell on Windows
set shell := ["powershell", "-NoProfile", "-Command"]

# Default: show available recipes
default:
    @just --list

# ── GERSite: Installation ────────────────────────────────────────────────────

# Install GERSite dependencies (DuckDB, Prefect, Marimo, H3, etc.)
install:
    uv sync --extra gers

# Install dev dependencies only (pytest, flake8, pylint)
install-dev:
    uv sync

# ── GERSite: Run pipeline flows ──────────────────────────────────────────────

# Run all three flows for an AOI  (e.g. just run miami_dade)
run aoi:
    uv run python flows/ingest_sources.py --aoi {{aoi}}
    uv run python flows/generate_bridges.py --aoi {{aoi}}
    uv run python flows/produce_gold_layer.py --aoi {{aoi}}

# Flow 1 only — ingest Bronze sources  (e.g. just ingest miami_dade)
ingest aoi:
    uv run python flows/ingest_sources.py --aoi {{aoi}}

# Flow 2 only — generate Silver bridge files  (e.g. just bridge miami_dade)
bridge aoi:
    uv run python flows/generate_bridges.py --aoi {{aoi}}

# Flow 3 only — produce Gold layer  (e.g. just gold miami_dade)
gold aoi:
    uv run python flows/produce_gold_layer.py --aoi {{aoi}}

# ── GERSite: Test AOIs ───────────────────────────────────────────────────────

# Full pipeline — Miami-Dade County, FL (dense US metro test)
run-miami:
    just run miami_dade

# Full pipeline — Puerto Rico (medium US territory test)
run-pr:
    just run puerto_rico

# Full pipeline — Saipan/CNMI (smallest AOI; fast smoke test)
run-saipan:
    just run saipan

# Full pipeline — Guam
run-guam:
    just run guam

# ── GERSite: Interactive notebooks ──────────────────────────────────────────

# Open Flow 1 as a Marimo notebook
nb-ingest:
    uv run marimo edit flows/ingest_sources.py

# Open Flow 2 as a Marimo notebook
nb-bridge:
    uv run marimo edit flows/generate_bridges.py

# Open Flow 3 as a Marimo notebook
nb-gold:
    uv run marimo edit flows/produce_gold_layer.py

# ── OpenPOIs (legacy) ────────────────────────────────────────────────────────

# Run the unit test suite
test:
    uv run pytest tests/ -v

# Lint source code and tests
lint:
    uv run flake8 src/ scripts/ tests/
    uv run pylint src/openpois/

# Build the Vue site for production
site-build:
    cd site && npm run build

# Serve the Vue site locally with hot reload (http://localhost:5173)
site-dev:
    cd site && npm run dev

# Full site preview with Sphinx docs (http://localhost:4173)
site-preview:
    uv run python scripts/build_taxonomy.py
    uv run sphinx-build -b html docs docs/_build/html -q
    cd site && npm run build
    Copy-Item -Recurse -Force docs/_build/html site/dist/docs
    uv run python -m http.server 4173 --directory site/dist

# ── Conda (legacy OpenPOIs env) ──────────────────────────────────────────────

# Create conda environment from environment.yml
conda-create:
    conda env create -f environment.yml

# Export current conda environment to environment.yml
conda-export:
    conda env export > environment.yml
