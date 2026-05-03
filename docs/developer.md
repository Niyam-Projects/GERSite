# Developer Guide

This guide covers setting up a local development environment for the GERSite
building conflation pipeline.

## Prerequisites

Install the following tools before cloning:

- **[uv](https://docs.astral.sh/uv/)** — Python package manager (`winget install astral-sh.uv`)
- **[just](https://just.systems/man/en/)** — task runner (`winget install Casey.Just`)
- **[DuckDB CLI](https://duckdb.org/docs/installation/)** *(optional)* — for ad-hoc SQL inspection

## Install

```powershell
git clone https://github.com/Niyam-Projects/GERSite.git
cd GERSite
just install        # uv sync --extra gers
```

For dev-only dependencies (pytest, flake8, pylint):

```powershell
just install-dev    # uv sync
```

## Running the Pipeline

Run all four flows end-to-end for an AOI:

```powershell
just run miami_dade
```

Or run individual flows:

| Command | Flow |
|---------|------|
| `just ingest <aoi>` | Flow 1 — ingest Bronze (Overture, FEMA, NSI) |
| `just bridge <aoi>` | Flow 2 — generate Silver bridge files |
| `just gold <aoi>` | Flow 3 — produce Gold GeoParquet |
| `just tiles <aoi>` | Flow 4 — generate PMTiles |

Available AOIs: `saipan`, `guam`, `puerto_rico`, `miami_dade`.

## Interactive Notebooks (Marimo)

Each flow is also a [Marimo](https://marimo.io/) notebook for interactive exploration:

```powershell
just nb-ingest      # flows/ingest_sources.py
just nb-bridge      # flows/generate_bridges.py
just nb-gold        # flows/produce_gold_layer.py
just nb-tiles       # flows/generate_tiles.py
```

## Tests

```powershell
just test           # uv run pytest tests/ -v
```

The test suite currently covers `lib/occupancy.py` via `tests/test_occupancy.py`.

## Linting

```powershell
just lint           # flake8 + pylint over flows/ lib/ tests/
```

Pylint and flake8 settings are configured in `pyproject.toml` under
`[tool.pylint.*]` and `[tool.flake8]`.

## Configuration

- **`config.gers.yaml`** — all pipeline settings (AOI bboxes, storage paths,
  DuckDB memory/threads, confidence thresholds, H3 resolution, attribution).
- **`aoi/`** — GeoJSON polygons for each study area.
- **`lib/`** — shared Python helpers imported by all flows.

## Building the Site

```powershell
just site-build     # cd site && npm run build
just site-dev       # hot-reload dev server at http://localhost:5173
```
