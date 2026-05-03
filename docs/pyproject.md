# pyproject.toml

`pyproject.toml` is the single configuration file for the GERSite Python project.
It covers package metadata, dependencies, and tool settings for `uv`, `flake8`,
and `pylint`.

## `[build-system]`

Uses `setuptools` as the build backend. Required by `uv` for editable installs.

## `[project]`

Standard PEP 621 metadata: name (`gersite`), description, Python version
constraints (`>=3.10`), and the runtime dependency list.

**Core runtime dependencies:**

| Package | Purpose |
|---------|---------|
| `duckdb` | SQL engine for all Bronze/Silver/Gold queries |
| `freestiler` | PMTiles generation from DuckDB queries |
| `geopandas` | GeoDataFrame I/O (FEMA GDB, NSI GPKG) |
| `h3` | H3 cell partitioning for Gold layer output |
| `marimo` | Interactive notebook UI for each flow |
| `prefect` | Flow orchestration and task logging |
| `pyarrow` | Parquet I/O and metadata writing |

## `[project.urls]`

Points to the `Niyam-Projects/GERSite` GitHub repository for documentation,
source, and issue tracking.

## `[tool.uv]`

Declares dev-only dependencies (not installed in production):
`pytest`, `flake8`, `pylint`.

Install everything with `uv sync`; install dev deps only with `uv sync` (no extras).

## `[tool.flake8]`

Line length: 90. Ignores `E203` (whitespace before `:`), `E251` (unexpected
spaces around keyword), and `W503` (line break before binary operator) to
align with Black's formatting style.

## `[tool.pylint.*]`

Configured in `[tool.pylint.format]` (line length 90), `[tool.pylint."messages control"]`
(disables noisy false-positive codes common in data science code), and
`[tool.pylint.basic]` (allows short variable names like `i`, `j`, `x`, `y`).

## Usage

```powershell
# Install all deps (runtime + dev)
uv sync

# Add a new runtime dependency
uv add <package>

# Upgrade all deps
uv sync --upgrade
```
