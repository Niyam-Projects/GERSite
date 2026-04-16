# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
make build_env       # Create conda environment from environment.yml
make install_package # Install openpois in editable mode (pip install -e .)
```

The conda environment is named `openpois` and requires Python 3.10+.

## Common Commands

```bash
pytest               # Run tests
make export_env      # Export conda environment to environment.yml after adding dependencies
```

Code style is enforced by Black (format on save in VSCode). Linting via flake8 and pylint, both configured in `pyproject.toml`.

## Architecture

**openpois** models POI (Point of Interest) stability over time using historical OpenStreetMap data. The workflow is:

1. **Download OSM history** — two options depending on scope:
   - **US + Puerto Rico (default, `src/openpois/io/osm_history_pbf.py`)** — downloads Geofabrik full-history PBFs (`us-internal.osh.pbf` + `puerto-rico-internal.osh.pbf`), runs `osmium tags-filter --omit-referenced` then `osmium time-filter`, and streams the result through pyosmium into `osm_versions.parquet` + `osm_changes.parquet`. Requires an OSM-account OAuth cookie jar for Geofabrik's internal server. Entry point: `scripts/osm_data/download_history.py`.
   - **City-scale fallback (`src/openpois/io/osm_history.py`)** — queries the Overpass API for element IDs in a bounding box, then fetches per-element histories from the OSM API. Seattle-scoped by default; Overpass cannot serve US-wide histories. Entry point: `scripts/osm_data/download.py`.
2. **Format observations** (`src/openpois/osm/format_observations.py`) — converts raw OSM version histories into observation records (one row per version) with flags for tag changes and deletions
3. **Model change rates** (`src/openpois/models/`) — fits an empirical Bayes model using PyTorch to estimate per-group POI change rates (λ) as a Poisson process
4. **Visualize stability** (`src/openpois/osm/change_plots.py`) — plots how long POI tags remain unchanged

The **scripts/** directory contains end-to-end pipelines that call library functions using settings from `config.yaml`. They are not part of the installed package and serve as reference implementations.

### Key classes and files

- `EventRate` (`models/event_rate.py`) — wraps a constant or time-varying λ; computes change probabilities via integration
- `ModelFitter` (`models/model_fitter.py`) — fits λ using PyTorch L-BFGS optimizer with optional priors; supports parameter draws for uncertainty
- `pytorch_setup()` / `prepare_data_for_model()` (`models/setup.py`) — initializes torch (GPU/CPU) and prepares filtered, grouped observation data
- `download_osm_history()` (`io/osm_history_pbf.py`) — US+PR history pipeline entry: Geofabrik full-history PBFs → osmium tags-filter (`--omit-referenced`) → osmium time-filter → pyosmium stream → `osm_versions.parquet` + `osm_changes.parquet`. Requires `download.osm.history_cookie_file` to point at a Netscape-format cookie jar with valid Geofabrik OAuth cookies.
- `download_element_histories()` (`io/osm_history.py`) — legacy city-scale entry point (Overpass, `download.osm.history_bbox` config key, Seattle-scoped; Overpass cannot serve US-wide histories)

### Configuration

`config.yaml` holds all shared settings (spatial boundary, date ranges, OSM tag keys, model hyperparameters, output directory paths with versioning). The `config_versioned` package (external dependency) reads this file. Scripts load config at startup; library functions accept parameters directly.

- `.get()` raises `ValueError` for null config values — pass `fail_if_none=False` for optional fields like `release_date: null`

## POI Snapshot Downloads

Three separate utilities download current snapshots covering the 50 US states + DC + Puerto Rico (separate from the historical OSM workflow):

### Spatial boundary (`src/openpois/io/boundary.py`)
- Single source of truth for the US+PR extent used by all three snapshot downloaders
- Downloads the Census 1:20M cartographic state shapefile (`cb_2023_us_state_20m`) on first use; cached under `directories.boundary`
- `get_us_pr_boundary()` returns `(boundary_gdf, coarse_bboxes)` — a single-row dissolved+buffered polygon (EPSG:4326) plus a list of bboxes for predicate pushdown
- Buffering is done in `EPSG:6933` (World Equal-Area Cylindrical) so the `coastline_buffer_m` (default 100 m) is accurate across CONUS / AK / HI / PR. Because `.dissolve()` removes internal state borders, the uniform outward buffer effectively only expands coastline; land-border expansion into CA/MX is negligible.
- `coarse_bboxes` splits the Aleutians at the antimeridian into two bboxes (Near Islands at +172°E vs. rest of AK at negative longitudes)

### OSM (`src/openpois/io/osm_snapshot.py`)
- `download_pbf` / `filter_pbf` / `parse_pbf_to_geodataframe` / `download_osm_snapshot`
- Two Geofabrik extracts: `us-latest.osm.pbf` (~11 GB, 50 states incl. AK+HI) + `puerto-rico-latest.osm.pbf` (PR is NOT in the US extract) → osmium tags-filter → pyosmium parse → concat → GeoParquet
- Geofabrik extracts are pre-cut to admin boundaries, so no polygon post-filter is needed
- `osmium` is in the conda env bin but NOT on shell PATH; code resolves it via `Path(sys.executable).parent / "osmium"`
- Run: `python scripts/osm_snapshot/download.py`

### Overture Maps (`src/openpois/io/overture.py`)
- DuckDB + httpfs + spatial extensions; queries public S3 directly, no auth
- **Two-stage spatial filter:** DuckDB `WHERE` clause ORs one disjunct per coarse bbox (predicate pushdown on Overture's `bbox` struct column), then a GeoPandas `sjoin(predicate='within')` post-filter against the exact US+PR polygon
- `taxonomy` field is a named STRUCT: use `taxonomy.hierarchy[1]` (not `taxonomy[1]`)
- `brand` is a singular struct (not array); geometry is native DuckDB GEOMETRY type requiring `LOAD spatial` and `ST_X()/ST_Y()`
- L0 category names (Feb 2026+): `food_and_drink`, `shopping`, `arts_and_entertainment`, `sports_and_recreation`, `health_care`
- Run: `python scripts/overture/download.py`

### Foursquare OS Places (`src/openpois/io/foursquare.py`)
- PyIceberg `RestCatalog`; requires `warehouse="places"` parameter
- Catalog: `uri=https://catalog.h3-hub.foursquare.com/iceberg`, namespace=`datasets`, tables=`places_os` / `categories_os`
- Table is **unpartitioned** (no `dt` column); release date inferred from `last_updated_at` in partition metadata
- Row filter: `country IN ('US', 'PR') AND date_closed IS NULL` — Foursquare uses ISO alpha-2 codes, so PR must be listed explicitly; PyIceberg has no spatial predicate support, so an exact `sjoin(predicate='within')` post-filter runs after the rows are loaded
- `fsq_category_ids` arrives as numpy/pyarrow array — use `len(x) == 0` not `if not x:`
- Token in `FSQ_PORTAL_TOKEN` env var; run: `python scripts/foursquare/download.py`
