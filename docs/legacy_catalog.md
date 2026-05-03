# Legacy Catalog — OpenPOIs Codebase

This file documents every file inherited from the `henryspatialanalysis/openpois` fork.
Each entry includes its purpose and a recommendation for the GERSite building pipeline:

- **Keep** — directly useful for building conflation; adapt in-place
- **Leverage** — contains reusable logic; port key parts to `lib/` or `flows/`
- **Remove** — POI-specific; no building-pipeline relevance; delete when legacy code is retired

---

## Root Files

| File | Purpose | Recommendation |
|---|---|---|
| `pyproject.toml` | `openpois` package metadata + linting config | **Keep** — extended for GERSite; retain openpois section until legacy removed |
| `config.yaml` | POI pipeline version/path/download/conflation config | **Leverage** — pattern reused in `config.gers.yaml`; POI-specific values will be removed |
| `environment.yml` | Conda environment pinning all legacy dependencies | **Keep** — preserve for anyone running legacy POI pipeline; GERSite uses `uv` |
| `Makefile` | Build/lint/site targets for openpois | **Remove** — POI targets; update/replace with GERSite-specific targets later |
| `README.md` | OpenPOIs public README | **Remove** — replace with GERSite building pipeline README |
| `CITATION.cff` | Academic citation for OpenPOIs | **Remove** — replace with GERSite citation |
| `LICENSE` | MIT license for code | **Keep** — code license is unchanged |

---

## `src/openpois/` — Library Source

### `src/openpois/conflation/`

| File | Purpose | Recommendation |
|---|---|---|
| `match.py` | BallTree centroid-based POI candidate matching and scoring | **Leverage** — the chunking pattern and greedy assignment logic are reusable; IoU replaces centroid distance for buildings |
| `merge.py` | Merges matched/unmatched OSM+Overture POIs into unified GeoDataFrame | **Leverage** — `_pick_geometries()` (geometry type rank preference) is directly portable to `lib/spatial_utils.py` |
| `chunking.py` | KD-tree spatial chunking driver for memory-bounded processing | **Leverage** — H3/geohash chunking approach; adapt for building-scale DuckDB chunking |
| `taxonomy.py` | POI taxonomy crosswalk (OSM tags ↔ Overture categories) | **Remove** — POI-specific; buildings don't use a taxonomy crosswalk |
| `dedup_overture.py` | Intra-Overture POI deduplication | **Remove** — POI-specific deduplication logic; buildings use IoU bridging instead |
| `data/` | CSV crosswalk files (OSM ↔ Overture taxonomy) | **Remove** — POI taxonomy data |
| `__init__.py` | Package exports | **Remove** (with module) |

### `src/openpois/io/`

| File | Purpose | Recommendation |
|---|---|---|
| `overture.py` | Downloads Overture Places snapshot for US+PR via DuckDB S3 scan | **Leverage** — the per-part S3 scan pattern, bbox prefilter, and ST_Within polygon filter are directly reusable for `theme=buildings` in `flows/ingest_sources.py` |
| `boundary.py` | Downloads + dissolves US+PR Census boundary polygon | **Leverage** — US+PR boundary logic reused in `lib/spatial_utils.py`; AOI GeoJSON files replace the Census download |
| `geohash_partition.py` | Geohash-partitioned GeoParquet writing | **Leverage** — H3 partitioning in `lib/spatial_utils.py` reuses the write pattern; geohash replaced by H3 |
| `osm_snapshot.py` | Reads/converts filtered OSM PBF to GeoParquet (POI filter) | **Remove** — POI-specific OSM reader; building PBF parsing deferred to deployment phase |
| `osm_history_pbf.py` | Parses OSM full-history PBF for the Bayesian turnover model | **Remove** — turnover model is POI-specific |
| `_osm_poi_handler.py` | Osmium handler for POI extraction | **Remove** — POI-specific handler |
| `credentials.py` | Geofabrik OAuth cookie and Source Cooperative S3 credentials | **Remove** — credentials pattern replaced by `lib/duckdb_helpers.py` StorageConfig |
| `source_coop.py` | Uploads partitioned GeoParquet + PMTiles to Source Cooperative | **Remove** — POI publish target; GERSite will use its own publish flow |
| `pmtiles.py` | Generates PMTiles archives from GeoParquet for web map | **Keep** (later) — tile generation pattern will be adapted for GERSite building map |
| `__init__.py` | Package exports | **Remove** (with module) |

### `src/openpois/models/`

| File | Purpose | Recommendation |
|---|---|---|
| `osm_models.py` | JAX Bayesian turnover models (Constant + RandomByType) | **Remove** — POI-specific; buildings don't use a turnover model |
| `jax_core.py` | JAX NUTS MCMC infrastructure | **Remove** — JAX dependency; POI-specific |
| `model_fitter.py` | Fits turnover model; generates predictions + diagnostics | **Remove** — POI-specific |
| `apply.py` | Applies fitted turnover model to OSM snapshot for per-POI confidence | **Remove** — POI-specific; building confidence uses IoU-based scoring instead |
| `diagnostics.py` | R-hat and ESS diagnostics for MCMC chains | **Remove** — POI-specific |
| `setup.py` | Loads fitted model params from CSV | **Remove** — POI-specific |
| `__init__.py` | Package exports | **Remove** (with module) |

### `src/openpois/osm/`

| File | Purpose | Recommendation |
|---|---|---|
| `format_observations.py` | Formats OSM history records into observations for the turnover model | **Remove** — POI-specific |
| `change_plots.py` | Visualizes OSM edit history | **Remove** — POI-specific |
| `__init__.py` | Package exports | **Remove** (with module) |

### `src/openpois/publish/`

| File | Purpose | Recommendation |
|---|---|---|
| `build_readme.py` | Generates per-version README for Source Cooperative upload | **Remove** — POI publish logic |
| `templates/` | Jinja2 templates for README generation | **Remove** — POI templates |
| `__init__.py` | Package exports | **Remove** (with module) |

---

## `scripts/` — Pipeline Scripts

| File | Purpose | Recommendation |
|---|---|---|
| `build_taxonomy.py` | Builds POI taxonomy crosswalk CSV from Overture + OSM tags | **Remove** — POI taxonomy |
| `check_taxonomy_sync.py` | CI check: taxonomy CSVs match source data | **Remove** — POI taxonomy |
| `conflation/conflate.py` | Main POI conflation driver (OSM × Overture) | **Leverage** — BallTree+chunking orchestration pattern; building pipeline replaces matching logic with IoU DuckDB joins |
| `conflation/format_for_upload.py` | Partitions conflated POI parquet for Source Cooperative | **Remove** — POI publish |
| `conflation/prepare_pmtiles.py` | Generates PMTiles from conflated POIs | **Remove** (for now) — adapt for buildings later |
| `conflation/summarize.py` | Generates per-label summary CSV for POI diagnostics | **Remove** — POI-specific summary |
| `models/osm_turnover.py` | CLI driver for JAX turnover model fit | **Remove** — POI model |
| `osm_data/download_history.py` | Downloads OSM full-history PBF | **Remove** — POI-specific |
| `osm_data/format_tabular.py` | Converts history PBF to model-ready parquet | **Remove** — POI model prep |
| `osm_data/data_viz.py` | Visualization for OSM history EDA | **Remove** — POI EDA |
| `osm_snapshot/download.py` | Downloads and filters current OSM PBF (POIs) | **Remove** — POI snapshot |
| `osm_snapshot/apply_model.py` | Applies turnover model to OSM snapshot | **Remove** — POI model |
| `osm_snapshot/format_for_upload.py` | Partitions OSM snapshot for publish | **Remove** — POI publish |
| `osm_snapshot/prepare_pmtiles.py` | POI OSM PMTiles generation | **Remove** — POI |
| `overture/download.py` | Downloads Overture Places snapshot (POI) | **Leverage** — calls `src/openpois/io/overture.py`; adapt for buildings |
| `publish/upload_to_source_coop.py` | Uploads all artifacts to Source Cooperative | **Remove** — POI publish |
| `snapshots/load_samples.py` | Loads sample snapshots for testing | **Remove** — POI testing |
| `exploratory/` | EDA notebooks for JAX model, stability curves | **Remove** — POI EDA |

---

## `tests/` — Unit Tests

| File | Purpose | Recommendation |
|---|---|---|
| `test_match.py` | BallTree POI matching tests | **Remove** — POI-specific; IoU matching will have new tests |
| `test_merge.py` | POI merge logic tests | **Remove** — POI-specific |
| `test_chunking.py` | KD-tree chunking tests | **Remove** — POI-specific |
| `test_taxonomy.py` | Taxonomy crosswalk tests | **Remove** — POI-specific |
| `test_taxonomy_sync.py` | Taxonomy CSV sync check | **Remove** — POI-specific |
| `test_dedup_overture.py` | Intra-Overture dedup tests | **Remove** — POI-specific |
| `test_osm_models.py` | JAX model tests | **Remove** — POI-specific |
| `test_constant_lambda_simulation.py` | MCMC simulation tests | **Remove** — POI-specific |
| `test_apply.py` | Model application tests | **Remove** — POI-specific |
| `test_format_observations.py` | OSM history formatting tests | **Remove** — POI-specific |
| `test_osm_history_pbf.py` | OSM PBF parser tests | **Remove** — POI-specific |
| `test_osm_snapshot.py` | OSM snapshot reader tests | **Remove** — POI-specific |
| `test_poi_handler.py` | Osmium POI handler tests | **Remove** — POI-specific |
| `test_overture_download.py` | Overture Places download tests | **Leverage** — DuckDB S3 scan test patterns reusable for buildings download |
| `test_geohash_partition.py` | Geohash partition writer tests | **Leverage** — write/read pattern; adapt for H3 partitioning |
| `test_setup.py` | Package import smoke test | **Remove** — POI package test |
| `__init__.py` | Test package | **Keep** |

---

## `site/` — Vue 3 + Vite Frontend

| Path | Purpose | Recommendation |
|---|---|---|
| `site/` (entire directory) | Interactive map viewer for POI data using MapLibre GL + PMTiles | **Keep** — adapt for building visualization in a future phase; no changes now |

---

## `docs/` — Sphinx Documentation

| File | Purpose | Recommendation |
|---|---|---|
| `docs/conf.py` | Sphinx config (openpois API docs) | **Remove** — POI docs |
| `docs/api.rst` | API reference RST | **Remove** — POI docs |
| `docs/index.rst` | Docs index | **Remove** — POI docs |
| `docs/workflows.rst` | Pipeline workflow docs | **Remove** — POI docs |
| `docs/developer.md` | Developer setup guide | **Leverage** — update for GERSite setup instructions |
| `docs/devcontainer.md` | Dev container setup | **Keep** — container setup may be adapted |
| `docs/pre-commit-config.md` | Pre-commit hook docs | **Keep** — linting config is unchanged |
| `docs/pylint.md` | Pylint config docs | **Keep** — linting config is unchanged |
| `docs/pyproject.md` | pyproject.toml docs | **Keep** — updated as pyproject evolves |
| `docs/vscode.md` | VSCode setup | **Keep** — IDE config |
| `docs/legacy_catalog.md` | **This file** — GERSite migration catalog | **Keep** |

---

## `.github/`

| File | Purpose | Recommendation |
|---|---|---|
| `.github/workflows/deploy-site.yml` | GitHub Actions: deploy Vue site to GitHub Pages | **Keep** — adapt for GERSite building site when ready |

---

## Summary Counts

| Recommendation | Count |
|---|---|
| **Remove** | ~47 files/modules |
| **Leverage** (port logic to `lib/` or `flows/`) | ~10 files |
| **Keep** (unchanged or minimally updated) | ~10 files |

> **Priority order for removal:** Models → POI-specific scripts → POI IO modules → Conflation (taxonomy, dedup, match) → Tests → Sphinx docs
