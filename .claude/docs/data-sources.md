# Data sources

Reference for every external data source openpois ingests. For the workflow that orchestrates these, see the skills under [.claude/skills/](../skills/).

## OSM history (Geofabrik full-history PBFs)

**Used by**: the historical modeling pipeline ([skills/model-history-pipeline](../skills/model-history-pipeline/SKILL.md)).

- **URLs**:
  - `download.osm.history_pbf_url` → `https://osm-internal.download.geofabrik.de/north-america/us-internal.osh.pbf`
  - `download.osm.pr_history_pbf_url` → `.../us/puerto-rico-internal.osh.pbf`
- **Auth**: OAuth — any OSM account works. Produce a Netscape-format cookie jar (browser export or Geofabrik's `oauth_cookie_client.py`). Path: `download.osm.history_cookie_file` (default `~/data/openpois/.creds/geofabrik_cookies.txt`).
- **Pipeline**: `osmium tags-filter --omit-referenced` → `osmium time-filter` → pyosmium streams to `osm_versions.parquet` + `osm_changes.parquet`.
- **Entry**: [src/openpois/io/osm_history_pbf.py](../../src/openpois/io/osm_history_pbf.py) (`download_osm_history`).
- **Config**: `download.osm.start_date`, `end_date`, `date_interval_days`, `filter_keys`, `extract_keys`.

## OSM snapshot (Geofabrik standard PBFs)

**Used by**: current-state snapshot (`osm_snapshot.parquet`).

- **URLs**:
  - US: `https://download.geofabrik.de/north-america/us-latest.osm.pbf` (~11 GB, 50 states incl. AK+HI)
  - PR: `https://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf` — **PR is not in the US extract**
- **Auth**: none (public).
- **Pipeline**: `osmium tags-filter` → pyosmium parse → concat US+PR → GeoParquet.
- **Entry**: [src/openpois/io/osm_snapshot.py](../../src/openpois/io/osm_snapshot.py).
- **Quirks**:
  - `osmium` is in the conda env's `bin/` but **not** on shell PATH. Code resolves via `Path(sys.executable).parent / "osmium"`.
  - Geofabrik extracts are pre-cut to admin boundaries → no polygon post-filter needed.

## Overture Maps

**Used by**: current-state Overture snapshot (`overture_snapshot.parquet`).

- **URL**: public S3 at `s3://overturemaps-us-west-2/`.
- **Auth**: none (DuckDB + httpfs queries directly).
- **Pipeline**: two-stage spatial filter — DuckDB `WHERE` clause ORs one disjunct per coarse bbox (predicate pushdown on Overture's `bbox` struct), then GeoPandas `sjoin(predicate='within')` against the exact US+PR polygon.
- **Entry**: [src/openpois/io/overture.py](../../src/openpois/io/overture.py).
- **Schema quirks (as of Feb 2026 schema)**:
  - `taxonomy` is a named STRUCT `{primary, hierarchy[], alternates[]}` — use `taxonomy.hierarchy[1]` **not** `taxonomy[1]`.
  - `brand` is a singular struct, **not** a `brands[]` array.
  - L0 category names: `food_and_drink`, `shopping`, `arts_and_entertainment`, `sports_and_recreation`, `health_care`.
  - Geometry is native DuckDB GEOMETRY — must `LOAD spatial;` and use `ST_X()` / `ST_Y()`.
- **Upcoming migration (~June 2026)**: L0/L1 hierarchy → flat `basic_category`. Crosswalk CSV + `assign_overture_shared_label` will need updating.

## Foursquare OS Places

**Used by**: current-state Foursquare snapshot (`foursquare_snapshot.parquet`).

- **Catalog**: `https://catalog.h3-hub.foursquare.com/iceberg` — PyIceberg `RestCatalog`.
- **Auth**: `FSQ_PORTAL_TOKEN` env var.
- **Params** (all in config.yaml):
  - `warehouse="places"` (required)
  - `namespace="datasets"`
  - `places_table="places_os"`, `categories_table="categories_os"`
- **Pipeline**: row filter `country IN ('US', 'PR') AND date_closed IS NULL` → PyIceberg scan → sjoin against exact US+PR polygon (PyIceberg has no spatial predicates).
- **Entry**: [src/openpois/io/foursquare.py](../../src/openpois/io/foursquare.py).
- **Quirks**:
  - Table is **unpartitioned** (no `dt` column); release date inferred from `last_updated_at` in partition metadata.
  - `fsq_category_ids` arrives as numpy/pyarrow array — use `len(x) == 0`, **not** `if not x:`.
  - PR uses alpha-2 `'PR'`, not `'US'` — silent drop regression pre-2026-04-16 when filter was `'US'`-only.

## Census boundary

**Used by**: all three snapshot downloaders (spatial clipping).

- **URL**: `download.general.boundary.source_url` → `https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip` (1:20M cartographic, 50 states + DC + PR).
- **Auth**: none.
- **Pipeline**: download ZIP → cache under `directories.boundary` (first-use) → dissolve → buffer outward by `coastline_buffer_m` (default 100 m) in EPSG:6933 (equal-area, so buffer accurate across CONUS/AK/HI/PR).
- **Entry**: [src/openpois/io/boundary.py](../../src/openpois/io/boundary.py) (`get_us_pr_boundary`).
- **Returns**: `(boundary_gdf, coarse_bboxes)` — single-row dissolved+buffered polygon (EPSG:4326) plus a list of bboxes for predicate pushdown.
- **Antimeridian**: Aleutians split into two bboxes (Near Islands at +172°E vs. rest of AK at negative longitudes).

## Legacy: Overpass-based OSM history

Still wired up but superseded by the PBF pipeline. Queries Overpass API for element IDs in a bbox, then fetches per-element histories from the OSM API.

- **Config**: `download.osm.history_bbox` (Seattle-scoped; Overpass can't serve US-wide histories).
- **Entry**: [src/openpois/io/osm_history.py](../../src/openpois/io/osm_history.py) (`download_element_histories`).
- **Script**: `scripts/osm_data/download.py`.
- **When to use**: city-scale testing, or if Geofabrik OAuth is unavailable.
