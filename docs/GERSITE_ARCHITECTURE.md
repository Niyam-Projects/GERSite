# GERSite — Architecture & Design Document

**Project Name:** GERSite  
**Naming:** GERS = Overture Maps Foundation Global Entity Reference System (the building ID prefix)  
**Goal:** Produce a unified, high-confidence US building footprint dataset by conflating Overture, FEMA USA Structures, and the USACE National Structure Inventory (NSI).

---

## 1. System Overview

GERSite ingests four data sources, links them without duplicating geometry (the "Linker-Merger" pattern), then emits a single Gold-layer GeoParquet partitioned by H3 cell. The Overture-to-OSM bridge file is staged for future activation but is not yet wired into the active flows.

```
                    ┌──────────────────────────────────────────┐
                    │              Source Inputs               │
                    │  Overture S3   FEMA ArcGIS   NSI USACE   │
                    └──────────┬───────────┬──────────┬────────┘
                               │           │          │
                   ┌───────────▼───────────▼──────────▼────────┐
                   │    Flow 1: Ingest Sources (Bronze Layer)    │
                   │  Per-AOI GeoParquet written to ~/data/gers/ │
                   └───────────────────────┬────────────────────┘
                                           │
                   ┌───────────────────────▼────────────────────┐
                   │  Flow 2: Generate Bridges (Silver Layer)    │
                   │  FEMA IoU bridge  •  NSI point-in-polygon   │
                   │  nsi_unmatched.parquet for risk review      │
                   └───────────────────────┬────────────────────┘
                                           │
                   ┌───────────────────────▼────────────────────┐
                   │   Flow 3: Produce Gold Layer                │
                   │  FULL OUTER JOIN → geometry resolution      │
                   │  NSI enrichment  →  confidence scoring      │
                   │  H3-partitioned GeoParquet                  │
                   └────────────────────────────────────────────┘
```

---

## 2. Technical Stack

| Component | Choice | Notes |
|---|---|---|
| Package management | `uv` (Astral) | `uv sync --extra gers` |
| Orchestration | Prefect 3.0+ | Local workers; ECS-ready |
| Compute engine | DuckDB ≥ 1.4 | `spatial` + `httpfs` extensions |
| Storage format | GeoParquet | H3-partitioned (res 4/7) |
| Flow files | Marimo notebooks | Dual-mode: notebook + CLI |
| Local data root | `~/data/gers/` | Swap to S3 prefix for cloud |

---

## 3. Repository Structure

```
.
├── flows/
│   ├── ingest_sources.py       # Flow 1 — Bronze ingestion
│   ├── generate_bridges.py     # Flow 2 — Silver bridge files
│   └── produce_gold_layer.py   # Flow 3 — Gold merge & scoring
├── lib/
│   ├── duckdb_helpers.py       # Connection factory, StorageConfig, AOI helpers
│   ├── spatial_utils.py        # IoU SQL, H3 partitioning, geometry resolution
│   └── scoring.py              # Confidence scoring, NSI risk flag
├── aoi/
│   ├── saipan.geojson          # Saipan/CNMI study area
│   ├── guam.geojson            # Guam study area
│   ├── puerto_rico.geojson     # Puerto Rico study area
│   └── miami_dade.geojson      # Miami-Dade County, FL study area
├── config.gers.yaml            # Central config (paths, thresholds, API URLs)
├── docs/
│   ├── GERSITE_ARCHITECTURE.md # ← this file
│   └── legacy_catalog.md       # OpenPOIs inheritance catalog
└── src/openpois/               # Inherited OpenPOIs library (untouched)
```

---

## 4. Data Sources

| Source | License | Scope | Access |
|---|---|---|---|
| [Overture Maps](https://overturemaps.org) — Buildings | ODbL 1.0 | Global | Public S3 (`overturemaps-us-west-2`) |
| [Overture-OSM Bridge](https://docs.overturemaps.org/attribution/) | ODbL 1.0 | Global | Bundled in Overture monthly release |
| [FEMA USA Structures](https://www.fema.gov/flood-maps/products-tools/products/building-data) | CC BY 4.0 | US | ArcGIS REST API |
| [NSI — National Structure Inventory](https://www.hec.usace.army.mil/confluence/nsi) | Public Domain | US | USACE HEC API |

### Overture-OSM Bridge (staged, not active)
Overture publishes a monthly bridge file that cross-references every Overture building ID against its contributing OpenStreetMap way ID(s). This file is downloaded and staged in Flow 1 but is **not yet joined** in Flows 2–3. It is reserved for a future enrichment step that will add `osm_id` and `osm_type` columns to the Gold layer. The `osm_id` / `osm_type` columns exist in the Gold schema today but are always `null`.

---

## 5. Study Area AOIs

| Key | Label | BBox (lon/lat WGS84) | Rationale |
|---|---|---|---|
| `saipan` | Saipan, CNMI | 145.65–145.88E, 15.06–15.32N | Small island; fast end-to-end test |
| `guam` | Guam | 144.56–145.00E, 13.18–13.70N | Slightly larger island; similar coverage check |
| `puerto_rico` | Puerto Rico | 67.30–65.22W, 17.87–18.52N | Medium US territory; tests negative-longitude path |
| `miami_dade` | Miami-Dade County, FL | 80.88–80.10W, 25.13–25.98N | Dense US metro; realistic scale and FEMA/NSI density |

---

## 6. Pipeline Data Flow

### Phase I — Bronze (Flow 1: `ingest_sources.py`)

Mirrors external sources to versioned local (or S3) paths under `bronze/`.

| Task | Input | Output |
|---|---|---|
| `ingest-overture-buildings` | Overture S3 (`theme=buildings/type=building`) | `bronze/overture/buildings/{aoi}/buildings.parquet` |
| `stage-overture-osm-bridge` | Overture S3 bridge file | `bronze/overture/osm_bridge.parquet` (staged only) |
| `ingest-fema-structures` | FEMA ArcGIS REST API (paginated) | `bronze/fema/structures/{aoi}/structures.parquet` |
| `ingest-nsi-structures` | USACE NSI REST API | `bronze/nsi/structures/{aoi}/structures.parquet` |

**Overture ingestion detail:** Uses the Overture `bbox` struct for S3-side predicate pushdown before an exact `ST_Intersects` polygon filter. DuckDB spatial auto-decodes GeoParquet WKB geometry — no `ST_GeomFromWKB()` wrapper needed.

**FEMA pagination:** ArcGIS REST API max 2,000 features/page; paginated via `resultOffset`.

### Phase II — Silver (Flow 2: `generate_bridges.py`)

Produces tabular Parquet bridge files that map IDs across sources **without copying geometry**.

#### FEMA-Base Bridge (`fema_bridge.parquet`)
Spatial IoU join between FEMA and Overture polygons. Uses a two-stage reciprocal best-match algorithm:
1. For each FEMA record, keep the Overture building with the highest IoU (`QUALIFY` per `fema_id`).
2. If multiple FEMA records claim the same Overture building, keep only the one with the highest IoU (`QUALIFY` per `overture_id`).

Result: strict one-to-one mapping. Only pairs with IoU ≥ 0.10 are retained.

Schema: `fema_id`, `overture_id`, `iou`, `fema_centroid_wkt`

#### NSI-Base Bridge (`nsi_bridge.parquet`)
Point-in-polygon join: NSI structure points against Overture building polygons. If a point falls in multiple overlapping polygons, it is assigned to the **smallest** Overture footprint (avoids assigning to container/campus boundaries).

Schema: `nsi_id`, `overture_id`, `nsi_occtype`, `nsi_val_struct`, `nsi_val_cont`, `nsi_point_wkt`

#### NSI Unmatched (`nsi_unmatched.parquet`)
NSI points with **no** Overture polygon match. These represent structures recorded in NSI that have no known building footprint. Written alongside the bridge file; consumed by Flow 3 to produce the NSI risk review output.

Schema: `nsi_id`, `nsi_occtype`, `nsi_val_struct`, `nsi_val_cont`, `nsi_point_wkt`

### Phase III — Gold (Flow 3: `produce_gold_layer.py`)

#### Task 1: Geometry Resolution
`FULL OUTER JOIN` Overture buildings with FEMA via `fema_bridge`. Geometry priority: **Overture > FEMA**. FEMA records with no bridge entry (IoU < 0.10) appear as `source='fema_only'` additive candidates.

#### Task 2: NSI Enrichment
`LEFT JOIN` against `nsi_bridge`. To prevent fan-out duplicates (multiple NSI points per building), the bridge is pre-aggregated to one row per `overture_id` — the record with the highest `nsi_val_struct` is kept as representative.

Adds columns: `nsi_id`, `nsi_occtype`, `nsi_val_struct`, `nsi_val_cont`, `has_nsi_match`

#### Task 3: Confidence Scoring + Output
Each row receives a `conflation_confidence` score:

| Score | Condition |
|---|---|
| **1.0** (High) | `overture_id` present AND `fema_iou` ≥ 0.80 |
| **0.6** (Medium) | `overture_id` present but no FEMA match (or IoU < 0.80) |
| **0.3** (Low) | `fema_id` only — additive candidate, no Overture footprint |

**NSI Risk Review:** Unmatched NSI points (no Overture/FEMA footprint) are written to `gold/{aoi}/nsi_review/nsi_unmatched.parquet` for manual review. This is a separate output — not a column on Gold layer buildings.

#### Gold Layer Schema

| Column | Type | Notes |
|---|---|---|
| `building_id` | string | `overture_id` or `fema_only_{fema_id}` |
| `overture_id` | string | null for FEMA-only rows |
| `fema_id` | string | null for Overture-only rows |
| `fema_iou` | float | IoU of the FEMA-Overture match |
| `osm_id` | string | **reserved — null** (future: Overture-OSM bridge) |
| `osm_type` | string | **reserved — null** |
| `height` | float | From Overture |
| `num_floors` | integer | From Overture |
| `overture_class` | string | From Overture (`residential`, `commercial`, …) |
| `names` | struct | From Overture |
| `nsi_id` | string | null if no NSI match |
| `nsi_occtype` | string | NSI occupancy type |
| `nsi_val_struct` | float | NSI structural replacement value (USD) |
| `nsi_val_cont` | float | NSI contents replacement value (USD) |
| `has_nsi_match` | boolean | |
| `source` | string | `overture`, `fema_only` |
| `conflation_confidence` | float | 1.0 / 0.6 / 0.3 |
| `geometry` | WKB | WGS84 (EPSG:4326) polygon |

Output partitioning: H3 resolution 4 (~111 km) for directory partition key; rows sorted by H3 resolution 7 (~1.2 km) within each partition.

---

## 7. Architecture Decisions

### Marimo as Prefect Flow Files
Each `flows/*.py` is simultaneously a valid Marimo notebook (interactive, reactive cells) and a runnable Python script. The Prefect `@task` / `@flow` decorators live inside Marimo cells and are registered normally when executed as a script. Notebook mode enables interactive exploration of intermediate results without rerunning the full pipeline.

### Bridge-File Pattern (no geometry duplication)
Following the OpenPOIs pattern: the Silver layer contains only ID cross-walk tables (Parquet), never geometry. This makes the bridge files tiny, inspectable, and reusable across pipeline runs without re-running the expensive spatial joins.

### Reciprocal Best-Match for FEMA Bridge
A naive one-directional QUALIFY (best Overture per FEMA) can leave a single Overture building claimed by multiple FEMA footprints, producing duplicate Gold rows. The two-stage CTE enforces a strict one-to-one mapping from both directions.

### NSI Deduplication at Merge Time
The NSI bridge is one-per-NSI-point, not one-per-building. A building complex can contain many NSI points. To prevent fan-out duplicates on the Gold merge, the bridge is aggregated to one row per `overture_id` immediately before the LEFT JOIN, keeping the highest-value record as representative.

### StorageConfig Path Abstraction
All paths are resolved through `lib/duckdb_helpers.StorageConfig`. Swapping `config.gers.yaml:storage.root` from `~/data/gers` to `s3://your-bucket/gers` is the only change required to run the pipeline cloud-natively against a DuckLake.

---

## 8. Licensing & Attribution

The Gold layer is a derivative work of ODbL-licensed data (Overture, OSM) and must be released under the same license. FEMA structures carry CC BY 4.0. NSI is Public Domain.

**Resultant license:** Open Database License (ODbL) 1.0

Required attribution metadata (written to Gold Parquet file metadata):
- *Overture Maps Foundation* — ODbL 1.0
- *OpenStreetMap contributors* — ODbL 1.0
- *FEMA USA Structures* — CC BY 4.0
- *USACE National Structure Inventory* — Public Domain

---

## 9. Future Work

| Item | Description |
|---|---|
| Activate Overture-OSM bridge | Wire `bronze/overture/osm_bridge.parquet` into Flow 2/3 to populate `osm_id` / `osm_type` |
| OSM land-use enrichment | Spatial join with OSM land-use polygons to flag `infra_context` (Industrial / Healthcare / Critical) |
| FEMA S3 bulk download | Replace ArcGIS REST pagination with FEMA bulk S3 parquet once available |
| Cloud deployment | Deploy Prefect workers as Docker containers on AWS ECS; swap `storage.root` to S3 |
| Vue map adaptation | Adapt `site/` frontend to visualize the Gold building layer (replacing POI map) |
| OpenPOIs cleanup | Remove inherited OpenPOIs code per `docs/legacy_catalog.md` recommendations; unfork repo |

---

## 10. Inherited Code Reference

The repository was forked from [`henryspatialanalysis/openpois`](https://github.com/henryspatialanalysis/openpois). All inherited code under `src/openpois/`, `scripts/`, `tests/`, and `site/` is untouched. See [`docs/legacy_catalog.md`](legacy_catalog.md) for a full catalog of inherited files with keep/leverage/remove recommendations.
