# GERSite — US Building Conflation Pipeline

**Maintained by [NiyamIT, Inc.](https://niyamit.com) · Author: Troy Schmidt**

**GERSite** conflates [Overture Maps](https://overturemaps.org) building footprints,
[FEMA USA Structures](https://www.fema.gov/flood-maps/products-tools/products/building-data),
and the [USACE National Structure Inventory (NSI)](https://www.hec.usace.army.mil/confluence/nsi)
into a single confidence-scored Gold layer GeoParquet, partitioned by H3 cell.

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: ODbL](https://img.shields.io/badge/Data-ODbL%20v1.0-orange.svg)](https://opendatacommons.org/licenses/odbl/1-0/)

📐 **Architecture doc:** [`docs/GERSITE_ARCHITECTURE.md`](docs/GERSITE_ARCHITECTURE.md)

---

## GERSite Quick Start (Local)

### Prerequisites

Install [just](https://just.systems/man/en/) (the command runner) and [uv](https://docs.astral.sh/uv/) (package manager):

```powershell
# Windows — install both with winget
winget install Casey.Just
winget install astral-sh.uv
```

Then install GERSite dependencies:

```powershell
just install     # uv sync --extra gers (DuckDB, Prefect, Marimo, H3, …)
just --list      # see all available recipes
```

### Configuration

All settings live in [`config.gers.yaml`](config.gers.yaml).
By default, data is written to `~/data/gers/` (bronze / silver / gold subdirectories).
Change `storage.root` in the config to redirect output, including to an S3 prefix.

### Run the full pipeline for one AOI

```powershell
just run miami_dade     # all three flows for Miami-Dade
just run puerto_rico    # all three flows for Puerto Rico
just run saipan         # smallest AOI — fast smoke test
just run guam
```

Or run individual flows:

```powershell
just ingest miami_dade  # Flow 1 — Bronze ingestion
just bridge miami_dade  # Flow 2 — Silver bridge files
just gold   miami_dade  # Flow 3 — Gold merge & scoring
```

### Supported AOIs

| Key | Area | Shortcut | Notes |
|---|---|---|---|
| `saipan` | Saipan, CNMI | `just run-saipan` | Smallest; fast smoke test |
| `guam` | Guam | `just run-guam` | Small island |
| `puerto_rico` | Puerto Rico | `just run-pr` | Medium US territory |
| `miami_dade` | Miami-Dade County, FL | `just run-miami` | Dense US metro; realistic scale |

### Testing with Miami-Dade and Puerto Rico

These two AOIs are the recommended local test targets — Miami-Dade for a dense US
mainland area and Puerto Rico for a medium-sized US territory:

```powershell
# Miami-Dade County, FL
just run-miami

# Puerto Rico
just run-pr
```

Expected output paths (`~/data/gers/`):
```
bronze/overture/buildings/miami_dade/buildings.parquet
bronze/fema/structures/miami_dade/structures.parquet
bronze/nsi/structures/miami_dade/structures.parquet
silver/bridges/miami_dade/fema_bridge.parquet
silver/bridges/miami_dade/nsi_bridge.parquet
silver/bridges/miami_dade/nsi_unmatched.parquet
gold/buildings/miami_dade/          ← H3-partitioned GeoParquet
gold/nsi_review/miami_dade/         ← unmatched NSI points for review
```

### Interactive notebook mode (Marimo)

Each flow file is also a [Marimo](https://marimo.io) notebook. Open any flow
interactively to explore intermediate results, inspect data, and run individual
tasks without executing the full pipeline:

```powershell
just nb-ingest   # flows/ingest_sources.py
just nb-bridge   # flows/generate_bridges.py
just nb-gold     # flows/produce_gold_layer.py
```

### Read the Gold layer

```python
import pyarrow.dataset as ds
import geopandas as gpd

# Point at the H3-partitioned output directory for one AOI
gold = ds.dataset("~/data/gers/gold/buildings/miami_dade", format="parquet")
print(f"{gold.count_rows():,} buildings")

# Load into GeoPandas (full AOI or filtered)
gdf = gpd.read_parquet("~/data/gers/gold/buildings/miami_dade")
print(gdf[["building_id", "source", "conflation_confidence", "geometry"]].head())
```

### Inspect confidence scores

```python
import pandas as pd
import geopandas as gpd

gdf = gpd.read_parquet("~/data/gers/gold/buildings/miami_dade")
print(gdf["conflation_confidence"].value_counts())
# 1.0 — Overture + FEMA, IoU >= 0.80   (high agreement)
# 0.6 — Overture only                   (no FEMA match)
# 0.3 — FEMA only                       (additive candidate)
```

---

## Provenance & Credits

**GERSite** was created by **Troy Schmidt** at **[NiyamIT, Inc.](https://niyamit.com)**.

This project was inspired by the open-source [OpenPOIs](https://github.com/henryspatialanalysis/openpois)
pipeline built by [Nathaniel Henry](https://github.com/henryspatialanalysis) at
[Henry Spatial Analysis](https://github.com/henryspatialanalysis). The OpenPOIs framework
introduced a rigorous, confidence-scored approach to conflating multiple open geospatial
datasets into a unified Gold layer — and that approach directly inspired GERSite's adaptation
of the same methodology for US building footprints, fusing Overture Maps, FEMA USA Structures,
and the USACE National Structure Inventory. We are grateful to Nathaniel Henry and Henry Spatial
Analysis for making that foundational work available under an open-source license.

### Data Source Attribution

| Dataset | Provider | License |
|---|---|---|
| [Overture Maps Buildings](https://overturemaps.org) | Overture Maps Foundation | ODbL v1.0 |
| [USA Structures](https://www.fema.gov/flood-maps/products-tools/products/building-data) | FEMA | Public Domain |
| [National Structure Inventory (NSI)](https://www.hec.usace.army.mil/confluence/nsi) | USACE | Public Domain |

### Licensing

GERSite is dual-licensed:

- **Code** — [MIT License](LICENSE). You can use, modify, and redistribute the
  pipeline and library freely.
- **Data** — [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  Output datasets are derivative works of Overture Maps and inherit ODbL terms. Any public use
  must attribute GERSite and the
  [Overture Maps Foundation](https://docs.overturemaps.org/attribution/).
  Derivative databases must be released under the same license.

### Contact

Bug reports, feature requests, and contributions are welcome via
[GitHub issues](https://github.com/Niyam-Projects/GERSite/issues).
