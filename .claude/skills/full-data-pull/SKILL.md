---
name: full-data-pull
description: Use when the user wants to refresh the three independent POI snapshots (OSM, Overture, Foursquare) and rate the OSM snapshot for conflation. Triggers: "refresh all snapshots", "do a new data pull", "download new OSM/Overture/Foursquare", "monthly data refresh", "pull the latest POI data". Does NOT include conflation or S3 upload — those live in conflate-snapshots.
---

# Full data pull

Downloads the three snapshot sources (50 US states + DC + PR) and applies the rating model to OSM so conflation can run.

## Prerequisites

- conda env `openpois` active.
- For Foursquare: `FSQ_PORTAL_TOKEN` env var set.
- For OSM: `osmium` in env bin (resolved automatically via `Path(sys.executable).parent / "osmium"`).
- Boundary cache at `directories.boundary` (auto-downloads on first use).
- A fitted model exists for the OSM rating step (see [skills/model-history-pipeline](../model-history-pipeline/SKILL.md)).

## Steps

1. **Bump versions in `config.yaml`** — sources release on independent cadences, don't force them to match:
   ```yaml
   versions:
     snapshot_osm: "YYYYMMDD"
     snapshot_overture: "YYYYMMDD"
     snapshot_foursquare: "YYYYMMDD"
   ```
   See [docs/data-versioning.md](../../docs/data-versioning.md).

2. **Run the three downloads** (independent — order doesn't matter, can run in parallel):

   ```bash
   python scripts/osm_snapshot/download.py     # ~11 GB US PBF + PR PBF → osm_snapshot.parquet
   python scripts/overture/download.py         # DuckDB over S3           → overture_snapshot.parquet
   python scripts/foursquare/download.py       # PyIceberg catalog        → foursquare_snapshot.parquet
   ```
   Per-source details, auth, and schema quirks are in [docs/data-sources.md](../../docs/data-sources.md).

3. **Apply the rating model to OSM** → `osm_snapshot_rated.parquet`:
   ```bash
   python scripts/osm_snapshot/apply_model.py
   ```
   Uses `osm_data.apply_model.model_stub` to pick the model family.

4. **Optional schema snapshot** — produces small CSV snippets for spec review:
   ```bash
   python scripts/snapshots/load_samples.py
   ```

## Verification

Hand off to [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md). Baseline totals (as of 2026-04-17):
- OSM: ~7.78M POIs
- Overture: ~13.05M POIs (jumped from ~7.23M after widening `download.overture.taxonomy_allowlist` to include `services_and_business` + `lifestyle_services` sub-branches)
- Foursquare: ~8.32M POIs

Flag >5% drops — Foursquare in particular has had silent country-filter regressions (PR alpha-2 code quirk).

## Next

- To publish, continue with [skills/conflate-snapshots](../conflate-snapshots/SKILL.md).
- To update the frontend after publishing, continue with [skills/update-site](../update-site/SKILL.md).
