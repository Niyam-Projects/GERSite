---
name: conflate-snapshots
description: Use when the user wants to match rated OSM POIs with Overture POIs into a unified dataset, partition it for web consumption, and push to S3. Triggers: "run conflation", "push new conflated data to S3", "bump conflation version", "reconflate with new parameters", "re-upload the partitioned parquet".
---

# Conflate snapshots + publish to S3

Taxonomy-aware matching between rated OSM and Overture, then partition and upload for web consumption.

## Prerequisites

- Rated OSM snapshot (`osm_snapshot_rated.parquet`) at `versions.snapshot_osm` — produced by [skills/full-data-pull](../full-data-pull/SKILL.md) step 3.
- Overture snapshot (`overture_snapshot.parquet`) at `versions.snapshot_overture`.
- AWS credentials configured for the `openpois-public` bucket (region `us-west-2`).

## Steps

1. **Bump `versions.conflation` and `versions.aws`** in `config.yaml`. These typically track together since the upload uses the conflation output.

2. **Review conflation parameters** (`config.yaml` → `conflation`):
   - `min_match_score` (default 0.50) — raises/lowers match acceptance
   - `max_radius_m`, `default_radius_m` — per-label radii come from `match_radii.csv`
   - Component weights: `distance_weight`, `name_weight`, `type_weight`, `identifier_weight`
   - Changing these reshapes match counts — run with `--test` first (Seattle bbox).

3. **Sync taxonomy if crosswalks changed** — see [docs/taxonomy-setup.md](../../docs/taxonomy-setup.md):
   ```bash
   python scripts/build_taxonomy.py   # regenerates site/public/taxonomy.html
   ```

4. **Run conflation** — ~15M POIs, ~16 GB RAM:
   ```bash
   python scripts/conflation/conflate.py            # full run
   python scripts/conflation/conflate.py --test     # Seattle bbox dry run
   ```
   Outputs: `conflated.parquet`, `match_diagnostics.parquet`.

5. **Match-rate sanity check**:
   ```bash
   python scripts/conflation/summarize.py
   ```
   Writes `summary_by_label.csv`.

6. **Partition for web** — geohash-4 partition, geohash-6 sort:
   ```bash
   python scripts/conflation/format_for_upload.py
   ```
   Outputs `conflated_partitioned/` (and OSM-only `osm_snapshot_partitioned/`).

7. **Upload to S3**:
   ```bash
   python scripts/conflation/upload_to_s3.py
   ```
   Pushes to `s3://openpois-public/snapshots/...` under `versions.aws`.

8. **Update latest-URL pointers** in `config.yaml`:
   ```yaml
   upload:
     latest_url_osm:       "https://openpois-public.s3.us-west-2.amazonaws.com/snapshots/osm/YYYYMMDD/osm_snapshot_partitioned/"
     latest_url_conflation: "https://openpois-public.s3.us-west-2.amazonaws.com/snapshots/conflated/YYYYMMDD/conflated_partitioned/"
   ```

## Verification

- `summary_by_label.csv` match rates should resemble the prior run; large drifts mean a parameter or crosswalk regression.
- `match_diagnostics.parquet` for per-pair forensics on surprising matches.
- See [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md).

## Next

- Bump the frontend: [skills/update-site](../update-site/SKILL.md).

## Key code

- Matching: [src/openpois/conflation/match.py](../../../src/openpois/conflation/match.py)
- Merging: [src/openpois/conflation/merge.py](../../../src/openpois/conflation/merge.py)
- Taxonomy assignment: [src/openpois/conflation/taxonomy.py](../../../src/openpois/conflation/taxonomy.py)
- Conflation algorithm docs: [scripts/conflation/README.md](../../../scripts/conflation/README.md)
