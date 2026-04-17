---
name: verify-pipeline-run
description: Use when the user wants a QA/sanity check on a recently completed pipeline run — row counts, parameter spot-checks, diffs vs. prior versions, or site-side verification. Triggers: "sanity check the run", "verify the new data looks right", "QA the model output", "diff against the last version", "did the upload work".
---

# Verify a pipeline run

Post-run QA runbook. Pick the subsection that matches what just ran.

## Snapshots (OSM / Overture / Foursquare)

Baseline row counts (2026-04-17):
- OSM: ~7.78M
- Overture: ~13.05M (up from ~7.23M after widening `taxonomy_allowlist`; pre-2026-04-17 runs will be lower)
- Foursquare: ~8.32M

Check:
```python
import pandas as pd
pd.read_parquet(path).shape[0]
```

Flag >5% drops. Known regression patterns:
- **Foursquare**: PR alpha-2 code — filter must be `country IN ('US', 'PR')`, not `'US'` only.
- **OSM**: PR is a *separate* PBF — confirm both `us-latest.osm.pbf` and `puerto-rico-latest.osm.pbf` got downloaded, filtered, and concat'd.
- **Overture**: coarse-bbox pushdown + final DuckDB `ST_Within` — drop means the Aleutian antimeridian split was lost or the Census boundary failed to load. If the run crashed with "Information loss on integer cast", the DuckDB pin was bumped off 1.4.1 (see [docs/data-sources.md](../../docs/data-sources.md) → Overture Maps).

## Model output

```
~/data/openpois/osm_turnover_model/{version}/
  fitted_params.csv     # λ and σ per group
  param_draws.csv       # uncertainty bounds
  predictions.csv       # predictions per POI
  fitted_model.pt       # torch state
```

Checks:
- Row count in `fitted_params.csv` ≈ number of groups (after `min_value_count` filter).
- λ values in a sensible range (spot-check against prior `fitted_params.csv`).
- `predictions.csv` head/tail — every POI should have a prediction; no NaNs.

## Rated snapshot

```
~/data/openpois/snapshots/osm/{version}/osm_snapshot_rated.parquet
```

Confirm `conf_mean`, `conf_lower`, `conf_upper` columns are populated for every row. NaNs indicate groups not covered by any `{stub}_by_*` or `{stub}_constant` fallback.

## Conflation

```
~/data/openpois/conflation/{version}/
  conflated.parquet
  match_diagnostics.parquet
  summary_by_label.csv
```

- Match rate per label in `summary_by_label.csv` should resemble prior run. Large drifts → parameter regression or crosswalk edit.
- `match_diagnostics.parquet` for per-pair forensics when specific matches look wrong.

## Site

- Open the deployed site (or `npm run dev` locally after a constants.js bump).
- Browser console: no CORS, no 404s on S3 URLs.
- Filter dropdown: each source (OSM / Overture / Foursquare / Conflated) loads.
- Popups non-empty; taxonomy legend rendered; PMTiles overlay visible at zoom 14+.

## Recording issues

Anything anomalous goes into [.claude/TODO.md](../../TODO.md) under **In progress** so follow-ups don't drop.
