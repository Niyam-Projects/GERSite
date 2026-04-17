# Data versioning

Every pipeline output is versioned via a single `versions:` block in [config.yaml](../../config.yaml). The external `config_versioned` package resolves these into filesystem paths.

## Source of truth

```yaml
versions:
  osm_data: "20260416"                 # historical PBF pipeline outputs
  model_output: "20260416_by_leisure"  # fitted model artifacts (suffix indicates variant)
  snapshot_osm: "20260416"             # OSM current-state snapshot
  snapshot_overture: "20260417"        # Overture snapshot
  snapshot_foursquare: "20260416"      # Foursquare snapshot
  aws: "20260416"                      # S3 prefix for uploaded data
  conflation: "20260417"               # conflated output
```

Each key corresponds to a `directories.<key>` entry in `config.yaml` with `versioned: true`.

## Path resolution

External `config_versioned.Config` API:

```python
config.get_dir_path("osm_data")
# → ~/data/openpois/osm_data/20260416/

config.get_file_path("osm_data", "osm_versions")
# → ~/data/openpois/osm_data/20260416/osm_versions.parquet
```

**Prefer `get_file_path` over composing `get_dir_path()` + `get()` manually.**

`.get()` raises `ValueError` on null values — pass `fail_if_none=False` for optional fields like `download.overture.release_date: null` and `download.foursquare.release_date: null`.

`config.write_self(section)` snapshots the effective config into the output directory — used by model and conflation scripts to record the state of a run.

## Naming conventions

- **Dates**: `YYYYMMDD`, e.g., `20260416`.
- **Model variants**: `{date}_by_{group_key}` (e.g., `20260416_by_leisure`, `20260416_by_amenity`) or `{date}_constant`. See [skills/iterate-model-types](../skills/iterate-model-types/SKILL.md).
- **Independent cadences**: snapshot versions can (and should) differ across sources — Overture releases ~monthly, Foursquare separately. Don't force them to match.

## External references (hand-update when bumping)

Version strings appear in these places outside `versions:` — grep before any cross-source version change:

| File | References |
|---|---|
| [config.yaml](../../config.yaml) | `upload.latest_url_osm`, `upload.latest_url_conflation` (full URL with date) |
| [site/src/constants.js](../../site/src/constants.js) | `OSM_S3_BASE`, `FSQ_S3_BASE`, `CONFLATED_S3_BASE` |
| [site/public/about.html](../../site/public/about.html) | Hardcoded S3 browse links in the data-access section |
| `osm_data.apply_model.model_stub` (config.yaml) | Which model family [scripts/osm_snapshot/apply_model.py](../../scripts/osm_snapshot/apply_model.py) ingests |

[skills/update-site](../skills/update-site/SKILL.md) covers the frontend side; [skills/conflate-snapshots](../skills/conflate-snapshots/SKILL.md) covers the upload + config side.

## Workflow

1. Bump the relevant `versions.*` keys before running a pipeline.
2. Run the pipeline — outputs land in the versioned directory.
3. After upload, update `upload.latest_url_*` and the frontend references.
4. Old versions stay on disk / S3 — delete manually when confident nothing references them.
