# Taxonomy setup

The unified taxonomy bridges OSM tags, Overture L0/L1/L2 categories, and a `shared_label` used throughout conflation and the frontend.

## Source CSVs

All four live at [src/openpois/conflation/data/](../../src/openpois/conflation/data/):

| File | Columns | Purpose |
|---|---|---|
| `taxonomy_crosswalk_openstreetmap.csv` | `osm_key`, `osm_value`, `shared_label` | Map OSM tag key/value pairs to a shared label. Wildcard `*` on `osm_value` is a fallback per key. |
| `taxonomy_crosswalk_overture_maps.csv` | `overture_l0`, `overture_l1`, `overture_l2`, `shared_label` | Map Overture hierarchy to a shared label. 4-tier cascade: (L0, L1, L2) → (L0, L2) → (L0, L1) → L0. |
| `match_radii.csv` | `shared_label`, `match_radius_m` | Per-label spatial match radius (meters). Private businesses ~50m, mid-size facilities ~75-100m, areal features ~150-200m. |
| `top_level_matches.csv` | `overture_l0`, `osm_key` | L0/key bitmask for the type-score "same broad group" check. |

## Code

[src/openpois/conflation/taxonomy.py](../../src/openpois/conflation/taxonomy.py) exposes:

- Loaders: `load_osm_crosswalk`, `load_overture_crosswalk`, `load_match_radii`, `load_top_level_matches`
- Assigners: `assign_osm_shared_label`, `assign_overture_shared_label`

OSM key priority order for label assignment: **shop > healthcare > leisure > amenity** (specific tags win over generic).

## Regenerating the site's taxonomy page

After editing any of the four CSVs:

```bash
python scripts/build_taxonomy.py
```

This renders [site/public/taxonomy.html](../../site/public/taxonomy.html) — an HTML table showing the full crosswalk + radii.

## Manual sync points

These are **not** automatic — forgetting them silently breaks the frontend:

1. **`site/src/constants.js` `CONFLATED_LABELS`** must match the `shared_label` column in `match_radii.csv`. Adding or removing labels in the CSV requires an edit here. Noted in [site/README.md](../../site/README.md).
2. **`site/public/taxonomy.html`** is generated — don't hand-edit; rerun `build_taxonomy.py`.

## Upcoming Overture migration (~June 2026)

Overture is deprecating the L0/L1/L2 `categories` hierarchy in favor of a flat `basic_category` field. When that happens:

- `taxonomy_crosswalk_overture_maps.csv` schema will need to change from `(overture_l0, overture_l1, overture_l2)` to `(basic_category)` or equivalent.
- `assign_overture_shared_label` in `taxonomy.py` will need updating to use the new field.
- `scripts/overture/download.py` → SQL queries against `taxonomy.hierarchy[1]` will need updating.

Track the migration status in the Overture Maps changelog.
