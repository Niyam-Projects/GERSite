---
name: update-site
description: Use when the user wants to bump the frontend to point at newly uploaded S3 data, or wants to run/preview/build the site locally. Triggers: "push new data to the site", "bump site to latest data version", "update constants.js", "deploy the site", "preview the site with new data", "rebuild site after data refresh".
---

# Update + verify the site

Vue 3 + Vite frontend lives in [site/](../../../site/). After a data pull + upload, the site's S3 URLs need a manual bump.

## Prerequisites

- New data published to S3 via [skills/conflate-snapshots](../conflate-snapshots/SKILL.md) (or the latest FSQ snapshot uploaded separately).
- Node + npm available (see `site/package.json` for engine requirements).

## Steps

1. **Update S3 URLs in [site/src/constants.js](../../../site/src/constants.js)** — each source may have a different date:
   - `OSM_S3_BASE` → `snapshots/osm/YYYYMMDD/osm_snapshot_partitioned`
   - `FSQ_S3_BASE` → `snapshots/foursquare/YYYYMMDD/foursquare_snapshot_partitioned`
   - `CONFLATED_S3_BASE` → `snapshots/conflated/YYYYMMDD/conflated_partitioned`
   - `OVERTURE_PMTILES_URL` → bump on monthly Overture release

2. **Update hardcoded links in [site/public/about.html](../../../site/public/about.html)** — the S3 browse paths in the data-access section must match constants.js.

3. **Sync category labels if taxonomy changed** — if `match_radii.csv` labels were added/removed, update `CONFLATED_LABELS` in `site/src/constants.js`. See [docs/taxonomy-setup.md](../../docs/taxonomy-setup.md).

4. **Local preview**:
   ```bash
   cd site && npm run dev
   ```
   Verify:
   - Map loads POIs at zoom 14+ without CORS/404 errors
   - Source filter dropdown (OSM / Overture / Foursquare / Conflated) toggles data
   - Taxonomy legend renders from `taxonomy.html`
   - POI popups show non-empty name/category/confidence

5. **Production build**:
   ```bash
   npm run build
   ```
   Inspect `dist/` output; flag large chunk-size increases if dependencies changed.

6. **Deploy** — per host's deployment mechanism (not scripted in-repo).

7. **Post-deploy check** — load the deployed site, open browser console, confirm no CORS or 404s on the new S3 URLs.

## Commit convention

Two separate commits, matching the recent history:
- "Push to new data version" — `config.yaml` and upload script changes
- "Update to latest data version" — `site/src/constants.js` + `site/public/about.html`

## Key files

- [site/src/constants.js](../../../site/src/constants.js) — S3 URLs, PMTiles URL, color ramps, zoom thresholds, CONFLATED_LABELS
- [site/public/about.html](../../../site/public/about.html) — hardcoded data-access links
- [site/vite.config.js](../../../site/vite.config.js) — code-split chunks (ol, duckdb, arrow, etc.)
- [site/README.md](../../../site/README.md) — maintenance notes on the two sync points
