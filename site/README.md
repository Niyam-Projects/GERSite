# Vue 3 + Vite

This template should help get you started developing with Vue 3 in Vite. The template uses Vue 3 `<script setup>` SFCs, check out the [script setup docs](https://v3.vuejs.org/api/sfc-script-setup.html#sfc-script-setup) to learn more.

Learn more about IDE Support for Vue in the [Vue Docs Scaling up Guide](https://vuejs.org/guide/scaling-up/tooling.html#ide-support).

## Maintenance

### Taxonomy

The OSM / Overture / conflated filters derive from arrays in
`src/taxonomy.generated.js`, which is regenerated from
`src/openpois/conflation/data/*.csv` by `scripts/build_taxonomy.py`. The file
is gitignored. Use the `sync-taxonomy` skill (or run `build_taxonomy.py`
followed by `check_taxonomy_sync.py`) whenever the CSVs change. Only the
display-label maps in `src/constants.js` are hand-maintained.

### S3 snapshot URLs

When a new snapshot is uploaded to S3, update the date strings in two places:

- `src/constants.js` — `OSM_S3_BASE`, `FSQ_S3_BASE`, and `CONFLATED_S3_BASE`
- `public/about.html` — hardcoded S3 paths and browse links in the data
  access section

The OSM/Foursquare and conflated snapshots may have different dates (e.g.
Foursquare `20260313`, OSM/conflated `20260416`), so update each URL individually.
