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

### PMTiles URLs

Public data is hosted on Source Cooperative. When a new version is
published, bump the URLs in `src/constants.js`:

- `OSM_PMTILES_URL` and `CONFLATED_PMTILES_URL` — version folder matches
  `versions.source_coop` in `config.yaml`
- `OVERTURE_PMTILES_URL` — bump on each monthly Overture release

Each URL is independent; OSM and conflated versions may differ.
