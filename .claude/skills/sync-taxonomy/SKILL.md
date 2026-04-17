---
name: sync-taxonomy
description: Use when the user edits the conflation taxonomy CSVs, or when the frontend filters / taxonomy reference page are suspected to be out of sync with the CSVs. Triggers: "sync taxonomy to the site", "update frontend after taxonomy change", "rebuild taxonomy.html", "check taxonomy drift", "constants.js out of sync", "regenerate taxonomy generated file".
---

# Sync taxonomy to the frontend

The conflation CSVs in [src/openpois/conflation/data/](../../../src/openpois/conflation/data/) are the single source of truth for three frontend artifacts:

1. [site/src/taxonomy.generated.js](../../../site/src/taxonomy.generated.js) — machine-readable `SHARED_LABELS`, `OSM_KEYS`, `OVERTURE_L0S` (gitignored; imported by `constants.js`).
2. [site/public/taxonomy.html](../../../site/public/taxonomy.html) — user-facing taxonomy reference page (gitignored).
3. [site/src/constants.js](../../../site/src/constants.js) — display-label maps `OSM_KEY_LABELS` / `OVERTURE_L0_LABELS` (hand-maintained).

When the CSVs change, (1) and (2) must be regenerated, and (3) must be audited for any new keys/L0s that need human-readable labels.

## Steps

1. **Check for drift**:
   ```bash
   python scripts/check_taxonomy_sync.py
   ```
   Exit 0 = clean, skip the rest. Exit 1 = proceed.

2. **Regenerate the generated artifacts**:
   ```bash
   python scripts/build_taxonomy.py
   ```
   Writes both `site/public/taxonomy.html` and `site/src/taxonomy.generated.js`.

3. **Re-run the check**:
   ```bash
   python scripts/check_taxonomy_sync.py
   ```
   Must exit 0 now. If it still fails, something is wrong with `build_taxonomy.py` — debug before continuing.

4. **Audit display labels** — the check prints `WARN:` lines if [constants.js](../../../site/src/constants.js) is missing entries in `OSM_KEY_LABELS` or `OVERTURE_L0_LABELS`. Add pretty labels for any new keys (e.g. `tourism: 'Tourism'`). Missing entries fall back to the raw key (e.g. `tourism`), which is ugly but not broken.

5. **Run the test**:
   ```bash
   pytest tests/test_taxonomy_sync.py
   ```

6. **Preview in a browser**:
   ```bash
   cd site && npm run dev
   ```
   Verify in each source view:
   - **OSM**: filter checkboxes match `OSM_KEYS` in the generated file; toggling a new key (e.g. `tourism`) filters POIs appropriately.
   - **Overture**: POIs from all 12 L0 categories render (lodging, education, etc.). If a new L0 appears, confirm it's enabled on initial load.
   - **Conflated**: all `SHARED_LABELS` appear as checkboxes; any new "Other *" labels are unchecked by default.
   - `/taxonomy.html` page renders with the new rows.

7. **Build**:
   ```bash
   cd site && npm run build
   ```
   Must succeed without new warnings.

## Key files

- [scripts/build_taxonomy.py](../../../scripts/build_taxonomy.py) — regenerates `taxonomy.html` + `taxonomy.generated.js`.
- [scripts/check_taxonomy_sync.py](../../../scripts/check_taxonomy_sync.py) — drift detector, exits 1 on mismatch.
- [tests/test_taxonomy_sync.py](../../../tests/test_taxonomy_sync.py) — pytest wrapper; runs in CI.
- [site/src/constants.js](../../../site/src/constants.js) — `OSM_KEY_LABELS`, `OVERTURE_L0_LABELS` (display labels only).
- [.claude/docs/taxonomy-setup.md](../../docs/taxonomy-setup.md) — background on the four CSVs.

## When this skill runs automatically

This is step 1 of [skills/update-site](../update-site/SKILL.md) — anytime the site is bumped to new data, taxonomy drift is caught first. It is also referenced from [skills/conflate-snapshots](../conflate-snapshots/SKILL.md).
