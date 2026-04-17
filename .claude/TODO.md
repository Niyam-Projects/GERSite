# openpois — running to-do

Short running list of in-progress / upcoming work. Edit freely; trim older completed items when the list gets long. Date items `YYYY-MM-DD` when added.

## In progress

## Upcoming

- [ ] Watch for a DuckDB release that fixes the WSL2 httpfs "Information loss on integer cast" crash (issue #21669, fix PR #21395). Once a tagged release ships with the fix and a full `scripts/overture/download.py` run on WSL2 completes, we can unpin from `duckdb==1.4.1` and revert the per-part download to a single-query DuckDB scan. Added 2026-04-17.
- [ ] Auto-check taxonomy changes whenever we switch to a new Overture Maps version (detect new/removed L0/L1/L2 categories vs. `taxonomy_crosswalk_overture_maps.csv` and flag gaps). Added 2026-04-16.
- [ ] Watch for Overture L0/L1 → flat `basic_category` migration (~June 2026). Crosswalk CSV + `assign_overture_shared_label` will need updating. See [docs/taxonomy-setup.md](docs/taxonomy-setup.md).

## Recently done

_(trim after a few weeks)_

- [x] Trim peak memory in `scripts/conflation/conflate.py` — 2026-04-17. Landed four changes in one PR: (1) chunk match dtypes narrowed to int32/float32; (2) chunk matches streamed to disk-only parquets, no `part_dfs` RAM accumulation; (3) pandas concat+sort+drop_duplicates replaced with DuckDB `ROW_NUMBER()` dedup over the checkpoint parquet glob with a bounded memory limit; (4) `osm_gdf`/`overture_gdf` dropped during chunked matching and reloaded with narrow merge-only columns before the merge step. Also added `log_rss()` phase-boundary instrumentation via `/proc/self/status` (no psutil dep). `--test` run peak 9.35 GB, down from 17 GB on the last full CONUS run; full-run measurement pending.
- [x] Instrument conflate.py phase boundaries — 2026-04-17. Shipped together with the memory trim above; `log_rss()` prints RSS + VmHWM at each phase (load, taxonomy, drop-gdfs, matching, merge reload, save).
- [x] Fix: CONUS Overture download crashed DuckDB on httpfs scans — 2026-04-17. Refactored [src/openpois/io/overture.py](../src/openpois/io/overture.py) to per-part resumable download + final filter-in-DuckDB; pinned `duckdb==1.4.1` to dodge bug #21669. Full run produced 13,054,244 POIs.

---

**Agent note:** When uncommitted changes are present in the repo, do not assume they belong in "In progress" here — confirm with the user first. This file is curated, not auto-synced to git status.
