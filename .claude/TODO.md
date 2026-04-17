# openpois — running to-do

Short running list of in-progress / upcoming work. Edit freely; trim older completed items when the list gets long. Date items `YYYY-MM-DD` when added.

## In progress

## Upcoming

- [ ] Watch for a DuckDB release that fixes the WSL2 httpfs "Information loss on integer cast" crash (issue #21669, fix PR #21395). Once a tagged release ships with the fix and a full `scripts/overture/download.py` run on WSL2 completes, we can unpin from `duckdb==1.4.1` and revert the per-part download to a single-query DuckDB scan. Added 2026-04-17.
- [ ] Auto-check taxonomy changes whenever we switch to a new Overture Maps version (detect new/removed L0/L1/L2 categories vs. `taxonomy_crosswalk_overture_maps.csv` and flag gaps). Added 2026-04-16.
- [ ] Watch for Overture L0/L1 → flat `basic_category` migration (~June 2026). Crosswalk CSV + `assign_overture_shared_label` will need updating. See [docs/taxonomy-setup.md](docs/taxonomy-setup.md).

## Recently done

_(trim after a few weeks)_

- [x] Fix: CONUS Overture download crashed DuckDB on httpfs scans — 2026-04-17. Refactored [src/openpois/io/overture.py](../src/openpois/io/overture.py) to per-part resumable download + final filter-in-DuckDB; pinned `duckdb==1.4.1` to dodge bug #21669. Full run produced 13,054,244 POIs.

---

**Agent note:** When uncommitted changes are present in the repo, do not assume they belong in "In progress" here — confirm with the user first. This file is curated, not auto-synced to git status.
