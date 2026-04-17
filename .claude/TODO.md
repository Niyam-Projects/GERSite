# openpois — running to-do

Short running list of in-progress / upcoming work. Edit freely; trim older completed items when the list gets long. Date items `YYYY-MM-DD` when added.

## In progress

## Upcoming

- [ ] Trim peak memory in `scripts/conflation/conflate.py` so the OSM(8.7M) × Overture(13M) run fits inside 16GB+4GB swap. Widened Overture allowlist (6.29M → 13.05M) overflowed WSL2 and rebooted the VM on 2026-04-17. Tactics: drop `osm_gdf`/`overture_gdf` down to the minimal column set `build_merge_parts` needs before the scoring pass, free normalized name arrays right after `compute_name_scores`, and narrow `osm_idx`/`overture_idx` to int32. See [src/openpois/conflation/match.py](../src/openpois/conflation/match.py) and [scripts/conflation/conflate.py](../scripts/conflation/conflate.py). Added 2026-04-17.
- [ ] Instrument setup / matching-reload phase of `scripts/conflation/conflate.py` to pin down the ~17 GB VmHWM spike observed on the 2026-04-17 chunked run (checkpoints reloaded, merge phase bounded — spike is upstream of both). Likely culprits: `pd.concat` of 128 chunk parquets, name/brand array construction holding dual refs, or taxonomy crosswalk transient. Add `psutil` RSS logging at each phase boundary so we can see exactly which step jumps. Added 2026-04-17.
- [ ] Watch for a DuckDB release that fixes the WSL2 httpfs "Information loss on integer cast" crash (issue #21669, fix PR #21395). Once a tagged release ships with the fix and a full `scripts/overture/download.py` run on WSL2 completes, we can unpin from `duckdb==1.4.1` and revert the per-part download to a single-query DuckDB scan. Added 2026-04-17.
- [ ] Auto-check taxonomy changes whenever we switch to a new Overture Maps version (detect new/removed L0/L1/L2 categories vs. `taxonomy_crosswalk_overture_maps.csv` and flag gaps). Added 2026-04-16.
- [ ] Watch for Overture L0/L1 → flat `basic_category` migration (~June 2026). Crosswalk CSV + `assign_overture_shared_label` will need updating. See [docs/taxonomy-setup.md](docs/taxonomy-setup.md).

## Recently done

_(trim after a few weeks)_

- [x] Fix: CONUS Overture download crashed DuckDB on httpfs scans — 2026-04-17. Refactored [src/openpois/io/overture.py](../src/openpois/io/overture.py) to per-part resumable download + final filter-in-DuckDB; pinned `duckdb==1.4.1` to dodge bug #21669. Full run produced 13,054,244 POIs.

---

**Agent note:** When uncommitted changes are present in the repo, do not assume they belong in "In progress" here — confirm with the user first. This file is curated, not auto-synced to git status.
