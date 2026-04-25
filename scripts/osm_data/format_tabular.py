"""
Reformat raw OSM version histories into modelling-ready observations, tagged
with a shared taxonomy label.

Reads osm_versions.parquet and osm_changes.parquet (produced by
osm_data/download_history.py). DuckDB streams them into an observation-per-
version intermediate via ``format_observations_duckdb``. Each observation
records the value of ``osm_data.tag_key`` (the change event — the tag whose
add/change/delete fires ``changed=1``), timestamps of the previous tag
assignment and the current observation, and the current values of every
``download.osm.filter_keys`` tag.

After DuckDB finishes, this script assigns zero or more ``shared_label``
values to each row using the conflation taxonomy crosswalk and explodes the
table so that a POI version contributing to multiple taxonomy categories
produces one row per category. Rows with no matching taxonomy category are
dropped. Wildcard ``key=*`` labels are only applied when no specific
crosswalk match fires anywhere on the row (see
``assign_osm_shared_label(..., return_all=True)`` for details).

Config keys used (config.yaml):
    directories.osm_data          — directory containing input and output files
    download.osm.filter_keys      — all tag keys collected (passed as keep_keys
                                    AND used by the taxonomy assignment)
    osm_data.tag_key              — single tag key whose changes define
                                    observation events (e.g. "name")

Prerequisites:
    Run osm_data/download_history.py first to produce osm_versions.parquet
    and osm_changes.parquet.

Output file (in osm_data directory):
    osm_observations.parquet — one row per (POI version, shared_label). Columns:
        id, version, tag_key, last_tag_timestamp, obs_timestamp, changed,
        shared_label, plus every filter_keys column for reference.
"""

import pandas as pd
from config_versioned import Config

from openpois.conflation.taxonomy import (
    assign_osm_shared_label,
    load_match_radii,
    load_osm_crosswalk,
)
from openpois.osm.format_observations import format_observations_duckdb


# ----------------------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

SAVE_DIR = config.get_dir_path("osm_data")
OSM_KEYS = config.get("download", "osm", "filter_keys")
TAG_KEY = config.get("osm_data", "tag_key")

CHANGES_PATH = config.get_file_path("osm_data", "osm_changes")
VERSIONS_PATH = config.get_file_path("osm_data", "osm_versions")
OUT_PATH = config.get_file_path("osm_data", "osm_observations")

# DuckDB execution limits. The sort operator spills past memory_limit so this
# caps peak RAM independent of input size. Threads default to os.cpu_count()
# when left as None.
DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = None


# ----------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    n_written = format_observations_duckdb(
        changes_path = CHANGES_PATH,
        versions_path = VERSIONS_PATH,
        output_path = OUT_PATH,
        tag_key = TAG_KEY,
        keep_keys = OSM_KEYS,
        duckdb_memory_limit = DUCKDB_MEMORY_LIMIT,
        duckdb_threads = DUCKDB_THREADS,
    )
    print(f"DuckDB wrote {n_written:,} raw observations to {OUT_PATH}")

    print("Loading raw observations for shared-label assignment ...")
    obs_df = pd.read_parquet(OUT_PATH)

    print("Assigning shared taxonomy labels (multi-label, exploded) ...")
    labels_per_row, _ = assign_osm_shared_label(
        obs_df,
        load_osm_crosswalk(),
        load_match_radii(),
        OSM_KEYS,
        return_all = True,
    )
    obs_df["shared_label"] = labels_per_row
    n_before_explode = len(obs_df)
    obs_df = obs_df.explode("shared_label", ignore_index = True)
    obs_df = obs_df.dropna(subset = ["shared_label"])
    obs_df = obs_df[obs_df["shared_label"] != ""]

    print(
        f"Exploded {n_before_explode:,} raw rows to "
        f"{len(obs_df):,} (POI, shared_label) rows"
    )
    print("Top shared labels by row count:")
    print(obs_df["shared_label"].value_counts().head(15).to_string())

    obs_df.to_parquet(OUT_PATH, index = False)
    print(f"Saved {len(obs_df):,} observations to {OUT_PATH}")
