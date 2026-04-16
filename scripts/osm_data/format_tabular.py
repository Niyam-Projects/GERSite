"""
Reformat raw OSM version histories into modelling-ready observations.

Reads osm_versions.parquet and osm_changes.parquet (produced by either
osm_data/download_history.py for US+PR or osm_data/download.py for a
Seattle-scoped Overpass run), then converts them into an observation-per-version
format suitable for the change-rate model. Each observation records the tag
value, the timestamps of the previous tag assignment and the current
observation, and a flag for whether the tag changed.

Config keys used (config.yaml):
    directories.osm_data          — directory containing input and output files
    download.osm.filter_keys      — all tag keys collected (passed as keep_keys)
    osm_data.tag_key              — single tag key to model (e.g. "amenity")

Prerequisites:
    Run osm_data/download_history.py (US+PR, PBF-based) or osm_data/download.py
    (Seattle, Overpass-based) first to produce osm_versions.parquet and
    osm_changes.parquet.

Output file (in osm_data directory):
    osm_observations_{tag_key}.csv — one row per version observation with columns:
        id, version, tag_key, last_tag_timestamp, obs_timestamp, changed,
        plus all keep_keys columns for grouping
"""

import pandas as pd
from config_versioned import Config

from openpois.osm.format_observations import format_observations


# ----------------------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

SAVE_DIR = config.get_dir_path("osm_data")
OSM_KEYS = config.get("download", "osm", "filter_keys")
TAG_KEY = config.get("osm_data", "tag_key")


# ----------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Read files
    changes_df = pd.read_parquet(config.get_file_path("osm_data", "osm_changes"))
    versions_df = pd.read_parquet(config.get_file_path("osm_data", "osm_versions"))

    # Format changes and versions into observations
    observations_df = format_observations(
        changes_df = changes_df,
        versions_df = versions_df,
        tag_key = TAG_KEY,
        keep_keys = OSM_KEYS,
    )

    # Save observations
    out_path = SAVE_DIR / f"osm_observations_{TAG_KEY}.csv"
    observations_df.to_csv(out_path, index = False)
    print(f"Saved {len(observations_df)} observations to {out_path}")
