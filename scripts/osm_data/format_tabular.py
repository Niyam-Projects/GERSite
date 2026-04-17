"""
Reformat raw OSM version histories into modelling-ready observations.

Reads osm_versions.parquet and osm_changes.parquet (produced by either
osm_data/download_history.py for US+PR or osm_data/download.py for a
Seattle-scoped Overpass run), then converts them into an observation-per-version
format suitable for the change-rate model. Each observation records the tag
value, the timestamps of the previous tag assignment and the current
observation, and a flag for whether the tag changed.

At US scale the versions + changes Parquets together exceed typical RAM, so
this script uses ``format_observations_duckdb`` to pivot changes wide, LEFT
JOIN them against versions, and ORDER BY (type, id, version) inside DuckDB,
spilling the sort to disk as needed. A Python scan then runs the per-POI
state machine over the sorted row stream and writes observations directly to
the output CSV — peak RSS stays bounded to roughly ``DUCKDB_MEMORY_LIMIT``.

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

from config_versioned import Config

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
OUT_PATH = SAVE_DIR / f"osm_observations_{TAG_KEY}.csv"

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
    print(f"Saved {n_written} observations to {OUT_PATH}")
