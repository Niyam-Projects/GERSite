"""
Download US+PR OSM full-history data for POI change-rate modelling.

Downloads Geofabrik full-history PBFs (US + PR), filters them to the configured
POI tag keys, slices to the configured date range with ``osmium time-filter``,
and streams each element's versions into osm_versions.parquet plus one row per
tag-level change into osm_changes.parquet. Those two Parquets feed
format_tabular.py unchanged.

Geofabrik's internal server requires OSM OAuth. Point ``history_cookie_file`` at
a Netscape-format cookie jar (export from a browser logged in at
osm-internal.download.geofabrik.de, or use Geofabrik's oauth_cookie_client.py).

Config keys used (config.yaml):
    download.osm.history_pbf_url      — Geofabrik US full-history PBF URL
    download.osm.pr_history_pbf_url   — Geofabrik PR full-history PBF URL
    download.osm.history_cookie_file  — cookie file for Geofabrik OAuth (or null)
    download.osm.filter_keys          — OSM tag keys to retain
    download.osm.start_date           — start of time-filter window
    download.osm.end_date             — end of time-filter window
    download.osm.overwrite_download   — re-download raw PBFs if present
    download.osm.overwrite_filter     — re-run tags-filter/time-filter if present
    download.osm.overwrite_parse      — re-run parse if Parquets are present
    download.osm.chunk_size           — rows per Parquet-writer flush
    download.osm.verbose              — print progress
    directories.osm_data              — output directory (versioned)

Output files (in osm_data directory):
    osm_versions.parquet — one row per element version
    osm_changes.parquet  — one row per per-version tag change (Added/Changed/Deleted)
"""
import datetime

from config_versioned import Config

from openpois.io.osm_history_pbf import download_osm_history

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

HISTORY_PBF_URL = config.get("download", "osm", "history_pbf_url")
PR_HISTORY_PBF_URL = config.get("download", "osm", "pr_history_pbf_url")
HISTORY_COOKIE_FILE = config.get(
    "download", "osm", "history_cookie_file", fail_if_none = False
)
FILTER_KEYS = config.get("download", "osm", "filter_keys")
START_DATE = datetime.datetime.combine(
    config.get("download", "osm", "start_date"), datetime.time.min
)
END_DATE = datetime.datetime.combine(
    config.get("download", "osm", "end_date"), datetime.time.min
)
OVERWRITE_DOWNLOAD = config.get("download", "osm", "overwrite_download")
OVERWRITE_FILTER = config.get("download", "osm", "overwrite_filter")
OVERWRITE_PARSE = config.get("download", "osm", "overwrite_parse")
CHUNK_SIZE = config.get("download", "osm", "chunk_size")
VERBOSE = config.get("download", "osm", "verbose")

SAVE_DIR = config.get_dir_path("osm_data")
SAVE_DIR.mkdir(parents = True, exist_ok = True)

RAW_PBF = config.get_file_path("osm_data", "raw_history_pbf")
FILTERED_PBF = config.get_file_path("osm_data", "filtered_history_pbf")
TIME_FILTERED_PBF = config.get_file_path(
    "osm_data", "time_filtered_history_pbf"
)
RAW_PR_PBF = config.get_file_path("osm_data", "raw_pr_history_pbf")
FILTERED_PR_PBF = config.get_file_path("osm_data", "filtered_pr_history_pbf")
TIME_FILTERED_PR_PBF = config.get_file_path(
    "osm_data", "time_filtered_pr_history_pbf"
)

US_VERSIONS = config.get_file_path("osm_data", "us_versions")
US_CHANGES = config.get_file_path("osm_data", "us_changes")
PR_VERSIONS = config.get_file_path("osm_data", "pr_versions")
PR_CHANGES = config.get_file_path("osm_data", "pr_changes")
OUTPUT_VERSIONS = config.get_file_path("osm_data", "osm_versions")
OUTPUT_CHANGES = config.get_file_path("osm_data", "osm_changes")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    download_osm_history(
        pbf_url = HISTORY_PBF_URL,
        raw_pbf_path = RAW_PBF,
        filtered_pbf_path = FILTERED_PBF,
        time_filtered_pbf_path = TIME_FILTERED_PBF,
        us_versions_path = US_VERSIONS,
        us_changes_path = US_CHANGES,
        pr_pbf_url = PR_HISTORY_PBF_URL,
        raw_pr_pbf_path = RAW_PR_PBF,
        filtered_pr_pbf_path = FILTERED_PR_PBF,
        time_filtered_pr_pbf_path = TIME_FILTERED_PR_PBF,
        pr_versions_path = PR_VERSIONS,
        pr_changes_path = PR_CHANGES,
        output_versions_path = OUTPUT_VERSIONS,
        output_changes_path = OUTPUT_CHANGES,
        filter_keys = FILTER_KEYS,
        start_date = START_DATE,
        end_date = END_DATE,
        cookie_file = HISTORY_COOKIE_FILE,
        overwrite_download = OVERWRITE_DOWNLOAD,
        overwrite_filter = OVERWRITE_FILTER,
        overwrite_parse = OVERWRITE_PARSE,
        chunk_size = CHUNK_SIZE,
        verbose = VERBOSE,
    )
