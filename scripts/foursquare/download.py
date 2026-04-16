"""
Download the current US+PR Foursquare OS Places snapshot as a GeoParquet file.

Authenticates to the Foursquare Places Portal Apache Iceberg REST catalog
using a portal token, loads the unpartitioned places_os table filtered to
places whose country is 'US' or 'PR' with no closed date, joins against
categories_os to resolve L1 category names, applies an exact within-polygon
filter against the US+PR Census boundary, and saves the result as a
GeoParquet file.

Authentication:
    Set the FSQ_PORTAL_TOKEN environment variable before running:
        export FSQ_PORTAL_TOKEN="<your_token>"
    Register at https://places.foursquare.com to obtain a token.

Config keys used (config.yaml):
    download.foursquare.release_date        — pinned release (null = auto-detect)
    download.foursquare.catalog_uri         — REST catalog endpoint URL
    download.foursquare.catalog_warehouse   — warehouse name ("places")
    download.foursquare.catalog_namespace   — namespace ("datasets")
    download.foursquare.places_table        — places table name ("places_os")
    download.foursquare.categories_table    — categories table name ("categories_os")
    download.foursquare.token_env_var       — env var name for the portal token
    download.foursquare.l1_category_names   — L1 category filter list
    download.general.boundary.source_url    — Census state-boundary zip URL
    download.general.boundary.coastline_buffer_m — outward coastline buffer (m)
    directories.boundary                    — cache directory for boundary file
    directories.snapshot_foursquare         — output directory

Output file:
    foursquare_snapshot.parquet — GeoParquet with US+PR POIs
        Columns: fsq_place_id, name, fsq_category_ids, geometry, source
"""
from config_versioned import Config
from openpois.io.boundary import get_us_pr_boundary
from openpois.io.foursquare import download_foursquare_snapshot

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

# None = auto-detect latest
RELEASE_DATE = config.get("download", "foursquare", "release_date", fail_if_none=False)
CATALOG_URI = config.get("download", "foursquare", "catalog_uri")
CATALOG_WAREHOUSE = config.get("download", "foursquare", "catalog_warehouse")
CATALOG_NAMESPACE = config.get("download", "foursquare", "catalog_namespace")
PLACES_TABLE = config.get("download", "foursquare", "places_table")
CATEGORIES_TABLE = config.get("download", "foursquare", "categories_table")
TOKEN_ENV_VAR = config.get("download", "foursquare", "token_env_var")
L1_CATEGORIES = config.get("download", "foursquare", "l1_category_names")
BOUNDARY_URL = config.get("download", "general", "boundary", "source_url")
COASTLINE_BUFFER_M = config.get(
    "download", "general", "boundary", "coastline_buffer_m"
)
BOUNDARY_DIR = config.get_dir_path("boundary")

SAVE_DIR = config.get_dir_path("snapshot_foursquare")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = config.get_file_path("snapshot_foursquare", "snapshot")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    boundary_gdf, _ = get_us_pr_boundary(
        source_url = BOUNDARY_URL,
        cache_dir = BOUNDARY_DIR,
        coastline_buffer_m = COASTLINE_BUFFER_M,
    )
    gdf = download_foursquare_snapshot(
        output_path=OUTPUT_PATH,
        l1_category_names=L1_CATEGORIES,
        catalog_uri=CATALOG_URI,
        catalog_warehouse=CATALOG_WAREHOUSE,
        catalog_namespace=CATALOG_NAMESPACE,
        places_table=PLACES_TABLE,
        categories_table=CATEGORIES_TABLE,
        token_env_var=TOKEN_ENV_VAR,
        boundary_gdf=boundary_gdf,
        release_date=RELEASE_DATE,
    )
    print(f"Saved {len(gdf):,} Foursquare POIs to {OUTPUT_PATH}")
