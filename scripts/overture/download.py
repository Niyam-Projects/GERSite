"""
Download the current US+PR Overture Maps Places snapshot as a GeoParquet file.

Queries Overture Maps GeoParquet files on public S3 using DuckDB's httpfs and
spatial extensions, filtering with a two-stage spatial filter (coarse bbox
prefilter in the DuckDB query, then exact within-polygon filter in Python
against the US+PR Census boundary) and by L0 taxonomy category. No
authentication required — Overture Maps data is publicly accessible.

Auto-detects the latest available Overture release from S3 unless a specific
release_date is pinned in config.yaml.

Config keys used (config.yaml):
    download.overture.release_date           — pinned release (null = auto-detect)
    download.overture.s3_bucket              — Overture Maps S3 bucket name
    download.overture.s3_region              — AWS region of the Overture bucket
    download.overture.taxonomy_l0_categories — L0 category filter list
    download.general.boundary.source_url     — Census state-boundary zip URL
    download.general.boundary.coastline_buffer_m — outward coastline buffer (m)
    directories.boundary                     — cache directory for boundary file
    directories.snapshot_overture            — output directory

Output file:
    overture_snapshot.parquet — GeoParquet with US+PR POIs
        Columns: overture_id, overture_name, taxonomy_l0, taxonomy_l1,
        taxonomy_l2, brand_name, confidence, geometry, source
"""
from config_versioned import Config
from openpois.io.boundary import get_us_pr_boundary
from openpois.io.overture import download_overture_snapshot

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

# None = auto-detect latest
RELEASE_DATE = config.get("download", "overture", "release_date", fail_if_none=False)
S3_BUCKET = config.get("download", "overture", "s3_bucket")
S3_REGION = config.get("download", "overture", "s3_region")
TAXONOMY_CATEGORIES = config.get("download", "overture", "taxonomy_l0_categories")
BOUNDARY_URL = config.get("download", "general", "boundary", "source_url")
COASTLINE_BUFFER_M = config.get(
    "download", "general", "boundary", "coastline_buffer_m"
)
BOUNDARY_DIR = config.get_dir_path("boundary")
SAVE_DIR = config.get_dir_path("snapshot_overture")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = config.get_file_path("snapshot_overture", "snapshot")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    boundary_gdf, coarse_bboxes = get_us_pr_boundary(
        source_url = BOUNDARY_URL,
        cache_dir = BOUNDARY_DIR,
        coastline_buffer_m = COASTLINE_BUFFER_M,
    )
    gdf = download_overture_snapshot(
        output_path=OUTPUT_PATH,
        taxonomy_l0_categories=TAXONOMY_CATEGORIES,
        boundary_gdf=boundary_gdf,
        coarse_bboxes=coarse_bboxes,
        bucket=S3_BUCKET,
        s3_region=S3_REGION,
        release_date=RELEASE_DATE,
    )
    print(f"Saved {len(gdf):,} Overture POIs to {OUTPUT_PATH}")
