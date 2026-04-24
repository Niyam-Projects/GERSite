"""
Partition the rated OSM snapshot by top-level tag for local queries.

Reads osm_snapshot_rated.parquet, derives a `primary_tag` per POI via first-
non-null across the configured `download.osm.filter_keys` priority order
(shop > healthcare > leisure > amenity > tourism > office > craft > historic,
matching the priority in `openpois.conflation.taxonomy.assign_osm_shared_label`),
adds a geohash sort key from each POI's centroid, and writes a Hive-style
dataset:

    osm_snapshot_partitioned/
        primary_tag=amenity/part-0.parquet
        primary_tag=shop/part-0.parquet
        ...

Rows within each partition are sorted by the `geohash` column so spatial
filters prune via Parquet row-group min/max stats. Queries like
``WHERE primary_tag = 'shop' AND shop = 'bakery'`` read a single partition
file.
"""
import geopandas as gpd
from config_versioned import Config

from openpois.io.geohash_partition import (
    add_geohash_column,
    compute_primary_osm_tag,
    write_label_partitioned_dataset,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("snapshot_osm", "rated_snapshot")
OUTPUT_DIR = config.get_file_path("snapshot_osm", "partitioned")
OVERWRITE = True

FILTER_KEYS = config.get("download", "osm", "filter_keys")
PRECISION_SORT = config.get("upload", "geohash_precision_sort")
PARTITION_COL = "primary_tag"

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Reading rated snapshot from {INPUT_PATH} ...")
    gdf = gpd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(gdf):,} POIs")

    print(f"Deriving {PARTITION_COL} from filter_keys {FILTER_KEYS} ...")
    gdf = compute_primary_osm_tag(
        gdf, filter_keys = FILTER_KEYS, out_col = PARTITION_COL
    )

    print(f"Computing geohash-{PRECISION_SORT} sort column from centroids ...")
    gdf = add_geohash_column(gdf, precision = PRECISION_SORT)

    write_label_partitioned_dataset(
        gdf,
        output_dir = OUTPUT_DIR,
        partition_col = PARTITION_COL,
        sort_col = "geohash",
        overwrite = OVERWRITE,
    )

    n_partitions = sum(1 for _ in OUTPUT_DIR.iterdir() if _.is_dir())
    print(
        f"Done. Wrote {len(gdf):,} rows across {n_partitions} "
        f"{PARTITION_COL} partitions."
    )
    print(f"Output: {OUTPUT_DIR}")
