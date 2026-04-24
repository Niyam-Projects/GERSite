"""
Partition the conflated POI dataset by destination type for local queries.

Reads conflated.parquet, adds a geohash sort key from each POI's centroid,
and writes a Hive-style dataset partitioned by `shared_label`:

    conflated_partitioned/
        shared_label=Pharmacy/part-0.parquet
        shared_label=Restaurant/part-0.parquet
        ...

Rows within each partition are sorted by the `geohash` column so spatial
filters prune via Parquet row-group min/max stats. Queries like
``WHERE shared_label = 'Pharmacy'`` read a single partition file.
"""
import geopandas as gpd
from config_versioned import Config

from openpois.io.geohash_partition import (
    add_geohash_column,
    write_label_partitioned_dataset,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("conflation", "conflated")
OUTPUT_DIR = config.get_file_path("conflation", "partitioned")
OVERWRITE = True

PRECISION_SORT = config.get("upload", "geohash_precision_sort")
PARTITION_COL = "shared_label"

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Reading conflated dataset from {INPUT_PATH} ...")
    gdf = gpd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(gdf):,} POIs")

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
