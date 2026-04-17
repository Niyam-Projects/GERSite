"""Utilities for spatially partitioning GeoDataFrames by geohash for web map serving."""
import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import pygeohash
import shapely


def add_geohash_columns(
    gdf: gpd.GeoDataFrame,
    precision_partition: int,
    precision_sort: int,
) -> gpd.GeoDataFrame:
    """Add geohash_prefix (partition key) and geohash_sort columns from centroids.

    Rows with null or empty geometries are dropped before computing hashes.
    Both columns are derived from the geometry centroid, so Points, Polygons,
    and MultiPolygons are all handled uniformly.

    Geohash is a prefix code, so the partition hash equals the first
    ``precision_partition`` characters of the sort hash. We encode once at
    the higher precision and derive the shorter prefix by string slicing,
    avoiding a second pass over N Shapely Points.
    """
    mask = ~gdf.geometry.is_empty & gdf.geometry.notna()
    if not mask.all():
        gdf = gdf[mask].reset_index(drop = True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Geometry is in a geographic CRS", UserWarning
        )
        centroids = shapely.centroid(gdf.geometry.to_numpy())
    lats = shapely.get_y(centroids)
    lons = shapely.get_x(centroids)
    del centroids

    sort_hashes = [
        pygeohash.encode(float(lat), float(lon), precision = precision_sort)
        for lat, lon in zip(lats, lons)
    ]
    gdf["geohash_sort"] = sort_hashes
    gdf["geohash_prefix"] = [h[:precision_partition] for h in sort_hashes]
    return gdf


def write_partitioned_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir,
    overwrite: bool = True,
) -> None:
    """Sort gdf spatially and write as a geohash-partitioned parquet dataset.

    Writes one parquet file per geohash_prefix value into a Hive-style directory
    layout (geohash_prefix=9q/part-0.parquet). Converts and writes one partition
    at a time to avoid duplicating the full dataset in memory.

    The geohash_prefix column becomes the Hive partition directory name and is
    dropped from the stored parquet files. The geohash_sort column is used only
    for row ordering and is also dropped before writing.
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            print(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )

    cols = [c for c in gdf.columns if c not in ("geohash_prefix", "geohash_sort")]
    output_dir.mkdir(parents = True, exist_ok = True)

    # Iterate without a global sort_values: that would double peak memory on
    # multi-GB frames. groupby(sort = False) hands us each partition as a view;
    # each small partition is sorted in-place before writing.
    groups = gdf.groupby("geohash_prefix", sort = False, observed = True)
    n_partitions = len(groups)
    print(f"Writing {n_partitions} partitions to {output_dir} ...")
    for i, (prefix, group) in enumerate(groups):
        partition_dir = output_dir / f"geohash_prefix={prefix}"
        partition_dir.mkdir()
        group.sort_values("geohash_sort")[cols].to_parquet(
            partition_dir / "part-0.parquet"
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_partitions} partitions written...")
