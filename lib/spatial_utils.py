"""
Spatial utilities for GERSite building conflation.

Provides:
- ``compute_iou_sql`` — DuckDB SQL expression for Intersection-over-Union.
- ``load_aoi_polygon`` — Loads an AOI GeoJSON polygon as a Shapely geometry.
- ``aoi_polygon_wkt`` — Returns the AOI polygon as WKT for DuckDB ST_Within.
- ``add_h3_columns`` — Adds H3 partition and sort columns to a GeoDataFrame.
- ``write_h3_partitioned_dataset`` — Writes a GeoDataFrame as an H3-partitioned
  GeoParquet dataset.

Geometry rank logic (for priority selection) adapted from
src/openpois/conflation/merge.py:_pick_geometries.
Partition write pattern adapted from src/openpois/io/geohash_partition.py.
Boundary loading adapted from src/openpois/io/boundary.py.
"""
from __future__ import annotations

import json
import io
import shutil
from pathlib import Path
from typing import Optional

import geopandas as gpd
import h3
import numpy as np
import pyarrow.parquet as pq
import shapely
from shapely.geometry import shape


# ---------------------------------------------------------------------------
# IoU SQL expression
# ---------------------------------------------------------------------------


def compute_iou_sql(
    geom_a: str = "a.geometry",
    geom_b: str = "b.geometry",
    alias: str = "iou",
) -> str:
    """Return a DuckDB SQL expression computing Intersection-over-Union (IoU).

    IoU = Area(A ∩ B) / Area(A ∪ B)

    The expression guards against zero-area unions to avoid division by zero.

    Args:
        geom_a: SQL column reference for the first geometry.
        geom_b: SQL column reference for the second geometry.
        alias: Output column alias.

    Returns:
        SQL expression string, suitable for use in a SELECT clause.

    Example:
        >>> sql = compute_iou_sql("fema.geometry", "overture.geometry", "iou")
        >>> # "ST_Area(ST_Intersection(fema.geometry, overture.geometry)) /
        >>> #  NULLIF(ST_Area(ST_Union(fema.geometry, overture.geometry)), 0) AS iou"
    """
    return (
        f"ST_Area(ST_Intersection({geom_a}, {geom_b})) / "
        f"NULLIF(ST_Area(ST_Union({geom_a}, {geom_b})), 0) AS {alias}"
    )


# ---------------------------------------------------------------------------
# AOI polygon helpers
# ---------------------------------------------------------------------------


def load_aoi_polygon(geojson_path: str | Path) -> shapely.Geometry:
    """Load an AOI polygon from a GeoJSON file.

    Args:
        geojson_path: Path to a GeoJSON Feature file.

    Returns:
        Shapely geometry of the AOI polygon.
    """
    with open(geojson_path, "r") as f:
        feature = json.load(f)
    return shape(feature["geometry"])


def aoi_polygon_wkt(geojson_path: str | Path) -> str:
    """Return the AOI polygon as WKT for use in DuckDB ST_Within / ST_Intersects.

    Args:
        geojson_path: Path to a GeoJSON Feature file.

    Returns:
        WKT string of the polygon.
    """
    return load_aoi_polygon(geojson_path).wkt


# ---------------------------------------------------------------------------
# Geometry rank selection
# ---------------------------------------------------------------------------

# Shapely type IDs → geometry rank (higher = preferred for building footprints)
_GEOM_RANK = np.ones(8, dtype=np.int8)
_GEOM_RANK[0] = 1   # Point
_GEOM_RANK[1] = 2   # LineString
_GEOM_RANK[3] = 3   # Polygon
_GEOM_RANK[6] = 4   # MultiPolygon


def pick_preferred_geometry(
    primary_geoms: np.ndarray,
    fallback_geoms: np.ndarray,
) -> np.ndarray:
    """Vectorized geometry selection: prefer the higher-rank geometry type.

    Rank order (ascending): Point < LineString < Polygon < MultiPolygon.
    Returns primary geometry on ties (Overture preferred over FEMA on equals).

    Adapted from src/openpois/conflation/merge.py:_pick_geometries.

    Args:
        primary_geoms: Array of Shapely geometries (higher priority source).
        fallback_geoms: Array of Shapely geometries (lower priority source).

    Returns:
        Array of selected geometries (same shape as inputs).
    """
    primary_types = shapely.get_type_id(primary_geoms)
    fallback_types = shapely.get_type_id(fallback_geoms)
    primary_ranks = _GEOM_RANK[primary_types]
    fallback_ranks = _GEOM_RANK[fallback_types]

    use_fallback = fallback_ranks > primary_ranks
    result = primary_geoms.copy()
    result[use_fallback] = fallback_geoms[use_fallback]
    return result


# ---------------------------------------------------------------------------
# H3 partitioning
# ---------------------------------------------------------------------------


def add_h3_columns(
    gdf: gpd.GeoDataFrame,
    partition_resolution: int = 4,
    sort_resolution: int = 7,
) -> gpd.GeoDataFrame:
    """Add H3 partition and sort columns derived from geometry centroids.

    H3 resolution reference (approximate edge lengths):
      res 4 ≈ 111 km  (coarse partition key, ~120K cells globally)
      res 7 ≈ 1.2 km  (fine sort key for row-group pruning)

    Args:
        gdf: Input GeoDataFrame (any geometry type, EPSG:4326).
        partition_resolution: H3 resolution for Hive partition column.
        sort_resolution: H3 resolution for within-partition row sort.

    Returns:
        GeoDataFrame with added 'h3_partition' and 'h3_sort' columns.
        Rows with null/empty geometries are dropped.
    """
    mask = ~gdf.geometry.is_empty & gdf.geometry.notna()
    if not mask.all():
        gdf = gdf[mask].reset_index(drop=True)

    centroids = shapely.centroid(gdf.geometry.to_numpy())
    lats = shapely.get_y(centroids)
    lons = shapely.get_x(centroids)

    gdf = gdf.copy()
    gdf["h3_partition"] = [
        h3.latlng_to_cell(float(lat), float(lon), partition_resolution)
        for lat, lon in zip(lats, lons)
    ]
    gdf["h3_sort"] = [
        h3.latlng_to_cell(float(lat), float(lon), sort_resolution)
        for lat, lon in zip(lats, lons)
    ]
    return gdf


def write_h3_partitioned_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir: str | Path,
    partition_resolution: int = 4,
    sort_resolution: int = 7,
    overwrite: bool = True,
    attribution_metadata: Optional[dict] = None,
) -> None:
    """Write a GeoDataFrame as an H3-partitioned GeoParquet dataset.

    Writes one Parquet file per H3 partition cell into a Hive-style layout:
    ``h3_partition=<cell>/part-0.parquet``. Within each partition, rows are
    sorted by h3_sort for spatial row-group pruning.

    Adapted from src/openpois/io/geohash_partition.py:write_partitioned_dataset.

    Args:
        gdf: GeoDataFrame to write (EPSG:4326).
        output_dir: Root directory for the partitioned output.
        partition_resolution: H3 resolution for partition key.
        sort_resolution: H3 resolution for within-partition sort.
        overwrite: If True, remove and recreate output_dir.
        attribution_metadata: Optional dict of attribution metadata to embed
            in Parquet file metadata (written as 'gers_attribution' key).
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} already exists. Pass overwrite=True to replace."
            )

    gdf = add_h3_columns(gdf, partition_resolution, sort_resolution)
    data_cols = [c for c in gdf.columns if c not in ("h3_partition", "h3_sort")]

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = gdf.groupby("h3_partition", sort=False, observed=True)
    print(f"Writing {len(groups)} H3 partitions to {output_dir} ...")

    extra_meta: dict[bytes, bytes] = {}
    if attribution_metadata:
        extra_meta[b"gers_attribution"] = json.dumps(attribution_metadata).encode()

    for cell, group in groups:
        group = group.sort_values("h3_sort")[data_cols].reset_index(drop=True)
        partition_dir = output_dir / f"h3_partition={cell}"
        partition_dir.mkdir(exist_ok=True)
        out_path = partition_dir / "part-0.parquet"

        # Write via a BytesIO buffer so we can inject custom schema metadata
        # without relying on any private GeoPandas internals.
        buf = io.BytesIO()
        group.to_parquet(buf, index=False, compression="zstd")
        buf.seek(0)
        arrow_table = pq.read_table(buf)
        if extra_meta:
            existing = arrow_table.schema.metadata or {}
            arrow_table = arrow_table.replace_schema_metadata({**existing, **extra_meta})
        pq.write_table(arrow_table, out_path, compression="zstd")
