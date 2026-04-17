#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Spatial chunking primitives for POI conflation.

Partitions the global POI extent into roughly balanced rectangular
chunks via recursive KD-tree bisection on pooled (OSM + Overture)
centroids. Chunks tile the global bbox with half-open intervals so
every POI lands in exactly one primary chunk. Buffered bboxes are
computed by expanding each core bbox by a user-supplied distance in
metres, using an equirectangular correction on longitude.

The chunked conflation driver in
:mod:`openpois.conflation.match.find_and_score_matches_chunked`
consumes the ``ChunkSpec`` list produced here. No matching or scoring
logic lives in this module; it is purely geometric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

_METRES_PER_DEG_LAT = 111_139.0


@dataclass(frozen = True)
class ChunkSpec:
    """One spatial chunk.

    - ``chunk_id`` is a dense 0-based index into the chunk list.
    - ``core_bbox`` tiles the global extent with half-open intervals
      (see :func:`assign_primary_chunk`).
    - ``buffered_bbox`` is ``core_bbox`` expanded by ``buffer_m``
      metres on each side, converted to degrees via equirectangular
      approximation.
    """

    chunk_id: int
    core_bbox: tuple[float, float, float, float]
    buffered_bbox: tuple[float, float, float, float]


# -----------------------------------------------------------------
# Centroid extraction
# -----------------------------------------------------------------


def extract_centroids_lonlat(geom_array) -> np.ndarray:
    """
    Extract ``(lon, lat)`` centroids from a shapely geometry array.

    Returns an ``(N, 2)`` float64 array with lon in column 0 and lat
    in column 1. Compatible with :func:`bbox_mask` and
    :func:`assign_primary_chunk`.
    """
    if len(geom_array) == 0:
        return np.zeros((0, 2), dtype = np.float64)
    centroids = shapely.centroid(geom_array)
    x = shapely.get_x(centroids)
    y = shapely.get_y(centroids)
    return np.column_stack([x, y]).astype(np.float64, copy = False)


def lonlat_to_latlon_rad(centroids_lonlat: np.ndarray) -> np.ndarray:
    """
    Convert an (N, 2) ``(lon, lat)`` array in degrees to the
    ``(lat_rad, lon_rad)`` layout expected by
    ``sklearn.neighbors.BallTree`` with ``metric='haversine'``.
    """
    if len(centroids_lonlat) == 0:
        return np.zeros((0, 2), dtype = np.float64)
    return np.column_stack(
        [
            np.deg2rad(centroids_lonlat[:, 1]),
            np.deg2rad(centroids_lonlat[:, 0]),
        ]
    )


# -----------------------------------------------------------------
# KD-tree bisection
# -----------------------------------------------------------------


def _axis_extents_m(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Return ``(x_extent_m, y_extent_m)`` for a lon/lat bbox.

    Uses an equirectangular correction: longitude degrees are scaled
    by ``cos(mean_lat)`` so that "longer axis in metres" is a fair
    comparison at high latitudes (Alaska) where 1° lon << 1° lat.
    """
    x_range_deg = bbox[2] - bbox[0]
    y_range_deg = bbox[3] - bbox[1]
    mean_lat = 0.5 * (bbox[1] + bbox[3])
    cos_lat = max(float(np.cos(np.deg2rad(mean_lat))), 0.01)
    return (x_range_deg * cos_lat, y_range_deg)


def compute_chunks(
    osm_centroids_lonlat: np.ndarray,
    overture_centroids_lonlat: np.ndarray,
    chunk_target_pois: int,
    buffer_m: float,
) -> list[ChunkSpec]:
    """
    Recursively KD-bisect the pooled centroids into balanced chunks.

    Pools both datasets so a chunk where one source is sparse still
    receives enough POIs to balance the other. Splits on the axis
    with the larger metres-extent at the median of the current
    points, producing rectangular bboxes that tile the global extent
    of the pooled input with no gaps or overlaps.

    Recursion stops for a branch when ``len(points) <= target`` or
    when a median split leaves one side empty (all points share the
    splitting coordinate — further splitting would not reduce size).

    Args:
        osm_centroids_lonlat: ``(n_osm, 2)`` array of ``(lon, lat)``.
        overture_centroids_lonlat: ``(n_ov, 2)`` array.
        chunk_target_pois: Target maximum POIs per chunk.
        buffer_m: Buffer width added to each core bbox to produce
            ``buffered_bbox``. Should equal the conflation
            ``max_radius_m`` so that per-chunk matching sees every
            candidate pair for a core POI.

    Returns:
        List of ``ChunkSpec`` with dense 0-based ``chunk_id``.
    """
    pooled = np.vstack(
        [osm_centroids_lonlat, overture_centroids_lonlat]
    )
    if len(pooled) == 0:
        raise ValueError(
            "Cannot compute chunks from empty centroid arrays."
        )

    xmin = float(pooled[:, 0].min())
    xmax = float(pooled[:, 0].max())
    ymin = float(pooled[:, 1].min())
    ymax = float(pooled[:, 1].max())
    global_bbox = (xmin, ymin, xmax, ymax)

    core_bboxes: list[tuple[float, float, float, float]] = []
    stack: list[
        tuple[tuple[float, float, float, float], np.ndarray]
    ] = [(global_bbox, np.arange(len(pooled)))]

    while stack:
        bbox, idx = stack.pop()
        if len(idx) <= chunk_target_pois:
            core_bboxes.append(bbox)
            continue

        x_extent_m, y_extent_m = _axis_extents_m(bbox)
        if x_extent_m >= y_extent_m:
            col = 0
            median = float(np.median(pooled[idx, col]))
            left_bbox = (bbox[0], bbox[1], median, bbox[3])
            right_bbox = (median, bbox[1], bbox[2], bbox[3])
        else:
            col = 1
            median = float(np.median(pooled[idx, col]))
            left_bbox = (bbox[0], bbox[1], bbox[2], median)
            right_bbox = (bbox[0], median, bbox[2], bbox[3])

        left_mask = pooled[idx, col] < median
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]

        # Degenerate: median split leaves one side empty because all
        # points share this coordinate. Further splitting can't help;
        # accept the oversized chunk.
        if len(left_idx) == 0 or len(right_idx) == 0:
            core_bboxes.append(bbox)
            continue

        stack.append((left_bbox, left_idx))
        stack.append((right_bbox, right_idx))

    # Stable, deterministic chunk_id ordering
    core_bboxes.sort()

    lat_buffer_deg = buffer_m / _METRES_PER_DEG_LAT
    chunks: list[ChunkSpec] = []
    for i, bbox in enumerate(core_bboxes):
        mean_lat = 0.5 * (bbox[1] + bbox[3])
        cos_lat = max(
            float(np.cos(np.deg2rad(mean_lat))), 0.01,
        )
        lon_buffer_deg = buffer_m / (
            _METRES_PER_DEG_LAT * cos_lat
        )
        buffered = (
            bbox[0] - lon_buffer_deg,
            bbox[1] - lat_buffer_deg,
            bbox[2] + lon_buffer_deg,
            bbox[3] + lat_buffer_deg,
        )
        chunks.append(
            ChunkSpec(
                chunk_id = i,
                core_bbox = bbox,
                buffered_bbox = buffered,
            )
        )
    return chunks


# -----------------------------------------------------------------
# Point-to-chunk assignment
# -----------------------------------------------------------------


def assign_primary_chunk(
    centroids_lonlat: np.ndarray,
    chunks: list[ChunkSpec],
) -> np.ndarray:
    """
    Map each centroid to the unique chunk whose core bbox contains
    it.

    Uses half-open intervals ``[xmin, xmax) × [ymin, ymax)`` for
    every chunk except where the upper bound equals the global
    extent — those are closed so that points lying exactly on the
    global rightmost / topmost boundary are claimed. Because the
    global bbox was derived from the pooled centroids' min/max,
    at least one point is guaranteed to sit on each global boundary.

    Args:
        centroids_lonlat: ``(N, 2)`` array of ``(lon, lat)``.
        chunks: Output of :func:`compute_chunks`.

    Returns:
        ``(N,)`` int array of chunk IDs.

    Raises:
        ValueError: If any centroid falls outside every chunk's
            core bbox (indicates a tiling gap — the chunks were not
            produced by :func:`compute_chunks` on the same data).
    """
    n = len(centroids_lonlat)
    result = np.full(n, -1, dtype = np.int32)
    if n == 0:
        return result

    x = centroids_lonlat[:, 0]
    y = centroids_lonlat[:, 1]

    global_xmax = max(c.core_bbox[2] for c in chunks)
    global_ymax = max(c.core_bbox[3] for c in chunks)

    for c in chunks:
        xmin, ymin, xmax, ymax = c.core_bbox
        x_upper = (x <= xmax) if xmax == global_xmax else (x < xmax)
        y_upper = (y <= ymax) if ymax == global_ymax else (y < ymax)
        in_chunk = (
            (x >= xmin) & x_upper & (y >= ymin) & y_upper
        )
        unassigned = result == -1
        result[in_chunk & unassigned] = c.chunk_id

    if (result == -1).any():
        n_unassigned = int((result == -1).sum())
        raise ValueError(
            f"{n_unassigned} centroid(s) did not fall into any "
            "chunk's core bbox; chunks and centroids disagree."
        )
    return result


def bbox_mask(
    centroids_lonlat: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Vectorized closed-interval bbox-containment test.

    Used to subset the full centroid arrays to each chunk's buffered
    bbox at matching time. Closed on all four sides because buffered
    bboxes overlap by design.
    """
    if len(centroids_lonlat) == 0:
        return np.zeros(0, dtype = bool)
    x = centroids_lonlat[:, 0]
    y = centroids_lonlat[:, 1]
    xmin, ymin, xmax, ymax = bbox
    return (
        (x >= xmin) & (x <= xmax)
        & (y >= ymin) & (y <= ymax)
    )
