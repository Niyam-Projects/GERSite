#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Spatial candidate matching and scoring for POI conflation.

1. ``find_spatial_candidates`` — BallTree-based radius search to find
   nearby (OSM, Overture) pairs within category-specific thresholds.
2. ``compute_match_scores`` — multi-component scoring (distance, name,
   type taxonomy, identifiers) for each candidate pair.
3. ``select_best_matches`` — greedy one-to-one assignment above a
   minimum composite score.
4. ``find_and_score_matches_chunked`` — KD-tree-bisection driver that
   runs the above three functions per spatial chunk with a buffer
   overlap, reconciling matches across chunks via the OSM-anchored
   emit rule and per-Overture max-score dedup.
"""
from __future__ import annotations

import gc
import re
import tempfile
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from rapidfuzz import fuzz
from sklearn.neighbors import BallTree

from openpois.conflation.chunking import (
    ChunkSpec,
    assign_primary_chunk,
    bbox_mask,
    compute_chunks,
    extract_centroids_lonlat,
    lonlat_to_latlon_rad,
)

EARTH_RADIUS_M = 6_371_000.0

_MATCH_PART_COLS = [
    "osm_idx",
    "overture_idx",
    "distance_m",
    "distance_score",
    "name_score",
    "type_score",
    "identifier_score",
    "composite_score",
]


# -----------------------------------------------------------------
# Spatial candidate search
# -----------------------------------------------------------------


def _extract_centroids_rad(geom_array) -> np.ndarray:
    """
    Extract (lat_rad, lon_rad) from a geometry array.

    BallTree with ``metric='haversine'`` expects [lat, lon] in
    radians. We suppress the geographic CRS centroid warning since
    we only need approximate centroids for radius search.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Geometry is in a geographic CRS"
        )
        centroids = shapely.centroid(geom_array)

    x = shapely.get_x(centroids)
    y = shapely.get_y(centroids)
    return np.column_stack([np.deg2rad(y), np.deg2rad(x)])


def find_spatial_candidates(
    osm_geom,
    overture_geom,
    osm_radii_m: np.ndarray,
    max_radius_m: float = 200.0,
    chunk_size: int = 500_000,
    osm_centroids_rad: np.ndarray | None = None,
    overture_centroids_rad: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Find spatially proximate (OSM, Overture) candidate pairs.

    Builds a single BallTree on Overture centroids and queries it
    in chunks of OSM centroids to control memory.

    Args:
        osm_geom: OSM geometry array (GeoSeries.values or similar).
        overture_geom: Overture geometry array.
        osm_radii_m: Per-OSM-POI match radius in meters.
        max_radius_m: Global upper bound on search radius.
        chunk_size: Number of OSM rows to query per batch.
        osm_centroids_rad: Precomputed OSM centroid array shaped
            ``(n_osm, 2)`` with (lat_rad, lon_rad). When provided,
            skips re-extracting from ``osm_geom``. Useful when the
            caller iterates chunks and has already computed full-
            dataset centroids; pass the subset corresponding to
            ``osm_geom`` here.
        overture_centroids_rad: Precomputed Overture centroid array
            in (lat_rad, lon_rad). Same semantics as above.

    Returns:
        DataFrame with columns: osm_idx, overture_idx, distance_m.
    """
    overture_coords = (
        overture_centroids_rad
        if overture_centroids_rad is not None
        else _extract_centroids_rad(overture_geom)
    )
    tree = BallTree(overture_coords, metric = "haversine")

    osm_coords = (
        osm_centroids_rad
        if osm_centroids_rad is not None
        else _extract_centroids_rad(osm_geom)
    )
    # Clip radii to max and convert to radians once
    osm_radii_rad = (
        np.minimum(osm_radii_m, max_radius_m) / EARTH_RADIUS_M
    )
    n_osm = len(osm_coords)

    all_osm = []
    all_ov = []
    all_dist = []

    for start in range(0, n_osm, chunk_size):
        end = min(start + chunk_size, n_osm)
        chunk_coords = osm_coords[start:end]
        chunk_radii_rad = osm_radii_rad[start:end]

        # Pass per-POI radii so the BallTree only returns
        # neighbours within each POI's actual radius — avoids
        # the huge over-query from using max_radius for all.
        ind, dist = tree.query_radius(
            chunk_coords,
            r = chunk_radii_rad,
            return_distance = True,
        )

        for local_i in range(len(chunk_coords)):
            ov_idx = ind[local_i]
            if len(ov_idx) > 0:
                global_i = start + local_i
                d_m = dist[local_i] * EARTH_RADIUS_M
                all_osm.append(
                    np.full(
                        len(ov_idx), global_i,
                        dtype = np.int64,
                    )
                )
                all_ov.append(
                    ov_idx.astype(np.int64)
                )
                all_dist.append(d_m)

    if not all_osm:
        return pd.DataFrame(
            columns = ["osm_idx", "overture_idx", "distance_m"]
        )
    return pd.DataFrame(
        {
            "osm_idx": np.concatenate(all_osm),
            "overture_idx": np.concatenate(all_ov),
            "distance_m": np.concatenate(all_dist),
        }
    )


# -----------------------------------------------------------------
# Name scoring (vectorized)
# -----------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_name(s) -> str:
    """Lowercase, strip, collapse whitespace."""
    if s is None or pd.isna(s):
        return ""
    s = str(s).lower().strip()
    return _WHITESPACE_RE.sub(" ", s)


def _normalize_name_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array of names. Vectorized via list comp."""
    return np.array(
        [_normalize_name(v) for v in arr], dtype = object,
    )


def _batch_token_set_ratio(
    a_arr: np.ndarray,
    b_arr: np.ndarray,
) -> np.ndarray:
    """
    Compute token_set_ratio / 100 for paired arrays.

    Returns NaN where either side is empty.
    """
    n = len(a_arr)
    scores = np.full(n, np.nan, dtype = np.float64)
    for i in range(n):
        if a_arr[i] and b_arr[i]:
            scores[i] = fuzz.token_set_ratio(
                a_arr[i], b_arr[i]
            ) / 100.0
    return scores


def compute_name_scores(
    osm_names: np.ndarray,
    osm_brands: np.ndarray,
    overture_names: np.ndarray,
    overture_brands: np.ndarray,
    osm_idx: np.ndarray,
    overture_idx: np.ndarray,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """
    Compute name match score for each candidate pair.

    Pre-normalizes source arrays once (~15M strings), then scores
    in chunks to limit peak memory. Takes the max of up to 4
    comparisons (name-name, brand-brand, name-brand cross).
    Returns 0.5 (neutral) when all are null.
    """
    # Normalize source arrays once (15M total, not 80M indexed)
    norm_on = _normalize_name_array(osm_names)
    norm_ob = _normalize_name_array(osm_brands)
    norm_vn = _normalize_name_array(overture_names)
    norm_vb = _normalize_name_array(overture_brands)

    n = len(osm_idx)
    scores = np.empty(n, dtype = np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        oi = osm_idx[start:end]
        vi = overture_idx[start:end]

        on = norm_on[oi]
        ob = norm_ob[oi]
        vn = norm_vn[vi]
        vb = norm_vb[vi]

        s1 = _batch_token_set_ratio(on, vn)
        s2 = _batch_token_set_ratio(ob, vb)
        s3 = _batch_token_set_ratio(on, vb)
        s4 = _batch_token_set_ratio(ob, vn)

        stacked = np.column_stack([s1, s2, s3, s4])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice", RuntimeWarning
            )
            chunk_scores = np.nanmax(stacked, axis = 1)
        scores[start:end] = np.where(
            np.isnan(chunk_scores), 0.5, chunk_scores
        )

    del norm_on, norm_ob, norm_vn, norm_vb
    return scores


# -----------------------------------------------------------------
# Type taxonomy scoring (vectorized)
# -----------------------------------------------------------------


def compute_type_scores(
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    osm_l0_bits: np.ndarray,
    overture_l0_bits: np.ndarray,
    osm_idx: np.ndarray,
    overture_idx: np.ndarray,
) -> np.ndarray:
    """
    Score how well the POI types match between OSM and Overture.

    - Exact shared_label match: 1.0
    - L0 broad-group overlap (bitmask): 0.5
    - Otherwise: 0.0
    """
    o_labels = osm_shared_labels[osm_idx]
    v_labels = overture_shared_labels[overture_idx]

    scores = np.zeros(len(osm_idx), dtype = np.float64)

    # Tier 1: exact shared_label match
    both_present = (o_labels != "") & (v_labels != "")
    exact = both_present & (o_labels == v_labels)
    scores[exact] = 1.0

    # Tier 2: L0 bitmask overlap for non-exact pairs
    not_exact = ~exact
    if not_exact.any():
        o_bits = osm_l0_bits[osm_idx[not_exact]]
        v_bits = overture_l0_bits[overture_idx[not_exact]]
        broad = (o_bits & v_bits) != 0
        idx = np.where(not_exact)[0]
        scores[idx[broad]] = 0.5

    return scores


# -----------------------------------------------------------------
# Identifier scoring
# -----------------------------------------------------------------


def compute_identifier_scores(
    osm_idx: np.ndarray,
    overture_idx: np.ndarray,
) -> np.ndarray:
    """
    Score identifier matches.

    Returns 0.5 (neutral) for all pairs. Overture schema does not
    currently expose website/phone/wikidata fields. This component
    can be extended when those fields become available.
    """
    return np.full(len(osm_idx), 0.5, dtype = np.float64)


# -----------------------------------------------------------------
# Composite scoring
# -----------------------------------------------------------------


def compute_match_scores(
    candidates: pd.DataFrame,
    osm_names: np.ndarray,
    osm_brands: np.ndarray,
    overture_names: np.ndarray,
    overture_brands: np.ndarray,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    osm_radii_m: np.ndarray,
    osm_l0_bits: np.ndarray,
    overture_l0_bits: np.ndarray,
    distance_weight: float = 0.25,
    name_weight: float = 0.30,
    type_weight: float = 0.25,
    identifier_weight: float = 0.20,
    score_chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    """
    Compute composite match scores for all candidate pairs.

    Name and type scoring are processed in chunks of
    ``score_chunk_size`` pairs to limit peak memory.

    Returns the candidates DataFrame with added score columns:
    distance_score, name_score, type_score, identifier_score,
    composite_score.
    """
    osm_idx = candidates["osm_idx"].to_numpy()
    overture_idx = candidates["overture_idx"].to_numpy()
    distance_m = candidates["distance_m"].to_numpy()
    n = len(candidates)

    # A) Distance score (cheap vectorized arithmetic)
    pair_radii = osm_radii_m[osm_idx]
    distance_score = np.clip(
        1.0 - (distance_m / pair_radii), 0.0, 1.0
    )
    del pair_radii

    # B) Name score (chunked internally)
    name_score = compute_name_scores(
        osm_names, osm_brands,
        overture_names, overture_brands,
        osm_idx, overture_idx,
        chunk_size = score_chunk_size,
    )

    # C) Type taxonomy score (chunked to limit string arrays)
    type_score = np.empty(n, dtype = np.float64)
    for start in range(0, n, score_chunk_size):
        end = min(start + score_chunk_size, n)
        type_score[start:end] = compute_type_scores(
            osm_shared_labels, overture_shared_labels,
            osm_l0_bits, overture_l0_bits,
            osm_idx[start:end],
            overture_idx[start:end],
        )

    # D) Identifier score (neutral placeholder)
    identifier_score = np.full(n, 0.5, dtype = np.float64)

    composite = (
        distance_weight * distance_score
        + name_weight * name_score
        + type_weight * type_score
        + identifier_weight * identifier_score
    )

    # Mutate in place to avoid copying the full DataFrame
    candidates["distance_score"] = distance_score
    candidates["name_score"] = name_score
    candidates["type_score"] = type_score
    candidates["identifier_score"] = identifier_score
    candidates["composite_score"] = composite
    return candidates


# -----------------------------------------------------------------
# Best-match selection (greedy one-to-one)
# -----------------------------------------------------------------


def select_best_matches(
    scored: pd.DataFrame,
    min_score: float = 0.67,
) -> pd.DataFrame:
    """
    Greedy one-to-one matching above a minimum composite score.

    Sorts candidates by composite_score descending, then iterates:
    assign each pair if neither the OSM POI nor the Overture POI has
    been assigned yet.

    Returns:
        DataFrame of selected matches with all score columns.
    """
    above = scored[scored["composite_score"] >= min_score].copy()
    if above.empty:
        return above

    above = above.sort_values(
        "composite_score", ascending = False
    ).reset_index(drop = True)

    used_osm: set[int] = set()
    used_overture: set[int] = set()
    keep = []

    osm_arr = above["osm_idx"].to_numpy()
    ov_arr = above["overture_idx"].to_numpy()
    for i in range(len(above)):
        oi = int(osm_arr[i])
        vi = int(ov_arr[i])
        if oi not in used_osm and vi not in used_overture:
            keep.append(i)
            used_osm.add(oi)
            used_overture.add(vi)

    return above.iloc[keep].reset_index(drop = True)


# -----------------------------------------------------------------
# Chunked driver
# -----------------------------------------------------------------


_MATCH_PART_DTYPES = {
    "osm_idx": np.int32,
    "overture_idx": np.int32,
    "distance_m": np.float32,
    "distance_score": np.float32,
    "name_score": np.float32,
    "type_score": np.float32,
    "identifier_score": np.float32,
    "composite_score": np.float32,
}


def _empty_match_part() -> pd.DataFrame:
    """Empty DataFrame with the canonical match-part schema/dtypes."""
    return pd.DataFrame(
        {col: np.array([], dtype = dt)
         for col, dt in _MATCH_PART_DTYPES.items()}
    )


def _match_one_chunk(
    chunk: ChunkSpec,
    osm_centroids_lonlat: np.ndarray,
    overture_centroids_lonlat: np.ndarray,
    osm_centroids_rad: np.ndarray,
    overture_centroids_rad: np.ndarray,
    osm_primary: np.ndarray,
    osm_radii_m: np.ndarray,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    osm_l0_bits: np.ndarray,
    overture_l0_bits: np.ndarray,
    osm_names: np.ndarray,
    osm_brands: np.ndarray,
    overture_names: np.ndarray,
    overture_brands: np.ndarray,
    distance_weight: float,
    name_weight: float,
    type_weight: float,
    identifier_weight: float,
    min_match_score: float,
    max_radius_m: float,
    chunk_size: int,
) -> pd.DataFrame:
    """
    Run the candidates → score → select sequence for one chunk and
    return the OSM-anchored filtered matches with global indices.

    Caller passes full-dataset arrays; this function subsets them via
    each chunk's buffered bbox and remaps the resulting local indices
    back to global. Returned DataFrame has the
    ``_MATCH_PART_DTYPES`` schema (int32 indices, float32 scores) or
    is empty. Geometries are not needed — centroids carry the spatial
    information, so the caller can free heavy GeoDataFrames before
    invoking this.
    """
    osm_mask = bbox_mask(osm_centroids_lonlat, chunk.buffered_bbox)
    ov_mask = bbox_mask(
        overture_centroids_lonlat, chunk.buffered_bbox
    )
    osm_global_idx = np.where(osm_mask)[0]
    ov_global_idx = np.where(ov_mask)[0]

    if len(osm_global_idx) == 0 or len(ov_global_idx) == 0:
        return _empty_match_part()

    osm_rad_sub = osm_centroids_rad[osm_global_idx]
    ov_rad_sub = overture_centroids_rad[ov_global_idx]
    osm_radii_sub = osm_radii_m[osm_global_idx]
    osm_labels_sub = osm_shared_labels[osm_global_idx]
    ov_labels_sub = overture_shared_labels[ov_global_idx]
    osm_l0_sub = osm_l0_bits[osm_global_idx]
    ov_l0_sub = overture_l0_bits[ov_global_idx]
    osm_names_sub = osm_names[osm_global_idx]
    osm_brands_sub = osm_brands[osm_global_idx]
    ov_names_sub = overture_names[ov_global_idx]
    ov_brands_sub = overture_brands[ov_global_idx]

    candidates = find_spatial_candidates(
        osm_geom = None,
        overture_geom = None,
        osm_radii_m = osm_radii_sub,
        max_radius_m = max_radius_m,
        chunk_size = chunk_size,
        osm_centroids_rad = osm_rad_sub,
        overture_centroids_rad = ov_rad_sub,
    )
    if candidates.empty:
        return _empty_match_part()

    scored = compute_match_scores(
        candidates = candidates,
        osm_names = osm_names_sub,
        osm_brands = osm_brands_sub,
        overture_names = ov_names_sub,
        overture_brands = ov_brands_sub,
        osm_shared_labels = osm_labels_sub,
        overture_shared_labels = ov_labels_sub,
        osm_radii_m = osm_radii_sub,
        osm_l0_bits = osm_l0_sub,
        overture_l0_bits = ov_l0_sub,
        distance_weight = distance_weight,
        name_weight = name_weight,
        type_weight = type_weight,
        identifier_weight = identifier_weight,
    )

    local_matches = select_best_matches(
        scored, min_score = min_match_score,
    )
    if local_matches.empty:
        return _empty_match_part()

    # Remap local indices to global
    osm_global = osm_global_idx[
        local_matches["osm_idx"].to_numpy()
    ]
    ov_global = ov_global_idx[
        local_matches["overture_idx"].to_numpy()
    ]

    # OSM-anchored emit rule: only this chunk emits pairs for OSM
    # points whose primary chunk is this chunk.
    keep = osm_primary[osm_global] == chunk.chunk_id
    result = pd.DataFrame(
        {
            "osm_idx": osm_global[keep].astype(np.int32),
            "overture_idx": ov_global[keep].astype(np.int32),
            "distance_m": local_matches["distance_m"]
            .to_numpy()[keep]
            .astype(np.float32),
            "distance_score": local_matches["distance_score"]
            .to_numpy()[keep]
            .astype(np.float32),
            "name_score": local_matches["name_score"]
            .to_numpy()[keep]
            .astype(np.float32),
            "type_score": local_matches["type_score"]
            .to_numpy()[keep]
            .astype(np.float32),
            "identifier_score": local_matches["identifier_score"]
            .to_numpy()[keep]
            .astype(np.float32),
            "composite_score": local_matches["composite_score"]
            .to_numpy()[keep]
            .astype(np.float32),
        }
    )
    return result


def _dedup_chunk_parquets(
    checkpoint_dir: Path,
    duckdb_memory_limit: str = "4GB",
) -> pd.DataFrame:
    """
    Stream-dedup chunk match parquets via DuckDB.

    Scans every ``chunk_*.parquet`` under ``checkpoint_dir``, ranks
    rows by ``(composite_score DESC, distance_m ASC, osm_idx ASC)``
    within each ``overture_idx``, and returns the top row per group.
    Uses a bounded DuckDB memory limit so spilling is preferred over
    OOMing — much cheaper than pandas concat + sort + drop_duplicates
    over hundreds of part files.

    Returns an empty DataFrame with the canonical match-part schema
    if no parquet rows exist.
    """
    glob_pattern = str(checkpoint_dir / "chunk_*.parquet")
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        # Probe row count first so we can short-circuit on empty input
        # (read_parquet errors on a glob that matches no files, but a
        # parquet with zero rows is fine).
        n_total = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{glob_pattern}')"
        ).fetchone()[0]
        if n_total == 0:
            return _empty_match_part()

        deduped = con.execute(
            f"""
            SELECT osm_idx, overture_idx, distance_m,
                   distance_score, name_score, type_score,
                   identifier_score, composite_score
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY overture_idx
                        ORDER BY composite_score DESC,
                                 distance_m ASC,
                                 osm_idx ASC
                    ) AS _rn
                FROM read_parquet('{glob_pattern}')
            )
            WHERE _rn = 1
            """
        ).fetch_df()
    finally:
        con.close()

    # Restore canonical narrow dtypes (DuckDB → pandas may widen).
    for col, dt in _MATCH_PART_DTYPES.items():
        if col in deduped.columns:
            deduped[col] = deduped[col].astype(dt)
    return deduped


def find_and_score_matches_chunked(
    osm_geom = None,
    overture_geom = None,
    *,
    osm_radii_m: np.ndarray,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    osm_l0_bits: np.ndarray,
    overture_l0_bits: np.ndarray,
    osm_names: np.ndarray,
    osm_brands: np.ndarray,
    overture_names: np.ndarray,
    overture_brands: np.ndarray,
    distance_weight: float,
    name_weight: float,
    type_weight: float,
    identifier_weight: float,
    min_match_score: float,
    max_radius_m: float,
    chunk_target_pois: int,
    chunk_size: int = 500_000,
    checkpoint_dir: Path | None = None,
    osm_centroids_lonlat: np.ndarray | None = None,
    overture_centroids_lonlat: np.ndarray | None = None,
    duckdb_memory_limit: str = "4GB",
) -> tuple[pd.DataFrame, dict]:
    """
    Spatially chunked matching driver.

    Splits the pooled centroid set into KD-bisected chunks of roughly
    ``chunk_target_pois`` each, runs the full matching pipeline on
    each chunk's buffered subset, and reconciles across chunks:

    - Each chunk emits only matched pairs whose OSM POI has its
      primary chunk equal to the current chunk (``osm-anchored emit``
      rule). This guarantees each matched pair is emitted at most
      once across the whole run.
    - Each chunk's matches are streamed to a parquet file under
      ``checkpoint_dir``; nothing is kept in a Python list. After all
      chunks finish, DuckDB scans the parquet glob and runs the
      per-Overture max-score dedup with a bounded memory budget so
      peak RSS stays small even for hundreds of chunks.
    - A safety assertion verifies each ``osm_idx`` appears at most
      once after dedup.

    Geometry is not needed when centroids are supplied: callers can
    free the source GeoDataFrames before invoking this function and
    pass ``osm_centroids_lonlat`` / ``overture_centroids_lonlat``
    directly. If centroids are not provided, they are computed from
    ``osm_geom`` / ``overture_geom`` (which must then be set).

    If ``checkpoint_dir`` is ``None``, a tempdir is created and
    cleaned up on exit. Pass an explicit dir to enable resume.

    Returns:
        ``(matches, summary)`` where ``matches`` has the
        ``_MATCH_PART_DTYPES`` schema and ``summary`` is a dict of
        observability counters including ``n_chunks``,
        ``min_chunk_pois``, ``max_chunk_pois``, and
        ``n_overture_dedup_drops``, plus the ``osm_primary`` /
        ``overture_primary`` arrays the chunked merge needs.
    """
    # Centroids computed once, reused for chunking + BallTree queries
    if osm_centroids_lonlat is None:
        if osm_geom is None:
            raise ValueError(
                "Provide either osm_geom or osm_centroids_lonlat."
            )
        print("  Precomputing OSM centroids ...")
        osm_centroids_lonlat = extract_centroids_lonlat(
            np.asarray(osm_geom)
        )
    if overture_centroids_lonlat is None:
        if overture_geom is None:
            raise ValueError(
                "Provide either overture_geom or "
                "overture_centroids_lonlat."
            )
        print("  Precomputing Overture centroids ...")
        overture_centroids_lonlat = extract_centroids_lonlat(
            np.asarray(overture_geom)
        )
    # Geometry no longer needed past this point — drop refs so the
    # caller's del + gc.collect can actually free the arrays.
    osm_geom = None
    overture_geom = None

    osm_centroids_rad = lonlat_to_latlon_rad(osm_centroids_lonlat)
    overture_centroids_rad = lonlat_to_latlon_rad(
        overture_centroids_lonlat
    )

    print(
        f"  Computing chunks (target ~{chunk_target_pois:,} "
        f"POIs/chunk, buffer {max_radius_m:.0f}m) ..."
    )
    chunks = compute_chunks(
        osm_centroids_lonlat = osm_centroids_lonlat,
        overture_centroids_lonlat = overture_centroids_lonlat,
        chunk_target_pois = chunk_target_pois,
        buffer_m = max_radius_m,
    )

    osm_primary = assign_primary_chunk(
        osm_centroids_lonlat, chunks,
    )
    # OSM primaries drive the emit rule during matching. Overture
    # primaries are not needed for matching, but are computed here so
    # the downstream per-chunk merge can slice unmatched Overture rows
    # without recomputing centroids.
    overture_primary = assign_primary_chunk(
        overture_centroids_lonlat, chunks,
    )
    osm_counts = np.bincount(
        osm_primary, minlength = len(chunks),
    )
    print(
        f"  {len(chunks)} chunks; OSM POIs per chunk: "
        f"min={int(osm_counts.min()):,}, "
        f"max={int(osm_counts.max()):,}, "
        f"mean={osm_counts.mean():.0f}"
    )

    # Always checkpoint to disk: per-chunk parquets are the input to
    # the DuckDB dedup pass and keep RSS independent of chunk count.
    cleanup_checkpoint = False
    if checkpoint_dir is None:
        checkpoint_dir = Path(
            tempfile.mkdtemp(prefix = "openpois_chunkmatch_")
        )
        cleanup_checkpoint = True
    else:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents = True, exist_ok = True)

    try:
        cumulative_matches = 0
        t_chunk_start = time.time()
        for chunk in chunks:
            part_path = (
                checkpoint_dir
                / f"chunk_{chunk.chunk_id:04d}.parquet"
            )
            if part_path.exists():
                # Resume: count rows for the running tally without
                # loading the frame into RAM.
                cumulative_matches += pq.ParquetFile(
                    part_path
                ).metadata.num_rows
            else:
                part = _match_one_chunk(
                    chunk = chunk,
                    osm_centroids_lonlat = osm_centroids_lonlat,
                    overture_centroids_lonlat = (
                        overture_centroids_lonlat
                    ),
                    osm_centroids_rad = osm_centroids_rad,
                    overture_centroids_rad = overture_centroids_rad,
                    osm_primary = osm_primary,
                    osm_radii_m = osm_radii_m,
                    osm_shared_labels = osm_shared_labels,
                    overture_shared_labels = overture_shared_labels,
                    osm_l0_bits = osm_l0_bits,
                    overture_l0_bits = overture_l0_bits,
                    osm_names = osm_names,
                    osm_brands = osm_brands,
                    overture_names = overture_names,
                    overture_brands = overture_brands,
                    distance_weight = distance_weight,
                    name_weight = name_weight,
                    type_weight = type_weight,
                    identifier_weight = identifier_weight,
                    min_match_score = min_match_score,
                    max_radius_m = max_radius_m,
                    chunk_size = chunk_size,
                )
                part.to_parquet(part_path, compression = "zstd")
                cumulative_matches += len(part)
                del part
                gc.collect()

            done = chunk.chunk_id + 1
            if done % 10 == 0 or done == len(chunks):
                elapsed = time.time() - t_chunk_start
                print(
                    f"  Chunk {done}/{len(chunks)}: "
                    f"{cumulative_matches:,} cumulative matches "
                    f"({elapsed:.0f}s)"
                )

        # Free chunk-loop transients before opening DuckDB.
        del osm_centroids_rad, overture_centroids_rad
        gc.collect()

        print("  Deduplicating chunk matches via DuckDB ...")
        global_matches = _dedup_chunk_parquets(
            checkpoint_dir,
            duckdb_memory_limit = duckdb_memory_limit,
        )
    finally:
        if cleanup_checkpoint:
            for p in checkpoint_dir.glob("chunk_*.parquet"):
                p.unlink()
            checkpoint_dir.rmdir()

    n_overture_dedup_drops = (
        cumulative_matches - len(global_matches)
    )

    if not global_matches.empty and not global_matches[
        "osm_idx"
    ].is_unique:
        dup_count = int(
            (global_matches["osm_idx"].duplicated()).sum()
        )
        raise AssertionError(
            f"{dup_count} duplicate osm_idx values in chunked "
            "matches after Overture dedup. This violates the "
            "OSM-anchored emit invariant and indicates a "
            "non-deterministic primary-chunk assignment."
        )

    summary = {
        "n_chunks": len(chunks),
        "min_chunk_pois": int(osm_counts.min()),
        "max_chunk_pois": int(osm_counts.max()),
        "n_overture_dedup_drops": int(n_overture_dedup_drops),
        # Arrays needed by the downstream chunked merge. Not included
        # in observability prints but passed straight to
        # ``build_merge_parts_chunked``.
        "osm_primary": osm_primary,
        "overture_primary": overture_primary,
    }
    return global_matches, summary
