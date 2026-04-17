#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Merge matched and unmatched POIs into a unified conflated dataset.

Produces a GeoDataFrame superset:
  - Matched pairs (OSM + Overture) with blended confidence.
  - Unmatched OSM POIs with their original confidence.
  - Unmatched Overture POIs with downweighted confidence.

Three entry points:
  - ``merge_matched_pois``: in-memory, for tests/small datasets.
  - ``build_merge_parts``: disk-backed, row-sliced. Writes multiple
    part parquets so peak memory is bounded by slice size.
  - ``build_merge_parts_chunked``: disk-backed, spatial-chunk-sliced.
    Reuses the ``osm_primary`` / ``overture_primary`` arrays produced
    by the chunked matching driver so each per-chunk part is small
    and independent.
"""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely


def _pick_geometries(
    osm_geoms: np.ndarray,
    overture_geoms: np.ndarray,
) -> np.ndarray:
    """
    Vectorized geometry selection: prefer higher-level geometry type,
    OSM on ties.
    """
    osm_types = shapely.get_type_id(osm_geoms)
    ov_types = shapely.get_type_id(overture_geoms)
    rank_table = np.ones(8, dtype = np.int8)
    rank_table[0] = 1  # Point
    rank_table[1] = 2  # LineString
    rank_table[3] = 3  # Polygon
    rank_table[6] = 4  # MultiPolygon
    osm_ranks = rank_table[osm_types]
    ov_ranks = rank_table[ov_types]
    use_overture = ov_ranks > osm_ranks
    result = osm_geoms.copy()
    result[use_overture] = overture_geoms[use_overture]
    return result


def _build_matched_gdf(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    osm_w: float,
    ov_w: float,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for matched pairs."""
    oi = matches["osm_idx"].to_numpy()
    vi = matches["overture_idx"].to_numpy()

    osm_conf = osm_gdf["conf_mean"].to_numpy()[oi].astype(float)
    ov_conf_raw = overture_gdf["confidence"].to_numpy()[vi]
    ov_conf = pd.to_numeric(
        ov_conf_raw, errors = "coerce"
    ).astype(float)
    ov_conf = np.where(np.isnan(ov_conf), 0.5, ov_conf)
    osm_higher = osm_conf >= ov_conf

    osm_names = osm_gdf["name"].to_numpy()[oi]
    ov_names = overture_gdf["overture_name"].to_numpy()[vi]
    names = np.where(
        osm_higher,
        osm_names,
        np.where(pd.notna(ov_names), ov_names, osm_names),
    )

    osm_brands = (
        osm_gdf["brand"].to_numpy()[oi]
        if "brand" in osm_gdf.columns
        else np.full(len(oi), None, dtype = object)
    )
    ov_brands = (
        overture_gdf["brand_name"].to_numpy()[vi]
        if "brand_name" in overture_gdf.columns
        else np.full(len(vi), None, dtype = object)
    )
    brands = np.where(
        osm_higher,
        osm_brands,
        np.where(pd.notna(ov_brands), ov_brands, osm_brands),
    )

    merged_conf = osm_conf * osm_w + ov_conf * ov_w

    osm_conf_lower = osm_gdf["conf_lower"].to_numpy()[oi].astype(
        float
    )
    osm_conf_upper = osm_gdf["conf_upper"].to_numpy()[oi].astype(
        float
    )
    conf_lower = osm_conf_lower * osm_w + ov_conf * ov_w
    conf_upper = osm_conf_upper * osm_w + ov_conf * ov_w

    osm_geoms = osm_gdf.geometry.to_numpy()[oi]
    ov_geoms = overture_gdf.geometry.to_numpy()[vi]
    geoms = _pick_geometries(osm_geoms, ov_geoms)

    osm_ids = osm_gdf["osm_id"].to_numpy()[oi]
    ov_ids = overture_gdf["overture_id"].to_numpy()[vi]

    unified_ids = np.array(
        [
            f"matched:{o}_{v}"
            for o, v in zip(osm_ids, ov_ids)
        ],
        dtype = object,
    )

    return gpd.GeoDataFrame(
        {
            "unified_id": unified_ids,
            "source": "matched",
            "osm_id": osm_ids,
            "overture_id": ov_ids,
            "name": names,
            "brand": brands,
            "shared_label": osm_shared_labels[oi],
            "conf_mean": merged_conf,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "match_score": matches["composite_score"].to_numpy(),
            "match_distance_m": matches["distance_m"].to_numpy(),
            "osm_name": osm_names,
            "overture_name": ov_names,
            "osm_brand": osm_brands,
            "overture_brand": ov_brands,
            "osm_conf_mean": osm_conf,
            "overture_confidence": ov_conf,
        },
        geometry = geoms,
        crs = osm_gdf.crs,
    )


def _build_unmatched_osm_gdf(
    osm_gdf: gpd.GeoDataFrame,
    idx: np.ndarray,
    osm_shared_labels: np.ndarray,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for unmatched OSM POIs at the given indices.

    Uses column-wise ``to_numpy()[idx]`` to avoid a full ``.iloc[idx]``
    copy — the old implementation held both the source frame and the
    full iloc copy in memory simultaneously.
    """
    n = len(idx)

    osm_ids = osm_gdf["osm_id"].to_numpy()[idx]
    names = osm_gdf["name"].to_numpy()[idx]
    brand_arr = (
        osm_gdf["brand"].to_numpy()[idx]
        if "brand" in osm_gdf.columns
        else np.full(n, None, dtype = object)
    )
    conf_mean = osm_gdf["conf_mean"].to_numpy()[idx].astype(float)
    conf_lower = osm_gdf["conf_lower"].to_numpy()[idx].astype(float)
    conf_upper = osm_gdf["conf_upper"].to_numpy()[idx].astype(float)
    geoms = osm_gdf.geometry.to_numpy()[idx]

    unified_ids = np.array(
        [f"osm:{x}" for x in osm_ids], dtype = object,
    )

    return gpd.GeoDataFrame(
        {
            "unified_id": unified_ids,
            "source": "osm",
            "osm_id": osm_ids,
            "overture_id": np.full(n, None, dtype = object),
            "name": names,
            "brand": brand_arr,
            "shared_label": osm_shared_labels[idx],
            "conf_mean": conf_mean,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "match_score": np.full(n, np.nan),
            "match_distance_m": np.full(n, np.nan),
            "osm_name": names,
            "overture_name": np.full(n, None, dtype = object),
            "osm_brand": brand_arr,
            "overture_brand": np.full(n, None, dtype = object),
            "osm_conf_mean": conf_mean,
            "overture_confidence": np.full(n, np.nan),
        },
        geometry = geoms,
        crs = osm_gdf.crs,
    )


def _build_unmatched_overture_gdf(
    overture_gdf: gpd.GeoDataFrame,
    idx: np.ndarray,
    overture_shared_labels: np.ndarray,
    w: float,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for unmatched Overture POIs at the given
    indices.
    """
    n = len(idx)

    ov_ids = overture_gdf["overture_id"].to_numpy()[idx]
    names = overture_gdf["overture_name"].to_numpy()[idx]
    brand_arr = (
        overture_gdf["brand_name"].to_numpy()[idx]
        if "brand_name" in overture_gdf.columns
        else np.full(n, None, dtype = object)
    )
    ov_conf_raw = overture_gdf["confidence"].to_numpy()[idx]
    ov_conf = pd.to_numeric(
        ov_conf_raw, errors = "coerce"
    ).astype(float)
    ov_conf = np.where(np.isnan(ov_conf), 0.5, ov_conf)
    geoms = overture_gdf.geometry.to_numpy()[idx]

    unified_ids = np.array(
        [f"overture:{x}" for x in ov_ids], dtype = object,
    )

    return gpd.GeoDataFrame(
        {
            "unified_id": unified_ids,
            "source": "overture",
            "osm_id": np.full(n, None, dtype = object),
            "overture_id": ov_ids,
            "name": names,
            "brand": brand_arr,
            "shared_label": overture_shared_labels[idx],
            "conf_mean": ov_conf * w,
            "conf_lower": np.full(n, np.nan),
            "conf_upper": np.full(n, np.nan),
            "match_score": np.full(n, np.nan),
            "match_distance_m": np.full(n, np.nan),
            "osm_name": np.full(n, None, dtype = object),
            "overture_name": names,
            "osm_brand": np.full(n, None, dtype = object),
            "overture_brand": brand_arr,
            "osm_conf_mean": np.full(n, np.nan),
            "overture_confidence": ov_conf,
        },
        geometry = geoms,
        crs = overture_gdf.crs,
    )


def _unmatched_idx(
    n: int, matched_idx: np.ndarray,
) -> np.ndarray:
    """Return the sorted indices in ``[0, n)`` not present in
    ``matched_idx``.
    """
    mask = np.ones(n, dtype = bool)
    if len(matched_idx) > 0:
        mask[matched_idx] = False
    return np.where(mask)[0]


# -----------------------------------------------------------------
# In-memory merge (for tests and small datasets)
# -----------------------------------------------------------------


def merge_matched_pois(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    overture_confidence_weight: float = 0.7,
) -> gpd.GeoDataFrame:
    """
    Build the unified conflated dataset from matches + unmatched.

    This in-memory version is suitable for tests and small datasets.
    For large datasets, use ``build_merge_parts`` (row-sliced) or
    ``build_merge_parts_chunked`` (spatial-chunk-sliced) +
    ``save_conflated_from_parts``.

    Returns:
        Conflated GeoDataFrame with unified schema.
    """
    w = overture_confidence_weight
    osm_w = 1.0 / (1.0 + w)
    ov_w = w / (1.0 + w)

    matched_osm_idx = matches["osm_idx"].to_numpy()
    matched_ov_idx = matches["overture_idx"].to_numpy()

    parts = []

    if len(matches) > 0:
        parts.append(
            _build_matched_gdf(
                osm_gdf, overture_gdf, matches,
                osm_shared_labels, osm_w, ov_w,
            )
        )

    parts.append(
        _build_unmatched_osm_gdf(
            osm_gdf,
            _unmatched_idx(len(osm_gdf), matched_osm_idx),
            osm_shared_labels,
        )
    )

    parts.append(
        _build_unmatched_overture_gdf(
            overture_gdf,
            _unmatched_idx(len(overture_gdf), matched_ov_idx),
            overture_shared_labels, w,
        )
    )

    # Normalize CRS across parts — OSM and Overture may declare the
    # same WGS84 lon/lat system with different authority strings
    # (e.g. "WGS 84" vs "WGS 84 (CRS84)"), which breaks pd.concat.
    target_crs = osm_gdf.crs
    for part in parts:
        if part.crs != target_crs:
            part.set_crs(
                target_crs, allow_override = True, inplace = True,
            )

    result = pd.concat(parts, ignore_index = True)
    return gpd.GeoDataFrame(result, crs = target_crs)


# -----------------------------------------------------------------
# Disk-backed merge (for large datasets)
# -----------------------------------------------------------------


def _write_part(
    gdf: gpd.GeoDataFrame, path: Path,
) -> None:
    gdf.to_parquet(path, compression = "zstd")


def _split_indices(
    idx: np.ndarray, n_slices: int,
) -> list[np.ndarray]:
    """Split an index array into ``n_slices`` roughly-equal contiguous
    ranges. Preserves order so downstream concat stays deterministic.
    """
    if n_slices <= 1 or len(idx) == 0:
        return [idx]
    return [s for s in np.array_split(idx, n_slices) if len(s) > 0]


def build_merge_parts(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    overture_confidence_weight: float = 0.7,
    n_slices: int = 4,
) -> list[Path]:
    """
    Build each merge subset, writing to temp parquet files.

    Unmatched OSM and Overture rows are split into ``n_slices``
    contiguous row ranges each, and each slice is built and written
    independently. This caps peak memory at roughly
    ``(1 / n_slices)`` of the full-dataset footprint for unmatched
    parts. The matched part is written as a single file (it is the
    smallest and already bounded by the number of matches).

    Returns:
        List of temp parquet file paths in concat order.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix = "openpois_merge_"))

    w = overture_confidence_weight
    osm_w = 1.0 / (1.0 + w)
    ov_w = w / (1.0 + w)

    matched_osm_idx = matches["osm_idx"].to_numpy()
    matched_ov_idx = matches["overture_idx"].to_numpy()
    part_paths: list[Path] = []

    # Part 1: matched pairs (single file)
    if len(matches) > 0:
        print(f"  Building {len(matches):,} matched pairs ...")
        part = _build_matched_gdf(
            osm_gdf, overture_gdf, matches,
            osm_shared_labels, osm_w, ov_w,
        )
        p = tmp_dir / "1_matched.parquet"
        _write_part(part, p)
        part_paths.append(p)
        del part
        gc.collect()

    # Part 2: unmatched OSM (sliced)
    unmatched_osm = _unmatched_idx(
        len(osm_gdf), matched_osm_idx,
    )
    osm_slices = _split_indices(unmatched_osm, n_slices)
    print(
        f"  Building {len(unmatched_osm):,} unmatched OSM POIs "
        f"in {len(osm_slices)} slice(s) ..."
    )
    for i, sl in enumerate(osm_slices):
        part = _build_unmatched_osm_gdf(
            osm_gdf, sl, osm_shared_labels,
        )
        p = tmp_dir / f"2_unmatched_osm_{i:02d}.parquet"
        _write_part(part, p)
        part_paths.append(p)
        del part
        gc.collect()

    # Part 3: unmatched Overture (sliced)
    unmatched_ov = _unmatched_idx(
        len(overture_gdf), matched_ov_idx,
    )
    ov_slices = _split_indices(unmatched_ov, n_slices)
    print(
        f"  Building {len(unmatched_ov):,} unmatched Overture "
        f"POIs in {len(ov_slices)} slice(s) ..."
    )
    for i, sl in enumerate(ov_slices):
        part = _build_unmatched_overture_gdf(
            overture_gdf, sl,
            overture_shared_labels, w,
        )
        p = tmp_dir / f"3_unmatched_overture_{i:02d}.parquet"
        _write_part(part, p)
        part_paths.append(p)
        del part
        gc.collect()

    return part_paths


def _group_idx_by_chunk(
    idx: np.ndarray,
    primary: np.ndarray,
    n_chunks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a subset of indices and a global ``primary`` array,
    return ``(idx_sorted_by_chunk, chunk_offsets)`` where
    ``chunk_offsets[c:c+2]`` slices out the indices for chunk ``c``.
    """
    if len(idx) == 0:
        return (
            np.empty(0, dtype = np.int64),
            np.zeros(n_chunks + 1, dtype = np.int64),
        )
    chunks = primary[idx]
    order = np.argsort(chunks, kind = "stable")
    idx_sorted = idx[order]
    chunks_sorted = chunks[order]
    offsets = np.searchsorted(
        chunks_sorted, np.arange(n_chunks + 1),
    ).astype(np.int64)
    return idx_sorted, offsets


def build_merge_parts_chunked(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    osm_primary: np.ndarray,
    overture_primary: np.ndarray,
    n_chunks: int,
    overture_confidence_weight: float = 0.7,
) -> list[Path]:
    """
    Build per-spatial-chunk merge parts, writing one parquet per chunk.

    Reuses the KD-bisected chunks produced by the chunked matching
    driver: for each chunk ``c`` we emit matched pairs whose OSM POI
    has ``osm_primary == c`` (the same OSM-anchored emit rule used
    during matching), unmatched OSM POIs with ``osm_primary == c``,
    and unmatched Overture POIs with ``overture_primary == c``.

    Peak memory per chunk is bounded by chunk size × 18-column
    schema, so this stays within a few hundred MB for ~200k-POI
    chunks regardless of total dataset size.

    Args:
        osm_gdf, overture_gdf: Full source frames.
        matches: Post-dedup match DataFrame (osm_idx unique).
        osm_shared_labels, overture_shared_labels: Parallel to source
            frames.
        osm_primary, overture_primary: ``(n,)`` int arrays assigning
            each row to its primary chunk. Produced by
            ``chunking.assign_primary_chunk``.
        n_chunks: Total number of chunks; used for offset arrays.
        overture_confidence_weight: Blend weight ``w`` (see
            ``_build_matched_gdf``).

    Returns:
        List of per-chunk part file paths, in ascending chunk order.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix = "openpois_merge_"))

    w = overture_confidence_weight
    osm_w = 1.0 / (1.0 + w)
    ov_w = w / (1.0 + w)

    # Sort matches by the OSM POI's primary chunk. The OSM-anchored
    # emit rule guarantees osm_idx is unique, so each match belongs to
    # exactly one chunk.
    if len(matches) > 0:
        matched_osm_idx = matches["osm_idx"].to_numpy()
        matched_ov_idx = matches["overture_idx"].to_numpy()
        match_chunk = osm_primary[matched_osm_idx]
        match_order = np.argsort(match_chunk, kind = "stable")
        matches_sorted = matches.iloc[match_order].reset_index(
            drop = True,
        )
        match_chunk_sorted = match_chunk[match_order]
        match_offsets = np.searchsorted(
            match_chunk_sorted, np.arange(n_chunks + 1),
        ).astype(np.int64)
    else:
        matched_osm_idx = np.empty(0, dtype = np.int64)
        matched_ov_idx = np.empty(0, dtype = np.int64)
        matches_sorted = matches
        match_offsets = np.zeros(
            n_chunks + 1, dtype = np.int64,
        )

    unmatched_osm = _unmatched_idx(len(osm_gdf), matched_osm_idx)
    osm_by_chunk, osm_offsets = _group_idx_by_chunk(
        unmatched_osm, osm_primary, n_chunks,
    )
    del unmatched_osm
    gc.collect()

    unmatched_ov = _unmatched_idx(
        len(overture_gdf), matched_ov_idx,
    )
    ov_by_chunk, ov_offsets = _group_idx_by_chunk(
        unmatched_ov, overture_primary, n_chunks,
    )
    del unmatched_ov
    gc.collect()

    part_paths: list[Path] = []
    total_matched = 0
    total_unmatched_osm = 0
    total_unmatched_ov = 0

    print(
        f"  Building {n_chunks} per-chunk merge parts ..."
    )
    for c in range(n_chunks):
        subparts: list[gpd.GeoDataFrame] = []

        m_start, m_end = (
            int(match_offsets[c]), int(match_offsets[c + 1]),
        )
        if m_end > m_start:
            matched_c = matches_sorted.iloc[m_start:m_end]
            subparts.append(
                _build_matched_gdf(
                    osm_gdf, overture_gdf, matched_c,
                    osm_shared_labels, osm_w, ov_w,
                )
            )
            total_matched += m_end - m_start

        o_start, o_end = (
            int(osm_offsets[c]), int(osm_offsets[c + 1]),
        )
        if o_end > o_start:
            osm_idx_c = osm_by_chunk[o_start:o_end]
            subparts.append(
                _build_unmatched_osm_gdf(
                    osm_gdf, osm_idx_c, osm_shared_labels,
                )
            )
            total_unmatched_osm += o_end - o_start

        v_start, v_end = (
            int(ov_offsets[c]), int(ov_offsets[c + 1]),
        )
        if v_end > v_start:
            ov_idx_c = ov_by_chunk[v_start:v_end]
            subparts.append(
                _build_unmatched_overture_gdf(
                    overture_gdf, ov_idx_c,
                    overture_shared_labels, w,
                )
            )
            total_unmatched_ov += v_end - v_start

        if not subparts:
            continue

        # OSM and Overture may be loaded with different CRS
        # representations for the same WGS84 lon/lat system
        # (e.g. "WGS 84" vs "WGS 84 (CRS84)"). Geometries are
        # already in lon/lat on both sides, so force a common CRS
        # without reprojecting.
        target_crs = osm_gdf.crs
        for sp in subparts:
            if sp.crs != target_crs:
                sp.set_crs(
                    target_crs, allow_override = True,
                    inplace = True,
                )

        chunk_gdf = pd.concat(subparts, ignore_index = True)
        p = tmp_dir / f"chunk_{c:04d}.parquet"
        _write_part(
            gpd.GeoDataFrame(chunk_gdf, crs = target_crs), p,
        )
        part_paths.append(p)
        del subparts, chunk_gdf
        gc.collect()

        done = c + 1
        if done % 25 == 0 or done == n_chunks:
            print(
                f"    {done}/{n_chunks} chunks written "
                f"(matched: {total_matched:,}, "
                f"unmatched OSM: {total_unmatched_osm:,}, "
                f"unmatched Overture: {total_unmatched_ov:,})"
            )

    return part_paths


def save_conflated_from_parts(
    part_paths: list[Path],
    output_path: Path,
) -> int:
    """
    Stream temp parquet parts into the final output file.

    Opens each part sequentially, unifies its schema against the
    writer, and appends its row groups. Only one part is held in
    memory at a time, so peak memory is bounded by the largest
    part — independent of the number of parts or the total dataset
    size. Skips Hilbert sorting to stay within memory limits.

    Returns:
        Number of POIs written.
    """
    if not part_paths:
        raise ValueError("No part paths provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    # First pass: read schemas and compute a promoted schema so the
    # writer can accept all parts even if some are missing optional
    # columns or have slightly different null-vs-typed fields.
    schemas = [pq.read_schema(p) for p in part_paths]
    unified_schema = pa.unify_schemas(
        schemas, promote_options = "permissive",
    )

    print(
        f"  Streaming {len(part_paths)} parts into "
        f"{output_path} ..."
    )
    n = 0
    writer = pq.ParquetWriter(
        output_path,
        unified_schema,
        compression = "zstd",
    )
    try:
        for i, p in enumerate(part_paths):
            table = pq.read_table(p)
            # Re-cast columns to the unified schema so row groups
            # written sequentially stay compatible.
            table = table.cast(unified_schema, safe = False)
            writer.write_table(table, row_group_size = 50_000)
            n += table.num_rows
            del table
            gc.collect()
            if (i + 1) % 25 == 0 or (i + 1) == len(part_paths):
                print(
                    f"    {i + 1}/{len(part_paths)} parts written "
                    f"({n:,} rows)"
                )
    finally:
        writer.close()

    # Clean up temp files
    for p in part_paths:
        p.unlink()
    part_paths[0].parent.rmdir()

    print(f"  Done.")
    return n


def save_conflated(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Hilbert-sort and save as GeoParquet (zstd, 50k row groups)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    print("Sorting by Hilbert curve index ...")
    hilbert_order = gdf.hilbert_distance()
    gdf = gdf.iloc[hilbert_order.argsort()].reset_index(drop = True)

    print(f"Saving conflated dataset to {output_path} ...")
    gdf.to_parquet(
        output_path,
        compression = "zstd",
        row_group_size = 50_000,
    )
    print(f"Done. Saved {len(gdf):,} POIs.")
