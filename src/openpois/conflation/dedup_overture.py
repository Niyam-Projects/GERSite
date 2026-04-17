#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Overture-internal deduplication via self-matching.

The Overture Maps snapshot contains near-duplicate POIs — the same
place represented multiple times (different provenance, minor
metadata drift). This module finds clusters of Overture POIs that
likely refer to the same place, picks one "best" representative per
cluster, and marks the rest with ``no_conflate = True`` so they are
excluded from downstream OSM × Overture conflation.

Pipeline:
    1. ``find_self_matches_chunked`` — BallTree radius search inside
       KD-bisected chunks, scoring each pair on distance, name, and
       type taxonomy. Pair emission is anchored to the lower-index
       POI's primary chunk so each pair is emitted exactly once.
    2. ``cluster_pairs_to_components`` — union-find over pairs above
       the match threshold, producing one component id per POI.
    3. ``pick_cluster_winners`` — within each multi-POI component,
       pick the winner by (confidence, completeness, index).
    4. ``mark_no_conflate`` — top-level entry point that glues the
       three phases together and returns a ``no_conflate`` boolean
       array parallel to the input frame.

Re-uses cross-source scoring primitives from
:mod:`openpois.conflation.match` (``compute_name_scores``,
``compute_type_scores``) and chunking primitives from
:mod:`openpois.conflation.chunking` so the memory pattern stays
consistent with the main conflation pass.
"""
from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import BallTree

from openpois.conflation.chunking import (
    ChunkSpec,
    assign_primary_chunk,
    bbox_mask,
    compute_chunks,
    extract_centroids_lonlat,
    lonlat_to_latlon_rad,
)
from openpois.conflation.match import (
    EARTH_RADIUS_M,
    compute_name_scores,
    compute_type_scores,
)

# Self-dedup composite weights. Renormalised from the cross-source
# weights by dropping ``identifier_score`` (a constant placeholder
# today) and rescaling the remaining three to sum to 1.
_DEDUP_DISTANCE_WEIGHT = 0.30
_DEDUP_NAME_WEIGHT = 0.50
_DEDUP_TYPE_WEIGHT = 0.20

_DEDUP_PART_DTYPES = {
    "idx_a": np.int32,
    "idx_b": np.int32,
    "distance_m": np.float32,
    "distance_score": np.float32,
    "name_score": np.float32,
    "type_score": np.float32,
    "composite_score": np.float32,
}


def _empty_dedup_part() -> pd.DataFrame:
    return pd.DataFrame(
        {c: np.array([], dtype = dt)
         for c, dt in _DEDUP_PART_DTYPES.items()}
    )


# -----------------------------------------------------------------
# Per-chunk self-matching
# -----------------------------------------------------------------


def _match_one_self_chunk(
    chunk: ChunkSpec,
    centroids_lonlat: np.ndarray,
    centroids_rad: np.ndarray,
    primary: np.ndarray,
    radii_m: np.ndarray,
    shared_labels: np.ndarray,
    l0_bits: np.ndarray,
    names: np.ndarray,
    brands: np.ndarray,
    min_match_score: float,
    max_radius_m: float,
    chunk_size: int,
) -> pd.DataFrame:
    """
    Find, score, and filter self-matches for one spatial chunk.

    Emit rule: a pair ``(a, b)`` with ``a < b`` (global indices) is
    emitted by this chunk only when ``primary[a] == chunk.chunk_id``.
    That anchors each pair to the smaller endpoint's primary chunk,
    guaranteeing exactly-once emission across the run.
    """
    mask = bbox_mask(centroids_lonlat, chunk.buffered_bbox)
    global_idx = np.where(mask)[0]
    if len(global_idx) < 2:
        return _empty_dedup_part()

    sub_rad = centroids_rad[global_idx]
    # Only query points whose primary chunk is this chunk (they are
    # guaranteed to be in the chunk's core, not just its buffer).
    local_primary_mask = (
        primary[global_idx] == chunk.chunk_id
    )
    query_local = np.where(local_primary_mask)[0]
    if len(query_local) == 0:
        return _empty_dedup_part()

    tree = BallTree(sub_rad, metric = "haversine")
    r_rad = max_radius_m / EARTH_RADIUS_M

    all_a: list[np.ndarray] = []
    all_b: list[np.ndarray] = []
    all_d: list[np.ndarray] = []

    # Query in batches to keep BallTree output per-batch bounded.
    for start in range(0, len(query_local), chunk_size):
        end = min(start + chunk_size, len(query_local))
        q_local = query_local[start:end]
        q_coords = sub_rad[q_local]
        ind, dist = tree.query_radius(
            q_coords, r = r_rad, return_distance = True,
        )
        for k in range(len(q_local)):
            nb_local = ind[k]
            if len(nb_local) == 0:
                continue
            global_i = global_idx[q_local[k]]
            nb_global = global_idx[nb_local]
            # Keep only neighbours with strictly larger global index;
            # also drops the self-hit at distance 0.
            keep = nb_global > global_i
            if not keep.any():
                continue
            nb_g_kept = nb_global[keep]
            d_m = dist[k][keep] * EARTH_RADIUS_M
            all_a.append(
                np.full(
                    len(nb_g_kept), global_i, dtype = np.int64,
                )
            )
            all_b.append(nb_g_kept.astype(np.int64))
            all_d.append(d_m)

    del tree
    if not all_a:
        return _empty_dedup_part()

    pairs = pd.DataFrame(
        {
            "idx_a": np.concatenate(all_a),
            "idx_b": np.concatenate(all_b),
            "distance_m": np.concatenate(all_d),
        }
    )

    # Score: distance (uses max of the two endpoints' radii for a
    # symmetric normalisation), name, type.
    idx_a = pairs["idx_a"].to_numpy()
    idx_b = pairs["idx_b"].to_numpy()
    pair_radii = np.maximum(radii_m[idx_a], radii_m[idx_b])
    distance_score = np.clip(
        1.0 - (pairs["distance_m"].to_numpy() / pair_radii),
        0.0, 1.0,
    )

    name_score = compute_name_scores(
        names, brands, names, brands,
        idx_a, idx_b,
    )
    type_score = compute_type_scores(
        shared_labels, shared_labels,
        l0_bits, l0_bits,
        idx_a, idx_b,
    )

    composite = (
        _DEDUP_DISTANCE_WEIGHT * distance_score
        + _DEDUP_NAME_WEIGHT * name_score
        + _DEDUP_TYPE_WEIGHT * type_score
    )
    keep = composite >= min_match_score
    if not keep.any():
        return _empty_dedup_part()

    return pd.DataFrame(
        {
            "idx_a": idx_a[keep].astype(np.int32),
            "idx_b": idx_b[keep].astype(np.int32),
            "distance_m": pairs["distance_m"].to_numpy()[keep]
            .astype(np.float32),
            "distance_score": distance_score[keep]
            .astype(np.float32),
            "name_score": name_score[keep].astype(np.float32),
            "type_score": type_score[keep].astype(np.float32),
            "composite_score": composite[keep].astype(np.float32),
        }
    )


# -----------------------------------------------------------------
# Chunked driver
# -----------------------------------------------------------------


def _collect_chunk_parquets(
    checkpoint_dir: Path,
    duckdb_memory_limit: str,
) -> pd.DataFrame:
    """Stream-concat per-chunk dedup parquets via DuckDB.

    No per-pair dedup step is needed (the self-anchored emit rule
    guarantees each pair is written exactly once), but DuckDB keeps
    peak RSS bounded when hundreds of parquets are involved.
    """
    glob_pattern = str(checkpoint_dir / "chunk_*.parquet")
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        n_total = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{glob_pattern}')"
        ).fetchone()[0]
        if n_total == 0:
            return _empty_dedup_part()
        df = con.execute(
            f"""
            SELECT idx_a, idx_b, distance_m,
                   distance_score, name_score, type_score,
                   composite_score
            FROM read_parquet('{glob_pattern}')
            """
        ).fetch_df()
    finally:
        con.close()

    for col, dt in _DEDUP_PART_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    return df


def find_self_matches_chunked(
    *,
    centroids_lonlat: np.ndarray,
    radii_m: np.ndarray,
    shared_labels: np.ndarray,
    l0_bits: np.ndarray,
    names: np.ndarray,
    brands: np.ndarray,
    min_match_score: float,
    max_radius_m: float,
    chunk_target_pois: int,
    chunk_size: int = 500_000,
    checkpoint_dir: Path | None = None,
    duckdb_memory_limit: str = "4GB",
) -> tuple[pd.DataFrame, dict]:
    """
    Spatially chunked Overture × Overture self-matching driver.

    Returns ``(pairs, summary)`` where ``pairs`` has the
    ``_DEDUP_PART_DTYPES`` schema, filtered to
    ``composite_score >= min_match_score``. ``summary`` contains
    chunking counters for logging.
    """
    if len(centroids_lonlat) == 0:
        return _empty_dedup_part(), {"n_chunks": 0}

    print(
        f"  Computing dedup chunks (target ~"
        f"{chunk_target_pois:,} POIs/chunk, buffer "
        f"{max_radius_m:.0f}m) ..."
    )
    # ``compute_chunks`` pools two input arrays; for self-dedup we
    # pass the same array twice so each POI is weighted once.
    empty = np.zeros((0, 2), dtype = np.float64)
    chunks = compute_chunks(
        osm_centroids_lonlat = centroids_lonlat,
        overture_centroids_lonlat = empty,
        chunk_target_pois = chunk_target_pois,
        buffer_m = max_radius_m,
    )
    primary = assign_primary_chunk(centroids_lonlat, chunks)
    counts = np.bincount(primary, minlength = len(chunks))
    print(
        f"  {len(chunks)} chunks; POIs per chunk: "
        f"min={int(counts.min()):,}, "
        f"max={int(counts.max()):,}, "
        f"mean={counts.mean():.0f}"
    )

    centroids_rad = lonlat_to_latlon_rad(centroids_lonlat)

    cleanup_checkpoint = False
    if checkpoint_dir is None:
        checkpoint_dir = Path(
            tempfile.mkdtemp(prefix = "openpois_selfdedup_")
        )
        cleanup_checkpoint = True
    else:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents = True, exist_ok = True)

    try:
        cumulative_pairs = 0
        t_chunk_start = time.time()
        for chunk in chunks:
            part_path = (
                checkpoint_dir
                / f"chunk_{chunk.chunk_id:04d}.parquet"
            )
            if part_path.exists():
                cumulative_pairs += pq.ParquetFile(
                    part_path
                ).metadata.num_rows
            else:
                part = _match_one_self_chunk(
                    chunk = chunk,
                    centroids_lonlat = centroids_lonlat,
                    centroids_rad = centroids_rad,
                    primary = primary,
                    radii_m = radii_m,
                    shared_labels = shared_labels,
                    l0_bits = l0_bits,
                    names = names,
                    brands = brands,
                    min_match_score = min_match_score,
                    max_radius_m = max_radius_m,
                    chunk_size = chunk_size,
                )
                part.to_parquet(part_path, compression = "zstd")
                cumulative_pairs += len(part)
                del part
                gc.collect()

            done = chunk.chunk_id + 1
            if done % 10 == 0 or done == len(chunks):
                elapsed = time.time() - t_chunk_start
                print(
                    f"  Chunk {done}/{len(chunks)}: "
                    f"{cumulative_pairs:,} cumulative pairs "
                    f"({elapsed:.0f}s)"
                )

        del centroids_rad
        gc.collect()

        print("  Concatenating chunk parquets via DuckDB ...")
        pairs = _collect_chunk_parquets(
            checkpoint_dir, duckdb_memory_limit,
        )
    finally:
        if cleanup_checkpoint:
            for p in checkpoint_dir.glob("chunk_*.parquet"):
                p.unlink()
            checkpoint_dir.rmdir()

    summary = {
        "n_chunks": len(chunks),
        "min_chunk_pois": int(counts.min()),
        "max_chunk_pois": int(counts.max()),
        "n_pairs": int(len(pairs)),
    }
    return pairs, summary


# -----------------------------------------------------------------
# Union-find over pairs
# -----------------------------------------------------------------


def cluster_pairs_to_components(
    pairs: pd.DataFrame,
    n_nodes: int,
) -> np.ndarray:
    """
    Connected-components over ``pairs`` producing a component id per
    node.

    ``pairs`` must have ``idx_a`` / ``idx_b`` int32 columns. Returns
    an ``(n_nodes,)`` int32 array of dense 0-based component labels.
    Singleton nodes each get their own label. Implemented via
    ``scipy.sparse.csgraph.connected_components`` — much faster than
    a Python union-find loop at 10M+ nodes.
    """
    if n_nodes == 0:
        return np.zeros(0, dtype = np.int32)
    if len(pairs) == 0:
        return np.arange(n_nodes, dtype = np.int32)

    a = pairs["idx_a"].to_numpy()
    b = pairs["idx_b"].to_numpy()
    data = np.ones(len(a), dtype = np.int8)
    graph = csr_matrix(
        (data, (a, b)), shape = (n_nodes, n_nodes),
    )
    _, labels = connected_components(
        graph, directed = False, return_labels = True,
    )
    return labels.astype(np.int32)


# -----------------------------------------------------------------
# Winner selection
# -----------------------------------------------------------------


def _completeness_score(overture_gdf: pd.DataFrame) -> np.ndarray:
    """Count non-null fields among a fixed set of columns.

    Used as a tiebreak when two POIs in a cluster have equal
    ``confidence``. Missing columns count as all-null (contribute 0).
    """
    cols = [
        "overture_name", "brand_name",
        "taxonomy_l1", "taxonomy_l2",
    ]
    score = np.zeros(len(overture_gdf), dtype = np.int8)
    for col in cols:
        if col not in overture_gdf.columns:
            continue
        score = score + overture_gdf[col].notna().to_numpy(
            dtype = np.int8
        )
    return score


def _coerce_confidence(overture_gdf: pd.DataFrame) -> np.ndarray:
    """
    Coerce Overture's ``confidence`` column to ``float32``. Missing or
    non-numeric entries are treated as -1.0 so they sort below any
    real confidence value.
    """
    if "confidence" not in overture_gdf.columns:
        return np.full(len(overture_gdf), -1.0, dtype = np.float32)
    conf = pd.to_numeric(
        overture_gdf["confidence"], errors = "coerce",
    )
    return conf.fillna(-1.0).to_numpy(dtype = np.float32)


def pick_cluster_winners(
    components: np.ndarray,
    confidence: np.ndarray,
    completeness: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean ``no_conflate`` array parallel to ``components``.

    Within each component (identified by shared root index), pick one
    winner by ``(confidence DESC, completeness DESC, node_idx ASC)``
    and mark every other member ``no_conflate = True``. Singleton
    components are untouched.
    """
    n = len(components)
    no_conflate = np.zeros(n, dtype = bool)
    if n == 0:
        return no_conflate

    # Group by component and find sizes.
    order = np.argsort(components, kind = "stable")
    sorted_comp = components[order]
    # Boundaries between component groups.
    boundaries = np.concatenate(
        [
            [0],
            np.where(sorted_comp[1:] != sorted_comp[:-1])[0] + 1,
            [n],
        ]
    )

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue  # singleton
        members = order[start:end]
        conf = confidence[members]
        comp = completeness[members]
        # Lexicographic sort: confidence DESC, completeness DESC,
        # member idx ASC. np.lexsort uses LAST key as primary, and
        # sorts ascending — negate to get descending.
        order_in_cluster = np.lexsort(
            (members, -comp.astype(np.int32), -conf)
        )
        winner_local = order_in_cluster[0]
        loser_mask = np.ones(len(members), dtype = bool)
        loser_mask[winner_local] = False
        no_conflate[members[loser_mask]] = True

    return no_conflate


# -----------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------


def mark_no_conflate(
    overture_gdf,
    shared_labels: np.ndarray,
    l0_bits: np.ndarray,
    *,
    min_match_score: float,
    max_radius_m: float,
    chunk_target_pois: int,
    chunk_size: int = 500_000,
    checkpoint_dir: Path | None = None,
    duckdb_memory_limit: str = "4GB",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """
    Mark Overture POIs as duplicates via self-matching.

    Returns ``(no_conflate, components, pairs, summary)``:
        - ``no_conflate``: ``(N,)`` bool array parallel to
          ``overture_gdf``. ``True`` means the POI lost to another
          POI in its cluster and should be dropped from downstream
          conflation.
        - ``components``: ``(N,)`` int32 array of dense 0-based
          cluster labels (singletons included).
        - ``pairs``: scored self-match pairs at or above
          ``min_match_score`` with a ``component_id`` column.
        - ``summary``: dict with chunking / cluster counters.
    """
    n = len(overture_gdf)
    if n == 0:
        return (
            np.zeros(0, dtype = bool),
            np.zeros(0, dtype = np.int32),
            _empty_dedup_part(),
            {"n_clusters": 0, "n_dropped": 0, "n_pairs": 0},
        )

    print(
        "\nOverture internal deduplication "
        f"(min_match_score={min_match_score}, "
        f"max_radius_m={max_radius_m}) ..."
    )
    centroids_lonlat = extract_centroids_lonlat(
        np.asarray(overture_gdf.geometry.values)
    )
    # Per-POI radius: use the taxonomy-derived match_radius_m capped
    # at max_radius_m. For type scoring only (distance search uses
    # max_radius_m uniformly for symmetric candidate finding).
    # ``shared_labels`` → radius mapping lives upstream; here we
    # trust ``radii_m`` is supplied at full-snapshot scale.
    # But the signature only passes shared_labels; we rely on the
    # distance_score normalisation using max(r_a, r_b) internally.
    radii_m = np.full(n, max_radius_m, dtype = np.float32)

    names = overture_gdf["overture_name"].to_numpy()
    brands = (
        overture_gdf["brand_name"].to_numpy()
        if "brand_name" in overture_gdf.columns
        else np.full(n, None, dtype = object)
    )

    pairs, chunk_summary = find_self_matches_chunked(
        centroids_lonlat = centroids_lonlat,
        radii_m = radii_m,
        shared_labels = shared_labels,
        l0_bits = l0_bits,
        names = names,
        brands = brands,
        min_match_score = min_match_score,
        max_radius_m = max_radius_m,
        chunk_target_pois = chunk_target_pois,
        chunk_size = chunk_size,
        checkpoint_dir = checkpoint_dir,
        duckdb_memory_limit = duckdb_memory_limit,
    )
    del centroids_lonlat, names, brands, radii_m
    gc.collect()

    print(
        f"  {len(pairs):,} candidate duplicate pairs above "
        f"threshold."
    )

    if len(pairs) == 0:
        no_conflate = np.zeros(n, dtype = bool)
        components = np.arange(n, dtype = np.int32)
        summary = {
            **chunk_summary,
            "n_clusters": int(n),
            "n_multi_clusters": 0,
            "n_dropped": 0,
        }
        return no_conflate, components, pairs, summary

    components = cluster_pairs_to_components(pairs, n)

    confidence = _coerce_confidence(overture_gdf)
    completeness = _completeness_score(overture_gdf)

    no_conflate = pick_cluster_winners(
        components, confidence, completeness,
    )
    n_dropped = int(no_conflate.sum())

    # Attach component ids to pairs for the audit artefact.
    pairs = pairs.copy()
    pairs["component_id"] = components[
        pairs["idx_a"].to_numpy()
    ]

    # Cluster-size histogram for observability.
    unique_comps, comp_sizes = np.unique(
        components, return_counts = True,
    )
    multi = comp_sizes[comp_sizes >= 2]
    if len(multi) > 0:
        print(
            f"  {len(multi):,} duplicate clusters "
            f"(sizes: min={int(multi.min())}, "
            f"max={int(multi.max())}, "
            f"mean={multi.mean():.1f}); "
            f"{n_dropped:,} POIs marked no_conflate."
        )
    else:
        print("  No multi-POI clusters formed.")

    summary = {
        **chunk_summary,
        "n_clusters": int(len(unique_comps)),
        "n_multi_clusters": int(len(multi)),
        "n_dropped": n_dropped,
    }
    return no_conflate, components, pairs, summary
