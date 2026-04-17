"""Tests for openpois.conflation.dedup_overture."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from openpois.conflation.dedup_overture import (
    cluster_pairs_to_components,
    find_self_matches_chunked,
    mark_no_conflate,
    pick_cluster_winners,
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _make_overture_gdf(
    coords: list[tuple[float, float]],
    names: list[str],
    brands: list[str | None] | None = None,
    confidences: list[float] | None = None,
    l1: list[str | None] | None = None,
    l2: list[str | None] | None = None,
) -> gpd.GeoDataFrame:
    n = len(coords)
    data = {
        "overture_id": [f"ov{i}" for i in range(n)],
        "overture_name": names,
        "brand_name": (
            brands if brands is not None else [None] * n
        ),
        "confidence": (
            confidences
            if confidences is not None
            else [0.5] * n
        ),
        "taxonomy_l1": l1 if l1 is not None else [None] * n,
        "taxonomy_l2": l2 if l2 is not None else [None] * n,
        "geometry": [Point(lon, lat) for lon, lat in coords],
    }
    return gpd.GeoDataFrame(data, crs = "EPSG:4326")


# -----------------------------------------------------------------
# Union-find / connected components
# -----------------------------------------------------------------


class TestClusterPairsToComponents:
    def test_no_pairs_singletons(self):
        pairs = pd.DataFrame(
            {"idx_a": np.array([], dtype = np.int32),
             "idx_b": np.array([], dtype = np.int32)}
        )
        comp = cluster_pairs_to_components(pairs, 5)
        assert len(np.unique(comp)) == 5

    def test_triangle_forms_one_component(self):
        # 0-1, 1-2, 0-2 → all in one component
        pairs = pd.DataFrame(
            {
                "idx_a": np.array([0, 1, 0], dtype = np.int32),
                "idx_b": np.array([1, 2, 2], dtype = np.int32),
            }
        )
        comp = cluster_pairs_to_components(pairs, 5)
        assert comp[0] == comp[1] == comp[2]
        # 3 and 4 are singletons, distinct from the triangle
        assert comp[3] != comp[0]
        assert comp[4] != comp[0]
        assert comp[3] != comp[4]

    def test_transitive_chain_unifies(self):
        # 0-1, 1-2, 2-3 chain → one component of size 4
        pairs = pd.DataFrame(
            {
                "idx_a": np.array([0, 1, 2], dtype = np.int32),
                "idx_b": np.array([1, 2, 3], dtype = np.int32),
            }
        )
        comp = cluster_pairs_to_components(pairs, 4)
        assert len(np.unique(comp)) == 1


# -----------------------------------------------------------------
# Winner selection
# -----------------------------------------------------------------


class TestPickClusterWinners:
    def test_singletons_untouched(self):
        comps = np.array([0, 1, 2, 3], dtype = np.int32)
        conf = np.array([0.1, 0.2, 0.3, 0.4], dtype = np.float32)
        compl = np.array([0, 0, 0, 0], dtype = np.int8)
        nc = pick_cluster_winners(comps, conf, compl)
        assert not nc.any()

    def test_higher_confidence_wins(self):
        # Both in component 0; idx 1 has higher confidence.
        comps = np.array([0, 0, 1], dtype = np.int32)
        conf = np.array([0.3, 0.9, 0.5], dtype = np.float32)
        compl = np.array([0, 0, 0], dtype = np.int8)
        nc = pick_cluster_winners(comps, conf, compl)
        assert nc[0] is np.True_ or bool(nc[0]) is True
        assert not bool(nc[1])  # winner
        assert not bool(nc[2])  # singleton

    def test_completeness_tiebreak(self):
        # Equal confidence → more-complete wins.
        comps = np.array([0, 0], dtype = np.int32)
        conf = np.array([0.5, 0.5], dtype = np.float32)
        compl = np.array([1, 3], dtype = np.int8)
        nc = pick_cluster_winners(comps, conf, compl)
        assert bool(nc[0]) is True  # less complete → dropped
        assert bool(nc[1]) is False  # more complete → kept

    def test_index_tiebreak_on_full_tie(self):
        # Equal confidence, equal completeness → lowest index wins.
        comps = np.array([0, 0, 0], dtype = np.int32)
        conf = np.array([0.5, 0.5, 0.5], dtype = np.float32)
        compl = np.array([1, 1, 1], dtype = np.int8)
        nc = pick_cluster_winners(comps, conf, compl)
        assert bool(nc[0]) is False
        assert bool(nc[1]) is True
        assert bool(nc[2]) is True


# -----------------------------------------------------------------
# Self-match chunked driver (small synthetic)
# -----------------------------------------------------------------


class TestFindSelfMatchesChunked:
    def test_two_near_duplicates_make_one_pair(self):
        # Two POIs ~11 m apart with identical name + taxonomy →
        # strong match, emitted once.
        coords = [(-122.335, 47.608), (-122.335, 47.6081)]
        n = 2
        shared_labels = np.array(["bank", "bank"], dtype = object)
        l0_bits = np.array([1, 1], dtype = np.uint16)
        names = np.array(["US Bank", "US Bank"], dtype = object)
        brands = np.array([None, None], dtype = object)
        radii = np.array([100.0, 100.0], dtype = np.float32)

        # compute centroids directly
        from openpois.conflation.chunking import (
            extract_centroids_lonlat,
        )
        centroids = extract_centroids_lonlat(
            np.array([Point(x, y) for x, y in coords])
        )

        pairs, summary = find_self_matches_chunked(
            centroids_lonlat = centroids,
            radii_m = radii,
            shared_labels = shared_labels,
            l0_bits = l0_bits,
            names = names,
            brands = brands,
            min_match_score = 0.5,
            max_radius_m = 100.0,
            chunk_target_pois = 1000,
        )
        assert len(pairs) == 1
        row = pairs.iloc[0]
        assert int(row["idx_a"]) == 0
        assert int(row["idx_b"]) == 1
        assert row["composite_score"] >= 0.75

    def test_far_points_no_pair(self):
        # Two POIs ~1 km apart → beyond 100 m max_radius.
        coords = [(-122.335, 47.608), (-122.335, 47.618)]
        centroids = np.array(
            [[-122.335, 47.608], [-122.335, 47.618]],
            dtype = np.float64,
        )
        n = 2
        pairs, _ = find_self_matches_chunked(
            centroids_lonlat = centroids,
            radii_m = np.array([100.0, 100.0], dtype = np.float32),
            shared_labels = np.array(
                ["bank", "bank"], dtype = object
            ),
            l0_bits = np.array([1, 1], dtype = np.uint16),
            names = np.array(
                ["US Bank", "US Bank"], dtype = object
            ),
            brands = np.array([None, None], dtype = object),
            min_match_score = 0.5,
            max_radius_m = 100.0,
            chunk_target_pois = 1000,
        )
        assert len(pairs) == 0


# -----------------------------------------------------------------
# End-to-end mark_no_conflate
# -----------------------------------------------------------------


class TestMarkNoConflate:
    def test_cluster_of_three_plus_singleton(self):
        # Three US Bank points within ~10-20m at same spot + one
        # unrelated singleton far away. Confidence: idx 1 wins.
        coords = [
            (-122.3350, 47.6080),  # 0
            (-122.3350, 47.60805),  # 1 (~5m from 0)
            (-122.3350, 47.60810),  # 2 (~11m from 0)
            (-122.3000, 47.6080),  # 3 (far)
        ]
        gdf = _make_overture_gdf(
            coords = coords,
            names = ["US Bank", "US Bank", "US Bank", "Other"],
            confidences = [0.5, 0.9, 0.7, 0.5],
            l1 = ["fin", "fin", "fin", "retail"],
            l2 = ["bank", "bank", "bank", "shop"],
        )
        shared_labels = np.array(
            ["bank", "bank", "bank", "shop"], dtype = object,
        )
        l0_bits = np.array([1, 1, 1, 2], dtype = np.uint16)

        no_conflate, components, pairs, summary = mark_no_conflate(
            gdf,
            shared_labels,
            l0_bits,
            min_match_score = 0.5,
            max_radius_m = 100.0,
            chunk_target_pois = 1000,
        )
        # The three US Bank points cluster, with idx 1 the winner.
        assert bool(no_conflate[0]) is True
        assert bool(no_conflate[1]) is False
        assert bool(no_conflate[2]) is True
        # Singleton unchanged.
        assert bool(no_conflate[3]) is False
        assert summary["n_dropped"] == 2
        assert summary["n_multi_clusters"] == 1
        # All three share a component.
        assert components[0] == components[1] == components[2]
        assert components[3] != components[0]

    def test_no_duplicates_no_drops(self):
        coords = [
            (-122.3350, 47.6080),
            (-122.3000, 47.6080),
            (-122.3000, 47.6200),
        ]
        gdf = _make_overture_gdf(
            coords = coords,
            names = ["A", "B", "C"],
        )
        shared_labels = np.array(["", "", ""], dtype = object)
        l0_bits = np.zeros(3, dtype = np.uint16)
        no_conflate, _, _, summary = mark_no_conflate(
            gdf,
            shared_labels,
            l0_bits,
            min_match_score = 0.5,
            max_radius_m = 100.0,
            chunk_target_pois = 1000,
        )
        assert not no_conflate.any()
        assert summary["n_dropped"] == 0
