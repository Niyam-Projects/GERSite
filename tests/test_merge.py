"""Tests for openpois.conflation.merge."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from openpois.conflation.merge import (
    build_merge_parts,
    build_merge_parts_chunked,
    merge_matched_pois,
    save_conflated_from_parts,
)


@pytest.fixture
def osm_gdf():
    return gpd.GeoDataFrame(
        {
            "osm_id": [100, 200, 300],
            "osm_type": ["node", "way", "relation"],
            "name": ["Coffee Shop", "Grocery Store", "Park"],
            "brand": ["Starbucks", None, None],
            "conf_mean": [0.8, 0.6, 0.9],
            "conf_lower": [0.7, 0.5, 0.85],
            "conf_upper": [0.9, 0.7, 0.95],
        },
        geometry = [
            Point(-122.335, 47.608),
            Point(-122.340, 47.610),
            Polygon([
                (-122.33, 47.60), (-122.33, 47.61),
                (-122.32, 47.61), (-122.32, 47.60),
                (-122.33, 47.60),
            ]),
        ],
        crs = "EPSG:4326",
    )


@pytest.fixture
def overture_gdf():
    return gpd.GeoDataFrame(
        {
            "overture_id": ["ov_a", "ov_b", "ov_c"],
            "overture_name": [
                "Starbucks Coffee", "Fresh Market", "City Park",
            ],
            "brand_name": ["Starbucks", None, None],
            "confidence": [0.95, 0.7, 0.8],
        },
        geometry = [
            Point(-122.3351, 47.6081),
            Point(-122.350, 47.620),
            Point(-122.325, 47.605),
        ],
        crs = "EPSG:4326",
    )


@pytest.fixture
def matches():
    return pd.DataFrame(
        {
            "osm_idx": [0],
            "overture_idx": [0],
            "distance_m": [15.0],
            "composite_score": [0.85],
        }
    )


class TestMergeMatchedPois:
    def test_output_has_all_sources(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
            overture_confidence_weight = 0.7,
        )
        sources = set(result["source"].unique())
        assert sources == {"matched", "osm", "overture"}

    def test_total_count(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        # 1 matched + 2 unmatched OSM + 2 unmatched Overture = 5
        assert len(result) == 5

    def test_confidence_blending(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        w = 0.7
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
            overture_confidence_weight = w,
        )
        matched_row = result[
            result["source"] == "matched"
        ].iloc[0]
        osm_conf = 0.8
        ov_conf = 0.95
        expected = osm_conf / (1 + w) + ov_conf * w / (1 + w)
        assert matched_row["conf_mean"] == pytest.approx(
            expected, abs = 0.01
        )

    def test_unmatched_overture_downweighted(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        w = 0.7
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
            overture_confidence_weight = w,
        )
        unmatched_ov = result[result["source"] == "overture"]
        for _, row in unmatched_ov.iterrows():
            assert row["conf_mean"] == pytest.approx(
                row["overture_confidence"] * w, abs = 0.01
            )

    def test_unmatched_osm_keeps_confidence(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        unmatched_osm = result[result["source"] == "osm"]
        # OSM ids 200 and 300 should be unmatched
        for _, row in unmatched_osm.iterrows():
            assert row["conf_mean"] == row["osm_conf_mean"]

    def test_geometry_preference_polygon_over_point(self):
        """When OSM has a Polygon and Overture has a Point,
        the merged geometry should be the Polygon."""
        poly = Polygon([
            (-122.33, 47.60), (-122.33, 47.61),
            (-122.32, 47.61), (-122.32, 47.60),
            (-122.33, 47.60),
        ])
        osm = gpd.GeoDataFrame(
            {
                "osm_id": [1],
                "osm_type": ["way"],
                "name": ["Park"],
                "brand": [None],
                "conf_mean": [0.9],
                "conf_lower": [0.85],
                "conf_upper": [0.95],
            },
            geometry = [poly],
            crs = "EPSG:4326",
        )
        overture = gpd.GeoDataFrame(
            {
                "overture_id": ["ov_1"],
                "overture_name": ["City Park"],
                "brand_name": [None],
                "confidence": [0.8],
            },
            geometry = [Point(-122.325, 47.605)],
            crs = "EPSG:4326",
        )
        m = pd.DataFrame(
            {
                "osm_idx": [0],
                "overture_idx": [0],
                "distance_m": [50.0],
                "composite_score": [0.75],
            }
        )
        result = merge_matched_pois(
            osm, overture, m,
            np.array(["Park"]), np.array(["Park"]),
        )
        matched = result[
            result["source"] == "matched"
        ].iloc[0]
        assert matched.geometry.geom_type == "Polygon"

    def test_expected_columns(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        expected_cols = {
            "unified_id", "source", "osm_id", "osm_type",
            "overture_id",
            "name", "brand", "shared_label",
            "conf_mean", "conf_lower", "conf_upper",
            "match_score", "match_distance_m",
            "osm_name", "overture_name",
            "osm_brand", "overture_brand",
            "osm_conf_mean", "overture_confidence",
            "geometry",
        }
        assert set(result.columns) == expected_cols

    def test_shared_label_values(
        self, osm_gdf, overture_gdf, matches,
    ):
        osm_labels = np.array(
            ["Cafe", "Supermarket", "Park"]
        )
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        assert "shared_label" in result.columns
        matched = result[
            result["source"] == "matched"
        ].iloc[0]
        assert matched["shared_label"] == "Cafe"
        # Unmatched Park should retain its label
        park_row = result[
            result["shared_label"] == "Park"
        ].iloc[0]
        assert park_row["shared_label"] == "Park"

    def test_osm_type_propagation(
        self, osm_gdf, overture_gdf, matches,
    ):
        """osm_type must flow through matched + unmatched-OSM rows and
        be null on unmatched-Overture rows so the frontend can route
        OpenStreetMap links to /node/, /way/, or /relation/ correctly.
        """
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        # Matched row (osm_idx=0 → node)
        matched = result[result["source"] == "matched"].iloc[0]
        assert matched["osm_type"] == "node"
        # Unmatched-OSM rows (idx 1,2 → way, relation)
        osm_only = result[result["source"] == "osm"].set_index("osm_id")
        assert osm_only.loc[200, "osm_type"] == "way"
        assert osm_only.loc[300, "osm_type"] == "relation"
        # Unmatched-Overture rows carry no osm_type
        ov_only = result[result["source"] == "overture"]
        assert ov_only["osm_type"].isna().all()


# -----------------------------------------------------------------
# Disk-backed row-sliced merge
# -----------------------------------------------------------------


def _read_parts(part_paths):
    """Concatenate part parquets into a single GeoDataFrame."""
    dfs = [gpd.read_parquet(p) for p in part_paths]
    return gpd.GeoDataFrame(
        pd.concat(dfs, ignore_index = True),
        crs = dfs[0].crs,
    )


class TestBuildMergeParts:
    def test_matches_in_memory_result(
        self, osm_gdf, overture_gdf, matches, tmp_path,
    ):
        """Row-sliced merge parts must reconstruct the same GDF as
        ``merge_matched_pois`` (order may differ; compare as a set)."""
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        in_mem = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        paths = build_merge_parts(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            n_slices = 2,
        )
        try:
            disk = _read_parts(paths)
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()

        assert set(disk.columns) == set(in_mem.columns)
        assert len(disk) == len(in_mem)
        assert (
            set(disk["unified_id"])
            == set(in_mem["unified_id"])
        )

    def test_empty_matches(
        self, osm_gdf, overture_gdf, tmp_path,
    ):
        """With zero matches, no matched part is written and all
        source rows appear as unmatched in the output."""
        empty = pd.DataFrame(
            {
                "osm_idx": pd.array([], dtype = np.int64),
                "overture_idx": pd.array([], dtype = np.int64),
                "distance_m": pd.array([], dtype = np.float64),
                "composite_score": pd.array(
                    [], dtype = np.float64
                ),
            }
        )
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        paths = build_merge_parts(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = empty,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            n_slices = 3,
        )
        try:
            disk = _read_parts(paths)
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()

        assert not any(
            "matched" == str(p.name).split("_")[1]
            for p in paths
        )
        assert len(disk) == len(osm_gdf) + len(overture_gdf)
        assert set(disk["source"].unique()) == {"osm", "overture"}

    def test_slice_count_produces_expected_parts(
        self, osm_gdf, overture_gdf, matches,
    ):
        """n_slices=3 with 2 unmatched rows on each side yields
        2 single-row slices per source (empty slices are dropped)."""
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        paths = build_merge_parts(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            n_slices = 3,
        )
        try:
            names = sorted(p.name for p in paths)
            # 1 matched + at most 3 osm + at most 3 overture
            assert names[0] == "1_matched.parquet"
            osm_parts = [
                n for n in names if n.startswith("2_")
            ]
            ov_parts = [
                n for n in names if n.startswith("3_")
            ]
            assert 1 <= len(osm_parts) <= 3
            assert 1 <= len(ov_parts) <= 3
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()


# -----------------------------------------------------------------
# Disk-backed per-chunk merge
# -----------------------------------------------------------------


class TestBuildMergePartsChunked:
    def test_matches_in_memory_result(
        self, osm_gdf, overture_gdf, matches,
    ):
        """Per-chunk merge must produce the same unified set as the
        in-memory ``merge_matched_pois`` when chunking is applied."""
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])

        # Assign every row to chunk 0 (trivial single-chunk case).
        osm_primary = np.zeros(len(osm_gdf), dtype = np.int32)
        ov_primary = np.zeros(len(overture_gdf), dtype = np.int32)

        in_mem = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        paths = build_merge_parts_chunked(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            osm_primary = osm_primary,
            overture_primary = ov_primary,
            n_chunks = 1,
        )
        try:
            disk = _read_parts(paths)
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()

        assert len(disk) == len(in_mem)
        assert (
            set(disk["unified_id"])
            == set(in_mem["unified_id"])
        )

    def test_multi_chunk_partitions_cleanly(
        self, osm_gdf, overture_gdf, matches,
    ):
        """Two chunks: each unified_id appears in exactly one part,
        and the union equals the in-memory result."""
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])

        # OSM rows 0,1 → chunk 0; row 2 → chunk 1.
        # Overture rows 0 → chunk 0; 1,2 → chunk 1.
        osm_primary = np.array([0, 0, 1], dtype = np.int32)
        ov_primary = np.array([0, 1, 1], dtype = np.int32)

        in_mem = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        paths = build_merge_parts_chunked(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            osm_primary = osm_primary,
            overture_primary = ov_primary,
            n_chunks = 2,
        )
        try:
            per_chunk = [gpd.read_parquet(p) for p in paths]
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()

        assert len(paths) == 2
        # Disjoint coverage of unified_ids
        id_sets = [set(df["unified_id"]) for df in per_chunk]
        assert id_sets[0].isdisjoint(id_sets[1])
        assert (
            id_sets[0] | id_sets[1]
            == set(in_mem["unified_id"])
        )

    def test_matched_emitted_once_by_osm_primary(
        self, osm_gdf, overture_gdf, matches,
    ):
        """The matched pair (OSM idx 0, Overture idx 0) must be
        emitted by OSM's primary chunk, not Overture's — even when
        they disagree."""
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])

        osm_primary = np.array([1, 0, 0], dtype = np.int32)
        ov_primary = np.array([0, 0, 1], dtype = np.int32)

        paths = build_merge_parts_chunked(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            osm_primary = osm_primary,
            overture_primary = ov_primary,
            n_chunks = 2,
        )
        try:
            per_chunk = [gpd.read_parquet(p) for p in paths]
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()

        # Two chunks, both should have parts (OSM 0 in chunk 1,
        # OSM 1,2 in chunk 0, Overture 0,1 vary).
        by_chunk = {
            int(p.name.split("_")[1].split(".")[0]): df
            for p, df in zip(paths, per_chunk)
        }
        matched_chunks = {
            c for c, df in by_chunk.items()
            if (df["source"] == "matched").any()
        }
        # Matched pair's OSM is at idx 0 → osm_primary[0] = 1.
        assert matched_chunks == {1}


class TestMixedCrsInputs:
    """OSM and Overture can load with different CRS strings for the
    same underlying WGS84 lon/lat system ("WGS 84" vs
    "WGS 84 (CRS84)"). All merge entry points must tolerate that
    without raising a CRS mismatch."""

    @pytest.fixture
    def osm_wgs84(self, osm_gdf):
        return osm_gdf.set_crs(
            "EPSG:4326", allow_override = True,
        )

    @pytest.fixture
    def overture_crs84(self, overture_gdf):
        return overture_gdf.set_crs(
            "OGC:CRS84", allow_override = True,
        )

    def test_merge_matched_pois_handles_mixed_crs(
        self, osm_wgs84, overture_crs84, matches,
    ):
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        result = merge_matched_pois(
            osm_wgs84, overture_crs84, matches,
            osm_labels, ov_labels,
        )
        assert len(result) == 5
        assert result.crs == osm_wgs84.crs

    def test_build_merge_parts_chunked_handles_mixed_crs(
        self, osm_wgs84, overture_crs84, matches,
    ):
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        osm_primary = np.zeros(
            len(osm_wgs84), dtype = np.int32,
        )
        ov_primary = np.zeros(
            len(overture_crs84), dtype = np.int32,
        )
        paths = build_merge_parts_chunked(
            osm_gdf = osm_wgs84,
            overture_gdf = overture_crs84,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            osm_primary = osm_primary,
            overture_primary = ov_primary,
            n_chunks = 1,
        )
        try:
            disk = gpd.read_parquet(paths[0])
        finally:
            for p in paths:
                p.unlink(missing_ok = True)
            paths[0].parent.rmdir()

        assert len(disk) == 5
        assert disk.crs == osm_wgs84.crs


class TestSaveConflatedFromParts:
    def test_streams_multiple_parts(
        self, osm_gdf, overture_gdf, matches, tmp_path,
    ):
        """Streaming writer reconstructs the full GDF from per-chunk
        parts and returns the correct row count."""
        osm_labels = np.array(["Cafe", "Supermarket", "Park"])
        ov_labels = np.array(["Cafe", "Supermarket", "Park"])
        osm_primary = np.array([0, 0, 1], dtype = np.int32)
        ov_primary = np.array([0, 1, 1], dtype = np.int32)

        in_mem = merge_matched_pois(
            osm_gdf, overture_gdf, matches,
            osm_labels, ov_labels,
        )
        paths = build_merge_parts_chunked(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_labels,
            overture_shared_labels = ov_labels,
            osm_primary = osm_primary,
            overture_primary = ov_primary,
            n_chunks = 2,
        )

        out = tmp_path / "conflated.parquet"
        n = save_conflated_from_parts(paths, out)

        assert n == len(in_mem)
        assert out.exists()
        round_trip = gpd.read_parquet(out)
        assert len(round_trip) == len(in_mem)
        assert (
            set(round_trip["unified_id"])
            == set(in_mem["unified_id"])
        )
        # Temp files and tmp_dir should be removed after streaming.
        for p in paths:
            assert not p.exists()
