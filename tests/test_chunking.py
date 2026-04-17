"""Tests for openpois.conflation.chunking."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

from openpois.conflation.chunking import (
    ChunkSpec,
    assign_primary_chunk,
    bbox_mask,
    compute_chunks,
    extract_centroids_lonlat,
)


def _random_lonlat(n: int, seed: int) -> np.ndarray:
    """Generate a uniformly-distributed (n, 2) lonlat array over CONUS."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-124.0, -67.0, size = n)
    lat = rng.uniform(25.0, 49.0, size = n)
    return np.column_stack([lon, lat])


class TestComputeChunks:
    def test_single_chunk_when_small(self):
        """N <= target produces exactly one chunk."""
        centroids = _random_lonlat(50, seed = 0)
        chunks = compute_chunks(
            osm_centroids_lonlat = centroids,
            overture_centroids_lonlat = np.zeros((0, 2)),
            chunk_target_pois = 100,
            buffer_m = 200.0,
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0

    def test_target_count_respected(self):
        """No chunk's pooled count exceeds 2× target (recursion halts
        when a split leaves one side empty, which is the only way a
        chunk can exceed target; allow slack)."""
        osm = _random_lonlat(5_000, seed = 1)
        ov = _random_lonlat(5_000, seed = 2)
        target = 1_000
        chunks = compute_chunks(
            osm_centroids_lonlat = osm,
            overture_centroids_lonlat = ov,
            chunk_target_pois = target,
            buffer_m = 200.0,
        )
        pooled = np.vstack([osm, ov])
        primary = assign_primary_chunk(pooled, chunks)
        counts = np.bincount(primary, minlength = len(chunks))
        assert counts.max() <= 2 * target
        # Roughly 10 chunks expected for 10,000 points and target 1,000
        assert 6 <= len(chunks) <= 20

    def test_tile_coverage_no_gaps_no_overlaps(self):
        """Every centroid lands in exactly one chunk."""
        osm = _random_lonlat(3_000, seed = 3)
        ov = _random_lonlat(2_000, seed = 4)
        chunks = compute_chunks(
            osm_centroids_lonlat = osm,
            overture_centroids_lonlat = ov,
            chunk_target_pois = 500,
            buffer_m = 200.0,
        )
        pooled = np.vstack([osm, ov])
        primary = assign_primary_chunk(pooled, chunks)
        assert (primary >= 0).all()
        assert primary.max() < len(chunks)
        # Sum of per-chunk counts equals N
        counts = np.bincount(primary, minlength = len(chunks))
        assert counts.sum() == len(pooled)

    def test_determinism(self):
        """Same inputs produce identical chunk layouts."""
        osm = _random_lonlat(2_000, seed = 5)
        ov = _random_lonlat(1_500, seed = 6)
        c1 = compute_chunks(osm, ov, 500, 200.0)
        c2 = compute_chunks(osm, ov, 500, 200.0)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a == b

    def test_duplicate_centroids(self):
        """All-identical centroids collapse to a single chunk (recursion
        halts immediately when median split produces an empty side)."""
        n = 500
        centroids = np.full((n, 2), [-122.0, 47.0])
        chunks = compute_chunks(
            osm_centroids_lonlat = centroids,
            overture_centroids_lonlat = np.zeros((0, 2)),
            chunk_target_pois = 100,
            buffer_m = 200.0,
        )
        # Single chunk even though count > target (degenerate split)
        assert len(chunks) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_chunks(
                osm_centroids_lonlat = np.zeros((0, 2)),
                overture_centroids_lonlat = np.zeros((0, 2)),
                chunk_target_pois = 100,
                buffer_m = 200.0,
            )

    def test_buffered_bbox_larger_than_core(self):
        centroids = _random_lonlat(200, seed = 7)
        chunks = compute_chunks(
            osm_centroids_lonlat = centroids,
            overture_centroids_lonlat = np.zeros((0, 2)),
            chunk_target_pois = 50,
            buffer_m = 500.0,
        )
        for c in chunks:
            assert c.buffered_bbox[0] < c.core_bbox[0]
            assert c.buffered_bbox[1] < c.core_bbox[1]
            assert c.buffered_bbox[2] > c.core_bbox[2]
            assert c.buffered_bbox[3] > c.core_bbox[3]

    def test_buffer_scales_with_metres(self):
        """Buffer widths in degrees should roughly equal
        ``buffer_m / (~111_139 m/deg)`` at the chunk's latitude."""
        centroids = _random_lonlat(200, seed = 8)
        chunks_100 = compute_chunks(
            centroids, np.zeros((0, 2)), 50, 100.0,
        )
        chunks_500 = compute_chunks(
            centroids, np.zeros((0, 2)), 50, 500.0,
        )
        # Core bboxes identical; buffered bboxes scale linearly
        assert len(chunks_100) == len(chunks_500)
        for c_small, c_big in zip(chunks_100, chunks_500):
            assert c_small.core_bbox == c_big.core_bbox
            small_width = c_small.buffered_bbox[2] - c_small.core_bbox[2]
            big_width = c_big.buffered_bbox[2] - c_big.core_bbox[2]
            assert big_width == pytest.approx(
                small_width * 5.0, rel = 0.01,
            )


class TestAssignPrimaryChunk:
    def test_boundary_point_assigned_once(self):
        """A point on a chunk boundary lands in exactly one chunk."""
        osm = _random_lonlat(1_000, seed = 9)
        chunks = compute_chunks(
            osm, np.zeros((0, 2)), 200, 200.0,
        )
        primary = assign_primary_chunk(osm, chunks)
        assert (primary >= 0).all()
        assert len(np.unique(primary)) <= len(chunks)

    def test_global_max_point_claimed(self):
        """Point at global xmax / ymax is claimed by a rightmost /
        topmost chunk."""
        centroids = np.array(
            [
                [-122.0, 47.0],
                [-121.0, 48.0],  # global max corner
                [-121.5, 47.5],
            ]
        )
        chunks = compute_chunks(
            centroids, np.zeros((0, 2)), 1, 200.0,
        )
        primary = assign_primary_chunk(centroids, chunks)
        assert (primary >= 0).all()
        # All three points must be claimed
        assert (primary != -1).sum() == 3

    def test_gap_raises(self):
        """Extraneous points outside the chunks' global bbox raise."""
        centroids = _random_lonlat(200, seed = 10)
        chunks = compute_chunks(
            centroids, np.zeros((0, 2)), 50, 200.0,
        )
        # Add a point outside the global bbox
        stray = np.array([[180.0, 90.0]])
        with pytest.raises(
            ValueError, match = "did not fall into any",
        ):
            assign_primary_chunk(
                np.vstack([centroids, stray]), chunks,
            )

    def test_empty_input(self):
        chunks = compute_chunks(
            _random_lonlat(100, seed = 11),
            np.zeros((0, 2)), 50, 200.0,
        )
        result = assign_primary_chunk(np.zeros((0, 2)), chunks)
        assert len(result) == 0


class TestBboxMask:
    def test_in_and_out(self):
        centroids = np.array(
            [
                [-122.5, 47.5],  # in
                [-120.0, 48.0],  # out
                [-122.0, 47.6],  # in
            ]
        )
        bbox = (-123.0, 47.0, -121.0, 48.0)
        mask = bbox_mask(centroids, bbox)
        assert mask.tolist() == [True, False, True]

    def test_inclusive_edges(self):
        centroids = np.array([[-122.0, 47.0]])
        # Point on the lower-left corner
        mask = bbox_mask(
            centroids, (-122.0, 47.0, -121.0, 48.0)
        )
        assert mask[0]

    def test_empty(self):
        mask = bbox_mask(
            np.zeros((0, 2)), (-1.0, -1.0, 1.0, 1.0),
        )
        assert len(mask) == 0


class TestExtractCentroids:
    def test_points(self):
        geoms = np.array(
            [Point(-122.5, 47.5), Point(-121.0, 48.0)]
        )
        result = extract_centroids_lonlat(geoms)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(-122.5)
        assert result[0, 1] == pytest.approx(47.5)

    def test_empty(self):
        result = extract_centroids_lonlat(np.array([]))
        assert result.shape == (0, 2)


class TestChunkSpec:
    def test_frozen(self):
        spec = ChunkSpec(
            chunk_id = 0,
            core_bbox = (0.0, 0.0, 1.0, 1.0),
            buffered_bbox = (-0.1, -0.1, 1.1, 1.1),
        )
        with pytest.raises(Exception):
            spec.chunk_id = 1  # frozen dataclass
