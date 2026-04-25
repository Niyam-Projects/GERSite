#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.osm.format_observations.
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openpois.osm.format_observations import format_observations_duckdb


VERSIONS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("changeset", pa.int64()),
    ("timestamp", pa.string()),
    ("user", pa.string()),
    ("uid", pa.int64()),
    ("type", pa.string()),
])

CHANGES_SCHEMA = pa.schema([
    ("key", pa.string()),
    ("value", pa.string()),
    ("change", pa.string()),
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("type", pa.string()),
])


def _make_versions(rows):
    return pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in VERSIONS_SCHEMA},
        schema=VERSIONS_SCHEMA,
    )


def _make_changes(rows):
    return pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in CHANGES_SCHEMA},
        schema=CHANGES_SCHEMA,
    )


def _synthetic_inputs():
    """
    Three POIs with name tags. Each has an Added, then a Changed on the
    tag_key, so each should yield 2 observations (the second with changed=1).
    """
    versions = []
    changes = []
    for elem_id in [100, 200, 300]:
        for ver in [1, 2]:
            versions.append({
                "id": elem_id, "version": ver, "changeset": 1000 + ver,
                "timestamp": f"2024-01-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            changes.append({
                "key": "name", "value": f"n{elem_id}.v{ver}",
                "change": "Added" if ver == 1 else "Changed",
                "id": elem_id, "version": ver, "type": "node",
            })
            changes.append({
                "key": "amenity", "value": "cafe",
                "change": "Added" if ver == 1 else "Changed",
                "id": elem_id, "version": ver, "type": "node",
            })
    return versions, changes


def _write_parquets(tmp_path, versions, changes):
    v_path = tmp_path / "versions.parquet"
    c_path = tmp_path / "changes.parquet"
    pq.write_table(_make_versions(versions), v_path)
    pq.write_table(_make_changes(changes), c_path)
    return v_path, c_path


class TestFormatObservationsDuckdb:
    """``format_observations_duckdb`` is the production entry point for the
    OSM history → observations pipeline; these tests pin its row count,
    column set, and state-machine semantics."""

    def test_synthetic_inputs_produce_expected_rows(self, tmp_path):
        versions, changes = _synthetic_inputs()
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        total = format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = ["amenity"],
            verbose = False,
        )
        # Two observations per POI (Added + Changed), three POIs
        assert total == 6
        out = pd.read_parquet(out_path)
        assert len(out) == 6
        assert set(out["id"]) == {100, 200, 300}
        assert set(out["version"]) == {1, 2}
        assert out["changed"].sum() == 6
        # Expected output columns
        expected = {
            "id", "version", "changeset", "obs_timestamp", "last_obs_timestamp",
            "last_tag_timestamp", "user", "last_tag_user",
            "tag_value", "last_tag_value", "changed", "deleted",
            "amenity", "amenity_last_value", "tag_key",
        }
        assert set(out.columns) == expected
        # last_tag_value reflects the PRE-update state: None on v1, v1's value on v2.
        out = out.sort_values(["id", "version"]).reset_index(drop = True)
        for poi_id in [100, 200, 300]:
            rows = out[out["id"] == poi_id].reset_index(drop = True)
            assert pd.isna(rows.loc[0, "last_tag_value"])
            assert rows.loc[1, "last_tag_value"] == f"n{poi_id}.v1"
            assert rows.loc[0, "tag_value"] == f"n{poi_id}.v1"
            assert rows.loc[1, "tag_value"] == f"n{poi_id}.v2"

    def test_tag_state_machine(self, tmp_path):
        """Added → Changed → visible=false → visible=true sequence.

        The re-added version should restore ``tag_value`` to the last SET
        value (``"bar"``), matching the original state-machine semantics.
        """
        versions = []
        changes = []
        seq = [
            ("Added",   "foo", None,    None),
            ("Changed", "bar", None,    None),
            (None,      None,  "false", "Added"),
            (None,      None,  "true",  "Changed"),
        ]
        for ver, (tag_ch, tag_val, vis_val, vis_ch) in enumerate(seq, start = 1):
            versions.append({
                "id": 42, "version": ver, "changeset": 1000 + ver,
                "timestamp": f"2024-01-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            if tag_ch is not None:
                changes.append({
                    "key": "name", "value": tag_val, "change": tag_ch,
                    "id": 42, "version": ver, "type": "node",
                })
            if vis_ch is not None:
                changes.append({
                    "key": "visible", "value": vis_val, "change": vis_ch,
                    "id": 42, "version": ver, "type": "node",
                })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = [],
            verbose = False,
        )
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        assert list(out["version"]) == [1, 2, 3, 4]
        assert list(out["tag_value"].fillna("")) == ["foo", "bar", "", "bar"]
        assert list(out["changed"]) == [1, 1, 1, 1]

    def test_keep_key_stickiness(self, tmp_path):
        """``{k}_last_value`` must persist across versions that don't touch ``k``."""
        versions = []
        changes = []
        seq = [
            ("Added",   "foo",  "Added",   "restaurant"),
            ("Changed", "foo2", None,      None),
            ("Changed", "foo3", "Changed", "bar"),
        ]
        for ver, (tag_ch, tag_val, kk_ch, kk_val) in enumerate(seq, start = 1):
            versions.append({
                "id": 7, "version": ver, "changeset": 1000 + ver,
                "timestamp": f"2024-02-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            changes.append({
                "key": "name", "value": tag_val, "change": tag_ch,
                "id": 7, "version": ver, "type": "node",
            })
            if kk_ch is not None:
                changes.append({
                    "key": "amenity", "value": kk_val, "change": kk_ch,
                    "id": 7, "version": ver, "type": "node",
                })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = ["amenity"],
            verbose = False,
        )
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        amenities = list(out["amenity"].fillna(""))
        lasts = list(out["amenity_last_value"].fillna(""))
        assert amenities == ["restaurant", "restaurant", "bar"]
        # v1: pre-change was None; v2: no change → last stays empty;
        # v3: last = "restaurant".
        assert lasts == ["", "", "restaurant"]

    def test_left_join_null_inheritance(self, tmp_path):
        """Versions with no relevant changes (LEFT-JOIN produces NULLs) should
        inherit prior state without crashing."""
        versions = []
        changes = []
        for ver in [1, 2, 3]:
            versions.append({
                "id": 99, "version": ver, "changeset": 2000 + ver,
                "timestamp": f"2024-03-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
        # Only v1 has tag/keep-key changes; v2 and v3 have no rows at all.
        changes.append({
            "key": "name", "value": "cafe", "change": "Added",
            "id": 99, "version": 1, "type": "node",
        })
        changes.append({
            "key": "amenity", "value": "cafe", "change": "Added",
            "id": 99, "version": 1, "type": "node",
        })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        total = format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = ["amenity"],
            verbose = False,
        )
        assert total == 3
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        assert list(out["tag_value"].fillna("")) == ["cafe", "cafe", "cafe"]
        assert list(out["amenity"].fillna("")) == ["cafe", "cafe", "cafe"]
        assert list(out["changed"]) == [1, 0, 0]
