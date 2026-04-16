#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.io.osm_history_pbf.

All external I/O (requests.get, subprocess.run, osmium.FileProcessor)
is mocked so tests run in milliseconds without network or filesystem access.
"""
from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

from openpois.io.osm_history_pbf import (
    _diff_tag_sets,
    _tag_set_for_version,
    download_history_pbf,
    filter_history_pbf,
    parse_history_pbf,
    time_filter_history_pbf,
)


# ---------------------------------------------------------------------------
# Helpers: fake pyosmium objects
# ---------------------------------------------------------------------------


def _make_tag(k: str, v: str) -> MagicMock:
    tag = MagicMock()
    tag.k = k
    tag.v = v
    return tag


def _make_tags(tag_dict: dict) -> list[MagicMock]:
    """Return an iterable of mock pyosmium Tag objects from a dict."""
    return [_make_tag(k, v) for k, v in tag_dict.items()]


def _make_location(lon: float | None, lat: float | None) -> MagicMock:
    loc = MagicMock()
    if lon is None or lat is None:
        loc.valid = MagicMock(return_value=False)
    else:
        loc.valid = MagicMock(return_value=True)
        loc.lon = lon
        loc.lat = lat
    return loc


def _make_version(
    kind: str,
    osm_id: int,
    version: int,
    changeset: int,
    tag_dict: dict,
    user: str = "alice",
    uid: int = 1,
    visible: bool = True,
    timestamp: datetime.datetime | None = None,
    lon: float | None = None,
    lat: float | None = None,
) -> MagicMock:
    obj = MagicMock()
    obj.id = osm_id
    obj.version = version
    obj.changeset = changeset
    obj.user = user
    obj.uid = uid
    obj.visible = visible
    obj.timestamp = (
        timestamp if timestamp is not None
        else datetime.datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    )
    obj.tags = _make_tags(tag_dict)
    obj.is_node = MagicMock(return_value=kind == "node")
    obj.is_way = MagicMock(return_value=kind == "way")
    obj.is_relation = MagicMock(return_value=kind == "relation")
    obj.location = _make_location(lon, lat)
    return obj


def _make_file_processor(objects: list) -> MagicMock:
    fp = MagicMock()
    fp.__iter__ = MagicMock(return_value=iter(objects))
    return fp


# ---------------------------------------------------------------------------
# _diff_tag_sets
# ---------------------------------------------------------------------------


class TestDiffTagSets:
    def test_added_only(self):
        rows = _diff_tag_sets(set(), {("amenity", "cafe")})
        assert rows == [{"key": "amenity", "value": "cafe", "change": "Added"}]

    def test_deleted_only(self):
        rows = _diff_tag_sets({("amenity", "cafe")}, set())
        assert rows == [{"key": "amenity", "value": "cafe", "change": "Deleted"}]

    def test_changed_key(self):
        rows = _diff_tag_sets(
            {("amenity", "cafe")},
            {("amenity", "restaurant")},
        )
        assert rows == [
            {"key": "amenity", "value": "restaurant", "change": "Changed"},
        ]

    def test_unchanged_tags_produce_no_rows(self):
        tags = {("amenity", "cafe"), ("name", "X")}
        assert _diff_tag_sets(tags, tags) == []

    def test_mixed_diff(self):
        prev = {("amenity", "cafe"), ("name", "Old")}
        curr = {("amenity", "restaurant"), ("cuisine", "italian")}
        rows = _diff_tag_sets(prev, curr)
        by_key = {r["key"]: r for r in rows}
        assert by_key["amenity"]["change"] == "Changed"
        assert by_key["amenity"]["value"] == "restaurant"
        assert by_key["cuisine"]["change"] == "Added"
        assert by_key["name"]["change"] == "Deleted"
        assert by_key["name"]["value"] == "Old"


# ---------------------------------------------------------------------------
# _tag_set_for_version
# ---------------------------------------------------------------------------


class TestTagSetForVersion:
    def test_node_includes_lat_lon_and_visible(self):
        obj = _make_version(
            kind="node",
            osm_id=1, version=1, changeset=100,
            tag_dict={"amenity": "cafe"},
            lon=-122.0, lat=47.0,
        )
        tags = _tag_set_for_version(obj)
        assert ("amenity", "cafe") in tags
        assert ("visible", "true") in tags
        assert ("lat", "47.0") in tags
        assert ("lon", "-122.0") in tags

    def test_way_excludes_lat_lon(self):
        obj = _make_version(
            kind="way",
            osm_id=2, version=1, changeset=100,
            tag_dict={"building": "yes"},
        )
        tags = _tag_set_for_version(obj)
        assert ("building", "yes") in tags
        assert ("visible", "true") in tags
        assert not any(k == "lat" for k, _ in tags)
        assert not any(k == "lon" for k, _ in tags)

    def test_invisible_flag(self):
        obj = _make_version(
            kind="node",
            osm_id=1, version=2, changeset=101,
            tag_dict={}, visible=False,
        )
        tags = _tag_set_for_version(obj)
        assert ("visible", "false") in tags


# ---------------------------------------------------------------------------
# download_history_pbf
# ---------------------------------------------------------------------------


class TestDownloadHistoryPbf:
    def test_skips_if_exists_and_no_overwrite(self, tmp_path):
        output = tmp_path / "out.osh.pbf"
        output.write_bytes(b"fake")
        with patch(
            "openpois.io.osm_history_pbf._load_cookie_session"
        ) as mock_session:
            result = download_history_pbf(
                url="http://example.com/x.osh.pbf",
                output_path=output,
                overwrite=False,
            )
        mock_session.assert_not_called()
        assert result == output

    def test_raises_if_cookie_file_missing(self, tmp_path):
        output = tmp_path / "out.osh.pbf"
        missing_cookie = tmp_path / "nope.txt"
        with pytest.raises(FileNotFoundError, match="cookie file not found"):
            download_history_pbf(
                url="http://example.com/x.osh.pbf",
                output_path=output,
                cookie_file=missing_cookie,
                overwrite=False,
            )

    def test_downloads_via_streaming_session(self, tmp_path):
        output = tmp_path / "subdir" / "out.osh.pbf"

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content = MagicMock(return_value=[b"hello"])

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch(
            "openpois.io.osm_history_pbf._load_cookie_session",
            return_value=mock_session,
        ):
            result = download_history_pbf(
                url="http://example.com/x.osh.pbf",
                output_path=output,
                overwrite=False,
            )

        mock_session.get.assert_called_once_with(
            "http://example.com/x.osh.pbf", stream=True, timeout=(30, None)
        )
        assert result == output
        assert output.exists()


# ---------------------------------------------------------------------------
# filter_history_pbf
# ---------------------------------------------------------------------------


class TestFilterHistoryPbf:
    def test_skips_if_output_exists_and_no_overwrite(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")
        output_pbf.write_bytes(b"fake")
        with patch(
            "openpois.io.osm_history_pbf.subprocess.run"
        ) as mock_run:
            result = filter_history_pbf(
                input_pbf, output_pbf, ["amenity"], overwrite=False
            )
        mock_run.assert_not_called()
        assert result == output_pbf

    def test_command_uses_omit_referenced_and_osh_format(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")

        with (
            patch("openpois.io.osm_history_pbf.subprocess.run") as mock_run,
            patch(
                "openpois.io.osm_history_pbf._resolve_osmium",
                return_value="/usr/bin/osmium",
            ),
        ):
            filter_history_pbf(
                input_pbf, output_pbf, ["amenity", "shop"], overwrite=False
            )

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "tags-filter"
        assert "--omit-referenced" in cmd
        assert "--output-format=osh.pbf" in cmd
        assert "nwr/amenity" in cmd
        assert "nwr/shop" in cmd
        assert mock_run.call_args[1].get("check") is True


# ---------------------------------------------------------------------------
# time_filter_history_pbf
# ---------------------------------------------------------------------------


class TestTimeFilterHistoryPbf:
    def test_skips_if_output_exists_and_no_overwrite(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")
        output_pbf.write_bytes(b"fake")
        with patch(
            "openpois.io.osm_history_pbf.subprocess.run"
        ) as mock_run:
            result = time_filter_history_pbf(
                input_pbf, output_pbf,
                datetime.date(2016, 1, 1),
                datetime.date(2025, 12, 31),
                overwrite=False,
            )
        mock_run.assert_not_called()
        assert result == output_pbf

    def test_passes_iso_formatted_timestamps(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")

        with (
            patch("openpois.io.osm_history_pbf.subprocess.run") as mock_run,
            patch(
                "openpois.io.osm_history_pbf._resolve_osmium",
                return_value="/usr/bin/osmium",
            ),
        ):
            time_filter_history_pbf(
                input_pbf, output_pbf,
                datetime.date(2016, 1, 1),
                datetime.date(2025, 12, 31),
                overwrite=False,
            )

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "time-filter"
        assert "2016-01-01T00:00:00Z" in cmd
        assert "2025-12-31T00:00:00Z" in cmd
        # Range must come AFTER the input path for osmium
        input_idx = cmd.index(str(input_pbf))
        assert cmd.index("2016-01-01T00:00:00Z") > input_idx
        assert cmd.index("2025-12-31T00:00:00Z") > input_idx


# ---------------------------------------------------------------------------
# parse_history_pbf
# ---------------------------------------------------------------------------


class TestParseHistoryPbf:
    def test_single_element_multiple_versions_emits_diffs(self, tmp_path):
        """Versions of the same element should diff against the previous one."""
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"

        objs = [
            _make_version(
                kind="node", osm_id=1, version=1, changeset=100,
                tag_dict={"amenity": "cafe"}, lon=-122.0, lat=47.0,
            ),
            _make_version(
                kind="node", osm_id=1, version=2, changeset=101,
                tag_dict={"amenity": "restaurant"}, lon=-122.0, lat=47.0,
            ),
            _make_version(
                kind="node", osm_id=1, version=3, changeset=102,
                tag_dict={}, visible=False,
                lon=-122.0, lat=47.0,
            ),
        ]

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor",
            return_value=_make_file_processor(objs),
        ):
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                chunk_size=10,
                overwrite=True,
                verbose=False,
            )

        versions = pd.read_parquet(v_path)
        changes = pd.read_parquet(c_path)

        assert len(versions) == 3
        assert list(versions["version"]) == [1, 2, 3]
        assert list(versions["changeset"]) == [100, 101, 102]
        assert (versions["type"] == "node").all()

        # Version 1 must produce Added rows (no prior state)
        v1_changes = changes.query("version == 1")
        assert (v1_changes["change"] == "Added").all()
        assert "amenity" in set(v1_changes["key"])

        # Version 2 changes amenity value (Changed)
        v2_changes = changes.query("version == 2")
        amenity_rows = v2_changes.query('key == "amenity"')
        assert len(amenity_rows) == 1
        assert amenity_rows.iloc[0]["change"] == "Changed"
        assert amenity_rows.iloc[0]["value"] == "restaurant"

        # Version 3 marks visible=false (Changed) and removes amenity (Deleted)
        v3_changes = changes.query("version == 3")
        visible_rows = v3_changes.query('key == "visible"')
        assert len(visible_rows) == 1
        assert visible_rows.iloc[0]["value"] == "false"
        assert visible_rows.iloc[0]["change"] == "Changed"
        amenity_del = v3_changes.query(
            'key == "amenity" and change == "Deleted"'
        )
        assert len(amenity_del) == 1

    def test_element_boundary_resets_prev_tags(self, tmp_path):
        """When (type, id) changes, the next version's diff is against empty."""
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"

        objs = [
            _make_version(
                kind="node", osm_id=1, version=1, changeset=100,
                tag_dict={"amenity": "cafe"}, lon=-122.0, lat=47.0,
            ),
            _make_version(
                kind="node", osm_id=2, version=1, changeset=101,
                tag_dict={"shop": "bakery"}, lon=-122.1, lat=47.1,
            ),
        ]

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor",
            return_value=_make_file_processor(objs),
        ):
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                chunk_size=10,
                overwrite=True,
                verbose=False,
            )

        changes = pd.read_parquet(c_path)
        # Every row for node id=2 v=1 must be "Added", never "Deleted"
        n2_rows = changes.query("id == 2 and version == 1")
        assert (n2_rows["change"] == "Added").all()
        # It should NOT inherit a "Deleted amenity=cafe" row from node 1
        assert not (
            (n2_rows["key"] == "amenity") & (n2_rows["change"] == "Deleted")
        ).any()

    def test_empty_input_writes_empty_parquets(self, tmp_path):
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor",
            return_value=_make_file_processor([]),
        ):
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                chunk_size=10,
                overwrite=True,
                verbose=False,
            )

        # Both files should exist and be readable
        assert v_path.exists()
        assert c_path.exists()
        assert pq.ParquetFile(str(v_path)).metadata.num_rows == 0
        assert pq.ParquetFile(str(c_path)).metadata.num_rows == 0

    def test_skips_if_both_outputs_exist_and_no_overwrite(self, tmp_path):
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"
        v_path.write_bytes(b"fake")
        c_path.write_bytes(b"fake")

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor"
        ) as mock_fp:
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                overwrite=False,
                verbose=False,
            )
        mock_fp.assert_not_called()
