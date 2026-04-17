#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module formats OSM changes and versions into observations, which can be more easily
queried and statistically analyzed.
"""

import csv
import os
import re
from pathlib import Path

import duckdb


_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_:]+$")


def _validate_key(k: str) -> str:
    """Allow only alphanumerics, underscores, and colons in interpolated keys.

    OSM tag keys such as ``addr:street`` are valid; anything else is rejected
    to avoid opening a SQL injection path through the pivot CTE.
    """
    if not isinstance(k, str) or not _SAFE_KEY_RE.match(k):
        raise ValueError(f"Unsafe tag key for SQL interpolation: {k!r}")
    return k


def _init_scan_state(keep_keys: list[str]) -> dict:
    return {
        "add_to_list": False,
        "last_tag_timestamp": None,
        "last_obs_timestamp": None,
        "last_tag_user": None,
        "last_tag_value": None,
        "tag_value": None,
        "keep_current": {k: None for k in keep_keys},
        "keep_last": {k: None for k in keep_keys},
    }


def _advance_scan_state(
    state: dict,
    row: tuple,
    col_idx: dict,
    tag_key: str,
    keep_keys: list[str],
) -> dict | None:
    """Run one row through the per-POI state machine.

    Returns the observation dict to emit, or ``None`` if this version is
    before the tag was first added (so ``add_to_list`` is still False).
    """
    elem_id = row[col_idx["id"]]
    version = row[col_idx["version"]]
    changeset = row[col_idx["changeset"]]
    obs_timestamp = row[col_idx["timestamp"]]
    user = row[col_idx["user"]]

    # `last_tag_value` on the emitted obs must reflect the PRE-update state;
    # the other `last_*` fields are updated below after `obs` is built.
    prev_last_tag_value = state["last_tag_value"]

    # Keep-keys: shift current → last only when this version's changeset
    # touches the key; otherwise current + last both stay sticky.
    for k in keep_keys:
        ch = row[col_idx[f"{k}__change"]]
        if ch is not None:
            state["keep_last"][k] = state["keep_current"][k]
            state["keep_current"][k] = row[col_idx[f"{k}__value"]]

    tag_val = row[col_idx[f"{tag_key}__value"]]
    tag_ch = row[col_idx[f"{tag_key}__change"]]
    vis_val = row[col_idx["visible__value"]]
    vis_ch = row[col_idx["visible__change"]]

    tag_added = tag_ch == "Added"
    tag_changed = tag_ch == "Changed"
    tag_deleted = tag_ch == "Deleted"
    poi_deleted = (vis_ch is not None) and (vis_val == "false")
    poi_re_added = (
        state["add_to_list"]
        and (vis_ch is not None)
        and (vis_val == "true")
    )
    any_change = (
        tag_added or tag_changed or tag_deleted or poi_deleted or poi_re_added
    )

    if tag_added:
        state["add_to_list"] = True
    if tag_added or tag_changed:
        state["last_tag_value"] = tag_val
        state["tag_value"] = tag_val
    if tag_deleted or poi_deleted:
        state["tag_value"] = None
    if poi_re_added:
        state["tag_value"] = state["last_tag_value"]

    if not state["add_to_list"]:
        return None

    obs = {
        "id": elem_id,
        "version": version,
        "changeset": changeset,
        "obs_timestamp": obs_timestamp,
        "last_obs_timestamp": state["last_obs_timestamp"],
        "last_tag_timestamp": state["last_tag_timestamp"],
        "user": user,
        "last_tag_user": state["last_tag_user"],
        "tag_value": state["tag_value"],
        "last_tag_value": prev_last_tag_value,
        "changed": int(any_change),
        "deleted": None,
        "tag_key": tag_key,
    }
    for k in keep_keys:
        obs[k] = state["keep_current"][k]
        obs[f"{k}_last_value"] = state["keep_last"][k]

    if any_change:
        state["last_tag_timestamp"] = obs_timestamp
        state["last_tag_user"] = user
    state["last_obs_timestamp"] = obs_timestamp
    return obs


def format_observations_duckdb(
    changes_path: Path,
    versions_path: Path,
    output_path: Path,
    tag_key: str,
    keep_keys: list[str],
    duckdb_memory_limit: str = "4GB",
    duckdb_threads: int | None = None,
    duckdb_temp_dir: Path | None = None,
    batch_rows: int = 100_000,
    verbose: bool = True,
) -> int:
    """
    Stream POI observations from Parquet inputs to CSV via DuckDB.

    DuckDB pivots the long-form ``osm_changes.parquet`` wide by tag key,
    LEFT-joins ``osm_versions.parquet`` on ``(type, id, version)``, and
    returns rows sorted by ``(type, id, version)``; the sort spills to
    ``duckdb_temp_dir`` past ``duckdb_memory_limit``. A Python scan then
    iterates the sorted stream through :func:`_advance_scan_state`,
    writing each observation directly to CSV.

    Peak RSS is bounded to roughly ``duckdb_memory_limit`` plus the
    DictWriter buffer, regardless of input size.

    Args:
        changes_path: Input ``osm_changes.parquet``.
        versions_path: Input ``osm_versions.parquet``.
        output_path: Destination CSV. Overwritten.
        tag_key: Tag key to model (e.g. ``"name"``).
        keep_keys: Tag keys to retain on each observation. Must not
            include special characters (validated against
            ``[A-Za-z0-9_:]+``).
        duckdb_memory_limit: DuckDB ``memory_limit`` setting. The sort
            operator spills to disk past this.
        duckdb_threads: DuckDB worker thread count. Defaults to
            ``os.cpu_count()``.
        duckdb_temp_dir: Sort-spill directory. Defaults to
            ``output_path.parent``.
        batch_rows: Rows pulled per ``fetchmany`` call.
        verbose: Print progress.

    Returns:
        Total number of observation rows written.
    """
    changes_path = Path(changes_path)
    versions_path = Path(versions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    tag_key = _validate_key(tag_key)
    keep_keys = [_validate_key(k) for k in keep_keys]
    # Pivot needs tag_key, 'visible', and all keep_keys (deduplicated).
    pivot_keys: list[str] = [tag_key, "visible"]
    for k in keep_keys:
        if k not in pivot_keys:
            pivot_keys.append(k)

    threads = duckdb_threads if duckdb_threads is not None else (os.cpu_count() or 1)
    temp_dir = (
        Path(duckdb_temp_dir) if duckdb_temp_dir is not None else output_path.parent
    )
    temp_dir.mkdir(parents = True, exist_ok = True)

    pivot_exprs: list[str] = []
    for k in pivot_keys:
        pivot_exprs.append(
            f"MAX(CASE WHEN key = '{k}' THEN value  END) AS \"{k}__value\""
        )
        pivot_exprs.append(
            f"MAX(CASE WHEN key = '{k}' THEN change END) AS \"{k}__change\""
        )
    pivot_select = ",\n            ".join(pivot_exprs)
    key_list_sql = ", ".join(f"'{k}'" for k in pivot_keys)
    pivot_cols_sql = ", ".join(
        f'p."{k}__value", p."{k}__change"' for k in pivot_keys
    )

    sql = f"""
    WITH pivoted AS (
        SELECT type, id, version,
            {pivot_select}
        FROM read_parquet('{changes_path.as_posix()}')
        WHERE key IN ({key_list_sql})
        GROUP BY type, id, version
    )
    SELECT v.type, v.id, v.version, v.changeset, v.timestamp, v."user",
           {pivot_cols_sql}
    FROM read_parquet('{versions_path.as_posix()}') v
    LEFT JOIN pivoted p USING (type, id, version)
    ORDER BY v.type, v.id, v.version
    """

    base_cols = ["type", "id", "version", "changeset", "timestamp", "user"]
    col_idx: dict = {c: i for i, c in enumerate(base_cols)}
    for k in pivot_keys:
        col_idx[f"{k}__value"] = len(col_idx)
        col_idx[f"{k}__change"] = len(col_idx)

    fieldnames = (
        [
            "id", "version", "changeset", "obs_timestamp", "last_obs_timestamp",
            "last_tag_timestamp", "user", "last_tag_user",
            "tag_value", "last_tag_value", "changed", "deleted",
        ]
        + keep_keys
        + [f"{k}_last_value" for k in keep_keys]
        + ["tag_key"]
    )

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute(f"SET threads TO {int(threads)}")
        con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
        if verbose:
            print(
                f"DuckDB streaming observations "
                f"(threads={threads}, mem={duckdb_memory_limit})..."
            )

        cursor = con.execute(sql)

        total = 0
        with open(output_path, "w", newline = "") as f:
            writer = csv.DictWriter(
                f, fieldnames = fieldnames, extrasaction = "ignore"
            )
            writer.writeheader()

            current_poi = None
            state = None
            while True:
                rows = cursor.fetchmany(batch_rows)
                if not rows:
                    break
                for row in rows:
                    poi_key = (row[col_idx["type"]], row[col_idx["id"]])
                    if poi_key != current_poi:
                        current_poi = poi_key
                        state = _init_scan_state(keep_keys)
                    obs = _advance_scan_state(
                        state, row, col_idx, tag_key, keep_keys
                    )
                    if obs is not None:
                        writer.writerow(obs)
                        total += 1
    finally:
        con.close()

    if verbose:
        print(f"Wrote {total:,} observations to {output_path}")
    return total
