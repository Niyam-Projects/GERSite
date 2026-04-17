#!/home/nathenry/miniforge3/envs/openpois/bin/python
"""
Apply OSM change-rate model predictions to the OSM POI snapshot.

Reads fitted model predictions for all versions that start with the configured
`model_stub`, then assigns a confidence estimate to every POI in the OSM
snapshot based on:
  1. Best-matching model group (by filter key priority order)
  2. Years since the element was last edited in OSM

Matching priority (first match wins):
  - For each key in `download > osm > filter_keys`, if a `<model_stub>_by_<key>`
    prediction set exists and the POI's value for that key appears in the
    predictions' group_name column, use those group-specific estimates.
  - If no filter key matches, fall back to `<model_stub>_constant`.

Output columns added to the snapshot:
  conf_mean, conf_lower, conf_upper  — confidence (1 - p_change) estimates
  t2_years                           — years since last OSM edit (rounded to
                                       0.1, capped at 10)
  model_version                      — which model version was used
  model_group                        — which group was matched, or "constant"

Streams input row-groups via pyarrow, computes predictions per batch in numpy,
and appends the new columns before writing each batch to the output parquet.
Input row order is preserved — the downstream `format_for_upload.py` step
re-sorts by geohash, so no intermediate spatial ordering is needed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from config_versioned import Config

from openpois.models.apply import (
    PREDICTIONS_FILE,
    constant_lookup,
    group_lookup,
    load_predictions,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

MODEL_STUB = config.get("osm_data", "apply_model", "model_stub")
FILTER_KEYS = config.get("download", "osm", "filter_keys")
SNAPSHOT_PATH = config.get_file_path("snapshot_osm", "snapshot")
OUTPUT_PATH = config.get_file_path("snapshot_osm", "rated_snapshot")

# Base directory containing all versioned model subdirectories
MODEL_BASE = Path(config.get_dir_path("model_output")).parent

BATCH_ROWS = 500_000
ROW_GROUP_SIZE = 50_000


# -----------------------------------------------------------------------------
# Per-batch prediction logic (numpy only)
# -----------------------------------------------------------------------------

def _compute_batch_predictions(
    df_lookup: pd.DataFrame,
    const_arr: np.ndarray,
    by_key_lookups: dict[str, tuple[list[str], np.ndarray]],
    constant_version: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Given only the `last_edited` + `FILTER_KEYS` columns of a batch, compute
    the 6 prediction columns plus a boolean `matched` mask (True where a
    per-key random-effects model was used, False where the constant fallback
    applied). Mirrors the logic of the original in-memory implementation.
    """
    n = len(df_lookup)
    today = pd.Timestamp.now(tz = "UTC")
    last_edited = df_lookup["last_edited"]
    if last_edited.dt.tz is None:
        last_edited = last_edited.dt.tz_localize("UTC")
    n_null = last_edited.isna().sum()
    if n_null:
        raise ValueError(
            f"{n_null} rows have a null last_edited timestamp. "
            "Remove or impute these rows before applying the model."
        )
    elapsed_secs = (today - last_edited).dt.total_seconds().to_numpy()
    elapsed_years = elapsed_secs / (365.25 * 86_400)
    t2_years = np.clip(np.round(elapsed_years * 10) / 10, 0.0, 10.0)
    t2_int_arr = np.round(t2_years * 10).astype(int)

    p_arr = const_arr[t2_int_arr].copy()
    model_version_arr = np.full(n, constant_version, dtype = object)
    model_group_arr = np.full(n, "constant", dtype = object)
    matched = np.zeros(n, dtype = bool)

    for key in FILTER_KEYS:
        if key not in by_key_lookups:
            continue
        groups, group_arr = by_key_lookups[key]
        group_to_idx = {g: i for i, g in enumerate(groups)}
        group_ids = (
            df_lookup[key].map(group_to_idx).fillna(-1).astype(int).to_numpy()
        )
        eligible = ~matched & (group_ids >= 0)
        eli_pos = np.where(eligible)[0]
        if len(eli_pos) == 0:
            continue
        p_arr[eli_pos] = group_arr[group_ids[eli_pos], t2_int_arr[eli_pos]]
        model_version_arr[eli_pos] = f"{MODEL_STUB}_by_{key}"
        model_group_arr[eli_pos] = df_lookup[key].to_numpy()[eli_pos]
        matched[eli_pos] = True

    return (
        {
            "t2_years": t2_years,
            "conf_mean": 1.0 - p_arr[:, 0],
            "conf_lower": 1.0 - p_arr[:, 2],   # 1 - p_upper
            "conf_upper": 1.0 - p_arr[:, 1],   # 1 - p_lower
            "model_version": model_version_arr,
            "model_group": model_group_arr,
        },
        matched,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Apply change-rate model predictions to the OSM POI snapshot."
    )
    parser.add_argument(
        "--test",
        action = "store_true",
        help = "Process only the first 10,000 rows of the snapshot for testing.",
    )
    args = parser.parse_args()

    print(f"Model stub: {MODEL_STUB}")
    print(f"Filter keys: {FILTER_KEYS}")

    # -- Load predictions -------------------------------------------------------
    constant_version = f"{MODEL_STUB}_constant"
    constant_dir = MODEL_BASE / constant_version
    if not constant_dir.is_dir():
        raise FileNotFoundError(
            f"Constant model directory not found: {constant_dir}"
        )
    const_arr = constant_lookup(load_predictions(constant_dir))
    print(f"Loaded constant model from {constant_dir.name}")

    by_key_lookups: dict[str, tuple[list[str], np.ndarray]] = {}
    for key in FILTER_KEYS:
        version = f"{MODEL_STUB}_by_{key}"
        version_dir = MODEL_BASE / version
        if version_dir.is_dir() and (version_dir / PREDICTIONS_FILE).exists():
            groups, arr = group_lookup(load_predictions(version_dir))
            by_key_lookups[key] = (groups, arr)
            print(f"  Loaded {version} ({len(groups)} groups)")
        else:
            print(f"  No predictions found for {version}; will skip")

    # -- Open input, build output schema ----------------------------------------
    print(f"\nReading OSM snapshot from {SNAPSHOT_PATH} ...")
    pf = pq.ParquetFile(SNAPSHOT_PATH)
    n_total = pf.metadata.num_rows
    print(f"  {n_total:,} POIs across {pf.num_row_groups} row groups")

    # Preserve the input GeoParquet file-level metadata (contains the `geo`
    # block that marks `geometry` as the primary geometry column + its CRS).
    # We only append new columns — existing schema + metadata carry through.
    input_schema = pf.schema_arrow
    new_fields = [
        pa.field("t2_years", pa.float64()),
        pa.field("conf_mean", pa.float64()),
        pa.field("conf_lower", pa.float64()),
        pa.field("conf_upper", pa.float64()),
        pa.field("model_version", pa.string()),
        pa.field("model_group", pa.string()),
    ]
    output_schema = pa.schema(
        list(input_schema) + new_fields,
        metadata = input_schema.metadata,
    )

    # -- Stream: read batch → append prediction columns → write -----------------
    lookup_cols = ["last_edited"] + [k for k in FILTER_KEYS if k in by_key_lookups]
    version_counts: dict[str, int] = {
        f"{MODEL_STUB}_by_{k}": 0 for k in by_key_lookups
    }
    version_counts[constant_version] = 0
    n_written = 0

    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    print(f"\nWriting to {OUTPUT_PATH} ...", flush = True)

    with pq.ParquetWriter(
        OUTPUT_PATH, output_schema, compression = "zstd"
    ) as writer:
        if args.test:
            print("  (--test mode: first 10,000 rows only)")
            batches = [next(pf.iter_batches(batch_size = 10_000))]
        else:
            batches = pf.iter_batches(batch_size = BATCH_ROWS)

        for batch in batches:
            tbl = pa.Table.from_batches([batch])
            df_lookup = tbl.select(lookup_cols).to_pandas()
            preds, matched = _compute_batch_predictions(
                df_lookup, const_arr, by_key_lookups, constant_version,
            )

            for key in by_key_lookups:
                version = f"{MODEL_STUB}_by_{key}"
                version_counts[version] += int(
                    (preds["model_version"] == version).sum()
                )
            version_counts[constant_version] += int(len(df_lookup) - matched.sum())

            for field in new_fields:
                tbl = tbl.append_column(
                    field.name,
                    pa.array(preds[field.name], type = field.type),
                )

            writer.write_table(tbl, row_group_size = ROW_GROUP_SIZE)
            n_written += batch.num_rows
            print(f"  {n_written:,}/{n_total:,} rows written", flush = True)

    print("\nModel version breakdown:")
    for version, count in sorted(version_counts.items(), key = lambda kv: -kv[1]):
        print(f"  {version}: {count:,}")
    print(f"\nDone. Saved {n_written:,} POIs.", flush = True)
