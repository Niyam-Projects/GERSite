#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb==1.5.2",
#   "geopandas",
#   "pandas",
#   "pyarrow",
#   "shapely",
# ]
# ///
"""
Minimal Reproducible Example: DuckDB CASE expression over geometry columns from two
GeoParquet files fails with Binder Error when the CRS metadata differs between the files,
even when the coordinate systems are functionally identical (OGC:CRS84 vs EPSG:4326).

DuckDB version: 1.5.2
Spatial extension: dc1996b

BUG DESCRIPTION
---------------
The GeoParquet 1.0 specification states:
  "When the CRS is OGC:CRS84, the 'crs' member SHOULD be omitted."

Overture Maps GeoParquet files follow this convention — they omit the `crs` field,
so DuckDB infers GEOMETRY('OGC:CRS84'). Other datasets (e.g. FEMA USA Structures)
include an explicit PROJJSON for EPSG:4326.

When both files are used in a FULL OUTER JOIN with a CASE expression to select geometry
from one side or the other, DuckDB refuses to unify the two geometry types:

    Binder Error: Cannot cast GEOMETRY with CRS 'OGC:CRS84' to GEOMETRY with
    different CRS 'EPSG:4326' without specifying allow_override = true.

OGC:CRS84 and EPSG:4326 are functionally identical (both are WGS84). The only
difference is axis-order convention, but all GeoParquet-encoded WKB uses lon/lat
regardless of the CRS label. This means there is nothing to "cast" — the geometries
are byte-for-byte identical in their WKB representation.

CASCADING EFFECT
----------------
A Binder-phase failure while the DuckDB spatial extension is loaded may corrupt the
extension's global state for the remainder of the Python process. Subsequent fresh
connections then fail with:

    INTERNAL Error: TransactionContext::ActiveTransaction called without active
    transaction. This error signals an assertion failure within DuckDB.

This cascade was observed in production (DuckDB 1.5.2, spatial extension dc1996b)
when a CASE-without-normalization query ran alongside other spatial operations in the
same Python process. The exact conditions for the cascade are difficult to reproduce
in isolation but are reliably avoided by using the workaround below.

WORKAROUND
----------
Run the FULL OUTER JOIN without geometry columns (DuckDB handles this fine), then
read geometry columns via geopandas (PyArrow-based, completely bypasses DuckDB
spatial extension), and merge in Python. See the end of this script.

GitHub issue: (link to be added after posting)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

print(f"DuckDB version  : {duckdb.__version__}")
print(f"Python          : {sys.version.split()[0]}")
print()

# ---------------------------------------------------------------------------
# Create two minimal GeoParquet files that reproduce the real-world trigger
# ---------------------------------------------------------------------------
# t1: Overture-style — omits 'crs' field per GeoParquet 1.0 spec (OGC:CRS84 default)
#                      DuckDB infers: GEOMETRY('OGC:CRS84')
# t2: FEMA-style     — has explicit EPSG:4326 PROJJSON in 'crs' field
#                      DuckDB infers: GEOMETRY('EPSG:4326')

# Use pre-built fixture files if available (e.g. downloaded from GitHub Gist),
# otherwise generate them in a temp directory.
_fixtures = Path(__file__).parent / "bug_fixtures"
_use_fixtures = (
    (_fixtures / "t1_overture_no_crs.parquet").exists()
    and (_fixtures / "t2_fema_epsg4326.parquet").exists()
)

if _use_fixtures:
    tmpdir = None
    t1_path = _fixtures / "t1_overture_no_crs.parquet"
    t2_path = _fixtures / "t2_fema_epsg4326.parquet"
    print(f"Using fixture files from {_fixtures}")
    print()
else:
    tmpdir = Path(tempfile.mkdtemp())
    t1_path = tmpdir / "overture_style_no_crs_field.parquet"
    t2_path = tmpdir / "fema_style_epsg4326.parquet"

# --- t1: Overture-style — NO 'crs' field in geo metadata ---
geo_meta_no_crs = {
    "version": "1.0.0",
    "primary_column": "geometry",
    "columns": {
        "geometry": {
            "encoding": "WKB",
            "geometry_types": ["Point"],
        }
    },
    # NOTE: 'crs' key is intentionally absent.
    # GeoParquet 1.0: "When the CRS is OGC:CRS84, the 'crs' member SHOULD be omitted."
}
if not _use_fixtures:
    pts1 = [shapely.Point(-80.3, 25.7), shapely.Point(-80.2, 25.6), shapely.Point(-80.1, 25.5)]
    t1_table = pa.table(
        {
            "overture_id": pa.array(["ov_a", "ov_b", "ov_c"], type=pa.string()),
            "value": pa.array([1, 2, 3], type=pa.int32()),
            "geometry": pa.array([shapely.to_wkb(p) for p in pts1], type=pa.binary()),
        },
        metadata={b"geo": json.dumps(geo_meta_no_crs).encode()},
    )
    pq.write_table(t1_table, t1_path)

    # --- t2: FEMA-style — explicit EPSG:4326 PROJJSON in 'crs' field ---
    gdf2 = gpd.GeoDataFrame(
        {"build_id": [1001, 1002, 1003], "score": [10, 20, 30]},
        geometry=[shapely.Point(-80.2, 25.6), shapely.Point(-80.1, 25.5), shapely.Point(-80.0, 25.4)],
        crs="EPSG:4326",
    )
    gdf2.to_parquet(t2_path)

# Confirm DuckDB sees the CRS mismatch
con_check = duckdb.connect()
con_check.execute("LOAD spatial")
t1_type = con_check.sql(f"SELECT typeof(geometry) FROM read_parquet('{t1_path}') LIMIT 1").fetchone()[0]
t2_type = con_check.sql(f"SELECT typeof(geometry) FROM read_parquet('{t2_path}') LIMIT 1").fetchone()[0]
con_check.close()
print(f"t1 geometry type (DuckDB): {t1_type}")
print(f"t2 geometry type (DuckDB): {t2_type}")
print(f"  → Both are WGS84, but DuckDB treats them as distinct GEOMETRY types")
print()

# ---------------------------------------------------------------------------
# Helper — each call uses a fresh connection so prior failures don't cascade
# ---------------------------------------------------------------------------

def run(label: str, sql: str) -> None:
    con = duckdb.connect()
    con.execute("LOAD spatial")
    print(f"[{label}]")
    print(f"  SQL: {sql.strip().splitlines()[0][:80]}...")
    try:
        rows = con.sql(sql).fetchall()
        print(f"  OK  — {len(rows)} rows returned")
    except Exception as e:
        for line in str(e).split("\n"):
            if line.strip():
                print(f"  FAIL — {line.strip()}")
                break
    finally:
        con.close()
        print()

# ---------------------------------------------------------------------------
# CONTROL cases — these work correctly
# ---------------------------------------------------------------------------

run(
    "CONTROL: Single-table ST_SetCRS on OGC:CRS84 file (works)",
    f"SELECT ST_SetCRS(geometry, 'EPSG:4326') FROM read_parquet('{t1_path}') LIMIT 3",
)

run(
    "CONTROL: FULL OUTER JOIN without geometry columns (works)",
    f"""
    SELECT t1.overture_id, t2.build_id
    FROM read_parquet('{t1_path}') t1
    FULL OUTER JOIN read_parquet('{t2_path}') t2
        ON t1.value = t2.score
    """,
)

# ---------------------------------------------------------------------------
# BUG: CASE expression over geometry columns from files with mismatched CRS
# This is the natural query a developer would write for geometry priority logic.
# ---------------------------------------------------------------------------

run(
    "BUG: FULL OUTER JOIN + CASE over geometry (no ST_SetCRS) → Binder Error",
    f"""
    SELECT
        COALESCE(t1.overture_id, 'fema_' || t2.build_id) AS building_id,
        CASE
            WHEN t1.overture_id IS NOT NULL THEN t1.geometry
            ELSE                                 t2.geometry
        END AS geometry
    FROM read_parquet('{t1_path}') t1
    FULL OUTER JOIN read_parquet('{t2_path}') t2
        ON t1.value = t2.score
    """,
)

# ---------------------------------------------------------------------------
# SAFE WORKAROUND: geometry-free JOIN + geopandas geometry reads + Python merge
# Bypasses DuckDB spatial extension for geometry columns entirely.
# ---------------------------------------------------------------------------

print("=" * 60)
print("SAFE WORKAROUND: geometry-free JOIN + geopandas geometry reads")
print("=" * 60)
print()

# Step 1: FULL OUTER JOIN without geometry columns (DuckDB, no spatial extension needed)
con_w = duckdb.connect()
join_df = con_w.sql(f"""
    SELECT
        COALESCE(t1.overture_id, 'fema_' || t2.build_id) AS building_id,
        t1.overture_id, t2.build_id,
        t1.value, t2.score
    FROM read_parquet('{t1_path}') t1
    FULL OUTER JOIN read_parquet('{t2_path}') t2
        ON t1.value = t2.score
""").df()
con_w.close()
print(f"Step 1 — JOIN rows: {len(join_df)}")

# Step 2: Read geometry via geopandas (PyArrow-based, no DuckDB spatial extension)
t1_geom = (
    gpd.read_parquet(t1_path, columns=["overture_id", "geometry"])
    .to_crs("EPSG:4326")
    .set_index("overture_id")["geometry"]
)
t2_geom = (
    gpd.read_parquet(t2_path, columns=["build_id", "geometry"])
    .to_crs("EPSG:4326")
    .set_index("build_id")["geometry"]
)
print(f"Step 2 — t1 geom: {len(t1_geom)} rows, t2 geom: {len(t2_geom)} rows")

# Step 3: Merge geometries in Python — prefer t1 (Overture), fall back to t2 (FEMA)
join_df = join_df.merge(t1_geom.rename("geom1"), left_on="overture_id", right_index=True, how="left")
join_df = join_df.merge(t2_geom.rename("geom2"), left_on="build_id", right_index=True, how="left")
has_t1 = join_df["overture_id"].notna()
join_df["geometry"] = pd.concat([
    join_df.loc[has_t1, "geom1"],
    join_df.loc[~has_t1, "geom2"],
]).reindex(join_df.index)
join_df = join_df.drop(columns=["geom1", "geom2"])

result_gdf = gpd.GeoDataFrame(join_df, geometry="geometry", crs="EPSG:4326")
print(f"Step 3 — GeoDataFrame: {len(result_gdf)} rows, CRS: {result_gdf.crs}")
print()
print(result_gdf[["building_id", "overture_id", "build_id"]].to_string(index=False))

if tmpdir is not None:
    shutil.rmtree(tmpdir)
print("\nDone.")
