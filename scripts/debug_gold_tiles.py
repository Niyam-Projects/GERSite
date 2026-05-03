"""
Debug script: iterate on freestiler approaches for zstd-compressed GeoParquet.

Run with:
    uv run python scripts/debug_gold_tiles.py

Tries each approach in sequence, printing success or error for each.
"""

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "lib"))

import yaml
from duckdb_helpers import StorageConfig

CONFIG_PATH = REPO_ROOT / "config.gers.yaml"
STORAGE = StorageConfig.from_config(CONFIG_PATH)
AOI = "miami_dade"

GOLD_FILE = Path(STORAGE.gold_path("buildings", aoi=AOI)) / "buildings.parquet"
OUT_DIR = Path(STORAGE.tiles_path("gold_buildings", aoi=AOI))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 0. Verify file exists and inspect schema
# ---------------------------------------------------------------------------
section("0. File check + schema")
if not GOLD_FILE.exists():
    print(f"ERROR: Gold file not found: {GOLD_FILE}")
    print("Run flows/produce_gold_layer.py --aoi saipan first.")
    sys.exit(1)

print(f"File: {GOLD_FILE}")
print(f"Size: {GOLD_FILE.stat().st_size / 1_048_576:.1f} MB")

import pyarrow.parquet as pq

meta = pq.read_metadata(str(GOLD_FILE))
schema = pq.read_schema(str(GOLD_FILE))
print(f"Row groups: {meta.num_row_groups}")
print(f"Rows: {meta.num_rows:,}")
print(f"Compression: {meta.row_group(0).column(0).compression}")
print(f"Columns: {schema.names}")

# ---------------------------------------------------------------------------
# 1. freestile_file — engine="duckdb" (what we just added to generate_tiles.py)
# ---------------------------------------------------------------------------
section("1. freestile_file(engine='duckdb')")
from freestiler import freestile_file

try:
    out = OUT_DIR / "_debug_engine_duckdb.pmtiles"
    out.unlink(missing_ok=True)
    freestile_file(
        input=str(GOLD_FILE),
        output=str(out),
        layer_name="buildings",
        min_zoom=0,
        max_zoom=14,
        tile_format="mvt",
        coalesce=True,
        overwrite=True,
        engine="duckdb",
    )
    print(f"SUCCESS → {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
except Exception:
    print("FAILED:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 2. freestile_query — DuckDB SQL with read_parquet + LOAD spatial
# ---------------------------------------------------------------------------
section("2. freestile_query — SELECT * FROM read_parquet(...)")
from freestiler import freestile_query

try:
    out = OUT_DIR / "_debug_query_star.pmtiles"
    out.unlink(missing_ok=True)
    sql = f"SELECT * FROM read_parquet('{GOLD_FILE.as_posix()}')"
    print(f"SQL: {sql}")
    freestile_query(
        query=sql,
        output=str(out),
        layer_name="buildings",
        min_zoom=0,
        max_zoom=14,
        tile_format="mvt",
        coalesce=True,
        overwrite=True,
    )
    print(f"SUCCESS → {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
except Exception:
    print("FAILED:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 3. freestile_query — with LOAD spatial; upfront, explicit geometry column
# ---------------------------------------------------------------------------
section("3. freestile_query — LOAD spatial; SELECT ST_GeomFromWKB(geometry)...")
try:
    out = OUT_DIR / "_debug_query_geomfromwkb.pmtiles"
    out.unlink(missing_ok=True)
    geom_col = "geometry"
    # Build column list: all non-geometry columns + cast geometry
    all_cols = [n for n in schema.names if n != geom_col]
    col_list = ", ".join(all_cols[:20])  # cap to avoid huge column lists
    sql = (
        "LOAD spatial; "
        f"SELECT {col_list}, ST_GeomFromWKB({geom_col}) AS geometry "
        f"FROM read_parquet('{GOLD_FILE.as_posix()}')"
    )
    print(f"SQL (truncated): ...ST_GeomFromWKB({geom_col})...")
    freestile_query(
        query=sql,
        output=str(out),
        layer_name="buildings",
        min_zoom=0,
        max_zoom=14,
        tile_format="mvt",
        coalesce=True,
        overwrite=True,
    )
    print(f"SUCCESS → {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
except Exception:
    print("FAILED:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 4. freestile_file — engine="auto" (original, for baseline comparison)
# ---------------------------------------------------------------------------
section("4. freestile_file(engine='auto') — original baseline")
try:
    out = OUT_DIR / "_debug_engine_auto.pmtiles"
    out.unlink(missing_ok=True)
    freestile_file(
        input=str(GOLD_FILE),
        output=str(out),
        layer_name="buildings",
        min_zoom=0,
        max_zoom=14,
        tile_format="mvt",
        coalesce=True,
        overwrite=True,
        engine="auto",
    )
    print(f"SUCCESS → {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
except Exception:
    print("FAILED:")
    traceback.print_exc()

section("Done")
