"""Generate minimal parquet fixtures for the DuckDB OGC:CRS84 vs EPSG:4326 bug report."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

OUT = Path(__file__).parent / "bug_fixtures"
OUT.mkdir(exist_ok=True)

# t1: Overture-style — NO 'crs' field (OGC:CRS84 per GeoParquet 1.0 spec)
# DuckDB reads this as GEOMETRY('OGC:CRS84')
geo_no_crs = {
    "version": "1.0.0",
    "primary_column": "geometry",
    "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Point"]}},
    # NOTE: 'crs' key intentionally absent — matches GeoParquet 1.0 spec
}
pts1 = [shapely.Point(-80.3, 25.7), shapely.Point(-80.2, 25.6)]
t1 = pa.table(
    {
        "overture_id": pa.array(["ov_001", "ov_002"], type=pa.string()),
        "value": pa.array([1, 2], type=pa.int32()),
        "geometry": pa.array([shapely.to_wkb(p) for p in pts1], type=pa.binary()),
    },
    metadata={b"geo": json.dumps(geo_no_crs).encode()},
)
pq.write_table(t1, OUT / "t1_overture_no_crs.parquet")

# t2: FEMA-style — explicit EPSG:4326 PROJJSON written by geopandas
# DuckDB reads this as GEOMETRY('EPSG:4326')
gdf2 = gpd.GeoDataFrame(
    {"build_id": ["fema_001", "fema_002"], "score": [1, 3]},
    geometry=[shapely.Point(-80.2, 25.6), shapely.Point(-80.1, 25.5)],
    crs="EPSG:4326",
)
gdf2.to_parquet(OUT / "t2_fema_epsg4326.parquet")

print("Written:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

# Confirm DuckDB sees the type mismatch
con = duckdb.connect()
con.execute("LOAD spatial")
t1_path = str(OUT / "t1_overture_no_crs.parquet")
t2_path = str(OUT / "t2_fema_epsg4326.parquet")
t1_type = con.sql(f"SELECT typeof(geometry) FROM read_parquet('{t1_path}') LIMIT 1").fetchone()[0]
t2_type = con.sql(f"SELECT typeof(geometry) FROM read_parquet('{t2_path}') LIMIT 1").fetchone()[0]
con.close()
print(f"\nt1 DuckDB type: {t1_type}")
print(f"t2 DuckDB type: {t2_type}")
print("\nOK — upload these two files to GitHub Gist alongside the MRE script.")
