#!/usr/bin/env python
"""
Standalone debug harness for resolve_building_geometries (Flow 3, Gold layer).

Extracts the DuckDB FULL OUTER JOIN logic from flows/produce_gold_layer.py so it
can be iterated on directly without Marimo / Prefect overhead.

Currently failing for miami_dade with an Internal DuckDB error.

Usage:
    python scripts/debug_resolve_geometries.py
    python scripts/debug_resolve_geometries.py --aoi miami_dade
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).parent.parent
LIB_PATH = REPO_ROOT / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))

from duckdb_helpers import StorageConfig, get_connection  # noqa: E402

import yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = REPO_ROOT / "config.gers.yaml"


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------


def _inspect_parquet(label: str, path: Path) -> int:
    """Log schema and row count of a parquet file. Returns row count."""
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    nrows = pf.metadata.num_rows
    log.info(f"  [{label}] rows={nrows:,}  path={path}")
    log.debug(f"  [{label}] schema:\n{schema.to_string()}")
    return nrows


def _run_probe_query(con, label: str, sql: str, fatal: bool = True) -> int | None:
    """Execute a probe COUNT query, log result, return count.

    If fatal=False the error is logged but not re-raised (useful for probes
    that are expected to fail and whose failure is itself diagnostic).
    """
    try:
        result = con.sql(sql).fetchone()[0]
        log.info(f"  Probe '{label}': {result:,} rows")
        return result
    except Exception as exc:
        log.error(f"  Probe '{label}' FAILED: {exc}")
        if fatal:
            raise
        return None


def _finish(gdf, logger):
    """Log final stats and return gdf."""
    has_overture = gdf["overture_id"].notna().to_numpy()
    has_fema = gdf["fema_id"].notna().to_numpy()
    gdf["osm_id"] = None
    gdf["osm_type"] = None
    logger.info(
        f"\n=== Geometry resolution complete ===\n"
        f"  Total buildings : {len(gdf):,}\n"
        f"  Overture records: {has_overture.sum():,}\n"
        f"  FEMA-only records: {has_fema.sum():,}"
    )
    return gdf


# ---------------------------------------------------------------------------
# Core logic (extracted from flows/produce_gold_layer.py)
# ---------------------------------------------------------------------------


def resolve_building_geometries(aoi: str, cfg: dict, storage: StorageConfig):
    """Geometry resolution — standalone debug version.

    Mirrors the logic in flows/produce_gold_layer.py::resolve_building_geometries
    with extra diagnostics around the failing DuckDB query.
    """
    import geopandas as gpd

    overture_path = Path(storage.bronze_path("overture_buildings")) / aoi / "buildings.parquet"
    fema_path = Path(storage.bronze_path("fema_structures")) / aoi / "structures.parquet"
    fema_bridge_path = (
        Path(storage.silver_path("fema_bridge")).parent / aoi / "fema_bridge.parquet"
    )

    # ---- File existence check -----------------------------------------------
    log.info("=== Input file check ===")
    for label, p in [
        ("overture", overture_path),
        ("fema", fema_path),
        ("fema_bridge", fema_bridge_path),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file missing: {p}\n"
                "Run Flow 1 + 2 first (produce_bronze_layer.py / produce_silver_layer.py)."
            )
        _inspect_parquet(label, p)

    # ---- DuckDB connection (diagnostics only) --------------------------------
    con = get_connection(
        memory_limit=cfg["duckdb"]["memory_limit"],
        threads=cfg["duckdb"]["threads"],
    )

    log.info(f"\n=== Probe queries for AOI: {aoi} ===")

    # Probe: row counts per table (no JOIN yet)
    _run_probe_query(con, "overture rows", f"SELECT COUNT(*) FROM read_parquet('{overture_path}')")
    _run_probe_query(con, "fema rows", f"SELECT COUNT(*) FROM read_parquet('{fema_path}')")
    _run_probe_query(con, "fema_bridge rows", f"SELECT COUNT(*) FROM read_parquet('{fema_bridge_path}')")

    # Probe: check for NULL geometries in each source
    _run_probe_query(con, "overture NULL geom", f"SELECT COUNT(*) FROM read_parquet('{overture_path}') WHERE geometry IS NULL")
    _run_probe_query(con, "fema NULL geom", f"SELECT COUNT(*) FROM read_parquet('{fema_path}') WHERE geometry IS NULL")

    # Probe: simple JOIN without ST_SetCRS to check if JOIN itself is the issue
    log.info("\n=== Probe: JOIN without geometry (isolate JOIN vs ST_SetCRS) ===")
    _run_probe_query(
        con,
        "FULL OUTER JOIN row count (no geometry)",
        f"""
        SELECT COUNT(*) FROM read_parquet('{overture_path}') o
        FULL OUTER JOIN read_parquet('{fema_bridge_path}') b ON o.overture_id = b.overture_id
        FULL OUTER JOIN read_parquet('{fema_path}') f ON b.fema_id = f.build_id
        """,
    )

    # Probe: log the actual CRS metadata on both geometry columns
    log.info("\n=== Probe: geometry column CRS metadata ===")
    try:
        o_crs = con.sql(f"SELECT typeof(geometry) FROM read_parquet('{overture_path}') LIMIT 1").fetchone()
        f_crs = con.sql(f"SELECT typeof(geometry) FROM read_parquet('{fema_path}') LIMIT 1").fetchone()
        log.info(f"  overture geometry type : {o_crs[0] if o_crs else 'n/a'}")
        log.info(f"  fema geometry type     : {f_crs[0] if f_crs else 'n/a'}")
    except Exception as exc:
        log.warning(f"  typeof() probe failed: {exc}")

    # Probe: CASE geometry WITHOUT ST_SetCRS — this is KNOWN to fail if there is
    # a CRS mismatch between the two geometry columns (OGC:CRS84 vs EPSG:4326).
    # IMPORTANT: A Binder-phase failure on a DuckDB connection that has the
    # spatial extension loaded can corrupt DuckDB's global extension state for
    # ALL subsequent connections in the same process, causing unrelated
    # "TransactionContext::ActiveTransaction" errors on fresh connections.
    # We therefore SKIP this particular probe at runtime and instead document
    # the finding from the first exploratory run:
    #
    #   overture geometry type : GEOMETRY('OGC:CRS84')
    #   fema geometry type     : GEOMETRY('EPSG:4326')
    #
    # Root cause: Overture GeoParquet uses OGC:CRS84 metadata; FEMA uses
    # EPSG:4326. DuckDB treats these as distinct types in CASE expressions even
    # though they refer to the same coordinate system.
    log.info("\n=== Probe: CASE geometry without ST_SetCRS ===")
    log.info("  SKIPPED — known to corrupt DuckDB global extension state in-process.")
    log.info("  Finding from prior run: Overture=GEOMETRY('OGC:CRS84'), FEMA=GEOMETRY('EPSG:4326')")
    log.info("  Root cause: DuckDB CRS type mismatch prevents CASE branch unification.")

    # Probe: ST_SetCRS on single-table scans (no JOIN) to isolate the error
    log.info("\n=== Probe: ST_SetCRS on single tables (no JOIN) ===")
    _run_probe_query(
        con,
        "ST_SetCRS on overture (no join)",
        f"SELECT COUNT(*) FROM (SELECT ST_SetCRS(geometry, 'EPSG:4326') FROM read_parquet('{overture_path}'))",
        fatal=False,
    )
    _run_probe_query(
        con,
        "ST_SetCRS on fema (no join)",
        f"SELECT COUNT(*) FROM (SELECT ST_SetCRS(geometry, 'EPSG:4326') FROM read_parquet('{fema_path}'))",
        fatal=False,
    )

    # Discard diagnostics connection before opening fresh ones for the real queries.
    try:
        con.close()
    except Exception:
        pass

    # ---- Full query (original from the flow) --------------------------------
    # Fresh connection — isolated from any state left by the probe above.
    log.info("\n=== Full query (original — uses ST_SetCRS on both branches) ===")
    orig_con = get_connection(
        memory_limit=cfg["duckdb"]["memory_limit"],
        threads=cfg["duckdb"]["threads"],
    )
    query = f"""
    SELECT
        COALESCE(o.overture_id, 'fema_only_' || f.build_id) AS building_id,
        o.overture_id,
        f.build_id AS fema_id,
        b.iou                           AS fema_iou,
        o.height,
        o.num_floors,
        o.class                         AS overture_class,
        o.names,
        CASE
            WHEN o.overture_id IS NOT NULL THEN ST_SetCRS(o.geometry, 'EPSG:4326')
            ELSE ST_SetCRS(f.geometry, 'EPSG:4326')
        END AS geometry,
        CASE
            WHEN o.overture_id IS NOT NULL THEN 'overture'
            WHEN f.build_id IS NOT NULL THEN 'fema_only'
        END AS source
    FROM read_parquet('{overture_path}') o
    FULL OUTER JOIN read_parquet('{fema_bridge_path}') b
        ON o.overture_id = b.overture_id
    FULL OUTER JOIN read_parquet('{fema_path}') f
        ON b.fema_id = f.build_id
    """

    original_failed = False
    arrow_tbl = None
    try:
        log.info("Executing original query...")
        arrow_tbl = orig_con.sql(query).to_arrow_table()
        log.info(f"Original query SUCCEEDED: {arrow_tbl.num_rows:,} rows returned")
        orig_con.close()
    except Exception as exc:
        log.error(f"Original query FAILED: {exc}")
        original_failed = True
        try:
            orig_con.close()
        except Exception:
            pass

    # ---- Fixed query A: strip CRS via ST_GeomFromWKB(ST_AsWKB(...)) ---------
    # OGC:CRS84 and EPSG:4326 are the same coordinate system; the CRS metadata
    # on the parquet geometry columns differs between Overture and FEMA.
    # Stripping CRS metadata before the CASE unifies the geometry type so
    # DuckDB can resolve the CASE expression without a Binder error.
    log.info("\n=== Fixed query A (strip CRS via ST_GeomFromWKB/ST_AsWKB before CASE) ===")
    fixed_a_con = get_connection(
        memory_limit=cfg["duckdb"]["memory_limit"],
        threads=cfg["duckdb"]["threads"],
    )
    fixed_a_query = f"""
    SELECT
        COALESCE(o.overture_id, 'fema_only_' || f.build_id) AS building_id,
        o.overture_id,
        f.build_id AS fema_id,
        b.iou                           AS fema_iou,
        o.height,
        o.num_floors,
        o.class                         AS overture_class,
        o.names,
        ST_SetCRS(
            CASE
                WHEN o.overture_id IS NOT NULL
                    THEN ST_GeomFromWKB(ST_AsWKB(o.geometry))
                ELSE ST_GeomFromWKB(ST_AsWKB(f.geometry))
            END,
            'EPSG:4326'
        ) AS geometry,
        CASE
            WHEN o.overture_id IS NOT NULL THEN 'overture'
            WHEN f.build_id IS NOT NULL THEN 'fema_only'
        END AS source
    FROM read_parquet('{overture_path}') o
    FULL OUTER JOIN read_parquet('{fema_bridge_path}') b
        ON o.overture_id = b.overture_id
    FULL OUTER JOIN read_parquet('{fema_path}') f
        ON b.fema_id = f.build_id
    """

    fixed_a_succeeded = False
    try:
        log.info("Executing fixed query A...")
        arrow_tbl_a = fixed_a_con.sql(fixed_a_query).to_arrow_table()
        log.info(f"Fixed query A SUCCEEDED: {arrow_tbl_a.num_rows:,} rows returned")
        fixed_a_succeeded = True
        fixed_a_con.close()
        if original_failed:
            log.info(">>> Using fixed query A result")
            arrow_tbl = arrow_tbl_a
    except Exception as exc:
        log.error(f"Fixed query A FAILED: {exc}")
        try:
            fixed_a_con.close()
        except Exception:
            pass

    # ---- Fixed query B: two-column geometry (no spatial func in JOIN) --------
    # Avoid spatial functions entirely inside the JOIN by fetching both geometry
    # columns separately, then resolving the correct one in Python.
    # The CASE expression only operates on non-spatial columns.
    if not fixed_a_succeeded:
        log.info("\n=== Fixed query B (two-column geometry — Python-side resolution) ===")
        fixed_b_con = get_connection(
            memory_limit=cfg["duckdb"]["memory_limit"],
            threads=cfg["duckdb"]["threads"],
        )
        fixed_b_query = f"""
        SELECT
            COALESCE(o.overture_id, 'fema_only_' || f.build_id) AS building_id,
            o.overture_id,
            f.build_id AS fema_id,
            b.iou                           AS fema_iou,
            o.height,
            o.num_floors,
            o.class                         AS overture_class,
            o.names,
            o.geometry                      AS overture_geometry,
            f.geometry                      AS fema_geometry,
            CASE
                WHEN o.overture_id IS NOT NULL THEN 'overture'
                WHEN f.build_id IS NOT NULL THEN 'fema_only'
            END AS source
        FROM read_parquet('{overture_path}') o
        FULL OUTER JOIN read_parquet('{fema_bridge_path}') b
            ON o.overture_id = b.overture_id
        FULL OUTER JOIN read_parquet('{fema_path}') f
            ON b.fema_id = f.build_id
        """

        fixed_b_succeeded = False
        try:
            log.info("Executing fixed query B...")
            arrow_tbl_b = fixed_b_con.sql(fixed_b_query).to_arrow_table()
            log.info(f"Fixed query B SUCCEEDED: {arrow_tbl_b.num_rows:,} rows returned")
            fixed_b_succeeded = True
            fixed_b_con.close()
        except Exception as exc:
            log.error(f"Fixed query B FAILED: {exc}")
            try:
                fixed_b_con.close()
            except Exception:
                pass

        if fixed_b_succeeded:
            log.info("=== Resolving geometry column in Python (two-column approach) ===")
            import pandas as pd
            df_b = arrow_tbl_b.to_pandas()
            # Pick overture_geometry when available, else fema_geometry
            import geopandas as gpd_local
            from shapely import from_wkb

            def _geom_from_col(row):
                geom_bytes = row["overture_geometry"] if row["overture_id"] is not None and not pd.isna(row["overture_id"]) else row["fema_geometry"]
                if geom_bytes is None:
                    return None
                # Geometry may already be a shapely object or bytes
                if hasattr(geom_bytes, "__geo_interface__"):
                    return geom_bytes
                if isinstance(geom_bytes, (bytes, bytearray)):
                    return from_wkb(geom_bytes)
                return geom_bytes

        if fixed_b_succeeded:
            log.info("=== Resolving geometry column in Python (two-column approach) ===")
            # Try to build GeoDataFrame directly from the two-col arrow table.
            # geopandas can read multi-geometry arrow tables.
            try:
                import geopandas as _gpd
                gdf_b = _gpd.GeoDataFrame.from_arrow(arrow_tbl_b)
                has_ov = gdf_b["overture_id"].notna()
                gdf_b["geometry"] = None
                gdf_b.loc[has_ov, "geometry"] = gdf_b.loc[has_ov, "overture_geometry"]
                gdf_b.loc[~has_ov, "geometry"] = gdf_b.loc[~has_ov, "fema_geometry"]
                gdf_b = gdf_b.drop(columns=["overture_geometry", "fema_geometry"])
                gdf_b = gdf_b.set_geometry("geometry").set_crs("EPSG:4326", allow_override=True)
                log.info(f"Fixed query B GeoDataFrame built: {len(gdf_b):,} rows")
                return _finish(gdf_b, log)
            except Exception as exc:
                log.error(f"Fixed query B GeoDataFrame conversion FAILED: {exc}")

    # ---- Fixed query C: Python-merge approach --------------------------------
    # FULL OUTER JOIN without geometry columns (avoids DuckDB assertion),
    # then merge geometries in Python from individual per-table reads.
    # This bypasses the DuckDB bug with geometry columns in FULL OUTER JOINs.
    log.info("\n=== Fixed query C (Python-merge: JOIN without geometry + per-table geometry reads) ===")
    fixed_c_con = get_connection(
        memory_limit=cfg["duckdb"]["memory_limit"],
        threads=cfg["duckdb"]["threads"],
    )
    fixed_c_query = f"""
    SELECT
        COALESCE(o.overture_id, 'fema_only_' || f.build_id) AS building_id,
        o.overture_id,
        f.build_id AS fema_id,
        b.iou                           AS fema_iou,
        o.height,
        o.num_floors,
        o.class                         AS overture_class,
        o.names,
        CASE
            WHEN o.overture_id IS NOT NULL THEN 'overture'
            WHEN f.build_id IS NOT NULL THEN 'fema_only'
        END AS source
    FROM read_parquet('{overture_path}') o
    FULL OUTER JOIN read_parquet('{fema_bridge_path}') b
        ON o.overture_id = b.overture_id
    FULL OUTER JOIN read_parquet('{fema_path}') f
        ON b.fema_id = f.build_id
    """

    try:
        log.info("  Step 1: FULL OUTER JOIN without geometry columns...")
        c1 = get_connection(memory_limit=cfg["duckdb"]["memory_limit"], threads=cfg["duckdb"]["threads"])
        arrow_join = c1.sql(fixed_c_query).to_arrow_table()
        c1.close()
        log.info(f"  JOIN rows: {arrow_join.num_rows:,}")

        # Steps 2+3: Read geometry columns directly via geopandas/pyarrow,
        # bypassing DuckDB entirely.  DuckDB's spatial extension enters a broken
        # state after several connections are opened/closed in the same process,
        # causing TransactionContext errors on geometry reads in fresh connections.
        # Geopandas reads GeoParquet natively via pyarrow, which is unaffected.
        log.info("  Step 2: Read overture_id → geometry via geopandas (bypass DuckDB)...")
        overture_geom_gdf = gpd.read_parquet(overture_path, columns=["overture_id", "geometry"])
        log.info(f"  Overture geometry rows: {len(overture_geom_gdf):,}")

        log.info("  Step 3: Read build_id → geometry via geopandas (bypass DuckDB)...")
        # FEMA parquet stores the column as BUILD_ID (original GDB case);
        # DuckDB auto-lowercases it to build_id.  Use the original case here.
        fema_geom_gdf = gpd.read_parquet(fema_path, columns=["BUILD_ID", "geometry"])
        fema_geom_gdf = fema_geom_gdf.rename(columns={"BUILD_ID": "build_id"})
        log.info(f"  FEMA geometry rows: {len(fema_geom_gdf):,}")

        fixed_c_con.close()

        log.info("  Step 4: Merge geometries in Python...")
        import pandas as _pd

        join_df = arrow_join.to_pandas()

        # Normalize both geometry series to EPSG:4326
        overture_geom_gdf = overture_geom_gdf.to_crs("EPSG:4326")
        fema_geom_gdf = fema_geom_gdf.to_crs("EPSG:4326")

        overture_geom = overture_geom_gdf.set_index("overture_id")["geometry"]
        fema_geom = fema_geom_gdf.set_index("build_id")["geometry"]

        join_df = join_df.merge(
            overture_geom.rename("overture_geom"), left_on="overture_id", right_index=True, how="left"
        )
        join_df = join_df.merge(
            fema_geom.rename("fema_geom"), left_on="fema_id", right_index=True, how="left"
        )

        has_ov = join_df["overture_id"].notna()
        join_df["geometry"] = _pd.concat([
            join_df.loc[has_ov, "overture_geom"],
            join_df.loc[~has_ov, "fema_geom"],
        ]).reindex(join_df.index)
        join_df = join_df.drop(columns=["overture_geom", "fema_geom"])

        gdf_c = gpd.GeoDataFrame(join_df, geometry="geometry", crs="EPSG:4326")
        log.info(f"  Fixed query C GeoDataFrame built: {len(gdf_c):,} rows")
        return _finish(gdf_c, log)

    except Exception as exc:
        log.error(f"Fixed query C FAILED: {exc}")
        try:
            fixed_c_con.close()
        except Exception:
            pass
        raise RuntimeError("All query strategies failed. See errors above.") from exc

    if original_failed and not fixed_a_succeeded:
        raise RuntimeError("All query strategies failed. See errors above.")

    # ---- GeoDataFrame conversion --------------------------------------------
    log.info("\n=== Converting Arrow table to GeoDataFrame ===")
    try:
        gdf = gpd.GeoDataFrame.from_arrow(arrow_tbl)
        gdf = gdf.set_geometry("geometry").set_crs("EPSG:4326")
    except Exception as exc:
        log.error(f"GeoDataFrame conversion FAILED:\n{exc}")
        raise

    return _finish(gdf, log)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="miami_dade", help="AOI name (default: miami_dade)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    storage = StorageConfig.from_config(CONFIG_PATH)

    log.info(f"Starting debug run for AOI: {args.aoi}")
    gdf = resolve_building_geometries(args.aoi, cfg, storage)
    log.info(f"Done. Result shape: {gdf.shape}")
    log.info(f"Columns: {list(gdf.columns)}")


if __name__ == "__main__":
    main()
