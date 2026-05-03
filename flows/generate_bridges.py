"""
Flow 2: Generate Bridges — GERSite Building Conflation Pipeline.

Generates Silver-layer bridge files that cross-walk source IDs without
moving geometries (the "Linker" pattern):

  - FEMA-Base bridge: IoU-based spatial join between FEMA structures and
    Overture buildings. Records with IoU >= min_iou become bridge rows;
    unmatched FEMA records become additive candidates in Flow 3.

  - NSI-Base bridge: Point-in-polygon join mapping NSI structure points
    to Overture building footprints.

The Overture-OSM bridge was downloaded and staged in Flow 1. It is NOT
used in this flow — it will be activated in a future deployment phase.

This file is a Marimo notebook.
  Script mode:   python flows/generate_bridges.py --aoi saipan
  Notebook mode: marimo edit flows/generate_bridges.py

Config keys used (config.gers.yaml):
  storage.*             — Bronze/silver paths
  bridges.fema_min_iou  — IoU threshold for FEMA bridge inclusion
  duckdb.*              — Memory and thread limits
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _setup():
    """Imports, config, and lib loading."""
    import sys
    from pathlib import Path

    import duckdb
    import marimo as mo
    import yaml
    from prefect import flow, task, get_run_logger

    REPO_ROOT = Path(__file__).parent.parent
    CONFIG_PATH = REPO_ROOT / "config.gers.yaml"
    LIB_PATH = REPO_ROOT / "lib"
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))

    from duckdb_helpers import StorageConfig, get_connection
    from spatial_utils import compute_iou_sql

    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)

    STORAGE = StorageConfig.from_config(CONFIG_PATH)
    AOI_CHOICES = list(CFG["aoi"].keys())

    mo.md("## Flow 2: Generate Bridges")
    return (
        CFG, CONFIG_PATH, LIB_PATH, REPO_ROOT, STORAGE, AOI_CHOICES,
        Path, duckdb, flow, get_run_logger, mo, sys, task, yaml,
        StorageConfig, get_connection, compute_iou_sql,
    )


@app.cell
def _aoi_selector(mo, AOI_CHOICES):
    """AOI selection."""
    aoi_dropdown = mo.ui.dropdown(
        options=AOI_CHOICES,
        value=AOI_CHOICES[0],
        label="Select study area (AOI)",
    )
    aoi_dropdown
    return (aoi_dropdown,)


@app.cell
def _resolve_aoi(aoi_dropdown, CFG, mo, sys):
    """Resolve AOI from UI or --aoi CLI argument."""
    aoi_name = None
    args = sys.argv[1:]
    if "--aoi" in args:
        idx = args.index("--aoi")
        if idx + 1 < len(args):
            aoi_name = args[idx + 1]
    if aoi_name is None:
        aoi_name = aoi_dropdown.value
    aoi_label = CFG["aoi"][aoi_name]["label"]
    mo.md(f"**Active AOI:** {aoi_label} (`{aoi_name}`)")
    return (aoi_name, aoi_label)


# ---------------------------------------------------------------------------
# Task: FEMA-Base bridge via IoU
# ---------------------------------------------------------------------------


@app.cell
def _task_fema_bridge(
    CFG, STORAGE, Path, get_connection, compute_iou_sql,
    flow, task, get_run_logger, mo,
):
    """FEMA-Base IoU bridge task."""

    @task(name="generate-fema-bridge", retries=1)
    def generate_fema_bridge(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        min_iou = cfg["bridges"]["fema_min_iou"]
        mem = cfg["duckdb"]["memory_limit"]
        threads = cfg["duckdb"]["threads"]

        fema_path = Path(storage.bronze_path("fema_structures")) / aoi / "structures.parquet"
        overture_path = Path(storage.bronze_path("overture_buildings")) / aoi / "buildings.parquet"
        out_path = Path(storage.silver_path("fema_bridge")).parent / aoi / "fema_bridge.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not fema_path.exists():
            raise FileNotFoundError(
                f"FEMA structures not found: {fema_path}. Run Flow 1 first."
            )
        if not overture_path.exists():
            raise FileNotFoundError(
                f"Overture buildings not found: {overture_path}. Run Flow 1 first."
            )

        iou_col = compute_iou_sql("f.geometry", "o.geometry", "iou")

        logger.info(f"Generating FEMA-Base IoU bridge for AOI {aoi} ...")
        con = get_connection(memory_limit=mem, threads=threads)

        # Two-stage reciprocal best-match:
        #   Stage 1 (best_per_fema): for each FEMA record keep the Overture
        #            building with the highest IoU.
        #   Stage 2 (reciprocal): if multiple FEMA records claim the same
        #            Overture building, keep only the one with the highest IoU.
        # This guarantees a one-to-one bridge on both sides.
        query = f"""
        COPY (
            WITH candidates AS (
                SELECT
                    f.fema_id,
                    o.overture_id,
                    {iou_col},
                    ST_AsText(ST_Centroid(f.geometry)) AS fema_centroid_wkt
                FROM read_parquet('{fema_path}') f
                JOIN read_parquet('{overture_path}') o
                  ON ST_Intersects(f.geometry, o.geometry)
                WHERE ST_Area(ST_Intersection(f.geometry, o.geometry)) /
                      NULLIF(ST_Area(ST_Union(f.geometry, o.geometry)), 0) >= {min_iou}
            ),
            best_per_fema AS (
                SELECT *
                FROM candidates
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY fema_id ORDER BY iou DESC
                ) = 1
            )
            SELECT fema_id, overture_id, iou, fema_centroid_wkt
            FROM best_per_fema
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY overture_id ORDER BY iou DESC
            ) = 1
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
        logger.info(f"FEMA bridge: {count:,} matches (IoU >= {min_iou}) → {out_path}")
        con.close()
        return out_path

    mo.md("### Task: FEMA-Base bridge — defined")
    return (generate_fema_bridge,)


# ---------------------------------------------------------------------------
# Task: NSI-Base bridge via point-in-polygon
# ---------------------------------------------------------------------------


@app.cell
def _task_nsi_bridge(
    CFG, STORAGE, Path, get_connection,
    flow, task, get_run_logger, mo,
):
    """NSI point-in-polygon bridge task."""

    @task(name="generate-nsi-bridge", retries=1)
    def generate_nsi_bridge(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        mem = cfg["duckdb"]["memory_limit"]
        threads = cfg["duckdb"]["threads"]

        nsi_path = Path(storage.bronze_path("nsi_structures")) / aoi / "structures.parquet"
        overture_path = Path(storage.bronze_path("overture_buildings")) / aoi / "buildings.parquet"
        out_path = Path(storage.silver_path("nsi_bridge")).parent / aoi / "nsi_bridge.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not nsi_path.exists():
            raise FileNotFoundError(
                f"NSI structures not found: {nsi_path}. Run Flow 1 first."
            )
        if not overture_path.exists():
            raise FileNotFoundError(
                f"Overture buildings not found: {overture_path}. Run Flow 1 first."
            )

        logger.info(f"Generating NSI-Base point-in-polygon bridge for AOI {aoi} ...")
        con = get_connection(memory_limit=mem, threads=threads)

        query = f"""
        COPY (
            SELECT
                n.nsi_id,
                o.overture_id,
                n.occtype                               AS nsi_occtype,
                n.val_struct                            AS nsi_val_struct,
                n.val_cont                              AS nsi_val_cont,
                ST_AsText(n.geometry)                   AS nsi_point_wkt
            FROM read_parquet('{nsi_path}') n
            JOIN read_parquet('{overture_path}') o
              ON ST_Within(n.geometry, o.geometry)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY n.nsi_id
                ORDER BY ST_Area(o.geometry) ASC
            ) = 1
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]

        # Also write unmatched NSI points — these represent NSI-recorded structures
        # that have no Overture building footprint. Used by Flow 3 to compute
        # the NSI risk flag (structures with NSI data but no known footprint).
        unmatched_path = out_path.parent / "nsi_unmatched.parquet"
        unmatched_query = f"""
        COPY (
            SELECT
                n.nsi_id,
                n.occtype       AS nsi_occtype,
                n.val_struct    AS nsi_val_struct,
                n.val_cont      AS nsi_val_cont,
                ST_AsText(n.geometry) AS nsi_point_wkt
            FROM read_parquet('{nsi_path}') n
            WHERE NOT EXISTS (
                SELECT 1 FROM read_parquet('{out_path}') b
                WHERE b.nsi_id = n.nsi_id
            )
        ) TO '{unmatched_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(unmatched_query)
        unmatched_count = con.execute(
            f"SELECT COUNT(*) FROM '{unmatched_path}'"
        ).fetchone()[0]

        logger.info(
            f"NSI bridge: {count:,} matched, {unmatched_count:,} unmatched "
            f"(no Overture footprint) → {out_path}"
        )
        con.close()
        return out_path

    mo.md("### Task: NSI-Base bridge — defined")
    return (generate_nsi_bridge,)


# ---------------------------------------------------------------------------
# Bridge diagnostics cell
# ---------------------------------------------------------------------------


@app.cell
def _bridge_diagnostics(
    STORAGE, CFG, aoi_name, Path, get_connection, mo,
):
    """Show bridge match rate statistics (notebook mode only)."""
    import os

    fema_bridge = Path(STORAGE.silver_path("fema_bridge")).parent / aoi_name / "fema_bridge.parquet"
    fema_bronze = Path(STORAGE.bronze_path("fema_structures")) / aoi_name / "structures.parquet"
    nsi_bridge = Path(STORAGE.silver_path("nsi_bridge")).parent / aoi_name / "nsi_bridge.parquet"
    nsi_bronze = Path(STORAGE.bronze_path("nsi_structures")) / aoi_name / "structures.parquet"

    stats = []
    if fema_bridge.exists() and fema_bronze.exists():
        con = get_connection()
        total = con.execute(f"SELECT COUNT(*) FROM '{fema_bronze}'").fetchone()[0]
        matched = con.execute(f"SELECT COUNT(*) FROM '{fema_bridge}'").fetchone()[0]
        con.close()
        stats.append(f"**FEMA bridge:** {matched:,} / {total:,} matched ({matched/total*100:.1f}%)")

    if nsi_bridge.exists() and nsi_bronze.exists():
        con = get_connection()
        total = con.execute(f"SELECT COUNT(*) FROM '{nsi_bronze}'").fetchone()[0]
        matched = con.execute(f"SELECT COUNT(*) FROM '{nsi_bridge}'").fetchone()[0]
        con.close()
        stats.append(f"**NSI bridge:** {matched:,} / {total:,} matched ({matched/total*100:.1f}%)")

    mo.md("### Bridge Diagnostics\n" + ("\n\n".join(stats) if stats else "_Run Flow 1 + Flow 2 first._"))
    return ()


# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------


@app.cell
def _flow_definition(
    generate_fema_bridge, generate_nsi_bridge,
    flow, CFG, STORAGE, mo,
):
    """Define the Prefect bridge-generation flow."""

    @flow(name="gers-generate-bridges", log_prints=True)
    def generate_bridges_flow(aoi: str = "saipan"):
        fema_bridge_file = generate_fema_bridge(aoi, CFG, STORAGE)
        nsi_bridge_file = generate_nsi_bridge(aoi, CFG, STORAGE)
        return {
            "fema_bridge": str(fema_bridge_file),
            "nsi_bridge": str(nsi_bridge_file),
        }

    mo.md(f"""
    ## Flow: `gers-generate-bridges`

    **Tasks:**
    1. `generate-fema-bridge` — IoU spatial join (Overture × FEMA)
    2. `generate-nsi-bridge` — Point-in-polygon join (NSI points → Overture footprints)

    Run:
    ```bash
    python flows/generate_bridges.py --aoi saipan
    ```
    """)
    return (generate_bridges_flow,)


@app.cell
def _run(generate_bridges_flow, aoi_name, mo):
    """Execute flow when run as a script."""
    if not mo.running_in_notebook():
        result = generate_bridges_flow(aoi=aoi_name)
        print(result)
    return ()


if __name__ == "__main__":
    app.run()
