"""
Flow 2: Generate Bridges — GERSite Building Conflation Pipeline.

Generates Silver-layer bridge files that cross-walk source IDs without
moving geometries (the "Linker" pattern):

  - FEMA-Base bridge: IoU-based spatial join between FEMA structures and
    Overture buildings. Records with IoU >= min_iou become bridge rows;
    unmatched FEMA records become additive candidates in Flow 3.

  - NSI-Base bridge: Three-pass join mapping NSI structure points to Overture
    building footprints. Pass 1 = strict point-in-polygon; Pass 2 = convex-hull
    PIP for concave buildings; Pass 3 = nearest-neighbor within a configurable
    buffer radius (bridges.nsi_nearest_buffer_m) for points just outside the
    footprint.

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

__generated_with = "0.23.4"
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
        AOI_CHOICES,
        CFG,
        Path,
        STORAGE,
        StorageConfig,
        compute_iou_sql,
        flow,
        get_connection,
        get_run_logger,
        mo,
        sys,
        task,
    )


@app.cell
def _aoi_selector(AOI_CHOICES, mo):
    """AOI selection."""
    aoi_dropdown = mo.ui.dropdown(
        options=AOI_CHOICES,
        value=AOI_CHOICES[0],
        label="Select study area (AOI)",
    )
    aoi_dropdown
    return (aoi_dropdown,)


@app.cell
def _resolve_aoi(CFG, aoi_dropdown, mo, sys):
    """Resolve AOI from UI or --aoi CLI argument."""
    aoi_name = None
    args = sys.argv[1:]
    if "--aoi" in args:
        idx = args.index("--aoi")
        if idx + 1 < len(args):
            aoi_name = args[idx + 1]
    if aoi_name is None:
        aoi_name = aoi_dropdown.value
    aoi_name = aoi_name.replace("-", "_")
    aoi_label = CFG["aoi"][aoi_name]["label"]
    mo.md(f"**Active AOI:** {aoi_label} (`{aoi_name}`)")
    return (aoi_name,)


@app.cell
def _run_controls(aoi_name, mo):
    """Run button — click to trigger bridge generation in notebook mode."""
    run_btn = mo.ui.run_button(label="▶  Run bridges")
    mo.hstack([mo.md(f"**AOI:** `{aoi_name}`"), run_btn], justify="start")
    return (run_btn,)


@app.cell
def _task_fema_bridge(
    Path,
    StorageConfig,
    compute_iou_sql,
    get_connection,
    get_run_logger,
    mo,
    task,
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

        # Two-stage match — keeps the best Overture building per FEMA record:
        #   Stage 1 (candidates): IoU-filtered cross join of FEMA × Overture.
        #   Stage 2 (best_per_fema): for each FEMA record keep the Overture
        #            building with the highest IoU. Each FEMA can only belong
        #            to one Overture building.
        # One Overture building may match multiple FEMA records (many-to-one
        # is intentional: a large Overture footprint can subsume several
        # smaller FEMA structures). Flow 3 collapses those to one Gold row.
        query = f"""
        COPY (
            WITH fema_src AS (
                SELECT build_id as fema_id, ST_SetCRS(geometry, 'EPSG:4326') AS geometry
                FROM read_parquet('{fema_path}')
            ),
            overture_src AS (
                SELECT overture_id, ST_SetCRS(geometry, 'EPSG:4326') AS geometry
                FROM read_parquet('{overture_path}')
            ),
            candidates AS (
                SELECT
                    f.fema_id,
                    o.overture_id,
                    {iou_col},
                    ST_AsText(ST_Centroid(f.geometry)) AS fema_centroid_wkt
                FROM fema_src f
                JOIN overture_src o
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
            -- Keep all FEMA matches per Overture building (many-to-one allowed).
            -- A large Overture footprint may subsume multiple FEMA structures;
            -- Flow 3 aggregates them into a single Gold record using best-IoU scoring.
            SELECT fema_id, overture_id, iou, fema_centroid_wkt
            FROM best_per_fema
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
        logger.info(f"FEMA bridge: {count:,} matches (IoU >= {min_iou}) → {out_path}")
        con.close()
        return out_path

    mo.md("### Task: FEMA-Base bridge — defined")
    return (generate_fema_bridge,)


@app.cell
def _task_nsi_bridge(
    Path,
    StorageConfig,
    get_connection,
    get_run_logger,
    mo,
    task,
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
            logger.warning(
                f"NSI structures not found for AOI {aoi} — skipping NSI-Overture bridge "
                f"(territory may not have NSI coverage). NSI columns will be null in Gold layer."
            )
            return out_path
        if not overture_path.exists():
            raise FileNotFoundError(
                f"Overture buildings not found: {overture_path}. Run Flow 1 first."
            )

        buffer_m = cfg["bridges"].get("nsi_nearest_buffer_m", 10)
        # Planar degree approximation (good enough for small search radii at low latitudes).
        buffer_deg = buffer_m / 111_320

        logger.info(f"Generating NSI-Base point-in-polygon bridge for AOI {aoi} ...")
        con = get_connection(memory_limit=mem, threads=threads)

        # Three-pass approach:
        #   Pass 1 (pip_matches): strict ST_Within — NSI point inside Overture polygon.
        #   Pass 2 (convex_hull_matches): ST_Within against convex hull of Overture polygon,
        #     used only for NSI points that failed Pass 1. This catches points that sit in the
        #     interior void of horseshoe/C-shaped buildings where the literal polygon has a
        #     concave cavity (e.g., a courtyard), making strict PIP fail.
        #   Pass 3 (nearest_matches): ST_DWithin within buffer_deg for NSI points that failed
        #     Passes 1 & 2. Picks the closest Overture building by ST_Distance. Handles points
        #     that lie just outside the building footprint (e.g., snapped to the wrong side).
        # All passes keep the best Overture building per NSI point.
        # A match_method column records which pass produced each row.
        query = f"""
        COPY (
            WITH nsi_src AS (
                SELECT nsi_id, occtype, val_struct, val_cont,
                       ST_SetCRS(geometry, 'EPSG:4326') AS geometry
                FROM read_parquet('{nsi_path}')
            ),
            overture_src AS (
                SELECT overture_id, ST_SetCRS(geometry, 'EPSG:4326') AS geometry
                FROM read_parquet('{overture_path}')
            ),
            pip_matches AS (
                SELECT
                    n.nsi_id,
                    o.overture_id,
                    n.occtype                               AS nsi_occtype,
                    n.val_struct                            AS nsi_val_struct,
                    n.val_cont                              AS nsi_val_cont,
                    ST_AsText(n.geometry)                   AS nsi_point_wkt,
                    'point_in_polygon'                      AS match_method
                FROM nsi_src n
                JOIN overture_src o
                  ON ST_Within(n.geometry, o.geometry)
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY n.nsi_id
                    ORDER BY ST_Area(o.geometry) ASC
                ) = 1
            ),
            convex_hull_matches AS (
                SELECT
                    n.nsi_id,
                    o.overture_id,
                    n.occtype                               AS nsi_occtype,
                    n.val_struct                            AS nsi_val_struct,
                    n.val_cont                              AS nsi_val_cont,
                    ST_AsText(n.geometry)                   AS nsi_point_wkt,
                    'convex_hull'                           AS match_method
                FROM nsi_src n
                JOIN overture_src o
                  ON ST_Within(n.geometry, ST_ConvexHull(o.geometry))
                WHERE NOT EXISTS (
                    SELECT 1 FROM pip_matches p WHERE p.nsi_id = n.nsi_id
                )
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY n.nsi_id
                    ORDER BY ST_Area(o.geometry) ASC
                ) = 1
            ),
            nearest_matches AS (
                SELECT
                    n.nsi_id,
                    o.overture_id,
                    n.occtype                               AS nsi_occtype,
                    n.val_struct                            AS nsi_val_struct,
                    n.val_cont                              AS nsi_val_cont,
                    ST_AsText(n.geometry)                   AS nsi_point_wkt,
                    'nearest_neighbor'                      AS match_method
                FROM nsi_src n
                JOIN overture_src o
                  ON ST_DWithin(n.geometry, o.geometry, {buffer_deg})
                WHERE NOT EXISTS (
                    SELECT 1 FROM pip_matches p WHERE p.nsi_id = n.nsi_id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM convex_hull_matches c WHERE c.nsi_id = n.nsi_id
                )
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY n.nsi_id
                    ORDER BY ST_Distance(n.geometry, o.geometry) ASC
                ) = 1
            )
            SELECT * FROM pip_matches
            UNION ALL
            SELECT * FROM convex_hull_matches
            UNION ALL
            SELECT * FROM nearest_matches
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
        pip_count = con.execute(
            f"SELECT COUNT(*) FROM '{out_path}' WHERE match_method = 'point_in_polygon'"
        ).fetchone()[0]
        hull_count = con.execute(
            f"SELECT COUNT(*) FROM '{out_path}' WHERE match_method = 'convex_hull'"
        ).fetchone()[0]
        nn_count = con.execute(
            f"SELECT COUNT(*) FROM '{out_path}' WHERE match_method = 'nearest_neighbor'"
        ).fetchone()[0]
        count = pip_count + hull_count + nn_count

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
            f"NSI bridge: {pip_count:,} PIP + {hull_count:,} convex-hull + "
            f"{nn_count:,} nearest-neighbor = "
            f"{count:,} matched, {unmatched_count:,} unmatched "
            f"(no Overture footprint within {buffer_m} m) → {out_path}"
        )
        con.close()
        return out_path

    mo.md("### Task: NSI-Base bridge — defined")
    return (generate_nsi_bridge,)


@app.cell
def _task_nsi_fema_bridge(
    Path,
    StorageConfig,
    get_connection,
    get_run_logger,
    mo,
    task,
):
    """NSI point-in-polygon bridge against FEMA building geometries."""

    @task(name="generate-nsi-fema-bridge", retries=1)
    def generate_nsi_fema_bridge(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        """Map NSI structure points onto FEMA building footprints.

        Produces a bridge analogous to the Overture NSI bridge but using FEMA
        building geometries as the polygon source. Flow 3 uses this to enrich
        FEMA-only Gold records (buildings with no Overture footprint) with NSI
        occupancy and structural-value attributes.
        """
        logger = get_run_logger()
        mem = cfg["duckdb"]["memory_limit"]
        threads = cfg["duckdb"]["threads"]

        nsi_path = Path(storage.bronze_path("nsi_structures")) / aoi / "structures.parquet"
        fema_path = Path(storage.bronze_path("fema_structures")) / aoi / "structures.parquet"
        out_path = Path(storage.silver_path("nsi_fema_bridge")).parent / aoi / "nsi_fema_bridge.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not nsi_path.exists():
            logger.warning(
                f"NSI structures not found for AOI {aoi} — skipping NSI-FEMA bridge "
                f"(territory may not have NSI coverage). FEMA-only buildings will have null NSI attributes."
            )
            return out_path
        if not fema_path.exists():
            raise FileNotFoundError(
                f"FEMA structures not found: {fema_path}. Run Flow 1 first."
            )

        logger.info(f"Generating NSI-FEMA point-in-polygon bridge for AOI {aoi} ...")
        con = get_connection(memory_limit=mem, threads=threads)

        query = f"""
        COPY (
            WITH nsi_src AS (
                SELECT nsi_id, occtype, val_struct, val_cont,
                       ST_SetCRS(geometry, 'EPSG:4326') AS geometry
                FROM read_parquet('{nsi_path}')
            ),
            fema_src AS (
                SELECT build_id AS fema_id, ST_SetCRS(geometry, 'EPSG:4326') AS geometry
                FROM read_parquet('{fema_path}')
            )
            SELECT
                n.nsi_id,
                f.fema_id,
                n.occtype                               AS nsi_occtype,
                n.val_struct                            AS nsi_val_struct,
                n.val_cont                              AS nsi_val_cont,
                ST_AsText(n.geometry)                   AS nsi_point_wkt
            FROM nsi_src n
            JOIN fema_src f
              ON ST_Within(n.geometry, f.geometry)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY n.nsi_id
                ORDER BY ST_Area(f.geometry) ASC
            ) = 1
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
        logger.info(f"NSI-FEMA bridge: {count:,} NSI points matched to FEMA buildings → {out_path}")
        con.close()
        return out_path

    mo.md("### Task: NSI-FEMA bridge — defined")
    return (generate_nsi_fema_bridge,)


@app.cell
def _bridge_diagnostics(Path, STORAGE, aoi_name, get_connection, mo):
    """Show bridge match rate statistics (notebook mode only)."""
    import os

    if mo.running_in_notebook():
        fema_bridge = Path(STORAGE.silver_path("fema_bridge")).parent / aoi_name / "fema_bridge.parquet"
        fema_bronze = Path(STORAGE.bronze_path("fema_structures")) / aoi_name / "structures.parquet"
        nsi_bridge = Path(STORAGE.silver_path("nsi_bridge")).parent / aoi_name / "nsi_bridge.parquet"
        nsi_bronze = Path(STORAGE.bronze_path("nsi_structures")) / aoi_name / "structures.parquet"
        stats = []
        if fema_bridge.exists() and fema_bronze.exists():
            con = get_connection()
            total = con.execute(f"SELECT COUNT(*) FROM '{fema_bronze}'").fetchone()[0]
            matched = con.execute(f"SELECT COUNT(DISTINCT fema_id) FROM '{fema_bridge}'").fetchone()[0]
            overture_count = con.execute(f"SELECT COUNT(DISTINCT overture_id) FROM '{fema_bridge}'").fetchone()[0]
            con.close()
            stats.append(
                f"**FEMA bridge:** {matched:,} / {total:,} FEMA matched ({matched/total*100:.1f}%) "
                f"→ {overture_count:,} Overture buildings (many-to-one allowed)"
            )

        nsi_fema_bridge = Path(STORAGE.silver_path("nsi_fema_bridge")).parent / aoi_name / "nsi_fema_bridge.parquet"
        if nsi_bridge.exists() and nsi_bronze.exists():
            con = get_connection()
            total = con.execute(f"SELECT COUNT(*) FROM '{nsi_bronze}'").fetchone()[0]
            matched = con.execute(f"SELECT COUNT(*) FROM '{nsi_bridge}'").fetchone()[0]
            pip = con.execute(f"SELECT COUNT(*) FROM '{nsi_bridge}' WHERE match_method = 'point_in_polygon'").fetchone()[0]
            hull = con.execute(f"SELECT COUNT(*) FROM '{nsi_bridge}' WHERE match_method = 'convex_hull'").fetchone()[0]
            nn = con.execute(f"SELECT COUNT(*) FROM '{nsi_bridge}' WHERE match_method = 'nearest_neighbor'").fetchone()[0]
            con.close()
            stats.append(
                f"**NSI-Overture bridge:** {matched:,} / {total:,} matched ({matched/total*100:.1f}%) "
                f"— {pip:,} PIP + {hull:,} convex-hull + {nn:,} nearest-neighbor"
            )

        if nsi_fema_bridge.exists() and nsi_bronze.exists():
            con = get_connection()
            total = con.execute(f"SELECT COUNT(*) FROM '{nsi_bronze}'").fetchone()[0]
            matched = con.execute(f"SELECT COUNT(*) FROM '{nsi_fema_bridge}'").fetchone()[0]
            con.close()
            stats.append(f"**NSI-FEMA bridge:** {matched:,} / {total:,} matched ({matched/total*100:.1f}%)")

        mo.md("### Bridge Diagnostics\n" + ("\n\n".join(stats) if stats else "_Run Flow 1 + Flow 2 first._"))
    return


@app.cell
def _flow_definition(
    CFG,
    STORAGE,
    flow,
    generate_fema_bridge,
    generate_nsi_bridge,
    generate_nsi_fema_bridge,
    mo,
):
    """Define the Prefect bridge-generation flow."""

    @flow(name="gers-generate-bridges", log_prints=True)
    def generate_bridges_flow(aoi: str = "saipan"):
        fema_bridge_file = generate_fema_bridge(aoi, CFG, STORAGE)
        nsi_bridge_file = generate_nsi_bridge(aoi, CFG, STORAGE)
        nsi_fema_bridge_file = generate_nsi_fema_bridge(aoi, CFG, STORAGE)
        return {
            "fema_bridge": str(fema_bridge_file),
            "nsi_bridge": str(nsi_bridge_file),
            "nsi_fema_bridge": str(nsi_fema_bridge_file),
        }

    mo.md(f"""
    ## Flow: `gers-generate-bridges`

    **Tasks:**
    1. `generate-fema-bridge` — IoU spatial join (Overture × FEMA, many-FEMA-per-Overture allowed)
    2. `generate-nsi-bridge` — PIP + convex-hull + nearest-neighbor fallback join (NSI points → Overture footprints)
    3. `generate-nsi-fema-bridge` — PIP join (NSI points → FEMA footprints, for FEMA-only enrichment)

    Run:
    ```bash
    python flows/generate_bridges.py --aoi saipan
    ```
    """)
    return (generate_bridges_flow,)


@app.cell
def _notebook_run(
    CFG,
    STORAGE,
    aoi_name,
    generate_fema_bridge,
    generate_nsi_bridge,
    generate_nsi_fema_bridge,
    mo,
    run_btn,
):
    """Execute tasks directly in notebook mode, gated by run button."""
    if mo.running_in_notebook():
        mo.stop(
            not run_btn.value,
            mo.md("☝️ Click **▶ Run bridges** above to start."),
        )
        _aoi = aoi_name
        _fema_bridge_file = generate_fema_bridge(_aoi, CFG, STORAGE)
        _nsi_bridge_file = generate_nsi_bridge(_aoi, CFG, STORAGE)
        _nsi_fema_bridge_file = generate_nsi_fema_bridge(_aoi, CFG, STORAGE)
        print({
            "fema_bridge": str(_fema_bridge_file),
            "nsi_bridge": str(_nsi_bridge_file),
            "nsi_fema_bridge": str(_nsi_fema_bridge_file),
        })
    return


@app.cell
def _run(aoi_name, generate_bridges_flow, mo):
    """Execute flow when run as a script."""
    if not mo.running_in_notebook():
        result = generate_bridges_flow(aoi=aoi_name)
        print(result)
    return


if __name__ == "__main__":
    app.run()
