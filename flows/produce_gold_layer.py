"""
Flow 3: Produce Gold Layer — GERSite Building Conflation Pipeline.

Merges all sources and bridge files into the unified Gold building layer:

  1. Geometry resolution: JOIN Overture buildings with FEMA via fema_bridge.
     Priority: Overture geometry > FEMA geometry (higher polygon rank wins).
     OSM ID column reserved (null) for future bridge activation.

  2. Additive step: FEMA records absent from fema_bridge are appended as
     new candidate features with source='fema_only' and confidence=0.3.

  3. NSI enrichment: LEFT JOIN the NSI bridge to add occupancy type and
     structural value attributes from the National Structure Inventory.

  4. Confidence scoring: Apply compute_confidence() and write Gold GeoParquet.
     from lib/scoring.py. Write final Gold GeoParquet partitioned by H3.

This file is a Marimo notebook.
  Script mode:   python flows/produce_gold_layer.py --aoi saipan
  Notebook mode: marimo edit flows/produce_gold_layer.py

Config keys used (config.gers.yaml):
  storage.*             — Bronze/silver/gold paths
  bridges.*             — IoU thresholds
  scoring.*             — Confidence level constants
  h3.*                  — H3 partition/sort resolutions
  attribution.*         — ODbL attribution metadata
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

    import geopandas as gpd
    import marimo as mo
    import numpy as np
    import pandas as pd
    import yaml
    from prefect import flow, task, get_run_logger

    REPO_ROOT = Path(__file__).parent.parent
    CONFIG_PATH = REPO_ROOT / "config.gers.yaml"
    LIB_PATH = REPO_ROOT / "lib"
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))

    from duckdb_helpers import StorageConfig, get_connection
    from occupancy import conflate_occupancy
    from scoring import compute_confidence, confidence_summary

    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)

    STORAGE = StorageConfig.from_config(CONFIG_PATH)
    AOI_CHOICES = list(CFG["aoi"].keys())

    mo.md("## Flow 3: Produce Gold Layer")
    return (
        AOI_CHOICES,
        CFG,
        Path,
        STORAGE,
        StorageConfig,
        compute_confidence,
        confidence_summary,
        conflate_occupancy,
        flow,
        get_connection,
        get_run_logger,
        gpd,
        mo,
        np,
        pd,
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
    aoi_label = CFG["aoi"][aoi_name]["label"]
    mo.md(f"**Active AOI:** {aoi_label} (`{aoi_name}`)")
    return (aoi_name,)


@app.cell
def _run_controls(aoi_name, mo):
    """Run button — click to trigger gold layer production in notebook mode."""
    run_btn = mo.ui.run_button(label="▶  Run gold layer")
    mo.hstack([mo.md(f"**AOI:** `{aoi_name}`"), run_btn], justify="start")
    return (run_btn,)


@app.cell
def _task_geometry_resolution(
    Path,
    StorageConfig,
    get_connection,
    get_run_logger,
    gpd,
    mo,
    pd,
    task,
):
    """Load Overture and FEMA via the bridge, resolve geometry priority."""

    @task(name="resolve-building-geometries", retries=1)
    def resolve_building_geometries(aoi: str, cfg: dict, storage: StorageConfig) -> gpd.GeoDataFrame:
        logger = get_run_logger()

        overture_path = Path(storage.bronze_path("overture_buildings")) / aoi / "buildings.parquet"
        fema_path = Path(storage.bronze_path("fema_structures")) / aoi / "structures.parquet"
        fema_bridge_path = (
            Path(storage.silver_path("fema_bridge")).parent / aoi / "fema_bridge.parquet"
        )

        for p in (overture_path, fema_path, fema_bridge_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file missing: {p}. Run Flow 1 + 2 first.")

        logger.info(f"Resolving geometries for AOI {aoi} ...")

        # Step 1: FULL OUTER JOIN without geometry columns.
        # DuckDB has an internal bug where reading geometry columns from GeoParquet
        # in a FULL OUTER JOIN triggers TransactionContext::ActiveTransaction errors,
        # regardless of spatial function used. Workaround: run the JOIN geometry-free,
        # then read geometry columns via geopandas (PyArrow-based, bypasses DuckDB spatial).
        con = get_connection(
            memory_limit=cfg["duckdb"]["memory_limit"],
            threads=cfg["duckdb"]["threads"],
        )
        join_query = f"""
        SELECT
            COALESCE(o.overture_id, 'fema_only_' || f.build_id) AS building_id,
            o.overture_id,
            f.build_id AS fema_id,
            b.iou                           AS fema_iou,
            o.height,
            o.num_floors,
            o.class                         AS overture_class,
            o.names,
            f.OCC_CLS                       AS fema_occ_cls,
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
        join_df = con.sql(join_query).df()
        con.close()

        # Aggregate multi-FEMA rows per Overture building.
        # Removing the one-FEMA-per-Overture constraint in the bridge means a large
        # Overture footprint may now have multiple rows here (one per matched FEMA).
        # Collapse them to one row per Overture by selecting the highest-IoU FEMA match
        # for attribute sourcing and recording fema_match_count for diagnostics.
        overture_mask = join_df["overture_id"].notna()
        fema_mask = join_df["fema_id"].notna()

        overture_with_fema = join_df[overture_mask & fema_mask]
        overture_only = join_df[overture_mask & ~fema_mask].copy()
        fema_only = join_df[~overture_mask & fema_mask].copy()

        if not overture_with_fema.empty:
            fema_counts = (
                overture_with_fema.groupby("overture_id", as_index=False)["fema_id"]
                .count()
                .rename(columns={"fema_id": "fema_match_count"})
            )
            best_per_overture = (
                overture_with_fema
                # Sort by IoU descending, then fema_id ascending for deterministic tie-breaking.
                .sort_values(["fema_iou", "fema_id"], ascending=[False, True], na_position="last")
                .drop_duplicates(subset="overture_id", keep="first")
                .merge(fema_counts, on="overture_id", how="left")
            )
        else:
            best_per_overture = overture_with_fema.copy()
            best_per_overture["fema_match_count"] = pd.Series(dtype="Int64")

        overture_only["fema_match_count"] = 0
        fema_only["fema_match_count"] = 1

        join_df = pd.concat(
            [best_per_overture, overture_only, fema_only], ignore_index=True
        )

        # Step 2: Read geometry columns via geopandas (bypasses DuckDB spatial extension).
        # FEMA parquet preserves original GDB column casing (BUILD_ID); DuckDB auto-lowercases.
        overture_geom = (
            gpd.read_parquet(overture_path, columns=["overture_id", "geometry"])
            .to_crs("EPSG:4326")
            .set_index("overture_id")["geometry"]
        )
        fema_geom = (
            gpd.read_parquet(fema_path, columns=["BUILD_ID", "geometry"])
            .rename(columns={"BUILD_ID": "build_id"})
            .to_crs("EPSG:4326")
            .set_index("build_id")["geometry"]
        )

        # Step 3: Merge geometries — prefer Overture, fall back to FEMA.
        join_df = join_df.merge(
            overture_geom.rename("overture_geom"), left_on="overture_id", right_index=True, how="left"
        )
        join_df = join_df.merge(
            fema_geom.rename("fema_geom"), left_on="fema_id", right_index=True, how="left"
        )
        has_ov = join_df["overture_id"].notna()
        join_df["geometry"] = pd.concat([
            join_df.loc[has_ov, "overture_geom"],
            join_df.loc[~has_ov, "fema_geom"],
        ]).reindex(join_df.index)
        join_df = join_df.drop(columns=["overture_geom", "fema_geom"])

        gdf = gpd.GeoDataFrame(join_df, geometry="geometry", crs="EPSG:4326")

        has_overture = gdf["overture_id"].notna().to_numpy()
        has_fema = gdf["fema_id"].notna().to_numpy()
        multi_fema = int((gdf.get("fema_match_count", pd.Series([0] * len(gdf))) > 1).sum())

        # Reserve OSM columns for future bridge activation
        gdf["osm_id"] = None
        gdf["osm_type"] = None

        logger.info(
            f"Geometry resolution complete: {len(gdf):,} buildings "
            f"({has_overture.sum():,} Overture, {has_fema.sum():,} FEMA matched, "
            f"{multi_fema:,} Overture buildings subsuming multiple FEMA footprints)"
        )
        return gdf

    mo.md("### Task: Geometry resolution — defined")
    return (resolve_building_geometries,)


@app.cell
def _task_nsi_enrichment(
    Path,
    StorageConfig,
    get_run_logger,
    gpd,
    mo,
    pd,
    task,
):
    """Join NSI bridge attributes onto the unified buildings GeoDataFrame."""

    @task(name="enrich-with-nsi")
    def enrich_with_nsi(
        gdf: gpd.GeoDataFrame,
        aoi: str,
        storage: StorageConfig,
    ) -> gpd.GeoDataFrame:
        logger = get_run_logger()
        nsi_bridge_path = (
            Path(storage.silver_path("nsi_bridge")).parent / aoi / "nsi_bridge.parquet"
        )

        if not nsi_bridge_path.exists():
            logger.warning(
                f"NSI bridge not found: {nsi_bridge_path}. "
                "NSI columns will be null. Run Flow 2 to generate."
            )
            gdf["nsi_id"] = None
            gdf["nsi_occtype"] = None
            gdf["nsi_val_struct"] = None
            gdf["nsi_val_cont"] = None
            gdf["nsi_point_count"] = 0
            gdf["has_nsi_match"] = False
            return gdf

        nsi_bridge = pd.read_parquet(nsi_bridge_path)

        # Aggregate multiple NSI points per building.
        # A single Overture (or FEMA) footprint may contain multiple NSI structure
        # points — e.g., a large multi-unit building or two detached units under one
        # roof polygon. Strategy:
        #   nsi_id / nsi_occtype  — from the highest-value point (primary representative)
        #   nsi_val_struct / nsi_val_cont — SUM across all points (total building value)
        #   nsi_point_count       — number of NSI points inside the footprint
        nsi_sorted = nsi_bridge.sort_values(
            ["nsi_val_struct", "nsi_id"], ascending=[False, True], na_position="last"
        )
        nsi_primary = nsi_sorted.drop_duplicates(subset="overture_id", keep="first")[
            ["overture_id", "nsi_id", "nsi_occtype"]
        ]
        nsi_agg = nsi_bridge.groupby("overture_id", as_index=False).agg(
            nsi_val_struct=("nsi_val_struct", "sum"),
            nsi_val_cont=("nsi_val_cont", "sum"),
            nsi_point_count=("nsi_id", "count"),
        )
        nsi_best = nsi_primary.merge(nsi_agg, on="overture_id", how="left")

        gdf = gdf.merge(nsi_best, on="overture_id", how="left")
        gdf["has_nsi_match"] = gdf["nsi_id"].notna()

        # Secondary NSI enrichment: FEMA-only buildings have no overture_id, so they
        # are never reached by the Overture-based NSI bridge join above.
        # Use the NSI-FEMA bridge (NSI points PIP-matched against FEMA geometries)
        # to fill NSI attributes for any FEMA-only rows still missing NSI data.
        nsi_fema_bridge_path = (
            Path(storage.silver_path("nsi_fema_bridge")).parent / aoi / "nsi_fema_bridge.parquet"
        )
        if nsi_fema_bridge_path.exists():
            nsi_fema_bridge = pd.read_parquet(nsi_fema_bridge_path)
            nsi_fema_sorted = nsi_fema_bridge.sort_values(
                ["nsi_val_struct", "nsi_id"], ascending=[False, True], na_position="last"
            )
            nsi_fema_primary = nsi_fema_sorted.drop_duplicates(subset="fema_id", keep="first")[
                ["fema_id", "nsi_id", "nsi_occtype"]
            ]
            nsi_fema_agg = nsi_fema_bridge.groupby("fema_id", as_index=False).agg(
                nsi_val_struct=("nsi_val_struct", "sum"),
                nsi_val_cont=("nsi_val_cont", "sum"),
                nsi_point_count=("nsi_id", "count"),
            )
            nsi_fema_best = nsi_fema_primary.merge(nsi_fema_agg, on="fema_id", how="left")

            fema_only_null_nsi = (gdf["source"] == "fema_only") & gdf["nsi_id"].isna()
            if fema_only_null_nsi.any():
                subset_idx = gdf.index[fema_only_null_nsi]
                # Preserve original index through the merge so values align correctly.
                subset = (
                    gdf.loc[subset_idx, ["fema_id"]]
                    .reset_index()
                    .merge(nsi_fema_best, on="fema_id", how="left")
                    .set_index("index")
                )
                nsi_cols = ["nsi_id", "nsi_occtype", "nsi_val_struct", "nsi_val_cont", "nsi_point_count"]
                gdf.loc[subset.index, nsi_cols] = subset[nsi_cols]
                gdf["has_nsi_match"] = gdf["nsi_id"].notna()
                fema_nsi_count = int(subset["nsi_id"].notna().sum())
                logger.info(
                    f"NSI-FEMA enrichment: {fema_nsi_count:,} FEMA-only buildings "
                    "matched with NSI records via FEMA geometry bridge"
                )
        else:
            logger.warning(
                f"NSI-FEMA bridge not found: {nsi_fema_bridge_path}. "
                "FEMA-only buildings will have null NSI attributes. Run Flow 2 to generate."
            )

        nsi_count = int(gdf["has_nsi_match"].sum())
        logger.info(f"NSI enrichment: {nsi_count:,} buildings matched with NSI records")
        return gdf

    mo.md("### Task: NSI enrichment — defined")
    return (enrich_with_nsi,)


@app.cell
def _task_scoring_and_output(
    Path,
    StorageConfig,
    compute_confidence,
    confidence_summary,
    conflate_occupancy,
    get_run_logger,
    gpd,
    mo,
    np,
    task,
):
    """Apply confidence scoring and write Gold layer as a single GeoParquet file."""

    @task(name="score-and-write-gold")
    def score_and_write_gold(
        gdf: gpd.GeoDataFrame,
        aoi: str,
        cfg: dict,
        storage: StorageConfig,
    ) -> Path:
        import json

        import pandas as pd
        import pyarrow.parquet as pq

        logger = get_run_logger()

        has_overture = gdf["overture_id"].notna().to_numpy()
        has_fema = gdf["fema_id"].notna().to_numpy()
        max_iou = np.where(has_fema, gdf["fema_iou"].fillna(0.0).to_numpy(), 0.0)

        high_iou = cfg["bridges"]["fema_high_iou"]
        gdf["conflation_confidence"] = compute_confidence(
            has_overture, has_fema, max_iou, high_iou_threshold=high_iou
        )

        summary = confidence_summary(gdf["conflation_confidence"].to_numpy())
        logger.info(
            f"Confidence summary — High: {summary['high']:,} ({summary['pct_high']}%), "
            f"Medium: {summary['medium']:,} ({summary['pct_medium']}%), "
            f"Low: {summary['low']:,} ({summary['pct_low']}%)"
        )

        # Occupancy conflation: merge FEMA OCC_CLS + NSI occtype into a
        # single canonical general_occupancy field with its own confidence score.
        _conflate_vec = np.frompyfunc(conflate_occupancy, 2, 2)
        _occ_cls = gdf.get("fema_occ_cls", pd.Series([None] * len(gdf)))
        _nsi_occ = gdf.get("nsi_occtype", pd.Series([None] * len(gdf)))
        _gen_occ, _occ_conf = _conflate_vec(
            _occ_cls.where(_occ_cls.notna(), None),
            _nsi_occ.where(_nsi_occ.notna(), None),
        )
        gdf["general_occupancy"] = pd.array(_gen_occ.tolist(), dtype=object)
        gdf["occupancy_confidence"] = pd.to_numeric(
            pd.array(_occ_conf.tolist(), dtype=object), errors="coerce"
        )

        occ_known = gdf["general_occupancy"].notna().sum()
        logger.info(
            f"Occupancy conflation: {occ_known:,} / {len(gdf):,} buildings "
            "have a general_occupancy classification"
        )

        # NSI risk review: truly unmatched NSI points — no Overture AND no FEMA footprint.
        # nsi_unmatched.parquet (from Flow 2) only anti-joins against the Overture bridge.
        # Filter here against nsi_fema_bridge so points matched via FEMA geometry don't
        # falsely appear as having no footprint.
        nsi_unmatched_path = (
            Path(storage.silver_path("nsi_bridge")).parent / aoi / "nsi_unmatched.parquet"
        )
        if nsi_unmatched_path.exists():
            nsi_unmatched = pd.read_parquet(nsi_unmatched_path)
            nsi_fema_check_path = (
                Path(storage.silver_path("nsi_fema_bridge")).parent / aoi / "nsi_fema_bridge.parquet"
            )
            if nsi_fema_check_path.exists():
                fema_matched_ids = pd.read_parquet(
                    nsi_fema_check_path, columns=["nsi_id"]
                )["nsi_id"]
                before = len(nsi_unmatched)
                nsi_unmatched = nsi_unmatched[~nsi_unmatched["nsi_id"].isin(fema_matched_ids)]
                logger.info(
                    f"NSI review: filtered {before - len(nsi_unmatched):,} points matched "
                    f"via FEMA bridge; {len(nsi_unmatched):,} truly unmatched remain"
                )
            review_dir = Path(storage.gold_path("nsi_review", aoi=aoi))
            review_dir.mkdir(parents=True, exist_ok=True)
            review_out = review_dir / "nsi_unmatched.parquet"
            # Convert WKT string column to real Point geometry for true GeoParquet output.
            nsi_unmatched_gdf = gpd.GeoDataFrame(
                nsi_unmatched.drop(columns=["nsi_point_wkt"]),
                geometry=gpd.GeoSeries.from_wkt(nsi_unmatched["nsi_point_wkt"]),
                crs="EPSG:4326",
            )
            nsi_unmatched_gdf.to_parquet(review_out, index=False, compression="zstd")
            logger.warning(
                f"NSI risk review: {len(nsi_unmatched_gdf):,} NSI structures have no "
                f"Overture/FEMA footprint → {review_out}"
            )

        out_dir = Path(storage.gold_path("buildings", aoi=aoi))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "buildings.parquet"

        # Ensure stable schema for the fema_match_count column before writing.
        if "fema_match_count" in gdf.columns:
            gdf["fema_match_count"] = gdf["fema_match_count"].astype("Int64")

        # Write GeoParquet then embed attribution metadata via pyarrow.
        gdf.to_parquet(out_path, index=False, compression="zstd")

        attribution = cfg.get("attribution", {})
        if attribution:
            table = pq.read_table(out_path)
            existing_meta = table.schema.metadata or {}
            new_meta = {**existing_meta, b"gers_attribution": json.dumps(attribution).encode()}
            pq.write_table(table.replace_schema_metadata(new_meta), out_path, compression="zstd")

        logger.info(f"Gold layer written: {len(gdf):,} buildings → {out_path}")
        return out_path

    mo.md("### Task: Confidence scoring + Gold output — defined")
    return (score_and_write_gold,)


@app.cell
def _gold_diagnostics(Path, STORAGE, aoi_name, mo):
    """Display Gold layer stats if output exists."""
    import pyarrow.parquet as pq

    gold_file = Path(STORAGE.gold_path("buildings", aoi=aoi_name)) / "buildings.parquet"

    if gold_file.exists():
        meta = pq.read_metadata(gold_file)
        total = meta.num_rows
        mo.md(f"""
    ### Gold Layer — `{aoi_name}`

    **Output:** `{gold_file}`
    **Total buildings:** {total:,}
        """)
    else:
        mo.md("_Gold layer not yet produced. Run all three flows first._")
    return


@app.cell
def _flow_definition(
    CFG,
    STORAGE,
    enrich_with_nsi,
    flow,
    mo,
    resolve_building_geometries,
    score_and_write_gold,
):
    """Define the full Gold layer production flow."""

    @flow(name="gers-produce-gold-layer", log_prints=True)
    def produce_gold_layer_flow(aoi: str = "saipan"):
        gdf = resolve_building_geometries(aoi, CFG, STORAGE)
        gdf = enrich_with_nsi(gdf, aoi, STORAGE)
        out_dir = score_and_write_gold(gdf, aoi, CFG, STORAGE)
        return {"gold_output": str(out_dir)}

    mo.md(f"""
    ## Flow: `gers-produce-gold-layer`

    **Tasks:**
    1. `resolve-building-geometries` — FULL OUTER JOIN Overture + FEMA; Overture > FEMA geometry priority
    2. `enrich-with-nsi` — LEFT JOIN NSI bridge; adds `nsi_occtype`, `nsi_val_struct`, `nsi_point_count`
    3. `score-and-write-gold` — Confidence scoring + H3-partitioned GeoParquet output

    Run:
    ```bash
    python flows/produce_gold_layer.py --aoi saipan
    ```
    """)
    return (produce_gold_layer_flow,)


@app.cell
def _notebook_run(
    CFG,
    STORAGE,
    aoi_name,
    enrich_with_nsi,
    mo,
    resolve_building_geometries,
    run_btn,
    score_and_write_gold,
):
    """Execute tasks directly in notebook mode, gated by run button."""
    if mo.running_in_notebook():
        mo.stop(
            not run_btn.value,
            mo.md("☝️ Click **▶ Run gold layer** above to start."),
        )
        _aoi = aoi_name
        _gdf = resolve_building_geometries(_aoi, CFG, STORAGE)
        _gdf = enrich_with_nsi(_gdf, _aoi, STORAGE)
        _out_dir = score_and_write_gold(_gdf, _aoi, CFG, STORAGE)
        print({"gold_output": str(_out_dir)})
    return


@app.cell
def _run(aoi_name, mo, produce_gold_layer_flow):
    """Execute flow when run as a script."""
    if not mo.running_in_notebook():
        result = produce_gold_layer_flow(aoi=aoi_name)
        print(result)
    return


if __name__ == "__main__":
    app.run()
