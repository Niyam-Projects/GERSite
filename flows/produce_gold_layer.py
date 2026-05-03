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
    from scoring import compute_confidence, confidence_summary
    from spatial_utils import (
        pick_preferred_geometry,
        write_h3_partitioned_dataset,
    )

    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)

    STORAGE = StorageConfig.from_config(CONFIG_PATH)
    AOI_CHOICES = list(CFG["aoi"].keys())

    mo.md("## Flow 3: Produce Gold Layer")
    return (
        CFG, CONFIG_PATH, LIB_PATH, REPO_ROOT, STORAGE, AOI_CHOICES,
        Path, flow, get_run_logger, gpd, mo, np, pd, sys, task, yaml,
        StorageConfig, get_connection,
        compute_confidence, confidence_summary,
        pick_preferred_geometry, write_h3_partitioned_dataset,
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
# Task: Geometry resolution — Overture + FEMA via bridge
# ---------------------------------------------------------------------------


@app.cell
def _task_geometry_resolution(
    CFG, STORAGE, Path, get_connection,
    flow, task, get_run_logger, mo, gpd, np, pd,
    pick_preferred_geometry,
):
    """Load Overture and FEMA via the bridge, resolve geometry priority."""

    @task(name="resolve-building-geometries", retries=1)
    def resolve_building_geometries(aoi: str, cfg: dict, storage: StorageConfig) -> gpd.GeoDataFrame:
        logger = get_run_logger()
        import shapely

        overture_path = Path(storage.bronze_path("overture_buildings")) / aoi / "buildings.parquet"
        fema_path = Path(storage.bronze_path("fema_structures")) / aoi / "structures.parquet"
        fema_bridge_path = (
            Path(storage.silver_path("fema_bridge")).parent / aoi / "fema_bridge.parquet"
        )

        for p in (overture_path, fema_path, fema_bridge_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file missing: {p}. Run Flow 1 + 2 first.")

        con = get_connection(
            memory_limit=cfg["duckdb"]["memory_limit"],
            threads=cfg["duckdb"]["threads"],
        )

        logger.info(f"Resolving geometries for AOI {aoi} ...")

        # FULL OUTER JOIN Overture + FEMA via bridge
        query = f"""
        SELECT
            COALESCE(o.overture_id, 'fema_only_' || f.fema_id) AS building_id,
            o.overture_id,
            f.fema_id,
            b.iou                           AS fema_iou,
            o.height,
            o.num_floors,
            o.class                         AS overture_class,
            o.names,
            -- Geometry columns (WKB → resolved in Python for rank selection)
            o.geometry                      AS overture_geom,
            f.geometry                      AS fema_geom,
            CASE
                WHEN o.overture_id IS NOT NULL THEN 'overture'
                WHEN f.fema_id IS NOT NULL THEN 'fema_only'
            END AS source
        FROM read_parquet('{overture_path}') o
        FULL OUTER JOIN read_parquet('{fema_bridge_path}') b
            ON o.overture_id = b.overture_id
        FULL OUTER JOIN read_parquet('{fema_path}') f
            ON b.fema_id = f.fema_id
        """
        df = con.execute(query).df()
        con.close()

        # Resolve geometry: Overture > FEMA (higher polygon rank wins)
        overture_geoms = shapely.from_wkb(
            df["overture_geom"].where(df["overture_geom"].notna())
        )
        fema_geoms = shapely.from_wkb(
            df["fema_geom"].where(df["fema_geom"].notna())
        )

        # Fill None with the other source's geometry
        has_overture = df["overture_id"].notna().to_numpy()
        has_fema = df["fema_id"].notna().to_numpy()

        resolved_geoms = np.where(
            has_overture & has_fema,
            pick_preferred_geometry(overture_geoms, fema_geoms),
            np.where(has_overture, overture_geoms, fema_geoms),
        )

        gdf = gpd.GeoDataFrame(
            df.drop(columns=["overture_geom", "fema_geom"]),
            geometry=resolved_geoms,
            crs="EPSG:4326",
        )

        # Reserve OSM columns for future bridge activation
        gdf["osm_id"] = None
        gdf["osm_type"] = None

        logger.info(
            f"Geometry resolution complete: {len(gdf):,} buildings "
            f"({has_overture.sum():,} Overture, {has_fema.sum():,} FEMA matched)"
        )
        return gdf

    mo.md("### Task: Geometry resolution — defined")
    return (resolve_building_geometries,)


# ---------------------------------------------------------------------------
# Task: NSI enrichment
# ---------------------------------------------------------------------------


@app.cell
def _task_nsi_enrichment(
    STORAGE, Path, task, get_run_logger, mo, gpd, pd,
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
            gdf["has_nsi_match"] = False
            return gdf

        nsi_bridge = pd.read_parquet(nsi_bridge_path)

        # Aggregate to one row per overture_id to prevent fan-out duplicates
        # when multiple NSI points fall within the same building footprint.
        # Use the record with the highest structural value as representative.
        nsi_best = (
            nsi_bridge
            .sort_values("nsi_val_struct", ascending=False, na_position="last")
            .drop_duplicates(subset="overture_id", keep="first")
        )[["overture_id", "nsi_id", "nsi_occtype", "nsi_val_struct", "nsi_val_cont"]]

        gdf = gdf.merge(nsi_best, on="overture_id", how="left")
        gdf["has_nsi_match"] = gdf["nsi_id"].notna()

        nsi_count = int(gdf["has_nsi_match"].sum())
        logger.info(f"NSI enrichment: {nsi_count:,} buildings matched with NSI records")
        return gdf

    mo.md("### Task: NSI enrichment — defined")
    return (enrich_with_nsi,)


# ---------------------------------------------------------------------------
# Task: Confidence scoring + Gold output
# ---------------------------------------------------------------------------


@app.cell
def _task_scoring_and_output(
    CFG, STORAGE, Path,
    task, get_run_logger, mo, gpd,
    compute_confidence, confidence_summary,
    write_h3_partitioned_dataset, np,
):
    """Apply confidence scoring and write partitioned Gold layer."""

    @task(name="score-and-write-gold")
    def score_and_write_gold(
        gdf: gpd.GeoDataFrame,
        aoi: str,
        cfg: dict,
        storage: StorageConfig,
    ) -> Path:
        import pandas as pd

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

        # NSI risk review: unmatched NSI points represent structures recorded in
        # NSI but absent from Overture/FEMA. These cannot appear as rows in the
        # Gold layer (no footprint → nothing to merge onto), so they are written
        # to a separate review file rather than a column on Gold layer buildings.
        nsi_unmatched_path = (
            Path(storage.silver_path("nsi_bridge")).parent / aoi / "nsi_unmatched.parquet"
        )
        if nsi_unmatched_path.exists():
            nsi_unmatched = pd.read_parquet(nsi_unmatched_path)
            review_dir = Path(storage.gold_path("nsi_review", aoi=aoi))
            review_dir.mkdir(parents=True, exist_ok=True)
            review_out = review_dir / "nsi_unmatched.parquet"
            nsi_unmatched.to_parquet(review_out, index=False, compression="zstd")
            logger.warning(
                f"NSI risk review: {len(nsi_unmatched):,} NSI structures have no "
                f"Overture/FEMA footprint → {review_out}"
            )

        out_dir = Path(storage.gold_path("buildings", aoi=aoi))

        attribution = cfg.get("attribution", {})
        write_h3_partitioned_dataset(
            gdf,
            output_dir=out_dir,
            partition_resolution=cfg["h3"]["partition_resolution"],
            sort_resolution=cfg["h3"]["sort_resolution"],
            overwrite=True,
            attribution_metadata=attribution,
        )
        logger.info(f"Gold layer written: {len(gdf):,} buildings → {out_dir}")
        return out_dir

    mo.md("### Task: Confidence scoring + Gold output — defined")
    return (score_and_write_gold,)


# ---------------------------------------------------------------------------
# Gold layer diagnostics cell
# ---------------------------------------------------------------------------


@app.cell
def _gold_diagnostics(STORAGE, CFG, aoi_name, Path, mo):
    """Display Gold layer stats if output exists."""
    import glob as glob_module

    gold_dir = Path(STORAGE.gold_path("buildings", aoi=aoi_name))
    part_files = list(gold_dir.rglob("part-*.parquet")) if gold_dir.exists() else []

    if part_files:
        import pyarrow.parquet as pq
        import pyarrow.dataset as ds
        dataset = ds.dataset(gold_dir, format="parquet")
        total = dataset.count_rows()
        mo.md(f"""
### Gold Layer — `{aoi_name}`

**Output:** `{gold_dir}`
**Partitions:** {len(part_files)}
**Total buildings:** {total:,}
        """)
    else:
        mo.md("_Gold layer not yet produced. Run all three flows first._")
    return ()


# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------


@app.cell
def _flow_definition(
    resolve_building_geometries, enrich_with_nsi, score_and_write_gold,
    flow, CFG, STORAGE, mo,
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
2. `enrich-with-nsi` — LEFT JOIN NSI bridge; adds `nsi_occtype`, `nsi_val_struct`
3. `score-and-write-gold` — Confidence scoring + H3-partitioned GeoParquet output

Run:
```bash
python flows/produce_gold_layer.py --aoi saipan
```
    """)
    return (produce_gold_layer_flow,)


@app.cell
def _run(produce_gold_layer_flow, aoi_name, mo):
    """Execute flow when run as a script."""
    if not mo.running_in_notebook():
        result = produce_gold_layer_flow(aoi=aoi_name)
        print(result)
    return ()


if __name__ == "__main__":
    app.run()
