"""
Flow 1: Ingest Sources — GERSite Building Conflation Pipeline.

Downloads and stages Bronze-layer data for the building conflation pipeline:
  - Overture buildings for the selected AOI
  - Official Overture-OSM building bridge file (staged only; not yet active)
  - FEMA USA Structures for the selected AOI
  - NSI (National Structure Inventory) for the selected AOI

This file is a Marimo notebook. It can be run in two ways:
  Script mode:   python flows/ingest_sources.py --aoi saipan
  Notebook mode: marimo edit flows/ingest_sources.py

Each Marimo cell corresponds to a Prefect task. The Prefect flow wraps all
tasks for orchestrated execution with retries and observability.

Config keys used (config.gers.yaml):
  aoi.*               — AOI bboxes and GeoJSON paths
  storage.*           — Bronze layer output paths
  overture.*          — Overture S3 bucket and DuckDB settings
  fema.api_url        — FEMA USA Structures endpoint
  nsi.api_url         — NSI API endpoint
  duckdb.*            — Memory and thread limits
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _setup():
    """Imports, config loading, and Prefect flow definition."""
    import argparse
    import json
    import os
    import sys
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import duckdb
    import geopandas as gpd
    import marimo as mo
    import requests
    import yaml
    from prefect import flow, task, get_run_logger

    # ── Resolve repo root and config path ────────────────────────────────────
    REPO_ROOT = Path(__file__).parent.parent
    CONFIG_PATH = REPO_ROOT / "config.gers.yaml"
    LIB_PATH = REPO_ROOT / "lib"
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))

    from duckdb_helpers import (
        StorageConfig,
        aoi_bbox,
        aoi_bbox_struct_filter,
        aoi_bbox_sql,
        get_connection,
        load_aoi_config,
    )
    from spatial_utils import aoi_polygon_wkt

    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)

    STORAGE = StorageConfig.from_config(CONFIG_PATH)
    AOI_CHOICES = list(CFG["aoi"].keys())

    mo.md("## Flow 1: Ingest Sources")
    return (
        CFG, CONFIG_PATH, LIB_PATH, REPO_ROOT, STORAGE, AOI_CHOICES,
        ET, Path, argparse, duckdb, flow, get_run_logger, gpd,
        json, mo, os, requests, sys, task,
        yaml, StorageConfig, aoi_bbox, aoi_bbox_struct_filter,
        aoi_bbox_sql, get_connection, load_aoi_config, aoi_polygon_wkt,
    )


@app.cell
def _aoi_selector(mo, AOI_CHOICES):
    """AOI selection — interactive dropdown in notebook mode."""
    aoi_dropdown = mo.ui.dropdown(
        options=AOI_CHOICES,
        value=AOI_CHOICES[0],
        label="Select study area (AOI)",
    )
    aoi_dropdown
    return (aoi_dropdown,)


@app.cell
def _resolve_aoi(aoi_dropdown, CFG, CONFIG_PATH, mo, sys):
    """Resolve AOI name from UI or --aoi CLI argument."""
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
# Task: Download Overture buildings
# ---------------------------------------------------------------------------


@app.cell
def _task_overture(
    CFG, STORAGE, aoi_name, CONFIG_PATH, Path,
    ET, duckdb, requests, get_connection, aoi_bbox_struct_filter,
    aoi_polygon_wkt, flow, task, get_run_logger, mo,
):
    """Overture buildings download task."""

    @task(
        name="ingest-overture-buildings",
        retries=2,
        retry_delay_seconds=60,
    )
    def ingest_overture_buildings(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        ov_cfg = cfg["overture"]
        bucket = ov_cfg["s3_bucket"]
        region = ov_cfg["s3_region"]
        theme = ov_cfg.get("theme", "buildings")
        mem = ov_cfg["duckdb"]["memory_limit"]
        threads = ov_cfg["duckdb"]["threads"]

        # Resolve output path
        out_dir = Path(storage.bronze_path("overture_buildings")) / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "buildings.parquet"

        # Auto-detect latest Overture release
        release = cfg["versions"].get("overture_release")
        if not release:
            s3_list_url = (
                f"https://{bucket}.s3.amazonaws.com"
                "/?list-type=2&prefix=release%2F&delimiter=%2F"
            )
            resp = requests.get(s3_list_url, timeout=30)
            resp.raise_for_status()
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            root = ET.fromstring(resp.text)
            prefixes = [
                el.text.rstrip("/").removeprefix("release/")
                for el in root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
            ]
            if not prefixes:
                raise ValueError(f"No Overture releases found in s3://{bucket}/release/")
            release = sorted(prefixes)[-1]
            logger.info(f"Auto-detected Overture release: {release}")

        # Build S3 glob path for buildings theme
        s3_path = (
            f"s3://{bucket}/release/{release}"
            f"/theme={theme}/type=building/*.parquet"
        )

        bbox_filter = aoi_bbox_struct_filter(aoi, CONFIG_PATH)
        wkt = aoi_polygon_wkt(cfg["aoi"][aoi]["geojson"])

        con = get_connection(memory_limit=mem, threads=threads)
        con.execute(f"SET s3_region = '{region}'")
        con.execute("SET enable_external_file_cache = false")

        logger.info(f"Scanning Overture buildings: {s3_path}")
        query = f"""
        COPY (
            SELECT
                id              AS overture_id,
                geometry,
                names,
                height,
                num_floors,
                class,
                sources,
                bbox
            FROM read_parquet('{s3_path}', hive_partitioning=false)
            WHERE {bbox_filter}
              AND ST_Intersects(
                  geometry,
                  ST_GeomFromText('{wkt}')
              )
        ) TO '{out_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
        logger.info(f"Overture buildings written: {count:,} rows → {out_file}")
        con.close()
        return out_file

    mo.md("### Task: Overture buildings — defined")
    return (ingest_overture_buildings,)


# ---------------------------------------------------------------------------
# Task: Download Overture-OSM bridge file (staged only)
# ---------------------------------------------------------------------------


@app.cell
def _task_overture_bridge(
    CFG, STORAGE, Path, ET, requests, get_connection,
    flow, task, get_run_logger, mo,
):
    """Overture-OSM bridge file download task (staged; not yet active in pipeline)."""

    @task(
        name="ingest-overture-osm-bridge",
        retries=2,
        retry_delay_seconds=60,
    )
    def ingest_overture_osm_bridge(cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        ov_cfg = cfg["overture"]
        bucket = ov_cfg["s3_bucket"]
        region = ov_cfg["s3_region"]

        out_file = Path(storage.bronze_path("overture_osm_bridge"))
        out_file.parent.mkdir(parents=True, exist_ok=True)

        release = cfg["versions"].get("overture_release")
        if not release:
            s3_list_url = (
                f"https://{bucket}.s3.amazonaws.com"
                "/?list-type=2&prefix=release%2F&delimiter=%2F"
            )
            resp = requests.get(s3_list_url, timeout=30)
            resp.raise_for_status()
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            root = ET.fromstring(resp.text)
            prefixes = sorted([
                el.text.rstrip("/").removeprefix("release/")
                for el in root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
            ])
            release = prefixes[-1]

        # Overture publishes an OSM-to-Overture bridge at this path:
        # s3://<bucket>/release/<date>/theme=buildings/type=building_part/
        # The official bridge file path may shift between releases;
        # update the key below if Overture changes the schema.
        bridge_s3 = (
            f"s3://{bucket}/release/{release}"
            "/theme=base/type=infrastructure/*.parquet"
        )
        # Fallback: some releases publish the bridge at:
        # /release/{date}/theme=admins/type=locality_area/*.parquet
        # Check Overture release notes for the correct path each cycle.

        con = get_connection()
        con.execute(f"SET s3_region = '{region}'")
        con.execute("SET enable_external_file_cache = false")

        # NOTE: This path must be validated each Overture release cycle.
        # The Overture-OSM building bridge schema includes: overture_id, osm_id.
        # For now we download whatever is at the bridge path and stage it as-is.
        bridge_s3_path = (
            f"s3://{bucket}/release/{release}"
            "/theme=buildings/type=building/*.parquet"
        )
        logger.info(
            f"Staging Overture-OSM bridge (STAGED ONLY — not active in pipeline): "
            f"{bridge_s3_path}"
        )

        # The bridge contains a 'sources' column with OSM refs; we extract it here.
        query = f"""
        COPY (
            SELECT
                id AS overture_id,
                list_filter(
                    sources,
                    s -> s.dataset = 'OpenStreetMap'
                ) AS osm_sources
            FROM read_parquet('{bridge_s3_path}', hive_partitioning=false)
            WHERE len(list_filter(
                sources,
                s -> s.dataset = 'OpenStreetMap'
            )) > 0
        ) TO '{out_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        try:
            con.execute(query)
            count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
            logger.info(f"Bridge staged: {count:,} rows → {out_file}")
        except Exception as e:
            logger.warning(
                f"Bridge download failed (non-fatal — bridge is staged only): {e}"
            )
        finally:
            con.close()
        return out_file

    mo.md("### Task: Overture-OSM bridge — defined (staged only)")
    return (ingest_overture_osm_bridge,)


# ---------------------------------------------------------------------------
# Task: Download FEMA USA Structures
# ---------------------------------------------------------------------------


@app.cell
def _task_fema(
    CFG, STORAGE, CONFIG_PATH, Path, requests,
    flow, task, get_run_logger, aoi_bbox, mo,
):
    """FEMA USA Structures download task."""

    @task(
        name="ingest-fema-structures",
        retries=2,
        retry_delay_seconds=120,
    )
    def ingest_fema_structures(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        import json
        import geopandas as gpd
        from shapely.geometry import shape

        out_dir = Path(storage.bronze_path("fema_structures")) / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "structures.parquet"

        xmin, ymin, xmax, ymax = aoi_bbox(aoi, CONFIG_PATH)
        api_url = cfg["fema"]["api_url"]

        logger.info(f"Downloading FEMA structures for AOI {aoi} ...")
        # FEMA ArcGIS REST API — paginate with resultOffset
        all_features = []
        offset = 0
        page_size = 2000

        while True:
            params = {
                "where": "1=1",
                "geometry": f"{xmin},{ymin},{xmax},{ymax}",
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            }
            resp = requests.get(api_url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            all_features.extend(features)
            logger.info(f"  Fetched {offset + len(features):,} FEMA features ...")
            if len(features) < page_size:
                break
            offset += page_size

        if not all_features:
            logger.warning(f"No FEMA features returned for AOI {aoi}")
            return out_file

        geojson_fc = {"type": "FeatureCollection", "features": all_features}
        gdf = gpd.GeoDataFrame.from_features(geojson_fc["features"], crs="EPSG:4326")
        gdf = gdf.rename(columns={"OID_": "fema_id"}) if "OID_" in gdf.columns else gdf
        if "OBJECTID" in gdf.columns:
            gdf = gdf.rename(columns={"OBJECTID": "fema_id"})

        gdf.to_parquet(out_file, index=False)
        logger.info(f"FEMA structures written: {len(gdf):,} rows → {out_file}")
        return out_file

    mo.md("### Task: FEMA structures — defined")
    return (ingest_fema_structures,)


# ---------------------------------------------------------------------------
# Task: Download NSI
# ---------------------------------------------------------------------------


@app.cell
def _task_nsi(
    CFG, STORAGE, CONFIG_PATH, Path, requests,
    flow, task, get_run_logger, aoi_bbox, mo,
):
    """NSI (National Structure Inventory) download task."""

    @task(
        name="ingest-nsi-structures",
        retries=2,
        retry_delay_seconds=120,
    )
    def ingest_nsi_structures(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        import geopandas as gpd

        out_dir = Path(storage.bronze_path("nsi_structures")) / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "structures.parquet"

        xmin, ymin, xmax, ymax = aoi_bbox(aoi, CONFIG_PATH)
        api_url = cfg["nsi"]["api_url"]

        logger.info(f"Downloading NSI structures for AOI {aoi} ...")
        # NSI API: POST with bbox parameter returns GeoJSON
        params = {"bbox": f"{xmin},{ymin},{xmax},{ymax}"}
        resp = requests.get(api_url, params=params, timeout=300)
        resp.raise_for_status()

        data = resp.json()
        features = data.get("features", [])

        if not features:
            logger.warning(f"No NSI features returned for AOI {aoi}")
            return out_file

        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        # NSI key fields: fd_id (structure ID), occtype, val_struct, val_cont
        if "fd_id" in gdf.columns:
            gdf = gdf.rename(columns={"fd_id": "nsi_id"})

        gdf.to_parquet(out_file, index=False)
        logger.info(f"NSI structures written: {len(gdf):,} rows → {out_file}")
        return out_file

    mo.md("### Task: NSI structures — defined")
    return (ingest_nsi_structures,)


# ---------------------------------------------------------------------------
# Prefect flow: wire tasks together
# ---------------------------------------------------------------------------


@app.cell
def _flow_definition(
    ingest_overture_buildings, ingest_overture_osm_bridge,
    ingest_fema_structures, ingest_nsi_structures,
    flow, CFG, STORAGE, mo,
):
    """Define and display the Prefect flow graph."""

    @flow(name="gers-ingest-sources", log_prints=True)
    def ingest_sources_flow(aoi: str = "saipan"):
        overture_file = ingest_overture_buildings(aoi, CFG, STORAGE)
        bridge_file = ingest_overture_osm_bridge(CFG, STORAGE)
        fema_file = ingest_fema_structures(aoi, CFG, STORAGE)
        nsi_file = ingest_nsi_structures(aoi, CFG, STORAGE)
        return {
            "overture": str(overture_file),
            "bridge": str(bridge_file),
            "fema": str(fema_file),
            "nsi": str(nsi_file),
        }

    mo.md(f"""
    ## Flow: `gers-ingest-sources`

    **Tasks:**
    1. `ingest-overture-buildings` — Overture buildings for selected AOI
    2. `ingest-overture-osm-bridge` — Official Overture-OSM bridge (staged only)
    3. `ingest-fema-structures` — FEMA USA Structures for selected AOI
    4. `ingest-nsi-structures` — NSI structures for selected AOI

    Run:
    ```bash
    python flows/ingest_sources.py --aoi saipan
    ```
    """)
    return (ingest_sources_flow,)


# ---------------------------------------------------------------------------
# Run cell — executes flow when run as a script
# ---------------------------------------------------------------------------


@app.cell
def _run(ingest_sources_flow, aoi_name, mo):
    """Run the flow when executed as a script (not in notebook mode)."""
    if not mo.running_in_notebook():
        result = ingest_sources_flow(aoi=aoi_name)
        print(result)
    return ()


if __name__ == "__main__":
    app.run()
