"""
Flow 1: Ingest Sources — GERSite Building Conflation Pipeline.

Downloads and stages Bronze-layer data for the building conflation pipeline:
  - Overture buildings for the selected AOI
  - Overture-OSM building bridge file (AOI-filtered; OSM cross-references for the selected AOI)
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
  fema.download_index_url — FEMA USA Structures state GDB download page
  nsi.download_base_url — NSI per-state GeoPackage download base URL
  duckdb.*            — Memory and thread limits
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _setup():
    """Imports, config loading, and Prefect flow definition."""
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
        AOI_CHOICES,
        CFG,
        CONFIG_PATH,
        ET,
        Path,
        STORAGE,
        StorageConfig,
        aoi_bbox,
        aoi_bbox_struct_filter,
        aoi_polygon_wkt,
        flow,
        get_connection,
        get_run_logger,
        mo,
        requests,
        task,
    )


@app.cell
def _aoi_selector(AOI_CHOICES, mo):
    """AOI selection — interactive dropdown in notebook mode."""
    aoi_dropdown = mo.ui.dropdown(
        options=AOI_CHOICES,
        value=AOI_CHOICES[0],
        label="Select study area (AOI)",
    )
    aoi_dropdown
    return (aoi_dropdown,)


@app.cell
def _resolve_aoi(AOI_CHOICES, CFG, aoi_dropdown, mo):
    """Resolve AOI name from UI or --aoi CLI argument."""
    import argparse

    if mo.running_in_notebook():
        aoi_name = aoi_dropdown.value
    else:
        parser = argparse.ArgumentParser(description="Ingest Sources flow")
        parser.add_argument(
            "--aoi",
            default=AOI_CHOICES[0],
            help="Study area to ingest (e.g. saipan, guam, miami_dade)",
        )
        args, _ = parser.parse_known_args()
        aoi_name = args.aoi

    aoi_name = aoi_name.replace("-", "_")
    aoi_label = CFG["aoi"][aoi_name]["label"]
    mo.md(f"**Active AOI:** {aoi_label} (`{aoi_name}`)")
    return (aoi_name,)


@app.cell
def _run_controls(aoi_name, mo):
    """Run button — click to trigger ingestion in notebook mode."""
    run_btn = mo.ui.run_button(label="▶  Run ingestion")
    mo.hstack([mo.md(f"**AOI:** `{aoi_name}`"), run_btn], justify="start")
    return (run_btn,)


@app.cell
def _task_overture(
    CONFIG_PATH,
    ET,
    Path,
    StorageConfig,
    aoi_bbox_struct_filter,
    aoi_polygon_wkt,
    get_connection,
    get_run_logger,
    mo,
    requests,
    task,
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

        # Overture publishes data in WGS84 / EPSG:4326 by spec — no reprojection needed.
        logger.info(f"Scanning Overture buildings: {s3_path}")
        query = f"""
        COPY (
            SELECT
                id                          AS overture_id,
                ST_MakeValid(geometry)      AS geometry,
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
            ORDER BY ST_Hilbert(geometry,
                                ST_Extent(ST_MakeEnvelope(-180, -90, 180, 90)))
        ) TO '{out_file}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
        logger.info(f"Overture buildings written: {count:,} rows → {out_file}")
        con.close()
        return out_file

    mo.md("### Task: Overture buildings — defined")
    return (ingest_overture_buildings,)


@app.cell
def _task_overture_bridge(
    CONFIG_PATH,
    ET,
    Path,
    StorageConfig,
    aoi_bbox_struct_filter,
    aoi_polygon_wkt,
    get_connection,
    get_run_logger,
    mo,
    requests,
    task,
):
    """Overture-OSM bridge file download task — AOI-filtered."""

    @task(
        name="ingest-overture-osm-bridge",
        retries=2,
        retry_delay_seconds=60,
    )
    def ingest_overture_osm_bridge(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()
        ov_cfg = cfg["overture"]
        bucket = ov_cfg["s3_bucket"]
        region = ov_cfg["s3_region"]
        mem = ov_cfg["duckdb"]["memory_limit"]
        threads = ov_cfg["duckdb"]["threads"]

        out_dir = Path(storage.bronze_path("overture_osm_bridge")) / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "bridge.parquet"

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

        # The bridge data lives in the same Overture buildings parquet;
        # we extract only records with OSM source IDs, spatially filtered to the AOI.
        bridge_s3_path = (
            f"s3://{bucket}/release/{release}"
            "/theme=buildings/type=building/*.parquet"
        )

        bbox_filter = aoi_bbox_struct_filter(aoi, CONFIG_PATH)
        wkt = aoi_polygon_wkt(cfg["aoi"][aoi]["geojson"])

        con = get_connection(memory_limit=mem, threads=threads)
        con.execute(f"SET s3_region = '{region}'")
        con.execute("SET enable_external_file_cache = false")

        logger.info(f"Scanning Overture-OSM bridge for {aoi}: {bridge_s3_path}")
        query = f"""
        COPY (
            SELECT
                id AS overture_id,
                list_filter(
                    sources,
                    s -> s.dataset = 'OpenStreetMap'
                ) AS osm_sources
            FROM read_parquet('{bridge_s3_path}', hive_partitioning=false)
            WHERE {bbox_filter}
              AND ST_Intersects(
                  geometry,
                  ST_GeomFromText('{wkt}')
              )
              AND len(list_filter(
                  sources,
                  s -> s.dataset = 'OpenStreetMap'
              )) > 0
        ) TO '{out_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
        try:
            con.execute(query)
            count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
            logger.info(f"Overture-OSM bridge written: {count:,} rows → {out_file}")
        except Exception as e:
            logger.warning(f"Bridge download failed: {e}")
            raise
        finally:
            con.close()
        return out_file

    mo.md("### Task: Overture-OSM bridge — defined")
    return (ingest_overture_osm_bridge,)


@app.cell
def _task_fema(
    CONFIG_PATH,
    Path,
    StorageConfig,
    aoi_bbox,
    aoi_polygon_wkt,
    get_connection,
    get_run_logger,
    mo,
    requests,
    task,
):
    """FEMA USA Structures download task.

    Downloads per-state zipped File Geodatabases from the FEMA GDB download
    site, extracts the .gdb, and clips to the AOI polygon.  The index page is
    scraped at runtime so the correct (dated) S3 URLs are always used.
    """

    @task(
        name="ingest-fema-structures",
        retries=2,
        retry_delay_seconds=120,
    )
    def ingest_fema_structures(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        import re
        import tempfile
        import zipfile

        import geopandas as gpd
        import pandas as pd
        from shapely import wkt as shapely_wkt

        logger = get_run_logger()

        out_dir = Path(storage.bronze_path("fema_structures")) / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "structures.parquet"

        xmin, ymin, xmax, ymax = aoi_bbox(aoi, CONFIG_PATH)
        aoi_polygon = shapely_wkt.loads(aoi_polygon_wkt(cfg["aoi"][aoi]["geojson"]))

        # ── Scrape the FEMA GDB index for current per-state S3 URLs ──────────
        # URL format: .../USA_Structures/{StateName}/Deliverable{YYYYMMDD}{ST}.zip
        # Dates change with each state update, so we parse the page at runtime.
        index_url = cfg["fema"]["download_index_url"]
        logger.info(f"Fetching FEMA download index from {index_url} ...")
        index_resp = requests.get(index_url, timeout=60)
        index_resp.raise_for_status()
        state_urls = {}
        for href in re.findall(
            r'href="(https://fema-femadata\.s3\.amazonaws\.com/[^"]+\.zip)"',
            index_resp.text,
        ):
            m = re.search(r"Deliverable\d+([A-Z]{2})\.zip$", href)
            if m:
                state_urls[m.group(1)] = href
        logger.info(f"Found {len(state_urls)} state/territory GDB downloads on index page")

        # ── Find which states/territories the AOI overlaps ───────────────────
        tiger_url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services"
            "/TIGERweb/State_County/MapServer/0/query"
        )
        tiger_resp = requests.get(
            tiger_url,
            params={
                "geometry": f"{xmin},{ymin},{xmax},{ymax}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "STUSAB",
                "f": "json",
            },
            timeout=60,
        )
        tiger_resp.raise_for_status()
        abbrev_list = [
            feat["attributes"]["STUSAB"]
            for feat in tiger_resp.json().get("features", [])
        ]
        if not abbrev_list:
            logger.warning(f"No state codes found for AOI {aoi} — no FEMA data written")
            return out_file

        logger.info(f"FEMA: downloading GDB data for states: {abbrev_list}")

        # ── Download, extract, and clip each state's GDB ─────────────────────
        all_gdfs = []
        for abbrev in abbrev_list:
            url = state_urls.get(abbrev)
            if not url:
                logger.warning(f"  No FEMA GDB URL found for state {abbrev} — skipping")
                continue

            logger.info(f"  Fetching {abbrev}: {url} ...")
            try:
                resp = requests.get(url, timeout=600, stream=True)
                if resp.status_code == 404:
                    logger.warning(f"  FEMA GDB not found for {abbrev} (HTTP 404) — skipping")
                    resp.close()
                    continue
                resp.raise_for_status()

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    zip_path = tmpdir_path / f"fema_{abbrev}.zip"

                    with open(zip_path, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            fh.write(chunk)
                    resp.close()

                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(tmpdir_path)

                    gdb_dirs = list(tmpdir_path.glob("**/*.gdb"))
                    if not gdb_dirs:
                        logger.warning(f"  No .gdb found in zip for {abbrev} — skipping")
                        continue
                    gdb_path = gdb_dirs[0]

                    # Coarse bbox filter at read time, then precise polygon clip
                    gdf_state = gpd.read_file(
                        str(gdb_path), layer=0, bbox=(xmin, ymin, xmax, ymax)
                    )
                    if gdf_state.crs and gdf_state.crs.to_epsg() != 4326:
                        gdf_state = gdf_state.to_crs("EPSG:4326")
                    if not gdf_state.empty:
                        gdf_state = gdf_state[gdf_state.geometry.intersects(aoi_polygon)]
                    if not gdf_state.empty:
                        logger.info(f"  {abbrev}: {len(gdf_state):,} structures in AOI")
                        all_gdfs.append(gdf_state)
                    else:
                        logger.info(f"  {abbrev}: no structures intersect AOI")

            except Exception as e:
                logger.warning(f"  Failed to process FEMA GDB for {abbrev}: {e} — skipping")
                continue

        if not all_gdfs:
            logger.warning(f"No FEMA structures found for AOI {aoi}")
            return out_file

        gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
        if "OBJECTID" in gdf.columns:
            gdf = gdf.rename(columns={"OBJECTID": "fema_id"})
        elif "fema_id" not in gdf.columns:
            # ESRI GDB FIDs are not exposed as regular columns by fiona/geopandas;
            # fall back to a sequential surrogate ID.
            gdf.insert(0, "fema_id", range(len(gdf)))

        # Ensure final output is EPSG:4326 (per-state reprojection already done above,
        # but guard here in case any concat edge case slips through).
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Write via DuckDB COPY TO: ST_MakeValid + Hilbert sort + ZSTD + row-group size.
        # Stage an intermediate GeoParquet so DuckDB can read geometry via GeoParquet
        # metadata (avoids manual WKB conversion).
        tmp_parquet = out_file.with_name("_tmp_" + out_file.name)
        try:
            gdf.to_parquet(tmp_parquet, index=False)
            con = get_connection()
            con.execute(f"""
                COPY (
                    SELECT * REPLACE (ST_MakeValid(geometry) AS geometry)
                    FROM read_parquet('{tmp_parquet}')
                    ORDER BY ST_Hilbert(geometry,
                                        ST_Extent(ST_MakeEnvelope(-180, -90, 180, 90)))
                ) TO '{out_file}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """)
            con.close()
        finally:
            tmp_parquet.unlink(missing_ok=True)

        count = len(gdf)
        logger.info(f"FEMA structures written: {count:,} rows → {out_file}")
        return out_file

    mo.md("### Task: FEMA structures — defined")
    return (ingest_fema_structures,)


@app.cell
def _task_nsi(
    CONFIG_PATH,
    Path,
    StorageConfig,
    aoi_bbox,
    aoi_polygon_wkt,
    get_connection,
    get_run_logger,
    mo,
    requests,
    task,
):
    """NSI (National Structure Inventory) download task.

    Downloads per-state GeoPackage zips from the NSI bulk download server,
    extracts the GeoPackage, and filters to the AOI polygon.  States/territories
    with no file on the server (HTTP 404) are skipped gracefully.
    """

    @task(
        name="ingest-nsi-structures",
        retries=2,
        retry_delay_seconds=120,
    )
    def ingest_nsi_structures(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        import tempfile
        import zipfile

        import geopandas as gpd
        import pandas as pd
        from shapely import wkt as shapely_wkt

        logger = get_run_logger()

        out_dir = Path(storage.bronze_path("nsi_structures")) / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "structures.parquet"

        xmin, ymin, xmax, ymax = aoi_bbox(aoi, CONFIG_PATH)
        aoi_polygon = shapely_wkt.loads(aoi_polygon_wkt(cfg["aoi"][aoi]["geojson"]))

        # ── Find which US states/territories the AOI overlaps ─────────────────
        tiger_url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services"
            "/TIGERweb/State_County/MapServer/0/query"
        )
        tiger_resp = requests.get(
            tiger_url,
            params={
                "geometry": f"{xmin},{ymin},{xmax},{ymax}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "STATE",
                "f": "json",
            },
            timeout=60,
        )
        tiger_resp.raise_for_status()
        fips_list = [
            feat["attributes"]["STATE"]
            for feat in tiger_resp.json().get("features", [])
        ]
        if not fips_list:
            logger.warning(f"No state FIPS codes found for AOI {aoi} — no NSI data written")
            return out_file

        logger.info(f"NSI: downloading data for state FIPS: {fips_list}")
        download_base = cfg["nsi"]["download_base_url"]

        # ── Download, extract, and filter each state's GeoPackage ─────────────
        all_gdfs = []
        for fips in fips_list:
            url = f"{download_base}/nsi_2022_{fips}.gpkg.zip"
            logger.info(f"  Fetching {url} ...")
            try:
                resp = requests.get(url, timeout=600, stream=True)
                if resp.status_code == 404:
                    logger.warning(f"  NSI data not found for FIPS {fips} (HTTP 404) — skipping")
                    resp.close()
                    continue
                resp.raise_for_status()

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    zip_path = tmpdir_path / f"nsi_{fips}.gpkg.zip"

                    with open(zip_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                    resp.close()

                    with zipfile.ZipFile(zip_path) as zf:
                        gpkg_entries = [n for n in zf.namelist() if n.endswith(".gpkg")]
                        if not gpkg_entries:
                            logger.warning(f"  No .gpkg in zip for FIPS {fips} — skipping")
                            continue
                        zf.extract(gpkg_entries[0], tmpdir_path)
                        gpkg_path = tmpdir_path / gpkg_entries[0]

                    # Coarse bbox filter at read time, then precise polygon filter
                    gdf_state = gpd.read_file(str(gpkg_path), bbox=(xmin, ymin, xmax, ymax))
                    if gdf_state.crs and gdf_state.crs.to_epsg() != 4326:
                        gdf_state = gdf_state.to_crs("EPSG:4326")
                    if not gdf_state.empty:
                        gdf_state = gdf_state[gdf_state.geometry.intersects(aoi_polygon)]
                    if not gdf_state.empty:
                        logger.info(f"  FIPS {fips}: {len(gdf_state):,} structures in AOI")
                        all_gdfs.append(gdf_state)
                    else:
                        logger.info(f"  FIPS {fips}: no structures intersect AOI")

            except Exception as e:
                logger.warning(f"  Failed to process NSI data for FIPS {fips}: {e} — skipping")
                continue

        if not all_gdfs:
            logger.warning(f"No NSI structures found for AOI {aoi}")
            return out_file

        gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
        if "fd_id" in gdf.columns:
            gdf = gdf.rename(columns={"fd_id": "nsi_id"})
        elif "nsi_id" not in gdf.columns:
            # Fallback surrogate ID if fd_id is absent from the GeoPackage.
            gdf.insert(0, "nsi_id", range(len(gdf)))

        # Ensure final output is EPSG:4326 (per-state reprojection already done above,
        # but guard here in case any concat edge case slips through).
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Write via DuckDB COPY TO: ST_MakeValid + Hilbert sort + ZSTD + row-group size.
        tmp_parquet = out_file.with_name("_tmp_" + out_file.name)
        try:
            gdf.to_parquet(tmp_parquet, index=False)
            con = get_connection()
            con.execute(f"""
                COPY (
                    SELECT * REPLACE (ST_MakeValid(geometry) AS geometry)
                    FROM read_parquet('{tmp_parquet}')
                    ORDER BY ST_Hilbert(geometry,
                                        ST_Extent(ST_MakeEnvelope(-180, -90, 180, 90)))
                ) TO '{out_file}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """)
            con.close()
        finally:
            tmp_parquet.unlink(missing_ok=True)

        count = len(gdf)
        logger.info(f"NSI structures written: {count:,} rows → {out_file}")
        return out_file

    mo.md("### Task: NSI structures — defined")
    return (ingest_nsi_structures,)


@app.cell
def _flow_definition(
    CFG,
    STORAGE,
    flow,
    ingest_fema_structures,
    ingest_nsi_structures,
    ingest_overture_buildings,
    ingest_overture_osm_bridge,
):
    @flow(name="gers-ingest-sources", log_prints=True)
    def ingest_sources_flow(aoi: str = "saipan"):
        overture_file = ingest_overture_buildings(aoi, CFG, STORAGE)
        bridge_file = ingest_overture_osm_bridge(aoi, CFG, STORAGE)
        fema_file = ingest_fema_structures(aoi, CFG, STORAGE)
        nsi_file = ingest_nsi_structures(aoi, CFG, STORAGE)
        return {
            "overture": str(overture_file),
            "bridge": str(bridge_file),
            "fema": str(fema_file),
            "nsi": str(nsi_file),
        }

    return (ingest_sources_flow,)


@app.cell
def _notebook_run(
    CFG,
    STORAGE,
    aoi_name,
    ingest_fema_structures,
    ingest_nsi_structures,
    ingest_overture_buildings,
    ingest_overture_osm_bridge,
    mo,
    run_btn,
):
    if mo.running_in_notebook():
        mo.stop(
            not run_btn.value,
            mo.md("☝️ Click **▶ Run ingestion** above to start."),
        )
        aoi = aoi_name
        overture_file = ingest_overture_buildings(aoi, CFG, STORAGE)
        bridge_file = ingest_overture_osm_bridge(aoi, CFG, STORAGE)
        fema_file = ingest_fema_structures(aoi, CFG, STORAGE)
        nsi_file = ingest_nsi_structures(aoi, CFG, STORAGE)
        print({
            "overture": str(overture_file),
            "bridge": str(bridge_file),
            "fema": str(fema_file),
            "nsi": str(nsi_file),
        })
    return


@app.cell
def _run(aoi_name, ingest_sources_flow, mo):
    """Run the flow when executed as a script (not in notebook mode)."""
    if not mo.running_in_notebook():
        result = ingest_sources_flow(aoi=aoi_name)
        print(result)
    return


if __name__ == "__main__":
    app.run()
