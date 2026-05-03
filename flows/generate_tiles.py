"""
Flow 4: Generate Tiles — GERSite Building Conflation Pipeline.

Converts Gold layer and Bronze source GeoParquet files into PMTiles archives
for use by the site viewer.

Uses **freestiler** (pip install freestiler) — pure-Python, Rust-backed tiler.
No external binaries required; works on Windows, macOS, and Linux.

Tasks produced:
  1. gold_buildings   — Gold layer GeoParquet → PMTiles (min_zoom 0, max_zoom 14)
  2. fema_buildings   — FEMA bronze GeoParquet → PMTiles (min_zoom 0, max_zoom 14)
  3. nsi_unmatched    — NSI unmatched points GeoParquet → PMTiles (min_zoom 0, max_zoom 16)

Output paths are configured under ``storage.tiles`` in config.gers.yaml.

This file is a Marimo notebook.
  Script mode:   python flows/generate_tiles.py --aoi saipan
  Notebook mode: marimo edit flows/generate_tiles.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _setup():
    """Imports, config, and lib loading."""
    import sys
    from pathlib import Path

    import marimo as mo
    import yaml
    from freestiler import freestile_query
    from prefect import flow, task, get_run_logger

    REPO_ROOT = Path(__file__).parent.parent
    CONFIG_PATH = REPO_ROOT / "config.gers.yaml"
    LIB_PATH = REPO_ROOT / "lib"
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))

    from duckdb_helpers import StorageConfig

    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)

    STORAGE = StorageConfig.from_config(CONFIG_PATH)
    AOI_CHOICES = list(CFG["aoi"].keys())

    mo.md("## Flow 4: Generate Tiles")
    return (
        AOI_CHOICES,
        CFG,
        Path,
        STORAGE,
        StorageConfig,
        flow,
        freestile_query,
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
    aoi_label = CFG["aoi"][aoi_name]["label"]
    mo.md(f"**Active AOI:** {aoi_label} (`{aoi_name}`)")
    return (aoi_name,)


@app.cell
def _run_controls(aoi_name, mo):
    """Run button — click to trigger tile generation in notebook mode."""
    run_btn = mo.ui.run_button(label="▶  Generate tiles")
    mo.hstack([mo.md(f"**AOI:** `{aoi_name}`"), run_btn], justify="start")
    return (run_btn,)


@app.cell
def _task_gold_tiles(
    Path,
    StorageConfig,
    freestile_query,
    get_run_logger,
    mo,
    task,
):
    """Task: Gold buildings GeoParquet → PMTiles."""

    @task(name="generate-gold-tiles", retries=1)
    def generate_gold_tiles(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()

        gold_file = Path(storage.gold_path("buildings", aoi=aoi)) / "buildings.parquet"
        if not gold_file.exists():
            raise FileNotFoundError(
                f"Gold buildings not found: {gold_file}. Run Flow 3 first."
            )

        out_path = Path(storage.tiles_path("gold_buildings", aoi=aoi)) / "buildings.pmtiles"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Tiling gold buildings from {gold_file} → {out_path}")
        freestile_query(
            query=f"SELECT * FROM read_parquet('{gold_file.as_posix()}')",
            output=str(out_path),
            layer_name="buildings",
            min_zoom=0,
            max_zoom=14,
            tile_format="mvt",
            coalesce=True,
            overwrite=True,
        )
        logger.info(f"Gold PMTiles written → {out_path}")
        return out_path

    mo.md("### Task: Gold buildings tiles — defined")
    return (generate_gold_tiles,)


@app.cell
def _task_fema_tiles(
    Path,
    StorageConfig,
    freestile_query,
    get_run_logger,
    mo,
    task,
):
    """Task: FEMA bronze GeoParquet → PMTiles."""

    @task(name="generate-fema-tiles", retries=1)
    def generate_fema_tiles(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()

        fema_path = Path(storage.bronze_path("fema_structures")) / aoi / "structures.parquet"
        if not fema_path.exists():
            raise FileNotFoundError(
                f"FEMA bronze structures not found: {fema_path}. Run Flow 1 first."
            )

        out_path = Path(storage.tiles_path("fema_buildings", aoi=aoi)) / "fema.pmtiles"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Tiling FEMA structures {fema_path} → {out_path}")
        freestile_query(
            query=f"SELECT * FROM read_parquet('{fema_path.as_posix()}')",
            output=str(out_path),
            layer_name="fema",
            min_zoom=0,
            max_zoom=14,
            tile_format="mvt",
            coalesce=True,
            overwrite=True,
        )
        logger.info(f"FEMA PMTiles written → {out_path}")
        return out_path

    mo.md("### Task: FEMA bronze tiles — defined")
    return (generate_fema_tiles,)


@app.cell
def _task_nsi_unmatched_tiles(
    Path,
    StorageConfig,
    freestile_query,
    get_run_logger,
    mo,
    task,
):
    """Task: NSI unmatched points GeoParquet → PMTiles."""

    @task(name="generate-nsi-unmatched-tiles", retries=1)
    def generate_nsi_unmatched_tiles(aoi: str, cfg: dict, storage: StorageConfig) -> Path:
        logger = get_run_logger()

        nsi_path = (
            Path(storage.gold_path("nsi_review", aoi=aoi)) / "nsi_unmatched.parquet"
        )
        if not nsi_path.exists():
            logger.warning(
                f"NSI unmatched not found: {nsi_path}. "
                "Run Flow 3 first (requires NSI bridge data)."
            )
            return nsi_path  # Return path even if absent — task succeeds gracefully

        out_path = Path(storage.tiles_path("nsi_unmatched", aoi=aoi)) / "nsi_unmatched.pmtiles"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Points — higher max zoom for precise placement
        logger.info(f"Tiling NSI unmatched points {nsi_path} → {out_path}")
        freestile_query(
            query=f"SELECT * FROM read_parquet('{nsi_path.as_posix()}')",
            output=str(out_path),
            layer_name="nsi_unmatched",
            min_zoom=0,
            max_zoom=16,
            tile_format="mvt",
            overwrite=True,
        )
        logger.info(f"NSI unmatched PMTiles written → {out_path}")
        return out_path

    mo.md("### Task: NSI unmatched tiles — defined")
    return (generate_nsi_unmatched_tiles,)


@app.cell
def _flow_definition(
    CFG,
    STORAGE,
    flow,
    generate_fema_tiles,
    generate_gold_tiles,
    generate_nsi_unmatched_tiles,
    mo,
):
    """Define the Prefect tile-generation flow."""

    @flow(name="gers-generate-tiles", log_prints=True)
    def generate_tiles_flow(aoi: str = "saipan"):
        gold_out = generate_gold_tiles(aoi, CFG, STORAGE)
        fema_out = generate_fema_tiles(aoi, CFG, STORAGE)
        nsi_out = generate_nsi_unmatched_tiles(aoi, CFG, STORAGE)
        return {
            "gold": str(gold_out),
            "fema": str(fema_out),
            "nsi_unmatched": str(nsi_out),
        }

    mo.md(f"""
    ## Flow: `gers-generate-tiles`

    Uses **freestiler** (pip install freestiler) — no external binaries needed.

    **Tasks:**
    1. `generate-gold-tiles` — Gold buildings GeoParquet → PMTiles
    2. `generate-fema-tiles` — FEMA bronze GeoParquet → PMTiles
    3. `generate-nsi-unmatched-tiles` — NSI unmatched points GeoParquet → PMTiles

    Run:
    ```bash
    python flows/generate_tiles.py --aoi saipan
    ```
    """)
    return (generate_tiles_flow,)


@app.cell
def _notebook_run(
    CFG,
    STORAGE,
    aoi_name,
    generate_fema_tiles,
    generate_gold_tiles,
    generate_nsi_unmatched_tiles,
    mo,
    run_btn,
):
    """Execute tasks directly in notebook mode, gated by run button."""
    if mo.running_in_notebook():
        mo.stop(
            not run_btn.value,
            mo.md("☝️ Click **▶ Generate tiles** above to start."),
        )
        gold_out = generate_gold_tiles(aoi_name, CFG, STORAGE)
        fema_out = generate_fema_tiles(aoi_name, CFG, STORAGE)
        nsi_out = generate_nsi_unmatched_tiles(aoi_name, CFG, STORAGE)
        print({"gold": str(gold_out), "fema": str(fema_out), "nsi_unmatched": str(nsi_out)})
    return


@app.cell
def _run(aoi_name, generate_tiles_flow, mo):
    """Script-mode execution gated on __main__ check."""
    import sys as _sys
    if not mo.running_in_notebook() and __name__ == "__main__":
        generate_tiles_flow(aoi=aoi_name)
    return


if __name__ == "__main__":
    app.run()
