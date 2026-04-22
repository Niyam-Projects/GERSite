"""
Build conflated.pmtiles from the conflated POI dataset.

Output is a single-zoom PMTiles archive (z14) keyed by the config's
``upload.pmtiles`` block. OpenLayers over-zooms z15-20 natively, and the site
never renders below z14, so tiling extra zoom levels would just waste disk and
wall time.

Intermediate FlatGeobuf is staged next to the output and deleted on success.
"""
from config_versioned import Config

from openpois.io.pmtiles import build_pmtiles

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("conflation", "conflated")
OUTPUT_PATH = config.get_file_path("conflation", "pmtiles")

LAYER_NAME = config.get("upload", "pmtiles", "conflated_layer_name")
PROPERTIES = config.get("upload", "pmtiles", "conflated_properties")
MIN_ZOOM = config.get("upload", "pmtiles", "min_zoom")
MAX_ZOOM = config.get("upload", "pmtiles", "max_zoom")
DROP_STRATEGY = config.get("upload", "pmtiles", "drop_strategy")

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Building conflated PMTiles from {INPUT_PATH}")
    print(f"  layer: {LAYER_NAME}")
    print(f"  zooms: Z{MIN_ZOOM}-z{MAX_ZOOM}")
    print(f"  drop:  --{DROP_STRATEGY}")
    print(f"  props: {', '.join(PROPERTIES)}")
    print(f"  -> {OUTPUT_PATH}")

    stats = build_pmtiles(
        input_parquet = INPUT_PATH,
        output_pmtiles = OUTPUT_PATH,
        layer_name = LAYER_NAME,
        properties = PROPERTIES,
        min_zoom = MIN_ZOOM,
        max_zoom = MAX_ZOOM,
        drop_strategy = DROP_STRATEGY,
    )

    print(
        f"Done. Wrote {stats['rows_written']:,} features, "
        f"{stats['pmtiles_bytes'] / 1e9:.2f} GB PMTiles."
    )
