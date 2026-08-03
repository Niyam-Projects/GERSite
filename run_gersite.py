"""
GERSite pipeline runner for Fargate containers.

Patches config.gers.yaml for the container environment (storage root,
geojson paths, memory limit), then runs the three production flows:
  1. ingest_sources    — download Overture, FEMA, NSI to bronze
  2. generate_bridges  — compute FEMA + NSI IoU bridge files (silver)
  3. produce_gold_layer — merge sources → gold GeoParquet

Skips generate_tiles (PMTiles output not needed for the asset-points pipeline).

Usage:
  python run_gersite.py --aoi florida
  python run_gersite.py --aoi puerto_rico
  python run_gersite.py --aoi us_virgin_islands
"""
from __future__ import annotations

import argparse
import os
import subprocess


def run_flow(script: str, aoi: str) -> None:
    venv_python = "/app/.venv/bin/python"
    result = subprocess.run(
        [venv_python, f"/app/flows/{script}", "--aoi", aoi],
        check=True,
    )
    return result


AOI_TO_STATE_CODE = {
    "florida":          "FL",
    "puerto_rico":      "PR",
    "us_virgin_islands": "VI",
    "guam":             "GU",
    "saipan":           "MP",
    "miami_dade":       "FL",
}


def upload_gold(aoi: str, bucket: str, region: str) -> None:
    state_code = AOI_TO_STATE_CODE.get(aoi, aoi.upper())
    subprocess.run(
        [
            "aws", "s3", "sync",
            f"/data/gers/gold/buildings/{aoi}/",
            f"s3://{bucket}/external/gersite/gold/buildings/state_code={state_code}/",
            "--region", region,
            "--only-show-errors",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GERSite building conflation pipeline (Fargate entry point)"
    )
    parser.add_argument(
        "--aoi",
        required=True,
        choices=["florida", "puerto_rico", "us_virgin_islands", "guam", "saipan", "miami_dade"],
        help="AOI name matching a key in config.gers.yaml",
    )
    args = parser.parse_args()
    aoi = args.aoi

    bucket = os.environ.get("S3_BUCKET", "geocube-files-prod")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    os.makedirs("/data/gers", exist_ok=True)

    print(f"=== GERSite pipeline: aoi={aoi} ===", flush=True)
    run_flow("ingest_sources.py", aoi)

    print("=== Generate bridges ===", flush=True)
    run_flow("generate_bridges.py", aoi)

    print("=== Produce gold layer ===", flush=True)
    run_flow("produce_gold_layer.py", aoi)

    print("=== Upload gold to S3 ===", flush=True)
    upload_gold(aoi, bucket, region)

    print(f"=== Done: {aoi} ===", flush=True)


if __name__ == "__main__":
    main()
