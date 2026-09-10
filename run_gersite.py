"""
GERSite container entrypoint — runs the building-conflation pipeline for one AOI.

Usage:
  python run_gersite.py --aoi <name>

Sequence (per the Prefect orchestration flow's documented architecture):
  ingest_sources.py --aoi <aoi>
  generate_bridges.py --aoi <aoi>
  produce_gold_layer.py --aoi <aoi>
  aws s3 sync /data/gers/gold/buildings/<aoi>/ s3://$S3_BUCKET/external/gersite/gold/buildings/<aoi>/
"""
import argparse
import os
import subprocess
import sys

STEPS = ["flows/ingest_sources.py", "flows/generate_bridges.py", "flows/produce_gold_layer.py"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GERSite pipeline for one AOI")
    parser.add_argument("--aoi", required=True, help="AOI name, e.g. florida, puerto_rico, miami_dade")
    args = parser.parse_args()

    for step in STEPS:
        print(f"=== Running {step} --aoi {args.aoi} ===", flush=True)
        result = subprocess.run([sys.executable, step, "--aoi", args.aoi])
        if result.returncode != 0:
            print(f"Step {step} failed with exit code {result.returncode}", flush=True)
            return result.returncode

    s3_bucket = os.environ.get("S3_BUCKET", "geocube-files-prod")
    local_path = f"/data/gers/gold/buildings/{args.aoi}/"
    s3_path = f"s3://{s3_bucket}/external/gersite/gold/buildings/{args.aoi}/"
    print(f"=== Syncing {local_path} -> {s3_path} ===", flush=True)
    sync_result = subprocess.run(["aws", "s3", "sync", local_path, s3_path, "--region", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")])
    return sync_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
