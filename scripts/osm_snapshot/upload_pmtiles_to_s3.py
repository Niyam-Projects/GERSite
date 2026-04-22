"""
Upload osm_snapshot.pmtiles to the public S3 bucket.

Key layout:
    s3://<bucket>/<s3_prefix>/<s3_version>/osm_snapshot.pmtiles

``s3_version`` defaults to ``versions.aws`` from config. Override with
``--s3-version YYYYMMDD`` when uploading to a different dataset version (e.g.,
to match the existing OSM parquet path when aws_version has been bumped past
it).

See the sibling ``upload_to_s3.py`` for AWS setup prerequisites.
"""
import argparse

from config_versioned import Config

from openpois.io.s3 import upload_single_file

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

PMTILES_PATH = config.get_file_path("snapshot_osm", "pmtiles")
S3_BUCKET = config.get("upload", "s3_bucket")
S3_PREFIX = config.get("upload", "s3_prefix_osm")
S3_REGION = config.get("upload", "s3_region")
DEFAULT_VERSION = config.get("versions", "aws")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--s3-version",
        default = DEFAULT_VERSION,
        help = (
            "Version segment in the S3 key path. Defaults to versions.aws in "
            "config.yaml. Pass an explicit date (e.g. 20260417) to land the "
            "PMTiles alongside an existing dataset at a different version."
        ),
    )
    args = parser.parse_args()

    if not PMTILES_PATH.exists():
        raise FileNotFoundError(
            f"{PMTILES_PATH} not found. Run prepare_pmtiles.py first."
        )

    s3_key = f"{S3_PREFIX}/{args.s3_version}/{PMTILES_PATH.name}"
    url = upload_single_file(
        local_path = PMTILES_PATH,
        bucket = S3_BUCKET,
        s3_key = s3_key,
        s3_region = S3_REGION,
        content_type = "application/octet-stream",
    )
    print(f"Uploaded OSM PMTiles: {url}")
