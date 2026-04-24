#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Load Source Cooperative temporary AWS credentials from a local .env.json file.

Source Coop credentials are short-lived (issued via the dashboard and scoped to
a single repository prefix). The file format is the JSON payload shown in the
Source Coop UI — four keys:

    {
      "aws_access_key_id": "ASIA...",
      "aws_secret_access_key": "...",
      "aws_session_token": "...",
      "region_name": "us-west-2"
    }

If the file has not been touched recently we warn the caller (but do not fail);
STS tokens typically last an hour or so.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REQUIRED_KEYS = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "region_name",
)

REFRESH_HINT = (
    "Regenerate temporary credentials at "
    "https://source.coop/repositories/henryspatialanalysis/openpois/manage "
    "and write them to .env.json at the repo root."
)

STALE_SECONDS = 60 * 60  # Source Coop tokens usually last ~1 hour.


def load_source_coop_credentials(env_file: Path | str | None = None) -> dict:
    """Read Source Coop temporary AWS credentials from ``.env.json``.

    ``env_file`` defaults to ``~/repos/openpois/.env.json``.
    Raises ``FileNotFoundError`` or ``ValueError`` with a refresh hint if the
    file is missing or malformed. Prints a warning if the file's mtime is
    older than ~1 hour (tokens may have expired).
    """
    if env_file is None:
        env_file = Path.home() / "repos" / "openpois" / ".env.json"
    env_file = Path(env_file).expanduser()

    if not env_file.exists():
        raise FileNotFoundError(
            f"Source Coop credentials file not found at {env_file}. "
            f"{REFRESH_HINT}"
        )

    with env_file.open() as f:
        creds = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in creds]
    if missing:
        raise ValueError(
            f"Source Coop credentials file {env_file} is missing keys: "
            f"{missing}. {REFRESH_HINT}"
        )

    mtime_age = time.time() - env_file.stat().st_mtime
    if mtime_age > STALE_SECONDS:
        minutes = int(mtime_age // 60)
        print(
            f"⚠️  {env_file} was last updated ~{minutes} minutes ago. "
            "Source Coop tokens usually expire within an hour — if uploads "
            f"fail with ExpiredToken, {REFRESH_HINT}"
        )

    return {k: creds[k] for k in REQUIRED_KEYS}
