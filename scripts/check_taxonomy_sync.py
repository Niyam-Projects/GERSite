#!/usr/bin/env python3
"""Check that the site's generated taxonomy file matches the conflation CSVs.

Run from the repo root:
    python scripts/check_taxonomy_sync.py

Exit code 0: no drift. Exit code 1: drift detected (diff printed to stderr).

Also warns (without failing) if display-label maps in constants.js are missing
entries for any newly added OSM key or Overture L0.
"""

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "src/openpois/conflation/data"
GENERATED_JS = REPO_ROOT / "site/src/taxonomy.generated.js"
CONSTANTS_JS = REPO_ROOT / "site/src/constants.js"


def canonical_sets():
    """Load the canonical sets from the CSVs."""
    osm = pd.read_csv(DATA_DIR / "taxonomy_crosswalk_openstreetmap.csv")
    overture = pd.read_csv(DATA_DIR / "taxonomy_crosswalk_overture_maps.csv")
    radii = pd.read_csv(DATA_DIR / "match_radii.csv")
    return {
        "SHARED_LABELS": set(radii["shared_label"].tolist()),
        "OSM_KEYS": set(osm["osm_key"].dropna().unique().tolist()),
        "OVERTURE_L0S": set(overture["overture_l0"].dropna().unique().tolist()),
    }


def parse_js_array(text, name):
    """Extract a string-array export from a JS source file."""
    match = re.search(
        rf"export\s+const\s+{re.escape(name)}\s*=\s*\[(.*?)\]",
        text,
        flags = re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not find export `{name}` in JS source")
    return set(re.findall(r"'([^']+)'", match.group(1)))


def parse_js_object_keys(text, name):
    """Extract the keys of a `const <name> = { ... }` object literal."""
    match = re.search(
        rf"const\s+{re.escape(name)}\s*=\s*\{{(.*?)\}}",
        text,
        flags = re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not find `const {name}` in JS source")
    return set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", match.group(1), flags = re.MULTILINE))


def report_diff(name, csv_set, js_set):
    """Print a unified diff for one array. Return True if drift."""
    if csv_set == js_set:
        return False
    missing = sorted(csv_set - js_set)
    extra = sorted(js_set - csv_set)
    print(f"DRIFT in {name}:", file = sys.stderr)
    for item in missing:
        print(f"  + {item!r} (in CSV, missing from JS)", file = sys.stderr)
    for item in extra:
        print(f"  - {item!r} (in JS, missing from CSV)", file = sys.stderr)
    return True


def check() -> int:
    if not GENERATED_JS.exists():
        print(
            f"ERROR: {GENERATED_JS} does not exist. "
            f"Run `python scripts/build_taxonomy.py` first.",
            file = sys.stderr,
        )
        return 1

    canon = canonical_sets()
    gen_text = GENERATED_JS.read_text(encoding = "utf-8")

    drift = False
    for name, csv_set in canon.items():
        js_set = parse_js_array(gen_text, name)
        drift = report_diff(name, csv_set, js_set) or drift

    if drift:
        print(
            "\nRerun `python scripts/build_taxonomy.py` to regenerate "
            f"{GENERATED_JS.relative_to(REPO_ROOT)}.",
            file = sys.stderr,
        )
        return 1

    # Non-fatal: warn if display-label maps miss any key.
    const_text = CONSTANTS_JS.read_text(encoding = "utf-8")
    for map_name, set_name in [
        ("OSM_KEY_LABELS", "OSM_KEYS"),
        ("OVERTURE_L0_LABELS", "OVERTURE_L0S"),
    ]:
        try:
            label_keys = parse_js_object_keys(const_text, map_name)
        except ValueError:
            continue
        missing_labels = sorted(canon[set_name] - label_keys)
        if missing_labels:
            print(
                f"WARN: {map_name} in constants.js is missing display labels "
                f"for: {missing_labels}. Falling back to raw keys.",
                file = sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(check())
