"""Guard against frontend taxonomy drift.

Fails when `site/src/taxonomy.generated.js` is stale relative to the
conflation data CSVs.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_taxonomy_sync.py"


def _load_check():
    spec = importlib.util.spec_from_file_location("check_taxonomy_sync", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_taxonomy_sync"] = module
    spec.loader.exec_module(module)
    return module


def test_taxonomy_sync_is_clean(capsys):
    """scripts/check_taxonomy_sync.py must exit 0 (no drift)."""
    mod = _load_check()
    if not mod.GENERATED_JS.exists():
        pytest.skip(
            f"{mod.GENERATED_JS.relative_to(REPO_ROOT)} not present; "
            f"run `python scripts/build_taxonomy.py` before running this test."
        )
    exit_code = mod.check()
    captured = capsys.readouterr()
    assert exit_code == 0, (
        "Frontend taxonomy is out of sync with conflation CSVs. "
        "Rerun `python scripts/build_taxonomy.py`.\n"
        f"stderr:\n{captured.err}"
    )
