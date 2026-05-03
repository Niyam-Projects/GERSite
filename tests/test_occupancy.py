"""Tests for lib/occupancy.py — occupancy conflation logic."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
LIB_PATH = REPO_ROOT / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))

from occupancy import (
    FEMA_OCCUPANCY_CLASSES,
    NSI_TO_FEMA,
    OCCUPANCY_CONF_AGREE,
    OCCUPANCY_CONF_DISAGREE,
    OCCUPANCY_CONF_FEMA,
    OCCUPANCY_CONF_NSI,
    conflate_occupancy,
    nsi_occtype_to_fema,
)


# ---------------------------------------------------------------------------
# nsi_occtype_to_fema
# ---------------------------------------------------------------------------

class TestNsiOcctypeToFema:
    def test_residential_codes(self):
        for code in ["RES1", "RES2", "RES3A", "RES3B", "RES3C", "RES3D", "RES3E", "RES3F", "RES4", "RES5", "RES6"]:
            assert nsi_occtype_to_fema(code) == "Residential", f"Expected Residential for {code}"

    def test_commercial_codes(self):
        for code in ["COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10"]:
            assert nsi_occtype_to_fema(code) == "Commercial", f"Expected Commercial for {code}"

    def test_industrial_codes(self):
        for code in ["IND1", "IND2", "IND3", "IND4", "IND5", "IND6"]:
            assert nsi_occtype_to_fema(code) == "Industrial", f"Expected Industrial for {code}"

    def test_single_category_codes(self):
        assert nsi_occtype_to_fema("AGR1") == "Agriculture"
        assert nsi_occtype_to_fema("REL1") == "Religion"
        assert nsi_occtype_to_fema("GOV1") == "Government"
        assert nsi_occtype_to_fema("GOV2") == "Government"
        assert nsi_occtype_to_fema("EDU1") == "Education"
        assert nsi_occtype_to_fema("EDU2") == "Education"

    def test_case_insensitive(self):
        assert nsi_occtype_to_fema("res1") == "Residential"
        assert nsi_occtype_to_fema("Com4") == "Commercial"

    def test_none_input(self):
        assert nsi_occtype_to_fema(None) is None

    def test_empty_string(self):
        assert nsi_occtype_to_fema("") is None

    def test_unknown_code(self):
        assert nsi_occtype_to_fema("UNKNOWN") is None
        assert nsi_occtype_to_fema("XYZ99") is None


# ---------------------------------------------------------------------------
# conflate_occupancy — four agreement scenarios
# ---------------------------------------------------------------------------

class TestConflateOccupancyScenarios:
    def test_both_agree_returns_high_confidence(self):
        occ, conf = conflate_occupancy("Residential", "RES1")
        assert occ == "Residential"
        assert conf == OCCUPANCY_CONF_AGREE  # 1.0

    def test_both_agree_commercial(self):
        occ, conf = conflate_occupancy("Commercial", "COM3")
        assert occ == "Commercial"
        assert conf == OCCUPANCY_CONF_AGREE

    def test_fema_only_medium_high_confidence(self):
        occ, conf = conflate_occupancy("Industrial", None)
        assert occ == "Industrial"
        assert conf == OCCUPANCY_CONF_FEMA  # 0.8

    def test_nsi_only_medium_confidence(self):
        occ, conf = conflate_occupancy(None, "COM1")
        assert occ == "Commercial"
        assert conf == OCCUPANCY_CONF_NSI  # 0.6

    def test_both_disagree_fema_wins_low_confidence(self):
        occ, conf = conflate_occupancy("Commercial", "RES1")
        assert occ == "Commercial"
        assert conf == OCCUPANCY_CONF_DISAGREE  # 0.3

    def test_both_none_returns_none(self):
        occ, conf = conflate_occupancy(None, None)
        assert occ is None
        assert conf is None

    def test_nsi_unknown_code_treated_as_no_nsi(self):
        # NSI code that cannot be mapped: behave as if NSI is absent
        occ, conf = conflate_occupancy("Government", "UNKNOWN")
        assert occ == "Government"
        assert conf == OCCUPANCY_CONF_FEMA  # 0.8 — only FEMA

    def test_nsi_unknown_and_no_fema_returns_none(self):
        occ, conf = conflate_occupancy(None, "UNKNOWN")
        assert occ is None
        assert conf is None

    def test_education_agree(self):
        occ, conf = conflate_occupancy("Education", "EDU2")
        assert occ == "Education"
        assert conf == OCCUPANCY_CONF_AGREE


# ---------------------------------------------------------------------------
# FEMA_OCCUPANCY_CLASSES completeness
# ---------------------------------------------------------------------------

class TestFemaOccupancyClasses:
    def test_contains_seven_canonical_classes(self):
        assert len(FEMA_OCCUPANCY_CLASSES) == 7

    def test_all_nsi_mappings_resolve_to_canonical_class(self):
        for nsi_code, fema_class in NSI_TO_FEMA.items():
            assert fema_class in FEMA_OCCUPANCY_CLASSES, (
                f"NSI code {nsi_code} maps to '{fema_class}' which is not in "
                f"FEMA_OCCUPANCY_CLASSES: {FEMA_OCCUPANCY_CLASSES}"
            )
