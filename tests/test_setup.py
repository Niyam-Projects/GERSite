#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Tests for ``openpois.models.setup.prepare_data_for_model``.

Covers the two data-prep invariants required by turnover-model-methodology.md:

* Per-row Δ is the *inter-observation* interval (t_k − t_{k−1}), not the
  duration since the individual's start (§1.2).
* ``is_first_interval`` marks the first surviving observation of each
  ``(POI, name-iteration)`` individual, which is where the ZIE δ term enters
  the likelihood (§1.7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from openpois.models.setup import prepare_data_for_model


def _hand_crafted_observations() -> pd.DataFrame:
    """Two individuals with multiple observation intervals each.

    Individual A (id=1, name-iteration tagged at v1): three observations at
    v1, v2, v3; no name change → two inter-observation rows (v2 and v3).
    Individual B (id=2, name-iteration tagged at u1): two observations at
    u1, u2; no name change → one inter-observation row (u2).
    """
    v1 = pd.Timestamp("2024-01-01")
    v2 = pd.Timestamp("2024-03-01")
    v3 = pd.Timestamp("2024-08-01")
    u1 = pd.Timestamp("2023-06-01")
    u2 = pd.Timestamp("2024-06-01")
    return pd.DataFrame({
        "id":                  [1,  1,  2],
        "obs_timestamp":       [v2, v3, u2],
        "last_obs_timestamp":  [v1, v2, u1],
        "last_tag_timestamp":  [v1, v1, u1],
        "changed":             [0,  0,  0],
    })


def test_delta_is_inter_observation_interval():
    """``tag_years`` is t_k − t_{k−1}, not t_k − t_tag (methodology §1.2)."""
    df = _hand_crafted_observations()
    prepared = prepare_data_for_model(df).sort_values("obs_timestamp")

    # Individual A, observation v2: Δ = v2 − v1.
    row_v2 = prepared.iloc[0]
    expected_v2 = (
        pd.Timestamp("2024-03-01") - pd.Timestamp("2024-01-01")
    ).days
    assert row_v2["tag_days"] == expected_v2

    # Individual A, observation v3: Δ = v3 − v2 (NOT v3 − v1).
    row_v3 = prepared.iloc[2]
    expected_v3 = (
        pd.Timestamp("2024-08-01") - pd.Timestamp("2024-03-01")
    ).days
    assert row_v3["tag_days"] == expected_v3
    wrong_cumulative = (
        pd.Timestamp("2024-08-01") - pd.Timestamp("2024-01-01")
    ).days
    assert row_v3["tag_days"] != wrong_cumulative


def test_is_first_interval_flag():
    """Flag is True iff last_obs_timestamp == last_tag_timestamp, once per id."""
    df = _hand_crafted_observations()
    prepared = prepare_data_for_model(df)

    # Exactly two rows flagged as first (one per individual).
    assert int(prepared["is_first_interval"].sum()) == 2

    # Per-individual: exactly one first-interval row each.
    per_id = prepared.groupby("id")["is_first_interval"].sum()
    np.testing.assert_array_equal(per_id.to_numpy(), np.array([1, 1]))

    # The flag lines up exactly with the equality invariant.
    manual = (
        prepared["last_obs_timestamp"] == prepared["last_tag_timestamp"]
    )
    np.testing.assert_array_equal(
        prepared["is_first_interval"].to_numpy(),
        manual.to_numpy(),
    )
