#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for JAX-based OSM turnover models.

Uses small synthetic frames + short NUTS runs so the suite stays fast. The
recovery tolerances are loose on purpose: ``num_draws = 300`` produces noisy
estimates, but we still want to catch outright regressions (wrong math, wrong
sign, broken priors).
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pandas as pd
import pytest

from openpois.models.jax_core import jax_rng
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import (
    MODEL_REGISTRY,
    ConstantModel,
    RandomByTypeModel,
    get_model_class,
)


NUM_DRAWS = 300


def _simulate_frame(
    key,
    n: int,
    true_log_lambda_by_group: np.ndarray,
    group_names: list[str],
) -> pd.DataFrame:
    """Build a DataFrame matching the model likelihood for each group."""
    key_group, key_dt, key_y = jrd.split(key, 3)
    g = np.asarray(
        jrd.randint(key_group, (n,), 0, len(group_names))
    )
    dt = np.asarray(
        jrd.uniform(key_dt, (n,), minval = 0.5, maxval = 5.0)
    )
    lam = np.exp(np.asarray(true_log_lambda_by_group)[g])
    p = 1.0 - np.exp(-lam * dt)
    y = np.asarray(jrd.bernoulli(key_y, jnp.asarray(p))).astype(np.int32)
    return pd.DataFrame({
        "tag_years": dt,
        "changed": y,
        "group_col": [group_names[i] for i in g],
    })


def _run_fitter(model, key) -> ModelFitter:
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_draws = NUM_DRAWS,
        param_likelihood = model.param_likelihood,
        rng_key = key,
    )
    fitter.fit()
    return fitter


def test_constant_model_recovery():
    """ConstantModel posterior should bracket the true log_lambda."""
    true_log_lambda = -1.2
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_frame(
        key_sim,
        n = 2_000,
        true_log_lambda_by_group = np.array([true_log_lambda]),
        group_names = ["only"],
    )
    model = ConstantModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = _run_fitter(model, key_fit)

    row = fitter.get_parameter_table().iloc[0]
    assert row["parameter"] == "log_lambda"
    assert row["lower"] <= true_log_lambda <= row["upper"], (
        f"true log_lambda {true_log_lambda:+.3f} not covered by "
        f"[{row['lower']:+.3f}, {row['upper']:+.3f}]"
    )
    assert abs(row["mean"] - true_log_lambda) < 0.3


def test_random_by_type_recovery():
    """RandomByTypeModel should recover per-group mean lambdas and log_sigma."""
    group_names = ["aaa", "bbb", "ccc"]
    true_log_lambda_0 = -1.0
    true_epsilons = np.array([-0.5, 0.0, 0.6])
    true_log_lambda_by_group = true_log_lambda_0 + true_epsilons
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_frame(
        key_sim,
        n = 3_000,
        true_log_lambda_by_group = true_log_lambda_by_group,
        group_names = group_names,
    )
    model = RandomByTypeModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "group": "group_col",
            "var_prior": (0.0, 1.0),
        },
    )
    fitter = _run_fitter(model, key_fit)

    # Recovered per-group mean log-lambda = log_lambda_0 + epsilon[group]
    draws = fitter.get_parameter_draws()
    group_log_lambdas = (
        draws["log_lambda_0"][:, None] + draws["epsilon"]
    )
    post_mean = np.asarray(jnp.mean(group_log_lambdas, axis = 0))
    for i, truth in enumerate(true_log_lambda_by_group):
        assert abs(post_mean[i] - truth) < 0.4, (
            f"group {group_names[i]}: posterior mean {post_mean[i]:+.3f} "
            f"far from truth {truth:+.3f}"
        )

    # param_ids / group_lookup shape check
    assert list(model.param_ids["param_name"]) == (
        ["log_lambda_0", "log_sigma"] + ["epsilon"] * len(group_names)
    )
    assert sorted(model.group_lookup["group_name"]) == sorted(group_names)


def test_predictions_schema_constant():
    """predict() output for ConstantModel has the expected columns + row count."""
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 400,
        true_log_lambda_by_group = np.array([-1.5]),
        group_names = ["only"],
    )
    model = ConstantModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = _run_fitter(model, jrd.fold_in(key, 1))

    times = jnp.arange(11) / 10.0
    preds = fitter.predict(data = model.build_predict_data(times))
    assert list(preds.columns) == ["p_mean", "p_lower", "p_upper"]
    assert len(preds) == 11
    # P(change) must be in [0, 1] and monotonically non-decreasing in time.
    pm = preds["p_mean"].values
    assert np.all((pm >= 0.0) & (pm <= 1.0))
    assert np.all(np.diff(pm) >= -1e-6)


def test_predictions_schema_random_by_type():
    """predict() output for RandomByTypeModel has one row per (group, time)."""
    group_names = ["aaa", "bbb"]
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 600,
        true_log_lambda_by_group = np.array([-1.2, -0.6]),
        group_names = group_names,
    )
    model = RandomByTypeModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "group": "group_col",
            "var_prior": (0.0, 1.0),
        },
    )
    fitter = _run_fitter(model, jrd.fold_in(key, 1))

    times = jnp.arange(5) / 10.0
    pred_data = model.build_predict_data(times)
    preds = fitter.predict(data = pred_data)
    assert list(preds.columns) == ["p_mean", "p_lower", "p_upper"]
    assert len(preds) == len(group_names) * len(times)
    assert int(pred_data["group"].max()) == len(group_names) - 1


def test_model_registry():
    """Registry exposes the supported models and rejects removed ones."""
    assert set(MODEL_REGISTRY) == {"constant", "random_by_type"}
    assert get_model_class("constant") is ConstantModel
    assert get_model_class("random_by_type") is RandomByTypeModel
    with pytest.raises(ValueError, match = "Unknown model"):
        get_model_class("pseudo_varying")


def test_random_by_type_requires_group():
    """Missing 'group' metadata key is a construction-time error."""
    df = pd.DataFrame({
        "tag_years": [1.0, 2.0],
        "changed": [0, 1],
        "group_col": ["a", "b"],
    })
    with pytest.raises(ValueError, match = "group"):
        RandomByTypeModel(dataset = df, metadata = {"dt_col": "tag_years"})
