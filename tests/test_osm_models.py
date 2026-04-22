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
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
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


def _build_random_by_type(df, reparam = None):
    metadata = {
        "dt_col": "tag_years",
        "group": "group_col",
        "var_prior": (0.0, 1.0),
    }
    if reparam is not None:
        metadata["reparam"] = reparam
    return RandomByTypeModel(dataset = df, metadata = metadata)


@pytest.mark.parametrize("reparam", ["non_centered", "centered"])
def test_random_by_type_recovery(reparam):
    """RandomByTypeModel should recover per-group mean lambdas under both parameterisations."""
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
    model = _build_random_by_type(df, reparam = reparam)
    fitter = _run_fitter(model, key_fit)

    # Recovered per-group mean log-lambda. ``derive_draws`` makes ``epsilon``
    # available regardless of parameterisation.
    draws = fitter.get_parameter_draws()
    assert "epsilon" in draws
    group_log_lambdas = (
        draws["log_lambda_0"][:, None] + draws["epsilon"]
    )
    post_mean = np.asarray(jnp.mean(group_log_lambdas, axis = 0))
    for i, truth in enumerate(true_log_lambda_by_group):
        assert abs(post_mean[i] - truth) < 0.4, (
            f"group {group_names[i]}: posterior mean {post_mean[i]:+.3f} "
            f"far from truth {truth:+.3f}"
        )

    # param_ids shape: epsilon rows always present; epsilon_raw present only
    # when sampled directly (non-centered).
    names = list(model.param_ids["param_name"])
    assert names[:2] == ["log_lambda_0", "log_sigma"]
    if reparam == "centered":
        assert names[2:] == ["epsilon"] * len(group_names)
    else:
        assert names[2:] == (
            ["epsilon_raw"] * len(group_names)
            + ["epsilon"] * len(group_names)
        )
    assert sorted(model.group_lookup["group_name"]) == sorted(group_names)


@pytest.mark.parametrize("reparam", ["non_centered", "centered"])
def test_random_by_type_sufficient_stats_matches_dense(reparam):
    """Sufficient-stats and dense log-likelihoods agree to tight tolerance."""
    import jax

    group_names = ["aaa", "bbb", "ccc", "ddd"]
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 1_500,
        true_log_lambda_by_group = np.array([-1.2, -0.8, -0.4, -0.1]),
        group_names = group_names,
    )
    md_dense = {
        "dt_col": "tag_years",
        "group": "group_col",
        "var_prior": (0.0, 1.0),
        "reparam": reparam,
        "use_sufficient_stats": False,
    }
    md_suff = {**md_dense, "use_sufficient_stats": True}
    model_dense = RandomByTypeModel(dataset = df, metadata = md_dense)
    model_suff = RandomByTypeModel(dataset = df, metadata = md_suff)

    # Pick a not-all-zero parameter point.
    log_sigma = jnp.array(-0.2)
    n_groups = len(group_names)
    if reparam == "centered":
        params = {
            "log_lambda_0": jnp.array(-1.0),
            "log_sigma": log_sigma,
            "epsilon": jnp.array([0.3, -0.2, 0.1, -0.4]),
        }
    else:
        params = {
            "log_lambda_0": jnp.array(-1.0),
            "log_sigma": log_sigma,
            "epsilon_raw": jnp.array([0.3, -0.2, 0.1, -0.4]),
        }

    # Dense path: sum(target*log_p + (1-t)*(-rate)) + param_likelihood.
    rate = model_dense.event_rate_fun(params, model_dense.data)
    log_p = jnp.log(-jnp.expm1(-rate))
    t = model_dense.target
    ll_dense = float(
        jnp.sum(t * log_p + (1.0 - t) * (-rate))
        + model_dense.param_likelihood(params)
    )

    # Suff-stats path: log_likelihood_fun(params, data, target) + prior.
    ll_suff = float(
        model_suff.log_likelihood_fun(
            params, model_suff.data, model_suff.target,
        )
        + model_suff.param_likelihood(params)
    )
    np.testing.assert_allclose(ll_suff, ll_dense, rtol = 1e-4, atol = 1e-3)

    # Gradients must agree too — this is what NUTS actually uses.
    def _wrap_dense(p):
        rate = model_dense.event_rate_fun(p, model_dense.data)
        log_p = jnp.log(-jnp.expm1(-rate))
        t = model_dense.target
        return (
            jnp.sum(t * log_p + (1.0 - t) * (-rate))
            + model_dense.param_likelihood(p)
        )

    def _wrap_suff(p):
        return (
            model_suff.log_likelihood_fun(p, model_suff.data, model_suff.target)
            + model_suff.param_likelihood(p)
        )

    g_dense = jax.grad(_wrap_dense)(params)
    g_suff = jax.grad(_wrap_suff)(params)
    for k in params:
        np.testing.assert_allclose(
            np.asarray(g_suff[k]), np.asarray(g_dense[k]),
            rtol = 1e-3, atol = 1e-3,
        )


def test_random_by_type_reparam_likelihoods_agree():
    """Centered and non-centered log-densities agree at matching parameter points."""
    group_names = ["aaa", "bbb"]
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 500,
        true_log_lambda_by_group = np.array([-1.2, -0.8]),
        group_names = group_names,
    )
    model_c = _build_random_by_type(df, reparam = "centered")
    model_nc = _build_random_by_type(df, reparam = "non_centered")

    # Pick an arbitrary (not-all-zero) point. epsilon = exp(log_sigma)*eps_raw.
    log_lambda_0 = jnp.array(-1.0)
    log_sigma = jnp.array(-0.3)
    epsilon_raw = jnp.array([0.5, -0.7])
    epsilon = jnp.exp(log_sigma) * epsilon_raw

    params_c = {
        "log_lambda_0": log_lambda_0,
        "log_sigma": log_sigma,
        "epsilon": epsilon,
    }
    params_nc = {
        "log_lambda_0": log_lambda_0,
        "log_sigma": log_sigma,
        "epsilon_raw": epsilon_raw,
    }

    # Event rates must match (deterministic transform).
    r_c = model_c.event_rate_fun(params_c, model_c.data)
    r_nc = model_nc.event_rate_fun(params_nc, model_nc.data)
    np.testing.assert_allclose(np.asarray(r_c), np.asarray(r_nc), rtol = 1e-5)

    # Full log-priors differ only by the Jacobian of the N(0, exp(σ)) vs
    # N(0, 1) parameterisation; here we compare in closed form.
    lp_c = float(model_c.param_likelihood(params_c))
    lp_nc = float(model_nc.param_likelihood(params_nc))
    # log p_c(epsilon | log_sigma) = log p_nc(eps_raw) - K * log_sigma
    # (standard change-of-variables for epsilon = exp(log_sigma) * eps_raw).
    k = len(group_names)
    expected_gap = float(k * log_sigma)
    np.testing.assert_allclose(lp_c - lp_nc, -expected_gap, atol = 1e-4)


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


def test_random_by_type_multichain_diagnostics():
    """Multi-chain NUTS produces per-chain draws plus R-hat / ESS diagnostics."""
    group_names = ["aaa", "bbb"]
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_frame(
        key_sim,
        n = 800,
        true_log_lambda_by_group = np.array([-1.0, -0.6]),
        group_names = group_names,
    )
    model = _build_random_by_type(df, reparam = "non_centered")
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_draws = NUM_DRAWS,
        num_chains = 2,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        rng_key = key_fit,
    )
    fitter.fit()

    # chain_draws keeps the (chain, draw, ...) axis; param_draws is flattened.
    log_lambda_chains = fitter.chain_draws["log_lambda_0"]
    assert log_lambda_chains.shape == (2, NUM_DRAWS)
    log_lambda_flat = fitter.param_draws["log_lambda_0"]
    assert log_lambda_flat.shape == (2 * NUM_DRAWS,)

    # Diagnostics DataFrame: R-hat and ESS finite for every sampled parameter.
    diag = fitter.diagnostics
    assert set(diag.columns) == {"parameter", "rhat", "ess_bulk"}
    assert not diag["rhat"].isna().any()
    assert (diag["rhat"] < 1.5).all(), (
        f"max rhat = {diag['rhat'].max():.3f} — suspiciously high for a "
        "well-specified model"
    )
    assert (diag["ess_bulk"] > 10.0).all()


def test_random_by_type_requires_group():
    """Missing 'group' metadata key is a construction-time error."""
    df = pd.DataFrame({
        "tag_years": [1.0, 2.0],
        "changed": [0, 1],
        "group_col": ["a", "b"],
    })
    with pytest.raises(ValueError, match = "group"):
        RandomByTypeModel(dataset = df, metadata = {"dt_col": "tag_years"})
