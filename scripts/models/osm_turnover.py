"""
Fit an empirical Bayes JAX model for OSM POI tag change rates.

Reads ``osm_observations.parquet`` (produced by ``osm_data/format_tabular.py``,
one row per (POI version, shared_label)) and fits a Poisson change-rate
model using BlackJAX NUTS. The model estimates a per-group change rate λ
(events per year). Predictions give the probability that a tag changes
within t years for t = 0.0, 0.1, ..., 10.0. Supports ``constant`` and
``random_by_type`` model specifications.

Random effects are grouped by shared taxonomy label
(``osm_turnover_model.group_key: shared_label`` — the default) so that all
POIs are compared apples-to-apples under a single unified model, instead of
one model per OSM tag key.

Config keys used (config.yaml):
    directories.osm_data                    — input data directory
    directories.model_output                — output directory for results
    osm_turnover_model.group_key            — column to group by (null =
                                              constant model; default
                                              "shared_label")
    osm_turnover_model.group_values         — subset of group values (null = all)
    osm_turnover_model.min_value_count      — minimum observations to include a group
    osm_turnover_model.default_model_type   — "constant" or "random_by_type"
                                              (overridable via --model-type)
    osm_turnover_model.var_prior            — (loc, scale) hyperprior on log_sigma
    osm_turnover_model.logit_delta_prior    — (loc, scale) prior on logit_delta_0
                                              intercept
    osm_turnover_model.logit_delta_var_prior — (loc, scale) tight hyperprior on
                                              log_tau (per-group δ scale)
    osm_turnover_model.n_warmup             — NUTS warmup steps (adaptation)
    osm_turnover_model.n_samples            — posterior draws retained
    osm_turnover_model.n_chains             — number of NUTS chains (vmapped)
    osm_turnover_model.save_full_model      — save param_draws and pickled fitter

Prerequisites:
    Run ``osm_data/format_tabular.py`` first.

Output files (in ``model_output`` directory):
    fitted_params.csv   — posterior summaries per parameter
    predictions.csv     — P(change) at t = 0.0..10.0 years per group
    diagnostics.csv     — per-parameter R-hat / bulk-ESS (multi-chain only)
    inference_data.nc   — ArviZ InferenceData (optional, if arviz installed)
    param_draws.csv     — posterior draws (if save_full_model = true)
"""

import argparse

import jax.numpy as jnp
import numpy as np
import pandas as pd
from config_versioned import Config

from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import get_model_class
from openpois.models.setup import prepare_data_for_model


# Globals
config = Config("~/repos/openpois/config.yaml")

MODEL_DIR = config.get_dir_path("model_output")
OBSERVATIONS_PATH = config.get_file_path("osm_data", "osm_observations")
GROUP_KEY = config.get("osm_turnover_model", "group_key", fail_if_none = False)
GROUP_VALUES = config.get("osm_turnover_model", "group_values", fail_if_none = False)
MIN_VALUE_COUNT = config.get(
    "osm_turnover_model", "min_value_count", fail_if_none = False
)
N_WARMUP = config.get("osm_turnover_model", "n_warmup", fail_if_none = False)
N_SAMPLES = config.get("osm_turnover_model", "n_samples", fail_if_none = False)
N_CHAINS = config.get("osm_turnover_model", "n_chains", fail_if_none = False)
# Back-compat: older configs used `n_draws` for both warmup and sampling.
_LEGACY_N_DRAWS = config.get(
    "osm_turnover_model", "n_draws", fail_if_none = False
)
if N_WARMUP is None:
    N_WARMUP = _LEGACY_N_DRAWS if _LEGACY_N_DRAWS is not None else 1_000
if N_SAMPLES is None:
    N_SAMPLES = _LEGACY_N_DRAWS if _LEGACY_N_DRAWS is not None else 1_000
if N_CHAINS is None:
    N_CHAINS = 1
SAVE_FULL_MODEL = config.get("osm_turnover_model", "save_full_model")


def flatten_param_draws(
    param_draws: dict[str, jnp.ndarray],
) -> pd.DataFrame:
    """
    Flatten the pytree from ``ModelFitter.get_parameter_draws`` into a
    DataFrame with one column per scalar parameter, matching the labels
    emitted by ``get_parameter_table`` (e.g. ``log_lambda``, ``epsilon[0]``).
    """
    columns: dict[str, np.ndarray] = {}
    for name, draws in param_draws.items():
        arr = np.asarray(draws)
        n_draws = arr.shape[0]
        flat = arr.reshape(n_draws, -1)
        param_shape = arr.shape[1:]
        for i in range(flat.shape[1]):
            if len(param_shape) == 0:
                label = name
            else:
                idx = np.unravel_index(i, param_shape)
                label = f"{name}[{','.join(str(k) for k in idx)}]"
            columns[label] = flat[:, i]
    return pd.DataFrame(columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Fit a JAX turnover model over OSM observations.",
    )
    parser.add_argument(
        "--model-type",
        choices = ["constant", "random_by_type"],
        default = None,
        help = (
            "Override osm_turnover_model.default_model_type for this run."
        ),
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents = True, exist_ok = True)
    config.write_self("model_output")

    # Data preparation ------------------------------------------------------>
    observations_df = pd.read_parquet(OBSERVATIONS_PATH)
    # t1_col defaults to "last_obs_timestamp" in prepare_data_for_model, so
    # tag_years is the inter-observation interval the per-row Bernoulli-on-
    # Poisson likelihood requires (methodology §1.2).
    obs_sub = prepare_data_for_model(
        data = observations_df,
        group_key = GROUP_KEY,
        group_values = GROUP_VALUES,
        min_value_count = MIN_VALUE_COUNT,
        t2_col = "obs_timestamp",
    )

    # Build model + fitter -------------------------------------------------->
    model_type = args.model_type or config.get(
        "osm_turnover_model", "default_model_type"
    )
    print(f"Model type: {model_type}")
    metadata = {
        "dt_col": "tag_years",
        "group": GROUP_KEY,
        "var_prior": tuple(
            config.get("osm_turnover_model", "var_prior")
        ),
    }
    logit_delta_prior = config.get(
        "osm_turnover_model", "logit_delta_prior", fail_if_none = False
    )
    if logit_delta_prior is not None:
        metadata["logit_delta_prior"] = tuple(logit_delta_prior)
    logit_delta_var_prior = config.get(
        "osm_turnover_model", "logit_delta_var_prior", fail_if_none = False
    )
    if logit_delta_var_prior is not None:
        metadata["logit_delta_var_prior"] = tuple(logit_delta_var_prior)
    model = get_model_class(model_type)(
        dataset = obs_sub,
        metadata = metadata,
    )

    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_warmup = N_WARMUP,
        num_samples = N_SAMPLES,
        num_chains = N_CHAINS,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        log_1md_fun = getattr(model, "log_1md_fun", None),
        verbose = True,
    )
    fitter.fit()

    # Fitted parameter summary --------------------------------------------->
    fitted_params = (
        fitter.get_parameter_table()
        .merge(model.param_ids, on = "parameter", how = "left")
    )
    if model.group_lookup is not None:
        fitted_params = fitted_params.merge(
            model.group_lookup, on = "group_id", how = "left"
        )

    # Predictions ----------------------------------------------------------->
    # Emit both regimes (methodology §4.2 Step G): the conditional formula
    # populates p_mean/p_lower/p_upper (δ-independent, right for rating
    # already-observed POIs); the fresh formula populates p_fresh_* (uses δ,
    # right for rating a hypothetical freshly tagged POI).
    predict_times = jnp.arange(101) / 10.0
    predict_data = model.build_predict_data(predict_times)
    conditional = fitter.predict(data = predict_data, mode = "conditional")
    fresh = (
        fitter.predict(data = predict_data, mode = "fresh")
        .rename(columns = {
            "p_mean": "p_fresh_mean",
            "p_lower": "p_fresh_lower",
            "p_upper": "p_fresh_upper",
        })
    )
    predictions = (
        pd.concat([conditional, fresh], axis = 1)
        .assign(t1 = 0.0, units = "years")
    )
    predictions["t2"] = np.asarray(predict_data["dt"])
    if model.group_lookup is not None:
        predictions["group"] = np.asarray(predict_data["group"])
        predictions = (
            predictions
            .merge(
                model.group_lookup.rename(columns = {"group_id": "group"}),
                on = "group",
                how = "left",
            )
            .sort_values(["group_name", "t2"], ascending = True)
        )

    # Save ----------------------------------------------------------------->
    config.write(fitted_params, "model_output", "fitted_params")
    config.write(predictions, "model_output", "predictions")
    if fitter.diagnostics is not None:
        config.write(fitter.diagnostics, "model_output", "diagnostics")
    try:
        idata = fitter.to_inference_data()
        idata.to_netcdf(
            str(config.get_file_path("model_output", "inference_data"))
        )
    except ImportError:
        print("arviz not installed — skipping inference_data.nc")
    if SAVE_FULL_MODEL:
        config.write(
            flatten_param_draws(fitter.get_parameter_draws()),
            "model_output",
            "param_draws",
        )
