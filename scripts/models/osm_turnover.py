"""
Fit an empirical Bayes JAX model for OSM POI tag change rates.

Reads ``osm_observations.csv`` (produced by ``osm_data/format_tabular.py``,
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
    osm_turnover_model.n_draws              — number of posterior draws
    osm_turnover_model.save_full_model      — save param_draws and pickled fitter

Prerequisites:
    Run ``osm_data/format_tabular.py`` first.

Output files (in ``model_output`` directory):
    fitted_params.csv   — posterior summaries per parameter
    predictions.csv     — P(change) at t = 0.0..10.0 years per group
    param_draws.csv     — posterior draws (if save_full_model = true)
    fitted_model.pkl    — pickled ModelFitter (if save_full_model = true)
"""

import argparse
import pickle

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
N_DRAWS = config.get("osm_turnover_model", "n_draws")
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
    observations_df = pd.read_csv(OBSERVATIONS_PATH)
    obs_sub = prepare_data_for_model(
        data = observations_df,
        group_key = GROUP_KEY,
        group_values = GROUP_VALUES,
        min_value_count = MIN_VALUE_COUNT,
        t1_col = "last_tag_timestamp",
        t2_col = "obs_timestamp",
    )

    # Build model + fitter -------------------------------------------------->
    model_type = args.model_type or config.get(
        "osm_turnover_model", "default_model_type"
    )
    print(f"Model type: {model_type}")
    model = get_model_class(model_type)(
        dataset = obs_sub,
        metadata = {
            "dt_col": "tag_years",
            "group": GROUP_KEY,
            "var_prior": tuple(
                config.get("osm_turnover_model", "var_prior")
            ),
        },
    )

    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_draws = N_DRAWS,
        param_likelihood = model.param_likelihood,
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
    predict_times = jnp.arange(101) / 10.0
    predict_data = model.build_predict_data(predict_times)
    predictions = (
        fitter.predict(data = predict_data)
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
    if SAVE_FULL_MODEL:
        config.write(
            flatten_param_draws(fitter.get_parameter_draws()),
            "model_output",
            "param_draws",
        )
        with open(
            config.get_file_path("model_output", "fitted_model"), "wb",
        ) as fh:
            pickle.dump(fitter, fh)
