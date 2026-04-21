#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
JAX-based models for OSM POI turnover rate estimation.

Each model class is self-contained: it ingests a raw observations DataFrame
plus a metadata dict, prepares the JAX arrays that ``ModelFitter`` needs, and
exposes ``event_rate_fun`` and ``param_likelihood`` as bound instance methods.

The fitted rate is interpreted as a Poisson event rate per observation; the
change probability is recovered inside ``ModelFitter`` via P = 1 - exp(-rate).
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy import stats


DEFAULT_LOG_LAMBDA_PRIOR_SCALE = 3.0
DEFAULT_VAR_PRIOR = (0.0, 1.0)


class ModelFactory(ABC):
    """
    Base class for OSM turnover models.

    Subclasses must implement ``build_model()`` to populate ``starting_params``,
    ``param_ids`` (and ``group_lookup`` for random-effects variants), plus any
    per-observation columns in ``self.data`` beyond ``dt``.
    """

    def __init__(self, dataset: pd.DataFrame, metadata: dict):
        """
        Args:
            dataset: Observations DataFrame (already filtered/prepared by
                ``prepare_data_for_model``).
            metadata: Model configuration. Required keys vary by subclass; all
                subclasses honor ``dt_col`` (default ``"tag_years"``).
        """
        self.raw_data = dataset
        self.metadata = metadata or {}
        self.data: dict[str, jnp.ndarray] = {}
        self.target: jnp.ndarray | None = None
        self.starting_params: dict[str, jnp.ndarray] = {}
        self.param_ids: pd.DataFrame | None = None
        self.group_lookup: pd.DataFrame | None = None
        self.validate_inputs()
        self.build_model()
        self.assign_targets()

    def validate_inputs(self):
        """Override to validate ``raw_data`` / ``metadata`` before build_model."""
        if not isinstance(self.raw_data, pd.DataFrame):
            raise ValueError("Raw data must be a pandas DataFrame")
        if 'changed' not in self.raw_data.columns:
            raise ValueError("Raw data must include a 'changed' column")

    @abstractmethod
    def build_model(self):
        """Populate ``starting_params``, ``param_ids``, and any extra data columns."""

    def assign_targets(self):
        """Create ``self.data['dt']`` and ``self.target`` as JAX arrays."""
        dt_col = self.metadata.get('dt_col', 'tag_years')
        if dt_col not in self.raw_data.columns:
            raise ValueError(
                f"dt_col '{dt_col}' not found in raw_data columns"
            )
        self.data['dt'] = jnp.asarray(
            self.raw_data[dt_col].to_numpy(), dtype = jnp.float32
        )
        self.target = jnp.asarray(
            self.raw_data['changed'].to_numpy(), dtype = jnp.float32
        )

    @abstractmethod
    def event_rate_fun(
        self,
        params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Poisson event rate per observation."""

    def param_likelihood(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Optional log-prior contribution. Default is flat (0.0)."""
        return jnp.asarray(0.0)

    @abstractmethod
    def build_predict_data(
        self,
        times: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Build the ``data`` dict passed to ``ModelFitter.predict`` for a time grid."""


# Constant rate model -------------------------------------------------------->


class ConstantModel(ModelFactory):
    """
    Constant change rate: λ = exp(log_lambda). One scalar parameter.

    Metadata keys:
        dt_col: Column containing per-observation interval length in years
            (default ``"tag_years"``).
        log_lambda_prior_scale: Standard deviation of the N(0, scale) prior on
            ``log_lambda`` (default ``DEFAULT_LOG_LAMBDA_PRIOR_SCALE``).
    """

    def build_model(self):
        """Define a single scalar ``log_lambda`` parameter."""
        self.starting_params = {"log_lambda": jnp.array(0.0)}
        self.param_ids = pd.DataFrame({
            "parameter": ["log_lambda"],
            "param_name": ["log_lambda"],
            "group_id": [np.nan],
        })
        self.group_lookup = None

    def event_rate_fun(self, params, data):
        return jnp.exp(params["log_lambda"]) * data["dt"]

    def param_likelihood(self, params):
        scale = self.metadata.get(
            "log_lambda_prior_scale", DEFAULT_LOG_LAMBDA_PRIOR_SCALE
        )
        return stats.norm.logpdf(
            params["log_lambda"], loc = 0.0, scale = scale
        ).sum()

    def build_predict_data(self, times):
        return {"dt": jnp.asarray(times, dtype = jnp.float32)}


# Random effects by group ---------------------------------------------------->


class RandomByTypeModel(ModelFactory):
    """
    Random-effects model: λ_i = exp(log_lambda_0 + ε_{g(i)}).

    Per-group epsilons are drawn from N(0, exp(log_sigma)). log_sigma has a
    hyperprior N(var_prior[0], var_prior[1]).

    Metadata keys:
        group: Column name in raw_data holding the grouping variable. **Required.**
        dt_col: Column containing per-observation interval length (default
            ``"tag_years"``).
        var_prior: ``(loc, scale)`` tuple for the hyperprior on ``log_sigma``
            (default ``DEFAULT_VAR_PRIOR``).
    """

    def validate_inputs(self):
        """Require a valid ``group`` metadata entry that is a column in raw_data."""
        super().validate_inputs()
        group_key = self.metadata.get("group") if self.metadata else None
        if group_key is None:
            raise ValueError("Key 'group' is required in metadata")
        if group_key not in self.raw_data.columns:
            raise ValueError(
                f"Group key '{group_key}' not found in raw data columns: "
                f"{', '.join(self.raw_data.columns.tolist())}"
            )

    def build_model(self):
        """Encode group IDs and allocate per-group epsilon parameters."""
        group_key = self.metadata["group"]
        self.raw_data = self.raw_data.dropna(subset = [group_key]).copy()
        self.raw_data["group_id"] = (
            self.raw_data[group_key].astype("category").cat.codes
        )
        self.data["group"] = jnp.asarray(
            self.raw_data["group_id"].to_numpy(), dtype = jnp.int32
        )
        self.group_lookup = (
            self.raw_data
            .loc[:, [group_key, "group_id"]]
            .rename(columns = {group_key: "group_name"})
            .drop_duplicates()
            .sort_values("group_id", ascending = True)
            .reset_index(drop = True)
        )
        n_groups = self.group_lookup.shape[0]
        self.starting_params = {
            "log_lambda_0": jnp.array(0.0),
            "log_sigma": jnp.array(0.0),
            "epsilon": jnp.zeros(n_groups),
        }
        self.param_ids = pd.DataFrame({
            "parameter": (
                ["log_lambda_0", "log_sigma"]
                + [f"epsilon[{i}]" for i in range(n_groups)]
            ),
            "param_name": (
                ["log_lambda_0", "log_sigma"] + ["epsilon"] * n_groups
            ),
            "group_id": [np.nan, np.nan] + list(range(n_groups)),
        })

    def event_rate_fun(self, params, data):
        log_lambda = (
            params["log_lambda_0"] + params["epsilon"][data["group"]]
        )
        return jnp.exp(log_lambda) * data["dt"]

    def param_likelihood(self, params):
        var_loc, var_scale = self.metadata.get("var_prior", DEFAULT_VAR_PRIOR)
        ll = stats.norm.logpdf(
            params["log_sigma"], loc = var_loc, scale = var_scale
        ).sum()
        ll = ll + stats.norm.logpdf(
            params["epsilon"],
            loc = 0.0,
            scale = jnp.exp(params["log_sigma"]),
        ).sum()
        return ll

    def build_predict_data(self, times):
        if self.group_lookup is None:
            raise RuntimeError(
                "group_lookup is unset; build_model must run first"
            )
        n_groups = self.group_lookup.shape[0]
        times = jnp.asarray(times, dtype = jnp.float32)
        n_periods = times.shape[0]
        return {
            "dt": jnp.tile(times, n_groups),
            "group": jnp.repeat(
                jnp.arange(n_groups, dtype = jnp.int32), n_periods
            ),
        }


# Registry ------------------------------------------------------------------->


MODEL_REGISTRY = {
    "constant": ConstantModel,
    "random_by_type": RandomByTypeModel,
}


def get_model_class(model_name: str) -> type[ModelFactory]:
    """
    Return a ``ModelFactory`` subclass by name from ``MODEL_REGISTRY``.

    Args:
        model_name: Registry key (``"constant"`` or ``"random_by_type"``).

    Returns:
        The corresponding ``ModelFactory`` subclass.

    Raises:
        ValueError: If ``model_name`` is not a registered model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: "
            f"{', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
