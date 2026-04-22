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
# Floor applied when taking the log of an empirical rate. Groups with zero
# observed changes would otherwise give log(0) = -inf on init.
_EMPIRICAL_RATE_FLOOR = 1e-8
# Bounds on the empirical-Bayes log_sigma initializer, to guard against
# single-observation groups driving the starting point to extremes.
_LOG_SIGMA_INIT_BOUNDS = (-3.0, 1.0)
# Supported parameterizations for RandomByTypeModel.
_VALID_REPARAMS = ("centered", "non_centered")
_DEFAULT_REPARAM = "non_centered"


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
        # Optional sufficient-statistics log-likelihood override. If set by
        # ``build_model``, the fitter bypasses the per-observation dense path
        # and calls ``log_likelihood_fun(params, data, target)`` directly.
        self.log_likelihood_fun: callable | None = None
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

    def derive_draws(
        self,
        draws: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """
        Augment posterior draws with any derived/back-transformed parameters.

        Default is an identity map. Override in subclasses that sample a
        reparameterised form (e.g. non-centered ``epsilon_raw``) to expose the
        natural parameter (``epsilon``) to downstream consumers.
        """
        return draws

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
        dt_col = self.metadata.get("dt_col", "tag_years")
        empirical_rate = (
            self.raw_data["changed"].mean() / self.raw_data[dt_col].mean()
        )
        log_lambda_init = float(
            np.log(max(empirical_rate, _EMPIRICAL_RATE_FLOOR))
        )
        self.starting_params = {"log_lambda": jnp.array(log_lambda_init)}
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

    Two equivalent parameterisations are supported via ``metadata["reparam"]``:

    * ``"centered"`` (legacy): the sampler traces ``epsilon`` directly. Simple,
      but the posterior is funnel-shaped near sparse groups and NUTS tends to
      diverge there.
    * ``"non_centered"`` (**default**): the sampler traces
      ``epsilon_raw ~ N(0, 1)`` and we reconstruct
      ``epsilon = exp(log_sigma) * epsilon_raw`` post-hoc. Removes the funnel,
      usually yielding zero divergences and higher ESS on small groups with
      no change to well-identified groups.

    Under ``non_centered`` the natural-scale ``epsilon`` draws are exposed via
    ``derive_draws`` so downstream consumers (``predict``, parameter tables,
    saved ``param_draws.csv``) see the same parameter names either way.

    Metadata keys:
        group: Column name in raw_data holding the grouping variable. **Required.**
        dt_col: Column containing per-observation interval length (default
            ``"tag_years"``).
        var_prior: ``(loc, scale)`` tuple for the hyperprior on ``log_sigma``
            (default ``DEFAULT_VAR_PRIOR``).
        reparam: ``"centered"`` or ``"non_centered"`` (default
            ``"non_centered"``).
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
        reparam = (self.metadata or {}).get("reparam", _DEFAULT_REPARAM)
        if reparam not in _VALID_REPARAMS:
            raise ValueError(
                f"metadata['reparam']={reparam!r} not in "
                f"{_VALID_REPARAMS}"
            )
        suff = (self.metadata or {}).get("use_sufficient_stats", True)
        if not isinstance(suff, bool):
            raise ValueError(
                f"metadata['use_sufficient_stats']={suff!r} must be a bool"
            )

    def build_model(self):
        """Encode group IDs and allocate per-group epsilon parameters."""
        group_key = self.metadata["group"]
        dt_col = self.metadata.get("dt_col", "tag_years")
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

        # Empirical-Bayes init so warmup doesn't burn steps escaping a bad
        # starting point. log_lambda_0: empirical pooled log-rate.
        # log_sigma: log(std of per-group empirical log-rates), bounded.
        overall_rate = (
            self.raw_data["changed"].mean() / self.raw_data[dt_col].mean()
        )
        log_lambda_0_init = float(
            np.log(max(overall_rate, _EMPIRICAL_RATE_FLOOR))
        )
        per_group = self.raw_data.groupby("group_id", observed = True).agg(
            changed_mean = ("changed", "mean"),
            dt_mean = (dt_col, "mean"),
        )
        per_group_rates = np.maximum(
            (per_group["changed_mean"] / per_group["dt_mean"]).to_numpy(),
            _EMPIRICAL_RATE_FLOOR,
        )
        per_group_log_rates = np.log(per_group_rates)
        if len(per_group_log_rates) > 1:
            empirical_log_sigma = float(np.log(
                max(float(np.std(per_group_log_rates)), 1e-3)
            ))
        else:
            empirical_log_sigma = 0.0
        log_sigma_init = float(np.clip(
            empirical_log_sigma,
            _LOG_SIGMA_INIT_BOUNDS[0],
            _LOG_SIGMA_INIT_BOUNDS[1],
        ))
        self._reparam = self.metadata.get("reparam", _DEFAULT_REPARAM)
        self._use_sufficient_stats = self.metadata.get(
            "use_sufficient_stats", True
        )
        eps_key = "epsilon" if self._reparam == "centered" else "epsilon_raw"
        self.starting_params = {
            "log_lambda_0": jnp.array(log_lambda_0_init),
            "log_sigma": jnp.array(log_sigma_init),
            eps_key: jnp.zeros(n_groups),
        }
        if self._use_sufficient_stats:
            self._build_sufficient_stats(
                dt_col = dt_col, n_groups = n_groups,
            )
            self.log_likelihood_fun = self._suff_stats_log_likelihood
        # Under non-centered, expose BOTH epsilon_raw (what NUTS sees) and the
        # back-transformed epsilon (what consumers expect) in param_ids.
        param_rows = ["log_lambda_0", "log_sigma"]
        param_names = ["log_lambda_0", "log_sigma"]
        group_ids: list = [np.nan, np.nan]
        if self._reparam == "non_centered":
            param_rows += [f"epsilon_raw[{i}]" for i in range(n_groups)]
            param_names += ["epsilon_raw"] * n_groups
            group_ids += list(range(n_groups))
        param_rows += [f"epsilon[{i}]" for i in range(n_groups)]
        param_names += ["epsilon"] * n_groups
        group_ids += list(range(n_groups))
        self.param_ids = pd.DataFrame({
            "parameter": param_rows,
            "param_name": param_names,
            "group_id": group_ids,
        })

    def _build_sufficient_stats(self, dt_col: str, n_groups: int):
        """
        Precompute per-group unchanged-time sums and changed-only arrays.

        The Bernoulli-on-Poisson log-likelihood factors as
            Σ_g [ -λ_g · Σ_{i: g(i)=g, y_i=0} dt_i ]
            + Σ_{i: y_i=1} log(1 - exp(-λ_{g(i)} · dt_i))
        so the unchanged contribution collapses from N terms to K terms, and
        the changed contribution runs over only the ~10 % of observations
        that flipped. Stashes three arrays into ``self.data`` for the jitted
        log-likelihood to gather.
        """
        dt_np = self.raw_data[dt_col].to_numpy().astype(np.float32)
        target_np = self.raw_data["changed"].to_numpy().astype(np.int32)
        group_np = self.raw_data["group_id"].to_numpy().astype(np.int32)

        mask_unchanged = target_np == 0
        sum_dt = np.zeros(n_groups, dtype = np.float32)
        np.add.at(
            sum_dt, group_np[mask_unchanged], dt_np[mask_unchanged],
        )

        mask_changed = ~mask_unchanged
        self.data["sum_dt_unchanged"] = jnp.asarray(sum_dt)
        self.data["group_changed"] = jnp.asarray(
            group_np[mask_changed], dtype = jnp.int32,
        )
        self.data["dt_changed"] = jnp.asarray(
            dt_np[mask_changed], dtype = jnp.float32,
        )

    def _log_lambda_per_group(self, params):
        """Shared helper: per-group log-rate under either parameterisation."""
        if self._reparam == "centered":
            return params["log_lambda_0"] + params["epsilon"]
        return (
            params["log_lambda_0"]
            + jnp.exp(params["log_sigma"]) * params["epsilon_raw"]
        )

    def event_rate_fun(self, params, data):
        log_lambda_g = self._log_lambda_per_group(params)
        return jnp.exp(log_lambda_g[data["group"]]) * data["dt"]

    def _suff_stats_log_likelihood(self, params, data, target):
        """
        Bernoulli-on-Poisson log-likelihood via K-vector + N_changed fold.

        ``target`` is intentionally unused — its information is captured by
        ``data["sum_dt_unchanged"]`` / ``data["group_changed"]`` /
        ``data["dt_changed"]``, built once in ``_build_sufficient_stats``.
        """
        del target
        lam_g = jnp.exp(self._log_lambda_per_group(params))
        ll_unchanged = -jnp.sum(lam_g * data["sum_dt_unchanged"])
        rate_c = lam_g[data["group_changed"]] * data["dt_changed"]
        ll_changed = jnp.sum(jnp.log(-jnp.expm1(-rate_c)))
        return ll_unchanged + ll_changed

    def param_likelihood(self, params):
        var_loc, var_scale = self.metadata.get("var_prior", DEFAULT_VAR_PRIOR)
        ll = stats.norm.logpdf(
            params["log_sigma"], loc = var_loc, scale = var_scale
        ).sum()
        if self._reparam == "centered":
            ll = ll + stats.norm.logpdf(
                params["epsilon"],
                loc = 0.0,
                scale = jnp.exp(params["log_sigma"]),
            ).sum()
        else:
            ll = ll + stats.norm.logpdf(
                params["epsilon_raw"], loc = 0.0, scale = 1.0,
            ).sum()
        return ll

    def derive_draws(self, draws):
        """Under non-centered, add ``epsilon = exp(log_sigma) * epsilon_raw``."""
        if self._reparam == "centered":
            return draws
        # draws["log_sigma"]: (n_draws,). draws["epsilon_raw"]: (n_draws, K).
        epsilon = (
            jnp.exp(draws["log_sigma"])[:, None] * draws["epsilon_raw"]
        )
        return {**draws, "epsilon": epsilon}

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
