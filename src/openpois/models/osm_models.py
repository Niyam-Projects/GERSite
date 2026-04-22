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

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy import stats


DEFAULT_LOG_LAMBDA_PRIOR_SCALE = 3.0
DEFAULT_VAR_PRIOR = (0.0, 1.0)
# Zero-inflated exponential (ZIE) δ prior on the log-odds scale, per
# turnover-model-methodology.md §1.7. Prior mean −3 → δ ≈ 5 %.
DEFAULT_LOGIT_DELTA_PRIOR = (-3.0, 1.0)
# Tight hyperprior on log_tau (random-effect scale for per-group logit_delta).
# Tau median ≈ exp(-2) ≈ 0.135 on the logit scale.
DEFAULT_LOGIT_DELTA_VAR_PRIOR = (-2.0, 0.5)
# Floor applied when taking the log of an empirical rate. Groups with zero
# observed changes would otherwise give log(0) = -inf on init.
_EMPIRICAL_RATE_FLOOR = 1e-8
# Bounds on the empirical-Bayes log_sigma initializer, to guard against
# single-observation groups driving the starting point to extremes.
_LOG_SIGMA_INIT_BOUNDS = (-3.0, 1.0)
# Supported parameterizations for RandomByTypeModel.
_VALID_REPARAMS = ("centered", "non_centered")
_DEFAULT_REPARAM = "non_centered"


def _empirical_rate_from_nonfirst(
    raw_data: pd.DataFrame, dt_col: str,
) -> float:
    """
    Pooled change rate using non-first-interval rows only.

    Per methodology §4.2 Step F: the δ-component changes at t = 0, so including
    first-interval rows in the empirical λ init would let instant-change mass
    inflate the starting point. Falls back to the full frame if no non-first
    rows exist (e.g. tests that don't emit the flag).
    """
    if "is_first_interval" in raw_data.columns:
        mask = ~raw_data["is_first_interval"].astype(bool)
        if mask.sum() > 0:
            sub = raw_data.loc[mask]
            return float(sub["changed"].mean() / sub[dt_col].mean())
    return float(raw_data["changed"].mean() / raw_data[dt_col].mean())


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
        """Create ``self.data['dt']``, ``self.target``, and ``is_first_interval``."""
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
        # is_first_interval flags per-individual first rows for the ZIE δ
        # extension (methodology §1.7). Default zeros for legacy frames that
        # don't carry the column — in that case δ has no effect on the
        # likelihood (no first-interval rows), so the sampler just exercises
        # the prior on logit_delta.
        if 'is_first_interval' in self.raw_data.columns:
            self.data['is_first_interval'] = jnp.asarray(
                self.raw_data['is_first_interval'].to_numpy().astype(bool),
                dtype = jnp.float32,
            )
        else:
            self.data['is_first_interval'] = jnp.zeros_like(self.target)

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
    Constant change rate with ZIE δ mixture.

    λ = exp(log_lambda); δ = sigmoid(logit_delta). A fraction δ of individuals
    change at t = 0 (methodology §1.7); the remaining 1−δ fraction follow
    Exponential(λ). Only the first interval of each individual carries the
    (1−δ) discount — see ``log_likelihood_fun``.

    Metadata keys:
        dt_col: Column containing per-observation interval length in years
            (default ``"tag_years"``).
        log_lambda_prior_scale: Standard deviation of the N(0, scale) prior on
            ``log_lambda`` (default ``DEFAULT_LOG_LAMBDA_PRIOR_SCALE``).
        logit_delta_prior: (loc, scale) tuple for the Normal prior on
            ``logit_delta`` (default ``DEFAULT_LOGIT_DELTA_PRIOR`` = (−3, 1)).
    """

    def build_model(self):
        """Define ``log_lambda`` and ``logit_delta`` parameters + ZIE likelihood."""
        dt_col = self.metadata.get("dt_col", "tag_years")
        empirical_rate = _empirical_rate_from_nonfirst(self.raw_data, dt_col)
        log_lambda_init = float(
            np.log(max(empirical_rate, _EMPIRICAL_RATE_FLOOR))
        )
        logit_delta_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        self.starting_params = {
            "log_lambda": jnp.array(log_lambda_init),
            "logit_delta": jnp.array(logit_delta_init),
        }
        self.param_ids = pd.DataFrame({
            "parameter": ["log_lambda", "logit_delta", "delta"],
            "param_name": ["log_lambda", "logit_delta", "delta"],
            "group_id": [np.nan, np.nan, np.nan],
        })
        self.group_lookup = None
        self.log_likelihood_fun = self._zie_log_likelihood

    @staticmethod
    def _zie_log_likelihood(params, data, target):
        """
        Bernoulli-on-Poisson log-likelihood with the ZIE δ discount on the
        first interval of each individual (methodology §4.2 Step D).

        First-interval rows (``is_first = 1``):
            y=0: log(1−δ) − λ·Δ
            y=1: log(1 − (1−δ)·exp(−λ·Δ))
        Non-first rows:
            y=0: −λ·Δ
            y=1: log(1 − exp(−λ·Δ))
        """
        lam = jnp.exp(params["log_lambda"])
        dt = data["dt"]
        is_first = data["is_first_interval"]
        rate = lam * dt
        # log(1−δ) = log(sigmoid(−logit_delta)) via log_sigmoid for numerical
        # stability at extreme logit_delta values.
        log_1md = jax.nn.log_sigmoid(-params["logit_delta"])

        log_p_std = jnp.log(-jnp.expm1(-rate))          # non-first, y=1
        log_1mp_std = -rate                              # non-first, y=0
        log_1mp_zie = log_1md - rate                     # first, y=0
        log_p_zie = jnp.log(-jnp.expm1(log_1md - rate))  # first, y=1

        log_p = jnp.where(is_first > 0, log_p_zie, log_p_std)
        log_1mp = jnp.where(is_first > 0, log_1mp_zie, log_1mp_std)
        per_obs = target * log_p + (1.0 - target) * log_1mp
        return jnp.sum(per_obs)

    def event_rate_fun(self, params, data):
        return jnp.exp(params["log_lambda"]) * data["dt"]

    def param_likelihood(self, params):
        scale = self.metadata.get(
            "log_lambda_prior_scale", DEFAULT_LOG_LAMBDA_PRIOR_SCALE
        )
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        lp = stats.norm.logpdf(
            params["log_lambda"], loc = 0.0, scale = scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["logit_delta"], loc = delta_loc, scale = delta_scale
        ).sum()
        return lp

    def derive_draws(self, draws):
        """Expose ``delta = sigmoid(logit_delta)`` alongside raw draws."""
        return {**draws, "delta": jax.nn.sigmoid(draws["logit_delta"])}

    def build_predict_data(self, times):
        return {"dt": jnp.asarray(times, dtype = jnp.float32)}


# Random effects by group ---------------------------------------------------->


class RandomByTypeModel(ModelFactory):
    """
    Random-effects model on both λ and δ, grouped by a shared label.

    Per-group log-rate:
        log λ_g = log_lambda_0 + ε_g,   ε_g ~ N(0, exp(log_sigma))
        log_sigma ~ N(var_prior[0], var_prior[1])

    Per-group zero-inflated mixture mass on the logit scale:
        logit δ_g = logit_delta_0 + η_g, η_g ~ N(0, exp(log_tau))
        log_tau ~ N(logit_delta_var_prior[0], logit_delta_var_prior[1])

    Two equivalent parameterisations are supported via ``metadata["reparam"]``
    and apply symmetrically to both ε and η:

    * ``"centered"`` (legacy): the sampler traces ``epsilon`` / ``eta``
      directly. Simple, but the posterior is funnel-shaped near sparse groups
      and NUTS tends to diverge there.
    * ``"non_centered"`` (**default**): the sampler traces
      ``epsilon_raw ~ N(0, 1)`` / ``eta_raw ~ N(0, 1)`` and we reconstruct
      ``epsilon = exp(log_sigma) * epsilon_raw`` and
      ``eta = exp(log_tau) * eta_raw`` post-hoc. Removes the funnel, usually
      yielding zero divergences and higher ESS on small groups with no change
      to well-identified groups.

    Under ``non_centered`` the natural-scale ``epsilon`` and ``eta`` draws are
    exposed via ``derive_draws`` along with the per-group ``logit_delta`` and
    ``delta``, so downstream consumers (``predict``, parameter tables, saved
    ``param_draws.csv``) see the same derived names either way.

    Metadata keys:
        group: Column name in raw_data holding the grouping variable. **Required.**
        dt_col: Column containing per-observation interval length (default
            ``"tag_years"``).
        var_prior: ``(loc, scale)`` tuple for the hyperprior on ``log_sigma``
            (default ``DEFAULT_VAR_PRIOR``).
        logit_delta_prior: ``(loc, scale)`` tuple for the prior on the
            intercept ``logit_delta_0`` (default ``DEFAULT_LOGIT_DELTA_PRIOR``).
        logit_delta_var_prior: ``(loc, scale)`` tuple for the tight hyperprior
            on ``log_tau`` (default ``DEFAULT_LOGIT_DELTA_VAR_PRIOR``).
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
        """Encode group IDs and allocate per-group epsilon + logit_delta."""
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
        # starting point. log_lambda_0: non-first-only pooled log-rate
        # (excludes δ-mass). log_sigma: log(std of per-group empirical
        # log-rates on non-first rows), bounded.
        if "is_first_interval" in self.raw_data.columns:
            init_df = self.raw_data.loc[
                ~self.raw_data["is_first_interval"].astype(bool)
            ]
            if init_df.empty:
                init_df = self.raw_data
        else:
            init_df = self.raw_data
        overall_rate = (
            init_df["changed"].mean() / init_df[dt_col].mean()
        )
        log_lambda_0_init = float(
            np.log(max(overall_rate, _EMPIRICAL_RATE_FLOOR))
        )
        per_group = (
            init_df.groupby("group_id", observed = True)
            .agg(
                changed_mean = ("changed", "mean"),
                dt_mean = (dt_col, "mean"),
            )
            .reindex(range(n_groups))
        )
        per_group["changed_mean"] = per_group["changed_mean"].fillna(
            float(init_df["changed"].mean())
        )
        per_group["dt_mean"] = per_group["dt_mean"].fillna(
            float(init_df[dt_col].mean())
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
        logit_delta_0_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        log_tau_init = float(
            self.metadata.get(
                "logit_delta_var_prior", DEFAULT_LOGIT_DELTA_VAR_PRIOR,
            )[0]
        )
        self._reparam = self.metadata.get("reparam", _DEFAULT_REPARAM)
        self._use_sufficient_stats = self.metadata.get(
            "use_sufficient_stats", True
        )
        eps_key = "epsilon" if self._reparam == "centered" else "epsilon_raw"
        eta_key = "eta" if self._reparam == "centered" else "eta_raw"
        self.starting_params = {
            "log_lambda_0": jnp.array(log_lambda_0_init),
            "log_sigma": jnp.array(log_sigma_init),
            "logit_delta_0": jnp.array(logit_delta_0_init),
            "log_tau": jnp.array(log_tau_init),
            eps_key: jnp.zeros(n_groups),
            eta_key: jnp.zeros(n_groups),
        }
        if self._use_sufficient_stats:
            self._build_sufficient_stats(
                dt_col = dt_col, n_groups = n_groups,
            )
            self.log_likelihood_fun = self._suff_stats_log_likelihood
        else:
            # Dense per-row ZIE path — needed whenever suff-stats is off so the
            # log-likelihood stays methodologically consistent with §4.2.
            self.log_likelihood_fun = self._zie_dense_log_likelihood
        # Under non-centered, expose BOTH *_raw (what NUTS sees) and the
        # back-transformed natural-scale draws (what consumers expect) in
        # param_ids. Per-group logit_delta and delta are always exposed so
        # end users can read off per-category instant-change fractions.
        param_rows = ["log_lambda_0", "log_sigma", "logit_delta_0", "log_tau"]
        param_names = list(param_rows)
        group_ids: list = [np.nan, np.nan, np.nan, np.nan]
        if self._reparam == "non_centered":
            param_rows += [f"epsilon_raw[{i}]" for i in range(n_groups)]
            param_names += ["epsilon_raw"] * n_groups
            group_ids += list(range(n_groups))
        param_rows += [f"epsilon[{i}]" for i in range(n_groups)]
        param_names += ["epsilon"] * n_groups
        group_ids += list(range(n_groups))
        if self._reparam == "non_centered":
            param_rows += [f"eta_raw[{i}]" for i in range(n_groups)]
            param_names += ["eta_raw"] * n_groups
            group_ids += list(range(n_groups))
        param_rows += [f"eta[{i}]" for i in range(n_groups)]
        param_names += ["eta"] * n_groups
        group_ids += list(range(n_groups))
        param_rows += [f"logit_delta[{i}]" for i in range(n_groups)]
        param_names += ["logit_delta"] * n_groups
        group_ids += list(range(n_groups))
        param_rows += [f"delta[{i}]" for i in range(n_groups)]
        param_names += ["delta"] * n_groups
        group_ids += list(range(n_groups))
        self.param_ids = pd.DataFrame({
            "parameter": param_rows,
            "param_name": param_names,
            "group_id": group_ids,
        })

    def _build_sufficient_stats(self, dt_col: str, n_groups: int):
        """
        Precompute first/non-first unchanged-time sums and changed-only arrays.

        The ZIE Bernoulli-on-Poisson log-likelihood factors as
            -Σ_g λ_g · (sum_dt_unchanged_nonfirst[g] + sum_dt_unchanged_first[g])
            + Σ_g n_first_unchanged_by_group[g] · log(1−δ_g)
            + Σ log(1 − exp(−λ_g·Δ))                over non-first y=1 rows
            + Σ log(1 − (1−δ_g)·exp(−λ_g·Δ))        over first y=1 rows
        (methodology §4.3). The unchanged contribution collapses from N terms
        to 3K scalars; the changed contribution runs over only the ~10 %
        of observations that flipped. Stashes six arrays into ``self.data``
        for the jitted log-likelihood to gather.
        """
        dt_np = self.raw_data[dt_col].to_numpy().astype(np.float32)
        target_np = self.raw_data["changed"].to_numpy().astype(np.int32)
        group_np = self.raw_data["group_id"].to_numpy().astype(np.int32)
        if "is_first_interval" in self.raw_data.columns:
            is_first_np = (
                self.raw_data["is_first_interval"].to_numpy().astype(bool)
            )
        else:
            is_first_np = np.zeros(len(self.raw_data), dtype = bool)

        mask_unchanged = target_np == 0
        mask_unchanged_first = mask_unchanged & is_first_np
        mask_unchanged_nonfirst = mask_unchanged & ~is_first_np
        sum_dt_first = np.zeros(n_groups, dtype = np.float32)
        sum_dt_nonfirst = np.zeros(n_groups, dtype = np.float32)
        n_first_unchanged_by_group = np.zeros(n_groups, dtype = np.float32)
        np.add.at(
            sum_dt_first,
            group_np[mask_unchanged_first],
            dt_np[mask_unchanged_first],
        )
        np.add.at(
            sum_dt_nonfirst,
            group_np[mask_unchanged_nonfirst],
            dt_np[mask_unchanged_nonfirst],
        )
        np.add.at(
            n_first_unchanged_by_group,
            group_np[mask_unchanged_first],
            1.0,
        )

        mask_changed = ~mask_unchanged
        mask_changed_first = mask_changed & is_first_np
        mask_changed_nonfirst = mask_changed & ~is_first_np

        self.data["sum_dt_unchanged_first"] = jnp.asarray(sum_dt_first)
        self.data["sum_dt_unchanged_nonfirst"] = jnp.asarray(sum_dt_nonfirst)
        self.data["n_first_unchanged_by_group"] = jnp.asarray(
            n_first_unchanged_by_group
        )
        self.data["group_changed_nonfirst"] = jnp.asarray(
            group_np[mask_changed_nonfirst], dtype = jnp.int32,
        )
        self.data["dt_changed_nonfirst"] = jnp.asarray(
            dt_np[mask_changed_nonfirst], dtype = jnp.float32,
        )
        self.data["group_changed_first"] = jnp.asarray(
            group_np[mask_changed_first], dtype = jnp.int32,
        )
        self.data["dt_changed_first"] = jnp.asarray(
            dt_np[mask_changed_first], dtype = jnp.float32,
        )

    def _log_lambda_per_group(self, params):
        """Shared helper: per-group log-rate under either parameterisation."""
        if self._reparam == "centered":
            return params["log_lambda_0"] + params["epsilon"]
        return (
            params["log_lambda_0"]
            + jnp.exp(params["log_sigma"]) * params["epsilon_raw"]
        )

    def _logit_delta_per_group(self, params):
        """Shared helper: per-group logit_delta under either parameterisation."""
        if self._reparam == "centered":
            return params["logit_delta_0"] + params["eta"]
        return (
            params["logit_delta_0"]
            + jnp.exp(params["log_tau"]) * params["eta_raw"]
        )

    def event_rate_fun(self, params, data):
        log_lambda_g = self._log_lambda_per_group(params)
        return jnp.exp(log_lambda_g[data["group"]]) * data["dt"]

    def log_1md_fun(self, params, data):
        """Per-observation log(1-δ_{g(i)}) for fresh-mode predictions."""
        logit_delta_g = self._logit_delta_per_group(params)
        return jax.nn.log_sigmoid(-logit_delta_g[data["group"]])

    def _zie_dense_log_likelihood(self, params, data, target):
        """
        Per-row ZIE Bernoulli-on-Poisson log-likelihood (methodology §4.2
        Step D). Used when sufficient stats are disabled.
        """
        rate = self.event_rate_fun(params, data)
        is_first = data["is_first_interval"]
        logit_delta_g = self._logit_delta_per_group(params)
        log_1md_per_row = jax.nn.log_sigmoid(-logit_delta_g[data["group"]])

        log_p_std = jnp.log(-jnp.expm1(-rate))
        log_1mp_std = -rate
        log_p_zie = jnp.log(-jnp.expm1(log_1md_per_row - rate))
        log_1mp_zie = log_1md_per_row - rate

        log_p = jnp.where(is_first > 0, log_p_zie, log_p_std)
        log_1mp = jnp.where(is_first > 0, log_1mp_zie, log_1mp_std)
        return jnp.sum(target * log_p + (1.0 - target) * log_1mp)

    def _suff_stats_log_likelihood(self, params, data, target):
        """
        ZIE Bernoulli-on-Poisson log-likelihood via K-vectors + changed folds.

        ``target`` is intentionally unused — its information is captured by
        the precomputed sufficient-stats arrays built in
        ``_build_sufficient_stats``. ``log_sigmoid(-logit_delta_g)`` gives
        ``log(1−δ_g)`` with numerical stability at extreme logits.
        """
        del target
        lam_g = jnp.exp(self._log_lambda_per_group(params))
        logit_delta_g = self._logit_delta_per_group(params)
        log_1md_g = jax.nn.log_sigmoid(-logit_delta_g)  # shape (K,)

        # y=0 contribution: exponential survival on all unchanged rows, plus
        # Σ_g n_first_unchanged[g] · log(1−δ_g) for first-interval y=0 rows.
        ll_unchanged = -jnp.sum(
            lam_g * (
                data["sum_dt_unchanged_nonfirst"]
                + data["sum_dt_unchanged_first"]
            )
        )
        ll_unchanged = ll_unchanged + jnp.sum(
            data["n_first_unchanged_by_group"] * log_1md_g
        )

        # y=1 non-first: log(1 − exp(−λ_g·Δ)).
        rate_nf = (
            lam_g[data["group_changed_nonfirst"]]
            * data["dt_changed_nonfirst"]
        )
        ll_changed_nf = jnp.sum(jnp.log(-jnp.expm1(-rate_nf)))

        # y=1 first: log(1 − (1−δ_g)·exp(−λ_g·Δ)) via log(−expm1(log(1−δ_g) − rate)).
        log_1md_f = log_1md_g[data["group_changed_first"]]
        rate_f = (
            lam_g[data["group_changed_first"]] * data["dt_changed_first"]
        )
        ll_changed_f = jnp.sum(jnp.log(-jnp.expm1(log_1md_f - rate_f)))

        return ll_unchanged + ll_changed_nf + ll_changed_f

    def param_likelihood(self, params):
        var_loc, var_scale = self.metadata.get("var_prior", DEFAULT_VAR_PRIOR)
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        tau_loc, tau_scale = self.metadata.get(
            "logit_delta_var_prior", DEFAULT_LOGIT_DELTA_VAR_PRIOR
        )
        ll = stats.norm.logpdf(
            params["log_sigma"], loc = var_loc, scale = var_scale
        ).sum()
        ll = ll + stats.norm.logpdf(
            params["logit_delta_0"], loc = delta_loc, scale = delta_scale
        ).sum()
        ll = ll + stats.norm.logpdf(
            params["log_tau"], loc = tau_loc, scale = tau_scale
        ).sum()
        if self._reparam == "centered":
            ll = ll + stats.norm.logpdf(
                params["epsilon"],
                loc = 0.0,
                scale = jnp.exp(params["log_sigma"]),
            ).sum()
            ll = ll + stats.norm.logpdf(
                params["eta"],
                loc = 0.0,
                scale = jnp.exp(params["log_tau"]),
            ).sum()
        else:
            ll = ll + stats.norm.logpdf(
                params["epsilon_raw"], loc = 0.0, scale = 1.0,
            ).sum()
            ll = ll + stats.norm.logpdf(
                params["eta_raw"], loc = 0.0, scale = 1.0,
            ).sum()
        return ll

    def derive_draws(self, draws):
        """Expose natural-scale epsilon/eta and per-group logit_delta/delta."""
        out = dict(draws)
        if self._reparam != "centered":
            # draws["log_sigma"]: (n_draws,). draws["epsilon_raw"]: (n_draws, K).
            out["epsilon"] = (
                jnp.exp(draws["log_sigma"])[:, None] * draws["epsilon_raw"]
            )
            out["eta"] = (
                jnp.exp(draws["log_tau"])[:, None] * draws["eta_raw"]
            )
        # Per-group logit_delta and delta (always K-vectors per draw).
        eta_natural = out["eta"]
        out["logit_delta"] = draws["logit_delta_0"][:, None] + eta_natural
        out["delta"] = jax.nn.sigmoid(out["logit_delta"])
        return out

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
