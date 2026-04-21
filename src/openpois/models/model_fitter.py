#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Fit POI-change models in JAX.

The likelihood is Bernoulli on a binary change indicator, with
P(change) = 1 - exp(-rate) derived from a Poisson event rate.
"""

import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pandas as pd

from openpois.models.jax_core import jax_rng, nuts_sample


class ModelFitter:
    """
    Fitter for POI change-rate models using BlackJAX NUTS.
    """

    EPSILON = 1e-6

    def __init__(
        self,
        event_rate_fun: callable,
        starting_params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
        target: jnp.ndarray,
        num_draws: int = 1_000,
        param_likelihood: callable | None = None,
        rng_key: jrd.KeyArray | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            event_rate_fun: Callable ``(params, data) -> rates`` returning a
                Poisson event rate per observation. ``data`` should bundle any
                covariates plus time bounds (e.g. ``t1``, ``t2``) the function
                needs.
            starting_params: Initial position for NUTS, as a pytree (dict) of
                arrays matching the signature expected by ``event_rate_fun``.
            data: Dict of arrays consumed by ``event_rate_fun``.
            target: Binary change indicator per observation.
            num_draws: Number of warmup steps and posterior draws.
            param_likelihood: Optional ``(params) -> log_prior`` contribution
                added to the Bernoulli log-likelihood.
            rng_key: JAX PRNG key. Defaults to a fresh key from ``jax_rng()``.
            verbose: Reserved for future use.
        """
        self.event_rate_fun = event_rate_fun
        self.starting_params = starting_params
        self.data = data
        self.target = target
        self.num_draws = num_draws
        self.param_likelihood = param_likelihood
        self.rng_key = rng_key if rng_key is not None else jax_rng()
        self.verbose = verbose
        self.param_draws = None
        self.model_finished = False

    def calculate_probs(
        self,
        params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Map parameters + data to change probabilities in (EPSILON, 1 - EPSILON).
        """
        change_rates = self.event_rate_fun(params, data)
        probs = 1.0 - jnp.exp(-change_rates)
        return jnp.clip(probs, self.EPSILON, 1.0 - self.EPSILON)

    def calculate_lp(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Log-posterior density at ``params`` given ``self.data``/``self.target``.

        Bernoulli likelihood on the binary change indicator plus the optional
        ``param_likelihood`` log-prior contribution.
        """
        probs = self.calculate_probs(params, self.data)
        lp = jnp.sum(
            self.target * jnp.log(probs)
            + (1.0 - self.target) * jnp.log(1.0 - probs)
        )
        if self.param_likelihood is not None:
            lp = lp + self.param_likelihood(params)
        return lp

    def fit(self):
        """
        Draw posterior samples via BlackJAX NUTS with window adaptation.

        Stores draws on ``self.param_draws`` as a pytree matching
        ``starting_params`` where each leaf has a leading draw axis.
        """
        self.param_draws = nuts_sample(
            log_density = self.calculate_lp,
            init_position = self.starting_params,
            num_draws = self.num_draws,
            key = self.rng_key,
        )
        self.model_finished = True

    def get_parameter_draws(self) -> dict[str, jnp.ndarray]:
        """
        Return the posterior parameter draws as a pytree.

        Raises:
            ValueError: If ``fit()`` has not been run yet.
        """
        if self.param_draws is None:
            raise ValueError("Run fit() first")
        return self.param_draws

    def get_parameter_table(self, ui_width: float = 0.95) -> pd.DataFrame:
        """
        Summarize posterior draws as one row per scalar parameter.

        Vector/matrix parameters are unrolled with ``name[i]`` / ``name[i,j]``
        labels.

        Args:
            ui_width: Width of the uncertainty interval, in (0, 1).

        Returns:
            DataFrame with columns ``parameter``, ``mean``, ``lower``, ``upper``.
        """
        if self.param_draws is None:
            raise ValueError("Run fit() first")
        if not (0.0 < ui_width < 1.0):
            raise ValueError("ui_width must be between 0 and 1")
        lb = (1.0 - ui_width) / 2.0
        ub = 1.0 - lb

        rows = []
        for name, draws in self.param_draws.items():
            draws_np = np.asarray(draws)
            n_draws = draws_np.shape[0]
            flat = draws_np.reshape(n_draws, -1)
            param_shape = draws_np.shape[1:]
            for i in range(flat.shape[1]):
                if len(param_shape) == 0:
                    label = name
                else:
                    idx = np.unravel_index(i, param_shape)
                    label = f"{name}[{','.join(str(k) for k in idx)}]"
                col = flat[:, i]
                rows.append({
                    "parameter": label,
                    "mean": float(np.mean(col)),
                    "lower": float(np.quantile(col, lb, method = "linear")),
                    "upper": float(np.quantile(col, ub, method = "linear")),
                })
        return pd.DataFrame(rows)

    def predict(
        self,
        data: dict[str, jnp.ndarray] | None = None,
        ui_width: float = 0.95,
    ) -> pd.DataFrame:
        """
        Posterior-predictive change probabilities.

        Vmaps ``calculate_probs`` over the stacked posterior draws in
        ``self.param_draws``.

        Args:
            data: Dict of arrays with the same keys as ``self.data``. Defaults
                to ``self.data``.
            ui_width: Width of the uncertainty interval, in (0, 1).

        Returns:
            DataFrame with one row per observation and columns
            ``p_mean``, ``p_lower``, ``p_upper``.
        """
        if self.param_draws is None:
            raise ValueError("Run fit() first")
        if not (0.0 < ui_width < 1.0):
            raise ValueError("ui_width must be between 0 and 1")
        if data is None:
            data = self.data

        def probs_for_draw(params):
            return self.calculate_probs(params, data)

        all_probs = jax.vmap(probs_for_draw)(self.param_draws)
        lb = (1.0 - ui_width) / 2.0
        ub = 1.0 - lb
        return pd.DataFrame({
            "p_mean": np.asarray(jnp.mean(all_probs, axis = 0)),
            "p_lower": np.asarray(
                jnp.quantile(all_probs, lb, axis = 0, method = "linear")
            ),
            "p_upper": np.asarray(
                jnp.quantile(all_probs, ub, axis = 0, method = "linear")
            ),
        })
