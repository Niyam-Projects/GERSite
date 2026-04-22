#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
MCMC diagnostics for multi-chain posterior draws.

Thin wrappers over ``blackjax.diagnostics`` that operate on the
``(num_chains, num_draws, *param_shape)`` pytrees produced by
``nuts_sample_multichain``.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from blackjax.diagnostics import (
    effective_sample_size,
    potential_scale_reduction,
)


_HAS_ARVIZ = importlib.util.find_spec("arviz") is not None


def _rhat_one(leaf: jnp.ndarray) -> jnp.ndarray:
    """R-hat per scalar element of a (chains, draws, *param_shape) leaf."""
    # blackjax expects (chains, draws) for a scalar and reduces over axes.
    # For vector/matrix parameters, flatten trailing dims and vmap.
    if leaf.ndim <= 2:
        return potential_scale_reduction(leaf)
    flat = leaf.reshape(leaf.shape[0], leaf.shape[1], -1)
    return jax.vmap(
        potential_scale_reduction, in_axes = -1, out_axes = -1,
    )(flat).reshape(leaf.shape[2:])


def _ess_one(leaf: jnp.ndarray) -> jnp.ndarray:
    """Bulk ESS per scalar element, operating on (chains, draws, ...)."""
    if leaf.ndim <= 2:
        return effective_sample_size(leaf)
    flat = leaf.reshape(leaf.shape[0], leaf.shape[1], -1)
    return jax.vmap(
        effective_sample_size, in_axes = -1, out_axes = -1,
    )(flat).reshape(leaf.shape[2:])


def summarize_chain_draws(
    chain_draws: dict[str, jnp.ndarray],
) -> pd.DataFrame:
    """
    Compute R-hat and bulk-ESS for every scalar element of ``chain_draws``.

    Args:
        chain_draws: Pytree where each leaf has shape
            ``(num_chains, num_draws, *param_shape)``.

    Returns:
        DataFrame with columns ``parameter``, ``rhat``, ``ess_bulk``. One row
        per scalar element (vectors/matrices are unrolled with
        ``name[i]``/``name[i,j]`` labels to match ``get_parameter_table``).
    """
    rows = []
    for name, leaf in chain_draws.items():
        leaf_arr = jnp.asarray(leaf)
        if leaf_arr.shape[0] < 2:
            # R-hat requires >= 2 chains. ESS still computes.
            rhat = np.full(leaf_arr.shape[2:] or (1,), np.nan, dtype = float)
        else:
            rhat = np.asarray(_rhat_one(leaf_arr))
        ess = np.asarray(_ess_one(leaf_arr))
        rhat_flat = np.atleast_1d(rhat).reshape(-1)
        ess_flat = np.atleast_1d(ess).reshape(-1)
        param_shape = leaf_arr.shape[2:]
        for i, (r, e) in enumerate(zip(rhat_flat, ess_flat)):
            if len(param_shape) == 0:
                label = name
            else:
                idx = np.unravel_index(i, param_shape)
                label = f"{name}[{','.join(str(k) for k in idx)}]"
            rows.append({
                "parameter": label,
                "rhat": float(r),
                "ess_bulk": float(e),
            })
    return pd.DataFrame(rows)


def to_inference_data(
    chain_draws: dict[str, jnp.ndarray],
    sampler_info = None,
):
    """
    Build an ``arviz.InferenceData`` from chain-shaped NUTS output.

    Populates ``posterior`` (one variable per pytree leaf) and, if
    ``sampler_info`` is provided, ``sample_stats`` with
    ``acceptance_rate``, ``diverging``, ``tree_depth`` (from
    ``num_integration_steps``), and ``energy``. ArviZ is an optional
    dependency; this raises a clear error if it's not installed.

    Args:
        chain_draws: Pytree with ``(num_chains, num_draws, *param_shape)``
            leaves.
        sampler_info: Optional NUTSInfo stacked across ``(chains, draws)``.

    Returns:
        ``arviz.InferenceData``.
    """
    if not _HAS_ARVIZ:
        raise ImportError(
            "arviz is required for to_inference_data; install with "
            "`pip install arviz`."
        )
    import arviz as az

    posterior = {
        name: np.asarray(leaf) for name, leaf in chain_draws.items()
    }
    data = {"posterior": posterior}
    if sampler_info is not None:
        data["sample_stats"] = {
            "acceptance_rate": np.asarray(sampler_info.acceptance_rate),
            "diverging": np.asarray(sampler_info.is_divergent),
            "tree_depth": np.asarray(sampler_info.num_integration_steps),
            "energy": np.asarray(sampler_info.energy),
        }
    return az.from_dict(data)


def flatten_chains(draws: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """
    Reshape ``(C, T, *S)`` leaves to ``(C*T, *S)``.

    Used so consumers that expect one leading draw axis don't have to know
    whether the sampler ran 1 or N chains.
    """
    def _flat(x):
        x = jnp.asarray(x)
        if x.ndim < 2:
            return x
        c, t = x.shape[0], x.shape[1]
        return x.reshape((c * t,) + x.shape[2:])
    return jax.tree_util.tree_map(_flat, draws)
