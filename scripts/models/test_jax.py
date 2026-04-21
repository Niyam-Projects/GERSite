"""
Test JAX core functions.
"""

import jax.numpy as jnp
import jax.random as jrd
from jax.scipy import stats

from functools import partial
from openpois.models.jax_core import (
    jax_rng,
    nuts_sample,
    generate_predictive_draws,
)

NUM_DRAWS = 250


def simulate_regression(n=2_000, p=2, n_new=4, seed=145_777):
    key = jrd.PRNGKey(seed)

    def simulate_covariates(key, n):
        x = jrd.normal(key, shape=(n, p))
        x = x.at[:, 1].set(x[:, 1] ** 2)
        return x

    # Split the key for reproducible results
    key, key_alpha, key_beta, key_log_sigma, key_x, key_y, key_xnew = jrd.split(key, 7)

    parameters = {
        'alpha': jrd.normal(key_alpha, ()) * 5.0,
        'beta': jrd.normal(key_beta, (p,)) * 2.5,
        'log_sigma': jrd.normal(key_log_sigma, ()) * 0.25,
    }
    x = simulate_covariates(key_x, n)
    mu = parameters['alpha'] + x @ parameters['beta']
    y = mu + jrd.normal(key_y, (n,)) * jnp.exp(parameters['log_sigma'])
    x_new = simulate_covariates(key_xnew, n_new)

    data = {
        'N': n,
        'P': p,
        'N_new': n_new,
        'x': x,         # jnp.ndarray
        'y': y,         # jnp.ndarray
        'x_new': x_new, # jnp.ndarray
    }
    return parameters, data


def log_posterior(params, data):
    lp = 0.0
    lp += jnp.sum(stats.norm.logpdf(params['alpha'], loc = 0.0, scale = 5.0))
    lp += jnp.sum(stats.norm.logpdf(params['beta'], loc = 0.0, scale = 2.5))
    lp += jnp.sum(stats.norm.logpdf(params['log_sigma'], loc = jnp.log(0.2), scale = 0.25))
    mu = params['alpha'] + data['x'] @ params['beta']
    lp += jnp.sum(stats.norm.logpdf(data['y'], loc = mu, scale = jnp.exp(params['log_sigma'])))
    return lp

def predict(key, params, data, add_sigma: bool = True):
    mu = params['alpha'] + data['x'] @ params['beta']
    if add_sigma:
        mu += jrd.normal(key, shape = (data['N'],)) * jnp.exp(params['log_sigma'])
    return mu


if __name__ == "__main__":
    key = jax_rng()
    # Simulate some data
    true_params, data = simulate_regression(seed = key[1])
    # Starting parameters
    initial_params = {
        'alpha': jnp.array(0.0),
        'beta': jnp.zeros(shape = data['P']),
        'log_sigma': jnp.array(0.0),
    }
    # Log posterior with data
    log_posterior_with_data = partial(log_posterior, data = data)
    # Sample
    param_draws = nuts_sample(
        log_density = log_posterior_with_data,
        init_position = initial_params,
        num_draws = NUM_DRAWS,
        key = key
    )
    # Predictive draws
    predict_with_data = partial(predict, data = data, add_sigma = True)
    predictive_draws = generate_predictive_draws(
        posterior_predictive = predict_with_data,
        param_draws = param_draws,
        num_draws = NUM_DRAWS,
        key = key
    )
