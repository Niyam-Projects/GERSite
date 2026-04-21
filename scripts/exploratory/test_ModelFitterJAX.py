"""
Simulation test for ModelFitterJAX.

Generates data that matches the model's likelihood -- one Bernoulli trial per
observation with P(y=1) = 1 - exp(-lambda * dt) -- and checks that NUTS
recovers the true log_lambda and produces well-calibrated predictions on
held-out data.
"""

import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from jax.scipy import stats

from openpois.models.jax_core import jax_rng
from openpois.models.model_fitter import ModelFitterJAX


NUM_DRAWS = 500
TRUE_LOG_LAMBDA = -1.5  # lambda ~= 0.223 per unit time
N_TRAIN = 4_000
N_TEST = 1_000


def simulate_poi_data(key, n, true_log_lambda):
    """
    Simulate n independent change indicators with random interval lengths.

    For each observation i: dt_i ~ Uniform(0.5, 5.0), then
    y_i ~ Bernoulli(1 - exp(-exp(true_log_lambda) * dt_i)).
    """
    key_dt, key_y = jrd.split(key, 2)
    dt = jrd.uniform(key_dt, shape = (n,), minval = 0.5, maxval = 5.0)
    p = 1.0 - jnp.exp(-jnp.exp(true_log_lambda) * dt)
    y = jrd.bernoulli(key_y, p).astype(jnp.int32)
    return {"dt": dt, "y": y}


def event_rate_fun(params, data):
    return jnp.exp(params["log_lambda"]) * data["dt"]


def param_likelihood(params, loc: float = 0.0, scale: float = 3.0):
    return jnp.sum(
        stats.norm.logpdf(params["log_lambda"], loc = loc, scale = scale)
    )


if __name__ == "__main__":
    key = jax_rng()
    key_train, key_test, key_fit = jrd.split(key, 3)

    train = simulate_poi_data(key_train, N_TRAIN, TRUE_LOG_LAMBDA)
    test = simulate_poi_data(key_test, N_TEST, TRUE_LOG_LAMBDA)

    fitter = ModelFitterJAX(
        event_rate_fun = event_rate_fun,
        starting_params = {"log_lambda": jnp.array(0.0)},
        data = {"dt": train["dt"]},
        target = train["y"],
        num_draws = NUM_DRAWS,
        param_likelihood = param_likelihood,
        rng_key = key_fit,
    )
    fitter.fit()

    # Parameter recovery.
    params_table = fitter.get_parameter_table()
    print(f"Parameter recovery (true log_lambda = {TRUE_LOG_LAMBDA:+.3f}):")
    print(params_table.to_string(index = False))

    row = params_table.loc[params_table["parameter"] == "log_lambda"].iloc[0]
    post_mean, post_lo, post_hi = row["mean"], row["lower"], row["upper"]
    assert post_lo <= TRUE_LOG_LAMBDA <= post_hi, (
        f"True log_lambda {TRUE_LOG_LAMBDA:+.3f} outside 95% UI "
        f"[{post_lo:+.3f}, {post_hi:+.3f}]"
    )
    assert abs(post_mean - TRUE_LOG_LAMBDA) < 0.1, (
        f"Posterior mean {post_mean:+.3f} far from truth "
        f"{TRUE_LOG_LAMBDA:+.3f}"
    )

    # Held-out prediction check against the known true probabilities.
    test_preds = fitter.predict(data = {"dt": test["dt"]})
    true_p = np.asarray(1.0 - jnp.exp(-jnp.exp(TRUE_LOG_LAMBDA) * test["dt"]))
    mae_p = float(np.mean(np.abs(test_preds["p_mean"].values - true_p)))
    coverage = float(np.mean(
        (test_preds["p_lower"].values <= true_p)
        & (true_p <= test_preds["p_upper"].values)
    ))
    print("\nHeld-out predictions vs. true P(change):")
    print(f"  mean |p_mean - true_p| = {mae_p:.4f}")
    print(f"  95% UI coverage of true p = {coverage:.3f}")
