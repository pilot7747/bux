"""Stateless functions that several layers share."""

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from bux.parallel.context import softcap


def activation(name: str, x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Return the activation `name` of every entry of x: one of
    `bux.config.Activation`, ``situ`` being ``4 * tanh(x / 4) * sigmoid(x)``.

    Raises:
        ValueError: `name` is no activation.
    """
    match name:
        case "relu":
            return jax.nn.relu(x)
        case "relu2":
            return jnp.square(jax.nn.relu(x))
        case "gelu":
            return jax.nn.gelu(x, approximate=False)
        case "gelu_tanh":
            return jax.nn.gelu(x, approximate=True)
        case "quick_gelu":
            return x * jax.nn.sigmoid(1.702 * x)
        case "swish":
            return jax.nn.swish(x)
        case "sigmoid":
            return jax.nn.sigmoid(x)
        case "tanh":
            return jnp.tanh(x)
        case "softplus":
            return jax.nn.softplus(x)
        case "situ":
            return softcap(x, 4.0) * jax.nn.sigmoid(x)
    raise ValueError(name)


def time_step_bias(
    key: PRNGKeyArray, n: int, low: float = 1e-3, high: float = 0.1
) -> Float[Array, " n"]:
    """Return softplus^-1 of n time steps log-uniform in [low, high], floored at 1e-4:
    how Mamba starts its time steps' bias, and fla the dt_bias of Gated DeltaNet and
    KDA."""
    log_dt = jax.random.uniform(key, (n,), minval=math.log(low), maxval=math.log(high))
    dt = jnp.maximum(jnp.exp(log_dt), 1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))
