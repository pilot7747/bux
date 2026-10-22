"""Stateless functions that several layers share."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def activation(name: str, x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    match name:
        case "relu":
            return jax.nn.relu(x)
        case "relu2":
            return jnp.square(jax.nn.relu(x))
        case "gelu":
            return jax.nn.gelu(x, approximate=False)
        case "gelu_tanh":
            return jax.nn.gelu(x, approximate=True)
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


def softcap(x: Float[Array, "*shape"], cap: float | None) -> Float[Array, "*shape"]:
    return x if cap is None else cap * jnp.tanh(x / cap)
