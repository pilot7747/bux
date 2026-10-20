"""Elementwise and plumbing layers: scale, linear, act and dropout."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.functional import activation
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


class Scale(nnx.Module):
    def __init__(self, c: cfg.ScaleConfig, width: int):
        self.value = nnx.Param(jnp.full(width if c.per_channel else (), c.init))

    def __call__(self, x: Float[Array, "*shape"], taps: Taps) -> Float[Array, "*shape"]:
        return x * self.value


@register("scale")
def _(
    c: cfg.ScaleConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Scale:
    return Scale(c, width)


class Linear(nnx.Module):
    def __init__(self, c: cfg.LinearConfig, width: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(width, c.features, use_bias=c.bias, rngs=rngs)

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps
    ) -> Float[Array, "*lead f"]:
        return self.proj(x)


@register("linear")
def _(
    c: cfg.LinearConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Linear:
    return Linear(c, width, ctx.rngs)


class Activation(nnx.Module):
    def __init__(self, fn: str):
        self.fn = fn

    def __call__(self, x: Float[Array, "*shape"], taps: Taps) -> Float[Array, "*shape"]:
        return activation(self.fn, x)


@register("act")
def _(
    c: cfg.ActivationConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Activation:
    return Activation(c.fn)


class Dropout(nnx.Module):
    def __init__(self, rate: float, rngs: nnx.Rngs):
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x: Float[Array, "*shape"], taps: Taps) -> Float[Array, "*shape"]:
        return self.dropout(x)


@register("dropout")
def _(
    c: cfg.DropoutConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Dropout:
    return Dropout(c.rate, ctx.rngs)
