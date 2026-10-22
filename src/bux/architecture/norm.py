"""Normalization: layernorm, rmsnorm and DyT. Norm statistics are in float32, whatever
dtype the run computes in (``precision``)."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


class Norm(nnx.Module):
    no_weight_decay = ("scale", "bias")

    def __init__(self, config: cfg.NormalizationConfig, width: int | tuple[int, ...]):
        self.kind = config.type
        self.eps = config.eps
        self.zero_centered = config.zero_centered
        self.scale_input = config.scale_input
        init = jnp.zeros if config.zero_centered else jnp.ones
        self.scale = nnx.Param(init(width)) if config.affine else None
        self.bias = nnx.Param(jnp.zeros(width)) if config.bias else None

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps | None = None
    ) -> Float[Array, "*lead d"]:
        gain = None
        if self.scale is not None:
            gain = 1 + self.scale if self.zero_centered else self.scale
        if gain is not None and self.scale_input:
            x = x * gain
        x = self.normalize(x)
        if gain is not None and not self.scale_input:
            x = x * gain
        if self.bias is not None:
            x = x + self.bias
        return x

    def normalize(self, x: Float[Array, "*lead d"]) -> Float[Array, "*lead d"]:
        """Return x over its RMS, centered first for a layernorm: the statistics in
        float32, the result in x's dtype."""
        y = x.astype(jnp.float32)
        if self.kind == "layernorm":
            y = y - y.mean(-1, keepdims=True)
        y = y * jax.lax.rsqrt(jnp.square(y).mean(-1, keepdims=True) + self.eps)
        return y.astype(x.dtype)


@register("layernorm", "rmsnorm")
def _(
    c: cfg.NormalizationConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Norm:
    return Norm(c, width)


class DyT(nnx.Module):
    no_weight_decay = ("alpha", "weight", "bias")

    def __init__(self, c: cfg.DyTConfig, width: int):
        self.alpha = nnx.Param(jnp.asarray(c.alpha_init))
        self.weight = nnx.Param(jnp.ones(width))
        self.bias = nnx.Param(jnp.zeros(width))

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps
    ) -> Float[Array, "*lead d"]:
        return jnp.tanh(self.alpha * x) * self.weight + self.bias


@register("dyt")
def _(c: cfg.DyTConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> DyT:
    return DyT(c, width)
