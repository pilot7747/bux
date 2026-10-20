"""Normalization: layernorm, rmsnorm and DyT."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.registry import Context, Taps, register
from bux.lang import ir


class Norm(nnx.Module):
    def __init__(self, c: cfg.NormalizationConfig, width: int | tuple[int, ...]):
        self.kind = c.type
        self.eps = c.eps
        self.zero_centered = c.zero_centered
        self.scale_input = c.scale_input
        init = jnp.zeros if c.zero_centered else jnp.ones
        self.scale = nnx.Param(init(width)) if c.affine else None
        self.bias = nnx.Param(jnp.zeros(width)) if c.bias else None

    def __call__(
        self, x: Float[Array, "*lead d"], taps: Taps | None = None
    ) -> Float[Array, "*lead d"]:
        gain = None
        if self.scale is not None:
            gain = 1 + self.scale if self.zero_centered else self.scale
        if gain is not None and self.scale_input:
            x = x * gain
        if self.kind == "layernorm":
            x = x - x.mean(-1, keepdims=True)
        x = x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)
        if gain is not None and not self.scale_input:
            x = x * gain
        if self.bias is not None:
            x = x + self.bias
        return x


@register("layernorm", "rmsnorm")
def _(
    c: cfg.NormalizationConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Norm:
    return Norm(c, width)


class DyT(nnx.Module):
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
