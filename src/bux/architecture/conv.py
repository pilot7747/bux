"""Depthwise causal convolution, as a token mixer and inside other layers."""

import jax.numpy as jnp
from flax import nnx

from bux import config as cfg
from bux.architecture.registry import Context, Taps, register


class CausalConv(nnx.Module):
    """Depthwise causal convolution over positions of (batch, seq, channels)."""

    def __init__(
        self,
        channels: int,
        size: int,
        bias: bool,
        rngs: nnx.Rngs,
        *,
        dilation: int = 1,
        zero: bool = False,
    ):
        init = nnx.initializers.zeros if zero else nnx.initializers.lecun_normal()
        self.weight = nnx.Param(init(rngs.params(), (size, channels)))
        self.bias = nnx.Param(jnp.zeros(channels)) if bias else None
        self.dilation = dilation

    def __call__(self, x, taps: Taps | None = None):
        size, t, r = self.weight.shape[0], x.shape[1], self.dilation
        padded = jnp.pad(x, ((0, 0), ((size - 1) * r, 0), (0, 0)))
        y = sum(self.weight[i] * padded[:, i * r : i * r + t] for i in range(size))
        return y if self.bias is None else y + self.bias


@register("conv")
def _(c: cfg.ConvConfig, width, ports, ctx: Context):
    return CausalConv(width, c.size, c.bias, ctx.rngs)
