"""Depthwise causal convolution, as a token mixer and inside other layers."""

import jax.numpy as jnp
from flax import nnx

from bux import config as cfg
from bux.architecture.registry import Context, Taps, positions, register


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
        terms = [self.weight[i] * padded[:, i * r : i * r + t] for i in range(size)]
        if (pos := positions(taps)) is not None:
            # Tap i looks (size - 1 - i) * r tokens back, never before the document.
            back = (size - 1 - jnp.arange(size)) * r
            terms = [
                jnp.where((pos >= n)[..., None], y, 0) for n, y in zip(back, terms)
            ]
        y = sum(terms)
        return y if self.bias is None else y + self.bias


@register("conv")
def _(c: cfg.ConvConfig, width, ports, ctx: Context):
    return CausalConv(width, c.size, c.bias, ctx.rngs)
