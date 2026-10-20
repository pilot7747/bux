"""Depthwise causal convolution, as a token mixer and inside other layers."""

import functools
import operator

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir
from bux.parallel import Name


class CausalConv(nnx.Module):
    """Depthwise causal convolution over positions, its channels named `channel`
    (``bux.parallel``)."""

    def __init__(
        self,
        channels: int,
        size: int,
        bias: bool,
        rngs: nnx.Rngs,
        *,
        dilation: int = 1,
        zero: bool = False,
        channel: Name = None,
    ):
        init = nnx.initializers.zeros if zero else nnx.initializers.lecun_normal()
        weight = init(rngs.params(), (size, channels))
        self.weight = nnx.Param(weight, names=(None, channel))
        self.bias = nnx.Param(jnp.zeros(channels), names=(channel,)) if bias else None
        self.dilation = dilation

    def __call__(
        self, x: Float[Array, "b t c"], taps: Taps | None = None
    ) -> Float[Array, "b t c"]:
        size, t, r = self.weight.shape[0], x.shape[1], self.dilation
        padded = jnp.pad(x, ((0, 0), ((size - 1) * r, 0), (0, 0)))
        terms = [self.weight[i] * padded[:, i * r : i * r + t] for i in range(size)]
        if (docs := segments(taps)) is not None:
            # Tap i looks (size - 1 - i) * r tokens back, never before the document.
            back = (size - 1 - jnp.arange(size)) * r
            terms = [
                jnp.where((docs.within >= n)[..., None], y, 0)
                for n, y in zip(back, terms)
            ]
        y = functools.reduce(operator.add, terms)
        return y if self.bias is None else y + self.bias


@register("conv")
def _(
    c: cfg.ConvConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> CausalConv:
    return CausalConv(width, c.size, c.bias, ctx.rngs)
