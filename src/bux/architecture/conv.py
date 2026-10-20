"""Depthwise causal convolution, as a token mixer and inside other layers."""

import functools
import itertools
import operator
from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir


def _convolve(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    bias: Float[Array, " c"] | None,
    dilation: int,
    taps: Taps | None,
) -> Float[Array, "b t c"]:
    size, t, r = weight.shape[0], x.shape[1], dilation
    padded = jnp.pad(x, ((0, 0), ((size - 1) * r, 0), (0, 0)))
    terms = [weight[i] * padded[:, i * r : i * r + t] for i in range(size)]
    if (docs := segments(taps)) is not None:
        # Tap i looks (size - 1 - i) * r tokens back, never before the document.
        back = (size - 1 - jnp.arange(size)) * r
        terms = [
            jnp.where((docs.within >= n)[..., None], y, 0) for n, y in zip(back, terms)
        ]
    y = functools.reduce(operator.add, terms)
    return y if bias is None else y + bias


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
        channel: parallel.Name = None,
    ):
        init = nnx.initializers.zeros if zero else nnx.initializers.lecun_normal()
        weight = init(rngs.params(), (size, channels))
        self.weight = nnx.Param(weight, names=(None, channel))
        self.bias = nnx.Param(jnp.zeros(channels), names=(channel,)) if bias else None
        self.dilation = dilation

    def __call__(
        self, x: Float[Array, "b t c"], taps: Taps | None = None
    ) -> Float[Array, "b t c"]:
        bias = None if self.bias is None else self.bias[...]
        return _convolve(x, self.weight[...], bias, self.dilation, taps)

    def side_by_side(
        self, xs: Sequence[Float[Array, "b t _"]], taps: Taps | None = None
    ) -> list[Float[Array, "b t _"]]:
        """The conv of inputs whose channels sit side by side in its own, as of their
        concatenation, but each on its own: q, k and v split over their own heads,
        which concatenating them would move. The parts of its weights do not split on
        shard boundaries, so it takes them whole."""
        cuts = list(itertools.accumulate(x.shape[-1] for x in xs))[:-1]
        weights = jnp.split(parallel.shard(self.weight[...], None, None), cuts, axis=1)
        biases = (
            [None] * len(xs)
            if self.bias is None
            else jnp.split(parallel.shard(self.bias[...], None), cuts)
        )
        return [
            _convolve(x, w, b, self.dilation, taps)
            for x, w, b in zip(xs, weights, biases)
        ]


@register("conv")
def _(
    c: cfg.ConvConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> CausalConv:
    return CausalConv(width, c.size, c.bias, ctx.rngs, channel="embed")
