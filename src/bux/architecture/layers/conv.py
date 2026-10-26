"""Depthwise causal convolution, as a token mixer and inside other layers."""

import functools
import itertools
import operator
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import kernels, parallel
from bux.architecture.layers import pallas_conv
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir

F32 = jnp.float32


def _convolve(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    bias: Float[Array, " c"] | None,
    dilation: int,
    taps: Taps | None,
    *,
    channels_last: bool = False,
) -> Float[Array, "b t c"]:
    """Return the convolution of x, by XLA. On a GPU its backward pass is bux's own:
    by the kernel where `channels_last` says XLA lays x out so, and it runs, else by
    XLA in a form it runs faster than autodiff's over documents, in any layout."""
    docs = segments(taps)
    within = None if docs is None else docs.within
    by_kernel = channels_last and pallas_conv.runs(x.shape)
    # Without documents' masks, autodiff's backward takes at most a tenth more time
    # than XLA's form, and less where the channels come last.
    if by_kernel or (kernels.gpu() and within is not None):
        return _with_own_backward(x, weight, bias, dilation, within, by_kernel)
    return _shifted_sum(x, weight, bias, dilation, within)


def _shifted_sum(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    bias: Float[Array, " c"] | None,
    dilation: int,
    within: Int[Array, "b t"] | None,
) -> Float[Array, "b t c"]:
    """Return the convolution of x by XLA: a sum of its shifted copies."""
    size, t, r = weight.shape[0], x.shape[1], dilation
    padded = jnp.pad(x, ((0, 0), ((size - 1) * r, 0), (0, 0)))
    terms = [weight[i] * padded[:, i * r : i * r + t] for i in range(size)]
    if within is not None:
        # Tap i looks (size - 1 - i) * r tokens back, never before the document.
        back = (size - 1 - jnp.arange(size)) * r
        terms = [jnp.where((within >= n)[..., None], y, 0) for n, y in zip(back, terms)]
    y = functools.reduce(operator.add, terms)
    return y if bias is None else y + bias


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 5))
def _with_own_backward(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    bias: Float[Array, " c"] | None,
    dilation: int,
    within: Int[Array, "b t"] | None,
    by_kernel: bool,
) -> Float[Array, "b t c"]:
    """Return `_shifted_sum`, whose gradients `_backward` computes."""
    return _shifted_sum(x, weight, bias, dilation, within)


def _forward(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    bias: Float[Array, " c"] | None,
    dilation: int,
    within: Int[Array, "b t"] | None,
    by_kernel: bool,
) -> tuple[Float[Array, "b t c"], tuple[Any, ...]]:
    """Return the convolution and what its backward pass reads."""
    return _shifted_sum(x, weight, bias, dilation, within), (x, weight, bias, within)


def _backward(
    dilation: int, by_kernel: bool, saved: tuple[Any, ...], dy: Float[Array, "b t c"]
) -> tuple[Any, ...]:
    """Return the gradients of x, the weight and the bias: by the kernel, or XLA."""
    x, weight, bias, within = saved
    by = pallas_conv.backward if by_kernel else _backward_by_xla
    dx, dw, db = by(x, weight, dy, within, dilation)
    return dx, dw, None if bias is None else db.astype(bias.dtype), None


_with_own_backward.defvjp(_forward, _backward)


def _backward_by_xla(
    x: Float[Array, "b t c"],
    weight: Float[Array, "size c"],
    dy: Float[Array, "b t c"],
    within: Int[Array, "b t"] | None,
    dilation: int,
) -> pallas_conv.Gradients:
    """Return the gradients of x, the weight and the bias from dy, by XLA, as the
    kernel computes them: every tap's from dy shifted ahead by as many positions as
    the tap looks back. Autodiff takes the weights' from x shifted back instead,
    whose sums over positions XLA runs at a quarter of this one's speed where the
    channels come first (Mamba-2's conv)."""
    x32 = x.astype(F32)
    dx, dw = jnp.zeros(x.shape, F32), []
    for i, g in enumerate(_dy_ahead(dy, within, weight.shape[0], dilation)):
        dw.append(jnp.sum(g * x32, axis=(0, 1)))
        dx += weight[i].astype(F32) * g
    db = jnp.sum(dy.astype(F32), axis=(0, 1))
    weights = jnp.stack(dw).astype(weight.dtype)
    return dx.astype(x.dtype), weights, db.astype(weight.dtype)


def _dy_ahead(
    dy: Float[Array, "b t c"], within: Int[Array, "b t"] | None, size: int, r: int
) -> list[Float[Array, "b t c"]]:
    """Return for each tap i dy ``(size - 1 - i) r`` positions ahead, in float32: 0
    past the row's end, and where that position's document starts later."""
    t, pad = dy.shape[1], (size - 1) * r
    padded = jnp.pad(dy, ((0, 0), (0, pad), (0, 0)))
    seen = None if within is None else jnp.pad(within, ((0, 0), (0, pad)))
    ahead = []
    for i in range(size):
        n = (size - 1 - i) * r
        g = padded[:, n : n + t]
        if seen is not None:
            g = jnp.where((seen[:, n : n + t] >= n)[..., None], g, 0)
        ahead.append(g.astype(F32))
    return ahead


class CausalConv(nnx.Module):
    """Depthwise causal convolution over positions, ``conv``, and the short conv
    inside other layers.

    Each channel c of position t is ``sum_i w[i, c] x[t - (size - 1 - i) r, c]``,
    plus a bias, for the dilation r: it sees itself and ``size - 1`` positions before
    it, never one before its document's start. Its channels are named `channel`
    (`bux.parallel`), which a mesh may split. On a GPU its backward pass is bux's
    own: `side_by_side`'s by a Pallas kernel
    (`~bux.architecture.layers.pallas_conv`), the others' by XLA in a form it runs
    faster than autodiff's.

    Args:
        channels: The channels convolved, each with weights of its own.
        size: The taps of each channel's kernel.
        bias: Whether to add a learned bias per channel.
        rngs: The random streams the weights start from.
        dilation: The distance between two taps, in positions.
        zero: Start the weights at 0, so that the conv starts as nothing.
        channel: The name of the channels' dimension (`bux.parallel`), or None to
            keep them whole.
    """

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
        init = nnx.initializers.zeros if zero else parallel.default_init()
        weight = init(rngs.params(), (size, channels))
        self.weight = nnx.Param(weight, names=(None, channel))
        self.bias = nnx.Param(jnp.zeros(channels), names=(channel,)) if bias else None
        self.dilation = dilation

    def __call__(
        self, x: Float[Array, "b t c"], taps: Taps | None = None
    ) -> Float[Array, "b t c"]:
        """Return the convolution of x, ``b t c``: b rows of t positions, c channels.

        Args:
            x: The channels at every position.
            taps: The forward pass's taps, for the documents in the rows; None for
                rows of one document each.
        """
        bias = None if self.bias is None else self.bias[...]
        return _convolve(x, self.weight[...], bias, self.dilation, taps)

    def side_by_side(
        self, xs: Sequence[Float[Array, "b t _"]], taps: Taps | None = None
    ) -> list[Float[Array, "b t _"]]:
        """Convolve inputs whose channels sit side by side in the conv's, as their
        concatenation would, but each on its own: q, k and v split over their own
        heads, which concatenating them would move. The parts of its weights do not
        split on shard boundaries, so it takes them whole."""
        cuts = list(itertools.accumulate(x.shape[-1] for x in xs))[:-1]
        weights = jnp.split(parallel.shard(self.weight[...], None, None), cuts, axis=1)
        biases = (
            [None] * len(xs)
            if self.bias is None
            else jnp.split(parallel.shard(self.bias[...], None), cuts)
        )
        # q, k and v go on in heads, laid out channels last, as the kernel takes x.
        return [
            _convolve(x, w, b, self.dilation, taps, channels_last=True)
            for x, w, b in zip(xs, weights, biases)
        ]


@register("conv")
def _(
    c: cfg.ConvConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> CausalConv:
    return CausalConv(width, c.size, c.bias, ctx.rngs, channel="embed")
