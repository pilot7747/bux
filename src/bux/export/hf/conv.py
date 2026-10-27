"""Causal depthwise convolution with its state in transformers' cache: bux's
``CausalConv``, which ``conv`` is and other layers convolve with, and the pattern of
every layer that keeps state between decoding steps.

A conv of `size` taps and dilation r reads ``(size - 1) r`` inputs before each
position. Decoding keeps them in the layer's cache slot, a ``LinearAttentionLayer``
(``cache.update_conv_state``): a prefill convolves its own inputs, zero-padded on the
left, and leaves the last of them; a decoding step convolves the kept ones and its
own (`history`). Padded tokens of a left-padded batch are zeroed first, so that they
leave nothing in the state.

A recurrent layer keeps its state in the same slot (`previous_state`, `keep_state`):
none before its first call, a prefill, which starts from zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Site


def real_tokens(ctx: Ctx, t: int) -> torch.Tensor | None:
    """Return 1 at the last `t` tokens transformers' mask keeps and 0 at its padding,
    b rows of t; None without a 2D mask. Generate makes the masks of a compileable
    cache before the model, by layer type: a recurrent layer's is that 2D mask."""
    mask = ctx.attention_mask
    if isinstance(mask, dict):
        mask = mask.get("linear_attention", mask.get("conv"))
    if mask is None or mask.ndim != 2:
        return None
    return mask[:, -t:]


def without_padding(x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    """Return x, b t ..., with the tokens transformers' mask pads zeroed."""
    keep = real_tokens(ctx, x.shape[1])
    if keep is None:
        return x
    keep = keep.to(x.dtype)
    return x * keep.view(*keep.shape, *(1,) * (x.ndim - 2))


def previous_state(ctx: Ctx, slot: int) -> torch.Tensor | None:
    """Return the recurrent state the cache keeps at `slot`: None without a cache, or
    before the layer's first call, which starts from zero."""
    if ctx.cache is None:
        return None
    layer = ctx.cache.layers[slot]
    return (
        layer.recurrent_states[0] if layer.is_recurrent_states_initialized[0] else None
    )


def keep_state(ctx: Ctx, slot: int, state: torch.Tensor) -> None:
    """Keep `state` in the cache at `slot`, for the next call; nothing without one."""
    if ctx.cache is not None:
        ctx.cache.update_recurrent_state(state, slot)


def history(
    x: torch.Tensor, ctx: Ctx, slot: int | None, state: int, keep: int
) -> torch.Tensor:
    """Return x, b t c, after the `keep` inputs before it, b keep+t c: zeros before a
    prefill, or those the cache slot's conv state `state` kept, while decoding. It
    keeps x's last `keep` there for the next call; without a slot or a cache, none."""
    xs = F.pad(x, (0, 0, keep, 0))
    if ctx.cache is None or slot is None or not keep:
        return xs
    decoding = ctx.cache.has_previous_state(slot, state)
    full = ctx.cache.update_conv_state(
        x.transpose(1, 2), slot, state, conv_kernel_size=keep
    )
    return full.transpose(1, 2) if decoding else xs


class CausalConv(nn.Module):
    """bux's ``CausalConv``: each channel c of position t is
    ``sum_i w[i, c] x[t - (size - 1 - i) r, c]``, plus a bias, for the dilation r.

    Args:
        channels: The channels convolved, each with its own weights.
        size: The taps of each channel's kernel.
        bias: Whether to add a learned bias per channel.
        dilation: The distance between two taps, in positions.
    """

    def __init__(self, channels: int, size: int, bias: bool, dilation: int = 1):
        super().__init__()
        self.size, self.dilation = size, dilation
        self.weight = nn.Parameter(torch.zeros(size, channels))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None

    def forward(
        self, x: torch.Tensor, ctx: Ctx, slot: int | None = None, state: int = 0
    ) -> torch.Tensor:
        """Convolve x, b t c, keeping its last inputs in the cache slot `slot` (its
        conv state `state`); without a slot or a cache, as one whole sequence."""
        x = without_padding(x, ctx)
        t, r = x.shape[1], self.dilation
        xs = history(x, ctx, slot, state, (self.size - 1) * r)
        y = sum(self.weight[i] * xs[:, i * r : i * r + t] for i in range(self.size))
        assert isinstance(y, torch.Tensor)
        return y if self.bias is None else y + self.bias


class Conv(CausalConv):
    """bux's ``conv``: a causal depthwise conv over positions as a layer, which keeps
    its inputs' tail as its state."""

    bux_types = ("conv",)

    @staticmethod
    def cache_type(c: dict, ports: list) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "conv"

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__(width, c["size"], c["bias"])

    def forward(
        self, x: torch.Tensor, ctx: Ctx, slot: int | None = None, state: int = 0
    ) -> torch.Tensor:
        return super().forward(
            x, ctx, ctx.cache_slot() if slot is None else slot, state
        )
