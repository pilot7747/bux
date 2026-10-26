"""Causal depthwise convolution with its state in transformers' cache: bux's
``CausalConv``, which ``conv`` is and other layers convolve with, and the pattern of
every layer that keeps state between decoding steps.

A conv of `size` taps and dilation r reads ``(size - 1) r`` inputs before each
position. Decoding keeps them in the layer's cache slot, a ``LinearAttentionLayer``
(``cache.update_conv_state``): a prefill convolves its own inputs, zero-padded on the
left, and leaves the last of them; a decoding step convolves the kept ones and its
own. Padded tokens of a left-padded batch are zeroed first, so that they leave
nothing in the state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Site


def without_padding(x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    """Return x, b t ..., with the tokens transformers' mask pads zeroed."""
    mask = ctx.attention_mask
    if mask is None or mask.ndim != 2:
        return x
    keep = mask[:, -x.shape[1] :].to(x.dtype)
    return x * keep.view(*keep.shape, *(1,) * (x.ndim - 2))


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
        keep = (self.size - 1) * r
        xs = F.pad(x, (0, 0, keep, 0))
        if ctx.cache is not None and slot is not None and keep:
            decoding = ctx.cache.has_previous_state(slot, state)
            full = ctx.cache.update_conv_state(
                x.transpose(1, 2), slot, state, conv_kernel_size=keep
            )
            if decoding:
                xs = full.transpose(1, 2)
        y = sum(self.weight[i] * xs[:, i * r : i * r + t] for i in range(self.size))
        assert isinstance(y, torch.Tensor)
        return y if self.bias is None else y + self.bias


class Conv(CausalConv):
    """bux's ``conv``: a causal depthwise conv over positions as a layer, which keeps
    its inputs' tail as its state."""

    bux_types = ("conv",)

    @staticmethod
    def cache_type(c: dict) -> str:
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
