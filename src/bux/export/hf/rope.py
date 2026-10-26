"""Rotary position embeddings, exported: bux's `bux.architecture.attention.rope`, with
position interpolation, YaRN and Llama 3's scaling, in float32."""

from __future__ import annotations

import math

import torch
from torch import nn


def rotated_dims(head_dim: int, rope: dict | None) -> int:
    """How many of a head's `head_dim` dims `rope` rotates, in whole pairs: 0 for no
    RoPE."""
    return 0 if rope is None else int(head_dim * rope["fraction"]) // 2 * 2


def attention_factor(yarn: dict) -> float:
    """YaRN's attention factor: as given, else 0.1 ln(factor) + 1."""
    if yarn["attention_factor"] is not None:
        return yarn["attention_factor"]
    return 0.1 * math.log(yarn["factor"]) + 1 if yarn["factor"] > 1 else 1.0


def logit_gain(rope: dict | None) -> float:
    """YaRN's ``attention_factor ** 2`` where it goes on the softmax scale, else 1."""
    s = None if rope is None else rope["scaling"]
    if s is not None and s["type"] == "yarn" and s["attention_factor_on"] == "softmax":
        return attention_factor(s) ** 2
    return 1.0


def _yarn(rope: dict, inv: torch.Tensor, dims: int) -> tuple[torch.Tensor, float]:
    """YaRN's frequencies, as bux's (and transformers') ramp between the pairs that
    turn `beta_fast` and `beta_slow` times over the original context."""
    s = rope["scaling"]

    def pair(turns: float) -> float:
        span = math.log(s["original_max_len"] / (turns * 2 * math.pi))
        return dims * span / (2 * math.log(rope["base"]))

    low = max(math.floor(pair(s["beta_fast"])), 0)
    high = min(math.ceil(pair(s["beta_slow"])), dims - 1)
    index = torch.arange(dims // 2, dtype=torch.float64)
    ramp = ((index - low) / ((high - low) or 1e-3)).clamp(0, 1)
    on_rotary = s["attention_factor_on"] == "rotary"
    gain = attention_factor(s) if on_rotary else 1.0
    return inv / s["factor"] * ramp + inv * (1 - ramp), gain


def rope_frequencies(rope: dict, dims: int) -> tuple[torch.Tensor, float]:
    """Return the inverse frequencies of `dims` rotated dims' pairs, and the gain on
    cos and sin, as `bux.architecture.attention.rope.frequencies` makes them."""
    inv = 1.0 / rope["base"] ** (torch.arange(0, dims, 2, dtype=torch.float64) / dims)
    s = rope["scaling"]
    if s is None:
        return inv, 1.0
    if s["type"] == "pi":
        return inv / s["factor"], 1.0
    if s["type"] == "yarn":
        return _yarn(rope, inv, dims)
    if s["type"] == "llama3":
        turns = s["original_max_len"] * inv / (2 * math.pi)
        span = s["high_freq_factor"] - s["low_freq_factor"]
        ramp = ((turns - s["low_freq_factor"]) / span).clamp(0, 1)
        return inv / s["factor"] * (1 - ramp) + inv * ramp, 1.0
    raise NotImplementedError(f"rope scaling {s['type']} not exported yet")


class Rope(nn.Module):
    """Rotates the first ``fraction`` of each head's dims by the tokens' positions,
    paired ``half`` (NeoX) or ``interleaved`` (GPT-J), in float32."""

    inv_freq: torch.Tensor

    def __init__(self, rope: dict, head_dim: int):
        super().__init__()
        self.config, self.layout = rope, rope["layout"]
        self.dims = rotated_dims(head_dim, rope)
        inv, self.gain = rope_frequencies(rope, self.dims)
        self.register_buffer("inv_freq", inv.float(), persistent=False)

    def reset(self) -> None:
        """Recompute the frequencies, which transformers' meta-device init leaves
        unset (``BuxPreTrainedModel._init_weights`` calls it)."""
        with torch.no_grad():
            self.inv_freq.copy_(rope_frequencies(self.config, self.dims)[0].float())

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate x, b t h k, at positions, b t."""
        if self.dims == 0:
            return x
        angles = positions[..., None].float() * self.inv_freq
        sin = (self.gain * angles.sin())[..., None, :]
        cos = (self.gain * angles.cos())[..., None, :]
        y, rest = x[..., : self.dims].float(), x[..., self.dims :]
        if self.layout == "interleaved":
            y1, y2 = y[..., 0::2], y[..., 1::2]
            y = torch.stack([y1 * cos - y2 * sin, y2 * cos + y1 * sin], -1).flatten(-2)
        else:
            y1, y2 = y.chunk(2, -1)
            y = torch.cat([y1 * cos - y2 * sin, y2 * cos + y1 * sin], -1)
        return torch.cat([y.to(x.dtype), rest], -1)
