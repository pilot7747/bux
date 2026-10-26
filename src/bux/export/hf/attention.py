"""Softmax attention, exported: RoPE and ``gqa``, through transformers' attention
interface (eager, sdpa, flash, flex) and its KV cache."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .common import Ctx, Norm, Site, unsupported


def rope_frequencies(rope: dict, dims: int) -> tuple[torch.Tensor, float]:
    """Return the inverse frequencies of `dims` rotated dims' pairs, and the gain on
    cos and sin, as `bux.architecture.attention.rope.frequencies` makes them."""
    inv = 1.0 / rope["base"] ** (torch.arange(0, dims, 2, dtype=torch.float64) / dims)
    s = rope["scaling"]
    if s is None:
        return inv, 1.0
    if s["type"] == "pi":
        return inv / s["factor"], 1.0
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
        self.dims = int(head_dim * rope["fraction"]) // 2 * 2
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


def make_mask(ctx: Ctx, kind: str, arg: Any) -> Any:
    """Return the pass's attention mask of `kind`, as transformers makes it for the
    first cache layer of that kind (``config.layer_types``)."""
    if kind != "full_attention":
        raise NotImplementedError(f"{kind} masks are not exported yet")
    b, t = ctx.tokens.shape
    like = torch.empty((b, t, 1), dtype=ctx.dtype, device=ctx.tokens.device)
    return create_causal_mask(
        config=ctx.config,
        inputs_embeds=like,
        attention_mask=ctx.attention_mask,
        past_key_values=ctx.cache,
        position_ids=ctx.positions,
        layer_idx=ctx.config.layer_types.index(kind),
    )


def eager_attention(
    module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention of q, b h t k, over k and v, b g s k: the eager fallback of
    transformers' attention interface. Returns b t h k, and the probabilities."""
    groups = module.num_key_value_groups
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    w = q @ k.transpose(2, 3) * scaling
    if mask is not None:
        w = w + mask
    w = w.softmax(-1, dtype=torch.float32).to(q.dtype)
    return (w @ v).transpose(1, 2).contiguous(), w


class Attention(nn.Module):
    """bux's ``gqa``: q, k and v projections, QK-norm then RoPE, causal softmax
    attention by transformers' interface, and the output projection."""

    bux_types = ("gqa",)

    @staticmethod
    def cache_type(c: dict) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "full_attention"

    @staticmethod
    def masks(c: dict) -> list[tuple[str, Any]]:
        """The masks it reads, as ``ctx.mask`` takes them."""
        return [("full_attention", None)]

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        unsupported(
            c,
            *("fused_qkv", "gated", "sinks", "softcap", "sliding_window_size"),
            *("kv_conv_size", "indexer", "relative_bias", "log_scale", "output_norm"),
        )
        if site.ports:
            raise NotImplementedError("gqa: ports not exported yet")
        h, g = c["num_heads"], c["num_kv_heads"]
        dk = c["head_dim"] or width // h
        dv = c["v_head_dim"] or dk
        bias = c["bias"]
        out_bias = bias if c["output_bias"] is None else c["output_bias"]
        self.h, self.g, self.dk, self.dv = h, g, dk, dv
        self.q = nn.Linear(width, h * dk, bias=bias)
        self.k = nn.Linear(width, g * dk, bias=bias)
        self.v = nn.Linear(width, g * dv, bias=bias)
        self.o = nn.Linear(h * dv, width, bias=out_bias)
        qk = c["qk_norm"]
        self.q_norm = Norm(qk, dk) if qk else None
        self.k_norm = Norm(qk, dk) if qk else None
        self.rope = Rope(c["rope"], dk) if c["rope"] else None
        self.scale = c["scale"] or dk**-0.5
        self.config = site.config
        self.num_key_value_groups, self.is_causal = h // g, True

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.dk)
        k = self.k(x).view(b, t, self.g, self.dk)
        v = self.v(x).view(b, t, self.g, self.dv)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, ctx.positions), self.rope(k, ctx.positions)
        q, k, v = (y.transpose(1, 2) for y in (q, k, v))
        mask, slot = ctx.mask(), ctx.cache_slot()
        if ctx.cache is not None:
            k, v = ctx.cache.update(k, v, slot)
        attend = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention
        )
        out, _ = attend(self, q, k, v, mask, dropout=0.0, scaling=self.scale)
        return self.o(out.reshape(b, t, -1))
