"""RWKV-7 time mixing, exported: bux's ``rwkv7``.

Its state is linear attention's, by the generalized delta rule
(``linear_attention.recurrence``, one position at a time). Its token shift reads the
token before, which decoding keeps in the layer's cache slot as a conv state of one
position, beside the recurrent state.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import Ctx, Norm, Site
from .conv import keep_state, previous_state, real_tokens, without_padding
from .linear_attention import LowRank, hold, l2norm, recurrence


class RWKV7(nn.Module):
    """bux's ``rwkv7``: the token shift, r, k and v, low-rank decay w, in-context
    learning rate a, value residual and gate, the generalized delta rule
    ``S_t = (diag(w_t) - (a_t kk_t) kk_t^T) S_{t-1} + k_t v_t^T``, a layernorm of
    each head plus the bonus ``(r k r_k) v``, the gate and the projection back.

    Its ``v`` port written pushes the values; read, mixes the tap's into its own (the
    value residual), popping it with ``$v!``.
    """

    bux_types = ("rwkv7",)

    @staticmethod
    def cache_type(c: dict) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "linear_attention"

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        if c["conv_size"] is not None:
            raise NotImplementedError("rwkv7 with a short conv (it shifts tokens)")
        h, hv = c["num_heads"], c["num_v_heads"] or c["num_heads"]
        dk, dv = c["head_dim"], c["v_head_dim"] or c["head_dim"]
        self.heads, self.v_heads, self.head_dim, self.v_head_dim = h, hv, dk, dv
        nk, nv = h * dk, hv * dv
        self.shift = nn.Parameter(torch.zeros(6, width))
        self.r = nn.Linear(width, nk, bias=False)
        self.k = nn.Linear(width, nk, bias=False)
        self.v = nn.Linear(width, nv, bias=False)
        self.w = LowRank(width, c["decay_rank"], nk, "tanh", bias=True)
        self.a = LowRank(width, c["a_rank"], nk, bias=True)
        self.reads = next(
            ((t, m == "pop") for n, t, m in site.ports if n == "v" and m != "write"),
            None,
        )
        self.writes = site.port("v", "write")
        reading = self.reads is not None
        self.v_mix = LowRank(width, c["v_rank"], nv, bias=True) if reading else None
        gate = c["output_gate"]
        self.g = None if gate is None else LowRank(width, c["gate_rank"], nv, gate)
        self.k_k = nn.Parameter(torch.zeros(nk))
        self.k_a = nn.Parameter(torch.zeros(nk))
        self.r_k = nn.Parameter(torch.zeros(hv, dk))
        norm = {
            "type": "layernorm",
            "eps": dv * c["norm_eps"],
            "affine": True,
            "bias": True,
            "zero_centered": False,
            "scale_input": False,
        }
        self.norm = Norm(norm, (hv, dv))
        self.o = nn.Linear(nv, width, bias=False)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        slot = ctx.cache_slot()
        x = without_padding(x, ctx)
        b, t, _ = x.shape
        h, hv, dk, dv = self.heads, self.v_heads, self.head_dim, self.v_head_dim
        shifted = x + (self.previous(x, ctx, slot) - x) * self.shift[:, None, None]
        xr, xw, xk, xv, xa, xg = shifted
        r, k, v = self.r(xr), self.k(xk), self.v(xv).reshape(b, t, hv, dv)
        if self.v_mix is not None and self.reads is not None:  # value residual
            mix = torch.sigmoid(self.v_mix(xv)).reshape(v.shape)
            v = v + (ctx.read(*self.reads) - v) * mix
        if self.writes is not None:
            ctx.push(self.writes, v)
        log_w = -math.exp(-0.5) * torch.sigmoid(self.w(xw))
        a = torch.sigmoid(self.a(xa))
        kk = k * self.k_k
        k = k * (1 + (a - 1) * self.k_a)

        def split(y: torch.Tensor) -> torch.Tensor:
            """Split into heads, each serving its group of value heads."""
            return y.reshape(b, t, h, dk).repeat_interleave(hv // h, 2)

        r, k, log_w, a, kk = map(split, (r, k, log_w, a, kk))
        kk = l2norm(kk, eps=1e-12)
        written, log_w, step = hold(real_tokens(ctx, t), k, log_w, kk * a)
        state = previous_state(ctx, slot)
        o, state = recurrence(r, written, v, log_w, state, -kk, step)
        keep_state(ctx, slot, state)
        o = self.norm(o) + (r * k * self.r_k).sum(-1, keepdim=True) * v
        o = o.reshape(b, t, -1)
        if self.g is not None:
            o = o * self.g(xg)
        return self.o(o)

    def previous(self, x: torch.Tensor, ctx: Ctx, slot: int) -> torch.Tensor:
        """Return every token's predecessor, zero before the first; while decoding,
        the one before this call's first, kept in the cache."""
        first = torch.zeros_like(x[:, :1])
        if ctx.cache is not None:
            decoding = ctx.cache.has_previous_state(slot, 0)
            kept = ctx.cache.update_conv_state(
                x.transpose(1, 2), slot, 0, conv_kernel_size=1
            )
            if decoding:
                first = kept[..., :1].transpose(1, 2)
        return torch.cat([first, x[:, :-1]], 1)
