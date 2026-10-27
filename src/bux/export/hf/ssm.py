"""State space mixers, exported: bux's ``mamba``, ``mamba2`` and ``short_conv``.

Mamba-2's selective scan is decayed linear attention (``linear_attention.decayed``),
by chunks in a prefill; Mamba's, a decay per channel and state dim, runs one
position at a time, as bux runs it. Their states and their convs' last inputs stay
in the layer's cache slot while decoding (``linear_attention``; ``conv`` for the
short conv, which keeps no other). Padded tokens of a left-padded batch take a time
step of 0, which leaves the state as it was.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Norm, Site, norm_config
from .conv import CausalConv, keep_state, previous_state, real_tokens, without_padding
from .linear_attention import decayed, hold, wide


def mamba_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Mamba's selective scan one position at a time, in float32 (`wide`):
    every channel's state ``S = exp(dt A) S + (dt x) B``, read as ``y = S @ C``.

    Args:
        x: The input of every channel, b t d.
        dt: Its time steps, b t d.
        A: The state matrix's diagonal, d n.
        B: What the states take in, b t n, shared by the channels.
        C: What they give out, b t n.
        state: S before the first position, b d n; None for zero.

    Returns:
        y, b t d in x's dtype, and the last state, in float32.
    """
    xs, dt, A, B, C = wide(x), wide(dt), wide(A), wide(B), wide(C)
    b, t, d = xs.shape
    s = xs.new_zeros(b, d, A.shape[-1]) if state is None else wide(state)
    ys = []
    for i in range(t):
        s = (dt[:, i, :, None] * A).exp() * s
        s = s + (dt[:, i] * xs[:, i])[..., None] * B[:, i, None]
        ys.append((s * C[:, i, None]).sum(-1))
    return torch.stack(ys, 1).to(x.dtype), s


class Mamba(nn.Module):
    """bux's ``mamba``, as Jamba has it: u and a gate z from the input, u convolved
    and SiLU'd, time steps of rank ``dt_rank``, B and C RMS-normalized, the selective
    scan of a state per channel (`mamba_scan`), D u, the gate and the projection
    back. ``dt_min`` and ``dt_max`` only start the time steps."""

    bux_types = ("mamba",)

    @staticmethod
    def cache_type(c: dict, ports: list) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "linear_attention"

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        d_inner, n = c["expand"] * width, c["state_size"]
        r = c["dt_rank"] or math.ceil(width / 16)
        self.rank, self.state_size = r, n
        self.in_proj = nn.Linear(width, 2 * d_inner, bias=False)
        self.conv = CausalConv(d_inner, c["conv_size"], True)
        self.x_proj = nn.Linear(d_inner, r + 2 * n, bias=False)
        self.dt_proj = nn.Parameter(torch.zeros(r, d_inner))
        self.dt_bias = nn.Parameter(torch.zeros(d_inner))
        rms = norm_config("rmsnorm")
        self.dt_norm = Norm(rms, r)
        self.b_norm, self.c_norm = Norm(rms, n), Norm(rms, n)
        self.A_log = nn.Parameter(torch.zeros(d_inner, n))
        self.D = nn.Parameter(torch.zeros(d_inner))
        self.out_proj = nn.Linear(d_inner, width, bias=False)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        slot = ctx.cache_slot()
        x = without_padding(x, ctx)
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u, ctx, slot))
        dt, B, C = self.x_proj(u).split(
            [self.rank, self.state_size, self.state_size], -1
        )
        dt = F.softplus(self.dt_norm(dt) @ self.dt_proj + self.dt_bias)
        (dt,) = hold(real_tokens(ctx, x.shape[1]), dt)
        A = -wide(self.A_log).exp()
        y, state = mamba_scan(
            u, dt, A, self.b_norm(B), self.c_norm(C), previous_state(ctx, slot)
        )
        keep_state(ctx, slot, state)
        return self.out_proj((y + u * self.D) * F.silu(z))


class Mamba2(nn.Module):
    """bux's ``mamba2``, as transformers' Mamba-2, Zamba2 and Nemotron-H compute it: a
    gate z, u, B and C convolved together and SiLU'd, a time step per head, the scan
    of a state per head by chunks, D u, the gate, an RMSNorm of each group of heads,
    and the projection back. ``dt_min`` and ``dt_max`` only start the time steps."""

    bux_types = ("mamba2",)

    @staticmethod
    def cache_type(c: dict, ports: list) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "linear_attention"

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        d_inner, n, g = c["expand"] * width, c["state_size"], c["num_groups"]
        if c["num_heads"] is not None:
            d_inner = c["num_heads"] * c["head_dim"]
        h = d_inner // c["head_dim"]
        self.inner, self.heads, self.head_dim = d_inner, h, c["head_dim"]
        self.groups, self.state_size = g, n
        self.in_proj = nn.Linear(width, 2 * d_inner + 2 * g * n + h, bias=False)
        self.conv = CausalConv(d_inner + 2 * g * n, c["conv_size"], True)
        self.dt_bias = nn.Parameter(torch.zeros(h))
        self.A_log = nn.Parameter(torch.zeros(h))
        self.D = nn.Parameter(torch.zeros(h))
        self.norm = Norm(norm_config("rmsnorm", 1e-5), (g, d_inner // g))
        self.out_proj = nn.Linear(d_inner, width, bias=False)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        slot = ctx.cache_slot()
        x = without_padding(x, ctx)
        b, t, _ = x.shape
        d, g, n, h = self.inner, self.groups, self.state_size, self.heads
        z, ubc, dt = self.in_proj(x).split([d, d + 2 * g * n, h], -1)
        ubc = F.silu(self.conv(ubc, ctx, slot))
        u, B, C = ubc.split([d, g * n, g * n], -1)
        u = u.reshape(b, t, h, self.head_dim)
        B, C = (y.reshape(b, t, g, n).repeat_interleave(h // g, 2) for y in (B, C))
        dt = F.softplus(wide(dt) + wide(self.dt_bias))
        (dt,) = hold(real_tokens(ctx, t), dt)
        A = -wide(self.A_log).exp()
        state = previous_state(ctx, slot)
        y, state = decayed(C, B, dt[..., None] * wide(u), (dt * A)[..., None], state)
        keep_state(ctx, slot, state)
        y = (y.to(u.dtype) + u * self.D[:, None]).reshape(b, t, d) * F.silu(z)
        y = self.norm(y.reshape(b, t, g, -1)).reshape(b, t, d)
        return self.out_proj(y)


class ShortConv(nn.Module):
    """bux's ``short_conv``, LFM2's: B, C and u from the input, and
    ``C * conv(B * u)`` projected back, the conv's last inputs its state."""

    bux_types = ("short_conv",)

    @staticmethod
    def cache_type(c: dict, ports: list) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "conv"

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        bias = c["bias"]
        self.in_proj = nn.Linear(width, 3 * width, bias=bias)
        self.conv = CausalConv(width, c["conv_size"], bias)
        self.out_proj = nn.Linear(width, width, bias=bias)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        B, C, u = self.in_proj(x).chunk(3, -1)
        return self.out_proj(C * self.conv(B * u, ctx, ctx.cache_slot()))
