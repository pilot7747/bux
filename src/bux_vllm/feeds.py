"""What ``gdn2`` and ``rwkv7`` feed bux's delta rule (`bux_vllm.delta`) from their
projections, and what they make of its outputs, in plain PyTorch, as
`bux.export.hf` computes it: the plugin's layers run it on vLLM's projections, and a
test checks it against the export without vLLM.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .delta import l2norm, wide

#: A rank's q/k heads, value heads, and their dims.
Heads = tuple[int, int, int, int]


def channel_decay(
    f: torch.Tensor, A_log: torch.Tensor, lower_bound: float | None
) -> torch.Tensor:
    """KDA's log decay per key channel of f, its projection plus dt_bias, (..., h,
    dk): ``-exp(A_log) softplus(f)``, or ``lower_bound * sigmoid(exp(A_log) f)``,
    in float32."""
    rate = wide(A_log).exp()[:, None]
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(rate * f)
    return -rate * F.softplus(f)


def shared(x: torch.Tensor, heads: Heads) -> torch.Tensor:
    """x of the q/k heads, (T, h d), as (T, hv, d): each head shared out to the
    group of value heads next to it."""
    h, hv = heads[:2]
    return x.unflatten(-1, (h, -1)).repeat_interleave(hv // h, 1)


def gdn2_rule(
    c: dict,
    heads: Heads,
    qkv: torch.Tensor,
    gates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> list[torch.Tensor]:
    """GDN-2's q, k, v, log decay, a and b at the value heads, (T, hv, d), from its
    convolved q, k and v, (T, C), and its projections of the decay, the erase gate
    and the write gate: q and k L2-normalized, q scaled by ``dk ** -0.5``, KDA's
    decay g, ``a = -exp(g) b k`` written at k, and the values ``w v``."""
    h, hv, dk, dv = heads
    f, erase, write = gates
    q, k, v = qkv.split([h * dk, h * dk, hv * dv], -1)
    q, k = shared(q, heads), shared(k, heads)
    q, k = l2norm(q) * dk**-0.5, l2norm(k)
    f = (wide(f) + wide(dt_bias)).unflatten(-1, (h, dk))
    g = shared(channel_decay(f, A_log, c["gate_lower_bound"]).flatten(1), heads)
    scale = 2.0 if c["negative_eigenvalues"] else 1.0
    b = shared(scale * wide(torch.sigmoid(erase)), heads)
    w = wide(torch.sigmoid(write)).unflatten(-1, (hv, dv))
    a = -g.exp() * b * wide(k)
    return [q, k, w * wide(v.unflatten(-1, (hv, dv))), g, a, k]


def rwkv7_mixes(x: torch.Tensor, prev: torch.Tensor, shift: torch.Tensor) -> tuple:
    """RWKV-7's six inputs, of r, w, k, v, a and g: each token moved toward the
    one before it by a learned amount per channel, ``shift``'s row."""
    return (x + (prev - x) * shift[:, None]).unbind(0)


def rwkv7_rule(
    heads: Heads,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw: tuple[torch.Tensor, torch.Tensor],
    k_k: torch.Tensor,
    k_a: torch.Tensor,
) -> list[torch.Tensor]:
    """RWKV-7's r, k, v, log decay, a and b at the value heads, (T, hv, d), from its
    r, k and v and the low-rank projections of its decay and learning rate (`raw`):
    ``log w = -exp(-0.5) sigmoid(.)``, the normalized key kk read at ``-kk`` and
    written at ``kk a``, and k scaled toward ``a k`` by ``k_a``."""
    log_w = -math.exp(-0.5) * torch.sigmoid(raw[0])
    a = torch.sigmoid(raw[1])
    kk = k * k_k
    k = k * (1 + (a - 1) * k_a)
    r, k, log_w, a, kk = (shared(y, heads) for y in (r, k, log_w, a, kk))
    kk = l2norm(kk, eps=1e-12)
    return [r, k, v, log_w, -kk, kk * a]


def rwkv7_bonus(
    r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, r_k: torch.Tensor
) -> torch.Tensor:
    """RWKV-7's bonus ``(r k r_k) v`` of each value head, (T, hv, dv)."""
    return (r * k * r_k).sum(-1, keepdim=True) * v
