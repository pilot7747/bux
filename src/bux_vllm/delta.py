"""bux's generalized delta rule in plain PyTorch, for the layers vLLM has no kernel
of (``gdn2``, ``rwkv7``): each value head keeps a state S of dk by dv,

    S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T q_t,

in float32, as `bux.export.hf.linear_attention` runs it. A batch of prompts runs by
chunks of `CHUNK` positions, matmuls within a chunk and the state carried across
(`rank_one`: the UT form of fla's DPLR kernels), its sequences padded to the longest
(`padded`), which leaves each one's state as it was. A batch of decoding steps runs
one position of every sequence at once (`step`). What each layer feeds the rule from
its projections is here too (`gdn2_rule`, `rwkv7_rule`), so that a test checks it
against the export without vLLM.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

#: A rank's q/k heads, value heads, and their dims.
Heads = tuple[int, int, int, int]

CHUNK = 64  # positions per chunk of a prompt


def wide(x: torch.Tensor) -> torch.Tensor:
    """Return x in float32, or in its own dtype where that is wider."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return ``x / sqrt(sum(x^2) + eps)`` over the last dim, in float32 (`wide`),
    the result in x's dtype."""
    y = wide(x)
    return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + eps)).to(x.dtype)


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


def step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One position of the rule for each sequence of a batch: q, k, a, b and the log
    decay are (B, h, dk), v (B, h, dv), the state (B, h, dk, dv). Returns the
    outputs, (B, h, dv) in v's dtype, and the new state, in float32."""
    s = wide(state)
    read = torch.einsum("bhk,bhkv->bhv", wide(a), s)
    s = wide(log_decay).exp()[..., None] * s + wide(b)[..., None] * read[..., None, :]
    s = s + wide(k)[..., None] * wide(v)[..., None, :]
    return torch.einsum("bhk,bhkv->bhv", wide(q), s).to(v.dtype), s


def rank_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the rule by chunks over sequences side by side, from `state`.

    Args:
        q: The queries, (B, T, h, dk).
        k: The keys, (B, T, h, dk).
        v: The values, (B, T, h, dv).
        log_decay: log w, (B, T, h, dk).
        a: What the rank-one term reads of the state, (B, T, h, dk).
        b: Where it writes it, (B, T, h, dk).
        state: Each sequence's S before its first position, (B, h, dk, dv).

    Returns:
        The outputs, (B, T, h, dv) in v's dtype, and the last states, in float32.
    """
    t = v.shape[1]
    parts = [_chunked(x) for x in (q, k, v, log_decay, a, b)]
    s = wide(state)
    outs = []
    for i in range(parts[0].shape[2]):
        out, s = _chunk(s, *(x[:, :, i] for x in parts))
        outs.append(out)
    out = torch.stack(outs, 2).flatten(2, 3)[:, :, :t].transpose(1, 2)
    return out.to(v.dtype), s


def _chunked(x: torch.Tensor) -> torch.Tensor:
    """Return x, (B, T, h, d), as (B, h, n, c, d) in float32: n chunks of c
    positions, the last padded with zeros, which leave a state as it was."""
    x = wide(x).transpose(1, 2)
    x = F.pad(x, (0, 0, 0, -x.shape[2] % CHUNK))
    return x.reshape(*x.shape[:2], -1, CHUNK, x.shape[-1])


def _pairwise(G: torch.Tensor, before: torch.Tensor | None = None) -> torch.Tensor:
    """Return the decays ``exp(G_t - G_s)`` from position s to t of a chunk, (B, h,
    c, c, d), 0 where s is after t, from the cumulative log decays G, (B, h, c, d);
    with `before`, G up to the position before each, ``exp(before_t - G_s)``, 0
    where s is not before t. No exponent is positive."""
    c = G.shape[-2]
    to = G if before is None else before
    after = torch.ones(c, c, dtype=torch.bool, device=G.device)
    after = after.triu(1 if before is None else 0)
    diff = to[..., :, None, :] - G[..., None, :, :]
    return diff.masked_fill(after[..., None], float("-inf")).exp()


def _inner(x: torch.Tensor, y: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Return ``sum_d x_t[d] y_s[d] D[t, s, d]`` for every two positions of a chunk."""
    return torch.einsum("bhtd,bhsd,bhtsd->bhts", x, y, D)


def _carry(
    s: torch.Tensor, k: torch.Tensor, G: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    """Return the state a chunk ends with, ``diag(exp(G_c)) S + sum_s diag(exp(G_c -
    G_s)) k_s u_s^T``, from the state it starts with and what it writes."""
    last = G[..., -1:, :]
    return (
        last.exp().transpose(-1, -2) * s + (k * (last - G).exp()).transpose(-1, -2) @ u
    )


def _chunk(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `rank_one` over one chunk from the state s: its outputs and last state.

    What each position reads of the state before it, ``x_t = a_t^T S_{t-1}``,
    solves ``(I - L) X = (A exp(G')) S + L_k V``, with ``L[t, s] = a_t^T
    diag(exp(G'_t - G_s)) b_s`` below the diagonal (L_k with k_s), for G' the log
    decay up to the position before t.
    """
    G = g.cumsum(-2)
    before = G - g
    D, R = _pairwise(G), _pairwise(G, before)
    eye = torch.eye(k.shape[-2], dtype=k.dtype, device=k.device)
    rhs = (a * before.exp()) @ s + _inner(a, k, R) @ v
    system = eye - _inner(a, b, R)
    x = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    out = (q * G.exp()) @ s + _inner(q, b, D) @ x + _inner(q, k, D) @ v
    last = (G[..., -1:, :] - G).exp()
    return out, _carry(s, b, G, x) + (k * last).transpose(-1, -2) @ v


def padded(x: torch.Tensor, starts: torch.Tensor, longest: int) -> torch.Tensor:
    """The tokens of a batch's sequences side by side, (N, ...), as (B, longest,
    ...): sequence i from ``starts[i]`` to ``starts[i + 1]``, then zeros."""
    lengths = starts[1:] - starts[:-1]
    at = torch.arange(longest, device=x.device)
    idx = (starts[:-1, None] + at).clamp(max=max(x.shape[0] - 1, 0))
    keep = (at < lengths[:, None]).view(*idx.shape, *(1,) * (x.ndim - 1))
    return x[idx] * keep.to(x.dtype)


def unpadded(y: torch.Tensor, starts: torch.Tensor, n: int) -> torch.Tensor:
    """The inverse of `padded`: the n tokens of (B, longest, ...) side by side."""
    tokens = torch.arange(n, device=y.device)
    seq = torch.searchsorted(starts[1:], tokens, right=True)
    return y[seq, tokens - starts[seq]]


def previous(
    x: torch.Tensor, starts: torch.Tensor, before: torch.Tensor
) -> torch.Tensor:
    """Each token's predecessor in its sequence, (N, ...), of the tokens of a
    batch's sequences side by side: sequence i's first token's is ``before[i]``."""
    prev = x.roll(1, 0)
    prev[starts[:-1]] = before.to(x.dtype)
    return prev


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
