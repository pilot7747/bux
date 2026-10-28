"""bux's generalized delta rule in plain PyTorch, for the layers vLLM has no kernel
of (``gdn2``, ``rwkv7``): each value head keeps a state S of dk by dv,

    S_t = diag(w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,    o_t = S_t^T q_t,

in float32, as `bux.export.hf.linear_attention` runs it. A batch of prompts runs by
chunks of `CHUNK` positions, matmuls within a chunk and the state carried across
(`rank_one`: the UT form of fla's DPLR kernels), its sequences padded to the longest
(`padded`), which leaves each one's state as it was. A batch of decoding steps runs
one position of every sequence at once (`step`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CHUNK = 64  # positions per chunk of a prompt


def wide(x: torch.Tensor) -> torch.Tensor:
    """Return x in float32, or in its own dtype where that is wider."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return ``x / sqrt(sum(x^2) + eps)`` over the last dim, in float32 (`wide`),
    the result in x's dtype."""
    y = wide(x)
    return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + eps)).to(x.dtype)


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
    outputs, (B, h, dv) in v's dtype, and the new state, in float32: the state's
    two rank-one updates in place, so that it is read and written few times."""
    s = wide(state)
    read = torch.einsum("bhk,bhkv->bhv", wide(a), s)
    new = (wide(log_decay).exp()[..., None] * s).flatten(0, 1)
    new.baddbmm_(wide(b).flatten(0, 1)[..., None], read.flatten(0, 1)[:, None])
    new.baddbmm_(wide(k).flatten(0, 1)[..., None], wide(v).flatten(0, 1)[:, None])
    new = new.unflatten(0, s.shape[:2])
    return torch.einsum("bhk,bhkv->bhv", wide(q), new).to(v.dtype), new


def rank_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    floor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the rule by chunks over sequences side by side, from `state`.

    With a `floor`, the chunks of a block of positions run side by side, all but the
    state they carry (`_side_by_side`); without, one after another (`_chunk`).

    Args:
        q: The queries, (B, T, h, dk).
        k: The keys, (B, T, h, dk).
        v: The values, (B, T, h, dv).
        log_decay: log w, (B, T, h, dk).
        a: What the rank-one term reads of the state, (B, T, h, dk).
        b: Where it writes it, (B, T, h, dk).
        state: Each sequence's S before its first position, (B, h, dk, dv).
        floor: How far a position's log decay may fall, ``-min(log w)``; None for
            no bound.

    Returns:
        The outputs, (B, T, h, dv) in v's dtype, and the last states, in float32.
    """
    batch, t, h = v.shape[:3]
    sub = blocks_of(floor)
    span = CHUNK * (max(1, BLOCK // (batch * h * CHUNK)) if sub else 1)
    s, outs = wide(state), []
    for start in range(0, t, span):
        parts = [
            _chunked(x[:, start : start + span]) for x in (q, k, v, log_decay, a, b)
        ]
        if sub:
            out, s = _side_by_side(s, *parts, sub=sub)
        else:
            out, s = _chunk(s, *(x[:, :, 0] for x in parts))
            out = out[:, :, None]
        outs.append(out.flatten(2, 3))
    out = torch.cat(outs, 2)[:, :, :t].transpose(1, 2)
    return out.to(v.dtype), s


#: Positions (times sequences and heads) that one block of chunks side by side
#: holds at once, which bounds its memory: about 13 KB each at dk = dv = 128.
BLOCK = 2**16

#: The largest exponent a factor of a decay takes, ``exp(80)`` about 5.5e34, which
#: float32 holds with room for a sum over the channels.
LIMIT = 80.0


def blocks_of(floor: float | None) -> int | None:
    """The rows of a chunk whose decays factor at one point (`_decayed`): the most
    that keep every factor under ``exp(LIMIT)``; None where they would be too few
    (under 4), or with no floor, for chunks one after another."""
    if floor is None or floor <= 0:
        return CHUNK if floor is not None else None
    rows = 2 ** math.floor(math.log2(LIMIT / floor)) if LIMIT / floor >= 1 else 0
    return min(rows, CHUNK) if rows >= 4 else None


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


def _decayed(
    x: torch.Tensor,
    y: torch.Tensor,
    to: torch.Tensor,
    G: torch.Tensor,
    before: torch.Tensor,
    sub: int,
    diagonal: bool,
) -> torch.Tensor:
    """Return ``sum_d x_t[d] y_s[d] exp(to_t[d] - G_s[d])`` for the positions s
    up to t of a chunk (before t, without the `diagonal`), (..., c, c), by matmuls.

    The decay factors at the point before each block of `sub` rows, its
    cumulative log decay r: ``exp(to_t - r) exp(r - G_s)``. Both are at most 1 for s
    before the block, and the second at most ``exp(sub * floor)`` within it
    (`blocks_of`); the pairs after t, which can be larger, are clipped and dropped.
    """
    c = x.shape[-2]
    ref = before.unflatten(-2, (c // sub, sub))[..., :1, :]  # (..., n, 1, d)
    left = (
        x.unflatten(-2, (c // sub, sub))
        * (to.unflatten(-2, (c // sub, sub)) - ref).exp()
    )
    right = y[..., None, :, :] * (ref - G[..., None, :, :]).clamp(max=LIMIT).exp()
    m = (left @ right.transpose(-1, -2)).flatten(-3, -2)
    keep = torch.ones(c, c, dtype=torch.bool, device=x.device).tril(
        0 if diagonal else -1
    )
    return m.masked_fill(~keep, 0.0)


def _side_by_side(
    s: torch.Tensor, *parts: torch.Tensor, sub: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `rank_one` over chunks side by side, (B, h, n, c, d), from the state s:
    their outputs, (B, h, n, c, dv), and the last state.

    `_chunk`'s algebra, with what a chunk reads of the state written as ``x = W S
    + U``: a chunk's outputs are ``P S + O`` and the state it leaves ``M S + N``,
    each computed for every chunk at once; only the states the chunks start from
    go one after another (`_scan`).
    """
    q, k, v, g, a, b = parts
    dk, dv = k.shape[-1], v.shape[-1]
    G = g.cumsum(-2)
    before = G - g
    D_qk = _decayed(q, k, G, G, before, sub, True)
    D_qb = _decayed(q, b, G, G, before, sub, True)
    R_ak = _decayed(a, k, before, G, before, sub, False)
    eye = torch.eye(k.shape[-2], dtype=k.dtype, device=k.device)
    system = eye - _decayed(a, b, before, G, before, sub, False)
    rhs = torch.cat([a * before.exp(), R_ak @ v], -1)
    W, U = torch.linalg.solve_triangular(
        system, rhs, upper=False, unitriangular=True
    ).split([dk, dv], -1)
    P, O = q * G.exp() + D_qb @ W, D_qb @ U + D_qk @ v
    last = G[..., -1:, :]
    written = (b * (last - G).exp()).transpose(-1, -2)
    M = torch.diag_embed(last[..., 0, :].exp()) + written @ W
    N = written @ U + (k * (last - G).exp()).transpose(-1, -2) @ v
    starts, s = _scan(s, M, N)
    return P @ starts + O, s


def _scan(
    s: torch.Tensor, M: torch.Tensor, N: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The states the chunks start from, (B, h, n, dk, dv), the first s, each next
    ``M_i S_i + N_i``; and the state after the last."""
    batch, h, n = M.shape[:3]
    M, N, s = M.flatten(0, 1), N.flatten(0, 1), s.flatten(0, 1)
    starts = []
    for i in range(n):
        starts.append(s)
        s = torch.baddbmm(N[:, i], M[:, i], s)
    out = torch.stack(starts, 1).unflatten(0, (batch, h))
    return out, s.unflatten(0, (batch, h))


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
