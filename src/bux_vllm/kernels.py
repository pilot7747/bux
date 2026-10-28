"""Triton kernels for the plugin's sparse attention (`bux_vllm.sparse`), which read
vLLM's paged caches in place where the PyTorch path (`bux_vllm.picks`) gathers the
keys into memory first: the indexer's score of every key of each query's request
(`scores`), and each query's softmax over the keys it picked (`attend`), in float32.

They run under ``BUX_VLLM_SPARSE_KERNELS=triton``, for the settings they take
(`takes`): DSA's single keys, MSA's blocks and picks per group, QSA's blocks (its
scores PyTorch's), weighted or summed heads, keys and values of up to 256 dims, not
MLA's latents. A prompt's queries, which share their keys, score them by PyTorch's
GEMM; decoded tokens, each over its own request's keys, by the kernel. The picks stay
PyTorch's, as bux ranks the scores. ``scripts/vllm_kernels.py`` checks them against
the PyTorch path on a GPU.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from . import picks
from .picks import Picked, Sparse

#: Whether the sparse ops run these kernels, read once.
ENABLED = os.environ.get("BUX_VLLM_SPARSE_KERNELS", "torch") == "triton"
MOST_DIMS = 256  # the widest key or value head the kernels take


def takes(spec: Sparse, dims: tuple[int, int]) -> bool:
    """Whether the kernels compute the layer of `spec`, its key and value heads of
    `dims`: not MLA's latents."""
    return ENABLED and max(dims) <= MOST_DIMS and not spec.latent


def _block(n: int) -> int:
    """The tile of `n` elements a dot takes: a power of two, 16 at least."""
    return max(16, triton.next_power_of_2(n))


@triton.jit
def _scores_kernel(
    q_ptr,
    w_ptr,
    cache_ptr,
    table_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_wt,
    stride_cb,
    stride_cs,
    stride_tt,
    stride_ot,
    stride_og,
    n_keys,
    first,
    heads: tl.constexpr,
    dims: tl.constexpr,
    page: tl.constexpr,
    relu: tl.constexpr,
    groups: tl.constexpr,
    block_h: tl.constexpr,
    block_i: tl.constexpr,
    block_s: tl.constexpr,
):
    """One query's scores of `block_s` keys: ``sum_j w_j ReLU(q_j . k_s)``, or each
    of the `groups` heads' from `first` on, ``w_j ReLU(q_j . k_s)``, with `groups`."""
    t, part = tl.program_id(0), tl.program_id(1)
    s = part * block_s + tl.arange(0, block_s)
    h, i = tl.arange(0, block_h), tl.arange(0, block_i)
    on_h, on_i, inside = h < heads, i < dims, s < n_keys
    q = tl.load(
        q_ptr + t * stride_qt + h[:, None] * stride_qh + i[None, :],
        mask=on_h[:, None] & on_i[None, :],
        other=0.0,
    ).to(tl.float32)
    w = tl.load(w_ptr + t * stride_wt + h, mask=on_h, other=0.0).to(tl.float32)
    block = tl.load(table_ptr + t * stride_tt + s // page, mask=inside, other=0)
    rows = block.to(tl.int64) * stride_cb + (s % page) * stride_cs
    k = tl.load(
        cache_ptr + rows[:, None] + i[None, :],
        mask=inside[:, None] & on_i[None, :],
        other=0.0,
    ).to(tl.float32)
    dots = tl.dot(q, tl.trans(k), input_precision="ieee")
    if relu:
        dots = tl.maximum(dots, 0.0)
    weighted = dots * w[:, None]
    if groups:  # each of this rank's groups by its own head
        mine = on_h & (h >= first) & (h < first + groups)
        at = out_ptr + t * stride_ot + (h - first)[:, None] * stride_og + s[None, :]
        tl.store(at, weighted, mask=mine[:, None] & inside[None, :])
    else:
        tl.store(out_ptr + t * stride_ot + s, tl.sum(weighted, axis=0), mask=inside)


def scores(
    q: torch.Tensor,
    w: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    n: int,
    relu: bool,
    groups: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Each query's indexer score of each of its request's first `n` keys, (Q, n) in
    float32, as `bux_vllm.picks.scores` gives it summed over the heads; or each
    head's of this rank's groups, (Q, G, n), as it gives them with `per_group`.

    Args:
        q: The indexer's queries, (Q, H, i), the last axis contiguous.
        w: Their heads' weights, (Q, H).
        cache: The paged index cache, (blocks, 1, block_size, i).
        rows: Each query's request's block table, (Q, blocks).
        n: How many keys to score.
        relu: Whether to ReLU the dot products.
        groups: The first of this rank's groups and how many, picking apart; None
            for all heads' scores summed.
    """
    first, count = groups or (0, 0)
    shape = (q.shape[0], count, n) if groups else (q.shape[0], n)
    out = torch.empty(shape, dtype=torch.float32, device=q.device)
    block_s = 64
    grid = (q.shape[0], triton.cdiv(n, block_s))
    _scores_kernel[grid](
        q,
        w,
        cache,
        rows,
        out,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        cache.stride(0),
        cache.stride(2),
        rows.stride(0),
        out.stride(0),
        out.stride(1) if groups else 0,
        n,
        first,
        heads=q.shape[1],
        dims=q.shape[2],
        page=cache.shape[2],
        relu=relu,
        groups=count,
        block_h=_block(q.shape[1]),
        block_i=_block(q.shape[2]),
        block_s=block_s,
    )
    return out


@triton.jit
def _attend_kernel(
    q_ptr,
    cache_ptr,
    table_ptr,
    keys_ptr,
    valid_ptr,
    sinks_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_cb,
    stride_cg,
    stride_cs,
    stride_tt,
    stride_kt,
    stride_kg,
    stride_ot,
    stride_oh,
    n_picks,
    scale,
    softcap,
    heads_per: tl.constexpr,
    dk: tl.constexpr,
    dv: tl.constexpr,
    page: tl.constexpr,
    capped: tl.constexpr,
    sunk: tl.constexpr,
    block_h: tl.constexpr,
    block_dk: tl.constexpr,
    block_dv: tl.constexpr,
    block_k: tl.constexpr,
):
    """One query's heads of one kv head's group, over its picked keys: an online
    softmax of the scaled, capped logits, the sinks beside them."""
    t, g = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, block_h)
    head = g * heads_per + h
    on_h = h < heads_per
    d, e = tl.arange(0, block_dk), tl.arange(0, block_dv)
    q = tl.load(
        q_ptr + t * stride_qt + head[:, None] * stride_qh + d[None, :],
        mask=on_h[:, None] & (d[None, :] < dk),
        other=0.0,
    ).to(tl.float32)
    top = tl.full([block_h], float("-inf"), tl.float32)
    total = tl.zeros([block_h], tl.float32)
    acc = tl.zeros([block_h, block_dv], tl.float32)
    for start in range(0, n_picks, block_k):
        j = start + tl.arange(0, block_k)
        inside = j < n_picks
        pick = t * stride_kt + g * stride_kg + j  # a group's own picks, or all's
        at = tl.load(keys_ptr + pick, mask=inside, other=0)
        ok = tl.load(valid_ptr + pick, mask=inside, other=0) != 0
        ok = ok & inside
        block = tl.load(table_ptr + t * stride_tt + at // page, mask=ok, other=0)
        rows = block.to(tl.int64) * stride_cb + g * stride_cg + (at % page) * stride_cs
        k = tl.load(
            cache_ptr + rows[:, None] + d[None, :],
            mask=ok[:, None] & (d[None, :] < dk),
            other=0.0,
        ).to(tl.float32)
        logits = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        if capped:
            logits = softcap * (2.0 * tl.sigmoid(2.0 * logits / softcap) - 1.0)
        logits = tl.where(ok[None, :], logits, float("-inf"))
        new = tl.maximum(top, tl.max(logits, axis=1))
        safe = tl.where(new == float("-inf"), 0.0, new)
        p = tl.exp(logits - safe[:, None])
        fade = tl.exp(top - safe)
        total = total * fade + tl.sum(p, axis=1)
        v = tl.load(
            cache_ptr + rows[:, None] + dk + e[None, :],
            mask=ok[:, None] & (e[None, :] < dv),
            other=0.0,
        )
        acc = acc * fade[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee").to(
            tl.float32
        )
        top = new
    kept = total > 0
    if sunk:
        sink = tl.load(sinks_ptr + head, mask=on_h, other=0.0).to(tl.float32)
        most = tl.maximum(top, sink)
        fade = tl.exp(top - most)
        total, acc = total * fade + tl.exp(sink - most), acc * fade[:, None]
    out = tl.where(kept[:, None], acc / tl.where(kept, total, 1.0)[:, None], 0.0)
    tl.store(
        out_ptr + t * stride_ot + head[:, None] * stride_oh + e[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=on_h[:, None] & (e[None, :] < dv),
    )


def attend(
    spec: Sparse,
    q: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    picked: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Each query's heads' outputs over the keys it picked, (Q, h, dv) in the cache's
    dtype, as `bux_vllm.picks.attend` gives them.

    Args:
        spec: The layer's settings: the key dims, scale, softcap and sinks.
        q: The queries, (Q, h, dk), the last axis contiguous.
        cache: The paged cache of keys and values, (blocks, g, block_size, dk + dv).
        rows: Each query's request's block table, (Q, blocks).
        picked: The picked keys' positions and which are valid, (Q, K) each, or
            (Q, g, K) for each kv head's group.
    """
    keys, valid = (t.contiguous() for t in picked)
    n, h, dk = q.shape
    g, dv = cache.shape[1], cache.shape[3] - spec.dk
    out = torch.empty(n, h, dv, dtype=cache.dtype, device=q.device)
    sinks = spec.sinks if spec.sinks is not None else q.new_zeros(1)
    _attend_kernel[(n, g)](
        q,
        cache,
        rows,
        keys,
        valid.view(torch.int8),
        sinks,
        out,
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        rows.stride(0),
        keys.stride(0),
        keys.stride(1) if keys.ndim == 3 else 0,
        out.stride(0),
        out.stride(1),
        keys.shape[-1],
        spec.scale,
        spec.softcap or 1.0,
        heads_per=h // g,
        dk=dk,
        dv=dv,
        page=cache.shape[2],
        capped=spec.softcap is not None,
        sunk=spec.sinks is not None,
        block_h=_block(h // g),
        block_dk=_block(dk),
        block_dv=_block(dv),
        block_k=32,
    )
    return out


def attend_segment(
    spec: Sparse,
    q: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor],
    rows: tuple[torch.Tensor, torch.Tensor],
    n: int,
) -> tuple[torch.Tensor, Picked]:
    """`bux_vllm.picks.attend_segment` by the kernels: each chunk of queries' scores
    (`scores`), its picks as bux ranks them, and its attention over them (`attend`);
    the outputs, and the picks padded to `bux_vllm.picks.most` keys a query."""
    count, shared = q.shape[0], rows[0].shape[0] == 1
    table, index_table = (r.expand(count, -1) for r in rows)
    span = torch.arange(n, device=q.device)
    keys = picks.gather(caches[1], rows[1], span[None])[0, 0] if shared else None
    per = picks._per_query(spec, q, index[0], caches[0], n, shared or not spec.mean)
    chunk, outs, kept = max(1, picks.BUDGET // per), [], []
    for a in range(0, count, chunk):
        cut = slice(a, a + chunk)
        iq, iw = (t[cut].contiguous() for t in index)
        where = (positions[cut], index_table[cut], n)
        picked = _choose(spec, (iq, iw), caches[1], keys, where)
        outs.append(attend(spec, q[cut].contiguous(), caches[0], table[cut], picked))
        kept.append(picks.padded(picked, picks.most(spec)))
    out_keys, valid = zip(*kept, strict=True)
    return torch.cat(outs), (torch.cat(out_keys), torch.cat(valid))


def attend_picked(
    spec: Sparse,
    q: torch.Tensor,
    picked: Picked,
    positions: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """`bux_vllm.picks.attend_picked` by the kernels: the queries' attention over the
    keys another call picked, of those their own window sees."""
    keys, valid = picked[0].clamp(0, n - 1), picked[1]
    see = picks.seen(positions, n, spec.window)
    view = see if keys.ndim == 2 else see[:, None].expand(-1, keys.shape[1], -1)
    table = rows.expand(q.shape[0], -1)
    return attend(
        spec, q.contiguous(), cache, table, (keys, valid & view.gather(-1, keys))
    )


def _choose(
    spec: Sparse,
    index: tuple[torch.Tensor, torch.Tensor],
    cache: torch.Tensor,
    keys: torch.Tensor | None,
    where: tuple[torch.Tensor, torch.Tensor, int],
) -> Picked:
    """The keys the queries keep, as bux picks them, from the indexer's queries and
    weights `index`: scored by PyTorch over `keys`, (n, i), which a request's
    queries share, or over each query's request's gathered (QSA's blocks, which
    PyTorch pools); else by `scores`. `where` are the queries' positions, their
    requests' block tables and the keys' count."""
    positions, table, n = where
    see = picks.seen(positions, n, spec.window)
    if keys is None and spec.mean:
        span = torch.arange(n, device=positions.device)
        keys = picks.gather(cache, table, span.expand(positions.shape[0], n))[:, 0]
    if keys is not None:
        return picks.choose(spec, *index, keys, see, positions)
    groups = spec.groups if spec.per_group else None
    s = scores(*index, cache, table, n, spec.relu, groups)
    if spec.block_size > 1:
        return picks.best_blocks(s, see, positions, spec.block_size, spec.top_k)
    return picks.tokens(s, see, spec.top_k)
