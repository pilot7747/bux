"""Triton kernels for the plugin's sparse attention (`bux_vllm.sparse`), which read
vLLM's paged caches in place where the PyTorch path (`bux_vllm.picks`) gathers the
keys into memory first: the indexer's score of every key of each query's request
(`scores`, and `prompt_scores` for a prompt's queries, a tile of them for each key
read), and each query's softmax over the keys it picked (`attend`), in float32.

They run by default in a 16-bit model, and under ``BUX_VLLM_SPARSE_KERNELS=triton``
in any (``torch`` keeps the PyTorch path), for the settings they take
(`takes`): DSA's single keys, MSA's blocks and picks per group, QSA's blocks (its
scores PyTorch's), weighted or summed heads, keys and values of up to 256 dims, not
MLA's latents. The picks stay PyTorch's, as bux ranks the scores. ``scripts/vllm_kernels.py`` checks them against
the PyTorch path on a GPU.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from . import picks
from .picks import Picked, Sparse

#: Which path the sparse ops take: ``triton``, ``torch``, or unset for the kernels
#: in 16-bit models alone; read once.
CHOICE = os.environ.get("BUX_VLLM_SPARSE_KERNELS")
MOST_DIMS = 256  # the widest key or value head the kernels take
BLOCKS = (16, 32, 64, 128)  # MSA's blocks whose best a prompt's kernel finds


def takes(spec: Sparse, dims: tuple[int, int], dtype: torch.dtype) -> bool:
    """Whether the kernels compute the layer of `spec`, its key and value heads of
    `dims` in a cache of `dtype`: not MLA's latents, and by `CHOICE`."""
    if CHOICE is None:
        on = dtype in (torch.bfloat16, torch.float16)
    else:
        on = CHOICE == "triton"
    return on and max(dims) <= MOST_DIMS and not spec.latent


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
    stride_cg,
    stride_cs,
    stride_tt,
    stride_ot,
    stride_og,
    n_keys,
    first,
    start,
    heads: tl.constexpr,
    dims: tl.constexpr,
    share: tl.constexpr,
    page: tl.constexpr,
    relu: tl.constexpr,
    groups: tl.constexpr,
    block_h: tl.constexpr,
    block_i: tl.constexpr,
    block_s: tl.constexpr,
):
    """One query's scores of `block_s` keys: ``sum_j w_j ReLU(q_j . k_s)``, or each
    of the `groups` heads' from `first` on, ``w_j ReLU(q_j . k_s)``, with `groups`;
    a key's channels `share` a kv head, from channel `start` of its row on."""
    t, part = tl.program_id(0), tl.program_id(1)
    s = part * block_s + tl.arange(0, block_s)
    h, i = tl.arange(0, block_h), tl.arange(0, block_i)
    on_h, on_i, inside = h < heads, i < dims, s < n_keys
    spread = (i // share) * stride_cg + start + i % share
    q = tl.load(
        q_ptr + t * stride_qt + h[:, None] * stride_qh + i[None, :],
        mask=on_h[:, None] & on_i[None, :],
        other=0.0,
    ).to(tl.float32)
    w = tl.load(w_ptr + t * stride_wt + h, mask=on_h, other=0.0).to(tl.float32)
    block = tl.load(table_ptr + t * stride_tt + s // page, mask=inside, other=0)
    rows = block.to(tl.int64) * stride_cb + (s % page) * stride_cs
    k = tl.load(
        cache_ptr + rows[:, None] + spread[None, :],
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
    cache: picks.Cache,
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
        cache: The call's paged cache, its indexer keys beside its keys and
            values.
        rows: Each query's request's block table, (Q, blocks).
        n: How many keys to score.
        relu: Whether to ReLU the dot products.
        groups: The first of this rank's groups and how many, picking apart; None
            for all heads' scores summed.
    """
    first, count = groups or (0, 0)
    shape = (q.shape[0], count, n) if groups else (q.shape[0], n)
    out = torch.empty(shape, dtype=torch.float32, device=q.device)
    block_s, pages = 64, cache.cache
    grid = (q.shape[0], triton.cdiv(n, block_s))
    _scores_kernel[grid](
        q,
        w,
        pages,
        rows,
        out,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        rows.stride(0),
        out.stride(0),
        out.stride(1) if groups else 0,
        n,
        first,
        cache.channels,
        heads=q.shape[1],
        dims=q.shape[2],
        share=pages.shape[3] - cache.channels,
        page=pages.shape[2],
        relu=relu,
        groups=count,
        block_h=_block(q.shape[1]),
        block_i=_block(q.shape[2]),
        block_s=block_s,
    )
    return out


@triton.jit
def _prompt_scores_kernel(
    q_ptr,
    w_ptr,
    cache_ptr,
    table_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_wt,
    stride_cb,
    stride_cg,
    stride_cs,
    stride_ot,
    stride_og,
    pos_ptr,
    window,
    n_queries,
    n_keys,
    first,
    start,
    heads: tl.constexpr,
    dims: tl.constexpr,
    share: tl.constexpr,
    page: tl.constexpr,
    relu: tl.constexpr,
    groups: tl.constexpr,
    ranked: tl.constexpr,
    best: tl.constexpr,
    block_t: tl.constexpr,
    block_i: tl.constexpr,
    block_s: tl.constexpr,
):
    """A tile of one request's queries' scores of a tile of its keys, ``sum_j w_j
    ReLU(q_j . k_s)``, or each of the `groups` heads' from `first` on: each head's
    dots by the tensor cores in the inputs' dtype, summed in float32. `ranked`
    stores their ranking keys (`_ranks`) instead, and `best` the best a query sees
    of the tile's keys, a block of MSA's (`_best`)."""
    part_t, part_s = tl.program_id(0), tl.program_id(1)
    t = part_t * block_t + tl.arange(0, block_t)
    s = part_s * block_s + tl.arange(0, block_s)
    i = tl.arange(0, block_i)
    on_t, on_s, on_i = t < n_queries, s < n_keys, i < dims
    spread = (i // share) * stride_cg + start + i % share
    block = tl.load(table_ptr + s // page, mask=on_s, other=0)
    rows = block.to(tl.int64) * stride_cb + (s % page) * stride_cs
    k = tl.load(
        cache_ptr + rows[:, None] + spread[None, :],
        mask=on_s[:, None] & on_i[None, :],
        other=0.0,
    ).to(q_ptr.dtype.element_ty)
    acc = tl.zeros([block_t, block_s], tl.float32)
    for j in tl.static_range(heads):
        q = tl.load(
            q_ptr + t[:, None] * stride_qt + j * stride_qh + i[None, :],
            mask=on_t[:, None] & on_i[None, :],
            other=0.0,
        )
        dots = tl.dot(q, tl.trans(k), input_precision="ieee")
        if relu:
            dots = tl.maximum(dots, 0.0)
        w = tl.load(w_ptr + t * stride_wt + j, mask=on_t, other=0.0).to(tl.float32)
        if groups:  # each of this rank's groups by its own head
            if (j >= first) & (j < first + groups):
                at = out_ptr + t * stride_ot + (j - first) * stride_og
                mine = dots * w[:, None]
                if best:
                    top = _best(mine, pos_ptr, t, s, on_t, on_s, window)
                    tl.store(at + part_s, top, mask=on_t)
                else:
                    if ranked:
                        mine = _ranks(mine, pos_ptr, t, s, on_t, window)
                    keep = on_t[:, None] & on_s[None, :]
                    tl.store(at[:, None] + s[None, :], mine, mask=keep)
        else:
            acc += dots * w[:, None]
    if not groups:
        at = out_ptr + t * stride_ot
        if best:
            top = _best(acc, pos_ptr, t, s, on_t, on_s, window)
            tl.store(at + part_s, top, mask=on_t)
        else:
            if ranked:
                acc = _ranks(acc, pos_ptr, t, s, on_t, window)
            keep = on_t[:, None] & on_s[None, :]
            tl.store(at[:, None] + s[None, :], acc, mask=keep)


@triton.jit
def _seen(pos_ptr, t, s, on_t, window):
    """Which keys of a tile each query sees: those up to it, and in its `window`
    (none below 0)."""
    p = tl.load(pos_ptr + t, mask=on_t, other=0)[:, None]
    seen = s[None, :] <= p
    return seen & ((window < 0) | (p - s[None, :] < window))


@triton.jit
def _best(score, pos_ptr, t, s, on_t, on_s, window):
    """Each query's best score of the tile's keys it sees, -inf for none."""
    seen = _seen(pos_ptr, t, s, on_t, window) & on_s[None, :]
    return tl.max(tl.where(seen, score, float("-inf")), axis=1)


@triton.jit
def _ranks(score, pos_ptr, t, s, on_t, window):
    """The ranking keys of a tile of scores, as `bux_vllm.picks.top` makes them: the
    score's ordered bits above the key's index's complement, -1 for a key the query
    does not see (after it, or out of its `window`; none below 0)."""
    u = score.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    order = tl.where((u >> 31) == 1, ~u & 0xFFFFFFFF, u | 0x80000000)
    key = (order << 31) | (0x7FFFFFFF - s[None, :].to(tl.int64))
    return tl.where(_seen(pos_ptr, t, s, on_t, window), key, -1)


def prompt_scores(
    q: torch.Tensor,
    w: torch.Tensor,
    cache: picks.Cache,
    rows: torch.Tensor,
    n: int,
    relu: bool,
    groups: tuple[int, int] | None = None,
    seen: tuple[torch.Tensor, int | None] | None = None,
    best: int = 0,
) -> torch.Tensor:
    """`scores` of one request's queries, which share its keys: tiles of queries by
    tiles of keys, each key read once a tile of queries; or, with `seen`, their
    ranking keys, as `bux_vllm.picks.top` ranks them, in int64, or with `best` too
    the best score each query sees of each block of `best` keys (MSA's, `BLOCKS`),
    -inf for none, in float32.

    Args:
        q: The indexer's queries, (Q, H, i), the last axis contiguous.
        w: Their heads' weights, (Q, H), contiguous.
        cache: The request's paged cache, its indexer keys beside its keys and
            values.
        rows: The request's block table, (1, blocks).
        n: How many keys to score.
        relu: Whether to ReLU the dot products.
        groups: The first of this rank's groups and how many, picking apart; None
            for all heads' scores summed.
        seen: The queries' positions, (Q,), and the layer's window (None for
            none), by which a query sees a key: to rank, -1 for a key it does not.
        best: The keys of MSA's blocks, to find each block's best score a query
            sees (with `seen`); 0 for none.
    """
    first, count = groups or (0, 0)
    block_t, block_s = 64, best or 64
    columns = triton.cdiv(n, block_s) if best else n
    shape = (q.shape[0], count, columns) if groups else (q.shape[0], columns)
    dtype = torch.int64 if seen is not None and not best else torch.float32
    out = torch.empty(shape, dtype=dtype, device=q.device)
    positions, window = seen if seen is not None else (rows, None)
    pages = cache.cache
    grid = (triton.cdiv(q.shape[0], block_t), triton.cdiv(n, block_s))
    _prompt_scores_kernel[grid](
        q,
        w,
        pages,
        rows,
        out,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        out.stride(0),
        out.stride(1) if groups else 0,
        positions,
        -1 if window is None else window,
        q.shape[0],
        n,
        first,
        cache.channels,
        heads=q.shape[1],
        dims=q.shape[2],
        share=pages.shape[3] - cache.channels,
        page=pages.shape[2],
        relu=relu,
        groups=count,
        ranked=seen is not None,
        best=bool(best),
        block_t=block_t,
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
    cache: picks.Cache,
    rows: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, Picked]:
    """`bux_vllm.picks.attend_segment` by the kernels: each chunk of queries' scores
    (`scores`), its picks as bux ranks them, and its attention over them (`attend`);
    the outputs, and the picks padded to `bux_vllm.picks.most` keys a query."""
    count, shared = q.shape[0], rows.shape[0] == 1
    table = rows.expand(count, -1)
    span = torch.arange(n, device=q.device)
    keys = cache.keys(rows, span[None])[0] if shared and spec.mean else None
    per = picks._per_query(spec, q, index[0], cache.kv, n, shared or not spec.mean)
    chunk, outs, kept = max(1, picks.BUDGET // per), [], []
    for a in range(0, count, chunk):
        cut = slice(a, a + chunk)
        iq, iw = (t[cut].contiguous() for t in index)
        where = (positions[cut], rows if shared else table[cut], n)
        picked = _choose(spec, (iq, iw), cache, keys, where)
        outs.append(attend(spec, q[cut].contiguous(), cache.kv, table[cut], picked))
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
    cache: picks.Cache,
    keys: torch.Tensor | None,
    where: tuple[torch.Tensor, torch.Tensor, int],
) -> Picked:
    """The keys the queries keep, as bux picks them, from the indexer's queries and
    weights `index`: QSA's blocks scored by PyTorch, which pools them, over `keys`,
    (n, i), which a request's queries share, or each query's request's gathered;
    else by `prompt_scores` for one request's queries, `scores` for single queries
    of several. `where` are the queries' positions, their requests' block tables
    (one row for one request's) and the keys' count."""
    positions, table, n = where
    groups = spec.groups if spec.per_group else None
    one = table.shape[0] == 1
    at, size = (positions.contiguous(), spec.window), spec.block_size
    if one and size == 1:  # DSA's, ranked in the kernel
        ranks = prompt_scores(*index, cache, table, n, spec.relu, groups, at)
        return picks.best_of(ranks, spec.top_k)
    see = picks.seen(positions, n, spec.window)
    if one and not spec.mean and size in BLOCKS:  # MSA's blocks' best, by the kernel
        best = prompt_scores(*index, cache, table, n, spec.relu, groups, at, size)
        return picks.blocks_of_best(best, see, positions, size, spec.top_k)
    if spec.mean:
        if keys is None:
            span = torch.arange(n, device=positions.device)
            keys = cache.keys(table, span.expand(positions.shape[0], n))
        return picks.choose(spec, *index, keys, see, positions)
    score = prompt_scores if one else scores
    s = score(*index, cache, table, n, spec.relu, groups)
    if spec.block_size > 1:
        return picks.best_blocks(s, see, positions, spec.block_size, spec.top_k)
    return picks.tokens(s, see, spec.top_k)
