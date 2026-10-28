"""Triton kernels for the plugin's sparse attention (`bux_vllm.sparse`), which read
vLLM's paged caches in place where the PyTorch path (`bux_vllm.picks`) gathers the
keys into memory first: the indexer's score of every key of each query's request
(`scores`, and `prompt_scores` for a prompt's queries, a tile of them for each key
read, and `pooled_ranks` for QSA's decoded tokens), each query's softmax over the keys it picked (`attend`), and a ``csa``
query's over its window and the compressed latents it keeps (`csa_attend`), in
float32.

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
import torch.nn.functional as F
import triton
import triton.language as tl

from . import compressed, picks
from .picks import Picked, Sparse

#: Which path the sparse ops take: ``triton``, ``torch``, or unset for the kernels
#: in 16-bit models alone; read once.
CHOICE = os.environ.get("BUX_VLLM_SPARSE_KERNELS")
MOST_DIMS = 256  # the widest key or value head the kernels take
BLOCKS = (16, 32, 64, 128)  # MSA's blocks whose best a prompt's kernel finds


def enabled(dtype: torch.dtype) -> bool:
    """Whether the kernels run in a model of `dtype`, by `CHOICE`: in a 16-bit one
    unless it names a path, in any under ``triton`` or ``native``."""
    if CHOICE is None:
        return dtype in (torch.bfloat16, torch.float16)
    return CHOICE in ("triton", "native")


def takes(spec: Sparse, dims: tuple[int, int], dtype: torch.dtype) -> bool:
    """Whether the kernels compute the layer of `spec`, its key and value heads of
    `dims`, in a model of `dtype`: not MLA's latents, and by `enabled`."""
    return enabled(dtype) and max(dims) <= MOST_DIMS and not spec.latent


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
    pos_ptr,
    window,
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
    block_h: tl.constexpr,
    block_i: tl.constexpr,
    block_s: tl.constexpr,
):
    """One query's scores of `block_s` keys: ``sum_j w_j ReLU(q_j . k_s)``, or each
    of the `groups` heads' from `first` on, ``w_j ReLU(q_j . k_s)``, with `groups`;
    a key's channels `share` a kv head, from channel `start` of its row on.
    `ranked` stores their ranking keys instead, and `best` the best a query sees
    of the tile's keys, a block of MSA's (as `_prompt_scores_kernel`)."""
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
    seen = inside
    if ranked or best:
        seen = inside & _sees(tl.load(pos_ptr + t), s, window)
    if groups:  # each of this rank's groups by its own head
        mine = on_h & (h >= first) & (h < first + groups)
        rows_at = out_ptr + t * stride_ot + (h - first) * stride_og
        if best:
            top = tl.max(tl.where(seen[None, :], weighted, float("-inf")), axis=1)
            tl.store(rows_at + part, top, mask=mine)
        else:
            if ranked:
                weighted = tl.where(seen[None, :], _order(weighted, s[None, :]), -1)
            keep = mine[:, None] & inside[None, :]
            tl.store(rows_at[:, None] + s[None, :], weighted, mask=keep)
    else:
        total = tl.sum(weighted, axis=0)
        if best:
            top = tl.max(tl.where(seen, total, float("-inf")), axis=0)
            tl.store(out_ptr + t * stride_ot + part, top)
        else:
            if ranked:
                total = tl.where(seen, _order(total, s), -1)
            tl.store(out_ptr + t * stride_ot + s, total, mask=inside)


def scores(
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
    """Each query's indexer score of each of its request's first `n` keys, (Q, n) in
    float32, as `bux_vllm.picks.scores` gives it summed over the heads; or each
    head's of this rank's groups, (Q, G, n), as it gives them with `per_group`; or
    with `seen` and `best`, their ranking keys or blocks' best, as `prompt_scores`
    gives them.

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
        seen: The queries' positions, (Q,), and the layer's window, as
            `prompt_scores` takes them.
        best: The keys of MSA's blocks, as `prompt_scores` takes them.
    """
    first, count = groups or (0, 0)
    block_s, pages = best or 64, cache.cache
    columns = triton.cdiv(n, block_s) if best else n
    shape = (q.shape[0], count, columns) if groups else (q.shape[0], columns)
    dtype = torch.int64 if seen is not None and not best else torch.float32
    out = torch.empty(shape, dtype=dtype, device=q.device)
    positions, window = seen if seen is not None else (rows, None)
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
        positions,
        -1 if window is None else window,
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
        block_h=_block(q.shape[1]),
        block_i=_block(q.shape[2]),
        block_s=block_s,
    )
    return out


@triton.jit
def _pooled_part(
    cache_ptr,
    table_ptr,
    t,
    stride_tt,
    stride_cb,
    stride_cg,
    stride_cs,
    b,
    on_b,
    c,
    on_c,
    start,
    share: tl.constexpr,
    page: tl.constexpr,
    size: tl.constexpr,
    block_b: tl.constexpr,
    block_c: tl.constexpr,
):
    """The mean of each block's `size` raw indexer keys at the channels c, (B, C),
    in float32, from the blocks b of query t's request."""
    spread = (c // share) * stride_cg + start + c % share
    out = tl.zeros([block_b, block_c], tl.float32)
    for r in tl.static_range(size):
        at = b * size + r
        block = tl.load(table_ptr + t * stride_tt + at // page, mask=on_b, other=0)
        rows = block.to(tl.int64) * stride_cb + (at % page) * stride_cs
        out += tl.load(
            cache_ptr + rows[:, None] + spread[None, :],
            mask=on_b[:, None] & on_c[None, :],
            other=0.0,
        ).to(tl.float32)
    return out / size


@triton.jit
def _gained(x, gain_ptr, c, on_c, zero_centered: tl.constexpr):
    """x, (B, C), times the norm's gain at the channels c: its weight, or 1 + it
    zero-centered, made in the weight's dtype, as `bux_vllm.common.Norm` makes it."""
    w = tl.load(gain_ptr + c, mask=on_c, other=0.0)
    if zero_centered:
        w = w + 1.0
    return x * w.to(tl.float32)[None, :]


@triton.jit
def _head_dots(q_ptr, t, stride_qt, stride_qh, h, on_h, c, on_c, k):
    """Each head's dots with the keys k, (B, C), over the channels c: (H, B)."""
    q = tl.load(
        q_ptr + t * stride_qt + h[:, None] * stride_qh + c[None, :],
        mask=on_h[:, None] & on_c[None, :],
        other=0.0,
    ).to(tl.float32)
    return tl.dot(q, tl.trans(k), input_precision="ieee")


@triton.jit
def _pooled_kernel(
    q_ptr,
    w_ptr,
    cache_ptr,
    table_ptr,
    out_ptr,
    pos_ptr,
    gain_ptr,
    bias_ptr,
    freq_ptr,
    stride_qt,
    stride_qh,
    stride_wt,
    stride_cb,
    stride_cg,
    stride_cs,
    stride_tt,
    stride_ot,
    n_blocks,
    start,
    window,
    eps,
    rope_gain,
    heads: tl.constexpr,
    dims: tl.constexpr,
    share: tl.constexpr,
    page: tl.constexpr,
    size: tl.constexpr,
    rotated: tl.constexpr,
    interleaved: tl.constexpr,
    centered: tl.constexpr,
    gained: tl.constexpr,
    zero_centered: tl.constexpr,
    scale_input: tl.constexpr,
    biased: tl.constexpr,
    relu: tl.constexpr,
    block_h: tl.constexpr,
    block_j: tl.constexpr,
    block_r: tl.constexpr,
    block_b: tl.constexpr,
):
    """One decoded token's ranking keys of a tile of QSA's complete blocks, as
    `bux_vllm.picks.choose` scores them: each block's `size` raw keys pooled, the
    indexer's key norm and RoPE at the block's first key (pairs of its first
    `rotated` channels), then ``sum_j w_j ReLU(q_j . k)``; -1 for a block it does
    not see all of. The channels go in three parts, each pair's first and second
    and the rest, so that RoPE pairs them in place."""
    t, part = tl.program_id(0), tl.program_id(1)
    b = part * block_b + tl.arange(0, block_b)
    on_b = b < n_blocks
    half = rotated // 2
    j, r = tl.arange(0, block_j), tl.arange(0, block_r)
    on_j, on_r = j < half, r < dims - rotated
    if interleaved:
        ca, cb = 2 * j, 2 * j + 1
    else:
        ca, cb = j, j + half
    cr = rotated + r
    ka = _pooled_part(
        cache_ptr,
        table_ptr,
        t,
        stride_tt,
        stride_cb,
        stride_cg,
        stride_cs,
        b,
        on_b,
        ca,
        on_j,
        start,
        share,
        page,
        size,
        block_b,
        block_j,
    )
    kb = _pooled_part(
        cache_ptr,
        table_ptr,
        t,
        stride_tt,
        stride_cb,
        stride_cg,
        stride_cs,
        b,
        on_b,
        cb,
        on_j,
        start,
        share,
        page,
        size,
        block_b,
        block_j,
    )
    kr = _pooled_part(
        cache_ptr,
        table_ptr,
        t,
        stride_tt,
        stride_cb,
        stride_cg,
        stride_cs,
        b,
        on_b,
        cr,
        on_r,
        start,
        share,
        page,
        size,
        block_b,
        block_r,
    )
    if gained and scale_input:
        ka = _gained(ka, gain_ptr, ca, on_j, zero_centered)
        kb = _gained(kb, gain_ptr, cb, on_j, zero_centered)
        kr = _gained(kr, gain_ptr, cr, on_r, zero_centered)
    if centered:
        mean = (tl.sum(ka, 1) + tl.sum(kb, 1) + tl.sum(kr, 1)) / dims
        ka = tl.where(on_j[None, :], ka - mean[:, None], 0.0)
        kb = tl.where(on_j[None, :], kb - mean[:, None], 0.0)
        kr = tl.where(on_r[None, :], kr - mean[:, None], 0.0)
    square = tl.sum(ka * ka, 1) + tl.sum(kb * kb, 1) + tl.sum(kr * kr, 1)
    scale = tl.rsqrt(square / dims + eps)[:, None]
    ka, kb, kr = ka * scale, kb * scale, kr * scale
    if gained and not scale_input:
        ka = _gained(ka, gain_ptr, ca, on_j, zero_centered)
        kb = _gained(kb, gain_ptr, cb, on_j, zero_centered)
        kr = _gained(kr, gain_ptr, cr, on_r, zero_centered)
    if biased:
        ka += tl.load(bias_ptr + ca, mask=on_j, other=0.0).to(tl.float32)[None, :]
        kb += tl.load(bias_ptr + cb, mask=on_j, other=0.0).to(tl.float32)[None, :]
        kr += tl.load(bias_ptr + cr, mask=on_r, other=0.0).to(tl.float32)[None, :]
    h = tl.arange(0, block_h)
    on_h = h < heads
    dots = _head_dots(q_ptr, t, stride_qt, stride_qh, h, on_h, cr, on_r, kr)
    if rotated:
        freq = tl.load(freq_ptr + j, mask=on_j, other=0.0)
        angles = (b * size).to(tl.float32)[:, None] * freq[None, :]
        sin, cos = rope_gain * tl.sin(angles), rope_gain * tl.cos(angles)
        ka, kb = ka * cos - kb * sin, kb * cos + ka * sin
        dots += _head_dots(q_ptr, t, stride_qt, stride_qh, h, on_h, ca, on_j, ka)
        dots += _head_dots(q_ptr, t, stride_qt, stride_qh, h, on_h, cb, on_j, kb)
    if relu:
        dots = tl.maximum(dots, 0.0)
    w = tl.load(w_ptr + t * stride_wt + h, mask=on_h, other=0.0).to(tl.float32)
    score = tl.sum(dots * w[:, None], axis=0)
    p = tl.load(pos_ptr + t)
    first = b * size
    full = on_b & (first + size - 1 <= p) & ((window < 0) | (p - first < window))
    tl.store(
        out_ptr + t * stride_ot + b, tl.where(full, _order(score, b), -1), mask=on_b
    )


def pooled_ranks(
    spec: Sparse,
    index: tuple[torch.Tensor, torch.Tensor],
    cache: picks.Cache,
    rows: torch.Tensor,
    positions: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Each decoded token's ranking keys of QSA's complete blocks of its request's
    first `n` keys, (Q, n // block_size) in int64, as `bux_vllm.picks.top` ranks
    `bux_vllm.picks.choose`'s scores of their pooled keys, -1 for a block it does not
    see all of: the pooling, the key norm and RoPE, in float32, in the kernel.

    Args:
        spec: The layer's settings, its indexer among them.
        index: The indexer's queries and weights, (Q, H, i) and (Q, H).
        cache: The call's paged cache, its indexer keys beside its keys and
            values.
        rows: Each token's request's block table, (Q, blocks).
        positions: The tokens' positions, (Q,).
        n: How many keys to read.
    """
    q, w = (x.contiguous() for x in index)
    ix, size, pages = spec.indexer, spec.block_size, cache.cache
    norm, rope = ix.k_norm, ix.rope
    count = n // size
    out = torch.empty(q.shape[0], count, dtype=torch.int64, device=q.device)
    zero = q.new_zeros(1, dtype=torch.float32)
    rotated = rope.dims if rope is not None else 0
    block_b = 16
    _pooled_kernel[(q.shape[0], triton.cdiv(max(count, 1), block_b))](
        q,
        w,
        pages,
        rows,
        out,
        positions.contiguous(),
        norm.weight if norm.weight is not None else zero,
        norm.bias if norm.bias is not None else zero,
        rope.inv_freq if rope is not None else zero,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        rows.stride(0),
        out.stride(0),
        count,
        cache.channels,
        -1 if spec.window is None else spec.window,
        norm.eps,
        rope.gain if rope is not None else 1.0,
        heads=q.shape[1],
        dims=q.shape[2],
        share=pages.shape[3] - cache.channels,
        page=pages.shape[2],
        size=size,
        rotated=rotated,
        interleaved=rope is not None and rope.layout == "interleaved",
        centered=norm.kind == "layernorm",
        gained=norm.weight is not None,
        zero_centered=norm.zero_centered,
        scale_input=norm.scale_input,
        biased=norm.bias is not None,
        relu=spec.relu,
        block_h=_block(q.shape[1]),
        block_j=_block(rotated // 2),
        block_r=_block(q.shape[2] - rotated),
        block_b=block_b,
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
def _sees(p, s, window):
    """Whether a query at position p sees the keys at s, broadcast: those up to it,
    and in its `window` (none below 0)."""
    return (s <= p) & ((window < 0) | (p - s < window))


@triton.jit
def _order(score, s):
    """The ranking keys of scores at the keys s, broadcast, as
    `bux_vllm.picks.top` makes them: the score's ordered bits above the key's
    index's complement."""
    u = score.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    order = tl.where((u >> 31) == 1, ~u & 0xFFFFFFFF, u | 0x80000000)
    return (order << 31) | (0x7FFFFFFF - s.to(tl.int64))


@triton.jit
def _seen(pos_ptr, t, s, on_t, window):
    """Which keys of a tile each query of a tile sees (`_sees`)."""
    p = tl.load(pos_ptr + t, mask=on_t, other=0)[:, None]
    return _sees(p, s[None, :], window)


@triton.jit
def _best(score, pos_ptr, t, s, on_t, on_s, window):
    """Each query's best score of the tile's keys it sees, -inf for none."""
    seen = _seen(pos_ptr, t, s, on_t, window) & on_s[None, :]
    return tl.max(tl.where(seen, score, float("-inf")), axis=1)


@triton.jit
def _ranks(score, pos_ptr, t, s, on_t, window):
    """The ranking keys of a tile of scores (`_order`), -1 for a key the query does
    not see (`_seen`)."""
    return tl.where(_seen(pos_ptr, t, s, on_t, window), _order(score, s[None, :]), -1)


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
    v_scale,
    heads_per: tl.constexpr,
    dk: tl.constexpr,
    dv: tl.constexpr,
    page: tl.constexpr,
    capped: tl.constexpr,
    sunk: tl.constexpr,
    fp8: tl.constexpr,
    block_h: tl.constexpr,
    block_dk: tl.constexpr,
    block_dv: tl.constexpr,
    block_k: tl.constexpr,
):
    """One query's heads of one kv head's group, over its picked keys: an online
    softmax of the scaled, capped logits, the sinks beside them. An `fp8` cache's
    values take bfloat16's tensor cores, their scale `v_scale` on the output (its
    keys' is in `scale`)."""
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
        if fp8:
            v = v.to(tl.bfloat16)
        step = tl.dot(p.to(v.dtype), v, input_precision="ieee").to(tl.float32)
        acc = acc * fade[:, None] + step
        top = new
    acc = acc * v_scale
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
    scales: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """Each query's heads' outputs over the keys it picked, (Q, h, dv) in the
    queries' dtype, as `bux_vllm.picks.attend` gives them.

    Args:
        spec: The layer's settings: the key dims, scale, softcap and sinks.
        q: The queries, (Q, h, dk), the last axis contiguous.
        cache: The paged cache of keys and values, (blocks, g, block_size, dk + dv),
            in the model's dtype or FP8 (e4m3 or e5m2).
        rows: Each query's request's block table, (Q, blocks).
        picked: The picked keys' positions and which are valid, (Q, K) each, or
            (Q, g, K) for each kv head's group.
        scales: An FP8 cache's scales of its keys and values, which it holds
            divided by them.
    """
    keys, valid = (t.contiguous() for t in picked)
    n, h, dk = q.shape
    g, dv = cache.shape[1], cache.shape[3] - spec.dk
    out = torch.empty(n, h, dv, dtype=q.dtype, device=q.device)
    fp8 = cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
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
        spec.scale * scales[0],
        spec.softcap or 1.0,
        scales[1],
        heads_per=h // g,
        dk=dk,
        dv=dv,
        page=cache.shape[2],
        capped=spec.softcap is not None,
        sunk=spec.sinks is not None,
        fp8=fp8,
        block_h=_block(h // g),
        block_dk=_block(dk),
        block_dv=_block(dv),
        block_k=32,
    )
    return out


@triton.jit
def _online(
    q,
    cache_ptr,
    rows,
    ok,
    d,
    on_d,
    top,
    total,
    acc,
    scale,
    softcap,
    capped: tl.constexpr,
):
    """An online softmax's step over a tile of keys, which are the values too (csa's
    latents), at `rows` of the cache where `ok`: the new running max, sum and
    output."""
    k = tl.load(
        cache_ptr + rows[:, None] + d[None, :],
        mask=ok[:, None] & on_d[None, :],
        other=0.0,
    )
    logits = tl.dot(q, tl.trans(k.to(tl.float32)), input_precision="ieee") * scale
    if capped:
        logits = softcap * (2.0 * tl.sigmoid(2.0 * logits / softcap) - 1.0)
    logits = tl.where(ok[None, :], logits, float("-inf"))
    new = tl.maximum(top, tl.max(logits, axis=1))
    safe = tl.where(new == float("-inf"), 0.0, new)
    p = tl.exp(logits - safe[:, None])
    fade = tl.exp(top - safe)
    total = total * fade + tl.sum(p, axis=1)
    step = tl.dot(p.to(k.dtype), k, input_precision="ieee").to(tl.float32)
    return new, total, acc * fade[:, None] + step


@triton.jit
def _csa_kernel(
    q_ptr,
    window_ptr,
    window_table_ptr,
    latents_ptr,
    latents_table_ptr,
    kept_ptr,
    seq_ptr,
    pos_ptr,
    sinks_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_wb,
    stride_wr,
    stride_wt,
    stride_lb,
    stride_lr,
    stride_lt,
    stride_kt,
    stride_ot,
    stride_oh,
    span,
    n_kept,
    scale,
    softcap,
    heads: tl.constexpr,
    dims: tl.constexpr,
    page: tl.constexpr,
    per_block: tl.constexpr,
    capped: tl.constexpr,
    sunk: tl.constexpr,
    block_h: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    """One token's heads of a tile: an online softmax over its window of `span`
    token latents, then the `n_kept` compressed latents it keeps (-1 for none), the
    sinks beside them; the latents are the keys and the values."""
    t, part = tl.program_id(0), tl.program_id(1)
    head = part * block_h + tl.arange(0, block_h)
    on_h = head < heads
    d = tl.arange(0, block_d)
    on_d = d < dims
    q = tl.load(
        q_ptr + t * stride_qt + head[:, None] * stride_qh + d[None, :],
        mask=on_h[:, None] & on_d[None, :],
        other=0.0,
    ).to(tl.float32)
    seq, p = tl.load(seq_ptr + t).to(tl.int64), tl.load(pos_ptr + t)
    top = tl.full([block_h], float("-inf"), tl.float32)
    total = tl.zeros([block_h], tl.float32)
    acc = tl.zeros([block_h, block_d], tl.float32)
    for start in range(0, span, block_k):
        j = start + tl.arange(0, block_k)
        near = p - (span - 1 - j)
        ok = (j < span) & (near >= 0)
        at = window_table_ptr + seq * stride_wt + near // page
        block = tl.load(at, mask=ok, other=0).to(tl.int64)
        rows = block * stride_wb + (near % page) * stride_wr
        top, total, acc = _online(
            q, window_ptr, rows, ok, d, on_d, top, total, acc, scale, softcap, capped
        )
    for start in range(0, n_kept, block_k):
        j = start + tl.arange(0, block_k)
        inside = j < n_kept
        kept = tl.load(kept_ptr + t * stride_kt + j, mask=inside, other=-1)
        ok = inside & (kept >= 0)
        at = latents_table_ptr + seq * stride_lt + kept // per_block
        block = tl.load(at, mask=ok, other=0).to(tl.int64)
        rows = block * stride_lb + (kept % per_block) * stride_lr
        top, total, acc = _online(
            q, latents_ptr, rows, ok, d, on_d, top, total, acc, scale, softcap, capped
        )
    kept_any = total > 0
    if sunk:
        sink = tl.load(sinks_ptr + head, mask=on_h, other=0.0).to(tl.float32)
        most = tl.maximum(top, sink)
        fade = tl.exp(top - most)
        total, acc = total * fade + tl.exp(sink - most), acc * fade[:, None]
    out = tl.where(
        kept_any[:, None], acc / tl.where(kept_any, total, 1.0)[:, None], 0.0
    )
    tl.store(
        out_ptr + t * stride_ot + head[:, None] * stride_oh + d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=on_h[:, None] & on_d[None, :],
    )


def takes_csa(dtype: torch.dtype, dims: int) -> bool:
    """Whether the kernel attends a ``csa`` call of heads of `dims` in a model of
    `dtype` (`csa_attend`), by `enabled`."""
    return enabled(dtype) and dims <= MOST_DIMS


def csa_attend(
    caches: compressed.Caches,
    steps: compressed.Steps,
    s: compressed.Settings,
    q: torch.Tensor,
    kept: torch.Tensor | None,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """`bux_vllm.compressed.attend` by a kernel, which reads the window's and the
    compressed latents in place: each query, q (T, h, d), over its window of token
    latents and the compressed latents it keeps (`kept`, by index, -1 for none), in
    one softmax with the sinks; (T, h, d)."""
    q = q.contiguous()
    n, h, d = q.shape
    span = min(s.window or steps.longest, steps.longest)
    window = caches.window
    latents = caches.compressed if kept is not None else None
    if latents is None or kept is None:
        latents, kept = window, q.new_full((n, 0), -1, dtype=torch.long)
    kept = kept.contiguous()
    out = torch.empty_like(q)
    block_h = min(_block(h), 64)
    _csa_kernel[(n, triton.cdiv(h, block_h))](
        q,
        window.cache,
        window.block_table,
        latents.cache,
        latents.block_table,
        kept,
        steps.seq.contiguous(),
        steps.positions.contiguous(),
        sinks if sinks is not None else q.new_zeros(1),
        out,
        q.stride(0),
        q.stride(1),
        window.cache.stride(0),
        window.cache.stride(1),
        window.block_table.stride(0),
        latents.cache.stride(0),
        latents.cache.stride(1),
        latents.block_table.stride(0),
        kept.stride(0),
        out.stride(0),
        out.stride(1),
        span,
        kept.shape[1],
        s.scale,
        s.softcap or 1.0,
        heads=h,
        dims=d,
        page=window.cache.shape[1],
        per_block=latents.cache.shape[1],
        capped=s.softcap is not None,
        sunk=sinks is not None,
        block_h=block_h,
        block_d=_block(d),
        block_k=32,
    )
    return out


def csa_select(
    caches: compressed.Caches,
    steps: compressed.Steps,
    s: compressed.Settings,
    q: torch.Tensor,
    w: torch.Tensor,
    picks_of: compressed.Picks | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`bux_vllm.compressed.select` by the kernels, which read the indexer keys in
    place: the decoded tokens' scores by `scores`, ranked by PyTorch, and each
    prompt chunk's ranking keys by `prompt_scores`; (T, top_k) by index, -1 for
    none, and no candidate pool. A call that picks inside a pool, or makes one
    (`picks_of`), picks by PyTorch's `bux_vllm.compressed.select`."""
    if picks_of is not None and (picks_of.inside is not None or picks_of.pooled):
        return compressed.select(caches, steps, s, q, w, picks_of)
    assert caches.index is not None and s.ratio is not None and s.top_k
    seen = (steps.positions + 1) // s.ratio
    most = steps.longest // s.ratio
    if most <= s.top_k:
        return compressed.every(seen, most), None
    kept = seen.new_full((seen.shape[0], s.top_k), -1)
    index = caches.index
    keys = picks.Cache(index.cache.unsqueeze(1), 0, index.cache.shape[-1])
    q, w = q.contiguous(), w.contiguous()
    if steps.decodes:
        c = slice(0, steps.decodes)
        rows = index.block_table[steps.seq[c]]
        sc = scores(q[c], w[c], keys, rows, most, s.relu)
        kept[c] = compressed.top(sc, seen[c], s.top_k)
    for start, n, seq, length in steps.prompts:
        c = slice(start, start + n)
        rows = index.block_table[seq : seq + 1]
        at = ((seen[c] - 1).contiguous(), None)  # a latent seen is one before seen
        ranks = prompt_scores(
            q[c], w[c], keys, rows, length // s.ratio, s.relu, None, at
        )
        picked, ok = picks.best_of(ranks, s.top_k)
        picked = torch.where(ok, picked, -1)
        kept[c] = F.pad(picked, (0, s.top_k - picked.shape[1]), value=-1)
    return kept, None


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
    per = _per_query(spec, index[0], n, shared)
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


def _per_query(spec: Sparse, iq: torch.Tensor, n: int, one: bool) -> int:
    """The elements of a query's largest intermediate on the kernels' path, of `n`
    keys, its request's alone if `one`: its picks, and its ranking keys (DSA's), its
    blocks' best scores and mask (MSA's), PyTorch's scores of every head and
    gathered keys (QSA's), or its scores, their mask and ranking keys."""
    groups = spec.groups[1] if spec.per_group else 1
    most = 2 * groups * picks.most(spec)  # int64 positions
    if spec.mean:
        return max(most, iq.shape[1] * n, 0 if one else n * iq.shape[2])
    if spec.block_size == 1:
        return max(most, 2 * groups * n)
    if spec.block_size in BLOCKS:
        return max(most, n + groups * -(-n // spec.block_size))
    return max(most, 3 * groups * n)


def attend_picked(
    spec: Sparse,
    q: torch.Tensor,
    picked: Picked,
    positions: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    n: int,
    scales: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """`bux_vllm.picks.attend_picked` by the kernels: the queries' attention over the
    keys another call picked, of those their own window sees; `scales` an FP8
    cache's (`attend`)."""
    keys, valid = picked[0].clamp(0, n - 1), picked[1]
    see = picks.seen(positions, n, spec.window)
    view = see if keys.ndim == 2 else see[:, None].expand(-1, keys.shape[1], -1)
    table = rows.expand(q.shape[0], -1)
    kept = (keys, valid & view.gather(-1, keys))
    return attend(spec, q.contiguous(), cache, table, kept, scales)


def _choose(
    spec: Sparse,
    index: tuple[torch.Tensor, torch.Tensor],
    cache: picks.Cache,
    keys: torch.Tensor | None,
    where: tuple[torch.Tensor, torch.Tensor, int],
) -> Picked:
    """The keys the queries keep, as bux picks them, from the indexer's queries and
    weights `index`: by `prompt_scores` for one request's queries, `scores` for
    single queries of several, DSA's keys ranked and MSA's blocks' best found in
    them; QSA's blocks scored by PyTorch, which pools them, over `keys`, (n, i),
    which a request's queries share, or each query's request's gathered. `where`
    are the queries' positions, their requests' block tables (one row for one
    request's) and the keys' count."""
    positions, table, n = where
    groups = spec.groups if spec.per_group else None
    one = table.shape[0] == 1
    at, size = (positions.contiguous(), spec.window), spec.block_size
    kernel = prompt_scores if one else scores
    if size == 1:  # DSA's, ranked in the kernel
        ranks = kernel(*index, cache, table, n, spec.relu, groups, at)
        return picks.best_of(ranks, spec.top_k)
    see = picks.seen(positions, n, spec.window)
    if spec.mean and not one:  # QSA's decoded tokens, pooled in the kernel
        ranks = pooled_ranks(spec, index, cache, table, positions, n)
        blocks, ok = picks.best_of(ranks, spec.top_k // size)
        return picks.with_own_block(blocks, ok, see, positions, spec.window, size)
    if not spec.mean and size in BLOCKS:  # MSA's blocks' best, by the kernel
        best = kernel(*index, cache, table, n, spec.relu, groups, at, size)
        return picks.blocks_of_best(best, see, positions, size, spec.top_k)
    if spec.mean:
        if keys is None:
            span = torch.arange(n, device=positions.device)
            keys = cache.keys(table, span.expand(positions.shape[0], n))
        return picks.choose(spec, *index, keys, see, positions)
    s = kernel(*index, cache, table, n, spec.relu, groups)  # MSA's other blocks
    return picks.best_blocks(s, see, positions, size, spec.top_k)
