"""Triton kernels that prepare a sparse call's step on vLLM's DSA kernels, each in
one launch where PyTorch takes one for each of its steps: the step's keys and
values written into the call's paged cache, in FP8 in an FP8 cache, with each
token's indexer key in FP8 and its float32 scale in DeepGEMM's paged layout
(`write`); the indexer's queries in FP8, their heads' weights times their scales
(`queries`); and each query's picks with their count, as vLLM's sparse GQA
attention reads them (`packed`).

They compute what `bux_vllm.dsa` and `bux_vllm.sparse.PagedCache` do in PyTorch,
bit for bit: the scales by IEEE division, the FP8 values rounded to nearest even
(e4m3 saturating at 448, as the PyTorch path clamps them). A padding token (slot
-1) writes nothing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
FLOOR = 1e-4  # the least scale of a key or query head, times FP8_MAX
SCALE_BYTES = 4  # a key's float32 scale, after a block's keys


def _block(n: int) -> int:
    """A power of two at least `n`."""
    return triton.next_power_of_2(max(n, 1))


@triton.jit
def _part(
    row_ptr,
    src_ptr,
    stride_sh,
    stride_cg,
    first,
    scale,
    most,
    heads: tl.constexpr,
    dims: tl.constexpr,
    fp8: tl.constexpr,
    block_g: tl.constexpr,
    block_d: tl.constexpr,
):
    """Store a token's `dims` channels of each kv head, from `src_ptr`, in its row
    of the cache, `row_ptr`, from channel `first` on: in an `fp8` cache divided by
    `scale` and clamped to `most`."""
    g = tl.arange(0, block_g)[:, None]
    d = tl.arange(0, block_d)[None, :]
    on = (g < heads) & (d < dims)
    x = tl.load(src_ptr + g * stride_sh + d, mask=on, other=0.0)
    if fp8:
        y = x.to(tl.float32)
        y = tl.div_rn(y, tl.zeros_like(y) + scale)
        x = tl.minimum(tl.maximum(y, -most), most)
    dst = row_ptr + g * stride_cg + first + d
    tl.store(dst, x.to(row_ptr.dtype.element_ty), mask=on)


@triton.jit
def _scaled(x, fp8_max, floor):
    """The FP8 scale of each row of x, (rows, dims): its largest magnitude, `floor`
    at least, over `fp8_max`, by IEEE division."""
    most = tl.maximum(tl.max(tl.abs(x), 1), floor)
    return tl.div_rn(most, tl.zeros_like(most) + fp8_max)


@triton.jit
def _index_key(
    base_ptr,
    key_ptr,
    offset,
    block_size,
    fp8_max,
    floor,
    dims: tl.constexpr,
    block_i: tl.constexpr,
):
    """Store a token's indexer key, from `key_ptr`, at `offset` of its block's
    bytes, `base_ptr`: in FP8 e4m3 at its scale (`_scaled`), and the scale in
    float32 after the block's keys."""
    i = tl.arange(0, block_i)[None, :]
    on = i < dims
    x = tl.load(key_ptr + i, mask=on, other=0.0).to(tl.float32)
    scale = _scaled(x, fp8_max, floor)
    values = tl.div_rn(x, tl.zeros_like(x) + scale[:, None])
    keys = (base_ptr + offset * dims + i).to(tl.pointer_type(tl.float8e4nv))
    tl.store(keys, values.to(tl.float8e4nv), mask=on)
    at = base_ptr + block_size * dims + offset * 4 + tl.arange(0, 1)
    tl.store(at.to(tl.pointer_type(tl.float32)), scale)


@triton.jit
def _write_kernel(
    k_ptr,
    v_ptr,
    key_ptr,
    slot_ptr,
    cache_ptr,
    raw_ptr,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_it,
    stride_cb,
    stride_cg,
    stride_cs,
    stride_rb,
    region,
    block_size,
    k_scale,
    v_scale,
    most,
    fp8_max,
    floor,
    heads: tl.constexpr,
    dk: tl.constexpr,
    dv: tl.constexpr,
    index_dims: tl.constexpr,
    fp8: tl.constexpr,
    block_g: tl.constexpr,
    block_k: tl.constexpr,
    block_v: tl.constexpr,
    block_i: tl.constexpr,
):
    """A token's keys and values at its slot of the cache, and its indexer key
    (with `index_dims`) at the slot's place in the region of each block that
    starts `region` bytes in; nothing for a padding token's slot of -1."""
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t)
    if slot >= 0:
        block, offset = slot // block_size, slot % block_size
        row = cache_ptr + block * stride_cb + offset * stride_cs
        k_row = k_ptr + t * stride_kt
        _part(
            row,
            k_row,
            stride_kh,
            stride_cg,
            0,
            k_scale,
            most,
            heads,
            dk,
            fp8,
            block_g,
            block_k,
        )
        v_row = v_ptr + t * stride_vt
        _part(
            row,
            v_row,
            stride_vh,
            stride_cg,
            dk,
            v_scale,
            most,
            heads,
            dv,
            fp8,
            block_g,
            block_v,
        )
        if index_dims > 0:
            base = raw_ptr + block * stride_rb + region
            key = key_ptr + t * stride_it
            _index_key(
                base, key, offset, block_size, fp8_max, floor, index_dims, block_i
            )


def write(
    cache: torch.Tensor,
    slots: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keys: torch.Tensor | None = None,
    first: int = 0,
    scales: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Write the tokens' keys and values, and their indexer keys, at vLLM's
    `slots`, (T,), in one launch.

    Args:
        cache: The keys and values, ``[blocks, g, block_size, dk + dv]``, in the
            model's dtype or FP8 (e4m3 or e5m2), within a call's whole cache.
        slots: Each token's slot; -1 for padding, which writes nothing.
        k: The keys, (T, g, dk), each head's contiguous.
        v: The values, (T, g, dv), each head's contiguous.
        keys: The indexer keys, (T, i), contiguous, or None: each in FP8 e4m3 with
            its float32 scale, as `bux_vllm.dsa.Region` lays them, in the head
            slots of `cache`'s storage from `first` on.
        first: The first head slot of the indexer keys.
        scales: An FP8 cache's scales of its keys and values, which it holds
            divided by them.
    """
    n, g, dk = k.shape
    fp8 = cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    most = torch.finfo(cache.dtype).max if fp8 else 0.0
    dims = 0 if keys is None else keys.shape[1]
    keys = k if keys is None else keys
    raw = cache.view(torch.uint8)
    slot_bytes = cache.stride(1) * cache.element_size()
    if not n:
        return
    _write_kernel[(n,)](
        k,
        v,
        keys,
        slots,
        cache,
        raw,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        keys.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        raw.stride(0),
        first * slot_bytes,
        cache.shape[2],
        scales[0],
        scales[1],
        most,
        FP8_MAX,
        FLOOR,
        heads=g,
        dk=dk,
        dv=v.shape[2],
        index_dims=dims,
        fp8=fp8,
        block_g=_block(g),
        block_k=_block(dk),
        block_v=_block(v.shape[2]),
        block_i=_block(dims),
        num_warps=1,
    )


@triton.jit
def _queries_kernel(
    q_ptr,
    w_ptr,
    q8_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_wt,
    stride_8t,
    stride_8h,
    stride_ot,
    fp8_max,
    floor,
    heads: tl.constexpr,
    padded: tl.constexpr,
    dims: tl.constexpr,
    block_h: tl.constexpr,
    block_d: tl.constexpr,
):
    """A token's query heads in FP8 at their scales (`_scaled`), and their
    weights times the scales; the heads past `heads`, up to `padded`, zeros."""
    t = tl.program_id(0)
    h = tl.arange(0, block_h)
    d = tl.arange(0, block_d)
    on = (h[:, None] < heads) & (d[None, :] < dims)
    q = tl.load(
        q_ptr + t * stride_qt + h[:, None] * stride_qh + d[None, :],
        mask=on,
        other=0.0,
    ).to(tl.float32)
    scale = _scaled(q, fp8_max, floor)
    q8 = tl.div_rn(q, tl.zeros_like(q) + scale[:, None])
    kept = (h[:, None] < padded) & (d[None, :] < dims)
    dst = q8_ptr + t * stride_8t + h[:, None] * stride_8h + d[None, :]
    tl.store(dst, q8.to(q8_ptr.dtype.element_ty), mask=kept)
    w = tl.load(w_ptr + t * stride_wt + h, mask=h < heads, other=0.0)
    tl.store(out_ptr + t * stride_ot + h, w.to(tl.float32) * scale, mask=h < padded)


def queries(
    q: torch.Tensor, w: torch.Tensor, heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The indexer's queries in FP8, (T, heads, D), and their heads' weights times
    their scales in float32, (T, heads), padded with heads of no weight to `heads`,
    as `bux_vllm.dsa.queries` gives them, in one launch.

    Args:
        q: The queries, (T, H, D), each head's contiguous.
        w: Their heads' weights, (T, H), each token's contiguous.
        heads: The heads to pad to, H at least.
    """
    n, h, d = q.shape
    q8 = torch.empty(n, heads, d, dtype=FP8, device=q.device)
    weights = torch.empty(n, heads, dtype=torch.float32, device=q.device)
    if not n:
        return q8, weights
    _queries_kernel[(n,)](
        q,
        w,
        q8,
        weights,
        q.stride(0),
        q.stride(1),
        w.stride(0),
        q8.stride(0),
        q8.stride(1),
        weights.stride(0),
        FP8_MAX,
        FLOOR,
        heads=h,
        padded=heads,
        dims=d,
        block_h=_block(heads),
        block_d=_block(d),
        num_warps=1,
    )
    return q8, weights


@triton.jit
def _packed_kernel(
    picked_ptr,
    out_ptr,
    req_ptr,
    stride_pt,
    stride_ot,
    k: tl.constexpr,
    whole: tl.constexpr,
    decoding: tl.constexpr,
    block_k: tl.constexpr,
):
    """A query's picks, then their count (those of 0 on, or all of them if
    `whole`), and its request: its own index among decoded tokens, else 0."""
    t = tl.program_id(0)
    j = tl.arange(0, block_k)
    on = j < k
    picked = tl.load(picked_ptr + t * stride_pt + j, mask=on, other=-1)
    tl.store(out_ptr + t * stride_ot + j, picked, mask=on)
    if whole:
        tl.store(out_ptr + t * stride_ot + k, k)
    else:
        tl.store(out_ptr + t * stride_ot + k, tl.sum((picked >= 0).to(tl.int32), 0))
    if decoding:
        tl.store(req_ptr + t, t)
    else:
        tl.store(req_ptr + t, 0)


def packed(
    picked: torch.Tensor, decoding: bool, whole: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Picks, (Q, K) int32 positions with -1 for none, as vLLM's sparse GQA
    attention reads them, in one launch: (Q, K + 1), their count last, those of 0
    on (DSA's, whose valid ones come first) or all K if `whole` (QSA's, read
    whole); and each query's request, (Q,) int32, its own index among `decoding`
    tokens, else 0 (one prompt's)."""
    n, k = picked.shape
    out = torch.empty(n, k + 1, dtype=torch.int32, device=picked.device)
    requests = torch.empty(n, dtype=torch.int32, device=picked.device)
    if not n:
        return out, requests
    _packed_kernel[(n,)](
        picked,
        out,
        requests,
        picked.stride(0),
        out.stride(0),
        k=k,
        whole=whole,
        decoding=decoding,
        block_k=_block(k),
        num_warps=4,
    )
    return out, requests
