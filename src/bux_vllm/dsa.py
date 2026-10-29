"""DSA's indexer on vLLM's own kernels, those vLLM runs DeepSeek-V3.2's indexer on:
a call's indexer keys cached in FP8 in DeepGEMM's paged layout, a prompt's queries
scored by DeepGEMM's MQA logits over the request's keys as vLLM gathers them
(``fp8_fp4_mqa_logits``), a decoded token's by its paged MQA logits
(``fp8_fp4_paged_mqa_logits``), and the best ``top_k`` kept by vLLM's top-k kernels.

bux's DSA indexer is DeepSeek's: a query scores a key ``sum_j w_j ReLU(q_j . k)``
and keeps the best ``top_k`` it sees. The kernels take it where its keys are single,
its dots ReLU'd, its heads' scores summed (no picks per group), its head dim 32, 64
or 128, its heads 128 at most, no window, and a top-k of 2048 at most (`takes`).
They score FP8 keys and queries, with a float32 scale for each key and each query
head, as vLLM's DeepSeek-V3.2 does, so a pick differs from bux's float32 one where
two scores are within FP8's rounding. They run in a 16-bit model; a float32 one
keeps the exact path (`bux_vllm.picks`).

QSA's indexer (Qwen3.8's, mean-pooled blocks) is DSA's over its blocks: a complete
block's key is the mean of its keys as projected, normalized and rotated to its
first position, and a query keeps the best ``top_k / block_size`` of the complete
blocks it sees, and its own block while that is unfinished. The keys as projected
are cached in the model's dtype (`Raw`); a block's key is written when its last
token comes, at the position of its index among the request's blocks (`pooled`), so
the kernels score a request's blocks as DSA's keys.

The keys live in the call's own cache (`bux_vllm.sparse.PagedCache`), in head slots
after its kv heads' (`Region`): each block's FP8 keys, then their scales, so that
they page with the keys and values, one cache a call, and prefix caching hits as
often as in a dense model. On a GPU, Triton kernels write a step's keys, quantize
its queries and pack its picks, each in one launch (`bux_vllm.caching`).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .picks import BUDGET, Sparse

#: Which path the sparse ops take (as `bux_vllm.kernels` reads it): ``native`` for
#: these kernels wherever they take the layer, or unset for them in 16-bit models.
CHOICE = os.environ.get("BUX_VLLM_SPARSE_KERNELS")
DIMS = (32, 64, 128)  # the indexer head dims DeepGEMM's MQA logits take
MOST_TOP_K = 2048  # the most keys vLLM's top-k kernels keep
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
SCALE_BYTES = 4  # a key's float32 scale, after a block's keys
PAGE = 64  # the tokens of a block that DeepGEMM's paged kernel reads


def takes(spec: Sparse, dims: int, heads: int, dtype: torch.dtype) -> bool:
    """Whether the kernels pick the keys of a layer of `spec`, whose indexer has
    `heads` of `dims`, in a model of `dtype` (by `CHOICE`). MLA's take them under
    ``native`` alone: their keys would take a head slot as wide as a latent, which
    doubles its cache."""
    if CHOICE is None:
        on = dtype in (torch.bfloat16, torch.float16) and not spec.latent
    else:
        on = CHOICE == "native"
    size = spec.block_size  # DSA's keys, or QSA's blocks within a cache block
    blocks = size == 1 or (spec.mean and PAGE % size == 0)
    fits = blocks and spec.relu and not spec.per_group and spec.window is None
    fits = fits and 0 < spec.top_k // spec.block_size <= MOST_TOP_K
    if not (on and fits and dims in DIMS and available()):
        return False
    return heads <= max(_head_sizes())


@functools.cache
def available() -> bool:
    """Whether the kernels run here: DeepGEMM installed, on an SM 9.0 or 10.x GPU."""
    if not torch.cuda.is_available():
        return False
    from vllm.utils.import_utils import has_deep_gemm

    return has_deep_gemm() and torch.cuda.get_device_capability()[0] in (9, 10)


def padded_heads(heads: int) -> int:
    """The query heads DeepGEMM scores with for an indexer of `heads`: as many,
    rounded up to a size its kernels take (`_head_sizes`), the rest of no weight."""
    return next(n for n in _head_sizes() if n >= heads)


@functools.cache
def _head_sizes() -> tuple[int, ...]:
    """The query heads DeepGEMM's MQA logits take: any power of two from 4 on SM
    10.x, 32 or 64 on SM 9.0."""
    sm90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
    return (32, 64) if sm90 else (4, 8, 16, 32, 64, 128)


def slots(dims: int, channels: int, itemsize: int) -> int:
    """The head slots of `channels` a token, of `itemsize` bytes each, that a
    `Region` of keys of `dims` takes."""
    return -(-(dims + SCALE_BYTES) // (channels * itemsize))


def quantized(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x, (..., D), in FP8 e4m3 and a float32 scale a row, as vLLM caches DeepSeek's
    indexer keys: the row's largest magnitude, 1e-4 at least, over 448."""
    scale = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-4) / FP8_MAX
    return (x.float() / scale).to(FP8), scale.squeeze(-1)


def queries(
    q: torch.Tensor, w: torch.Tensor, heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The indexer's queries, (T, H, D), in FP8, and their heads' weights, (T, H),
    times each query head's scale in float32, as DeepGEMM takes them; both padded
    with heads of no weight to `heads`. On a GPU by one Triton kernel, bit for bit
    (`bux_vllm.caching.queries`)."""
    if q.is_cuda:
        from . import caching

        return caching.queries(q, w, heads)
    q8, scale = quantized(q)
    weights = w.float() * scale
    pad = heads - q.shape[1]
    if pad:
        q8 = F.pad(q8.view(torch.uint8), (0, 0, 0, pad)).view(FP8)
        weights = F.pad(weights, (0, pad))
    return q8.contiguous(), weights.contiguous()


@dataclass(frozen=True)
class Region:
    """The FP8 indexer keys of a call's cache, ``[blocks, heads, block_size, C]``,
    in its head slots from `first` on, which must lie together within a block (a
    head-major layout): a block's bytes there are its ``block_size`` keys of `dims`,
    then their float32 scales, as DeepGEMM's paged kernel reads them.

    Attributes:
        cache: The call's cache.
        first: Its first slot of keys, after the kv heads'.
        dims: The keys' dims.
    """

    cache: torch.Tensor
    first: int
    dims: int

    @property
    def size(self) -> int:
        """The tokens of a block."""
        return self.cache.shape[2]

    @property
    def raw(self) -> torch.Tensor:
        """Each block's bytes of its slots, (blocks, bytes); a view, or an error."""
        c = self.cache.shape[3] * self.cache.dtype.itemsize
        part = self.cache[:, self.first : self.first + slots(self.dims, c, 1)]
        part = part.view(torch.uint8)
        return part.view(part.shape[0], -1)

    def write(self, slots: torch.Tensor, keys: torch.Tensor) -> None:
        """Write the tokens' keys, (T, dims), at vLLM's `slots`, (T,), in FP8 with
        their scales: padding's (slot -1) into vLLM's null block, block 0."""
        values, scales = quantized(keys)
        slots = slots.clamp(min=0)
        block, offset = slots // self.size, slots % self.size
        n, d, raw = self.size, self.dims, self.raw
        raw[:, : n * d].unflatten(1, (n, d))[block, offset] = values.view(torch.uint8)
        sf = raw[:, n * d : n * (d + SCALE_BYTES)].unflatten(1, (n, SCALE_BYTES))
        sf[block, offset] = scales.view(torch.uint8).view(-1, SCALE_BYTES)

    def paged(self) -> torch.Tensor:
        """The keys as DeepGEMM's paged MQA logits read them, (blocks, block_size,
        1, dims + 4) bytes."""
        raw, n, row = self.raw, self.size, self.dims + SCALE_BYTES
        return raw.as_strided((raw.shape[0], n, 1, row), (raw.stride(0), row, row, 1))

    def gathered(self) -> torch.Tensor:
        """The keys as vLLM's gather reads them, (blocks, block_size, dims + 4)."""
        raw, n, row = self.raw, self.size, self.dims + SCALE_BYTES
        return raw.as_strided((raw.shape[0], n, row), (raw.stride(0), row, 1))

    def dequantized(self, rows: torch.Tensor, n: int) -> torch.Tensor:
        """The first `n` keys of the request whose block table is `rows`, (1,
        blocks), in float32, (n, dims): what the kernels score."""
        at = torch.arange(n, device=rows.device)
        block, offset = rows[0, at // self.size].long(), at % self.size
        size, d, raw = self.size, self.dims, self.raw
        values = raw[:, : size * d].unflatten(1, (size, d))[block, offset]
        sf = raw[:, size * d : size * (d + SCALE_BYTES)].unflatten(1, (size, 4))
        scales = sf[block, offset].contiguous().view(torch.float32)
        return values.view(FP8).float() * scales


@dataclass(frozen=True)
class Raw:
    """QSA's indexer keys as projected, in a call's cache, ``[blocks, heads,
    block_size, C]``, in its head slots from `first` on (together within a block):
    a block's ``block_size`` keys of `dims` in `dtype`, the model's.

    Attributes:
        cache: The call's cache.
        first: Its first slot of keys.
        dims: The keys' dims.
        dtype: Their dtype.
    """

    cache: torch.Tensor
    first: int
    dims: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def keys(self) -> torch.Tensor:
        """The keys, (blocks, block_size, dims); a view, or an error."""
        c, size = self.cache.shape[3] * self.cache.dtype.itemsize, self.dtype.itemsize
        part = self.cache[:, self.first : self.first + raw_slots(self.dims, c, size)]
        raw = part.view(torch.uint8)
        raw = raw.view(raw.shape[0], -1)[:, : self.cache.shape[2] * self.dims * size]
        return raw.view(self.dtype).unflatten(1, (self.cache.shape[2], self.dims))

    def write(self, slots: torch.Tensor, keys: torch.Tensor) -> None:
        """Write the tokens' keys, (T, dims), at vLLM's `slots`, (T,)."""
        slots, size = slots.clamp(min=0), self.cache.shape[2]
        self.keys[slots // size, slots % size] = keys.to(self.dtype)

    def at(self, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys at `positions`, (T, S), of the requests whose block tables are
        `rows`, (T, blocks), in float32: (T, S, dims)."""
        size = self.cache.shape[2]
        block = rows.gather(1, positions // size).long()
        return self.keys[block, positions % size].float()


def raw_slots(dims: int, channels_bytes: int, itemsize: int = 2) -> int:
    """The head slots of `channels_bytes` a token that a `Raw` of `dims` of
    `itemsize` bytes takes."""
    return -(-(itemsize * dims) // channels_bytes)


def pooled(
    raw: Raw,
    region: Region,
    indexer: Any,
    rows: torch.Tensor,
    positions: torch.Tensor,
    size: int,
) -> None:
    """Write the keys of the blocks of `size` that the tokens at `positions`, (T,),
    of the requests whose block tables are `rows`, (T, blocks), complete into
    `region`, at the position of each block's index: the mean of its keys as
    projected, from `raw`, normalized and rotated by `indexer` to the block's first
    position, as bux pools them; nothing for a token that completes none."""
    first = positions // size * size
    keys = raw.at(rows, first[:, None] + torch.arange(size, device=rows.device))
    mean = indexer.k_norm(keys.mean(1))  # in float32, as bux pools them
    key = indexer._rotate(mean[:, None], first)[:, 0]
    at = positions // size
    slot = rows.gather(1, (at // region.size)[:, None])[:, 0].long() * region.size
    slot = slot + at % region.size
    done = (positions + 1) % size == 0
    region.write(torch.where(done, slot, -1), key)


def block_picks(
    blocks: torch.Tensor, positions: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """QSA's picks as `bux_vllm.picks` keeps them, (Q, (k + 1) size): the keys of
    the picked complete `blocks`, (Q, k) int32 (-1 for none), and of the query's own
    block while it is unfinished, those it sees valid."""
    own = positions // size
    finished = (positions + 1) % size == 0
    own = torch.where(finished, -1, own)[:, None]
    blocks = torch.cat([blocks.long(), own], -1)
    keys = blocks[..., None] * size + torch.arange(size, device=blocks.device)
    seen = (blocks >= 0)[..., None] & (keys <= positions[:, None, None])
    return keys.flatten(1).clamp(min=0), seen.flatten(1)


def prompt(
    region: Region,
    index: tuple[torch.Tensor, torch.Tensor],
    rows: torch.Tensor,
    n: int,
    ends: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """One request's queries' picks, (Q, top_k) int32 key positions, -1 for none:
    its first `n` keys gathered by vLLM from `region` by its block table `rows`,
    (1, blocks), scored by DeepGEMM's MQA logits a chunk of queries at a time, and
    the best of the first `ends`, (Q,), each query sees kept by vLLM's top-k.
    `index` are the queries and weights `queries` gives."""
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    q8, weights = index
    out = torch.full((len(ends), top_k), -1, dtype=torch.int32, device=q8.device)
    if n == 0:
        return out
    keys = q8.new_empty(n, region.dims)
    scales = torch.empty(n, SCALE_BYTES, dtype=torch.uint8, device=q8.device)
    bounds = torch.tensor([0, n], dtype=torch.int32, device=q8.device)
    ops.cp_gather_indexer_k_quant_cache(region.gathered(), keys, scales, rows, bounds)
    kv = (keys, scales.view(torch.float32).squeeze(-1))
    start = torch.zeros_like(ends, dtype=torch.int32)
    end = ends.to(torch.int32)
    chunk = max(1, BUDGET // n)
    for a in range(0, len(ends), chunk):
        c = slice(a, a + chunk)
        logits = fp8_fp4_mqa_logits(
            (q8[c], None), kv, weights[c], start[c], end[c], False
        )
        count, (s0, s1) = logits.shape[0], logits.stride()
        ops.top_k_per_row_prefill(
            logits, start[c], end[c], out[c], count, s0, s1, top_k
        )
    return out


@dataclass(frozen=True)
class Decoding:
    """A step's decoded tokens, as the paged kernel takes them.

    Attributes:
        lens: Each one's keys (its position + 1), (B, 1) int32.
        schedule: DeepGEMM's schedule of them over the SMs.
        longest: The most keys a token may have, ``max_model_len``.
    """

    lens: torch.Tensor
    schedule: torch.Tensor
    longest: int


def schedule(lens: torch.Tensor, block_size: int) -> torch.Tensor:
    """DeepGEMM's schedule of decoded tokens of `lens` keys, (B, 1), over the GPU's
    SMs, for blocks of `block_size` keys."""
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    sms = torch.cuda.get_device_properties(lens.device).multi_processor_count
    return get_paged_mqa_logits_metadata(lens, block_size, sms)


def decode(
    region: Region,
    index: tuple[torch.Tensor, torch.Tensor],
    table: torch.Tensor,
    step: Decoding,
    top_k: int,
    backend: str,
) -> torch.Tensor:
    """Decoded tokens' picks, (B, top_k) int32 key positions, -1 for none: each
    token's keys read in place from `region` by its request's block table `table`,
    (B, blocks), scored by DeepGEMM's paged MQA logits, and the best kept by vLLM's
    decode top-k (`backend`, vLLM's ``sparse_indexer_topk_backend``)."""
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

    q8, weights = index
    paged = region.paged()
    logits = fp8_fp4_paged_mqa_logits(
        (q8[:, None], None),
        paged,
        weights,
        step.lens,
        table,
        step.schedule,
        step.longest,
        False,
    )
    out = torch.empty(len(q8), top_k, dtype=torch.int32, device=q8.device)
    get_indexer_topk(backend)(logits, step.lens, 1, out, top_k, step.longest)
    return out


def attends(spec: Sparse, q: torch.Tensor, kv: torch.Tensor) -> bool:
    """Whether vLLM's own sparse GQA attention, Qwen4's QSA kernel (`attend`),
    attends a layer of `spec` over its picks: bfloat16 queries, keys and values of
    the same power of two of dims, 16 or more, a bfloat16 or e4m3 cache, groups of
    4 query heads or more (as measured), and no sinks, softcap or latents, which it
    has not."""
    h, dk = q.shape[1], q.shape[2]
    group = h // kv.shape[1]
    dims = kv.shape[3] == 2 * dk and dk >= 16 and dk & (dk - 1) == 0
    cache = kv.dtype in (torch.bfloat16, FP8)
    plain = spec.sinks is None and spec.softcap is None and not spec.latent
    return q.dtype == torch.bfloat16 and dims and cache and plain and group >= 4


def attend(
    spec: Sparse,
    q: torch.Tensor,
    kv: torch.Tensor,
    rows: torch.Tensor,
    picked: torch.Tensor,
    prompt: bool,
    scales: tuple[float, float],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Each query's heads' outputs, (Q, h, d), over its `picked` keys, (Q, K) int32
    positions with -1 for none, by vLLM's own sparse GQA attention (Qwen4's QSA
    kernel), which reads the paged `kv`, (blocks, g, block_size, 2 d), in place by
    the block table `rows`: one request's, (1, blocks), for a prompt's queries,
    else each one's; into `out` where given. DSA's picks have their valid ones
    first, as the kernel bounds a query's tiles by their count; QSA's are read
    whole (`bux_vllm.caching.packed`). Its output gate is passed open, and a scale
    other than ``d ** -0.5`` goes on the queries."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    from . import caching

    dk = q.shape[2]
    if spec.scale != dk**-0.5:
        q = (q.float() * (spec.scale * dk**0.5)).to(q.dtype)
    indices, req = caching.packed(picked.to(torch.int32), not prompt, spec.mean)
    nhd = kv.transpose(1, 2)
    k, v = nhd[..., :dk], nhd[..., dk:]
    fp8 = (scales[0], scales[1]) if kv.dtype == FP8 else (None, None)
    gate = torch.full_like(q, torch.inf)  # sigmoid(+inf) = 1: no gate
    return qsa_sparse_paged_attention(
        q.contiguous(), k, v, indices, rows, req, prompt, out, *fp8, output_gate=gate
    )


def reserve(device: torch.device, longest: int, dims: int) -> None:
    """Claim, in vLLM's profiling run, what a step's kernels take beyond it: the
    top-k's workspace, and a prompt chunk's logits and gathered keys at their most,
    which the profile counts as they are freed."""
    from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
    from vllm.v1.worker.workspace import current_workspace_manager

    workspace = ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
    current_workspace_manager().get_simultaneous(workspace)
    most = 4 * BUDGET + longest * (dims + SCALE_BYTES)
    torch.empty(most, dtype=torch.uint8, device=device)


def as_picks(picked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Picks of int32 positions, -1 for none, as `bux_vllm.picks` keeps them: the
    positions (0 for none) and which are valid."""
    valid = picked >= 0
    return picked.clamp(min=0).long(), valid
