"""CPLM's copy pointer in vLLM: the first head's paged cache of keys and tokens, and
the op that gives vLLM's sampler the pointer's distribution (`bux_vllm.copying`).

Each step the op, ``vllm::bux_copy_pointer``, writes the step's tokens' rows into
the cache (`PointerCache`), then for each request's last token of the step, the
token vLLM samples, attends to the request's tokens before it and the sink, and
writes p_copy, a row of the vocabulary, and a_sink into buffers that the head's
`logits` mixes with its softmax (`bux_vllm.copying.mixed`). vLLM splits its
piecewise CUDA graphs around the op, as around attention, and captures a step of
decoded tokens alone as one graph, whose replays read each step's requests from
vLLM's buffers (`bux_vllm.copying.Steps`).

In a 16-bit model a Triton kernel attends in one pass over the keys, a request's
positions split over programs: each adds each position's mass, its score shifted
by a bound the query fixes, into its token's entry of the request's row, and the
masses' sum, which the logits divide by. A float32 model takes PyTorch's path,
which gathers the request's rows first.
"""

from __future__ import annotations

from functools import cache
from typing import Any, ClassVar

import torch
import triton
import triton.language as tl
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from . import copying, sparse

#: The op, as vLLM's compilation names it.
OP = "vllm::bux_copy_pointer"


class StepsBuilder(AttentionMetadataBuilder[copying.Steps]):
    """Gives a `PointerCache` the step's `bux_vllm.copying.Steps`: views of vLLM's
    buffers, which the op reads each request's last token and position from. A step
    of decoded tokens alone is one CUDA graph; the op reads nothing on the CPU."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: Any
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> copying.Steps:
        return self._steps(common_attn_metadata)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> copying.Steps:
        return self._steps(common_attn_metadata)

    def _steps(self, cm: CommonAttentionMetadata) -> copying.Steps:
        r = cm.num_reqs
        starts, lens = cm.query_start_loc[: r + 1], cm.seq_lens[:r]
        return copying.Steps(
            cm.slot_mapping, cm.block_table_tensor, cm.num_actual_tokens, starts, lens
        )


class PointerBackend(sparse.PagedBackend):
    """The backend of a `PointerCache`: its metadata alone."""

    @staticmethod
    def get_name() -> str:
        return "BUX_POINTER"

    @staticmethod
    def get_builder_cls() -> type[StepsBuilder]:
        return StepsBuilder


class PointerCache(nn.Module, AttentionLayerBase):
    """The pointer's paged cache: a row of `bux_vllm.copying.row_width` channels a
    token, its key and its token, in vLLM's paged KV cache, allocated as for full
    attention, ``[blocks, 1, block_size, row]``: in the model's dtype, or bytes under
    an FP8 KV cache, its keys in FP8 at a scale of 1.

    Args:
        dim: The pointer's key channels.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of the cache, unique in the model.
        window: How far back a query looks, in positions.
    """

    def __init__(self, dim: int, vllm_config: Any, prefix: str, window: int):
        super().__init__()
        self.dim, self.prefix, self.window = dim, prefix, window
        self.longest = vllm_config.model_config.max_model_len  # positions a query sees
        self.dtype, self.fp8 = sparse.cache_dtype(vllm_config, "a copy pointer", True)
        self.kv_cache = torch.tensor([])
        self.stash: copying.Prompts | None = None  # the last step run on the host
        self.graphed: list[copying.Rows] = []  # the head inputs of full graphs
        ops = vllm_config.compilation_config.splitting_ops
        if ops and OP not in ops:  # split the graphs at it, as at attention
            ops.append(OP)
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PointerBackend

    def get_kv_cache_spec(self, vllm_config: Any) -> FullAttentionSpec:
        row = copying.row_width(self.dim)
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=row - row // 2,
            head_size_v=row // 2,
            dtype=self.dtype,
        )

    @property
    def rows(self) -> torch.Tensor:
        """The cache, ``[blocks, 1, block_size, row]``, as vLLM lays out a cache of
        one head (its rows' channels contiguous)."""
        return self.kv_cache


def copy_pointer(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    tokens: torch.Tensor,
    sink: torch.Tensor,
    p_copy: torch.Tensor,
    a_sink: torch.Tensor,
    total: torch.Tensor,
    layer_name: str,
) -> None:
    """The op: write the step's keys, (T, c), and `tokens`, (T,), into the cache of
    `layer_name`, then each request's p_copy and a_sink, for its last token of the
    step, as masses into the first rows of `p_copy`, (R_max, vocab), and `a_sink`,
    (R_max,), and their sum into `total`, (R_max,); nothing in vLLM's profiling
    run, which has no metadata. It keeps where the
    head's input `x` lies, for the logits of prompt rows (`bux_vllm.copying.Prompts`).
    """
    metas = sparse._metadata()
    if metas is None or layer_name not in metas:
        return
    cache = get_forward_context().no_compile_layers[layer_name]
    steps: copying.Steps = metas[layer_name]
    if torch.cuda.is_current_stream_capturing():  # a full graph, replayed in place
        if (rows := copying.Rows.of(x)) not in cache.graphed:
            cache.graphed.append(rows)
    else:
        cache.stash = copying.Prompts(copying.Rows.of(x), steps)
    # Every request's, but past the buffers' rows, which vLLM's runs within
    # max_num_seqs never reach.
    requests = min(steps.requests, p_copy.shape[0])
    q, sink, table, position = copying.stepped(
        cache.rows, steps, q, k, tokens, sink, requests, cache.fp8
    )
    if q.dtype == torch.float32:
        copy, share = copying.pointed(
            q,
            sink,
            cache.rows,
            table,
            position,
            p_copy.shape[1],
            cache.fp8,
            cache.longest,
            cache.window,
        )
        p_copy[:requests].copy_(copy)
        a_sink[:requests].copy_(share)
        total[:requests].fill_(1.0)
        return
    p_copy[:requests].zero_()
    total[:requests].zero_()
    _pointed(q, sink, cache, table, position, p_copy, a_sink, total)


def _pointed(
    q: torch.Tensor,
    sink: torch.Tensor,
    cache: PointerCache,
    table: torch.Tensor,
    position: torch.Tensor,
    p_copy: torch.Tensor,
    a_sink: torch.Tensor,
    total: torch.Tensor,
) -> None:
    """`bux_vllm.copying.pointed` by the Triton kernel, for keys of any width: into
    the zeroed rows of `p_copy` and `total`, its masses and their sum, and into
    `a_sink`, the sink's mass, which ``total`` then normalizes
    (`bux_vllm.copying.mixed`).

    One pass over the keys (`bux_vllm.copying.shift`): each mass is
    ``exp(s - c)``, c fixed by the query and the sink before any key is read, and a
    request's positions split over programs (`_splits`). A query whose scores may
    span more than float32's masses hold, past a gain of about ``64 / sqrt(c)``,
    takes two passes, its max first, in its first program.

    Every address is in bounds whatever the cache holds: vLLM's dummy and capture
    runs point padded requests at the null block, whose rows no step wrote, so the
    kernel clamps each block to the cache's, each digit to a byte and each token to
    the vocabulary, and reads no position past the block table's."""
    rows = cache.rows
    requests, dim = q.shape
    if requests == 0:
        return
    keys = rows.view(cache.fp8) if cache.fp8 is not None else rows
    longest = table.shape[1] * rows.shape[2]
    splits, per = _splits(requests, min(longest, cache.window), q.device)
    _pointer_kernel[(requests, splits)](
        q.contiguous(),
        sink.float().contiguous(),
        rows,
        keys,
        table,
        position.to(torch.int32).contiguous(),
        p_copy,
        a_sink,
        total,
        rows.stride(0),
        rows.stride(2),
        table.stride(0),
        p_copy.stride(0),
        p_copy.shape[1],
        rows.shape[0],
        longest,
        cache.window,
        dim**-0.5,
        copying.KEY_NORM,
        copying.TOP,
        copying.FLOOR,
        D=dim,
        DP=triton.next_power_of_2(dim),
        B=rows.shape[2],
        BN=BN,
        PER=per,
        DIGITS=copying.DIGITS,
    )


#: The positions a program of the kernel scores at once.
BN = 64


@cache
def _sms(device: torch.device) -> int:
    """The streaming multiprocessors of `device`."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _splits(requests: int, longest: int, device: torch.device) -> tuple[int, int]:
    """`bux_vllm.copying.splits` on `device`, of chunks of `BN` positions."""
    return copying.splits(requests, longest, _sms(device), BN)


@triton.jit
def _scores(
    KEYS,
    TABLE_ROW,
    at,
    end,
    q,
    d,
    on,
    stride_block,
    stride_slot,
    blocks,
    scale,
    B: tl.constexpr,
):
    """The scores of positions `at` before `end`, -inf past it, and their rows'
    offsets in the cache."""
    seen = at < end
    block = tl.load(TABLE_ROW + at // B, mask=seen, other=0)
    block = tl.minimum(tl.maximum(block, 0), blocks - 1).to(tl.int64)
    base = block * stride_block + (at % B) * stride_slot
    k = tl.load(
        KEYS + base[:, None] + d[None, :], mask=seen[:, None] & on[None, :], other=0.0
    )
    s = tl.sum(k.to(tl.float32) * q[None, :], 1) * scale
    return tl.where(seen, s, -float("inf")), base


@triton.jit
def _tokens(
    ROWS, base, seen, vocab, D: tl.constexpr, BN: tl.constexpr, DIGITS: tl.constexpr
):
    """The tokens of the cache's rows at `base`, from their digits."""
    token = tl.zeros([BN], tl.int64)
    for i in tl.static_range(DIGITS):
        digit = tl.load(ROWS + base + D + i, mask=seen, other=0).to(tl.float32)
        # A byte, or 0 for anything else (NaN too), of a row no step wrote
        digit = tl.where((digit >= 0.0) & (digit <= 255.0), digit, 0.0)
        token += digit.to(tl.int64) << (8 * i)
    return tl.minimum(tl.maximum(token, 0), vocab - 1)


@triton.jit
def _pointer_kernel(
    Q,
    SINK,
    ROWS,
    KEYS,
    TABLE,
    POSITION,
    P_COPY,
    A_SINK,
    TOTAL,
    stride_block,
    stride_slot,
    stride_table,
    stride_copy,
    vocab,
    blocks,
    longest,
    window,
    scale,
    key_norm,
    top,
    floor,
    D: tl.constexpr,
    DP: tl.constexpr,
    B: tl.constexpr,
    BN: tl.constexpr,
    PER: tl.constexpr,
    DIGITS: tl.constexpr,
):
    """A program of a request's query: `PER` of the positions within `window`
    before it, from its split's first, each one's mass ``exp(s - c)`` added into
    its token's entry of the request's row, and their sum into its total, the
    sink's too in the first program; or, past the bound float32 holds, the whole
    request in its first program, by its max. The adds are relaxed: sums, whose
    order no ordering would fix."""
    r, j = tl.program_id(0), tl.program_id(1)
    end = tl.minimum(tl.load(POSITION + r), longest)  # the block table's positions
    lo = tl.maximum(end - window, 0)
    d = tl.arange(0, DP)  # a power of two of channels, D of them the key's
    on = d < D
    q = tl.load(Q + r * D + d, mask=on, other=0.0).to(tl.float32)
    s0 = tl.load(SINK + r)
    bound = tl.sqrt(tl.sum(q * q, 0)) * key_norm  # no score is higher
    keys, table, out = KEYS, TABLE + r * stride_table, P_COPY + r * stride_copy
    if 2 * bound <= top + floor:  # every score within float32's masses
        c = tl.maximum(s0, bound - top)
        first = lo + j * PER
        mass = s0 * 0.0
        for start in range(first, tl.minimum(first + PER, end), BN):
            at = start + tl.arange(0, BN)
            s, base = _scores(
                keys,
                table,
                at,
                end,
                q,
                d,
                on,
                stride_block,
                stride_slot,
                blocks,
                scale,
                B,
            )
            e = tl.exp(s - c)
            mass += tl.sum(e, 0)
            token = _tokens(ROWS, base, at < end, vocab, D, BN, DIGITS)
            tl.atomic_add(out + token, e, mask=at < end, sem="relaxed")
        if j == 0:
            tl.store(A_SINK + r, tl.exp(s0 - c))
            mass += tl.exp(s0 - c)
        tl.atomic_add(TOTAL + r, mass, sem="relaxed")
    elif j == 0:
        m, total = s0, s0 * 0.0 + 1.0
        for start in range(lo, end, BN):
            at = start + tl.arange(0, BN)
            s, _ = _scores(
                keys,
                table,
                at,
                end,
                q,
                d,
                on,
                stride_block,
                stride_slot,
                blocks,
                scale,
                B,
            )
            peak = tl.maximum(m, tl.max(s, 0))
            total = total * tl.exp(m - peak) + tl.sum(tl.exp(s - peak), 0)
            m = peak
        for start in range(lo, end, BN):
            at = start + tl.arange(0, BN)
            s, base = _scores(
                keys,
                table,
                at,
                end,
                q,
                d,
                on,
                stride_block,
                stride_slot,
                blocks,
                scale,
                B,
            )
            token = _tokens(ROWS, base, at < end, vocab, D, BN, DIGITS)
            tl.atomic_add(out + token, tl.exp(s - m), mask=at < end, sem="relaxed")
        tl.store(A_SINK + r, tl.exp(s0 - m))
        tl.atomic_add(TOTAL + r, total, sem="relaxed")


def mixture(
    logits: torch.Tensor,
    p_copy: torch.Tensor,
    a_sink: torch.Tensor,
    total: torch.Tensor,
    copy: int,
) -> torch.Tensor:
    """`bux_vllm.copying.mixed` in one Triton launch, a program a row, in the
    logits' dtype: the head's log-softmax over the rows but `copy` and the gate in
    a first pass over the row, the mixture in a second, in float32. Its dozen
    PyTorch ops cost vLLM's sampler about 7 times the head's log-softmax alone,
    each a launch outside graphs; a program's blocks of `MIX_BLOCK` take a row of
    50304 in 4 loads a pass."""
    logits = logits.contiguous()
    rows, vocab = logits.shape
    out = torch.empty_like(logits)
    if rows:
        _mixture_kernel[(rows,)](
            logits,
            p_copy,
            a_sink,
            total,
            out,
            p_copy.stride(0),
            vocab,
            copy,
            BV=MIX_BLOCK,
            num_warps=16,
        )
    return out


#: The logits of a row the mixture's kernel reads at once.
MIX_BLOCK = 16384


@triton.jit
def _logaddexp(a, b):
    """log(e^a + e^b), -inf where both are."""
    top = tl.maximum(a, b)
    top = tl.where(top == -float("inf"), 0.0, top)
    return top + tl.log(tl.exp(a - top) + tl.exp(b - top))


@triton.jit
def _mixture_kernel(
    LOGITS, P_COPY, A_SINK, TOTAL, OUT, stride_copy, vocab, copy, BV: tl.constexpr
):
    """A row's mixture: ``log(p_lm (1 - a + a a_sink) + a p_copy)``, its masses over
    `TOTAL`, -inf at <copy>."""
    r = tl.program_id(0).to(tl.int64)
    row, out, masses = LOGITS + r * vocab, OUT + r * vocab, P_COPY + r * stride_copy
    m = tl.full([], -float("inf"), tl.float32)
    total = tl.full([], 0.0, tl.float32)
    for start in range(0, vocab, BV):  # the log-softmax over every row but <copy>
        v = start + tl.arange(0, BV)
        z = tl.load(row + v, mask=(v < vocab) & (v != copy), other=-float("inf"))
        z = z.to(tl.float32)
        top = tl.maximum(m, tl.max(z, 0))
        total = total * tl.exp(m - top) + tl.sum(tl.exp(z - top), 0)
        m = top
    lse_lm = m + tl.log(total)
    gate = tl.load(row + copy).to(tl.float32)
    lse = _logaddexp(lse_lm, gate)
    log_gate = gate - lse - tl.log(tl.load(TOTAL + r).to(tl.float32))
    log_sink = tl.log(tl.load(A_SINK + r).to(tl.float32))
    kept = _logaddexp(lse_lm - lse, log_gate + log_sink)
    for start in range(0, vocab, BV):
        v = start + tl.arange(0, BV)
        z = tl.load(row + v, mask=v < vocab, other=0.0).to(tl.float32)
        p = tl.load(masses + v, mask=v < vocab, other=0.0).to(tl.float32)
        y = _logaddexp(z - lse_lm + kept, log_gate + tl.log(p))
        tl.store(out + v, tl.where(v == copy, -float("inf"), y), mask=v < vocab)


def _fake(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    tokens: torch.Tensor,
    sink: torch.Tensor,
    p_copy: torch.Tensor,
    a_sink: torch.Tensor,
    total: torch.Tensor,
    layer_name: str,
) -> None:
    """The op as torch.compile traces it: it writes `p_copy`, `a_sink` and `total`
    alone."""


direct_register_custom_op(
    op_name="bux_copy_pointer",
    op_func=copy_pointer,
    mutates_args=["p_copy", "a_sink", "total"],
    fake_impl=_fake,
)
