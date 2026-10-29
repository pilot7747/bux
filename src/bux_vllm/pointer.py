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

In a 16-bit model a Triton kernel attends, one program a request: a pass over the
request's blocks for the softmax's max and sum, then one that adds each position's
share into its token's entry of the request's row. A float32 model takes PyTorch's
path, which gathers the request's rows first.
"""

from __future__ import annotations

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
    layer_name: str,
) -> None:
    """The op: write the step's keys, (T, c), and `tokens`, (T,), into the cache of
    `layer_name`, then each request's p_copy and a_sink, for its last token of the
    step, into the first rows of `p_copy`, (R_max, vocab), and `a_sink`, (R_max,);
    nothing in vLLM's profiling run, which has no metadata. It keeps where the
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
        return
    p_copy[:requests].zero_()
    _pointed(q, sink, cache, table, position, p_copy, a_sink)


def _pointed(
    q: torch.Tensor,
    sink: torch.Tensor,
    cache: PointerCache,
    table: torch.Tensor,
    position: torch.Tensor,
    p_copy: torch.Tensor,
    a_sink: torch.Tensor,
) -> None:
    """`bux_vllm.copying.pointed` by the Triton kernel, into the zeroed rows of
    `p_copy` and into `a_sink`, for keys of any width: the kernel masks the
    power of two of channels it loads to the keys'. Every address it takes is in
    bounds whatever the cache holds: vLLM's dummy and capture runs point padded
    requests at the null block, whose rows no step wrote, so it clamps each block
    to the cache's, each digit to a byte and each token to the vocabulary, and
    reads no position past the block table's."""
    rows = cache.rows
    requests, dim = q.shape
    if requests == 0:
        return
    keys = rows.view(cache.fp8) if cache.fp8 is not None else rows
    _pointer_kernel[(requests,)](
        q.contiguous(),
        sink.float().contiguous(),
        rows,
        keys,
        table,
        position.to(torch.int32).contiguous(),
        p_copy,
        a_sink,
        rows.stride(0),
        rows.stride(2),
        table.stride(0),
        p_copy.stride(0),
        p_copy.shape[1],
        rows.shape[0],
        table.shape[1] * rows.shape[2],
        cache.window,
        dim**-0.5,
        D=dim,
        DP=triton.next_power_of_2(dim),
        B=rows.shape[2],
        BN=64,
        DIGITS=copying.DIGITS,
    )


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
    stride_block,
    stride_slot,
    stride_table,
    stride_copy,
    vocab,
    blocks,
    longest,
    window,
    scale,
    D: tl.constexpr,
    DP: tl.constexpr,
    B: tl.constexpr,
    BN: tl.constexpr,
    DIGITS: tl.constexpr,
):
    """One request's query over its positions before it within `window`: the
    softmax's max and sum in a first pass, then each position's share added into
    its token's entry."""
    r = tl.program_id(0)
    end = tl.minimum(tl.load(POSITION + r), longest)  # the block table's positions
    lo = tl.maximum(end - window, 0)
    d = tl.arange(0, DP)  # a power of two of channels, D of them the key's
    on = d < D
    q = tl.load(Q + r * D + d, mask=on, other=0.0).to(tl.float32)
    s0 = tl.load(SINK + r)
    m, total = s0, s0 * 0.0 + 1.0
    for start in range(lo, end, BN):
        at = start + tl.arange(0, BN)
        seen = at < end
        block = tl.load(TABLE + r * stride_table + at // B, mask=seen, other=0)
        block = tl.minimum(tl.maximum(block, 0), blocks - 1).to(tl.int64)
        base = block * stride_block + (at % B) * stride_slot
        k = tl.load(
            KEYS + base[:, None] + d[None, :],
            mask=seen[:, None] & on[None, :],
            other=0.0,
        )
        s = tl.sum(k.to(tl.float32) * q[None, :], 1) * scale
        s = tl.where(seen, s, -float("inf"))
        top = tl.maximum(m, tl.max(s, 0))
        total = total * tl.exp(m - top) + tl.sum(tl.exp(s - top), 0)
        m = top
    for start in range(lo, end, BN):
        at = start + tl.arange(0, BN)
        seen = at < end
        block = tl.load(TABLE + r * stride_table + at // B, mask=seen, other=0)
        block = tl.minimum(tl.maximum(block, 0), blocks - 1).to(tl.int64)
        base = block * stride_block + (at % B) * stride_slot
        k = tl.load(
            KEYS + base[:, None] + d[None, :],
            mask=seen[:, None] & on[None, :],
            other=0.0,
        )
        s = tl.sum(k.to(tl.float32) * q[None, :], 1) * scale
        share = tl.where(seen, tl.exp(s - m) / total, 0.0)
        token = tl.zeros([BN], tl.int64)
        for i in tl.static_range(DIGITS):
            digit = tl.load(ROWS + base + D + i, mask=seen, other=0).to(tl.float32)
            # A byte, or 0 for anything else (NaN too), of a row no step wrote
            digit = tl.where((digit >= 0.0) & (digit <= 255.0), digit, 0.0)
            token += digit.to(tl.int64) << (8 * i)
        token = tl.minimum(tl.maximum(token, 0), vocab - 1)
        tl.atomic_add(P_COPY + r * stride_copy + token, share, mask=seen)
    tl.store(A_SINK + r, tl.exp(s0 - m) / total)


def _fake(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    tokens: torch.Tensor,
    sink: torch.Tensor,
    p_copy: torch.Tensor,
    a_sink: torch.Tensor,
    layer_name: str,
) -> None:
    """The op as torch.compile traces it: it writes `p_copy` and `a_sink` alone."""


direct_register_custom_op(
    op_name="bux_copy_pointer",
    op_func=copy_pointer,
    mutates_args=["p_copy", "a_sink"],
    fake_impl=_fake,
)
