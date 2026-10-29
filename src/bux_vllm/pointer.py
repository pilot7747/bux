"""CPLM's copy pointer in vLLM: the first head's paged cache of keys and tokens, and
the op that gives vLLM's sampler the pointer's distribution (`bux_vllm.copying`).

Each step the op, ``vllm::bux_copy_pointer``, writes the step's tokens' rows into
the cache (`PointerCache`), then for each request's last token of the step, the
token vLLM samples, attends to the request's tokens before it and the sink, and
writes p_copy, a row of the vocabulary, and a_sink into buffers that the head's
`logits` mixes with its softmax (`bux_vllm.copying.mixed`). vLLM splits its
piecewise CUDA graphs around the op, as around attention, and captures a step of
decoded tokens alone as one graph (`sparse.PagedMetadataBuilder`).

In a 16-bit model a Triton kernel attends, one program a request: a pass over the
request's blocks for the softmax's max and sum, then one that adds each position's
share into its token's entry of the request's row. A float32 model takes PyTorch's
path, which gathers the request's rows first.
"""

from __future__ import annotations

from dataclasses import dataclass
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

from . import copying, picks, sparse

#: The op, as vLLM's compilation names it.
OP = "vllm::bux_copy_pointer"


@dataclass
class Steps:
    """A step's tokens, as the pointer's op reads them.

    Attributes:
        slot_mapping: Each token's slot, (T,); -1 for padding.
        block_table_tensor: Each request's blocks, (R, blocks).
        num_actual_tokens: The step's tokens, padding left out.
        last: Each request's last token in the step, (R,), which vLLM samples.
        position: That token's position in its request, (R,).
    """

    slot_mapping: torch.Tensor
    block_table_tensor: torch.Tensor
    num_actual_tokens: int
    last: torch.Tensor
    position: torch.Tensor


class StepsBuilder(AttentionMetadataBuilder[Steps]):
    """Gives a `PointerCache` the step's `Steps`: each request's last token is the
    one before the next request's first, at the position before its length. A step
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
    ) -> Steps:
        return self._steps(common_attn_metadata)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Steps:
        return self._steps(common_attn_metadata)

    def _steps(self, cm: CommonAttentionMetadata) -> Steps:
        requests = cm.num_reqs
        last = cm.query_start_loc[1 : requests + 1] - 1
        position = cm.seq_lens[:requests] - 1
        return Steps(
            cm.slot_mapping, cm.block_table_tensor, cm.num_actual_tokens, last, position
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
    """

    def __init__(self, dim: int, vllm_config: Any, prefix: str):
        super().__init__()
        self.dim, self.prefix = dim, prefix
        self.dtype, self.fp8 = sparse.cache_dtype(vllm_config, "a copy pointer", True)
        self.kv_cache = torch.tensor([])
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
    nothing in vLLM's profiling run, which has no metadata."""
    metas = sparse._metadata()
    if metas is None or layer_name not in metas:
        return
    cache = get_forward_context().no_compile_layers[layer_name]
    steps: Steps = metas[layer_name]
    n = steps.num_actual_tokens
    rows = copying.encode(k[:n], tokens[:n], cache.dtype, cache.fp8)
    picks.write(cache.rows, steps.slot_mapping[:n], rows)
    last, requests = steps.last.long(), steps.last.shape[0]
    table = steps.block_table_tensor[:requests]
    if q.dtype == torch.float32:
        copy, share = copying.pointed(
            q[last],
            sink[last],
            cache.rows,
            table,
            steps.position,
            p_copy.shape[1],
            cache.fp8,
        )
        p_copy[:requests].copy_(copy)
        a_sink[:requests].copy_(share)
        return
    p_copy[:requests].zero_()
    _pointed(q[last], sink[last], cache, table, steps.position, p_copy, a_sink)


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
    `p_copy` and into `a_sink`."""
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
        dim**-0.5,
        D=dim,
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
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    BN: tl.constexpr,
    DIGITS: tl.constexpr,
):
    """One request's query over its positions before it: the softmax's max and sum
    in a first pass, then each position's share added into its token's entry."""
    r = tl.program_id(0)
    end = tl.load(POSITION + r)
    d = tl.arange(0, D)
    q = tl.load(Q + r * D + d).to(tl.float32)
    s0 = tl.load(SINK + r)
    m, total = s0, s0 * 0.0 + 1.0
    for start in range(0, end, BN):
        at = start + tl.arange(0, BN)
        seen = at < end
        block = tl.load(TABLE + r * stride_table + at // B, mask=seen, other=0)
        base = block.to(tl.int64) * stride_block + (at % B) * stride_slot
        k = tl.load(KEYS + base[:, None] + d[None, :], mask=seen[:, None], other=0.0)
        s = tl.sum(k.to(tl.float32) * q[None, :], 1) * scale
        s = tl.where(seen, s, -float("inf"))
        top = tl.maximum(m, tl.max(s, 0))
        total = total * tl.exp(m - top) + tl.sum(tl.exp(s - top), 0)
        m = top
    for start in range(0, end, BN):
        at = start + tl.arange(0, BN)
        seen = at < end
        block = tl.load(TABLE + r * stride_table + at // B, mask=seen, other=0)
        base = block.to(tl.int64) * stride_block + (at % B) * stride_slot
        k = tl.load(KEYS + base[:, None] + d[None, :], mask=seen[:, None], other=0.0)
        s = tl.sum(k.to(tl.float32) * q[None, :], 1) * scale
        share = tl.where(seen, tl.exp(s - m) / total, 0.0)
        token = tl.zeros([BN], tl.int64)
        for i in tl.static_range(DIGITS):
            digit = tl.load(ROWS + base + D + i, mask=seen, other=0).to(tl.float32)
            token += digit.to(tl.int64) << (8 * i)
        token = tl.minimum(token, vocab - 1)
        tl.atomic_add(P_COPY + r * stride_copy + token, share, mask=seen)
    tl.store(A_SINK + r, tl.exp(s0 - m) / total)


def _fake(
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
