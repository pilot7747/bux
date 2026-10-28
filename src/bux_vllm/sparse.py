"""A ``gqa`` layer's sparse attention in vLLM: its keys and values and its
indexer's keys in vLLM's paged KV cache, and each query's attention over the keys
its indexer picks, as bux computes them (`bux_vllm.picks`).

Each call of a sparse layer keeps one of vLLM's caches, `PagedCache`: its keys and
values, as a dense layer's, each kv head's row with its share of the indexer's key
beside them (`bux_vllm.picks.Cache`). vLLM allocates its blocks as for full
attention, so they grow with the sequence, and in blocks of as many tokens as the
model's other attention caches, so they serve prefix caching as those do (a cache
of indexer keys alone, smaller a token, would get blocks of more tokens, which a
prefix would have to fill to be cached). The plugin writes and reads it itself, in
one op, ``vllm::bux_sparse_attention``: vLLM splits its piecewise CUDA graphs around
it (`split_graphs`), as around attention, and captures a step of decoded tokens
alone as one graph (`PagedMetadataBuilder`). The op also gives the keys each query
picked, which a layer that reads them through its ``index`` port attends over with
``vllm::bux_picked_attention``, keeping no indexer keys of its own.

A layer with a relative bias attends to every key it sees, the bias on its logits,
in ``vllm::bux_biased_attention`` over a cache of the same kind: no attention kernel
of vLLM's adds a bias that depends on the query and on how far behind it a key is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
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
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import FullAttentionSpec

from . import compressed, kernels, picks
from .common import own_cache_dtype
from .kv import reads, writes

#: The ops, as vLLM's compilation names them.
OPS = (
    "vllm::bux_sparse_attention",
    "vllm::bux_picked_attention",
    "vllm::bux_biased_attention",
)


@dataclass
class Batch:
    """A step's tokens, as a `PagedCache`'s ops read them.

    Attributes:
        slot_mapping: Each token's slot, (T,); -1 for padding.
        block_table_tensor: Each request's blocks, (requests, blocks).
        num_actual_tokens: The step's tokens, padding left out.
        segments: Its queries by `bux_vllm.picks.Segment`: its decoded tokens, one
            a request, then each prompt chunk.
    """

    slot_mapping: torch.Tensor
    block_table_tensor: torch.Tensor
    num_actual_tokens: int
    segments: list[picks.Segment]


class PagedMetadataBuilder(AttentionMetadataBuilder[Batch]):
    """Gives a `PagedCache` the step's `Batch`, its decoded tokens first (it has vLLM
    order them so). A step of decoded tokens alone is one CUDA graph: as captured,
    they read up to ``max_model_len`` keys, which every replay may have, not the
    capture's longest request's, and the ops read nothing on the CPU."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: Any
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)
        self.longest = vllm_config.model_config.max_model_len

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Batch:
        return self._batch(common_attn_metadata, common_attn_metadata.max_seq_len)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> Batch:
        return self._batch(common_attn_metadata, self.longest)

    def _batch(self, cm: CommonAttentionMetadata, longest: int) -> Batch:
        """The step `cm`, its decoded tokens over `longest` keys at most."""
        decodes, _, decode_tokens, _ = split_decodes_and_prefills(cm)
        chunks = _prompts(cm, decodes) if decodes < cm.num_reqs else []
        segments = picks.segments(decode_tokens, chunks, longest)
        return Batch(
            cm.slot_mapping, cm.block_table_tensor, cm.num_actual_tokens, segments
        )


def _prompts(cm: CommonAttentionMetadata, decodes: int) -> list:
    """Each prompt chunk of the step after its `decodes` decoded requests: its first
    token, tokens, request and the request's length with it."""
    lengths = cm.seq_lens_cpu_upper_bound
    lengths = cm.seq_lens.cpu() if lengths is None else lengths
    starts = cm.query_start_loc_cpu[: cm.num_reqs + 1].tolist()
    return compressed.prompt_chunks(starts, lengths.tolist(), decodes)


class PagedBackend(AttentionBackend):
    """The backend of a `PagedCache`: its metadata alone, as the plugin attends."""

    supported_dtypes: ClassVar = [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return "BUX_PAGED"

    @staticmethod
    def get_impl_cls() -> Any:
        raise NotImplementedError("a bux paged cache attends in bux's own op")

    @staticmethod
    def get_builder_cls() -> type[PagedMetadataBuilder]:
        return PagedMetadataBuilder


class PagedCache(nn.Module, AttentionLayerBase):
    """A sparse call's `heads` vectors of `channels` a token, and its indexer's key
    of `index_dims` spread over them, in vLLM's paged KV cache, allocated as for
    full attention, which `kv_cache` views as ``[blocks, heads, block_size,
    channels + ceil(index_dims / heads)]`` (`bux_vllm.picks.Cache`).

    Args:
        heads: The kv heads of a token.
        channels: Each head's channels: a key's and a value's, or MLA's latent.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of the cache, unique in the model.
        sparse: The settings of the call, which the op attends with.
        index_dims: The channels of its indexer's key; 0 for a call that reads its
            picks from its ``index`` port.
        what: The layer, as its errors name it.

    Raises:
        NotImplementedError: ``kv_cache_dtype`` is other than the model's, such as
            fp8, which the plugin's ops do not read (`own_cache_dtype`).
    """

    def __init__(
        self,
        heads: int,
        channels: int,
        vllm_config: Any,
        prefix: str,
        sparse: picks.Sparse,
        index_dims: int = 0,
        what: str = "a sparse layer",
    ):
        super().__init__()
        self.dtype = own_cache_dtype(vllm_config, what)
        self.heads, self.channels, self.prefix = heads, channels, prefix
        self.index_dims, self.share = index_dims, -(-index_dims // heads)
        self.sparse = sparse
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PagedBackend

    def get_kv_cache_spec(self, vllm_config: Any) -> FullAttentionSpec:
        row = self.channels + self.share
        half = row // 2  # a key's and a value's dims, as vLLM sums them
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.heads,
            head_size=row - half,
            head_size_v=half,
            dtype=self.dtype,
        )

    @property
    def rows(self) -> picks.Cache:
        """The cache, its keys and values apart from its indexer keys."""
        return picks.Cache(self.kv_cache, self.channels, self.index_dims)


def layer_caches(
    spec: picks.Sparse,
    heads: int,
    channels: int,
    index_dims: int,
    names: dict[str, list[str]],
    vllm_config: Any,
    what: str,
) -> nn.ModuleList:
    """Each call's `PagedCache` of the layer `what`, by the names of its states: its
    keys and values, `heads` vectors of `channels` a token, attended with `spec`,
    and its indexer's keys, `index_dims` a token (0 for a layer that reads its
    picks from its ``index`` port)."""
    caches = nn.ModuleList(
        PagedCache(heads, channels, vllm_config, name, spec, index_dims, what)
        for name in names["full_attention"]
    )
    split_graphs(vllm_config)
    return caches


def attend(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    index: tuple[torch.Tensor, ...] | None,
    out: torch.Tensor,
    cache: Any,
    ports: list,
    ctx: Any,
) -> torch.Tensor:
    """A sparse layer's call: its queries, keys and values (`tensors`) attend into
    `out` by the indexer's picks, `index` its queries, head weights and keys of the
    step's tokens, or by the picks its ``index`` port reads (None), over the call's
    `PagedCache` `cache`; the picks go on to the taps of the port it writes. Returns
    `out`."""
    q, k, v = (t.contiguous() for t in tensors)
    taps = writes(ports, "index")
    if index is None:
        read = reads(ports, "index")
        assert read is not None, "a sparse layer reads its picks or has an indexer"
        kept = ctx.read(*read)
        torch.ops.vllm.bux_picked_attention(
            q, k, v, *kept, ctx.positions, out, cache.prefix
        )
    else:
        kept = _no_picks(cache.sparse, q, len(taps))
        torch.ops.vllm.bux_sparse_attention(
            q,
            k,
            v,
            *(t.contiguous() for t in index),
            ctx.positions,
            out,
            *kept,
            cache.prefix,
        )
    for tap in taps:
        ctx.push(tap, kept)
    return out


def _no_picks(
    spec: picks.Sparse, q: torch.Tensor, taps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the sparse op writes the step's picks, (T, [G,] K): none unless the
    layer pushes them to `taps`."""
    groups = (spec.groups[1],) if spec.per_group else ()
    shape = (q.shape[0] if taps else 0, *groups, picks.most(spec))
    return q.new_zeros(shape, dtype=torch.long), q.new_zeros(shape, dtype=torch.bool)


def split_graphs(vllm_config: Any) -> None:
    """Have vLLM split its compiled graphs at the ops, as at attention, where it
    splits them at all."""
    ops = vllm_config.compilation_config.splitting_ops
    if ops:
        ops.extend(op for op in OPS if op not in ops)


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_w: torch.Tensor,
    index_k: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    picked_keys: torch.Tensor,
    picked_valid: torch.Tensor,
    layer_name: str,
) -> None:
    """The op: write the step's keys and values, (T, g, dk) and (T, g, dv), and its
    indexer keys, (T, i), into the cache of the call `layer_name`, then write each
    query's attention over the keys it picks into `output`, (T, h, dv), and its
    picks into `picked_keys` and `picked_valid`, (T, [G,] K), unless they are
    empty; zeros in vLLM's profiling run, which has no metadata."""
    metas = _metadata()
    picked_valid.zero_()
    if metas is None or layer_name not in metas:
        output.zero_()
        return
    cache = get_forward_context().no_compile_layers[layer_name]
    meta = metas[layer_name]
    n = meta.num_actual_tokens
    output[n:].zero_()
    rows = picks.merged(torch.cat([k, v], -1)[:n], index_k[:n])
    picks.write(cache.kv_cache, meta.slot_mapping[:n], rows)
    table, tensors = meta.block_table_tensor, (q, index_q, index_w, positions)
    for part in meta.segments:
        at = part.tokens
        q_, iq, iw, pos = (t[at] for t in tensors)
        out, (kept, valid) = _segments(cache)(
            cache.sparse, q_, (iq, iw), pos, cache.rows, table[part.requests], part.keys
        )
        output[at] = out.to(output.dtype)
        if picked_keys.numel():
            picked_keys[at], picked_valid[at] = kept, valid


def picked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    picked_keys: torch.Tensor,
    picked_valid: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """The op of a call that reads its ``index`` port: write the step's keys and
    values, (T, g, dk) and (T, g, dv), into the cache of the call `layer_name`, then
    write each query's attention over the keys another call picked for it,
    `picked_keys` and `picked_valid` (T, [G,] K), into `output`, (T, h, dv); zeros
    in vLLM's profiling run, which has no metadata."""
    metas = _metadata()
    if metas is None or layer_name not in metas:
        output.zero_()
        return
    cache = get_forward_context().no_compile_layers[layer_name]
    meta = metas[layer_name]
    n = meta.num_actual_tokens
    output[n:].zero_()
    picks.write(cache.kv_cache, meta.slot_mapping[:n], torch.cat([k, v], -1)[:n])
    table = meta.block_table_tensor
    for part in meta.segments:
        at = part.tokens
        picked = (picked_keys[at], picked_valid[at])
        attend = kernels.attend_picked if _kernels(cache) else picks.attend_picked
        out = attend(
            cache.sparse,
            q[at],
            picked,
            positions[at],
            cache.rows.kv,
            table[part.requests],
            part.keys,
        )
        output[at] = out.to(output.dtype)


def biased_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    table: torch.Tensor,
    gain: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """The op of a call with a relative bias: write the step's keys and values,
    (T, g, dk) and (T, g, dv), into the cache of the call `layer_name`, then write
    each query's attention over every key it sees into `output`, (T, h, dv), the
    bias of its heads' projections `r`, (T, h, rank), and the bias `table` on its
    logits, then its length scale `gain`, (T,), unless empty; zeros in vLLM's
    profiling run, which has no metadata."""
    metas = _metadata()
    if metas is None or layer_name not in metas:
        output.zero_()
        return
    cache = get_forward_context().no_compile_layers[layer_name]
    meta = metas[layer_name]
    n = meta.num_actual_tokens
    output[n:].zero_()
    picks.write(cache.kv_cache, meta.slot_mapping[:n], torch.cat([k, v], -1)[:n])
    table_rows = meta.block_table_tensor
    for part in meta.segments:
        at = part.tokens
        bias = (r[at], table, gain[at] if gain.numel() else None)
        rows = table_rows[part.requests]
        out = picks.attend_all(
            cache.sparse, q[at], bias, positions[at], cache.kv_cache, rows, part.keys
        )
        output[at] = out.to(output.dtype)


def _kernels(cache: PagedCache) -> bool:
    """Whether the Triton kernels attend over the call's `cache` (`bux_vllm.kernels`)."""
    dims = (cache.sparse.dk, cache.channels - cache.sparse.dk)
    return kernels.takes(cache.sparse, dims, cache.kv_cache.dtype)


def _segments(cache: PagedCache) -> Any:
    """How a segment of the call's queries picks and attends: by the kernels where
    they take its settings, else by PyTorch."""
    return kernels.attend_segment if _kernels(cache) else picks.attend_segment


def _metadata() -> dict | None:
    """The step's attention metadata by layer name; None in vLLM's profiling run."""
    metas = get_forward_context().attn_metadata
    if isinstance(metas, list):
        metas = metas[0]
    return metas if isinstance(metas, dict) else None


def _fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_w: torch.Tensor,
    index_k: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    picked_keys: torch.Tensor,
    picked_valid: torch.Tensor,
    layer_name: str,
) -> None:
    """The op as torch.compile traces it: it writes `output` and the picks alone."""


def _fake_picked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    picked_keys: torch.Tensor,
    picked_valid: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """The reader's op as torch.compile traces it: it writes `output` alone."""


direct_register_custom_op(
    op_name="bux_sparse_attention",
    op_func=sparse_attention,
    mutates_args=["output", "picked_keys", "picked_valid"],
    fake_impl=_fake,
)


def _fake_biased(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    table: torch.Tensor,
    gain: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """The biased op as torch.compile traces it: it writes `output` alone."""


direct_register_custom_op(
    op_name="bux_biased_attention",
    op_func=biased_attention,
    mutates_args=["output"],
    fake_impl=_fake_biased,
)
direct_register_custom_op(
    op_name="bux_picked_attention",
    op_func=picked_attention,
    mutates_args=["output"],
    fake_impl=_fake_picked,
)
