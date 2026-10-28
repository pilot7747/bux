"""A ``gqa`` layer's sparse attention in vLLM: its keys and values and its
indexer's keys in vLLM's paged KV cache, and each query's attention over the keys
its indexer picks, as bux computes them (`bux_vllm.picks`).

Each call of a sparse layer keeps two of vLLM's caches, `PagedCache`: its keys and
values, as a dense layer's, and its indexer's keys, one vector a token. vLLM
allocates their blocks as for full attention, so they grow with the sequence and
serve prefix caching. The plugin writes and reads them itself, in one op,
``vllm::bux_sparse_attention``, which runs eagerly: vLLM splits its CUDA graphs
around it (`split_graphs`), as around attention, and captures no whole-step graph
of a model with one (`PagedMetadataBuilder`). The op also gives the keys each query
picked, which a layer that reads them through its ``index`` port attends over with
``vllm::bux_picked_attention``, keeping no indexer keys of its own.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from . import kernels, picks
from .kv import reads, writes

#: The ops, as vLLM's compilation names them.
OPS = ("vllm::bux_sparse_attention", "vllm::bux_picked_attention")


class PagedMetadataBuilder(AttentionMetadataBuilder[CommonAttentionMetadata]):
    """Gives a `PagedCache` the step's common metadata as it is: its block table,
    slots, sequence lengths and queries' starts. It supports no CUDA graph of a
    whole step (the builders' default), as the op reads them on the CPU."""

    def __init__(
        self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: Any
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CommonAttentionMetadata:
        return common_attn_metadata


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
    """`heads` vectors of `channels` a token in vLLM's paged KV cache, allocated as
    for full attention, which `kv_cache` views as ``[blocks, heads, block_size,
    channels]``.

    Args:
        heads: The vectors of a token.
        channels: Each vector's channels: a key's and a value's, or an indexer key's.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of the cache, unique in the model.
        sparse: The settings of the layer's call whose keys and values it keeps,
            which the op attends with; None for an indexer's cache.
        index_name: The name of that call's indexer cache.

    Raises:
        NotImplementedError: The KV cache is quantized, which the plugin's sparse
            attention does not read.
    """

    def __init__(
        self,
        heads: int,
        channels: int,
        vllm_config: Any,
        prefix: str,
        sparse: picks.Sparse | None = None,
        index_name: str = "",
    ):
        super().__init__()
        cache = vllm_config.cache_config
        if cache.cache_dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise NotImplementedError(
                f"gqa: an indexer with a {cache.cache_dtype} KV cache"
            )
        self.heads, self.channels, self.prefix = heads, channels, prefix
        self.dtype = vllm_config.model_config.dtype
        self.sparse, self.index_name = sparse, index_name
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PagedBackend

    def get_kv_cache_spec(self, vllm_config: Any) -> FullAttentionSpec:
        half = self.channels // 2  # a key's and a value's dims, as vLLM sums them
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.heads,
            head_size=self.channels - half,
            head_size_v=half,
            dtype=self.dtype,
        )


def layer_caches(
    spec: picks.Sparse,
    heads: int,
    channels: int,
    index_dims: int,
    names: dict[str, list[str]],
    vllm_config: Any,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Each call's caches of a sparse layer, by the names of its states: of its keys
    and values, `heads` vectors of `channels` a token, attended with `spec`, and of
    its indexer's keys, `index_dims` a token (none for 0, a layer that reads its
    picks from its ``index`` port)."""
    main = names["full_attention"]
    index = names["index"] if index_dims else [""] * len(main)
    caches = nn.ModuleList(
        PagedCache(heads, channels, vllm_config, name, spec, index_name)
        for name, index_name in zip(main, index, strict=True)
    )
    index_caches = nn.ModuleList(
        PagedCache(1, index_dims, vllm_config, name) for name in index if index_dims
    )
    split_graphs(vllm_config)
    return caches, index_caches


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
        kept = _no_picks(cache.spec, q, len(taps))
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
    spec: picks.Sparse | None, q: torch.Tensor, taps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the sparse op writes the step's picks, (T, [G,] K): none unless the
    layer pushes them to `taps`."""
    assert spec is not None
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
    indexer keys, (T, i), into the caches of the call `layer_name`, then write each
    query's attention over the keys it picks into `output`, (T, h, dv), and its
    picks into `picked_keys` and `picked_valid`, (T, [G,] K), unless they are
    empty; zeros in vLLM's profiling run, which has no metadata."""
    metas = _metadata()
    picked_valid.zero_()
    if metas is None or layer_name not in metas:
        output.zero_()
        return
    context = get_forward_context()
    cache = context.no_compile_layers[layer_name]
    index = context.no_compile_layers[cache.index_name]
    meta, index_meta = metas[layer_name], metas[cache.index_name]
    n = meta.num_actual_tokens
    output[n:].zero_()
    picks.write(cache.kv_cache, meta.slot_mapping[:n], torch.cat([k, v], -1)[:n])
    picks.write(index.kv_cache, index_meta.slot_mapping[:n], index_k[:n, None])
    caches = (cache.kv_cache, index.kv_cache)
    tables = (meta.block_table_tensor, index_meta.block_table_tensor)
    tensors = (q, index_q, index_w, positions)
    for part in picks.segments(*_bounds(meta), q.device):
        rows = (tables[0][part.requests], tables[1][part.requests])
        at = part.tokens
        q_, iq, iw, pos = (t[at] for t in tensors)
        out, (kept, valid) = _segments(cache)(
            cache.sparse, q_, (iq, iw), pos, caches, rows, part.keys
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
    for part in picks.segments(*_bounds(meta), q.device):
        at = part.tokens
        picked = (picked_keys[at], picked_valid[at])
        attend = kernels.attend_picked if _kernels(cache) else picks.attend_picked
        out = attend(
            cache.sparse,
            q[at],
            picked,
            positions[at],
            cache.kv_cache,
            table[part.requests],
            part.keys,
        )
        output[at] = out.to(output.dtype)


def _kernels(cache: PagedCache) -> bool:
    """Whether the Triton kernels attend over the call's `cache` (`bux_vllm.kernels`)."""
    assert cache.sparse is not None
    dims = (cache.sparse.dk, cache.channels - cache.sparse.dk)
    return kernels.takes(cache.sparse, dims)


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


def _bounds(meta: CommonAttentionMetadata) -> tuple[list[int], list[int]]:
    """The requests' first queries and the end, and the most keys of each, on the
    CPU."""
    upper = meta.seq_lens_cpu_upper_bound
    if upper is None:
        upper = meta.seq_lens.cpu()
    reqs = meta.num_reqs
    return meta.query_start_loc_cpu[: reqs + 1].tolist(), upper[:reqs].tolist()


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
direct_register_custom_op(
    op_name="bux_picked_attention",
    op_func=picked_attention,
    mutates_args=["output"],
    fake_impl=_fake_picked,
)
