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

Where vLLM's DeepSeek-V3.2 kernels take a layer's indexer (`bux_vllm.dsa`), its
keys are FP8 in head slots of their own after the kv heads', in DeepGEMM's paged
layout, in blocks of 64 tokens (`IndexedBackend`). Such a cache, and one of a layer
that reads its ``index`` port, takes an FP8 KV cache: its keys and values in e4m3
or e5m2, at a scale of 1, as vLLM's attention keeps them.

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
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from . import caching, compressed, dsa, kernels, msa, picks
from .common import own_cache_dtype
from .kv import reads, writes

#: The ops, as vLLM's compilation names them.
OPS = (
    "vllm::bux_sparse_attention",
    "vllm::bux_picked_attention",
    "vllm::bux_biased_attention",
)
#: The FP8 KV caches a sparse call's cache takes, by vLLM's ``kv_cache_dtype``.
FP8_CACHES = {
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
PAGE = dsa.PAGE


@dataclass
class Batch:
    """A step's tokens, as a `PagedCache`'s ops read them.

    Attributes:
        slot_mapping: Each token's slot, (T,); -1 for padding.
        block_table_tensor: Each request's blocks, (requests, blocks).
        num_actual_tokens: The step's tokens, padding left out.
        segments: Its queries by `bux_vllm.picks.Segment`: its decoded tokens, one
            a request, then each prompt chunk.
        decoding: Its decoded tokens as vLLM's DSA kernels take them, by the
            indexer's block size (their keys or blocks), for an `IndexedBackend`'s
            caches; None without.
        seq_lens: Each request's keys, (requests,) int32, as MSA's kernels read
            them.
    """

    slot_mapping: torch.Tensor
    block_table_tensor: torch.Tensor
    num_actual_tokens: int
    segments: list[picks.Segment]
    decoding: dict[int, dsa.Decoding] | None = None
    seq_lens: torch.Tensor | None = None


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
        batch = Batch(
            cm.slot_mapping, cm.block_table_tensor, cm.num_actual_tokens, segments
        )
        batch.seq_lens = cm.seq_lens
        return batch


class IndexedMetadataBuilder(PagedMetadataBuilder):
    """A `PagedMetadataBuilder` for caches whose indexer runs on vLLM's DSA kernels
    (`IndexedBackend`): with each decoded token's keys and DeepGEMM's schedule of
    them, in buffers that a CUDA graph's replays read in place."""

    def __init__(
        self, kv_cache_spec: Any, layer_names: list[str], vllm_config: Any, device: Any
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        context = vllm_config.compilation_config.static_forward_context
        self.sizes = sorted({context[name].sparse.block_size for name in layer_names})
        most = vllm_config.scheduler_config.max_num_batched_tokens
        shape = (most, 1)
        self.lens = {
            n: torch.zeros(shape, dtype=torch.int32, device=device) for n in self.sizes
        }
        self.plans: dict[int, torch.Tensor] = {}

    def _batch(self, cm: CommonAttentionMetadata, longest: int) -> Batch:
        batch = super()._batch(cm, longest)
        decoding = [part for part in batch.segments if part.decoding]
        if decoding:
            count = decoding[0].tokens.stop
            batch.decoding = {n: self._decoding(cm, count, n) for n in self.sizes}
        return batch

    def _decoding(
        self, cm: CommonAttentionMetadata, count: int, size: int
    ) -> dsa.Decoding:
        """The step's `count` decoded tokens over their keys, or their complete
        blocks of `size`: how many each sees, and DeepGEMM's schedule of them."""
        lens = self.lens[size][:count]
        lens[:, 0].copy_(cm.seq_lens[:count] // size)
        plan = dsa.schedule(lens, PAGE)
        if size not in self.plans:
            self.plans[size] = torch.empty_like(plan)
        self.plans[size].copy_(plan)
        return dsa.Decoding(lens, self.plans[size], self.longest // size)


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
    supported_kv_cache_dtypes: ClassVar = ["auto", "float16", "bfloat16", *FP8_CACHES]

    @staticmethod
    def get_name() -> str:
        return "BUX_PAGED"

    @staticmethod
    def get_impl_cls() -> Any:
        raise NotImplementedError("a bux paged cache attends in bux's own op")

    @staticmethod
    def get_builder_cls() -> type[PagedMetadataBuilder]:
        return PagedMetadataBuilder


class IndexedBackend(PagedBackend):
    """The backend of a `PagedCache` whose indexer runs on vLLM's DSA kernels: its
    blocks of `PAGE` tokens, as DeepGEMM's paged kernel reads them (vLLM splits
    larger blocks into such), and a layout that keeps a block's head slots
    together, where the indexer keys lie."""

    @staticmethod
    def get_name() -> str:
        return "BUX_PAGED_DSA"

    @staticmethod
    def get_builder_cls() -> type[IndexedMetadataBuilder]:
        return IndexedMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec: Any = None) -> list:
        return [PAGE]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


class BlockedBackend(PagedBackend):
    """The backend of a `PagedCache` whose indexer and attention run on vLLM's MSA
    kernels (`bux_vllm.msa`): its blocks of 128 tokens, one of MSA's to a page, and
    a layout that keeps a block's head slots together, where the index keys lie."""

    @staticmethod
    def get_name() -> str:
        return "BUX_PAGED_MSA"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec: Any = None) -> list:
        return [msa.BLOCK]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


class PagedCache(nn.Module, AttentionLayerBase):
    """A sparse call's `heads` vectors of `channels` a token, and its indexer's keys
    of `index_dims`, in vLLM's paged KV cache, allocated as for full attention, which
    `kv_cache` views as ``[blocks, slots, block_size, row]``: each kv head's
    channels, and the indexer key's spread over them (`bux_vllm.picks.Cache`), or,
    where vLLM's DSA kernels pick (`native`), the FP8 keys in slots of their own
    after the kv heads' (`bux_vllm.dsa.Region`).

    Args:
        heads: The kv heads of a token.
        channels: Each head's channels: a key's and a value's, or MLA's latent.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of the cache, unique in the model.
        sparse: The settings of the call, which the op attends with.
        index_dims: The channels of its indexer's key; 0 for a call that reads its
            picks from its ``index`` port.
        what: The layer, as its errors name it.
        native: Whether vLLM's DSA kernels pick its keys (`bux_vllm.dsa.takes`).
        blocked: Whether vLLM's MSA kernels pick and attend (`bux_vllm.msa.takes`).
        fp8: Whether its ops read an FP8 KV cache (the Triton kernels' do, over
            keys and values alone or with FP8 indexer keys).

    Raises:
        NotImplementedError: ``kv_cache_dtype`` is other than the model's, but for
            fp8 where the call takes it (`cache_dtype`).
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
        native: bool = False,
        fp8: bool = False,
        blocked: bool = False,
    ):
        super().__init__()
        self.model_dtype = vllm_config.model_config.dtype
        self.dtype, self.fp8 = cache_dtype(vllm_config, what, fp8)
        self.heads, self.channels, self.prefix = heads, channels, prefix
        self.index_dims, self.native, self.sparse = index_dims, native, sparse
        self.blocked = blocked
        self.share = 0 if native or blocked else -(-index_dims // heads)
        size, raw = self.dtype.itemsize, (native and sparse.mean) or blocked
        item = self.model_dtype.itemsize
        self.raw_slots = dsa.raw_slots(index_dims, channels * size, item) if raw else 0
        keys = dsa.slots(index_dims, channels, size) if native else 0
        self.slots = self.raw_slots + keys
        self.scales = (1.0, 1.0)  # an FP8 cache's keys' and values' scales
        self.longest = vllm_config.model_config.max_model_len
        kernel = getattr(vllm_config, "kernel_config", None)
        self.topk_backend = getattr(kernel, "sparse_indexer_topk_backend", "auto")
        quant = get_kv_quant_mode(vllm_config.cache_config.cache_dtype)
        self.quant = quant if self.fp8 is not None else KVQuantMode.NONE
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.blocked:
            return BlockedBackend
        return IndexedBackend if self.native else PagedBackend

    def get_kv_cache_spec(self, vllm_config: Any) -> FullAttentionSpec:
        row = self.channels + self.share
        half = row // 2  # a key's and a value's dims, as vLLM sums them
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.heads + self.slots,
            head_size=row - half,
            head_size_v=half,
            dtype=self.dtype,
            kv_quant_mode=self.quant,
        )

    @property
    def rows(self) -> picks.Cache:
        """The cache, its keys and values apart from its indexer keys."""
        if self.slots:
            return picks.Cache(self.kv_cache[:, : self.heads], self.channels)
        return picks.Cache(self.kv_cache, self.channels, self.index_dims)

    @property
    def kv(self) -> torch.Tensor:
        """The keys and values, ``[blocks, heads, block_size, channels]``, as FP8
        in an FP8 cache."""
        kv = self.rows.kv
        return kv if self.fp8 is None else kv.view(self.fp8)

    @property
    def region(self) -> dsa.Region:
        """The FP8 indexer keys (QSA's blocks'), where vLLM's DSA kernels pick."""
        first = self.heads + self.raw_slots
        return dsa.Region(self.kv_cache, first, self.index_dims)

    @property
    def raw(self) -> dsa.Raw:
        """QSA's indexer keys as projected, where vLLM's DSA kernels pick; MSA's
        index keys, where its kernels do."""
        return dsa.Raw(self.kv_cache, self.heads, self.index_dims, self.model_dtype)

    def write(
        self,
        slots: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keys: torch.Tensor | None = None,
    ) -> None:
        """Write the tokens' keys and values, (T, heads, dk) and (T, heads, dv), at
        vLLM's `slots`, (T,): in FP8, divided by their scales, in an FP8 cache; and
        their indexer keys for vLLM's DSA kernels, (T, i), into the `region`. On a
        GPU in one launch (`bux_vllm.caching.write`), bit for bit."""
        if k.is_cuda:
            first = self.heads + self.raw_slots
            caching.write(self.kv, slots, k, v, keys, first, self.scales)
            return
        rows = torch.cat([k, v], -1)
        if self.fp8 is not None:  # its bytes, as indexing writes them
            most = torch.finfo(self.fp8).max
            dk = self.sparse.dk
            k, v = rows[..., :dk] / self.scales[0], rows[..., dk:] / self.scales[1]
            rows = torch.cat([k, v], -1).float().clamp(-most, most)
            rows = rows.to(self.fp8).view(torch.uint8)
        picks.write(self.rows.kv, slots, rows)
        if keys is not None:
            self.region.write(slots, keys)


def cache_dtype(
    vllm_config: Any, what: str, fp8: bool
) -> tuple[torch.dtype, torch.dtype | None]:
    """The dtype a sparse call's cache keeps, as vLLM allocates it, and its FP8
    dtype or None: under ``kv_cache_dtype`` fp8 (or fp8_e4m3, fp8_e5m2), bytes read
    as e4m3 (e5m2) where the call takes it (`fp8`), else the model's.

    Raises:
        NotImplementedError: ``kv_cache_dtype`` is other than the model's, where
            the call does not take it (`bux_vllm.common.own_cache_dtype`).
    """
    name = vllm_config.cache_config.cache_dtype
    if fp8 and name in FP8_CACHES:
        return torch.uint8, FP8_CACHES[name]
    return own_cache_dtype(vllm_config, what), None


def paths(
    spec: picks.Sparse,
    ix: dict | None,
    dims: tuple[int, int],
    dtype: torch.dtype,
    kv_heads: int = 0,
) -> tuple[bool, bool, bool]:
    """Whether vLLM's DSA kernels pick a call's keys, by its indexer's config `ix`
    (`bux_vllm.dsa.takes`), in a model of `dtype`; whether its MSA kernels pick and
    attend instead (`bux_vllm.msa.takes`, the layer of `kv_heads`); and whether its
    ops take an FP8 KV cache: theirs, and the Triton kernels' over keys and values
    alone, of heads of `dims` (a layer that reads its ``index`` port), but not MLA's
    latents."""
    native = ix is not None and dsa.takes(spec, ix["head_dim"], ix["num_heads"], dtype)
    blocked = ix is not None and not native and dims[0] == dims[1]
    blocked = blocked and msa.takes(spec, ix, kv_heads, dims[0], dtype)
    alone = ix is None and kernels.takes(spec, dims, dtype)
    return native, blocked, (native or blocked or alone) and not spec.latent


def layer_caches(
    spec: picks.Sparse,
    heads: int,
    channels: int,
    index_dims: int,
    names: dict[str, list[str]],
    vllm_config: Any,
    what: str,
    native: bool = False,
    fp8: bool = False,
    blocked: bool = False,
) -> nn.ModuleList:
    """Each call's `PagedCache` of the layer `what`, by the names of its states: its
    keys and values, `heads` vectors of `channels` a token, attended with `spec`,
    and its indexer's keys, `index_dims` a token (0 for a layer that reads its
    picks from its ``index`` port), FP8 ones where vLLM's DSA kernels pick them
    (`native`), or bfloat16 ones where its MSA kernels do (`blocked`); in an FP8 KV
    cache where the call's ops take it (`fp8`)."""
    args = (spec, index_dims, what, native, fp8, blocked)
    caches = nn.ModuleList(
        PagedCache(heads, channels, vllm_config, name, *args)
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
    q, k, v = tensors[0].contiguous(), tensors[1], tensors[2]  # writes take strides
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
        native = cache.native  # the native ops take each row's strides
        torch.ops.vllm.bux_sparse_attention(
            q,
            k,
            v,
            *(t if native else t.contiguous() for t in index),
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
    cache = get_forward_context().no_compile_layers[layer_name]
    if metas is None or layer_name not in metas:
        output.zero_()
        if cache.native:
            dsa.reserve(q.device, cache.longest, cache.index_dims)
        if cache.blocked:
            msa.reserve(q, cache.sparse.groups[1], cache.longest)
        return
    meta = metas[layer_name]
    n = meta.num_actual_tokens
    output[n:].zero_()
    _written(cache, meta, k[:n], v[:n], index_k[:n])
    table, tensors = meta.block_table_tensor, (q, index_q, index_w, positions)
    if cache.native:
        _pooled(cache, meta, positions)
        heads = dsa.padded_heads(index_q.shape[1])
        index_q, index_w = dsa.queries(index_q, index_w, heads)
        tensors = (q, index_q, index_w, positions)
    for part in meta.segments:
        at = part.tokens
        q_, iq, iw, pos = (t[at] for t in tensors)
        rows = table[part.requests]
        if cache.blocked:
            out, chosen = _blocked(cache, meta, part, q_, iq, rows)
            output[at] = out.to(output.dtype)
            if picked_keys.numel():
                picked_keys[at], picked_valid[at] = msa.as_picks(chosen, pos)
            continue
        wanted, into = picked_keys.numel() > 0, output[at]
        args = (q_, (iq, iw), pos, rows, wanted, into)
        out, kept = _segment(cache, meta, part, *args)
        if out is not into:
            into.copy_(out)
        if kept is not None:
            picked_keys[at], picked_valid[at] = kept


def _written(
    cache: PagedCache,
    meta: Batch,
    k: torch.Tensor,
    v: torch.Tensor,
    index_k: torch.Tensor,
) -> None:
    """Write the step's keys and values, (n, g, dk) and (n, g, dv), and its indexer
    keys, (n, i), into `cache`: in FP8 where vLLM's DSA kernels pick, as projected
    for QSA's and MSA's, else spread over the rows."""
    slots = meta.slot_mapping[: k.shape[0]]
    if not (cache.native or cache.blocked):
        kv = torch.cat([k, v], -1)
        picks.write(cache.kv_cache, slots, picks.merged(kv, index_k))
        return
    if cache.sparse.mean or cache.blocked:
        cache.write(slots, k, v)
        cache.raw.write(slots, index_k)
    else:
        cache.write(slots, k, v, index_k)


def _blocked(
    cache: PagedCache,
    meta: Batch,
    part: picks.Segment,
    q: torch.Tensor,
    iq: torch.Tensor,
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A segment's outputs, (Q, h, dv), and picks, (g, Q, k) block ids, by vLLM's MSA
    kernels (`bux_vllm.msa`): its index queries `iq`, (Q, H, i), of this rank's kv
    heads' groups, over the call's index keys, and its queries over its kv heads."""
    spec = cache.sparse
    first, count = spec.groups
    iq = iq[:, first : first + count].contiguous()
    attending = (spec.top_k // msa.BLOCK, spec.scale, cache.scales)
    keys, kv = cache.raw.keys, cache.kv
    if part.decoding:
        assert meta.seq_lens is not None, "decoded tokens have their keys' lengths"
        lens = meta.seq_lens[: q.shape[0]]
        return msa.decode(keys, iq, q, kv, rows, lens, cache.longest, attending)
    return msa.prompt(keys, iq, q, kv, rows, part.keys, attending)


def _pooled(cache: PagedCache, meta: Batch, positions: torch.Tensor) -> None:
    """Write the keys of the blocks the step's tokens complete, for QSA's indexer
    on vLLM's DSA kernels (`bux_vllm.dsa.pooled`), before any query picks."""
    spec = cache.sparse
    if not spec.mean:
        return
    table = meta.block_table_tensor
    for part in meta.segments:
        pos = positions[part.tokens]
        rows = table[part.requests].expand(len(pos), -1)
        dsa.pooled(cache.raw, cache.region, spec.indexer, rows, pos, spec.block_size)


def _segment(
    cache: PagedCache,
    meta: Batch,
    part: picks.Segment,
    q: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    rows: torch.Tensor,
    wanted: bool,
    into: torch.Tensor,
) -> tuple[torch.Tensor, picks.Picked | None]:
    """A segment's outputs, (Q, h, dv), and its picks if `wanted`: by vLLM's DSA
    kernels where they pick (`_native`, which may attend `into` the op's output),
    else by the Triton kernels or PyTorch (`_segments`)."""
    if cache.native:
        return _native(cache, meta, part, q, index, positions, rows, wanted, into)
    run = _segments(cache)
    out, kept = run(cache.sparse, q, index, positions, cache.rows, rows, part.keys)
    return out, kept if wanted else None


def _native(
    cache: PagedCache,
    meta: Batch,
    part: picks.Segment,
    q: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    rows: torch.Tensor,
    wanted: bool,
    into: torch.Tensor,
) -> tuple[torch.Tensor, picks.Picked | None]:
    """A segment's picks by vLLM's DSA kernels (decoded tokens' over their paged
    keys, a prompt chunk's over its request's gathered ones; QSA's over its blocks,
    and its own unfinished block), padded to `bux_vllm.picks.most` if `wanted`, and
    its attention over them: by vLLM's own sparse GQA attention where it takes the
    layer (`bux_vllm.dsa.attends`), `into` the op's output where it fits, else by
    the plugin's Triton kernel, or by PyTorch over MLA's latents."""
    spec, region, size = cache.sparse, cache.region, cache.sparse.block_size
    top_k = spec.top_k // size
    if part.decoding:
        assert meta.decoding is not None, "decoded tokens have their keys' lengths"
        step = meta.decoding[size]
        chosen = dsa.decode(region, index, rows, step, top_k, cache.topk_backend)
    else:
        ends = (positions + 1) // size  # the keys, or complete blocks, each sees
        chosen = dsa.prompt(region, index, rows, part.keys // size, ends, top_k)
    attends = not spec.latent and dsa.attends(spec, q, cache.kv)
    kept = None
    if spec.mean:
        kept = dsa.block_picks(chosen, positions, size)
    elif wanted or not attends:
        kept = dsa.as_picks(chosen)
    if spec.latent:
        assert kept is not None
        kv = cache.rows.kv
        out = picks.attend_picked(spec, q, kept, positions, kv, rows, part.keys)
        return out, kept
    if attends:
        if spec.mean and kept is not None:  # QSA's picks, read whole
            chosen = torch.where(kept[1], kept[0], -1)
        fits = into.dtype == q.dtype and into.shape == q.shape
        args = (not part.decoding, cache.scales, into if fits else None)
        out = dsa.attend(spec, q, cache.kv, rows, chosen, *args)
        return out, kept if wanted else None
    assert kept is not None
    table = rows.expand(q.shape[0], -1)
    out = kernels.attend(spec, q.contiguous(), cache.kv, table, kept, cache.scales)
    return out, kept if wanted else None


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
    cache.write(meta.slot_mapping[:n], k[:n], v[:n])
    table = meta.block_table_tensor
    for part in meta.segments:
        at = part.tokens
        picked = (picked_keys[at], picked_valid[at])
        args = (cache.sparse, q[at], picked, positions[at])
        rows = table[part.requests]
        if _kernels(cache):
            out = kernels.attend_picked(*args, cache.kv, rows, part.keys, cache.scales)
        else:
            out = picks.attend_picked(*args, cache.rows.kv, rows, part.keys)
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
    return kernels.takes(cache.sparse, dims, cache.model_dtype)


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
