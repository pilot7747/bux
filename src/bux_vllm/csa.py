"""Compressed sparse attention in vLLM: bux's ``csa``, its latents in paged caches
of vLLM's own kinds, attending by `bux_vllm.compressed`.

vLLM 0.32 serves DeepSeek-V4 by layers and kernels of its own
(``vllm.models.deepseek_v4``), which bux's ``csa`` does not map onto. At a ratio of 4
their compressor pools overlapping windows of 8 tokens, by two sets of weights, with
a learned position bias on the gate; bux pools each group of ``compress_ratio``
tokens alone, with none (DeepSeek-V4.1's compressor, which vLLM takes at ratios 1 and
2 alone). Their indexer compresses the input by a compressor of its own and scores
FP8 or MXFP4 keys; bux's projects its keys from the layer's compressed latents and
scores them in full precision. Their kernels take heads of 512 with the last 64 dims
rotated, interleaved; bux's take any width, its first ``rope.fraction`` rotated.

So this layer keeps bux's arithmetic, on vLLM's paged caches: the window's token
latents (a sliding-window cache of one vector a token, ``SlidingWindowMLASpec``),
the compressed latents and their indexer keys (each one vector a
``compress_ratio`` tokens, ``MLAAttentionSpec`` with ``tokens_per_state``), and the
projections the next compressed latent pools (a window of ``compress_ratio``
tokens), as vLLM keeps DeepSeek-V4's. Its projections run in the compiled graph; what reads or writes the
caches runs in two ops of its own outside it (`POOL`, `ATTEND`), which split the
graph, as vLLM's attention does. A batch of decoded tokens alone has shapes of
``max_model_len``, not its longest sequence's, so that vLLM captures its steps in
one CUDA graph; a batch with prompts runs the ops between the graph's pieces.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from . import compressed, kernels
from .common import (
    ACTIVATIONS,
    Ctx,
    Layer,
    Site,
    logit_gain,
    own_cache_dtype,
    sharded_loader,
    unsupported,
)
from .linear import aligned, zero_padding

#: The ops that pool the compressed latents, and that attend, outside the graph.
POOL, ATTEND = "bux_csa_pool", "bux_csa"

#: The bytes the ops' chunks of queries take at most, which vLLM's profiling run,
#: which gives them no metadata, sets aside.
WORKSPACE = 20 * compressed.CHUNK


def check(c: dict, site: Site) -> None:
    """Raise for what the plugin's ``csa`` does not run.

    Raises:
        NotImplementedError: A relative bias, a length scale, an indexer of
            blocks, ``o_groups`` or heads that the ranks do not split, speculative
            decoding, decode context parallelism, or a KV cache dtype other than
            the model's.
    """
    unsupported(c, "relative_bias", "log_scale")
    ix = c["indexer"]
    if ix and ix["block_size"] > 1:
        raise NotImplementedError("csa: an indexer of blocks (block_size > 1)")
    tp = get_tensor_model_parallel_world_size()
    groups = c["o_groups"] if c["o_lora_rank"] else 1
    if c["num_heads"] % tp or (c["o_lora_rank"] and groups % tp):
        raise NotImplementedError(f"csa: heads or o_groups that {tp} ranks split")
    vllm = site.vllm_config
    if vllm.num_speculative_tokens:
        raise NotImplementedError("csa: speculative decoding")
    if vllm.parallel_config.decode_context_parallel_size > 1:
        raise NotImplementedError("csa: decode context parallelism")
    own_cache_dtype(vllm, "csa")


# --- the caches, their backend and metadata ----------------------------------------


@dataclass
class Metadata:
    """A batch, as one of a call's caches sees it.

    Attributes:
        slot_mapping: Each token's row, for a cache of a row a token, (T,).
        block_table: Each sequence's blocks, (sequences, blocks).
        token_to_req: Each token's sequence, (T,).
        num_actual_tokens: The tokens of the batch, padding left out.
        decode_tokens: How many of the first tokens are decoded ones.
        prompts: Each prompt chunk's first token, tokens, sequence and length.
        max_seq_len: The longest sequence's length, or more: ``max_model_len`` for
            a batch of decoded tokens alone, whose shapes a CUDA graph fixes.
    """

    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    token_to_req: torch.Tensor
    num_actual_tokens: int
    decode_tokens: int
    prompts: list[tuple[int, int, int, int]]
    max_seq_len: int

    def steps(self, positions: torch.Tensor) -> compressed.Steps:
        """The batch's tokens at `positions`, padding left out."""
        n = self.num_actual_tokens
        seq = self.token_to_req[:n].long()
        return compressed.Steps(
            positions[:n].long(),
            seq,
            self.decode_tokens,
            self.prompts,
            self.max_seq_len,
        )


def prompts(
    cm: CommonAttentionMetadata, decodes: int
) -> list[tuple[int, int, int, int]]:
    """Each prompt chunk of the batch `cm` after its `decodes` decoded sequences:
    its first token, tokens, sequence and the sequence's length with it."""
    lengths = cm.seq_lens_cpu_upper_bound
    lengths = cm.seq_lens.cpu() if lengths is None else lengths
    starts = cm.query_start_loc_cpu[: cm.num_reqs + 1].tolist()
    return compressed.prompt_chunks(starts, lengths.tolist(), decodes)


class Builder(AttentionMetadataBuilder[Metadata]):
    """The metadata of a call's caches: vLLM's slot mapping and block table, each
    token's sequence, and the batch's decoded tokens first (it reorders them so).
    A batch of decoded tokens alone is one CUDA graph."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)
        tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.requests = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.longest = vllm_config.model_config.max_model_len

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Metadata:
        cm = common_attn_metadata
        decodes, _, decode_tokens, _ = split_decodes_and_prefills(cm)
        chunks = prompts(cm, decodes)
        return Metadata(
            cm.slot_mapping,
            cm.block_table_tensor,
            cm.token_to_req_indices(self.requests),
            cm.num_actual_tokens,
            decode_tokens,
            chunks,
            cm.max_seq_len if chunks else self.longest,
        )


class Backend(AttentionBackend):
    """The backend of a call's caches: the metadata `Builder` makes. The layer's
    ops attend, so it has no implementation of its own."""

    @staticmethod
    def get_name() -> str:
        return "BUX_CSA"

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[Builder]:
        return Builder


class Rows(nn.Module, AttentionLayerBase):
    """One of a call's paged caches, vLLM's layer `prefix`: rows of `width` values,
    one for each `per` tokens, of its last `window` tokens (None for all)."""

    def __init__(
        self,
        prefix: str,
        width: int,
        per: int,
        window: int | None,
        dtype: torch.dtype,
        vllm_config: Any,
    ):
        super().__init__()
        self.prefix, self.width, self.per, self.window = prefix, width, per, window
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        _register(vllm_config.compilation_config, prefix, self)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)  # (blocks, 1, rows, width)

    def get_kv_cache_spec(self, vllm_config: Any) -> KVCacheSpec:
        tokens = vllm_config.cache_config.block_size
        one = {"num_kv_heads": 1, "max_tp_shards": 1, "head_size": self.width}
        if self.window is not None:
            return SlidingWindowMLASpec(
                block_size=tokens, sliding_window=self.window, dtype=self.dtype, **one
            )
        return MLAAttentionSpec(
            block_size=compressed.block_size(tokens, self.per),
            tokens_per_state=self.per,
            dtype=self.dtype,
            **one,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Backend

    def paged(self, meta: dict) -> compressed.Paged:
        """The cache in the batch of `meta`, vLLM's metadata by layer."""
        return compressed.Paged(self.kv_cache, meta[self.prefix].block_table)


class Call(Rows):
    """A ``csa`` call's caches: the token latents' window, vLLM's layer `prefix`,
    with the compressed latents', their indexer keys' and the pooling's beside it
    (`compressed.cache_rows`), and the `settings` its ops attend by."""

    def __init__(
        self,
        c: dict,
        settings: compressed.Settings,
        prefix: str,
        dtype: torch.dtype,
        vllm_config: Any,
        ports: list | None = None,
    ):
        rows = compressed.cache_rows(c, ports)
        super().__init__(prefix, *rows["window"], dtype, vllm_config)
        self.settings = settings
        self.compressed, self.index, self.state = (
            Rows(f"{prefix}.{name}", *rows[name], dtype, vllm_config)
            if name in rows
            else None
            for name in ("compressed", "index", "state")
        )

    def caches(
        self, meta: dict, n: int, writer: "Call | None" = None
    ) -> compressed.Caches:
        """Its caches in the batch of `meta`, of n tokens: with the compressed
        latents and indexer keys of the call `writer`, where it reads its ``kv``
        port."""
        out = compressed.Caches(self.paged(meta), meta[self.prefix].slot_mapping[:n])
        latents = self if writer is None else writer
        if latents.compressed is not None:
            out.compressed = latents.compressed.paged(meta)
        if latents.index is not None:
            out.index = latents.index.paged(meta)
        if self.state is not None:
            out.state = self.state.paged(meta)
            out.state_slots = meta[self.state.prefix].slot_mapping[:n]
        return out


# --- the ops -----------------------------------------------------------------------


def _batch(name: str, device: torch.device) -> tuple[Call, dict | None]:
    """The call whose window is vLLM's layer `name`, and the batch's metadata by
    layer: None in a run without any (vLLM's profiling), which sets aside on
    `device` the memory the ops' chunks take."""
    context = get_forward_context()
    call, meta = context.no_compile_layers[name], context.attn_metadata
    if meta:
        return call, meta
    torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
    return call, None


def pool_op(
    both: torch.Tensor, pooled: torch.Tensor, positions: torch.Tensor, layer_name: str
) -> None:
    """`POOL`: write the projections and gates, both (T, 2d), to the call's
    pooling cache, and the pooled group each token ends to `pooled` (T, d)."""
    call, meta = _batch(layer_name, both.device)
    if meta is None:
        return
    m = meta[layer_name]
    n, ratio = m.num_actual_tokens, call.settings.ratio
    assert ratio is not None
    steps = m.steps(positions)
    pooled[:n] = compressed.pool(call.caches(meta, n), steps, both[:n], ratio)


def attend_op(
    q: torch.Tensor,
    latent: torch.Tensor,
    rows: torch.Tensor | None,
    index_q: torch.Tensor | None,
    index_w: torch.Tensor | None,
    positions: torch.Tensor,
    sinks: torch.Tensor | None,
    picked: torch.Tensor | None,
    inside: torch.Tensor | None,
    out: torch.Tensor,
    kept: torch.Tensor,
    pool: torch.Tensor,
    layer_name: str,
    writer: str,
) -> None:
    """`ATTEND`: write the token latents and compressed rows to the call's caches,
    pick the compressed latents each query keeps (or take the `picked` its
    ``index`` port reads, or pick inside the pool its ``candidates`` port reads),
    and attend to them and the window; the heads' outputs to `out` (T, h, d)
    (`compressed.mix`), and the picks and the pool made to `kept` (T, top_k) and
    `pool` (T, k), unless they are empty. A call that reads its ``kv`` port reads
    the compressed latents of the call `writer`."""
    call, meta = _batch(layer_name, q.device)
    if meta is None:
        return
    m = meta[layer_name]
    n = m.num_actual_tokens
    index = None
    if index_q is not None and index_w is not None:
        index = (index_q[:n], index_w[:n])
    picks = compressed.Picks(
        None if picked is None else picked[:n],
        None if inside is None else inside[:n],
        pool.numel() > 0,
    )
    theirs = get_forward_context().no_compile_layers[writer] if writer else None
    kernel = kernels.takes_csa(q.dtype, q.shape[-1])
    out[:n], own, made = compressed.mix(
        call.caches(meta, n, theirs),
        m.steps(positions),
        call.settings,
        q[:n],
        latent[:n],
        None if rows is None else rows[:n],
        index,
        sinks,
        picks,
        kernels.csa_attend if kernel else None,
        kernels.csa_select if kernel else None,
    )
    if kept.numel() and own is not None:
        kept[:n] = torch.nn.functional.pad(
            own, (0, kept.shape[1] - own.shape[1]), value=-1
        )
    if pool.numel() and made is not None:
        pool[:n] = made


direct_register_custom_op(op_name=POOL, op_func=pool_op, mutates_args=["pooled"])
direct_register_custom_op(
    op_name=ATTEND, op_func=attend_op, mutates_args=["out", "kept", "pool"]
)


def _register(compilation: Any, prefix: str, layer: nn.Module) -> None:
    """Make `layer` vLLM's layer `prefix`, and the ops places the graph splits at."""
    if prefix in compilation.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation.static_forward_context[prefix] = layer
    ops = compilation.splitting_ops
    for op in (POOL, ATTEND):
        if ops is not None and f"vllm::{op}" not in ops:
            ops.append(f"vllm::{op}")


def _sharded(*shape: int, dtype: torch.dtype | None = None) -> nn.Parameter:
    """A parameter split over ranks along its first dim: this rank's `shape`."""
    p = nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
    p.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]
    return p


# --- the layer ---------------------------------------------------------------------


class CompressedSparseAttention(Layer):
    """bux's ``csa``: queries from the input or a normalized latent, a token latent
    and, every ``compress_ratio`` tokens, a compressed latent pooled under a softmax
    gate, the indexer's top-k of those, one softmax over them, the window and the
    sinks, the outputs rotated back, normalized and gated, and the output projection
    (dense, or grouped and low-rank).

    The query heads split over tensor-parallel ranks, and the output projection's
    groups with them; the latents, the compressed latents and the indexer are every
    rank's, as bux's are. Each call keeps caches of its own, over the same weights.

    Its ports share the compressed latents with later layers, DeepSeek-V4.1's: one
    that reads ``kv`` attends to the compressed latents (and their indexer keys) of
    the call that wrote it, keeping its window alone; one that reads ``index``
    takes that call's picks; one that reads ``candidates`` picks inside its
    candidate pool, which a call that writes ``candidates`` makes.
    """

    bux_types = ("csa",)
    stateful = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        check(c, site)
        tp = get_tensor_model_parallel_world_size()
        self.path, self.d, self.ratio = site.path, c["head_dim"], c["compress_ratio"]
        self.h, self.ports = c["num_heads"] // tp, site.ports
        if not compressed.compresses(c, site.ports):
            self.ratio = None  # its own: the compressed latents are the kv port's
        self.parts = compressed.Parts(c, site.ports)
        kw = {"bias": False, "return_bias": False, "quant_config": site.quant}
        self._inputs(c, width, site.path, kw)
        self._outputs(c, width, site.path, tp, kw)
        self.sinks = _sharded(self.h, dtype=torch.float32) if c["sinks"] else None
        scale = (c["scale"] or self.d**-0.5) * logit_gain(c["rope"])
        settings, dtype = (
            compressed.Settings.of(c, scale),
            own_cache_dtype(site.vllm_config, "csa"),
        )
        self.names, self.shares, self.settings = (
            list(site.caches),
            site.shares,
            settings,
        )
        self.calls = nn.ModuleList(
            Call(c, settings, name, dtype, site.vllm_config, site.ports)
            for name in site.caches
        )

    def _inputs(self, c: dict, width: int, path: str, kw: dict) -> None:
        """The projections of the input, of the linear layers' arguments `kw`: the
        queries', split by head, and every rank's by one GEMM (`down`): the query
        latent's, the token latent's, the compressed latent's and its gate's, and
        the indexer's head weights."""
        d, latent, ix = c["head_dim"], c["q_lora_rank"], c["indexer"]
        self.down_parts, self.down_split = compressed.down(c, self.ports)
        sizes, split = aligned(
            [n for _, n in self.down_parts], [n for _, n in self.down_split], 1
        )
        if len(split) > len(self.down_split):  # a part of its own, which none reads
            self.down_split = [*self.down_split, ("padding", split[-1])]
        self.down = MergedColumnParallelLinear(
            width, sizes, disable_tp=True, prefix=f"{path}.down", **kw
        )
        zero_padding(self.down, sum(sizes) - sum(n for _, n in self.down_parts))
        self.q = ColumnParallelLinear(
            latent or width, c["num_heads"] * d, prefix=f"{path}.q", **kw
        )
        self.index_q = None
        if ix and not compressed.reads(self.ports, "index"):
            self.index_q = ReplicatedLinear(
                latent or width,
                ix["num_heads"] * ix["head_dim"],
                prefix=f"{path}.indexer.q",
                **kw,
            )

    def _outputs(self, c: dict, width: int, path: str, tp: int, kw: dict) -> None:
        """The gate on the heads' outputs, by head, and its head bias; the output
        projection, row-parallel, after a grouped low-rank one (``o_a``, its groups
        split by rank, in the model's dtype) where ``o_lora_rank`` sets it."""
        h, d, rank = c["num_heads"], self.d, c["o_lora_rank"]
        self.gate = self.gate_bias = None
        self.gate_activation = ACTIVATIONS[c["gate_activation"]]
        if c["gated"]:
            n = h if c["gate_per_head"] else h * d
            self.gate = ColumnParallelLinear(width, n, prefix=f"{path}.gate", **kw)
            if c["gate_head_bias"]:
                self.gate_bias = _sharded(self.h)
        groups = c["o_groups"]
        self.o_a = _sharded(groups // tp, h * d // groups, rank) if rank else None
        self.o = RowParallelLinear(
            groups * rank if rank else h * d, width, prefix=f"{path}.o", **kw
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            f"{name}.weight": (self.down.weight, (i,))
            for i, (name, _) in enumerate(self.down_parts)
        }
        out |= {name: (p, ()) for name, p in self.parts.named_parameters()}
        for name, layer in (("q", self.q), ("indexer.q", self.index_q)):
            if layer is not None:
                out[f"{name}.weight"] = (layer.weight, ())
        for name in ("gate", "o"):
            if (layer := getattr(self, name)) is not None:
                out[f"{name}.weight"] = (layer.weight, ())
        for name in ("gate_bias", "o_a", "sinks"):
            if (p := getattr(self, name)) is not None:
                out[name] = (p, ())
        return out

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        t, pos = x.shape[0], ctx.positions
        name = self.names[ctx.call(self.path)]
        down = self._down(x)
        query = x
        if self.parts.q_a_norm is not None:
            query = self.parts.q_a_norm(down["q_a"])
        q = self.parts.queries(self.q(query).view(t, self.h, self.d), pos)
        latent = self.parts.latents(down["kv"], pos)
        rows = self._compressed(down, pos, name)
        index_q, index_w = self._index(down, query, pos)
        picked, inside = self._read(ctx, "index"), self._read(ctx, "candidates")
        kept, pool = self._made(q, "index"), self._made(q, "candidates")
        out = x.new_zeros(t, self.h, self.d)
        torch.ops.vllm.bux_csa(
            q,
            latent,
            rows,
            index_q,
            index_w,
            pos,
            self.sinks,
            picked,
            inside,
            out,
            kept,
            pool,
            name,
            self._writer(name),
        )
        self._write(ctx, "index", picked if picked is not None else kept)
        self._write(ctx, "candidates", pool)
        return self._output(x, self.parts.unrotate(out, pos))

    def _writer(self, name: str) -> str:
        """The call whose compressed latents the call `name` reads through its
        ``kv`` port, by vLLM's name; empty for its own."""
        if not self.shares:
            return ""
        return self.shares[self.names.index(name)] or ""

    def _read(self, ctx: Ctx, port: str) -> torch.Tensor | None:
        """What the layer reads from its port `port`; None if it reads none."""
        tap = next(
            ((t, m == "pop") for n, t, m in self.ports if n == port and m != "write"),
            None,
        )
        return None if tap is None else ctx.read(*tap)

    def _made(self, q: torch.Tensor, port: str) -> torch.Tensor:
        """Where the op writes what the layer pushes to its port `port` (its picks,
        or its candidate pool), (T, k) by index; empty if it writes none."""
        s = self.settings
        if not compressed.writes(self.ports, port):
            return q.new_zeros(0, dtype=torch.long)
        width = s.top_k if port == "index" else s.pool[1] if s.pool else 0
        return q.new_full((q.shape[0], width or 0), -1, dtype=torch.long)

    def _write(self, ctx: Ctx, port: str, x: torch.Tensor | None) -> None:
        """Push `x` to the taps of the layer's port `port`."""
        for n, tap, mode in self.ports:
            if n == port and mode == "write" and x is not None:
                ctx.push(tap, x)

    def _down(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The parts of `down`'s output, by name (``down_split``)."""
        sizes = [n for _, n in self.down_split]
        return dict(zip([n for n, _ in self.down_split], self.down(x).split(sizes, -1)))

    def _compressed(
        self, down: dict[str, torch.Tensor], pos: torch.Tensor, name: str
    ) -> torch.Tensor | None:
        """The rows of the compressed cache (`compressed.Parts.compressed`), from the
        projection of each token, or of the group of compress_ratio it ends, pooled
        by `POOL`; None without compressed latents."""
        if self.ratio is None:
            return None
        kv = down["compress"]
        if self.ratio > 1:
            kv = kv.new_zeros(kv.shape[0], self.d)
            torch.ops.vllm.bux_csa_pool(down["compress"], kv, pos, name)
        return self.parts.compressed(kv, pos)

    def _index(
        self, down: dict[str, torch.Tensor], query: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The indexer's query heads and their weights; None without an indexer."""
        ix = self.parts.indexer
        if ix is None or self.index_q is None:
            return None, None
        return ix.queries(self.index_q(query), down.get("indexer.weights"), pos)

    def _output(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The heads' outputs, (T, h, d), normalized, gated and projected back."""
        t = x.shape[0]
        if self.parts.out_norm is not None:
            out = self.parts.out_norm(out)
        if self.gate is not None:
            gate = self.gate(x).view(t, self.h, -1)
            if self.gate_bias is not None:
                gate = gate + self.gate_bias.to(gate.dtype)[:, None]
            out = out * self.gate_activation(gate)
        if self.o_a is None:
            return self.o(out.reshape(t, -1))
        grouped = out.reshape(t, self.o_a.shape[0], -1)
        return self.o(torch.einsum("tgi,gir->tgr", grouped, self.o_a).reshape(t, -1))
