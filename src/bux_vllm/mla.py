"""Multi-head latent attention in vLLM: bux's ``mla`` by vLLM's own MLA layer
(``MLAAttention``), its latent KV cache and its MLA backends.

vLLM caches what bux's MLA projects keys and values up from: each token's normalized
``kv_lora_rank`` latent and the ``qk_rope_head_dim`` key dims all heads share,
rotated, as one head of ``kv_lora_rank + qk_rope_head_dim``. It attends to a
prompt's new tokens with the keys and values projected up by ``kv_b`` (its MLA
prefill backend), and decodes in the latent space, with ``kv_b``'s halves absorbed
into the queries and the output (its MLA decode backend), as DeepSeek's MLA runs in
vLLM. The layer computes the rest as bux does: the query (from the input, or from
its normalized latent), the latents' norms and RoPE in float32, by bux's
frequencies, the scale, and the norm and gate of the heads' outputs.

vLLM picks the backends by the model's MLA dims, which it reads from the config
(`bux_vllm.config`, `bux_vllm.arch`), and the GPU. Two of its checks hold every
model to a few dims (DeepSeek's among them), whatever the backend takes: the MLA
metadata's head size (the latent and the rotated dims), and the prefill backends'
dims. For other dims, the layer gives vLLM's MLA the decoding backend vLLM picks,
its metadata checked by that backend's own head sizes (`any_head_size`), and
FlashAttention's prefill, which takes any dims up to 128 (`prefill_backend`). A
prefill that attends to cached context (a prompt's earlier chunks, a cached prefix)
gathers its latents by vLLM's op, whose kernel takes the same few sizes: for the
others the plugin gathers them in PyTorch (`extend_gather`, `bux_vllm.latent`).

With an indexer (DeepSeek-V3.2's), or reading another layer's picks through its
``index`` port, each query attends to the picked keys alone, in the latent space,
by the plugin's sparse attention ops over caches of its own (`bux_vllm.sparse`):
the same latent and shared key dims a token, beside the indexer's keys.

The projections take the engine's quantization config
(`bux_vllm.common.quantization`), as vLLM's DeepSeek's do: its MLA dequantizes
``kv_b``'s halves to absorb them. A sparse layer's ``kv_b`` stays in the model's
dtype, since the sparse ops read its weight themselves.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import torch
from torch import nn
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.mla_attention import MLACommonBackend
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.v1.attention.backend import subclass_attention_backend
from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend
from vllm.v1.attention.backends.mla.prefill.selector import get_mla_prefill_backend
from vllm.v1.attention.selector import get_attn_backend

from . import kv, latent, picks, sparse
from .attention import sink_logits
from .common import ACTIVATIONS, Ctx, Layer, Norm, Rope, Site, logit_gain, unsupported
from .config import LATENT, states
from .config import sparse as is_sparse
from .indexer import Indexer, rotated

#: The norm of MLA's latents: bux's default ``rmsnorm``.
LATENT_NORM = {
    "type": "rmsnorm",
    "eps": 1e-6,
    "affine": True,
    "bias": False,
    "zero_centered": False,
    "scale_input": False,
}


def check(c: dict, site: Site) -> None:
    """Raise for what vLLM's MLA, or the plugin's sparse attention in the latent
    space, does not run, or the model's config does not say.

    Raises:
        NotImplementedError: A setting that needs the keys projected up while
            decoding (QK-norm), or that neither takes (a relative bias, a length
            scale, the gate's head bias, ports but a sparse layer's ``index``);
            for vLLM's MLA, a window, soft-capping or sinks, dims other than the
            config's, a dtype other than bfloat16 or float16, or vLLM not running
            the model as MLA.
    """
    unsupported(c, "qk_norm", "relative_bias", "log_scale", "gate_head_bias")
    sparse_ = is_sparse(c, site.ports)
    names = {name for name, _, _ in site.ports} - ({"index"} if sparse_ else set())
    if names:
        raise NotImplementedError(f"mla: the {', '.join(sorted(names))} port")
    if sparse_:
        return
    unsupported(c, "sliding_window_size", "softcap", "sinks")
    if any(c[n] != getattr(site.config, n, None) for n in LATENT):
        raise NotImplementedError("mla: dims other than the config's not in vLLM")
    model = site.vllm_config.model_config
    if model.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"mla in {model.dtype}: vLLM's MLA takes 16 bits")
    if not model.use_mla:
        raise NotImplementedError("mla: vLLM runs the model without MLA")


def prefill_backend(vllm_config: Any) -> type | None:
    """The MLA prefill backend for the model's dims: None, vLLM's own pick, where it
    has one; FlashAttention's otherwise.

    Raises:
        ValueError: The engine's config names a prefill backend that does not take
            the dims.
    """
    try:
        get_mla_prefill_backend(vllm_config)
    except ValueError:
        if vllm_config.attention_config.mla_prefill_backend is not None:
            raise
        return FlashAttnPrefillBackend
    return None


def decode_backend(head_size: int, heads: int, cache_config: Any) -> Any:
    """The MLA backend for a latent cache of `head_size` a token: None, vLLM's own
    pick, where MLA's metadata takes the size; vLLM's pick by `any_head_size`
    otherwise, as `MLAAttention` would pick it (``--attention-backend`` too)."""
    if MLACommonBackend.supports_head_size(head_size):
        return None
    dtype, cache_dtype = torch.get_default_dtype(), cache_config.cache_dtype
    picked = get_attn_backend(
        head_size, dtype, cache_dtype, use_mla=True, num_heads=heads
    )
    return any_head_size(picked)


@cache
def any_head_size(backend: Any) -> Any:
    """`backend`, its metadata checked by the head sizes it takes itself rather
    than by every MLA backend's few (``MLACommonMetadata``'s check): vLLM's own
    subclass of a backend with a builder of its own, as for chunked attention."""

    extend_gather()

    class Builder(backend.get_builder_cls()):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.metadata_cls = checked_by(self.metadata_cls, backend)

    return subclass_attention_backend("Bux", backend, Builder)


@cache
def checked_by(metadata: type, backend: Any) -> type:
    """The MLA metadata class `metadata`, its head size checked by `backend`'s."""

    def check(self: Any) -> None:
        size = self.head_dim
        if size is not None and not backend.supports_head_size(size):
            raise ValueError(f"{backend.get_name()} takes no head size {size}")

    return type(metadata.__name__, (metadata,), {"__post_init__": check})


def extend_gather() -> None:
    """Make vLLM's ``gather_and_maybe_dequant_cache``, which every MLA prefill with
    cached context calls (and DCP's, and sparse MLA's), take a cache of any size:
    its kernel for `latent.KERNEL_SIZES`, `latent.gather` for the others."""
    kernel = ops.gather_and_maybe_dequant_cache
    if kernel.__module__ == __name__:
        return

    def gather(
        src_cache: torch.Tensor, dst: torch.Tensor, *args: Any, **kw: Any
    ) -> None:
        if src_cache.shape[-1] in latent.KERNEL_SIZES:
            kernel(src_cache, dst, *args, **kw)
        else:
            latent.gather(src_cache, dst, *args, **kw)

    ops.gather_and_maybe_dequant_cache = gather


class LatentAttention(Layer):
    """bux's ``mla``: queries from the input or a normalized latent, keys and values
    from a normalized latent, by vLLM's MLA attention and its latent cache, then
    each head's output norm and gate, and the row-parallel output projection.

    The heads split over tensor-parallel ranks; the latents are every rank's. Each
    call keeps a cache of its own, over the same weights. A sparse layer (an
    indexer, or the ``index`` port) attends by the plugin's ops instead (`_sparse`).
    """

    bux_types = ("mla",)
    stateful = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        check(c, site)
        tp = get_tensor_model_parallel_world_size()
        h, n, r = c["num_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"]
        dv, rank, latent = c["v_head_dim"], c["kv_lora_rank"], c["q_lora_rank"]
        self.path, self.h, self.n, self.r, self.dv = site.path, h // tp, n, r, dv
        self.rank, self.latent = rank, latent
        path, kw = site.path, {"bias": False, "quant_config": site.quant}
        self._down(width, path, kw)
        self.q = ColumnParallelLinear(
            latent or width, h * (n + r), return_bias=False, prefix=f"{path}.q", **kw
        )
        # It returns its bias too, as vLLM's MLA calls it. The sparse ops read its
        # weight (`_sparse`), which stays in the model's dtype for them.
        sparse_ = is_sparse(c, site.ports)
        self.kv_b = ColumnParallelLinear(
            rank,
            h * (n + dv),
            bias=False,
            quant_config=None if sparse_ else site.quant,
            prefix=f"{path}.kv_b",
        )
        self.o = RowParallelLinear(
            h * dv, width, return_bias=False, prefix=f"{path}.o", **kw
        )
        self.q_a_norm = Norm(LATENT_NORM, latent) if latent else None
        self.kv_a_norm = Norm(LATENT_NORM, rank)
        self.rope = Rope(c["rope"], r) if c["rope"] else None
        self._gate(c, width, h * dv, path, kw)
        self.out_norm = Norm(c["output_norm"], dv) if c["output_norm"] else None
        self.scale = (c["scale"] or (n + r) ** -0.5) * logit_gain(c["rope"])
        self.sinks = sink_logits(self.h) if c["sinks"] else None
        self.ports, self.spec, self.indexer = site.ports, None, None
        if sparse_:
            self._sparse_caches(c, width, site)
        else:
            self.caches = nn.ModuleList(self._cache(name, site) for name in site.caches)

    def _down(self, width: int, path: str, kw: dict) -> None:
        """Make the down projections, every rank's, of the linear layers' arguments
        `kw`: the query latent's and the key and value latent's by one GEMM where
        there is a query latent."""
        self.qkv_a = self.kv_a = None
        out = self.rank + self.r
        if self.latent:
            self.qkv_a = MergedColumnParallelLinear(
                width,
                [self.latent, out],
                return_bias=False,
                disable_tp=True,
                prefix=f"{path}.qkv_a",
                **kw,
            )
        else:
            self.kv_a = ReplicatedLinear(
                width, out, return_bias=False, prefix=f"{path}.kv_a", **kw
            )

    def _gate(self, c: dict, width: int, heads_dv: int, path: str, kw: dict) -> None:
        """Make the gate on the heads' outputs, ``act(W_g x)``, a value a channel or
        a head (``gate_per_head``), column-parallel by head; None without one."""
        self.gate = None
        self.gate_activation = ACTIVATIONS[c["gate_activation"]]
        if c["gated"]:
            n = c["num_heads"] if c["gate_per_head"] else heads_dv
            self.gate = ColumnParallelLinear(
                width, n, return_bias=False, prefix=f"{path}.gate", **kw
            )

    def _cache(self, name: str, site: Site) -> MLAAttention:
        """vLLM's MLA attention of one call, its latent cache named `name`."""
        return MLAAttention(
            num_heads=self.h,
            scale=self.scale,
            qk_nope_head_dim=self.n,
            qk_rope_head_dim=self.r,
            v_head_dim=self.dv,
            q_lora_rank=self.latent,
            kv_lora_rank=self.rank,
            kv_b_proj=self.kv_b,
            cache_config=site.vllm_config.cache_config,
            prefix=name,
            attn_backend=decode_backend(
                self.rank + self.r, self.h, site.vllm_config.cache_config
            ),
            prefill_backend_cls=prefill_backend(site.vllm_config),
        )

    def _sparse_caches(self, c: dict, width: int, site: Site) -> None:
        """Make the indexer, and each call's cache of its latents (one head of
        ``kv_lora_rank + qk_rope_head_dim`` a token, as vLLM's MLA keeps them) with
        the indexer's keys beside them, which the sparse attention ops read: of the
        latents alone for a layer that reads its ``index`` port."""
        kinds = states(c, site.ports)  # a call's states, side by side in site.caches
        names = {kind: site.caches[i :: len(kinds)] for i, kind in enumerate(kinds)}
        ix = None if kv.reads(site.ports, "index") else c["indexer"]
        if ix is not None:
            unsupported(ix, "per_group")  # its groups are kv heads, which MLA's are not
            dims = rotated(self.r, c["rope"])
            self.indexer = Indexer(ix, width, c["rope"], dims, self.latent or width)
        settings = (c["sliding_window_size"], self.scale, c["softcap"], self.sinks)
        dims = self.rank + self.r
        self.spec = picks.layer_spec(
            ix, self.indexer, settings, dims, (0, 1), self.rank
        )
        self.caches = sparse.layer_caches(
            self.spec, 1, dims, ix["head_dim"] if ix else 0, names, site.vllm_config
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {}
        if self.qkv_a is not None:
            out["q_a.weight"] = (self.qkv_a.weight, (0,))
            out["kv_a.weight"] = (self.qkv_a.weight, (1,))
        if self.kv_a is not None:
            out["kv_a.weight"] = (self.kv_a.weight, ())
        for name in ("q", "kv_b", "o", "gate"):
            if (layer := getattr(self, name)) is not None:
                out[f"{name}.weight"] = (layer.weight, ())
        for name in ("q_a_norm", "kv_a_norm", "out_norm"):
            if (norm := getattr(self, name)) is not None:
                out |= {f"{name}.{k}": v for k, v in norm.targets().items()}
        if self.sinks is not None:
            out["sinks"] = (self.sinks, ())
        if self.indexer is not None:
            params = self.indexer.named_parameters()
            out |= {f"indexer.{name}": (p, ()) for name, p in params}
        return out

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        t = x.shape[0]
        query, kv = self._latents(x)
        kv_c, k_pe = kv.split([self.rank, self.r], -1)
        q = self.q(query).view(t, self.h, self.n + self.r)
        q, k_pe = self._rotate(q, k_pe[:, None], ctx)
        call, latent = ctx.call(self.path), self.kv_a_norm(kv_c)
        if self.spec is not None:
            keys = torch.cat([latent, k_pe[:, 0]], -1)[:, None]  # one head of both
            out = self._sparse(x, query, q, keys, call, ctx)
        else:
            shape = (t, self.h * self.dv)
            out = self.caches[call](q, latent, k_pe, output_shape=shape)
        out = self._gated(x, out.view(t, self.h, self.dv))
        return self.o(out.reshape(t, -1))

    def _sparse(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        q: torch.Tensor,
        keys: torch.Tensor,
        call: int,
        ctx: Ctx,
    ) -> torch.Tensor:
        """The heads' outputs, (T, h, dv), of attending in the latent space to the
        keys the indexer picks (or the ``index`` port's), `keys` the step's latents
        with their shared rotated dims, (T, 1, kv_lora_rank + r): each head's
        queries absorb its key projection (``kv_b``'s first ``qk_nope_head_dim``
        rows of it), and its value projection takes the latents they weigh, in
        float32, as DeepSeek's MLA decodes."""
        t, w = q.shape[0], self.kv_b.weight
        queries = picks.absorbed(q, w, self.n)
        index = None
        if self.indexer is not None:
            at, ix = ctx.positions, self.indexer
            index = (ix.queries(query, at), ix.head_weights(x), ix.keys(x, at))
        out = queries.new_empty(t, self.h, self.rank)
        values = keys.new_empty(t, 1, 0)  # the latents are the values too
        tensors = (queries, keys, values)
        sparse.attend(tensors, index, out, self.caches[call], self.ports, ctx)
        return picks.projected(out, w, self.n).to(x.dtype)

    def _latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What the queries come from (the normalized query latent, or x), and the
        key and value latent with the shared key dims, (T, kv_lora_rank + r)."""
        if self.kv_a is not None:
            return x, self.kv_a(x)
        assert self.qkv_a is not None and self.q_a_norm is not None
        q_c, kv = self.qkv_a(x).split([self.latent, self.rank + self.r], -1)
        return self.q_a_norm(q_c), kv

    def _rotate(
        self, q: torch.Tensor, k_pe: torch.Tensor, ctx: Ctx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the queries' last r dims, (T, h, n + r), and the shared key dims,
        (T, 1, r), by the positions."""
        if self.rope is None:
            return q, k_pe
        q_pe = self.rope(q[..., self.n :], ctx.positions)
        q = torch.cat([q[..., : self.n], q_pe], -1)
        return q, self.rope(k_pe, ctx.positions)

    def _gated(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The heads' outputs, (T, h, dv), normalized and gated."""
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is None:
            return out
        gate = self.gate_activation(self.gate(x))
        return out * gate.view(out.shape[0], self.h, -1)
