"""Softmax attention in vLLM: bux's ``gqa``, its projections split by head under
tensor parallelism, attending by vLLM's attention layer and its paged KV cache, with
its ``v`` port (the value residual, `bux_vllm.kv`), its conv on keys and values
(`bux_vllm.conv`) and its indexer, which picks the keys each query attends to
(`bux_vllm.sparse`). Its projections take the engine's quantization config
(`bux_vllm.common.quantization`); its norms, conv and indexer stay in the model's
dtype."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention import Attention as PagedAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from . import kv, picks, sparse
from .common import (
    ACTIVATIONS,
    Ctx,
    Layer,
    Norm,
    Rope,
    Site,
    load_into,
    logit_gain,
    sharded_loader,
)
from .config import states
from .conv import CausalConv
from .indexer import Indexer, rotated


class Attention(Layer):
    """bux's ``gqa``: q, k and v by one column-parallel projection, the causal conv
    on k and v added to them (``kv_conv_size``), the values mixed with those of its
    ``v`` port (the value residual) and pushed to it, QK-norm then RoPE (in float32,
    as bux), the queries scaled by their position (``power_scale``, ``log_scale``),
    causal softmax attention with its sliding window, soft-capping and sinks by
    vLLM's attention layer, each head's output norm and gate (with a bias per head),
    and the row-parallel output projection.

    With an indexer, each query attends to the keys the indexer picks alone, as
    bux's do (DSA, and its blocks: QSA's, MSA's), by the plugin's sparse attention
    op over vLLM's paged caches (`bux_vllm.sparse`). The ``index`` port passes the
    picks on: a layer that reads it attends to the keys another's indexer picked,
    of those its own window sees, and runs no indexer. With a relative bias, each
    query attends to every key it sees, the bias and the length scale on its
    logits, by the plugin's op over a paged cache of its own.

    With ``k_eq_v`` (Gemma 4's global layers) its values are its keys as
    projected, after the conv, RMS-normalized without a gain: as vLLM's Gemma 4
    does, bux's key projection loads into both the k and the v shard of the
    projection, whose v output goes unused, and the KV cache holds both.

    Each call of the layer keeps a KV cache of its own, a conv state beside it with
    ``kv_conv_size``, and its indexer's keys with an indexer: a shared layer that
    runs three times makes three of vLLM's attention layers (and convs), over the
    same weights. A layer that reads its ``kv`` port projects its queries alone and
    attends to its writer's KV cache, which vLLM shares with it
    (``kv_sharing_target_layer_name``).
    """

    bux_types = ("gqa",)
    stateful = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        kv.check_ports(c, site.ports)
        self.rel, self.rel_proj, self.rel_table = c["relative_bias"], None, None
        if self.rel and (c["indexer"] or site.ports):
            raise NotImplementedError("gqa: a relative bias beside an indexer or ports")
        self.reads_kv = kv.reads(site.ports, "kv") is not None
        self.reads_index = kv.reads(site.ports, "index") is not None
        tp = get_tensor_model_parallel_world_size()
        h, g = c["num_heads"], c["num_kv_heads"]
        dk, dv = kv.head_dims(c, width)
        self.path, self.dk, self.dv, self.fused = site.path, dk, dv, c["fused_qkv"]
        self.h, self.g = h // tp, max(1, g // tp)  # this rank's
        self.kv_first = kv.kv_heads(g, tp, get_tensor_model_parallel_rank())[0]
        self.c, self.scale = c, (c["scale"] or dk**-0.5) * logit_gain(c["rope"])
        self.k_eq_v = c["k_eq_v"] and not self.reads_kv  # values from the keys
        self._projections(c, width, site)
        self._norms(c, tp)
        self.rope = Rope(c["rope"], dk) if c["rope"] else None
        self._value_residual(c, site.ports)
        self.sinks = sink_logits(self.h) if c["sinks"] else None
        kinds = states(c, site.ports)  # a call's states, side by side in site.caches
        names = {kind: site.caches[i :: len(kinds)] for i, kind in enumerate(kinds)}
        self.indexer = None
        if c["indexer"] or self.reads_index:
            self._sparse_caches(c, width, site, names)
        elif self.rel:
            self._biased_caches(c, width, site, names)
        else:
            shares = site.shares or [None] * len(names["full_attention"])
            self._caches(c, site, names["full_attention"], shares)
        self._convs(c, site, names.get("conv", []))

    def _projections(self, c: dict, width: int, site: Site) -> None:
        """Make q, k and v's projection, column-parallel by head (q's alone for a
        layer that reads its ``kv`` port), the gate's, and the row-parallel output
        projection, each with the engine's quantization config."""
        h, g, dk, dv = c["num_heads"], c["num_kv_heads"], self.dk, self.dv
        path, kw = site.path, {"return_bias": False, "quant_config": site.quant}
        self.qkv, self.q = None, None
        if self.reads_kv:
            self.q = ColumnParallelLinear(
                width, h * dk, bias=c["bias"], prefix=f"{path}.q", **kw
            )
        else:
            self.qkv = QKVParallelLinear(
                width,
                dk,
                h,
                g,
                bias=c["bias"],
                v_head_size=dv,
                prefix=f"{path}.qkv",
                **kw,
            )
        self._gate(c, width, h * dv, site)
        out_bias = c["bias"] if c["output_bias"] is None else c["output_bias"]
        self.o = RowParallelLinear(
            h * dv, width, bias=out_bias, prefix=f"{path}.o", **kw
        )

    def _norms(self, c: dict, tp: int) -> None:
        """Make the QK-norm, of each head or of all heads together, and the norm of
        each head's output.

        Raises:
            NotImplementedError: QK-norm of all heads where ranks copy kv heads.
        """
        qk, out, h, g = (
            c["qk_norm"],
            c["output_norm"],
            c["num_heads"],
            c["num_kv_heads"],
        )
        self.all_heads = bool(qk and c["qk_norm_all_heads"])
        if self.all_heads and g % tp:
            raise NotImplementedError("gqa: qk_norm_all_heads with kv heads copied")
        if self.all_heads:
            self.q_norm = AllHeads(qk, h, self.dk, tp)
            self.k_norm = AllHeads(qk, g, self.dk, tp)
        else:
            self.q_norm = Norm(qk, self.dk) if qk else None
            self.k_norm = Norm(qk, self.dk) if qk else None
        if self.reads_kv:  # its writer's keys are normalized
            self.k_norm = None
        self.out_norm = Norm(out, self.dv) if out else None
        self.v_norm = Norm(kv.value_norm(qk), self.dk) if self.k_eq_v else None

    def _caches(
        self, c: dict, site: Site, names: list[str], shares: list[str | None]
    ) -> None:
        """Make each call's vLLM attention layer, with its KV cache, or attending
        to the KV cache it `shares`, a writer's through the ``kv`` port."""
        self.caches = nn.ModuleList(
            PagedAttention(
                self.h,
                self.dk,
                self.scale,
                num_kv_heads=self.g,
                cache_config=site.vllm_config.cache_config,
                logits_soft_cap=c["softcap"],
                per_layer_sliding_window=c["sliding_window_size"],
                prefix=name,
                sinks=self.sinks,
                head_size_v=self.dv,
                kv_sharing_target_layer_name=share,
            )
            for name, share in zip(names, shares, strict=True)
        )

    def _sparse_caches(
        self, c: dict, width: int, site: Site, names: dict[str, list[str]]
    ) -> None:
        """Make the indexer, and each call's cache of keys, values and indexer keys,
        which the sparse attention ops read (`bux_vllm.sparse`): of keys and values
        alone for a layer that reads its ``index`` port."""
        ix = None if self.reads_index else c["indexer"]
        if ix is not None:
            rotated_dims = rotated(self.dk, c["rope"])
            self.indexer = Indexer(ix, width, c["rope"], rotated_dims)
        settings = (c["sliding_window_size"], self.scale, c["softcap"], self.sinks)
        groups = (self.kv_first, self.g)
        spec = picks.layer_spec(ix, self.indexer, settings, self.dk, groups)
        what = "gqa with an indexer" if ix else "gqa reading its index port"
        dtype = site.vllm_config.model_config.dtype
        dims, heads = (self.dk, self.dv), c["num_kv_heads"]
        native, blocked, fp8 = sparse.paths(spec, ix, dims, dtype, heads)
        self.caches = sparse.layer_caches(
            spec,
            self.g,
            self.dk + self.dv,
            ix["head_dim"] if ix else 0,
            names,
            site.vllm_config,
            what,
            native,
            fp8,
            blocked,
        )

    def _biased_caches(
        self, c: dict, width: int, site: Site, names: dict[str, list[str]]
    ) -> None:
        """Make the relative bias, each head's projection (column-parallel by head)
        and the table every rank keeps, and each call's paged cache of keys and
        values, which the biased attention op reads (`bux_vllm.sparse`)."""
        rank, extent = self.rel["rank"], self.rel["extent"]
        self.rel_proj = ColumnParallelLinear(
            width,
            c["num_heads"] * rank,
            bias=c["bias"],
            return_bias=False,
            quant_config=site.quant,
            prefix=f"{site.path}.rel_proj",
        )
        table = torch.zeros(rank, extent)
        self.rel_table = nn.Parameter(table, requires_grad=False)
        settings = (c["sliding_window_size"], self.scale, c["softcap"], self.sinks)
        spec = picks.layer_spec(None, None, settings, self.dk, (self.kv_first, self.g))
        self.caches = sparse.layer_caches(
            spec,
            self.g,
            self.dk + self.dv,
            0,
            names,
            site.vllm_config,
            "gqa with a relative bias",
        )

    def _convs(self, c: dict, site: Site, names: list[str]) -> None:
        """Make each call's conv on keys and values, with its state, under
        ``kv_conv_size``: all calls' over the same weights."""
        self.convs, self.conv_weight = None, None
        if (size := c["kv_conv_size"]) and names:  # none for a kv reader
            channels = self.g * (self.dk + (0 if self.k_eq_v else self.dv))
            convs = [
                CausalConv(channels, size, site.vllm_config, name) for name in names
            ]
            self.conv_weight = convs[0].weight
            for other in convs[1:]:
                other.weight = self.conv_weight
            self.convs = nn.ModuleList(convs)

    def _gate(self, c: dict, width: int, heads_dv: int, site: Site) -> None:
        """Make the gate on the heads' outputs, ``act(W_g x)``, a value a channel or
        a head (``gate_per_head``), column-parallel by head; None without one."""
        self.gate, self.gate_bias = None, None
        self.gate_activation = ACTIVATIONS[c["gate_activation"]]
        if c["gated"]:
            n = c["num_heads"] if c["gate_per_head"] else heads_dv
            self.gate = ColumnParallelLinear(
                width,
                n,
                bias=c["bias"],
                return_bias=False,
                quant_config=site.quant,
                prefix=f"{site.path}.gate",
            )
        if c["gated"] and c["gate_head_bias"]:
            self.gate_bias = nn.Parameter(torch.zeros(self.h), requires_grad=False)
            self.gate_bias.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]

    def _value_residual(self, c: dict, layer_ports: list) -> None:
        """Keep the layer's ports, and the learned weights of its value residual
        where it reads the ``v`` port and ``value_residual`` learns them."""
        self.ports, self.mix = layer_ports, c["value_residual"]
        self.v_mix = None
        if kv.reads(layer_ports, "v") and self.mix != "fixed":
            shape = (2,) if self.mix == "learned" else ()
            w = torch.full(shape, 0.5, dtype=torch.float32)
            self.v_mix = nn.Parameter(w, requires_grad=False)

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {}
        for kind in ("weight", "bias"):
            if (fused := getattr(self.qkv, kind, None)) is not None:
                shards = {"qkv": None} if self.fused else {s: s for s in "qkv"}
                out |= {f"{n}.{kind}": (fused, (s,)) for n, s in shards.items()}
            for name in ("q", "gate", "o", "rel_proj"):
                layer = getattr(self, name)
                if layer is not None and getattr(layer, kind) is not None:
                    out[f"{name}.{kind}"] = (getattr(layer, kind), ())
        if self.k_eq_v:  # bux's keys load into the v shard too, as Gemma 4's
            out.pop("v.weight", None), out.pop("v.bias", None)
        for name in ("q_norm", "k_norm", "out_norm"):
            if (norm := getattr(self, name)) is not None:
                out |= {f"{name}.{k}": v for k, v in norm.targets().items()}
        if self.sinks is not None:
            out["sinks"] = (self.sinks, ())
        if self.v_mix is not None:
            out["v_mix"] = (self.v_mix, ())
        if self.gate_bias is not None:
            out["gate_bias"] = (self.gate_bias, ())
        if self.indexer is not None:
            params = self.indexer.named_parameters()
            out |= {f"indexer.{name}": (p, ()) for name, p in params}
        if self.rel_table is not None:
            out["rel_table"] = (self.rel_table, ())
        return out

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name in ("k_conv.weight", "v_conv.weight") and self.conv_weight is not None:
            return self._load_conv(self.conv_weight, name[0], tensor)
        if self.k_eq_v and self.qkv is not None and name.split(".")[0] in ("k", "qkv"):
            return self._load_keys_as_values(name, tensor)
        return super().load(name, tensor)

    def _load_keys_as_values(self, name: str, tensor: torch.Tensor) -> list:
        """Load bux's key projection (or the q and k of its fused one) into the k
        shard of ``qkv`` and into its v shard too, as vLLM loads Gemma 4's
        ``attention_k_eq_v`` layers: the v shard's output goes unused, but it keeps
        an FP8 scale of the whole projection the keys'."""
        assert self.qkv is not None
        param = getattr(self.qkv, name.split(".")[1])
        if name.startswith("qkv."):  # bux's fused q and k, as rows
            queries = self.c["num_heads"] * self.dk
            load_into(param, tensor[:queries], "q")
            tensor = tensor[queries:]
        for shard in ("k", "v"):
            load_into(param, tensor, shard)
        return [param]

    def _load_conv(
        self, weight: nn.Parameter, which: str, taps: torch.Tensor
    ) -> list[nn.Parameter]:
        """Load bux's taps of the conv on the keys (`which` ``k``) or the values
        (``v``) into this rank's rows of the conv's `weight`: its kv heads' keys,
        then their values."""
        dims, start = (self.dk, 0) if which == "k" else (self.dv, self.g * self.dk)
        rows = kv.conv_rows(taps, dims, self.kv_first, self.g)
        weight.data[start : start + rows.shape[0]].copy_(rows)
        return [weight]

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        t, call = x.shape[0], ctx.call(self.path)
        if self.reads_kv:  # its writer's keys and values, as vLLM shares them
            assert self.q is not None
            q = self._queries(self.q(x), ctx).reshape(t, -1).contiguous()
            out = self.caches[call](q, None, None).view(t, self.h, self.dv)
            return self.o(self._gated(x, out).reshape(t, -1))
        q, k, v = self._project(x, call)
        v = self._v_port(v, ctx)
        q, k = self._queries(q, ctx), self._normed(k, self.k_norm, self.g)
        if self.rope is not None:
            k = self.rope(k, ctx.positions)
        if self.indexer is not None or self.reads_index:
            out = self._sparse(x, q, k, v, call, ctx)
        elif self.rel:
            out = self._biased(x, q, k, v, call, ctx)
        else:
            q, k = q.reshape(t, -1).contiguous(), k.reshape(t, -1).contiguous()
            out = self.caches[call](q, k, v.contiguous()).view(t, self.h, self.dv)
        return self.o(self._gated(x, out).reshape(t, -1))

    def _queries(self, q: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """The projected queries, (T, h dk), as heads, (T, h, dk): QK-norm then RoPE
        (in float32, as bux), and their scale by position (``power_scale``,
        ``log_scale``)."""
        q = self._normed(q, self.q_norm, self.h)
        if self.rope is not None:
            q = self.rope(q, ctx.positions)
        scales = kv.power_scale if self.rel else kv.query_scale  # biased: its op's
        if (scale := scales(self.c, ctx.positions)) is not None:
            q = (q * scale[:, None, None]).to(q.dtype)
        return q

    def _normed(self, x: torch.Tensor, norm: Norm | None, heads: int) -> torch.Tensor:
        """Projected queries or keys, (T, heads dk), as heads, (T, heads, dk), by
        their QK-norm: each head's, or all heads' together before they split
        (``qk_norm_all_heads``)."""
        t = x.shape[0]
        if norm is not None and self.all_heads:
            return norm(x).view(t, heads, self.dk)
        x = x.view(t, heads, self.dk)
        return x if norm is None else norm(x)

    def _sparse(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        call: int,
        ctx: Ctx,
    ) -> torch.Tensor:
        """The heads' outputs, (T, h, dv), of attending to the keys the indexer
        picks, or the ``index`` port's picks, by the sparse attention ops over the
        `call`-th call's caches (`bux_vllm.sparse.attend`)."""
        index = None
        if self.indexer is not None:
            at = ctx.positions
            ix = self.indexer
            index = (ix.queries(x, at), ix.head_weights(x), ix.keys(x, at))
        out = q.new_empty(q.shape[0], self.h, self.dv)
        values = v.view(-1, self.g, self.dv)
        cache = self.caches[call]
        return sparse.attend((q, k, values), index, out, cache, self.ports, ctx)

    def _biased(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        call: int,
        ctx: Ctx,
    ) -> torch.Tensor:
        """The heads' outputs, (T, h, dv), of attending to every key with the
        relative bias, then the length scale, on the logits, by the biased op over
        the `call`-th call's cache."""
        assert self.rel_proj is not None and self.rel_table is not None
        t, at = q.shape[0], ctx.positions
        r = self.rel_proj(x).view(t, self.h, -1)
        gain = kv.length_scale(self.c, at)
        gain = q.new_empty(0) if gain is None else gain
        out = q.new_empty(t, self.h, self.dv)
        values = v.view(-1, self.g, self.dv)
        tensors = (q, k, values, r, self.rel_table, gain)
        torch.ops.vllm.bux_biased_attention(
            *(t.contiguous() for t in tensors), at, out, self.caches[call].prefix
        )
        return out

    def _project(self, x: torch.Tensor, call: int) -> list[torch.Tensor]:
        """q, k and v of x, (T, _) each, with the conv added to k and v (by the
        `call`-th call's state) under ``kv_conv_size``."""
        assert self.qkv is not None, "a layer that reads kv projects q alone"
        if self.k_eq_v:
            return self._project_keys_as_values(x, call)
        q, keys_values = self.qkv(x).split(
            [self.h * self.dk, self.g * (self.dk + self.dv)], -1
        )
        if self.convs is not None:
            keys_values = keys_values + self.convs[call](keys_values)
        return [q, *keys_values.split([self.g * self.dk, self.g * self.dv], -1)]

    def _project_keys_as_values(self, x: torch.Tensor, call: int) -> list:
        """q, k and v of x, (T, _) each, v made from k (``k_eq_v``): each head's
        key, after the conv, RMS-normalized without a gain."""
        assert self.qkv is not None and self.v_norm is not None
        q, k, _ = self.qkv(x).split(
            [self.h * self.dk, self.g * self.dk, self.g * self.dv], -1
        )
        if self.convs is not None:
            k = k + self.convs[call](k)
        v = self.v_norm(k.view(-1, self.g, self.dk)).view(k.shape)
        return [q, k, v]

    def _v_port(self, v: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """The values, (T, g dv), with those the ``v`` port reads mixed in, pushed
        to the taps it writes, as bux mixes and pushes them."""
        read = kv.reads(self.ports, "v")
        if read is not None:
            v = kv.mix_values(self.mix, self.v_mix, v, ctx.read(*read))
        for tap in kv.writes(self.ports, "v"):
            ctx.push(tap, v)
        return v

    def _gated(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The heads' outputs, (T, h, dv), normalized and gated."""
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is None:
            return out
        gate = self.gate(x).view(out.shape[0], self.h, -1)
        if self.gate_bias is not None:
            gate = gate + self.gate_bias.to(gate.dtype)[:, None]
        return out * self.gate_activation(gate)


class AllHeads(Norm):
    """bux's QK-norm of all heads together (``qk_norm_all_heads``): statistics over
    every rank's heads, summed across ranks under tensor parallelism, and this
    rank's share of the gain and bias, a channel each.

    Args:
        c: The norm's config, as JSON.
        heads: The heads it normalizes together, on all ranks.
        dims: Each head's dims.
        tp: The ranks they split over.
    """

    def __init__(self, c: dict, heads: int, dims: int, tp: int):
        super().__init__(c, heads // tp * dims)
        self.total = heads * dims
        for p in (self.weight, self.bias):
            if p is not None:
                p.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x, (T, this rank's channels), over the RMS of all ranks' channels, centered
        first for a layernorm, in float32."""
        y = x.float()
        if self.kind == "layernorm":
            y = y - _ranks_sum(y) / self.total
        y = y * torch.rsqrt(_ranks_sum(y.square()) / self.total + self.eps)
        return y.to(x.dtype)


def _ranks_sum(x: torch.Tensor) -> torch.Tensor:
    """The sum of x's last axis, (T, 1), over every rank's share of it: reduced as
    4 columns, the sum and zeros, as vLLM's custom all-reduce takes rows of 16
    bytes at least."""
    sums = F.pad(x.sum(-1, keepdim=True), (0, 3))
    return tensor_model_parallel_all_reduce(sums)[:, :1]


def sink_logits(heads: int) -> nn.Parameter:
    """This rank's heads' sink logits, in float32 as bux's softmax takes them."""
    sinks = nn.Parameter(torch.empty(heads, dtype=torch.float32), requires_grad=False)
    sinks.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]
    return sinks
