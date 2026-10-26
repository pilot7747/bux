"""Softmax attention in vLLM: bux's ``gqa``, its projections split by head under
tensor parallelism, attending by vLLM's attention layer and its paged KV cache."""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention as PagedAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from .common import (
    ACTIVATIONS,
    Ctx,
    Layer,
    Norm,
    Rope,
    Site,
    logit_gain,
    sharded_loader,
    unsupported,
)


class Attention(Layer):
    """bux's ``gqa``: q, k and v by one column-parallel projection, QK-norm then
    RoPE (in float32, as bux), causal softmax attention with its sliding window,
    soft-capping and sinks by vLLM's attention layer, each head's output norm and
    gate, and the row-parallel output projection.

    Each call of the layer keeps a KV cache of its own: a shared layer that runs
    three times makes three of vLLM's attention layers, over the same weights.
    """

    bux_types = ("gqa",)
    stateful = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        unsupported(c, "kv_conv_size", "indexer", "relative_bias", "log_scale")
        if c["qk_norm"] and c["qk_norm_all_heads"]:
            # Its statistics would sum over every rank's heads under tensor parallelism.
            raise NotImplementedError("gqa: qk_norm_all_heads not in vLLM yet")
        if site.ports:
            raise NotImplementedError("gqa: ports not in vLLM yet")
        tp = get_tensor_model_parallel_world_size()
        h, g = c["num_heads"], c["num_kv_heads"]
        dk = c["head_dim"] or width // h
        dv = c["v_head_dim"] or dk
        self.path, self.dk, self.dv, self.fused = site.path, dk, dv, c["fused_qkv"]
        self.h, self.g = h // tp, max(1, g // tp)  # this rank's
        self.qkv = QKVParallelLinear(
            width,
            dk,
            h,
            g,
            bias=c["bias"],
            v_head_size=dv,
            return_bias=False,
            prefix=f"{site.path}.qkv",
        )
        self._gate(c, width, h * dv, site.path)
        out_bias = c["bias"] if c["output_bias"] is None else c["output_bias"]
        self.o = RowParallelLinear(
            h * dv, width, bias=out_bias, return_bias=False, prefix=f"{site.path}.o"
        )
        qk, out = c["qk_norm"], c["output_norm"]
        self.q_norm = Norm(qk, dk) if qk else None
        self.k_norm = Norm(qk, dk) if qk else None
        self.out_norm = Norm(out, dv) if out else None
        self.rope = Rope(c["rope"], dk) if c["rope"] else None
        self.sinks = _sinks(self.h) if c["sinks"] else None
        scale = (c["scale"] or dk**-0.5) * logit_gain(c["rope"])
        self.caches = nn.ModuleList(
            PagedAttention(
                self.h,
                dk,
                scale,
                num_kv_heads=self.g,
                cache_config=site.vllm_config.cache_config,
                logits_soft_cap=c["softcap"],
                per_layer_sliding_window=c["sliding_window_size"],
                prefix=name,
                sinks=self.sinks,
                head_size_v=dv,
            )
            for name in site.caches
        )

    def _gate(self, c: dict, width: int, heads_dv: int, path: str) -> None:
        """Make the gate on the heads' outputs, ``act(W_g x)``, a value a channel or
        a head (``gate_per_head``), column-parallel by head; None without one."""
        self.gate = None
        self.gate_activation = ACTIVATIONS[c["gate_activation"]]
        if c["gated"]:
            n = c["num_heads"] if c["gate_per_head"] else heads_dv
            self.gate = ColumnParallelLinear(
                width, n, bias=c["bias"], return_bias=False, prefix=f"{path}.gate"
            )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {}
        for kind in ("weight", "bias"):
            if (fused := getattr(self.qkv, kind)) is not None:
                shards = {"qkv": None} if self.fused else {s: s for s in "qkv"}
                out |= {f"{n}.{kind}": (fused, (s,)) for n, s in shards.items()}
            for name in ("gate", "o"):
                layer = getattr(self, name)
                if layer is not None and getattr(layer, kind) is not None:
                    out[f"{name}.{kind}"] = (getattr(layer, kind), ())
        for name in ("q_norm", "k_norm", "out_norm"):
            if (norm := getattr(self, name)) is not None:
                out |= {f"{name}.{k}": v for k, v in norm.targets().items()}
        if self.sinks is not None:
            out["sinks"] = (self.sinks, ())
        return out

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        t = x.shape[0]
        q, k, v = self.qkv(x).split(
            [self.h * self.dk, self.g * self.dk, self.g * self.dv], -1
        )
        q, k = q.view(t, self.h, self.dk), k.view(t, self.g, self.dk)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, ctx.positions), self.rope(k, ctx.positions)
        attend = self.caches[ctx.call(self.path)]
        q, k = q.reshape(t, -1).contiguous(), k.reshape(t, -1).contiguous()
        out = attend(q, k, v.contiguous()).view(t, self.h, self.dv)
        return self.o(self._gated(x, out).reshape(t, -1))

    def _gated(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The heads' outputs, (T, h, dv), normalized and gated."""
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is None:
            return out
        gate = self.gate_activation(self.gate(x))
        return out * gate.view(out.shape[0], self.h, -1)


def _sinks(heads: int) -> nn.Parameter:
    """This rank's heads' sink logits, in float32 as bux's softmax takes them."""
    sinks = nn.Parameter(torch.empty(heads, dtype=torch.float32), requires_grad=False)
    sinks.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]
    return sinks
