"""Lightning attention in vLLM: bux's ``lightning``, MiniMax-01's linear attention, by
vLLM's own lightning kernels, as vLLM runs MiniMax-Text-01.

q, k and v are projected by heads, SiLU'd in float32 and unscaled; each head decays
its state by a fixed ``exp(-s)``, whose rate s weakens with the layer's depth among
the model's token mixers (`bux_vllm.linear.slopes`, the depth from the export's sites); one RMSNorm
takes all heads together, its statistics summed over the ranks, then the gate and
the output projection. Each head's state, dk by dk, stays in vLLM's Mamba cache (its
``LINEAR`` kind), in float32 as bux keeps it, and the recurrence runs out of the
compiled graph, in the op the plugin's recurrent layers share (`bux_vllm.recurrent`).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
    MiniMaxText01LinearKernel,
    clear_linear_attention_cache_for_new_sequences,
    linear_attention_decode,
    linear_attention_prefill_and_mix,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .attention import AllHeads
from .common import ACTIVATIONS, Ctx, Layer, Site
from .gdn2 import rms
from .linear import slopes
from .recurrent import MIX, split_op

BLOCK = 256  # the positions of a block of vLLM's lightning prefill kernel


def state_shape(c: dict, tp: int) -> tuple:
    """The shape of a ``lightning`` layer's state on one of `tp` ranks: each of its
    heads' (dk, dk)."""
    return MambaStateShapeCalculator.linear_attention_state_shape(
        c["num_heads"], tp, c["head_dim"]
    )


def check(c: dict) -> None:
    """Raise for what vLLM's lightning kernels do not take.

    Raises:
        NotImplementedError: A conv, value heads of other numbers or dims than the
            q/k heads', or heads of a width that is no multiple of 64.
    """
    if c["conv_size"] is not None:
        raise NotImplementedError("lightning: a conv not in vLLM yet")
    if (c["num_v_heads"] or c["num_heads"], c["v_head_dim"] or c["head_dim"]) != (
        c["num_heads"],
        c["head_dim"],
    ):
        raise NotImplementedError("lightning: value heads other than q's and k's")
    if c["head_dim"] % 64:
        raise NotImplementedError("lightning: a head_dim vLLM's kernels do not take")


class State(nn.Module, MambaBase):
    """vLLM's layer of one ``lightning`` call: its heads' states in the Mamba cache,
    and the recurrence of `mix` by vLLM's lightning kernels, out of the graph.

    Args:
        rates: This rank's heads' decay rates, (h,).
        c: The layer's config, as JSON.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of the state, unique in the model.
    """

    def __init__(self, rates: torch.Tensor, c: dict, vllm_config: Any, prefix: str):
        super().__init__()
        self.c, self.prefix = c, prefix
        self.register_buffer("rates", rates, persistent=False)
        self.kv_cache = (torch.tensor([]),)
        compilation = vllm_config.compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self
        split_op(compilation, MIX)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, get_tensor_model_parallel_world_size())

    def get_state_dtype(self) -> tuple:
        return (torch.float32,)

    def mix(self, xs: list[torch.Tensor], out: torch.Tensor) -> None:
        """A step: the recurrence of q, k and v, (T, h, d) in float32, from each
        sequence's state (zero for a new one), keeping the last; the heads' outputs,
        (T, h d), into `out`; nothing in vLLM's profiling run, which has no
        metadata."""
        meta = get_forward_context().attn_metadata
        meta = meta.get(self.prefix) if isinstance(meta, dict) else None
        if meta is None:
            return
        n = meta.num_prefill_tokens + meta.num_decode_tokens
        q, k, v = (x[:n] for x in xs)
        state, idx = self.kv_cache[0], meta.state_indices_tensor
        clear_linear_attention_cache_for_new_sequences(state, idx, meta)
        if meta.num_prefills:
            out[:n] = linear_attention_prefill_and_mix(
                q,
                k,
                v,
                state,
                idx,
                meta,
                self.rates,
                BLOCK,
                decode_fn=self._decode,
                prefix_fn=MiniMaxText01LinearKernel.jit_linear_forward_prefix,
            )
        else:
            out[:n] = self._decode(q, k, v, state, idx, meta)

    def _decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        idx: torch.Tensor,
        meta: Any,
    ) -> torch.Tensor:
        """A step of each decoding sequence, which vLLM puts first."""
        return linear_attention_decode(
            q,
            k,
            v,
            state,
            self.rates,
            idx,
            q_end=meta.num_decode_tokens,
            slot_end=meta.num_decodes,
        )


class Lightning(Layer):
    """bux's ``lightning``: q, k and v by one column-parallel projection, SiLU'd in
    float32, the recurrence (`State`), the RMSNorm of all heads (`AllHeads`), the
    gate's activation, and the row-parallel output projection."""

    bux_types = ("lightning",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        check(c)
        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        h, d = c["num_heads"], c["head_dim"]
        self.h, self.d = h // tp, d
        self.qkv = QKVParallelLinear(
            width, d, h, h, bias=False, return_bias=False, prefix=f"{site.path}.qkv"
        )
        self.gate, self.gate_activation = None, None
        if c["output_gate"] is not None:
            self.gate_activation = ACTIVATIONS[c["output_gate"]]
            self.gate = ColumnParallelLinear(
                width, h * d, bias=False, return_bias=False, prefix=f"{site.path}.gate"
            )
        self.norm = AllHeads(rms(c["norm_eps"]), h, d, tp)
        self.o = RowParallelLinear(
            h * d, width, bias=False, return_bias=False, prefix=f"{site.path}.o"
        )
        depth = site.config.bux_sites[site.path]["extra"]
        rates = slopes(h, depth["layer"], depth["layers"])
        mine = rates[rank * self.h : (rank + 1) * self.h].contiguous()
        self.state = State(mine, c, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            f"{n}.weight": (self.qkv.weight, (n,)) for n in "qkv"
        }
        out["o.weight"] = (self.o.weight, ())
        if self.gate is not None:
            out["gate.weight"] = (self.gate.weight, ())
        return out | {f"norm.{k}": v for k, v in self.norm.targets().items()}

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        t = x.shape[0]
        qkv = F.silu(self.qkv(x).float()).view(t, 3, self.h, self.d)
        xs = [qkv[:, i].contiguous() for i in range(3)]
        out = x.new_zeros(t, self.h * self.d, dtype=torch.float32)
        torch.ops.vllm.bux_mix(xs, out, self.state.prefix, "mix")
        o = self.norm(out.to(x.dtype))
        if self.gate is not None and self.gate_activation is not None:
            o = o * self.gate_activation(self.gate(x))
        return self.o(o)
