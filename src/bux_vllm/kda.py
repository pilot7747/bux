"""Kimi Delta Attention in vLLM: bux's ``kda`` by the kernels vLLM serves Kimi
Linear with (KDA by chunks for a prefill, its recurrence for decoding, the causal
conv's), its state in vLLM's Mamba cache.

vLLM's own KDA layer gives every value head a q and a k head of its own, where bux's
share one among a group of value heads (``num_v_heads``). This layer keeps bux's: it
projects and convolves q and k at their own heads, so that its weights and its conv
state are bux's, and repeats them out to the value heads for the kernels alone. Its
conv and recurrence run outside the compiled graph, as vLLM's attention does: in an
op of their own (`OP`), which splits the graph.
"""

from functools import cache
from typing import Any

import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KimiGDNMergedColumnParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.utils.torch_utils import direct_register_custom_op

from . import deltanet
from .common import Ctx, Layer, Site
from .linear import kda_projection, linear_dims, to_value_heads, zero_padding
from .recurrent import Recurrent, float32, low_rank_tensor, split_op

#: The op the conv and recurrence run in, outside the compiled graph.
OP = "bux_kda"

#: The lowest gate_lower_bound vLLM's KDA kernels hold to (Kimi's -5).
LOWEST_BOUND = -5.0


def takes(c: dict) -> bool:
    """Whether vLLM's kernels run the ``kda`` config `c`: with a conv and an output
    gate, value heads as wide as q and k's, and a lower bound of `LOWEST_BOUND` or
    none; the delta rule of bux's own runs the others (`bux_vllm.deltanet`)."""
    bound = c["gate_lower_bound"]
    return (
        c["conv_size"] is not None
        and c["output_gate"] is not None
        and (c["v_head_dim"] or c["head_dim"]) == c["head_dim"]
        and (bound is None or bound >= LOWEST_BOUND)
    )


def state_shape(c: dict, width: int, tp: int) -> tuple:
    """The shapes of a ``kda`` layer's state on one of `tp` ranks: its conv's last
    inputs of q, k and v, and each value head's (dv, dk) state."""
    if not takes(c):
        return deltanet.state_shape(c, width, tp)
    h, hv, dk, dv = linear_dims(c)
    return MambaStateShapeCalculator.kda_state_shape(
        tp, hv, dv, h, dk, conv_kernel_size=c["conv_size"]
    )


@cache
def kernels() -> tuple[Any, Any]:
    """vLLM's KDA kernels, by chunks and recurrent, imported once a model has KDA."""
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
    )

    return chunk_kda_with_fused_gate, fused_recurrent_kda


def kda_op(
    qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    """`OP`: the conv and recurrence of the layer `layer_name` (`Mixer.mix`)."""
    get_forward_context().no_compile_layers[layer_name].mix(qkv, g, beta, out)


direct_register_custom_op(op_name=OP, op_func=kda_op, mutates_args=["out"])


class Mixer(Recurrent):
    """vLLM's layer of one ``kda`` call (`Recurrent`): its weights by rank, the
    projections in the compiled graph and `mix` out of it.

    Raises:
        NotImplementedError: Speculative decoding.
    """

    what = "kda"

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str):
        super().__init__(vllm_config, prefix)
        split_op(vllm_config.compilation_config, OP)
        self.c = c
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        self.group = hv // h
        self._projections(c, width)
        self.conv_taps(c)
        self.A_log, self.dt_bias = float32(self.hv), float32(self.hv * dk)
        self.norm = FusedRMSNormGated(
            dv, eps=c["norm_eps"], activation=c["output_gate"]
        )
        self.o = RowParallelLinear(
            hv * dv, width, bias=False, return_bias=False, prefix=f"{prefix}.o"
        )

    def _projections(self, c: dict, width: int) -> None:
        """q, k, v, a dense output gate, the decay's down projection (on every rank)
        and beta, by one merged projection; the decay's up projection, and a
        low-rank output gate's two."""
        _, hv, dk, dv = linear_dims(c)
        rank = c["gate_rank"] or dk
        self.dense_gate = c["output_gate_rank"] is None
        projection = kda_projection(c, self.tp)
        self.split, self.parts = projection.split, projection.parts
        self.in_proj = _KimiGDNMergedColumnParallelLinear(
            width,
            projection.sizes,
            replicated_shard_id=projection.whole,  # the decay's down projection
            tp_size=self.tp,
            bias=False,
            return_bias=False,
            prefix=f"{self.prefix}.in_proj",
        )
        zero_padding(self.in_proj, sum(self.split[self.parts :]))
        self.f_up = ColumnParallelLinear(
            rank, hv * dk, bias=False, return_bias=False, prefix=f"{self.prefix}.f_up"
        )
        if not self.dense_gate:
            r = c["output_gate_rank"]
            self.gate_down = ReplicatedLinear(
                width, r, bias=False, return_bias=False, prefix=f"{self.prefix}.g_a"
            )
            self.gate_up = ColumnParallelLinear(
                r, hv * dv, bias=False, return_bias=False, prefix=f"{self.prefix}.g_b"
            )

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, 0, self.tp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        n = x.shape[0]
        qkv, *rest = self.in_proj(x).split(self.split, -1)[: self.parts]
        gate, f, beta = rest if self.dense_gate else (None, *rest)
        out = x.new_zeros(n, self.hv, self.dv)
        torch.ops.vllm.bux_kda(qkv, self.f_up(f), beta, out, self.prefix)
        if gate is None:  # made after the op, so that no matmul's output crosses it
            gate = self.gate_up(self.gate_down(x))
        out = self.norm(out, gate.unflatten(-1, (self.hv, self.dv)))
        return self.o(out.flatten(1))

    def mix(
        self, qkv: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Convolve q, k and v and run the delta rule from each sequence's state, its
        decay from `g` and step from `beta`: write the heads' outputs, (T, hv, dv),
        to `out`. In a run without metadata (vLLM's profiling), nothing."""
        m = self.batch()
        if m is None:
            return
        n = self.tokens(m)
        q, k, v = self._heads(self.conv_tokens(m, qkv[:n]))
        gates = {
            "raw_g": g[:n].unflatten(-1, (self.hv, self.dk))[None],
            "raw_beta": beta[:n][None],
            "A_log": self.A_log,
            "lower_bound": self.c["gate_lower_bound"],
        }
        run = self._chunked if m.num_prefills else self._recurrent
        out[:n] = run(m, q, k, v, gates)[0]

    def _heads(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """q and k at the value heads, and v, each (1, T, hv, d)."""
        h, dk = self.h, self.dk
        q, k, v = x.split([h * dk, h * dk, self.hv * self.dv], -1)
        q, k = (to_value_heads(y, h, self.group) for y in (q, k))
        return q, k, v.reshape(1, -1, self.hv, self.dv).contiguous()

    def _chunked(self, m: Any, q: Any, k: Any, v: Any, gates: dict) -> torch.Tensor:
        """A batch with prompts: KDA by chunks, from each sequence's state (zero for
        a new one), keeping the last."""
        state = self.kv_cache[1]
        idx, has = m.non_spec_state_indices_tensor, m.has_initial_state
        chunked, _ = kernels()
        o, last = chunked(
            q=q,
            k=k,
            v=v,
            g_bias=self.dt_bias,
            initial_state=gather_initial_states(state, idx, has),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=m.non_spec_query_start_loc,
            **gates,
        )
        state[idx] = last.to(state.dtype)
        return o

    def _recurrent(self, m: Any, q: Any, k: Any, v: Any, gates: dict) -> torch.Tensor:
        """A batch of decoding steps: KDA's step, in place on each sequence's state."""
        n = m.num_decodes
        _, recurrent = kernels()
        o, _ = recurrent(
            q=q,
            k=k,
            v=v,
            dt_bias=self.dt_bias,
            initial_state=self.kv_cache[1],
            cu_seqlens=m.non_spec_query_start_loc[: n + 1],
            ssm_state_indices=m.non_spec_state_indices_tensor[:n],
            **gates,
        )
        return o


class KimiDeltaAttention(Layer):
    """bux's ``kda`` (`Mixer`): q, k and v convolved with SiLU, q and k
    L2-normalized at their own heads, then each serving its group of value heads, q
    scaled by ``head_dim ** -0.5``, the delta rule with a decay per key channel
    (``-exp(A_log) softplus(f + dt_bias)``, or ``lower_bound * sigmoid(exp(A_log)
    (f + dt_bias))``, f low rank) and ``sigmoid(b)`` steps, each value head's
    RMSNorm times the gate's activation (a dense gate, or a low-rank one), and the
    output projection; by bux's own rule where vLLM's kernels do not take the
    config (`takes`)."""

    bux_types = ("kda",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        own = deltanet.Mixer if not takes(c) else Mixer
        self.mixer = own(c, width, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        if isinstance(m, deltanet.Mixer):
            return m.targets()
        merged = ["q.weight", "k.weight", "v.weight"]
        merged += ["gate.weight"] if m.dense_gate else []
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            n: (m.in_proj.weight, (i,)) for i, n in enumerate(merged)
        }
        out["f.down"] = (m.in_proj.weight, (len(merged),))
        out["b.weight"] = (m.in_proj.weight, (len(merged) + 1,))
        out |= {
            "f.up": (m.f_up.weight, ()),
            "A_log": (m.A_log, ()),
            "dt_bias": (m.dt_bias, ()),
            "norm.weight": (m.norm.weight, ()),
            "o.weight": (m.o.weight, ()),
        }
        if not m.dense_gate:
            out["gate.down"] = (m.gate_down.weight, ())
            out["gate.up"] = (m.gate_up.weight, ())
        return out

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if isinstance(self.mixer, deltanet.Mixer):
            return deltanet.load(self, self.mixer, name, tensor)
        if name == "conv.weight":
            return [self.mixer.load_conv(tensor)]
        return super().load(name, low_rank_tensor(name, tensor))

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)
