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
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KimiGDNMergedColumnParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .common import Ctx, Layer, Site, sharded_loader
from .linear import conv_channels, conv_shard, linear_dims, to_value_heads

#: The op the conv and recurrence run in, outside the compiled graph.
OP = "bux_kda"

#: The lowest gate_lower_bound vLLM's KDA kernels hold to (Kimi's -5).
LOWEST_BOUND = -5.0


def check(c: dict) -> None:
    """Raise if vLLM's kernels cannot run the ``kda`` config `c`.

    Raises:
        NotImplementedError: No conv or no output gate, value heads of another
            width than q and k's, or a lower bound under `LOWEST_BOUND`.
    """
    if c["conv_size"] is None or c["output_gate"] is None:
        raise NotImplementedError("kda: no conv or no output gate not in vLLM yet")
    if (c["v_head_dim"] or c["head_dim"]) != c["head_dim"]:
        raise NotImplementedError("kda: v_head_dim other than head_dim not in vLLM yet")
    bound = c["gate_lower_bound"]
    if bound is not None and bound < LOWEST_BOUND:
        raise NotImplementedError(f"kda: gate_lower_bound under {LOWEST_BOUND}")


def state_shape(c: dict, tp: int) -> tuple:
    """The shapes of a ``kda`` layer's state on one of `tp` ranks: its conv's last
    inputs of q, k and v, and each value head's (dv, dk) state."""
    h, hv, dk, dv = linear_dims(c)
    return MambaStateShapeCalculator.kda_state_shape(
        tp, hv, dv, h, dk, conv_kernel_size=c["conv_size"]
    )


def state_dtype(model_config: Any, cache_config: Any) -> tuple:
    """The dtypes of a ``kda`` layer's state: the conv's the model's (or the cache's),
    the recurrence's float32 (or ``mamba_ssm_cache_dtype``)."""
    return MambaStateDtypeCalculator.kda_state_dtype(
        model_config.dtype,
        cache_config.mamba_cache_dtype,
        cache_config.mamba_ssm_cache_dtype,
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


class Mixer(nn.Module, MambaBase):
    """vLLM's layer of one ``kda`` call: its weights by rank, its state in the Mamba
    cache under `prefix`, the projections in the compiled graph and `mix` out of it.

    Raises:
        NotImplementedError: Speculative decoding, or what `check` raises for.
    """

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str):
        super().__init__()
        check(c)
        if vllm_config.num_speculative_tokens:
            raise NotImplementedError("kda: speculative decoding not in vLLM yet")
        self.c, self.prefix = c, prefix
        self.model_config, self.cache_config = (
            vllm_config.model_config,
            vllm_config.cache_config,
        )
        self.tp = get_tensor_model_parallel_world_size()
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        self.group = hv // h
        self._projections(c, width)
        self._weights(c)
        self.norm = FusedRMSNormGated(
            dv, eps=c["norm_eps"], activation=c["output_gate"]
        )
        self.o = RowParallelLinear(
            hv * dv, width, bias=False, return_bias=False, prefix=f"{prefix}.o"
        )
        _register(vllm_config.compilation_config, prefix, self)

    def _projections(self, c: dict, width: int) -> None:
        """q, k, v, a dense output gate, the decay's down projection (on every rank)
        and beta, by one merged projection; the decay's up projection, and a
        low-rank output gate's two."""
        _, hv, dk, dv = linear_dims(c)
        rank = c["gate_rank"] or dk
        self.dense_gate = c["output_gate_rank"] is None
        gate = [hv * dv] if self.dense_gate else []
        sizes = [*conv_channels(c), *gate, rank, hv]
        self.split = [sum(conv_channels(c)) // self.tp, *(s // self.tp for s in gate)]
        self.split += [rank, hv // self.tp]
        self.in_proj = _KimiGDNMergedColumnParallelLinear(
            width,
            sizes,
            replicated_shard_id=len(sizes) - 2,
            tp_size=self.tp,
            bias=False,
            return_bias=False,
            prefix=f"{self.prefix}.in_proj",
        )
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

    def _weights(self, c: dict) -> None:
        """The conv's taps, A_log and dt_bias (float32, as vLLM's KDA keeps them)."""
        channels = sum(conv_channels(c)) // self.tp
        self.conv = nn.Parameter(
            torch.empty(channels, 1, c["conv_size"]), requires_grad=False
        )
        f32 = torch.float32
        self.A_log = nn.Parameter(torch.empty(self.hv, dtype=f32), requires_grad=False)
        self.dt_bias = nn.Parameter(
            torch.empty(self.hv * self.dk, dtype=f32), requires_grad=False
        )
        for p in (self.A_log, self.dt_bias):
            p.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, self.tp)

    def get_state_dtype(self) -> tuple:
        return state_dtype(self.model_config, self.cache_config)

    def load_conv(self, tensor: torch.Tensor) -> nn.Parameter:
        """Load bux's conv taps, (k, C), as this rank's; return the parameter."""
        rank = get_tensor_model_parallel_rank()
        self.conv.data.copy_(conv_shard(tensor, conv_channels(self.c), rank, self.tp))
        return self.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        n = x.shape[0]
        qkv, *rest = self.in_proj(x).split(self.split, -1)
        if self.dense_gate:
            gate, f, beta = rest
        else:
            (f, beta), gate = rest, self.gate_up(self.gate_down(x))
        out = x.new_zeros(n, self.hv, self.dv)
        torch.ops.vllm.bux_kda(qkv, self.f_up(f), beta, out, self.prefix)
        out = self.norm(out, gate.unflatten(-1, (self.hv, self.dv)))
        return self.o(out.flatten(1))

    def mix(
        self, qkv: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Convolve q, k and v and run the delta rule from each sequence's state, its
        decay from `g` and step from `beta`: write the heads' outputs, (T, hv, dv),
        to `out`. In a run without metadata (vLLM's profiling), nothing."""
        meta = get_forward_context().attn_metadata
        m = meta.get(self.prefix) if isinstance(meta, dict) else None
        if m is None or m.num_prefills + m.num_decodes == 0:
            return
        conv_state, state = self.kv_cache
        if not is_conv_state_dim_first():  # the kernels take (C, taps)
            conv_state = conv_state.transpose(-1, -2)
        n = m.num_actual_tokens if m.num_prefills else m.num_decodes
        gates = (g[:n].unflatten(-1, (self.hv, self.dk))[None], beta[:n][None])
        step = self._prefill if m.num_prefills else self._decode
        o = step(qkv[:n], gates, m, conv_state, state)
        out[:n] = o[0]

    def _heads(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """q and k at the value heads, and v, each (1, T, hv, d)."""
        h, dk = self.h, self.dk
        q, k, v = x.split([h * dk, h * dk, self.hv * self.dv], -1)
        q, k = (to_value_heads(y, h, self.group) for y in (q, k))
        return q, k, v.reshape(1, -1, self.hv, self.dv).contiguous()

    def _gates(self) -> dict:
        """What both kernels take of the decay: A_log, dt_bias and the bound."""
        return {"A_log": self.A_log, "lower_bound": self.c["gate_lower_bound"]}

    def _prefill(
        self, x: torch.Tensor, gates: tuple, m: Any, conv_state: Any, state: Any
    ) -> torch.Tensor:
        """A batch with prompts: the conv and KDA by chunks, from each sequence's
        state (zero for a new one), keeping the last."""
        idx, has = m.non_spec_state_indices_tensor, m.has_initial_state
        x = causal_conv1d_fn(
            x.transpose(0, 1),
            self.conv.flatten(1),
            None,
            conv_states=conv_state,
            query_start_loc=m.non_spec_query_start_loc,
            cache_indices=idx,
            has_initial_state=has,
            activation="silu",
            metadata=m,
        ).transpose(0, 1)
        q, k, v = self._heads(x)
        chunked, _ = kernels()
        o, last = chunked(
            q=q,
            k=k,
            v=v,
            raw_g=gates[0],
            raw_beta=gates[1],
            g_bias=self.dt_bias,
            initial_state=gather_initial_states(state, idx, has),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=m.non_spec_query_start_loc,
            **self._gates(),
        )
        state[idx] = last.to(state.dtype)
        return o

    def _decode(
        self, x: torch.Tensor, gates: tuple, m: Any, conv_state: Any, state: Any
    ) -> torch.Tensor:
        """A batch of decoding steps: the conv's step and KDA's, in place on each
        sequence's state."""
        n = m.num_decodes
        idx = m.non_spec_state_indices_tensor[:n]
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        x = causal_conv1d_update(
            x,
            conv_state,
            self.conv.flatten(1),
            None,
            activation="silu",
            conv_state_indices=idx,
            validate_data=True,
            out=y,
        )
        q, k, v = self._heads(x)
        _, recurrent = kernels()
        o, _ = recurrent(
            q=q,
            k=k,
            v=v,
            raw_g=gates[0],
            raw_beta=gates[1],
            dt_bias=self.dt_bias,
            initial_state=state,
            cu_seqlens=m.non_spec_query_start_loc[: n + 1],
            ssm_state_indices=idx,
            **self._gates(),
        )
        return o


def _register(compilation: Any, prefix: str, layer: nn.Module) -> None:
    """Make `layer` vLLM's layer `prefix`, and `OP` a place the graph splits at."""
    if prefix in compilation.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation.static_forward_context[prefix] = layer
    ops = compilation.splitting_ops
    if ops is not None and f"vllm::{OP}" not in ops:
        ops.append(f"vllm::{OP}")


class KimiDeltaAttention(Layer):
    """bux's ``kda`` (`Mixer`): q, k and v convolved with SiLU, q and k
    L2-normalized at their own heads, then each serving its group of value heads, q
    scaled by ``head_dim ** -0.5``, the delta rule with a decay per key channel
    (``-exp(A_log) softplus(f + dt_bias)``, or ``lower_bound * sigmoid(exp(A_log)
    (f + dt_bias))``, f low rank) and ``sigmoid(b)`` steps, each value head's
    RMSNorm times the gate's activation (a dense gate, or a low-rank one), and the
    output projection."""

    bux_types = ("kda",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
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
        if name == "conv.weight":
            return [self.mixer.load_conv(tensor)]
        if name.endswith((".down", ".up")):  # bux's x A B; vLLM's layers hold A^T, B^T
            tensor = tensor.T
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)
