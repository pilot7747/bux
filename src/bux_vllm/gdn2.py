"""Gated DeltaNet-2 in vLLM: bux's ``gdn2``, its projections by vLLM's parallel
layers, its conv by vLLM's causal conv kernels, and its delta rule, which no kernel
of vLLM's computes (an erase gate per key channel makes it a DPLR rule, not KDA's),
in PyTorch (`bux_vllm.delta`), its state in vLLM's Mamba cache (`Recurrent`).
"""

from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

from . import feeds
from .common import ACTIVATIONS, Ctx, Layer, Norm, Site
from .linear import gdn2_projection, linear_dims, zero_padding
from .recurrent import LowRank, Recurrent, float32, low_rank_tensor


def state_shape(c: dict, width: int, tp: int) -> tuple:
    """The shapes of a ``gdn2`` layer's state on one of `tp` ranks: its conv's last
    inputs of q, k and v, and each value head's (dv, dk) state."""
    h, hv, dk, dv = linear_dims(c)
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp, h, hv, dk, dv, c["conv_size"]
    )


def rms(eps: float) -> dict:
    """The config of an RMSNorm of `eps` with a gain, as bux's norms' JSON."""
    return {
        "type": "rmsnorm",
        "eps": eps,
        "affine": True,
        "bias": False,
        "zero_centered": False,
        "scale_input": False,
    }


class Mixer(Recurrent):
    """vLLM's layer of one ``gdn2`` call: q, k, v and the erase and write gates by
    one merged projection, the decay's and the output gate's low-rank ones, the conv
    and the rule (`Recurrent.convolve`, `Recurrent.recur`) out of the compiled
    graph, and the rest in it.

    Raises:
        NotImplementedError: No conv, or speculative decoding.
    """

    what = "gdn2"

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str):
        if c["conv_size"] is None:
            raise NotImplementedError("gdn2: no conv not in vLLM yet")
        super().__init__(vllm_config, prefix)
        self.c = c
        bound = c["gate_lower_bound"]
        self.floor = None if bound is None else -bound
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        sizes, self.split, _ = gdn2_projection(c, self.tp)  # q, k, v, erase, write
        self.in_proj = MergedColumnParallelLinear(
            width, sizes, bias=False, return_bias=False, prefix=f"{prefix}.in_proj"
        )
        zero_padding(self.in_proj, sum(self.split[3:]))
        self.f = LowRank(width, c["gate_rank"] or dv, h * dk, f"{prefix}.f")
        self.gate = None
        if c["output_gate"] is not None:
            rank = c["output_gate_rank"] or dv
            self.gate = LowRank(width, rank, hv * dv, f"{prefix}.gate", bias=True)
        self.conv_taps(c)
        self.A_log, self.dt_bias = float32(self.h), float32(self.h * dk)
        self.norm = Norm(rms(c["norm_eps"]), dv)
        self.o = RowParallelLinear(
            hv * dv, width, bias=False, return_bias=False, prefix=f"{prefix}.o"
        )

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, 0, self.tp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        qkv, erase, write = self.in_proj(x).split(self.split, -1)[:3]
        qkv = self.run("convolve", [qkv], qkv.new_zeros(qkv.shape))
        heads = (self.h, self.hv, self.dk, self.dv)
        gates = (self.f(x), erase, write)
        xs = feeds.gdn2_rule(self.c, heads, qkv, gates, self.A_log, self.dt_bias)
        o = self.run("recur", xs, x.new_zeros(x.shape[0], self.hv, self.dv))
        o = self.norm(o).flatten(1)
        if self.gate is not None:
            o = o * ACTIVATIONS[self.c["output_gate"]](self.gate(x))
        return self.o(o)


class GatedDeltaNet2(Layer):
    """bux's ``gdn2`` (`Mixer`): q, k and v convolved with SiLU, q and k
    L2-normalized and shared out to their value heads, q scaled by ``dk ** -0.5``,
    ``S_t = (I - k_t (b_t * k_t)^T) diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T`` with
    KDA's decay, the erase gate b (times 2 with negative eigenvalues) and the write
    gate w, each value head's RMSNorm, the low-rank output gate, and the output
    projection."""

    bux_types = ("gdn2",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        merged = ["q.weight", "k.weight", "v.weight", "erase.weight", "write.weight"]
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            n: (m.in_proj.weight, (i,)) for i, n in enumerate(merged)
        }
        out |= m.f.targets("f")
        if m.gate is not None:
            out |= m.gate.targets("gate")
        return out | {
            "A_log": (m.A_log, ()),
            "dt_bias": (m.dt_bias, ()),
            "norm.weight": (m.norm.weight, ()),
            "o.weight": (m.o.weight, ()),
        }

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            return [self.mixer.load_conv(tensor)]
        return super().load(name, low_rank_tensor(name, tensor))

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)
