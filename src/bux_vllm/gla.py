"""Gated linear attention in vLLM: bux's ``gla``, its projections by vLLM's parallel
layers, its conv (if it has one) by vLLM's causal conv kernels, and its recurrence
by bux's delta rule with no rank-one term (`bux_vllm.delta`, fla's kernels where they
run), its state in vLLM's Mamba cache (`Recurrent`).
"""

from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from . import feeds
from .common import ACTIVATIONS, Ctx, Layer, Norm, Site
from .gdn2 import rms, state_shape
from .linear import gla_projection, linear_dims, zero_padding
from .recurrent import LowRank, Recurrent, low_rank_tensor


class Mixer(Recurrent):
    """vLLM's layer of one ``gla`` call: q, k and v by one merged projection, the
    decay's low-rank one, the conv if it has one and the recurrence
    (`Recurrent.convolved`, `Recurrent.recur`) out of the compiled graph, and the
    rest in it.

    Raises:
        NotImplementedError: Speculative decoding.
    """

    what = "gla"

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str):
        super().__init__(vllm_config, prefix)
        self.c = c
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        projection = gla_projection(c, self.tp)  # q, k and v
        self.split = projection.split
        self.in_proj = MergedColumnParallelLinear(
            width,
            projection.sizes,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.in_proj",
        )
        zero_padding(self.in_proj, sum(self.split[1:]))
        self.f = LowRank(width, c["gate_rank"], hv * dk, f"{prefix}.f", bias=True)
        self.gate = None
        if c["output_gate"] is not None:
            self.gate = ColumnParallelLinear(
                width, hv * dv, bias=False, return_bias=False, prefix=f"{prefix}.gate"
            )
        self.conv_taps(c)
        self.norm = Norm(rms(c["norm_eps"]), dv)
        self.o = RowParallelLinear(
            hv * dv, width, bias=False, return_bias=False, prefix=f"{prefix}.o"
        )

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, 0, self.tp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        qkv = self.convolved(self.in_proj(x).split(self.split, -1)[0], silu=False)
        heads = (self.h, self.hv, self.dk, self.dv)
        xs = feeds.gla_rule(heads, qkv, self.f(x))
        o = self.run("recur", xs, x.new_zeros(x.shape[0], self.hv, self.dv))
        o = self.norm(o).flatten(1)
        if self.gate is not None:
            o = o * ACTIVATIONS[self.c["output_gate"]](self.gate(x))
        return self.o(o)


class GatedLinearAttention(Layer):
    """bux's ``gla`` (`Mixer`): q, k and v (convolved and SiLU'd if it has a conv),
    q and k shared out to their value heads, q scaled by ``dk ** -0.5``,
    ``S_t = diag(w_t) S_{t-1} + k_t v_t^T`` with ``log w = log sigmoid(f) / 16`` per
    key channel of each value head, f a low-rank projection with a bias, each value
    head's RMSNorm, the output gate, and the output projection."""

    bux_types = ("gla",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            f"{n}.weight": (m.in_proj.weight, (i,)) for i, n in enumerate("qkv")
        }
        out |= m.f.targets("f")
        if m.gate is not None:
            out["gate.weight"] = (m.gate.weight, ())
        return out | {"norm.weight": (m.norm.weight, ()), "o.weight": (m.o.weight, ())}

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            return [self.mixer.load_conv(tensor)]
        return super().load(name, low_rank_tensor(name, tensor))

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)
