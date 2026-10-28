"""DeltaNet in vLLM: bux's ``deltanet``, its projections by vLLM's parallel layers,
its conv by vLLM's causal conv kernels, and its delta rule by bux's own
(`bux_vllm.delta`, fla's kernels where they run), with no decay, its state in vLLM's
Mamba cache (`Recurrent`).
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
from .linear import deltanet_projection, linear_dims, zero_padding
from .recurrent import Recurrent


class Mixer(Recurrent):
    """vLLM's layer of one ``deltanet`` call: q, k, v and the step by one merged
    projection, the conv and the rule (`Recurrent.convolve`, `Recurrent.recur`) out
    of the compiled graph, and the rest in it.

    Raises:
        NotImplementedError: No conv, or speculative decoding.
    """

    what = "deltanet"
    floor = 0.0  # no decay

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str):
        if c["conv_size"] is None:
            raise NotImplementedError("deltanet: no conv not in vLLM yet")
        super().__init__(vllm_config, prefix)
        self.c = c
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        sizes, self.split, _ = deltanet_projection(c, self.tp)  # q, k, v, step
        self.in_proj = MergedColumnParallelLinear(
            width, sizes, bias=False, return_bias=False, prefix=f"{prefix}.in_proj"
        )
        zero_padding(self.in_proj, sum(self.split[2:]))
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
        qkv, beta = self.in_proj(x).split(self.split, -1)[:2]
        qkv = self.run("convolve", [qkv], qkv.new_zeros(qkv.shape))
        heads = (self.h, self.hv, self.dk, self.dv)
        xs = feeds.deltanet_rule(heads, qkv, beta)
        o = self.run("recur", xs, x.new_zeros(x.shape[0], self.hv, self.dv))
        o = self.norm(o).flatten(1)
        if self.gate is not None:
            o = o * ACTIVATIONS[self.c["output_gate"]](self.gate(x))
        return self.o(o)


class DeltaNet(Layer):
    """bux's ``deltanet`` (`Mixer`): q, k and v convolved with SiLU, q and k
    L2-normalized and shared out to their value heads, q scaled by ``dk ** -0.5``,
    ``S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`` with a step
    ``beta = sigmoid(W_b x)`` per value head, each value head's RMSNorm, the
    output gate, and the output projection."""

    bux_types = ("deltanet",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        merged = ["q.weight", "k.weight", "v.weight", "b.weight"]
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            n: (m.in_proj.weight, (i,)) for i, n in enumerate(merged)
        }
        if m.gate is not None:
            out["gate.weight"] = (m.gate.weight, ())
        return out | {"norm.weight": (m.norm.weight, ()), "o.weight": (m.o.weight, ())}

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            return [self.mixer.load_conv(tensor)]
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)
