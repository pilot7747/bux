"""Lightning attention in vLLM: bux's ``lightning``, MiniMax-01's linear attention, by
bux's delta rule with no rank-one term (`bux_vllm.delta`, fla's DPLR kernels where
they run), as ``gla`` runs.

q, k and v are projected by heads (convolved first, with a conv), SiLU'd and
unscaled; each value head's state decays by a fixed ``exp(-s)``, whose rate s
weakens with the layer's depth among the model's token mixers
(`bux_vllm.linear.slopes`, the depth from the export's sites); one RMSNorm takes all
heads together, its statistics summed over the ranks (`AllHeads`), then the gate and
the output projection. Each value head's state stays in vLLM's Mamba cache, in
float32 (`Recurrent`).
"""

from typing import Any

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from . import feeds
from .attention import AllHeads
from .common import ACTIVATIONS, Ctx, Layer, Site
from .gdn2 import rms, state_shape
from .linear import gla_projection, linear_dims, slopes, zero_padding
from .recurrent import Recurrent


class Mixer(Recurrent):
    """vLLM's layer of one ``lightning`` call: q, k and v by one merged projection,
    the conv if it has one and the recurrence (`Recurrent.convolved`,
    `Recurrent.recur`) out of the compiled graph, and the rest in it.

    Args:
        c: The layer's config, as JSON.
        width: The width of the stream.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of the state, unique in the model.
        rates: This rank's value heads' decay rates, (hv,).

    Raises:
        NotImplementedError: Speculative decoding.
    """

    what = "lightning"

    def __init__(
        self, c: dict, width: int, vllm_config: Any, prefix: str, rates: torch.Tensor
    ):
        super().__init__(vllm_config, prefix)
        self.c = c
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        self.register_buffer("rates", rates, persistent=False)
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
        self.gate = None
        if c["output_gate"] is not None:
            self.gate = ColumnParallelLinear(
                width, hv * dv, bias=False, return_bias=False, prefix=f"{prefix}.gate"
            )
        self.conv_taps(c)
        self.norm = AllHeads(rms(c["norm_eps"]), hv, dv, self.tp)
        self.o = RowParallelLinear(
            hv * dv, width, bias=False, return_bias=False, prefix=f"{prefix}.o"
        )

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, 0, self.tp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        qkv = self.convolved(self.in_proj(x).split(self.split, -1)[0])
        heads = (self.h, self.hv, self.dk, self.dv)
        xs = feeds.lightning_rule(heads, qkv, self.rates)
        o = self.run("recur", xs, x.new_zeros(x.shape[0], self.hv, self.dv))
        o = self.norm(o.flatten(1))
        if self.gate is not None:
            o = o * ACTIVATIONS[self.c["output_gate"]](self.gate(x))
        return self.o(o)


class Lightning(Layer):
    """bux's ``lightning`` (`Mixer`): q, k and v SiLU'd (convolved first, with a
    conv), q and k shared out to their value heads, ``S_t = exp(-s) S_{t-1} + k_t
    v_t^T`` with a rate s per value head by the layer's depth, the RMSNorm of all
    heads, the gate's activation, and the output projection."""

    bux_types = ("lightning",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        _, hv, _, _ = linear_dims(c)
        depth = site.config.bux_sites[site.path]["extra"]
        rates = slopes(hv, depth["layer"], depth["layers"])
        tp = get_tensor_model_parallel_world_size()
        mine, rank = hv // tp, get_tensor_model_parallel_rank()
        share = rates[rank * mine : (rank + 1) * mine].contiguous()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0], share)

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            f"{n}.weight": (m.in_proj.weight, (i,)) for i, n in enumerate("qkv")
        }
        if m.gate is not None:
            out["gate.weight"] = (m.gate.weight, ())
        out["o.weight"] = (m.o.weight, ())
        return out | {f"norm.{k}": v for k, v in m.norm.targets().items()}

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            return [self.mixer.load_conv(tensor)]
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)
