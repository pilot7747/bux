"""The gated delta rule in vLLM by bux's own (`bux_vllm.delta`, fla's kernels where
they run): bux's ``deltanet``, with no decay, and the ``gdn`` and ``kda`` layers that
vLLM's own kernels do not take (`runs`), with theirs. Their projections run on vLLM's
parallel layers, their conv (if they have one) on vLLM's causal conv kernels, and
their state stays in vLLM's Mamba cache (`Recurrent`).
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
from .recurrent import LowRank, Recurrent, float32, low_rank_tensor


class Mixer(Recurrent):
    """vLLM's layer of one call of the rule: q, k, v, the step and ``gdn``'s decay by
    one merged projection, ``kda``'s decay and a low-rank output gate by low-rank
    ones, the conv if it has one and the rule (`Recurrent.convolved`,
    `Recurrent.recur`) out of the compiled graph, and the rest in it.

    Raises:
        NotImplementedError: Speculative decoding.
    """

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str):
        self.what = c["type"]
        super().__init__(vllm_config, prefix)
        self.c = c
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        projection = deltanet_projection(c, self.tp)  # q, k, v, step[, decay]
        self.split, self.parts = projection.split, projection.parts
        kw = {"bias": False, "return_bias": False, "quant_config": self.quant}
        self.in_proj = MergedColumnParallelLinear(
            width, projection.sizes, prefix=f"{prefix}.in_proj", **kw
        )
        zero_padding(self.in_proj, sum(self.split[self.parts :]))
        self._decay(c, width)
        self._gate(c, width)
        self.conv_taps(c)
        self.norm = Norm(rms(c["norm_eps"]), dv)
        self.o = RowParallelLinear(hv * dv, width, prefix=f"{prefix}.o", **kw)

    def _decay(self, c: dict, width: int) -> None:
        """The decay's parameters: none for ``deltanet``, a rate per value head for
        ``gdn``, and a low-rank projection to every key channel for ``kda``."""
        self.f, self.A_log, self.dt_bias = None, None, None
        self.floor = 0.0 if c["type"] == "deltanet" else None
        _, hv, dk, _ = linear_dims(c)
        if c["type"] == "gdn":
            self.A_log, self.dt_bias = float32(self.hv), float32(self.hv)
        if c["type"] == "kda":
            rank = c["gate_rank"] or dk
            self.f = LowRank(width, rank, hv * dk, f"{self.prefix}.f", quant=self.quant)
            self.A_log, self.dt_bias = float32(self.hv), float32(self.hv * dk)
            bound = c["gate_lower_bound"]
            self.floor = None if bound is None else -bound

    def _gate(self, c: dict, width: int) -> None:
        """The output gate: none, dense, or ``kda``'s low-rank one."""
        _, hv, _, dv = linear_dims(c)
        self.gate: nn.Module | None = None
        if c["output_gate"] is None:
            return
        prefix = f"{self.prefix}.gate"
        if c.get("output_gate_rank"):
            rank = c["output_gate_rank"]
            self.gate = LowRank(width, rank, hv * dv, prefix, quant=self.quant)
            return
        self.gate = ColumnParallelLinear(
            width,
            hv * dv,
            bias=False,
            return_bias=False,
            quant_config=self.quant,
            prefix=prefix,
        )

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, 0, self.tp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        qkv, beta, *decay = self.in_proj(x).split(self.split, -1)[: self.parts]
        qkv = self.convolved(qkv)
        heads = (self.h, self.hv, self.dk, self.dv)
        xs = feeds.deltanet_rule(heads, qkv, beta, self._log_decay(x, decay))
        o = self.run("recur", xs, x.new_zeros(x.shape[0], self.hv, self.dv))
        o = self.norm(o).flatten(1)
        if self.gate is not None:
            o = o * ACTIVATIONS[self.c["output_gate"]](self.gate(x))
        return self.o(o)

    def _log_decay(self, x: torch.Tensor, decay: list) -> torch.Tensor | None:
        """The log decay, (T, hv, dk) in float32, from ``gdn``'s projection in
        `decay` or ``kda``'s of x; None for ``deltanet``'s."""
        if self.A_log is None or self.dt_bias is None:
            return None
        if self.f is not None:
            return feeds.kda_decay(self.f(x), self.A_log, self.dt_bias, self.c)
        return feeds.gdn_decay(decay[0], self.A_log, self.dt_bias, self.dk)

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        """Its parameters by bux's names."""
        merged = ["q.weight", "k.weight", "v.weight", "b.weight", "a.weight"]
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            n: (self.in_proj.weight, (i,))
            for i, n in enumerate(merged[: self.parts + 2])
        }
        if self.f is not None:
            out |= self.f.targets("f")
        if self.A_log is not None and self.dt_bias is not None:
            out |= {"A_log": (self.A_log, ()), "dt_bias": (self.dt_bias, ())}
        if isinstance(self.gate, LowRank):
            out |= self.gate.targets("gate")
        elif self.gate is not None:
            out["gate.weight"] = (self.gate.weight, ())
        return out | {
            "norm.weight": (self.norm.weight, ()),
            "o.weight": (self.o.weight, ()),
        }

    def load(self, name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        """bux's tensor `name` as the layer loads it; None for the conv's taps,
        which it loads itself."""
        if name == "conv.weight":
            self.load_conv(tensor)
            return None
        return low_rank_tensor(name, tensor)


class DeltaNet(Layer):
    """bux's ``deltanet`` (`Mixer`): q, k and v convolved (if it has a conv) and
    SiLU'd, q and k L2-normalized and shared out to their value heads, q scaled by
    ``dk ** -0.5``, ``S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`` with
    a step ``beta = sigmoid(W_b x)`` per value head, each value head's RMSNorm, the
    output gate, and the output projection."""

    bux_types = ("deltanet",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        return self.mixer.targets()

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        return load(self, self.mixer, name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)


def load(
    layer: Layer, mixer: Mixer, name: str, tensor: torch.Tensor
) -> list[nn.Parameter]:
    """Load bux's tensor `name` into the layer whose `mixer` runs the rule; return
    the parameters it fills."""
    loaded = mixer.load(name, tensor)
    if loaded is None:
        assert mixer.conv is not None
        return [mixer.conv]
    return Layer.load(layer, name, loaded)
