"""RWKV-7 time mixing in vLLM: bux's ``rwkv7``, its projections by vLLM's parallel
layers, and its generalized delta rule, which no kernel of vLLM's computes, in
PyTorch (`bux_vllm.delta`). Its state is in vLLM's Mamba cache (`Recurrent`): the
token before, which the token shift reads, as a conv state of one position, and
each value head's (dv, dk) state.
"""

import math
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

from . import delta, feeds, kv
from .common import Ctx, Layer, Norm, Site, sharded_loader
from .linear import linear_dims
from .recurrent import LowRank, Recurrent, low_rank_tensor


def state_shape(c: dict, width: int, tp: int) -> tuple:
    """The shapes of an ``rwkv7`` layer's state on one of `tp` ranks: the token
    before, all of its width on every rank, and each value head's (dv, dk) state."""
    _, hv, dk, dv = linear_dims(c)
    (shift,) = MambaStateShapeCalculator.short_conv_state_shape(1, width, 2)
    return shift, (hv // tp, dv, dk)


def sharded(*shape: int) -> nn.Parameter:
    """A parameter of this rank's share of the first dim, in the model's dtype."""
    p = nn.Parameter(torch.empty(*shape), requires_grad=False)
    p.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]
    return p


def layernorm(eps: float) -> dict:
    """The config of a layernorm of `eps` with a gain and a bias, as bux's JSON."""
    return {
        "type": "layernorm",
        "eps": eps,
        "affine": True,
        "bias": True,
        "zero_centered": False,
        "scale_input": False,
    }


class Mixer(Recurrent):
    """vLLM's layer of one ``rwkv7`` call: the token shift and the rule
    (`shift`, `Recurrent.recur`) out of the compiled graph, the rest in it.

    Raises:
        NotImplementedError: A short conv, which RWKV-7 has none of, or
            speculative decoding.
    """

    what = "rwkv7"
    floor = math.exp(-0.5)  # log w = -exp(-0.5) sigmoid(.)

    def __init__(self, c: dict, width: int, vllm_config: Any, prefix: str, ports: list):
        if c["conv_size"] is not None:
            raise NotImplementedError("rwkv7 with a short conv (it shifts tokens)")
        super().__init__(vllm_config, prefix)
        self.c, self.width = c, width
        h, hv, dk, dv = linear_dims(c)
        self.h, self.hv, self.dk, self.dv = h // self.tp, hv // self.tp, dk, dv
        self.reads, self.writes = kv.reads(ports, "v"), kv.writes(ports, "v")
        self._projections(c, width, prefix)
        self.shift = nn.Parameter(torch.empty(6, width), requires_grad=False)
        self.k_k, self.k_a = sharded(h * dk // self.tp), sharded(h * dk // self.tp)
        self.r_k = sharded(self.hv, dk)
        self.norm = Norm(layernorm(dv * c["norm_eps"]), (self.hv, dv))
        for p in (self.norm.weight, self.norm.bias):
            assert p is not None
            p.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]
        self.o = RowParallelLinear(
            hv * dv, width, bias=False, return_bias=False, prefix=f"{prefix}.o"
        )

    def _projections(self, c: dict, width: int, prefix: str) -> None:
        """r, k and v; the low-rank decay w, learning rate a, value residual's mix
        and gate g."""
        h, hv, dk, dv = linear_dims(c)
        nk, nv = h * dk, hv * dv

        def column(name: str, n: int) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                width, n, bias=False, return_bias=False, prefix=f"{prefix}.{name}"
            )

        self.r, self.k, self.v = column("r", nk), column("k", nk), column("v", nv)
        self.w = LowRank(width, c["decay_rank"], nk, f"{prefix}.w", "tanh", True)
        self.a = LowRank(width, c["a_rank"], nk, f"{prefix}.a", bias=True)
        self.v_mix = None
        if self.reads is not None:
            self.v_mix = LowRank(width, c["v_rank"], nv, f"{prefix}.v_mix", bias=True)
        self.g = None
        if c["output_gate"] is not None:
            gate = c["output_gate"]
            self.g = LowRank(width, c["gate_rank"], nv, f"{prefix}.g", gate)

    def get_state_shape(self) -> tuple:
        return state_shape(self.c, self.width, self.tp)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """The layer's output, (T, width), from the tokens' inputs."""
        prev = self.run("shifted", [x], torch.zeros_like(x))
        xr, xw, xk, xv, xa, xg = feeds.rwkv7_mixes(x, prev, self.shift)
        heads = (self.h, self.hv, self.dk, self.dv)
        raw = (self.w(xw), self.a(xa))
        v = self._values(xv, ctx)
        xs = feeds.rwkv7_rule(heads, self.r(xr), self.k(xk), v, raw, self.k_k, self.k_a)
        o = self.run("recur", xs, x.new_zeros(x.shape[0], self.hv, self.dv))
        o = self.norm(o) + feeds.rwkv7_bonus(xs[0], xs[1], v, self.r_k)
        o = o.flatten(1)
        if self.g is not None:
            o = o * self.g(xg)
        return self.o(o)

    def _values(self, xv: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """The values, (T, hv, dv), with those the ``v`` port reads mixed in by the
        learned amount (the value residual), pushed to the taps it writes."""
        v = self.v(xv).unflatten(-1, (self.hv, self.dv))
        if self.v_mix is not None and self.reads is not None:
            mix = torch.sigmoid(self.v_mix(xv)).reshape(v.shape)
            v = v + (ctx.read(*self.reads) - v) * mix
        for tap in self.writes:
            ctx.push(tap, v)
        return v

    def shifted(self, xs: list[torch.Tensor], out: torch.Tensor) -> None:
        """A step: each token's predecessor in its sequence, (T, width), into `out`;
        a sequence's first takes the token before it from the cache (zero for a new
        sequence), and its last stays there."""
        m = self.batch()
        if m is None:
            return
        n = self.tokens(m)
        x = xs[0][:n]
        cache = self.kv_cache[0].flatten(1)  # (slots, width), either layout
        idx = m.non_spec_state_indices_tensor[: m.num_prefills + m.num_decodes]
        before, last = cache[idx], x
        if m.num_prefills:
            has = m.has_initial_state[:, None]
            before = torch.where(has, before, torch.zeros_like(before))
            starts = m.non_spec_query_start_loc[: idx.shape[0] + 1].long()
            before, last = delta.previous(x, starts, before), x[starts[1:] - 1]
        kept = (idx > 0)[:, None]  # padding's slots stay as they were
        cache[idx] = torch.where(kept, last.to(cache.dtype), cache[idx])
        out[:n] = before


class RWKV7(Layer):
    """bux's ``rwkv7`` (`Mixer`): the token shift, r, k and v, the low-rank decay w,
    in-context learning rate a, value residual and gate, the generalized delta rule
    ``S_t = (diag(w_t) - (a_t kk_t) kk_t^T) S_{t-1} + k_t v_t^T``, a layernorm of
    each head plus the bonus ``(r k r_k) v``, the gate and the output projection."""

    bux_types = ("rwkv7",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.mixer = Mixer(c, width, site.vllm_config, site.caches[0], site.ports)

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        out: dict[str, tuple[nn.Parameter, tuple]] = {
            f"{n}.weight": (getattr(m, n).weight, ()) for n in ("r", "k", "v", "o")
        }
        for name in ("w", "a", "v_mix", "g"):
            if (low := getattr(m, name)) is not None:
                out |= low.targets(name)
        out |= {n: (getattr(m, n), ()) for n in ("shift", "k_k", "k_a", "r_k")}
        assert m.norm.weight is not None and m.norm.bias is not None
        return out | {
            "norm.weight": (m.norm.weight, ()),
            "norm.bias": (m.norm.bias, ()),
        }

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        return super().load(name, low_rank_tensor(name, tensor))

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        assert ctx is not None
        return self.mixer(x, ctx)
