"""Recurrent layers in vLLM, by its own Mamba, Mamba-2 and Gated DeltaNet layers:
bux's ``mamba``, ``mamba2`` and ``gdn``, their state in vLLM's Mamba cache, as the plugin's own
recurrent layers' is (``kda``, ``gdn2``, ``rwkv7``: `bux_vllm.recurrent`).

A model with them is a hybrid, whose class tells vLLM the shape and dtype of a
layer's state from the config, before it builds the model (`state_shapes`,
`state_dtypes`): every recurrent layer of a model must then be of one type and
shape. vLLM's layers take bux's weights after a reshape: bux keeps a conv's taps by
channels, (k, C), and vLLM by channels, (C, 1, k).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

from . import deltanet, gdn2, gla, kda, lightning, recurrent, rwkv7
from .common import Ctx, Layer, Site
from .linear import conv_weight, linear_dims
from .plan import live, plan

#: The recurrent layer types the plugin computes itself, by their modules.
OWN = {
    "kda": kda,
    "gdn2": gdn2,
    "rwkv7": rwkv7,
    "deltanet": deltanet,
    "gla": gla,
    "lightning": lightning,
}

#: The recurrent layer types, which keep their state in vLLM's Mamba cache.
RECURRENT = ("mamba", "mamba2", "gdn", *OWN)


def mamba2_dims(c: dict, width: int) -> tuple[int, int]:
    """Mamba-2's inner channels and heads, as `bux.architecture.recurrent.ssm`
    splits them."""
    inner = c["num_heads"] * c["head_dim"] if c["num_heads"] else c["expand"] * width
    return inner, inner // c["head_dim"]


def mamba_dims(c: dict, width: int) -> tuple[int, int]:
    """Mamba's inner channels and the rank of its time steps, as bux's."""
    return c["expand"] * width, c["dt_rank"] or math.ceil(width / 16)


class Mamba(Layer):
    """bux's ``mamba`` by vLLM's ``MambaMixer``, as Jamba runs it: u and the gate z
    projected, u convolved with a bias and SiLU'd, the time steps of rank
    ``dt_rank``, B and C RMS-normalized (eps 1e-6), the selective scan with D u, the
    SiLU(z) gate, and the output projection."""

    bux_types = ("mamba",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        inner, rank = mamba_dims(c, width)
        vllm = site.vllm_config
        self.mixer = MambaMixer(
            hidden_size=width,
            ssm_state_size=c["state_size"],
            conv_kernel_size=c["conv_size"],
            intermediate_size=inner,
            time_step_rank=rank,
            use_conv_bias=True,
            use_bias=False,
            use_rms_norm=True,
            rms_norm_eps=1e-6,
            model_config=vllm.model_config,
            cache_config=vllm.cache_config,
            prefix=site.caches[0],
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        out = {
            "in_proj.weight": (m.in_proj.weight, ()),
            "conv.weight": (m.conv1d.weight, ()),
            "conv.bias": (m.conv1d.bias, ()),
            "x_proj.weight": (m.x_proj.weight, ()),
            "dt_proj": (m.dt_proj.weight, ()),
            "dt_bias": (m.dt_proj.bias, ()),
            "A_log": (m.A, ()),  # its loader takes -exp(A_log)
            "D": (m.D, ()),
            "out_proj.weight": (m.out_proj.weight, ()),
        }
        for x in ("dt", "b", "c"):  # bux's dt_norm, vLLM's dt_layernorm
            out[f"{x}_norm.weight"] = (getattr(m, f"{x}_layernorm").weight, ())
        return out

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            tensor = conv_weight(tensor)
        if name == "dt_proj":  # bux's is (rank, inner); vLLM's (inner, rank)
            tensor = tensor.T.contiguous()
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        out = torch.empty_like(x)
        self.mixer(x, out)
        return out


class Mamba2(Layer):
    """bux's ``mamba2`` by vLLM's ``MambaMixer2``: the same projection of z, u, B, C
    and dt, conv with a bias and SiLU, SSD scan, D skip, SiLU(z) gate and RMSNorm
    of each group of heads (eps 1e-5), and output projection."""

    bux_types = ("mamba2",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        inner, heads = mamba2_dims(c, width)
        vllm = site.vllm_config
        self.mixer = MambaMixer2(
            hidden_size=width,
            ssm_state_size=c["state_size"],
            conv_kernel_size=c["conv_size"],
            intermediate_size=inner,
            use_conv_bias=True,
            use_bias=False,
            n_groups=c["num_groups"],
            num_heads=heads,
            head_dim=c["head_dim"],
            rms_norm_eps=1e-5,
            model_config=vllm.model_config,
            cache_config=vllm.cache_config,
            prefix=site.caches[0],
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        return {
            "in_proj.weight": (m.in_proj.weight, ()),
            "conv.weight": (m.conv1d.weight, ()),
            "conv.bias": (m.conv1d.bias, ()),
            "A_log": (m.A, ()),  # its loader takes -exp(A_log)
            "D": (m.D, ()),
            "dt_bias": (m.dt_bias, ()),
            "norm.weight": (m.norm.weight, ()),
            "out_proj.weight": (m.out_proj.weight, ()),
        }

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            tensor = conv_weight(tensor)
        if name == "norm.weight":  # bux's is (groups, channels of a group)
            tensor = tensor.reshape(-1)
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)


class GatedDeltaNet(Layer):
    """bux's ``gdn`` by vLLM's Gated DeltaNet (Qwen3.5's, its projections apart): q,
    k and v convolved with SiLU, q and k L2-normalized, q scaled by
    ``head_dim ** -0.5``, the delta rule with ``exp(-exp(A_log) softplus(a +
    dt_bias))`` decay and ``sigmoid(b)`` steps, each value head's RMSNorm times the
    gate's activation, and the output projection."""

    bux_types = ("gdn",)
    stateful = once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        if c["conv_size"] is None or c["output_gate"] is None:
            raise NotImplementedError("gdn: no conv or no output gate not in vLLM yet")
        h, hv, dk, dv = linear_dims(c)
        config = SimpleNamespace(
            hidden_size=width,
            hidden_act="silu",
            rms_norm_eps=c["norm_eps"],
            linear_num_key_heads=h,
            linear_num_value_heads=hv,
            linear_key_head_dim=dk,
            linear_value_head_dim=dv,
            linear_conv_kernel_dim=c["conv_size"],
            output_gate_type=c["output_gate"],
        )
        self.mixer = QwenGatedDeltaNetAttention(
            config,
            site.vllm_config,
            prefix=site.caches[0],
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        m = self.mixer
        qkvz, ba = m.in_proj_qkvz.weight, m.in_proj_ba.weight
        return {
            "q.weight": (qkvz, (0,)),
            "k.weight": (qkvz, (1,)),
            "v.weight": (qkvz, (2,)),
            "gate.weight": (qkvz, (3,)),
            "b.weight": (ba, (0,)),
            "a.weight": (ba, (1,)),
            "conv.weight": (m.conv1d.weight, ()),
            "A_log": (m.A_log, ()),
            "dt_bias": (m.dt_bias, ()),
            "norm.weight": (m.norm.weight, ()),
            "o.weight": (m.out_proj.weight, ()),
        }

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "conv.weight":
            tensor = conv_weight(tensor)
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.mixer(x)


def _recurrent(config: Any) -> tuple[str, dict, int] | None:
    """The type, config and width of the model's recurrent layers, which must all be
    of one type and shape; None for a model without them.

    Raises:
        NotImplementedError: Its recurrent layers differ.
    """
    calls = live(plan(config.bux, config.bux_reserved).calls)
    layers = [c.layer for c in calls if c.layer["config"]["type"] in RECURRENT]
    if len({repr((n["config"], n["width"])) for n in layers}) > 1:
        raise NotImplementedError("recurrent layers of several shapes not in vLLM yet")
    if not layers:
        return None
    return layers[0]["config"]["type"], layers[0]["config"], layers[0]["width"]


def state_shapes(vllm_config: Any) -> tuple:
    """The shapes of the state of each of the model's recurrent layers: its conv's
    and its scan's, on one rank."""
    found = _recurrent(vllm_config.model_config.hf_config)
    if found is None:
        return ()
    kind, c, width = found
    tp = vllm_config.parallel_config.tensor_parallel_size
    spec = vllm_config.num_speculative_tokens
    if kind in OWN:
        return OWN[kind].state_shape(c, width, tp)
    if kind == "gdn":
        h, hv, dk, dv = linear_dims(c)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp, h, hv, dk, dv, c["conv_size"], spec
        )
    if kind == "mamba":
        if spec:
            raise NotImplementedError("mamba: speculative decoding")
        inner, _ = mamba_dims(c, width)
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp, inner, c["state_size"], c["conv_size"]
        )
    inner, heads = mamba2_dims(c, width)
    return MambaStateShapeCalculator.mamba2_state_shape(
        tp_world_size=tp,
        intermediate_size=inner,
        n_groups=c["num_groups"],
        num_heads=heads,
        head_dim=c["head_dim"],
        state_size=c["state_size"],
        conv_kernel=c["conv_size"],
        num_spec=spec,
    )


def state_dtypes(vllm_config: Any) -> tuple:
    """The dtypes of the state of each of the model's recurrent layers."""
    cache = vllm_config.cache_config
    found = _recurrent(vllm_config.model_config.hf_config)
    if found is not None and found[0] in OWN:
        return recurrent.state_dtype(vllm_config.model_config, cache)
    if found is not None and found[0] == "mamba":
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            cache.mamba_cache_dtype,
            cache.mamba_ssm_cache_dtype,
        )
    return MambaStateDtypeCalculator.mamba2_state_dtype(
        vllm_config.model_config.dtype,
        cache.mamba_cache_dtype,
        cache.mamba_ssm_cache_dtype,
    )


def copy_funcs(kind: str) -> tuple:
    """How vLLM copies the state of a layer of vLLM's `kind` (``MAMBA1``,
    ``MAMBA2``, ``GDN_ATTN``: ``gdn``'s and the plugin's own recurrent layers', or
    ``SHORT_CONV``: a conv on keys and values) for prefix caching."""
    if kind == "MAMBA1":
        return MambaStateCopyFuncCalculator.mamba1_state_copy_func()
    if kind == "GDN_ATTN":
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
    if kind == "SHORT_CONV":
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()
    return MambaStateCopyFuncCalculator.mamba2_state_copy_func()
