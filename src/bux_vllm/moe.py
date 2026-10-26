"""Mixtures of experts in vLLM: bux's ``moe``, its routed experts by vLLM's fused
MoE kernels (`FusedMoEFactory`), routed by bux's own rule.

The router runs as bux's does (`bux_vllm.routing`): vLLM takes the picks and
weights from it (``custom_routing_function``) and runs the experts.
Under tensor parallelism each rank keeps every expert's share of the hidden units,
as vLLM splits them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import ReplicatedLinear

from .basic import MLP
from .common import Ctx, Layer, Site, unsupported
from .routing import Router, expert_shards

#: vLLM's MoE activations, by a bux expert's type and activation, without a limit
#: or an offset.
ACTIVATIONS = {
    ("gated_mlp", "swish"): "silu",
    ("gated_mlp", "gelu"): "gelu",
    ("gated_mlp", "gelu_tanh"): "gelu_tanh",
    ("gated_mlp", "relu2"): "relu2",
    ("mlp", "swish"): "silu_no_mul",
    ("mlp", "gelu"): "gelu_no_mul",
    ("mlp", "gelu_tanh"): "gelu_tanh_no_mul",
    ("mlp", "relu2"): "relu2_no_mul",
}

#: The gate's activation ``x sigmoid(alpha x)``, by its name, of vLLM's clamped
#: SwiGLU: ``act(min(gate, limit)) * (clip(up, limit) + offset)``.
SIGMOID_GATES = {"swish": 1.0, "quick_gelu": 1.702}
FLOAT32_MAX = 3.4028234e38  # a limit that clips nothing


def experts_activation(expert: dict) -> dict:
    """vLLM's activation of the routed experts `expert` (an MLP's config), with its
    clamp and offset, as `FusedMoEFactory`'s arguments.

    Raises:
        NotImplementedError: vLLM has no such activation.
    """
    unsupported(expert, "up_softcap")
    kind = (expert["type"], expert["activation"])
    clamped = expert["limit"] is not None or expert["up_offset"]
    if not clamped and kind in ACTIVATIONS:
        return {"activation": ACTIVATIONS[kind]}
    if kind[0] != "gated_mlp" or kind[1] not in SIGMOID_GATES:
        raise NotImplementedError(f"moe: {kind} experts, so clamped, not in vLLM yet")
    limit = FLOAT32_MAX if expert["limit"] is None else expert["limit"]
    return {
        "activation": "swigluoai_uninterleave",  # gate then up, as w13 holds them
        "swiglu_limit": limit,
        "swiglu_alpha": SIGMOID_GATES[kind[1]],
        "swiglu_beta": expert["up_offset"],
    }


class MoE(Layer):
    """bux's ``moe``: the router (bux's rule, `Router`), the routed experts by vLLM's
    fused kernels, and the shared experts as one MLP, gated by a sigmoid under
    ``shared_expert_gate``. Training's settings (the balance loss, the bias's
    update, random logits) change nothing here."""

    bux_types = ("moe",)
    once = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        unsupported(
            c,
            "shared_expert_sink",
            "latent_size",
            "latent_input_norm",
            "latent_output_norm",
            "expert_norm",
        )
        expert = c["expert"]
        n, self.gated = c["num_experts"], expert["type"] == "gated_mlp"
        self.router = Router(c)
        weight = torch.empty(n, width)
        self.router_weight = nn.Parameter(weight, requires_grad=False)
        self.router_proj_bias = (
            nn.Parameter(torch.empty(n), requires_grad=False)
            if c["router_bias"]
            else None
        )
        self.experts = FusedMoEFactory(
            num_experts=n,
            top_k=c["top_k"],
            hidden_size=width,
            intermediate_size=expert["intermediate_size"],
            renormalize=c["normalize_top_k"],
            prefix=f"{site.path}.experts",
            custom_routing_function=self.router,
            **experts_activation(expert),
            has_bias=expert["bias"],
            router_logits_dtype=torch.float32,
            # vLLM's names of the shards, which a non-gated MoE has no up of
            ckpt_names=("gate_proj", "down_proj", "up_proj" if self.gated else None),
        )
        self.shared = None
        if c["num_shared_experts"]:
            shared = c["shared_expert"] or expert
            size = shared["intermediate_size"] * c["num_shared_experts"]
            inner = Site(f"{site.path}.shared", site.vllm_config, site.config)
            self.shared = MLP(shared, width, inner, size=size)
        self.shared_gate = (
            ReplicatedLinear(width, 1, bias=False, return_bias=False)
            if c["shared_expert_gate"]
            else None
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out = {"router.weight": (self.router_weight, ())}
        if self.router_proj_bias is not None:
            out["router.bias"] = (self.router_proj_bias, ())
        if self.router.bias is not None:
            out["router_bias"] = (self.router.bias, ())
        if self.shared is not None:
            out |= {f"shared.{k}": v for k, v in self.shared.targets().items()}
        if self.shared_gate is not None:
            out["shared_gate.weight"] = (self.shared_gate.weight, ())
        return out

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        shards = expert_shards(name, tensor, self.gated)
        if shards is None:
            return super().load(name, tensor)
        routed = self.experts.routed_experts
        bias = name.endswith("bias")
        stem = "w2" if shards[0][0] == "w2" else "w13"
        param = getattr(routed, f"{stem}_{'bias' if bias else 'weight'}")
        for shard, stacked in shards:
            for e in range(stacked.shape[0]):
                where = f"experts.{e}.{shard}.{'bias' if bias else 'weight'}"
                routed.weight_loader(param, stacked[e].contiguous(), where, shard, e)
        return [param]

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        logits = F.linear(x.float(), self.router_weight.float())
        if self.router_proj_bias is not None:
            logits = logits + self.router_proj_bias.float()
        y = self.experts(x, logits)
        if self.shared is not None:
            shared = self.shared(x)
            if self.shared_gate is not None:
                shared = shared * torch.sigmoid(self.shared_gate(x))
            y = y + shared
        return y
