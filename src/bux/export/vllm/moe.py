"""Mixtures of experts in vLLM: bux's ``moe``, its routed experts by vLLM's fused
MoE kernels (`FusedMoEFactory`), routed by bux's own rule.

The router runs as bux's does: logits in float32, scored by a softmax, a sigmoid or
``sqrt(softplus)``, the aux-free bias added to the scores that pick, groups limited
by their best two experts' sum, the top-k weights normalized and scaled. vLLM takes
the picks and weights from it (``custom_routing_function``) and runs the experts.
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

#: vLLM's MoE activations, by a bux expert's type and activation.
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


class Router(nn.Module):
    """bux's routing of one MoE layer, as vLLM's custom routing function: returns
    each token's top-k experts and their weights from the router's logits."""

    def __init__(self, c: dict):
        super().__init__()
        self.c = c
        aux_free = c["load_balancing"] == "aux_free"
        bias = torch.zeros(c["num_experts"], dtype=torch.float32)
        self.bias = nn.Parameter(bias, requires_grad=False) if aux_free else None

    def scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Every expert's score of each token, from its logits, in float32."""
        logits = logits.float() * self.c["router_logit_scale"]
        if self.c["router"] == "softmax":
            return logits.softmax(-1)
        if self.c["router"] == "sigmoid":
            return logits.sigmoid()
        return F.softplus(logits).sqrt()

    def limit_groups(self, choice: torch.Tensor) -> torch.Tensor:
        """The scores at -inf outside each token's best groups of experts, a group
        scored by the sum of its best two."""
        n, groups, keep = choice.shape[0], self.c["num_groups"], self.c["top_k_groups"]
        grouped = choice.view(n, groups, -1)
        best = grouped.topk(min(2, grouped.shape[-1]), -1).values.sum(-1)
        kept = torch.zeros_like(best, dtype=torch.bool)
        kept.scatter_(-1, best.topk(keep, -1).indices, True)
        return grouped.masked_fill(~kept[..., None], -torch.inf).view(n, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scores(gating_output)
        choice = scores if self.bias is None else scores + self.bias
        if self.c["num_groups"] > 1:
            choice = self.limit_groups(choice)
        idx = choice.topk(topk, -1).indices
        weights = scores.gather(-1, idx)
        if self.c["normalize_top_k"]:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights * self.c["routed_scaling_factor"], idx


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
        unsupported(expert, "limit", "up_softcap", "up_offset")
        kind = (expert["type"], expert["activation"])
        if kind not in ACTIVATIONS:
            raise NotImplementedError(f"moe: {kind} experts not in vLLM yet")
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
            activation=ACTIVATIONS[kind],
            has_bias=expert["bias"],
            router_logits_dtype=torch.float32,
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
        if name in ("up", "down", "up_bias", "down_bias"):
            return self.load_experts(name, tensor)
        return super().load(name, tensor)

    def load_experts(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        """Load bux's stacked experts, ``up`` (E, d, f, p) with each hidden unit's up
        and gate side by side, and ``down`` (E, f, d), and their biases, into
        vLLM's w13 (gate then up) and w2, expert by expert."""
        routed = self.experts.routed_experts
        param = routed.w2_bias if name == "down_bias" else routed.w2_weight
        if name.startswith("up"):
            param = routed.w13_bias if name == "up_bias" else routed.w13_weight
        for e in range(tensor.shape[0]):
            for shard, w in self.shards(name, tensor[e]):
                where = f"experts.{e}.{shard}.{'bias' if 'bias' in name else 'weight'}"
                routed.weight_loader(param, w.contiguous(), where, shard, e)
        return [param]

    def shards(self, name: str, w: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """One expert's tensor `name` as vLLM's shards, in its (out, in) layout:
        w1 the gate, w3 up, w2 down."""
        if name == "down":
            return [("w2", w.T)]
        if name == "down_bias":
            return [("w2", w)]
        weight = name == "up"
        up = w[..., 0].T if weight else w[..., 0]
        if not self.gated:
            return [("w1", up)]
        gate = w[..., 1].T if weight else w[..., 1]
        return [("w1", gate), ("w3", up)]

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
