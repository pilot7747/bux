"""How an MoE layer routes its tokens, and its experts' weights as vLLM's fused
kernels take them: plain PyTorch, which `bux_vllm.moe` hands vLLM.

The router runs as bux's does (`bux.architecture.experts.moe`): logits in float32,
scored by a softmax, a sigmoid or ``sqrt(softplus)``, the aux-free bias added to the
scores that pick, groups limited by their best two experts' sum, the top-k weights
normalized (with the shared experts' scores, where the router scores them) and
scaled.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Router(nn.Module):
    """bux's routing of one MoE layer, as vLLM's custom routing function: returns
    each token's top-k experts and their weights from the router's logits."""

    def __init__(self, c: dict):
        super().__init__()
        self.c = c
        by_bias = c["load_balancing"] != "aux_loss"  # aux_free, smebu
        bias = torch.zeros(c["num_experts"], dtype=torch.float32)
        self.bias = nn.Parameter(bias, requires_grad=False) if by_bias else None

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
        weights, idx, _ = self.route(gating_output, topk)
        return weights, idx

    def route(
        self, logits: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each token's top-k experts and their weights, from the router's logits,
        and the weights of the shared experts that the router scores too, last,
        under ``shared_expert_sink`` (none, otherwise): normalized together and
        scaled, in float32."""
        n = self.c["num_experts"]
        scores = self.scores(logits)
        scores, sinks = scores[..., :n], scores[..., n:]
        choice = scores if self.bias is None else scores + self.bias
        if self.c["num_groups"] > 1:
            choice = self.limit_groups(choice)
        idx = choice.topk(k, -1).indices
        weights = scores.gather(-1, idx)
        if self.c["normalize_top_k"]:
            total = weights.sum(-1, keepdim=True) + sinks.sum(-1, keepdim=True)
            weights, sinks = weights / total, sinks / total
        scale = self.c["routed_scaling_factor"]
        return weights * scale, idx, sinks * scale


def expert_shards(
    name: str, t: torch.Tensor, gated: bool
) -> list[tuple[str, torch.Tensor]] | None:
    """The routed experts' tensor `name` as vLLM's shards, stacked by expert in its
    (out, in) layout (w1 the gate, w3 up, w2 down); None for another weight.

    It reads bux's layout, ``up`` (E, d, f, p) with each hidden unit's up and gate
    side by side and ``down`` (E, f, d), and transformers', the export's
    (`bux.export.hf.moe`): ``experts.gate_up_proj`` (E, 2f, d), the gate first, or
    ``experts.up_proj`` (E, f, d), and ``experts.down_proj`` (E, d, f); and their
    biases alike."""
    match name.removeprefix("experts."):
        case "up" | "up_bias":
            up = t[..., 0].transpose(1, 2) if name == "up" else t[..., 0]
            if not gated:
                return [("w1", up)]
            gate = t[..., 1].transpose(1, 2) if name == "up" else t[..., 1]
            return [("w1", gate), ("w3", up)]
        case "down":
            return [("w2", t.transpose(1, 2))]
        case "down_bias" | "down_proj" | "down_proj_bias":
            return [("w2", t)]
        case "up_proj" | "up_proj_bias":
            return [("w1", t)]
        case "gate_up_proj" | "gate_up_proj_bias":
            gate, up = t.chunk(2, 1)
            return [("w1", gate), ("w3", up)]
    return None
