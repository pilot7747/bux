"""How an MoE layer routes its tokens, and its experts' weights as vLLM's fused
kernels take them: plain PyTorch, which `bux_vllm.moe` hands vLLM.

The router runs as bux's does (`bux.architecture.experts.moe`): logits in float32,
scored by a softmax, a sigmoid or ``sqrt(softplus)``, the aux-free bias added to the
scores that pick, groups limited by their best two experts' sum, the top-k weights
normalized (with the shared experts' scores, where the router scores them) and
scaled. Zero experts are scored after the routed ones; `zero_split` computes their
output, and `given_logits` hands vLLM's fused MoE the routed experts' picks.
"""

from __future__ import annotations

from collections.abc import Callable

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
        scored = c["num_experts"] + c["num_zero_experts"]
        bias = torch.zeros(scored, dtype=torch.float32)
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
        scaled, in float32. A pick of ``num_experts`` or more is a zero expert's."""
        n = self.c["num_experts"] + self.c["num_zero_experts"]
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


def zero_split(
    z: torch.Tensor, weights: torch.Tensor, idx: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A token's picks split into the zero experts' output, each token times its zero
    picks' weights summed, and the picks the fused MoE runs: a zero expert's as the
    phantom expert ``num_experts``, of weight 0, which `phantom_map` skips.

    Args:
        z: The tokens as the experts take them, (T, d).
        weights: The router's weights of each token's picks, (T, k).
        idx: The picks, (T, k), zero experts' from `num_experts` on.
        num_experts: The routed experts.

    Returns:
        The zero experts' output, (T, d), and the fused MoE's weights and picks.
    """
    zero = idx >= num_experts
    output = z * (weights * zero).sum(-1, keepdim=True).to(z.dtype)
    return output, weights.masked_fill(zero, 0), idx.masked_fill(zero, num_experts)


def given_logits(
    weights: torch.Tensor, idx: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Logits that hand vLLM's fused MoE each token's picks (`pick_given`): their
    weights where `idx` picks a routed expert, -inf elsewhere, and -inf in the
    phantom expert's column, ``num_experts``, which the zero experts' picks become.

    Args:
        weights: The router's weights of each token's picks, (T, k), final.
        idx: The picks, (T, k), zero experts' from `num_experts` on.
        num_experts: The routed experts.

    Returns:
        The logits, (T, num_experts + 1), in float32.
    """
    shape = (idx.shape[0], num_experts + 2)  # the zero experts' picks go last
    every = torch.full(shape, -torch.inf, dtype=torch.float32, device=idx.device)
    at = torch.where(idx >= num_experts, num_experts + 1, idx.long())
    return every.scatter(-1, at, weights.float())[:, : num_experts + 1]


def pick_given(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's custom routing of `given_logits`: each token's routed picks with their
    weights, and, for each pick of a zero expert, the phantom expert, its last
    column, of weight 0."""
    values, ids = gating_output.topk(topk, -1)
    given = values.isfinite()
    phantom = gating_output.shape[-1] - 1
    return torch.where(given, values, 0.0), torch.where(given, ids, phantom)


def phantom_map(num_experts: int) -> torch.Tensor:
    """The expert map that skips the phantom expert ``num_experts``: each routed
    expert at its own slot and the phantom at -1, whose picks vLLM's Triton kernels
    then neither multiply nor sum, as they skip another rank's experts under expert
    parallelism."""
    experts = torch.arange(num_experts + 1, dtype=torch.int32)
    return torch.where(experts < num_experts, experts, -1).to(torch.int32)


def pick_rows(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's custom routing of rows that each name one expert by one-hot logits
    (`apart`): that expert, weighted 1."""
    ids = gating_output.argmax(-1, keepdim=True)
    return torch.ones(ids.shape, dtype=torch.float32, device=ids.device), ids


def apart(
    experts: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    weights: torch.Tensor,
    idx: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """The routed experts' output where each expert's is normalized before the
    router weighs it (``expert_norm``), which a fused MoE's sum over a token's
    picks leaves no room for: each pick is a row of its own, which `experts` runs
    through its one expert and returns whole, then `norm`, the weights and their
    sum.

    Args:
        experts: vLLM's fused MoE of one pick a row, routed by `pick_rows`, called
            as ``experts(rows, logits)``: under tensor parallelism it sums each
            row's output over the ranks, as the norm needs it.
        norm: The expert norm.
        z: The tokens, (T, d).
        weights: The router's weights of each token's picks, (T, k).
        idx: The picks, (T, k).
        num_experts: The routed experts, the one-hot logits' width.

    Returns:
        Each token's weighted sum of its experts' normalized outputs, (T, d), in
        z's dtype, summed in float32.
    """
    k = idx.shape[-1]
    rows = z.repeat_interleave(k, 0)
    logits = F.one_hot(idx.reshape(-1).long(), num_experts).float()
    y = norm(experts(rows, logits)).float().view(-1, k, z.shape[-1])
    return (y * weights[..., None].float()).sum(1).to(z.dtype)


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
