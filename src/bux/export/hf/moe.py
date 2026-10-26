"""Mixtures of experts, exported: bux's router, and its routed experts through
transformers' experts interface (``eager``, ``batched_mm``, ``grouped_mm``).

The routed experts' weights are stacked as transformers' MoE models keep theirs:
``gate_up_proj`` (E, 2f, d), the gate's rows above the up's, and ``down_proj``
(E, d, f), so that its grouped kernels run them (``experts_implementation``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.integrations.moe import use_experts_implementation

from .basic import MLP, activate
from .common import Ctx, Norm, Site

#: How the router scores the experts from its logits (`bux.config.MoEConfig.router`).
SCORES = {
    "softmax": lambda x: x.softmax(-1),
    "sigmoid": torch.sigmoid,
    "sqrtsoftplus": lambda x: F.softplus(x).sqrt(),
}


def top_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """The indices of the k largest entries of x's last dimension, the lower index
    first among equal ones, as ``jax.lax.top_k`` picks them."""
    return x.sort(stable=True, dim=-1, descending=True).indices[..., :k]


def limit_groups(choice: torch.Tensor, groups: int, keep: int) -> torch.Tensor:
    """The scores at -inf outside the `keep` best of `groups` groups of experts, each
    group scored by the sum of its best two (DeepSeek-V3)."""
    *lead, n = choice.shape
    grouped = choice.reshape(*lead, groups, n // groups)
    best = grouped.topk(min(2, n // groups), -1).values.sum(-1)
    kept = torch.zeros_like(best, dtype=torch.bool).scatter(-1, top_k(best, keep), True)
    return grouped.masked_fill(~kept[..., None], -torch.inf).reshape(*lead, n)


def _norm(c: dict | None, width: int) -> Norm | None:
    return None if c is None else Norm(c, width)


class Experts(nn.Module):
    """The routed experts of bux's ``moe``, stacked as transformers' are: an FFN each,
    ``mlp`` or ``gated_mlp`` with its biases, clip, cap and offset (`activate`), then
    the expert norm. Called as ``experts(x, top_k_index, top_k_weights)`` on n rows,
    it returns each row's sum of its experts' outputs times their weights; this is
    the eager path, and transformers' kernels replace it.

    Args:
        config: The model's config, whose ``_experts_implementation`` picks the path.
        c: The experts' FFN (`bux.config.MLPConfig`), as JSON.
        num_experts: E.
        width: The width the experts read and write, d (in LatentMoE, the latent's).
        norm: The norm of each expert's output, as JSON; None for none.
    """

    def __init__(
        self, config: Any, c: dict, num_experts: int, width: int, norm: dict | None
    ):
        super().__init__()
        n, f, gated = num_experts, c["intermediate_size"], c["type"] == "gated_mlp"
        self.ffn, self.num_experts = c, n
        up = "gate_up_proj" if gated else "up_proj"
        rows = 2 * f if gated else f
        setattr(self, up, nn.Parameter(torch.zeros(n, rows, width)))
        self.down_proj = nn.Parameter(torch.zeros(n, width, f))
        if c["bias"]:
            setattr(self, f"{up}_bias", nn.Parameter(torch.zeros(n, rows)))
            self.down_proj_bias = nn.Parameter(torch.zeros(n, width))
        self.post_expert_norm = _norm(norm, width)

    def configure(self) -> Experts:
        """Set what transformers' kernels read of the experts, which its decorator
        sets to its defaults as it builds them; return them."""
        self.has_gate = self.ffn["type"] == "gated_mlp"
        self.has_bias = self.ffn["bias"]
        self.has_post_expert_norm = self.post_expert_norm is not None
        return self

    def act_fn(self, up: torch.Tensor) -> torch.Tensor:
        """An ungated expert's hidden units from its up projection."""
        return activate(self.ffn, up, None)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """A gated expert's hidden units from its gate and up projections, in one."""
        gate, up = gate_up.chunk(2, -1)
        return activate(self.ffn, up, gate)

    def expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """The output of expert `e` on the rows x."""
        up = "gate_up_proj" if self.has_gate else "up_proj"
        h = F.linear(x, getattr(self, up)[e], self._bias(f"{up}_bias", e))
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)
        y = F.linear(h, self.down_proj[e], self._bias("down_proj_bias", e))
        return y if self.post_expert_norm is None else self.post_expert_norm(y)

    def _bias(self, name: str, e: int) -> torch.Tensor | None:
        return getattr(self, name)[e] if self.has_bias else None

    def forward(
        self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        for e in top_k_index.unique().tolist():
            row, pick = torch.where(top_k_index == e)
            y = self.expert(x[row], e) * top_k_weights[row, pick, None]
            out.index_add_(0, row, y.to(out.dtype))
        return out


# In place: the forward runs by ``config._experts_implementation``.
use_experts_implementation(Experts)


class MoE(nn.Module):
    """bux's ``moe``: the router picks each token's top-k experts, as bux does, and
    their outputs add up by its weights; around them, shared experts and LatentMoE.

    Scores and the top-k choice are in float32, the router's logits from x and its
    kernel in x's dtype, as bux makes them; equal scores pick the lower expert, as
    bux's top-k does. Settings that only matter in training are left out: the
    balance loss (``aux_loss_coef``, ``aux_loss_scope``), how the load-balancing bias
    moves (``bias_update_rate``, ``center_bias``, SMEBU's ``bias_softclamp`` and
    ``bias_momentum``; the bias itself is a buffer, ``router_bias``, which ``aux_free``
    and ``smebu`` both add to the scores that choose), ``random_logits``, and dropout.
    """

    bux_types = ("moe",)
    router_bias: torch.Tensor | None

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.config, n = c, c["num_experts"]
        sinks = c["num_shared_experts"] if c["shared_expert_sink"] else 0
        self.router = nn.Linear(width, n + sinks, bias=c["router_bias"])
        by_bias = c["load_balancing"] != "aux_loss"  # aux_free, smebu
        self.register_buffer("router_bias", torch.zeros(n) if by_bias else None)
        latent = c["latent_size"] or width
        lat = c["latent_size"] is not None
        self.to_latent = nn.Linear(width, latent, bias=False) if lat else None
        self.from_latent = nn.Linear(latent, width, bias=False) if lat else None
        self.latent_input_norm = _norm(c["latent_input_norm"], latent)
        self.latent_output_norm = _norm(c["latent_output_norm"], latent)
        expert, norm = c["expert"], c["expert_norm"]
        self.experts = Experts(site.config, expert, n, latent, norm).configure()
        self.shared = self._shared_experts(c, width)
        gate = c["shared_expert_gate"]
        self.shared_gate = nn.Linear(width, 1, bias=False) if gate else None

    @staticmethod
    def _shared_experts(c: dict, width: int) -> MLP | None:
        """The shared experts, one MLP of all their hidden units, or None."""
        if not c["num_shared_experts"]:
            return None
        shared = c["shared_expert"] or c["expert"]
        size = shared["intermediate_size"] * c["num_shared_experts"]
        return MLP(shared, width, size=size)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        idx, weights, sinks = self.route(x)
        z = x if self.to_latent is None else self.to_latent(x)
        if self.latent_input_norm is not None:
            z = self.latent_input_norm(z)
        rows = z.reshape(-1, z.shape[-1])
        k = idx.shape[-1]
        y = self.experts(rows, idx.reshape(-1, k), weights.reshape(-1, k))
        y = y.reshape(z.shape)
        if self.latent_output_norm is not None:
            y = self.latent_output_norm(y)
        if self.from_latent is not None:
            y = self.from_latent(y)
        return y if self.shared is None else y + self._shared(x, sinks)

    def route(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Each token's top-k experts and their weights, and the shared experts'
        weights if the router scores them too, as bux's ``MoE._route``."""
        c, n = self.config, self.config["num_experts"]
        scores = SCORES[c["router"]](self._logits(x) * c["router_logit_scale"])
        scores, sinks = scores[..., :n], scores[..., n:]
        choice = (
            scores if self.router_bias is None else scores + self.router_bias.float()
        )
        if c["num_groups"] > 1:
            choice = limit_groups(choice, c["num_groups"], c["top_k_groups"])
        idx = top_k(choice, c["top_k"])
        weights = scores.gather(-1, idx)
        if c["normalize_top_k"]:
            total = weights.sum(-1, keepdim=True) + sinks.sum(-1, keepdim=True)
            weights, sinks = weights / total, sinks / total
        scale = c["routed_scaling_factor"]
        weights, sinks = (w.to(x.dtype) * scale for w in (weights, sinks))
        return idx, weights, sinks if sinks.shape[-1] else None

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """The router's logits in float32, of x and its kernel in x's dtype, whose
        products float32 holds exactly."""
        kernel = self.router.weight.to(x.dtype).float()
        bias = self.router.bias
        return F.linear(x.float(), kernel, None if bias is None else bias.float())

    def _shared(self, x: torch.Tensor, sinks: torch.Tensor | None) -> torch.Tensor:
        """The shared experts' output; where the router scores them, each one's share
        of the hidden units weighted by its score."""
        assert self.shared is not None
        if sinks is None:
            y = self.shared(x)
        else:
            h = self.shared.hidden(x)
            h = h * sinks.repeat_interleave(h.shape[-1] // sinks.shape[-1], -1)
            y = self.shared.down(h)
        if self.shared_gate is not None:
            y = y * torch.sigmoid(self.shared_gate(x))
        return y

    @classmethod
    def bux_weights(cls, tensors: dict, c: dict) -> dict:
        """Restack bux's routed experts as transformers': up (E, d, f, p), its up
        and gate side by side, to ``gate_up_proj`` (E, 2f, d), the gate first, or
        ``up_proj`` (E, f, d); down (E, f, d) to ``down_proj`` (E, d, f); and their
        biases and the expert norm alike."""
        out = dict(tensors)
        up, down = out.pop("up"), out.pop("down")
        gated = c["expert"]["type"] == "gated_mlp"
        name = "experts.gate_up_proj" if gated else "experts.up_proj"
        out[name] = _gate_first(up).transpose(0, 2, 1)
        out["experts.down_proj"] = down.transpose(0, 2, 1)
        if "up_bias" in out:
            out[f"{name}_bias"] = _gate_first(out.pop("up_bias"))
            out["experts.down_proj_bias"] = out.pop("down_bias")
        for key in [k for k in out if k.startswith("expert_norm.")]:
            out[key.replace("expert_norm.", "experts.post_expert_norm.")] = out.pop(key)
        return out


def _gate_first(a: np.ndarray) -> np.ndarray:
    """bux's projections side by side in the last axis, up then gate, as one of the
    gate's columns then the up's: (..., f, p) to (..., p f)."""
    return np.concatenate([a[..., i] for i in reversed(range(a.shape[-1]))], -1)
