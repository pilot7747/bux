"""The embedding, the value embeddings, the head, the dense MLP and the elementwise
layers, exported."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import ACTIVATIONS, Ctx, Norm, Site


class TokenEmbedding(nn.Module):
    """bux's ``embed``: the token table, times sqrt(d) if asked, normalized if it has
    a norm, plus learned positions. The model's first is the one tied heads read;
    dropout, a training setting, is left out."""

    bux_types = ("embed",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        vocab, d = site.config.vocab_size, site.config.hidden_size
        self.tokens = nn.Embedding(vocab, d)
        most = c["max_positions"]
        self.positions = nn.Embedding(most, d) if most else None
        self.multiplier = d**0.5 if c["scale_by_sqrt_dim"] else 1.0
        self.norm = Norm(c["norm"], d) if c["norm"] else None

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """The tokens' rows, scaled and normalized: what nextn reads."""
        x = self.tokens(ids) * self.multiplier
        return x if self.norm is None else self.norm(x)

    def forward(self, ids: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        ctx.embedding = ctx.embedding or self
        x = self.embed_tokens(ids)
        return x if self.positions is None else x + self.positions(ctx.positions)


class ValueEmbedding(nn.Module):
    """bux's ``value_embed``: a row per token, pushed to its ``ve`` port for a
    ``gqa`` layer to add to its values; the stream goes on as it is."""

    bux_types = ("value_embed",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        self.table = nn.Embedding(site.config.vocab_size, c["features"])
        self.taps = [tap for name, tap, mode in site.ports if mode == "write"]

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        rows = self.table(ctx.tokens)
        for tap in self.taps:
            ctx.push(tap, rows)
        return x


class Head(nn.Module):
    """bux's ``head``: records ``softcap(x W + b)`` as an output, and passes x on."""

    bux_types = ("head",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        vocab = site.config.vocab_size
        self.tied, self.softcap = c["tied"], c["softcap"]
        self.proj = None if self.tied else nn.Linear(width, vocab, bias=c["bias"])
        tied_bias = self.tied and c["bias"]
        self.bias = nn.Parameter(torch.zeros(vocab)) if tied_bias else None

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        if self.proj is None:
            assert ctx.embedding is not None, "a tied head needs an embedding before it"
            logits = F.linear(x, ctx.embedding.tokens.weight, self.bias)
        else:
            logits = self.proj(x)
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        ctx.heads.append(logits)
        return x


def activate(c: dict, up: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """An MLP's hidden units, as bux's ``activate``: ``act(up)``, or
    ``act(gate) * (up + up_offset)`` when gated; `limit` clips both (the gate from
    above only), and `up_softcap` caps up."""
    limit, cap = c["limit"], c["up_softcap"]
    if limit is not None:
        up = up.clamp(-limit, limit)
    if cap is not None:
        up = cap * torch.tanh(up / cap)
    act = ACTIVATIONS[c["activation"]]
    if gate is None:
        return act(up)
    if limit is not None:
        gate = gate.clamp(max=limit)
    return act(gate) * (up + c["up_offset"])


class MLP(nn.Module):
    """bux's ``mlp`` and ``gated_mlp``: ``down(act(up x))``, or
    ``down(act(gate x) * up x)``, with the clip, cap and offset of `activate`; dropout,
    a training setting, is left out. An MoE's shared experts are one, of all their
    hidden units (`size`)."""

    bux_types = ("mlp", "gated_mlp")

    def __init__(self, c: dict, width: int, site: Site | None = None, size: int = 0):
        super().__init__()
        n, bias = size or c["intermediate_size"], c["bias"]
        self.config = c
        self.up = nn.Linear(width, n, bias=bias)
        gated = c["type"] == "gated_mlp"
        self.gate = nn.Linear(width, n, bias=bias) if gated else None
        self.down = nn.Linear(n, width, bias=bias)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The hidden units the down projection takes."""
        gate = None if self.gate is None else self.gate(x)
        return activate(self.config, self.up(x), gate)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.down(self.hidden(x))


class Linear(nn.Module):
    """bux's ``linear``: ``x W + b``."""

    bux_types = ("linear",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        self.proj = nn.Linear(width, c["features"], bias=c["bias"])

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.proj(x)


class Act(nn.Module):
    """bux's ``act``: an activation, elementwise."""

    bux_types = ("act",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        self.fn = ACTIVATIONS[c["fn"]]

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.fn(x)


class Scale(nn.Module):
    """bux's ``scale``: a learned multiplier, one or one per channel."""

    bux_types = ("scale",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        shape = (width,) if c["per_channel"] else ()
        self.value = nn.Parameter(torch.full(shape, float(c["init"])))

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return x * self.value


class DyT(nn.Module):
    """bux's ``dyt``: ``weight * tanh(alpha * x) + bias``, a norm without
    statistics."""

    bux_types = ("dyt",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(c["alpha_init"])))
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return torch.tanh(self.alpha * x) * self.weight + self.bias


class Dropout(nn.Module):
    """bux's ``dropout``: nothing, at inference."""

    bux_types = ("dropout",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return x
