"""The embedding, the head, the dense MLP and the elementwise layers, exported."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import ACTIVATIONS, Ctx, Site, unsupported


class Embedding(nn.Module):
    """bux's ``embed``: the token table, times sqrt(d) if asked, plus learned
    positions. The model's first is the one tied heads read."""

    bux_types = ("embed",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        unsupported(c, "norm")
        vocab, d = site.config.vocab_size, site.config.hidden_size
        self.tokens = nn.Embedding(vocab, d)
        most = c["max_positions"]
        self.positions = nn.Embedding(most, d) if most else None
        self.multiplier = d**0.5 if c["scale_by_sqrt_dim"] else 1.0

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the token embedding, scaled: what nextn reads."""
        return self.tokens(ids) * self.multiplier

    def forward(self, ids: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        ctx.embedding = ctx.embedding or self
        x = self.embed_tokens(ids)
        return x if self.positions is None else x + self.positions(ctx.positions)


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


class MLP(nn.Module):
    """bux's ``mlp`` and ``gated_mlp``: ``down(act(up x))``, or
    ``down(act(gate x) * up x)``."""

    bux_types = ("mlp", "gated_mlp")

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        unsupported(c, "limit", "up_softcap", "up_offset")
        n, bias = c["intermediate_size"], c["bias"]
        self.act = ACTIVATIONS[c["activation"]]
        self.up = nn.Linear(width, n, bias=bias)
        gated = c["type"] == "gated_mlp"
        self.gate = nn.Linear(width, n, bias=bias) if gated else None
        self.down = nn.Linear(n, width, bias=bias)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        up = self.up(x)
        h = self.act(up) if self.gate is None else self.act(self.gate(x)) * up
        return self.down(h)


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


class Dropout(nn.Module):
    """bux's ``dropout``: nothing, at inference."""

    bux_types = ("dropout",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return x
