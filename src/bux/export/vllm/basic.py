"""The embedding, the head, the dense MLP and the elementwise layers, in vLLM.

Under tensor parallelism the embedding and the head split the vocabulary, an MLP its
hidden units (a column-parallel up and gate, a row-parallel down), and the other
layers are whole on every rank.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .common import ACTIVATIONS, Ctx, Layer, Site, unsupported


class Embedding(Layer):
    """bux's ``embed``: the token table, split by vocabulary, times sqrt(d) if asked,
    plus learned positions. The model's first is the one tied heads read, and the one
    vLLM's ``inputs_embeds`` stand for."""

    bux_types = ("embed",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        unsupported(c, "norm")
        vocab, d = site.config.vocab_size, site.config.hidden_size
        self.tokens = VocabParallelEmbedding(vocab, d, prefix=f"{site.path}.tokens")
        most = c["max_positions"]
        self.positions = (
            nn.Parameter(torch.empty(most, d), requires_grad=False) if most else None
        )
        self.multiplier = d**0.5 if c["scale_by_sqrt_dim"] else 1.0

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out = {"tokens.weight": (self.tokens.weight, ())}
        if self.positions is not None:
            out["positions.weight"] = (self.positions, ())
        return out

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The tokens' rows of the table, times the multiplier."""
        x = self.tokens(ids)
        return x if self.multiplier == 1.0 else x * self.multiplier

    def forward(self, ids: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        first = ctx.embedding is None
        ctx.embedding = ctx.embedding or self
        given = ctx.inputs_embeds if first else None
        x = self.embed(ids) if given is None else given
        return x if self.positions is None else x + self.positions[ctx.positions]


class Head(Layer):
    """bux's ``head``: ``softcap(x W + b)``, split by vocabulary. The model's first
    head is its output: the model returns its input, and vLLM takes its logits
    (`logits`) of the tokens it samples."""

    bux_types = ("head",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        vocab = site.config.vocab_size
        self.tied = c["tied"]
        lm_head = ParallelLMHead(
            vocab, width, bias=c["bias"], prefix=f"{site.path}.lm_head"
        )
        if self.tied:
            assert site.embedding is not None, "a tied head needs an embedding before"
            lm_head = lm_head.tie_weights(site.embedding.tokens)
        self.lm_head = lm_head
        self.processor = LogitsProcessor(vocab, soft_cap=c["softcap"])

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out = {} if self.tied else {"proj.weight": (self.lm_head.weight, ())}
        if self.lm_head.bias is not None:
            out["bias" if self.tied else "proj.bias"] = (self.lm_head.bias, ())
        return out

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        if ctx.hidden is None:
            ctx.hidden = x
        return x

    def logits(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """The logits of `hidden`, the head's input, gathered over the ranks."""
        return self.processor(self.lm_head, hidden, self.lm_head.bias)


class MLP(Layer):
    """bux's ``mlp`` and ``gated_mlp``: ``down(act(up x))``, or
    ``down(act(gate x) * up x)``, gate and up as one column-parallel projection.

    Args:
        size: Hidden units in place of the config's, as an MoE's shared experts take
            them together.
    """

    bux_types = ("mlp", "gated_mlp")

    def __init__(
        self, c: dict, width: int, site: Site | None = None, size: int | None = None
    ):
        super().__init__()
        unsupported(c, "limit", "up_softcap", "up_offset")
        n, bias = size or c["intermediate_size"], c["bias"]
        self.act = ACTIVATIONS[c["activation"]]
        self.gated = c["type"] == "gated_mlp"
        prefix = site.path if site is not None else ""
        if self.gated:
            self.up: nn.Module = MergedColumnParallelLinear(
                width, [n, n], bias=bias, return_bias=False, prefix=f"{prefix}.up"
            )
        else:
            self.up = ColumnParallelLinear(
                width, n, bias=bias, return_bias=False, prefix=f"{prefix}.up"
            )
        self.down = RowParallelLinear(
            n, width, bias=bias, return_bias=False, prefix=f"{prefix}.down"
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {}
        for kind in ("weight", "bias"):
            up, down = getattr(self.up, kind), getattr(self.down, kind)
            if up is None:
                continue
            if self.gated:  # gate is the first half of the fused projection
                out[f"gate.{kind}"] = (up, (0,))
                out[f"up.{kind}"] = (up, (1,))
            else:
                out[f"up.{kind}"] = (up, ())
            out[f"down.{kind}"] = (down, ())
        return out

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The hidden units of x: ``act(up x)``, or ``act(gate x) * up x``."""
        h = self.up(x)
        if not self.gated:
            return self.act(h)
        gate, up = h.chunk(2, -1)
        return self.act(gate) * up

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.down(self.hidden(x))


class Linear(Layer):
    """bux's ``linear``: ``x W + b``, whole on every rank."""

    bux_types = ("linear",)

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.proj = ReplicatedLinear(
            width,
            c["features"],
            bias=c["bias"],
            return_bias=False,
            prefix=f"{site.path}.proj",
        )

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.proj(x)


class Act(Layer):
    """bux's ``act``: an activation, elementwise."""

    bux_types = ("act",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        self.fn = ACTIVATIONS[c["fn"]]

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return self.fn(x)


class Scale(Layer):
    """bux's ``scale``: a learned multiplier, one or one per channel."""

    bux_types = ("scale",)

    def __init__(self, c: dict, width: int, site: Site | None = None):
        super().__init__()
        shape = (width,) if c["per_channel"] else ()
        self.value = nn.Parameter(torch.empty(shape), requires_grad=False)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return x * self.value


class Dropout(Layer):
    """bux's ``dropout``: nothing, at inference."""

    bux_types = ("dropout",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        return x
