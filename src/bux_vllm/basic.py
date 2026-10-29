"""The embedding, the value embeddings, the head, the dense MLP and the elementwise
layers, in vLLM.

Under tensor parallelism the embeddings and the head split the vocabulary, an MLP its
hidden units (a column-parallel up and gate, a row-parallel down), and the other
layers are whole on every rank. Under vLLM's online FP8 weights the MLPs' and
``linear``'s projections take the engine's quantization config
(`bux_vllm.common.quantization`); the embeddings and the head stay in the model's
dtype, as vLLM keeps its own models'.
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

from . import copying
from .common import ACTIVATIONS, Ctx, Layer, Norm, Site, unsupported


class Embedding(Layer):
    """bux's ``embed``: the token table, split by vocabulary, times sqrt(d) if asked
    and normalized if asked, plus learned positions. The model's first is the one
    tied heads read, and the one vLLM's ``inputs_embeds`` stand for."""

    bux_types = ("embed",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        vocab, d = site.config.vocab_size, site.config.hidden_size
        self.tokens = VocabParallelEmbedding(vocab, d, prefix=f"{site.path}.tokens")
        most = c["max_positions"]
        self.positions = (
            nn.Parameter(torch.empty(most, d), requires_grad=False) if most else None
        )
        self.multiplier = d**0.5 if c["scale_by_sqrt_dim"] else 1.0
        self.norm = Norm(c["norm"], d) if c["norm"] else None

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out = {"tokens.weight": (self.tokens.weight, ())}
        if self.positions is not None:
            out["positions.weight"] = (self.positions, ())
        if self.norm is not None:
            out |= {f"norm.{k}": v for k, v in self.norm.targets().items()}
        return out

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The tokens' rows of the table, times the multiplier, normalized."""
        x = self.tokens(ids)
        x = x if self.multiplier == 1.0 else x * self.multiplier
        return x if self.norm is None else self.norm(x)

    def forward(self, ids: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        given = None
        if ctx.embedding is None:  # the model's first: vLLM's inputs_embeds are its
            ctx.embedding, given = self, ctx.inputs_embeds
        x = self.embed(ids) if given is None else given
        return x if self.positions is None else x + self.positions[ctx.positions]


class ValueEmbedding(Layer):
    """bux's ``value_embed``: the step's tokens' rows of its table, split by
    vocabulary as the embedding's (in the model's dtype, unquantized), pushed to its
    ``ve`` port for a ``gqa`` layer to add to its values; the stream goes on as it
    is. Each rank gets every row's whole width, and keeps its kv heads' share."""

    bux_types = ("value_embed",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        vocab = site.config.vocab_size
        prefix = f"{site.path}.table"
        self.table = VocabParallelEmbedding(vocab, c["features"], prefix=prefix)
        self.taps = [tap for _, tap, mode in site.ports if mode == "write"]

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        return {"table.weight": (self.table.weight, ())}

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        assert ctx.input_ids is not None, "value_embed looks up the step's token ids"
        rows = self.table(ctx.input_ids)
        for tap in self.taps:
            ctx.push(tap, rows)
        return x


class Head(Layer):
    """bux's ``head``: ``softcap(x W + b)``, split by vocabulary. The model's first
    head is its output: the model returns its input, and vLLM takes its logits
    (`logits`) of the tokens it samples.

    With a copy pointer (CPLM), its logits are the log of the mixture of its
    softmax and the pointer (`bux_vllm.copying.mixed`), which its forward computes
    for each request's last token of the step (`bux_vllm.pointer`): its queries and
    keys whole on every rank, in the model's dtype."""

    bux_types = ("head",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        vocab = site.config.vocab_size
        self.pointer = None
        if c["pointer"] is not None:
            self.pointer = Pointer(c["pointer"], width, vocab, site)
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
        if self.pointer is not None:
            out |= {f"pointer.{k}": v for k, v in self.pointer.targets().items()}
        return out

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        if ctx.hidden is None:
            ctx.hidden = x
            if self.pointer is not None:
                self.pointer(x, ctx)
        return x

    def logits(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """The logits of `hidden`, the head's input, gathered over the ranks: with a
        copy pointer, the log of its mixture, of the requests' rows the forward
        pointed for, in order, as vLLM samples one a request."""
        logits = self.processor(self.lm_head, hidden, self.lm_head.bias)
        if self.pointer is None or logits is None:
            return logits
        return self.pointer.mixed(logits)


class Pointer(Layer):
    """A head's copy pointer: its query and key projections, the query's gain, the
    sink's key, its paged cache (`bux_vllm.pointer.PointerCache`), and the buffers
    of each request's p_copy and a_sink, which its forward fills and `mixed` reads.

    Raises:
        NotImplementedError: A ``window``, which the cache's op does not cap yet, or
            speculative decoding, which samples more than one row a request.
    """

    p_copy: torch.Tensor
    a_sink: torch.Tensor

    def __init__(self, c: dict, width: int, vocab: int, site: Site):
        super().__init__()
        from .pointer import PointerCache

        unsupported(c, "window")
        if site.vllm_config.num_speculative_tokens:
            raise NotImplementedError("head: a copy pointer with speculative decoding")
        dim, self.dim = c["dim"], c["dim"]
        self.copy = vocab - 1 if c["token"] is None else c["token"]
        self.q = ReplicatedLinear(width, dim, bias=False, return_bias=False)
        self.k = ReplicatedLinear(width, dim, bias=False, return_bias=False)
        self.gain = nn.Parameter(torch.ones(()), requires_grad=False)
        self.sink = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.cache = PointerCache(dim, site.vllm_config, site.caches[0])
        rows = site.vllm_config.scheduler_config.max_num_seqs
        copy = torch.zeros(rows, vocab, dtype=torch.float32)
        self.register_buffer("p_copy", copy, persistent=False)
        self.register_buffer("a_sink", torch.ones(rows), persistent=False)

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        return {
            "q.weight": (self.q.weight, ()),
            "k.weight": (self.k.weight, ()),
            "gain": (self.gain, ()),
            "sink": (self.sink, ()),
        }

    def forward(self, x: torch.Tensor, ctx: Ctx) -> None:
        """Point for the step's tokens `x`, the head's input: write their keys and
        tokens, and each request's p_copy and a_sink for its last token."""
        assert ctx.input_ids is not None, "a copy pointer copies the step's tokens"
        q = _rms_normed(self.q(x)) * self.gain.to(x.dtype)
        k = _rms_normed(self.k(x))
        sink = (q.float() @ self.sink.float()) * self.dim**-0.5
        torch.ops.vllm.bux_copy_pointer(
            q, k, ctx.input_ids, sink, self.p_copy, self.a_sink, self.cache.prefix
        )

    def mixed(self, logits: torch.Tensor) -> torch.Tensor:
        """The log of the mixture, of the head's `logits` of each request's row."""
        n = logits.shape[0]
        mix = copying.mixed(logits, self.p_copy[:n], self.a_sink[:n], self.copy)
        return mix.to(logits.dtype)


def _rms_normed(x: torch.Tensor) -> torch.Tensor:
    """x over its RMS, in float32, as bux's pointer normalizes its queries and keys,
    an epsilon of float32's."""
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + torch.finfo(y.dtype).eps)
    return y.to(x.dtype)


class MLP(Layer):
    """bux's ``mlp`` and ``gated_mlp``: ``down(act(up x))``, or
    ``down(act(gate x) * (up x + up_offset))``, gate and up as one column-parallel
    projection; `limit` clips both (the gate from above only), and `up_softcap`
    caps up, as `bux.architecture.layers.mlp.activate` does.

    Args:
        size: Hidden units in place of the config's, as an MoE's shared experts take
            them together.
    """

    bux_types = ("mlp", "gated_mlp")

    def __init__(
        self, c: dict, width: int, site: Site | None = None, size: int | None = None
    ):
        super().__init__()
        n, bias = size or c["intermediate_size"], c["bias"]
        self.act = ACTIVATIONS[c["activation"]]
        self.limit, self.cap, self.offset = c["limit"], c["up_softcap"], c["up_offset"]
        self.gated = c["type"] == "gated_mlp"
        prefix = site.path if site is not None else ""
        kw = {"bias": bias, "return_bias": False}
        kw["quant_config"] = site.quant if site is not None else None
        if self.gated:
            self.up: nn.Module = MergedColumnParallelLinear(
                width, [n, n], prefix=f"{prefix}.up", **kw
            )
        else:
            self.up = ColumnParallelLinear(width, n, prefix=f"{prefix}.up", **kw)
        self.down = RowParallelLinear(n, width, prefix=f"{prefix}.down", **kw)

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
        """The hidden units of x: ``act(up x)``, or ``act(gate x) * up x``, clipped,
        capped and offset."""
        h = self.up(x)
        gate, up = h.chunk(2, -1) if self.gated else (None, h)
        if self.limit is not None:
            up = up.clamp(-self.limit, self.limit)
        if self.cap is not None:
            up = self.cap * torch.tanh(up / self.cap)
        if gate is None:
            return self.act(up)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
        return self.act(gate) * (up + self.offset)

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
            quant_config=site.quant,
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
