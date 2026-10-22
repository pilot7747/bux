"""The token embedding and the layers that read it: the LM head and nextn."""

from typing import Literal

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.functional import softcap
from bux.architecture.norm import Norm
from bux.architecture.registry import (
    NEXTN,
    OUTPUTS,
    TOKENS,
    Context,
    Taps,
    positions,
    register,
)
from bux.architecture.residual import widen, widening
from bux.lang import ir

# nnx.Embed's init.
EMBED = nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


def _table(rows: int, d: int, row: str | None, rngs: nnx.Rngs) -> nnx.Embed:
    """A table of `rows` vectors of d, its rows named `row`."""
    init = parallel.named(row, "embed", init=EMBED)
    return nnx.Embed(rows, d, embedding_init=init, rngs=rngs)


def lookup(
    table: Float[Array, "n d"], ids: Int[Array, "*ids"], row: str | None
) -> Float[Array, "*ids d"]:
    """The rows `ids` of `table`, whose rows are named `row`, gathered from the table
    whole but for its names, into rows and positions as a layer splits them: a gather
    from a table split over `fsdp` makes XLA rematerialize it in full, and into
    positions split as the stream is, under sequence parallelism, gather the table."""
    rows = jnp.take(parallel.shard(table, row, "embed"), ids, axis=0)
    return parallel.shard(rows, *("batch", "seq")[2 - ids.ndim :], None)


class Embedding(nnx.Module):
    """Token ids to the stream; with `copies`, to the stream a residual algebra widens,
    which starts as that many copies of it. The token table splits over the vocabulary
    (``bux.parallel``), and what it gives is the stream."""

    no_muon = True

    def __init__(
        self,
        c: cfg.EmbeddingConfig,
        vocab: int,
        d: int,
        rngs: nnx.Rngs,
        copies: int | None = None,
    ):
        self.tokens = _table(vocab, d, "vocab", rngs)
        self.positions = (
            _table(c.max_positions, d, None, rngs) if c.max_positions else None
        )
        self.multiplier = d**0.5 if c.scale_by_sqrt_dim else 1.0
        self.norm = Norm(c.norm, d) if c.norm is not None else None
        self.dropout = nnx.Dropout(c.dropout, rngs=rngs)
        self.copies = copies

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        """The token embedding, scaled and normalized: what nextn sees."""
        x = lookup(self.tokens.embedding[...], tokens, "vocab") * self.multiplier
        return x if self.norm is None else self.norm(x)

    def __call__(
        self, tokens: Int[Array, "b t"], taps: Taps
    ) -> Float[Array, "b t *copies d"]:
        x = self.embed_tokens(tokens)
        if self.positions is not None:
            pos = positions(taps)
            pos = jnp.arange(tokens.shape[-1]) if pos is None else pos
            x = x + lookup(self.positions.embedding[...], pos, None)
        x = self.dropout(x)
        return parallel.stream(x if self.copies is None else widen(x, self.copies))


@register("embed")
def _(
    c: cfg.EmbeddingConfig, width: ir.Width, ports: tuple[ir.Port, ...], ctx: Context
) -> Embedding:
    assert ctx.arch.vocab_size is not None
    copies = widening(ctx.arch.residual) if ctx.depth == 0 else None
    embedding = Embedding(c, ctx.arch.vocab_size, ctx.arch.d_model, ctx.rngs, copies)
    if ctx.embedding is None:
        ctx.embedding = embedding
    return embedding


class Head(nnx.Module):
    """The stream to logits, split over the vocabulary, tied to the embedding or not:
    the loss reduces over them where they are."""

    no_muon = True

    def __init__(self, c: cfg.LMHeadConfig, width: int, ctx: Context):
        vocab = ctx.arch.vocab_size
        assert vocab is not None
        self.embedding = ctx.embedding if c.tied else None
        self.proj = (
            None
            if c.tied
            else nnx.Linear(
                width,
                vocab,
                use_bias=c.bias,
                kernel_init=parallel.named("embed", "vocab"),
                bias_init=parallel.named("vocab", init=nnx.initializers.zeros),
                rngs=ctx.rngs,
            )
        )
        self.bias = (
            nnx.Param(jnp.zeros(vocab), names=("vocab",)) if c.tied and c.bias else None
        )
        self.softcap = c.softcap

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps
    ) -> Float[Array, "b t vocab"]:
        if self.embedding is not None:
            logits = self.embedding.tokens.attend(x)
        else:
            assert self.proj is not None
            logits = self.proj(x)
        if self.bias is not None:
            logits = logits + self.bias
        logits = parallel.shard(softcap(logits, self.softcap), "batch", "seq", "vocab")
        taps.setdefault(OUTPUTS, []).append(logits)
        return logits


@register("head")
def _(
    c: cfg.LMHeadConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Head:
    return Head(c, width, ctx)


class NextN(nnx.Module):
    """Mixes the stream with the embedding of the token k positions ahead, where this
    is the k-th nextn to run in the forward pass (shared modules run more than once).

    On the n copies R of a widened stream it mixes the embedding into each one,
    W_e norm(e) + W_h norm(R)_j, where norm(R) normalizes the copies together, as one
    vector of n * d with a gain for every copy and channel, so that they keep their
    sizes relative to each other. On one stream under an algebra that widens the
    stream, it widens its output, as `embed` does.
    """

    def __init__(
        self, c: cfg.NextNConfig, width: int | Literal["streams"], ctx: Context
    ):
        assert ctx.embedding is not None
        d, copies = ctx.arch.d_model, widening(ctx.arch.residual)
        self.embedding = ctx.embedding
        if width == "streams":  # n copies in, n copies out
            assert copies is not None
            self.hidden_norm, self.copies = Norm(c.norm, copies * d), None
            width = d
        else:  # one stream in, widened at depth 0
            self.hidden_norm = Norm(c.norm, width)
            self.copies = None if ctx.depth else copies
        self.embed_norm = Norm(c.norm, d)
        self.proj = nnx.Linear(d + width, width, use_bias=False, rngs=ctx.rngs)

    def __call__(
        self, x: Float[Array, "b t *copies d"], taps: Taps
    ) -> Float[Array, "b t *widened d"]:
        runs = taps.setdefault(NEXTN, [])
        runs.append(None)
        k, tokens = len(runs), taps[TOKENS][0]
        # The last k positions (all, in a shorter row) have no token that far ahead;
        # pad them with 0.
        ahead = jnp.pad(tokens[:, k:], ((0, 0), (0, min(k, tokens.shape[1]))))
        e = self.embed_norm(self.embedding.embed_tokens(ahead))
        if x.ndim == 4:  # the copies of a widened stream
            h = self.hidden_norm(x.reshape(*x.shape[:2], -1)).reshape(x.shape)
            e = jnp.broadcast_to(e[:, :, None], x.shape)
        else:
            h = self.hidden_norm(x)
        y = self.proj(jnp.concatenate([e, h], -1))
        return parallel.stream(y if self.copies is None else widen(y, self.copies))


@register("nextn")
def _(
    c: cfg.NextNConfig,
    width: int | Literal["streams"],
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> NextN:
    return NextN(c, width, ctx)
