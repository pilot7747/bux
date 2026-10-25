"""The token embedding and the layers that read it: the LM head and nextn."""

from typing import Literal

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.common.sharding import linear, projection
from bux.architecture.layers.norm import Norm
from bux.architecture.logits import HeadOutput, Projection
from bux.architecture.registry import Context, Taps, positions, register
from bux.architecture.residual import widen, widening
from bux.lang import ir

#: How a table of embeddings starts, as `flax.nnx.Embed`'s do: normal, of variance
#: 1 / d.
EMBED = nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


def _table(rows: int, d: int, row: str | None, rngs: nnx.Rngs) -> nnx.Embed:
    """Make a table of `rows` vectors of size d, its rows named `row`."""
    init = parallel.named(row, "embed", init=EMBED)
    return nnx.Embed(rows, d, embedding_init=init, rngs=rngs)


def lookup(
    table: Float[Array, "n d"], ids: Int[Array, "*ids"], row: str | None
) -> Float[Array, "*ids d"]:
    """Return the rows `ids` of `table`, whose rows are named `row`. The table splits
    only by its names: XLA would rematerialize one split over `fsdp` in full to gather
    from it. The rows split as inside a layer: under sequence parallelism, rows split
    as the stream is would gather the table."""
    rows = jnp.take(parallel.shard(table, row, "embed"), ids, axis=0)
    return parallel.shard(rows, *("batch", "seq")[2 - ids.ndim :], None)


class Embedding(nnx.Module):
    """The token embedding, ``embed``: token ids to the stream, by a table split over
    the vocabulary (`bux.parallel`) that a tied head and nextn read from the model's
    first one.

    A token's row is scaled by ``sqrt(d)`` with `scale_by_sqrt_dim` (Gemma) and
    normalized if the config has a norm (Inkling); then learned absolute positions
    are added, if any (GPT-2), each token's position in its document, and dropout
    applied. With `copies`, for a residual algebra that widens the stream, it returns
    that many copies of it. Muon leaves its table to the other optimizer
    (``no_muon``).

    Args:
        config: Its settings (`bux.config.EmbeddingConfig`).
        vocab: The rows of the table, the architecture's ``vocab_size``.
        d: The width of the stream, ``d_model``.
        rngs: The random streams the tables start from, and dropout draws from.
        copies: How many copies of the stream to return, for a widening residual
            algebra; None for the stream alone.
    """

    no_muon = True

    def __init__(
        self,
        config: cfg.EmbeddingConfig,
        vocab: int,
        d: int,
        rngs: nnx.Rngs,
        copies: int | None = None,
    ):
        self.tokens = _table(vocab, d, "vocab", rngs)
        max_positions = config.max_positions
        self.positions = _table(max_positions, d, None, rngs) if max_positions else None
        self.multiplier = d**0.5 if config.scale_by_sqrt_dim else 1.0
        self.norm = Norm(config.norm, d) if config.norm is not None else None
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.copies = copies

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        """Return the token embedding, scaled and normalized: what nextn reads."""
        x = lookup(self.tokens.embedding[...], tokens, "vocab") * self.multiplier
        return x if self.norm is None else self.norm(x)

    def __call__(
        self, tokens: Int[Array, "b t"], taps: Taps
    ) -> Float[Array, "b t *copies d"]:
        """Return the stream that starts from the token ids.

        Args:
            tokens: The token ids, b rows of t.
            taps: The forward pass's taps, which give the tokens' positions in their
                documents.

        Returns:
            The embedding, ``b t d``, or ``b t n d`` in n copies.
        """
        x = self.embed_tokens(tokens)
        if self.positions is not None:
            pos = positions(taps)
            pos = jnp.arange(tokens.shape[-1]) if pos is None else pos
            x = x + lookup(self.positions.embedding[...], pos, None)
        x = self.dropout(x)
        if self.copies is not None:
            x = widen(x, self.copies)
        return parallel.shard_stream(x)


@register("embed")
def _(
    config: cfg.EmbeddingConfig,
    width: ir.Width,
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> Embedding:
    vocab, d = ctx.arch.vocab_size, ctx.arch.d_model
    assert vocab is not None
    copies = widening(ctx.arch.residual) if ctx.depth == 0 else None
    embedding = Embedding(config, vocab, d, ctx.rngs, copies)
    if ctx.built.embedding is None:
        ctx.built.embedding = embedding
    return embedding


class Head(nnx.Module):
    """The language-model head, ``head``: the stream to logits, split over the
    vocabulary.

    The logits are ``x W``, or, tied, ``x E^T`` with E the first embedding's token
    table, in the stream's dtype with float32 accumulation, then the bias and the
    softcap. The head does not make them: it leaves its input and its projection in
    the taps (`~bux.architecture.logits.HeadOutput`), from which the objective makes
    the logits a chunk at a time, and passes the stream on. Muon leaves its matrix to
    the other optimizer (``no_muon``).

    Args:
        config: Its settings (`bux.config.LMHeadConfig`).
        width: The width of the stream it reads.
        ctx: The build's context: the vocabulary, the random streams, and the first
            embedding a tied head reads.
    """

    no_muon = True

    def __init__(self, config: cfg.LMHeadConfig, width: int, ctx: Context):
        vocab, tied, bias = ctx.arch.vocab_size, config.tied, config.bias
        assert vocab is not None
        self.embedding = ctx.built.embedding if tied else None
        self.proj = (
            None
            if tied
            else linear(
                width, vocab, ("embed", "vocab"), bias, ctx.rngs, in_compute=True
            )
        )
        self.bias = (
            nnx.Param(jnp.zeros(vocab), names=("vocab",)) if tied and bias else None
        )
        self.softcap = config.softcap

    def projection(self) -> Projection:
        """Return the map from the stream to logits, of the weights as they are now."""
        if self.embedding is not None:
            kernel, bias = self.embedding.tokens.embedding[...].T, self.bias
        else:
            assert self.proj is not None
            kernel, bias = self.proj.kernel[...], self.proj.bias
        cap = None if self.softcap is None else jnp.asarray(self.softcap, jnp.float32)
        return Projection(kernel, None if bias is None else bias[...], cap)

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Leave x and the projection in ``taps.heads``, and return x."""
        taps.heads.append(HeadOutput(x, self.projection()))
        return x


@register("head")
def _(
    c: cfg.LMHeadConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Head:
    return Head(c, width, ctx)


class NextN(nnx.Module):
    """Multi-token prediction's input (DeepSeek-V3), ``nextn``: the stream mixed with
    the embedding of the token k positions ahead, ``W [norm(e); norm(h)]``, for the
    k-th nextn to run in the forward pass (shared modules run more than once). A head
    after it predicts one token further ahead than the one before it. The last k
    positions, which have no token that far ahead, embed token 0.

    On the n copies R of a widened stream it mixes the embedding into each one,
    ``W_e norm(e) + W_h norm(R)_j``. ``norm(R)`` normalizes the copies together, as
    one vector of ``n * d`` with a gain for every copy and channel, so that they keep
    their sizes relative to each other. On one stream under an algebra that widens
    the stream, it widens its output, as `embed` does.

    Args:
        config: Its settings (`bux.config.NextNConfig`): the norm of its inputs.
        width: The width of the stream it reads, or ``"streams"`` for a widened one.
        ctx: The build's context: the first embedding, whose table it reads, the
            residual algebra, and the random streams.
    """

    def __init__(
        self, config: cfg.NextNConfig, width: int | Literal["streams"], ctx: Context
    ):
        assert ctx.built.embedding is not None
        d, copies = ctx.arch.d_model, widening(ctx.arch.residual)
        self.embedding = ctx.built.embedding
        if width == "streams":  # n copies in, n copies out
            assert copies is not None
            self.hidden_norm, self.copies = Norm(config.norm, copies * d), None
            width = d
        else:  # one stream in, widened at depth 0
            self.hidden_norm = Norm(config.norm, width)
            self.copies = None if ctx.depth else copies
        self.embed_norm = Norm(config.norm, d)
        self.proj = projection(d + width, width, "mtp", use_bias=False, rngs=ctx.rngs)

    def __call__(
        self, x: Float[Array, "b t *copies d"], taps: Taps
    ) -> Float[Array, "b t *widened d"]:
        """Return the stream mixed with the embedding of the tokens ahead.

        Args:
            x: The stream, ``b t d``, or its n copies, ``b t n d``.
            taps: The forward pass's taps: the input tokens, and how many nextn
                layers ran before this one.

        Returns:
            The mixed stream, ``b t d``; or ``b t n d``, for n copies in, or one
            stream widened by the residual algebra.
        """
        taps.nextn_runs += 1
        offset, tokens = taps.nextn_runs, taps.tokens
        assert tokens is not None
        # The last `offset` positions (all, in a shorter row) have no token that far
        # ahead; pad them with 0.
        pad = min(offset, tokens.shape[1])
        ahead = jnp.pad(tokens[:, offset:], ((0, 0), (0, pad)))
        e = self.embed_norm(self.embedding.embed_tokens(ahead))
        if x.ndim == 4:  # the copies of a widened stream
            h = self.hidden_norm(x.reshape(*x.shape[:2], -1)).reshape(x.shape)
            e = jnp.broadcast_to(e[:, :, None], x.shape)
        else:
            h = self.hidden_norm(x)
        y = self.proj(jnp.concatenate([e, h], -1))
        return parallel.shard_stream(
            y if self.copies is None else widen(y, self.copies)
        )


@register("nextn")
def _(
    c: cfg.NextNConfig,
    width: int | Literal["streams"],
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> NextN:
    return NextN(c, width, ctx)
