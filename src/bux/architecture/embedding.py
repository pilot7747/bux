"""The token embedding and the layers that read it: the LM head and nextn."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
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
from bux.lang import ir


class Embedding(nnx.Module):
    def __init__(self, c: cfg.EmbeddingConfig, vocab: int, d: int, rngs: nnx.Rngs):
        self.tokens = nnx.Embed(vocab, d, rngs=rngs)
        self.positions = (
            nnx.Embed(c.max_positions, d, rngs=rngs) if c.max_positions else None
        )
        self.multiplier = d**0.5 if c.scale_by_sqrt_dim else 1.0
        self.norm = Norm(c.norm, d) if c.norm is not None else None
        self.dropout = nnx.Dropout(c.dropout, rngs=rngs)

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        """The token embedding, scaled and normalized: what nextn sees."""
        x = self.tokens(tokens) * self.multiplier
        return x if self.norm is None else self.norm(x)

    def __call__(self, tokens: Int[Array, "b t"], taps: Taps) -> Float[Array, "b t d"]:
        x = self.embed_tokens(tokens)
        if self.positions is not None:
            pos = positions(taps)
            x = x + self.positions(jnp.arange(tokens.shape[-1]) if pos is None else pos)
        return self.dropout(x)


@register("embed")
def _(
    c: cfg.EmbeddingConfig, width: ir.Width, ports: tuple[ir.Port, ...], ctx: Context
) -> Embedding:
    assert ctx.arch.vocab_size is not None
    embedding = Embedding(c, ctx.arch.vocab_size, ctx.arch.d_model, ctx.rngs)
    if ctx.embedding is None:
        ctx.embedding = embedding
    return embedding


class Head(nnx.Module):
    def __init__(self, c: cfg.LMHeadConfig, width: int, ctx: Context):
        vocab = ctx.arch.vocab_size
        assert vocab is not None
        self.embedding = ctx.embedding if c.tied else None
        self.proj = (
            None if c.tied else nnx.Linear(width, vocab, use_bias=c.bias, rngs=ctx.rngs)
        )
        self.bias = nnx.Param(jnp.zeros(vocab)) if c.tied and c.bias else None
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
        logits = softcap(logits, self.softcap)
        taps.setdefault(OUTPUTS, []).append(logits)
        return logits


@register("head")
def _(
    c: cfg.LMHeadConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Head:
    return Head(c, width, ctx)


class NextN(nnx.Module):
    """Mixes the stream with the embedding of the token k positions ahead, where this
    is the k-th nextn to run in the forward pass (shared modules run more than once)."""

    def __init__(self, c: cfg.NextNConfig, width: int, ctx: Context):
        assert ctx.embedding is not None
        self.embedding = ctx.embedding
        self.hidden_norm = Norm(c.norm, width)
        self.embed_norm = Norm(c.norm, ctx.arch.d_model)
        self.proj = nnx.Linear(
            ctx.arch.d_model + width, width, use_bias=False, rngs=ctx.rngs
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        runs = taps.setdefault(NEXTN, [])
        runs.append(None)
        k, tokens = len(runs), taps[TOKENS][0]
        # The last k positions have no token that far ahead; pad them with 0.
        ahead = jnp.pad(tokens[:, k:], ((0, 0), (0, k)))
        e = self.embed_norm(self.embedding.embed_tokens(ahead))
        return self.proj(jnp.concatenate([e, self.hidden_norm(x)], -1))


@register("nextn")
def _(
    c: cfg.NextNConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> NextN:
    return NextN(c, width, ctx)
