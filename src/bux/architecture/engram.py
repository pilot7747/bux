"""Engram: memory looked up by hashed n-grams of the input tokens."""

import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, ArrayLike, Float, Int, Int32, UInt32

from bux import config as cfg
from bux.architecture.conv import CausalConv
from bux.architecture.norm import Norm
from bux.architecture.registry import TOKENS, Context, Taps, positions, register
from bux.lang import ir


def _next_primes(start: int, count: int, taken: set[int]) -> list[int]:
    """The `count` smallest primes >= start that are not taken yet, which takes them."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if n not in taken and all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
            taken.add(n)
        n += 1
    return found


def _ngram_rows(
    ids: Int[Array, "b t"],
    tokens: Int[Array, "b t"],
    multipliers: UInt32[Array, " n"],
    primes: UInt32[Array, "n-1 heads"],
    offsets: UInt32[Array, "n-1 heads"],
    boundary: int | None,
    pad: Int[ArrayLike, ""],
    docs: Int[Array, "b t"] | None = None,
) -> Int32[Array, "b t (n-1)*heads"]:
    """Table rows of the n-grams of 2 to n ids ending at every position, one a head.

    The n-gram of the last j + 1 ids hashes to XOR_{i <= j} ids[t - i] * m_i, in wrapping
    uint32 arithmetic. Positions before the sequence, before the document (by `docs`,
    the positions in it), or at and before the last boundary token before t, count as
    `pad`, so that n-grams do not cross documents.
    """
    b, t = ids.shape
    pos = jnp.arange(t)
    start = jnp.zeros((b, t), jnp.int32)
    if boundary is not None:
        after = jnp.where(tokens == boundary, pos + 1, 0)
        start = jnp.pad(jax.lax.cummax(after, axis=1)[:, :-1], ((0, 0), (1, 0)))
    if docs is not None:
        start = jnp.maximum(start, pos - docs)
    h, rows = jnp.zeros((b, t), jnp.uint32), []
    for i in range(multipliers.shape[0]):
        prev = jnp.take(ids, jnp.maximum(pos - i, 0), axis=1)
        prev = jnp.where(pos - i >= start, prev, pad).astype(jnp.uint32)
        h = h ^ (prev * multipliers[i])
        if i:
            rows.append(h[..., None] % primes[i - 1] + offsets[i - 1])
    return jnp.concatenate(rows, -1).astype(jnp.int32)


class Engram(nnx.Module):
    """Hashed n-gram rows of the input tokens, gated by the stream and added to it. On a
    widened stream, with an axis of copies, every copy has a key of its own, and they
    share the value."""

    def __init__(
        self, c: cfg.EngramConfig, width: int | Literal["streams"], ctx: Context
    ):
        d, vocab = ctx.arch.d_model, ctx.arch.vocab_size
        assert vocab is not None
        copies = getattr(ctx.arch.residual, "expansion", 1) if width == "streams" else 1
        orders, heads = c.max_ngram_size - 1, c.num_heads
        primes = _next_primes(c.table_size, orders * heads, ctx.primes)
        offsets = np.cumsum([0, *primes[:-1]])
        self.primes = nnx.Variable(
            jnp.asarray(primes, jnp.uint32).reshape(orders, heads)
        )
        self.offsets = nnx.Variable(
            jnp.asarray(offsets, jnp.uint32).reshape(orders, heads)
        )
        ctx.engrams += 1
        rng = np.random.default_rng(10007 * ctx.engrams)
        odd = rng.integers(0, 2**31, c.max_ngram_size) * 2 + 1
        self.multipliers = nnx.Variable(jnp.asarray(odd, jnp.uint32))
        # A stand-in for the tokenizer's normalized classes, which a checkpoint replaces.
        self.classes = (
            nnx.Variable(jnp.arange(vocab) % c.vocab_classes)
            if c.vocab_classes
            else None
        )
        init = nnx.initializers.normal(1.0)
        self.table = nnx.Param(init(ctx.rngs.params(), (sum(primes), c.head_dim)))
        self.kv = nnx.Linear(
            orders * heads * c.head_dim, (copies + 1) * d, use_bias=False, rngs=ctx.rngs
        )
        self.q_norm, self.k_norm = Norm(c.norm, (copies, d)), Norm(c.norm, (copies, d))
        self.conv = (
            CausalConv(
                copies * d,
                c.conv_size,
                False,
                ctx.rngs,
                dilation=c.max_ngram_size,
                zero=True,
            )
            if c.conv_size
            else None
        )
        self.conv_norm = Norm(c.norm, (copies, d)) if c.conv_size else None
        self.boundary = c.boundary_token

    def __call__(
        self, x: Float[Array, "b t *copies d"], taps: Taps
    ) -> Float[Array, "b t *copies d"]:
        tokens = taps[TOKENS][0]
        ids, pad = tokens, self.boundary or 0
        if self.classes is not None:
            ids, pad = self.classes[...][tokens], self.classes[...][pad]
        rows = _ngram_rows(
            ids,
            tokens,
            self.multipliers[...],
            self.primes[...],
            self.offsets[...],
            self.boundary,
            pad,
            positions(taps),
        )
        h = x if x.ndim == 4 else x[..., None, :]
        (b, t, n, d), kv = (
            h.shape,
            self.kv(self.table[...][rows].reshape(*rows.shape[:2], -1)),
        )
        key, value = kv[..., : n * d].reshape(b, t, n, d), kv[..., n * d :]
        a = (self.q_norm(h) * self.k_norm(key)).sum(-1) / math.sqrt(d)
        gate = jax.nn.sigmoid(jnp.sign(a) * jnp.sqrt(jnp.maximum(jnp.abs(a), 1e-6)))
        u = gate[..., None] * value[..., None, :]
        if self.conv is not None and self.conv_norm is not None:
            conv = self.conv(self.conv_norm(u).reshape(b, t, n * d), taps)
            u = u + jax.nn.silu(conv).reshape(u.shape)
        y = h + u
        return y if x.ndim == 4 else y[..., 0, :]


@register("engram")
def _(
    c: cfg.EngramConfig,
    width: int | Literal["streams"],
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> Engram:
    return Engram(c, width, ctx)
