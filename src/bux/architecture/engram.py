"""Engram: memory looked up by hashed n-grams of the input tokens."""

import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, ArrayLike, Float, Int, Int32, UInt32

from bux import config as cfg
from bux import parallel
from bux.architecture.conv import CausalConv
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir


def _next_primes(start: int, count: int, taken: set[int]) -> list[int]:
    """Return the `count` smallest primes >= start not taken yet, and take them."""
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
    within: Int[Array, "b t"] | None = None,
) -> Int32[Array, "b t (n-1)*heads"]:
    """Return the table rows of the n-grams of 2 to n ids ending at every position,
    one per head.

    The n-gram of the last j + 1 ids hashes to XOR_{i <= j} ids[t - i] * m_i, in wrapping
    uint32 arithmetic. Positions before the sequence, before the document (by `within`,
    how many tokens of it come before each), or at and before the last boundary token
    before t, count as `pad`, so that n-grams do not cross documents.
    """
    b, t = ids.shape
    pos = jnp.arange(t)
    start = jnp.zeros((b, t), jnp.int32)
    if boundary is not None:
        after = jnp.where(tokens == boundary, pos + 1, 0)
        start = jnp.pad(jax.lax.cummax(after, axis=1)[:, :-1], ((0, 0), (1, 0)))
    if within is not None:
        start = jnp.maximum(start, pos - within)
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
    share the value. The table's rows split as the vocabulary does (``bux.parallel``),
    and the rest is whole."""

    no_weight_decay = ("table",)

    def __init__(
        self, config: cfg.EngramConfig, width: int | Literal["streams"], ctx: Context
    ):
        d, vocab = ctx.arch.d_model, ctx.arch.vocab_size
        assert vocab is not None
        copies = getattr(ctx.arch.residual, "expansion", 1) if width == "streams" else 1
        n, heads, head_dim = config.max_ngram_size, config.num_heads, config.head_dim
        orders = n - 1
        primes = _next_primes(config.table_size, orders * heads, ctx.built.primes)
        offsets = np.cumsum([0, *primes[:-1]])
        self.primes = nnx.Variable(
            jnp.asarray(primes, jnp.uint32).reshape(orders, heads)
        )
        self.offsets = nnx.Variable(
            jnp.asarray(offsets, jnp.uint32).reshape(orders, heads)
        )
        rng = np.random.default_rng(10007 * (ctx.built.count["engram"] + 1))
        odd = rng.integers(0, 2**31, n) * 2 + 1
        self.multipliers = nnx.Variable(jnp.asarray(odd, jnp.uint32))
        # A stand-in for the tokenizer's normalized classes, which a checkpoint replaces.
        classes = config.vocab_classes
        self.classes = nnx.Variable(jnp.arange(vocab) % classes) if classes else None
        init = nnx.initializers.normal(1.0)
        table = init(ctx.rngs.params(), (sum(primes), head_dim))
        self.table = nnx.Param(table, names=("vocab", None))
        self.kv = nnx.Linear(
            orders * heads * head_dim, (copies + 1) * d, use_bias=False, rngs=ctx.rngs
        )
        norm, size = config.norm, config.conv_size
        self.q_norm, self.k_norm = Norm(norm, (copies, d)), Norm(norm, (copies, d))
        self.conv = (
            CausalConv(copies * d, size, False, ctx.rngs, dilation=n, zero=True)
            if size
            else None
        )
        self.conv_norm = Norm(norm, (copies, d)) if size else None
        self.boundary = config.boundary_token

    def __call__(
        self, x: Float[Array, "b t *copies d"], taps: Taps
    ) -> Float[Array, "b t *copies d"]:
        tokens = taps.tokens
        assert tokens is not None
        ids, pad = tokens, self.boundary or 0
        if self.classes is not None:
            ids, pad = self.classes[...][tokens], self.classes[...][pad]
        docs = segments(taps)
        rows = _ngram_rows(
            ids,
            tokens,
            self.multipliers[...],
            self.primes[...],
            self.offsets[...],
            self.boundary,
            pad,
            None if docs is None else docs.within,
        )
        h = x if x.ndim == 4 else x[..., None, :]
        # Gathered from the table whole over `fsdp`, as the embedding gathers; then
        # the keys and the value side by side, whole.
        table = parallel.shard(self.table[...], "vocab", None)
        memory = table[rows].reshape(*rows.shape[:2], -1)
        memory = parallel.shard(memory, "batch", "seq", None)
        kv = parallel.shard(self.kv(memory), "batch", "seq", None)
        b, t, n, d = h.shape
        key, value = kv[..., : n * d].reshape(b, t, n, d), kv[..., n * d :]
        a = (self.q_norm(h) * self.k_norm(key)).sum(-1) / math.sqrt(d)
        gate = jax.nn.sigmoid(jnp.sign(a) * jnp.sqrt(jnp.maximum(jnp.abs(a), 1e-6)))
        u = gate[..., None] * value[..., None, :]
        if self.conv is not None and self.conv_norm is not None:
            conv = self.conv(self.conv_norm(u).reshape(b, t, n * d), taps)
            u = u + jax.nn.silu(conv).reshape(u.shape)
        y = h + u
        return parallel.shard_stream(y if x.ndim == 4 else y[..., 0, :])


@register("engram")
def _(
    c: cfg.EngramConfig,
    width: int | Literal["streams"],
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> Engram:
    return Engram(c, width, ctx)
