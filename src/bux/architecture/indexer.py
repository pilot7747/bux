"""The lightning indexer of DeepSeek sparse attention: which keys each query attends to.

Keys sit in slots along a row, in order, and `order` says which key of its document
each slot holds (-1 for a slot with none, as between compressed latents). Blocks of
keys are counted from a document's first key, so none spans two documents.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import config as cfg
from bux.architecture.norm import Norm
from bux.architecture.rope import rotate

# DeepSeek-V3.2's indexer keys, projected from the layer's input, are layer-normalized.
_STREAM_KEY_NORM = cfg.NormalizationConfig(type="layernorm", bias=True)


def _groups(order: Int[Array, "b s"], size: int) -> Int[Array, "b s"]:
    """Which block of `size` keys each slot is in, counted along the row: a block starts
    at every size-th key of a document."""
    starts = (order >= 0) & (order % size == 0)
    return jnp.maximum(jnp.cumsum(starts, axis=1) - 1, 0)


def _per_block(
    values: Shaped[Array, "b t s"],
    block: Int[Array, "b s"],
    reduce: Literal["sum", "max"],
) -> Shaped[Array, "b t s"]:
    """Every query's sum or maximum of the values of each block's slots, at the block's
    index; -inf for the maximum of a block with none."""
    op = jax.ops.segment_sum if reduce == "sum" else jax.ops.segment_max

    def row(v: Shaped[Array, "t s"], ids: Int[Array, " s"]) -> Shaped[Array, "t s"]:
        return op(v.T, ids, num_segments=v.shape[-1]).T

    return jax.vmap(row)(values, block)


def _spread(
    per: Shaped[Array, "b t n"], block: Int[Array, "b s"]
) -> Shaped[Array, "b t s"]:
    """What each slot's block has."""
    b, t, _ = per.shape
    index = jnp.broadcast_to(block[:, None], (b, t, block.shape[1]))
    return jnp.take_along_axis(per, index, axis=-1)


def _top(scores: Float[Array, "b t n"], k: int) -> Bool[Array, "b t n"]:
    """Each query's k highest scores, of those above -inf."""
    b, t, n = scores.shape
    if min(k, n) == 0:
        return jnp.zeros(scores.shape, bool)
    _, index = jax.lax.top_k(scores, min(k, n))
    rows, queries = jnp.arange(b)[:, None, None], jnp.arange(t)[None, :, None]
    picked = jnp.zeros(scores.shape, bool).at[rows, queries, index].set(True)
    return picked & (scores > -jnp.inf)


def _newest(
    seen: Bool[Array, "b t s"], block: Int[Array, "b s"]
) -> Bool[Array, "b t s"]:
    """Whether each block (at its index) holds the newest key each query sees."""
    s = seen.shape[-1]
    last = jnp.argmax(jnp.where(seen, jnp.arange(s), -1), axis=-1)
    own = jnp.take_along_axis(block, last, axis=1)
    return (jnp.arange(s) == own[..., None]) & seen.any(-1, keepdims=True)


class Indexer(nnx.Module):
    """The lightning indexer of DeepSeek sparse attention (DSA, DeepSeek-V3.2): a small
    attention with `num_heads` query heads of its own, which scores every key a query
    may see by sum_j w_j ReLU(q_j . k_s) against one key a slot, and keeps the `top_k`
    best. The weights w_j come from the input when it is `weighted`; else the heads
    are summed. Its selection is a mask, which no gradient goes through, so the
    layer's loss does not train it (DSA trains it to match the attention, with a loss
    of its own).

    With block_size > 1 it scores blocks of keys by their mean key and keeps whole ones,
    with the block of the query's newest key (Qwen3.8 QSA). It can export a candidate
    pool of coarse blocks, and keep keys only inside one it imports (DeepSeek-V4.1).
    """

    def __init__(
        self,
        c: cfg.SparseIndexerConfig,
        width: int,
        query_width: int,
        key_width: int | None,
        key_norm: cfg.NormalizationConfig,
        rope: cfg.RoPEConfig | None,
        rope_dims: int,
        rngs: nnx.Rngs,
    ):
        """Queries come from `query_width` wide inputs, and keys from `key_width` wide
        ones, or from another layer when None. `rope` turns the first `rope_dims` dims
        of both."""
        h, i = c.num_heads, c.head_dim
        self.heads, self.head_dim = h, i
        self.top_k, self.block_size = c.top_k, c.block_size
        self.candidate_block_size = c.candidate_block_size
        self.candidate_top_k = c.candidate_top_k

        # The randomly initialized layers; reordering them changes a seed's init.
        self.q = nnx.Linear(query_width, h * i, use_bias=False, rngs=rngs)
        self.k = (
            nnx.Linear(key_width, i, use_bias=False, rngs=rngs) if key_width else None
        )
        self.weights = (
            nnx.Linear(width, h, use_bias=False, rngs=rngs) if c.weighted else None
        )

        self.k_norm = Norm(key_norm, i) if key_width else None
        self.rope = None if rope is None else rope.model_copy(update={"fraction": 1.0})
        self.rope_dims = min(rope_dims, i) // 2 * 2

    def keys(
        self, source: Float[Array, "b s e"], pos: Int[Array, "b s"] | None
    ) -> Float[Array, "b s i"]:
        """The indexer's keys, one a slot, from what they are projected from."""
        assert self.k is not None and self.k_norm is not None
        k = self.k_norm(self.k(source))
        return self._rotate(k[:, :, None], pos)[:, :, 0]

    def scores(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"],
    ) -> Float[Array, "b t s"]:
        """Every query's score for every key."""
        b, t, _ = x.shape
        q = self._rotate(self.q(query).reshape(b, t, self.heads, self.head_dim), pos)
        dots = jax.nn.relu(jnp.einsum("bthi,bsi->bths", q, keys)) * self.head_dim**-0.5
        if self.weights is None:
            return dots.sum(2)
        w = self.weights(x) * self.heads**-0.5
        return jnp.einsum("bths,bth->bts", dots, w)

    def __call__(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"],
        allowed: Bool[Array, "b t s"],
        order: Int[Array, "b s"],
        candidates: Bool[Array, "b t s"] | None = None,
        pool: bool = False,
    ) -> tuple[Bool[Array, "b t s"], Bool[Array, "b t s"] | None]:
        """The keys each query attends to, of those it is `allowed` and inside the
        `candidates` if given; and the candidate pool, if asked for."""
        if candidates is not None:
            allowed = allowed & candidates
        if self.block_size > 1:
            return self._blocks(x, query, pos, keys, allowed, order), None
        scores = jax.lax.stop_gradient(self.scores(x, query, pos, keys))
        scores = jnp.where(allowed, scores, -jnp.inf)
        return _top(scores, self.top_k), self._pool(scores, order) if pool else None

    def _blocks(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"],
        allowed: Bool[Array, "b t s"],
        order: Int[Array, "b s"],
    ) -> Bool[Array, "b t s"]:
        """Whole blocks of block_size keys: the top_k // block_size best of those whose
        keys a query sees all of, scored by their mean key, and the block of the newest
        key it sees (which can hold keys after it)."""
        block = _groups(order, self.block_size)
        exists = (order >= 0)[:, None, :]
        count = _per_block(exists.astype(jnp.int32), block, "sum")
        sums = _per_block(keys.transpose(0, 2, 1) * exists, block, "sum")
        means = (sums / jnp.maximum(count, 1)).transpose(0, 2, 1)
        seen = _per_block(allowed.astype(jnp.int32), block, "sum")
        own = _newest(allowed, block)
        scores = jax.lax.stop_gradient(self.scores(x, query, pos, means))
        scores = jnp.where((seen == count) & (count > 0) & ~own, scores, -jnp.inf)
        picked = _top(scores, self.top_k // self.block_size) | own
        return _spread(picked, block) & allowed

    def _pool(
        self, scores: Float[Array, "b t s"], order: Int[Array, "b s"]
    ) -> Bool[Array, "b t s"]:
        """The candidate pool: the candidate_top_k best blocks of candidate_block_size
        keys, each scored by its best key, and always the block of the newest key a
        query sees, which may not be full yet."""
        block = _groups(order, self.candidate_block_size)
        seen = scores > -jnp.inf
        best = jnp.where(
            _newest(seen, block), jnp.inf, _per_block(scores, block, "max")
        )
        return _spread(_top(best, self.candidate_top_k), block) & seen

    def _rotate(
        self, x: Float[Array, "b t h i"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h i"]:
        """x with its first rope_dims dims turned by the positions."""
        n = self.rope_dims
        if self.rope is None or n == 0:
            return x
        return jnp.concatenate([rotate(x[..., :n], self.rope, pos), x[..., n:]], -1)


def from_stream(
    c: cfg.SparseIndexerConfig,
    width: int,
    query_width: int,
    rope: cfg.RoPEConfig | None,
    rope_dims: int,
    rngs: nnx.Rngs,
) -> Indexer:
    """DeepSeek-V3.2's indexer, for keys that are the tokens: its keys are projected
    from the layer's input and layer-normalized, and rotated in RoPE's half layout
    whatever the layer's."""
    half = None if rope is None else rope.model_copy(update={"layout": "half"})
    return Indexer(
        c, width, query_width, width, _STREAM_KEY_NORM, half, rope_dims, rngs
    )
