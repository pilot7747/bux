"""The lightning indexer of DeepSeek sparse attention, which picks the keys each query
attends to, and its loss.

Keys sit in slots along a row, in order, and `order` says which key of its document
each slot holds (-1 for a slot with none, as between compressed latents). Blocks of
keys are counted from a document's first key, so none spans two documents.

It scores a chunk of queries at a time (``sparse.chunked``), so that no t by t floats
are made: it keeps each query's top-k by index (``sparse.Kept``), and its loss scores
each chunk again.
"""

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import config as cfg
from bux import parallel
from bux.architecture.norm import Norm
from bux.architecture.rope import rotate
from bux.architecture.sharding import projection, shard_heads
from bux.architecture.sparse import Kept, Span, by_rows, chunked, chunks, mask

# DeepSeek-V3.2's indexer keys, projected from the layer's input, are layer-normalized.
_STREAM_KEY_NORM = cfg.NormalizationConfig(type="layernorm", bias=True)


class Scored(NamedTuple):
    """What the indexer kept by its scores, for its loss to score again: its inputs,
    the keys it scored (a block's at the block's index), and what it kept of them,
    by index (-1 for none), with `block` (which block each key is in) blocks."""

    x: Float[Array, "b t d"]
    query: Float[Array, "b t e"]
    pos: Int[Array, "b t"]
    keys: Float[Array, "b s i"]
    kept: Int[Array, "b t k"]
    block: Int[Array, "b s"] | None = None


Target = Callable[[Any], Float[Array, "b c s"]]  # a chunk's attention to the keys


def _kl(
    target: Float[Array, "b t n"],
    scores: Float[Array, "b t n"],
    mask: Bool[Array, "b t n"],
) -> Float[Array, "b t"]:
    """Return every query's KL(target || softmax(scores)) over the slots `mask` keeps,
    the target normalized over them: 0 for a query with none, as 0 log 0 is."""
    p = jnp.where(mask, target, 0)
    p = p / jnp.maximum(p.sum(-1, keepdims=True), jnp.finfo(p.dtype).tiny)
    log_q = jax.nn.log_softmax(jnp.where(mask, scores, jnp.finfo(scores.dtype).min))
    log_p = jnp.log(jnp.where(p > 0, p, 1))
    return jnp.where(p > 0, p * (log_p - log_q), 0).sum(-1)


def _starts(order: Int[Array, "b s"], size: int) -> Bool[Array, "b s"]:
    """Return whether each slot holds the first key of a block of `size` keys: every
    size-th key of a document."""
    return (order >= 0) & (order % size == 0)


def _groups(order: Int[Array, "b s"], size: int) -> Int[Array, "b s"]:
    """Return which block of `size` keys each slot is in, counted along the row."""
    return jnp.maximum(jnp.cumsum(_starts(order, size), axis=1) - 1, 0)


def _per_block(
    values: Shaped[Array, "b t s"],
    block: Int[Array, "b s"],
    reduce: Literal["sum", "max"],
) -> Shaped[Array, "b t s"]:
    """Return every query's sum or maximum of the values of each block's slots, at the
    block's index; -inf for the maximum of a block with none."""
    op = jax.ops.segment_sum if reduce == "sum" else jax.ops.segment_max

    def row(v: Shaped[Array, "t s"], ids: Int[Array, " s"]) -> Shaped[Array, "t s"]:
        return op(v.T, ids, num_segments=v.shape[-1]).T

    return jax.vmap(row)(values, block)


def _top(scores: Float[Array, "b t n"], top_k: int) -> Int[Array, "b t k"]:
    """Return the indices of each query's top_k highest scores, of those above -inf,
    and -1 past them; of equal ones, the first. By each device's rows: lax.top_k
    gathers every row onto each device otherwise."""
    k = min(top_k, scores.shape[-1])

    def top(scores: Float[Array, "b t n"]) -> Int[Array, "b t k"]:
        values, index = jax.lax.top_k(scores, k)
        return jnp.where(values > -jnp.inf, index, -1)

    return by_rows(top, scores)


def _newest(seen: Bool[Array, "b t s"], block: Int[Array, "b s"]) -> Int[Array, "b t"]:
    """Return the block of the newest key each query sees; -1 for a query that sees
    none."""
    s = seen.shape[-1]
    last = jnp.argmax(jnp.where(seen, jnp.arange(s), -1), axis=-1)
    return jnp.where(seen.any(-1), jnp.take_along_axis(block, last, axis=1), -1)


def _positions(
    pos: Int[Array, "b t"] | None, x: Float[Array, "b t d"]
) -> Int[Array, "b t"]:
    """Return the positions, those of the row's slots if none are given: a chunk of
    queries is not the first of the row."""
    if pos is not None:
        return pos
    b, t, _ = x.shape
    return jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t))


class Indexer(nnx.Module):
    """The lightning indexer of DeepSeek sparse attention (DSA, DeepSeek-V3.2): a small
    attention with `num_heads` query heads and one key per slot, which scores every key
    a query may attend to by sum_j w_j ReLU(q_j . k_s) and keeps the `top_k` best. The
    weights w_j come from the input when it is `weighted`; else the heads are summed.
    Its keep mask passes no gradient and its inputs are detached, so only `loss`
    trains it, and that loss trains nothing else.

    With block_size > 1 it scores blocks of keys and keeps whole ones (Qwen3.8 QSA): a
    block's key is the mean of its keys as projected, normalized, then rotated to the
    position of its first. A query scores the blocks it may attend to all of, and also
    keeps the keys of its own unfinished block. It can export a candidate pool of
    coarse blocks, and keep keys only inside one it imports (DeepSeek-V4.1).

    `config.norm`, if set, normalizes each query head, and the keys instead of
    `key_norm`.

    Its query heads split over `heads`, as the layer's do; its keys, a single head, are
    every device's.
    """

    def __init__(
        self,
        config: cfg.SparseIndexerConfig,
        width: int,
        query_width: int,
        key_width: int | None,
        key_norm: cfg.NormalizationConfig,
        rope: cfg.RoPEConfig | None,
        rope_dims: int,
        rngs: nnx.Rngs,
        query: parallel.Name = "embed",
    ):
        """Project queries from `query_width` wide inputs, whose dimension is `query`
        (``bux.parallel``), and keys from `key_width` wide ones, or take keys from
        another layer when it is None. `rope` rotates the first `rope_dims` dims of
        both."""
        h, i = config.num_heads, config.head_dim
        self.heads, self.head_dim = h, i
        self.top_k, self.block_size = config.top_k, config.block_size
        self.candidate_block_size = config.candidate_block_size
        self.candidate_top_k = config.candidate_top_k
        self.loss_weight = config.loss_weight

        def linear(n_in: int, n_out: int, *names: parallel.Name) -> nnx.Linear:
            init = parallel.named(*names)
            return projection(n_in, n_out, use_bias=False, kernel_init=init, rngs=rngs)

        # The randomly initialized layers; reordering them changes a seed's init.
        self.q = linear(query_width, h * i, query, ("heads", h))
        self.k = linear(key_width, i, None, None) if key_width else None
        self.weights = linear(width, h, "embed", "heads") if config.weighted else None

        self.q_norm = Norm(config.norm, i) if config.norm else None
        pools = config.block_size > 1  # it normalizes the blocks of keys it is given
        self.k_norm = Norm(config.norm or key_norm, i) if key_width or pools else None
        self.rope = None if rope is None else rope.model_copy(update={"fraction": 1.0})
        self.rope_dims = min(rope_dims, i) // 2 * 2

    def keys(
        self, source: Float[Array, "b s e"], pos: Int[Array, "b s"] | None
    ) -> Float[Array, "b s i"]:
        """Project the indexer's keys, one per slot, from `source`: left as projected if
        it pools them in blocks, else normalized and rotated."""
        assert self.k is not None and self.k_norm is not None
        k = self.k(jax.lax.stop_gradient(source))
        if self.block_size > 1:
            return k
        return self._rotate(self.k_norm(k)[:, :, None], pos)[:, :, 0]

    def scores(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"],
    ) -> Float[Array, "b t s"]:
        """Return every query's score for every key."""
        x, query = jax.lax.stop_gradient((x, query))
        dots = jnp.einsum("bthi,bsi->bths", self._queries(query, pos), keys)
        return self._weigh(x, dots)

    def _scores_at(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        index: Int[Array, "b t k"],
    ) -> Float[Array, "b t k"]:
        """Return each query's scores for the keys at its `index`, its kept ones."""
        x, query = jax.lax.stop_gradient((x, query))
        picked = jax.vmap(lambda k, i: k[i])(keys, jnp.maximum(index, 0))
        dots = jnp.einsum("bthi,btki->bthk", self._queries(query, pos), picked)
        return self._weigh(x, dots)

    def _queries(
        self, query: Float[Array, "b t e"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h i"]:
        """Project, normalize and rotate the query heads."""
        b, t, _ = query.shape
        q = self.q(query).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        return shard_heads(self._rotate(q, pos))

    def _weigh(
        self, x: Float[Array, "b t d"], dots: Float[Array, "b t h s"]
    ) -> Float[Array, "b t s"]:
        """Return the heads' ReLU'd dot products summed, by the weights if learned."""
        dots = jax.nn.relu(dots) * self.head_dim**-0.5
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
        span: Span,
        order: Int[Array, "b s"],
        candidates: Kept | None = None,
        export_candidates: bool = False,
    ) -> tuple[Kept, Kept | None, Scored]:
        """Return the keys each query keeps, of those of its `span` and the
        `candidates` (if given); the candidate pool, if asked for; and what it
        scored."""
        x, query = jax.lax.stop_gradient((x, query))
        pos = _positions(pos, x)
        if self.block_size > 1:
            kept, scored = self._blocks(x, query, pos, keys, span, order)
            return kept, None, scored
        s, exists = keys.shape[1], order >= 0
        ranked_keys = jax.lax.stop_gradient(keys)
        pooled = _groups(order, self.candidate_block_size)
        among = None if candidates is None else candidates.block

        def pick(xs: tuple) -> tuple[Int[Array, "b c k"], Int[Array, "b c p"] | None]:
            x, query, pos, span, index = xs
            inside = None if index is None else Kept(index, among)
            seen = mask(span, s, exists, inside)
            ranked = jnp.where(seen, self.scores(x, query, pos, ranked_keys), -jnp.inf)
            if not export_candidates:
                return _top(ranked, self.top_k), None
            return _top(ranked, self.top_k), self._candidates(ranked, pooled)

        index = None if candidates is None else candidates.index
        xs = (x, query, pos, span, index)
        index, best = chunked(pick, xs, chunks(x.shape[1], (self.heads + 1) * s))
        pool = None if best is None else Kept(best, pooled)
        return Kept(index), pool, Scored(x, query, pos, keys, index)

    def loss(
        self, scored: Scored, target: Target, xs: Any, width: int
    ) -> Float[Array, ""]:
        """Return KL(p || softmax(I)) over what the indexer kept, averaged over the
        queries (DeepSeek-V3.2, eq. 4): I its scores, and p the attention summed over
        heads and normalized over the kept keys, which `target` gives for a chunk of
        `xs` (arrays of (b, t, ...)), `width` elements a query. A block's p is its most
        attended key's (Qwen3.8 QSA, eqs. 17 and 20). It scores each chunk of
        queries' kept keys again, and so does its backward pass."""

        def kl(both: tuple) -> Float[Array, "b c"]:
            (x, query, pos, kept), xs = both
            p = jax.lax.stop_gradient(target(xs).astype(jnp.float32))
            if scored.block is not None:
                p = _per_block(p, scored.block, "max")
            p = jnp.take_along_axis(p, jnp.maximum(kept, 0), axis=-1)
            scores = self._scores_at(x, query, pos, scored.keys, kept)
            return _kl(p, scores.astype(jnp.float32), kept >= 0)

        own = (scored.x, scored.query, scored.pos, scored.kept)
        k = scored.kept.shape[-1] * max(self.heads, self.head_dim)
        count = chunks(scored.x.shape[1], max(k, width))
        return chunked(jax.checkpoint(kl), (own, xs), count).mean()

    def _blocks(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        span: Span,
        order: Int[Array, "b s"],
    ) -> tuple[Kept, Scored]:
        """Keep whole blocks of block_size keys: the top_k // block_size best of those
        the span has all of, and the span's keys of the query's own block until it is
        complete. Return them, and what it scored."""
        s, size = keys.shape[1], self.block_size
        block = _groups(order, size)
        block_keys = self._block_keys(keys, pos, order, block)
        exists, ranked_keys = order >= 0, jax.lax.stop_gradient(block_keys)

        def pick(xs: tuple) -> tuple[Int[Array, "b c k"], Int[Array, "b c"]]:
            x, query, pos, span = xs
            seen = mask(span, s, exists)
            full = _per_block(seen.astype(jnp.int32), block, "sum") == size
            scores = self.scores(x, query, pos, ranked_keys)
            picked = _top(jnp.where(full, scores, -jnp.inf), self.top_k // size)
            own = _newest(seen, block)
            unfinished = ~jnp.take_along_axis(full, jnp.maximum(own, 0)[..., None], -1)
            return picked, jnp.where(unfinished[..., 0], own, -1)

        xs = (x, query, pos, span)
        picked, own = chunked(pick, xs, chunks(x.shape[1], (self.heads + 2) * s))
        kept = Kept(jnp.concatenate([picked, own[..., None]], -1), block)
        return kept, Scored(x, query, pos, block_keys, picked, block)

    def _block_keys(
        self,
        keys: Float[Array, "b s i"],
        pos: Int[Array, "b s"],
        order: Int[Array, "b s"],
        block: Int[Array, "b s"],
    ) -> Float[Array, "b s i"]:
        """Return each block's key, at its index: the mean of its block_size keys,
        normalized, then rotated to the position of its first key's slot."""
        assert self.k_norm is not None
        exists = (order >= 0)[:, None, :]
        sums = _per_block(keys.transpose(0, 2, 1) * exists, block, "sum")
        means = self.k_norm(sums.transpose(0, 2, 1) / self.block_size)
        first = jnp.where(_starts(order, self.block_size), pos, 0)[:, None]
        first = _per_block(first, block, "sum")[:, 0]
        return self._rotate(means[:, :, None], first)[:, :, 0]

    def _candidates(
        self, ranked: Float[Array, "b t s"], block: Int[Array, "b s"]
    ) -> Int[Array, "b t p"]:
        """Return the candidate pool, by index: the candidate_top_k best of the
        `block`s of candidate_block_size keys, each scored by its best key, and always
        the block of the newest key a query may attend to, which may not be full
        yet."""
        s = ranked.shape[-1]
        newest = jnp.arange(s) == _newest(ranked > -jnp.inf, block)[..., None]
        best = jnp.where(newest, jnp.inf, _per_block(ranked, block, "max"))
        return _top(best, self.candidate_top_k)

    def _rotate(
        self, x: Float[Array, "b t h i"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h i"]:
        """Rotate the first rope_dims dims of `x` by the positions."""
        n = self.rope_dims
        if self.rope is None or n == 0:
            return x
        return jnp.concatenate([rotate(x[..., :n], self.rope, pos), x[..., n:]], -1)


def from_stream(
    config: cfg.SparseIndexerConfig,
    width: int,
    query_width: int,
    rope: cfg.RoPEConfig | None,
    rope_dims: int,
    rngs: nnx.Rngs,
    query: parallel.Name = "embed",
) -> Indexer:
    """Make DeepSeek-V3.2's indexer, for keys that are the tokens: its keys are
    projected from the layer's input and layer-normalized (unless `config.norm` is
    set), and rotated in the indexer's rope_layout whatever the layer's."""
    if rope is not None:
        rope = rope.model_copy(update={"layout": config.rope_layout})
    return Indexer(
        config,
        width,
        query_width,
        width,
        _STREAM_KEY_NORM,
        rope,
        rope_dims,
        rngs,
        query,
    )
