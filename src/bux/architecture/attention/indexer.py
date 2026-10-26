"""The lightning indexer of DeepSeek sparse attention, which picks the keys each query
attends to, and its loss.

Keys sit in slots along a row, in order, and `order` says which key of its document
each slot holds (-1 for a slot with none, as between compressed latents). Blocks of
keys are counted from a document's first key, so none spans two documents. With
``per_group`` a query picks for each kv head's group of query heads apart, and its
scores and picks have a group axis, ``(b, t, g, ...)``, after the queries'.

`Indexer` ranks a chunk of queries at a time
(`~bux.architecture.attention.sparse.chunked`), so that no t by t floats are made: on
a GPU by `~bux.architecture.attention.scoring`'s kernel. It keeps each query's top-k
by index (`~bux.architecture.attention.sparse.Kept`), on a B200 or B300 by cuDNN's
radix kernel (`~bux.architecture.attention.indexer_cudnn`), named ``attention.index``
for a remat policy to keep. Its loss scores the kept keys again: on a B200 or B300 by
cuDNN's kernels; on another GPU by `~bux.architecture.attention.scoring`'s, which
gather each query's up to its last, and whose backward pass adds their gradients into
the keys atomically.
"""

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Shaped, UInt

from bux import config as cfg
from bux import parallel
from bux.architecture.attention import indexer_cudnn, scoring, sparse
from bux.architecture.attention.rope import rotate
from bux.architecture.attention.sparse import (
    ELEMENTS,
    Kept,
    Span,
    chunked,
    chunks,
    mask,
)
from bux.architecture.common.sharding import projection, shard_heads
from bux.architecture.layers.norm import Norm
from bux.architecture.repeats import remat
from bux.softmax import log_softmax
from bux.train import flops

# DeepSeek-V3.2's indexer keys, projected from the layer's input, are layer-normalized.
_STREAM_KEY_NORM = cfg.NormalizationConfig(type="layernorm", bias=True)
# The most picks of a chunk of queries the loss's cuDNN kernels take: in bigger chunks
# XLA keeps up to 10 GiB more alive (4096 x 8 rows) for 3% less time.
LOSS_PICKS = 1 << 23


class Scored(NamedTuple):
    """What the indexer kept by its scores, for its loss (`Indexer.loss`) to score
    again."""

    #: The layer's input, from which the indexer weighs its heads.
    x: Float[Array, "b t d"]
    #: What the indexer projects its queries from.
    query: Float[Array, "b t e"]
    #: Each query's position in its document.
    pos: Int[Array, "b t"]
    #: The keys it scored; with `block` and mean pooling, a block's at its index.
    keys: Float[Array, "b s i"]
    #: What each query kept of them, by index, for each group if per group; -1 for
    #: none.
    kept: Int[Array, "b t *g k"]
    #: Which block each key is in, if it kept blocks; else None.
    block: Int[Array, "b s"] | None = None
    #: The slots each query may attend to, for a loss over its kept keys' tokens.
    span: sparse.Span | None = None


Target = Callable[[Any], Float[Array, "b c *g s"]]  # a chunk's attention to the keys


def _kl(
    target: Float[Array, "b t *g n"],
    log_q: Float[Array, "b t *g n"],
    mask: Bool[Array, "b t *g n"],
) -> Float[Array, "b t *g"]:
    """Return every query's KL(target || q) over the slots `mask` keeps, q's log
    `log_q`, the target normalized over them: 0 for a query with none, as 0 log 0
    is."""
    p = jnp.where(mask, target, 0)
    p = p / jnp.maximum(p.sum(-1, keepdims=True), jnp.finfo(p.dtype).tiny)
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
    values: Shaped[Array, "b t *g s"],
    block: Int[Array, "b s"],
    reduce: Literal["sum", "max"],
) -> Shaped[Array, "b t *g s"]:
    """Return every query's sum or maximum of the values of each block's slots, at the
    block's index; -inf for the maximum of a block with none. A group axis joins the
    queries' for it: with the groups an axis of their own, XLA rematerializes the
    segment op in full under context parallelism."""
    if values.ndim == 4:
        b, t, g, s = values.shape
        joined = _per_block(values.reshape(b, t * g, s), block, reduce)
        return joined.reshape(b, t, g, s)
    op = jax.ops.segment_sum if reduce == "sum" else jax.ops.segment_max

    def row(v: Shaped[Array, "t s"], ids: Int[Array, " s"]) -> Shaped[Array, "t s"]:
        return op(v.T, ids, num_segments=v.shape[-1]).T

    return jax.vmap(row)(values, block)


def _like(
    seen: Bool[Array, "b t s"], scores: Shaped[Array, "b t *g s"]
) -> Bool[Array, "b t ... s"]:
    """Return `seen`, of every query, with the group axis of `scores` if it has one,
    of size 1."""
    return seen if scores.ndim == seen.ndim else seen[:, :, None]


def _top(
    scores: Float[Array, "b t *g n"],
    top_k: int,
    span: Span | None = None,
    cudnn: bool = True,
) -> Int[Array, "b t *g k"]:
    """Return the indices of each query's top_k highest scores, of those above -inf,
    in order of index, and -1 past them; of equal ones, the first. No score is above
    -inf outside a query's `span` (None: every slot's).

    On a B200 or B300 cuDNN's kernel keeps them (`indexer_cudnn`), where `cudnn`
    says it may: not for MSA's picks (`Indexer._by_cudnn`). Else XLA finds
    the k-th highest by its bits, one count of those at least as high a bit, and
    keeps those above it and the first of those equal to it: XLA's top-k sorts every
    row on a GPU, 6 times slower at 32k keys, and in the global view gathers every
    row onto each device."""
    n = scores.shape[-1]
    k = min(top_k, n)
    if cudnn and indexer_cudnn.runs(k):
        rows = scores.shape[:-1]
        every = Span(jnp.zeros(rows, jnp.int32), jnp.full(rows, n, jnp.int32))
        start, end = every if span is None else span
        return _first(indexer_cudnn.top_k(scores, start, end, k))
    key = _ordered(scores)

    def bit(i: Int[Array, ""], kth: UInt[Array, "b t *g"]) -> UInt[Array, "b t *g"]:
        higher = kth | (jnp.uint32(1) << (31 - i).astype(jnp.uint32))
        enough = (key >= higher[..., None]).sum(-1) >= k
        return jnp.where(enough, higher, kth)

    kth = jax.lax.fori_loop(0, 32, bit, jnp.zeros(scores.shape[:-1], jnp.uint32))
    above, tied = key > kth[..., None], key == kth[..., None]
    room = k - above.sum(-1, keepdims=True)
    kept = (above | (tied & (jnp.cumsum(tied, -1) <= room))) & (scores > -jnp.inf)
    return _indices(kept, k)


def _weighed(
    dots: Float[Array, "b t h s"], weights: Float[Array, "b t h"], relu: bool = True
) -> Float[Array, "b t s"]:
    """Return the heads' dot products, ReLU'd if `relu`, summed by their weights."""
    return jnp.einsum("bths,bth->bts", jax.nn.relu(dots) if relu else dots, weights)


def _dense(k: int, n: int) -> bool:
    """Return whether the loss scores all n keys again rather than gather the k each
    query keeps: when it keeps a quarter of them or more. The gather's gradient adds
    into a key once for every query that keeps it, and on a B300 those adds cost more
    than scoring every key below that (at 4k keys, top 2048: 5 times as much)."""
    return 4 * k >= n


def _ordered(x: Float[Array, "*s"]) -> UInt[Array, "*s"]:
    """Return unsigned integers in the order of the floats `x`."""
    u = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    return jnp.where(u >> 31 == 1, ~u, u | jnp.uint32(1 << 31))


def _indices(kept: Bool[Array, "b t *g n"], k: int) -> Int[Array, "b t *g k"]:
    """Return the indices of each row's (at most) k kept slots, -1 past them."""
    n = kept.shape[-1]
    slot = jnp.where(kept, jnp.cumsum(kept, -1) - 1, k)

    def row(slot: Int[Array, " n"]) -> Int[Array, " k"]:
        index = jnp.arange(n, dtype=jnp.int32)
        return jnp.full(k + 1, -1, jnp.int32).at[slot].set(index)[:k]

    for _ in range(kept.ndim - 1):
        row = jax.vmap(row)
    return row(slot)


def _first(index: Int[Array, "b t k"]) -> Int[Array, "b t k"]:
    """Return each row's indices but -1 first, in their order, then the -1s."""
    at = _indices(index >= 0, index.shape[-1])
    return jnp.where(at >= 0, jnp.take_along_axis(index, jnp.maximum(at, 0), -1), -1)


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
    """The lightning indexer of DeepSeek sparse attention (DSA, DeepSeek-V3.2): it
    picks the keys each query of a sparse attention layer attends to.

    A small attention with ``num_heads`` query heads and one key per slot, it scores
    every key a query may attend to by ``sum_j w_j ReLU(q_j . k_s)`` and keeps the
    ``top_k`` best. The weights ``w_j`` come from the input when it is ``weighted``;
    else the heads are summed. Its keep mask passes no gradient and its inputs are
    detached, so only `loss` trains it, and that loss trains nothing else.

    With ``block_size > 1`` it scores blocks of keys and keeps whole ones. Pooled by
    their mean (Qwen3.8 QSA), a block's key is the mean of its keys as projected,
    normalized, then rotated to the position of its first; a query scores the blocks
    it may attend to all of, and also keeps the keys of its own unfinished block.
    Pooled by their max (MiniMax MSA), a block scores the best of its keys the query
    may attend to, and the query keeps its own block among its ``top_k //
    block_size``. It can export a candidate pool of coarse blocks, and keep keys only
    inside one it imports (DeepSeek-V4.1).

    Without ``relu`` the dot products are plain. With ``per_group`` each of its heads
    picks for the group of the layer's query heads that share a kv head, by its own
    scores, unweighted (MiniMax MSA).

    ``config.norm``, if set, normalizes each query head, and the keys instead of
    `key_norm`.

    Its query heads split over ``heads``, as the layer's do, or ``kv_heads`` with
    ``per_group``; its keys, a single head, are every device's.

    Args:
        config: the indexer's settings.
        width: the width d of the layer's input, from which it weighs its heads.
        query_width: the width of what it projects its queries from: the layer's
            input, or its query latent.
        key_width: the width of what it projects its keys from (`keys`); None when
            it takes its keys from another layer, as CSA's compressed latents.
        key_norm: the keys' norm, unless ``config.norm`` sets one.
        rope: the RoPE settings of its queries and keys; None for none.
        rope_dims: how many of their first dims RoPE rotates.
        rngs: draws the initial parameters.
        query: the name of its queries' input dimension (`bux.parallel`):
            ``embed``, or ``latent`` for a query latent.
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
        h, i = config.num_heads, config.head_dim
        self.heads, self.head_dim = h, i
        self.top_k, self.block_size = config.top_k, config.block_size
        self.max_pool = config.block_size > 1 and config.pool == "max"
        self.per_group, self.relu = config.per_group, config.relu
        self.split: parallel.Name = "kv_heads" if config.per_group else "heads"
        self.candidate_block_size = config.candidate_block_size
        self.candidate_top_k = config.candidate_top_k
        self.loss_weight = config.loss_weight

        def linear(
            n_in: int, n_out: int, role: str, *names: parallel.Name
        ) -> nnx.Linear:
            init = parallel.named(*names)
            return projection(
                n_in, n_out, role, use_bias=False, kernel_init=init, rngs=rngs
            )

        # The randomly initialized layers; reordering them changes a seed's init.
        self.q = linear(query_width, h * i, "q", query, (self.split, h))
        self.k = linear(key_width, i, "kv", None, None) if key_width else None
        self.weights = (
            linear(width, h, "q", "embed", "heads") if config.weighted else None
        )

        self.q_norm = Norm(config.norm, i) if config.norm else None
        pools = self._mean_blocks()  # it normalizes the blocks of keys it is given
        self.k_norm = Norm(config.norm or key_norm, i) if key_width or pools else None
        self.rope = None if rope is None else rope.model_copy(update={"fraction": 1.0})
        self.rope_dims = min(rope_dims, i) // 2 * 2

    def keys(
        self, source: Float[Array, "b s e"], pos: Int[Array, "b s"] | None
    ) -> Float[Array, "b s i"]:
        """Project the indexer's keys, one per slot, from `source`.

        Args:
            source: what the keys are projected from, ``key_width`` wide: the layer's
                input, or CSA's compressed latents. It passes no gradient.
            pos: each slot's position, which RoPE rotates its key by; None for its
                index in the row.

        Returns:
            The keys, ``head_dim`` wide, as projected if the indexer mean-pools them
            in blocks, else normalized and rotated.
        """
        assert self.k is not None and self.k_norm is not None
        k = self.k(jax.lax.stop_gradient(source))
        if self._mean_blocks():
            return k
        return self._rotate(self.k_norm(k)[:, :, None], pos)[:, :, 0]

    def scores(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"],
    ) -> Float[Array, "b t *g s"]:
        """Return every query's score for every key, ``sum_j w_j ReLU(q_j . k_s)``, by
        XLA: a t by s intermediate of every head. Without ``relu`` the dot products
        are plain; with ``per_group`` each head's are the scores of its group.

        Args:
            x: the layer's input, from which the head weights come.
            query: what the queries are projected from.
            pos: each query's position; None for its index in the row.
            keys: the keys, from `keys`.

        Returns:
            The scores, (b, t, s), or (b, t, g, s) for g groups with ``per_group``.
            They pass no gradient to `x` or `query`.
        """
        x, query = jax.lax.stop_gradient((x, query))
        q, w = self._queries(query, pos), self._head_weights(x)
        with flops.over_keys():
            dots = jnp.einsum("bthi,bsi->bths", q, keys)
            if self.per_group:
                return (jax.nn.relu(dots) if self.relu else dots) * w[..., None]
            return _weighed(dots, w, self.relu)

    def _scores_at(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        index: Int[Array, "b t k"],
    ) -> Float[Array, "b t k"]:
        """Return each query's scores for the keys at its `index`, its kept ones: by
        ``scoring``'s kernels where they run, else gathered by XLA in float32, as the
        gradient adds into a key once for every query that keeps it, which in
        bfloat16 a GPU does one at a time."""
        x, query = jax.lax.stop_gradient((x, query))
        q, w = self._queries(query, pos), self._head_weights(x)
        if self._picked_by_kernel(index.shape[-1]):
            return scoring.picked(q, w, keys, index)
        rows = jax.vmap(lambda k, i: k[i])
        picked = rows(keys.astype(jnp.float32), jnp.maximum(index, 0))
        with flops.over_picks():
            return _weighed(jnp.einsum("bthi,btki->bthk", q, picked), w, self.relu)

    def _ranking(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        end: Int[Array, "b t"],
    ) -> Float[Array, "b t *g s"]:
        """Return the scores that rank the keys, passing no gradient: by
        ``scoring``'s kernel where it runs. Keys from a query's `end` on score
        anything."""
        x, query, keys = jax.lax.stop_gradient((x, query, keys))
        if not self._ranked_by_kernel(x.shape[1], keys.shape[1]):
            return self.scores(x, query, pos, keys)
        q, w = self._queries(query, pos), self._head_weights(x)
        return scoring.scores(q, w, keys, end)

    def _ranked_by_kernel(self, t: int, s: int) -> bool:
        """Return whether ``scoring``'s kernel ranks s keys for t queries: it sums the
        heads' ReLU'd dot products, as a pick for all heads with ``relu`` does."""
        sums = self.relu and not self.per_group
        return sums and scoring.runs(t, s, self.head_dim)

    def _per_query(self, t: int, s: int) -> int:
        """Return the elements a query makes in the largest intermediate of ranking s
        keys: those of every head unless the kernel runs, and its picks' for each
        group."""
        groups = self.heads if self.per_group else 1
        kernel = self._ranked_by_kernel(t, s)
        return (4 * groups if kernel else self.heads + 4 * groups) * s

    def _queries(
        self, query: Float[Array, "b t e"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h i"]:
        """Project, normalize and rotate the query heads."""
        b, t, _ = query.shape
        q = self.q(query).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        return shard_heads(self._rotate(q, pos), self.split)

    def _head_weights(self, x: Float[Array, "b t d"]) -> Float[Array, "b t h"]:
        """Return each head's weight in a query's scores, with the head dims' scale."""
        scale = self.head_dim**-0.5
        if self.weights is None:
            return jnp.full((*x.shape[:2], self.heads), scale, x.dtype)
        return self.weights(x) * (scale * self.heads**-0.5)

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
        """Pick the keys each query keeps: its top-k of those it may attend to.

        Args:
            x: the layer's input, b rows of t queries, d wide.
            query: what the queries are projected from.
            pos: each query's position in its document; None for its index in the
                row.
            keys: the keys, one a slot, from `keys`.
            span: the slots each query may attend to.
            order: which key of its document each slot holds; -1 for none.
            candidates: a candidate pool another layer exported, which the kept keys
                must lie inside; None for none.
            export_candidates: whether to return a candidate pool of this indexer's.

        Returns:
            The keys each query keeps, by index (with ``block_size > 1``, whole
            blocks), for each group with ``per_group``; the candidate pool, if asked
            for, else None; and what it scored, for `loss`.
        """
        x, query = jax.lax.stop_gradient((x, query))
        pos = _positions(pos, x)
        if self.block_size > 1:
            blocks = self._max_blocks if self.max_pool else self._blocks
            kept, scored = blocks(x, query, pos, keys, span, order)
            return kept, None, scored
        s, exists = keys.shape[1], order >= 0
        ranked_keys = jax.lax.stop_gradient(keys)
        pooled = _groups(order, self.candidate_block_size)
        among = None if candidates is None else candidates.block

        def pick(
            xs: tuple,
        ) -> tuple[Int[Array, "b c *g k"], Int[Array, "b c p"] | None]:
            x, query, pos, span, index = xs
            inside = None if index is None else Kept(index, among)
            seen = mask(span, s, exists, inside)
            scores = self._ranking(x, query, pos, ranked_keys, span.end)
            ranked = jnp.where(_like(seen, scores), scores, -jnp.inf)
            top = _top(ranked, self.top_k, span, self._by_cudnn())
            if not export_candidates:
                return top, None
            return top, self._candidates(ranked, pooled)

        index = None if candidates is None else candidates.index
        xs, t = (x, query, pos, span, index), x.shape[1]
        index, best = chunked(pick, xs, chunks(t, self._per_query(t, s)))
        index = remat.label("attention.index", index)
        pool = None if best is None else Kept(best, pooled)
        return Kept(index), pool, Scored(x, query, pos, keys, index, span=span)

    def loss(
        self, scored: Scored, target: Target, xs: Any, width: int, picks: bool = False
    ) -> Float[Array, ""]:
        """Return the indexer's loss: ``KL(p || softmax(I))`` over what it kept,
        averaged over the queries (DeepSeek-V3.2, eq. 4).

        I is the indexer's scores, and p the attention summed over heads and
        normalized over the kept keys. A block's p is its most attended key's (Qwen3.8
        QSA, eqs. 17 and 20). It scores each chunk of queries' kept keys again, and so
        does its backward pass; p it takes once. With ``per_group`` or max-pooled
        blocks, the loss is over each kept key the query sees, for each group
        (MiniMax MSA).

        Args:
            scored: what the indexer kept, from its call.
            target: gives a chunk of queries' attention to the keys, from that
                chunk of `xs`, for each group with ``per_group``; it passes no
                gradient.
            xs: what `target` takes, arrays of (b, t, ...) split into chunks of
                queries.
            width: the elements a query makes of `target`'s output, which sets the
                chunks.
            picks: whether `target` gives the attention to each query's kept keys,
                in their order, rather than to every key (CSA's, by cuDNN's kernel).

        Returns:
            The loss, a scalar, which trains only the indexer's parameters.
        """
        if self.per_group or self.max_pool:
            return self._token_loss(scored, target, xs, width)

        def kept_target(both: tuple) -> Float[Array, "b c k"]:
            (*_, kept), xs = both
            p = target(xs).astype(jnp.float32)
            if picks:
                return p
            if scored.block is not None:
                p = _per_block(p, scored.block, "max")
            return jnp.take_along_axis(p, jnp.maximum(kept, 0), axis=-1)

        def kl(both: tuple) -> Float[Array, "b c"]:
            (x, query, pos, kept), p = both
            log_q = self._log_predicted(x, query, pos, scored.keys, kept)
            return _kl(p, log_q, kept >= 0)

        own = (scored.x, scored.query, scored.pos, scored.kept)
        b, t, _ = scored.x.shape
        k, n = scored.kept.shape[-1], scored.keys.shape[1]
        cudnn = self._loss_by_cudnn(k, scored.keys.dtype)
        dense = _dense(k, n) and not cudnn
        per = self.heads * n if dense else k * max(self.heads, self.head_dim)
        count = chunks(t, max(per, width))
        p = jax.lax.stop_gradient(chunked(kept_target, (own, xs), count))
        if cudnn:
            count = chunks(t, b * k * ELEMENTS // LOSS_PICKS)
        elif not dense and self._picked_by_kernel(k):
            count = 1  # its kernels make no intermediate of every head
        return chunked(jax.checkpoint(kl), (own, p), count).mean()

    def _log_predicted(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        index: Int[Array, "b t k"],
    ) -> Float[Array, "b t k"]:
        """Return each query's log-softmax of its scores for its kept keys: by
        cuDNN's kernels where they take them, else of `_scores_of`'s scores."""
        if not self._loss_by_cudnn(index.shape[-1], keys.dtype):
            scores = self._scores_of(x, query, pos, keys, index).astype(jnp.float32)
            unkept = jnp.finfo(jnp.float32).min
            return log_softmax(jnp.where(index >= 0, scores, unkept))
        x, query = jax.lax.stop_gradient((x, query))
        q, w = self._queries(query, pos), self._head_weights(x)
        return indexer_cudnn.log_softmax(q.astype(keys.dtype), w, keys, index)

    def _loss_by_cudnn(self, k: int, dtype: Any) -> bool:
        """Return whether cuDNN's kernels score each query's k picks, of keys in
        `dtype`, for the loss, where `_by_cudnn` lets them: on a B200 or B300, faster
        than both ``scoring``'s kernels and scoring every key (`_dense`)."""
        if not self._by_cudnn():
            return False
        return indexer_cudnn.takes(self.heads, self.head_dim, k, dtype)

    def _by_cudnn(self) -> bool:
        """Return whether cuDNN's DSA kernels may pick and score for it
        (`indexer_cudnn`): one pick for all heads, of ReLU'd dot products, as DSA's
        and QSA's. MSA's settings (``pool=max``, ``per_group``, ``relu=false``) run
        by XLA."""
        return self.relu and not (self.per_group or self.max_pool)

    def _scores_of(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        index: Int[Array, "b t k"],
    ) -> Float[Array, "b t k"]:
        """Return each query's scores for its kept keys: of every key, then taken,
        where `_dense` says that is cheaper; else of the kept keys alone."""
        if not _dense(index.shape[-1], keys.shape[1]):
            return self._scores_at(x, query, pos, keys, index)
        every = self.scores(x, query, pos, keys)
        return jnp.take_along_axis(every, jnp.maximum(index, 0), axis=-1)

    def _token_loss(
        self, scored: Scored, target: Target, xs: Any, width: int
    ) -> Float[Array, ""]:
        """Return `loss` over the keys the query sees of those it kept, each its own
        token, for each group with ``per_group``, averaged over the queries and the
        groups (MiniMax MSA).

        p is the attention to them summed over the group's heads (every head without
        groups), the mean of their probabilities once normalized. A chunk of queries
        scores every key and takes p, and does both again in its backward pass, so
        that no t by t floats are kept.
        """
        assert scored.span is not None
        s = scored.keys.shape[1]

        def kl(both: tuple) -> Float[Array, "b c *g"]:
            (x, query, pos, kept, span), xs = both
            p = jax.lax.stop_gradient(target(xs)).astype(jnp.float32)
            scores = self.scores(x, query, pos, scored.keys).astype(jnp.float32)
            over = mask(span, s, None, Kept(kept, scored.block))
            unkept = jnp.finfo(jnp.float32).min
            return _kl(p, log_softmax(jnp.where(over, scores, unkept)), over)

        own = (scored.x, scored.query, scored.pos, scored.kept, scored.span)
        count = chunks(scored.x.shape[1], max(width, self.heads * s))
        return chunked(jax.checkpoint(kl), (own, xs), count).mean()

    def _mean_blocks(self) -> bool:
        """Return whether it pools the keys in blocks by their mean (Qwen3.8 QSA)."""
        return self.block_size > 1 and not self.max_pool

    def _picked_by_kernel(self, k: int) -> bool:
        """Return whether ``scoring``'s kernels score each query's k picks: they stop
        at its last, so a query of a short document takes few steps. They ReLU the
        dot products."""
        return self.relu and scoring.picked_runs(self.heads, self.head_dim, k)

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
            scores = self._ranking(x, query, pos, ranked_keys, span.end)
            blocks = jnp.where(full, scores, -jnp.inf)
            picked = _top(blocks, self.top_k // size, cudnn=self._by_cudnn())
            own = _newest(seen, block)
            unfinished = ~jnp.take_along_axis(full, jnp.maximum(own, 0)[..., None], -1)
            return picked, jnp.where(unfinished[..., 0], own, -1)

        xs, t = (x, query, pos, span), x.shape[1]
        picked, own = chunked(pick, xs, chunks(t, self._per_query(t, s)))
        picked, own = remat.label("attention.index", (picked, own))
        kept = Kept(jnp.concatenate([picked, own[..., None]], -1), block)
        return kept, Scored(x, query, pos, block_keys, picked, block)

    def _max_blocks(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"],
        keys: Float[Array, "b s i"],
        span: Span,
        order: Int[Array, "b s"],
    ) -> tuple[Kept, Scored]:
        """Keep whole blocks of block_size keys, each scored by the best of its keys
        the span has: the query's own block and the best others, top_k // block_size
        in all, for each group with ``per_group`` (MiniMax MSA). Return them, and what
        it scored."""
        s, size = keys.shape[1], self.block_size
        block = _groups(order, size)
        exists, ranked_keys = order >= 0, jax.lax.stop_gradient(keys)

        def pick(xs: tuple) -> Int[Array, "b c *g k"]:
            x, query, pos, span = xs
            seen = mask(span, s, exists)
            scores = self._ranking(x, query, pos, ranked_keys, span.end)
            scores = jnp.where(_like(seen, scores), scores, -jnp.inf)
            best = _per_block(scores, block, "max")
            own = jnp.arange(s) == _newest(seen, block)[..., None]
            best = jnp.where(_like(own, best), jnp.inf, best)
            return _top(best, self.top_k // size, cudnn=self._by_cudnn())

        xs, t = (x, query, pos, span), x.shape[1]
        kept = chunked(pick, xs, chunks(t, self._per_query(t, s)))
        kept = remat.label("attention.index", kept)
        return Kept(kept, block), Scored(x, query, pos, keys, kept, block, span)

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
        return _top(best, self.candidate_top_k, cudnn=self._by_cudnn())

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
    """Make DeepSeek-V3.2's indexer, for keys that are the tokens.

    Its keys are projected from the layer's input and layer-normalized (unless
    ``config.norm`` is set), and rotated in the indexer's ``rope_layout`` whatever the
    layer's.

    Args:
        config: the indexer's settings.
        width: the width d of the layer's input.
        query_width: the width of what it projects its queries from.
        rope: the layer's RoPE settings; None for none.
        rope_dims: how many of the queries' and keys' first dims RoPE rotates.
        rngs: draws the initial parameters.
        query: the name of its queries' input dimension, as `Indexer` takes it.

    Returns:
        The `Indexer`.
    """
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
