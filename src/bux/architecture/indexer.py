"""The lightning indexer of DeepSeek sparse attention, which picks the keys each query
attends to, and its loss.

Keys sit in slots along a row, in order, and `order` says which key of its document
each slot holds (-1 for a slot with none, as between compressed latents). Blocks of
keys are counted from a document's first key, so none spans two documents.
"""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import config as cfg
from bux import parallel
from bux.architecture.norm import Norm
from bux.architecture.rope import rotate
from bux.architecture.sharding import projection, shard_heads

# DeepSeek-V3.2's indexer keys, projected from the layer's input, are layer-normalized.
_STREAM_KEY_NORM = cfg.NormalizationConfig(type="layernorm", bias=True)


class Scores(NamedTuple):
    """The indexer's scores and which of them it kept, for its loss: one per key, or
    with `block` (which block each key is in) one per block of keys, at the block's
    index."""

    values: Float[Array, "b t n"]
    kept: Bool[Array, "b t n"]
    block: Int[Array, "b s"] | None = None


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


def _spread(
    per: Shaped[Array, "b t n"], block: Int[Array, "b s"]
) -> Shaped[Array, "b t s"]:
    """Return, for each slot, what its block has."""
    b, t, _ = per.shape
    index = jnp.broadcast_to(block[:, None], (b, t, block.shape[1]))
    return jnp.take_along_axis(per, index, axis=-1)


def _top(scores: Float[Array, "b t n"], top_k: int) -> Bool[Array, "b t n"]:
    """Return each query's top_k highest scores, of those above -inf; of equal ones, the
    first."""
    n = scores.shape[-1]
    if min(top_k, n) == 0:
        return jnp.zeros(scores.shape, bool)
    # A stable sort and a scatter mapped over rows split with the rows; lax.top_k and
    # a scatter at broadcast indices gather every row onto each device.
    index = jnp.argsort(-scores, axis=-1, stable=True)[..., : min(top_k, n)]

    def pick(top: Int[Array, " top_k"]) -> Bool[Array, " n"]:
        return jnp.zeros(n, bool).at[top].set(True)

    return jax.vmap(jax.vmap(pick))(index) & (scores > -jnp.inf)


def _newest(
    mask: Bool[Array, "b t s"], block: Int[Array, "b s"]
) -> Bool[Array, "b t s"]:
    """Return whether each block (at its index) holds the newest key `mask` keeps for
    each query."""
    s = mask.shape[-1]
    last = jnp.argmax(jnp.where(mask, jnp.arange(s), -1), axis=-1)
    newest = jnp.take_along_axis(block, last, axis=1)
    return (jnp.arange(s) == newest[..., None]) & mask.any(-1, keepdims=True)


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
        b, t, _ = x.shape
        x, query = jax.lax.stop_gradient((x, query))
        q = self.q(query).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = shard_heads(self._rotate(q, pos))
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
        mask: Bool[Array, "b t s"],
        order: Int[Array, "b s"],
        candidates: Bool[Array, "b t s"] | None = None,
        export_candidates: bool = False,
    ) -> tuple[Bool[Array, "b t s"], Bool[Array, "b t s"] | None, Scores]:
        """Return the keys each query keeps, of those `mask` and the `candidates` (if
        given) allow; the candidate pool, if asked for; and the scores."""
        if candidates is not None:
            mask = mask & candidates
        if self.block_size > 1:
            keep_mask, scored = self._blocks(x, query, pos, keys, mask, order)
            return keep_mask, None, scored
        scores = self.scores(x, query, pos, keys)
        ranked = jnp.where(mask, jax.lax.stop_gradient(scores), -jnp.inf)
        keep_mask = _top(ranked, self.top_k)
        pool = self._candidates(ranked, order) if export_candidates else None
        return keep_mask, pool, Scores(scores, keep_mask)

    def loss(
        self, scores: Scores, attention: Float[Array, "b t s"]
    ) -> Float[Array, ""]:
        """Return KL(p || softmax(I)) over what the indexer kept, averaged over the
        queries (DeepSeek-V3.2, eq. 4): I its scores, and p the `attention` summed over
        heads and normalized over the kept keys. A block's p is its most attended key's
        (Qwen3.8 QSA, eqs. 17 and 20)."""
        p = jax.lax.stop_gradient(attention.astype(jnp.float32))
        if scores.block is not None:
            p = _per_block(p, scores.block, "max")
        return _kl(p, scores.values.astype(jnp.float32), scores.kept).mean()

    def _blocks(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"],
        mask: Bool[Array, "b t s"],
        order: Int[Array, "b s"],
    ) -> tuple[Bool[Array, "b t s"], Scores]:
        """Keep whole blocks of block_size keys: the top_k // block_size best of those
        `mask` keeps all of, and the masked-in keys of the query's own block until it
        is complete. Return the keep mask and the blocks' scores."""
        block = _groups(order, self.block_size)
        counts = _per_block(mask.astype(jnp.int32), block, "sum")
        full = counts == self.block_size
        scores = self.scores(x, query, pos, self._block_keys(keys, pos, order, block))
        ranked = jnp.where(full, jax.lax.stop_gradient(scores), -jnp.inf)
        picked = _top(ranked, self.top_k // self.block_size)
        keep_mask = _spread(picked | (_newest(mask, block) & ~full), block) & mask
        return keep_mask, Scores(scores, picked, block)

    def _block_keys(
        self,
        keys: Float[Array, "b s i"],
        pos: Int[Array, "b s"] | None,
        order: Int[Array, "b s"],
        block: Int[Array, "b s"],
    ) -> Float[Array, "b s i"]:
        """Return each block's key, at its index: the mean of its block_size keys,
        normalized, then rotated to the position of its first key's slot."""
        assert self.k_norm is not None
        exists = (order >= 0)[:, None, :]
        sums = _per_block(keys.transpose(0, 2, 1) * exists, block, "sum")
        means = self.k_norm(sums.transpose(0, 2, 1) / self.block_size)
        if pos is None:
            pos = jnp.broadcast_to(jnp.arange(order.shape[1]), order.shape)
        first = jnp.where(_starts(order, self.block_size), pos, 0)[:, None]
        first = _per_block(first, block, "sum")[:, 0]
        return self._rotate(means[:, :, None], first)[:, :, 0]

    def _candidates(
        self, scores: Float[Array, "b t s"], order: Int[Array, "b s"]
    ) -> Bool[Array, "b t s"]:
        """Return the candidate pool: the candidate_top_k best blocks of
        candidate_block_size keys, each scored by its best key, and always the block of
        the newest key a query may attend to, which may not be full yet."""
        block = _groups(order, self.candidate_block_size)
        mask = scores > -jnp.inf
        best = jnp.where(
            _newest(mask, block), jnp.inf, _per_block(scores, block, "max")
        )
        return _spread(_top(best, self.candidate_top_k), block) & mask

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
