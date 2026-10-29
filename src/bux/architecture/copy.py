"""CPLM's copy pointer: a head's next-token distribution mixed with a pointer over
the earlier tokens of each document (modded-nanogpt's CPLM).

    p(y) = p_lm(y) (1 - a (1 - a_sink)) + a p_copy(y)

- a: the head's softmax mass on the ``<copy>`` row, a row of the vocabulary that no
  token is (`~bux.config.CopyPointerConfig`); p_lm: the softmax over the others.
- p_copy(y): the attention of one head over the positions before the query in its
  document on those whose token is y; a_sink: its attention on a learned sink,
  whose share goes back to p_lm.

A head with a pointer (`CopyPointer`) leaves a `Pointer` in its output, and the loss
and the evals take the mixture's negative log-likelihood (`Pointer.sums`), in log
space, its logits a chunk of positions at a time as ``methods.loss = chunks`` takes
the softmax's: bits per byte are then those of the mixture. Its pointer's scores
are float32, a chunk of queries at a time against every key of the row.
"""

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import config as cfg
from bux import initializers, parallel
from bux.architecture import observe
from bux.architecture.common.sharding import linear
from bux.architecture.logits import (
    Chunks,
    Grads,
    Projection,
    Sums,
    _chunk_grad,
    _gathered,
    _scaled,
    _shard,
    _whole,
)
from bux.architecture.registry import Taps, segments

if TYPE_CHECKING:
    from bux.architecture.logits import HeadOutput

#: The most queries of a row whose scores the pointer makes at once.
QUERY_CHUNK = 512
#: A window that sees the whole row.
WHOLE = 2**30


class Pointer(NamedTuple):
    """What a head's copy pointer leaves the loss, at each position of the pass."""

    query: Float[Array, "b t c"]  #: Its query, RMS-normed, times the gain.
    key: Float[Array, "b t c"]  #: Its key, RMS-normed.
    sink: Float[Array, "b t"]  #: Its query's logit of the sink, in float32.
    tokens: Int[Array, "b t"]  #: The token at each position, which a source holds.
    docs: Int[Array, "b t"]  #: The document of each position in its row.
    copy: Int[Array, ""]  #: The <copy> row of the vocabulary.
    window: Int[Array, ""]  #: How far back a query looks, in positions.

    def logs(
        self, targets: Int[Array, "b t"]
    ) -> tuple[Float[Array, "b t"], Float[Array, "b t"]]:
        """Return each position's log p_copy of its target, -inf where no earlier
        position of its document holds it, and its log a_sink.

        A query attends to the positions strictly before it in its document, within
        its window, and to the sink; its scores are ``q . k / sqrt(c)``, in float32,
        a chunk of `QUERY_CHUNK` queries at a time, made again in the backward.

        Raises:
            NotImplementedError: Positions split over ``seq``, whose keys the
                queries of another device would need.
        """
        b, t, c = self.query.shape
        if parallel.splits(("seq",), (t,))[0] > 1:
            raise NotImplementedError("a copy pointer with positions split over seq")
        n = max(d for d in range(1, min(t, QUERY_CHUNK) + 1) if t % d == 0)
        at = jnp.broadcast_to(jnp.arange(t), (b, t))

        def chunks(a: Shaped[Array, "b t *r"]) -> Shaped[Array, "k b n *r"]:
            return jnp.moveaxis(a.reshape(b, t // n, n, *a.shape[2:]), 1, 0)

        @jax.checkpoint
        def chunk(xs: tuple) -> tuple[Float[Array, "b n"], Float[Array, "b n"]]:
            q, sink, doc, pos, target = xs
            scores = jnp.einsum(
                "bnc,btc->bnt", q, self.key, preferred_element_type=jnp.float32
            )
            scores = scores * c**-0.5
            behind = pos[..., None] - jnp.arange(t)
            valid = (behind > 0) & (behind <= self.window)
            valid = valid & (doc[..., None] == self.docs[:, None, :])
            match = valid & (target[..., None] == self.tokens[:, None, :])
            every = logadd(masked_lse(scores, valid), sink)
            return masked_lse(scores, match) - every, sink - every

        xs = tuple(chunks(a) for a in (self.query, self.sink, self.docs, at, targets))
        log_copy, log_sink = jax.lax.map(chunk, xs)
        return tuple(  # ty: ignore[invalid-return-type]
            jnp.moveaxis(a, 0, 1).reshape(b, t) for a in (log_copy, log_sink)
        )

    def sums(
        self,
        head: "HeadOutput",
        targets: Int[Array, "b t"],
        weights: Float[Array, "b t"],
        z: bool,
    ) -> Sums:
        """Return the weighted sums of the mixture's negative log-likelihood of
        `targets`, and of the z-loss (of the head's softmax, <copy> in it) if `z`:
        the head's logits a chunk of positions at a time (``Chunks``), their
        gradients made with them, as ``methods.loss = chunks`` makes the softmax's
        (`_mixture_sums`)."""
        log_copy, log_sink = self.logs(targets)
        x, p = head.hidden, head.projection
        given = (x, targets, weights, log_copy, log_sink, self.copy)
        return _mixture_sums(p, *given, z or observe.active() is not None)


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _mixture_sums(
    p: Projection,
    x: Float[Array, "b t d"],
    targets: Int[Array, "b t"],
    weights: Float[Array, "b t"],
    log_copy: Float[Array, "b t"],
    log_sink: Float[Array, "b t"],
    copy: Int[Array, ""],
    z: bool,
) -> Sums:
    """Return the mixture's sums, a chunk at a time, without their gradients."""
    chunks, whole = Chunks.of(x.shape[0], x.shape[1], p.kernel.shape[1]), _gathered(p)

    def step(total: Sums, xs: tuple) -> tuple[Sums, None]:
        x, target, weight, log_c, log_s = xs
        logits = _shard(whole(x), None, "vocab")
        nll, lse = mixture_nll(logits, target, copy, log_c, log_s)
        sums = Sums(jnp.sum(weight * nll), jnp.sum(weight * lse**2) * z)
        return jax.tree.map(jnp.add, total, sums), None

    given = (x, targets, weights, log_copy, log_sink)
    start = Sums(jnp.zeros(()), jnp.zeros(()))
    return jax.lax.scan(step, start, tuple(chunks.split(a) for a in given))[0]


def _mixture_sums_fwd(
    p: Projection,
    x: Float[Array, "b t d"],
    targets: Int[Array, "b t"],
    weights: Float[Array, "b t"],
    log_copy: Float[Array, "b t"],
    log_sink: Float[Array, "b t"],
    copy: Int[Array, ""],
    z: bool,
) -> tuple[Sums, tuple]:
    """Return the sums and each one's gradients, a chunk at a time: of the head's
    kernel, added up in float32, of its bias, of x, and the mixture's of the
    pointer's logs."""
    b, t, d = x.shape
    vocab = p.kernel.shape[1]
    chunks, whole = Chunks.of(b, t, vocab), _gathered(p)
    bias = None if whole.bias is None else whole.bias.astype(jnp.float32)

    def step(carry: tuple, xs: tuple) -> tuple[tuple, tuple]:
        sums, kernels = carry
        chunk, grads, logs = _mixture_grads(whole, bias, *xs, copy, z)
        sums = jax.tree.map(jnp.add, sums, chunk)
        kernels = jax.tree.map(jnp.add, kernels, [(g.kernel, g.bias) for g in grads])
        return (sums, kernels), ([g.x for g in grads], logs)

    zeros = _shard(jnp.zeros((chunks.rows, chunks.seq, d, vocab)), None, "vocab")
    kernel = (zeros, None if bias is None else jnp.zeros_like(bias))
    start = (Sums(jnp.zeros(()), jnp.zeros(())), [kernel] * (1 + z))
    given = (x, targets, weights, log_copy, log_sink)
    (sums, kernels), (dxs, logs) = jax.lax.scan(
        step, start, tuple(chunks.split(a) for a in given)
    )
    grads = [_whole(p, *k, chunks.join(dx, b, t)) for k, dx in zip(kernels, dxs)]
    logs = tuple(chunks.join(g, b, t) for g in logs)
    return sums, (grads[0], grads[1] if z else None, logs)


def _mixture_sums_bwd(z: bool, saved: tuple, cotangent: Sums) -> tuple:
    """Return the gradients of `_mixture_sums`: each sum's, times its cotangent."""
    ce, z_grads, (d_copy, d_sink) = saved
    grads = _scaled(ce, cotangent.ce)
    if z_grads is not None:
        grads = jax.tree.map(jnp.add, grads, _scaled(z_grads, cotangent.z))
    logs = (cotangent.ce * d_copy, cotangent.ce * d_sink)
    return Projection(grads.kernel, grads.bias, None), grads.x, None, None, *logs, None


_mixture_sums.defvjp(_mixture_sums_fwd, _mixture_sums_bwd)


def _mixture_grads(
    p: Projection,
    bias: Float[Array, " vocab"] | None,
    x: Float[Array, "B C c d"],
    targets: Int[Array, "B C c"],
    weights: Float[Array, "B C c"],
    log_c: Float[Array, "B C c"],
    log_s: Float[Array, "B C c"],
    copy: Int[Array, ""],
    z: bool,
) -> tuple[Sums, list[Grads], tuple[Float[Array, "B C c"], Float[Array, "B C c"]]]:
    """Return a chunk's sums, their gradients from its logits (the mixture's, and the
    z-loss's if `z`), and the mixture's of the pointer's logs. The mixture reads
    three logits a position, the target's, <copy>'s and the logsumexp of the
    others', so its gradient of the logits is theirs: the others' softmax times
    the last one's."""
    raw = _shard(p.matmul(x), None, "vocab")
    logits, finish = jax.vjp(p.finish, raw, bias)
    logits = logits.astype(jnp.float32)
    rows = jnp.arange(logits.shape[-1])
    hit, is_copy = rows == targets[..., None], rows == copy
    left = hit | is_copy
    scores = (
        jnp.sum(jnp.where(hit, logits, 0), -1),
        jnp.sum(jnp.where(is_copy, logits, 0), -1),
        _others(logits, left),
    )
    nll, vjp = jax.vjp(_mixture, *scores, log_c, log_s)
    d_target, d_gate, d_others, d_log_c, d_log_s = vjp(weights)
    others = scores[2]
    finite = jnp.isfinite(others)
    rest = jnp.exp(logits - jnp.where(finite, others, 0)[..., None])
    d_logits = jnp.where(left | ~finite[..., None], 0, d_others[..., None] * rest)
    d_logits = d_logits + jnp.where(hit, d_target[..., None], 0)
    d_logits = d_logits + jnp.where(is_copy, d_gate[..., None], 0)
    lse = logadd(logadd(scores[0], others), scores[1])
    grads = [_chunk_grad(p, finish, x, d_logits)]
    if z:
        probs = jnp.exp(logits - lse[..., None])
        grads.append(_chunk_grad(p, finish, x, (2 * weights * lse)[..., None] * probs))
    sums = Sums(jnp.sum(weights * nll), jnp.sum(weights * lse**2) * z)
    return sums, grads, (d_log_c, d_log_s)


def log_probs(head: "HeadOutput") -> Float[Array, "b t vocab"]:
    """Return the log of the mixture over the vocabulary at every position of a head
    with a copy pointer, -inf at <copy>, which is no token: what a server samples
    from, in float32. It makes the pointer's attention whole, t by t, for short rows
    (the vLLM plugin's parity check).

    Raises:
        ValueError: If the head has no copy pointer.
    """
    pointer = head.pointer
    if pointer is None:
        raise ValueError("a head without a copy pointer: take its logits")
    logits = head.projection(head.hidden).astype(jnp.float32)
    _, t, vocab = logits.shape
    scores = (
        jnp.einsum(
            "bic,bjc->bij",
            pointer.query,
            pointer.key,
            preferred_element_type=jnp.float32,
        )
        * pointer.query.shape[-1] ** -0.5
    )
    behind = jnp.arange(t)[:, None] - jnp.arange(t)
    valid = (behind > 0) & (behind <= pointer.window)
    valid = valid & (pointer.docs[:, :, None] == pointer.docs[:, None, :])
    every = logadd(masked_lse(scores, valid), pointer.sink)
    share = jnp.where(valid, jnp.exp(scores - every[..., None]), 0)
    tokens = jax.nn.one_hot(pointer.tokens, vocab, dtype=jnp.float32)
    log_copy = jnp.log(jnp.einsum("bij,bjv->biv", share, tokens))
    log_sink = pointer.sink - every
    rows = jnp.arange(vocab)
    others = jnp.where(rows == pointer.copy, -jnp.inf, logits)
    lse_lm = jax.nn.logsumexp(others, -1, keepdims=True)
    gate = jnp.sum(jnp.where(rows == pointer.copy, logits, 0), -1, keepdims=True)
    lse = jnp.logaddexp(lse_lm, gate)
    kept = jnp.logaddexp(lse_lm - lse, gate - lse + log_sink[..., None])
    out = jnp.logaddexp(others - lse_lm + kept, gate - lse + log_copy)
    return jnp.where(rows == pointer.copy, -jnp.inf, out)


def mixture_nll(
    logits: Float[Array, "*n vocab"],
    targets: Int[Array, "*n"],
    copy: Int[Array, ""],
    log_copy: Float[Array, "*n"],
    log_sink: Float[Array, "*n"],
) -> tuple[Float[Array, "*n"], Float[Array, "*n"]]:
    """Return the mixture's negative log-likelihood of each target, and the log of
    the softmax's normalizer, <copy> in it, in float32 and in log space.

    With a the softmax's mass on `copy` and p_lm the softmax over the other rows,
    ``-log(p_lm(y) (1 - a + a a_sink) + a p_copy(y))``. Where a is 0, it is the
    cross-entropy of a softmax without the row, ``softplus(others - target)`` as
    `~bux.architecture.logits.Scores` takes it, to the bit.

    Args:
        logits: The head's logits, the <copy> row among them.
        targets: The token each position predicts, never `copy`.
        copy: The <copy> row.
        log_copy: Each position's log p_copy of its target (`Pointer.logs`).
        log_sink: Each position's log a_sink.
    """
    logits = logits.astype(jnp.float32)
    rows = jnp.arange(logits.shape[-1])
    hit, is_copy = rows == targets[..., None], rows == copy
    target = jnp.sum(jnp.where(hit, logits, 0), -1)
    gate = jnp.sum(jnp.where(is_copy, logits, 0), -1)
    others = _others(logits, hit | is_copy)
    nll = _mixture(target, gate, others, log_copy, log_sink)
    return nll, logadd(logadd(target, others), gate)


def _mixture(
    target: Float[Array, "*n"],
    gate: Float[Array, "*n"],
    others: Float[Array, "*n"],
    log_copy: Float[Array, "*n"],
    log_sink: Float[Array, "*n"],
) -> Float[Array, "*n"]:
    """Return the mixture's negative log-likelihood from the target's logit,
    <copy>'s, the logsumexp of the others' and the pointer's logs (`mixture_nll`)."""
    log_lm = -jax.nn.softplus(others - target)  # log p_lm(y), over the other rows
    lse_lm = logadd(target, others)  # the normalizer without <copy>
    lse = logadd(lse_lm, gate)
    log_gate = gate - lse
    kept = logadd(lse_lm - lse, log_gate + log_sink)  # log(1 - a + a a_sink)
    return -logadd(log_lm + kept, log_gate + log_copy)


def _others(
    logits: Float[Array, "*n vocab"], left: Bool[Array, "*n vocab"]
) -> Float[Array, "*n"]:
    """Return the logsumexp of the logits `left` leaves out, as
    `~bux.architecture.logits._others` takes it: -inf where they all underflow."""
    top = jax.lax.stop_gradient(jnp.max(logits, -1, keepdims=True))
    total = jnp.sum(jnp.where(left, 0, jnp.exp(logits - top)), -1)
    some = total > 0
    return jnp.where(some, top[..., 0] + jnp.log(jnp.where(some, total, 1)), -jnp.inf)


def logadd(a: Float[Array, "*n"], b: Float[Array, "*n"]) -> Float[Array, "*n"]:
    """Return log(e^a + e^b), whose gradient stays finite where one of them is -inf:
    the term's share is then 0. It is a, to the bit, where b is -inf."""
    top = jax.lax.stop_gradient(jnp.maximum(a, b))
    top = jnp.where(jnp.isfinite(top), top, 0)
    return top + jnp.log(jnp.exp(a - top) + jnp.exp(b - top))


def masked_lse(
    scores: Float[Array, "*n t"], mask: Bool[Array, "*n t"]
) -> Float[Array, "*n"]:
    """Return the logsumexp of the scores `mask` keeps, -inf where it keeps none,
    with a gradient that stays finite there."""
    some = jnp.any(mask, -1)
    top = jax.lax.stop_gradient(jnp.max(jnp.where(mask, scores, -jnp.inf), -1))
    top = jnp.where(some, top, 0)
    e = jnp.where(
        mask, jnp.exp(jnp.where(mask, scores, top[..., None]) - top[..., None]), 0
    )
    total = jnp.sum(e, -1)
    return jnp.where(some, top + jnp.log(jnp.where(some, total, 1)), -jnp.inf)


class CopyPointer(nnx.Module):
    """The copy pointer of a head (`~bux.config.CopyPointerConfig`): its query and
    key projections of the head's input, the query's gain, and the sink's key.

    The projections start by LeCun's uniform, as modded-nanogpt's, unless the
    model's ``init`` says otherwise; the gain at 1, the sink's key from a normal
    of ``dim ** -0.5``.

    Args:
        config: Its settings.
        width: The width of the stream the head reads.
        vocab: The size of the vocabulary, whose last row is <copy> by default.
        rngs: Draws the initial parameters.
    """

    def __init__(
        self, config: cfg.CopyPointerConfig, width: int, vocab: int, rngs: nnx.Rngs
    ):
        own = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")
        init = initializers.weight("hidden", own=own)
        dim = config.dim
        self.q = linear(
            width, dim, ("embed", None), False, rngs, in_compute=True, init=init
        )
        self.k = linear(
            width, dim, ("embed", None), False, rngs, in_compute=True, init=init
        )
        self.gain = nnx.Param(jnp.ones(()))
        sink = jax.random.normal(rngs.params(), (dim,)) * dim**-0.5
        self.sink = nnx.Param(sink)
        self.copy = vocab - 1 if config.token is None else config.token
        if not 0 <= self.copy < vocab:
            raise ValueError(f"copy_pointer(token={self.copy}) is no row of {vocab}")
        self.window = WHOLE if config.window is None else config.window

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Pointer:
        """Return the pointer's view of the head's input `x`, of the pass's tokens and
        documents.

        Raises:
            ValueError: If the pass has no tokens, which the pointer copies.
        """
        if taps.tokens is None:
            raise ValueError("a copy pointer copies the pass's tokens, which it lacks")
        b, t, _ = x.shape
        q = _rms_normed(self.q(x)) * self.gain[...].astype(x.dtype)
        k = _rms_normed(self.k(x))
        sink = (
            jnp.einsum(
                "btc,c->bt",
                q,
                self.sink[...].astype(q.dtype),
                preferred_element_type=jnp.float32,
            )
            * q.shape[-1] ** -0.5
        )
        docs = segments(taps)
        ids = jnp.ones((b, t), jnp.int32) if docs is None else docs.ids
        copy, window = (jnp.asarray(v, jnp.int32) for v in (self.copy, self.window))
        return Pointer(q, k, sink, taps.tokens, ids, copy, window)


def _rms_normed(x: Float[Array, "b t c"]) -> Float[Array, "b t c"]:
    """Return x over its RMS, computed in float32, as torch's ``rms_norm`` without a
    gain: an epsilon of float32's."""
    y = x.astype(jnp.float32)
    ms = jnp.mean(y * y, -1, keepdims=True)
    return (y * jax.lax.rsqrt(ms + jnp.finfo(jnp.float32).eps)).astype(x.dtype)
