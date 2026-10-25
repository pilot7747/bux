"""The lightning indexer's scores of every key, ``sum_j w_j ReLU(q_j·k)``, with which
it ranks them (`~bux.architecture.attention.indexer`); and the attention's
probabilities of every key, summed over heads, which its loss learns from
(`bux.architecture.attention.sparse.probabilities`).

On a GPU a Pallas kernel computes them, one program per block of queries and of keys,
which keeps the block's scores in registers from head to head and skips keys after
every query's span; XLA would write and read every head's (queries, keys) instead.
Elsewhere, or at shapes the kernel does not take, XLA. The scores pass no gradient:
the indexer's loss scores its kept keys alone. On a mesh a `shard_map` gives each
device its rows, queries and heads, and sums the heads' scores.

The probabilities' kernel is the same, a head's exp(logit - lse) in place of its
weighted ReLU: XLA would write and read every head's logits, 16 floats for each one
the loss reads with 16 heads. The loss's scores of each query's picks are a program a
query that gathers its picks' keys, up to its last (`picked`), differentiated by
another that adds each pick's share of the keys' gradient into it atomically.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import jax_internals, kernels, parallel

QUERIES, KEYS = 64, 128  # the most of each a program takes
PROBABILITY_KEYS = 64  # of `probabilities`
PICKS = 64  # of `picked`, the most of a query's picks a step of its program takes
WARPS = 4
F32 = jnp.float32


def runs(t: int, s: int, i: int) -> bool:
    """Return whether `scores`' kernel runs: on a GPU, where blocks of a power of two
    (16 or more) divide the t queries and s keys, and the i head dims are a power of
    two, 16 or more."""
    pow2 = i >= 16 and i & (i - 1) == 0
    return kernels.gpu() and pow2 and None not in (_block(t, QUERIES), _block(s, KEYS))


def scores(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    end: Int[Array, "b t"],
) -> Float[Array, "b t s"]:
    """Return every query's score of every key by the kernel: the sum over heads of w
    times the ReLU'd dot product.

    Args:
        q: the indexer's queries: b rows of t, h heads of i dims.
        w: each query's head weights.
        keys: the keys, one a slot, i wide.
        end: the slot after each query's last; keys from it on score 0 or anything,
            as the query does not see them.

    Returns:
        The scores, (b, t, s) in float32. They pass no gradient, nor do the
        parameters behind them.
    """
    q, w, keys = jax.lax.stop_gradient((q, w, keys))  # nor the parameters behind them
    split = parallel.spec(("batch", "seq", "heads", None), q.shape)
    if not parallel.axes_of(split):
        return _kernel(q, w, keys, end)
    ins = (split, P(*split[:3]), P(split[0], None, None), P(split[0], split[1]))
    return on_own_queries(_kernel, split, ins)(q, w, keys, end)


def on_own_queries(
    kernel: Callable[..., Float[Array, "b t s"]], split: P, ins: tuple[Any, ...]
) -> Callable[..., Float[Array, "b t s"]]:
    """Return `kernel` in a shard_map manual over the axes of `split`.

    Each device scores its rows' queries by its heads, and the scores sum over the
    devices of the heads.

    Args:
        kernel: what makes each query's scores of the keys, (b, t, s).
        split: the queries' spec, (b, t, h, ...).
        ins: how the kernel's inputs split.

    Returns:
        The kernel on the whole arrays, its scores split by rows and queries.
    """
    heads = split[2]

    def device(*args: Any) -> Float[Array, "b t s"]:
        own = kernel(*args)
        return own if heads is None else jax.lax.psum(own, heads)

    return jax.shard_map(
        device,
        in_specs=ins,
        out_specs=P(split[0], split[1], None),
        axis_names=set(parallel.axes_of(split)),
        check_vma=False,  # a kernel's outputs do not say what varies
    )


def _block(n: int, most: int) -> int | None:
    """Return the largest power of two up to `most`, and 16 or more, that divides n."""
    size = most
    while size >= 16 and n % size:
        size //= 2
    return size if size >= 16 else None


def _kernel(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    end: Int[Array, "b t"],
) -> Float[Array, "b t s"]:
    """Run the kernel; interpreted off a GPU, as tests run it."""
    from jax.experimental.pallas import triton as plgpu

    b, t, h, i = q.shape
    s = keys.shape[1]
    bq, bk = _block(t, QUERIES), _block(s, KEYS)
    assert bq is not None and bk is not None

    # Pallas refs, which index and load as arrays do not.
    def kernel(end_ref: Any, q_ref: Any, w_ref: Any, k_ref: Any, out_ref: Any) -> None:
        seen = pl.program_id(2) * bk < jnp.max(end_ref[...])

        @pl.when(~seen)
        def _() -> None:
            out_ref[...] = jnp.zeros((bq, bk), jnp.float32)

        @pl.when(seen)
        def _() -> None:
            k = k_ref[...]

            def head(
                j: Int[Array, ""], acc: Float[Array, "bq bk"]
            ) -> Float[Array, "bq bk"]:
                dims = (((1,), (1,)), ((), ()))
                dots = jax.lax.dot_general(
                    q_ref[j], k, dims, preferred_element_type=jnp.float32
                )
                return acc + jnp.maximum(dots, 0) * w_ref[j][:, None]

            zeros = jnp.zeros((bq, bk), jnp.float32)
            out_ref[...] = jax.lax.fori_loop(0, h, head, zeros)

    queries = (None, bq)
    return pl.pallas_call(
        kernel,
        grid=(b, t // bq, s // bk),
        in_specs=[
            pl.BlockSpec(queries, lambda z, m, n: (z, m)),
            pl.BlockSpec((None, h, bq, i), lambda z, m, n: (z, 0, m, 0)),
            pl.BlockSpec((None, h, bq), lambda z, m, n: (z, 0, m)),
            pl.BlockSpec((None, bk, i), lambda z, m, n: (z, n, 0)),
        ],
        out_specs=pl.BlockSpec((None, bq, bk), lambda z, m, n: (z, m, n)),
        out_shape=jax.ShapeDtypeStruct((b, t, s), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
        interpret=kernels.interpreted(),
        name="indexer_scores",
    )(
        end.astype(jnp.int32),
        q.transpose(0, 2, 1, 3),
        w.transpose(0, 2, 1).astype(jnp.float32),
        keys.astype(q.dtype),
    )


# --- the attention's probabilities, the indexer's target ------------------------------


def probabilities_runs(t: int, s: int, d: int) -> bool:
    """Return whether `probabilities`' kernel runs: on a GPU, where blocks of a power
    of two divide the t queries and s keys, for heads of a power of two from 16 to 128
    dims. On a B300 it takes half XLA's time at 128 (GQA's, MQA's), and 1.4 times
    it at MLA's 192, padded to 256."""
    blocks = None not in (_block(t, QUERIES), _block(s, PROBABILITY_KEYS))
    pow2 = 16 <= d <= 128 and d & (d - 1) == 0
    return kernels.gpu() and blocks and pow2


def probabilities(
    q: Float[Array, "b t h d"],
    k: Float[Array, "b s g d"],
    lse: Float[Array, "b t h"],
    start: Int[Array, "b t"],
    end: Int[Array, "b t"],
    scale: float,
    cap: float | None,
) -> Float[Array, "b t s"]:
    """Return each query's attention to each key, summed over heads, by the kernel:
    ``exp(logit - lse)`` of each head, its logits scaled and soft-capped as the
    attention's.

    Args:
        q: the attention's queries: b rows of t, h heads of d dims.
        k: its keys: s slots of g kv heads. Query head j takes kv head
            ``j // (h / g)``, as in GQA.
        lse: each query's logsumexp in the attention's softmax.
        start: each query's first slot.
        end: the slot after each query's last.
        scale: the logits' scale.
        cap: the logits' tanh soft cap; None for none.

    Returns:
        The probabilities, (b, t, s) in float32: 0 for a key outside the query's
        span. They pass no gradient.
    """
    q, k, lse = jax.lax.stop_gradient((q, k, lse))
    b, t, h, _ = q.shape
    s, g = k.shape[1], k.shape[2]
    bq, bk = _block(t, QUERIES), _block(s, PROBABILITY_KEYS)
    assert bq is not None and bk is not None
    d = q.shape[-1]

    # Pallas refs, which index and load as arrays do not.
    def kernel(
        start_ref: Any, end_ref: Any, q_ref: Any, k_ref: Any, lse_ref: Any, out_ref: Any
    ) -> None:
        first = pl.program_id(2) * bk
        start, end = start_ref[...], end_ref[...]
        seen = (first < jnp.max(end)) & (first + bk > jnp.min(start))

        @pl.when(~seen)
        def _() -> None:
            out_ref[...] = jnp.zeros((bq, bk), jnp.float32)

        @pl.when(seen)
        def _() -> None:
            def head(
                j: Int[Array, ""], acc: Float[Array, "bq bk"]
            ) -> Float[Array, "bq bk"]:
                dims = (((1,), (1,)), ((), ()))
                dots = jax.lax.dot_general(
                    q_ref[j], k_ref[j // (h // g)], dims, preferred_element_type=F32
                )
                logits = dots * scale
                if cap is not None:
                    logits = cap * jnp.tanh(logits / cap)
                return acc + jnp.exp(logits - lse_ref[j][:, None])

            total = jax.lax.fori_loop(0, h, head, jnp.zeros((bq, bk), F32))
            slot = first + jnp.arange(bk)[None, :]
            inside = (slot >= start[:, None]) & (slot < end[:, None])
            out_ref[...] = jnp.where(inside, total, 0.0)

    from jax.experimental.pallas import triton as plgpu

    queries = (None, bq)
    return pl.pallas_call(
        kernel,
        grid=(b, t // bq, s // bk),
        in_specs=[
            pl.BlockSpec(queries, lambda z, m, n: (z, m)),
            pl.BlockSpec(queries, lambda z, m, n: (z, m)),
            pl.BlockSpec((None, h, bq, d), lambda z, m, n: (z, 0, m, 0)),
            pl.BlockSpec((None, g, bk, d), lambda z, m, n: (z, 0, n, 0)),
            pl.BlockSpec((None, h, bq), lambda z, m, n: (z, 0, m)),
        ],
        out_specs=pl.BlockSpec((None, bq, bk), lambda z, m, n: (z, m, n)),
        out_shape=jax.ShapeDtypeStruct((b, t, s), F32),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
        interpret=kernels.interpreted(),
        name="attention_probabilities",
    )(
        start.astype(jnp.int32),
        end.astype(jnp.int32),
        q.transpose(0, 2, 1, 3),
        k.astype(q.dtype).transpose(0, 2, 1, 3),
        lse.transpose(0, 2, 1).astype(F32),
    )


# --- the scores of each query's picks, for the indexer's loss -------------------------


def picked_runs(h: int, i: int, k: int) -> bool:
    """Return whether `picked`'s kernels run: on a GPU, for h heads and i head dims of
    a power of two (16 or more), and k picks in blocks of a power of two, 16 or
    more."""
    pow2 = all(n >= 16 and n & (n - 1) == 0 for n in (h, i))
    return kernels.gpu() and pow2 and _block(k, PICKS) is not None


def picked(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> Float[Array, "b t k"]:
    """Return each query's scores of the keys it picked, by the kernels: the sum over
    heads of w times the ReLU'd dot product.

    A program a query gathers its picks' keys, a block at a time up to its last pick,
    and the backward pass adds each pick's share of the keys' gradient into it
    atomically. On a mesh a ``shard_map`` gives each device its rows and queries,
    every head and every key.

    Args:
        q: the indexer's queries: b rows of t, h heads of i dims.
        w: each query's head weights.
        keys: the keys, one a slot, i wide.
        index: each query's picks, k slots; -1 for none.

    Returns:
        The scores, (b, t, k) in float32; 0 for a pick of -1.
    """
    split = parallel.spec(("batch", "seq", None, None), q.shape)
    if not parallel.axes_of(split):
        return _picked(q, w, keys, index)
    rows = P(split[0], split[1])
    ins = (split, P(*split[:3]), P(split[0], None, None), P(*rows, None))
    return on_own_queries(_picked, split, ins)(q, w, keys, index)


@jax.custom_vjp
def _picked(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> Float[Array, "b t k"]:
    """Return `picked` on a device's queries, forward and backward by the kernels."""
    return _picked_scores(q, w, keys, index)


def _picked_fwd(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> tuple[Float[Array, "b t k"], tuple]:
    return _picked_scores(q, w, keys, index), (q, w, keys, index)


def _picked_bwd(saved: tuple, d: Float[Array, "b t k"]) -> tuple:
    q, w, keys, index = saved
    dq, dw, dkeys = _picked_grads(q, w, keys, index, d)
    return dq, dw.astype(w.dtype), dkeys.astype(keys.dtype), None


_picked.defvjp(_picked_fwd, _picked_bwd)


def _dot(
    a: Float[Array, "m n"], b: Float[Array, "o p"], over: tuple[int, int]
) -> Float[Array, "x y"]:
    """Return the product of a and b over their dims `over`, in float32."""
    dims = (((over[0],), (over[1],)), ((), ()))
    return jax.lax.dot_general(a, b, dims, preferred_element_type=F32)


def _blocks(picks: Int[Array, " k"], size: int) -> Int[Array, ""]:
    """Return how many blocks of `size` picks hold every pick but -1s after them."""
    last = jnp.max(jnp.where(picks >= 0, jnp.arange(picks.shape[0]) + 1, 0))
    return (last + size - 1) // size


def _rows(
    picks: Int[Array, " n"], width: int
) -> tuple[Int[Array, "n i"], Int[Array, "n i"]]:
    """Return the row and column of each element of the rows `picks` of a ref `width`
    wide, as Triton's loads and atomic adds take them: both of the same shape."""
    n = picks.shape[0]
    rows = jnp.broadcast_to(picks[:, None], (n, width))
    return rows, jnp.broadcast_to(jnp.arange(width)[None, :], (n, width))


def _gathered(keys_ref: Any, picks: Int[Array, " n"]) -> Float[Array, "n i"]:
    """Load the keys at `picks` (a pick of -1 the first key's) from a ref of a row's."""
    return keys_ref[_rows(jnp.maximum(picks, 0), keys_ref.shape[1])]


def _picked_scores(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> Float[Array, "b t k"]:
    """Run the forward kernel: a program a query, a block of its picks at a time."""
    from jax.experimental.pallas import triton as plgpu

    b, t, h, i = q.shape
    k, bk = index.shape[-1], _block(index.shape[-1], PICKS)
    assert bk is not None

    # Pallas refs, which index and load as arrays do not.
    def kernel(
        index_ref: Any, q_ref: Any, w_ref: Any, keys_ref: Any, out_ref: Any
    ) -> None:
        q, w = q_ref[...], w_ref[...]
        out_ref[...] = jnp.zeros((k,), F32)

        def block(n: Int[Array, ""], _: None) -> None:
            picks = index_ref[pl.ds(n * bk, bk)]
            dots = _dot(q, _gathered(keys_ref, picks), (1, 1))
            scores = (jnp.maximum(dots, 0) * w[:, None]).sum(0)
            out_ref[pl.ds(n * bk, bk)] = jnp.where(picks >= 0, scores, 0.0)

        jax.lax.fori_loop(0, _blocks(index_ref[...], bk), block, None)

    query = lambda z, m: (z, m, 0)
    return pl.pallas_call(
        kernel,
        grid=(b, t),
        in_specs=[
            pl.BlockSpec((None, None, k), query),
            pl.BlockSpec((None, None, h, i), lambda z, m: (z, m, 0, 0)),
            pl.BlockSpec((None, None, h), query),
            pl.BlockSpec((None, keys.shape[1], i), lambda z, m: (z, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, k), query),
        out_shape=jax.ShapeDtypeStruct((b, t, k), F32),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=2),
        interpret=kernels.interpreted(),
        name="indexer_picked_scores",
    )(index.astype(jnp.int32), q, w.astype(F32), keys.astype(q.dtype))


def _picked_grads(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
    d: Float[Array, "b t k"],
) -> tuple[Float[Array, "b t h i"], Float[Array, "b t h"], Float[Array, "b s i"]]:
    """Run the backward kernel: a program a query, a block of its picks at a time,
    each pick's share of the keys' gradient added into it atomically, in float32."""
    from jax.experimental.pallas import triton as plgpu

    jax_internals.relax_pallas_atomics()
    b, t, h, i = q.shape
    s, k, bk = keys.shape[1], index.shape[-1], _block(index.shape[-1], PICKS)
    assert bk is not None

    # Pallas refs, which index and load as arrays do not.
    def kernel(
        index_ref: Any,
        q_ref: Any,
        w_ref: Any,
        keys_ref: Any,
        d_ref: Any,
        _: Any,
        dq_ref: Any,
        dw_ref: Any,
        dkeys_ref: Any,
    ) -> None:
        q, w = q_ref[...], w_ref[...]

        def block(n: Int[Array, ""], grads: tuple) -> tuple:
            dq, dw = grads
            picks = index_ref[pl.ds(n * bk, bk)]
            rows = _gathered(keys_ref, picks)
            dots = _dot(q, rows, (1, 1))
            d = jnp.where(picks >= 0, d_ref[pl.ds(n * bk, bk)], 0.0)
            g = jnp.where(dots > 0, w[:, None] * d[None, :], 0.0)  # (h, picks)
            dq += _dot(g, rows.astype(F32), (1, 0))
            dw += (jnp.maximum(dots, 0) * d[None, :]).sum(1)
            drows = _dot(g, q.astype(F32), (0, 0))
            spare = s + jnp.arange(bk)  # rows after the keys' for picks of -1
            plgpu.atomic_add(
                dkeys_ref, _rows(jnp.where(picks >= 0, picks, spare), i), drows
            )
            return dq, dw

        zeros = jnp.zeros((h, i), F32), jnp.zeros((h,), F32)
        dq, dw = jax.lax.fori_loop(0, _blocks(index_ref[...], bk), block, zeros)
        dq_ref[...], dw_ref[...] = dq.astype(dq_ref.dtype), dw

    query = lambda z, m: (z, m, 0)
    heads = pl.BlockSpec((None, None, h, i), lambda z, m: (z, m, 0, 0))
    row = pl.BlockSpec((None, s, i), lambda z, m: (z, 0, 0))
    grads = pl.BlockSpec((None, s + bk, i), lambda z, m: (z, 0, 0))
    dq, dw, dkeys = pl.pallas_call(
        kernel,
        grid=(b, t),
        in_specs=[
            pl.BlockSpec((None, None, k), query),
            heads,
            pl.BlockSpec((None, None, h), query),
            row,
            pl.BlockSpec((None, None, k), query),
            grads,
        ],
        out_specs=[heads, pl.BlockSpec((None, None, h), query), grads],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(w.shape, F32),
            jax.ShapeDtypeStruct((b, s + bk, i), F32),
        ],
        input_output_aliases={5: 2},  # the keys' gradient starts at zeros
        compiler_params=plgpu.CompilerParams(num_warps=2 * WARPS, num_stages=2),
        interpret=kernels.interpreted(),
        name="indexer_picked_grads",
    )(
        index.astype(jnp.int32),
        q,
        w.astype(F32),
        keys.astype(q.dtype),
        d.astype(F32),
        jnp.zeros((b, s + bk, i), F32),
    )
    return dq, dw, dkeys[:, :s]
