"""The lightning indexer's top-k and its loss's scores by cuDNN-frontend's DSA kernels
(CuTe DSL, in the cuda extra), on a B200 or B300, through ``cudnn.jax.call`` on XLA's
stream (`~bux.architecture.attention.indexer`).

`top_k` is cuDNN's radix top-k: a block of threads a row finds its k-th highest score
by 8 bits at a time, from a histogram in shared memory, keeps those above it and, of
those equal to it, the first, and returns them in order of index, as the indexer's
XLA path does. It reads a row up to the slot after its query's last. It keeps k a row
however few the query sees, and picks the rest among the ties at the k-th highest one
at a time: so a key the query does not see goes in as one of the lowest floats, each
slot's its own (`_unseen`), not as -inf, and `top_k` drops the keys it kept of those.
A query whose span has k slots or fewer, as in a short document, keeps every key it
sees without the kernel.

`log_softmax` is the log-softmax of the scores of each query's picks, for the
indexer's loss: forward, cuDNN's sparse score recompute (the softmax, its scores kept
in registers from head to head); backward, the GEMM kernel of cuDNN's indexer
backward, which gathers each block of 128 picks' keys by TMA and adds their gradient
into the keys in float32, from the scores' gradient. That gradient rounds to bfloat16
before the GEMMs, as flash attention's does.

On a mesh a ``shard_map`` gives the kernels each device's rows and queries, every
head and every key.
"""

import functools
from typing import Any, no_type_check

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import kernels, parallel
from bux.architecture.attention import scoring

try:  # cudnn-frontend's CuTe DSL kernels, where the cuda extra is installed (Linux)
    import cutlass
    import cutlass.utils
    from cudnn.deepseek_sparse_attention.indexer_backward import (
        indexer_backward_sm100 as backward,
    )
    from cudnn.deepseek_sparse_attention.indexer_top_k import (
        indexer_top_k_decode_varlen as radix,
    )
    from cudnn.deepseek_sparse_attention.score_recompute import (
        sparse_score_recompute_sm100 as recompute,
    )
    from cudnn.jax import call, zeros_init
    from cutlass import cute
except ImportError:
    cutlass = None

MOST = 2048  # the most a row the top-k keeps, and the most picks the loss's take
SMS = 148  # rows past which the top-k spills to its scratch, as cudnn-frontend tunes it
SCRATCH = 2  # int32 words of the top-k's scratch a float32 score takes
PICKS = 128  # the picks a step of the loss's kernels takes
HEADS = (32, 64)  # the loss's kernels' heads: 128 overflow the backward's smem
DIMS = 128  # their head dims
SM = ("10.0", "10.3")  # the GPUs they run on: B200, B300
F32, BF16 = jnp.float32, jnp.bfloat16
TINY = jnp.finfo(F32).tiny  # the least probability whose log `log_softmax` gives


def runs(k: int) -> bool:
    """Return whether the top-k kernel keeps k a row here: on a B200 or B300 with
    cudnn-frontend, for k up to 2048."""
    return _here() and 0 < k <= MOST


def takes(h: int, i: int, k: int, dtype: Any) -> bool:
    """Return whether the loss's kernels take queries of h heads of i dims and k
    picks each, in `dtype`, here: on a B200 or B300 with cudnn-frontend; 32 or 64
    heads of 128 dims in bfloat16; picks in blocks of 128, up to 2048."""
    picks = k % PICKS == 0 and 0 < k <= MOST
    return _here() and h in HEADS and i == DIMS and dtype == BF16 and picks


def _here() -> bool:
    """Return whether the kernels run here: on a B200 or B300, with cudnn-frontend."""
    return cutlass is not None and kernels.gpu() and _sm()


@functools.cache
def _sm() -> bool:
    """Return whether this host's GPUs are ones the kernels run on."""
    device = jax.local_devices()[0]
    return getattr(device, "compute_capability", "") in SM


# --- the top-k ------------------------------------------------------------------------


def top_k(
    scores: Float[Array, "b t n"],
    start: Int[Array, "b t"],
    end: Int[Array, "b t"],
    k: int,
) -> Int[Array, "b t k"]:
    """Return the indices of each query's k highest scores of those above -inf, by
    the kernel: in order of index, and of equal ones the first; -1 for the rest.

    Args:
        scores: each query's scores of n keys; -inf for those it does not see, and
            for every one outside its span. One among the n lowest floats counts as
            -inf.
        start: each query's first slot.
        end: the slot after each query's last, which the kernel reads up to.
        k: how many to keep, up to n.

    Returns:
        The indices, (b, t, k). A -1 may come before a kept one, where the kernel
        kept a key the query does not see.
    """
    split = parallel.spec(("batch", "seq", None), scores.shape)
    run = functools.partial(_top_k, k=k)
    if not parallel.axes_of(split):
        return run(scores, start, end)
    rows = P(*split[:2])
    return kernels.per_device(run, (split, rows, rows), split)(scores, start, end)


def _top_k(
    scores: Float[Array, "b t n"],
    start: Int[Array, "b t"],
    end: Int[Array, "b t"],
    k: int,
) -> Int[Array, "b t k"]:
    """Run the kernel on a device's rows whose span is longer than k, and drop the
    unseen keys it kept; keep every key the others see, as there are k at most."""
    b, t, n = scores.shape
    unseen = _unseen(n)
    rows = jnp.where(scores > -jnp.inf, scores.astype(F32), unseen).reshape(b * t, n)
    few = end - start <= k
    lengths = jnp.where(few, 0, jnp.clip(end, 0, n)).reshape(b * t).astype(jnp.int32)
    picks, values = _radix(rows, lengths, k)
    kept = jnp.where(values > unseen[-1], picks, -1).reshape(b, t, k)
    return jnp.where(few[..., None], _seen(scores, start, end, k), kept)


def _seen(
    scores: Float[Array, "b t n"],
    start: Int[Array, "b t"],
    end: Int[Array, "b t"],
    k: int,
) -> Int[Array, "b t k"]:
    """Return the k slots from each query's start where it sees a key, -1 for the
    rest."""
    n = scores.shape[-1]
    slots = start[..., None] + jnp.arange(k, dtype=jnp.int32)
    inside = slots < jnp.minimum(end, n)[..., None]
    seen = jnp.take_along_axis(scores, jnp.clip(slots, 0, n - 1), -1) > -jnp.inf
    return jnp.where(inside & seen, slots, -1)


def _unseen(n: int) -> Float[Array, " n"]:
    """Return the n lowest floats, from the lowest up: what each of n slots holds
    for the kernel where its query does not see the key, so that no two tie."""
    lowest = jnp.uint32(0xFF7FFFFF)  # -3.4e38, the bits of the lowest finite float
    bits = lowest - jnp.arange(n, dtype=jnp.uint32)
    return jax.lax.bitcast_convert_type(bits, F32)


def _radix(
    rows: Float[Array, "r n"], lengths: Int[Array, " r"], k: int
) -> tuple[Int[Array, "r k"], Float[Array, "r k"]]:
    """Return the kernel's k of each row, those of its first `lengths` slots, in
    order of index, and their scores; -1 and -inf past a row shorter than k. It runs
    on as many rows at a time as its scratch indexes by int32."""
    most = (2**31 - 1) // (SCRATCH * rows.shape[1])
    parts = [
        _radix_rows(rows[at : at + most], lengths[at : at + most], k)
        for at in range(0, rows.shape[0], most)
    ]
    picks, values = zip(*parts, strict=True)
    return jnp.concatenate(picks), jnp.concatenate(values)


def _radix_rows(
    rows: Float[Array, "r n"], lengths: Int[Array, " r"], k: int
) -> tuple[Int[Array, "r k"], Float[Array, "r k"]]:
    """Run the kernel on rows whose scratch int32 indexes."""
    (r, n), out = rows.shape, jax.ShapeDtypeStruct
    picks, values, _ = call(
        _jit(_select),
        output_shape_dtype=(
            out((r, k), jnp.int32),
            out((r, k), F32),
            out((r, SCRATCH, n), jnp.int32),
        ),
        kernel=_radix_kernel(n, k, r > SMS),
    )(rows, lengths)
    return picks, values


# --- the loss's log-softmax of each query's picks' scores -----------------------------


def log_softmax(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> Float[Array, "b t k"]:
    """Return each query's log-softmax of its scores of the keys it picked, the sum
    over heads of w times the ReLU'd dot product, by the kernels.

    Args:
        q: the indexer's queries: b rows of t, h heads of i dims, in bfloat16.
        w: each query's head weights.
        keys: the keys, one a slot, i wide, in bfloat16.
        index: each query's picks, k slots; -1 for none.

    Returns:
        The log-probabilities, (b, t, k) in float32, at least log(tiny): -87.3, which
        passes no gradient; for a pick of -1 too.
    """
    split = parallel.spec(("batch", "seq", None, None), q.shape)
    if not parallel.axes_of(split):
        return _log_softmax(q, w, keys, index)
    rows = P(split[0], split[1])
    ins = (split, P(*split[:3]), P(split[0], None, None), P(*rows, None))
    return scoring.on_own_queries(_log_softmax, split, ins)(q, w, keys, index)


@jax.custom_vjp
def _log_softmax(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> Float[Array, "b t k"]:
    """Return `log_softmax` on a device's queries, forward and backward by the
    kernels."""
    return jnp.log(jnp.maximum(_softmax(q, w, keys, index), TINY))


def _log_softmax_fwd(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> tuple[Float[Array, "b t k"], tuple]:
    p = _softmax(q, w, keys, index)
    return jnp.log(jnp.maximum(p, TINY)), (q, w, keys, index, p)


def _log_softmax_bwd(saved: tuple, d: Float[Array, "b t k"]) -> tuple:
    q, w, keys, index, p = saved
    d = jnp.where(p >= TINY, d, 0.0)  # a log clipped at TINY's passes none
    dq, dw, dkeys = _grads(q, w, keys, index, d - p * d.sum(-1, keepdims=True))
    return dq, dw.astype(w.dtype), dkeys.astype(keys.dtype), None


_log_softmax.defvjp(_log_softmax_fwd, _log_softmax_bwd)


def _softmax(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
) -> Float[Array, "b t k"]:
    """Return each query's softmax of its picks' scores by cuDNN's score recompute;
    0 for a pick of -1."""
    (b, t, h, i), k = q.shape, index.shape[-1]
    return call(
        _jit(_recompute),
        output_shape_dtype=jax.ShapeDtypeStruct((b, t, k), F32),
        kernel=_recompute_kernel(h, i, k),
    )(q, keys, w.astype(BF16), index.astype(jnp.int32))


def _grads(
    q: Float[Array, "b t h i"],
    w: Float[Array, "b t h"],
    keys: Float[Array, "b s i"],
    index: Int[Array, "b t k"],
    d: Float[Array, "b t k"],
) -> tuple[Float[Array, "b t h i"], Float[Array, "b t h"], Float[Array, "b s i"]]:
    """Return the gradients of q, w (in bfloat16) and the keys (in float32) from
    that of the picks' scores, d, by the GEMM kernel of cuDNN's indexer backward."""
    (b, t, h, i), (s, k) = q.shape, (keys.shape[1], index.shape[-1])
    out = jax.ShapeDtypeStruct
    return call(
        _jit(_backward),
        output_shape_dtype=(
            out(q.shape, q.dtype),
            out(w.shape, BF16),
            out(keys.shape, F32),
        ),
        initialized_outputs={2: zeros_init},  # the keys' gradient, added into
        compile_options="--opt-level 2",  # as cudnn-frontend compiles it
        kernel=_backward_kernel(h, i, k, b * t, b * s),
    )(q, w.astype(BF16), keys, d.astype(F32), index.astype(jnp.int32))


# --- the kernels, and their adapters, which CuTe DSL compiles ------------------------


@functools.cache
def _radix_kernel(n: int, k: int, many: bool) -> Any:
    """Return the top-k kernel for rows of up to n scores, k kept; `many` rows spill
    what shared memory does not hold to the scratch."""
    assert cutlass is not None
    return radix.IndexerTopKKernelVarlenDecode(
        cutlass.Float32,
        1 << (n - 1).bit_length(),  # it compiles for a power of two of columns
        k,
        return_val=True,
        large_occupancy=many,
        tie_break=1,  # of equal scores, the first
        deterministic=True,  # and in order of index
    )


@functools.cache
def _recompute_kernel(h: int, i: int, k: int) -> Any:
    """Return the score recompute's kernel for h heads of i dims and k picks a
    query: a tile of a query's heads by 128 picks, four tiles of keys in flight."""
    assert cutlass is not None
    return recompute.SparseScoreRecomputeSm100(
        head_dim=i,
        qhead_per_kvhead=h,  # one key a slot, MQA's
        m_block_size=h,
        n_block_size=PICKS,
        topk=k,
        kv_stage=4,  # what 227 KB of shared memory holds beside 64 heads, 2048 picks
        score_type="indexer",
        topk_indices_global=False,  # each row's own slots
    )


@functools.cache
def _backward_kernel(h: int, i: int, k: int, rows: int, keys: int) -> Any:
    """Return the indexer backward's GEMM kernel for h heads of i dims, k picks a
    query, `rows` queries and `keys` keys in all, on every SM."""
    assert cutlass is not None
    return backward.IndexerBackwardSm100(
        head_dim=i,
        heads=h,
        block_I=PICKS,
        topk=k,
        total_seqlen_k=keys,
        total_rows=rows,
        persistent_grid_size=cutlass.utils.HardwareInfo().get_max_active_clusters(1),
        topk_indices_global=False,  # each row's own slots
    )


@functools.cache
def _jit(adapter: Any) -> Any:
    """Return `adapter` compiled by CuTe DSL, once."""
    return cute.jit(adapter)


# CuTe DSL takes an adapter's annotations for its own types: these have none.


@no_type_check
def _select(stream, scores, lengths, picks, values, scratch, *, kernel):  # noqa: ANN001, ANN202
    """The top-k on rows of scores, each read up to its length."""
    kernel(scores, None, scratch, None, lengths, picks, values, stream)


@no_type_check
def _recompute(stream, q, keys, w, index, out, *, kernel):  # noqa: ANN001, ANN202
    """The score recompute, its scores' scale 1: w has the heads' scale."""
    kernel(q, keys, w, index, out, None, 1.0, stream)


@no_type_check
def _backward(stream, q, w, keys, d, index, dq, dw, dkeys, *, kernel):  # noqa: ANN001, ANN202
    """The indexer backward's GEMMs, from the scores' gradient d."""
    kernel(q, w, keys, dq, dw, dkeys, d, index, 1.0, stream)
