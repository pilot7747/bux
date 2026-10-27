"""CSA's attention by cudnn-frontend's DSA kernels (DeepSeek sparse attention, in CuTe
DSL) on a B200 or B300, through ``cudnn.jax.call`` on XLA's stream: the forward and
backward of a layer's parts and its sinks in one softmax, and the indexer's target at
its picks.

The kernels attend 64 query heads to one latent a slot, 512 wide, which is the key and
the value at once, as CSA's latents are; each query to the slots its list holds. So a
layer's parts become one flat array of latents, each part's rows one after another,
and each query's list holds its slots of every part by their place there (`_lists`):
the ones of its window, then the ones it keeps; -1 past them, up to a multiple of 64.
Heads go 64 at a time, the last group padded with heads of zeros, whose gradients are
zero. The forward kernel's logsumexp leaves the sinks out, and is what the backward
kernel takes.

`takes` says where they run, `attend` runs the attention on each device's rows,
queries and heads and every slot, and `probabilities` the target.
"""

import functools
from collections.abc import Sequence
from functools import partial
from typing import Any, no_type_check

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, Shaped

from bux import kernels, parallel
from bux.architecture.attention import sparse

DIMS = 512  # the latents' width, the kernels'
HEADS = 64  # the heads a call takes
SLOTS = 64  # a list's slots go in tiles of this many
SM = ("10.0", "10.3")  # the GPUs they run on: B200, B300
MOST = 1 << 31  # elements of an array the kernels address
SHARED = 227 * 1024  # bytes of shared memory a block has on a B200 or B300
DTYPES = (jnp.bfloat16, jnp.float16)  # the attention's
TARGET_DTYPES = (jnp.bfloat16,)  # the target's
F32 = jnp.float32


try:  # cudnn-frontend and CuTe DSL, where the cuda extra is installed (Linux)
    import cutlass
    from cudnn.deepseek_sparse_attention.score_recompute import (
        sparse_score_recompute_sm100 as _scores,
    )
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
        dsa_bwd_sm100 as _backward,
    )
    from cudnn.deepseek_sparse_attention.sparse_attention_forward import (
        dsa_fwd_sm100_head64 as _forward,
    )
    from cudnn.jax import call, zeros_init
    from cutlass import cute
except ImportError:
    cutlass = None


def runs() -> bool:
    """Return whether the kernels run here: on a B200 or B300 with cudnn-frontend,
    under ``methods.attention = auto``, outside `bux.kernels.off`."""
    auto = parallel.method_name("attention") == "auto"
    return cutlass is not None and auto and kernels.gpu() and _sm()


@functools.cache
def _sm() -> bool:
    """Return whether this host's GPUs are ones the kernels run on."""
    device = jax.local_devices()[0]
    return getattr(device, "compute_capability", "") in SM


def takes(
    q: Float[Array, "b t h d"],
    parts: Sequence[sparse.Part],
    windows: Sequence[int | None],
    cap: float | None,
) -> bool:
    """Return whether the kernels take parts whose keys are their values, CSA's
    latents: queries they take (`_takes_queries`), 2^16 of them at most on a device, a
    latent of 512 dims a slot of the queries' dtype, and the slots of each part
    bounded: a window of them, or the ones a query keeps by index (not by block, nor
    for each group).

    Args:
        q: the queries: b rows of t, h heads of d dims.
        parts: the latents, and the slots each query attends to.
        windows: each part's window, the most slots of its span a query sees; None
            for one without.
        cap: the logits' tanh soft cap; None for none.
    """
    latents = all(p.k.shape[2:] == (1, DIMS) and p.k.dtype == q.dtype for p in parts)
    bounded = all(map(_bounded, parts, windows, strict=True))
    fits = _queries(q) * HEADS * DIMS <= MOST
    return latents and bounded and fits and _takes_queries(q, cap)


def _takes_queries(q: Float[Array, "b t h d"], cap: float | None) -> bool:
    """Return whether the kernels take the queries: where they run (`runs`), in
    bfloat16 or float16, of 512 dims, and no soft cap."""
    typed = q.dtype in DTYPES
    return typed and q.shape[-1] == DIMS and cap is None and runs()


def _bounded(part: sparse.Part, window: int | None) -> bool:
    """Return whether a query sees a bounded number of `part`'s slots: those it keeps
    by index for all its heads (not by block, nor for each group, as MSA's), or those
    of its `window`."""
    if part.kept is None:
        return window is not None
    return part.kept.block is None and not sparse.per_group(part)


def _queries(q: Float[Array, "b t h d"]) -> int:
    """Return how many queries a device has of `q`'s rows."""
    on = parallel.active()
    if on is None:
        return q.shape[0] * q.shape[1]
    split = parallel.spec(("batch", "seq"), q.shape[:2])
    return q.shape[0] * q.shape[1] // on.size(*parallel.axes_of(split))


# --- the attention ----------------------------------------------------------------


def attend(
    q: Float[Array, "b t h d"],
    parts: Sequence[sparse.Part],
    sinks: Float[Array, " h"] | None,
    scale: float,
    windows: Sequence[int | None],
) -> tuple[Float[Array, "b t h d"], Float[Array, "b t h"]]:
    """Return `~bux.architecture.attention.sparse.attend` by the kernels, for parts
    whose keys are their values: every query's softmax-weighted latents of the slots
    it attends to in each part, the parts and the sinks in one softmax.

    On a mesh each device runs them on its rows, queries and heads, and every slot.
    A query's slots are each counted once, as an indexer keeps them.

    Args:
        q: the queries: b rows of t, h heads of 512 dims.
        parts: the latents, and the slots each query attends to (`takes`).
        sinks: each head's learned sink logit; None for none.
        scale: the logits' scale.
        windows: each part's window, as `takes` has them.

    Returns:
        The output, (b, t, h, 512), and each query's logsumexp, (b, t, h), which
        passes no gradient.
    """

    def run(
        q: Float[Array, "b t h d"],
        parts: tuple[sparse.Part, ...],
        sinks: Float[Array, " h"] | None,
        causal: bool,  # the lists say what each query sees
    ) -> tuple[Float[Array, "b t h d"], Float[Array, "b t h"]]:
        return _attend(q, parts, sinks, scale, tuple(windows))

    return sparse.on_devices(run, q, tuple(parts), sinks)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _attend(
    q: Float[Array, "b t h d"],
    parts: tuple[sparse.Part, ...],
    sinks: Float[Array, " h"] | None,
    scale: float,
    windows: tuple[int | None, ...],
) -> tuple[Float[Array, "b t h d"], Float[Array, "b t h"]]:
    """Run `attend` on a device's queries."""
    return _attend_fwd(q, parts, sinks, scale, windows)[0]


def _attend_fwd(
    q: Float[Array, "b t h d"],
    parts: tuple[sparse.Part, ...],
    sinks: Float[Array, " h"] | None,
    scale: float,
    windows: tuple[int | None, ...],
) -> tuple[tuple[Float[Array, "b t h d"], Float[Array, "b t h"]], tuple]:
    b, t, h, d = q.shape
    latents, lists = _lists(parts, windows)
    groups = zip(_heads(q.reshape(b * t, h, d)), _sink_groups(sinks, h), strict=True)
    outs, lses = zip(
        *(_forward_call(qg, latents, lists, sg, scale) for qg, sg in groups)
    )
    out = _ungrouped(outs, h).reshape(q.shape)
    lse = _ungrouped(lses, h).reshape(b, t, h)  # of the slots alone, for the backward
    return (out, _with_sinks(lse, sinks)), (q, parts, sinks, out, lse)


def _attend_bwd(
    scale: float, windows: tuple[int | None, ...], saved: tuple, cotangents: tuple
) -> tuple:
    """Return `_attend`'s gradients by the backward kernel, each part's latents' from
    the flat array's. The logsumexp's cotangent is dropped: it is for what is
    detached."""
    q, parts, sinks, out, lse = saved
    b, t, h, _ = q.shape
    latents, lists = _lists(parts, windows)
    dout = cotangents[0].astype(q.dtype)
    flat = [x.reshape(b * t, h, -1) for x in (q, out, dout, lse[..., None])]
    groups = zip(*map(_heads, flat), _sink_groups(sinks, h), strict=True)
    grads = [
        _backward_call(qg, latents, og, dg, lg[..., 0], sg, lists, scale)
        for qg, og, dg, lg, sg in groups
    ]
    dq = _ungrouped([g[0] for g in grads], h).reshape(q.shape)
    dlatents = sum(g[1].astype(F32) for g in grads)
    dsinks = _ungrouped([g[2] for g in grads], h)
    dsinks = None if sinks is None else dsinks.astype(sinks.dtype)
    return dq, _split(jnp.asarray(dlatents).astype(q.dtype), parts), dsinks


_attend.defvjp(_attend_fwd, _attend_bwd)


def _with_sinks(
    lse: Float[Array, "b t h"], sinks: Float[Array, " h"] | None
) -> Float[Array, "b t h"]:
    """Return each query's logsumexp of its slots and its head's sink, from the
    kernel's of its slots alone, which is +inf for a query with none."""
    lse = jnp.where(lse == jnp.inf, -jnp.inf, lse)
    return lse if sinks is None else jnp.logaddexp(lse, sinks.astype(F32))


# --- the slots, the latents and the heads, as the kernels take them ------------------


def _lists(
    parts: Sequence[sparse.Part], windows: Sequence[int | None]
) -> tuple[Float[Array, "n d"], Int[Array, "bt k"]]:
    """Return every part's latents, each part's rows one after another, and each
    query's list of the slots it attends to, by their place there: every part's in
    turn; -1 past them, up to a multiple of `SLOTS`."""
    latents, lists, first = [], [], 0
    for part, window in zip(parts, windows, strict=True):
        b, s = part.k.shape[:2]
        latents.append(part.k.reshape(b * s, -1))
        slot = _seen(part, window)
        row = first + s * jnp.arange(b, dtype=jnp.int32)[:, None, None]
        lists.append(jnp.where(slot >= 0, row + slot, -1))
        first += b * s
    index = jnp.concatenate(lists, -1)
    index = index.reshape(-1, index.shape[-1])
    pad = ((0, 0), (0, -index.shape[-1] % SLOTS))
    return jnp.concatenate(latents), jnp.pad(index, pad, constant_values=-1)


def _seen(part: sparse.Part, window: int | None) -> Int[Array, "b t k"]:
    """Return the slots each query attends to of `part`, -1 for none: of its span,
    those that exist, of the ones it keeps, or else of its `window` up to its
    span's end."""
    start, end = (x[..., None] for x in part.span)
    if part.kept is None:
        assert window is not None
        slot = end - 1 - jnp.arange(window, dtype=jnp.int32)
    else:
        slot = part.kept.index
    seen = (slot >= start) & (slot < end)
    if part.exists is not None:
        seen &= jax.vmap(lambda e, i: e[i])(part.exists, jnp.maximum(slot, 0))
    return jnp.where(seen, slot, -1).astype(jnp.int32)


def _split(
    dlatents: Float[Array, "n d"], parts: Sequence[sparse.Part]
) -> tuple[sparse.Part, ...]:
    """Return each part's share of the flat latents' gradient, as its keys'."""
    grads, first = [], 0
    for part in parts:
        n = part.k.shape[0] * part.k.shape[1]
        dk = dlatents[first : first + n].reshape(part.k.shape)
        grads.append(sparse.Part(dk, None, None, None, None))  # ty: ignore
        first += n
    return tuple(grads)


def _heads(x: Shaped[Array, "n h e"]) -> list[Shaped[Array, "n 64 e"]]:
    """Return x's heads `HEADS` at a time, the last group padded with zeros."""
    pad = [(0, 0), (0, -x.shape[1] % HEADS), (0, 0)]
    return jnp.split(jnp.pad(x, pad), -(-x.shape[1] // HEADS), axis=1)


def _sink_groups(sinks: Float[Array, " h"] | None, h: int) -> list[Float[Array, " 64"]]:
    """Return the sinks of each group of `_heads` in float32, -inf for none."""
    full = jnp.full(h, -jnp.inf, F32) if sinks is None else sinks.astype(F32)
    padded = jnp.pad(full, (0, -h % HEADS), constant_values=-jnp.inf)
    return list(padded.reshape(-1, HEADS))


def _ungrouped(groups: Sequence[Shaped[Array, "*g"]], h: int) -> Shaped[Array, "*s"]:
    """Return the first h heads of groups of `_heads`, together again: their heads
    along axis 1, or 0 for the sinks'."""
    axis = 1 if groups[0].ndim > 1 else 0
    return jax.lax.slice_in_dim(jnp.concatenate(groups, axis), 0, h, axis=axis)


# --- the indexer's target ------------------------------------------------------------


def probabilities_take(q: Float[Array, "b t h d"], cap: float | None) -> bool:
    """Return whether the kernel scores the indexer's target at its picks
    (`probabilities`): where the attention's kernels run, in bfloat16, 128 heads at
    most, and none of them split over devices, since it normalizes the sum of every
    head's."""
    split = parallel.axes_of(parallel.spec(("heads",), q.shape[2:3]))
    whole = not split and q.shape[2] <= 2 * HEADS
    return q.dtype in TARGET_DTYPES and whole and _takes_queries(q, cap)


def probabilities(
    q: Float[Array, "b t h d"],
    part: sparse.Part,
    lse: Float[Array, "b t h"],
    scale: float,
) -> Float[Array, "b t k"]:
    """Return each query's attention to the slots it keeps of `part`, summed over
    heads and normalized over the slots, by the kernel: the indexer's target at its
    picks (`probabilities_take`).

    As `~bux.architecture.attention.sparse.probabilities` but at the picks, not every
    slot, and normalized; 0 for a query whose picks take less than 1e-10 of its
    attention.

    Args:
        q: the attention's queries: b rows of t, h heads of 512 dims.
        part: the latents, and the slots each query keeps of its span.
        lse: each query's logsumexp in the attention's softmax.
        scale: the logits' scale.

    Returns:
        The probabilities, (b, t, k) in float32 in the order of the kept slots, 0
        for one that is -1 or outside the span. They pass no gradient.
    """
    q, part, lse = jax.lax.stop_gradient((q, part, lse))
    latents, picks = part.k[:, :, 0], _seen(part, None)
    run = partial(_score_call, scale=scale)
    split = parallel.spec(("batch", "seq", "heads", None), q.shape)
    if not parallel.axes_of(split):
        return run(q, latents, lse, picks)
    queries = P(split[0], split[1], None)
    ins = (split, P(split[0], None, None), queries, queries)
    return kernels.per_device(run, ins, queries)(q, latents, lse, picks)


# --- the kernels' calls ----------------------------------------------------------------


def _forward_call(
    q: Float[Array, "n 64 d"],
    latents: Float[Array, "m d"],
    lists: Int[Array, "n k"],
    sinks: Float[Array, " 64"],
    scale: float,
) -> tuple[Float[Array, "n 64 d"], Float[Array, "n 64"]]:
    """Return cuDNN's forward of 64 heads: each query's softmax-weighted latents of
    its list's slots beside its head's sink, and its logsumexp of the slots alone,
    +inf for a query with none."""
    n = q.shape[0]
    outs = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((n, HEADS), F32),  # each query's largest logit
        jax.ShapeDtypeStruct((n, HEADS), F32),
    )
    out, _, lse = call(
        _jit(_fwd),
        output_shape_dtype=outs,
        compile_options="--opt-level 2",  # cuDNN's: faster than CuTe DSL's 3 here
        kernel=_forward_kernel(),
        scale=scale,
    )(q, latents, lists, sinks)
    return out, lse


def _backward_call(
    q: Float[Array, "n 64 d"],
    latents: Float[Array, "m d"],
    out: Float[Array, "n 64 d"],
    dout: Float[Array, "n 64 d"],
    lse: Float[Array, "n 64"],
    sinks: Float[Array, " 64"],
    lists: Int[Array, "n k"],
    scale: float,
) -> tuple[Float[Array, "n 64 d"], Float[Array, "m d"], Float[Array, " 64"]]:
    """Return cuDNN's backward of 64 heads: the gradients of q, the latents and the
    sinks, from the forward's output and logsumexp of the slots alone."""
    assert cutlass is not None
    n, m = q.shape[0], latents.shape[0]
    kernel = _backward_kernel(q.dtype, lists.shape[1])
    f32 = cutlass.Float32
    outs = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(latents.shape, latents.dtype),
        jax.ShapeDtypeStruct((HEADS,), F32),
        # Scratch: each query's logsumexp and rowsum(dout * out), and dlatents in f32.
        jax.ShapeDtypeStruct(
            kernel._get_workspace_size_LSE_OdO(n, DIMS, HEADS, 1, f32), jnp.uint8
        ),
        jax.ShapeDtypeStruct(
            kernel._get_workspace_size_dKV(m, DIMS, 1, f32), jnp.uint8
        ),
    )
    dq, dlatents, dsinks, _, _ = call(
        _jit(_bwd),
        output_shape_dtype=outs,
        # The latents' gradient as cudnn-frontend 1.31 has it, the sinks' (atomic adds).
        initialized_outputs={1: zeros_init, 2: zeros_init},
        kernel=kernel,
        scale=scale,
    )(q, latents, out, dout, lse, sinks, lists)
    return dq, dlatents, dsinks


def _score_call(
    q: Float[Array, "b t h d"],
    latents: Float[Array, "b s d"],
    lse: Float[Array, "b t h"],
    picks: Int[Array, "b t k"],
    scale: float,
) -> Float[Array, "b t k"]:
    """Return cuDNN's target at the picks, its heads padded to 64 or 128 by heads that
    attend to nothing, and the picks to whole tiles."""
    b, t, h, _ = q.shape
    k = picks.shape[-1]
    heads = HEADS if h <= HEADS else 2 * HEADS
    q = jnp.pad(q, ((0, 0), (0, 0), (0, heads - h), (0, 0)))
    lse = jnp.pad(
        lse.astype(F32), ((0, 0), (0, 0), (0, heads - h)), constant_values=jnp.inf
    )
    kernel = _score_kernel(heads, -(-k // 128) * 128)
    picks = jnp.pad(picks, ((0, 0), (0, 0), (0, kernel.topk - k)), constant_values=-1)
    out = call(
        _jit(_score),
        output_shape_dtype=jax.ShapeDtypeStruct((b, t, kernel.topk), F32),
        compile_options="--opt-level 2",  # cuDNN's for its tiles of 128 picks
        kernel=kernel,
        scale=scale,
    )(q, latents.astype(q.dtype), lse, picks, jnp.zeros((1, 1), jnp.int32))
    return out[..., :k]


@functools.cache
def _forward_kernel() -> Any:
    """Return cuDNN's forward kernel of 64 heads of 512 dims."""
    return _forward.SparseAttentionForwardSm100Head64(head_dim=DIMS, indexer_topk=0)


@functools.cache
def _backward_kernel(dtype: Any, slots: int) -> Any:
    """Return cuDNN's backward kernel of 512 dims, lists of `slots`, for `dtype`."""
    assert cutlass is not None
    element = cutlass.BFloat16 if dtype == jnp.bfloat16 else cutlass.Float16
    return _backward.FlashAttentionDSABackwardSm100(
        element_dtype=element,
        head_dim=DIMS,
        head_dim_v=DIMS,
        block_tile=SLOTS,
        max_topk=slots,
    )


@functools.cache
def _score_kernel(heads: int, picks: int) -> Any:
    """Return cuDNN's kernel of the target for `heads` heads of 512 dims and `picks`
    picks a query: cudnn-frontend's tiles for them (tuned on a B200), and as many
    stages of latents as shared memory holds, the picks in it if they fit."""
    tile = 256 if heads == HEADS or picks > 512 else 128  # of the latents' dims
    stage = 128 * tile * 2  # bytes of a stage of latents: 128 picks
    fixed = heads * DIMS * 2 + heads * 8 + 2048  # the queries, logsumexps, barriers

    def stages(taken: int) -> int:
        return min(4, max(1, (SHARED - taken) // stage))

    inside = fixed + picks * 8 + stages(fixed + picks * 8) * stage <= SHARED
    taken = fixed + (picks * 8 if inside else 0)
    return _scores.SparseScoreRecomputeSm100(
        head_dim=DIMS,
        qhead_per_kvhead=heads,
        m_block_size=heads,
        n_block_size=128,
        k_block_size=tile,
        topk=picks,
        kv_stage=stages(taken),
        score_type="attention",
        have_topk_length=False,
        topk_in_smem=inside,
        topk_indices_global=False,
    )


@functools.cache
def _jit(adapter: Any) -> Any:
    """Return `adapter` compiled by CuTe DSL, once."""
    return cute.jit(adapter)


# CuTe DSL takes an adapter's annotations for its own types: these have none.


@no_type_check
def _fwd(stream, *operands, kernel, scale):  # noqa: ANN001, ANN002, ANN202
    """The forward kernel, without the logsumexp of a prefix of the lists."""
    q, latents, lists, sinks, out, top, lse = operands
    scale = cutlass.Float32(scale)
    kernel(q, latents, lists, out, top, lse, None, sinks, None, scale, stream)


@no_type_check
def _bwd(stream, *operands, kernel, scale):  # noqa: ANN001, ANN002, ANN202
    """The backward kernel, on one row of every query."""
    q, latents, out, dout, lse, sinks, lists = operands[:7]
    dq, dlatents, dsinks, scratch, latents_scratch = operands[7:]
    heads = (cutlass.Int32(HEADS), cutlass.Int32(1))  # of one row
    shape = (q.shape[0], latents.shape[0], cutlass.Int32(DIMS), heads)
    ins = (q, latents, out, dout, lse, sinks, lists, None)
    outs = (dq, dlatents, dsinks, scratch, latents_scratch)
    kernel(shape, *ins, *outs, cutlass.Float32(scale), stream)


@no_type_check
def _score(stream, *operands, kernel, scale):  # noqa: ANN001, ANN002, ANN202
    """The target's kernel, the picks local to each row, each query's count of them
    unused (cudnn-frontend 1.31 takes an array all the same)."""
    q, latents, lse, picks, unused, out = operands
    kernel(q, latents, lse, picks, out, unused, cutlass.Float32(scale), stream)
