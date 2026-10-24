"""Attention over the keys each query keeps (an indexer's top-k, CSA's latents), with
no t by t floats.

Keys sit in slots. A `Span` says which of them each query may attend to, `Kept` which
of those it keeps, by index; a `Part` is keys and values with both. A layer's parts
(CSA's window and its latents) share one softmax, and the sinks, merged by their
logsumexp.

On a GPU, tokamax's flash kernel runs each part with its kept keys as a boolean mask:
dense compute, exact, and t² bits of mask forward and bytes backward. Under
`methods.attention = xla`, off a GPU, or inside ``bux.kernels.off()``, XLA runs it a
chunk of queries at a time. Both return each query's logsumexp, from which the
backward pass computes the probabilities again, as `probabilities` does for the
indexer's loss.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import jax_internals, kernels, parallel
from bux.architecture.attention import scoring
from bux.architecture.common.segments import Segments
from bux.parallel.context import softcap
from bux.parallel.flash import MOSAIC, finite, tokamax_lse
from bux.train import flops

KEPT = "mask of kept keys (an indexer's top-k, or CSA's compression)"
ELEMENTS = 1 << 27  # the most a chunk of queries makes of an intermediate
HEAD_DIMS = 256  # the widest heads tokamax's kernels take
F32 = jnp.float32


class Span(NamedTuple):
    """The slots each query may attend to: from `start` to before `end`."""

    start: Int[Array, "b t"]
    end: Int[Array, "b t"]


class Kept(NamedTuple):
    """The slots each query keeps, by index; or with `block` (which block each slot is
    in) the blocks, by the block's. -1 for none. The index port carries it."""

    index: Int[Array, "b t k"]
    block: Int[Array, "b s"] | None = None


class Part(NamedTuple):
    """Keys and values, and the slots each query attends to: those of its span that
    exist, of those it keeps."""

    k: Float[Array, "b s g dk"]
    v: Float[Array, "b s g dv"]
    span: Span
    exists: Bool[Array, "b s"] | None = None
    kept: Kept | None = None


def token_span(rows: Segments, window: int | None) -> Span:
    """Return the span of keys at the queries' positions: from the first of each
    query's document, and fewer than `window` behind it, to itself."""
    index = jnp.arange(rows.within.shape[1], dtype=jnp.int32)
    start = index - rows.within
    if window is not None:
        start = jnp.maximum(start, index - window + 1)
    return Span(start, jnp.broadcast_to(index + 1, start.shape))


def hits(index: Int[Array, "b t k"], n: int) -> Bool[Array, "b t n"]:
    """Return which of n slots, or blocks, each query's `index` holds."""

    def row(i: Int[Array, " k"]) -> Bool[Array, " n"]:
        return jnp.zeros(n + 1, bool).at[jnp.where(i >= 0, i, n)].set(True)[:n]

    return jax.vmap(jax.vmap(row))(index)


def mask(
    span: Span,
    s: int,
    exists: Bool[Array, "b s"] | None = None,
    kept: Kept | None = None,
) -> Bool[Array, "b t s"]:
    """Return which of s slots each query attends to: of its span, those that exist,
    of those it keeps."""
    slot = jnp.arange(s)
    seen = (slot >= span.start[..., None]) & (slot < span.end[..., None])
    if exists is not None:
        seen &= exists[:, None]
    if kept is None:
        return seen
    hit = hits(kept.index, s)
    if kept.block is not None:
        block = jnp.broadcast_to(kept.block[:, None], hit.shape)
        hit = jnp.take_along_axis(hit, block, axis=-1)
    return seen & hit


def visible(part: Part) -> Bool[Array, "b t s"]:
    """Return which slots of `part` each query attends to."""
    return mask(part.span, part.k.shape[1], part.exists, part.kept)


def rows(part: Part) -> tuple[Span, Int[Array, "b t k"] | None]:
    """Return what `part` has for each query: its span, and the index it keeps."""
    return part.span, None if part.kept is None else part.kept.index


def at(part: Part, span: Span, index: Int[Array, "b t k"] | None) -> Part:
    """Return `part` for some of its queries, given their `rows`."""
    kept = None if part.kept is None or index is None else Kept(index, part.kept.block)
    return part._replace(span=span, kept=kept)


# --- chunks of queries ----------------------------------------------------------------


def chunks(t: int, per_query: int) -> int:
    """Return how many chunks to split t queries into: the fewest whose queries make
    at most ELEMENTS elements, `per_query` each, and split over the devices as all t
    do."""
    whole = t // _devices(t)
    fits = (n for n in range(1, whole) if whole % n == 0)
    return next((n for n in fits if t // n * per_query <= ELEMENTS), whole)


def _devices(t: int) -> int:
    """Return how many devices t positions split over."""
    on = parallel.active()
    if on is None:
        return 1
    return on.size(*parallel.axes_of(parallel.spec(("seq",), (t,))))


def chunked[R](f: Callable[[Any], R], xs: Any, n: int) -> R:
    """Return f(xs), xs and what it returns arrays of (b, t, ...), for n chunks of
    queries one at a time: every n-th query in one, so that a chunk splits over the
    devices as the positions do."""
    if n == 1:
        return f(xs)
    return jax.tree.map(_join, jax.lax.map(f, jax.tree.map(partial(_part, n=n), xs)))


def scanned[C, R](
    f: Callable[[C, Any], tuple[C, R]], carry: C, xs: Any, n: int
) -> tuple[C, R]:
    """Return `chunked` of f(carry, xs) -> (carry, out), the carry from chunk to
    chunk."""
    if n == 1:
        return f(carry, xs)
    carry, out = jax.lax.scan(f, carry, jax.tree.map(partial(_part, n=n), xs))
    return carry, jax.tree.map(_join, out)


def _part(x: Shaped[Array, "b t *r"], n: int) -> Shaped[Array, "chunks b c *r"]:
    b, t = x.shape[:2]
    return jnp.moveaxis(x.reshape(b, t // n, n, *x.shape[2:]), 2, 0)


def _join(x: Shaped[Array, "chunks b c *r"]) -> Shaped[Array, "b t *r"]:
    n, b, c = x.shape[:3]
    return jnp.moveaxis(x, 0, 2).reshape(b, c * n, *x.shape[3:])


# --- attention ------------------------------------------------------------------------


class _Static(NamedTuple):
    scale: float
    softcap: float | None
    flash: bool  # tokamax's kernels; else XLA's chunks
    causal: bool  # the queries at the keys' slots, as when positions are not split


def attend(
    q: Float[Array, "b t h dk"],
    parts: Sequence[Part],
    sinks: Float[Array, " h"] | None,
    scale: float,
    cap: float | None,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
    """Return every query's softmax-weighted values over the slots it attends to in
    each part, the parts and the sinks in one softmax in float32; and each query's
    logsumexp, which passes no gradient. With `seq` over `context` a device has its
    queries and every key: all-gather context parallelism."""
    static, parts = _Static(scale, cap, _flash(q), True), tuple(parts)
    split = parallel.spec(("batch", "seq", "heads", None), q.shape)
    names = set(parallel.axes_of(split))
    if parallel.active() is None or not names:
        return _attend(q, parts, sinks, static)
    parts = tuple(_heads_like(p, split, q.shape[2]) for p in parts)
    static = static._replace(causal=split[1] is None)
    return jax.shard_map(
        lambda q, parts, sinks: _attend(q, parts, sinks, static),
        in_specs=(
            split,
            tuple(_specs(p, split) for p in parts),
            _sink_spec(sinks, split),
        ),
        out_specs=(split, P(*split[:3])),
        axis_names=names,
        check_vma=False,
    )(q, parts, sinks)


def _flash(q: Float[Array, "b t h dk"]) -> bool:
    """Return whether the parts run by tokamax: under `tokamax`, or `auto` on a GPU in
    bfloat16 or float16, which its kernels take. `xla` runs them by XLA; the other
    methods have no mask of kept keys."""
    on = parallel.active()
    name = on.methods.attention if on else "auto"
    if name not in ("xla", "auto", "tokamax"):
        raise NotImplementedError(
            f"methods.attention = {name} has no {KEPT}; xla, tokamax and auto have"
        )
    if name == "tokamax" and MOSAIC is None:
        raise NotImplementedError(
            "methods.attention = tokamax needs tokamax, of the cuda extra (Linux)"
        )
    half = q.dtype in (jnp.bfloat16, jnp.float16)
    return name != "xla" and MOSAIC is not None and kernels.gpu() and half


def _heads_like(part: Part, split: P, heads: int) -> Part:
    """Return `part`, its kv heads repeated if they split otherwise than the queries'
    and there is more than one."""
    g = part.k.shape[2]
    keys = parallel.spec(("batch", None, "kv_heads", None), part.k.shape)
    if g == 1 or keys[2] == split[2]:
        return part
    k, v = (jnp.repeat(x, heads // g, axis=2) for x in (part.k, part.v))
    return part._replace(k=k, v=v)


def _specs(part: Part, split: P) -> Part:
    """Return how a device's share of `part` splits: its rows of everything, its
    queries of what each query has, and its heads; every key."""
    batch, seq, heads = split[:3]
    keys = P(batch, None, heads if part.k.shape[2] > 1 else None, None)
    queries, slots = P(batch, seq), P(batch, None)
    kept = None
    if part.kept is not None:
        block = None if part.kept.block is None else slots
        kept = Kept(P(batch, seq, None), block)  # ty: ignore
    exists = None if part.exists is None else slots
    return Part(keys, keys, Span(queries, queries), exists, kept)  # ty: ignore


def _sink_spec(sinks: Float[Array, " h"] | None, split: P) -> P | None:
    return None if sinks is None else P(split[2])


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _attend(
    q: Float[Array, "b t h dk"],
    parts: tuple[Part, ...],
    sinks: Float[Array, " h"] | None,
    static: _Static,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
    """Run `attend` on a device's queries."""
    return _forward(q, parts, sinks, static)


def _forward(
    q: Float[Array, "b t h dk"],
    parts: tuple[Part, ...],
    sinks: Float[Array, " h"] | None,
    static: _Static,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
    """Return `_attend`: each part's output and logsumexp, combined."""
    outs, lses = zip(*(_runs(p, static)[0](q, p, static) for p in parts))
    return _combine(outs, lses, sinks, q.dtype)


def _runs(part: Part, static: _Static) -> tuple[Callable, Callable]:
    """Return the forward and backward of `part`: tokamax's, for heads it takes (its
    kernels run out of shared memory past 256 dims, as CSA's 512 do), else XLA's. A
    part's forward and backward are each exact alone."""
    if static.flash and max(part.k.shape[-1], part.v.shape[-1]) <= HEAD_DIMS:
        return _flash_fwd, _flash_bwd
    return _xla_fwd, _xla_bwd


def _combine(
    outs: Sequence[Float[Array, "b t h dv"]],
    lses: Sequence[Float[Array, "b t h"]],
    sinks: Float[Array, " h"] | None,
    dtype: jnp.dtype,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
    """Return the output and logsumexp of the softmax over every part and the sinks,
    from each part's."""
    if len(outs) == 1 and sinks is None:
        return outs[0].astype(dtype), lses[0]
    every = [*lses] + (
        [] if sinks is None else [jnp.broadcast_to(sinks, lses[0].shape)]
    )
    lse = jax.nn.logsumexp(jnp.stack(every).astype(F32), axis=0)
    shares = [jnp.where(jnp.isfinite(x), jnp.exp(x - finite(lse)), 0) for x in lses]
    out = sum(w[..., None] * o.astype(F32) for w, o in zip(shares, outs))
    return jnp.asarray(out).astype(dtype), lse


def _attend_fwd(
    q: Float[Array, "b t h dk"],
    parts: tuple[Part, ...],
    sinks: Float[Array, " h"] | None,
    static: _Static,
) -> tuple[tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]], tuple]:
    out, lse = _forward(q, parts, sinks, static)
    return (out, lse), (q, parts, sinks, out, lse)


def _attend_bwd(static: _Static, saved: tuple, cotangents: tuple) -> tuple:
    """Return the gradients of `_attend`, each part's from the softmax of them all.
    The logsumexp's cotangent is dropped: it is for what is detached."""
    q, parts, sinks, out, lse = saved
    dout = cotangents[0]
    grads = [_runs(p, static)[1](q, p, out, dout, lse, static) for p in parts]
    dq = grads[0][0] if len(grads) == 1 else sum(g[0].astype(F32) for g in grads)
    dparts = tuple(Part(dk, dv, None, None, None) for _, dk, dv in grads)  # ty: ignore
    dsinks = None if sinks is None else _sink_grads(sinks, out, dout, lse)
    return jnp.asarray(dq).astype(q.dtype), dparts, dsinks


_attend.defvjp(_attend_fwd, _attend_bwd)


def _sink_grads(
    sinks: Float[Array, " h"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b t h"],
) -> Float[Array, " h"]:
    """Return the sinks' gradient: a sink takes exp(sink - lse) of each query's
    probability, so its gradient is minus that times dout . out, summed over queries."""
    delta = (dout.astype(F32) * out.astype(F32)).sum(-1)
    share = jnp.exp(sinks.astype(F32) - lse)
    return -(share * delta).sum((0, 1)).astype(sinks.dtype)


def probabilities(
    q: Float[Array, "b t h dk"],
    part: Part,
    lse: Float[Array, "b t h"],
    scale: float,
    cap: float | None,
) -> Float[Array, "b t s"]:
    """Return each query's probability of each slot of `part` in its span, kept or
    not, summed over heads, in the softmax whose logsumexp is `lse`; 0 for slots
    outside the span or that do not exist. By ``scoring``'s kernel where it runs."""
    t, s, d = q.shape[1], part.k.shape[1], part.k.shape[-1]
    if part.exists is None and scoring.probabilities_run(t, s, d):
        return _probabilities_kernel(q, part, lse, scale, cap)
    with flops.over_keys():
        _, logits = _logits(q, part.k, scale, cap)
    p = jnp.exp(logits - _grouped(lse[..., None], part.k.shape[2]))
    seen = mask(part.span, s, part.exists)
    return jnp.where(seen[:, None, None], p, 0).sum((1, 2))


def _probabilities_kernel(
    q: Float[Array, "b t h dk"],
    part: Part,
    lse: Float[Array, "b t h"],
    scale: float,
    cap: float | None,
) -> Float[Array, "b t s"]:
    """Return `probabilities` by ``scoring``'s kernel, in a shard_map that gives each
    device its rows, queries and heads, and every key, and sums the heads'."""
    run = partial(scoring.probabilities, scale=scale, cap=cap)
    split = parallel.spec(("batch", "seq", "heads", None), q.shape)
    names = set(parallel.axes_of(split))
    if parallel.active() is None or not names:
        return run(q, part.k, lse, *part.span)
    k, heads = _heads_like(part, split, q.shape[2]).k, split[2]

    def device(
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b s g dk"],
        lse: Float[Array, "b t h"],
        span: Span,
    ) -> Float[Array, "b t s"]:
        own = run(q, k, lse, *span)
        return own if heads is None else jax.lax.psum(own, heads)

    keys, rows = _specs(part._replace(k=k), split).k, P(split[0], split[1])
    return jax.shard_map(
        device,
        in_specs=(split, keys, P(*split[:3]), Span(rows, rows)),  # ty: ignore
        out_specs=P(split[0], split[1], None),
        axis_names=names,
        check_vma=False,
    )(q, k, lse, part.span)


# --- by XLA, a chunk of queries at a time ---------------------------------------------


def _grouped(x: Shaped[Array, "b t h e"], g: int) -> Shaped[Array, "b g r t e"]:
    """Group the query heads of `x` by the kv head they share, g groups."""
    b, t, h, e = x.shape
    return x.reshape(b, t, g, h // g, e).transpose(0, 2, 3, 1, 4)


def _ungrouped(x: Shaped[Array, "b g r t e"]) -> Shaped[Array, "b t h e"]:
    b, g, r, t, e = x.shape
    return x.transpose(0, 3, 1, 2, 4).reshape(b, t, g * r, e)


def _logits(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    scale: float,
    cap: float | None,
) -> tuple[Float[Array, "b g r t s"], Float[Array, "b g r t s"]]:
    """Return every query's logits for every key in float32, before the softcap and
    after it."""
    grouped = _grouped(q, k.shape[2])
    raw = jnp.einsum("bgrtd,bsgd->bgrts", grouped, k, preferred_element_type=F32)
    raw = raw * scale
    return raw, softcap(raw, cap)


def _xla_fwd(
    q: Float[Array, "b t h dk"], part: Part, static: _Static
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
    """Return a part's softmax-weighted values and logsumexp, a chunk at a time."""

    def chunk(
        xs: tuple,
    ) -> tuple[Float[Array, "b c h dv"], Float[Array, "b c h"]]:
        q, span, index = xs
        _, logits = _logits(q, part.k, static.scale, static.softcap)
        seen = visible(at(part, span, index))[:, None, None]
        logits = jnp.where(seen, logits, -jnp.inf)
        lse = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        p = jnp.exp(logits - finite(lse)).astype(part.v.dtype)
        out = jnp.einsum("bgrcs,bsgv->bgrcv", p, part.v)
        return _ungrouped(out), _ungrouped(lse)[..., 0]

    s, h = part.k.shape[1], q.shape[2]
    with flops.over_keys():
        return chunked(chunk, (q, *rows(part)), chunks(q.shape[1], h * s))


def _xla_bwd(
    q: Float[Array, "b t h dk"],
    part: Part,
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b t h"],
    static: _Static,
) -> tuple[
    Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
]:
    """Return a part's share of the gradients, flash attention's backward pass from
    the softmax's `lse`, a chunk at a time."""
    k, v, g = part.k, part.v, part.k.shape[2]

    def chunk(carry: tuple, xs: tuple) -> tuple[tuple, Float[Array, "b c h dk"]]:
        q, out, dout, lse, span, index = xs
        with flops.recomputed():
            raw, logits = _logits(q, k, static.scale, static.softcap)
        seen = visible(at(part, span, index))[:, None, None]
        p = jnp.where(seen, jnp.exp(logits - _grouped(finite(lse)[..., None], g)), 0)
        do, o = (_grouped(x.astype(F32), g) for x in (dout, out))
        dp = jnp.einsum("bgrcv,bsgv->bgrcs", do, v, preferred_element_type=F32)
        ds = p * (dp - (do * o).sum(-1, keepdims=True))
        if static.softcap is not None:
            ds = ds * (1 - jnp.tanh(raw / static.softcap) ** 2)
        ds = ds * static.scale
        dq = jnp.einsum("bgrcs,bsgd->bgrcd", ds, k, preferred_element_type=F32)
        dk = jnp.einsum(
            "bgrcs,bgrcd->bsgd", ds, _grouped(q, g), preferred_element_type=F32
        )
        dv = jnp.einsum("bgrcs,bgrcv->bsgv", p, do)
        return (carry[0] + dk, carry[1] + dv), _ungrouped(dq).astype(q.dtype)

    zeros = jnp.zeros(k.shape, F32), jnp.zeros(v.shape, F32)
    xs = (q, out, dout, lse, *rows(part))
    n = chunks(q.shape[1], q.shape[2] * k.shape[1])
    with flops.over_keys():
        (dk, dv), dq = scanned(chunk, zeros, xs, n)
    return dq, dk.astype(k.dtype), dv.astype(v.dtype)


# --- by tokamax's flash kernel --------------------------------------------------------


def _flash_fwd(
    q: Float[Array, "b t h dk"], part: Part, static: _Static
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
    """Return a part's softmax-weighted values and logsumexp by tokamax's kernel; 0
    and -inf for a query that attends to nothing."""
    assert MOSAIC is not None
    out, (m, l) = MOSAIC(
        q, part.k, part.v, return_residuals=True, **_flash_args(part, static)
    )
    seen = (l > 0).transpose(0, 2, 1)
    return jnp.where(seen[..., None], out, 0), tokamax_lse(m, l).transpose(0, 2, 1)


def _flash_bwd(
    q: Float[Array, "b t h dk"],
    part: Part,
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b t h"],
    static: _Static,
) -> tuple[
    Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
]:
    """Return a part's share of the gradients by tokamax's backward kernels, given
    the softmax's statistics: P = exp(logits - lse), and rowsum(dout * out)."""
    assert MOSAIC is not None
    args, lse = _flash_args(part, static), finite(lse).transpose(0, 2, 1)
    return jax_internals.tokamax_vjp(MOSAIC, lse, out, dout, q, part.k, part.v, args)


def _flash_args(part: Part, static: _Static) -> dict[str, Any]:
    """Return tokamax's arguments for `part`: the kept and existing slots as a
    boolean mask, the span as each query's first key and, with the positions split,
    the key after its last."""
    keep = (
        None if part.kept is None else mask(part.span, part.k.shape[1], None, part.kept)
    )
    keep = None if keep is None else keep[:, None]
    if part.exists is not None:
        exists = part.exists[:, None, None]
        keep = exists if keep is None else keep & exists
    end = None if static.causal else part.span.end[:, None]
    return {
        "mask": jax_internals.tokamax_mask(
            keep, k_start=part.span.start[:, None], k_end=end, is_causal=static.causal
        ),
        "logits_scale": static.scale,
        "logits_soft_cap": static.softcap,
        "logits_dtype": F32,
    }
