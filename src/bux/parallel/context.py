"""The methods of `methods.attention`: `xla`, `ulysses`, `cudnn`, `tokamax` and `auto`,
and bux.parallel.ring's `ring`.

A method maps queries (b, t, h, dk), keys (b, s, g, dk) and values (b, s, g, dv) at
the queries' positions to each query's softmax-weighted values (b, t, h, dv), the
softmax in float32. `AttentionArgs` holds the mask, scale, softcap and sinks; a method
that lacks one raises NotImplementedError naming it. The kernels are
bux.parallel.flash's.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from bux.parallel import flash, methods
from bux.parallel.flash import Keys
from bux.parallel.layout import shard, spec
from bux.parallel.mesh import active

AXIS = "context"
MIN = float(jnp.finfo(jnp.float32).min)  # the logit of a masked key
Body = Callable[..., Float[Array, "b t h dv"]]  # (q, k, v, rows, sinks) -> out


class AttentionArgs(NamedTuple):
    """The mask, scale, softcap and sinks that a method takes beside q, k and v."""

    ids: Int[Array, "b t"] | None  # each token's document (Segments.ids), or none
    window: int | None  # a query attends to no key `window` or more behind it
    scale: float  # on every logit
    softcap: float | None  # logits to softcap * tanh(logits / softcap)
    sinks: Float[Array, " h"] | None  # a per-head logit beside the keys', on no value


def causal_mask_at(
    window: int | None,
    q_index: Int[Array, " t"],
    k_index: Int[Array, " s"],
    ids: tuple[Int[Array, "b t"], Int[Array, "b s"]] | None = None,
) -> Bool[Array, "*b t s"]:
    """Return which keys each query may attend to, by their indices in the row: none
    ahead of it, none `window` or more behind, none of another document by `ids`."""
    offset = q_index[:, None] - k_index[None, :]
    mask = offset >= 0
    if window is not None:
        mask &= offset < window
    if ids is None:
        return mask
    return mask & (ids[0][:, :, None] == ids[1][:, None, :])


def causal_mask(
    t: int, window: int | None, ids: Int[Array, "b t"] | None
) -> Bool[Array, "*b t t"]:
    """Return `causal_mask_at` for t queries and the t keys at their positions."""
    index = jnp.arange(t)
    return causal_mask_at(window, index, index, None if ids is None else (ids, ids))


def softcap(x: Float[Array, "*s"], cap: float | None) -> Float[Array, "*s"]:
    return x if cap is None else cap * jnp.tanh(x / cap)


def shard_logits(logits: Float[Array, "b h t s"]) -> Float[Array, "b h t s"]:
    return shard(logits, "batch", "heads", "seq", None)


def _per_query_head(kv: Float[Array, "b s g e"], heads: int) -> Float[Array, "b s h e"]:
    """Repeat keys or values for every query head of their group."""
    return jnp.repeat(kv, heads // kv.shape[2], axis=2)


def scaled_logits(
    q: Float[Array, "b t h dk"], k: Float[Array, "b s g dk"], scale: float
) -> Float[Array, "b h t s"]:
    """Return every query's dot product with every key, times `scale`, in float32;
    the keys stay whole along their positions."""
    k = _per_query_head(k, q.shape[2])
    logits = jnp.einsum("bthk,bshk->bhts", q, k, preferred_element_type=jnp.float32)
    return shard_logits(logits * scale)


def masked_softmax(
    logits: Float[Array, "b h t s"],
    mask: Bool[Array, "*lead t s"],
    sinks: Float[Array, " h"] | None,
) -> Float[Array, "b h t s"]:
    """Return the softmax over the keys `mask` keeps, beside each head's sink logit if
    any, in float32; a query that keeps no key gets zeros."""
    logits = jnp.where(mask, logits.astype(jnp.float32), MIN)
    if sinks is None:
        probs = jax.nn.softmax(logits)
    else:
        probs = _softmax_with_sinks(logits, sinks.astype(jnp.float32))
    return jnp.where(mask, probs, 0)


def _softmax_with_sinks(
    logits: Float[Array, "b h t s"], sinks: Float[Array, " h"]
) -> Float[Array, "b h t s"]:
    """Return the softmax over the keys and a per-head sink logit, which takes a share
    of the probability and attends to nothing."""
    b, h, t, _ = logits.shape
    sink = jnp.broadcast_to(sinks[None, :, None, None], (b, h, t, 1))
    return jax.nn.softmax(jnp.concatenate([logits, sink], -1))[..., :-1]


def weighted_values(
    probs: Float[Array, "b h t s"], v: Float[Array, "b s g dv"]
) -> Float[Array, "b t h dv"]:
    """Return every query's values, weighted by its probabilities."""
    v = _per_query_head(v, probs.shape[1])
    return jnp.einsum("bhts,bshv->bthv", probs.astype(v.dtype), v)


@methods.attention.register("xla")
def xla(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Compute attention from all t by t logits at once, the reference method. With
    `seq` over `context`, a device has its queries' rows of the logits and every key
    and value: all-gather context parallelism."""
    mask = causal_mask(q.shape[1], args.window, args.ids)[..., None, :, :]
    logits = softcap(scaled_logits(q, k, args.scale), args.softcap)
    return weighted_values(masked_softmax(logits, mask, args.sinks), v)


# --- on each device -----------------------------------------------------------------


def _axes(split: str | tuple[str, ...] | None) -> set[str]:
    """Return the mesh axes of one dimension of a partition spec."""
    return set() if split is None else {split} if isinstance(split, str) else set(split)


def _free_axes() -> set[str]:
    """Return the axes of the run's mesh that no shard_map being traced maps."""
    on = active()
    assert on is not None
    return set(on.mesh.axis_names) - set(jax.sharding.get_abstract_mesh().manual_axes)


def splits_positions(q: Float[Array, "b t h dk"]) -> bool:
    """Return whether the run's layout splits the positions of `q` over `context`."""
    split = spec(("batch", "seq", "heads", None), q.shape)
    return len(split) > 1 and AXIS in _axes(split[1])


def _specs(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    seq: bool,
) -> tuple[Float[Array, "b s G dk"], Float[Array, "b s G dv"], P, P]:
    """Return k and v, repeated if their heads split otherwise than the queries', and
    how to split q, and k and v: rows and heads as the layout says, positions over
    `context` if `seq`, else whole."""
    queries = spec(("batch", None, "heads", None), q.shape)
    keys = spec(("batch", None, "kv_heads", None), k.shape)
    if keys[2] != queries[2]:
        k, v = (_per_query_head(x, q.shape[2]) for x in (k, v))
        keys = queries
    positions = AXIS if seq else None
    return k, v, P(queries[0], positions, queries[2]), P(keys[0], positions, keys[2])


def on_devices(
    body: Body,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    rows: Int[Array, "b t"] | None,
    sinks: Float[Array, " h"] | None,
    seq: bool,
) -> Float[Array, "b t h dv"]:
    """Return body(q, k, v, rows, sinks) of each device's part of them, rows and
    heads split as the layout says and positions over `context` if `seq`: a shard_map
    manual over every axis. XLA cannot split a kernel's call, nor an all-to-all over
    auto axes without gathering every dimension they split, and cuDNN's partitioner
    fails beside manual axes. Kernels need check_vma=False, with which the shard_map
    sums the gradient of the sinks over the devices that share them."""
    if active() is None:
        return body(q, k, v, rows, sinks)
    k, v, queries, keys = _specs(q, k, v, seq)
    return jax.shard_map(
        body,
        in_specs=(queries, keys, keys, P(*queries[:2]), P(queries[2])),
        out_specs=queries,
        axis_names=_free_axes(),
        check_vma=False,
    )(q, k, v, rows, sinks)


def document_starts(ids: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """Return the index of the first token of each token's document: the first key it
    attends to."""
    index = jnp.arange(ids.shape[1])
    new = jnp.concatenate(
        [jnp.ones_like(ids[:, :1], bool), ids[:, 1:] != ids[:, :-1]], 1
    )
    return jax.lax.cummax(jnp.where(new, index, 0), axis=1)


def first_keys(args: AttentionArgs, b: int, t: int) -> Int[Array, "b t"] | None:
    """Return the first key each of t queries sees: its document's first token, and
    none `window` or more behind; None if every query sees its row from the start."""
    start = None if args.ids is None else document_starts(args.ids)
    if args.window is None:
        return start
    behind = jnp.broadcast_to(jnp.maximum(jnp.arange(t) - args.window + 1, 0), (b, t))
    return behind if start is None else jnp.maximum(start, behind)


# --- whole rows: the kernels ----------------------------------------------------------


def whole_rows(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Attend with whole rows, on one device, by the fastest kernel that takes the
    call: cuDNN's, else tokamax's (a soft cap, sinks, float32), else `xla`."""
    if flash.cudnn_takes(q, k, v, args.softcap, args.sinks):
        return _cudnn(q, k, v, args)
    if flash.tokamax_runs():
        return _flash(flash.TOKAMAX, q, k, v, args)
    return xla(q, k, v, args)


def _flash(
    kernel: flash.Kernel,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Attend by a flash kernel, a window as each query's first key: tokamax's
    backward pass then skips what is ahead of the queries, as with no window."""
    keys = Keys(True, first_keys(args, *q.shape[:2]))
    return flash.attend(kernel, q, k, v, keys, args.sinks, args.scale, args.softcap)


def _cudnn(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Attend by cuDNN, or by tokamax (else xla) for a row of more documents than
    cuDNN's packed layout holds."""
    cudnn = partial(flash.cudnn, ids=args.ids, window=args.window, scale=args.scale)
    if args.ids is None:
        return cudnn(q, k, v, documents=1)
    if flash.tokamax_runs():
        other = partial(_flash, flash.TOKAMAX, args=args)
    else:
        other = partial(xla, args=args)
    if not (most := documents_bound(q, v)):
        return other(q, k, v)
    fits = flash.documents_in(args.ids) <= most
    return jax.lax.cond(fits, partial(cudnn, documents=most), other, q, k, v)


def documents_bound(q: Float[Array, "b t h dk"], v: Float[Array, "b t g dv"]) -> int:
    """Return the most documents a row that cuDNN's packed layout takes: its backward
    pass takes time in proportion to them and the row, and 8 bytes of memory for each
    of b documents' t positions of every head's channels. So up to t / 256, as
    tokamax's backward takes t^2 / 2, and 2^30 such numbers (8 GiB)."""
    b, t, h, dk = q.shape
    most = min(t // 256, 2**30 // (b * t * h * max(dk, v.shape[-1])))
    return max(1, 2 ** (most.bit_length() - 1)) if most > 0 else 0


# --- ulysses ----------------------------------------------------------------------


@methods.attention.register("ulysses")
def ulysses(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """All-to-all over `context` from a device's positions of every head to every
    position of its own heads, attend with those whole rows by the fastest kernel that
    takes the call (`whole_rows`), and all-to-all back. Context must divide a device's
    heads; kv heads it does not divide are repeated."""
    if active() is None:
        return whole_rows(q, k, v, args)
    body = partial(_ulysses, args=args._replace(ids=None, sinks=None))
    return on_devices(body, q, k, v, args.ids, args.sinks, seq=True)


def _ulysses(
    q: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    args: AttentionArgs,
) -> Float[Array, "b n h dv"]:
    """Run a device's part of `ulysses`: its n positions of its h heads in, and out."""
    c, me = jax.lax.axis_size(AXIS), jax.lax.axis_index(AXIS)
    h, g = q.shape[2], k.shape[2]
    if h % c:
        raise ValueError(
            f"methods.attention = ulysses: context = {c} does not divide the {h} "
            "heads of a device"
        )
    if g % c:
        k, v = (jnp.repeat(x, h // g, axis=2) for x in (k, v))
    if ids is not None:
        ids = jax.lax.all_gather(ids, AXIS, axis=1, tiled=True)
    if sinks is not None:
        sinks = jax.lax.dynamic_slice_in_dim(sinks, me * (h // c), h // c)
    q, k, v = (_to_heads(x) for x in (q, k, v))
    out = whole_rows(q, k, v, args._replace(ids=ids, sinks=sinks))
    return _to_positions(out)


def _to_heads(x: Float[Array, "b n h e"]) -> Float[Array, "b t hc e"]:
    """Move a device's positions of every head to every position of its own heads:
    device j of c gets the j-th c-th of the heads, so a query head's kv head goes
    with it."""
    return jax.lax.all_to_all(x, AXIS, split_axis=2, concat_axis=1, tiled=True)


def _to_positions(x: Float[Array, "b t hc e"]) -> Float[Array, "b n h e"]:
    """Undo `_to_heads`."""
    return jax.lax.all_to_all(x, AXIS, split_axis=1, concat_axis=2, tiled=True)


# --- cudnn, tokamax and auto ------------------------------------------------------


@methods.attention.register("cudnn")
def cudnn(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Run cuDNN's fused attention (jax's) on each device's whole rows, documents in
    its packed layout, which tokamax takes a row of more of (`documents_bound`).
    bfloat16 or float16, and equal head dims of keys and values up to 256; no soft
    cap, no sinks, and no positions over `context`: `ulysses` gives a device whole
    rows of fewer heads."""
    for lacks, feature in (
        (args.softcap is not None, "softcap"),
        (args.sinks is not None, "sinks"),
        (splits_positions(q), "context parallelism (ulysses runs it on whole rows)"),
    ):
        if lacks:
            raise NotImplementedError(f"methods.attention = cudnn has no {feature}")
    if not flash.cudnn_takes(q, k, v, None, None):
        raise NotImplementedError(
            "methods.attention = cudnn needs cuDNN on a GPU, bfloat16 or float16, "
            "and keys' and values' head dims equal, up to 256 in steps of 8"
        )
    body = partial(_rows, args=args, attend=_cudnn)
    return on_devices(body, q, k, v, args.ids, None, seq=False)


def _rows(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    ids: Int[Array, "b t"] | None,
    sinks: Float[Array, " h"] | None,
    args: AttentionArgs,
    attend: Callable[..., Float[Array, "b t h dv"]],
) -> Float[Array, "b t h dv"]:
    """Run `attend` on a device's whole rows, by their ids and sinks."""
    return attend(q, k, v, args._replace(ids=ids, sinks=sinks))


@methods.attention.register("tokamax")
def tokamax(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Run tokamax's flash attention (its Mosaic kernels, on a GPU), a document and a
    window by each query's first key, sinks by the logsumexp. With `seq` over
    `context`, a device gathers every key and value and attends with its queries:
    all-gather CP in a kernel, whose backward pass does every block."""
    if not flash.tokamax_runs():
        raise NotImplementedError(
            "methods.attention = tokamax needs tokamax's GPU kernels: a GPU, and the "
            "cuda extra (Linux)"
        )
    return gathered(flash.TOKAMAX, q, k, v, args)


def gathered(
    kernel: flash.Kernel,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Attend by `kernel` on each device, every key and value gathered if `context`
    splits the positions."""
    seq = splits_positions(q)
    start = first_keys(args, *q.shape[:2])
    body = partial(
        _gathered, kernel=kernel, seq=seq, scale=args.scale, cap=args.softcap
    )
    return on_devices(body, q, k, v, start, args.sinks, seq)


def _gathered(
    q: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    start: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    kernel: flash.Kernel,
    seq: bool,
    scale: float,
    cap: float | None,
) -> Float[Array, "b n h dv"]:
    """Run a device's part of `gathered`: its queries against every key up to theirs."""
    keys = Keys(True, start)
    if seq:
        k, v = (jax.lax.all_gather(x, AXIS, axis=1, tiled=True) for x in (k, v))
        n = q.shape[1]
        end = jax.lax.axis_index(AXIS) * n + jnp.arange(n) + 1
        keys = Keys(False, start, jnp.broadcast_to(end, q.shape[:2]))
    return flash.attend(kernel, q, k, v, keys, sinks, scale, cap)


@methods.attention.register("auto")
def auto(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """The default. On a GPU, the fastest kernel that takes the call on each device's
    whole rows (`whole_rows`): cuDNN's (on a B300 1.3x tokamax's speed causal at
    t = 32k, 4x with packed documents), else tokamax's. With `seq` over `context`,
    `ring` for documents, else `ulysses` where context divides a device's heads.
    `xla` where neither kernel runs, and for what only xla has."""
    if not (flash.tokamax_runs() or flash.cudnn_runs()):
        return xla(q, k, v, args)
    if splits_positions(q):
        return _context_parallel(q, k, v, args)
    body = partial(_rows, args=args, attend=whole_rows)
    return on_devices(body, q, k, v, args.ids, args.sinks, seq=False)


def _context_parallel(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """`ring` for documents, whose blocks no query sees it skips (on 8 B300s, 2k
    documents in rows of 128k: 3.4x ulysses' speed, as cuDNN's packed layout takes
    too few of them); else `ulysses`, whose kernel sees whole rows (one causal
    document a row: 1.5x ring's), where context divides a device's heads."""
    on = active()
    assert on is not None
    heads = q.shape[2] // on.size(*_axes(spec(("heads",), q.shape[2:3])[0]))
    if args.ids is None and heads % on.size(AXIS) == 0:
        return ulysses(q, k, v, args)
    return methods.attention.find("ring")(q, k, v, args)
