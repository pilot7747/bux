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
from bux.parallel.flash import Keys, per_query_head
from bux.parallel.layout import as_split, axes_of, shard, spec
from bux.parallel.mesh import active, free_axes
from bux.train import flops

AXIS = "context"
MIN = float(jnp.finfo(jnp.float32).min)  # the logit of a masked key
# Below, a 1.3B on a B300 trains on packed documents as fast by tokamax as by cuDNN's
# packed layout, whose fallback past its bound (a lax.cond) costs 5%.
CUDNN_ROWS = 8192
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


def scaled_logits(
    q: Float[Array, "b t h dk"], k: Float[Array, "b s g dk"], scale: float
) -> Float[Array, "b h t s"]:
    """Return every query's dot product with every key, times `scale`, in float32;
    the keys stay whole along their positions."""
    k = per_query_head(k, q.shape[2])
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
    v = per_query_head(v, probs.shape[1])
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
    t = q.shape[1]
    mask = causal_mask(t, args.window, args.ids)[..., None, :, :]
    with flops.attention(args.window), flops.over_keys():  # for MFU
        logits = softcap(scaled_logits(q, k, args.scale), args.softcap)
        return weighted_values(masked_softmax(logits, mask, args.sinks), v)


# --- on each device -----------------------------------------------------------------


def splits_positions(q: Float[Array, "b t h dk"]) -> bool:
    """Return whether the run's layout splits the positions of `q` over `context`."""
    split = spec(("batch", "seq", "heads", None), q.shape)
    return len(split) > 1 and AXIS in as_split(split[1])


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
        k, v = (per_query_head(x, q.shape[2]) for x in (k, v))
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
        axis_names=free_axes(),
        check_vma=False,
    )(q, k, v, rows, sinks)


# Rows up to this long run XLA's own cummax and cumsum, longer ones (long context, split
# over `context`) a prefix scan of log2(t) steps. There XLA's take 2 and 25 ms a row of
# 128k on a B300; in rows of 4096 the scan took a 1f1b pipeline of the 1.3B 1.6% slower.
SCAN_ROWS = 8192


def running_max(x: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """Return the largest of each row's values up to each of its tokens."""
    if x.shape[1] <= SCAN_ROWS:
        return jax.lax.cummax(x, axis=1)
    return jax.lax.associative_scan(jnp.maximum, x, axis=1)


def running_sum(x: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """Return the sum of each row's values up to each of its tokens."""
    if x.shape[1] <= SCAN_ROWS:
        return jnp.cumsum(x, axis=1)
    return jax.lax.associative_scan(jnp.add, x, axis=1)


def document_starts(ids: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """Return the index of the first token of each token's document: the first key it
    attends to."""
    index = jnp.arange(ids.shape[1])
    new = jnp.concatenate(
        [jnp.ones_like(ids[:, :1], bool), ids[:, 1:] != ids[:, :-1]], 1
    )
    return running_max(jnp.where(new, index, 0))


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
    call: `causal_kernel`'s where each row is one document, else `masked_rows`'."""
    return by_documents(causal_kernel(q, k, v, args), masked_rows, q, k, v, args)


def by_documents(
    kernel: flash.Kernel | None,
    masked: Callable[..., Float[Array, "b t h dv"]],
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Attend by `kernel`, causally, where each row is one document, else by
    `masked(q, k, v, args)`: a lax.cond, as the rows of a run without documents have
    ids too, all one. By `masked` alone where `kernel` is None."""
    if kernel is None:
        return masked(q, k, v, args)
    causal = partial(_flash, kernel, args=args._replace(ids=None))
    if args.ids is None:
        return causal(q, k, v)
    one = flash.documents_in(args.ids) == 1
    return jax.lax.cond(one, causal, partial(masked, args=args), q, k, v)


def causal_kernel(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> flash.Kernel | None:
    """Return the kernel for causal rows of one document, within no window: cuDNN's,
    None where it does not take the call."""
    whole = args.window is None
    if whole and flash.cudnn_kernel_takes(q, k, v, Keys(True), args.softcap):
        return flash.CUDNN
    return None


def masked_rows(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Attend with whole rows of documents, or within a window, by the fastest kernel
    that takes the call: cuDNN's in rows of `CUDNN_ROWS` or more and with a window,
    else tokamax's (a soft cap, sinks), else cuDNN's; in float32, tokamax's Triton
    kernel, or `xla` with sinks."""
    cudnn = flash.cudnn_takes(q, k, v, args.softcap, args.sinks)
    if cudnn and (q.shape[1] >= CUDNN_ROWS or args.window is not None):
        return _cudnn(q, k, v, args)
    if flash.tokamax_takes(q):
        return _flash(flash.TOKAMAX, q, k, v, args)
    if args.sinks is None and flash.triton_takes(q):
        keys = Keys(True, first_keys(args, *q.shape[:2]))
        return flash.triton(q, k, v, keys, args.scale, args.softcap)
    return _cudnn(q, k, v, args) if cudnn else xla(q, k, v, args)


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
    cuDNN's packed layout has room for."""
    cudnn = partial(flash.cudnn, ids=args.ids, window=args.window, scale=args.scale)
    if args.ids is None:
        return cudnn(q, k, v, documents=1)
    if flash.tokamax_takes(q):
        other = partial(_flash, flash.TOKAMAX, args=args)
    else:
        other = partial(xla, args=args)
    if not (most := documents_bound(*q.shape[:3], max(q.shape[-1], v.shape[-1]))):
        return other(q, k, v)
    fits = flash.documents_in(args.ids) <= most
    return jax.lax.cond(fits, partial(cudnn, documents=most), other, q, k, v)


def documents_bound(b: int, t: int, h: int, d: int) -> int:
    """Return the most documents a row that cuDNN's packed layout makes room for, in
    b rows of t positions of h heads of d channels: its backward pass takes time in
    proportion to the room and the row, and 8 bytes of memory for each of the room's
    positions of every head's channels. So up to t / 256, as tokamax's backward pass
    takes t^2 / 2, and 2^30 such numbers (8 GiB)."""
    most = min(t // 256, 2**30 // (b * t * h * d))
    return 2 ** (most.bit_length() - 1) if most > 0 else 0


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
    takes the call (`whole_rows`), and all-to-all back. Context must divide a
    device's heads; kv heads it does not divide are repeated."""
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
        k, v = (per_query_head(x, h) for x in (k, v))
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
    """Run cuDNN's fused attention (jax's) on each device's whole rows: rows of one
    document by its causal kernels (`flash.CUDNN`), documents in its packed layout,
    which tokamax takes a row of more of (`documents_bound`). bfloat16 or float16,
    and equal head dims of keys and values up to 256; no soft cap, no sinks, and no
    positions over `context`: `ulysses` gives a device whole rows of fewer heads."""
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
    body = partial(_rows, args=args, attend=_cudnn_rows)
    return on_devices(body, q, k, v, args.ids, None, seq=False)


def _cudnn_rows(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Run `cudnn` on a device's whole rows: by `causal_kernel`'s where each row is
    one document, else in cuDNN's packed layout."""
    return by_documents(causal_kernel(q, k, v, args), _cudnn, q, k, v, args)


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
    if not flash.tokamax_takes(q):
        raise NotImplementedError(
            "methods.attention = tokamax needs tokamax's GPU kernels: a GPU, the cuda "
            "extra (Linux), and bfloat16 or float16"
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
    whole rows (`whole_rows`): cuDNN's causal kernels where each row is one document
    (with sinks too), else cuDNN's in long rows or with a window (on a B300 4x
    tokamax's speed with packed documents), else tokamax's, which takes sinks and a
    soft cap; in float32 its Triton kernel, or `xla`. With `seq` over `context`,
    `ulysses` where context divides a device's heads, else `ring`. `xla` where
    neither kernel runs, and for what only xla has."""
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
    """`ulysses`, whose kernel sees whole rows, where context divides a device's
    heads; else `ring`. On 8 B300s, rows of 32k over context=8, a 218M model trains
    at 273k tok/s on packed documents by ulysses and 221k by ring, at 252k and 175k
    on rows of one document."""
    on = active()
    assert on is not None
    heads = q.shape[2] // on.size(*axes_of(spec(("heads",), q.shape[2:3])))
    if heads % on.size(AXIS) == 0:
        return ulysses(q, k, v, args)
    return methods.attention.find("ring")(q, k, v, args)
