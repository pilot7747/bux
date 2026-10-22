"""The methods of `methods.attention`: `xla`, `ring`, `ulysses`, `tokamax` and `auto`.

A method maps queries (b, t, h, dk), keys (b, s, g, dk) and values (b, s, g, dv) at
the queries' positions to each query's softmax-weighted values (b, t, h, dv), the
softmax in float32. `AttentionArgs` holds the mask, scale, softcap and sinks; a method
that lacks one raises NotImplementedError naming it.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from bux import kernels
from bux.parallel import methods
from bux.parallel.layout import shard, spec
from bux.parallel.mesh import active

try:  # tokamax's kernels, where its extra is installed (Linux)
    from tokamax._src.ops.attention import api as flash
    from tokamax._src.ops.attention import base as masks
except ImportError:
    flash = masks = None

AXIS = "context"
POSITIONS = P(None, AXIS)  # a (b, t, ...) array, its positions over context
MIN = float(jnp.finfo(jnp.float32).min)  # the logit of a masked key
KERNELS = ("mosaic_gpu", "mosaic_tpu", "triton", "xla")  # tokamax's, fastest first


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


# --- ring -------------------------------------------------------------------------


@methods.attention.register("ring")
def ring(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Keep each device's queries while the keys and values go round `context` one
    block at a time (ppermute) into an online softmax, skipping blocks wholly ahead of
    the queries or a window or more behind. The backward pass sends them round again
    with their gradients, so a device holds only its block and the one passing by."""
    if active() is None:
        return xla(q, k, v, args)
    body = partial(_ring, args=args._replace(ids=None, sinks=None), split=k.shape[-1])
    return jax.shard_map(
        body,
        in_specs=(POSITIONS, POSITIONS, POSITIONS, P()),
        out_specs=POSITIONS,
        axis_names={AXIS},
    )(q, jnp.concatenate([k, v], -1), args.ids, args.sinks)


class Device(NamedTuple):
    """A device's place in the ring, and the document ids of its queries and keys."""

    me: Int[Array, ""]
    c: int
    n: int  # positions per block
    q_ids: Int[Array, "b n"] | None
    k_ids: Int[Array, "b t"] | None

    @classmethod
    def of(cls, ids: Int[Array, "b n"] | None, n: int) -> "Device":
        me, c = jax.lax.axis_index(AXIS), jax.lax.axis_size(AXIS)
        if ids is None:
            return cls(me, c, n, None, None)
        return cls(me, c, n, ids, jax.lax.all_gather(ids, AXIS, axis=1, tiled=True))

    def mask(self, j: Int[Array, ""], window: int | None) -> Bool[Array, "*b n s"]:
        """Return which keys of block `j` each of the device's queries may attend to."""
        n, ids = self.n, None
        if self.q_ids is not None and self.k_ids is not None:
            k_ids = jax.lax.dynamic_slice_in_dim(self.k_ids, j * n, n, axis=1)
            ids = (self.q_ids, k_ids)
        q_index, k_index = self.me * n + jnp.arange(n), j * n + jnp.arange(n)
        return causal_mask_at(window, q_index, k_index, ids)

    def skips(self, j: Int[Array, ""], window: int | None) -> Bool[Array, ""]:
        """Return whether block `j` is wholly ahead of the device's queries, or a
        window or more behind."""
        ahead = j > self.me
        if window is None:
            return ahead
        return ahead | ((self.me - j - 1) * self.n + 1 >= window)

    def send[T](self, x: T) -> T:
        """Send `x` to the next device of the ring, the last device's to the first."""
        c = self.c
        return jax.lax.ppermute(x, AXIS, [(a, (a + 1) % c) for a in range(c)])


def _round[C, B](
    visit: Callable[[C, B, Int[Array, ""]], tuple[C, B]],
    carry: C,
    block: B,
    device: Device,
    window: int | None,
) -> tuple[C, B]:
    """Call `visit` on each block as it comes round, the device's own first, except
    those it skips; `block` (the keys and values and their companions) is sent on."""

    def step(state: tuple[C, B], i: Int[Array, ""]) -> tuple[tuple[C, B], None]:
        carry, block = state
        block = device.send(block)
        j = (device.me - i) % device.c  # whose block it is
        skip = device.skips(j, window)
        go = jax.lax.cond(skip, lambda *s: s, lambda c, b: visit(c, b, j), carry, block)
        return go, None

    first = visit(carry, block, device.me)
    return jax.lax.scan(step, first, jnp.arange(1, device.c))[0]


class OnlineSoftmax(NamedTuple):
    """The online softmax of each query head (b, g, r, n): the largest logit so far,
    the sum of exp(logit - largest), and the values so weighted."""

    max_logit: Float[Array, "b g r n"]
    exp_sum: Float[Array, "b g r n"]
    out: Float[Array, "b g r n dv"]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _ring(
    q: Float[Array, "b n h dk"],
    kv: Float[Array, "b n g e"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    args: AttentionArgs,
    split: int,
) -> Float[Array, "b n h dv"]:
    """Run a device's part of `ring`: its n queries against each block of n keys and
    values (`kv`, split at `split`) as it comes round."""
    return _ring_forward(q, kv, ids, sinks, args, split)[0]


def _ring_forward(
    q: Float[Array, "b n h dk"],
    kv: Float[Array, "b n g e"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    args: AttentionArgs,
    split: int,
) -> tuple[Float[Array, "b n h dv"], Float[Array, "b g r n"]]:
    """Return `_ring`'s output, and each query's log of the sum of exp(logit)."""
    b, n, h, _ = q.shape
    g, device = kv.shape[2], Device.of(ids, n)
    grouped = _group_heads(q, g)

    def visit(
        state: OnlineSoftmax, kv: Float[Array, "b n g e"], j: Int[Array, ""]
    ) -> tuple[OnlineSoftmax, Float[Array, "b n g e"]]:
        k, v, mask = kv[..., :split], kv[..., split:], device.mask(j, args.window)
        return _add_block(state, grouped, k, v, mask, args), kv

    start = _initial_softmax(sinks, (b, g, h // g, n), kv.shape[-1] - split)
    state, _ = _round(visit, start, kv, device, args.window)
    out = _ungroup_heads(state.out / state.exp_sum[..., None]).astype(kv.dtype)
    return out, state.max_logit + jnp.log(state.exp_sum)


def _ring_fwd(
    q: Float[Array, "b n h dk"],
    kv: Float[Array, "b n g e"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    args: AttentionArgs,
    split: int,
) -> tuple[Float[Array, "b n h dv"], tuple]:
    out, lse = _ring_forward(q, kv, ids, sinks, args, split)
    return out, (q, kv, ids, sinks, out, lse)


def _ring_bwd(
    args: AttentionArgs, split: int, saved: tuple, dout: Float[Array, "b n h dv"]
) -> tuple:
    """Return the gradients of `_ring`: the blocks go round again with their
    gradients, which are sent home at the end."""
    q, kv, ids, sinks, out, lse = saved
    g, device = kv.shape[2], Device.of(ids, q.shape[1])
    grouped, do = _group_heads(q, g), _group_heads(dout.astype(jnp.float32), g)
    delta = (do * _group_heads(out.astype(jnp.float32), g)).sum(-1)

    def visit(
        dq: Float[Array, "b g r n dk"],
        block: tuple[Float[Array, "b n g e"], Float[Array, "b n g e"]],
        j: Int[Array, ""],
    ) -> tuple[
        Float[Array, "b g r n dk"],
        tuple[Float[Array, "b n g e"], Float[Array, "b n g e"]],
    ]:
        kv, dkv = block
        k, v, mask = kv[..., :split], kv[..., split:], device.mask(j, args.window)
        ddq, dk, dv = _block_grads(grouped, k, v, do, lse, delta, mask, args)
        return dq + ddq, (kv, dkv + jnp.concatenate([dk, dv], -1))

    start = jnp.zeros_like(grouped, jnp.float32), (kv, jnp.zeros_like(kv, jnp.float32))
    dq, (_, dkv) = _round(visit, *start, device, args.window)
    dsinks = None if sinks is None else _sink_grads(sinks, lse, delta)
    dq = _ungroup_heads(dq).astype(q.dtype)
    return dq, device.send(dkv).astype(kv.dtype), None, dsinks


_ring.defvjp(_ring_fwd, _ring_bwd)


def _group_heads(x: Float[Array, "b n h e"], g: int) -> Float[Array, "b g r n e"]:
    """Group the query heads of `x` by the kv head they share, g groups."""
    b, n, h, e = x.shape
    return x.reshape(b, n, g, h // g, e).transpose(0, 2, 3, 1, 4)


def _ungroup_heads(x: Float[Array, "b g r n e"]) -> Float[Array, "b n h e"]:
    b, g, r, n, e = x.shape
    return x.transpose(0, 3, 1, 2, 4).reshape(b, n, g * r, e)


def _initial_softmax(
    sinks: Float[Array, " h"] | None, shape: tuple[int, int, int, int], dv: int
) -> OnlineSoftmax:
    """Return the online softmax of queries of `shape` (b, g, r, n) before any key:
    only the sinks, if any."""
    if sinks is None:
        max_logit, exp_sum = jnp.full(shape, MIN), jnp.zeros(shape)
    else:
        max_logit = jnp.broadcast_to(_grouped_sinks(sinks, shape), shape)
        exp_sum = jnp.ones(shape)
    return OnlineSoftmax(max_logit, exp_sum, jnp.zeros((*shape, dv)))


def _grouped_sinks(
    sinks: Float[Array, " h"], shape: tuple[int, ...]
) -> Float[Array, "1 g r 1"]:
    return sinks.reshape(shape[1:3]).astype(jnp.float32)[None, :, :, None]


def _block_logits(
    q: Float[Array, "b g r n dk"],
    k: Float[Array, "b s g dk"],
    mask: Bool[Array, "*lead n s"],
    args: AttentionArgs,
) -> tuple[Float[Array, "b g r n s"], Float[Array, "b g r n s"]]:
    """Return a block's logits in float32, before the softcap and after it, the latter
    at MIN for masked keys."""
    raw = jnp.einsum("bgrnk,bsgk->bgrns", q, k, preferred_element_type=jnp.float32)
    raw = raw * args.scale
    mask = mask[:, None, None] if mask.ndim == 3 else mask
    return raw, jnp.where(mask, softcap(raw, args.softcap), MIN)


def _add_block(
    state: OnlineSoftmax,
    q: Float[Array, "b g r n dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    mask: Bool[Array, "*lead n s"],
    args: AttentionArgs,
) -> OnlineSoftmax:
    """Return `state` with a block of keys and values added in."""
    _, logits = _block_logits(q, k, mask, args)
    max_logit = jnp.maximum(state.max_logit, logits.max(-1))
    p = jnp.where(logits > MIN, jnp.exp(logits - max_logit[..., None]), 0)
    rescale = jnp.exp(state.max_logit - max_logit)
    values = jnp.einsum(
        "bgrns,bsgv->bgrnv", p.astype(v.dtype), v, preferred_element_type=jnp.float32
    )
    return OnlineSoftmax(
        max_logit,
        state.exp_sum * rescale + p.sum(-1),
        state.out * rescale[..., None] + values,
    )


def _block_grads(
    q: Float[Array, "b g r n dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    do: Float[Array, "b g r n dv"],
    lse: Float[Array, "b g r n"],
    delta: Float[Array, "b g r n"],
    mask: Bool[Array, "*lead n s"],
    args: AttentionArgs,
) -> tuple[
    Float[Array, "b g r n dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
]:
    """Return a block's share of the gradients of q, k and v: flash attention's
    backward pass, the probabilities recomputed from `lse`."""
    raw, logits = _block_logits(q, k, mask, args)
    p = jnp.exp(logits - lse[..., None])  # 0 where masked
    dv = jnp.einsum("bgrns,bgrnv->bsgv", p, do)
    dp = jnp.einsum("bgrnv,bsgv->bgrns", do, v, preferred_element_type=jnp.float32)
    ds = p * (dp - delta[..., None])
    if args.softcap is not None:
        ds = ds * (1 - jnp.tanh(raw / args.softcap) ** 2)
    ds = ds * args.scale
    dq = jnp.einsum("bgrns,bsgk->bgrnk", ds, k, preferred_element_type=jnp.float32)
    dk = jnp.einsum("bgrns,bgrnk->bsgk", ds, q, preferred_element_type=jnp.float32)
    return dq, dk, dv


def _sink_grads(
    sinks: Float[Array, " h"],
    lse: Float[Array, "b g r n"],
    delta: Float[Array, "b g r n"],
) -> Float[Array, " h"]:
    """Return the sinks' gradient: a sink takes exp(sink - lse) of each query's
    probability, so its gradient is minus that times do . out, summed over queries."""
    share = jnp.exp(_grouped_sinks(sinks, lse.shape) - lse)
    d = jax.lax.psum(-(share * delta).sum((0, 3)), AXIS)
    return d.reshape(-1).astype(sinks.dtype)


# --- ulysses ----------------------------------------------------------------------


@methods.attention.register("ulysses")
def ulysses(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """All-to-all over `context` from a device's positions of every head to every
    position of its own heads, run `xla`, and all-to-all back. Context must divide a
    device's heads; kv heads it does not divide are repeated.

    The shard_map is manual over every axis that splits q: an all-to-all in a
    shard_map of auto axes gathers every dimension they split."""
    if active() is None:
        return xla(q, k, v, args)
    queries, keys = _ulysses_specs(q, k)
    if keys[2] != queries[2]:  # kv heads split otherwise than their queries
        k, v = (jnp.repeat(x, q.shape[2] // k.shape[2], axis=2) for x in (k, v))
        keys = queries
    rows = P(queries[0], AXIS)
    return jax.shard_map(
        partial(_ulysses, args=args._replace(ids=None, sinks=None)),
        in_specs=(queries, keys, keys, rows, P(queries[2])),
        out_specs=queries,
        axis_names=_mesh_axes(queries),
    )(q, k, v, args.ids, args.sinks)


def _ulysses_specs(
    q: Float[Array, "b t h dk"], k: Float[Array, "b s g dk"]
) -> tuple[P, P]:
    """Return how ulysses splits q, and k and v: rows and heads as the layout says,
    and positions over `context`."""
    queries = spec(("batch", None, "heads", None), q.shape)
    keys = spec(("batch", None, "kv_heads", None), k.shape)
    return P(queries[0], AXIS, queries[2], None), P(keys[0], AXIS, keys[2], None)


def _mesh_axes(s: P) -> set[str]:
    """Return the mesh axes `s` splits over."""
    return {a for d in s if d for a in ((d,) if isinstance(d, str) else d)}


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
    return _to_positions(xla(q, k, v, args._replace(ids=ids, sinks=sinks)))


def _to_heads(x: Float[Array, "b n h e"]) -> Float[Array, "b t hc e"]:
    """Move a device's positions of every head to every position of its own heads:
    device j of c gets the j-th c-th of the heads, so a query head's kv head goes
    with it."""
    return jax.lax.all_to_all(x, AXIS, split_axis=2, concat_axis=1, tiled=True)


def _to_positions(x: Float[Array, "b t hc e"]) -> Float[Array, "b n h e"]:
    """Undo `_to_heads`."""
    return jax.lax.all_to_all(x, AXIS, split_axis=1, concat_axis=2, tiled=True)


# --- tokamax ----------------------------------------------------------------------


@methods.attention.register("tokamax")
def tokamax(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Run tokamax's flash attention by the first of its kernels that runs here
    (Mosaic, then Triton, then XLA), each document from its first token (k_start).
    Its q_sharding gives a device its queries of its heads and all their keys and
    values: with `seq` over `context`, all-gather CP in a kernel. No sinks."""
    if flash is None or masks is None:
        raise NotImplementedError(
            "methods.attention = tokamax needs tokamax, of the cuda extra (Linux)"
        )
    if args.sinks is not None:
        raise NotImplementedError("methods.attention = tokamax has no sinks")
    k, v, shardings = _flash_shardings(q, k, v)
    first = None if args.ids is None else _document_starts(args.ids)[:, None]
    causal = masks.CausalMask(window_size=args.window)
    kwargs = dict(
        mask=masks.Mask(k_start=first, causal_mask=causal),
        logits_scale=args.scale,
        logits_soft_cap=args.softcap,
        logits_dtype=jnp.float32,
        **shardings,
    )
    errors = []
    for name in (n for n in KERNELS if n in flash.IMPLEMENTATIONS):
        try:
            return flash.IMPLEMENTATIONS[name](q, k, v, **kwargs)
        except NotImplementedError as e:
            errors.append(e)
    raise ExceptionGroup("methods.attention = tokamax: no kernel runs here", errors)


@methods.attention.register("auto")
def auto(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Run `tokamax` on a GPU, which makes no t by t logits (7 times faster than `xla`
    at t = 4096 on a B300), unless `args` has sinks; else `xla`. The default."""
    if flash is not None and args.sinks is None and kernels.gpu():
        return tokamax(q, k, v, args)
    return xla(q, k, v, args)


def _flash_shardings(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
) -> tuple[
    Float[Array, "b s G dk"], Float[Array, "b s G dv"], dict[str, NamedSharding]
]:
    """Return k and v, repeated if their heads split otherwise than the queries', and
    tokamax's q_sharding and k_sharding, the keys whole along their positions."""
    on = active()
    if on is None:
        return k, v, {}
    queries = spec(("batch", "seq", "heads", None), q.shape)
    keys = spec(("batch", None, "kv_heads", None), k.shape)
    if keys[2] != queries[2]:
        k, v = (jnp.repeat(x, q.shape[2] // k.shape[2], axis=2) for x in (k, v))
        keys = P(queries[0], None, queries[2], None)
    specs = {"q_sharding": queries, "k_sharding": keys}
    return k, v, {name: NamedSharding(on.mesh, s) for name, s in specs.items()}


def _document_starts(ids: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """Return the index of the first token of each token's document: the first key it
    attends to."""
    index = jnp.arange(ids.shape[1])
    new = jnp.concatenate(
        [jnp.ones_like(ids[:, :1], bool), ids[:, 1:] != ids[:, :-1]], 1
    )
    return jax.lax.cummax(jnp.where(new, index, 0), axis=1)
