"""Context parallelism for attention: what the methods of `methods.attention` take,
and `ring`, `ulysses` and `tokamax`.

A method is ``(q, k, v, visibility) -> out``: queries (b, t, h, k), keys and values
(b, t, g, k) and (b, t, g, v) of g kv heads at the queries' positions, and every
query's average of the values of the keys it sees, by the softmax of their logits
taken in float32, (b, t, h, v). A query sees no key ahead of it; the `Visibility`
says the rest. `xla`, the default, is in bux.architecture.attention, with what only
it has: masks of kept keys, relative bias, length scale and attention dropout. A
method that lacks a feature of the visibility raises NotImplementedError naming it.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from bux.parallel import methods
from bux.parallel.layout import spec
from bux.parallel.mesh import active

try:  # tokamax's kernels, where its extra is installed (Linux)
    from tokamax._src.ops.attention import api as flash
    from tokamax._src.ops.attention import base as masks
except ImportError:
    flash = masks = None

AXIS = "context"
POSITIONS = P(None, AXIS)  # a (b, t, ...) array, its positions over context
MIN = float(jnp.finfo(jnp.float32).min)  # the logit of a key not seen
KERNELS = ("mosaic_gpu", "mosaic_tpu", "triton", "xla")  # tokamax's, fastest first


class Visibility(NamedTuple):
    """Which keys each query sees, and how it weighs them."""

    ids: Int[Array, "b t"] | None  # each token's document (Segments.ids), or none
    window: int | None  # a query sees no key `window` or more behind it
    scale: float  # on every logit
    softcap: float | None  # logits to softcap * tanh(logits / softcap)
    sinks: Float[Array, " h"] | None  # a logit a head, beside the keys', on no value


def sees(
    window: int | None,
    queries: Int[Array, " t"],
    keys: Int[Array, " s"],
    ids: tuple[Int[Array, "b t"], Int[Array, "b s"]] | None = None,
) -> Bool[Array, "*b t s"]:
    """Which keys, at the indices `keys` of their rows, each query at `queries` sees:
    none ahead of it, none `window` or more behind, and none of another document, by
    the `ids` of the queries' and the keys' tokens."""
    offset = queries[:, None] - keys[None, :]
    visible = offset >= 0
    if window is not None:
        visible &= offset < window
    if ids is None:
        return visible
    return visible & (ids[0][:, :, None] == ids[1][:, None, :])


def _xla(
    q: Float[Array, "b t h k"],
    k: Float[Array, "b t g k"],
    v: Float[Array, "b t g v"],
    seen: Visibility,
) -> Float[Array, "b t h v"]:
    return methods.attention.find("xla")(q, k, v, seen)


def _softcap(x: Float[Array, "*s"], cap: float | None) -> Float[Array, "*s"]:
    return x if cap is None else cap * jnp.tanh(x / cap)


# --- ring -------------------------------------------------------------------------


@methods.attention.register("ring")
def ring(
    q: Float[Array, "b t h k"],
    k: Float[Array, "b t g k"],
    v: Float[Array, "b t g v"],
    seen: Visibility,
) -> Float[Array, "b t h v"]:
    """Each device of `context` keeps its own queries, while the keys and values go
    round the devices a block at a time by ppermute, each block added to an online
    softmax; a block wholly ahead of the queries, or a window or more behind them,
    is skipped. The backward pass sends them round again, their gradients with them,
    so that no device holds more than its own block and the one passing through."""
    if active() is None:
        return _xla(q, k, v, seen)
    body = partial(_ring, seen=seen._replace(ids=None, sinks=None), split=k.shape[-1])
    return jax.shard_map(
        body,
        in_specs=(POSITIONS, POSITIONS, POSITIONS, P()),
        out_specs=POSITIONS,
        axis_names={AXIS},
    )(q, jnp.concatenate([k, v], -1), seen.ids, seen.sinks)


class Device(NamedTuple):
    """A device's place in the ring, and the documents of its queries and of all."""

    me: Int[Array, ""]
    c: int
    n: int  # positions a block
    ids: Int[Array, "b n"] | None
    every: Int[Array, "b t"] | None

    @classmethod
    def of(cls, ids: Int[Array, "b n"] | None, n: int) -> "Device":
        me, c = jax.lax.axis_index(AXIS), jax.lax.axis_size(AXIS)
        if ids is None:
            return cls(me, c, n, None, None)
        return cls(me, c, n, ids, jax.lax.all_gather(ids, AXIS, axis=1, tiled=True))

    def sees(self, j: Int[Array, ""], window: int | None) -> Bool[Array, "*b n s"]:
        """Which keys of block `j` each of the device's queries sees."""
        n, own = self.n, None
        if self.ids is not None and self.every is not None:
            own = (self.ids, jax.lax.dynamic_slice_in_dim(self.every, j * n, n, axis=1))
        return sees(window, self.me * n + jnp.arange(n), j * n + jnp.arange(n), own)

    def skips(self, j: Int[Array, ""], window: int | None) -> Bool[Array, ""]:
        """Whether no query of the device sees a key of block `j`: all are ahead of
        it, or a window or more behind."""
        ahead = j > self.me
        if window is None:
            return ahead
        return ahead | ((self.me - j - 1) * self.n + 1 >= window)

    def send[T](self, x: T) -> T:
        """`x` to the next device of the ring, and the last device's to the first."""
        c = self.c
        return jax.lax.ppermute(x, AXIS, [(a, (a + 1) % c) for a in range(c)])


def _round[C, B](
    visit: Callable[[C, B, Int[Array, ""]], tuple[C, B]],
    carry: C,
    block: B,
    device: Device,
    window: int | None,
) -> tuple[C, B]:
    """`visit` each block as it comes round, the device's own first, but those of
    which it sees nothing; `block` (the keys and values, and what goes with them)
    is sent on between visits."""

    def step(state: tuple[C, B], i: Int[Array, ""]) -> tuple[tuple[C, B], None]:
        carry, block = state
        block = device.send(block)
        j = (device.me - i) % device.c  # whose block it is
        skip = device.skips(j, window)
        go = jax.lax.cond(skip, lambda *s: s, lambda c, b: visit(c, b, j), carry, block)
        return go, None

    first = visit(carry, block, device.me)
    return jax.lax.scan(step, first, jnp.arange(1, device.c))[0]


class Softmax(NamedTuple):
    """An online softmax, of each query of each head (b, g, r, n): the largest logit
    so far, the sum of exp(logit - largest), and the values so weighted."""

    top: Float[Array, "b g r n"]
    total: Float[Array, "b g r n"]
    out: Float[Array, "b g r n v"]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _ring(
    q: Float[Array, "b n h k"],
    kv: Float[Array, "b n g e"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    seen: Visibility,
    split: int,
) -> Float[Array, "b n h v"]:
    """A device's part of `ring`: its n queries against each block of n keys and
    values (`kv`, split at `split`) as it comes round."""
    return _ring_forward(q, kv, ids, sinks, seen, split)[0]


def _ring_forward(
    q: Float[Array, "b n h k"],
    kv: Float[Array, "b n g e"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    seen: Visibility,
    split: int,
) -> tuple[Float[Array, "b n h v"], Float[Array, "b g r n"]]:
    """`_ring`'s output, and each query's log of the sum of exp(logit)."""
    b, n, h, _ = q.shape
    g, device = kv.shape[2], Device.of(ids, n)
    grouped = _grouped(q, g)

    def visit(
        state: Softmax, kv: Float[Array, "b n g e"], j: Int[Array, ""]
    ) -> tuple[Softmax, Float[Array, "b n g e"]]:
        k, v, visible = kv[..., :split], kv[..., split:], device.sees(j, seen.window)
        return _add(state, grouped, k, v, visible, seen), kv

    start = _start(sinks, (b, g, h // g, n), kv.shape[-1] - split)
    state, _ = _round(visit, start, kv, device, seen.window)
    out = state.out / state.total[..., None]
    return _ungrouped(out).astype(kv.dtype), state.top + jnp.log(state.total)


def _ring_fwd(
    q: Float[Array, "b n h k"],
    kv: Float[Array, "b n g e"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    seen: Visibility,
    split: int,
) -> tuple[Float[Array, "b n h v"], tuple]:
    out, lse = _ring_forward(q, kv, ids, sinks, seen, split)
    return out, (q, kv, ids, sinks, out, lse)


def _ring_bwd(
    seen: Visibility, split: int, saved: tuple, dout: Float[Array, "b n h v"]
) -> tuple:
    """The gradients of `_ring`: each block's keys and values go round again, their
    gradients with them, and those go home at the end."""
    q, kv, ids, sinks, out, lse = saved
    g, device = kv.shape[2], Device.of(ids, q.shape[1])
    grouped, do = _grouped(q, g), _grouped(dout.astype(jnp.float32), g)
    delta = (do * _grouped(out.astype(jnp.float32), g)).sum(-1)

    def visit(
        dq: Float[Array, "b g r n k"],
        block: tuple[Float[Array, "b n g e"], Float[Array, "b n g e"]],
        j: Int[Array, ""],
    ) -> tuple[
        Float[Array, "b g r n k"],
        tuple[Float[Array, "b n g e"], Float[Array, "b n g e"]],
    ]:
        kv, dkv = block
        k, v, visible = kv[..., :split], kv[..., split:], device.sees(j, seen.window)
        ddq, dk, dv = _block_grads(grouped, k, v, do, lse, delta, visible, seen)
        return dq + ddq, (kv, dkv + jnp.concatenate([dk, dv], -1))

    start = jnp.zeros_like(grouped, jnp.float32), (kv, jnp.zeros_like(kv, jnp.float32))
    dq, (_, dkv) = _round(visit, *start, device, seen.window)
    dsinks = None if sinks is None else _sink_grads(sinks, lse, delta)
    dq = _ungrouped(dq).astype(q.dtype)
    return dq, device.send(dkv).astype(kv.dtype), None, dsinks


_ring.defvjp(_ring_fwd, _ring_bwd)


def _grouped(x: Float[Array, "b n h e"], g: int) -> Float[Array, "b g r n e"]:
    """Queries, or what is of them, by the g groups of heads that share a kv head."""
    b, n, h, e = x.shape
    return x.reshape(b, n, g, h // g, e).transpose(0, 2, 3, 1, 4)


def _ungrouped(x: Float[Array, "b g r n e"]) -> Float[Array, "b n h e"]:
    b, g, r, n, e = x.shape
    return x.transpose(0, 3, 1, 2, 4).reshape(b, n, g * r, e)


def _start(
    sinks: Float[Array, " h"] | None, shape: tuple[int, int, int, int], dv: int
) -> Softmax:
    """An online softmax of queries of `shape` (b, g, r, n) that has seen no key:
    only the sinks, if any."""
    if sinks is None:
        top, total = jnp.full(shape, MIN), jnp.zeros(shape)
    else:
        top, total = jnp.broadcast_to(_sinks(sinks, shape), shape), jnp.ones(shape)
    return Softmax(top, total, jnp.zeros((*shape, dv)))


def _sinks(
    sinks: Float[Array, " h"], shape: tuple[int, ...]
) -> Float[Array, "1 g r 1"]:
    return sinks.reshape(shape[1:3]).astype(jnp.float32)[None, :, :, None]


def _logits(
    q: Float[Array, "b g r n k"],
    k: Float[Array, "b s g k"],
    visible: Bool[Array, "*rows n s"],
    seen: Visibility,
) -> tuple[Float[Array, "b g r n s"], Float[Array, "b g r n s"]]:
    """A block's logits in float32, before the soft cap and after it, at MIN for
    the keys not seen."""
    raw = jnp.einsum("bgrqd,bsgd->bgrqs", q, k, preferred_element_type=jnp.float32)
    raw = raw * seen.scale
    visible = visible[:, None, None] if visible.ndim == 3 else visible
    return raw, jnp.where(visible, _softcap(raw, seen.softcap), MIN)


def _add(
    state: Softmax,
    q: Float[Array, "b g r n k"],
    k: Float[Array, "b s g k"],
    v: Float[Array, "b s g v"],
    visible: Bool[Array, "*rows n s"],
    seen: Visibility,
) -> Softmax:
    """`state`, with a block of keys and values added in."""
    _, logits = _logits(q, k, visible, seen)
    top = jnp.maximum(state.top, logits.max(-1))
    p = jnp.where(logits > MIN, jnp.exp(logits - top[..., None]), 0)
    fade = jnp.exp(state.top - top)
    values = jnp.einsum(
        "bgrqs,bsgd->bgrqd", p.astype(v.dtype), v, preferred_element_type=jnp.float32
    )
    return Softmax(
        top, state.total * fade + p.sum(-1), state.out * fade[..., None] + values
    )


def _block_grads(
    q: Float[Array, "b g r n k"],
    k: Float[Array, "b s g k"],
    v: Float[Array, "b s g v"],
    do: Float[Array, "b g r n v"],
    lse: Float[Array, "b g r n"],
    delta: Float[Array, "b g r n"],
    visible: Bool[Array, "*rows n s"],
    seen: Visibility,
) -> tuple[Float[Array, "b g r n k"], Float[Array, "b s g k"], Float[Array, "b s g v"]]:
    """What a block adds to the gradients of the queries, and its keys' and values':
    flash attention's backward pass, the probabilities recomputed from `lse`."""
    raw, logits = _logits(q, k, visible, seen)
    p = jnp.exp(logits - lse[..., None])  # 0 where not seen
    dv = jnp.einsum("bgrqs,bgrqd->bsgd", p, do)
    dp = jnp.einsum("bgrqd,bsgd->bgrqs", do, v, preferred_element_type=jnp.float32)
    ds = p * (dp - delta[..., None])
    if seen.softcap is not None:
        ds = ds * (1 - jnp.tanh(raw / seen.softcap) ** 2)
    ds = ds * seen.scale
    dq = jnp.einsum("bgrqs,bsgd->bgrqd", ds, k, preferred_element_type=jnp.float32)
    dk = jnp.einsum("bgrqs,bgrqd->bsgd", ds, q, preferred_element_type=jnp.float32)
    return dq, dk, dv


def _sink_grads(
    sinks: Float[Array, " h"],
    lse: Float[Array, "b g r n"],
    delta: Float[Array, "b g r n"],
) -> Float[Array, " h"]:
    """A sink takes exp(sink - lse) of each query's probability from its keys, so
    its gradient is minus that times the query's do . out, summed over every query."""
    share = jnp.exp(_sinks(sinks, lse.shape) - lse)
    d = jax.lax.psum(-(share * delta).sum((0, 3)), AXIS)
    return d.reshape(-1).astype(sinks.dtype)


# --- ulysses ----------------------------------------------------------------------


@methods.attention.register("ulysses")
def ulysses(
    q: Float[Array, "b t h k"],
    k: Float[Array, "b t g k"],
    v: Float[Array, "b t g v"],
    seen: Visibility,
) -> Float[Array, "b t h v"]:
    """All-to-alls over `context`, from a device's own positions of every head to
    every position of its own heads; `xla` on those; and back. Context must divide a
    device's heads; kv heads it does not divide are repeated for their queries.

    The shard_map is manual over every axis that splits q, not over `context` alone:
    an all-to-all in a shard_map of auto axes gathers every dimension they split."""
    if active() is None:
        return _xla(q, k, v, seen)
    queries, keys = _manual(q, k)
    if keys[2] != queries[2]:  # kv heads split otherwise than their queries
        k, v = (jnp.repeat(x, q.shape[2] // k.shape[2], axis=2) for x in (k, v))
        keys = queries
    rows = P(queries[0], AXIS)
    return jax.shard_map(
        partial(_ulysses, seen=seen._replace(ids=None, sinks=None)),
        in_specs=(queries, keys, keys, rows, P(queries[2])),
        out_specs=queries,
        axis_names=_axes(queries),
    )(q, k, v, seen.ids, seen.sinks)


def _manual(q: Float[Array, "b t h k"], k: Float[Array, "b t g k"]) -> tuple[P, P]:
    """How ulysses splits q, and k and v: rows and heads as the layout says, and
    positions over `context`."""
    queries = spec(("batch", None, "heads", None), q.shape)
    keys = spec(("batch", None, "kv_heads", None), k.shape)
    return P(queries[0], AXIS, queries[2], None), P(keys[0], AXIS, keys[2], None)


def _axes(s: P) -> set[str]:
    """The mesh axes `s` splits over."""
    return {a for d in s if d for a in ((d,) if isinstance(d, str) else d)}


def _ulysses(
    q: Float[Array, "b n h k"],
    k: Float[Array, "b n g k"],
    v: Float[Array, "b n g v"],
    ids: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    seen: Visibility,
) -> Float[Array, "b n h v"]:
    """A device's part of `ulysses`: its n positions of its h heads in, and out."""
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
    return _to_positions(_xla(q, k, v, seen._replace(ids=ids, sinks=sinks)))


def _to_heads(x: Float[Array, "b n h e"]) -> Float[Array, "b t hc e"]:
    """A device's own positions of every head to every position of its own heads:
    the j-th of `context` devices gets the j-th c-th of the heads, so that a query
    head's kv head goes where it does."""
    return jax.lax.all_to_all(x, AXIS, split_axis=2, concat_axis=1, tiled=True)


def _to_positions(x: Float[Array, "b t hc e"]) -> Float[Array, "b n h e"]:
    """`_to_heads`, undone."""
    return jax.lax.all_to_all(x, AXIS, split_axis=1, concat_axis=2, tiled=True)


# --- tokamax ----------------------------------------------------------------------


@methods.attention.register("tokamax")
def tokamax(
    q: Float[Array, "b t h k"],
    k: Float[Array, "b t g k"],
    v: Float[Array, "b t g v"],
    seen: Visibility,
) -> Float[Array, "b t h v"]:
    """tokamax's flash attention, by the first of its kernels that runs here (Mosaic,
    then Triton, then XLA), each document from its first token on (tokamax's
    k_start). Its q_sharding gives each device its own queries of its own heads and
    every key and value of them: with `seq` over `context`, all-gather CP in a
    kernel. No sinks."""
    if flash is None or masks is None:
        raise NotImplementedError(
            "methods.attention = tokamax needs tokamax, of the cuda extra (Linux)"
        )
    if seen.sinks is not None:
        raise NotImplementedError("methods.attention = tokamax has no sinks")
    k, v, shardings = _flash_shardings(q, k, v)
    first = None if seen.ids is None else _starts(seen.ids)[:, None]
    causal = masks.CausalMask(window_size=seen.window)
    kwargs = dict(
        mask=masks.Mask(k_start=first, causal_mask=causal),
        logits_scale=seen.scale,
        logits_soft_cap=seen.softcap,
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


def _flash_shardings(
    q: Float[Array, "b t h k"], k: Float[Array, "b t g k"], v: Float[Array, "b t g v"]
) -> tuple[Float[Array, "b t G k"], Float[Array, "b t G v"], dict[str, NamedSharding]]:
    """k and v, repeated for their queries if their heads split otherwise; and
    tokamax's q_sharding and k_sharding, keys whole along their positions."""
    on = active()
    if on is None:
        return k, v, {}
    queries = spec(("batch", "seq", "heads", None), q.shape)
    keys = spec(("batch", None, "kv_heads", None), k.shape)
    if keys[2] != queries[2]:
        k, v = (jnp.repeat(x, q.shape[2] // k.shape[2], axis=2) for x in (k, v))
        keys = P(queries[0], None, queries[2], None)
    split = {"q_sharding": queries, "k_sharding": keys}
    return k, v, {name: NamedSharding(on.mesh, s) for name, s in split.items()}


def _starts(ids: Int[Array, "b t"]) -> Int[Array, "b t"]:
    """The index of the first token of each token's document: the first key it
    sees."""
    index = jnp.arange(ids.shape[1])
    new = jnp.concatenate(
        [jnp.ones_like(ids[:, :1], bool), ids[:, 1:] != ids[:, :-1]], 1
    )
    return jax.lax.cummax(jnp.where(new, index, 0), axis=1)
