"""`ring` of `methods.attention`: context parallelism by blocks of keys, a flash
kernel's call each (bux.parallel.flash).

A device keeps its n queries and gathers every device's block of n keys and values
once (on a B300 node an all-gather moves ~6x the bytes a second of a ring's
ppermutes). It attends to each block by a kernel's call, merging the blocks by their
logsumexps, and skips a block no query sees: one ahead of them all, or before the
first key of each (its document's first token, or its window's). The backward pass
runs each block's kernel backward with the final output and logsumexp, and sums each
block's gradients onto its device.

In the positions' order a row of one document balances badly: device c - 1 attends
to c blocks, device 0 to one. So without documents or a window the queries go zigzag:
device i holds the i-th and the (2c-1-i)-th of 2c chunks, and attends to its own
block and to half of every other's.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Shaped

from bux.parallel import flash, methods
from bux.parallel.context import (
    AXIS,
    AttentionArgs,
    first_keys,
    on_devices,
    whole_rows,
)
from bux.parallel.flash import F32, Keys
from bux.parallel.mesh import active


class Static(NamedTuple):
    """What a device's part of the ring runs by: the kernel, the scale and soft cap
    of its logits, and whether its queries go zigzag."""

    kernel: flash.Kernel
    scale: float
    cap: float | None
    zigzag: bool


@methods.attention.register("ring")
def ring(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: AttentionArgs,
) -> Float[Array, "b t h dv"]:
    """Keep each device's queries, gather the keys and values, and attend to each
    device's block of them by tokamax's kernel (jnp's on a CPU), merged by their
    logsumexps; skip a block no query sees, and balance a row of one document by
    zigzag queries. Sinks fold into the final logsumexp."""
    on = active()
    if on is None:
        return whole_rows(q, k, v, args)
    kernel = flash.TOKAMAX if flash.tokamax_runs() else flash.JNP
    start = first_keys(args, *q.shape[:2])
    zigzag = start is None and q.shape[1] // on.size(AXIS) % 2 == 0
    static = Static(kernel, args.scale, args.softcap, zigzag)
    return on_devices(partial(_device, static=static), q, k, v, start, args.sinks, True)


def _device(
    q: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    start: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    static: Static,
) -> Float[Array, "b n h dv"]:
    """Run a device's part of `ring`: its n queries against every block of n keys."""
    if not static.zigzag:
        return _ring(q, k, v, start, sinks, static)
    return from_zigzag(_ring(to_zigzag(q), k, v, None, sinks, static))


class Piece(NamedTuple):
    """The queries and the keys of a block that a kernel's call attends to."""

    queries: slice
    keys: slice


class Plan(NamedTuple):
    """A device's place in the ring: which device it is of c, of n positions each."""

    me: Int[Array, ""]
    c: int
    n: int
    zigzag: bool

    @classmethod
    def here(cls, n: int, zigzag: bool) -> "Plan":
        return cls(jax.lax.axis_index(AXIS), jax.lax.axis_size(AXIS), n, zigzag)

    def pieces(self) -> tuple[Piece, ...]:
        """Return what of another device's block each branch but the first, which
        skips it, attends to. Zigzag: all queries to its first chunk (a block before
        the device's), or the second chunk of queries to all of it (after)."""
        whole = slice(None)
        if not self.zigzag:
            return (Piece(whole, whole),)
        half = self.n // 2
        return Piece(whole, slice(None, half)), Piece(slice(half, None), whole)

    def branch(
        self, j: Int[Array, ""], start: Int[Array, "b n"] | None
    ) -> Int[Array, ""]:
        """Return the branch of block j: 0 skips it, as wholly ahead of the queries or
        before the first key of each."""
        if self.zigzag:
            return jnp.where(j < self.me, 1, 2)
        seen = j < self.me
        if start is not None:
            seen &= start.min() < (j + 1) * self.n
        return seen.astype(jnp.int32)

    def keys(
        self, j: Int[Array, ""], start: Int[Array, "b m"] | None, causal: bool = False
    ) -> Keys:
        """Return which keys of block j each query sees, from its first key on."""
        if start is None:
            return Keys(causal)
        return Keys(causal, jnp.clip(start - j * self.n, 0, self.n))

    def blocks(self, x: Float[Array, "b n g e"]) -> Float[Array, "c b n g e"]:
        """Gather every device's block of x, as the queries are laid out."""
        blocks = jax.lax.all_gather(x, AXIS)
        return _reorder(blocks, _zigzag_order(self.c)) if self.zigzag else blocks

    def scatter(self, dx: Float[Array, "c b n g e"]) -> Float[Array, "b n g e"]:
        """Sum each block's gradients onto its device."""
        if self.zigzag:
            dx = _reorder(dx, np.argsort(_zigzag_order(self.c)))
        return jax.lax.psum_scatter(dx, AXIS, scatter_dimension=0)


def _rows(start: Int[Array, "b n"] | None, piece: Piece) -> Int[Array, "b m"] | None:
    return None if start is None else start[:, piece.queries]


State = tuple[Float[Array, "b n h dv"], Float[Array, "b h n"]]  # out and lse so far
Grads = tuple[
    Float[Array, "b n h dk"], Float[Array, "b n g dk"], Float[Array, "b n g dv"]
]  # dq so far, and a block's dk and dv


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _ring(
    q: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    start: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    static: Static,
) -> Float[Array, "b n h dv"]:
    return _ring_fwd(q, k, v, start, sinks, static)[0]


def _ring_fwd(
    q: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    start: Int[Array, "b n"] | None,
    sinks: Float[Array, " h"] | None,
    static: Static,
) -> tuple[Float[Array, "b n h dv"], tuple]:
    plan = Plan.here(q.shape[1], static.zigzag)
    ks, vs, me = plan.blocks(k), plan.blocks(v), plan.me
    attend = partial(static.kernel.forward, scale=static.scale, cap=static.cap)
    out, lse = attend(q, ks[me], vs[me], plan.keys(me, start, causal=True))
    pieces = [partial(_attend, p, plan, attend, q, start) for p in plan.pieces()]
    branches = [_skip, *pieces]

    def step(state: State, i: Int[Array, ""]) -> tuple[State, None]:
        j = (me - i) % plan.c
        which = plan.branch(j, start)
        return jax.lax.switch(which, branches, *state, ks[j], vs[j], j), None

    state = (out.astype(F32), lse)
    out, lse = jax.lax.scan(step, state, jnp.arange(1, plan.c))[0]
    out, lse = flash.with_sinks(out, lse, sinks)
    out = out.astype(q.dtype)
    return out, (q, k, v, start, sinks, out, lse)


def _skip(
    out: Float[Array, "b n h dv"], lse: Float[Array, "b h n"], *_: Shaped[Array, "..."]
) -> State:
    return out, lse


def _attend(
    piece: Piece,
    plan: Plan,
    attend: Callable[..., flash.Output],
    q: Float[Array, "b n h dk"],
    start: Int[Array, "b n"] | None,
    out: Float[Array, "b n h dv"],
    lse: Float[Array, "b h n"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    j: Int[Array, ""],
) -> State:
    """Return the output and logsumexp so far with a piece of block j merged in."""
    qs, keys = piece.queries, plan.keys(j, _rows(start, piece))
    o, l = attend(q[:, qs], k[:, piece.keys], v[:, piece.keys], keys)
    o, l = flash.merge(out[:, qs], lse[..., qs], o, l)
    return out.at[:, qs].set(o), lse.at[..., qs].set(l)


def _ring_bwd(static: Static, saved: tuple, dout: Float[Array, "b n h dv"]) -> tuple:
    """Return the gradients of `_ring`: each block's from its kernel's backward pass
    with the final output and logsumexp, summed onto its device at the end."""
    q, k, v, start, sinks, out, lse = saved
    plan = Plan.here(q.shape[1], static.zigzag)
    ks, vs, me = plan.blocks(k), plan.blocks(v), plan.me
    grads = partial(static.kernel.backward, scale=static.scale, cap=static.cap)
    dq, dk, dv = grads(q, ks[me], vs[me], out, dout, lse, plan.keys(me, start, True))
    seen = (q, out, dout, lse, start)
    pieces = [partial(_differentiate, p, plan, grads, *seen) for p in plan.pieces()]
    branches = [_no_grads, *pieces]

    def step(carry: tuple, i: Int[Array, ""]) -> tuple[tuple, None]:
        dq, dks, dvs = carry
        j = (me - i) % plan.c
        which = plan.branch(j, start)
        dq, dk, dv = jax.lax.switch(which, branches, dq, ks[j], vs[j], j)
        return (dq, dks.at[j].add(dk), dvs.at[j].add(dv)), None

    dks = jnp.zeros(ks.shape, F32).at[me].set(dk.astype(F32))
    dvs = jnp.zeros(vs.shape, F32).at[me].set(dv.astype(F32))
    carry = (dq.astype(F32), dks, dvs)
    dq, dks, dvs = jax.lax.scan(step, carry, jnp.arange(1, plan.c))[0]
    dk, dv = plan.scatter(dks), plan.scatter(dvs)
    dsinks = flash.sink_grads(sinks, lse, out, dout)
    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), None, dsinks


def _no_grads(
    dq: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    j: Int[Array, ""],
) -> Grads:
    return dq, jnp.zeros(k.shape, F32), jnp.zeros(v.shape, F32)


def _differentiate(
    piece: Piece,
    plan: Plan,
    grads: Callable[..., flash.Grads],
    q: Float[Array, "b n h dk"],
    out: Float[Array, "b n h dv"],
    dout: Float[Array, "b n h dv"],
    lse: Float[Array, "b h n"],
    start: Int[Array, "b n"] | None,
    dq: Float[Array, "b n h dk"],
    k: Float[Array, "b n g dk"],
    v: Float[Array, "b n g dv"],
    j: Int[Array, ""],
) -> Grads:
    """Return dq so far with a piece of block j's share added, and its dk and dv."""
    qs, ks, keys = piece.queries, piece.keys, plan.keys(j, _rows(start, piece))
    rows = (q[:, qs], k[:, ks], v[:, ks], out[:, qs], dout[:, qs], lse[..., qs])
    ddq, dk, dv = grads(*rows, keys)
    dk, dv = (jnp.zeros(x.shape, F32).at[:, ks].set(d) for x, d in ((k, dk), (v, dv)))
    return dq.at[:, qs].add(ddq), dk, dv


_ring.defvjp(_ring_fwd, _ring_bwd)


# --- zigzag -------------------------------------------------------------------------


def _zigzag_order(c: int) -> Int[np.ndarray, " c2"]:
    """Return the chunks of the zigzag blocks in order: block i holds chunks i and
    2c-1-i of 2c."""
    return np.asarray([x for i in range(c) for x in (i, 2 * c - 1 - i)])


def _reorder(
    blocks: Float[Array, "c b n g e"], order: Int[np.ndarray, " c2"]
) -> Float[Array, "c b n g e"]:
    """Return blocks of the chunks of `blocks`, two a block, taken in `order`."""
    c, b, n = blocks.shape[:3]
    rest = blocks.shape[3:]
    chunks = blocks.reshape(c, b, 2, n // 2, *rest).swapaxes(1, 2)
    chunks = chunks.reshape(2 * c, b, n // 2, *rest)[order]
    return chunks.reshape(c, 2, b, n // 2, *rest).swapaxes(1, 2).reshape(blocks.shape)


def _dest(x: int, c: int) -> int:
    """Return the device that holds chunk x of 2c, zigzag."""
    return x if x < c else 2 * c - 1 - x


def to_zigzag(x: Float[Array, "b n *r"]) -> Float[Array, "b n *r"]:
    """Move a device's chunks 2i and 2i+1 of 2c (contiguous) to the devices that hold
    them zigzag: device i chunks i and 2c-1-i, in that order."""
    me, c = jax.lax.axis_index(AXIS), jax.lax.axis_size(AXIS)
    half = x.shape[1] // 2
    first, second = (
        jax.lax.ppermute(part, AXIS, [(a, _dest(2 * a + i, c)) for a in range(c)])
        for i, part in enumerate((x[:, :half], x[:, half:]))
    )
    even = me % 2 == 0
    parts = [jnp.where(even, first, second), jnp.where(even, second, first)]
    return jnp.concatenate(parts, 1)


def from_zigzag(y: Float[Array, "b n *r"]) -> Float[Array, "b n *r"]:
    """Undo `to_zigzag`."""
    me, c = jax.lax.axis_index(AXIS), jax.lax.axis_size(AXIS)
    half = y.shape[1] // 2
    even = me % 2 == 0
    first, second = y[:, :half], y[:, half:]
    parts = (jnp.where(even, first, second), jnp.where(even, second, first))
    back = [
        jax.lax.ppermute(part, AXIS, [(_dest(2 * a + i, c), a) for a in range(c)])
        for i, part in enumerate(parts)
    ]
    return jnp.concatenate(back, 1)
