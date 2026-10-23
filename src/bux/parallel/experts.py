"""Expert parallelism: the methods of ``methods.moe`` that split the experts over the
axes `layout.experts` gives them (`expert`), each device running its own experts on
the picks of every device in its group.

    all_gather         gather the group's tokens, run this device's experts on them,
                       and reduce-scatter each token's sum back
    all_to_all         send each pick to its expert's device and back, in a buffer per
                       peer with room for every pick: dropless
    ragged_all_to_all  send each pick to its expert's device and back, a ragged slice
                       per expert of each peer, so that the rows arrive sorted by
                       expert. XLA:CPU lacks it
    ragged_tokens      send each token once to each device its picks are on, with
                       those picks, and back as their weighted sum

Each is a body that runs on one device's tokens, which ``expert_parallel`` runs in a
shard_map manual over the axes that split the tokens, so that each device sorts its
own. Routing ran before, in the global view. `methods.capacity` bounds the rows a
device takes (``_room``), and `precision.dispatch` has the ragged methods send the
tokens in MXFP8.
"""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, Shaped, UInt8

from bux import config as cfg
from bux.architecture.experts.grouped_blockscaled import dequantized
from bux.architecture.experts.moe import (
    Kernels,
    MoE,
    RowNorm,
    axes_of,
    experts,
    in_range,
    on_each_device,
    permuted,
    routed,
    run_sorted,
    sort_by_expert,
    token_rows,
    token_sums,
)
from bux.architecture.precision import blockscaled, matmuls
from bux.architecture.precision.blockscaled import Blocks
from bux.architecture.registry import Taps, report
from bux.architecture.repeats import remat
from bux.parallel import methods
from bux.parallel.layout import spec
from bux.parallel.mesh import active

Axes = tuple[str, ...]
# What a body returns: each token's sum of its experts' outputs, and the share of the
# group's picks that a capacity dropped.
Combined = tuple[Float[Array, "b t d"], Float[Array, ""]]
# A body: (z, idx, weights, kernels, *, axes, ffn, norm) -> Combined, of one device's
# tokens and its own experts' kernels, the experts split over `axes`.
Body = Callable[..., Combined]
# jax.lax.ragged_all_to_all's signature: another exchange is a function.
RaggedAllToAll = Callable[..., Shaped[Array, "m d"]]
# What the ragged exchanges send and bring back, which `dots` keeps (``remat``): else
# its backward would exchange them again.
EXCHANGED = "experts.exchanged"
# The pieces a ragged exchange sends in, at least: XLA's one-shot kernel on a node
# copies each with one block of 128 threads, ~90 GB/s with 64 on 8 B300s.
PIECES = 2048


def expert_parallel(body: Body) -> Callable[..., Float[Array, "b t d"]]:
    """Return the method of ``methods.moe`` that runs `body` on every device, or
    `sorted` where no axis splits the experts. Under a capacity, a training forward
    reports the share of picks it dropped as `dropped`."""

    def run(
        moe: MoE,
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
        taps: Taps | None = None,
    ) -> Float[Array, "b t d"]:
        ep = axes_of(spec(("experts",), (moe.config.num_experts,)))
        if not ep:
            return run_sorted(moe, z, idx, weights, taps)
        tokens = _token_spec(z, ep)
        ffn, norm = moe.config.expert, moe.expert_norm

        def local(*args: object) -> Combined:
            y, dropped = body(*args, axes=ep, ffn=ffn, norm=norm)
            return y, jax.lax.pmean(dropped, axes_of(tokens))

        each = on_each_device(local, tokens, P(ep), (tokens, P()))
        y, dropped = each(z, idx, weights, moe.kernels())
        if _capacity() is not None:
            report(taps, "dropped", 0.0, dropped)
        return y

    return run


def _token_spec(z: Float[Array, "b t d"], ep: Axes) -> P:
    """Return the tokens' spec: rows over `batch` and positions over `seq`, and rows
    over the experts' axes `ep` too where those leave them whole."""
    batch, seq, _ = spec(("batch", "seq", None), z.shape)
    taken = axes_of(P(batch, seq))
    rows = axes_of(P(batch)) + tuple(a for a in ep if a not in taken)
    on = active()
    assert on is not None  # `ep` came from a mesh
    if z.shape[0] % on.size(*rows):
        raise ValueError(
            f"the experts split over {ep}, and so do the tokens: "
            f"{z.shape[0]} rows do not split over {rows}"
        )
    return P(rows or None, seq, None)


# --- A capacity ------------------------------------------------------------------------


def _capacity() -> float | None:
    """Return the run's `methods.capacity`."""
    on = active()
    return None if on is None else on.methods.capacity


def _room(n: int, peers: int) -> int:
    """Return how many rows a device takes from its group's `peers`, each of which has
    `n` to send: `methods.capacity` times n, or all that could come."""
    most, capacity = n * peers, _capacity()
    return most if capacity is None else min(most, math.ceil(capacity * n))


def _fit(sizes: Int[Array, " g"], room: int) -> Int[Array, " g"]:
    """Return how many rows of each group, the groups in turn, fit in the first
    `room`."""
    return jnp.clip(room - (jnp.cumsum(sizes) - sizes), 0, sizes)


def _routed_within(
    x: Float[Array, "*v d"],
    idx: Int[Array, "*v k"],
    weights: Float[Array, "*v k"],
    kernels: Kernels,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None,
    room: int,
) -> tuple[Float[Array, "m d"], Int[Array, ""]]:
    """Return each row's sum of its picks' expert outputs, weighted, as ``routed``
    does, but of the first `room` picks by expert alone, flat; and how many ran. A
    pick outside the experts adds nothing, as in ``routed``."""
    (d, k), local = (x.shape[-1], idx.shape[-1]), kernels[0].shape[0]
    order, inverse, sizes = sort_by_expert(idx, local)
    order, sizes = order[:room], _fit(sizes, room)
    rows = token_rows(x.reshape(-1, d), order, inverse, k)
    made = experts(rows, sizes, *kernels, ffn, norm)
    kept = jnp.where(in_range(idx, local), weights, 0).reshape(-1)
    return token_sums(made * kept[order][:, None], inverse, order, k), sizes.sum()


# --- all_gather ------------------------------------------------------------------------


def all_gather(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None,
) -> Combined:
    """Gather the group's tokens onto each of its devices, run this device's experts
    on them, and reduce-scatter each token's sum back to its own device. A device runs
    `methods.capacity` times the picks it has, its last experts' first dropped."""
    t, d, local = z.shape[1], z.shape[2], kernels[0].shape[0]
    peers, first = jax.lax.axis_size(axes), jax.lax.axis_index(axes) * local
    gather = functools.partial(jax.lax.all_gather, axis_name=axes, tiled=True)
    group_z, group_weights = gather(z), gather(weights)
    group_idx = gather(idx) - first  # numbered from this device's first expert
    room = _room(idx.size, peers)
    y, ran = _routed_within(group_z, group_idx, group_weights, kernels, ffn, norm, room)
    mine = in_range(group_idx, local).sum()
    summed = jax.lax.psum_scatter(y.reshape(-1, t, d), axes, tiled=True)
    return summed, (mine - ran) / idx.size


# --- all_to_all ------------------------------------------------------------------------


def all_to_all(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None,
) -> Combined:
    """Send each pick to its expert's device and back, in a buffer for each peer with
    room for every pick, so that none drops. A device runs the rows it receives as
    tokens of one pick each (``routed`` with k = 1)."""
    d, k, local = z.shape[-1], idx.shape[-1], kernels[0].shape[0]
    buffers = _buffers(idx, local, jax.lax.axis_size(axes))
    rows = _swap(z.reshape(-1, d)[buffers.pick // k], axes)
    expert = _swap(buffers.expert, axes)[..., None]
    ones = jnp.ones(expert.shape, z.dtype)
    out = _swap(routed(rows, expert, ones, *kernels, ffn, norm), axes)
    per_pick = out[buffers.peer, buffers.slot].reshape(*idx.shape, d)
    return jnp.einsum("btkd,btk->btd", per_pick, weights), jnp.zeros(())


class Buffers(NamedTuple):
    """A device's n picks laid out in a buffer of n slots for each peer, [p, n]: the
    pick in each slot (any, past the peer's picks) and its expert on that peer
    (`local`, none, past them); and each pick's peer and slot."""

    pick: Int[Array, "p n"]
    expert: Int[Array, "p n"]
    peer: Int[Array, " n"]
    slot: Int[Array, " n"]


def _buffers(idx: Int[Array, "b t k"], local: int, peers: int) -> Buffers:
    """Return the buffers of the picks `idx`, of `local` experts on each of `peers`."""
    flat, n = idx.reshape(-1), idx.size
    order, inverse, sizes = sort_by_expert(idx, local * peers)
    sent = sizes.reshape(peers, local).sum(-1)  # the picks for each peer
    start = jnp.cumsum(sent) - sent  # where each peer's start, sorted
    slot = jnp.arange(n)
    pick = order[jnp.minimum(start[:, None] + slot, n - 1)]
    expert = jnp.where(slot < sent[:, None], flat[pick] % local, local)
    peer = flat // local
    return Buffers(pick, expert, peer, inverse - start[peer])


def _swap(x: Shaped[Array, "n ..."], axes: Axes) -> Shaped[Array, "n ..."]:
    """Return what the devices of `axes` send this one, in their order: each sends
    the i-th of as many slices of its x to the i-th device (an all-to-all)."""
    return jax.lax.all_to_all(x, axes, split_axis=0, concat_axis=0, tiled=True)


# --- ragged_all_to_all -----------------------------------------------------------------


class Plan(NamedTuple):
    """A ragged all-to-all's offsets and sizes (``jax.lax.ragged_all_to_all``'s
    arguments after its buffers), a slice an entry: slice s goes to peer
    s // (slices a peer)."""

    input_offsets: Int[Array, " s"]
    send_sizes: Int[Array, " s"]
    output_offsets: Int[Array, " s"]
    recv_sizes: Int[Array, " s"]


class Arrivals(NamedTuple):
    """Where the rows land, [j, l, i]: on device j, the rows device i sends for its
    destination l (an expert of j, or j itself), where they start and how many fit."""

    starts: Int[Array, "p l p"]
    kept: Int[Array, "p l p"]


class Route(NamedTuple):
    """How a ragged method's rows go to their destinations and back: `there` sends
    this device's rows, `back` returns what it made of the rows it took, `arrivals`
    says where they land, and counts[i, e] how many device i sends destination e."""

    there: Plan
    back: Plan
    arrivals: Arrivals
    counts: Int[Array, "p e"]


def ragged_all_to_all(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None,
    exchange: RaggedAllToAll = jax.lax.ragged_all_to_all,
) -> Combined:
    """Send each pick to its expert's device and back, a ragged slice for each expert
    of each peer: a device's rows arrive sorted by its experts, which run on them as
    they are. A device takes `methods.capacity` times the rows it sends and drops the
    rest: its last experts' picks first, and of an expert's, the last peers'."""
    d, k, n, local = z.shape[-1], idx.shape[-1], idx.size, kernels[0].shape[0]
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    order, inverse, sizes = sort_by_expert(idx, local * peers)
    rows = _room(n, peers)
    route = _route(sizes, rows, device, axes)
    picks = token_rows(z.reshape(-1, d), order, inverse, k)
    arrived = _exchanged_tokens(exchange, picks, rows, route.there, axes)
    made = experts(arrived, route.arrivals.kept[device].sum(1), *kernels, ffn, norm)
    returned = _exchanged(exchange, made, n, route.back, axes)
    came_back = _came_back(route.there, idx.reshape(-1)[order])
    per_pick = permuted(jnp.where(came_back[:, None], returned, 0), inverse, order)
    y = jnp.einsum("btkd,btk->btd", per_pick.reshape(*idx.shape, d), weights)
    return y, 1 - route.arrivals.kept.sum() / route.counts.sum()


def _route(
    sizes: Int[Array, " e"], rows: int, device: Int[Array, ""], axes: Axes
) -> Route:
    """Return the route of this device's rows, sorted by destination, sizes[e] of them
    for destination e (an expert, or a device), to `rows` rows on each device."""
    local = sizes.shape[0] // jax.lax.axis_size(axes)  # the destinations a device
    counts = jax.lax.all_gather(sizes, axes)  # [i, e]: device i's rows for e
    starts = jnp.cumsum(counts, 1) - counts  # [i, e]: where they start, sorted, on i
    arrivals = _arrivals(counts, rows)
    there = _plan_there(arrivals, starts[device], device)
    theirs = jax.lax.dynamic_slice_in_dim(starts, device * local, local, axis=1)
    back = _plan_back(arrivals, theirs, device)
    return Route(there, back, arrivals, counts)


def _arrivals(counts: Int[Array, "p e"], rows: int) -> Arrivals:
    """Lay out what each device takes, from the rows counts[i, e] of each device i for
    each destination e: on device j, those for its destination 0 from each device in
    turn, then those for its destination 1, and so on, to `rows` rows; past them none
    fit."""
    peers = counts.shape[0]
    counts = counts.reshape(peers, peers, -1).transpose(1, 2, 0)  # [j, l, i]
    ends = jnp.cumsum(counts.reshape(peers, -1), axis=1).reshape(counts.shape)
    starts = ends - counts
    kept = jnp.clip(rows - starts, 0, counts)
    return Arrivals(jnp.minimum(starts, rows), kept)


def _plan_there(
    arrivals: Arrivals, starts: Int[Array, " e"], device: Int[Array, ""]
) -> Plan:
    """Return the plan that sends this device's rows to their destinations, from
    starts[e] on for destination e, as they are sorted here."""
    return Plan(
        starts,
        arrivals.kept[:, :, device].reshape(-1),
        arrivals.starts[:, :, device].reshape(-1),
        arrivals.kept[device].T.reshape(-1),
    )


def _plan_back(
    arrivals: Arrivals, theirs: Int[Array, "p l"], device: Int[Array, ""]
) -> Plan:
    """Return the plan that sends back what this device made of the rows it took, to
    theirs[i, l] on device i for its destination l, where device i sorted those."""
    return Plan(
        arrivals.starts[device].T.reshape(-1),
        arrivals.kept[device].T.reshape(-1),
        theirs.reshape(-1),
        arrivals.kept[:, :, device].reshape(-1),
    )


def _came_back(there: Plan, to: Int[Array, " n"]) -> Bool[Array, " n"]:
    """Return which of the rows, sorted by their destination `to`, came back: the
    first there.send_sizes[e] of destination e's, which start at
    there.input_offsets[e], `there` sending a slice a destination."""
    at = jnp.arange(to.size) - there.input_offsets[to]
    return at < there.send_sizes[to]


# Sending: rows of any dtype, and the tokens in MXFP8 where `precision.dispatch` says.


def _exchanged(
    exchange: RaggedAllToAll,
    x: Shaped[Array, "n e"],
    rows: int,
    plan: Plan,
    axes: Axes,
) -> Shaped[Array, "rows e"]:
    """Return what `plan` sends of x, in `rows` rows, named for ``remat``."""
    return remat.label(EXCHANGED, _sent(exchange, x, rows, plan, axes))


def _exchanged_tokens(
    exchange: RaggedAllToAll,
    x: Float[Array, "n d"],
    rows: int,
    plan: Plan,
    axes: Axes,
) -> Float[Array, "rows d"]:
    """Return `_exchanged` of the tokens x: in MXFP8 where `precision.dispatch` says
    and x's rows are whole blocks, back in x's dtype where they arrive."""
    if matmuls.in_force().dispatch == "compute" or x.shape[-1] % blockscaled.MX_BLOCK:
        return _exchanged(exchange, x, rows, plan, axes)
    sent = _sent_in_mxfp8(exchange, rows, axes, x.shape, x, plan)
    return remat.label(EXCHANGED, sent)


def _sent(
    exchange: RaggedAllToAll,
    x: Shaped[Array, "n e"],
    rows: int,
    plan: Plan,
    axes: Axes,
) -> Shaped[Array, "rows e"]:
    """Return what `plan` sends of x, in `rows` rows."""
    return _sent_into(exchange, x, _buffer(rows, x), plan, axes)


def _sent_into(
    exchange: RaggedAllToAll,
    x: Shaped[Array, "n e"],
    into: Shaped[Array, "m e"],
    plan: Plan,
    axes: Axes,
) -> Shaped[Array, "m e"]:
    """Return `into` with what `plan` sends of x in it. XLA's kernel copies each slice
    with one block of threads, so the slices go in PIECES pieces in all."""
    parts = max(1, PIECES // plan.send_sizes.size)
    return exchange(x, into, *_in_pieces(plan, parts), axis_name=axes)


def _buffer(rows: int, like: Shaped[Array, "*s d"]) -> Shaped[Array, "rows d"]:
    """Return `rows` rows, as wide as `like`, to receive into, left as they are: the
    experts make 0 of the rows past theirs, and what does not come back is masked."""
    return jax.lax.empty((rows, like.shape[-1]), like.dtype)


def _in_pieces(plan: Plan, parts: int) -> Plan:
    """Return the same exchange, each slice cut into `parts` pieces in a row."""
    at = jnp.arange(parts)

    def cut(sizes: Int[Array, " s"]) -> tuple[Int[Array, "s p"], Int[Array, "s p"]]:
        piece = (sizes[:, None] + parts - 1) // parts
        starts = jnp.minimum(at * piece, sizes[:, None])
        return starts, jnp.minimum(piece, sizes[:, None] - starts)

    starts, sizes = cut(plan.send_sizes)
    received = cut(plan.recv_sizes)[1]
    return Plan(
        (plan.input_offsets[:, None] + starts).reshape(-1),
        sizes.reshape(-1),
        (plan.output_offsets[:, None] + starts).reshape(-1),
        received.reshape(-1),
    )


def _reversed(plan: Plan, axes: Axes) -> Plan:
    """Return the plan that sends back what `plan` sends, each row where it came from:
    what a ragged all-to-all's transpose sends, as jax's."""
    return Plan(
        _swap(plan.output_offsets, axes),
        plan.recv_sizes,
        _swap(plan.input_offsets, axes),
        plan.send_sizes,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _sent_in_mxfp8(
    exchange: RaggedAllToAll,
    rows: int,
    axes: Axes,
    shape: tuple[int, int],
    x: Float[Array, "n d"],
    plan: Plan,
) -> Float[Array, "rows d"]:
    """Return `_sent` of x sent in MXFP8, E4M3 values and a power-of-two scale per 32:
    a row of d + d / 32 bytes a token, about half of bfloat16's. Its gradient is
    `_sent`'s, in the gradient's dtype."""
    packed = _packed(blockscaled.mx_quantize(x))
    arrived = _sent(exchange, packed, rows, plan, axes)
    return _unpacked(arrived, shape[-1]).astype(x.dtype)


def _sent_in_mxfp8_forward(
    exchange: RaggedAllToAll,
    rows: int,
    axes: Axes,
    shape: tuple[int, int],
    x: Float[Array, "n d"],
    plan: Plan,
) -> tuple[Float[Array, "rows d"], Plan]:
    return _sent_in_mxfp8(exchange, rows, axes, shape, x, plan), plan


def _sent_in_mxfp8_backward(
    exchange: RaggedAllToAll,
    rows: int,
    axes: Axes,
    shape: tuple[int, int],
    plan: Plan,
    g: Float[Array, "rows d"],
) -> tuple[Float[Array, "n d"], None]:
    zeros = jnp.zeros(shape, g.dtype)  # what is not sent has no gradient
    return _sent_into(exchange, g, zeros, _reversed(plan, axes), axes), None


_sent_in_mxfp8.defvjp(_sent_in_mxfp8_forward, _sent_in_mxfp8_backward)


def _packed(blocks: Blocks) -> UInt8[Array, "n e"]:
    """Return each row of MXFP8 `blocks` as bytes: its E4M3 values, then its scales."""
    values = jax.lax.bitcast_convert_type(blocks.values, jnp.uint8)
    scales = jax.lax.bitcast_convert_type(blocks.scales, jnp.uint8)
    return jnp.concatenate([values, scales], axis=-1)


def _unpacked(packed: UInt8[Array, "m e"], d: int) -> Float[Array, "m d"]:
    """Return the tokens of `packed`, E4M3 values then their scales, dequantized."""
    values = jax.lax.bitcast_convert_type(packed[:, :d], jnp.float8_e4m3fn)
    scales = jax.lax.bitcast_convert_type(packed[:, d:], jnp.float8_e8m0fnu)
    return dequantized(Blocks(values, scales, None))


# --- ragged_tokens ---------------------------------------------------------------------


def ragged_tokens(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None,
    exchange: RaggedAllToAll = jax.lax.ragged_all_to_all,
) -> Combined:
    """Send each token once to each device that holds any of its experts, with its
    picks there, and back as their weighted sum: a row per device it visits, not per
    pick, as ``ragged_all_to_all`` sends. A device runs `methods.capacity` times the
    picks it has, and drops the rest as ``ragged_all_to_all`` does; it takes as many
    tokens, or the most that could come, a token from each device."""
    d, k, n, local = z.shape[-1], idx.shape[-1], idx.size, kernels[0].shape[0]
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    room = _room(n, peers)  # the picks a device runs
    rows = min(room, peers * n // k)  # the tokens it takes: a token visits it once
    visit = _visits(idx.reshape(-1, k), local, peers)
    order, inverse, sizes = sort_by_expert(visit, peers)  # by the device visited
    route = _route(sizes, rows, device, axes)
    tokens = token_rows(z.reshape(-1, d), order, inverse, k)
    x = _exchanged_tokens(exchange, tokens, rows, route.there, axes)
    choices = _choices_there(idx.reshape(-1, k), weights.reshape(-1, k), local)
    local_idx, local_weights = (
        _exchanged(exchange, a[order], rows, route.there, axes) for a in choices
    )
    came = (jnp.arange(rows) < route.arrivals.kept[device].sum())[:, None]
    local_idx = jnp.where(came, local_idx, local)
    local_weights = jnp.where(came, local_weights, 0)
    made, ran = _routed_within(x, local_idx, local_weights, kernels, ffn, norm, room)
    returned = _exchanged(exchange, made, n, route.back, axes)
    went = visit.reshape(-1)[order]  # the device each visit, sorted, went to
    came_back = (went < peers) & _came_back(route.there, went)
    y = token_sums(jnp.where(came_back[:, None], returned, 0), inverse, order, k)
    return y.reshape(z.shape), 1 - ran / n


def _visits(idx: Int[Array, "n k"], local: int, peers: int) -> Int[Array, "n k"]:
    """Return the device of each pick that is its token's first there, the device it
    visits; and `peers`, none, for the others."""
    on = idx // local
    same = on[:, :, None] == on[:, None, :]  # [t, s, r]: picks s and r on one device
    earlier = jnp.tri(idx.shape[-1], k=-1, dtype=bool)  # r before s
    return jnp.where((same & earlier).any(-1), peers, on)


def _choices_there(
    idx: Int[Array, "n k"], weights: Float[Array, "n k"], local: int
) -> tuple[Int[Array, "m k"], Float[Array, "m k"]]:
    """Return, for each pick, its token's picks on the pick's device: their experts
    there (`local`, none, for the picks elsewhere) and their weights (0)."""
    on = idx // local
    there = on[:, :, None] == on[:, None, :]  # [t, s, r]: pick r on pick s's device
    expert = jnp.where(there, idx[:, None, :] % local, local)
    weight = jnp.where(there, weights[:, None, :], 0)
    return expert.reshape(-1, idx.shape[-1]), weight.reshape(-1, idx.shape[-1])


methods.moe.register("all_gather")(expert_parallel(all_gather))
methods.moe.register("all_to_all")(expert_parallel(all_to_all))
methods.moe.register("ragged_all_to_all")(expert_parallel(ragged_all_to_all))
methods.moe.register("ragged_tokens")(expert_parallel(ragged_tokens))
