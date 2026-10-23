"""Expert parallelism: methods of ``methods.moe`` that split the experts over the
axes `layout.experts` gives them (`expert`), each device running its own.

    all_gather         each device gathers its group's tokens, runs its experts on
                       them, and a psum_scatter brings each token its sum back
    all_to_all         each pick goes to its expert's device and back, in one buffer
                       per peer that holds every local pick: dropless
    ragged_all_to_all  each pick goes to its expert's device and back, in one ragged
                       slice per expert of each peer, so that a device's rows arrive
                       sorted by its experts; `methods.capacity` bounds them. XLA:CPU
                       lacks it

Each runs a body on one device's tokens (``method``), in a shard_map manual over the
axes that split the tokens, so that every device sorts its own. The experts' kernels
enter split over the experts' axes and whole over the others (FSDP's gathered
first); `tensor` stays Auto. Routing, the balance loss and the router bias ran
before, in the global view.
"""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import config as cfg
from bux.architecture.moe import (
    Kernels,
    MoE,
    Rows,
    experts,
    permuted,
    pick_rows,
    routed,
    run_sorted,
    sort_by_expert,
    sum_rows,
)
from bux.architecture.registry import Taps, report
from bux.parallel import methods
from bux.parallel.layout import spec
from bux.parallel.mesh import active

Axes = tuple[str, ...]
# What a body returns: each token's sum of its experts' outputs, and the share of the
# group's picks that a capacity dropped.
Routed = tuple[Float[Array, "b t d"], Float[Array, ""]]
# What a method runs on one device: (z, idx, weights, kernels, *, axes, ffn, norm),
# its own tokens and its own experts' kernels, the experts split over `axes`.
Body = Callable[..., Routed]
# jax.lax.ragged_all_to_all's signature: another exchange is a function.
Exchange = Callable[..., Shaped[Array, "m d"]]


def method(body: Body) -> Callable[..., Float[Array, "b t d"]]:
    """Return a method of ``methods.moe`` that runs `body` on every device, or
    `sorted` where no axis splits the experts. Under a capacity, a training forward
    reports the share of picks it dropped as `dropped`."""

    def run(
        moe: MoE,
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
        taps: Taps | None = None,
    ) -> Float[Array, "b t d"]:
        ep = _axes(spec(("experts",), (moe.config.num_experts,)))
        if not ep:
            return run_sorted(moe, z, idx, weights, taps)
        tokens = _tokens(z, ep)
        ffn, norm = moe.config.expert, moe.expert_norm

        def local(*args: object) -> Routed:
            y, dropped = body(*args, axes=ep, ffn=ffn, norm=norm)
            return y, jax.lax.pmean(dropped, _axes(tokens))

        # Untyped by what varies, as `sorted`'s: tokamax's kernel takes
        # jnp.searchsorted, which fails in a typed shard_map on jax 0.12.
        y, dropped = jax.shard_map(
            local,
            in_specs=(tokens, tokens, tokens, P(ep)),
            out_specs=(tokens, P()),
            axis_names=set(_axes(tokens)),
            check_vma=False,
        )(z, idx, weights, moe.kernels())
        if _capacity() is not None:
            report(taps, "dropped", 0.0, dropped)
        return y

    return run


def _capacity() -> float | None:
    """Return the run's `methods.capacity`."""
    on = active()
    return None if on is None else on.methods.capacity


def _axes(split: P) -> Axes:
    """Return the mesh axes `split` splits over, in order."""
    return tuple(a for s in split if s for a in ((s,) if isinstance(s, str) else s))


def _tokens(z: Float[Array, "b t d"], ep: Axes) -> P:
    """Return the tokens' spec: rows over `batch` and positions over `seq`, and rows
    over the experts' axes `ep` too where those leave them whole."""
    batch, seq, _ = spec(("batch", "seq", None), z.shape)
    taken = _axes(P(batch, seq))
    rows = _axes(P(batch)) + tuple(a for a in ep if a not in taken)
    on = active()
    assert on is not None  # `ep` came from a mesh
    if z.shape[0] % on.size(*rows):
        raise ValueError(
            f"the experts split over {ep}, and so do the tokens: "
            f"{z.shape[0]} rows do not split over {rows}"
        )
    return P(rows or None, seq, None)


# --- all_gather ------------------------------------------------------------------------


def gathered(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: Rows | None,
) -> Routed:
    """Gather the group's tokens onto each of its devices, run each device's experts
    on them, and reduce-scatter the sums back."""
    first = jax.lax.axis_index(axes) * kernels[0].shape[0]
    group = functools.partial(jax.lax.all_gather, axis_name=axes, tiled=True)
    y = routed(group(z), group(idx) - first, group(weights), *kernels, ffn, norm)
    return jax.lax.psum_scatter(y, axes, tiled=True), jnp.zeros(())


# --- all_to_all ------------------------------------------------------------------------


def _buffers(
    idx: Int[Array, "b t k"], local: int, peers: int
) -> tuple[Int[Array, "p n"], Int[Array, "p n"], tuple[Int[Array, " n"], ...]]:
    """Lay the n picks out in a buffer of n slots per peer: return the pick in each
    slot (any, past the peer's), its expert on that peer (`local`, none, past them),
    and each pick's (peer, slot)."""
    flat, n = idx.reshape(-1), idx.size
    order, sizes = sort_by_expert(idx, local * peers)
    sent = sizes.reshape(peers, local).sum(-1)
    start = jnp.cumsum(sent) - sent
    slot = jnp.arange(n)
    pick = order[jnp.minimum(start[:, None] + slot, n - 1)]
    expert = jnp.where(slot < sent[:, None], flat[pick] % local, local)
    peer = flat // local
    return pick, expert, (peer, jnp.argsort(order) - start[peer])


def exchanged(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: Rows | None,
) -> Routed:
    """Send each pick to its expert's device and back, through fixed-size buffers:
    each peer's holds every local pick, so that none drops."""
    d, k, local = z.shape[-1], idx.shape[-1], kernels[0].shape[0]
    pick, expert, slot = _buffers(idx, local, jax.lax.axis_size(axes))
    swap = functools.partial(
        jax.lax.all_to_all, axis_name=axes, split_axis=0, concat_axis=0, tiled=True
    )
    rows, expert = swap(z.reshape(-1, d)[pick // k]), swap(expert)[..., None]
    ones = jnp.ones(expert.shape, z.dtype)
    out = swap(routed(rows, expert, ones, *kernels, ffn, norm))
    per_pick = out[slot].reshape(*idx.shape, d)
    return jnp.einsum("btkd,btk->btd", per_pick, weights), jnp.zeros(())


# --- ragged_all_to_all -----------------------------------------------------------------


class Slices(NamedTuple):
    """The slices of a ragged exchange by expert, [j, l, i]: the picks device i has
    for expert l of device j, where they start on device j, and how many fit."""

    starts: Int[Array, "p l p"]
    kept: Int[Array, "p l p"]


class Ragged(NamedTuple):
    """``jax.lax.ragged_all_to_all``'s arguments after its buffers, a slice an entry:
    slice s goes to peer s // (slices a peer)."""

    input_offsets: Int[Array, " s"]
    send_sizes: Int[Array, " s"]
    output_offsets: Int[Array, " s"]
    recv_sizes: Int[Array, " s"]


def _rows(n: int, peers: int, capacity: float | None) -> int:
    """Return how many rows a device of `peers` takes, sending `n`: `capacity` times
    as many, or all that could come."""
    most = n * peers
    return most if capacity is None else min(most, math.ceil(capacity * n))


def _slices(every: Int[Array, "p E"], rows: int) -> Slices:
    """Lay out what each device takes, from the picks every[i, e] of each device i
    for each expert e: on device j, those of its expert 0 from each device in turn,
    then those of its expert 1, and so on, to `rows` rows; past them none fit."""
    peers = every.shape[0]
    counts = every.reshape(peers, peers, -1).transpose(1, 2, 0)  # [j, l, i]
    ends = jnp.cumsum(counts.reshape(peers, -1), axis=1).reshape(counts.shape)
    starts = ends - counts
    kept = jnp.clip(rows - starts, 0, counts)
    return Slices(jnp.minimum(starts, rows), kept)


def _dispatch(
    slices: Slices, first: Int[Array, " E"], device: Int[Array, ""]
) -> Ragged:
    """Return the exchange that sends this device's picks to their experts, from
    first[e] on for expert e, as they are sorted here."""
    return Ragged(
        first,
        slices.kept[:, :, device].reshape(-1),
        slices.starts[:, :, device].reshape(-1),
        slices.kept[device].T.reshape(-1),
    )


def _combine(
    slices: Slices, theirs: Int[Array, "p l"], device: Int[Array, ""]
) -> Ragged:
    """Return the exchange that sends what this device's experts made back, to
    theirs[i, l] on device i for its expert l, where device i sorted those picks."""
    return Ragged(
        slices.starts[device].T.reshape(-1),
        slices.kept[device].T.reshape(-1),
        theirs.reshape(-1),
        slices.kept[:, :, device].reshape(-1),
    )


def _came_back(sent: Ragged, expert: Int[Array, " n"]) -> Bool[Array, " n"]:
    """Return which of the picks, sorted by their `expert`, came back: the first
    sent.send_sizes[e] of expert e's, which start at sent.input_offsets[e]."""
    at = jnp.arange(expert.size) - sent.input_offsets[expert]
    return at < sent.send_sizes[expert]


def _buffer(rows: int, like: Shaped[Array, "*s d"]) -> Shaped[Array, "rows d"]:
    """Return `rows` rows, as wide as `like`, to receive into, left as they are: the
    experts make 0 of the rows past theirs, and what does not come back is masked."""
    return jax.lax.empty((rows, like.shape[-1]), like.dtype)


def ragged(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: Rows | None,
    exchange: Exchange = jax.lax.ragged_all_to_all,
) -> Routed:
    """Send each pick to its expert's device and back, a ragged slice for each expert
    of each peer: a device's rows arrive sorted by its experts, which run on them as
    they are. A device takes `methods.capacity` times the rows it sends and drops
    the rest: its last experts' picks first, and of an expert's, the last peers'."""
    d, k, n, local = z.shape[-1], idx.shape[-1], idx.size, kernels[0].shape[0]
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    order, sizes = sort_by_expert(idx, local * peers)
    back = jnp.argsort(order)
    every = jax.lax.all_gather(sizes, axes)  # [i, e]: device i's picks of expert e
    starts = jnp.cumsum(every, 1) - every  # [i, e]: where they start, sorted, on i
    rows = _rows(n, peers, _capacity())
    slices = _slices(every, rows)
    sent = _dispatch(slices, starts[device], device)
    picks = pick_rows(z.reshape(-1, d), order, back, k)
    arrived = exchange(picks, _buffer(rows, z), *sent, axis_name=axes)
    made = experts(arrived, slices.kept[device].sum(1), *kernels, ffn, norm)
    theirs = jax.lax.dynamic_slice_in_dim(starts, device * local, local, axis=1)
    combine = _combine(slices, theirs, device)
    returned = exchange(made, _buffer(n, z), *combine, axis_name=axes)
    kept = _came_back(sent, idx.reshape(-1)[order])
    per_pick = permuted(jnp.where(kept[:, None], returned, 0), back, order)
    y = jnp.einsum("btkd,btk->btd", per_pick.reshape(*idx.shape, d), weights)
    return y, 1 - slices.kept.sum() / every.sum()


# --- ragged_tokens ---------------------------------------------------------------------


def _visits(idx: Int[Array, "n k"], local: int, peers: int) -> Int[Array, "n k"]:
    """Return the device of each pick that is its token's first there, the device it
    visits; and `peers`, none, for the others."""
    on = idx // local
    same = on[:, :, None] == on[:, None, :]  # [t, s, r]: picks s and r on one device
    earlier = jnp.tri(idx.shape[-1], k=-1, dtype=bool)  # r before s
    return jnp.where((same & earlier).any(-1), peers, on)


def _picked_there(
    idx: Int[Array, "n k"], weights: Float[Array, "n k"], local: int
) -> tuple[Int[Array, "m k"], Float[Array, "m k"]]:
    """Return, for each pick, its token's picks on its device: their experts there
    (`local`, none, for the picks elsewhere) and their weights (0)."""
    on = idx // local
    there = on[:, :, None] == on[:, None, :]  # [t, s, r]: pick r on pick s's device
    experts = jnp.where(there, idx[:, None, :] % local, local)
    picked = jnp.where(there, weights[:, None, :], 0)
    return experts.reshape(-1, idx.shape[-1]), picked.reshape(-1, idx.shape[-1])


def _fit(sizes: Int[Array, " g"], rows: int) -> Int[Array, " g"]:
    """Return how many rows of each group fit in the first `rows`."""
    return jnp.clip(rows - (jnp.cumsum(sizes) - sizes), 0, sizes)


def _visited(
    x: Float[Array, "v d"],
    idx: Int[Array, "v k"],
    weights: Float[Array, "v k"],
    arrived: Int[Array, ""],
    kernels: Kernels,
    ffn: cfg.MLPConfig,
    norm: Rows | None,
    room: int,
) -> tuple[Float[Array, "v d"], Int[Array, ""]]:
    """Return each of the first `arrived` tokens' sum of its picks' expert outputs,
    weighted, as ``routed`` does, but of the first `room` picks by expert alone; and
    how many picks ran."""
    k, local = idx.shape[-1], kernels[0].shape[0]
    came = (jnp.arange(x.shape[0]) < arrived)[:, None]
    idx, weights = jnp.where(came, idx, local), jnp.where(came, weights, 0)
    order, sizes = sort_by_expert(idx, local)
    back = jnp.argsort(order)
    order, sizes = order[:room], _fit(sizes, room)
    made = experts(pick_rows(x, order, back, k), sizes, *kernels, ffn, norm)
    weighted = made * weights.reshape(-1)[order][:, None]
    return sum_rows(weighted, back, order, k), sizes.sum()


def tokens(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: Rows | None,
    exchange: Exchange = jax.lax.ragged_all_to_all,
) -> Routed:
    """Send each token once to each device that holds any of its experts, with its
    picks there, and back as their weighted sum: a row per device it goes to, not
    per pick, as ``ragged`` sends. A device runs `methods.capacity` times the picks
    it has, and drops the rest as ``ragged`` does; it takes as many tokens, or the
    most that could come, a token from each device."""
    d, k, n, local = z.shape[-1], idx.shape[-1], idx.size, kernels[0].shape[0]
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    room = _rows(n, peers, _capacity())  # the picks a device runs
    visit = _visits(idx.reshape(-1, k), local, peers)
    order, sizes = sort_by_expert(visit, peers)  # each device's visits, in turn
    back = jnp.argsort(order)
    every = jax.lax.all_gather(sizes, axes)  # [i, j]: device i's visits to device j
    starts = jnp.cumsum(every, 1) - every
    rows = min(room, peers * n // k)  # a token visits a device at most once
    slices = _slices(every, rows)
    sent = _dispatch(slices, starts[device], device)

    def send(a: Shaped[Array, "n e"]) -> Shaped[Array, "rows e"]:
        return exchange(a, _buffer(rows, a), *sent, axis_name=axes)

    x = send(pick_rows(z.reshape(-1, d), order, back, k))
    choices = _picked_there(idx.reshape(-1, k), weights.reshape(-1, k), local)
    picked, chosen = (send(a[order]) for a in choices)
    arrived = slices.kept[device].sum()
    made, ran = _visited(x, picked, chosen, arrived, kernels, ffn, norm, room)
    theirs = jax.lax.dynamic_slice_in_dim(starts, device, 1, axis=1)
    combine = _combine(slices, theirs, device)
    returned = exchange(made, _buffer(n, z), *combine, axis_name=axes)
    gone = visit.reshape(-1)[order]  # where each visit went, sorted
    kept = (gone < peers) & _came_back(sent, gone)
    y = sum_rows(jnp.where(kept[:, None], returned, 0), back, order, k)
    return y.reshape(z.shape), 1 - ran / n


methods.moe.register("all_gather")(method(gathered))
methods.moe.register("all_to_all")(method(exchanged))
methods.moe.register("ragged_all_to_all")(method(ragged))
methods.moe.register("ragged_tokens")(method(tokens))
