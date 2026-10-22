"""Expert parallelism: methods of ``methods.moe`` that split the experts over the
axes `layout.experts` gives them (`expert`), each device running its own.

    all_gather         each device gathers its group's tokens, runs its experts on
                       them, and a psum_scatter brings each token its sum back
    all_to_all         each pick goes to its expert's device and back, in one buffer
                       per peer that holds every local pick: dropless
    ragged_all_to_all  each pick goes to its expert's device and back, in one ragged
                       slice per peer; XLA:CPU lacks it

Each runs a body on one device's tokens (``method``), in a shard_map manual over the
axes that split the tokens, so that every device sorts its own. The experts' kernels
enter split over the experts' axes and whole over the others (FSDP's gathered
first); `tensor` stays Auto. Routing, the balance loss and the router bias ran
before, in the global view.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux.architecture.moe import (
    Kernels,
    MoE,
    Rows,
    pick_rows,
    routed,
    run_sorted,
    sort_by_expert,
)
from bux.parallel import methods
from bux.parallel.layout import spec
from bux.parallel.mesh import active

Axes = tuple[str, ...]
# What a method runs on one device: (z, idx, weights, kernels, *, axes, ffn, norm),
# its own tokens and its own experts' kernels, the experts split over `axes`.
Body = Callable[..., Float[Array, "b t d"]]
# jax.lax.ragged_all_to_all's signature: another exchange is a function.
Exchange = Callable[..., Float[Array, "m d"]]


def method(body: Body) -> Callable[..., Float[Array, "b t d"]]:
    """Return a method of ``methods.moe`` that runs `body` on every device, or
    `sorted` where no axis splits the experts."""

    def run(
        moe: MoE,
        z: Float[Array, "b t d"],
        idx: Int[Array, "b t k"],
        weights: Float[Array, "b t k"],
    ) -> Float[Array, "b t d"]:
        ep = _axes(spec(("experts",), (moe.config.num_experts,)))
        if not ep:
            return run_sorted(moe, z, idx, weights)
        tokens = _tokens(z, ep)
        ffn, norm = moe.config.expert, moe.expert_norm
        local = functools.partial(body, axes=ep, ffn=ffn, norm=norm)
        return jax.shard_map(
            local,
            in_specs=(tokens, tokens, tokens, P(ep)),
            out_specs=tokens,
            axis_names=set(_axes(tokens)),
        )(z, idx, weights, moe.kernels())

    return run


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
) -> Float[Array, "b t d"]:
    """Gather the group's tokens onto each of its devices, run each device's experts
    on them, and reduce-scatter the sums back."""
    first = jax.lax.axis_index(axes) * kernels[0].shape[0]
    group = functools.partial(jax.lax.all_gather, axis_name=axes, tiled=True)
    y = routed(group(z), group(idx) - first, group(weights), *kernels, ffn, norm)
    return jax.lax.psum_scatter(y, axes, tiled=True)


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
) -> Float[Array, "b t d"]:
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
    return jnp.einsum("btkd,btk->btd", per_pick, weights)


# --- ragged_all_to_all -----------------------------------------------------------------


def _arrived(counts: Int[Array, "p l"], m: int) -> Int[Array, " m"]:
    """Return the local expert of each of `m` rows that arrive from the peers in
    turn, each peer's sorted by expert, counts[i, j] of them from peer i for expert j;
    `l`, none, past them."""
    ends = jnp.cumsum(counts.reshape(-1))
    # How many groups end at or before each row (jnp.searchsorted fails in a
    # shard_map on jax 0.12).
    ended = jnp.cumsum(jnp.zeros(m, ends.dtype).at[ends].add(1, mode="drop"))
    local = counts.shape[1]
    return jnp.where(ended < counts.size, ended % local, local)


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
) -> Float[Array, "b t d"]:
    """Send each peer its picks as one slice and back, after the slices of the peers
    before it: a buffer holds what every peer could send, but only what they do send
    moves."""
    d, k, n, local = z.shape[-1], idx.shape[-1], idx.size, kernels[0].shape[0]
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    order, sizes = sort_by_expert(idx, local * peers)
    every = jax.lax.all_gather(sizes, axes)  # each device's picks of each expert
    sent = every.reshape(peers, peers, local).sum(-1)  # [i, j]: from device i to j
    before, ahead = jnp.cumsum(sent, 0) - sent, jnp.cumsum(sent, 1) - sent
    # Buffers like z, so that they vary over the devices as it does: the exchange
    # types its result as its buffer.
    received = exchange(
        pick_rows(z.reshape(-1, d), order, jnp.argsort(order), k),
        jnp.zeros_like(z, shape=(peers * n, d)),
        *(ahead[device], sent[device], before[device], sent[:, device]),
        axis_name=axes,
    )
    counts = jax.lax.dynamic_slice_in_dim(every, device * local, local, axis=1)
    expert = _arrived(counts, peers * n)[None, :, None]
    ones = jnp.ones(expert.shape, z.dtype)
    out = routed(received[None], expert, ones, *kernels, ffn, norm)[0]
    returned = exchange(
        out,
        jnp.zeros_like(z, shape=(n, d)),
        *(before[:, device], sent[:, device], ahead[:, device], sent[device]),
        axis_name=axes,
    )
    per_pick = returned[jnp.argsort(order)].reshape(*idx.shape, d)
    return jnp.einsum("btkd,btk->btd", per_pick, weights)


methods.moe.register("all_gather")(method(gathered))
methods.moe.register("all_to_all")(method(exchanged))
methods.moe.register("ragged_all_to_all")(method(ragged))
