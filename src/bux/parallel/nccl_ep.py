"""`methods(moe=nccl_ep)`: expert parallelism by NCCL EP, a DeepEP of NCCL's device
API, through TransformerEngine's JAX bindings, on GPUs only (docs/parallelism.md,
"TransformerEngine's NCCL EP"; the Docker image has them).

A token's path, as in ``experts.ragged_tokens``: it goes once to each device that
holds any of its experts (and over InfiniBand once to each other node, whose NVLink
takes it on), and arrives as a row for each of its picks there, grouped by expert,
each expert's rows padded to a multiple of 128. The experts run on the rows as
they lie; each output, times its pick's weight, goes back, and each token sums its
own. Under a capacity the picks that do not fit drop before they go, as in
``ragged_all_to_all``. TE's one-device primitives send them, per device in
``expert_parallel``'s shard_map, with a custom_vjp each way: TE's global-view ops
hang across nodes.

NCCL EP keeps its buffers outside the step, sized for the run, so ``start`` runs once
a process before the step traces (``methods.prepare``).
"""

import functools
import math
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, UInt8

from bux import config as cfg
from bux.architecture.experts.moe import Kernels, RowNorm, experts, sort_by_expert
from bux.architecture.precision import matmuls
from bux.architecture.repeats import remat
from bux.parallel import methods
from bux.parallel.experts import (
    EXCHANGED,
    Arrivals,
    Axes,
    Combined,
    _arrivals,
    expert_parallel,
    token_spec,
)
from bux.parallel.experts import _room as experts_room
from bux.parallel.layout import axes_of, spec
from bux.parallel.mesh import active

if TYPE_CHECKING:
    from bux.lang import ir

# Each expert's rows start at a multiple of this: NCCL EP's high-throughput kernels
# take no less, and the experts' grouped GEMMs are content with it.
ALIGNED = 128
# What NCCL EP was started with in this process: experts, picks, tokens, d, rows.
started: tuple[int, int, int, int, int] | None = None


def transformer_engine() -> ModuleType:
    """Return TE's EP primitives (`transformer_engine.jax.cpp_extensions.ep`), or
    raise how to get them. It imports once jax's GPU backend has loaded NCCL, which
    TE's core library loads too."""
    try:
        from transformer_engine.jax.cpp_extensions import ep
    except ImportError as e:
        raise RuntimeError(
            "methods(moe=nccl_ep) needs TransformerEngine's JAX bindings, on GPUs, "
            "which bux's Docker image has (docs/parallelism.md)"
        ) from e
    return ep


# --- Started once a process, before the step traces ---------------------------------


@methods.before("moe", "nccl_ep")
def start(run: "ir.Run") -> None:
    """Start NCCL EP for `run`, once a process: its groups along the experts' axis,
    with buffers for the most tokens a device sends and the rows it takes. Nothing
    where no axis splits the experts, which then run as `sorted`."""
    global started
    num_experts, top_k = _moe_shape(run)
    ep = axes_of(spec(("experts",), (num_experts,)))
    if not ep:
        return
    _check(run, ep)
    transformer_engine()  # or say how to get it
    tokens = _tokens_a_device(run, ep)
    peers = _size(ep)
    rows = _room(tokens * top_k, num_experts // peers, peers)
    wanted = (num_experts, top_k, tokens, run.arch.d_model, rows)
    if started != wanted:
        _bootstrap(ep[0], num_experts, tokens, run.arch.d_model, rows)
        started = wanted


def _moe_shape(run: "ir.Run") -> tuple[int, int]:
    """Return the experts and picks of the run's MoE layers, all alike: NCCL EP is
    sized for one."""
    shapes = {
        (layer.config.num_experts, layer.config.top_k)
        for layer in run.arch.layers()
        if isinstance(layer.config, cfg.MoEConfig)
    }
    if len(shapes) != 1:
        raise ValueError(
            f"methods(moe=nccl_ep) takes one MoE shape a run, not {shapes}"
        )
    return shapes.pop()


def _check(run: "ir.Run", ep: Axes) -> None:
    """Raise where NCCL EP cannot go: experts over more than one axis, or tokens other
    than bfloat16, as in MXFP8."""
    if len(ep) > 1:
        raise ValueError(
            f"methods(moe=nccl_ep) splits the experts over one axis, not {ep}"
        )
    precision = run.train.precision
    if precision.compute != "bfloat16" or precision.dispatch != "compute":
        raise ValueError(
            "methods(moe=nccl_ep) sends the tokens in bfloat16 alone: "
            "precision(compute=bfloat16, dispatch=compute)"
        )


def _size(axes: Axes) -> int:
    """Return how many devices `axes` of the mesh in force have."""
    on = active()
    assert on is not None  # the experts split over a mesh's axes
    return on.size(*axes)


def _tokens_a_device(run: "ir.Run", ep: Axes) -> int:
    """Return the most tokens a device sends at once: its share of a step's rows,
    whole, as an eval step takes them; a training microbatch or chunk sends fewer."""
    t = run.train
    shape = (t.batch, t.seq_len, run.arch.d_model)
    return t.batch * t.seq_len // _size(axes_of(token_spec(shape, ep)))


def _room(picks: int, local: int, peers: int) -> int:
    """Return how many rows a device takes from `peers` devices of `picks` picks each:
    the picks ``experts._room`` lets it take, and room for each of its `local`
    experts' blocks to pad to a multiple of 128."""
    rows = experts_room(picks, peers) + local * (ALIGNED - 1)
    return math.ceil(rows / ALIGNED) * ALIGNED


def _bootstrap(axis: str, num_experts: int, tokens: int, d: int, rows: int) -> None:
    """Start NCCL EP's group of each line of devices along `axis`, on a communicator
    TE takes from XLA; first stop the one before, if any."""
    from transformer_engine.jax.ep import ep_bootstrap, ep_finalize
    from transformer_engine.jax.sharding import MeshResource, global_shard_guard

    if started is not None:
        ep_finalize()
    on = active()
    assert on is not None
    with on.mesh, global_shard_guard(MeshResource(ep_resource=axis)):
        ep_bootstrap(
            world_size=None,
            rank=None,
            num_experts=num_experts,
            max_tokens_per_rank=tokens,
            recv_capacity_per_rank=rows,
            hidden_dim=d,
            drop_on_overflow=False,  # bux drops what does not fit first (``_kept``)
        )


# --- A token's path ----------------------------------------------------------------


def nccl_ep(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    *,
    axes: Axes,
    ffn: cfg.MLPConfig,
    norm: RowNorm | None,
) -> Combined:
    """Send each token once to each device that holds any of its experts, a row for
    each of its picks there, grouped by expert; run the experts on the rows as they
    lie; send each output back times its pick's weight, and sum each token's. A device
    takes `methods.capacity` times the rows of a fair share, and the picks past it
    drop before they go, as in ``ragged_all_to_all``."""
    (d, k), local = (z.shape[-1], idx.shape[-1]), kernels[0].shape[0]
    tokens, peers = z.size // d, jax.lax.axis_size(axes)
    if matmuls.in_force().dispatch != "compute":
        raise ValueError("methods(moe=nccl_ep) sends the tokens in bfloat16, not MXFP8")
    assert started is not None, "methods.prepare starts NCCL EP before a step traces"
    rows = _room(idx.size, local, peers)
    kept, arrivals = _kept(idx, local, axes)
    picks = jnp.where(kept, idx, -1).reshape(tokens, k).astype(jnp.int32)  # -1: none
    chosen = jnp.where(kept, weights, 0).reshape(tokens, k).astype(jnp.float32)
    sent = _dispatched(rows, tokens, k, z.reshape(tokens, d), picks, chosen)
    held = _picks_in(sent.sizes, arrivals.kept[jax.lax.axis_index(axes)].sum(-1), rows)
    row_weights = jnp.where(held, sent.weights, 0)
    arrived = remat.label(EXCHANGED, sent.rows)
    made = experts(arrived, sent.sizes, row_weights, *kernels, ffn, norm)
    y = remat.label(EXCHANGED, _combined(rows, tokens, k, sent.handle, made))
    return y.reshape(z.shape), 1 - arrivals.kept.sum() / (idx.size * peers)


def _kept(
    idx: Int[Array, "b t k"], local: int, axes: Axes
) -> tuple[Bool[Array, "b t k"], Arrivals]:
    """Return which of this device's picks fit in what their experts' devices take, as
    ``ragged_all_to_all`` keeps them: of a device's experts, the last ones' picks drop
    first, and of an expert's, the last peers'; and where the kept picks land
    (``experts._arrivals``). NCCL EP would drop what does not fit too, but then its
    weights' gradients go wrong."""
    peers, device = jax.lax.axis_size(axes), jax.lax.axis_index(axes)
    _, inverse, sizes = sort_by_expert(idx, local * peers)
    counts = jax.lax.all_gather(sizes, axes)  # [i, e]: device i's picks of expert e
    arrivals = _arrivals(counts, experts_room(idx.size, peers))
    fits = arrivals.kept[:, :, device].reshape(-1)  # [e]: how many of mine fit
    flat = idx.reshape(-1)
    at = inverse - (jnp.cumsum(sizes) - sizes)[flat]  # each pick among its expert's
    return (at < fits[flat]).reshape(idx.shape), arrivals


def _picks_in(
    sizes: Int[Array, " l"], picks: Int[Array, " l"], rows: int
) -> Bool[Array, " r"]:
    """Return which of the `rows` that arrived hold a pick: expert l's block of
    sizes[l] rows starts where the one before ended, and its first picks[l] rows are
    picks. The rest of a block is 0 and the rows past the last are left as they were;
    the weights of both are garbage."""
    starts = jnp.cumsum(sizes) - sizes
    at = jnp.arange(rows)[:, None]
    return ((at >= starts) & (at < starts + picks)).any(-1)


# --- TE's one-device primitives, each way with its gradient ------------------------


class Dispatched(NamedTuple):
    """What arrives at a device: `rows` of the picks of its experts, sizes[l] of them
    (padded) its expert l's in turn, each pick's weight, and the `handle` that sends
    their outputs back."""

    rows: Float[Array, "r d"]
    weights: Float[Array, " r"]
    handle: UInt8[Array, "1 h"]
    sizes: Int[Array, " l"]


def _route(k: int) -> dict[str, int]:
    """Return what each of TE's primitives takes of the route: picks, and alignment."""
    return {"top_k": k, "dispatch_output_per_expert_alignment": ALIGNED}


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _dispatched(
    rows: int,
    tokens: int,
    k: int,
    x: Float[Array, "n d"],
    idx: Int[Array, "n k"],
    weights: Float[Array, "n k"],
) -> Dispatched:
    """Return what NCCL EP's dispatch brings this device, in `rows` rows, of the
    `tokens` tokens x of each device of its group, by their `k` picks `idx` and the
    picks' weights: TE's prepare, which tells each device what comes, then its
    dispatch. Its gradient sends the rows' gradients back the way they came."""
    return _dispatched_forward(rows, tokens, k, x, idx, weights)[0]


def _dispatched_forward(
    rows: int,
    tokens: int,
    k: int,
    x: Float[Array, "n d"],
    idx: Int[Array, "n k"],
    weights: Float[Array, "n k"],
) -> tuple[Dispatched, UInt8[Array, "1 h"]]:
    te = transformer_engine()
    sizes, _, handle = te.EpPreparePrimitive.inner_primitive.bind(
        idx, **_route(k), is_outer=False
    )
    arrived, arrived_weights = te.EpDispatchPrimitive.inner_primitive.bind(
        handle,
        idx,
        x,
        weights,
        **_route(k),
        recv_capacity_per_rank=rows,
        is_outer=False,
    )
    sent = Dispatched(arrived[0], arrived_weights[0], handle, sizes[0])
    return sent, handle


def _dispatched_backward(
    rows: int, tokens: int, k: int, handle: UInt8[Array, "1 h"], g: Dispatched
) -> tuple[Float[Array, "n d"], None, Float[Array, "n k"]]:
    gx, gw = transformer_engine().EpDispatchBwdPrimitive.inner_primitive.bind(
        handle,
        g.rows[None],
        g.weights[None],
        **_route(k),
        out_leading_shape=(tokens,),
        out_partition_spec=None,
    )
    return gx, None, gw


_dispatched.defvjp(_dispatched_forward, _dispatched_backward)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _combined(
    rows: int,
    tokens: int,
    k: int,
    handle: UInt8[Array, "1 h"],
    made: Float[Array, "r d"],
) -> Float[Array, "n d"]:
    """Return each of this device's `tokens` tokens summed from the rows `made` of it
    on its experts' devices, sent back by `handle` (NCCL EP's combine). Its gradient
    sends each token's gradient to its rows the way they came."""
    return _combined_forward(rows, tokens, k, handle, made)[0]


def _combined_forward(
    rows: int,
    tokens: int,
    k: int,
    handle: UInt8[Array, "1 h"],
    made: Float[Array, "r d"],
) -> tuple[Float[Array, "n d"], UInt8[Array, "1 h"]]:
    y = transformer_engine().EpCombinePrimitive.inner_primitive.bind(
        handle,
        made[None],
        **_route(k),
        out_leading_shape=(tokens,),
        out_partition_spec=None,
    )
    return y, handle


def _combined_backward(
    rows: int, tokens: int, k: int, handle: UInt8[Array, "1 h"], g: Float[Array, "n d"]
) -> tuple[None, Float[Array, "r d"]]:
    g_made = transformer_engine().EpCombineBwdPrimitive.inner_primitive.bind(
        handle, g, **_route(k), recv_capacity_per_rank=rows, is_outer=False
    )
    return None, g_made[0]


_combined.defvjp(_combined_forward, _combined_backward)


methods.moe.register("nccl_ep")(expert_parallel(nccl_ep))
