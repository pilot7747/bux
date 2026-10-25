"""``methods(moe=nccl_ep)``: expert parallelism by NCCL EP, a DeepEP of NCCL's device
API, through TransformerEngine's JAX bindings, on GPUs only. bux's Docker image has
them, and :doc:`/parallelism/experts` explains the method.

A token's path, as in ``experts.ragged_tokens``: it goes once to each device that
holds any of its experts (and over InfiniBand once to each other node, whose NVLink
takes it on), and arrives as a row for each of its picks there, grouped by expert,
each expert's rows padded to a multiple of 128. The experts run on the rows as
they lie; each output, times its pick's weight, goes back, and each token sums its
own. Under a capacity the picks that do not fit drop before they go, as in
``ragged_all_to_all``. TE's one-device primitives send them, per device in
``expert_parallel``'s shard_map, with a custom_vjp each way: TE's global-view ops
hang across nodes.

NCCL EP keeps its buffers outside the step, sized for the run, so `start` runs once
a process before the step traces (`bux.parallel.methods.prepare`). Under
``methods.moe_stream`` its kernels run on a stream of their own (``STREAM``), beside
the compute stream's, each call started as early as its inputs allow and waited for
as late as its results are needed (`bux.jax_internals.on_stream`).
"""

import functools
import math
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Shaped, UInt8

from bux import config as cfg
from bux import jax_internals, passes
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
    chunk_of,
    expert_parallel,
    in_chunks,
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
# The stream NCCL EP's kernels run on under `methods.moe_stream`, one of their own:
# XLA runs the compute stream's kernels beside them, and theirs one after another.
STREAM = "gpu_stream:1"
# Of NCCL EP's calls ready at once, XLA starts first those that send rows to the
# experts (the dispatch, and the combine's gradient), then those that send them back:
# in chunks, the next chunk's dispatch then runs beside this one's experts, not queued
# on NCCL EP's stream behind this one's combine (``jax_internals.on_stream``).
THERE, BACK = 1, 0
# bux's pass before XLA's scheduler that takes a stream's mark off the parts of a
# stream call's results (``jax_internals.parts_off_the_stream``): else each part of a
# call's results on NCCL EP's stream waits there for the calls after it.
PARTS = "nccl_ep_parts_off_the_stream"
passes.register(PARTS, stage=passes.BEFORE_SCHEDULING, key="1")(
    jax_internals.parts_off_the_stream
)


class Started(NamedTuple):
    """What NCCL EP was started with in this process: the run's experts and picks a
    token, the most tokens a device sends at once and their width, and the rows each
    dispatch brings a device, padded, of which NCCL EP takes no fewer."""

    experts: int
    picks: int
    tokens: int
    d: int
    rows: int


started: Started | None = None


def transformer_engine() -> ModuleType:
    """Return TE's EP primitives (`transformer_engine.jax.cpp_extensions.ep`), or
    raise how to get them. It imports once jax's GPU backend has loaded NCCL, which
    TE's core library loads too."""
    try:
        from transformer_engine.jax.cpp_extensions import ep
    except ImportError as e:
        raise RuntimeError(
            "methods(moe=nccl_ep) needs TransformerEngine's JAX bindings, on GPUs, "
            "which bux's Docker image has (docs/parallelism/experts.md)"
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
    passes.use(PARTS, run.train.methods.moe_stream)
    tokens = _tokens_a_device(run, ep)
    peers = _size(ep)
    rows = _room(tokens * top_k, num_experts // peers, peers)
    wanted = Started(num_experts, top_k, tokens, run.arch.d_model, rows)
    if started != wanted:
        _bootstrap(ep[0], wanted)
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
    """Return the most tokens a device sends at once: a chunk of its share of a
    microbatch's rows (``experts.chunk_of``). An eval step, which takes a step's rows
    whole, sends them in pieces of that many (``nccl_ep``)."""
    t = run.train
    rows = t.batch // t.methods.microbatches
    shape = (rows, t.seq_len, run.arch.d_model)
    return chunk_of(rows * t.seq_len // _size(axes_of(token_spec(shape, ep))))


def _room(picks: int, local: int, peers: int) -> int:
    """Return how many rows a device takes from `peers` devices of `picks` picks each:
    the picks ``experts._room`` lets it take, and room for each of its `local`
    experts' blocks to pad to a multiple of 128."""
    rows = experts_room(picks, peers) + local * (ALIGNED - 1)
    return math.ceil(rows / ALIGNED) * ALIGNED


def _bootstrap(axis: str, sizes: Started) -> None:
    """Start NCCL EP's group of each line of devices along `axis`, on a communicator
    TE takes from XLA, with buffers of `sizes`; first stop the one before, if any."""
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
            num_experts=sizes.experts,
            max_tokens_per_rank=sizes.tokens,
            recv_capacity_per_rank=sizes.rows,
            hidden_dim=sizes.d,
            drop_on_overflow=False,  # bux drops what does not fit first (``_kept``)
        )


# --- A token's path ----------------------------------------------------------------


def nccl_ep(
    z: Float[Array, "b t d"],
    idx: Int[Array, "b t k"],
    weights: Float[Array, "b t k"],
    kernels: Kernels,
    **settings: object,
) -> Combined:
    """Send each token once to each device that holds any of its experts, and back as
    its picks' weighted sum, by NCCL EP's kernels (a DeepEP of NCCL's, through
    TransformerEngine), on GPUs: rows grouped by expert as they arrive, each
    expert's padded to a multiple of 128. ``methods.moe_stream`` runs them on a
    stream of their own.

    It runs `sent_once` of a device's tokens, in as many pieces one after another
    as NCCL EP was started for (`~bux.parallel.experts.in_chunks`): an eval step's,
    a step's rows whole, send more tokens than a training microbatch's."""
    assert started is not None, "methods.prepare starts NCCL EP before a step traces"
    tokens = z.size // z.shape[-1]
    pieces = -(-tokens // started.tokens)
    if tokens % pieces:
        raise ValueError(
            f"methods(moe=nccl_ep) sends {tokens} tokens of a device in pieces of at "
            f"most {started.tokens}, which do not divide them"
        )
    return in_chunks(sent_once, pieces)(z, idx, weights, kernels, **settings)


def sent_once(
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
    rows = started.rows  # what every dispatch brings: NCCL EP takes no fewer
    kept, arrivals = _kept(idx, local, axes)
    picks = jnp.where(kept, idx, -1).reshape(tokens, k).astype(jnp.int32)  # -1: none
    chosen = jnp.where(kept, weights, 0).reshape(tokens, k).astype(jnp.float32)
    sent = _dispatched(rows, tokens, k, z.reshape(tokens, d), picks, chosen)
    held = _picks_in(sent.sizes, arrivals.kept[jax.lax.axis_index(axes)].sum(-1), rows)
    row_weights = jnp.where(held, sent.weights, 0)
    arrived = remat.label(EXCHANGED, sent.rows)
    made = experts(arrived, sent.sizes, row_weights, *kernels, ffn, norm)
    y = remat.label(EXCHANGED, _combined(rows, tokens, k, sent.handle, picks, made))
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


# A route's handle: TE's prepare writes the route into it, and TE keys its own state
# of the route by the handle's address, which a copy changes. So no handle outlives
# the pass it is prepared in: the combine takes its dispatch's, and each call of a
# backward a handle prepared anew from the picks its forward kept, in whatever order
# microbatches' forwards and backwards run.
Handle = UInt8[Array, "1 h"]


class Dispatched(NamedTuple):
    """What arrives at a device: `rows` of the picks of its experts, sizes[l] of them
    (padded) its expert l's in turn, each pick's weight, and the `handle` that sends
    their outputs back."""

    rows: Float[Array, "r d"]
    weights: Float[Array, " r"]
    handle: Handle
    sizes: Int[Array, " l"]


def _route(k: int) -> dict[str, int]:
    """Return what each of TE's primitives takes of the route: picks, and alignment."""
    return {"top_k": k, "dispatch_output_per_expert_alignment": ALIGNED}


def _bind(way: int, primitive: Any, *args: Shaped[Array, "..."], **params: Any) -> Any:
    """Return TE's one-device `primitive` of `args`, which sends rows `way`, THERE
    or BACK (``_on_its_stream``)."""
    return _on_its_stream(lambda: _bound(primitive, *args, **params), way)


def _bound(primitive: Any, *args: Shaped[Array, "..."], **params: Any) -> Any:
    return primitive.inner_primitive.bind(*args, **params)


def _on_its_stream[T](run: Callable[[], T], way: int) -> T:
    """Return run(), under `methods.moe_stream` the primitives it binds on NCCL EP's
    stream as one async call, which sends rows `way`: the compute stream waits for
    each call where it uses its results, so a call a primitive would make it wait for
    each one in turn."""
    on = active()
    if on is None or not on.methods.moe_stream:
        return run()
    return jax_internals.on_stream(STREAM, run, way)


def _prepared(k: int, idx: Int[Array, "n k"]) -> tuple[Int[Array, "1 l"], Handle]:
    """Return TE's prepare of the `k` picks idx of each token: the rows each of this
    device's experts takes, padded, and the handle of their route (``Handle``)."""
    te = transformer_engine()
    sizes, _, handle = _bound(te.EpPreparePrimitive, idx, **_route(k), is_outer=False)
    return sizes, handle


def _prepared_again(k: int, idx: Int[Array, "n k"], way: int) -> Handle:
    """Return a handle of the route of the picks idx for a call of a backward, which
    takes none from its forward (``Handle``): TE's prepare in a call of its own,
    which XLA can start as soon as the backward starts, its picks long kept."""
    return _on_its_stream(lambda: _prepared(k, idx)[1], way)


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
) -> tuple[Dispatched, Int[Array, "n k"]]:
    def prepared_and_sent() -> tuple[
        Float[Array, "1 r d"], Float[Array, "1 r"], Handle, Int[Array, "1 l"]
    ]:
        sizes, handle = _prepared(k, idx)
        arrived, arrived_weights = _bound(
            transformer_engine().EpDispatchPrimitive,
            handle,
            idx,
            x,
            weights,
            **_route(k),
            recv_capacity_per_rank=rows,
            is_outer=False,
        )
        return arrived, arrived_weights, handle, sizes

    arrived, arrived_weights, handle, sizes = _on_its_stream(prepared_and_sent, THERE)
    return Dispatched(arrived[0], arrived_weights[0], handle, sizes[0]), idx


def _dispatched_backward(
    rows: int, tokens: int, k: int, idx: Int[Array, "n k"], g: Dispatched
) -> tuple[Float[Array, "n d"], None, Float[Array, "n k"]]:
    gx, gw = _bind(
        BACK,
        transformer_engine().EpDispatchBwdPrimitive,
        _prepared_again(k, idx, BACK),
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
    handle: Handle,
    idx: Int[Array, "n k"],
    made: Float[Array, "r d"],
) -> Float[Array, "n d"]:
    """Return each of this device's `tokens` tokens summed from the rows `made` of it
    on its experts' devices, sent back by `handle`, its dispatch's of the picks `idx`
    (NCCL EP's combine). Its gradient sends each token's gradient to its rows the way
    they came."""
    return _combined_forward(rows, tokens, k, handle, idx, made)[0]


def _combined_forward(
    rows: int,
    tokens: int,
    k: int,
    handle: Handle,
    idx: Int[Array, "n k"],
    made: Float[Array, "r d"],
) -> tuple[Float[Array, "n d"], Int[Array, "n k"]]:
    y = _bind(
        BACK,
        transformer_engine().EpCombinePrimitive,
        handle,
        made[None],
        **_route(k),
        out_leading_shape=(tokens,),
        out_partition_spec=None,
    )
    return y, idx


def _combined_backward(
    rows: int, tokens: int, k: int, idx: Int[Array, "n k"], g: Float[Array, "n d"]
) -> tuple[None, None, Float[Array, "r d"]]:
    g_made = _bind(
        THERE,
        transformer_engine().EpCombineBwdPrimitive,
        _prepared_again(k, idx, THERE),
        g,
        **_route(k),
        recv_capacity_per_rank=rows,
        is_outer=False,
    )
    return None, None, g_made[0]


_combined.defvjp(_combined_forward, _combined_backward)


methods.moe.register("nccl_ep")(expert_parallel(nccl_ep))
