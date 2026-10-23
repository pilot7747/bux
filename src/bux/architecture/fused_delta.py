"""The gated delta rule by cuDNN's fused kernels (Frost, in cudnn-frontend, which
tokamax's cuda extra installs): forward and backward, a tenth of ``chunkwise``'s
time on a B300, keeping only their inputs for the backward pass.

One decay a head (Gated DeltaNet, DeltaNet) runs by Frost's GDN kernel, which takes
each pair's decay as the exp of a difference of summed log decays, never positive: it
runs any gate. cudnn-frontend gives it no JAX call, so a `buffer_callback` runs
cuDNN's graph of it on XLA's stream (`_graph_call`).

A decay per key channel (KDA) runs by Frost's KDA kernel, through cudnn-frontend's
own JAX call. It factors each decay through the start of a tile of 16 positions, so it
overflows where a tile decays a channel below exp(-87): a gate bounded at -5
(`gate_lower_bound`, Kimi K3) always fits it. Any other runs by it on the (row, head)
pairs where 16 positions decay no channel by more than exp(-80), and by ``chunkwise``
on the rest (`_patched`), up to a quarter of the pairs; past that, ``chunkwise`` runs
it all.

The kernels run on one SM100 or SM103 GPU at a time, on token-packed rows and their
documents' bounds (cu_seqlens); on a mesh, a `shard_map` gives them each device's rows
and heads. They take heads of 64 or 128, and every position of a row: not
`methods.recurrence = pass_state`, which splits them.
"""

import contextlib
import functools
import threading
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from bux import kernels, parallel
from bux.architecture.segments import Segments

try:  # cudnn-frontend and its JAX kernels, where the cuda extra is installed (Linux)
    import cudnn
    from cudnn.frost.device import device_context
    from cudnn.linear_attention import jax_api as frost
    from jax.experimental.buffer_callback import buffer_callback
except ImportError:
    frost = None

DIMS = (64, 128)  # the head dims the kernels take
SM = ("10.0", "10.3")  # the GPUs they run on: B200, B300
FLOOR = -5.0  # the least log decay a position may have, for KDA's kernel to run
TILE = 16  # positions per tile of KDA's kernel, from each sequence's start
# Documents a row of 4096 tokens may hold for the kernels: an empty sequence costs as
# much as 10 tokens, so cu_seqlens has room for this many, and more go by `otherwise`.
DOCS = 32

Rule = Callable[..., Float[Array, "b t h dv"]]  # linear_attention's delta_rule


def runs(dk: int, dv: int, dtype: Any) -> bool:
    """Return whether a kernel runs a layer of these head dims, its q, k and v of
    `dtype`, by the run's methods, here: 16-bit values only."""
    on = parallel.active()
    method = on.methods.recurrence if on else "scan"
    shapes = dk in DIMS and dv in DIMS and dtype in (jnp.bfloat16, jnp.float16)
    here = kernels.gpu() and _sm()
    return frost is not None and method == "scan" and shapes and here


@functools.cache
def _sm() -> bool:
    """Return whether the default device is one the kernels run on."""
    device = jax.devices()[0]
    return getattr(device, "compute_capability", "") in SM


def delta_rule(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    log_decay: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    docs: Segments | None,
    floor: float | None,
    otherwise: Rule,
) -> Float[Array, "b t h dv"]:
    """Run `linear_attention.delta_rule` by a kernel, in v's dtype, where `runs`; by
    `otherwise` for rows of more documents than cu_seqlens holds, and for a decay per
    channel steeper than KDA's kernel takes, where `floor`, the least log decay, does
    not bound it."""
    docs = docs or Segments.of(jnp.broadcast_to(jnp.arange(q.shape[1]), q.shape[:2]))
    xs = (q, k, v, log_decay, beta, *docs)
    bounded = log_decay.shape[-1] == 1 or (floor is not None and floor >= FLOOR)
    rule = functools.partial(_rows, otherwise, bounded)
    if parallel.active() is None:
        return rule(*xs)
    rows = parallel.spec(("batch", "seq", "heads", None), q.shape)
    if rows[1] is not None:  # positions split over devices, which XLA gathers
        return otherwise(q, k, v, log_decay, beta, docs)
    specs = (rows,) * 4 + (P(*rows[:3]),) + (P(*rows[:2]),) * 3
    # Untyped by what varies, which the kernels' outputs do not say; nothing here
    # communicates.
    return jax.shard_map(rule, in_specs=specs, out_specs=rows, check_vma=False)(*xs)


def _rows(
    otherwise: Rule,
    bounded: bool,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    positions: Int[Array, "b t"],
    ids: Int[Array, "b t"],
    within: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Run one device's rows by a kernel if cu_seqlens holds their documents; with
    gates the kernel may not take, unless more than `STEEP` of the (row, head) pairs
    are too steep for it."""
    b, t, h, _ = q.shape
    room = b * max(1, t * DOCS // 4096)
    fits = jnp.sum(within == 0) <= room
    docs = Segments(positions, ids, within)
    kernel = functools.partial(_kernel, room)
    if not bounded:
        steep = _steep(g)
        fits &= jnp.sum(steep) <= _patches(b, h)
        kernel = functools.partial(_patched, kernel, otherwise, docs, steep)

    def fallback(
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b t h dk"],
        v: Float[Array, "b t h dv"],
        g: Float[Array, "b t h #dk"],
        beta: Float[Array, "b t h"],
        within: Int[Array, "b t"],
    ) -> Float[Array, "b t h dv"]:
        return otherwise(q, k, v, g, beta, docs).astype(v.dtype)

    return jax.lax.cond(fits, kernel, fallback, q, k, v, g, beta, within)


STEEP = 1 / 4  # the share of (row, head) pairs the chunks may run for KDA's kernel


def _patches(b: int, h: int) -> int:
    """Return how many (row, head) pairs of b rows of h heads `_patched` runs by the
    chunks."""
    return max(1, int(b * h * STEEP))


def _steep(g: Float[Array, "b t h dk"]) -> Bool[Array, "b h"]:
    """Return the (row, head) pairs where `TILE` positions decay a channel by more
    than exp(TILE * FLOOR), which a tile of KDA's kernel may not: a log decay of -5 a
    position on average, where -5.4 loses precision and -5.6 overflows."""
    window, strides = (1, TILE, 1, 1), (1, 1, 1, 1)
    edges = ((0, 0), (TILE - 1, 0), (0, 0), (0, 0))
    g = g.astype(jnp.float32)
    sums = jax.lax.reduce_window(g, 0.0, jax.lax.add, window, strides, edges)
    return jnp.min(sums, axis=(1, 3)) < TILE * FLOOR


def _patched(
    kernel: Rule,
    otherwise: Rule,
    docs: Segments,
    steep: Bool[Array, "b h"],
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h dk"],
    beta: Float[Array, "b t h"],
    within: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Run the kernel on every (row, head) pair, the `steep` ones with no decay, and
    replace those by `otherwise`'s chunks, run on them alone. Slots past the steep
    pairs run the first pair again, and add nothing."""
    b, _, h, _ = q.shape
    flat = jnp.nonzero(steep.reshape(-1), size=_patches(b, h), fill_value=0)[0]
    rows, heads = flat // h, flat % h
    used = jnp.arange(flat.size) < jnp.sum(steep)

    def picked(x: Float[Array, "b t h ..."]) -> Float[Array, "n t 1 ..."]:
        return x[rows, :, heads][:, :, None]

    safe = jnp.where(steep[:, None, :, None], 0.0, g)
    out = jnp.where(steep[:, None, :, None], 0.0, kernel(q, k, v, safe, beta, within))
    few = Segments(*(x[rows] for x in docs))
    patch = otherwise(*map(picked, (q, k, v, g, beta)), few)[:, :, 0]
    return out.at[rows, :, heads].add(jnp.where(used[:, None, None], patch, 0.0))


Rows = tuple[Array, ...]  # q, k, v, the log decay and beta (b, t, h, ...), and within
Tokens = tuple[Array, ...]  # the same packed end to end (THD), and cu_seqlens


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _kernel(
    room: int,
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    within: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Run a kernel on the rows packed end to end, each document a sequence, in
    `room` sequences, the last ones empty: GDN's for one decay a head, else KDA's.
    Its backward pass keeps the rows as they come, of the shapes the chunks keep
    them in, so that the two branches of `_rows`' cond share that memory."""
    return _kernel_forward(room, q, k, v, g, beta, within)[0]


def _kernel_forward(room: int, *rows: Array) -> tuple[Float[Array, "b t h dv"], Rows]:
    v = rows[2]
    return _run(*_packed(room, *rows)).reshape(v.shape), rows


def _kernel_backward(
    room: int, rows: Rows, d_out: Float[Array, "b t h dv"]
) -> tuple[Array | None, ...]:
    tokens = _packed(room, *rows)
    grads = _grads(*tokens, d_out.reshape(tokens[2].shape).astype(tokens[0].dtype))
    return (*(d.reshape(x.shape).astype(x.dtype) for d, x in zip(grads, rows)), None)


_kernel.defvjp(_kernel_forward, _kernel_backward)


def _packed(room: int, *rows: Array) -> Tokens:
    """Return the rows packed end to end, q, k and v in v's dtype, the log decays and
    steps in float32, and the documents' bounds cu_seqlens, `room` + 1 of them."""
    q, k, v, g, beta, within = rows
    b, t, h, _ = q.shape
    q, k, v = (x.reshape(b * t, *x.shape[2:]).astype(v.dtype) for x in (q, k, v))
    starts = jnp.nonzero(within.reshape(-1) == 0, size=room, fill_value=b * t)[0]
    cu = jnp.append(starts, b * t).astype(jnp.int32)
    gates = g.reshape(b * t, h, -1).astype(jnp.float32)
    gates = gates[..., 0] if gates.shape[-1] == 1 else gates
    return q, k, v, gates, beta.reshape(b * t, h).astype(jnp.float32), cu


def _run(*tokens: Array) -> Array:
    """Return the outputs of the packed tokens by GDN's kernel for one decay a head
    (gates of n by h), else KDA's."""
    if tokens[3].ndim == 2:
        return _graph_call("gdn", [_like(tokens[2])], *tokens)[0]
    assert frost is not None
    return frost.kimi_delta_attention_fwd(*tokens, scale=1.0)[0]


def _grads(*tokens: Array) -> tuple[Array, ...]:
    """Return the gradients of q, k, v, the log decays and the steps of the packed
    tokens, of which the last is the outputs': by the kernels' backward passes,
    which compute the chunks' states again."""
    *xs, d_out = tokens
    if xs[3].ndim == 2:
        return _graph_call("gdn_bwd", [_like(x) for x in xs[:5]], *xs, d_out)
    assert frost is not None
    saved = frost.KdaResidual((*xs, None, None, None), None, frost.KdaConfig(scale=1.0))
    return tuple(frost.kimi_delta_attention_bwd(saved, d_out)[:5])


def _like(x: Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


# --- Gated DeltaNet: Frost's GDN kernel by cuDNN's graph, on XLA's stream -------------


def _graph_call(
    op: str, results: list[jax.ShapeDtypeStruct], *xs: Array
) -> tuple[Array, ...]:
    """Run cuDNN's graph of the node `op` on `xs`, as a callback that XLA hands its
    buffers and its stream; the graph's workspace is one more of its results."""
    shapes = tuple((x.shape, x.dtype.name) for x in xs)
    with _ONE_AT_A_TIME:
        workspace = _graph(op, shapes, 0).get_workspace_size()
    scratch = jax.ShapeDtypeStruct((max(workspace, 1),), jnp.uint8)

    def run(ctx: Any, outs: tuple[Any, ...], *ins: Any) -> None:
        device = ins[0].__dlpack_device__()[1]
        first = (op, shapes, device) not in _RAN
        *outputs, workspace = outs
        with _ONE_AT_A_TIME if first else contextlib.nullcontext():
            graph = _graph(op, shapes, device)
            pack = dict(zip(graph.ports, (*ins, *outputs), strict=True))
            handle = _handle(device, ctx.stream)
            graph.execute(pack, workspace=workspace, handle=handle)
        _RAN.add((op, shapes, device))

    return tuple(buffer_callback(run, (*results, scratch))(*xs)[:-1])


# XLA calls each device's callback from that device's thread. A graph's first run
# builds it and compiles its kernels, which two threads must not do at once (8 GPUs
# crashed so); later runs only launch them.
_ONE_AT_A_TIME = threading.Lock()
_RAN: set[tuple] = set()  # the graphs that have run once


# Each graph's inputs, and its outputs with the input whose dtype each takes, in the
# order `_graph_call` passes their buffers.
INPUTS = ("q", "k", "v", "g", "beta", "cu_seqlens")
NODES = {
    "gdn": (INPUTS, {"O": "v"}),
    "gdn_bwd": (
        (*INPUTS, "dO"),
        {"dQ": "q", "dK": "k", "dV": "v", "dG": "g", "dBeta": "beta"},
    ),
}


@functools.cache
def _graph(
    op: str, shapes: tuple[tuple[tuple[int, ...], str], ...], device: int
) -> Any:
    """Return cuDNN's graph of one node `op` of inputs of these shapes and dtypes,
    built for `device`, q unscaled, with its tensors in `_graph_call`'s order as
    `ports`."""
    inputs, outputs = NODES[op]
    graph = cudnn.pygraph()
    tensors = {
        name: graph.tensor(list(shape), data_type=_cudnn_type(dtype), name=name)
        for name, (shape, dtype) in zip(inputs, shapes, strict=True)
    }
    made = getattr(graph, op)(**tensors, scale=1.0, name=op)
    for (name, like), tensor in zip(outputs.items(), made, strict=False):
        tensor.set_output(True).set_data_type(tensors[like].get_data_type())
        tensors[name] = tensor
    with device_context(device):
        graph.build()
    graph.ports = [tensors[name] for name in (*inputs, *outputs)]
    return graph


def _cudnn_type(dtype: str) -> Any:
    """Return cuDNN's data type of a numpy dtype's name."""
    names = {"bfloat16": "BFLOAT16", "float16": "HALF", "float32": "FLOAT"}
    return getattr(cudnn.data_type, names.get(dtype, dtype.upper()))


@functools.cache
def _handles(device: int) -> Any:
    """Return a cuDNN handle of `device`."""
    with device_context(device):
        return cudnn.create_handle()


def _handle(device: int, stream: int) -> Any:
    """Return `device`'s cuDNN handle, set to run on `stream`."""
    handle = _handles(device)
    cudnn.set_stream(handle=handle, stream=stream)
    return handle
