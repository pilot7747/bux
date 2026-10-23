"""The gated delta rule by cuDNN's fused kernels (Frost, in cudnn-frontend, which
tokamax's cuda extra installs): forward and backward in a fifth (GDN) to a tenth (KDA)
of ``chunkwise``'s time on a B300, keeping only their inputs for the backward pass.

One decay a head (Gated DeltaNet, DeltaNet) runs by Frost's GDN kernel, which takes
each pair's decay as the exp of a difference of summed log decays, never positive: it
runs any gate. cudnn-frontend gives it no JAX call, so a `buffer_callback` runs
cuDNN's graph of it on XLA's stream (`_graph_call`).

A decay per key channel (KDA) runs by Frost's KDA kernel, through cudnn-frontend's
own JAX call. It factors each decay through the start of a tile of 16 positions, so it
loses precision where a tile decays a channel by exp(-86), and overflows at exp(-90):
it runs a gate bounded at -5 (`gate_lower_bound`, Kimi K3), and no other. Within 12
steps of bux's start, KDA's unbounded gates are that steep in 30 to 90% of the (row,
head) pairs of most layers.

The kernels run on one SM100 or SM103 GPU at a time, on token-packed rows and their
documents' bounds (cu_seqlens); on a mesh, a `shard_map` gives them each device's rows
and heads. They take heads of 64 or 128, and every position of a row: not
`methods.recurrence = pass_state`, which splits them.
"""

import functools
import threading
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, Shaped

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
FLOOR = -5.0  # the least log decay a channel may have, for KDA's kernel to run
# Documents a row of 4096 tokens may hold for the kernels: an empty sequence costs as
# much as 10 tokens, so cu_seqlens has room for this many, and more go by `otherwise`.
DOCS = 32

Rule = Callable[..., Float[Array, "b t h dv"]]  # linear_attention's delta_rule


def runs(dk: int, dv: int, dtype: Any, decays: int, floor: float | None) -> bool:
    """Return whether a kernel runs a layer of these head dims, its q, k and v of
    `dtype` and `decays` log decays a head, never below `floor`, by the run's methods,
    here: 16-bit values only, and a decay per channel bounded at `FLOOR`."""
    on = parallel.active()
    method = on.methods.recurrence if on else "scan"
    shapes = dk in DIMS and dv in DIMS and dtype in (jnp.bfloat16, jnp.float16)
    gates = decays == 1 or (floor is not None and floor >= FLOOR)
    here = kernels.gpu() and _sm()
    return frost is not None and method == "scan" and shapes and gates and here


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
    otherwise: Rule,
) -> Float[Array, "b t h dv"]:
    """Run `linear_attention.delta_rule` by a kernel, in v's dtype, where `runs`; by
    `otherwise` for rows of more documents than cu_seqlens holds."""
    docs = docs or Segments.of(jnp.broadcast_to(jnp.arange(q.shape[1]), q.shape[:2]))
    xs = (q, k, v, log_decay, beta, *docs)
    rule = functools.partial(_rows, otherwise)
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
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t h dk"],
    v: Float[Array, "b t h dv"],
    g: Float[Array, "b t h #dk"],
    beta: Float[Array, "b t h"],
    positions: Int[Array, "b t"],
    ids: Int[Array, "b t"],
    within: Int[Array, "b t"],
) -> Float[Array, "b t h dv"]:
    """Run one device's rows by a kernel if cu_seqlens holds their documents."""
    b, t = within.shape
    room = b * max(1, t * DOCS // 4096)
    fits = jnp.sum(within == 0) <= room
    docs = Segments(positions, ids, within)
    kernel = functools.partial(_kernel, room)

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


Rows = tuple[
    Shaped[Array, "..."], ...
]  # q, k, v, the log decay and beta (b, t, h, ...), and within
Tokens = tuple[
    Shaped[Array, "..."], ...
]  # the same packed end to end (THD), and cu_seqlens


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


def _kernel_forward(
    room: int, *rows: Shaped[Array, "..."]
) -> tuple[Float[Array, "b t h dv"], Rows]:
    v = rows[2]
    return _run(*_packed(room, *rows)).reshape(v.shape), rows


def _kernel_backward(
    room: int, rows: Rows, d_out: Float[Array, "b t h dv"]
) -> tuple[Shaped[Array, "..."] | None, ...]:
    tokens = _packed(room, *rows)
    grads = _grads(*tokens, d_out.reshape(tokens[2].shape).astype(tokens[0].dtype))
    return (*(d.reshape(x.shape).astype(x.dtype) for d, x in zip(grads, rows)), None)


_kernel.defvjp(_kernel_forward, _kernel_backward)


def _packed(room: int, *rows: Shaped[Array, "..."]) -> Tokens:
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


def _run(*tokens: Shaped[Array, "..."]) -> Shaped[Array, "..."]:
    """Return the outputs of the packed tokens by GDN's kernel for one decay a head
    (gates of n by h), else KDA's."""
    if tokens[3].ndim == 2:
        return _graph_call("gdn", [_like(tokens[2])], *tokens)[0]
    assert frost is not None
    return frost.kimi_delta_attention_fwd(*tokens, scale=1.0)[0]


def _grads(*tokens: Shaped[Array, "..."]) -> tuple[Shaped[Array, "..."], ...]:
    """Return the gradients of q, k, v, the log decays and the steps of the packed
    tokens, of which the last is the outputs': by the kernels' backward passes,
    which compute the chunks' states again."""
    *xs, d_out = tokens
    if xs[3].ndim == 2:
        return _graph_call("gdn_bwd", [_like(x) for x in xs[:5]], *xs, d_out)
    assert frost is not None
    saved = frost.KdaResidual((*xs, None, None, None), None, frost.KdaConfig(scale=1.0))
    return tuple(frost.kimi_delta_attention_bwd(saved, d_out)[:5])


def _like(x: Shaped[Array, "..."]) -> jax.ShapeDtypeStruct:
    """Return the shape and dtype of a result like `x`."""
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


# --- Gated DeltaNet: Frost's GDN kernel by cuDNN's graph, on XLA's stream -------------


def _graph_call(
    op: str, results: list[jax.ShapeDtypeStruct], *xs: Shaped[Array, "..."]
) -> tuple[Shaped[Array, "..."], ...]:
    """Run cuDNN's graph of the node `op` on `xs`, as a callback that XLA hands its
    buffers and its stream; the graph's workspace is one more of its results."""
    shapes = tuple((x.shape, x.dtype.name) for x in xs)
    with _ONE_AT_A_TIME:
        workspace = _graph(op, shapes, 0).get_workspace_size()
    scratch = jax.ShapeDtypeStruct((max(workspace, 1),), jnp.uint8)

    def run(ctx: Any, outs: tuple[Any, ...], *ins: Any) -> None:
        device = ins[0].__dlpack_device__()[1]
        *outputs, workspace = outs
        with _ONE_AT_A_TIME:
            graph = _graph(op, shapes, device)
            pack = dict(zip(graph.ports, (*ins, *outputs), strict=True))
            handle = _handle(device, ctx.stream)
            graph.execute(pack, workspace=workspace, handle=handle)

    return tuple(buffer_callback(run, (*results, scratch))(*xs)[:-1])


# XLA calls each device's callback from that device's thread. Two threads must not
# build and compile a graph at once (8 GPUs crashed so); and one at a time, 8 GPUs
# train faster than with every callback contending for the GIL: 828k tokens a second
# against 726k.
_ONE_AT_A_TIME = threading.Lock()


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
