"""Every part of JAX's private or unstable API that bux uses, and of tokamax's and
jax-triton's: the only module that imports or touches them.

Each entry says what bux uses it for and what breaks if it changes. They held on
the versions of `VERIFIED_ON`, and ``tests/test_jax_internals.py`` fails on any
other: it checks each entry by what bux relies on, on a CPU, and on a GPU those that
need one. After an upgrade, run it (on a GPU node too) and ``scripts/gpu_smoke/*.py``,
fix here what fails, and set `VERIFIED_ON` to the new versions.

The entries, by what uses them:

- Jaxprs (`calls_a_jaxpr`, `scan_body`, `bind`, ...): `bux.parallel.split` takes a
  backward apart by them, and `bux.train.flops` counts them.
- A vjp rebuilt from its residuals (`rebuild_vjp`): `bux.parallel.scheduled` keeps
  them a microbatch apart.
- Kernels: cuDNN's attention and scaled matmuls, XLA's buffers and streams for
  custom calls (`buffer_callback`, `on_stream`), jax-triton's and Pallas's
  lowering, and tokamax's flash attention.
- jax.distributed's processes (`processes`, `key_value_store`): `bux.runtime`,
  before anything starts a backend.
- XLA's scheduled HLO (`hlo_module`, `key_compiles_by`): `bux.passes` and
  `bux.train.reorder`, which changes a step's schedule.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax._src import cache_key, distributed
from jax._src.api import VJP, NotNeeded, NotSaveable
from jax._src.cudnn import fused_attention_stablehlo as cudnn_fused
from jax._src.cudnn.scaled_matmul_stablehlo import _scaled_matmul_p
from jax._src.lib import hlo
from jax.experimental import buffer_callback as callbacks
from jax.experimental.xla_metadata import xla_metadata_call2
from jax.extend import core
from jaxtyping import Array, Float, Int, Shaped

#: The packages whose internals below were checked, at these versions.
VERIFIED_ON = {
    "jax": "0.12.0",
    "jaxlib": "0.12.0",
    "jax-triton": "0.4.1",
    "tokamax": "0.0.17",
    "nvidia-cudnn-frontend": "1.31.0",
}


# --- Jaxprs: split.py takes a backward apart by them, flops.py counts them --------
#
# `core` is jax.extend.core: Jaxpr, ClosedJaxpr, JaxprEqn, Literal. A param renamed
# fails loudly where it is read; a primitive renamed only stops matching, which
# split.py and flops.py would miss in silence.

#: What a remat policy is called with, by each equation.
Primitive = core.Primitive
#: The primitives of jax.jit and jax.checkpoint, which call a jaxpr.
CALLS = frozenset({"jit", "remat2"})


def calls_a_jaxpr(eqn: core.JaxprEqn) -> bool:
    """Whether `eqn` calls a jaxpr, as jit and checkpoint do. A renamed one is not
    inlined, and split.py sends it whole to one part of the backward."""
    return eqn.primitive.name in CALLS


def called_jaxpr(eqn: core.JaxprEqn) -> core.ClosedJaxpr:
    """Return the jaxpr that the call `eqn` runs, which split.py inlines."""
    inner = eqn.params["jaxpr"]
    return inner if isinstance(inner, core.ClosedJaxpr) else core.ClosedJaxpr(inner, ())


def is_scan(eqn: core.JaxprEqn) -> bool:
    """Whether `eqn` is a lax.scan, which split.py splits in two (else the input's
    part runs it whole) and flops.py counts by its length."""
    return eqn.primitive.name == "scan"


def is_cond(eqn: core.JaxprEqn) -> bool:
    """Whether `eqn` is a lax.cond (or lax.switch), which split.py splits branch by
    branch (else the input's part runs it whole)."""
    return eqn.primitive.name == "cond"


def scan_body(eqn: core.JaxprEqn) -> core.ClosedJaxpr:
    """Return the body of the scan `eqn`: of its constants, carries and one pass's
    inputs, to its carries and one pass's outputs."""
    return eqn.params["jaxpr"]


def scan_counts(eqn: core.JaxprEqn) -> tuple[int, int]:
    """Return how many constants and carries the scan `eqn` takes: its first inputs,
    before what it scans over. Its param "ft_in" says, a tree of them (jax 0.12)."""
    consts, carry, _ = eqn.params["ft_in"].unpack()
    return len(consts), len(carry)


def scan_options(eqn: core.JaxprEqn) -> dict[str, Any]:
    """Return the scan `eqn`'s length, direction and unrolling, as lax.scan takes
    them: split.py runs each part's scan so."""
    p = eqn.params
    return {"length": p["length"], "reverse": p["reverse"], "unroll": p["unroll"]}


def cond_branches(eqn: core.JaxprEqn) -> Sequence[core.ClosedJaxpr]:
    """Return the branches of the cond `eqn`, by the index it takes first."""
    return eqn.params["branches"]


def bind(eqn: core.JaxprEqn, ins: Sequence[Any]) -> list[Any]:
    """Run the equation `eqn` on `ins`, under its context (its mesh, its dtype of
    compute), as jax.core.eval_jaxpr does: its outputs, as a list. split.py runs
    each part of a backward so."""
    params = eqn.primitive.get_bind_params(eqn.params)
    with eqn.ctx.manager:
        out = eqn.primitive.bind(*ins, **params)
    return out if eqn.primitive.multiple_results else [out]


# --- A vjp rebuilt from its residuals: scheduled.py keeps them a microbatch apart --
#
# `VJP` is what jax.vjp returns, a pytree of its residuals.


def rebuild_vjp(
    like: Callable[..., Any], residuals: Sequence[Any], unsaved: Any
) -> VJP:
    """Return the vjp whose structure `like` has, from its residuals' leaves: a
    jax.vjp is a pytree of its residuals. Its first primal, which
    jax.vjp(..., saveable_args=(False, ...)) did not save, goes back in from
    `unsaved`, where the vjp's `args_res` marks a leaf NotSaveable; NotNeeded marks
    one its backward does not read."""
    vjp = jax.tree.unflatten(jax.tree.structure(like), residuals)

    def marker(v: object) -> bool:
        return isinstance(v, NotSaveable | NotNeeded)

    held, tree = jax.tree.flatten(vjp.args_res[0], is_leaf=marker)
    values = [
        u if isinstance(h, NotSaveable) else h
        for h, u in zip(held, jax.tree.leaves(unsaved), strict=True)
    ]
    return vjp.replace(args_res=[jax.tree.unflatten(tree, values), *vjp.args_res[1:]])


# --- Kernels: cuDNN's, Triton's and tokamax's ------------------------------------


def scaled_matmul_unpartitioned(
    a: Shaped[Array, "1 m k"],
    b: Shaped[Array, "1 n k"],
    a_scales: Shaped[Array, "1 m kb"],
    b_scales: Shaped[Array, "1 n kb"],
    out: Any,
) -> Float[Array, "1 m n"]:
    """Return jax.nn.scaled_matmul(a, b, a_scales, b_scales) in `out` by its GEMM's
    primitive, `_scaled_matmul_p`, bound as it is: jax.nn.scaled_matmul's custom
    partitioning cannot place it inside a shard_map (blockscaled.py). Its name,
    "scaled_matmul", is one of remat.py's products of matrices."""
    args = (a, b, a_scales, b_scales)
    return _scaled_matmul_p.bind(*args, preferred_element_type=jnp.dtype(out))[0]


def cudnn_attention_here() -> bool:
    """Whether JAX's cuDNN is one its fused attention takes (flash.py asks, on a
    GPU)."""
    try:
        cudnn_fused.check_cudnn_version()
    except RuntimeError:
        return False
    return True


def cudnn_attention(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    lengths: Int[Array, "b m"] | None,
    starts: Int[Array, "b m1"] | None,
    scale: float,
    window: int | None,
) -> Float[Array, "b t h dv"]:
    """Return cuDNN's causal attention, within `window`, and with `lengths` and
    `starts` of each row's documents each by itself (cuDNN's packed layout): by
    JAX's own call of it, as jax.nn.dot_product_attention takes no packed layout."""
    masks = cudnn_fused.MaskType
    mask = masks.CAUSAL if lengths is None else masks.PADDING_CAUSAL
    return cudnn_fused.dot_product_attention(
        q,
        k,
        v,
        q_seqlen=lengths,
        kv_seqlen=lengths,
        q_offsets=starts,
        kv_offsets=starts,
        scale=scale,
        mask_type=mask,
        sliding_window_length=window,
    )


def cudnn_attention_and_lse(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    scale: float,
) -> tuple[Float[Array, "b t h dv"], Float[Array, "b h t"]]:
    """Return cuDNN's causal attention of whole rows, and each query's logsumexp in
    float32, which its training forward keeps for its backward: by JAX's own call of
    it (``return_residual``), whose backward ignores the logsumexp's cotangent."""
    return cudnn_fused.dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask_type=cudnn_fused.MaskType.CAUSAL,
        return_residual=True,
    )


def cudnn_attention_backward(
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b t g dk"],
    v: Float[Array, "b t g dv"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    lse: Float[Array, "b h t"],
    scale: float,
) -> tuple[
    Float[Array, "b t h dk"], Float[Array, "b t g dk"], Float[Array, "b t g dv"]
]:
    """Return the gradients of q, k and v of cuDNN's causal attention of whole rows,
    from its output and each query's logsumexp: by the primitive of JAX's backward
    (``_dot_product_attention_bwd_rule``), which takes what its forward kept."""
    unused = jnp.zeros(0, q.dtype)  # no bias, documents or pages
    grads = cudnn_fused._dot_product_attention_bwd_p_wrapper.bind(
        q,
        k,
        v,
        *[unused] * 7,
        lse,
        out,
        dout,
        scale=scale,
        seed=42,
        dropout_rate=0.0,
        variadic_args=(False, False),
        mask_type=cudnn_fused.MaskType.CAUSAL,
        layout=cudnn_fused.AttentionLayout.BTNH.value,
        sliding_window_length=None,
        score_mod=None,
    )
    return grads[0], grads[1], grads[2]


def buffer_callback(
    run: Callable[..., None], results: Sequence[jax.ShapeDtypeStruct]
) -> Callable[..., tuple[Shaped[Array, "..."], ...]]:
    """Return a function of arrays that calls `run(context, outs, *ins)` with XLA's
    buffers of them, to write the `outs` of `results`' shapes in place; on a GPU,
    `context.stream` is the stream XLA runs on, and `ins[0].__dlpack_device__()` the
    device. fused_delta.py runs cuDNN's graphs of the delta rule so."""
    return callbacks.buffer_callback(run, results)


def on_stream(stream: str, run: Callable[[], Any], first: int = 0) -> Any:
    """Return run(), its ops on the GPU stream `stream` ("gpu_stream:1"): XLA makes
    them one async call there, which the compute stream's kernels run beside, by
    --xla_gpu_experimental_stream_annotation (on by default). A CPU ignores it.
    nccl_ep.py runs NCCL EP's kernels so.

    XLA's scheduler starts the call as soon as its inputs are ready and waits for it
    as late as its results are needed, so that whatever else is ready runs beside it
    (the hint force_delay_async); of such calls ready at once, a higher `first`
    starts first (scheduler_delay_priority). jax's compute_on takes no hints. In a
    shard_map its results come off the stream as they are only where XLA runs
    ``parts_off_the_stream`` before its scheduler, as nccl_ep.py has it."""
    hinted = {
        "_xla_stream_annotation": stream.removeprefix("gpu_stream:"),
        "inlineable": "false",
        "scheduler_hint": "force_delay_async",
        "scheduler_delay_priority": str(first),
    }
    return xla_metadata_call2(run, hinted)()


# What marks an instruction for a GPU stream, and a module that ``parts_off_the_stream``
# changes.
STREAM_ATTRIBUTE = b"_xla_stream_annotation"
# Field numbers of xla/service/hlo.proto: HloModuleProto.computations,
# HloComputationProto.instructions, HloInstructionProto.opcode and
# .frontend_attributes, and the entries of FrontendAttributes.map.
COMPUTATIONS, INSTRUCTIONS, OPCODE, ATTRIBUTES, ENTRIES = 3, 2, 2, 68, 1
LENGTH = 2  # protobuf's wire type of a length-delimited field


def parts_off_the_stream(serialized: bytes) -> bytes | None:
    """Return the module `serialized` (an HloModuleProto, before XLA schedules it)
    with no get-tuple-element marked for a GPU stream; None where none is. In a
    shard_map XLA's SPMD partitioner copies a stream call's attributes onto the
    get-tuple-elements of its results, and XLA then makes each its own async call on
    the stream, queued there behind the calls issued after it: the compute stream
    would wait for a chunk's next dispatch before it takes this one's rows."""
    if STREAM_ATTRIBUTE not in serialized:
        return None
    edited = _edited(serialized, {COMPUTATIONS: _computation})
    return None if edited == serialized else edited


def _computation(message: bytes) -> bytes:
    if STREAM_ATTRIBUTE not in message:
        return message
    return _edited(message, {INSTRUCTIONS: _instruction})


def _instruction(message: bytes) -> bytes:
    if STREAM_ATTRIBUTE not in message or _opcode(message) != b"get-tuple-element":
        return message
    return _edited(message, {ATTRIBUTES: _attributes})


def _attributes(message: bytes) -> bytes:
    return _edited(message, {ENTRIES: _unless_stream})


def _unless_stream(entry: bytes) -> bytes | None:
    """Return an entry of a frontend attributes' map, or None for the stream's."""
    key = next(payload for number, _, payload in _fields(entry) if number == 1)
    return None if key == STREAM_ATTRIBUTE else entry


def _opcode(instruction: bytes) -> bytes | None:
    return next((p for n, _, p in _fields(instruction) if n == OPCODE), None)


Edit = Callable[[bytes], bytes | None]


def _edited(message: bytes, edits: dict[int, Edit]) -> bytes:
    """Return the protobuf `message` with each length-delimited field whose number
    has an edit replaced by the edit of its payload, or dropped where it gives None;
    the other fields as they are."""
    out = bytearray()
    for number, raw, payload in _fields(message):
        edit = edits.get(number)
        if edit is None or payload is None:
            out += raw
        elif (new := edit(payload)) is not None:
            out += _varint(number << 3 | LENGTH) + _varint(len(new)) + new
    return bytes(out)


def _fields(message: bytes) -> Iterator[tuple[int, bytes, bytes | None]]:
    """Yield each field of a protobuf message: its number, its bytes, and its
    payload where it is length-delimited."""
    at = 0
    while at < len(message):
        start = at
        tag, at = _read_varint(message, at)
        kind, payload = tag & 7, None
        if kind == 0:
            _, at = _read_varint(message, at)
        elif kind == LENGTH:
            size, at = _read_varint(message, at)
            payload, at = message[at : at + size], at + size
        else:  # 64 or 32 bits
            at += {1: 8, 5: 4}[kind]
        yield tag >> 3, message[start:at], payload


def _read_varint(data: bytes, at: int) -> tuple[int, int]:
    """Return the varint at `at` in data, and where it ends."""
    value = shift = 0
    while True:
        byte = data[at]
        value, shift, at = value | (byte & 0x7F) << shift, shift + 7, at + 1
        if byte < 0x80:
            return value, at


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _keeps_its_inputs(
    used: list[bool], live: list[bool], eqn: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn]:
    """Dead-code elimination's rule of a Triton call: keep it whole."""
    return [True] * len(eqn.invars), eqn


def keep_triton_calls_whole() -> None:
    """Have dead-code elimination keep jax-triton's calls whole: jax-triton 0.4.1
    registers a rule of jax 0.11's signature, which jax 0.12 calls with one more
    argument, so any DCE of a jaxpr with one (a scan's, remat's) fails. Called by
    grouped_triton.py, where jax-triton is installed."""
    import jax_triton
    from jax._src.interpreters import partial_eval

    partial_eval.dce_rules[jax_triton.triton_lib.triton_kernel_call_p] = (
        _keeps_its_inputs
    )


def relax_pallas_atomics() -> None:
    """Have Pallas's Triton lowering make its atomic adds relaxed, as CUDA's atomicAdd
    is: it asks acquire-release of every one (lowering._atomic_rmw's default), which
    sums need no more than atomicity of, and on a B300 the indexer's loss's backward
    kernel (scoring.py), whose adds are most of it, took 1.6x the time."""
    from jax._src.lib.triton import dialect
    from jax._src.pallas.triton import lowering

    defaults = lowering._atomic_rmw.__defaults__
    assert defaults is not None
    mask, _, scope = defaults
    relaxed = dialect.MemSemantic.RELAXED
    lowering._atomic_rmw.__defaults__ = (mask, relaxed, scope)


def tokamax_attention(implementation: str) -> Any:
    """Return tokamax's flash attention op of `implementation` ("mosaic_gpu",
    "triton"), None without tokamax. Called with return_residuals=True, it gives
    the output and each row's largest logit and sum of exps (flash.py, sparse.py)."""
    try:  # tokamax, of the cuda extra (Linux)
        from tokamax._src.ops.attention import api
    except ImportError:
        return None
    return api.IMPLEMENTATIONS.get(implementation)


def tokamax_mask(*args: Any, **kwargs: Any) -> Any:
    """Return tokamax's attention mask: a boolean mask, if any, then k_start, k_end
    (each query's first key and the key after its last) and is_causal."""
    from tokamax._src.ops.attention import base

    return base.Mask(*args, **kwargs)


def tokamax_vjp(
    op: Any,
    lse: Float[Array, "b h t"],
    out: Float[Array, "b t h dv"],
    dout: Float[Array, "b t h dv"],
    q: Float[Array, "b t h dk"],
    k: Float[Array, "b s g dk"],
    v: Float[Array, "b s g dv"],
    args: dict[str, Any],
) -> tuple[
    Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
]:
    """Return the gradients of q, k and v by tokamax's backward kernels of `op`,
    from the forward's `out` and each row's logsumexp `lse`, as residuals (lse, 1):
    P = exp(s - lse). `args` are the forward's keyword arguments."""
    bound = op.bind(q, k, v, **args)
    kwargs = {n: a for n, a in bound.kwargs.items() if n != "return_residuals"}
    grads = op.vjp((lse, jnp.ones_like(lse)), out, dout, *bound.args, **kwargs)
    return grads["q"], grads["k"], grads["v"]


# --- jax.distributed's processes: runtime.py, before anything starts a backend ----


def processes() -> tuple[int, int]:
    """Return this process's index and how many joined the job, (0, 1) before
    jax.distributed.initialize: from its state, which jax.process_index() would
    read only after starting the backends."""
    state = distributed.global_state
    return state.process_id or 0, state.num_processes or 1


def key_value_store() -> Any:
    """Return jax.distributed's client, None before the processes join: its
    key_value_set and blocking_key_value_get pass strings between them."""
    return distributed.global_state.client


# --- XLA's scheduled HLO: reorder.py changes a step's schedule ---------------------
#
# jaxlib's bindings of an HloModule: its schedule() and make_nonfusion_computations();
# a schedule's sequence(computation) and set_sequence; an instruction's name, opcode,
# operands(), users(), control_predecessors() and async_wrapped_root(). A method
# renamed fails as the pass runs, which then leaves XLA's schedule as it was.

#: jaxlib's enum of HLO opcodes, which an instruction's ``opcode`` is.
Opcode = hlo.HloOpcode


def hlo_module(serialized: bytes) -> Any:
    """Return the HloModule of a serialized HloModuleProto, as XLA hands a pass of
    jax.extend.xla; its as_serialized_hlo_module_proto() gives one back."""
    return hlo.HloModule.from_serialized_hlo_module_proto(serialized)


def hlo_module_from_text(text: str) -> Any:
    """Return the HloModule that HLO's text form `text` reads as, scheduled in the
    order of its instructions where its header says is_scheduled=true."""
    return hlo.hlo_module_from_text(text)


def key_compiles_by(text: str) -> None:
    """Make `text` part of the key of every program JAX's persistent cache keeps
    from now on (cache_key.custom_hook, which its key reads at each compile): a pass
    of jax.extend.xla changes what a program compiles to, which the key, of XLA's
    flags and options, does not see."""
    cache_key.custom_hook = lambda: text  # ty: ignore
