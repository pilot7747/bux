"""Every part of JAX's private or unstable API that bux uses, and of tokamax's and
jax-triton's: the only module that imports or touches them.

Each entry says what bux uses it for and what breaks if it changes. They held on
the versions of `VERIFIED_ON`, and tests/test_jax_internals.py fails on any other:
it checks each entry by what bux relies on, on a CPU, and on a GPU those that need
one. After an upgrade, run it (on a GPU node too) and scripts/gpu_smoke/*.py, fix
here what fails, and set `VERIFIED_ON` to the new versions.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax._src import distributed
from jax._src.api import VJP, NotNeeded, NotSaveable
from jax._src.cudnn import fused_attention_stablehlo as cudnn_fused
from jax._src.cudnn.scaled_matmul_stablehlo import _scaled_matmul_p
from jax.experimental import buffer_callback as callbacks
from jax.extend import core
from jaxtyping import Array, Float, Int, Shaped

# The packages whose internals below were checked, at these versions.
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

Primitive = core.Primitive  # what a remat policy is called with, by each equation
CALLS = frozenset({"jit", "remat2"})  # the primitives of jax.jit and jax.checkpoint


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
    """Return a vjp of `like`'s structure from its residuals' leaves, its first
    primal, which jax.vjp(..., saveable_args=(False, ...)) left out, put back as
    `unsaved`. A jax.vjp is a pytree of its residuals; what it did not save are
    NotSaveable in its `args_res`, and what it does not need NotNeeded."""
    vjp = jax.tree.unflatten(jax.tree.structure(like), residuals)

    def sentinel(v: object) -> bool:
        return isinstance(v, NotSaveable | NotNeeded)

    held, tree = jax.tree.flatten(vjp.args_res[0], is_leaf=sentinel)
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


def buffer_callback(
    run: Callable[..., None], results: Sequence[jax.ShapeDtypeStruct]
) -> Callable[..., tuple[Shaped[Array, "..."], ...]]:
    """Return a function of arrays that calls `run(context, outs, *ins)` with XLA's
    buffers of them, to write the `outs` of `results`' shapes in place; on a GPU,
    `context.stream` is the stream XLA runs on, and `ins[0].__dlpack_device__()` the
    device. fused_delta.py runs cuDNN's graphs of the delta rule so."""
    return callbacks.buffer_callback(run, results)


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
