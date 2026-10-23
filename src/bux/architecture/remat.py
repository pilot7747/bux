"""Rematerialization: the methods of `methods.remat`, and the calls they apply to.

A method is a jax.checkpoint policy: what the forward pass keeps for the backward one,
which computes the rest again. `none` keeps everything, so nothing is redone, but in a
scanned pass, which keeps all but what elementwise ops make (``unfused``). Each
residual branch outside any other is rematerialized (``rematted``), and so is each
scanned pass as a whole; nothing inside either is rematerialized again.

Layers name what is worth keeping (``checkpoint_name``), for the policies that keep
it by name: `attention`, softmax attention's output and what its backward takes
(``named``); `recurrence`, the same of the recurrent mixers; `qkv`, attention's
projections; `mlp`, an MLP's up and gate projections; and `experts`, the MoE's.
"""

import contextlib
import contextvars
from collections.abc import Callable, Generator
from typing import Any

import jax
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.extend.core import Primitive

from bux import parallel
from bux.architecture import pure
from bux.architecture.registry import Stream, Taps
from bux.parallel import methods

methods.remat.register("none")(jax.checkpoint_policies.everything_saveable)
methods.remat.register("full")(jax.checkpoint_policies.nothing_saveable)
methods.remat.register("dots")(jax.checkpoint_policies.dots_saveable)
methods.remat.register("dots_no_batch")(
    jax.checkpoint_policies.dots_with_no_batch_dims_saveable
)

MIXERS = ("attention", "recurrence")  # what the *_attention methods keep
_names = jax.checkpoint_policies.save_only_these_names
methods.remat.register("save_attention")(_names(*MIXERS))
methods.remat.register("save_qkv_attention")(_names(*MIXERS, "qkv"))
methods.remat.register("dots_attention")(
    jax.checkpoint_policies.save_from_both_policies(
        jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        _names(*MIXERS, "experts"),
    )
)
methods.remat.register("offload_attention")(
    jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=[],
        names_which_can_be_offloaded=list(MIXERS),
        offload_src="device",
        offload_dst="pinned_host",
    )
)

# Set while a rematerialized call traces, so that what it calls is not again.
_REMATTED = contextvars.ContextVar("rematted", default=False)
# Set while a call traces under a policy of the run's: what `named` names, only it
# keeps. Under `unfused`, naming would keep what it makes the backward compute again.
_NAMING = contextvars.ContextVar("naming", default=False)

Policy = Callable[..., bool]


def policy() -> Policy | None:
    """Return the run's remat policy; None where nothing rematerializes: with `none`,
    or inside a call that does already."""
    found = parallel.method("remat")
    none = found is jax.checkpoint_policies.everything_saveable
    return None if none or _REMATTED.get() else found


@contextlib.contextmanager
def off() -> Generator[None]:
    """Rematerialize nothing while the block traces: inside a call that rematerializes
    already, or in a trial pass that never runs."""
    token = _REMATTED.set(True)
    try:
        yield
    finally:
        _REMATTED.reset(token)


# Elementwise ops and those that only move or retype data: XLA fuses them into the ops
# that read them, so unrolled layers never keep what they make for the backward.
CHEAP = frozenset(
    """abs add add_any and clamp cos div eq erf exp exp2 expm1 ge gt integer_pow
    is_finite le log log1p logistic lt max min mul ne neg not or pow rsqrt select_n
    sign sin sqrt square sub tanh xor
    broadcast_in_dim concatenate convert_element_type copy copy_p dynamic_slice
    expand_dims iota pad reduce_precision reshape rev slice squeeze transpose""".split()  # noqa: SIM905
)


def unfused(prim: Primitive, *_: object, **params: object) -> bool:
    """The policy of a scanned pass where the run has none: keep all but what `CHEAP`
    ops make, and a sharding constraint on one device (on more, it may communicate),
    which the backward computes again. A scan keeps every value its backward reads,
    stacked by pass, where XLA would fuse those into their readers."""
    if prim.name == "sharding_constraint":
        return getattr(params["sharding"], "num_devices", 2) > 1
    return prim.name not in CHEAP


def named[R](name: str, f: Callable[..., R], *args: Any) -> R:
    """Return f(*args), naming its output and the residuals its backward takes
    `name`, inside a call the run's policy rematerializes: one that keeps the name computes
    none of f again. A kernel's residuals, as flash attention's log-sum-exp, are no
    products of matrices, so `dots` computes such a kernel again whole. `f` closes
    over no array but constants: jax differentiates it by `args` alone."""
    if not _NAMING.get():
        return f(*args)
    call = jax.custom_vjp(f)

    def forward(*args: Any) -> tuple[R, Any]:
        out, pull = jax.vjp(f, *args)
        return jax.tree.map(lambda a: checkpoint_name(a, name), (out, pull))

    call.defvjp(forward, lambda pull, cotangent: pull(cotangent))
    return call(*args)


def checkpoint[**A, R](
    f: Callable[A, R], prevent_cse: bool, scanned: bool = False
) -> Callable[A, R]:
    """Return `f` rematerialized by the run's remat policy, or if `scanned` by
    `unfused` where there is none; `f` itself otherwise."""
    found = policy()
    if found is None and scanned and not _REMATTED.get():
        found = unfused
    if found is None:
        return f

    def inner(*args: A.args, **kwargs: A.kwargs) -> R:
        token = _NAMING.set(found is not unfused)
        try:
            with off():
                return f(*args, **kwargs)
        finally:
            _NAMING.reset(token)

    return jax.checkpoint(inner, policy=found, prevent_cse=prevent_cse)


def rematted(module: nnx.Module, x: Stream, taps: Taps) -> Stream:
    """Call `module(x, taps)`, rematerialized by the run's remat policy."""
    if policy() is None:
        return module(x, taps)
    graphdef, state = nnx.split(module)
    call = pure.Pure(graphdef, taps)
    y, rest, changes, losses = checkpoint(call, prevent_cse=True)(
        state, x, pure.NOTHING
    )
    nnx.update(module, rest)
    pure.apply(taps, changes, call, [losses])
    return y
