"""Rematerialization: the methods of `methods.remat`, and the calls they apply to.

A method is a jax.checkpoint policy: what the forward pass keeps for the backward one,
which computes the rest again. `none` keeps everything, so nothing is redone. Each
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

MIXERS = ("attention", "recurrence")
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


def named[R](name: str, f: Callable[..., R], *args: Any) -> R:
    """Return f(*args), naming its output and the residuals its backward takes
    `name`, inside a rematerialized call: a policy that keeps the name then computes
    none of f again. A kernel's residuals, as flash attention's log-sum-exp, are no
    products of matrices, so `dots` computes such a kernel again whole. `f` closes
    over no array but constants: jax differentiates it by `args` alone."""
    if not _REMATTED.get():
        return f(*args)
    call = jax.custom_vjp(f)

    def forward(*args: Any) -> tuple[R, Any]:
        out, pull = jax.vjp(f, *args)
        return jax.tree.map(lambda a: checkpoint_name(a, name), (out, pull))

    call.defvjp(forward, lambda pull, cotangent: pull(cotangent))
    return call(*args)


def checkpoint[**A, R](f: Callable[A, R], prevent_cse: bool) -> Callable[A, R]:
    """Return `f` rematerialized by the run's remat policy; `f` itself if there is none."""
    if (found := policy()) is None:
        return f

    def inner(*args: A.args, **kwargs: A.kwargs) -> R:
        with off():
            return f(*args, **kwargs)

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
