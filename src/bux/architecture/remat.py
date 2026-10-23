"""Rematerialization: `methods.remat`, what the forward pass keeps for the backward
one, which computes the rest again; and the calls it applies to.

    methods = methods(remat=remat(save=[attention, mlp], first=8))

A policy keeps what it names in `save`, moves what it names in `offload` to the host's
memory until the backward reads it, and computes everything else again
(``keeping``). Each residual branch outside any other is rematerialized as a unit
(``rematted``), and so is each scanned pass; nothing inside either is again. With
`first`, only the first passes of each repeat are, and the rest keep everything
(``at_pass``). A policy is also a name in `methods.remat` (``PRESETS``): `none` keeps
everything, but in a scanned pass, which keeps all but what elementwise ops make
(``unfused``).

Layers name what is worth keeping (``checkpoint_name``):

    attention.in    softmax attention's q, k and v, as its kernel takes them (``named``)
    attention.out   its output, and what else its backward takes: the log-sum-exp
    attention.index the keys sparse attention's indexer kept for each query
    recurrence.in   a recurrent mixer's inputs, as its kernel takes them
    recurrence.out  its output, and what else its backward takes: the chunks' states
    qkv             attention's projections, MLA's latents
    mlp             an MLP's up and gate projections
    experts         an MoE's grouped products

A name covers those under it: `attention` is all three of attention's. `dots` names
every product of matrices, the experts' too, and `dots_no_batch` those without batch
dimensions.
"""

import contextlib
import contextvars
import functools
from collections.abc import Callable, Generator
from typing import Any

import jax
from flax import nnx
from jax.ad_checkpoint import Offloadable, checkpoint_name, name_p
from jax.extend.core import Primitive

from bux import parallel
from bux.architecture import pure
from bux.architecture.registry import Stream, Taps
from bux.parallel import methods
from bux.train.config import RematConfig

Policy = Callable[..., Any]  # of jax.checkpoint: True, False or an Offloadable

# The products of matrices: jax's MXFP8 GEMM is `scaled_matmul_wrapper` as
# jax.nn.scaled_matmul binds it, and `scaled_matmul` as blockscaled binds it inside a
# shard_map, which jax's own policies miss.
SCALED = frozenset({"scaled_matmul", "scaled_matmul_wrapper"})
GEMMS = SCALED | {"dot_general", "conv_general_dilated"}
DOTS = ("dots", "dots_no_batch")
OFFLOAD = Offloadable(src="device", dst="pinned_host")


def _dots(prim: Primitive, args: tuple, params: dict) -> bool:
    return prim.name in GEMMS


def _dots_no_batch(prim: Primitive, args: tuple, params: dict) -> bool:
    if prim.name == "dot_general":
        _, (lhs_batch, rhs_batch) = params["dimension_numbers"]
        return not lhs_batch and not rhs_batch
    return prim.name in SCALED and args[0].shape[0] == 1  # bux's GEMMs: a batch of 1


KINDS = {"dots": _dots, "dots_no_batch": _dots_no_batch}


def _under(name: str, scope: str) -> bool:
    """Return whether `scope` covers `name`: it is the name, or the name starts with
    it and a dot."""
    return name == scope or name.startswith(scope + ".")


def _covers(names: tuple[str, ...], prim: Primitive, args: tuple, params: dict) -> bool:
    """Return whether `names` cover what `prim` makes: a value named one of them or
    under one, or a product of matrices of a kind among them."""
    if prim is name_p:
        return any(_under(params["name"], scope) for scope in names)
    return any(KINDS[kind](prim, args, params) for kind in names if kind in KINDS)


@functools.cache
def keeping(save: tuple[str, ...] = (), offload: tuple[str, ...] = ()) -> Policy:
    """Return the policy that keeps what `save` names, moves what `offload` names to
    the host's memory, and computes the rest again. The experts' grouped products are
    products of matrices to `dots` and `dots_no_batch`."""
    if set(save) & set(DOTS):
        save = (*save, "experts")

    def policy(prim: Primitive, *args: Any, **params: Any) -> bool | Offloadable:
        if _covers(offload, prim, args, params):
            return OFFLOAD
        return _covers(save, prim, args, params)

    return policy


MIXERS = ("attention", "recurrence")  # what the *_attention presets keep
PRESETS = {
    "full": RematConfig(),
    "dots": RematConfig(save=("dots",)),
    "dots_no_batch": RematConfig(save=("dots_no_batch",)),
    "save_attention": RematConfig(save=MIXERS),
    "save_qkv_attention": RematConfig(save=(*MIXERS, "qkv")),
    "dots_attention": RematConfig(save=("dots_no_batch", *MIXERS, "experts")),
    "offload_attention": RematConfig(offload=MIXERS),
}
methods.remat.register("none")(jax.checkpoint_policies.everything_saveable)
for _name, _preset in PRESETS.items():
    methods.remat.register(_name)(keeping(_preset.save, _preset.offload))

# Set while a rematerialized call traces, so that what it calls is not again.
_REMATTED = contextvars.ContextVar("rematted", default=False)
# Set while a call traces under a policy of the run's: what `named` names, only it
# keeps. Under `unfused`, naming would keep what it makes the backward compute again.
_NAMING = contextvars.ContextVar("naming", default=False)
# The pass of the outermost repeat that traces, which `first` counts; None outside.
_PASS = contextvars.ContextVar[int | None]("pass", default=None)


def setting() -> str | RematConfig:
    """Return the run's `methods.remat`, or none outside ``parallel.use``."""
    on = parallel.active()
    return on.methods.remat if on else "none"


def _found(remat: str | RematConfig) -> Policy:
    if isinstance(remat, str):
        return methods.remat.find(remat)
    return keeping(remat.save, remat.offload)


def policy() -> Policy | None:
    """Return the run's remat policy; None where nothing rematerializes: with `none`,
    inside a call that does already, or in a pass past the run's `first`."""
    remat = setting()
    found = _found(remat)
    none = found is jax.checkpoint_policies.everything_saveable
    return None if none or _REMATTED.get() or _past_first(remat) else found


def _past_first(remat: str | RematConfig) -> bool:
    first, at = getattr(remat, "first", None), _PASS.get()
    return first is not None and at is not None and at >= first


@contextlib.contextmanager
def at_pass(i: int) -> Generator[None]:
    """Trace the block as pass i of a repeat, for `first`; inside a pass of an outer
    repeat, as that pass still."""
    token = _PASS.set(i) if _PASS.get() is None else None
    try:
        yield
    finally:
        if token is not None:
            _PASS.reset(token)


def spans(start: int, stop: int) -> list[tuple[int, int]]:
    """Return passes start to stop of a repeat as runs that rematerialize alike: those
    before the run's `first`, and the rest."""
    first = getattr(setting(), "first", None)
    if first is None or not start < first < stop:
        return [(start, stop)]
    return [(start, first), (first, stop)]


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


def label[T](name: str, tree: T) -> T:
    """Return `tree`, its arrays named `name` inside a call the run's policy
    rematerializes (``named``)."""
    if not _NAMING.get():
        return tree
    return jax.tree.map(lambda a: checkpoint_name(a, name), tree)


def named[R](name: str, f: Callable[..., R], *args: Any) -> R:
    """Return f(*args), naming what its backward takes, inside a call the run's policy
    rematerializes: `name.in` its inputs, and `name.out` its output and what else, so
    that a policy that keeps `name` computes none of f again. A kernel's residuals, as
    flash attention's log-sum-exp, are no products of matrices, so `dots` computes
    such a kernel again whole. `f` closes over no array but constants: jax
    differentiates it by `args` alone."""
    if not _NAMING.get():
        return f(*args)
    call = jax.custom_vjp(f)

    def forward(*args: Any) -> tuple[R, Any]:
        out, pull = jax.vjp(f, *args)
        inputs = {id(a) for a in jax.tree.leaves(args)}

        def labelled(a: jax.Array) -> jax.Array:
            part = "in" if id(a) in inputs else "out"
            return checkpoint_name(a, f"{name}.{part}")

        return jax.tree.map(labelled, (out, pull))

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
