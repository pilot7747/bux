"""The graph side: one small module per IR node kind, assembled by ``build_node``.

A residual branch outside any other is the residual algebra's (``residual``), when the
architecture sets one; inside a branch, or without one, it is a plain skip. What a
branch returns is the stream, which splits as the layout's `stream` says.

A repeat is built as the run's `repeat` method says: ``unroll``, a module a pass, or
``scan``, which stacks the passes into one ``Scanned`` module where it can, its
variables gaining a leading `layers` dimension, and runs them by `jax.lax.scan`. The
run's `remat` method rematerializes each branch outside any other in the backward
pass, and a scanned pass as a whole.
"""

import contextvars
import functools
import operator
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Shaped

from bux import parallel
from bux.architecture import residual
from bux.architecture.registry import LOSSES, Context, Loss, Stream, Taps, build_layer
from bux.lang import ir
from bux.parallel import methods, stream
from bux.parallel.layout import LAYERS, NAMES


class Sequence(nnx.Module):
    """Its items in a row; those that are `branches` outside any other, rematerialized
    as the run's `remat` says (``rematted``)."""

    def __init__(self, items: list[nnx.Module], branches: tuple[bool, ...] = ()):
        self.items = nnx.List(items)
        self.branches = branches or (False,) * len(items)

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        for item, branch in zip(self.items, self.branches, strict=True):
            x = rematted(item, x, taps) if branch else item(x, taps)
        return x


class ResidualBranch(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return stream(x + self.body(x, taps))


class Parallel(nnx.Module):
    def __init__(self, branches: list[nnx.Module], op: str):
        self.branches = nnx.List(branches)
        self.op = op

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        outs = [branch(x, taps) for branch in self.branches]
        return functools.reduce(
            operator.add if self.op == "sum" else operator.mul, outs
        )


class Negate(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return -self.body(x, taps)


class Constant(nnx.Module):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x: Stream, taps: Taps) -> Shaped[Array, ""]:
        return jnp.asarray(self.value, dtype=x.dtype)


class Identity(nnx.Module):
    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return x


class TapWrite(nnx.Module):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        taps.setdefault(self.name, []).append(x)
        return x


class TapRead(nnx.Module):
    def __init__(self, name: str, pop: bool):
        self.name = name
        self.pop = pop

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        stack = taps[self.name]
        return stack.pop() if self.pop else stack[-1]


def build_node(node: ir.Node, ctx: Context) -> nnx.Module:
    if node.share is not None:
        if node.share not in ctx.shared:
            ctx.shared[node.share] = _build(node, ctx)
        return ctx.shared[node.share]
    return _build(node, ctx)


def _build(node: ir.Node, ctx: Context) -> nnx.Module:
    match node:
        case ir.Layer():
            return build_layer(node, ctx)
        case ir.Seq(items=items):
            built = [build_node(item, ctx) for item in items]
            return Sequence(built, _branches(items, ctx))
        case ir.Residual(body=body):
            return _residual(body, ctx)
        case ir.Sum(terms=terms):
            return Parallel([build_node(t, ctx) for t in terms], "sum")
        case ir.Prod(factors=factors):
            return Parallel([build_node(f, ctx) for f in factors], "prod")
        case ir.Neg(body=body):
            return Negate(build_node(body, ctx))
        case ir.Const(value=value):
            return Constant(value)
        case ir.Identity():
            return Identity()
        case ir.Repeat(body=body, times=times):
            return parallel.method("repeat")(body, times, ctx)
        case ir.Tap(name=name):
            return TapWrite(name)
        case ir.Ref(name=name, pop=pop):
            return TapRead(name, pop)
    raise AssertionError(node)


def _residual(body: ir.Node, ctx: Context) -> nnx.Module:
    """The branch around `body`: the algebra's at depth 0, else a plain skip."""
    ctx.depth += 1
    try:
        inner = build_node(body, ctx)
    finally:
        ctx.depth -= 1
    if ctx.arch.residual is None or ctx.depth > 0:
        return ResidualBranch(inner)
    return residual.branch(ctx.arch.residual, inner, ctx)


def _branches(nodes: tuple[ir.Node, ...], ctx: Context) -> tuple[bool, ...]:
    """Which of `nodes`, in a row, are residual branches outside any other."""
    return tuple(isinstance(n, ir.Residual) and ctx.depth == 0 for n in nodes)


# --- Repeats: methods.repeat -----------------------------------------------------------
# A method builds `times` passes of `body`, and the module that runs them.


@methods.repeat.register("unroll")
def unroll(body: ir.Node, times: int, ctx: Context) -> nnx.Module:
    """A module a pass, in a row."""
    return Sequence(_passes(body, times, ctx), _branches((body,) * times, ctx))


@methods.repeat.register("scan")
def scan(body: ir.Node, times: int, ctx: Context) -> nnx.Module:
    """The passes, built as ``unroll`` builds them, stacked into one ``Scanned`` module
    if they can be (``scannable``); in a row if not."""
    passes = _passes(body, times, ctx)
    if scannable(passes):
        return Scanned(passes)
    return Sequence(passes, _branches((body,) * times, ctx))


def _passes(body: ir.Node, times: int, ctx: Context) -> list[nnx.Module]:
    return [build_node(body, ctx) for _ in range(times)]


def scannable(passes: list[nnx.Module]) -> bool:
    """Whether `passes` stack into one: of one graphdef and one shape of each variable,
    sharing no module or variable (a `shared` definition, nextn's embedding), and
    leaving the stream as wide as they found it, which attnres's grows."""
    if len(passes) < 2:
        return False
    first = passes[0]
    kind = nnx.graphdef(first), _shapes(first)
    same = all((nnx.graphdef(p), _shapes(p)) == kind for p in passes[1:])
    grows = any(isinstance(n, residual.AttentionResidual) for n in _nodes(first))
    return same and not grows and not _shares(passes)


def _nodes(module: nnx.Module) -> list[Any]:
    return [node for _, node in nnx.iter_graph(module)]


def _variables(module: nnx.Module) -> list[nnx.Variable]:
    return [n for n in _nodes(module) if isinstance(n, nnx.Variable)]


def _shapes(module: nnx.Module) -> list[tuple]:
    return [(v.shape, v.get_value().dtype) for v in _variables(module)]


def _shares(passes: list[nnx.Module]) -> bool:
    """Whether a module or variable is in more than one of `passes`."""
    seen: set[int] = set()
    for p in passes:
        mine = {id(n) for n in _nodes(p) if isinstance(n, nnx.Variable | nnx.Pytree)}
        if mine & seen:
            return True
        seen |= mine
    return False


def _stack(passes: list[nnx.Module]) -> nnx.Module:
    """The first of `passes`, each of its variables the stack of every pass's, along a
    leading dimension named `layers`."""
    for vs in zip(*map(_variables, passes), strict=True):
        names = vs[0].get_metadata().get(NAMES) or (None,) * len(vs[0].shape)
        vs[0].set_value(jnp.stack([v.get_value() for v in vs]))
        vs[0].set_metadata(names=(LAYERS, *names))
    return passes[0]


class Scanned(nnx.Module):
    """A repeat of `times` passes run as one: `layers` is the first pass, each of its
    variables the stack of every pass's, and a call runs them by `jax.lax.scan`
    (``run``)."""

    def __init__(self, passes: list[nnx.Module]):
        self.times = len(passes)
        self.layers = _stack(passes)

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        graphdef, state = nnx.split(self.layers)
        y, rest = run(graphdef, state, self.times, x, taps)
        nnx.update(self.layers, rest)
        return y


def run(
    graphdef: nnx.GraphDef, state: nnx.State, times: int, x: Stream, taps: Taps
) -> tuple[Stream, nnx.State]:
    """The `times` layers that `graphdef` and the stacked `state` make, run on `x`;
    and their state other than parameters, as they leave it. They run by
    `jax.lax.scan` from the first layer whose pass leaves `x`'s shape and the taps'
    as it found them: the first, or the second where the first starts a tap (mhc's
    single-pass read). One by one if neither does, as when they push to a tap."""
    done: list[nnx.State] = []
    for i in range(times):
        if i < 2 and (stacks := _steady(graphdef, _at(state, i), x, taps)) is not None:
            x, rest = _scan(graphdef, _since(state, i), times - i, x, taps, stacks)
            return x, _join([*done, rest])
        x, rest = _layer(graphdef, _at(state, i), x, taps)
        done.append(jax.tree.map(lambda a: a[None], rest))
    return x, _join(done)


def _at(state: nnx.State, i: int) -> nnx.State:
    return jax.tree.map(lambda a: a[i], state)


def _since(state: nnx.State, i: int) -> nnx.State:
    return jax.tree.map(lambda a: a[i:], state)


def _join(parts: list[nnx.State]) -> nnx.State:
    """Stacked states, one after the other."""
    if len(parts) == 1:
        return parts[0]
    return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)


def _steady(
    graphdef: nnx.GraphDef, layer: nnx.State, x: Stream, taps: Taps
) -> Taps | None:
    """The stacks of `taps` that a pass of `layer` changes, which a scan carries, if it
    leaves their shapes and `x`'s as it found them; None if it does not."""
    call = Pure(graphdef, taps)
    token = _REMATTED.set(True)  # a trial remats nothing
    try:
        y, _, changed, _ = jax.eval_shape(call, layer, x, {})
    finally:
        _REMATTED.reset(token)
    if _type(y) != _type(x) or not changed.keys() <= taps.keys():
        return None
    stacks = {k: taps[k] for k in changed}
    return stacks if _type(changed) == _type(stacks) else None


def _type(tree: Any) -> tuple:
    """The structure of `tree`, and the shape and dtype of every array in it."""
    leaves, structure = jax.tree.flatten(jax.eval_shape(lambda t: t, tree))
    return structure, [(a.shape, a.dtype, a.weak_type) for a in leaves]


def _layer(
    graphdef: nnx.GraphDef, layer: nnx.State, x: Stream, taps: Taps
) -> tuple[Stream, nnx.State]:
    """One layer, rematerialized as a whole as the run's `remat` says."""
    call = Pure(graphdef, taps)
    y, rest, changed, losses = _rematted(call, prevent_cse=True)(layer, x, {})
    _apply(taps, changed, call.reports, [losses])
    return y, rest


def _scan(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    times: int,
    x: Stream,
    taps: Taps,
    stacks: Taps,
) -> tuple[Stream, nnx.State]:
    """The layers by `jax.lax.scan`, each rematerialized as a whole as the run's
    `remat` says, carrying `x` and `stacks`."""
    call = Pure(graphdef, taps)
    layer = _rematted(call, prevent_cse=False)  # a scan keeps XLA from merging them

    def step(
        carry: tuple[Stream, Taps], s: nnx.State
    ) -> tuple[tuple[Stream, Taps], tuple[nnx.State, list[Float[Array, ""]]]]:
        y, rest, changed, losses = layer(s, *carry)
        return (y, changed), (rest, losses)

    (x, changed), (rest, losses) = jax.lax.scan(step, (x, stacks), state, times)
    _apply(taps, changed, call.reports, [[v[i] for v in losses] for i in range(times)])
    return x, rest


# --- Pure calls, for jax.checkpoint and jax.lax.scan ------------------------------------


class Pure:
    """The call of the module that `graphdef` and a state make, as a function of arrays
    alone: ``(state, x, stacks) -> (y, rest, changed, losses)``.

    It reads `taps`, with `stacks` in place of theirs, and returns the stream, its
    state other than parameters (an MoE's router bias), the stacks it changed and
    those of `stacks`, and the values of the losses it reports; `reports` keeps their
    names and weights as it traces. The taps it is given do not change.
    """

    def __init__(self, graphdef: nnx.GraphDef, taps: Taps):
        self.graphdef, self.taps = graphdef, taps
        self.reports: list[tuple[str, float]] = []

    def __call__(
        self, state: nnx.State, x: Stream, stacks: Taps
    ) -> tuple[Stream, nnx.State, Taps, list[Float[Array, ""]]]:
        before = {k: list(v) for k, v in self.taps.items()} | stacks
        if LOSSES in before:
            before[LOSSES] = []
        local = {k: list(v) for k, v in before.items()}
        module = nnx.merge(self.graphdef, state)
        y = module(x, local)
        losses = local.get(LOSSES, [])
        self.reports = [(loss.name, loss.weight) for loss in losses]
        changed = {
            k: v
            for k, v in local.items()
            if k != LOSSES and (k in stacks or not _same(before.get(k), v))
        }
        rest = nnx.state(module, nnx.Not(nnx.Param))
        return y, rest, changed, [loss.value for loss in losses]


def _same(before: list | None, after: list) -> bool:
    return (
        before is not None
        and len(before) == len(after)
        and all(a is b for a, b in zip(before, after))
    )


def _apply(
    taps: Taps,
    changed: Taps,
    reports: list[tuple[str, float]],
    losses: list[list[Float[Array, ""]]],
) -> None:
    """Into `taps`, what pure calls changed: their stacks, and the losses each
    reported, call by call."""
    taps.update(changed)
    if not reports:
        return
    for values in losses:
        taps[LOSSES].extend(
            Loss(name, weight, v) for (name, weight), v in zip(reports, values)
        )


# --- Remat: methods.remat --------------------------------------------------------------
# A method is a jax.checkpoint policy: what the forward pass keeps for the backward
# one, which computes the rest again. `none` keeps everything, so nothing is redone.

methods.remat.register("none")(jax.checkpoint_policies.everything_saveable)
methods.remat.register("full")(jax.checkpoint_policies.nothing_saveable)
methods.remat.register("dots")(jax.checkpoint_policies.dots_saveable)
methods.remat.register("dots_no_batch")(
    jax.checkpoint_policies.dots_with_no_batch_dims_saveable
)

# Set while a rematerialized call traces, so that what it calls is not again.
_REMATTED = contextvars.ContextVar("rematted", default=False)

Policy = Callable[..., bool]


def _policy() -> Policy | None:
    """The run's remat policy; None where nothing rematerializes: with `none`, or
    inside a call that does already."""
    policy = parallel.method("remat")
    none = policy is jax.checkpoint_policies.everything_saveable
    return None if none or _REMATTED.get() else policy


def _rematted[**A, R](f: Callable[A, R], prevent_cse: bool) -> Callable[A, R]:
    """`f`, rematerialized by the run's remat policy; `f` itself where there is none."""
    if (policy := _policy()) is None:
        return f

    def inner(*args: A.args, **kwargs: A.kwargs) -> R:
        token = _REMATTED.set(True)
        try:
            return f(*args, **kwargs)
        finally:
            _REMATTED.reset(token)

    return jax.checkpoint(inner, policy=policy, prevent_cse=prevent_cse)


def rematted(module: nnx.Module, x: Stream, taps: Taps) -> Stream:
    """`module(x, taps)`, rematerialized by the run's remat policy."""
    if _policy() is None:
        return module(x, taps)
    graphdef, state = nnx.split(module)
    call = Pure(graphdef, taps)
    y, rest, changed, losses = _rematted(call, prevent_cse=True)(state, x, {})
    nnx.update(module, rest)
    _apply(taps, changed, call.reports, [losses])
    return y
