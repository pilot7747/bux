"""The graph side: one small module per IR node kind, assembled by `build_node`.

- ``ir.Seq``, composition: `Sequence`, the items one after another.
- ``ir.Residual``, ``[f]``: `ResidualBranch`, ``x + f(x)``; or the residual
  algebra's branch, if the architecture sets one and the branch is outside any
  other (`bux.architecture.residual`). A branch inside another is always the plain
  ``x + f(x)``.
- ``ir.Sum`` and ``ir.Prod``, ``f + g`` and ``f * g``: `Parallel`.
- ``ir.Neg``, ``ir.Const`` and ``ir.Identity``: `Negate`, `Constant` and `Identity`.
- ``ir.Tap`` and ``ir.Ref``, ``@name`` and ``$name``: `TapWrite` and `TapRead`.
- ``ir.Repeat``, ``f^n``: what the run's ``methods.repeat`` builds
  (`bux.architecture.repeats.repeat`).
- ``ir.Layer``: what the builder registered for its config's type builds
  (`bux.architecture.registry`).

Each residual branch outside any other is rematerialized as the run's
``methods.remat`` says (`bux.architecture.repeats.remat`), and observed by its name
on the steps a run observes (`bux.architecture.observe`).
"""

import functools
import operator

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Shaped

from bux import parallel
from bux.architecture import observe, residual
from bux.architecture.registry import Context, Stream, Taps, build_layer
from bux.architecture.repeats import remat, tracing
from bux.lang import ir
from bux.parallel import shard_stream


class Sequence(nnx.Module):
    """Run the items one after another: ``f g`` is ``g(f(x))``.

    Those that are residual branches outside any other are numbered, for the
    precision's first and last (`~bux.architecture.repeats.tracing.branch`),
    rematerialized as the run's ``methods.remat`` says
    (`~bux.architecture.repeats.remat.rematted`), and observed by their names.

    Args:
        items: The modules, in the order they run.
        branches: For each item, whether it is a residual branch outside any other;
            none of them, by default.
        names: For each item, what its metrics are named by
            (`~bux.architecture.observe.item_name`): a branch's and a layer's, None
            for an item whose own items are named; none, by default.
    """

    def __init__(
        self,
        items: list[nnx.Module],
        branches: tuple[bool, ...] = (),
        names: tuple[str | None, ...] = (),
    ):
        self.items = nnx.List(items)
        self.branches = branches or (False,) * len(items)
        self.names = names or (None,) * len(items)

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        for item, branch, name in zip(
            self.items, self.branches, self.names, strict=True
        ):
            x = run_branch(item, x, taps, name) if branch else item(x, taps)
        return x


def run_branch(
    item: nnx.Module, x: Stream, taps: Taps, name: str | None = None
) -> Stream:
    """Run a residual branch outside any other: numbered, rematerialized, and
    observed as `name` (`bux.architecture.observe.stream`)."""
    with tracing.branch(), observe.branch(name):
        y = remat.rematted(item, x, taps)
        observe.stream(taps, x, y)
    return y


class ResidualBranch(nnx.Module):
    """``[f]``, the plain residual branch: ``x + f(x)``, the sum split as the stream
    is (`bux.parallel.shard_stream`)."""

    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return shard_stream(x + self.body(x, taps))


class Parallel(nnx.Module):
    """Branches on the same input, their outputs summed (``op="sum"``, ``f + g``) or
    multiplied (``"prod"``, ``f * g``), left to right."""

    def __init__(self, branches: list[nnx.Module], op: str):
        self.branches = nnx.List(branches)
        self.op = op

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        outs = [branch(x, taps) for branch in self.branches]
        return functools.reduce(
            operator.add if self.op == "sum" else operator.mul, outs
        )


class Negate(nnx.Module):
    """``-f``: the body's output, negated."""

    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return -self.body(x, taps)


class Constant(nnx.Module):
    """A number, as a scalar of the stream's dtype: the 0.5 of ``f * 0.5``."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, x: Stream, taps: Taps) -> Shaped[Array, ""]:
        return jnp.asarray(self.value, dtype=x.dtype)


class Identity(nnx.Module):
    """``_``: the stream, unchanged."""

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return x


class TapWrite(nnx.Module):
    """``@name``: push the stream onto the tap `name`, and pass it on."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        taps.setdefault(self.name, []).append(x)
        return x


class TapRead(nnx.Module):
    """``$name``: the top of the tap `name`, in place of the stream; with `pop`
    (``$name!``), taken off it."""

    def __init__(self, name: str, pop: bool):
        self.name = name
        self.pop = pop

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        stack = taps[self.name]
        return stack.pop() if self.pop else stack[-1]


def build_node(node: ir.Node, ctx: Context) -> nnx.Module:
    """Build the module for `node`; a node with a share key is built once and reused.

    Args:
        node: A node of an architecture's graph, and with it everything under it.
        ctx: The build's context, which carries what the nodes built so far leave
            for later ones.

    Returns:
        The node's module, called as ``module(x, taps)``.
    """
    if node.share is None:
        return _build(node, ctx)
    if node.share not in ctx.shared:
        ctx.shared[node.share] = _build(node, ctx)
    return ctx.shared[node.share]


def _build(node: ir.Node, ctx: Context) -> nnx.Module:
    match node:
        case ir.Layer():
            return build_layer(node, ctx)
        case ir.Seq(items=items):
            built = [build_node(item, ctx) for item in items]
            return Sequence(built, branches(items, ctx), names(items, ctx))
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
    """Build the branch around `body`: the algebra's at depth 0, else a plain skip."""
    ctx.depth += 1
    try:
        inner = build_node(body, ctx)
    finally:
        ctx.depth -= 1
    if ctx.arch.residual is None or ctx.depth > 0:
        return ResidualBranch(inner)
    return residual.branch(ctx.arch.residual, inner, ctx)


def branches(nodes: tuple[ir.Node, ...], ctx: Context) -> tuple[bool, ...]:
    """Return which of `nodes`, in a row, are residual branches outside any other."""
    return tuple(isinstance(n, ir.Residual) and ctx.depth == 0 for n in nodes)


def names(nodes: tuple[ir.Node, ...], ctx: Context) -> tuple[str | None, ...]:
    """Return what the metrics of each of `nodes`, in a row, are named by
    (`bux.architecture.observe.item_name`)."""
    flags = branches(nodes, ctx)
    return tuple(observe.item_name(n, b) for n, b in zip(nodes, flags, strict=True))
