"""The graph side: one small module per IR node kind, assembled by ``build_node``.

A residual branch [f] adds f(x) to the stream x. The architecture's residual algebra,
if it sets one, replaces that for a branch outside any other (``residual``); a branch
inside another is always the plain x + f(x). A repeat is built by the run's `repeat`
method (``repeat``), and each branch outside any other is rematerialized as the run's
`remat` says (``remat``).
"""

import functools
import operator

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Shaped

from bux import parallel
from bux.architecture import residual
from bux.architecture.registry import Context, Stream, Taps, build_layer
from bux.architecture.repeats import remat, tracing
from bux.lang import ir
from bux.parallel import shard_stream


class Sequence(nnx.Module):
    """Run the items one after another. Those that are `branches` outside any other
    are numbered, for the precision's first and last (``tracing.branch``), and
    rematerialized as the run's `remat` says (``remat.rematted``)."""

    def __init__(self, items: list[nnx.Module], branches: tuple[bool, ...] = ()):
        self.items = nnx.List(items)
        self.branches = branches or (False,) * len(items)

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        for item, branch in zip(self.items, self.branches, strict=True):
            x = run_branch(item, x, taps) if branch else item(x, taps)
        return x


def run_branch(item: nnx.Module, x: Stream, taps: Taps) -> Stream:
    """Run a residual branch outside any other: numbered, and rematerialized."""
    with tracing.branch():
        return remat.rematted(item, x, taps)


class ResidualBranch(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        return shard_stream(x + self.body(x, taps))


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
    """Build the module for `node`; a node with a share key is built once and reused."""
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
            return Sequence(built, branches(items, ctx))
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
