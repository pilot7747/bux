"""The graph side: one small module per IR node kind, assembled by ``build_node``.

A residual branch outside any other is the residual algebra's (``residual``), when the
architecture sets one; inside a branch, or without one, it is a plain skip. What a
branch returns is the stream, which splits as the layout's `stream` says.
"""

import functools
import operator

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Shaped

from bux.architecture import residual
from bux.architecture.registry import Context, Stream, Taps, build_layer
from bux.lang import ir
from bux.parallel import stream


class Sequence(nnx.Module):
    def __init__(self, items: list[nnx.Module]):
        self.items = nnx.List(items)

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        for item in self.items:
            x = item(x, taps)
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
            return Sequence([build_node(item, ctx) for item in items])
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
            return Sequence([build_node(body, ctx) for _ in range(times)])
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
