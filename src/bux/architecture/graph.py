"""The graph side: one small module per IR node kind, assembled by ``build_node``."""

import functools
import operator

import jax.numpy as jnp
from flax import nnx

from bux.architecture.registry import Context, Taps, build_layer
from bux.lang import ir


class Sequence(nnx.Module):
    def __init__(self, items: list[nnx.Module]):
        self.items = nnx.List(items)

    def __call__(self, x, taps: Taps):
        for item in self.items:
            x = item(x, taps)
        return x


class ResidualBranch(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x, taps: Taps):
        return x + self.body(x, taps)


class Parallel(nnx.Module):
    def __init__(self, branches: list[nnx.Module], op: str):
        self.branches = nnx.List(branches)
        self.op = op

    def __call__(self, x, taps: Taps):
        outs = [branch(x, taps) for branch in self.branches]
        return functools.reduce(
            operator.add if self.op == "sum" else operator.mul, outs
        )


class Negate(nnx.Module):
    def __init__(self, body: nnx.Module):
        self.body = body

    def __call__(self, x, taps: Taps):
        return -self.body(x, taps)


class Constant(nnx.Module):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x, taps: Taps):
        return jnp.asarray(self.value, dtype=x.dtype)


class Identity(nnx.Module):
    def __call__(self, x, taps: Taps):
        return x


class TapWrite(nnx.Module):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, x, taps: Taps):
        taps.setdefault(self.name, []).append(x)
        return x


class TapRead(nnx.Module):
    def __init__(self, name: str, pop: bool):
        self.name = name
        self.pop = pop

    def __call__(self, x, taps: Taps):
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
            return ResidualBranch(build_node(body, ctx))
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
