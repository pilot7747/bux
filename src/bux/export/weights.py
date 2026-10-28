"""A bux model's weights, by the paths of the exported model's modules.

Each exported layer mirrors its NNX module's attributes, so one rule names them all:
a ``nnx.Linear``'s kernel, transposed, and an ``nnx.Embed``'s table become
``weight``, as do a norm's scale; any other parameter keeps its attribute's name, and
submodules nest by theirs (``q_norm.weight``). The few variables besides parameters
that inference reads (`STATE`) go too. An exported layer whose own layout differs
converts its tensors after (``bux_weights``, `bux.export.hf.common`). A residual
algebra's branch holds its own parameters by the same rule, its body's being its
layers'.
"""

from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import jax
import numpy as np
from flax import nnx
from jaxtyping import Shaped

from bux.architecture import Model
from bux.architecture.experts.moe import MoE
from bux.architecture.layers.norm import Norm
from bux.export.plan import Plan
from bux.lang import ir

#: A layer's tensors, by their names under its module.
Tensors = dict[str, Shaped[np.ndarray, "..."]]

#: Variables besides parameters that inference reads, by module class: an MoE's
#: aux-free bias, which picks its experts.
STATE: dict[type, tuple[str, ...]] = {MoE: ("router_bias",)}

#: Converts a layer's tensors to its exported layout: ``(tensors, config) -> tensors``.
Convert = Callable[[Tensors, dict], Tensors]


def tensors(
    plan: Plan, model: Model, convert: Callable[[str], Convert | None] = lambda _: None
) -> Tensors:
    """Return the weights of `model`, built from ``plan.arch``, by exported path.

    Args:
        plan: The exported model's module tree (`bux.export.plan.plan`).
        model: The bux model, built with ``methods.repeat`` ``unroll`` (the default).
        convert: The converter of a layer type's tensors to its exported layout, or
            None to keep them as the rule names them.

    Returns:
        Every parameter once, a shared module's at its own path, as numpy arrays.
    """
    out: Tensors = {}
    seen: set[int] = set()
    for call in plan.calls:
        if not call.first:
            continue
        module = module_at(plan.arch.graph, model.graph, call.trail)
        own = {name: ref.array() for name, ref in refs(module, "", seen)}
        if (fn := convert(call.layer.config.type)) is not None:
            own = fn(own, call.layer.config.model_dump(mode="json"))
        out |= {f"{call.path}.{name}": array for name, array in own.items()}
    for branch in plan.branches:
        if branch.first:
            module = module_at(plan.arch.graph, model.graph, branch.trail)
            out |= {f"{branch.path}.{k}": r.array() for k, r in _algebra(module, seen)}
    return out


class Ref(NamedTuple):
    """A variable the rule names, and whether its exported tensor is its transpose (a
    ``nnx.Linear``'s kernel)."""

    variable: nnx.Variable
    transposed: bool = False

    def array(self) -> Shaped[np.ndarray, "..."]:
        """Return the variable as the rule exports it."""
        x = np.asarray(self.variable[...])
        return x.T if self.transposed else x

    def set(self, x: Shaped[np.ndarray, "..."]) -> None:
        """Set the variable from `x`, as the rule exports it, in the variable's dtype."""
        old = self.variable[...]
        x, want = np.asarray(x), old.shape[::-1] if self.transposed else old.shape
        if x.shape != want:
            raise ValueError(f"a tensor of shape {x.shape}, where {want} goes")
        x = np.asarray(x.T if self.transposed else x, old.dtype)
        self.variable[...] = jax.device_put(x, old.sharding)


def _algebra(m: nnx.Module, seen: set[int]) -> Iterator[tuple[str, Ref]]:
    """Yield the parameters of a residual algebra's branch, by the rule, but its
    body's, which are its layers'."""
    for name, value in vars(m).items():
        if name != "body":
            yield from _attribute(value, name, seen)


def module_at(node: ir.Node, module: Any, trail: tuple[int, ...]) -> nnx.Module:
    """Return the module of the node at `trail` under `node`, whose module is `module`:
    a sequence's items, a sum's or product's branches, a residual's, negation's or
    image's body, and a repeat's passes, as `bux.architecture.graph` builds them."""
    for i in trail:
        match node:
            case ir.Seq(items=items):
                node, module = items[i], module.items[i]
            case ir.Sum(terms=items) | ir.Prod(factors=items):
                node, module = items[i], module.branches[i]
            case ir.Residual(body=body) | ir.Neg(body=body) | ir.Image(body=body):
                node, module = body, module.body
            case ir.Repeat(body=body):
                node, module = body, module.items[i]
            case _:
                raise AssertionError(node)
    return module


def refs(m: object, prefix: str, seen: set[int]) -> Iterator[tuple[str, Ref]]:
    """Yield the variables of `m` by the rule, each module's once."""
    if id(m) in seen:
        return
    seen.add(id(m))
    match m:
        case nnx.Linear():
            yield f"{prefix}weight", Ref(m.kernel, transposed=True)
            if m.bias is not None:
                yield f"{prefix}bias", Ref(m.bias)
        case nnx.Embed():
            yield f"{prefix}weight", Ref(m.embedding)
        case Norm():
            for name, p in (("weight", m.scale), ("bias", m.bias)):
                if p is not None:
                    yield f"{prefix}{name}", Ref(p)
        case _:
            state = STATE.get(type(m), ())
            for name, value in vars(m).items():
                if name in state and isinstance(value, nnx.Variable):
                    yield f"{prefix}{name}", Ref(value)
                else:
                    yield from _attribute(value, f"{prefix}{name}", seen)


def _attribute(value: object, name: str, seen: set[int]) -> Iterator[tuple[str, Ref]]:
    """Yield the variables an attribute holds: itself, a module's, or a list's."""
    if isinstance(value, nnx.Param):
        yield name, Ref(value)
    elif isinstance(value, nnx.Module):
        yield from refs(value, f"{name}.", seen)
    elif isinstance(value, list | tuple | nnx.List):
        for i, item in enumerate(value):
            yield from _attribute(item, f"{name}.{i}", seen)
