"""A bux model's weights, by the paths of the exported model's modules.

Each exported layer mirrors its NNX module's attributes, so one rule names them all:
a ``nnx.Linear``'s kernel, transposed, and an ``nnx.Embed``'s table become
``weight``, as do a norm's scale; any other parameter keeps its attribute's name, and
submodules nest by theirs (``q_norm.weight``). The few variables besides parameters
that inference reads (`STATE`) go too. An exported layer whose own layout differs
converts its tensors after (``bux_weights``, `bux.export.hf.common`).
"""

from collections.abc import Callable, Iterator
from typing import Any

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
        own = dict(_named(module, "", seen))
        if (fn := convert(call.layer.config.type)) is not None:
            own = fn(own, call.layer.config.model_dump(mode="json"))
        out |= {f"{call.path}.{name}": array for name, array in own.items()}
    return out


def module_at(node: ir.Node, module: Any, trail: tuple[int, ...]) -> nnx.Module:
    """Return the module of the node at `trail` under `node`, whose module is `module`:
    a sequence's items, a sum's or product's branches, a residual's or negation's body,
    and a repeat's passes, as `bux.architecture.graph` builds them."""
    for i in trail:
        match node:
            case ir.Seq(items=items):
                node, module = items[i], module.items[i]
            case ir.Sum(terms=items) | ir.Prod(factors=items):
                node, module = items[i], module.branches[i]
            case ir.Residual(body=body) | ir.Neg(body=body):
                node, module = body, module.body
            case ir.Repeat(body=body):
                node, module = body, module.items[i]
            case _:
                raise AssertionError(node)
    return module


def _named(m: object, prefix: str, seen: set[int]) -> Iterator[tuple[str, np.ndarray]]:
    """Yield the parameters of `m` by the rule, each module's once."""
    if id(m) in seen:
        return
    seen.add(id(m))
    match m:
        case nnx.Linear():
            yield f"{prefix}weight", np.asarray(m.kernel[...]).T
            if m.bias is not None:
                yield f"{prefix}bias", np.asarray(m.bias[...])
        case nnx.Embed():
            yield f"{prefix}weight", np.asarray(m.embedding[...])
        case Norm():
            for name, p in (("weight", m.scale), ("bias", m.bias)):
                if p is not None:
                    yield f"{prefix}{name}", np.asarray(p[...])
        case _:
            state = STATE.get(type(m), ())
            for name, value in vars(m).items():
                if name in state and isinstance(value, nnx.Variable):
                    yield f"{prefix}{name}", np.asarray(value[...])
                else:
                    yield from _attribute(value, f"{prefix}{name}", seen)


def _attribute(
    value: object, name: str, seen: set[int]
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield the parameters an attribute holds: itself, a module's, or a list's."""
    if isinstance(value, nnx.Param):
        yield name, np.asarray(value[...])
    elif isinstance(value, nnx.Module):
        yield from _named(value, f"{name}.", seen)
    elif isinstance(value, list | tuple | nnx.List):
        for i, item in enumerate(value):
            yield from _attribute(item, f"{name}.{i}", seen)
