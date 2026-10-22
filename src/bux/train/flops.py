"""FLOPs, counted from a program, and the peak of the devices that run it, for MFU.

``count`` traces a function on shapes alone, so nothing runs, and adds up its matrix
products and convolutions: 2·m·n·k each, times the length of every scan around them,
the larger branch of a cond, a while loop's body once. So a new layer counts with no
formula of its own. A kernel (a pallas_call, a cuDNN call) hides its FLOPs: trace
inside ``bux.kernels.off``, as ``trainer.step_flops`` does.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import jax
from jax.extend import core

# Dense bf16 tensor FLOPs a second of one device, by a name in its device_kind:
# NVIDIA's datasheets, the SXM parts.
PEAK = {
    "B300": 2.25e15,
    "B200": 2.25e15,
    "H200": 989e12,
    "H100": 989e12,
    "A100": 312e12,
}


def count(f: Callable[..., Any], *args: Any) -> float:
    """Return the FLOPs of `f(*args)`, `args` arrays or their shapes."""
    return flops(jax.make_jaxpr(f)(*args).jaxpr)


def flops(jaxpr: core.Jaxpr) -> float:
    """Return the FLOPs of the matrix products and convolutions in `jaxpr`."""
    return sum(_equation(e) for e in jaxpr.eqns)


def _equation(e: core.JaxprEqn) -> float:
    match e.primitive.name:
        case "dot_general":
            (_, contracting), (_, batch) = e.params["dimension_numbers"]
            return _product(e, (*contracting, *batch))
        case "ragged_dot_general":
            numbers = e.params["ragged_dot_dimension_numbers"]
            (_, contracting), (_, batch) = numbers.dot_dimension_numbers
            return _product(e, (*contracting, *batch, *numbers.rhs_group_dimensions))
        case "conv_general_dilated":
            rhs = _shape(e.invars[1])  # out features, in features / groups, window
            each = math.prod(rhs) / rhs[e.params["dimension_numbers"].rhs_spec[0]]
            return 2 * math.prod(_shape(e.outvars[0])) * each
        case "scan":
            return e.params["length"] * flops(e.params["jaxpr"].jaxpr)
        case "cond":
            return max(flops(b.jaxpr) for b in e.params["branches"])
    return sum(flops(j) for j in _inner(e.params))  # jit, shard_map, custom_vjp, ...


def _inner(params: dict[str, Any]) -> Iterator[core.Jaxpr]:
    """The jaxprs an equation calls."""
    for value in params.values():
        for v in value if isinstance(value, tuple | list) else (value,):
            if isinstance(v, core.ClosedJaxpr):
                yield v.jaxpr
            elif isinstance(v, core.Jaxpr):
                yield v


def _product(e: core.JaxprEqn, rhs_shared: Sequence[int]) -> float:
    """2 for every pair of the lhs's elements and the rhs's free ones: those not in
    `rhs_shared`, which the lhs's elements meet one at a time."""
    rhs = _shape(e.invars[1])
    free = math.prod(n for d, n in enumerate(rhs) if d not in rhs_shared)
    return 2 * math.prod(_shape(e.invars[0])) * free


def _shape(v: Any) -> tuple[int, ...]:
    """The shape of an equation's input or output."""
    return v.aval.shape


def peak(kind: str) -> float | None:
    """Return the dense bf16 FLOPs a second of a device of `kind` (its device_kind),
    or None if it is not in PEAK."""
    return next((p for name, p in PEAK.items() if name in kind), None)
