"""FLOPs, counted from a program, and the peak of the devices that run it, for MFU.

``count`` traces a function on shapes alone, so nothing runs, and adds up its matrix
products and convolutions: 2·m·n·k each, times the length of every scan around them,
the larger branch of a cond, a while loop's body once. So a new layer counts with no
formula of its own. A kernel (a pallas_call, a cuDNN call) hides its FLOPs: trace
inside ``bux.kernels.off``, as ``trainer.step_flops`` does.

Products inside ``visible(share)`` count at that share of their FLOPs, forward and
backward: attention's logits and weighted values, of which a causal mask keeps about
half (``causal_share``), as Megatron counts them, and a flash kernel computes no more.
"""

import math
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import jax

from bux import jax_internals
from bux.jax_internals import core

_VISIBLE = re.compile(r"visible=([0-9.e+-]+)")

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


def visible(share: float) -> Any:
    """A scope whose matrix products count at `share` of their FLOPs: the share of a
    block of logits that a mask keeps."""
    return jax.named_scope(f"visible={share!r}")


def causal_share(t: int, window: int | None = None) -> float:
    """Return the share of t × t logits that t queries see of their keys: none ahead,
    and none `window` or more behind. (t + 1) / 2t causal."""
    w = t if window is None else min(window, t)
    seen = w * (w + 1) // 2 + (t - w) * w  # min(i + 1, w) for each query i
    return seen / (t * t)


def flops(jaxpr: core.Jaxpr, share: float = 1.0) -> float:
    """Return the FLOPs of the matrix products and convolutions in `jaxpr`, those of
    a `visible` scope at its share, the rest at `share`."""
    return sum(_equation(e, _share(e, share)) for e in jaxpr.eqns)


def _share(e: core.JaxprEqn, outer: float) -> float:
    """The share an equation counts at: its innermost `visible` scope's, else `outer`."""
    scopes = _VISIBLE.findall(str(e.source_info.name_stack))
    return float(scopes[-1]) if scopes else outer


def _equation(e: core.JaxprEqn, share: float) -> float:
    match e.primitive.name:
        case "dot_general":
            (_, contracting), (_, batch) = e.params["dimension_numbers"]
            return share * _product(e, (*contracting, *batch))
        case "ragged_dot_general":
            numbers = e.params["ragged_dot_dimension_numbers"]
            (_, contracting), (_, batch) = numbers.dot_dimension_numbers
            shared = (*contracting, *batch, *numbers.rhs_group_dimensions)
            return share * _product(e, shared)
        case "conv_general_dilated":
            rhs = _shape(e.invars[1])  # out features, in features / groups, window
            each = math.prod(rhs) / rhs[e.params["dimension_numbers"].rhs_spec[0]]
            return share * 2 * math.prod(_shape(e.outvars[0])) * each
        case "scan":
            length = jax_internals.scan_options(e)["length"]
            return length * flops(jax_internals.scan_body(e).jaxpr, share)
        case "cond":
            branches = jax_internals.cond_branches(e)
            return max(flops(b.jaxpr, share) for b in branches)
    # jit, shard_map, custom_vjp, ...
    return sum(flops(j, share) for j in _inner(e.params))


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
