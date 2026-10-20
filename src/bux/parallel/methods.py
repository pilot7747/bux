"""How each part that communicates is computed: a registry of methods per field of
``methods``, keyed by name. Another method is a function:

    from bux.parallel import methods

    @methods.attention.register("mine")
    def _(q, k, v, visibility): ...

and ``methods = methods(attention=mine)`` picks it. The WP that wires a call site
fixes its methods' signature, and registers the default where the layer lives.
"""

from collections.abc import Callable
from typing import Any

from bux.parallel.mesh import active
from bux.train.config import MethodsConfig
from bux.train.registry import Registry

Method = Callable[..., Any]


def _registry(kind: str) -> Registry[Any]:
    return Registry[Any](f"{kind} method", f"bux.parallel.methods.{kind}")


attention = _registry("attention")
moe = _registry("moe")
recurrence = _registry("recurrence")
repeat = _registry("repeat")
remat = _registry("remat")
pipeline = _registry("pipeline")
orthogonalize = _registry("orthogonalize")

REGISTRIES = {
    "attention": attention,
    "moe": moe,
    "recurrence": recurrence,
    "repeat": repeat,
    "remat": remat,
    "pipeline": pipeline,
    "orthogonalize": orthogonalize,
}


def method(kind: str) -> Method:
    """The run's method for `kind`, from its registry; the default outside ``use``."""
    on = active()
    return REGISTRIES[kind].find(getattr(on.methods if on else MethodsConfig(), kind))


def validate(c: MethodsConfig) -> None:
    """Raise if `c` names a method that is not registered. A default that is not is
    a call site not wired yet, and fine."""
    default = MethodsConfig()
    for kind, registry in REGISTRIES.items():
        name = getattr(c, kind)
        if name != getattr(default, kind):
            registry.find(name)
