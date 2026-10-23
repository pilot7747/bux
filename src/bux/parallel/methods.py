"""Registries of methods, one per field of ``methods``: how each part that communicates
runs. A new method is a function:

    from bux.parallel import methods

    @methods.attention.register("mine")
    def _(q, k, v, args): ...

and ``methods = methods(attention=mine)`` picks it. Whoever adds a kind of method
defines its signature, and registers its default next to the layer that calls it.
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
loss = _registry("loss")

REGISTRIES = {
    "attention": attention,
    "moe": moe,
    "recurrence": recurrence,
    "repeat": repeat,
    "remat": remat,
    "pipeline": pipeline,
    "orthogonalize": orthogonalize,
    "loss": loss,
}


def method(kind: str) -> Method:
    """Return the run's method for `kind`, or the default one outside ``use``."""
    on = active()
    return REGISTRIES[kind].find(getattr(on.methods if on else MethodsConfig(), kind))


def validate(config: MethodsConfig) -> None:
    """Raise if `config` names a method that is not registered.

    An unregistered default is fine: its call site is not wired yet.
    """
    default = MethodsConfig()
    for kind, registry in REGISTRIES.items():
        name = getattr(config, kind)
        if name != getattr(default, kind):
            registry.find(name)
