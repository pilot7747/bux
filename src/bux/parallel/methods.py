"""Registries of methods, one per field of ``methods``: how each part that communicates
runs. A new method is a function:

    from bux.parallel import methods

    @methods.attention.register("mine")
    def _(q, k, v, args): ...

and ``methods = methods(attention=mine)`` picks it. Whoever adds a kind of method
defines its signature, and registers its default next to the layer that calls it.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from bux.parallel.mesh import active
from bux.train.config import MethodsConfig
from bux.train.registry import Registry

if TYPE_CHECKING:
    from bux.lang import ir

Method = Callable[..., Any]
Preparation = Callable[["ir.Run"], None]


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
tensor = _registry("tensor")

REGISTRIES = {
    "attention": attention,
    "moe": moe,
    "recurrence": recurrence,
    "repeat": repeat,
    "remat": remat,
    "pipeline": pipeline,
    "orthogonalize": orthogonalize,
    "loss": loss,
    "tensor": tensor,
}


# What a method needs done before a run's step traces, by (kind, name): as NCCL EP's
# buffers, which live outside the program, sized for the run.
PREPARATIONS: dict[tuple[str, str], Preparation] = {}


def before(kind: str, name: str) -> Callable[[Preparation], Preparation]:
    """Register `prepare(run)`, which ``prepare`` calls for a run whose method of
    `kind` is `name`."""

    def register(prepare: Preparation) -> Preparation:
        PREPARATIONS[kind, name] = prepare
        return prepare

    return register


def prepare(run: "ir.Run") -> None:
    """Do what the methods in force need done before `run`'s step traces
    (``before``). Call it inside ``use``."""
    on = active()
    config = on.methods if on else MethodsConfig()
    for kind in REGISTRIES:
        name = getattr(config, kind)
        if isinstance(name, str) and (preparation := PREPARATIONS.get((kind, name))):
            preparation(run)


def method(kind: str) -> Method:
    """Return the run's method for `kind`, or the default one outside ``use``."""
    on = active()
    return REGISTRIES[kind].find(getattr(on.methods if on else MethodsConfig(), kind))


def validate(config: MethodsConfig) -> None:
    """Raise if `config` names a method that is not registered.

    An unregistered default is fine: its call site is not wired yet. A method given
    as data, as `remat(...)` is, has no name to find.
    """
    default = MethodsConfig()
    for kind, registry in REGISTRIES.items():
        name = getattr(config, kind)
        if isinstance(name, str) and name != getattr(default, kind):
            registry.find(name)
