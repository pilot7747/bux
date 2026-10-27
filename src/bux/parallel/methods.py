"""Registries of methods, one per field of ``methods``, which say how each part that
communicates runs.

A new method is a function::

    from bux.parallel import methods

    @methods.attention.register("mine")
    def _(q, k, v, args): ...

and ``methods = methods(attention=mine)`` picks it; a function registered under a
taken name replaces bux's own. Whoever adds a kind of method defines its signature,
and registers its default next to the layer that calls it. The registries and what
calls their methods:

- `attention`: softmax attention's core, ``(q, k, v, args)``
  (`bux.parallel.context`, `bux.parallel.ring`).
- `moe`: the routed experts of an MoE (`bux.parallel.experts`,
  `bux.parallel.nccl_ep`, `bux.architecture.experts.moe`).
- `recurrence`: a linear attention's recurrence over positions
  (`bux.parallel.recurrence`).
- `repeat`: how a repeat builds its passes (`bux.architecture.repeats.repeat`).
- `remat`: a rematerialization policy by name (`bux.architecture.repeats.remat`).
- `pipeline`: how a pipelined repeat runs its stages (`bux.parallel.pipeline`,
  `bux.parallel.scheduled`).
- `orthogonalize`: how Muon reaches whole matrices (`bux.train.optim`).
- `loss`: how the loss takes a head's logits (`bux.architecture.logits`).
- `tensor`: how tensor parallelism's collectives meet the projections they feed
  (`bux.parallel.tensor`).

A layer calls the run's method through `method`, which outside
`~bux.parallel.settings.use` gives the default, and reads its setting by `choice`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from bux.parallel.mesh import active
from bux.train.config import MethodsConfig
from bux.train.registry import Registry

if TYPE_CHECKING:
    from bux.lang import ir

Method = Callable[..., Any]  # a method, of the signature its kind defines
Preparation = Callable[["ir.Run"], None]  # what a method needs done before a step


def _registry(kind: str) -> Registry[Any]:
    return Registry[Any](f"{kind} method", f"bux.parallel.methods.{kind}")


#: The methods of softmax attention's core, ``(q, k, v, args) -> out``, for
#: ``methods.attention``.
attention = _registry("attention")
#: The methods of an MoE's routed experts, for ``methods.moe``.
moe = _registry("moe")
#: The methods of a linear attention's recurrence, for ``methods.recurrence``.
recurrence = _registry("recurrence")
#: The ways a repeat builds its passes, for ``methods.repeat``.
repeat = _registry("repeat")
#: The rematerialization policies by name, for ``methods.remat``.
remat = _registry("remat")
#: The ways a pipelined repeat runs its stages, for ``methods.pipeline``.
pipeline = _registry("pipeline")
#: The ways Muon reaches whole matrices, for ``methods.orthogonalize``.
orthogonalize = _registry("orthogonalize")
#: The ways the loss takes a head's logits, for ``methods.loss``.
loss = _registry("loss")
#: The ways tensor parallelism's collectives meet their projections, for
#: ``methods.tensor``.
tensor = _registry("tensor")

#: Every registry, by the field of ``methods`` that names its method.
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
    """Return a decorator that registers ``prepare(run)``, which `prepare` calls for
    a run whose method of `kind` is `name`.

    Example::

        @methods.before("moe", "nccl_ep")
        def start(run: ir.Run) -> None: ...  # size NCCL EP's buffers for the run
    """

    def register(prepare: Preparation) -> Preparation:
        PREPARATIONS[kind, name] = prepare
        return prepare

    return register


def prepare(run: "ir.Run") -> None:
    """Do what the methods in force need done before `run`'s step traces
    (`before`). Call it inside `~bux.parallel.settings.use`."""
    on = active()
    config = on.methods if on else MethodsConfig()
    for kind in REGISTRIES:
        name = getattr(config, kind)
        if isinstance(name, str) and (preparation := PREPARATIONS.get((kind, name))):
            preparation(run)


def method(kind: str) -> Method:
    """Return the run's method for `kind`, or the default one outside
    `~bux.parallel.settings.use`.

    Args:
        kind: A field of ``methods`` that names a method: a key of `REGISTRIES`.

    Raises:
        NotImplementedError: If no method is registered under the name.

    Example::

        attend = parallel.method("attention")
        out = attend(q, k, v, args)
    """
    return REGISTRIES[kind].find(choice(kind))


def choice(kind: str) -> Any:
    """Return the run's setting of the ``methods`` field `kind`, a method's name or
    another choice, or its default outside `~bux.parallel.settings.use`.

    Args:
        kind: A field of ``methods``.

    Example::

        recompute = parallel.choice("states") == "recompute"
    """
    on = active()
    return getattr(on.methods if on else MethodsConfig(), kind)


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
