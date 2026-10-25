"""What the type of a training config means to the trainer: the builder registered
for it.

Configs are plain data (`bux.train.config`); the code that does the work is found
by their ``type`` tag, as a layer's is (`bux.architecture.registry.register`). A
builder registered under a taken tag replaces the one there, so a variant of an
optimizer is a function, not a fork::

    from bux.train.optim import optimizers

    @optimizers.register("adamw")
    def _(c: AdamWConfig, multiplier: Schedule, model: nnx.Module):
        return optax.adamw(lambda step: c.lr * multiplier(step), b1=c.b1, b2=c.b2)

A new tag needs its config in `bux.train.config` too, where the compiler finds it.
The registries of training are `bux.train.optim.schedules`,
`bux.train.optim.optimizers`, `bux.train.optim.ns_coefficients` and
`bux.train.data.sources`; `bux.parallel.methods` holds one per kind of method.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tagged(Protocol):
    """A config with a ``type`` tag, which names its builder."""

    @property
    def type(self) -> str:
        """The tag a ``.bux`` file calls the config by."""
        ...


class Registry[T]:
    """Builders of `T`, by the type of the config they build from.

    Args:
        kind: What it builds, in words, for its errors: ``"optimizer"``.
        where: Its import path, for its errors: ``"bux.train.optim.optimizers"``.

    Attributes:
        builders: Each builder by the tag it is registered under.

    Example::

        schedules = Registry[Schedule]("schedule", "bux.train.optim.schedules")

        @schedules.register("constant")
        def _(c: ConstantSchedule, steps: int) -> Schedule:
            return lambda step: jnp.asarray(1.0)

        curve = schedules.build(ConstantSchedule(), 1000)
    """

    def __init__(self, kind: str, where: str):
        self.kind, self.where = kind, where  # what it builds, and its import path
        self.builders: dict[str, Callable[..., T]] = {}

    def register[B: Callable[..., Any]](self, *types: str) -> Callable[[B], B]:
        """Return a decorator that registers a builder under each of `types`,
        replacing what was there, and returns the builder as it is.

        Args:
            *types: The tags it builds from.
        """

        def decorate(builder: B) -> B:
            for t in types:
                self.builders[t] = builder
            return builder

        return decorate

    def find(self, tag: str) -> Callable[..., T]:
        """Return the builder registered under `tag`.

        Raises:
            NotImplementedError: If none is, with how to register one.
        """
        builder = self.builders.get(tag)
        if builder is None:
            raise NotImplementedError(
                f"the trainer has no {tag!r} {self.kind} yet; "
                f"add one with @{self.where}.register({tag!r})"
            )
        return builder

    def build(self, config: Tagged, *args: Any) -> T:
        """Return what `config` builds: its builder's result, given `config` and the
        builder's other arguments, `args`.

        Raises:
            NotImplementedError: If no builder is registered under its ``type``.
        """
        return self.find(config.type)(config, *args)
