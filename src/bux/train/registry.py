"""What the type of a training config means to the trainer: the builder registered
for it.

Configs are plain data (``bux.train.config``); the code that does the work is found
by their ``type`` tag, as a layer's is (``bux.architecture.register``). A builder
registered under a taken tag replaces the one there, so a variant of an optimizer is
a function, not a fork:

    from bux.train.optim import optimizers

    @optimizers.register("adamw")
    def _(c: AdamWConfig, multiplier: Schedule, model: nnx.Module):
        return optax.adamw(lambda step: c.lr * multiplier(step), b1=c.b1, b2=c.b2)

A new tag needs its config in ``bux.train.config`` too, where the compiler finds it.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tagged(Protocol):
    @property
    def type(self) -> str: ...


class Registry[T]:
    """Builders of `T`, by the type of the config they build from."""

    def __init__(self, kind: str, where: str):
        self.kind, self.where = kind, where  # what it builds, and its import path
        self.builders: dict[str, Callable[..., T]] = {}

    def register[B: Callable[..., Any]](self, *types: str) -> Callable[[B], B]:
        def decorate(builder: B) -> B:
            for t in types:
                self.builders[t] = builder
            return builder

        return decorate

    def build(self, config: Tagged, *args: Any) -> T:
        """What `config` builds, given the builder's other arguments."""
        builder = self.builders.get(config.type)
        if builder is None:
            raise NotImplementedError(
                f"the trainer has no {config.type!r} {self.kind} yet; "
                f"add one with @{self.where}.register({config.type!r})"
            )
        return builder(config, *args)
