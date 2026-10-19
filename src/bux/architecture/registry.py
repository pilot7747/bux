"""The leaf registry, the context builders get, and the taps modules are called with."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flax import nnx

from bux.lang import ir

if TYPE_CHECKING:
    from bux.architecture.embedding import Embedding

Taps = dict[str, list]
# Reserved tap keys, which no tap name can collide with.
TOKENS, OUTPUTS, NEXTN = "<tokens>", "<outputs>", "<nextn>"


@dataclass
class Context:
    arch: ir.Architecture
    rngs: nnx.Rngs
    shared: dict[str, nnx.Module] = field(default_factory=dict)
    embedding: Embedding | None = None
    engrams: int = 0  # engram layers built so far
    primes: set[int] = field(default_factory=set)  # engram table sizes taken


Builder = Callable[..., nnx.Module]
_REGISTRY: dict[str, Builder] = {}


def register(*types: str) -> Callable[[Builder], Builder]:
    def decorate(builder: Builder) -> Builder:
        for t in types:
            _REGISTRY[t] = builder
        return builder

    return decorate


def build_layer(layer: ir.Layer, ctx: Context) -> nnx.Module:
    config = layer.config
    builder = _REGISTRY.get(config.type)
    if builder is None:
        raise NotImplementedError(
            f"the reference backend has no '{config.type}' layer yet; "
            f"add one with @bux.architecture.register({config.type!r})"
        )
    return builder(config, layer.width, layer.ports, ctx)
