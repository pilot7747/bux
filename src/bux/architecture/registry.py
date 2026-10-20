"""The leaf registry, the context builders get, and the taps modules are called with."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flax import nnx
from jaxtyping import Array, Int, Shaped

from bux.lang import ir

if TYPE_CHECKING:
    # Quoted where used: jaxtyping, which checks annotations as the tests run, reads
    # them as a module loads, and embedding.py imports this one.
    from bux.architecture.embedding import Embedding

# What a module takes and passes on: the input tokens into the first, then the stream.
Stream = Shaped[Array, "*_"]
Taps = dict[str, list]
# Reserved tap keys, which no tap name can collide with.
TOKENS, POSITIONS, OUTPUTS, NEXTN = "<tokens>", "<positions>", "<outputs>", "<nextn>"


def positions(taps: Taps | None) -> Int[Array, "b t"] | None:
    """Where each token is in its document, so a 0 starts one; None when every row is
    one document."""
    return taps[POSITIONS][0] if taps and POSITIONS in taps else None


@dataclass
class Context:
    arch: ir.Architecture
    rngs: nnx.Rngs
    shared: dict[str, nnx.Module] = field(default_factory=dict)
    embedding: "Embedding | None" = None
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
