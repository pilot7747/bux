"""The leaf registry, the context builders get, and the taps modules are called with.

A layer adds a term to the training loss from its forward, in one line:

    report(taps, "balance_loss", weight, value)  # value: a scalar array

The objective adds weight * value to the loss, and logs the values, summed over the
layers that report the name, as a metric of that name. Only a training step's forward
(``Model.training_outputs``) collects them; elsewhere, eval included, ``report`` does
nothing. ``training(taps)`` says which forward it is, for a layer that skips a costly
loss outside training, or updates state of its own as it trains (an MoE's router bias).
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeGuard

from flax import nnx
from jaxtyping import Array, Float, Int, Shaped

from bux.architecture.segments import Segments
from bux.lang import ir

if TYPE_CHECKING:
    # Quoted where used: jaxtyping, which checks annotations as the tests run, reads
    # them as a module loads, and embedding.py imports this one.
    from bux.architecture.embedding import Embedding

# What a module takes and passes on: the input tokens into the first, then the stream.
Stream = Shaped[Array, "*_"]
Taps = dict[str, list]
# Reserved tap keys, which no tap name can collide with.
TOKENS, SEGMENTS, OUTPUTS, NEXTN = "<tokens>", "<segments>", "<outputs>", "<nextn>"
LOSSES = "<losses>"


class Loss(NamedTuple):
    """A term a layer adds to the training loss: weight * value, logged as `name`."""

    name: str
    weight: float
    value: Float[Array, ""]


def training(taps: Taps | None) -> TypeGuard[Taps]:
    """Whether this forward is a training step's, which collects reported losses."""
    return taps is not None and LOSSES in taps


def report(
    taps: Taps | None, name: str, weight: float, value: Float[Array, ""]
) -> None:
    """Add weight * value to the training loss, and value to the metric `name`."""
    if training(taps):
        taps[LOSSES].append(Loss(name, weight, value))


def segments(taps: Taps | None) -> Segments | None:
    """The documents of the rows; None when every row is one document."""
    return taps[SEGMENTS][0] if taps and SEGMENTS in taps else None


def positions(taps: Taps | None) -> Int[Array, "b t"] | None:
    """Where each token is in its document; None when every row is one document."""
    s = segments(taps)
    return None if s is None else s.positions


@dataclass
class Context:
    arch: ir.Architecture
    rngs: nnx.Rngs
    shared: dict[str, nnx.Module] = field(default_factory=dict)
    embedding: "Embedding | None" = None
    built: Counter[str] = field(default_factory=Counter)  # layers so far, by type
    primes: set[int] = field(default_factory=set)  # engram table sizes taken
    depth: int = 0  # residual branches the node being built is inside
    branches: int = 0  # branches the residual algebra has built so far


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
    module = builder(config, layer.width, layer.ports, ctx)
    ctx.built[config.type] += 1
    return module
