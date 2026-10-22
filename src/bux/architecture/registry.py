"""The layer registry, the context builders get, and the taps modules are called with.

A layer adds a term to the training loss from its forward, in one line:

    report(taps, "balance_loss", weight, value)  # value: a scalar array

The objective adds weight * value to the loss, and logs the values, summed over the
layers that report the name, as a metric of that name. Only a training step's forward
(``Model.training_outputs``) collects them; elsewhere, eval included, ``report`` does
nothing. ``training(taps)`` says which forward it is, for a layer that skips a costly
loss outside training, or updates state of its own as it trains (an MoE's router bias).
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

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


class Loss(NamedTuple):
    """A term a layer adds to the training loss: weight * value, logged as `name`."""

    name: str
    weight: float
    value: Float[Array, ""]


class Taps(dict[str, list]):
    """The tap stacks of one forward pass, by name, and what else the pass carries.

    `@name` pushes the stream onto the stack `name`, and `$name` reads its top. The
    attributes are the pass's input `tokens` and their `segments` (None when every row
    is one document); the `logits` of every head run so far; the `losses` layers
    report, which only a `training` forward collects; how many nextn layers have run
    (`nextn_runs`); and the read coefficients the last single-pass branch computed for
    the next one (`pending_read`).
    """

    def __init__(
        self,
        stacks: Mapping[str, list] | None = None,
        *,
        tokens: Int[Array, "b t"] | None = None,
        segments: Segments | None = None,
        training: bool = False,
    ):
        super().__init__(stacks or {})
        self.tokens, self.segments, self.training = tokens, segments, training
        self.losses: list[Loss] = []
        self.logits: list[Float[Array, "b t vocab"]] = []
        self.nextn_runs = 0
        self.pending_read: Float[Array, "b t n"] | None = None


def training(taps: Taps | None) -> TypeGuard[Taps]:
    """Return whether this is a training step's forward, which collects losses."""
    return taps is not None and taps.training


def report(
    taps: Taps | None, name: str, weight: float, value: Float[Array, ""]
) -> None:
    """Add weight * value to the training loss, and value to the metric `name`."""
    if training(taps):
        taps.losses.append(Loss(name, weight, value))


def segments(taps: Taps | None) -> Segments | None:
    """Return the documents packed into the rows; None when each row is one document."""
    return None if taps is None else taps.segments


def positions(taps: Taps | None) -> Int[Array, "b t"] | None:
    """Return each token's position in its document; None when each row is one."""
    s = segments(taps)
    return None if s is None else s.positions


@dataclass
class Built:
    """What the layers built so far leave for the ones built after them."""

    count: Counter[str] = field(default_factory=Counter)  # by layer type
    embedding: "Embedding | None" = None  # the first; tied heads and nextn read it
    primes: set[int] = field(default_factory=set)  # engram table sizes taken


@dataclass
class Context:
    """What a builder gets besides its layer's config, width and ports.

    `arch` is the whole architecture, and `rngs` draws the initial parameters. `depth`
    counts the residual branches around the layer, and `built` holds what earlier
    layers left for it. `shared` and `branches` are ``build_node``'s bookkeeping: the
    modules built for each share key, and the residual branches built so far.
    """

    arch: ir.Architecture
    rngs: nnx.Rngs
    depth: int = 0
    built: Built = field(default_factory=Built)
    shared: dict[str, nnx.Module] = field(default_factory=dict)
    branches: int = 0


@runtime_checkable
class Builder(Protocol):
    """Build one layer's module: ``builder(config, width, ports, ctx)``.

    `config` is the layer's config, `width` the width of the stream it reads (an int,
    or "streams" for a widened one), `ports` the taps it reads and writes, and `ctx`
    the ``Context``.
    """

    def __call__(
        self, config: Any, width: Any, ports: tuple[ir.Port, ...], ctx: Context, /
    ) -> nnx.Module: ...


_REGISTRY: dict[str, Builder] = {}


def register(*types: str) -> Callable[[Builder], Builder]:
    """Register a builder for the layer config types `types`, replacing any before it.

    A layer module registers its builders when it is imported, and ``bux.architecture``
    imports all of bux's own.
    """

    def decorate(builder: Builder) -> Builder:
        for t in types:
            _REGISTRY[t] = builder
        return builder

    return decorate


def build_layer(layer: ir.Layer, ctx: Context) -> nnx.Module:
    """Build `layer` with the builder registered for its config's type."""
    config = layer.config
    builder = _REGISTRY.get(config.type)
    if builder is None:
        raise NotImplementedError(
            f"the reference backend has no '{config.type}' layer yet; "
            f"add one with @bux.architecture.register({config.type!r})"
        )
    module = builder(config, layer.width, layer.ports, ctx)
    ctx.built.count[config.type] += 1
    return module
