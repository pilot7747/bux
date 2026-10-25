"""The layer registry, the context builders get, and the taps modules are called with.

A layer's module comes from the builder registered for its config's ``type``
(`register`), which gets the config, the width the layer reads, its ports and a
`Context`. The module is called as ``module(x, taps)``, and reads what else the
forward pass carries from its `Taps`: the documents in the rows (`segments`,
`positions`), and whether it is a training step's forward (`training`).

A layer adds a term to the training loss from its forward, in one line::

    report(taps, "balance_loss", weight, value)  # value: a scalar array

The objective adds ``weight * value`` to the loss, and logs the values, summed over
the layers that report the name, as a metric of that name. Only a training step's
forward (``Model.training_heads``) collects them; elsewhere, eval included, `report`
does nothing. ``training(taps)`` says which forward it is, for a layer that skips a
costly loss outside training, or updates state of its own as it trains (an MoE's
router bias).
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

from bux.architecture.common.segments import Segments
from bux.lang import ir

if TYPE_CHECKING:
    # Quoted where used: jaxtyping, which checks annotations as the tests run, reads
    # them as a module loads, and embedding.py imports this one.
    from bux.architecture.layers.embedding import Embedding
    from bux.architecture.logits import HeadOutput

#: What a module takes and passes on. The input tokens go into the first, then the
#: stream, ``b t d``, or ``b t n d`` widened by a residual algebra.
Stream = Shaped[Array, "*_"]


class Loss(NamedTuple):
    """A term a layer adds to the training loss: ``weight * value``, logged as `name`."""

    name: str  #: The metric the value is logged as, summed over the layers.
    weight: float  #: What the value is multiplied by in the loss.
    value: Float[Array, ""]  #: The term, a scalar.


class Taps(dict[str, list]):
    """The tap stacks of one forward pass, by name, and what else the pass carries.

    ``@name`` pushes the stream onto the stack `name`, and ``$name`` reads its top
    (``$name!`` pops it); a layer's ports push and read them too
    (`bux.architecture.common.ports`). A forward makes one, and every module is
    called with it, ``module(x, taps)``.

    Args:
        stacks: The stacks to start from, by tap name.
        tokens: The pass's input token ids, b rows of t, which layers that look at
            tokens (nextn, engram) read.
        segments: The documents packed into the rows; None when each row is one.
        training: Whether this is a training step's forward, which collects losses.

    Attributes:
        tokens: The pass's input token ids.
        segments: The documents packed into the rows, or None.
        training: Whether this is a training step's forward.
        losses: The losses layers reported (`report`), in the order they ran.
        heads: The output of every head run so far, in order
            (`~bux.architecture.logits.HeadOutput`).
        nextn_runs: How many nextn layers have run: the next looks one token
            further ahead.
        pending_read: The read coefficients, ``b t n``, the last single-pass hc or
            mhc branch computed for the next one; None before the first.
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
        self.heads: list[HeadOutput] = []
        self.nextn_runs = 0
        self.pending_read: Float[Array, "b t n"] | None = None


def training(taps: Taps | None) -> TypeGuard[Taps]:
    """Return whether this is a training step's forward, which collects losses.

    A layer asks it to skip a costly loss outside training, or to update state of
    its own only as it trains, as an MoE does its router bias.

    Args:
        taps: The forward pass's taps; None outside a model's forward.
    """
    return taps is not None and taps.training


def report(
    taps: Taps | None, name: str, weight: float, value: Float[Array, ""]
) -> None:
    """Add ``weight * value`` to the training loss, and `value` to the metric `name`.

    Only a training step's forward collects the term; elsewhere, eval included, this
    does nothing. A name may not be one of the objective's own terms (``loss``,
    ``z_loss``, ``mtp``): the objective raises if it is.

    Args:
        taps: The forward pass's taps; None outside a model's forward.
        name: The metric to log the value as, summed over the layers that report it.
        weight: What the value is multiplied by in the loss.
        value: The term, a scalar.

    Example:
        An MoE layer's balance loss::

            report(taps, "balance_loss", config.aux_loss_weight, loss)
    """
    if training(taps):
        taps.losses.append(Loss(name, weight, value))


def segments(taps: Taps | None) -> Segments | None:
    """Return the documents packed into the rows; None when each row is one document.

    Args:
        taps: The forward pass's taps; None outside a model's forward.
    """
    return None if taps is None else taps.segments


def positions(taps: Taps | None) -> Int[Array, "b t"] | None:
    """Return each token's position in its document; None when each row is one.

    Args:
        taps: The forward pass's taps; None outside a model's forward.
    """
    s = segments(taps)
    return None if s is None else s.positions


@dataclass
class Built:
    """What the layers built so far leave for the ones built after them.

    Attributes:
        count: How many layers of each config type have been built.
        embedding: The model's first embedding, whose table tied heads and nextn
            read.
        primes: The table sizes engram layers have taken, so that no two tables are
            alike.
    """

    count: Counter[str] = field(default_factory=Counter)  # by layer type
    embedding: "Embedding | None" = None  # the first; tied heads and nextn read it
    primes: set[int] = field(default_factory=set)  # engram table sizes taken


@dataclass
class Context:
    """What a builder gets besides its layer's config, width and ports.

    One context goes through the whole build of a model, so a builder can read what
    the builders before it left (`built`).

    Attributes:
        arch: The whole architecture: ``d_model``, ``vocab_size``, the residual
            algebra and the graph.
        rngs: The random streams the initial parameters are drawn from.
        depth: How many residual branches are around the layer being built: 0 for
            one outside any branch, where a residual algebra's widened stream is.
        built: What the layers built so far left for later ones.
        shared: `~bux.architecture.graph.build_node`'s bookkeeping: the module
            built for each share key.
        branches: How many residual branches outside any other have been built: the
            residual algebra numbers its branches by it.
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
    ``"tokens"`` for the embedding, or ``"streams"`` for a stream widened by the
    residual algebra), `ports` the taps it reads and writes, and `ctx` the
    `Context`. It returns the module, which is called as ``module(x, taps)``.
    """

    def __call__(
        self, config: Any, width: Any, ports: tuple[ir.Port, ...], ctx: Context, /
    ) -> nnx.Module: ...


_REGISTRY: dict[str, Builder] = {}


def register(*types: str) -> Callable[[Builder], Builder]:
    """Register a builder for the layer config types `types`, replacing any before it.

    A layer module registers its builders when it is imported, and
    `bux.architecture` imports all of bux's own; a module of yours must be imported
    before the model is built. Registering a type bux has replaces bux's builder, so
    another implementation of a layer plugs in as a new layer does.

    Args:
        types: The config ``type`` tags the builder builds, as in
            `bux.config.LayerConfig`.

    Returns:
        A decorator that registers the builder and returns it unchanged.

    Example:
        A builder of a new layer, whose config ``DiffAttentionConfig`` is in
        `bux.config.LayerConfig` under the tag ``diffattn``::

            from bux.architecture import Context, register


            @register("diffattn")
            def _(c: DiffAttentionConfig, width: int, ports: tuple, ctx: Context):
                return DiffAttention(c, width, rngs=ctx.rngs)
    """

    def decorate(builder: Builder) -> Builder:
        for t in types:
            _REGISTRY[t] = builder
        return builder

    return decorate


def build_layer(layer: ir.Layer, ctx: Context) -> nnx.Module:
    """Build `layer` with the builder registered for its config's type.

    Raises:
        NotImplementedError: No builder is registered for the type.
    """
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
