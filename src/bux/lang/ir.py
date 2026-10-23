"""The architecture IR: what ``bux.lang.compile`` produces and backends consume.

A graph is a tree of nodes, each a function of the incoming stream ``x``:

    Layer       config(x)                      a leaf from ``bux.config``
    Seq         items[-1](...items[0](x))      composition, left to right
    Residual    x + body(x)                    a residual branch, ``[body]``
    Sum         sum(t(x) for t in terms)
    Prod        prod(f(x) for f in factors)    gating, or scaling by a Const
    Neg         -body(x)
    Const       value                          a scalar, only inside Sum/Prod
    Identity    x                              ``_``
    Repeat      body applied `times` times     fresh parameters each time
    Tap         x, and pushes x onto the tap   ``@name``
    Ref         top of the tap, ignoring x     ``$name``; ``$name!`` also pops it

Branches of Sum/Prod run left to right, which only matters for taps. Every Layer
knows its input ``width``: an int, "tokens" for embeddings, or "streams" for layers
that act on a residual stream widened by the residual algebra. Nodes with the same
``share`` key are a single module. ``Residual`` is kept distinct from ``_ + body`` so
that backends can swap the residual algebra (e.g. hyper-connections) for it alone.
"""

import hashlib
import json
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from bux.config import LayerConfig, ResidualConfig
from bux.train.config import PrecisionConfig, TrainConfig

# "streams" is a residual stream widened by the residual algebra (hc, mhc, gr, attnres).
Width = int | Literal["tokens", "logits", "streams"]


class _Node(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Name of the definition the node came from, for printing and diagrams.
    label: str | None = None
    share: str | None = None


class Port(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tap: str
    mode: Literal["write", "read", "pop"]


class Layer(_Node):
    kind: Literal["layer"] = "layer"
    config: LayerConfig
    width: Width | None = None
    ports: tuple[Port, ...] = ()


class Seq(_Node):
    kind: Literal["seq"] = "seq"
    items: tuple[Node, ...]


class Residual(_Node):
    kind: Literal["residual"] = "residual"
    body: Node


class Sum(_Node):
    kind: Literal["sum"] = "sum"
    terms: tuple[Node, ...]


class Prod(_Node):
    kind: Literal["prod"] = "prod"
    factors: tuple[Node, ...]


class Neg(_Node):
    kind: Literal["neg"] = "neg"
    body: Node


class Const(_Node):
    kind: Literal["const"] = "const"
    value: float


class Identity(_Node):
    kind: Literal["identity"] = "identity"


class Repeat(_Node):
    kind: Literal["repeat"] = "repeat"
    body: Node
    times: int


class Tap(_Node):
    kind: Literal["tap"] = "tap"
    name: str


class Ref(_Node):
    kind: Literal["ref"] = "ref"
    name: str
    pop: bool = False


Node = Annotated[
    Layer | Seq | Residual | Sum | Prod | Neg | Const | Identity | Repeat | Tap | Ref,
    Field(discriminator="kind"),
]

NODE_TYPES = (Layer, Seq, Residual, Sum, Prod, Neg, Const, Identity, Repeat, Tap, Ref)

for _cls in NODE_TYPES:
    _cls.model_rebuild()


class Architecture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    d_model: int
    vocab_size: int | None = None
    # None: plain additive residuals.
    residual: ResidualConfig | None = None
    graph: Node

    def layers(self) -> Iterator[Layer]:
        """Return every layer in execution order, with repeats unrolled."""
        return unrolled_layers(self.graph)


def children(node: Node) -> tuple[Node, ...]:
    match node:
        case Seq(items=items):
            return items
        case Sum(terms=items) | Prod(factors=items):
            return items
        case Residual(body=body) | Neg(body=body) | Repeat(body=body):
            return (body,)
        case _:
            return ()


def walk(node: Node) -> Iterator[Node]:
    """Yield the nodes of the tree as written, in pre-order; a repeat's body once."""
    yield node
    for child in children(node):
        yield from walk(child)


def unrolled_layers(node: Node) -> Iterator[Layer]:
    if isinstance(node, Layer):
        yield node
    elif isinstance(node, Repeat):
        for _ in range(node.times):
            yield from unrolled_layers(node.body)
    else:
        for child in children(node):
            yield from unrolled_layers(child)


def replace_children(node: Node, new: tuple[Node, ...]) -> Node:
    match node:
        case Seq():
            return node.model_copy(update={"items": new})
        case Sum():
            return node.model_copy(update={"terms": new})
        case Prod():
            return node.model_copy(update={"factors": new})
        case Residual() | Neg() | Repeat():
            return node.model_copy(update={"body": new[0]})
        case _:
            return node


def strip(node: Node, *, labels: bool = True, widths: bool = True) -> Node:
    """Return the node without labels, inferred widths or both, to compare structure.

    Without labels, a sequence directly inside a sequence (a sum inside a sum, a product
    inside a product) is flattened into it, as the compiler does with unlabeled groups.
    """
    update: dict[str, object] = {}
    if labels:
        update["label"] = None
    if widths and isinstance(node, Layer):
        update["width"] = None
    kids = tuple(strip(c, labels=labels, widths=widths) for c in children(node))
    if labels and isinstance(node, Seq | Sum | Prod):
        same = type(node)
        kids = tuple(
            k
            for c in kids
            for k in (children(c) if type(c) is same and c.share is None else (c,))
        )
    node = replace_children(node, kids)
    return node.model_copy(update=update)


# Training settings that change how a run is watched or computed, not what it computes.
_UNCOMPUTED = {"log", "eval", "checkpoint", "mesh", "layout", "methods"}
_NEWER = {"precision": PrecisionConfig()}
# Fields of a setting newer than run ids, by what runs computed before them.
_NEWER_FIELDS = {"optimizer": {"ns_coefficients": "jordan", "ns_dtype": "float32"}}


def _without_newer_fields(train: dict) -> dict:
    """`train`, dumped, without the `_NEWER_FIELDS` that compute as runs did before."""
    for setting, fields in _NEWER_FIELDS.items():
        given = train.get(setting) or {}
        for field, before in fields.items():
            if given.get(field) == before:
                del given[field]
    return train


class Run(BaseModel):
    """One training run: an architecture and how to train it, with every setting resolved.

    This is what a run's directory keeps, next to the same as .bux source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: (
        str  # the file, and the alternatives it took: speedrun[mqa, optimizer.lr=0.01]
    )
    arch: Architecture
    train: TrainConfig
    # One per axis of the sweep, in the order written: "mqa", "optimizer.lr=0.01".
    choices: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """A hash of what the run computes, not of how it is written, watched or split
        over devices. Renaming a definition, reaching the same settings through a sweep,
        logging, evaluating or checkpointing at other intervals, or another mesh, layout
        or methods keep it."""
        arch = self.arch.model_copy(update={"graph": strip(self.arch.graph)})
        # A setting or a field newer than run ids is left out where it computes as runs
        # did before it (a setting at its default), so that ids made before it resume.
        new = {k for k, v in _NEWER.items() if getattr(self.train, k) == v}
        dumped = self.train.model_dump(mode="json", exclude=_UNCOMPUTED | new)
        train = _without_newer_fields(dumped)
        what = {"arch": arch.model_dump(mode="json"), "train": train}
        blob = json.dumps(what, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:10]
