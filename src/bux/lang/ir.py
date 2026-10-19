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
knows its input ``width`` (an int, or "tokens" for embeddings). Nodes with the same
``share`` key are a single module. ``Residual`` is kept distinct from ``_ + body`` so
that backends can swap the residual algebra (e.g. hyper-connections) for it alone.
"""

from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from bux.config import LayerConfig, ResidualConfig

Width = int | Literal["tokens", "logits"]


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
        """All leaves in execution order, with repeats unrolled."""
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
    """Pre-order traversal of the tree as written (repeats are visited once)."""
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
    """The node without display labels and/or inferred widths, for structural comparison.

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
