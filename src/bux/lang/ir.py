"""The architecture IR: what `bux.lang.compile` produces and backends consume.

An `Architecture` is a model's settings and its graph. A graph is a tree of nodes,
each a function of the incoming stream ``x``::

    Layer       config(x)                      a leaf from bux.config
    Seq         items[-1](...items[0](x))      composition, left to right
    Residual    x + body(x)                    a residual branch, [body]
    Sum         sum(t(x) for t in terms)
    Prod        prod(f(x) for f in factors)    gating, or scaling by a Const
    Neg         -body(x)
    Const       value                          a scalar, only inside Sum/Prod
    Identity    x                              _
    Repeat      body applied `times` times     fresh parameters each time
    Tap         x, and pushes x onto the tap   @name
    Ref         top of the tap, ignoring x     $name; $name! also pops it

Branches of `Sum` and `Prod` run left to right, which only matters for taps. Every
`Layer` knows its input `width`: an int, ``"tokens"`` for embeddings, or
``"streams"`` for layers that act on a residual stream widened by the residual
algebra. ``Residual`` is kept distinct from ``_ + body`` so that backends can swap
the residual algebra (e.g. hyper-connections) for it alone.

Every node also has a ``label``, the name of the definition it came from (for
printing and diagrams), and a ``share`` key: nodes with the same key are a single
module, with one set of parameters (``shared block = ...``). Nodes are frozen
pydantic models, so a graph compares, hashes and serializes as data
(``arch.model_dump_json()``, what ``bux json`` prints), and `walk`, `strip` and
`replace_children` take one apart and put it back together.
"""

import hashlib
import json
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bux.config import FanInitConfig, Init, LayerConfig, ResidualConfig
from bux.train.config import PrecisionConfig, TrainConfig

#: What a layer reads, as the compiler infers it.
#:
#: A width, ``"tokens"`` (ids, for an embedding), ``"logits"`` (after the LM head,
#: which nothing follows), or ``"streams"``, a residual stream widened by the
#: residual algebra (hc, mhc, gr, attnres).
Width = int | Literal["tokens", "logits", "streams"]


class _Node(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Name of the definition the node came from, for printing and diagrams.
    label: str | None = None
    share: str | None = None


class Port(BaseModel):
    """A layer's internal tensor, sent through a tap: ``gqa(v=@v1)``, ``gqa(v=$v1)``.

    Attributes:
        name: The port, one of the layer config's ``ports``: ``"v"``, ``"kv"``,
            ``"index"``, ...
        tap: The tap it goes through.
        mode: ``"write"`` exports the tensor (``@tap``), ``"read"`` imports it
            (``$tap``), and ``"pop"`` imports it and pops the tap (``$tap!``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tap: str
    mode: Literal["write", "read", "pop"]


class Layer(_Node):
    """A leaf: one layer, ``config(x)``.

    Attributes:
        config: The layer's settings, one of `bux.config.LayerConfig`.
        width: What the layer reads, as the compiler inferred it (`Width`).
        ports: The internal tensors it exports or imports through taps.
    """

    kind: Literal["layer"] = "layer"
    config: LayerConfig
    width: Width | None = None
    ports: tuple[Port, ...] = ()


class Seq(_Node):
    """Composition, left to right: ``a b c`` is ``c(b(a(x)))``."""

    kind: Literal["seq"] = "seq"
    items: tuple[Node, ...]


class Residual(_Node):
    """A residual branch, ``[body]``: ``x + body(x)``, or the residual algebra's."""

    kind: Literal["residual"] = "residual"
    body: Node


class Sum(_Node):
    """Parallel branches, summed: ``a + b``, and ``a - b`` with a `Neg`."""

    kind: Literal["sum"] = "sum"
    terms: tuple[Node, ...]


class Prod(_Node):
    """Parallel branches, multiplied elementwise: ``a * b`` (gating), or ``0.5 * a``."""

    kind: Literal["prod"] = "prod"
    factors: tuple[Node, ...]


class Neg(_Node):
    """Negation, ``-a``: ``-body(x)``."""

    kind: Literal["neg"] = "neg"
    body: Node


class Const(_Node):
    """A scalar `value`, only as a term of a `Sum` or a factor of a `Prod`."""

    kind: Literal["const"] = "const"
    value: float


class Identity(_Node):
    """The stream as it is, ``_``: ``x``."""

    kind: Literal["identity"] = "identity"


class Repeat(_Node):
    """``body^times``: `times` copies of `body` in sequence, each with its own
    parameters (unless `body` is shared)."""

    kind: Literal["repeat"] = "repeat"
    body: Node
    times: int


class Tap(_Node):
    """``@name``: pushes the stream onto the tap `name`, and passes it on."""

    kind: Literal["tap"] = "tap"
    name: str


class Ref(_Node):
    """``$name``: the top of the tap `name`, ignoring the stream; ``$name!`` also
    pops it (`pop`)."""

    kind: Literal["ref"] = "ref"
    name: str
    pop: bool = False


#: Any node of a graph, told apart by its ``kind`` when it is read from JSON.
Node = Annotated[
    Layer | Seq | Residual | Sum | Prod | Neg | Const | Identity | Repeat | Tap | Ref,
    Field(discriminator="kind"),
]

#: Every class of node, for ``isinstance(value, NODE_TYPES)``.
NODE_TYPES = (Layer, Seq, Residual, Sum, Prod, Neg, Const, Identity, Repeat, Tap, Ref)

for _cls in NODE_TYPES:
    _cls.model_rebuild()


class Architecture(BaseModel):
    """A compiled model: its top-level settings and its graph.

    `bux.lang.compile` returns one, and a backend builds it:
    ``bux.architecture.Model(arch, rngs=nnx.Rngs(0))``. It is frozen pydantic data,
    so it round-trips through JSON (``Architecture.model_validate_json``) and prints
    back to source (`bux.lang.printer.to_source`).

    Attributes:
        d_model: The width of the residual stream.
        vocab_size: The number of token ids that embeddings and heads take; None for
            a graph with neither, such as a block compiled alone.
        residual: The residual algebra every `Residual` merges by; None for plain
            additive residuals.
        init: How the weight matrices start, by role (`bux.config.Init`); None
            for ``normal(std=0.02)``, but where a layer starts one its own way
            (`bux.initializers`).
        graph: The model, a tree of nodes.

    Example::

        arch = compile(source)
        [layer.config.type for layer in arch.layers()][:3]
        # ['embed', 'rmsnorm', 'gqa']
        [layer.width for layer in arch.layers()][:3]  # ['tokens', 64, 64]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    d_model: int
    vocab_size: int | None = None
    residual: ResidualConfig | None = None
    init: Init | None = None
    graph: Node

    def layers(self) -> Iterator[Layer]:
        """Return every layer in execution order, with repeats unrolled."""
        return unrolled_layers(self.graph)


def children(node: Node) -> tuple[Node, ...]:
    """Return the nodes directly inside `node`, in order; none for a leaf."""
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
    """Yield every layer under `node` in execution order, a repeat's body each time."""
    if isinstance(node, Layer):
        yield node
    elif isinstance(node, Repeat):
        for _ in range(node.times):
            yield from unrolled_layers(node.body)
    else:
        for child in children(node):
            yield from unrolled_layers(child)


def replace_children(node: Node, new: tuple[Node, ...]) -> Node:
    """Return a copy of `node` with `new` in place of its `children`, in their order;
    a leaf as it is. With `children`, it rebuilds a tree bottom up."""
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
_UNCOMPUTED = {"log", "eval", "checkpoint", "observe", "profile"}
_UNCOMPUTED |= {"mesh", "layout", "methods", "compiler"}  # how it splits and compiles
_UNCOMPUTED |= {"objective", "search"}  # how a search ranks and proposes runs
# Settings newer than run ids, as runs computed before them: in float32, which was
# the default then.
_NEWER = {"precision": PrecisionConfig(compute="float32")}
# Settings newer than run ids that are not configs, at the values runs computed with
# before them, dumped.
_NEWER_VALUES = {"batch_ramp": None, "curriculum": []}
# Fields of a setting newer than run ids, by what runs computed before them.
_NEWER_FIELDS = {
    "precision": {
        "matmuls": "compute",
        "reduce": "compute",
        **PrecisionConfig().model_dump(
            mode="json", include={"roles", "first", "last", "passes", "switch"}
        ),
        "moments": "float32",
        "dispatch": "compute",
    },
    "optimizer": {
        "ns_coefficients": "jordan",
        "ns_dtype": "float32",
        "ns_algorithm": "standard",
        "ns_restarts": None,
    },
    "schedule": {"horizon": 1.0},
}
# Fields of a setting that only change how a run is watched.
_UNCOMPUTED_FIELDS = {"precision": ("eval",)}
# Fields of a layer's settings, or of the model's init, newer than run ids, by its
# type, at what they computed before them.
# HyGA's gates (QuadraticAttentionConfig's gate_input_rank, ...), unset.
_HYGA = {"gate_input_rank": None, "gate_output_rank": None, "gate_cross_head": False}
_NEWER_LAYER_FIELDS = {
    "mlp": {"up_offset": 0.0},
    "gated_mlp": {"up_offset": 0.0},
    "moe": {
        "router_bias": False,
        "center_bias": False,
        "bias_softclamp": 2.0,
        "bias_momentum": 0.5,
        "num_zero_experts": 0,
        "ffn_experts_per_token": None,
    },
    "mamba2": {"num_heads": None},
    "gqa": {
        "gate_head_bias": False,
        "power_scale": None,
        "power_scale_length": 2048,
        "qk_norm_all_heads": False,
        "exclusive": False,
        **_HYGA,
        "value_embed_gate": 6,
        "k_eq_v": False,
    },
    "mla": {"gate_head_bias": False, **_HYGA},
    "csa": {"gate_head_bias": False, **_HYGA},
    "indexer": {"pool": "mean", "per_group": False, "relu": True},
    "layernorm": {"gain_init": 1.0},
    "rmsnorm": {"gain_init": 1.0},
    "normal": {"embed": None, "head": None, "out": None},  # the model's init
}
# How every weight matrix started before the model's `init` was a normal, and before
# it could be set, as flax's layers start them: runs that set it keep their ids. But
# routed experts', which started sqrt(E) times smaller, so that an MoE's id is new.
_INIT_BEFORE = FanInitConfig(
    type="fan_in", embed=FanInitConfig(type="fan_in", distribution="gaussian")
)


def _starts_as_before(arch: "Architecture") -> bool:
    """Whether `arch`'s weights start as they did before its ``init`` could be set
    (`_INIT_BEFORE`)."""
    moe = any(layer.config.type == "moe" for layer in arch.layers())
    return arch.init == _INIT_BEFORE and not moe


def _experts_as_they_run(
    arch: "Architecture", config: TrainConfig, train: dict
) -> dict:
    """`train`, dumped, with the format a model's routed experts multiply in where
    `roles` leaves it to `matmuls` (``PrecisionConfig.experts``), as they did not
    before: so that a run whose experts now take its matmuls' format has an id of its
    own, and a model without them keeps its own."""
    if any(layer.config.type == "moe" for layer in arch.layers()):
        train["precision"]["roles"]["experts"] = config.precision.experts()
    return train


def _without_newer_fields(train: dict) -> dict:
    """`train`, dumped, without the `_NEWER_FIELDS` that compute as runs did before,
    nor the `_UNCOMPUTED_FIELDS`."""
    for setting, fields in _NEWER_FIELDS.items():
        given = train.get(setting) or {}
        for field, before in fields.items():
            if field in given and given[field] == before:
                del given[field]
    for setting, names in _UNCOMPUTED_FIELDS.items():
        for name in names:
            (train.get(setting) or {}).pop(name, None)
    return train


def _without_newer_layer_fields(node: Any) -> Any:
    """`node`, part of an architecture dumped, without the `_NEWER_LAYER_FIELDS` that
    compute as layers did before, at any depth: an MoE's experts are inside it."""
    if isinstance(node, list):
        return [_without_newer_layer_fields(n) for n in node]
    if not isinstance(node, dict):
        return node
    before = _NEWER_LAYER_FIELDS.get(node.get("type"), {})
    return {
        k: _without_newer_layer_fields(v)
        for k, v in node.items()
        if not (k in before and v == before[k])
    }


class Run(BaseModel):
    """One training run: an architecture and how to train it, with every setting resolved.

    `bux.lang.compile_runs` returns one per combination of a sweep's alternatives,
    and `bux.train.trainer.train` trains one into a directory named for its name and
    `id`, ``runs/smoke-5a4f0f7d85/``. The directory keeps it as ``run.json``, and as
    ``run.bux`` (`bux.lang.printer.run_to_source`), which compiles back to the same
    run.

    Attributes:
        name: The file, and the alternatives it took:
            ``speedrun[mqa, optimizer.lr=0.01]``.
        arch: The model.
        train: How to train it: data, optimizer, schedule, steps, mesh, ...
        choices: The alternative taken on each axis of the sweep, in the order
            written: ``("mqa", "optimizer.lr=0.01")``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    arch: Architecture
    train: TrainConfig
    choices: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """A hash of what the run computes, not of how it is written, watched or split
        over devices: ten hex digits, such as ``"5a4f0f7d85"``.

        Renaming a definition, reaching the same settings through a sweep, logging,
        evaluating, checkpointing or profiling at other steps, or another mesh,
        layout, methods or compiler options keep it. So does a setting newer than run
        ids, where it computes as runs did before it, so that their checkpoints
        resume."""
        arch = self.arch.model_copy(update={"graph": strip(self.arch.graph)})
        # A setting or a field newer than run ids is left out where it computes as runs
        # did before it, so that ids made before it resume.
        dumped = self.train.model_dump(mode="json", exclude=_UNCOMPUTED)
        dumped = _experts_as_they_run(self.arch, self.train, dumped)
        train = _without_newer_fields(dumped)
        for setting, before in _NEWER.items():
            old = _without_newer_fields({setting: before.model_dump(mode="json")})
            if train[setting] == old[setting]:
                del train[setting]
        for setting, value in _NEWER_VALUES.items():
            if train[setting] == value:
                del train[setting]
        dumped_arch = _without_newer_layer_fields(arch.model_dump(mode="json"))
        # Before, an init of None left the key out, and started weights as
        # `_INIT_BEFORE` does: a None now, which is a normal, keeps it.
        if _starts_as_before(self.arch):
            del dumped_arch["init"]
        what = {"arch": dumped_arch, "train": train}
        blob = json.dumps(what, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:10]
