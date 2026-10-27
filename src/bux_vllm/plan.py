"""Where every layer of an exported model goes: `bux.export.plan`'s module tree and
calls, from the IR as ``config.json`` holds it (``arch.model_dump(mode="json")``).

The vLLM model builds its modules at these paths, so that the checkpoint's weights,
named by `bux.export.plan`, load into them. It is plain Python over the IR's JSON,
importing neither bux nor torch; a test checks that it places every layer of every
example architecture where `bux.export.plan` does.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

#: An IR node as JSON: a dict with its ``kind``, ``label`` and ``share``.
Node = dict[str, Any]
#: A node's place in its scope: the child indices from the scope's root to it.
Pos = tuple[int, ...]


def children(node: Node) -> list[Node]:
    """Return the nodes directly inside `node`, in order; none for a leaf."""
    kind = node["kind"]
    if kind == "seq":
        return node["items"]
    if kind == "sum":
        return node["terms"]
    if kind == "prod":
        return node["factors"]
    if kind in ("residual", "neg", "repeat"):
        return [node["body"]]
    return []


def walk(node: Node) -> Iterator[Node]:
    """Yield the nodes of the tree as written, in pre-order; a repeat's body once."""
    yield node
    for child in children(node):
        yield from walk(child)


def at(root: Node, pos: Pos) -> Node:
    """Return the node at `pos` under `root`."""
    for i in pos:
        root = children(root)[i]
    return root


@dataclass
class Scope:
    """The modules of one block: the model's, a repeat's body, or a shared
    definition's (`bux.export.plan.Scope`).

    Attributes:
        root: The node the scope computes.
        key: The shared definition the scope is, if it is one.
        names: The attribute of each layer, repeat and residual algebra's branch in
            it, by its `Pos`.
        blocks: The scope of each repeat's body, by the repeat's `Pos`.
        reserved: The names no module takes, as the export's config lists them
            (``bux_reserved``, `bux.export.plan.RESERVED`).
    """

    root: Node
    key: str | None = None
    names: dict[Pos, str] = field(default_factory=dict)
    blocks: dict[Pos, Scope] = field(default_factory=dict)
    reserved: frozenset[str] = frozenset()

    def name(self, hint: str) -> str:
        """Return a fresh attribute name from `hint`: ``norm``, then ``norm_1``."""
        hint = hint if hint.isidentifier() and not keyword.iskeyword(hint) else "layer"
        taken = set(self.names.values()) | self.reserved
        name, i = hint, 1
        while name in taken:
            name, i = f"{hint}_{i}", i + 1
        return name


@dataclass(frozen=True)
class Call:
    """One call of a layer, as the model runs it (`bux.export.plan.LayerCall`).

    Attributes:
        path: The module's path, ``model.layers.3.attn``: the prefix of its weights.
        layer: The layer's node.
        first: Whether this is its module's first call.
    """

    path: str
    layer: Node
    first: bool


@dataclass
class Plan:
    """A graph's module tree, and the calls of its layers.

    Attributes:
        top: The model's scope.
        shared: The scope of each shared definition, by its key.
        calls: Every layer call, in the order they run.
    """

    top: Scope
    shared: dict[str, Scope]
    calls: list[Call]


def plan(arch: Node, reserved: Iterable[str] = ()) -> Plan:
    """Return the module tree of the IR `arch` (an architecture, as JSON), and the
    calls of its layers; `reserved` are the names no module takes, as the export's
    config lists them (``bux_reserved``)."""
    graph = arch["graph"]
    algebra = (arch.get("residual") or {}).get("type")
    depths = _share_depths(graph, 0, {})
    taken = frozenset(reserved)
    shared: dict[str, Scope] = {}
    for node in walk(graph):
        key = node.get("share")
        if key is not None and key not in shared:
            depth = depths[key]
            shared[key] = scope(node, key, algebra=algebra, depth=depth, reserved=taken)
    top = scope(graph, graph.get("share"), top=True, algebra=algebra, reserved=taken)
    out = Plan(top, shared, [])
    out.calls = list(_calls(out, graph, (), top, "model.", set(), True))
    return out


def _share_depths(node: Node, depth: int, out: dict[str, int]) -> dict[str, int]:
    """Return how many residual branches are around each shared definition's first
    use."""
    key = node.get("share")
    if key is not None:
        if key in out:
            return out
        out[key] = depth
    inner = depth + (node["kind"] == "residual")
    for child in children(node):
        _share_depths(child, inner, out)
    return out


def scope(
    root: Node,
    key: str | None,
    *,
    top: bool = False,
    algebra: str | None = None,
    depth: int = 0,
    reserved: frozenset[str] = frozenset(),
) -> Scope:
    """Return the scope of the module computing `root`; `key` is the shared
    definition it is, whose root it names rather than calls. Under a residual
    `algebra`, a residual branch outside any other (`depth` counts those around
    `root`) is a module too, named by it. The model's scope (`top`) names its repeat
    of the most passes ``layers``; no module takes a name of `reserved`."""
    s = Scope(root, key, reserved=reserved)
    hints: dict[Pos, str] = {}

    def visit(node: Node, pos: Pos, depth: int) -> None:
        if node.get("share") is not None and not (key is not None and pos == ()):
            return
        kind = node["kind"]
        if kind == "layer":
            hints[pos] = node.get("label") or node["config"]["type"]
        elif kind == "repeat":
            body = node["body"]
            if body.get("share") is None:
                label = body.get("label")
                hints[pos] = f"{label}s" if label else "blocks"
                s.blocks[pos] = scope(
                    body, None, algebra=algebra, depth=depth, reserved=reserved
                )
        else:
            if kind == "residual" and algebra and depth == 0:
                hints[pos] = algebra
            for i, child in enumerate(children(node)):
                visit(child, (*pos, i), depth + (kind == "residual"))

    visit(root, (), depth)
    if top and s.blocks and "layers" not in hints.values():
        hints[max(s.blocks, key=lambda p: at(root, p)["times"])] = "layers"
    for pos, hint in hints.items():
        s.names[pos] = s.name(hint)
    return s


def is_call(p: Plan, s: Scope, node: Node, pos: Pos) -> bool:
    """Whether `node`, at `pos` in `s`, is a call of a shared definition's module,
    which only that definition's own scope computes."""
    key = node.get("share")
    return key is not None and not (s is p.shared.get(key) and pos == ())


def _calls(
    p: Plan, node: Node, pos: Pos, s: Scope, prefix: str, seen: set[str], first: bool
) -> Iterator[Call]:
    """Yield the calls of the layers under `node`, at `pos` in scope `s` whose
    modules' paths start with `prefix`."""
    key = node.get("share")
    if key is not None and not (s is p.shared.get(key) and pos == ()):
        again = key in seen
        seen.add(key)
        path = f"model.shared.{key}"
        inner = path if node["kind"] == "layer" else f"{path}."
        yield from _calls(p, node, (), p.shared[key], inner, seen, first and not again)
        return
    kind = node["kind"]
    if kind == "layer":
        path = prefix if pos == () and key is not None else prefix + s.names[pos]
        yield Call(path.rstrip("."), node, first)
    elif kind == "repeat":
        body = node["body"]
        for i in range(node["times"]):
            if body.get("share") is not None:
                yield from _calls(p, body, (*pos, 0), s, prefix, seen, first)
            else:
                inner = f"{prefix}{s.names[pos]}.{i}."
                yield from _calls(p, body, (), s.blocks[pos], inner, seen, first)
    else:
        for i, child in enumerate(children(node)):
            yield from _calls(p, child, (*pos, i), s, prefix, seen, first)


def live(calls: list[Call]) -> list[Call]:
    """Return the calls up to the first head's, which make the model's output: the
    layers after it (a multi-token prediction module's) change nothing vLLM samples.

    Raises:
        ValueError: The graph has no head.
    """
    for i, call in enumerate(calls):
        if call.layer["config"]["type"] == "head":
            return calls[: i + 1]
    raise ValueError("a model needs a head for its logits")
