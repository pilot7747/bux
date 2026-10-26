"""Where every layer of an architecture goes in an exported model: the module tree
that every exporter shares, so that one set of weights loads into each.

It imports bux.lang alone, no JAX and no torch, so that a runtime's plugin can place
the layers just as `bux.export` named their weights.

The model is a tree of scopes, each a module of its own (a generated class):

- the model's, at ``model.``;
- each repeat's body, a list of them at ``<scope>.<name>.<pass>.``;
- each shared definition's, once, at ``model.shared.<key>``, which every use calls.

A scope names its layers and repeats by their labels (``attn``, ``norm_1``), and the
model's longest repeat ``layers``, as transformers' models name theirs. `calls`
unrolls the graph into the calls of the layers in the order they run, each with its
module's path, which is also the prefix of its weights.
"""

import keyword
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from bux.lang import ir

#: A node's place in its scope: the child indices from the scope's root to it.
Pos = tuple[int, ...]

#: What a layer type's exporter adds to its layers' `Site` extras, from the plan and
#: the call: ``SITE_EXTRAS["engram"] = lambda plan, call: {"table_sizes": ...}``.
SITE_EXTRAS: dict[str, Callable[["Plan", "Call"], dict]] = {}


@dataclass
class Scope:
    """The modules of one generated class: the model's, a repeat's body, or a shared
    definition's.

    Attributes:
        root: The node the scope computes.
        names: The attribute of each layer and repeat in it, by its `Pos`; a shared
            node in it is a call of ``model.shared``, not named here.
        blocks: The scope of each repeat's body, by the repeat's `Pos`; none for a
            repeat of a shared body, which calls it each pass.
    """

    root: ir.Node
    names: dict[Pos, str] = field(default_factory=dict)
    blocks: dict[Pos, "Scope"] = field(default_factory=dict)

    def name(self, hint: str) -> str:
        """Return a fresh attribute name from `hint`: ``norm``, then ``norm_1``."""
        hint = hint if hint.isidentifier() and not keyword.iskeyword(hint) else "layer"
        taken = set(self.names.values()) | {"shared"}
        name, i = hint, 1
        while name in taken:
            name, i = f"{hint}_{i}", i + 1
        return name


@dataclass(frozen=True)
class Call:
    """One call of a layer, as the model runs it.

    Attributes:
        path: The module's path, ``model.layers.3.attn``: the prefix of its weights.
        layer: The layer's node.
        trail: The child indices from the graph's root to the layer, a repeat's pass
            counted as its child: the way to its module in bux's model.
        depth: How many residual branches are around it.
        first: Whether this is its module's first call; a shared module is called
            again, by its later uses.
    """

    path: str
    layer: ir.Layer
    trail: tuple[int, ...]
    depth: int
    first: bool


@dataclass
class Plan:
    """An architecture's module tree, and the calls of its layers.

    Attributes:
        arch: The architecture.
        top: The model's scope.
        shared: The scope of each shared definition, by its key.
        calls: Every layer call, in the order they run.
    """

    arch: ir.Architecture
    top: Scope
    shared: dict[str, Scope]
    calls: list[Call]

    def sites(self) -> dict[str, dict]:
        """Return each module's site, by its path: its depth, how many layers of its
        type run before it, its ports, and its type's `SITE_EXTRAS`."""
        seen: Counter[str] = Counter()
        out = {}
        for call in self.calls:
            if not call.first:
                continue
            kind = call.layer.config.type
            hook = SITE_EXTRAS.get(kind)
            out[call.path] = {
                "depth": call.depth,
                "nth": seen[kind],
                "ports": [[p.name, p.tap, p.mode] for p in call.layer.ports],
                "extra": hook(self, call) if hook else {},
            }
            seen[kind] += 1
        return out


def plan(arch: ir.Architecture) -> Plan:
    """Return the module tree of `arch` and the calls of its layers."""
    shared: dict[str, Scope] = {}
    for node in ir.walk(arch.graph):
        if node.share is not None and node.share not in shared:
            shared[node.share] = scope(node, own=True)
    top = scope(arch.graph, own=arch.graph.share is not None, top=True)
    out = Plan(arch, top, shared, [])
    out.calls = list(_calls(out, arch.graph, (), top, (), "model.", 0, set()))
    return out


def scope(root: ir.Node, *, own: bool, top: bool = False) -> Scope:
    """Return the scope of the module computing `root`: `own` says whether `root` is
    the shared node the scope is for, rather than a call of it."""
    s = Scope(root)

    def visit(node: ir.Node, pos: Pos) -> None:
        if node.share is not None and not (own and pos == ()):
            return
        match node:
            case ir.Layer():
                s.names[pos] = s.name(node.label or node.config.type)
            case ir.Repeat(body=body):
                if body.share is None:
                    s.names[pos] = s.name(f"{body.label}s" if body.label else "blocks")
                    s.blocks[pos] = scope(body, own=False)
            case _:
                for i, child in enumerate(ir.children(node)):
                    visit(child, (*pos, i))

    visit(root, ())
    if top:
        _name_layers(s)
    return s


def _name_layers(s: Scope) -> None:
    """Name the scope's repeat of the most passes ``layers``."""
    repeats = [p for p in s.blocks if p in s.names]
    if repeats and "layers" not in s.names.values():
        s.names[max(repeats, key=lambda p: _passes(s.root, p))] = "layers"


def _passes(root: ir.Node, pos: Pos) -> int:
    """Return how many passes the repeat at `pos` under `root` makes."""
    for i in pos:
        root = ir.children(root)[i]
    assert isinstance(root, ir.Repeat)
    return root.times


def _calls(
    p: Plan,
    node: ir.Node,
    pos: Pos,
    s: Scope,
    trail: tuple[int, ...],
    prefix: str,
    depth: int,
    seen: set[str],
    first: bool = True,
) -> Iterator[Call]:
    """Yield the calls of the layers under `node`, at `pos` in scope `s` whose
    modules' paths start with `prefix`."""
    key = node.share
    if key is not None and not (s is p.shared.get(key) and pos == ()):
        again = key in seen
        seen.add(key)
        path = f"model.shared.{key}"
        inner = path if isinstance(node, ir.Layer) else f"{path}."
        yield from _calls(
            p, node, (), p.shared[key], trail, inner, depth, seen, first and not again
        )
        return
    match node:
        case ir.Layer():
            path = prefix if pos == () and key is not None else prefix + s.names[pos]
            yield Call(path.rstrip("."), node, trail, depth, first)
        case ir.Repeat(body=body, times=times):
            for i in range(times):
                if body.share is not None:
                    yield from _calls(
                        p, body, (*pos, 0), s, (*trail, i), prefix, depth, seen, first
                    )
                else:
                    inner = f"{prefix}{s.names[pos]}.{i}."
                    yield from _calls(
                        p,
                        body,
                        (),
                        s.blocks[pos],
                        (*trail, i),
                        inner,
                        depth,
                        seen,
                        first,
                    )
        case _:
            inside = depth + isinstance(node, ir.Residual)
            for i, child in enumerate(ir.children(node)):
                yield from _calls(
                    p, child, (*pos, i), s, (*trail, i), prefix, inside, seen, first
                )
