"""Where every layer of an architecture goes in an exported model: the module tree
that every exporter shares, so that one set of weights loads into each.

It imports bux.lang, bux.config and numpy alone, no JAX and no torch, so that a
runtime's plugin can place the layers just as `bux.export` named their weights.

The model is a tree of scopes, each a module of its own (a generated class):

- the model's, at ``model.``;
- each repeat's body, a list of them at ``<scope>.<name>.<pass>.``;
- each shared definition's, once, at ``model.shared.<key>``, which every use calls.

A scope names its layers and repeats by their labels (``attn``, ``norm_1``), and the
model's longest repeat ``layers``, as transformers' models name theirs; under a
residual algebra, each residual branch outside any other is a module too, named by
the algebra (``mhc``, ``mhc_1``), which holds the algebra's parameters. `calls`
unrolls the graph into the calls of the layers in the order they run, each with its
module's path, which is also the prefix of its weights.
"""

import keyword
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from bux import config as cfg
from bux.config import EngramConfig
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
        names: The attribute of each layer, repeat and residual algebra's branch in
            it, by its `Pos`; a shared node in it is a call of ``model.shared``, not
            named here.
        blocks: The scope of each repeat's body, by the repeat's `Pos`; none for a
            repeat of a shared body, which calls it each pass.
        depth: How many residual branches are around its root: for a shared
            definition, around its first use, where bux builds it.
    """

    root: ir.Node
    names: dict[Pos, str] = field(default_factory=dict)
    blocks: dict[Pos, "Scope"] = field(default_factory=dict)
    depth: int = 0

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


@dataclass(frozen=True)
class Branch:
    """A residual branch the residual algebra makes, whose module holds the
    algebra's parameters around the branch's body.

    Attributes:
        path: The module's path, ``model.layers.3.mhc``.
        trail: The child indices from the graph's root to the ``ir.Residual``.
        first: Whether this is its module's first call.
    """

    path: str
    trail: tuple[int, ...]
    first: bool


@dataclass
class Plan:
    """An architecture's module tree, and the calls of its layers.

    Attributes:
        arch: The architecture.
        top: The model's scope.
        shared: The scope of each shared definition, by its key.
        calls: Every layer call, in the order they run.
        branches: Every residual algebra's branch, in the order they run.
    """

    arch: ir.Architecture
    top: Scope
    shared: dict[str, Scope]
    calls: list[Call]
    branches: list[Branch] = field(default_factory=list)

    def stop(self) -> int | None:
        """Return the item of the graph's top sequence that runs the first head, if
        any runs after it: a model asked for the first head's logits alone, as it
        generates, stops after it. None when there is nothing to skip."""
        root = self.arch.graph
        head = next((c for c in self.calls if c.layer.config.type == "head"), None)
        if head is None or not isinstance(root, ir.Seq) or root.share is not None:
            return None
        item = head.trail[0]
        return item if item < len(root.items) - 1 else None

    def generation(self) -> list[Call]:
        """Return the calls a model runs for the first head's logits alone: those
        up to `stop`, which keep state in the cache as it generates."""
        stop = self.stop()
        return [c for c in self.calls if stop is None or c.trail[0] <= stop]

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
    algebra = None if arch.residual is None else arch.residual.type
    depths = _share_depths(arch.graph, 0, {})
    shared: dict[str, Scope] = {}
    for node in ir.walk(arch.graph):
        if node.share is not None and node.share not in shared:
            depth = depths[node.share]
            shared[node.share] = scope(node, own=True, algebra=algebra, depth=depth)
    own = arch.graph.share is not None
    top = scope(arch.graph, own=own, top=True, algebra=algebra)
    out = Plan(arch, top, shared, [])
    for item in _calls(out, arch.graph, (), top, (), "model.", 0, set()):
        if isinstance(item, Call):
            out.calls.append(item)
        else:
            out.branches.append(item)
    return out


def _share_depths(node: ir.Node, depth: int, out: dict[str, int]) -> dict[str, int]:
    """Return how many residual branches are around each shared definition's first
    use, where bux builds its module."""
    if node.share is not None:
        if node.share in out:
            return out
        out[node.share] = depth
    inner = depth + isinstance(node, ir.Residual)
    for child in ir.children(node):
        _share_depths(child, inner, out)
    return out


def scope(
    root: ir.Node,
    *,
    own: bool,
    top: bool = False,
    algebra: str | None = None,
    depth: int = 0,
) -> Scope:
    """Return the scope of the module computing `root`: `own` says whether `root` is
    the shared node the scope is for, rather than a call of it, `algebra` names the
    residual algebra if any, and `depth` counts the residual branches around it. The
    model's scope (`top`) names its repeat of the most passes ``layers``."""
    s = Scope(root, depth=depth)
    hints: dict[Pos, str] = {}

    def visit(node: ir.Node, pos: Pos, depth: int) -> None:
        if node.share is not None and not (own and pos == ()):
            return
        match node:
            case ir.Layer():
                hints[pos] = node.label or node.config.type
            case ir.Repeat(body=body):
                if body.share is None:
                    hints[pos] = f"{body.label}s" if body.label else "blocks"
                    s.blocks[pos] = scope(body, own=False, algebra=algebra, depth=depth)
            case _:
                if isinstance(node, ir.Residual) and algebra and depth == 0:
                    hints[pos] = algebra
                inner = depth + isinstance(node, ir.Residual)
                for i, child in enumerate(ir.children(node)):
                    visit(child, (*pos, i), inner)

    visit(root, (), depth)
    if top and s.blocks and "layers" not in hints.values():
        hints[max(s.blocks, key=lambda p: _passes(root, p))] = "layers"
    for pos, hint in hints.items():
        s.names[pos] = s.name(hint)
    return s


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
) -> Iterator[Call | Branch]:
    """Yield the calls of the layers and the algebra's branches under `node`, at
    `pos` in scope `s` whose modules' paths start with `prefix`."""
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
            if isinstance(node, ir.Residual) and pos in s.names:
                yield Branch(prefix + s.names[pos], trail, first)
            inside = depth + isinstance(node, ir.Residual)
            for i, child in enumerate(ir.children(node)):
                yield from _calls(
                    p, child, (*pos, i), s, (*trail, i), prefix, inside, seen, first
                )


def _next_primes(start: int, count: int, taken: set[int]) -> list[int]:
    """Return the `count` least primes >= start not taken yet, and take them, as
    `bux.architecture.layers.engram` does."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if n not in taken and all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
            taken.add(n)
        n += 1
    return found


def engram_hashes(p: Plan, call: Call) -> dict:
    """Return what an engram layer hashes by, as bux builds it: the prime size of each
    n-gram size's and head's table, the least that no engram built before it took,
    and the odd multipliers of its n tokens, drawn from a seed of its index."""
    taken: set[int] = set()
    engrams = [c for c in p.calls if c.first and c.layer.config.type == "engram"]
    for nth, c in enumerate(engrams):
        config = c.layer.config
        assert isinstance(config, EngramConfig)
        n, heads = config.max_ngram_size, config.num_heads
        primes = _next_primes(config.table_size, (n - 1) * heads, taken)
        if c is call:
            odd = np.random.default_rng(10007 * (nth + 1)).integers(0, 2**31, n)
            rows = [primes[i * heads : (i + 1) * heads] for i in range(n - 1)]
            return {"primes": rows, "multipliers": (odd * 2 + 1).tolist()}
    raise AssertionError(f"{call.path} is no engram's first call")


SITE_EXTRAS["engram"] = engram_hashes


def mixer_depth(p: Plan, call: Call) -> dict:
    """Return the index of `call`'s layer among the token mixers the model runs, from
    0, and how many token mixers it runs: what
    ``bux.architecture.recurrent.linear_attention.depth`` builds a layer by."""
    mixers = [
        c.path for c in p.calls if isinstance(c.layer.config, cfg.AttentionConfig)
    ]
    return {"layer": mixers.index(call.path), "layers": max(len(mixers), 1)}


SITE_EXTRAS["lightning"] = mixer_depth  # its decay rates weaken with depth
