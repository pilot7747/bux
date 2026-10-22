"""Draw an Architecture as a circuit of the residual stream.

The stream runs left to right along the top. A branch forks off below it and merges
back: ``⊕`` sums, ``⊗`` multiplies. Labeled blocks are drawn as ``┤name├`` and then
drawn themselves underneath, so a 48-layer hybrid stays one screen wide::

    ──┬───────────⊕──┬──────────⊕──
      └─norm──attn─┘  └─norm──ffn─┘
"""

from collections.abc import Sequence
from dataclasses import dataclass

from bux import config as cfg
from bux.lang import ir


@dataclass(frozen=True)
class _Box:
    rows: tuple[str, ...]
    port: int  # the row the stream enters and leaves on

    @property
    def width(self) -> int:
        return max(len(r) for r in self.rows)


def _text(text: str) -> _Box:
    return _Box((text,), 0)


def _hcat(boxes: list[_Box], sep: str = "──") -> _Box:
    above = max(b.port for b in boxes)
    below = max(len(b.rows) - b.port for b in boxes)
    rows = [""] * (above + below)
    for k, box in enumerate(boxes):
        pad = above - box.port
        for y in range(above + below):
            i = y - pad
            cell = box.rows[i] if 0 <= i < len(box.rows) else ""
            fill = "─" if y == above else " "
            cell = cell + fill * (box.width - len(cell))
            joint = (sep if y == above else " " * len(sep)) if k else ""
            rows[y] += joint + cell
    return _Box(tuple(rows), above)


def _branches(terms: list[_Box], fed: list[bool], merge: str) -> _Box:
    """Stack the terms between a fork on the left and a merge on the right; those
    that start from a tap are not connected to the fork."""
    width = max(term.width for term in terms)
    rows: list[str] = []
    ports: list[int] = []
    for term in terms:
        ports.append(len(rows) + term.port)
        for i, row in enumerate(term.rows):
            rows.append(row + ("─" if i == term.port else " ") * (width - len(row)))
    feeds = [port for port, is_fed in zip(ports, fed) if is_fed]
    out = [
        _fork_edge(y, feeds) + row + _merge_edge(y, ports, merge)
        for y, row in enumerate(rows)
    ]
    return _Box(tuple(out), ports[0])


def _fork_edge(y: int, feeds: list[int]) -> str:
    if feeds and y == feeds[0]:
        return "┬─" if len(feeds) > 1 else "──"
    if y in feeds[1:-1]:
        return "├─"
    if len(feeds) > 1 and y == feeds[-1]:
        return "└─"
    if feeds and feeds[0] < y < feeds[-1]:
        return "│ "
    return "  "


def _merge_edge(y: int, ports: list[int], merge: str) -> str:
    if y == ports[0]:
        return "─" + merge
    if y in ports[1:-1]:
        return "─┤"
    if y == ports[-1]:
        return "─┘"
    if ports[0] < y < ports[-1]:
        return " │"
    return "  "


def _frame(box: _Box, label: str) -> _Box:
    inner = box.width + 2
    top = f"╭─{label}" + "─" * max(0, inner - len(label) - 1) + "╮"
    rows = [top]
    for i, row in enumerate(box.rows):
        row = row + ("─" if i == box.port else " ") * (box.width - len(row))
        rows.append(
            ("┤─" if i == box.port else "│ ") + row + ("─├" if i == box.port else " │")
        )
    rows.append("╰" + "─" * inner + "╯")
    return _Box(tuple(rows), box.port + 1)


def _starts_with_ref(node: ir.Node) -> bool:
    match node:
        case ir.Ref():
            return True
        case ir.Seq(items=items):
            return _starts_with_ref(items[0])
        case ir.Neg(body=body):
            return _starts_with_ref(body)
        case _:
            return False


class _Renderer:
    def __init__(self, collapse: bool):
        self.collapse = collapse
        self.blocks: dict[str, ir.Node] = {}  # labeled blocks met so far, in order

    def leaf_text(self, node: ir.Layer) -> str:
        if node.label is not None:
            text = node.label
        elif isinstance(node.config, cfg.ActivationConfig):
            text = node.config.fn
        else:
            text = node.config.type
        if node.ports:
            marks = {"write": "@", "read": "$", "pop": "$"}
            text += (
                "("
                + ", ".join(f"{p.name}={marks[p.mode]}{p.tap}" for p in node.ports)
                + ")"
            )
        return text

    def collapsed(self, node: ir.Node) -> str | None:
        if not self.collapse or node.label is None or isinstance(node, ir.Layer):
            return None
        self.blocks.setdefault(node.label, node)
        return f"┤{node.label}├"

    def box(self, node: ir.Node) -> _Box:
        if (name := self.collapsed(node)) is not None:
            return _text(name)
        match node:
            case ir.Layer():
                return _text(self.leaf_text(node))
            case ir.Seq(items=items):
                return _hcat([self.box(item) for item in items])
            case ir.Residual(body=body):
                return _branches([_text(""), self.box(body)], [True, True], "⊕")
            case ir.Sum(terms=terms):
                return self.parallel(terms, "⊕")
            case ir.Prod(factors=factors):
                scales = [f for f in factors if isinstance(f, ir.Const)]
                others = [f for f in factors if not isinstance(f, ir.Const)]
                box = (
                    self.parallel(others, "⊗")
                    if len(others) > 1
                    else self.box(others[0])
                )
                return (
                    _hcat([box, *(_text(f"×{c.value:g}") for c in scales)])
                    if scales
                    else box
                )
            case ir.Neg(body=body):
                return _hcat([_text("−"), self.box(body)], "─")
            case ir.Const(value=value):
                return _text(f"{value:g}")
            case ir.Identity():
                return _text("")
            case ir.Repeat(body=body, times=times):
                if (name := self.collapsed(body)) is not None:
                    return _text(f"{name}×{times}")
                if isinstance(body, ir.Layer):
                    return _text(f"{self.leaf_text(body)}×{times}")
                return _frame(self.box(body), f" ×{times} ")
            case ir.Tap(name=name):
                return _text(f"@{name}")
            case ir.Ref(name=name, pop=pop):
                return _text(f"${name}{'!' if pop else ''}")
        raise AssertionError(node)

    def parallel(self, terms: Sequence[ir.Node], merge: str) -> _Box:
        # Fed branches first, so the fork sits on the row where the stream enters.
        order = sorted(terms, key=_starts_with_ref)
        return _branches(
            [self.box(t) for t in order],
            [not _starts_with_ref(t) for t in order],
            merge,
        )

    def draw(self, title: str, node: ir.Node) -> str:
        box = _hcat([_text(""), self.box(node), _text("")])
        return "\n".join([title, *(r.rstrip() for r in box.rows)])


def render(arch: ir.Architecture, *, expand: bool = False) -> str:
    """Draw the model's circuit, then one per labeled block it uses (unless `expand`)."""
    renderer = _Renderer(collapse=not expand)
    root = arch.graph.model_copy(update={"label": None})
    sections = [renderer.draw(arch.graph.label or "model", root)]
    drawn: set[str] = set()
    while pending := [name for name in renderer.blocks if name not in drawn]:
        for name in pending:
            drawn.add(name)
            block = renderer.blocks[name].model_copy(update={"label": None})
            sections.append(renderer.draw(name, block))
    return "\n\n".join(sections) + "\n"
