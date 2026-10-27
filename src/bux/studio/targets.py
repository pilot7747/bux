"""Where the studio's targets are written: a node of the drawn graph, a setting, or
a field of either, found in the file's text.

The page names a node by its path in the graph (``"n.1.0"``), and a setting by its
name. `File` holds the trace of the run that the page picked, and resolves them: a
node to its IR node and the expression of the file it is written as (`expression`),
and a field of a config to the argument that writes it, or to the call where the
language would write it (`place`). `bux.studio.edits` splices there, and
`bux.studio.forms` shows what is found there.

A span at or past the end of the file's own text is in a module: read-only, but a
field written there can be set by an amendment in this file (``attn.num_heads =
12``), which `place` names.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from bux import config as cfg
from bux.lang import ir
from bux.lang import syntax as ast
from bux.lang.compiler import Trace, Written
from bux.lang.syntax import BuxError, Span
from bux.studio.picks import traced

#: A name that an amendment or a definition can start with.
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LABEL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\Z")


class EditError(Exception):
    """An edit or a target that the studio cannot apply, with why, for the user."""


@dataclass(frozen=True)
class Located:
    """An expression of the file, with what it is written inside.

    Attributes:
        expr: The expression.
        parents: What it is inside, outermost first: its statement, then each
            expression around it (a `~bux.lang.syntax.Call` for an argument).
    """

    expr: ast.Expr
    parents: tuple[ast.Statement | ast.Expr, ...]

    @property
    def parent(self) -> ast.Statement | ast.Expr:
        """Return what the expression is directly inside."""
        return self.parents[-1]

    @property
    def statement(self) -> ast.Statement:
        """Return the statement the expression is in."""
        st = self.parents[0]
        assert isinstance(st, ast.Statement)
        return st


@dataclass(frozen=True)
class Site:
    """An argument that writes a field: ``num_kv_heads=8`` of the call at `level`."""

    arg: ast.Arg
    level: Written


@dataclass(frozen=True)
class Place:
    """Where a field of a config is written, and where this file would write it.

    Attributes:
        found: The argument that writes it, the latest amendment first; None for a
            field at its default, or set by the value of a setting around it.
        into: The latest call (or bare type name) in this file that the field can be
            written into, before `found`; None if this file has none.
        rest: The field's key relative to `into`'s config.
    """

    found: Site | None
    into: Written | None
    rest: tuple[str, ...]


class File:
    """The trace of the run of a file that the page picked, and its spans.

    Args:
        source: The file's text.
        picks: The option each axis of the sweep takes, as `bux.studio.picks`.
        path: Where the file is, which ``use`` searches from; None if not saved.

    Raises:
        EditError: If the file does not compile, with the compiler's message.
    """

    def __init__(self, source: str, picks: Mapping[str, int], path: str | None):
        try:
            self.trace: Trace = traced(source, picks, path=path)
        except BuxError as err:
            raise EditError(err.message) from None
        self.length = len(source)

    @property
    def source(self) -> str:
        """Return every file of the program joined, this one first."""
        return self.trace.program.source

    @property
    def own(self) -> str:
        """Return this file's own text."""
        return self.trace.program.units[0].text

    def local(self, span: Span) -> bool:
        """Return whether `span` is in this file, not in a module."""
        return span.start < self.length

    def text(self, span: Span) -> str:
        """Return the source at `span`."""
        return self.source[span.start : span.end]

    def origin(self, span: Span) -> str:
        """Return the name of the file that `span` is in."""
        return self.trace.program.unit(span.start).filename

    def module(self, span: Span) -> str:
        """Return the module that `span` is in, as a message names it."""
        return Path(self.origin(span)).name

    def where(self, span: Span) -> dict:
        """Return a `Where` of the page: the span in this file, else the module's
        file, and the 1-based line in its own file."""
        unit = self.trace.program.unit(span.start)
        line = unit.text.count("\n", 0, span.start - unit.start) + 1
        if self.local(span):
            return {"span": [span.start, span.end], "file": None, "line": line}
        return {"span": None, "file": unit.filename, "line": line}

    def statements(self) -> list[ast.Statement]:
        """Return this file's own statements, in the order written."""
        return self.trace.program.main

    def require_local(self, span: Span, what: str) -> None:
        """Raise unless `span` is in this file.

        Raises:
            EditError: Naming the module that `what` is written in.
        """
        if not self.local(span):
            raise EditError(
                f"{what} is written in {self.module(span)}: redefine it here to "
                f"change it"
            )


# --- Nodes ---------------------------------------------------------------------


def node_path(node: str) -> tuple[int, ...]:
    """Return the path of a node id: ``"n.0.2"`` is ``(0, 2)``, ``"n"`` the root.

    Raises:
        EditError: For an id that is not of this form.
    """
    head, *parts = node.split(".")
    if head != "n" or not all(p.isdigit() for p in parts):
        raise EditError(f"no node {node!r}")
    return tuple(int(p) for p in parts)


def node_at(f: File, node: str) -> ir.Node:
    """Return the IR node of the graph that `node` names.

    Raises:
        EditError: If the graph has no such node.
    """
    found = f.trace.arch.graph
    for k in node_path(node):
        kids = ir.children(found)
        if k >= len(kids):
            raise EditError(f"no node {node!r}")
        found = kids[k]
    return found


def node_span(f: File, node: str) -> Span:
    """Return where the node is written.

    Raises:
        EditError: For a node the compiler inserted, which is written nowhere.
    """
    node_at(f, node)
    span = f.trace.nodes.get(node_path(node))
    if span is None:
        raise EditError("the compiler inserted this node; it is not in the file")
    return span


def expression(f: File, node: str) -> Located:
    """Return the expression of this file that `node` is written as; for the root,
    the body of the entry's definition.

    Raises:
        EditError: For a node written in a module, or as a letter of a pattern
            string (``"DDDA"``).
    """
    span = node_span(f, node)
    f.require_local(span, "this node")
    for st in f.statements():
        if st.name_span == span:
            return Located(st.body, (st,))
    found = find(f.statements(), span)
    if found is None:
        raise EditError(
            "this layer is a letter of a pattern string: edit the pattern's text, or "
            "the definition of its letter"
        )
    return found


def find(statements: list[ast.Statement], span: Span) -> Located | None:
    """Return the outermost expression of `statements` written at `span`."""
    for st in statements:
        if holds(st.span, span) and (found := _find_in(st, span, ())) is not None:
            return found
    return None


def _find_in(
    node: ast.Statement | ast.Expr, span: Span, parents: tuple
) -> Located | None:
    for child in children(node):
        if child.span == span:
            return Located(child, (*parents, node))
        if holds(child.span, span) and (
            found := _find_in(child, span, (*parents, node))
        ):
            return found
    return None


def holds(outer: Span, inner: Span) -> bool:
    """Return whether `inner` is within `outer`."""
    return outer.start <= inner.start and inner.end <= outer.end


def children(node: ast.Statement | ast.Expr) -> Iterator[ast.Expr]:
    """Yield the expressions directly inside `node`, in the order written."""
    match node:
        case ast.Statement(defaults=defaults, body=body):
            yield from (default for _, default in defaults)
            yield body
        case ast.Seq(items=items) | ast.ListExpr(items=items):
            yield from items
        case ast.Alt(options=options):
            yield from options
        case ast.BinOp(left=left, right=right):
            yield left
            yield right
        case ast.Residual(body=body) | ast.Neg(operand=body) | ast.Attr(base=body):
            yield body
        case ast.Call(args=args):
            yield from (arg.value for arg in args)


def resolve(f: File, found: Located) -> Located:
    """Follow a name to the body of its definition, and on, as long as it is a plain
    definition: ``blocks`` to ``block^32``.

    Raises:
        EditError: For a definition in a module.
    """
    seen: set[str] = set()
    while isinstance(found.expr, ast.Name):
        name = found.expr.name
        params = found.statement.params or ()
        st = f.trace.statements.get(name)
        if name in params or name in seen or st is None or st.params is not None:
            break
        if st.amends:
            break
        f.require_local(st.span, name)
        seen.add(name)
        found = Located(st.body, (st,))
    return found


@dataclass(frozen=True)
class Count:
    """How the count of a repeat, ``block^n``, is written.

    Attributes:
        repeat: The ``block^n``, perhaps in a definition that the node names.
        number: The integer that says n, perhaps in a definition that the count
            names (``block^layers``); None if n is not written as a plain number.
        why: Why n cannot be set here, when `number` is None.
    """

    repeat: Located
    number: ast.Lit | None
    why: str | None = None


def count(f: File, node: str) -> Count | None:
    """Return how the count of the repeat that `node` is written as is written;
    None for a node not written as a repeat.

    Raises:
        EditError: As `expression` does.
    """
    found = _macro_body(f, resolve(f, expression(f, node)))
    repeat = found.expr
    if not (isinstance(repeat, ast.BinOp) and repeat.op == "^"):
        return None
    n = repeat.right
    if isinstance(n, ast.Name):
        if n.name in (found.statement.params or ()):
            why = f"set by the parameter {n.name} of the macro {found.statement.name}"
            return Count(found, None, why)
        n = resolve(f, Located(n, (*found.parents, repeat))).expr
    if isinstance(n, ast.Lit) and isinstance(n.value, int):
        return Count(found, n, None)
    span = extent(f.source, repeat.right)
    return Count(found, None, f"written as {f.text(span)}: edit it there")


def _macro_body(f: File, found: Located) -> Located:
    """Follow a call of a macro to its body: ``b(3)`` to ``rmsnorm^k``."""
    if not isinstance(found.expr, ast.Call):
        return found
    st = f.trace.statements.get(found.expr.name)
    if st is None or st.params is None:
        return found
    f.require_local(st.span, f"the macro {st.name}")
    return resolve(f, Located(st.body, (st,)))


# --- Extents ------------------------------------------------------------------------


def extent(source: str, expr: ast.Expr) -> Span:
    """Return all of what `expr` is written as, the brackets that group it included:
    ``(a + b)``, and the first label of labeled alternatives, ``mha: 12 | mqa: 1``."""
    span = bare(source, expr)
    while True:
        i = _before(source, span.start)
        j = _after(source, span.end)
        if i < 0 or j >= len(source) or source[i] != "(" or source[j] != ")":
            return span
        if i > 0 and (source[i - 1].isalnum() or source[i - 1] == "_"):
            return span  # a call's bracket: f(x)
        span = Span(i, j + 1)


def grouped(source: str, expr: ast.Expr) -> bool:
    """Return whether `expr` is written in brackets of its own: ``(a + b)``."""
    return extent(source, expr) != bare(source, expr)


def bare(source: str, expr: ast.Expr) -> Span:
    """Return what `expr` is written as, without the brackets that group it: the
    syntax's spans leave out those of its first and last parts, ``(a b)^2``."""
    match expr:
        case ast.Alt(options=options, labels=labels):
            first = extent(source, options[0]).start
            first = labeled(source, first) if labels else first
            return Span(first, extent(source, options[-1]).end)
        case ast.BinOp(left=left, right=right):
            return Span(extent(source, left).start, extent(source, right).end)
        case ast.Seq(items=items):
            return Span(extent(source, items[0]).start, extent(source, items[-1]).end)
        case ast.Neg(operand=operand):
            return Span(expr.span.start, extent(source, operand).end)
        case ast.Attr(base=base):
            return Span(extent(source, base).start, expr.span.end)
    return expr.span


def labeled(source: str, start: int) -> int:
    """Return where the label of an alternative that starts at `start` starts:
    ``mqa: 1``; `start` if it has none."""
    m = _LABEL.search(source, max(0, start - 256), start)
    return m.start() if m else start


def _before(source: str, i: int) -> int:
    """Return the index of the last non-blank character before `i`, or -1."""
    i -= 1
    while i >= 0 and source[i].isspace():
        i -= 1
    return i


def _after(source: str, i: int) -> int:
    """Return the index of the first non-blank character at or after `i`."""
    while i < len(source) and source[i].isspace():
        i += 1
    return i


def options(source: str, alt: ast.Alt) -> list[str]:
    """Return the alternatives of `alt` as written, labeled ones as ``mqa: 1``."""
    texts = [_text(source, extent(source, o)) for o in alt.options]
    if alt.labels:
        return [f"{label}: {text}" for label, text in zip(alt.labels, texts)]
    return texts


def _text(source: str, span: Span) -> str:
    return source[span.start : span.end]


# --- Fields ------------------------------------------------------------------------


def tag(config: cfg.Config) -> str:
    """Return the name that a ``.bux`` file calls `config` by: ``gqa``."""
    return getattr(config, "type")  # noqa: B009


def levels(written: Written | None) -> Iterator[Written]:
    """Yield where a config is written, then each call it amends, latest first."""
    while written is not None:
        yield written
        written = written.base


def value_at(value: object, key: tuple[str, ...]) -> object:
    """Return the field at the dotted `key` of a config, an index of a list as a
    part of it; None if there is none."""
    for part in key:
        if isinstance(value, tuple | list) and part.isdigit():
            value = value[int(part)] if int(part) < len(value) else None
        elif isinstance(value, cfg.Config) and part in type(value).model_fields:
            value = getattr(value, part)
        else:
            return None
    return value


def place(f: File, written: Written | None, key: tuple[str, ...]) -> Place:
    """Return where the field `key` of the config at `written` is written, and the
    latest call of this file that could write it.

    A key's prefix written as a call inline (``rope=rope(base=1e4)``) is followed into
    it; one written as anything else (``qk_norm=norm``) sets every field under it.
    """
    into: Written | None = None
    for level in levels(written):
        if into is None and f.local(level.span):
            into = level
        if level.call is None:
            continue
        if (arg := argument(level.call, key)) is not None:
            return Place(Site(arg, level), into, key)
        for i in range(len(key) - 1, 0, -1):
            if (outer := argument(level.call, key[:i])) is not None:
                inner = _inline(f, level, outer, key[:i])
                if inner is None:
                    return Place(None, into, key)
                below = place(f, inner, key[i:])
                if below.found is not None and f.local(below.found.arg.span):
                    return below
                if below.into is not None:
                    return below
                return Place(below.found, into, key)
    return Place(None, into, key)


def argument(call: ast.Call, key: tuple[str, ...]) -> ast.Arg | None:
    """Return the argument of `call` with exactly this key, if any."""
    return next((arg for arg in call.args if arg.key == key), None)


def _inline(
    f: File, level: Written, arg: ast.Arg, key: tuple[str, ...]
) -> Written | None:
    """Return where the config that `arg` gives is written, if within `arg` itself."""
    nested = value_at(level.config, key)
    found = f.trace.written(nested) if isinstance(nested, cfg.Config) else None
    if found is None or not holds(extent(f.source, arg.value), found.span):
        return None
    return found


def macro_parameter(f: File, arg: ast.Arg) -> tuple[str, str] | None:
    """Return the parameter and the macro that set `arg`'s value, if it is written in
    a macro's body with one of its parameters: ``num_heads=h``."""
    for st in f.trace.statements.values():
        if st.params is None or not (st.span.start <= arg.span.start < st.span.end):
            continue
        for name in names(arg.value):
            if name in st.params:
                return name, st.name
    return None


def names(expr: ast.Expr) -> Iterator[str]:
    """Yield every name that `expr` uses."""
    if isinstance(expr, ast.Name):
        yield expr.name
    if isinstance(expr, ast.Call):
        yield expr.name
    for child in children(expr):
        yield from names(child)


def blocked(f: File, site: Site) -> str | None:
    """Return why the argument at `site` cannot be edited here, or None if it can."""
    if (found := macro_parameter(f, site.arg)) is not None:
        param, macro = found
        return f"set by the parameter {param} of the macro {macro}"
    return None


def amendable(f: File, label: str | None) -> str | None:
    """Return `label` if this file can amend the definition it names: a plain
    definition, not a macro; else None."""
    if label is None or not IDENT.match(label):
        return None
    st = f.trace.statements.get(label)
    return label if st is not None and st.params is None else None


def amendments(f: File, name: str) -> list[ast.Statement]:
    """Return this file's amendments of `name`, in the order written."""
    return [st for st in f.statements() if st.amends and st.name == name]


def statement_of(f: File, span: Span) -> ast.Statement | None:
    """Return this file's statement that holds `span`."""
    return next(
        (st for st in f.statements() if st.span.start <= span.start < st.span.end),
        None,
    )
