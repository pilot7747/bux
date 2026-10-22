"""Diagnostics, completion and hover for .bux source.

Pure functions of the source text, so they work without an editor;
``bux.lang.lsp.server`` only converts positions. Offsets index into the source string.
Every call checks the file again with ``bux.lang.check``, which takes milliseconds for
a model file.
"""

import inspect
import itertools
import re
import types
import typing
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

from pydantic.fields import FieldInfo

from bux import config as cfg
from bux.lang import ir
from bux.lang import syntax as ast
from bux.lang.compiler import ACTIVATIONS, LAYERS, TYPES, check
from bux.lang.printer import config_to_source, field_order, value_to_source
from bux.lang.syntax import Span


@dataclass(frozen=True)
class Diagnostic:
    start: int
    end: int
    message: str


@dataclass(frozen=True)
class Completion:
    label: str
    # setting, value, layer, config, activation, definition, macro, parameter, tap, keyword
    kind: str
    detail: str = ""
    documentation: str = ""
    insert_text: str | None = None
    sort: str = "1"  # editors order items by this, then by label


@dataclass(frozen=True)
class Hover:
    start: int
    end: int
    markdown: str


def diagnostics(source: str, path: str | None = None) -> list[Diagnostic]:
    """Return the errors in the file at `path`. Those in modules show at the `use`."""
    filename = path or "<string>"
    report = check(source, filename=filename)
    out = []
    for err in report.errors:
        span = err.span or Span(0, 0)
        hint = f"\nhint: {err.hint}" if err.hint else ""
        message = err.message + hint
        if err.filename != filename:
            via = report.via.get(err.filename)
            if via is None:
                continue
            line = (err.source or "").count("\n", 0, span.start) + 1
            message = f"{Path(err.filename).name}:{line}: {message}"
            span = via
        out.append(Diagnostic(span.start, span.end, message))
    return out


# --- What the configs say about themselves ----------------------------------------

_FIELD = re.compile(r"\s+(\w+)\s*:[^#]*?(?:#\s*(.*))?$")


@cache
def _notes(cls: type[cfg.Config]) -> dict[str, str]:
    """Return each field's comment in bux/config.py, which documents it."""
    notes: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if not (isinstance(klass, type) and issubclass(klass, cfg.Config)):
            continue
        try:
            lines = inspect.getsource(klass).splitlines()[1:]
        except OSError, TypeError:
            continue
        above: list[str] = []
        for line in lines:
            text = line.strip()
            if text.startswith("#"):
                above.append(text.lstrip("#").strip())
                continue
            m = _FIELD.match(line)
            note = " ".join([*above, m.group(2) or ""]).strip() if m else ""
            if m and note and m.group(1) in klass.model_fields:
                notes[m.group(1)] = note
            above = []
    return notes


def _members(annotation: object) -> list[object]:
    """Return a union's members, with Annotated unwrapped."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _members(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [m for a in typing.get_args(annotation) for m in _members(a)]
    return [annotation]


def _config_classes(annotation: object) -> list[type[cfg.Config]]:
    return [
        m
        for m in _members(annotation)
        if isinstance(m, type) and issubclass(m, cfg.Config)
    ]


def _tags(cls: type[cfg.Config]) -> tuple[str, ...]:
    tag = cls.model_fields.get("type")
    return typing.get_args(tag.annotation) if tag is not None else ()


def _type(annotation: object) -> str:
    """Write a field's type as its values are written: int | none, relu | gelu, rope."""
    parts = []
    for m in _members(annotation):
        if typing.get_origin(m) is typing.Literal:
            parts += [str(a) for a in typing.get_args(m)]
        elif m is type(None):
            parts.append("none")
        elif isinstance(m, type) and issubclass(m, cfg.Config):
            parts += _tags(m) or [m.__name__]
        else:
            parts.append(getattr(m, "__name__", str(m)))
    return " | ".join(parts)


def _default(f: FieldInfo) -> str:
    return value_to_source(f.get_default(call_default_factory=True))


def _field_doc(cls: type[cfg.Config], name: str) -> str:
    f = cls.model_fields[name]
    text = f"`{name}: {_type(f.annotation)}"
    text += "` (required)" if f.is_required() else f" = {_default(f)}`"
    if note := _notes(cls).get(name):
        text += f" — {note}"
    return text


def _summary(cls: type[cfg.Config]) -> str:
    doc = cls.__dict__.get("__doc__")
    return inspect.cleandoc(doc).split("\n")[0] if doc else ""


@cache
def _type_doc(name: str) -> str:
    cls = TYPES[name]
    parts = [f"**{name}** — {'layer' if name in LAYERS else 'setting'}"]
    if doc := cls.__dict__.get("__doc__"):
        parts.append(inspect.cleandoc(doc))
    if fields := [f for f in field_order(cls) if f != "type"]:
        parts.append("\n".join(f"- {_field_doc(cls, f)}" for f in fields))
    if cls.ports and "Ports" not in (doc or ""):
        parts.append("Ports: " + ", ".join(f"`{p}`" for p in sorted(cls.ports)))
    return "\n\n".join(parts)


def _activation_doc(name: str) -> str:
    return (
        f"**{name}** — activation\n\nA layer that applies it elementwise "
        f"(`linear(features=h) {name}`), or a value for settings such as "
        f"`activation={name}`."
    )


# --- The file, indexed ---------------------------------------------------------------


class _Index:
    """The parsed file and what `check` evaluated, for looking names up."""

    def __init__(self, source: str, path: str | None = None):
        self.report = check(source, filename=path or "<string>")
        # Every file joined, this one first, so that all statements' spans index into it.
        self.source = self.report.source or source
        self.defs: dict[str, ast.Statement] = {}
        for st in self.report.statements:
            self.defs.setdefault(st.name, st)
        for name, st in self.report.inherited.items():
            self.defs.setdefault(name, st)
        self.macros = {n for n, st in self.defs.items() if st.params is not None}

    def resolve(self, name: str, inside: ast.Statement | None) -> str:
        """Classify a name: param, definition, type, activation, or bare (undefined)."""
        if inside is not None and name in (inside.params or ()):
            return "param"
        builtin = name in TYPES or name in ACTIVATIONS
        # Inside its own definition a name means the builtin it shadows: mlp = mlp(...).
        if name in self.defs and not (
            builtin and inside is not None and inside.name == name
        ):
            return "definition"
        if name in TYPES:
            return "type"
        return "activation" if name in ACTIVATIONS else "bare"

    def config_class(
        self, name: str, inside: ast.Statement | None
    ) -> type[cfg.Config] | None:
        """Return the config class a name stands for: a builtin, or a definition of (an
        override of) one."""
        seen: set[str] = set()
        while True:
            kind = self.resolve(name, inside)
            if kind == "type":
                return TYPES[name]
            if kind != "definition" or name in seen or name in self.macros:
                return None
            seen.add(name)
            value = self.report.values.get(name)
            if isinstance(value, ir.Layer):
                value = value.config
            if isinstance(value, cfg.Config):
                return type(value)
            # Not evaluated, e.g. while it has an error: follow what it is written as.
            st = self.defs[name]
            if not isinstance(st.body, ast.Call | ast.Name):
                return None
            name, inside = st.body.name, st

    def setting_class(
        self, callee: str, key: tuple[str, ...], inside: ast.Statement | None
    ) -> type[cfg.Config] | None:
        """Return the class whose fields follow `callee(key.`: RoPEConfig for gqa(rope."""
        cls = self.config_class(callee, inside)
        for part in key:
            f = cls.model_fields.get(part) if cls is not None else None
            classes = _config_classes(f.annotation) if f is not None else []
            cls = classes[0] if len(classes) == 1 else None
        return cls

    def signature(self, st: ast.Statement) -> str:
        head = self.source[st.name_span.start : st.body.span.start]
        return " ".join(head.split()).rstrip("= ")

    def is_block(self, name: str) -> bool:
        """Return whether a definition is a block or macro, which no setting takes."""
        seen: set[str] = set()
        while name in self.defs and name not in seen:
            seen.add(name)
            st = self.defs[name]
            if st.params is not None:
                return True
            if name in self.report.values:
                return isinstance(self.report.values[name], ir.NODE_TYPES)
            # Not evaluated, e.g. while it has an error: judge by how it is written.
            match st.body:
                case (
                    ast.Seq() | ast.Residual() | ast.Identity() | ast.Tap() | ast.Ref()
                ):
                    return True
                case ast.Call(name=callee) | ast.Name(name=callee):
                    if self.resolve(callee, st) != "definition":
                        return False
                    name = callee
                case _:
                    return False
        return False


@dataclass(frozen=True)
class _Ctx:
    statement: ast.Statement
    setting: bool = (
        False  # the whole value of a layer setting, where a bare name is a string
    )


def _occurrences(ix: _Index) -> Iterator[tuple[Span, tuple]]:
    """Yield every name-like thing in the file, with what it is."""
    for st in ix.report.statements:
        ctx = _Ctx(st)
        yield st.name_span, ("def", st)
        for name, span in zip(st.params or (), st.param_spans):
            yield span, ("param", st, name)
        for _, default in st.defaults:
            yield from _walk(default, ctx, ix)
        yield from _walk(st.body, ctx, ix)


def _walk(expr: ast.Expr, ctx: _Ctx, ix: _Index) -> Iterator[tuple[Span, tuple]]:
    inner = replace(ctx, setting=False)
    match expr:
        case ast.Name(name=name):
            yield expr.span, ("ref", name, ctx)
        case ast.Call(name=name, args=args):
            yield expr.name_span, ("ref", name, inner)
            configures = ix.config_class(name, ctx.statement) is not None
            for arg in args:
                for i, span in enumerate(arg.key_spans):
                    yield span, ("key", name, (arg.key or ())[: i + 1], ctx)
                setting = arg.key is not None and configures
                yield from _walk(arg.value, replace(ctx, setting=setting), ix)
        case ast.Attr(base=base):
            yield expr.field_span, ("field", base, expr.field, ctx)
            yield from _walk(base, inner, ix)
        case ast.Tap(name=name):
            yield expr.span, ("tap", name, "write")
        case ast.Ref(name=name, pop=pop):
            yield expr.span, ("tap", name, "pop" if pop else "read")
        case ast.Lit(value=str() as text) if not ctx.setting:
            # A pattern: each letter refers to a one-letter definition.
            for k, ch in enumerate(text):
                if not ch.isspace():
                    start = expr.span.start + 1 + k
                    yield Span(start, start + 1), ("ref", ch, inner)
        case ast.ListExpr(items=items):
            for item in items:
                yield from _walk(item, ctx, ix)
        case ast.Seq(items=items):
            for item in items:
                yield from _walk(item, inner, ix)
        case ast.Residual(body=body) | ast.Neg(operand=body):
            yield from _walk(body, inner, ix)
        case ast.Alt(options=options):
            for option in options:
                yield from _walk(option, ctx, ix)
        case ast.BinOp(left=left, right=right):
            yield from _walk(left, inner, ix)
            yield from _walk(right, inner, ix)


def _expr_class(
    ix: _Index, expr: ast.Expr, inside: ast.Statement
) -> type[cfg.Config] | None:
    match expr:
        case ast.Name(name=name) | ast.Call(name=name):
            return ix.config_class(name, inside)
        case ast.Attr(base=base, field=name):
            cls = _expr_class(ix, base, inside)
            f = cls.model_fields.get(name) if cls is not None else None
            classes = _config_classes(f.annotation) if f is not None else []
            return classes[0] if len(classes) == 1 else None
    return None


# --- Hover -----------------------------------------------------------------------------


def hover(source: str, offset: int, path: str | None = None) -> Hover | None:
    ix = _Index(source, path)
    best: tuple[Span, tuple] | None = None
    for span, what in _occurrences(ix):
        if span.start <= offset < span.end and (
            best is None or span.end - span.start < best[0].end - best[0].start
        ):
            best = (span, what)
    if best is None:
        return None
    text = _hover_text(ix, best[1])
    return Hover(best[0].start, best[0].end, text) if text else None


def _hover_text(ix: _Index, what: tuple) -> str | None:
    match what:
        case ("def", st):
            return _definition_doc(ix, st)
        case ("param", st, name):
            return f"parameter **{name}** of `{ix.signature(st)}`"
        case ("ref", name, ctx):
            match ix.resolve(name, ctx.statement):
                case "param":
                    return f"parameter **{name}** of `{ix.signature(ctx.statement)}`"
                case "definition":
                    return _definition_doc(ix, ix.defs[name])
                case "type":
                    return _type_doc(name)
                case "activation":
                    return _activation_doc(name)
            if ctx.setting:
                return f'`{name}` is not defined, so it is the string `"{name}"`'
        case ("key", callee, key, ctx):
            if (
                ix.resolve(callee, ctx.statement) == "definition"
                and callee in ix.macros
            ):
                return f"parameter **{key[0]}** of `{ix.signature(ix.defs[callee])}`"
            cls = ix.setting_class(callee, key[:-1], ctx.statement)
            if cls is not None and key[-1] in cls.model_fields:
                return _field_doc(cls, key[-1])
        case ("field", base, name, ctx):
            cls = _expr_class(ix, base, ctx.statement)
            if cls is not None and name in cls.model_fields:
                return _field_doc(cls, name)
        case ("tap", name, _):
            counts = Counter(
                w[2] for _, w in _occurrences(ix) if w[:2] == ("tap", name)
            )
            uses = ", ".join(
                f"{verb} in {counts[mode]} place{'s' * (counts[mode] != 1)}"
                for mode, verb in (
                    ("write", "written"),
                    ("read", "read"),
                    ("pop", "popped"),
                )
                if counts[mode]
            )
            return f"tap **{name}** — {uses}"
    return None


def _value_text(value: object) -> str | None:
    """Print a definition's value as source, unless it is a block: its source shows it best."""
    if isinstance(value, ir.Layer):
        value = value.config
    if isinstance(value, cfg.Config):
        return config_to_source(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool | int | float | list):
        return value_to_source(value)
    return None


def _definition_doc(ix: _Index, st: ast.Statement) -> str:
    parts = [f"```bux\n{ix.source[st.span.start : st.span.end]}\n```"]
    if st.params is None:
        shown = _value_text(ix.report.values.get(st.name))
        written = " ".join(ix.source[st.body.span.start : st.body.span.end].split())
        if shown is not None and shown != written:
            parts.append(f"= `{shown}`")
    if st.shared:
        parts.append("Shared: every use is the same module, with the same parameters.")
    if (origin := ix.report.origin.get(st.name)) and ix.report.inherited.get(
        st.name
    ) is st:
        parts.append(f"Defined in `{Path(origin).name}`.")
    return "\n\n".join(parts)


# --- Completion ----------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_STRINGS = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
_HEADER = re.compile(r"(?:shared[ \t]+)?([A-Za-z_]\w*)(?:\(([^)]*)\))?[ \t]*=")


@dataclass
class _Open:
    """An open bracket before the cursor."""

    callee: str | None  # the name of a call; None for a group, a list or a residual
    arg: list[ast.Token] = field(default_factory=list)  # the argument being typed
    keys: list[str] = field(default_factory=list)  # keys of the arguments before it


def complete(source: str, offset: int, path: str | None = None) -> list[Completion]:
    before = source[:offset]
    word = _WORD.search(before)
    start = word.start() if word else offset
    line = before[before.rfind("\n") + 1 : start]
    if not line or "#" in _STRINGS.sub("", line):
        return []  # a new statement's name, or a comment
    if line.endswith(("@", "$")):
        return _tap_items(source, start)
    ix = _Index(source, path)
    stmt_start = ast.statement_start(source, start)
    current = _current(ix, stmt_start)
    tokens = ast.tokenize(source, stmt_start, start, errors=[])[:-1]
    last = tokens[-1] if tokens else None
    if (
        last is not None
        and last.kind == "STRING"
        and last.span.end == start
        and (len(last.text) < 2 or last.text[-1] != last.text[0])
    ):
        return _pattern_items(ix)  # inside an unterminated string: a pattern
    opens = _open_brackets(tokens)
    inner = opens[-1] if opens and opens[-1].callee is not None else None
    key = _key(inner.arg) if inner is not None else None
    if inner is not None and key is None and (dotted := _dotted(inner.arg)) is not None:
        return _key_items(
            ix, typing.cast(str, inner.callee), dotted, inner.keys, current
        )
    if line.endswith("."):
        names = _names_before_dot(tokens)
        return _field_items(ix, names, current) if names else []
    if inner is not None and key is not None:
        callee = typing.cast(str, inner.callee)
        # A setting's value: what its type allows first, then definitions. Macro
        # arguments are often layers, so they get everything.
        macro = callee in ix.macros and ix.resolve(callee, current) == "definition"
        items = _value_items(ix, callee, key, current)
        return _unique(items + _expression_items(ix, current, values=not macro))
    return _expression_items(ix, current)


def _names_before_dot(tokens: list[ast.Token]) -> list[str]:
    names = []
    for i in range(len(tokens) - 1, 0, -2):
        dot, name = tokens[i], tokens[i - 1]
        if dot.text != "." or name.kind != "NAME" or dot.space_before:
            break
        names.insert(0, name.text)
        if name.space_before or i < 2 or tokens[i - 2].text != ".":
            break
    return names


def _current(ix: _Index, stmt_start: int) -> ast.Statement | None:
    """Return the statement the cursor is in, or its header while it does not parse."""
    for st in ix.report.statements:
        if st.span.start == stmt_start:
            return st
    m = _HEADER.match(ix.source, stmt_start)
    if m is None:
        return None
    params = None
    if m.group(2) is not None:
        params = tuple(
            p.split("=")[0].strip() for p in m.group(2).split(",") if p.strip()
        )
    here = Span(stmt_start, stmt_start)
    return ast.Statement(here, m.group(1), here, params, ast.Identity(here))


def _open_brackets(tokens: list[ast.Token]) -> list[_Open]:
    opens: list[_Open] = []
    prev = None
    for tok in tokens:
        op = tok.text if tok.kind == "OP" else None
        if tok.kind == "NEWLINE" or op == ";":
            opens.clear()
        elif op in ("(", "["):
            if opens:
                opens[-1].arg.append(tok)
            callee = None
            if op == "(" and prev is not None and prev.kind == "NAME":
                callee = None if tok.space_before else prev.text
            opens.append(_Open(callee))
        elif op in (")", "]"):
            if opens:
                opens.pop()
            if opens:
                opens[-1].arg.append(tok)
        elif op == "," and opens:
            if key := _key(opens[-1].arg):
                opens[-1].keys.append(key[0])
            opens[-1].arg = []
        elif opens:
            opens[-1].arg.append(tok)
        prev = tok
    return opens


def _key(arg: list[ast.Token]) -> tuple[str, ...] | None:
    """Return the key of an argument, if any: (rope, base) for `rope.base = ...`."""
    names = []
    for i, tok in enumerate(arg):
        if i % 2 == 0:
            if tok.kind != "NAME":
                return None
            names.append(tok.text)
        elif tok.text == "=":
            return tuple(names)
        elif tok.text != ".":
            return None
    return None


def _dotted(arg: list[ast.Token]) -> tuple[str, ...] | None:
    """Return the key typed so far, if that is all the argument has: () or (rope,)."""
    if len(arg) % 2:
        return None
    names = []
    for name, dot in zip(arg[::2], arg[1::2]):
        if name.kind != "NAME" or dot.text != ".":
            return None
        names.append(name.text)
    return tuple(names)


def _setting_item(
    cls: type[cfg.Config], name: str, i: int, detail: str = ""
) -> Completion:
    f = cls.model_fields[name]
    if not detail:
        detail = _type(f.annotation) + ("" if f.is_required() else f" = {_default(f)}")
    return Completion(
        name,
        "setting",
        detail=detail,
        documentation=_notes(cls).get(name, ""),
        insert_text=f"{name}=",
        sort=f"{0 if f.is_required() else 1}{i:03}",
    )


def _key_items(
    ix: _Index,
    callee: str,
    dotted: tuple[str, ...],
    given: list[str],
    current: ast.Statement | None,
) -> list[Completion]:
    if ix.resolve(callee, current) == "definition" and callee in ix.macros:
        if dotted:
            return []
        st = ix.defs[callee]
        params = [
            Completion(p, "parameter", ix.signature(st), insert_text=f"{p}=", sort="0")
            for p in st.params or ()
            if p not in given
        ]
        return _unique(params + _expression_items(ix, current))
    cls = ix.setting_class(callee, dotted, current)
    if cls is None:
        return []
    return [
        _setting_item(cls, name, i)
        for i, name in enumerate(field_order(cls))
        if name != "type" and (dotted or name not in given)
    ]


def _value_items(
    ix: _Index, callee: str, key: tuple[str, ...], current: ast.Statement | None
) -> list[Completion]:
    cls = ix.setting_class(callee, key[:-1], current)
    f = cls.model_fields.get(key[-1]) if cls is not None else None
    if f is None:
        return []
    items = []
    for m in _members(f.annotation):
        if typing.get_origin(m) is typing.Literal:
            items += [Completion(str(v), "value", sort="0") for v in typing.get_args(m)]
        elif m is bool:
            items += [Completion(b, "keyword", sort="0") for b in ("true", "false")]
        elif m is type(None):
            items.append(Completion("none", "keyword", sort="0"))
        elif isinstance(m, type) and issubclass(m, cfg.Config):
            items += [
                Completion(tag, "config", _summary(m), _type_doc(tag), sort="0")
                for tag in _tags(m)
                if tag in TYPES  # not a mix, which is written as a sum
            ]
            items += [
                _definition_item(ix, st, sort="0")
                for name, st in ix.defs.items()
                if (current is None or name != current.name)
                and ix.config_class(name, current) is m
            ]
    return items


def _expression_items(
    ix: _Index, current: ast.Statement | None, values: bool = False
) -> list[Completion]:
    """List the names usable in an expression; with `values`, not layers or activations."""
    params = current.params if current is not None else None
    items = [Completion(p, "parameter", sort="0") for p in params or ()]
    for name, st in ix.defs.items():
        if current is not None and name == current.name:
            continue
        if not (values and ix.is_block(name)):  # a setting cannot take a block
            items.append(_definition_item(ix, st))
    if not values:
        items += [
            Completion(
                tag,
                "layer" if tag in LAYERS else "config",
                _summary(cls),
                _type_doc(tag),
                sort="2",
            )
            for tag, cls in TYPES.items()
        ]
        items += [
            Completion(a, "activation", documentation=_activation_doc(a), sort="2")
            for a in sorted(ACTIVATIONS)
        ]
        items.append(Completion("_", "keyword", "identity", sort="3"))
    return items


def _definition_item(ix: _Index, st: ast.Statement, sort: str = "1") -> Completion:
    if st.params is not None:
        return Completion(st.name, "macro", ix.signature(st), sort=sort)
    detail = _value_text(ix.report.values.get(st.name))
    if detail is None:
        written = " ".join(ix.source[st.body.span.start : st.body.span.end].split())
        detail = "block" if ix.is_block(st.name) else written
    return Completion(st.name, "definition", detail, _definition_doc(ix, st), sort=sort)


def _field_items(
    ix: _Index, names: list[str], current: ast.Statement | None
) -> list[Completion]:
    """Complete after `name.`: the settings of the config it stands for, and values."""
    if ix.resolve(names[0], current) == "param":
        return []
    cls = ix.config_class(names[0], current)
    value = (
        ix.report.values.get(names[0])
        if ix.resolve(names[0], current) == "definition"
        else None
    )
    if isinstance(value, ir.Layer):
        value = value.config
    for part in names[1:]:
        f = cls.model_fields.get(part) if cls is not None else None
        classes = _config_classes(f.annotation) if f is not None else []
        cls = classes[0] if len(classes) == 1 else None
        value = getattr(value, part, None) if isinstance(value, cfg.Config) else None
        if isinstance(value, cfg.Config):
            cls = type(value)
    if cls is None:
        return []
    items = []
    for i, name in enumerate(field_order(cls)):
        if name == "type":
            continue
        detail = ""
        if isinstance(value, cfg.Config):
            detail = f"= {value_to_source(getattr(value, name))}"
        items.append(replace(_setting_item(cls, name, i, detail), insert_text=None))
    return items


def _tap_items(source: str, start: int) -> list[Completion]:
    tokens = ast.tokenize(source, errors=[])
    names = {
        name.text
        for sigil, name in itertools.pairwise(tokens)
        if sigil.text in ("@", "$") and name.kind == "NAME" and name.span.start != start
    }
    return [Completion(n, "tap", sort="0") for n in sorted(names)]


def _pattern_items(ix: _Index) -> list[Completion]:
    return [
        _definition_item(ix, st, sort="0")
        for name, st in ix.defs.items()
        if len(name) == 1 and name.isalpha()
    ]


def _unique(items: list[Completion]) -> list[Completion]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item.label not in seen:
            seen.add(item.label)
            out.append(item)
    return out
