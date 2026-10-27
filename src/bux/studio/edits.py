"""The studio's edits: each op of the page, as a splice of the file's own text.

The file is the truth, so an edit never prints the file again: it replaces the few
characters that say what changes, where the file says it, and keeps the rest
(comments, macros, patterns, alignment) as it was. `apply` takes the ops of the
page's JSON API::

    {"op": "set", "target": {"node": "n.1.0.0.0.1"}, "key": "num_kv_heads",
     "value": "1"}                                     # attn = gqa(..., num_kv_heads=1)
    {"op": "insert", "node": "n.2", "where": "after", "text": "rmsnorm"}
    {"op": "axis", "axis": "gqa|mha|mqa", "options": ["mha: 12", "mqa: 1"]}

and returns the new source, which compiles (`bux.lang.check`), or raises `EditError`
with why not. A setting is set where the file writes it: the argument of the latest
call that writes it, or the call of this file where the language would write it, or
an amendment of this file (``attn.num_kv_heads = 1``) for one written in a module.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher

from bux import config as cfg
from bux.lang import ir
from bux.lang import syntax as ast
from bux.lang.compiler import (
    ACTIVATIONS,
    LAYERS,
    SETTINGS,
    TRAIN_SETTINGS,
    Written,
    check,
)
from bux.lang.syntax import BuxError, Span
from bux.studio.picks import axis_id, squeezed
from bux.studio.targets import (
    Count,
    EditError,
    File,
    Located,
    Site,
    amendable,
    amendments,
    bare,
    blocked,
    children,
    count,
    expression,
    extent,
    find,
    grouped,
    labeled,
    levels,
    node_at,
    place,
    resolve,
    statement_of,
    tag,
)

__all__ = ["EditError", "apply"]

# How tightly each kind of expression binds, loosest first (bux.lang.syntax).
ALT, SUM, PROD, SEQ, UNARY, POWER, ATOM = range(7)
_LABELED = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*\Z", re.DOTALL)
_HEAD = re.compile(r"[ \t]*(?:shared[ \t]+)?[A-Za-z_][\w.]*?([ \t]*)=(?!=)")
# What separates the items of a list: of a call's arguments, a composition, a sweep.
_COMMA, _ARROW, _BAR = map(re.compile, (r"[ \t]*,", r"[ \t]*(?:->|→)", r"[ \t]*\|"))
_NOTE = re.compile(r"[ \t]*#[^\n]*")  # a comment that ends a line
_REST = re.compile(r"[ \t]*(?:#[^\n]*)?(?:\n|\Z)")  # nothing more on the line, but one
_SEMICOLON = re.compile(r"[ \t]*;[ \t]*")


@dataclass(frozen=True)
class Splice:
    """Replace ``source[start:end]`` with `text`."""

    start: int
    end: int
    text: str

    @classmethod
    def at(cls, span: Span, text: str) -> Splice:
        """Return the splice that replaces `span` with `text`."""
        return cls(span.start, span.end, text)


def spliced(source: str, splices: list[Splice]) -> str:
    """Return `source` with `splices` made, which must not overlap."""
    out, end = [], len(source)
    for s in sorted(splices, key=lambda s: (s.start, s.end), reverse=True):
        if s.end > end:
            raise AssertionError(f"overlapping splices at {s.start}")
        out += [source[s.end : end], s.text]
        end = s.start
    out.append(source[:end])
    return "".join(reversed(out))


def apply(
    source: str, ops: list[dict], *, picks: Mapping[str, int], path: str | None
) -> str:
    """Apply the page's edit ops to `source`, one after another.

    Each op applies to the source the one before it made, traced again: node ids
    and axis ids are those of that source's run that `picks` selects.

    Args:
        source: The file's text.
        ops: The ops, as the page's JSON API writes them (``bux.studio`` plan).
        picks: The option each axis of the sweep takes, by axis id.
        path: Where the file is, which ``use`` searches from; None if not saved.

    Returns:
        The new source. It compiles, but after an ``insert`` or ``retype`` whose
        layer still needs a required setting.

    Raises:
        EditError: If an op cannot apply, or makes a file that does not compile,
            with the reason, for the user.
    """
    for op in ops:
        kind = op.get("op")
        edit = _OPS.get(kind) if isinstance(kind, str) else None
        if edit is None:
            raise EditError(f"unknown edit {kind!r}")
        f = File(source, picks, path)
        edited = spliced(source, edit(f, op))
        _verify(edited, path, lenient=kind in ("insert", "retype"))
        source = edited
    return source


def _verify(source: str, path: str | None, *, lenient: bool) -> None:
    """Raise unless `source` compiles; with `lenient`, missing settings are fine."""
    errors = check(source, filename=path or "<string>").errors
    if not errors or (lenient and all(" needs '" in e.message for e in errors)):
        return
    raise EditError(_message(errors[0]))


def _message(err: BuxError) -> str:
    return f"{err.message} ({err.hint})" if err.hint else err.message


def _source(value: object) -> str | None:
    """Return a value of the page as source: JSON's true is ``true``."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _field(op: dict, name: str) -> object:
    if name not in op:
        raise EditError(f"the {op.get('op')} edit needs {name!r}")
    return op[name]


# --- set ----------------------------------------------------------------------------


def _set(f: File, op: dict) -> list[Splice]:
    target = _field(op, "target")
    key, value = op.get("key"), _source(op.get("value"))
    if not isinstance(target, dict):
        raise EditError("a target is {node}, {setting} or {model}")
    if "node" in target:
        return _set_node(f, str(target["node"]), key, value)
    if "setting" in target:
        return _set_setting(f, str(target["setting"]), key, value)
    if target.get("model"):
        if not key:
            raise EditError("say which setting: d_model, vocab_size, residual or init")
        name, _, rest = str(key).partition(".")
        return _set_setting(f, name, rest or None, value)
    raise EditError("a target is {node}, {setting} or {model}")


def _set_node(
    f: File, node_id: str, key: str | None, value: str | None
) -> list[Splice]:
    node = node_at(f, node_id)
    if not key:
        raise EditError("say which setting of the node to set")
    if isinstance(node, ir.Layer):
        if isinstance(node.config, cfg.ActivationConfig):
            raise EditError("an activation has no settings: retype it instead")
        parts = tuple(key.split("."))
        if parts[0] not in type(node.config).model_fields:
            raise EditError(f"{tag(node.config)} has no setting {key!r}")
        written = f.trace.written(node.config)
        return _set_field(f, written, parts, value, amendable(f, node.label))
    if key == "times":
        return _repeat_node(f, node_id, 1 if value is None else _count(value))
    if node.label is not None and key == node.label:
        return _set_definition(f, node.label, value)
    raise EditError(f"this node has no setting {key!r}")


def _count(value: object) -> int:
    try:
        return int(str(value))
    except ValueError:
        raise EditError(
            f"a repeat count is a positive integer, not {value!r}"
        ) from None


def _set_field(
    f: File,
    written: Written | None,
    key: tuple[str, ...],
    value: str | None,
    amend: str | None,
) -> list[Splice]:
    """Set the field `key` of the config at `written`, where the file writes it;
    `amend` names the definition to amend for a config written in a module."""
    found = place(f, written, key)
    dotted = ".".join(key)
    if found.found is not None and f.local(found.found.arg.span):
        if (why := blocked(f, found.found)) is not None:
            raise EditError(f"{dotted} is {why}")
        if value is None:
            return _remove_arg(f, found.found)
        return [Splice.at(extent(f.source, found.found.arg.value), value)]
    if value is None:
        if found.found is not None:
            module = f.module(found.found.arg.span)
            raise EditError(
                f"{dotted} is written in {module}: set it to its default here instead"
            )
        return []
    if found.into is not None:
        return [_write_into(f, found.into, found.rest, value)]
    if amend is None:
        span = found.found.arg.span if found.found else None
        span = span or (written.span if written is not None else None)
        module = f.module(span) if span is not None else "a module"
        raise EditError(
            f"this is written in {module} with no name to amend: name it there, or "
            f"redefine it here"
        )
    return [_amendment(f, amend, key, value)]


def _write_into(f: File, level: Written, key: tuple[str, ...], value: str) -> Splice:
    """Write ``key=value`` into the call (or bare type name) at `level`."""
    arg = f"{'.'.join(key)}={value}"
    if level.call is None:
        return Splice(level.span.end, level.span.end, f"({arg})")
    if level.call.amend:
        return _amendment(f, level.call.name, key, value)
    return _insert_arg(f.source, level.call, arg)


def _insert_arg(source: str, call: ast.Call, arg: str) -> Splice:
    """Add `arg` after the last argument of `call`: on a line of its own where the
    last one has its own line, before a trailing comma, after the last one's comment."""
    if not call.args:
        return Splice(call.span.end - 1, call.span.end - 1, arg)
    last = _arg_span(source, call.args[-1])
    line = source.rfind("\n", 0, last.start) + 1
    indent = source[line : last.start]
    if indent.strip():  # the arguments share lines
        return Splice(last.end, last.end, f", {arg}")
    if comma := _COMMA.match(source, last.end):
        return _after_item(source, comma.end(), "", f"\n{indent}{arg},")
    return _after_item(source, last.end, ",", f"\n{indent}{arg}")


def _after_item(source: str, end: int, glue: str, text: str) -> Splice:
    """Return the splice that puts `glue` right after an item that ends at `end`, and
    `text` after the comment that ends the item's line, if it has one."""
    note = _NOTE.match(source, end)
    if note is None:
        return Splice(end, end, glue + text)
    return Splice(end, note.end(), glue + source[end : note.end()] + text)


def _remove_arg(f: File, site: Site) -> list[Splice]:
    call = site.level.call
    assert call is not None
    if call.amend:
        st = statement_of(f, site.arg.span)
        assert st is not None
        return [_remove_statement(f.own, st)]
    return _remove_args(f.own, call, [site.arg])


def _arg_span(source: str, arg: ast.Arg) -> Span:
    """Return all of `arg`, its value's brackets included."""
    return Span(arg.span.start, extent(source, arg.value).end)


def _remove_args(source: str, call: ast.Call, gone: list[ast.Arg]) -> list[Splice]:
    """Remove the arguments `gone` from `call`, with their commas; all of them
    leave the bare name: ``rope(base=1)`` -> ``rope``."""
    args = [_arg_span(source, a) for a in call.args]
    out = [any(a is g for g in gone) for a in call.args]
    if all(out):
        return [Splice(call.name_span.end, call.span.end, "")]
    splices, i = [], 0
    while i < len(args):
        if not out[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(args) and out[j + 1]:
            j += 1
        splices += _cut(source, args, i, j, _COMMA)
        i = j + 1
    return splices


def _cut(
    source: str, spans: list[Span], i: int, j: int, sep: re.Pattern
) -> list[Splice]:
    """Remove the items `i` to `j` of a list written at `spans`, and a separator beside
    them, `sep`: on one line, the blanks to a neighbor; on lines of their own, those
    lines. A comment goes with the item whose line it ends. `source` is the file's
    own text (`File.own`), which ends where its last line does: the program's goes
    on, past a newline, into the modules it uses."""
    first, last = spans[i], spans[j]
    before = spans[i - 1].end if i > 0 else None
    after = spans[j + 1].start if j + 1 < len(spans) else None
    if after is not None and "\n" not in source[last.end : after]:
        return [Splice(first.start, after, "")]
    end = m.end() if (m := sep.match(source, last.end)) else last.end
    note = _NOTE.match(source, end)
    if before is not None and "\n" not in source[before : first.start]:
        if note is None:
            return [Splice(before, last.end, "")]
        if after is not None:  # the next item comes up to its line
            return [Splice(first.start, after, "")]
        return [Splice(before, note.end(), "")]
    start = _line_start(source, first.start)
    if source[start : first.start].strip():  # behind a separator, or the definition
        if after is not None:
            return [Splice(first.start, after, "")]
        assert before is not None
        if not _REST.match(source, end):  # a bracket follows: the line stays
            lead = (
                start
                + len(source[start : first.start])
                - len(source[start : first.start].lstrip())
            )
            return [Splice(lead, end, "")]
        gap = source.rfind("\n", before, first.start)
        return [Splice(gap, note.end() if note else end, "")]
    splices = []
    lasts = after is None and before is not None and end == last.end
    if lasts and (m := sep.match(source, before)):  # the separator before them
        splices.append(Splice(before, m.end(), ""))
    if rest := _REST.match(source, end):
        return [*splices, Splice(start, rest.end(), "")]
    return [*splices, Splice(first.start, end, "")]  # a bracket follows them


# --- Statements ------------------------------------------------------------------------


def _line_start(source: str, i: int) -> int:
    return source.rfind("\n", 0, i) + 1


def _line_end(source: str, i: int) -> int:
    end = source.find("\n", i)
    return len(source) if end == -1 else end


def _statement_span(source: str, st: ast.Statement) -> Span:
    """Return all of `st`, the brackets around its value included."""
    body = st.body
    if st.amends and isinstance(body, ast.Call):
        return Span(st.span.start, extent(source, body.args[-1].value).end)
    return Span(st.span.start, max(st.span.end, extent(source, body).end))


def _remove_statement(source: str, st: ast.Statement) -> Splice:
    """Remove `st`, with its lines if it has them to itself."""
    span = _statement_span(source, st)
    start, end = _line_start(source, span.start), _line_end(source, span.end)
    rest = source[span.end : end].strip()
    if source[start : span.start].strip() or (rest and not rest.startswith("#")):
        return _remove_inline(source, span, start)
    end = min(end + 1, len(source))
    # Between blank lines, or last after one, it takes one of them along.
    if source[:start].endswith("\n\n") and (end == len(source) or source[end] == "\n"):
        start -= 1
    return Splice(start, end, "")


def _remove_inline(source: str, span: Span, line: int) -> Splice:
    """Remove a statement that shares its line, which starts at `line`, with the
    ``;`` that separates it from the one beside it."""
    if m := _SEMICOLON.match(source, span.end):
        return Splice(span.start, m.end(), "")
    if m := re.search(r"[ \t]*;[ \t]*\Z", source[line : span.start]):
        return Splice(line + m.start(), span.end, "")
    return Splice.at(span, "")


def _column(source: str, st: ast.Statement) -> int | None:
    """Return the column that the ``=`` of `st` is aligned to with its neighbors', if
    its block of statements aligns them."""
    start = _line_start(source, st.span.start)
    m = _HEAD.match(source, start)
    if m is None:
        return None
    column = m.end() - 1 - start
    if len(m.group(1)) > 1:
        return column
    for line in (
        _line_start(source, start - 1),
        _line_end(source, _statement_span(source, st).end) + 1,
    ):
        other = _HEAD.match(source, line) if 0 <= line < len(source) else None
        if (
            other is not None
            and other.end() - 1 - line == column
            and len(other.group(1)) > 1
        ):
            return column
    return None


def _statement(head: str, value: str, column: int | None) -> str:
    pad = " " * max(1, (column or 0) - len(head))
    return f"{head}{pad}= {value}"


def _after(source: str, anchor: ast.Statement, head: str, value: str) -> Splice:
    """Return the splice that adds ``head = value`` on a line after `anchor`'s."""
    end = _line_end(source, _statement_span(source, anchor).end)
    line = _statement(head, value, _column(source, anchor))
    return Splice(end, end, f"\n{line}")


def _append(source: str, line: str) -> Splice:
    """Return the splice that adds `line` at the end, after a blank line."""
    n = len(source)
    if not source:
        return Splice(0, 0, f"{line}\n")
    if source.endswith("\n\n"):
        return Splice(n, n, f"{line}\n")
    return Splice(n, n, ("\n" if source.endswith("\n") else "\n\n") + f"{line}\n")


def _amendment(f: File, name: str, key: tuple[str, ...], value: str) -> Splice:
    """Return the splice that amends `name` in this file: after its amendments, if
    it has some, else at the end."""
    head = f"{name}.{'.'.join(key)}"
    earlier = amendments(f, name)
    if earlier:
        return _after(f.own, earlier[-1], head, value)
    return _append(f.own, f"{head} = {value}")


def _set_setting(
    f: File, name: str, key: str | None, value: str | None
) -> list[Splice]:
    current = _setting(f, name)
    if key and key.startswith(f"{name}.") and not _has_field(current, name):
        key = key.removeprefix(f"{name}.")
    if not key or (key == name and not _has_field(current, name)):
        return _set_definition(f, name, value)
    parts = tuple(key.split("."))
    if isinstance(current, tuple | list) and parts[0].isdigit():
        return _set_item(f, name, current, parts, value)
    if isinstance(current, cfg.Config) and parts[0] in type(current).model_fields:
        written = f.trace.written(current)
        if written is None:  # at its default: written here, whole
            if value is None:
                return []
            return _set_definition(f, name, f"{tag(current)}({key}={value})")
        return _set_field(f, written, parts, value, name)
    raise EditError(f"{name} has no setting {key!r}")


def _has_field(value: object, name: str) -> bool:
    return isinstance(value, cfg.Config) and name in type(value).model_fields


def _setting(f: File, name: str) -> object:
    """Return the value of the top-level setting `name`, None if it has none."""
    if name in SETTINGS:
        return getattr(f.trace.arch, name)
    if name in TRAIN_SETTINGS and f.trace.train is not None:
        return getattr(f.trace.train, name)
    return None


def _set_item(
    f: File,
    name: str,
    items: tuple | list,
    key: tuple[str, ...],
    value: str | None,
) -> list[Splice]:
    """Set a field of one config of a list, such as a phase of the curriculum."""
    index = int(key[0])
    if index >= len(items) or not isinstance(items[index], cfg.Config):
        raise EditError(f"{name} has no item {index}")
    written = f.trace.written(items[index])
    if written is None:
        raise EditError(f"{name} {index} is not written as a call: set the whole list")
    if len(key) > 1:
        return _set_field(f, written, key[1:], value, None)
    f.require_local(written.span, f"{name} {index}")
    if value is None:
        raise EditError(f"remove {name} {index} from the list's text")
    return [Splice.at(written.span, value)]


def _set_definition(f: File, name: str, value: str | None) -> list[Splice]:
    """Set the whole of `name`: its definition here, or a new one, which takes the
    place of a module's; None removes this file's."""
    plain = [st for st in f.statements() if st.name == name and not st.amends]
    amended = amendments(f, name)
    source = f.own
    if value is None:
        if not plain and not amended and name in f.trace.statements:
            module = f.module(f.trace.statements[name].span)
            raise EditError(f"{name} is written in {module}: set it here instead")
        return [_remove_statement(source, st) for st in plain + amended]
    removed = [_remove_statement(source, st) for st in amended]
    if plain:
        if plain[-1].params is not None:
            raise EditError(f"{name} is a macro: edit its text")
        return [Splice.at(extent(source, plain[-1].body), value), *removed]
    if amended:
        first = amended[0]
        line = _statement(name, value, _column(source, first))
        return [Splice.at(_statement_span(source, first), line), *removed[1:]]
    return [_new_statement(f, name, value)]


def _new_statement(f: File, name: str, value: str) -> Splice:
    """Return the splice that defines `name` in this file: after the file's last
    setting of its kind, or before the model, or at the end."""
    group = (
        TRAIN_SETTINGS
        if name in TRAIN_SETTINGS
        else SETTINGS
        if name in SETTINGS
        else ()
    )
    mine = [st for st in f.statements() if st.name in group]
    if mine:
        return _after(f.own, mine[-1], name, value)
    entry = next((st for st in f.statements() if st.name == "model"), None)
    if not group and entry is not None:
        start = _line_start(f.own, entry.span.start)
        return Splice(start, start, f"{name} = {value}\n")
    return _append(f.own, f"{name} = {value}")


# --- Structure ---------------------------------------------------------------------------


def precedence(source: str, expr: ast.Expr) -> int:
    """Return how tightly `expr` binds as written: brackets bind it tightest."""
    if grouped(source, expr):
        return ATOM
    match expr:
        case ast.Alt():
            return ALT
        case ast.BinOp(op="+" | "-"):
            return SUM
        case ast.BinOp(op="^"):
            return POWER
        case ast.BinOp():
            return PROD
        case ast.Seq():
            return SEQ
        case ast.Neg():
            return UNARY
    return ATOM


def slot(parent: ast.Statement | ast.Expr, expr: ast.Expr) -> int:
    """Return how tightly what takes the place of `expr` in `parent` must bind, so
    that it reads back as one part there. A composition can join one it is in."""
    match parent:
        case ast.Seq():
            return SEQ
        case ast.BinOp(op="^"):
            return ATOM if expr is parent.left else UNARY
        case ast.BinOp(op="+" | "-"):
            return SUM if expr is parent.left else PROD
        case ast.BinOp():
            return PROD if expr is parent.left else SEQ
        case ast.Neg():
            return UNARY
        case ast.Attr():
            return ATOM
        case ast.Alt():
            return SUM
    return ALT


def _leads(parent: ast.Statement | ast.Expr, expr: ast.Expr) -> bool:
    """Return whether `expr` comes first where it is: only a first item of a
    composition may start with ``-``, as ``a -b`` reads as ``a - b``."""
    return not isinstance(parent, ast.Seq) or parent.items[0] is expr


def fit(text: str, prec: int, need: int, *, first: bool = True) -> str:
    """Return `text`, in brackets if it binds looser than `need`, or starts with ``-``
    where it is not `first` in a composition."""
    if prec < need or (not first and text.lstrip().startswith("-")):
        return f"({text})"
    return text


@dataclass(frozen=True)
class _Snippet:
    """Source text that an op puts into the file, parsed."""

    text: str
    prec: int


def _snippet(text: str) -> _Snippet:
    """Parse `text` as an expression.

    Raises:
        EditError: If it is not one.
    """
    source = f"__x = {text.strip()}"
    try:
        [st] = ast.parse(source)
    except (BuxError, ValueError) as err:
        message = (
            err.message if isinstance(err, BuxError) else "more than one statement"
        )
        raise EditError(f"cannot read {text!r}: {message}") from None
    return _Snippet(text.strip(), precedence(source, st.body))


def _text(f: File, expr: ast.Expr) -> str:
    span = extent(f.source, expr)
    return f.source[span.start : span.end]


def _insert(f: File, op: dict) -> list[Splice]:
    node, where = str(_field(op, "node")), _field(op, "where")
    if where not in ("before", "after"):
        raise EditError("insert where: before or after")
    # Before -b, ``a - c b`` is as good as ``a + c (-b)``; after it, not.
    found = _whole(f, node) if where == "after" else expression(f, node)
    new = _snippet(str(_field(op, "text")))
    expr, parent, source = found.expr, found.parent, f.source
    span = extent(source, expr)
    if isinstance(parent, ast.Seq):
        sep = _separator(source, parent, expr, where == "after")
        lead = where == "before" and parent.items[0] is expr
        text = fit(new.text, new.prec, SEQ, first=lead)
        if where == "after":
            return [Splice(span.end, span.end, sep + text)]
        own = fit(_text(f, expr), precedence(source, expr), SEQ, first=False)
        return [Splice.at(span, text + sep + own)]
    after = where == "after"
    own = fit(_text(f, expr), precedence(source, expr), SEQ, first=after)
    text = fit(new.text, new.prec, SEQ, first=not after)
    joined = f"{own} {text}" if after else f"{text} {own}"
    return [Splice.at(span, fit(joined, SEQ, slot(parent, expr)))]


def _separator(source: str, seq: ast.Seq, expr: ast.Expr, after: bool) -> str:
    """Return what separates `expr` from the item beside it, on the side of an
    insert where it has one: a space, or an arrow."""
    spans = [extent(source, item) for item in seq.items]
    i = next(k for k, item in enumerate(seq.items) if item is expr)
    k = i if (after and i + 1 < len(spans)) or i == 0 else i - 1
    gap = source[spans[k].end : spans[k + 1].start]
    return " -> " if "->" in gap or "→" in gap else " "


def _delete(f: File, op: dict) -> list[Splice]:
    found = expression(f, str(_field(op, "node")))
    expr, parent, source = found.expr, found.parent, f.own
    if isinstance(parent, ast.Seq):
        spans = [extent(source, item) for item in parent.items]
        i = next(k for k, item in enumerate(parent.items) if item is expr)
        return _cut(source, spans, i, i, _ARROW)
    if isinstance(parent, ast.BinOp) and parent.op in ("+", "-", "*"):
        return [_delete_operand(f, found)]
    raise EditError(f"this is the only part of {_what(parent)}: delete that instead")


def _delete_operand(f: File, found: Located) -> Splice:
    """Replace ``a + b`` by the operand that stays: ``a``; ``-b`` of ``a - b``."""
    binop = found.parent
    assert isinstance(binop, ast.BinOp)
    source = f.source
    other = binop.right if found.expr is binop.left else binop.left
    text, prec = _text(f, other), precedence(source, other)
    if binop.op == "-" and other is binop.right:
        text, prec = "-" + fit(text, prec, UNARY), UNARY
    grand = found.parents[-2]
    first = _leads(grand, binop)
    return Splice.at(
        extent(source, binop), fit(text, prec, slot(grand, binop), first=first)
    )


def _what(parent: ast.Statement | ast.Expr) -> str:
    match parent:
        case ast.Statement(name=name):
            return f"the definition of {name}"
        case ast.Residual():
            return "its residual branch"
        case ast.BinOp(op="^"):
            return "its repeat"
        case ast.Neg():
            return "its negation"
        case ast.Alt():
            return "an alternative of the sweep"
        case ast.Call(name=name):
            return f"an argument of {name}"
    return "what it is in"


def _whole(f: File, node: str) -> Located:
    """Return the expression that `node` is written as, to change it whole: not the
    negation that ``a - b`` writes, whose text is b's.

    Raises:
        EditError: For that negation, as `expression` does.
    """
    found = expression(f, node)
    if isinstance(node_at(f, node), ir.Neg) and not isinstance(found.expr, ast.Neg):
        raise EditError(
            "this negation is the - of a - b, written as b alone: write it a + -b "
            "to change it, or change b"
        )
    return found


def _wrap(f: File, op: dict) -> list[Splice]:
    found = _whole(f, str(_field(op, "node")))
    inner = bare(f.source, found.expr)
    text = f.source[inner.start : inner.end]
    return [Splice.at(extent(f.source, found.expr), f"[{text}]")]


def _unwrap(f: File, op: dict) -> list[Splice]:
    found = resolve(f, expression(f, str(_field(op, "node"))))
    expr = found.expr
    if not isinstance(expr, ast.Residual):
        raise EditError("this is not a residual branch, [f]")
    text, prec = _text(f, expr.body), precedence(f.source, expr.body)
    need = slot(found.parent, expr)
    first = _leads(found.parent, expr)
    return [Splice.at(extent(f.source, expr), fit(text, prec, need, first=first))]


# --- repeat ------------------------------------------------------------------------------


def _repeat(f: File, op: dict) -> list[Splice]:
    return _repeat_node(f, str(_field(op, "node")), _count(_field(op, "times")))


def _repeat_node(f: File, node_id: str, times: int) -> list[Splice]:
    if times < 1:
        raise EditError("a repeat count is a positive integer")
    if isinstance(node_at(f, node_id), ir.Const):
        raise EditError("a number is not repeated: set it, or repeat what it scales")
    _whole(f, node_id)
    written = count(f, node_id)
    if written is not None:
        return _set_times(f, written, times)
    if isinstance(node_at(f, node_id), ir.Repeat):
        raise EditError("this repeat's count is not written as block^n")
    if times == 1:
        return []
    used = expression(f, node_id)
    text = fit(_text(f, used.expr), precedence(f.source, used.expr), ATOM)
    need = slot(used.parent, used.expr)
    return [Splice.at(extent(f.source, used.expr), fit(f"{text}^{times}", POWER, need))]


def _set_times(f: File, written: Count, times: int) -> list[Splice]:
    """Set the count of ``block^n``; one leaves ``block``."""
    repeat = written.repeat.expr
    assert isinstance(repeat, ast.BinOp)
    if times == 1:
        return [Splice.at(extent(f.source, repeat), _text(f, repeat.left))]
    if written.number is None:
        raise EditError(f"the count is {written.why}")
    return [Splice.at(extent(f.source, written.number), str(times))]


# --- retype ------------------------------------------------------------------------------


def _retype(f: File, op: dict) -> list[Splice]:
    node_id, tag = str(_field(op, "node")), str(_field(op, "type"))
    node = node_at(f, node_id)
    if not isinstance(node, ir.Layer):
        raise EditError("only a layer has a type")
    if tag not in LAYERS and tag not in ACTIVATIONS:
        raise EditError(f"unknown layer {tag!r}")
    chain = list(levels(f.trace.written(node.config)))
    if not chain:  # an activation, written as its name
        found = expression(f, node_id)
        if not isinstance(found.expr, ast.Name):
            raise EditError("this layer's type is not written here")
        return [Splice.at(found.expr.span, tag)]
    root = chain[-1]
    f.require_local(root.span, "this layer's type")
    whole = root.call is None or tag in ACTIVATIONS  # the call goes, or there is none
    name = root.span if whole or root.call is None else root.call.name_span
    cls = None if tag in ACTIVATIONS else LAYERS[tag]
    known = set(cls.model_fields) | set(cls.ports) if cls is not None else set()
    splices = [Splice.at(name, tag)]
    for level in chain[:-1] if whole else chain:
        if level.call is not None and f.local(level.span):
            splices += _drop_unknown(f, level.call, known)
    return splices


def _drop_unknown(f: File, call: ast.Call, known: set[str]) -> list[Splice]:
    """Remove the arguments of `call` that are not among the settings `known`."""
    gone = [arg for arg in call.args if arg.key is not None and arg.key[0] not in known]
    if not gone:
        return []
    if call.amend:
        return [
            _remove_statement(f.own, st)
            for arg in gone
            if (st := statement_of(f, arg.span)) is not None
        ]
    return _remove_args(f.own, call, gone)


# --- axis -------------------------------------------------------------------------------


def _axis(f: File, op: dict) -> list[Splice]:
    axis_key, texts = str(_field(op, "axis")), _field(op, "options")
    if not isinstance(texts, list) or not texts:
        raise EditError("an axis needs one option or more")
    texts = [str(t).strip() for t in texts]
    axis = next((a for a in f.trace.axes if axis_id(a) == axis_key), None)
    if axis is None:
        raise EditError(f"no axis {axis_key!r} in this run")
    f.require_local(axis.span, "this axis")
    found = find(f.statements(), axis.span)
    if found is None or not isinstance(found.expr, ast.Alt):
        raise EditError("this axis is not written here")
    labels = [m.group(1) if (m := _LABELED.match(t)) else None for t in texts]
    if axis.labeled and set(labels) != set(axis.options) and _sites(f, axis.key) > 1:
        raise EditError(
            f"the labels {', '.join(axis.options)} are written in more than one "
            f"place: change them in each"
        )
    if len(texts) == 1:
        return _end_sweep(f, found, texts[0])
    if any(labels) and not all(labels):
        raise EditError("label every alternative or none of them")
    return _rewrite(f.own, found.expr, texts)


def _end_sweep(f: File, found: Located, text: str) -> list[Splice]:
    """Replace the alternatives at `found` with the one `text`; one of them kept as
    it is keeps the comments it has, and so do its brackets."""
    alt, source = found.expr, f.own
    assert isinstance(alt, ast.Alt)
    m = _LABELED.match(text)
    new = _snippet(m.group(2) if m else text)
    span = extent(source, alt)
    [k] = _matched(source, alt, [text])
    if k is None or "#" not in f.text(span):
        need, first = slot(found.parent, alt), _leads(found.parent, alt)
        return [Splice.at(span, fit(new.text, new.prec, need, first=first))]
    spans, n = _option_spans(source, alt), len(alt.options)
    splices = _cut(source, spans, 0, k - 1, _BAR) if k > 0 else []
    splices += _cut(source, spans, k + 1, n - 1, _BAR) if k + 1 < n else []
    if alt.labels:  # its label goes
        splices.append(Splice(spans[k].start, extent(source, alt.options[k]).start, ""))
    return splices


def _rewrite(source: str, alt: ast.Alt, texts: list[str]) -> list[Splice]:
    """Make the options of `alt` `texts`: each new or changed one where it is
    written, each gone one with its line, and the rest as they are."""
    spans = _option_spans(source, alt)
    old = list(range(len(spans)))
    new = [
        k if k is not None else (j,) for j, k in enumerate(_matched(source, alt, texts))
    ]
    splices = []
    matcher = SequenceMatcher(None, old, new, autojunk=False)
    for kind, i1, i2, j1, j2 in matcher.get_opcodes():
        if kind == "equal":
            continue
        m = min(i2 - i1, j2 - j1)
        for i, j in zip(range(i1, i1 + m), range(j1, j1 + m), strict=True):
            splices.append(Splice.at(spans[i], _option(texts[j])))
        if i2 - i1 > m:
            splices += _cut(source, spans, i1 + m, i2 - 1, _BAR)
        if j2 - j1 > m:
            added = [_option(t) for t in texts[j1 + m : j2]]
            splices.append(_add_options(source, spans, i1 + m - 1, added))
    return splices


def _option(text: str) -> str:
    """Return an option as source that reads back as one option: ``(a | b)``."""
    m = _LABELED.match(text)
    new = _snippet(m.group(2) if m else text)
    value = fit(new.text, new.prec, SUM)
    return f"{m.group(1)}: {value}" if m else value


def _option_spans(source: str, alt: ast.Alt) -> list[Span]:
    """Return where each option of `alt` is written, its label and brackets included."""
    spans = [extent(source, o) for o in alt.options]
    if not alt.labels:
        return spans
    return [Span(labeled(source, s.start), s.end) for s in spans]


def _matched(source: str, alt: ast.Alt, texts: list[str]) -> list[int | None]:
    """Return which option of `alt` each of `texts` is, if any: as written, or as the
    page's sweep shows it, its blanks squeezed."""
    labels = alt.labels or (None,) * len(alt.options)
    known = []
    for label, o in zip(labels, alt.options, strict=True):
        written = {squeezed(source, s) for s in (o.span, extent(source, o))}
        known.append({f"{label}: {t}" if label else t for t in written})
    found, used = [], set()
    for text in texts:
        blanks = " ".join(text.split())
        k = next(
            (k for k, ks in enumerate(known) if k not in used and blanks in ks), None
        )
        used.add(k)
        found.append(k)
    return found


def _add_options(source: str, spans: list[Span], k: int, texts: list[str]) -> Splice:
    """Return the splice that adds the options `texts` after option `k` (before the
    first, for -1), as the alternatives beside it are separated: on lines of their
    own, ``| b`` or ``a |``, or on the same line."""
    if k < 0:
        return Splice(spans[0].start, spans[0].start, "".join(f"{t} | " for t in texts))
    gap = (
        source[spans[k - 1].end : spans[k].start]
        if k
        else source[spans[0].end : spans[1].start]
    )
    if "\n" not in gap:
        return Splice(spans[k].end, spans[k].end, "".join(gap + t for t in texts))
    line = gap[gap.rfind("\n") + 1 :]
    if "|" in line:
        return _after_item(
            source, spans[k].end, "", "".join(f"\n{line}{t}" for t in texts)
        )
    lines = f" |\n{line}".join(texts)
    return _after_item(source, spans[k].end, " |", f"\n{line}{lines}")


def _sites(f: File, labels: object) -> int:
    """Return in how many places of the program alternatives have these labels."""
    spans = set()
    for layer in f.trace.program.layers:
        for st in layer:
            todo: list[ast.Expr] = [st.body, *(d for _, d in st.defaults)]
            while todo:
                expr = todo.pop()
                if isinstance(expr, ast.Alt) and frozenset(expr.labels) == labels:
                    spans.add(expr.span)
                todo += children(expr)
    return len(spans)


_OPS: dict[str, Callable[[File, dict], list[Splice]]] = {
    "set": _set,
    "repeat": _repeat,
    "insert": _insert,
    "delete": _delete,
    "wrap": _wrap,
    "unwrap": _unwrap,
    "retype": _retype,
    "axis": _axis,
}
