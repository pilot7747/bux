"""What the configs say about themselves: their documentation, for the editor and the docs.

A config documents itself in ``bux/config.py`` and ``bux/train/config.py``: its
docstring says what it is, and the comment above or beside each field says what the
field does. ``bux-lsp`` shows them on hover and completion, and the reference pages of
the documentation are generated from them, so a field documented once is documented
everywhere.
"""

import inspect
import re
import types
import typing
from functools import cache

from pydantic.fields import FieldInfo

from bux import config as cfg
from bux.lang.compiler import LAYERS, TYPES
from bux.lang.printer import field_order, value_to_source

_FIELD = re.compile(r"\s+(\w+)\s*:[^#]*?(?:#\s*(.*))?$")


@cache
def notes(cls: type[cfg.Config]) -> dict[str, str]:
    """Return each field's comment in the source, the one above it and beside it."""
    found: dict[str, str] = {}
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
                found[m.group(1)] = note
            above = []
    return found


def members(annotation: object) -> list[object]:
    """Return a union's members, with Annotated unwrapped."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return members(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [m for a in typing.get_args(annotation) for m in members(a)]
    return [annotation]


def config_classes(annotation: object) -> list[type[cfg.Config]]:
    """Return the config classes a field's annotation takes."""
    return [
        m
        for m in members(annotation)
        if isinstance(m, type) and issubclass(m, cfg.Config)
    ]


def tags(cls: type[cfg.Config]) -> tuple[str, ...]:
    """Return the names a config is called by in a .bux file: its ``type`` tags."""
    tag = cls.model_fields.get("type")
    return typing.get_args(tag.annotation) if tag is not None else ()


def type_text(annotation: object) -> str:
    """Write a field's type as its values are written: int | none, relu | gelu, rope."""
    parts = []
    for m in members(annotation):
        if typing.get_origin(m) is typing.Literal:
            parts += [str(a) for a in typing.get_args(m)]
        elif m is type(None):
            parts.append("none")
        elif isinstance(m, type) and issubclass(m, cfg.Config):
            parts += tags(m) or [m.__name__]
        else:
            parts.append(getattr(m, "__name__", str(m)))
    return " | ".join(parts)


def default_text(f: FieldInfo) -> str:
    """Write a field's default as a .bux file would."""
    return value_to_source(f.get_default(call_default_factory=True))


def field_doc(cls: type[cfg.Config], name: str) -> str:
    """Return one line on a field: its type, its default, and its comment."""
    f = cls.model_fields[name]
    text = f"`{name}: {type_text(f.annotation)}"
    text += "` (required)" if f.is_required() else f" = {default_text(f)}`"
    if note := notes(cls).get(name):
        text += f" — {note}"
    return text


def summary(cls: type[cfg.Config]) -> str:
    """Return the first line of a config's own docstring, or nothing."""
    doc = cls.__dict__.get("__doc__")
    return inspect.cleandoc(doc).split("\n")[0] if doc else ""


@cache
def type_doc(name: str) -> str:
    """Return the Markdown that documents the config called `name`."""
    cls = TYPES[name]
    parts = [f"**{name}** — {'layer' if name in LAYERS else 'setting'}"]
    if doc := cls.__dict__.get("__doc__"):
        parts.append(inspect.cleandoc(doc))
    if fields := [f for f in field_order(cls) if f != "type"]:
        parts.append("\n".join(f"- {field_doc(cls, f)}" for f in fields))
    if cls.ports and "Ports" not in (doc or ""):
        parts.append("Ports: " + ", ".join(f"`{p}`" for p in sorted(cls.ports)))
    return "\n\n".join(parts)


def activation_doc(name: str) -> str:
    """Return the Markdown that documents the activation `name`."""
    return (
        f"**{name}** — activation\n\nA layer that applies it elementwise "
        f"(`linear(features=h) {name}`), or a value for settings such as "
        f"`activation={name}`."
    )
