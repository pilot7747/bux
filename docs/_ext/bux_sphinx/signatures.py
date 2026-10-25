"""Signatures with annotations as the source writes them.

autodoc evaluates annotations, so an alias such as ``Schedule`` or ``Real[ArrayLike,
""]`` shows as the union it stands for. Python 3.14 reads them as strings instead
(``annotationlib.Format.STRING``), which is what a reader of the source sees. A
pydantic config shows no signature, since its fields are its settings: its docstring
ends with a link to its entry in the language reference instead. A class shows the
constructor bux defines for it, its own or a base's, and a protocol none. Defaults show
as the source writes them too, never by their value's repr.
"""

import annotationlib
import ast
import inspect
import os
import re
import sys
import textwrap
from collections import Counter
from collections.abc import Callable
from functools import cache
from typing import Any

from pydantic import BaseModel
from sphinx.application import Sphinx

from bux.config import Config
from bux.lang import reference
from bux.lang.compiler import TYPES
from bux_sphinx import registries


class Written:
    """Text that formats as itself, unquoted: an annotation, or a default."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


def defaults(obj: Callable[..., Any]) -> dict[str, str]:
    """Return the source text of each default of `obj`'s parameters, by name."""
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(obj)))
        tree = ast.parse(source)
    except OSError, TypeError, SyntaxError:
        return {}
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        ),
        None,
    )
    if fn is None:
        return {}
    a = fn.args
    positional = [*a.posonlyargs, *a.args][
        len(a.posonlyargs) + len(a.args) - len(a.defaults) :
    ]
    pairs = [*zip(positional, a.defaults), *zip(a.kwonlyargs, a.kw_defaults)]
    return {
        arg.arg: text
        for arg, d in pairs
        if d is not None and (text := ast.get_source_segment(source, d))
    }


_PLAIN = (int, float, str, bool, type(None))


def safe(value: object) -> str:
    """Write a default that has no source text without what it holds at build time:
    a literal as itself, anything else by its name, never by its repr (``os.environ``
    would print the build machine's whole environment)."""
    if isinstance(value, _PLAIN) or (
        isinstance(value, tuple) and all(isinstance(v, _PLAIN) for v in value)
    ):
        return repr(value)
    if isinstance(value, os._Environ):
        return "os.environ"
    if repr(value) == "<factory>":  # a dataclass field's default_factory
        return "..."
    return getattr(value, "__name__", None) or type(value).__name__ + "(...)"


_NAMES = re.compile(r"""'[^']*'|"[^"]*"|\b([A-Za-z_]\w*)""")


@cache
def ambiguous() -> frozenset[str]:
    """Return the names of the classes that more than one module of bux defines."""
    registries.import_all()
    counts = Counter(
        name
        for module in list(sys.modules.values())
        if getattr(module, "__name__", "").startswith("bux.")
        for name, value in vars(module).items()
        if isinstance(value, type) and value.__module__ == module.__name__
    )
    return frozenset(name for name, n in counts.items() if n > 1)


def qualified(text: str, scope: dict[str, Any]) -> str:
    """Write each class of bux in `text` whose name is ambiguous by its full path,
    which Sphinx links to and shows by its name alone."""

    def full(m: re.Match[str]) -> str:
        value = scope.get(m.group(1) or "")
        if m.group(1) not in ambiguous() or not isinstance(value, type):
            return m.group(0)
        return f"{value.__module__}.{value.__qualname__}"

    return _NAMES.sub(full, text)


def as_written(obj: Callable[..., Any], drop_first: bool) -> tuple[str, str] | None:
    """Return `obj`'s parameters and return annotation as the source writes them,
    without its first parameter if `drop_first` (``self`` or ``cls``)."""
    try:
        sig = inspect.signature(obj, annotation_format=annotationlib.Format.STRING)
    except TypeError, ValueError:
        return None
    scope = getattr(inspect.unwrap(obj), "__globals__", {})
    source = defaults(obj)
    empty = inspect.Parameter.empty

    def note(a: Any) -> Any:
        return empty if a is empty else Written(qualified(a, scope))

    params = [
        p.replace(
            annotation=note(p.annotation),
            default=empty
            if p.default is empty
            else Written(source.get(p.name) or safe(p.default)),
        )
        for p in sig.parameters.values()
    ][1 if drop_first else 0 :]
    ret = note(sig.return_annotation)
    text = str(sig.replace(parameters=params, return_annotation=empty))
    return text, "" if ret is empty else repr(ret)


def constructor(cls: type) -> Callable[..., Any] | None:
    """Return the ``__init__`` or ``__new__`` that bux defines for `cls`, its own or
    a base's; None if bux defines none."""
    for klass in cls.__mro__:
        if not klass.__module__.startswith("bux"):
            continue
        for name in ("__init__", "__new__"):
            if name in vars(klass):
                return vars(klass)[name]
    return None


def process(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str] | None:
    if isinstance(obj, type) and (
        issubclass(obj, BaseModel) or getattr(obj, "_is_protocol", False)
    ):
        return "", ""
    if signature is None or what not in ("function", "method", "class"):
        return None
    if what == "class":
        made = constructor(obj)
        found = as_written(made, drop_first=True) if made else None
        return (found[0], "") if found else None
    first = next(iter(inspect.signature(obj).parameters), None)
    return as_written(obj, drop_first=what == "method" and first == "self")


def skip(
    app: Sphinx, what: str, name: str, obj: object, skip: bool, options: Any
) -> bool | None:
    """Skip pydantic's own attributes of a config: model_config, model_fields."""
    return True if name.startswith("model_") and what == "class" else None


def link(
    app: Sphinx, what: str, name: str, obj: object, options: Any, lines: list[str]
) -> None:
    """End a config's docstring with the names a .bux file calls it by."""
    if not (isinstance(obj, type) and issubclass(obj, Config)):
        return
    if names := [t for t in reference.tags(obj) if t in TYPES]:
        calls = ", ".join(f":bux:`{t}`" for t in names)
        lines += ["", f"Called in a ``.bux`` file as {calls}."]


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-signature", process)
    app.connect("autodoc-skip-member", skip)
    app.connect("autodoc-process-docstring", link)
