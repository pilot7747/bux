"""Signatures with annotations as the source writes them.

autodoc evaluates annotations, so an alias such as ``Schedule`` or ``Real[ArrayLike,
""]`` shows as the union it stands for. Python 3.14 reads them as strings instead
(``annotationlib.Format.STRING``), which is what a reader of the source sees. A
pydantic config shows no signature, since its fields are its settings: its docstring
ends with a link to its entry in the language reference instead.
"""

import annotationlib
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from sphinx.application import Sphinx

from bux.config import Config
from bux.lang import reference
from bux.lang.compiler import TYPES


class Written:
    """Text that formats as itself, unquoted: an annotation, or a default."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text


def written(value: object) -> object:
    """Return a default that formats readably: a dtype or a class by its name."""
    return Written(value.__name__) if isinstance(value, type) else value


def as_written(obj: Callable[..., Any]) -> tuple[str, str] | None:
    """Return `obj`'s parameters and return annotation as the source writes them."""
    try:
        sig = inspect.signature(obj, annotation_format=annotationlib.Format.STRING)
    except TypeError, ValueError:
        return None
    empty = inspect.Parameter.empty
    params = [
        p.replace(
            annotation=empty if p.annotation is empty else Written(p.annotation),
            default=empty if p.default is empty else written(p.default),
        )
        for p in sig.parameters.values()
    ]
    ret = sig.return_annotation
    text = str(sig.replace(parameters=params, return_annotation=empty))
    return text, "" if ret is empty else str(ret)


def process(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str] | None:
    if isinstance(obj, type) and issubclass(obj, BaseModel):
        return "", ""
    if signature is None or what not in ("function", "method", "class"):
        return None
    if what == "class" and "__init__" not in vars(obj):
        return None
    found = as_written(obj.__init__ if what == "class" else obj)
    if found is None:
        return None
    params, ret = found
    if what == "class" or (what == "method" and params.startswith("(self")):
        params = params.replace("(self, ", "(", 1).replace("(self)", "()", 1)
    return params, ret if what != "class" else ""


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
