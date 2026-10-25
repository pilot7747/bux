"""Submodules before attributes of their package's, for autosummary.

autosummary finds ``pkg.name`` as the package's attribute before its submodule, so a
package that re-exports a function under the name of the module that defines it
(``bux.parallel.mesh``, the function and the module) documents the function in place
of the module. A name that is a module of bux is that module.
"""

import importlib
import sys
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext import autosummary

_by_name = autosummary._import_by_name


def submodule_first(name: str, grouped_exception: bool = True) -> tuple[Any, Any, str]:
    """Return the module `name` if bux has one, else what autosummary finds."""
    if name.startswith("bux.") and name not in sys.modules:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    if name.startswith("bux.") and name in sys.modules:
        return sys.modules[name], None, name
    return _by_name(name, grouped_exception)


def setup(app: Sphinx) -> None:
    setattr(autosummary, "_import_by_name", submodule_first)  # noqa: B010 (a patch)
