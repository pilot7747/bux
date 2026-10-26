"""The documentation the code carries, which the docs are generated from.

Every config a .bux file calls says what it is, and every one of its settings what it
does: they are the language reference and the editor's hover (bux.lang.reference).
Every method, optimizer, schedule, source and format a registry holds says what it
does: the registries' reference lists it (docs/_ext/bux_sphinx/registries.py).
"""

import importlib
import inspect
import pkgutil

import pytest

import bux
from bux.lang import reference
from bux.lang.compiler import TYPES
from bux.lang.printer import field_order
from bux.train import config as tcfg
from bux.train.registry import Registry

CONFIGS = sorted(set(TYPES.values()), key=lambda c: c.__name__)


@pytest.mark.parametrize("cls", CONFIGS, ids=lambda c: c.__name__)
def test_every_config_says_what_it_is(cls):
    assert cls.__dict__.get("__doc__"), f"{cls.__name__} has no docstring"


@pytest.mark.parametrize("cls", [*CONFIGS, tcfg.TrainConfig], ids=lambda c: c.__name__)
def test_every_setting_says_what_it_does(cls):
    notes = reference.notes(cls)
    missing = [f for f in field_order(cls) if f != "type" and f not in notes]
    assert not missing, f"{cls.__name__}: no comment above or beside {missing}"


def registries() -> list[Registry]:
    """Return every registry, once every module has registered its builders."""
    for module in pkgutil.walk_packages(bux.__path__, "bux."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # an optional extra, such as pyspark
            continue
    from bux.architecture.precision import matmuls
    from bux.parallel import methods
    from bux.train import data, optim

    found = [v for v in vars(methods).values() if isinstance(v, Registry)]
    found += [optim.optimizers, optim.schedules]
    found += [optim.ns_coefficients, optim.ns_algorithms]
    return [*found, data.sources, matmuls.formats]


def from_tests(builder: object) -> bool:
    """Whether a test registered `builder`, as test_repeat does a method of its own."""
    module = getattr(builder, "__module__", "") or ""
    return module.startswith(("test_", "tests.", "conftest"))


def test_every_registered_builder_says_what_it_does():
    missing = [
        f"{registry.kind}: {name}"
        for registry in registries()
        for name, builder in registry.builders.items()
        if not (name in registry.docs or inspect.getdoc(builder) or from_tests(builder))
    ]
    assert not missing, f"registered without a docstring: {missing}"
