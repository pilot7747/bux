"""The names each of bux's registries knows, with where each is defined.

Every module of bux is imported first, since a module registers its builders as it
is imported. A builder's description is the first paragraph of its docstring.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Mapping

import bux
from bux_sphinx.configs import cell

Row = tuple[str, str, str]  # a name, the module that defines it, what it is


def import_all() -> None:
    """Import every module of bux, so that every builder is registered."""
    for module in pkgutil.walk_packages(bux.__path__, "bux."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # an optional extra (pyspark, cudnn) not installed
            continue


def builders(
    entries: Mapping[str, object], docs: Mapping[str, str] | None = None
) -> list[Row]:
    """Return a row for each builder, from what it was registered with (`docs`) or
    else its docstring. One of another library's shows no module: bux registered it."""
    rows = []
    for name, builder in entries.items():
        module = getattr(builder, "__module__", "") or ""
        doc = (docs or {}).get(name) or inspect.getdoc(builder) or ""
        first = doc.split("\n\n")[0].replace("\n", " ")
        rows.append((name, module if module.startswith("bux") else "", first))
    return rows


def section(title: str, where: str, rows: list[Row]) -> str:
    lines = [f"## {title}", "", f"`{where}`", ""]
    lines += ["| Name | Defined in | Description |", "|---|---|---|"]
    for name, module, text in sorted(rows):
        lines.append(f"| `{name}` | {f'`{module}`' if module else ''} | {cell(text)} |")
    return "\n".join(lines) + "\n"


def registries() -> list[tuple[str, str, list[Row]]]:
    """Return every registry: a title, its import path and its rows."""
    import_all()
    from bux import initializers, passes
    from bux.architecture.precision import matmuls
    from bux.architecture.repeats import remat
    from bux.lang.printer import config_to_source
    from bux.parallel import methods
    from bux.train import data, optim, profilers, xla

    kinds = sorted(k for k, v in vars(methods).items() if hasattr(v, "builders"))
    found = [
        (
            f"`methods.{k}`",
            getattr(methods, k).where,
            builders(getattr(methods, k).builders, getattr(methods, k).docs),
        )
        for k in kinds
    ]
    for title, registry in [
        ("Optimizers", optim.optimizers),
        ("Schedules", optim.schedules),
        ("Newton-Schulz coefficients", optim.ns_coefficients),
        ("Data sources", data.sources),
        ("Matmul formats", matmuls.formats),
        ("Inits", initializers.inits),
    ]:
        found.append(
            (title, registry.where, builders(registry.builders, registry.docs))
        )
    keeps = [(k, "", f"`{config_to_source(v)}`") for k, v in remat.PRESETS.items()]
    flags = [
        (k, "", " ".join(f"`{o}={v}`" for o, v in options.items()) or "XLA's defaults")
        for k, options in xla.PRESETS.items()
    ]
    tools = [(k, "", f"`{' '.join(c)}`") for k, c in profilers.TOOLS.items()]
    found += [
        ("Remat presets", "bux.architecture.repeats.remat.PRESETS", keeps),
        ("Compiler presets", "bux.train.xla.PRESETS", flags),
        ("Profilers", "bux.train.profilers.TOOLS", tools),
        (
            "HLO passes",
            "bux.passes.PASSES",
            builders({k: v.run for k, v in passes.PASSES.items()}),
        ),
    ]
    return found


def pages() -> dict[str, str]:
    return {"registries.md": "\n".join(section(*r) for r in registries())}
