"""The language reference: a section for every config a .bux file can call.

The sections follow the ``# --- Title ---`` comments of the config modules, and each
entry is what ``bux.lang.reference`` makes of a config: its docstring, and a table of
its fields with their types, defaults and comments.
"""

import inspect
import re
from types import ModuleType

from bux import config as cfg
from bux.lang import reference
from bux.lang.compiler import ACTIVATIONS, LAYERS, TYPES
from bux.lang.printer import field_order
from bux.train import config as tcfg

_SECTION = re.compile(r"# --- (.+?) -+$")
_CLASS = re.compile(r"class (\w+)\(")


def label(name: str) -> str:
    """Return the label of the reference entry of the config called `name`."""
    return f"bux-{name}"


def cell(text: str) -> str:
    """Escape `text` for a cell of a Markdown table."""
    return text.replace("|", "\\|").replace("\n", " ")


def fields(cls: type[cfg.Config]) -> list[str]:
    """Return the Markdown table of a config's fields."""
    rows = ["| Setting | Type | Default | Description |", "|---|---|---|---|"]
    notes = reference.notes(cls)
    for name in field_order(cls):
        if name == "type":
            continue
        f = cls.model_fields[name]
        default = "required" if f.is_required() else f"`{reference.default_text(f)}`"
        kind = reference.type_text(f.annotation)
        rows.append(
            f"| `{name}` | `{cell(kind)}` | {cell(default)} | {cell(notes.get(name, ''))} |"
        )
    return rows if len(rows) > 2 else []


def entry(cls: type[cfg.Config], names: list[str]) -> str:
    """Return the reference entry of `cls`, called by each of `names`."""
    kind = "layer" if names[0] in LAYERS else "setting"
    lines = [f"({label(n)})=" for n in names]
    lines += [f"### {', '.join(f'`{n}`' for n in names)}", ""]
    lines += [f"*{kind}* · {{py:class}}`{cls.__module__}.{cls.__name__}`", ""]
    if doc := cls.__dict__.get("__doc__"):
        lines += [inspect.cleandoc(doc), ""]
    if table := fields(cls):
        lines += [*table, ""]
    if cls.ports:
        lines += ["Ports: " + ", ".join(f"`{p}`" for p in sorted(cls.ports)), ""]
    return "\n".join(lines)


def sections(module: ModuleType) -> list[tuple[str, str, list[type[cfg.Config]]]]:
    """Return the module's sections: a title, the comment under it, its configs."""
    found: list[tuple[str, str, list[type[cfg.Config]]]] = []
    intro: list[str] | None = None
    for line in inspect.getsource(module).splitlines():
        if m := _SECTION.match(line):
            found.append((m.group(1), "", []))
            intro = []
        elif intro is not None and line.startswith("#"):
            intro.append(line.lstrip("#").strip())
        elif intro is not None:
            title, _, configs = found[-1]
            found[-1] = (title, " ".join(intro), configs)
            intro = None
        if (m := _CLASS.match(line)) and found:
            cls = getattr(module, m.group(1))
            if [t for t in reference.tags(cls) if t in TYPES]:
                found[-1][2].append(cls)
    return [s for s in found if s[2]]


def page(module: ModuleType) -> str:
    """Return the generated reference page of every config in `module`."""
    out = []
    for title, intro, configs in sections(module):
        out += [f"## {title}", ""] + ([intro, ""] if intro else [])
        out += [entry(c, [t for t in reference.tags(c) if t in TYPES]) for c in configs]
    return "\n".join(out)


def settings() -> str:
    """Return the table of a run's top-level settings, the fields of TrainConfig."""
    return "\n".join(fields(tcfg.TrainConfig)) + "\n"


def activations() -> str:
    """Return the list of the activations the language knows."""
    return "\n".join(f"- `{n}`" for n in sorted(ACTIVATIONS)) + "\n"


def pages() -> dict[str, str]:
    return {
        "layers.md": page(cfg),
        "training.md": page(tcfg),
        "settings.md": settings(),
        "activations.md": activations(),
    }
