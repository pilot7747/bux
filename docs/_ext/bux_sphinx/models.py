"""The model zoo: every example architecture, described by its file's header comment.

A section per folder of ``examples/architectures`` (a model family), an entry per
file: its header comment, what ``bux check`` counts of it, and its source.
"""

from pathlib import Path

from bux.__main__ import summary
from bux.lang import compile_file

_FAMILIES = {"": "Classics and primitives"}


def header(path: Path) -> str:
    """Return the comment the file starts with, as a paragraph."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line[1:].removeprefix(" "))
    return "\n".join(lines).strip()


def counts(path: Path) -> str:
    """Return what ``bux check`` prints of the file, or why it does not compile."""
    try:
        return summary(compile_file(str(path)))
    except Exception as e:  # noqa: BLE001 (the page says so instead)
        return f"does not compile on its own: {e}".splitlines()[0]


def entry(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent.parent)
    return "\n".join(
        [
            f"(model-{path.stem})=",
            f"### `{path.stem}`",
            "",
            header(path),
            "",
            "```text",
            counts(path),
            "```",
            "",
            f":::{{dropdown}} {rel}",
            f"```{{literalinclude}} /../{rel}",
            ":language: bux",
            "```",
            ":::",
            "",
        ]
    )


def pages(root: Path) -> dict[str, str]:
    out = []
    folders = sorted({p.parent for p in root.rglob("*.bux")}, key=lambda p: p != root)
    for folder in folders:
        name = "" if folder == root else folder.relative_to(root).as_posix()
        out += [f"## {_FAMILIES.get(name, name.capitalize())}", ""]
        out += [entry(p, root) for p in sorted(folder.glob("*.bux"))]
    return {"models.md": "\n".join(out)}
