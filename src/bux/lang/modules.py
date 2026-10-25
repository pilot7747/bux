"""Modules: finding what ``use`` names, and joining files into one program.

A module is a .bux file, found by its name and never by a path: ``use llama3_8b``
finds ``llama3_8b.bux``. The search (`find`) looks in the folder of the file that
uses it, then under every folder of the search path, subfolders included:

1. the ``path`` list under ``[tool.bux]`` in the nearest pyproject.toml, relative
   to the folder it is in;
2. then the folders in ``BUX_PATH``, separated as in ``PATH`` (``:``, or ``;`` on
   Windows).

The first folder that has the module wins. Folders organize files without becoming
part of their names, so a name must be unique under each one. Hidden folders,
``__pycache__`` and ``node_modules`` are skipped. A pyproject.toml such as::

    [tool.bux]
    path = ["examples"]

lets any file of the project ``use llama3_8b``, wherever it is under ``examples/``.

A program (`load`) is the main file, the modules it uses (and the ones those use),
and the command line's settings, as layers of definitions: modules first, in the
order they are used, then the main file, then the command line. A layer's
definitions take the place of the same names below it, everywhere, including inside
the definitions below.
"""

import difflib
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from bux.lang import syntax as ast
from bux.lang.syntax import BuxError, Span

#: The filename of the command line's settings, as their errors show it.
COMMAND_LINE = "<command line>"
_SKIP = frozenset({"__pycache__", "node_modules"})

#: What finds a module, as ``loader(name, importer)``.
#:
#: It returns the source and filename of the module `name`, for the file
#: `importer` that uses it, or raises a `~bux.lang.syntax.BuxError`. `find`
#: searches files; `from_mapping` makes one over sources in memory.
Loader = Callable[[str, str], tuple[str, str]]


def _search_path(start: Path) -> list[Path]:
    for folder in (start, *start.parents):
        pyproject = folder / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as f:
                tool = tomllib.load(f).get("tool", {}).get("bux", {})
            return [folder / p for p in tool.get("path", [])]
    return []


def _roots(importer: str) -> list[tuple[Path, bool]]:
    """Return where to look, and whether in subfolders: the folder of the file that
    uses the module on its own, then the search path."""
    # An editor's file may not be saved yet; "<string>" and the like have no folder.
    here = (Path.cwd() if importer.startswith("<") else Path(importer).parent).resolve()
    env = [Path(p) for p in os.environ.get("BUX_PATH", "").split(os.pathsep) if p]
    roots = [(here, False)]
    for root in _search_path(here) + env:
        if root.is_dir() and (root.resolve(), True) not in roots:
            roots.append((root.resolve(), True))
    return roots


def _modules(root: Path, recursive: bool) -> dict[str, list[Path]]:
    """Return the .bux files under `root` by name, skipping hidden folders."""
    found: dict[str, list[Path]] = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _SKIP)
        if not recursive:
            dirs.clear()
        for name in sorted(files):
            if name.endswith(".bux"):
                found.setdefault(name[:-4], []).append(Path(folder) / name)
    return found


def _check_name(name: str) -> None:
    if "/" in name or "\\" in name or name.endswith(".bux"):
        stem = Path(name).stem
        raise BuxError(
            f"modules are found by name, not by path: '{name}'",
            hint=f"use {stem}, with its folder on the search path ([tool.bux] path)",
        )


def find(name: str, importer: str) -> tuple[str, str]:
    """Return the source and filename of the module `name`: the default `Loader`.

    It searches the folder of `importer`, then the search path, as the module's
    description says.

    Args:
        name: The module, as ``use`` names it: ``llama3_8b``, never a path.
        importer: The filename of the file that uses it. A name in angle brackets,
            such as ``<string>``, searches the working directory first.

    Returns:
        The module's text, and its path. The path is relative to the working
        directory if it is under it, else absolute.

    Raises:
        BuxError: If no folder has the module (with a hint of a close name), or one
            has two files of that name.
    """
    known: list[str] = []
    for root, recursive in _roots(importer):
        modules = _modules(root, recursive)
        matches = modules.get(name, [])
        if len(matches) > 1:
            listed = ", ".join(str(p.relative_to(root)) for p in matches)
            raise BuxError(
                f"there are {len(matches)} modules named '{name}' under {root}: {listed}",
                hint="rename one of them",
            )
        if matches:
            path = matches[0]
            # Relative to the working directory, like the file named on the command line.
            shown = (
                os.path.relpath(path) if path.is_relative_to(Path.cwd()) else str(path)
            )
            return path.read_text(encoding="utf-8"), shown
        known += modules
    close = difflib.get_close_matches(name, known, n=1)
    raise BuxError(
        f"no module named '{name}'",
        hint=f"did you mean '{close[0]}'?"
        if close
        else "add its folder to [tool.bux] path in pyproject.toml",
    )


def from_mapping(modules: Mapping[str, str]) -> Loader:
    """Return a `Loader` over sources held in memory, by module name.

    `bux.lang.compile` and the others make one of a mapping given as `modules`, so
    a test or a notebook can ``use`` modules that are not files.

    Args:
        modules: Each module's source, by its name: ``{"lib": "norm = rmsnorm"}``.

    Returns:
        A loader that names each module's file ``<name>.bux``, and raises a
        `~bux.lang.syntax.BuxError` for a name the mapping does not hold.
    """

    def find_in(name: str, importer: str) -> tuple[str, str]:
        if name not in modules:
            close = difflib.get_close_matches(name, list(modules), n=1)
            raise BuxError(
                f"no module named '{name}'",
                hint=f"did you mean '{close[0]}'?" if close else None,
            )
        return modules[name], f"{name}.bux"

    return find_in


@dataclass(frozen=True)
class Unit:
    """One file of a program, at offset `start` of the joined source.

    Attributes:
        filename: The file's name, as errors show it.
        text: Its source.
        start: Where it starts in `Program.source`.
        via: The ``use`` in the main file that brought it in, perhaps through other
            modules; None for the main file.
    """

    filename: str
    text: str
    start: int
    via: Span | None = None


@dataclass
class Program:
    """Every file of a program, parsed, with the definitions in layers.

    Attributes:
        source: The files joined, the main one first, so that offsets into it are
            the main file's own.
        units: The files, the main one first, in the order they were loaded.
        layers: The definitions, a list a layer, the lowest first: each module's,
            then its amendments, then the main file's, then the command line's.
        main: The main file's own statements.
        uses: The main file's ``use`` statements.
        errors: The errors, from a recovering load; else empty.
        broken: The names of definitions that failed to parse.
        incomplete: Whether a module could not be found or parsed, so that names it
            might define are not errors.
    """

    source: str
    units: list[Unit]
    layers: list[list[ast.Statement]]
    main: list[ast.Statement]
    uses: list[ast.Use]
    errors: list[BuxError] = field(default_factory=list)
    broken: set[str] = field(default_factory=set)
    incomplete: bool = False

    def unit(self, offset: int) -> Unit:
        """Return the file that `offset` into `source` is in."""
        return next(u for u in reversed(self.units) if u.start <= offset)

    def locate(self, err: BuxError) -> BuxError:
        """Make the span of `err` local to the file it is in, and attach that file."""
        if err.source is not None:
            return err
        unit = self.units[0]
        if err.span is not None:
            unit = self.unit(err.span.start)
            err.span = Span(err.span.start - unit.start, err.span.end - unit.start)
        return err.attach(unit.text, unit.filename)


def _layers(statements: list[ast.Statement]) -> list[list[ast.Statement]]:
    """Return a file's definitions, then its amendments on top, one per name."""
    plain = [st for st in statements if not st.amends]
    amends: dict[str, ast.Statement] = {}
    for st in statements:
        if not st.amends:
            continue
        if st.name in amends:
            prev = amends[st.name]
            call, more = cast(ast.Call, prev.body), cast(ast.Call, st.body)
            body = replace(
                call, span=call.span.to(more.span), args=call.args + more.args
            )
            st = replace(prev, span=prev.span.to(st.span), body=body)
        amends[st.name] = st
    return [plain, list(amends.values())] if amends else [plain]


def load(
    source: str,
    filename: str = "<string>",
    *,
    overrides: Sequence[str] = (),
    loader: Loader | None = None,
    recover: bool = False,
) -> Program:
    """Parse a file, the modules it uses and command-line settings into a `Program`.

    Each module is loaded once, however many files use it. Modules that the command
    line uses (``"use recipe"``) go between the file and the command line's settings.

    Args:
        source: The main file's text.
        filename: Its name: modules are searched for from its folder.
        overrides: The command line's statements, which redefine the file's own.
        loader: What finds a module: `find` by default.
        recover: Collect errors in `Program.errors` and keep what parses, as
            `bux.lang.check` needs, instead of raising the first.

    Returns:
        The program, with nothing evaluated yet.

    Raises:
        BuxError: Without `recover`, at the first syntax error, missing module or
            cycle of modules that use each other, attached to its file.
    """
    linker = _Linker(source, filename, loader or find, recover)
    program = linker.program
    main = linker.parse(0, len(source))
    chain = (_file_key(filename),)
    program.main, program.uses = main.statements, main.uses
    program.layers = linker.use(main.uses, filename, chain, None)
    program.layers += _layers(main.statements)
    if overrides:
        # Modules used on the command line go between the file and the settings there.
        extra = linker.add("\n".join(overrides), COMMAND_LINE, None)
        program.layers += linker.use(extra.uses, filename, chain, None)
        program.layers += _layers(extra.statements)
    return program


def _file_key(name: str) -> str:
    """Return which file `name` is, however written."""
    return name if name.startswith("<") else str(Path(name).resolve())


class _Linker:
    def __init__(self, source: str, filename: str, loader: Loader, recover: bool):
        self.program = Program(source, [Unit(filename, source, 0)], [], [], [])
        self.loader = loader
        self.recover = recover
        self.placed: set[str] = set()

    def fail(self, err: BuxError) -> None:
        if not self.recover:
            raise self.program.locate(err)
        self.program.errors.append(err)

    def parse(self, start: int, stop: int) -> ast.Parsed:
        program = self.program
        try:
            parsed = ast.parse_file(program.source, start, stop, recover=self.recover)
        except BuxError as err:
            self.fail(err)
            return ast.Parsed([], [])
        for err in parsed.errors:
            self.fail(err)
        program.broken |= parsed.broken
        # A module's broken statements could define anything.
        program.incomplete |= bool(parsed.errors) and start > 0
        return parsed

    def add(self, text: str, name: str, via: Span | None) -> ast.Parsed:
        start = len(self.program.source) + 1
        self.program.source += "\n" + text
        self.program.units.append(Unit(name, text, start, via))
        return self.parse(start, len(self.program.source))

    def use(
        self,
        uses: list[ast.Use],
        importer: str,
        chain: tuple[str, ...],
        via: Span | None,
    ) -> list[list[ast.Statement]]:
        """Return the layers of the modules in `uses`, each above the modules it uses."""
        layers: list[list[ast.Statement]] = []
        for statement in uses:
            try:
                _check_name(statement.module)
                text, name = self.loader(statement.module, importer)
            except BuxError as err:
                self.program.incomplete = True
                self.fail(BuxError(err.message, statement.module_span, err.hint))
                continue
            key = _file_key(name)
            if key in chain:
                self.program.incomplete = True
                cycle = " -> ".join(Path(n).stem for n in (*chain, name))
                error = BuxError(
                    f"modules use each other: {cycle}", statement.module_span
                )
                self.fail(error)
                continue
            if key in self.placed:
                continue
            self.placed.add(key)
            parsed = self.add(text, name, via or statement.span)
            layers += self.use(parsed.uses, name, (*chain, key), via or statement.span)
            layers += _layers(parsed.statements)
        return layers
