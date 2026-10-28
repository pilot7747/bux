"""The view model: what the page draws of a file, as the plan's ``View``.

`view` checks the source, and with no errors traces the run the page picked
(`bux.studio.picks.traced`) and turns its trace into JSON: the model's graph as a tree
of nodes with ids by path (``"n"``, ``"n.0.2"``), each with where it is written; the
model's facts; the file's definitions; the training of the run (`bux.studio.training`);
and the sweep's axes, runs and search dimensions. It needs no JAX::

    v = view(open("examples/architectures/llama3_8b.bux").read(), {}, path=None)
    v["model"]["graph"]["items"][1]["kind"]  # "repeat": block^32
"""

import dataclasses
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from bux import config as cfg
from bux.lang import ir, modules
from bux.lang import syntax as ast
from bux.lang.compiler import (
    LAYERS,
    SETTINGS,
    TRAIN_SETTINGS,
    Axis,
    Dimension,
    Report,
    Trace,
    check,
    compile_runs,
)
from bux.lang.printer import config_to_source
from bux.lang.syntax import BuxError, Span
from bux.studio import internals, palette, training
from bux.studio.picks import axis_id, squeezed, traced, written_id
from bux.train import config as tcfg

#: A JSON object, as the API sends it.
type Json = dict[str, Any]

#: The most runs a view lists; `Sweep.total` counts them all.
RUNS = 256


def view(source: str, picks: Mapping[str, int], *, path: str | None) -> Json:
    """Return the plan's ``View`` of `source`, the run that `picks` selects.

    Args:
        source: The file's text, as the page has it, perhaps not saved.
        picks: The option each axis of the sweep takes, by `axis_id`; those left
            out take their first.
        path: Where the file is, which names it and which ``use`` searches from;
            None for a file not saved yet.

    Returns:
        ``ok`` and the errors, the model, the run, the sweep and the outline. With
        errors, the model, the run and the sweep are empty; a module has no model.
    """
    report = check(source, filename=path or "<string>")
    outline = _outline(report)
    if report.errors:
        return _view(errors(report, path), outline)
    if report.arch is None:
        return _view([], outline)
    try:
        t = traced(source, picks, path=path)
    except BuxError as err:
        return _view([_error(err, None)], outline)
    sweep, name = _sweep(t, source, picks, path)
    run = training.run_view(t, name=name) if t.train is not None else None
    model = {
        "name": _stem(path),
        "facts": facts(t.arch),
        "graph": graph(t, len(source)),
    }
    return {"ok": True, "errors": [], "model": model, "run": run, "sweep": sweep} | {
        "outline": outline
    }


def _view(errors: list[Json], outline: list[Json]) -> Json:
    """Return a view without a model: of a module, or of a file with errors."""
    return {
        "ok": not errors,
        "errors": errors,
        "model": None,
        "run": None,
        "sweep": _no_sweep(),
        "outline": outline,
    }


def _stem(path: str | None) -> str:
    return Path(path).stem if path else "untitled"


# --- Errors ----------------------------------------------------------------------------


def errors(report: Report, path: str | None) -> list[Json]:
    """Return the plan's ``Error`` of each error of `report`. One in a module is at
    the ``use`` that brought it in, as `bux.lang.lsp.features.diagnostics` puts it,
    its message prefixed with the module's file and line, and its `file` the
    module's."""
    filename = path or "<string>"
    found = []
    for err in report.errors:
        if err.filename == filename:
            found.append(_error(err, None))
        elif (via := report.via.get(err.filename)) is not None:
            found.append(_error(err, err.filename, via, report.source))
    return found


def _error(
    err: BuxError, module: str | None, via: Span | None = None, source: str = ""
) -> Json:
    """Return `err` as the plan's ``Error``; at `via` in `source` for an error in
    `module`."""
    span = err.span or Span(0, 0)
    message, text = err.message, err.source or ""
    if module is not None and via is not None:
        line = text.count("\n", 0, span.start) + 1
        message = f"{Path(module).name}:{line}: {message}"
        span, text = via, source
    line, col = _line_col(text, span.start)
    return {
        "start": span.start,
        "end": span.end,
        "line": line,
        "col": col,
        "message": message,
        "hint": err.hint,
        "file": module,
    }


def _line_col(text: str, offset: int) -> tuple[int, int]:
    """Return the line and column of `offset` in `text`, both from 1."""
    offset = min(offset, len(text))
    return text.count("\n", 0, offset) + 1, offset - text.rfind("\n", 0, offset)


# --- The model -------------------------------------------------------------------------


def facts(arch: ir.Architecture) -> Json:
    """Return the model's facts, as ``bux check``'s summary says them: its width,
    vocabulary, layers, residual algebra and init, and its layers by type."""
    counts = Counter(layer.config.type for layer in arch.layers())
    return {
        "d_model": arch.d_model,
        "vocab_size": arch.vocab_size,
        "layers": counts.total(),
        "residual": None if arch.residual is None else config_to_source(arch.residual),
        "init": None if arch.init is None else config_to_source(arch.init),
        "by_type": [[t, n] for t, n in counts.most_common()],
    }


def graph(t: Trace, size: int) -> Json:
    """Return the model's graph as the plan's tree of ``Node``, for a file of `size`
    characters: spans past it are in modules."""
    return _Drawer(t, size).node(t.arch.graph, ())


class _Drawer:
    """Turns the nodes of a trace into the view's, each layer's look worked out once."""

    def __init__(self, t: Trace, size: int):
        self.t, self.size = t, size
        self.strings = _strings(t.program, size)
        # By the config's id and the node's width: uses of a definition share them.
        self.looks: dict[tuple[int, ir.Width | None], Json] = {}

    def node(self, node: ir.Node, path: tuple[int, ...]) -> Json:
        nid = "n" + "".join(f".{k}" for k in path)
        span = self.t.nodes.get(path)
        here = span is not None and span.start < self.size
        out: Json = {
            "id": nid,
            "kind": node.kind,
            "label": node.label,
            "shared": node.share is not None,
            "span": [span.start, span.end] if here and span else None,
            "origin": self.origin(span) if span is not None and not here else None,
            "editable": here and span is not None and not self.in_pattern(span),
        }
        kids = [self.node(c, (*path, k)) for k, c in enumerate(ir.children(node))]
        match node:
            case ir.Layer():
                out |= self.layer(node, nid, span is None)
            case ir.Seq():
                out["items"] = kids
            case ir.Sum():
                out["terms"] = kids
            case ir.Prod():
                out["factors"] = kids
            case ir.Repeat(times=times):
                out |= {"body": kids[0], "times": times}
            case ir.Residual() | ir.Neg():
                out["body"] = kids[0]
            case ir.Image(token=token):
                out |= {"body": kids[0], "token": token}
            case ir.Const(value=value):
                out["value"] = value
            case ir.Tap(name=name):
                out["name"] = name
            case ir.Ref(name=name, pop=pop):
                out |= {"name": name, "pop": pop}
        return out

    def layer(self, node: ir.Layer, nid: str, unwritten: bool) -> Json:
        config = node.config
        ctx = palette.Context(self.t.arch.d_model, self.t.arch.vocab_size, node.width)
        key = (id(config), node.width)
        if key not in self.looks:
            self.looks[key] = self.look(config, ctx)
        look = self.looks[key]
        inside = look["internals"]
        return look | {
            "ports": [p.model_dump() for p in node.ports],
            "internals": None
            if inside is None
            else internals.numbered(inside, nid + ":i"),
            "inserted": unwritten and config.type == "merge",
        }

    def look(self, config: cfg.LayerConfig, ctx: palette.Context) -> Json:
        """Return what every use of `config` at this width shares: its title,
        category, chips, width, where it is written and its inside, without ids."""
        w = self.t.written(config)
        tag = config.type
        return {
            "type": tag,
            "title": palette.title(config),
            "category": palette.category(tag),
            "chips": palette.chips_of(config, ctx),
            "width": ctx.width,
            "written": [w.span.start, w.span.end]
            if w and w.span.start < self.size
            else None,
            "internals": _inside(config, ctx),
        }

    def origin(self, span: Span) -> str:
        """Return the name of the module `span` is in."""
        return Path(self.t.program.unit(span.start).filename).stem

    def in_pattern(self, span: Span) -> bool:
        """Whether `span` is a letter of a pattern string, which is not edited alone."""
        return any(s.start < span.start and span.end < s.end for s in self.strings)


def _inside(config: cfg.LayerConfig, ctx: palette.Context) -> Json | None:
    """Return the inside diagram of `config`, without ids, or None."""
    if config.type not in internals.internals.builders:
        return None
    return internals.internals.build(config, ctx)


def _strings(program: modules.Program, size: int) -> list[Span]:
    """Return the spans of the string literals of the file, the first `size`
    characters of the program."""
    return [
        e.span
        for st in _statements(program)
        if st.span.start < size
        for e in _exprs(st)
        if isinstance(e, ast.Lit) and isinstance(e.value, str)
    ]


def _statements(program: modules.Program) -> Iterator[ast.Statement]:
    """Yield every statement of every file of `program`, amendments merged."""
    for layer in program.layers:
        yield from layer


def _exprs(node: object) -> Iterator[ast.Expr]:
    """Yield every expression under `node`: a statement, an expression, an argument
    or a tuple of them."""
    if isinstance(node, tuple):
        for item in node:
            yield from _exprs(item)
        return
    if isinstance(node, ast.Expr):
        yield node
    if isinstance(node, ast.Expr | ast.Arg | ast.Statement):
        for f in dataclasses.fields(node):
            if f.name not in ("span", "name_span", "key_spans", "param_spans"):
                yield from _exprs(getattr(node, f.name))


# --- The outline -----------------------------------------------------------------------


def _outline(report: Report) -> list[Json]:
    """Return the file's definitions, then those it inherits from modules, each with
    its kind, where it is in this file, and the module it is in otherwise."""
    found: dict[str, Json] = {}
    for st in report.statements:
        found.setdefault(
            st.name, _entry(st, report, [st.span.start, st.span.end], None)
        )
    for name, st in report.inherited.items():
        if name not in found:
            found[name] = _entry(st, report, None, Path(report.origin[name]).stem)
    return list(found.values())


def _entry(
    st: ast.Statement, report: Report, span: list[int] | None, origin: str | None
) -> Json:
    return {"name": st.name, "kind": _kind(st, report), "span": span, "origin": origin}


def _kind(st: ast.Statement, report: Report) -> str:
    """Return what a definition is: a macro, a setting, a layer, a block or a value."""
    if st.params is not None:
        return "macro"
    if st.name in SETTINGS or st.name in TRAIN_SETTINGS:
        return "setting"
    value = report.values.get(st.name)
    if isinstance(value, ir.Layer) or (
        isinstance(value, cfg.Config) and getattr(value, "type", None) in LAYERS
    ):
        return "layer"
    if isinstance(value, ir.NODE_TYPES) or st.name == "model":
        return "block"
    return "value"


# --- The sweep -------------------------------------------------------------------------


def _no_sweep() -> Json:
    return {
        "axes": [],
        "runs": [],
        "total": 0,
        "truncated": False,
        "dimensions": [],
        "search": None,
    }


def _sweep(
    t: Trace, source: str, picks: Mapping[str, int], path: str | None
) -> tuple[Json, str]:
    """Return the sweep of the run `t`, and the run's name."""
    size = len(source)
    alts = _alternatives(t.program)
    runs = _runs(source, path) if t.train is not None else ()
    parts = {part: pick for alt in alts.values() for part, pick in alt.parts().items()}
    taken = {part for run in runs for part in run[2]}
    axes = [_axis(a, alts.get(a.span), picks, size) for a in t.axes if _live(a, taken)]
    listed = [
        {"name": name, "id": rid, "picks": dict(parts[p] for p in took if p in parts)}
        for name, rid, took in runs
    ]
    sweep = {
        "axes": axes,
        "runs": listed[:RUNS],
        "total": len(runs),
        "truncated": len(runs) > RUNS,
        "dimensions": [_dimension(d, size) for d in t.dimensions],
        "search": _search(t.train) if t.train is not None and t.dimensions else None,
    }
    return sweep, _name(t, axes, listed, path)


def _name(t: Trace, axes: list[Json], runs: list[Json], path: str | None) -> str:
    """Return the name of the run the axes pick: the sweep's run that takes their
    options, else one named for them."""
    want = {a["id"]: a["pick"] for a in axes}
    for run in runs:
        if run["picks"] == want:
            return run["name"]
    parts = [a.part(a.pick(want[axis_id(a)])) for a in t.axes if axis_id(a) in want]
    stem = _stem(path) if path else "run"
    return stem + (f"[{', '.join(parts)}]" if parts else "")


@dataclass(frozen=True)
class _Alt:
    """An ``a | b`` as written: where it is, its labels, and its options' text."""

    where: str
    labels: tuple[str, ...]
    texts: tuple[str, ...]  # each option, its blanks squeezed as the compiler does

    def id(self) -> str:
        """Return its axis's `axis_id`."""
        return written_id(self.where, self.labels, self.texts)

    def written(self) -> list[str]:
        """Return its options as written, labeled ones with their label."""
        if not self.labels:
            return list(self.texts)
        return [f"{a}: {b}" for a, b in zip(self.labels, self.texts, strict=True)]

    def parts(self) -> dict[str, tuple[str, int]]:
        """Return the axis and option that each part of a run's name stands for:
        a label, or ``where=option`` (`bux.lang.compiler.Axis.part`)."""
        names = self.labels or tuple(f"{self.where}={o}" for o in self.texts)
        return {part: (self.id(), k) for k, part in enumerate(names)}


def _alternatives(program: modules.Program) -> dict[Span, _Alt]:
    """Return every ``a | b`` of the program, by its span."""
    found = {}
    for st in _statements(program):
        for e in _exprs(st):
            if isinstance(e, ast.Alt):
                texts = tuple(squeezed(program.source, o.span) for o in e.options)
                found[e.span] = _Alt(e.where, e.labels, texts)
    return found


def _live(axis: Axis, taken: set[str]) -> bool:
    """Whether some run takes an option of `axis`, or there are no runs: an axis
    that a later definition overrides makes no runs of its own."""
    options = range(len(axis.options))
    return not taken or any(axis.part(axis.pick(k)) in taken for k in options)


def _axis(axis: Axis, alt: _Alt | None, picks: Mapping[str, int], size: int) -> Json:
    key = axis_id(axis)
    pick = picks.get(key, 0)
    return {
        "id": key,
        "where": axis.where,
        "labeled": axis.labeled,
        "pick": pick if 0 <= pick < len(axis.options) else 0,
        "options": list(axis.options) if alt is None else alt.written(),
        "labels": list(axis.options) if axis.labeled else None,
        "span": _span(axis.span, size),
    }


def _span(span: Span, size: int) -> list[int] | None:
    return [span.start, span.end] if span.start < size else None


def _dimension(d: Dimension, size: int) -> Json:
    space = d.space
    choice = isinstance(space, tcfg.ChoiceConfig)
    return {
        "where": d.where,
        "type": space.type,
        "low": None if choice else space.low,
        "high": None if choice else space.high,
        "options": list(space.options) if choice else None,
        "log": isinstance(space, tcfg.LogUniformConfig) or getattr(space, "log", False),
        "span": _span(d.key, size),
    }


def _search(t: tcfg.TrainConfig) -> Json:
    """Return the search's method and trials, and its halving's rungs in words."""
    s, h = t.search, t.search.halving
    halving = None
    if h is not None:
        rungs = ", ".join(palette.si(h.steps(t.steps, r)) for r in range(h.rungs))
        halving = f"rungs of {rungs} steps, eta {h.eta}"
    return {
        "method": config_to_source(s.method),
        "trials": s.trials,
        "halving": halving,
    }


@lru_cache(maxsize=16)
def _runs(
    source: str, path: str | None
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Return every run of the sweep: its name, id and the parts of its name; none
    if a run does not compile (`view` shows the run picked)."""
    try:
        runs = compile_runs(source, filename=path or "<string>")
    except BuxError:
        return ()
    return tuple((run.name, run.id, run.choices) for run in runs)
