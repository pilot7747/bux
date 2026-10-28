"""The W&B dashboard of the timeline ablations: a saved workspace view and a report of
the project their runs log to, made or updated in place.

    uv run --with wandb-workspaces --with wandb python scripts/timeline/dashboard.py \\
        [--notes notes.md] [--keep results.jsonl]

It reads ``WANDB_API_KEY`` and ``WANDB_BASE_URL`` from the environment, and needs
none of bux. Run it again as runs finish: it builds a section for each axis file the
project has runs of (a W&B group is a file), a line per variant (the axis besides the
rung's, the learning rate's, the seed's and the token scan's) beside its parent's
(`PARENTS`), and a grid for each rung. Before any run, it uses `FILES`.

The views show finished runs only, one of each name: of runs that share a name, the
one that trained all its steps, then the one ``--keep`` lists (a JSONL whose lines
carry a W&B run id under ``"wandb"``, as fit.py's results), then the newest. Token
scans (the ``budget`` axis) have grids of their own.

A bux run's config has ``axes``, what it took on each axis by the axis's name
(``{"model": "rope", "d_model": "d768"}``), ``params``, ``active_params`` and its
settings under ``train`` and ``arch``. The results' panels read the W&B Tables and
images (`TABLES`, `PLOTS`) of the newest finished run of the group ``analysis``.
``--notes`` fills the report's text from a Markdown file of ``## name`` sections,
as dashboard_notes.py writes them: ``intro``, ``headline``, ``best``, ``factorial``,
``swa``, ``stability``, ``winners``, ``long``, ``eras``, ``results``, ``axes``,
``caveats``, ``axis: <file>`` for each file's, and ``secondary: ...`` for the
secondary metrics'.

The report opens on the story: today's best architecture (the candidates and the
hybrid at each rung, seed by seed), the factorial screen, the sliding-window
frontier against the hybrid, and the stability findings, before the leaderboard
and a section per axis file.
"""

import argparse
import contextlib
import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces import expr
from wandb_workspaces.workspaces import internal

PROJECT = "bux-timeline"
VIEW = "nw-timeline-v"  # the saved view's internal name, by which a rerun updates it
TITLE = "Timeline: 2020 to late 2026, one change at a time"
RUNG = "d_model"  # the axis of the rungs (rungs.bux), as the leaderboard groups them
RUNGS = ("d512", "d768", "d1024", "d1280")  # before a file has runs
SCAN = "budget"  # the axis of the token scans: t5, t10, t40, t80
NOT_VARIANTS = {"optimizer.lr", "steps", "seed", SCAN}  # besides variants and rungs
# The axis files in the report's order, and the axis their variants are on (before
# the project has runs).
FILES = {
    "positions": "model",
    "norms": "model",
    "attention": "model",
    "ffn": "ffn",
    "moe": "ffn",
    "mixers": "model",
    "embeddings": "model",
    "residuals": "residual",
    "structure": "shape",
    "interactions": "model",
    "zoo": "model",
    "zoo_moe": "fr_experts",
    "sparse": "model",
    "milestones": "model",
    "stacks": "model",
    "stacks2": "residual",
    "factorial": "residual",
    "audit": "residual",
    "swa": "residual",
    "baseline": None,
}
REFERENCE = "positions/rope"  # the parent of an axis file's variants: file/label
# Each file's parent family where the analysis's leaderboard does not name one.
PARENTS: dict[str, str | None] = {
    **dict.fromkeys(("positions", "milestones"), "baseline"),
    **dict.fromkeys(
        ("stacks", "stacks2", "factorial", "audit"), "milestones/m2026_hybrid"
    ),
    **dict.fromkeys(("swa", "zoo_moe"), "milestones/m2026_frontier"),
    "baseline": None,
}
HYBRID = ("milestones/m2026_hybrid", "factorial/r0v0k0f0p0s0")  # one model, two files
CANDIDATES = ("factorial/r1v1k1f0p0s0", "factorial/r1v1k0f0p0s0")
FRONTIER = ("milestones/m2026_frontier",)
SWA = ("swa",)  # a file alone is every family of it
FIXED_LR = "2e-3"  # the factorial's learning rate, which the comparisons hold
HIGH_LR = "8e-3"  # the top of the stability claims' scans
# The stability findings' families: who survives 8e-3, logit growth, massive
# activations, and the switch to rows of 16k.
SURVIVORS = ("norms/qknorm", "attention/gated", "residuals/hc", "residuals/mhc")
SURVIVORS += ("norms/sandwich", "norms/postnorm")
LOGITS = ("positions/rope", "norms/qknorm")
CREST = ("positions/rope", "attention/gated", "interactions/gated_qk")
CREST += ("milestones/m2026_hybrid", "factorial/r1v1k1f0p0s0")
SWITCH = ("positions/rope", "positions/rope_noyarn", "sparse/csa")
ANALYSIS = "analysis"  # the group of the runs that log the results' tables
TABLES = {
    "best": "Today's best architecture at equal serving cost, against the hybrid and 2020",
    "factorial": "The factorial's main effects",
    "leaderboard": "Every change at its target: Δ by rung, multiplier, (a), (b)",
    "extrapolation_unlimited": "(a) Unlimited data: GPU-hours to the parent's loss",
    "extrapolation_limited": "(b) Limited data: loss on the same tokens",
    "multipliers": "Effective-parameter multipliers at each rung",
    "long_context": "Repositories after 8k tokens against their first tokens",
    "laws": "The fitted laws, L = E + A N^-alpha + B D^-beta",
    "lr": "Each variant's best learning rate",
    "points": "The points the laws fit",
}
PLOTS = [
    f"plots/{k}-{m}"
    for k in ("ladder", "unlimited", "limited")
    for m in ("objective", "repo_bpb")
]
OBJECTIVE = "objective"  # 0.5 code_bpb + 0.3 repo_bpb + 0.2 web_bpb
EVALS = ("code_bpb", "repo_bpb", "web_bpb")
POSITIONS = ("repo_bpb_0", "repo_bpb_2k", "repo_bpb_8k")  # repo_bpb by position
LR = "train.optimizer.lr"
X = "tokens"
COLUMNS = (
    *(f"summary:{m}" for m in (OBJECTIVE, *EVALS, "tokens_per_s", "mfu")),
    f"config:axes.value.{RUNG}",
    "config:train.value.optimizer.lr",
    "config:active_params.value",
    "config:params.value",
)
BEST = [expr.Ordering(expr.Summary(OBJECTIVE), ascending=True)]
LEAST = {
    "groupby_aggfunc": "min",
    "aggregate": True,
}  # a group's best run at each point
BAND = LEAST | {"groupby_rangefunc": "minmax"}  # shaded to the group's range


@dataclass
class File:
    """An axis file's runs: the axis of its variants, the axis of its rungs, the rungs
    it has runs at, and whether it has token scans."""

    variant: str | None
    rung: str = RUNG
    rungs: list[str] = field(default_factory=list)
    scans: bool = False


@dataclass
class Study:
    """What the project has runs of: its axis files, where each rung's long phase
    starts and ends in tokens, the runs the views hide, the token scans' labels, the
    newest analysis run with its tables and each family's parent, and each family's
    learning rates."""

    files: dict[str, File] = field(default_factory=dict)
    phases: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    hidden: set[str] = field(default_factory=set)
    scans: set[str] = field(default_factory=set)
    analysis: str | None = None
    tables: set[str] = field(default_factory=set)  # the analysis run's
    refs: dict[str, str] = field(default_factory=dict)  # each family's parent
    lrs: dict[str, set[str]] = field(default_factory=dict)  # each family's lr labels
    seeds: set[str] = field(default_factory=set)  # the seed axis's labels


def study(api: wandb.Api, path: str, keep: set[str]) -> Study:
    """Return what the finished runs at `path` are, or `FILES` before any."""
    found, named, analyses = Study(), dict[str, list](), []
    for run in api.runs(path, filters={"state": "finished"}, per_page=500):
        if run.group == ANALYSIS:
            analyses.append(run)
            continue
        named.setdefault(run.name, []).append(run)
        _add(found, run)
    if analyses:
        newest = max(analyses, key=lambda r: r.created_at)
        found.analysis, found.tables = newest.id, set(_tables(newest))
        found.refs = _refs(newest)
    for runs in named.values():
        runs.sort(key=lambda r: (_complete(r), r.id in keep, r.created_at))
        found.hidden |= {r.id for r in runs[:-1]}
    if not found.files:
        found.files = {name: File(v, rungs=list(RUNGS)) for name, v in FILES.items()}
    for f in found.files.values():
        f.rungs.sort(key=lambda r: int(r[1:]))
    order = list(FILES)
    rank = {
        name: order.index(name) if name in order else len(order) for name in found.files
    }
    found.files = dict(sorted(found.files.items(), key=lambda item: rank[item[0]]))
    return found


def _tables(run: Any) -> list[str]:
    """The names of the W&B Tables `run` logged."""
    logged = run.summary.items()
    return [
        k
        for k, v in logged
        if getattr(v, "get", None) and v.get("_type") == "table-file"
    ]


def _refs(run: Any) -> dict[str, str]:
    """Each family's parent, from the leaderboard table `run` logged."""
    table = run.summary.get("leaderboard")
    if not table:
        return {}
    with tempfile.TemporaryDirectory() as folder:
        path = run.file(table["path"]).download(root=folder, replace=True).name
        logged = json.loads(Path(path).read_text())
    family, ref = (logged["columns"].index(c) for c in ("family", "reference"))
    return {row[family]: row[ref] for row in logged["data"] if row[ref]}


def _add(found: Study, run: Any) -> None:
    """Add what `run` says of its file: its variants' axis, its rung, its scan, its
    learning rate and its seed."""
    axes = run.config.get("axes") or {}
    rung = next((k for k, v in axes.items() if re.fullmatch(r"d\d+", v)), None)
    rest = [k for k in axes if k != rung and k not in NOT_VARIANTS]
    f = found.files.setdefault(run.group, File(next(iter(rest), None)))
    if f.variant and f.variant in axes:
        lrs = found.lrs.setdefault(f"{run.group}/{axes[f.variant]}", set())
        lrs.add(axes.get("optimizer.lr", FIXED_LR))
    if "seed" in axes:
        found.seeds.add(axes["seed"])
    if SCAN in axes:
        f.scans = True
        found.scans.add(axes[SCAN])
    elif rung is not None and axes[rung] not in f.rungs:
        f.rung = rung
        f.rungs.append(axes[rung])
        found.phases.setdefault(axes[rung], _long_phase(run.config.get("train") or {}))


def _complete(run: Any) -> bool:
    """Whether `run` trained every step it was set to."""
    steps = (run.config.get("train") or {}).get("steps") or 0
    return (run.summary.get("_step") or 0) >= steps


def _long_phase(train: dict) -> tuple[float | None, float | None]:
    """Return the tokens where a run's last phase starts and where the run ends."""
    steps, rows, length = train.get("steps"), train.get("batch"), train.get("seq_len")
    phases = train.get("curriculum") or []
    if not (steps and rows and length and phases):
        return (None, None)
    at = phases[-1].get("at", 0)
    start = at * steps if 0 < at < 1 else at  # a share of the steps, or a step
    return (start * rows * length, steps * rows * length)


def kept(ids: Path | None) -> set[str]:
    """The W&B run ids that the JSONL file `ids` lists, under ``"wandb"``."""
    if ids is None:
        return set()
    lines = (line.partition("{")[2] for line in ids.read_text().splitlines())
    return {json.loads("{" + line).get("wandb") for line in lines if line}


def notes(path: Path | None) -> dict[str, str]:
    """The Markdown sections of `path` by their ``##`` heading, none without one."""
    if path is None:
        return {}
    parts = re.split(r"^## (.+)$", path.read_text(), flags=re.MULTILINE)
    return {name.strip(): text.strip() for name, text in zip(parts[1::2], parts[2::2])}


# --- Filters ----------------------------------------------------------------------------


def shown(s: Study) -> str:
    """The filter of the runs every view shows: finished, one of each name, no token
    scan."""
    rules = [
        "State = 'finished'",
        *(f"Config('axes.{SCAN}') != '{t}'" for t in s.scans),
    ]
    if s.hidden:
        rules.append(f"Metric('ID') not in [{', '.join(map(repr, sorted(s.hidden)))}]")
    return " and ".join(rules)


def family(name: str, s: Study) -> str:
    """The filter of a family's runs, ``milestones/m2026_hybrid``, or of a whole
    file's, ``swa``."""
    group, _, label = name.partition("/")
    mine = f"Metric('Group') = '{group}'"
    variant = s.files[group].variant if group in s.files else FILES.get(group)
    if not label or not variant:
        return mine
    return f"({mine} and Config('axes.{variant}') = '{label}')"


def families(names: Iterable[str], s: Study) -> str:
    """The filter of the runs of any of the families `names`."""
    return "(" + " or ".join(family(n, s) for n in names) + ")"


def parents(file: str, s: Study) -> set[str]:
    """The families an axis file's variants are measured against: the analysis's
    leaderboard's, else `PARENTS`, else `REFERENCE`."""
    named = {ref for fam, ref in s.refs.items() if fam.split("/")[0] == file}
    named -= {fam for fam in s.refs if fam.split("/")[0] == file}
    if named:
        return named
    parent = PARENTS.get(file, REFERENCE)
    return set() if parent is None else {parent}


def group_is(file: str, s: Study) -> str:
    """The filter of an axis file's runs and its parents'."""
    return families([file, *sorted(parents(file, s))], s)


def one_lr(s: Study) -> str:
    """The filter of the runs at `FIXED_LR`, or with no learning-rate axis."""
    others = {lr for lrs in s.lrs.values() for lr in lrs} - {FIXED_LR}
    return " and ".join(
        f"Config('axes.optimizer.lr') != '{lr}'" for lr in sorted(others)
    )


def at_rung(file: str, s: Study, rung: str | None) -> str:
    """The filter of an axis file's runs and its parent's at `rung`."""
    rules = [shown(s), group_is(file, s)]
    if rung is not None:
        rules.append(f"Config('axes.{s.files[file].rung}') = '{rung}'")
    return " and ".join(rules)


# --- Panels -----------------------------------------------------------------------------


def curve(y: str | Iterable[str], title: str, **kw: Any) -> wr.LinePlot:
    """A line plot of `y` against the tokens trained on."""
    ys = [y] if isinstance(y, str) else list(y)
    return wr.LinePlot(title=title, x=X, y=ys, title_x="tokens", **kw)


def layers(pattern: str, title: str) -> wr.LinePlot:
    """A line plot of every layer's metric that `pattern` matches, by step."""
    return wr.LinePlot(title=title, x="Step", metric_regex=pattern, max_runs_to_show=8)


def bars(metrics: Iterable[str], title: str, **kw: Any) -> wr.BarPlot:
    """A bar of each run, or of each group, for its latest `metrics`."""
    kw.setdefault("max_bars_to_show", 40)
    return wr.BarPlot(title=title, metrics=list(metrics), orientation="h", **kw)


def scatter(x: Any, y: str, title: str, *, log_x: bool = True) -> wr.ScatterPlot:
    """Each run's final `y` against `x`, a config or a summary metric."""
    return wr.ScatterPlot(title=title, x=x, y=wr.SummaryMetric(y), log_x=log_x)


def lr_scatter(y: str = OBJECTIVE) -> wr.ScatterPlot:
    """Each run's final `y` against its learning rate."""
    return scatter(wr.Config(LR), y, f"{y} by learning rate")


def coordinates() -> wr.ParallelCoordinatesPlot:
    """Each run's rung, learning rate, size, speed and objective."""
    col = wr.ParallelCoordinatesPlotColumn
    return wr.ParallelCoordinatesPlot(
        title="rung, lr, size, speed and objective",
        columns=[
            col(metric=wr.Config(f"axes.{RUNG}"), display_name="rung"),
            col(metric=wr.Config(LR), display_name="lr", log=True),
            col(metric=wr.Config("active_params"), display_name="active", log=True),
            col(metric=wr.SummaryMetric("tokens_per_s"), display_name="tok/s"),
            col(metric=wr.SummaryMetric(OBJECTIVE), display_name="objective"),
        ],
    )


def evals(**kw: Any) -> list[wr.LinePlot]:
    """The objective and each eval's bits per byte, by tokens."""
    return [curve(m, m, **kw) for m in (OBJECTIVE, *EVALS)]


def training() -> list[wr.LinePlot]:
    """The training loss smoothed and raw, its z-loss, the gradient's norm, the lr."""
    smooth = {"smoothing_type": "exponential", "smoothing_factor": 0.8}
    return [
        curve("loss", "loss, smoothed", **smooth),
        curve("loss", "loss, raw: spikes"),
        curve("grad_norm", "grad_norm, before clipping"),
        curve(["z_loss", "balance_loss"], "z_loss and balance_loss"),
        curve("lr", "lr"),
    ]


def stability() -> list[wr.LinePlot]:
    """The observed metrics that say a run is going wrong, every layer's."""
    return [
        layers("attn/logit_max/.*", "max attention logit (logit growth)"),
        layers("attn/entropy/.*", "attention entropy (collapse toward 0)"),
        layers("act/max_abs/.*", "largest activation (massive activations)"),
        layers("act/rms/.*", "residual stream rms by depth"),
        layers("z_loss/.*", "head log Z^2 (logit drift)"),
        layers("update/ratio/.*", "update / param by layer"),
        layers("grad/norm/.*", "gradient norm by layer"),
        layers("router/max_load/.*", "router max load (imbalance)"),
    ]


def speed(**kw: Any) -> list[wr.LinePlot | wr.BarPlot]:
    """Throughput and memory, by run and over training."""
    return [
        bars(["tokens_per_s"], "tokens/s", **kw),
        bars(["mfu"], "MFU", **kw),
        curve("tokens_per_s", "tokens/s over training", **kw),
        curve("time/step_ms", "step time, ms", **kw),
        bars(["mem/peak_gib"], "peak memory per GPU, GiB", **kw),
        curve(["time/data_ms", "time/wait_ms"], "host: data and wait, ms per step"),
    ]


def results(s: Study) -> list[wr.WeavePanelSummaryTable | wr.MediaBrowser]:
    """The analysis's tables that it logs, and its plots."""
    tables = [wr.WeavePanelSummaryTable(table_name=t) for t in TABLES if t in s.tables]
    plots = wr.MediaBrowser(title="fits", media_keys=PLOTS, num_columns=2)
    return [*tables, plots]


# --- The workspace ----------------------------------------------------------------------


def section(name: str, panels: list, *, is_open: bool = False) -> ws.Section:
    """A workspace section of `panels`, three a row, all on one page."""
    layout = ws.SectionLayoutSettings(columns=3, rows=max(2, -(-len(panels) // 3)))
    return ws.Section(name=name, panels=panels, is_open=is_open, layout_settings=layout)


def workspace(entity: str, project: str, s: Study) -> ws.Workspace:
    """The saved view: the shown runs grouped by file and rung, a line or bar per
    group at its best run, with the panels by topic."""
    leaderboard = [
        bars([OBJECTIVE], "best objective of each file and rung", **LEAST),
        bars(EVALS, "code, repo, web bpb", **LEAST),
        scatter(
            wr.Config("active_params"), OBJECTIVE, "objective by active parameters"
        ),
        coordinates(),
        wr.RunComparer(diff_only="split"),
    ]
    long = [bars(POSITIONS, "repo_bpb by position", **LEAST)]
    long.append(curve(list(POSITIONS), "repo_bpb by position", **LEAST))
    view = ws.Workspace(
        entity=entity,
        project=project,
        name="timeline",
        sections=[
            section("Leaderboard", leaderboard, is_open=True),
            section("Evals", evals(**BAND), is_open=True),
            section("Long context", long),
            section("Training", training()),
            section("Stability", stability()),
            section("LR sensitivity", [lr_scatter(), lr_scatter("repo_bpb")]),
            section("Throughput", speed(**LEAST)),
            section("Results", results(s)),
        ],
        settings=ws.WorkspaceSettings(x_axis=X, max_runs=100),
        runset_settings=ws.RunsetSettings(
            filters=shown(s),
            groupby=[expr.Metric("Group"), expr.Config(f"axes.{RUNG}")],
            order=BEST,
            pinned_columns=list(COLUMNS),
        ),
    )
    view._internal_name = VIEW
    with contextlib.suppress(ValueError):  # a first run makes it
        view._internal_id = internal.get_view_dict(entity, project, VIEW)["id"]
    return view


# --- The report -------------------------------------------------------------------------


def runset(e: str, p: str, name: str, filters: str, **kw: Any) -> wr.Runset:
    """A runset of the project's runs that `filters` keeps, best objective first."""
    order = [wr.OrderBy(wr.SummaryMetric(OBJECTIVE), ascending=True)]
    kw.setdefault("pinned_columns", list(COLUMNS))
    return wr.Runset(entity=e, project=p, name=name, filters=filters, order=order, **kw)


def grid(runs: wr.Runset, panels: list, columns: int = 3) -> wr.PanelGrid:
    """A grid of `panels`, over `runs`, `columns` a row."""
    w = 24 // columns
    for i, panel in enumerate(panels):
        panel.layout = wr.Layout(x=w * (i % columns), y=8 * (i // columns), w=w, h=8)
    return wr.PanelGrid(runsets=[runs], panels=panels)


def text(n: dict[str, str], name: str, todo: str) -> wr.MarkdownBlock | wr.CalloutBlock:
    """The note `name`, or a block saying what to fill."""
    if name in n:
        return wr.MarkdownBlock(text=n[name])
    return wr.CalloutBlock(text=f"To fill: {todo}")


def ablation(e: str, p: str, file: str, rung: str | None, s: Study) -> wr.PanelGrid:
    """The grid of one axis file at one rung, beside its parent: a line per variant,
    its best run at each point, shaded to its runs' range, and its long context."""
    f = s.files[file]
    group = ["Group", f"config.axes.{f.variant}"] if f.variant else ["Group"]
    columns = [f"config:axes.value.{f.variant}", *COLUMNS] if f.variant else COLUMNS
    name = f"{file} {rung or ''}".strip()
    filters = at_rung(file, s, rung)
    runs = runset(e, p, name, filters, groupby=group, pinned_columns=list(columns))
    start, end = s.phases.get(rung or "", (None, None))
    return grid(
        runs,
        [
            curve(OBJECTIVE, "objective", **BAND),
            curve("repo_bpb", "repo_bpb, rows of 16k", **BAND),
            bars([OBJECTIVE, *EVALS], "final, best lr", **LEAST),
            bars(POSITIONS, "repo_bpb by position", **LEAST),
            curve(
                "repo_bpb", "repo_bpb in the 16k phase", range_x=(start, end), **BAND
            ),
            lr_scatter(),
            bars(["tokens_per_s", "mfu"], "tokens/s and MFU", aggregate=True),
        ],
    )


def scans(e: str, p: str, file: str, s: Study) -> wr.PanelGrid:
    """The grid of an axis file's token scans: each run's objective by its tokens."""
    labels = ", ".join(map(repr, sorted(s.scans)))
    filters = f"State = 'finished' and Metric('Group') = '{file}'"
    filters += f" and Config('axes.{SCAN}') in [{labels}]"
    panels = [
        scatter(wr.SummaryMetric(X), OBJECTIVE, "objective by tokens"),
        scatter(wr.SummaryMetric(X), "repo_bpb", "repo_bpb by tokens"),
        curve(OBJECTIVE, "objective"),
    ]
    return grid(runset(e, p, f"{file} token scans", filters), panels)


def per_file(e: str, p: str, s: Study, n: dict[str, str]) -> list:
    """A collapsed heading per axis file: its note, a grid per rung, its scans."""
    blocks: list = [wr.H1(text="Per axis")]
    for file, f in s.files.items():
        against = ", ".join(sorted(parents(file, s)))
        beside = f", beside {against}" if against else ""
        how = f"A line per {f.variant or 'run'}{beside}: its best run at each point."
        inner: list = [text(n, f"axis: {file}", f"what {file} found."), wr.P(text=how)]
        for rung in f.rungs or [None]:
            inner += [wr.H3(text=rung or "every rung"), ablation(e, p, file, rung, s)]
        if f.scans:
            inner += [wr.H3(text="token scans"), scans(e, p, file, s)]
        blocks.append(wr.H2(text=file, collapsed_blocks=inner))
    return blocks


def unstable(e: str, p: str, s: Study) -> list:
    """A collapsed heading per axis file, with its runs' signs of instability."""
    blocks: list = [wr.H1(text="Stability by axis")]
    for file in s.files:
        panels = [
            curve("loss", "loss, raw: spikes"),
            curve("grad_norm", "grad_norm, before clipping"),
            curve(["z_loss", "balance_loss"], "z_loss and balance_loss"),
            layers("attn/logit_max/.*", "max attention logit"),
            layers("act/max_abs/.*", "largest activation"),
            layers("update/ratio/.*", "update / param by layer"),
            lr_scatter(),
            lr_scatter("repo_bpb"),
        ]
        runs = runset(e, p, file, f"{shown(s)} and Metric('Group') = '{file}'")
        blocks.append(wr.H2(text=file, collapsed_blocks=[grid(runs, panels)]))
    return blocks


def kept_at(s: Study, rung: str | None, *rules: str, show: bool = True) -> str:
    """The filter of the runs at `FIXED_LR`, at `rung` if given, that `rules` keep
    too; of the shown ones alone with `show`."""
    rung_rule = f"Config('axes.{RUNG}') = '{rung}'" if rung else ""
    kept = (shown(s) if show else "", one_lr(s), rung_rule, *rules)
    return " and ".join(r for r in kept if r)


def by_seed(e: str, p: str, name: str, filters: str) -> wr.Runset:
    """A runset of `filters`'s runs, seed by seed, so that a seed's runs sit side by
    side."""
    order = [wr.OrderBy(wr.Config("axes.seed"), ascending=True)]
    order.append(wr.OrderBy(wr.Metric("Name"), ascending=True))
    return wr.Runset(entity=e, project=p, name=name, filters=filters, order=order)


def table(name: str, s: Study, e: str, p: str) -> list:
    """A grid of the analysis's table `name`, where the analysis logs one."""
    if name not in s.tables:
        return []
    runs = runset(e, p, ANALYSIS, f"Metric('ID') = '{s.analysis}'")
    return [grid(runs, [wr.WeavePanelSummaryTable(table_name=name)], columns=1)]


def best(e: str, p: str, s: Study, n: dict[str, str]) -> list:
    """Today's best architecture: the analysis's table, and the candidates' and the
    hybrid's runs at each rung, seed by seed, and across the rungs."""
    arms = families((*CANDIDATES, *HYBRID), s)
    rungs = s.files["factorial"].rungs if "factorial" in s.files else list(RUNGS)
    blocks: list = [
        wr.H1(text="Today's best architecture"),
        text(n, "best", "the best."),
        *([text(n, "secondary: flags", "")] if "secondary: flags" in n else []),
    ]
    blocks += table("best", s, e, p)
    note = "The candidates and the hybrid (factorial r0v0k0f0p0s0 and milestones "
    note += f"m2026_hybrid, one model), at lr {FIXED_LR}, a run per seed, seed by seed."
    blocks.append(wr.P(text=note))
    for rung in rungs:
        panels = [
            curve(OBJECTIVE, "objective"),
            curve("repo_bpb_8k", "repo_bpb past 8k"),
            curve("loss", "loss", smoothing_type="exponential", smoothing_factor=0.8),
            bars([OBJECTIVE], "final objective, seed by seed"),
            bars(["repo_bpb_8k"], "final repo_bpb past 8k, seed by seed"),
            bars(EVALS, "final code, repo, web bpb"),
        ]
        runs = by_seed(e, p, f"candidates {rung}", kept_at(s, rung, arms))
        blocks += [wr.H3(text=rung), grid(runs, panels)]
    across = [
        scatter(wr.Config("arch.d_model"), OBJECTIVE, "objective by width"),
        scatter(wr.Config("arch.d_model"), "repo_bpb_8k", "repo_bpb past 8k by width"),
        bars(["repo_bpb_8k"], "repo_bpb past 8k, by rung", aggregate=True),
    ]
    group = [f"config.axes.{RUNG}", "Group", "config.axes.residual"]
    runs = runset(e, p, "candidates by rung", kept_at(s, None, arms), groupby=group)
    blocks += [wr.H3(text="by rung"), grid(runs, across)]
    return blocks


def factorial(e: str, p: str, s: Study, n: dict[str, str]) -> list:
    """The factorial screen: its note and main effects, and each arm's finals at
    d512 over its seeds."""
    blocks: list = [
        wr.H1(text="The factorial screen"),
        text(n, "factorial", "the factorial."),
    ]
    blocks += table("factorial", s, e, p)
    arms = families(("factorial", HYBRID[0]), s)
    mean = {"aggregate": True, "groupby_aggfunc": "mean", "groupby_rangefunc": "minmax"}
    panels = [
        bars([OBJECTIVE], "final objective, mean over seeds", **mean),
        bars(["repo_bpb_8k"], "final repo_bpb past 8k", **mean),
        bars(EVALS, "final code, repo, web bpb", **mean),
        curve(OBJECTIVE, "objective", **mean),
    ]
    group = ["Group", "config.axes.residual"]
    runs = runset(e, p, "factorial d512", kept_at(s, "d512", arms), groupby=group)
    note = "Every arm at d512, the mean of its seeds; milestones is the hybrid's arm."
    return [*blocks, wr.P(text=note), grid(runs, panels)]


def frontier(e: str, p: str, s: Study, n: dict[str, str]) -> list:
    """The sliding-window frontier against the KDA hybrid: the frontier, its arms,
    the hybrid and the candidates at each rung, the mean of their seeds."""
    blocks: list = [
        wr.H1(text="Sliding windows against the hybrid"),
        text(n, "swa", "SWA."),
    ]
    arms = families((*FRONTIER, *SWA, HYBRID[0], *CANDIDATES), s)
    mean = {"aggregate": True, "groupby_aggfunc": "mean", "groupby_rangefunc": "minmax"}
    group = ["Group", "config.axes.model", "config.axes.residual"]
    rungs = s.files["swa"].rungs if "swa" in s.files else list(RUNGS)[:2]
    for rung in rungs:
        panels = [
            curve(OBJECTIVE, "objective", **mean),
            curve("repo_bpb", "repo_bpb, rows of 16k", **mean),
            curve("repo_bpb_8k", "repo_bpb past 8k", **mean),
            bars([OBJECTIVE, "repo_bpb_8k"], "final objective and past 8k", **mean),
            bars(["tokens_per_s", "mfu"], "tokens/s and MFU", **mean),
        ]
        runs = runset(e, p, f"windows {rung}", kept_at(s, rung, arms), groupby=group)
        blocks += [wr.H3(text=rung), grid(runs, panels)]
    return blocks


def claims(e: str, p: str, s: Study, n: dict[str, str]) -> list:
    """The stability section: the secondary metrics' notes, and a grid for each of
    their findings, at d512 on seed 0 but where a rung is named."""
    blocks: list = [wr.H1(text="Stability"), text(n, "stability", "stability.")]
    blocks.append(text(n, "secondary: summary", "the secondary metrics."))
    others = sorted(s.seeds - {"0"})
    seed0 = " and ".join(f"Config('axes.seed') != '{x}'" for x in others)
    for title, notes_, grids in _findings(s, seed0):
        blocks += [wr.H2(text=title), *(text(n, k, title) for k in notes_ if k in n)]
        for subtitle, rules, panels in grids:
            runs = by_seed(e, p, subtitle, f"{shown(s)} and {rules}")
            blocks += [wr.H3(text=subtitle), grid(runs, panels, 2)]
    return blocks


def _findings(s: Study, seed0: str) -> list[tuple[str, tuple[str, ...], list]]:
    """Each stability finding: its title, its notes, and its grids, each a title, the
    filter of its runs and its panels."""

    def at(lr: str) -> str:
        return f"Config('axes.optimizer.lr') = '{lr}'"

    def rung(r: str) -> str:
        return f"Config('axes.{RUNG}') = '{r}'"

    # An OR of ANDs: W&B's filters take one level of parentheses.
    ends = [(name, HIGH_LR) for name in SURVIVORS] + [(REFERENCE, FIXED_LR)]
    survive = " or ".join(f"({family(name, s)} and {at(lr)})" for name, lr in ends)
    survive = f"({survive})"
    logits = f"{families(LOGITS, s)} and ({at('2e-3')} or {at('4e-3')})"
    early = {"range_x": (0, 3000)}
    switch = s.phases.get("d512", (None, None))[1]
    near = {"range_x": (0.85 * switch, switch)} if switch else {}
    candidates = families((*CANDIDATES, HYBRID[0], HYBRID[1]), s)
    return [
        (
            "8e-3: who survives",
            ("secondary: stability",),
            [
                (
                    f"d512 at {HIGH_LR}, the reference at {FIXED_LR}",
                    f"{rung('d512')} and {survive}",
                    [
                        curve("loss", "8e-3: who survives, loss"),
                        curve("grad_norm", "8e-3: who survives, grad_norm"),
                    ],
                )
            ],
        ),
        (
            "Attention logit growth",
            (),
            [
                (
                    "the reference and QK-norm at 2e-3 and 4e-3, d512",
                    f"{rung('d512')} and {logits} and {seed0}",
                    [
                        layers_at(
                            "attn/logit_max/.*", "attn/logit_max, every layer", early
                        ),
                        layers_at(
                            "attn/entropy/.*", "attn/entropy, every layer", early
                        ),
                    ],
                )
            ],
        ),
        (
            "Massive activations",
            ("secondary: fp8",),
            [
                (
                    "the stream's largest entry and its rms, d512",
                    kept_at(s, "d512", families(CREST, s), seed0, show=False),
                    [
                        layers("act/max_abs/.*", "act/max_abs, every branch"),
                        layers("act/rms/.*", "act/rms, every branch"),
                    ],
                )
            ],
        ),
        (
            "The candidates against the hybrid",
            ("secondary: candidates", "secondary: moe"),
            [
                (
                    f"candidates and the hybrid, {r}, seed 0",
                    kept_at(s, r, candidates, seed0, show=False),
                    [
                        curve("grad_norm", "grad_norm: hc clips more"),
                        layers("act/rms/.*", "act/rms of the stream, every branch"),
                        layers("router/max_load/.*", "router/max_load, every MoE"),
                        layers("router/dead/.*", "router/dead, every MoE"),
                    ],
                )
                for r in ("d512", "d1024")
            ],
        ),
        (
            "The 16k switch",
            ("secondary: switch",),
            [
                (
                    "rope, rope without YaRN, and CSA, d512, around 90% of training",
                    kept_at(s, "d512", families(SWITCH, s), show=False),
                    [
                        curve("loss", "loss at the switch", **near),
                        curve("repo_bpb_8k", "repo_bpb past 8k"),
                    ],
                )
            ],
        ),
    ]


def layers_at(pattern: str, title: str, kw: dict) -> wr.LinePlot:
    """`layers`, over a range of steps."""
    panel = layers(pattern, title)
    panel.range_x = kw.get("range_x", (None, None))
    return panel


def report(e: str, p: str, s: Study, n: dict[str, str], api: wandb.Api) -> wr.Report:
    """The report: the headline, the leaderboard, each axis file, long context,
    stability, speed and the results; an existing report of the same title is
    updated."""
    every = f"{shown(s)} and Metric('Group') != '{ANALYSIS}'"
    by_file = ["Group", f"config.axes.{RUNG}"]
    leader = runset(e, p, "best of each file and rung", every, groupby=by_file)
    analysis = f"Metric('ID') = '{s.analysis}'" if s.analysis else "State = 'none'"
    blocks = [
        wr.TableOfContents(),
        text(
            n, "intro", "the question, the baseline, the recipe and how to read this."
        ),
        text(n, "headline", "the headline."),
        *best(e, p, s, n),
        *factorial(e, p, s, n),
        *frontier(e, p, s, n),
        *claims(e, p, s, n),
        wr.H1(text="Leaderboard"),
        text(n, "winners", "the winners at equal serving cost."),
        wr.P(text="Each axis file at each rung: its best run's objective."),
        grid(leader, [bars([OBJECTIVE], "best objective", **LEAST)], columns=1),
        wr.P(text="Every run, best objective first."),
        grid(
            runset(e, p, "every run", every),
            [bars([OBJECTIVE], "objective"), bars(EVALS, "evals"), coordinates()],
        ),
        *per_file(e, p, s, n),
        wr.H1(text="Long context"),
        wr.P(
            text="repo_bpb reads rows of 16k from the start; the last tenth trains on them."
        ),
        grid(
            runset(
                e, p, "every run at d768", f"{every} and Config('axes.{RUNG}') = 'd768'"
            ),
            [
                bars(POSITIONS, "repo_bpb by position"),
                scatter(
                    wr.SummaryMetric("repo_bpb_0"),
                    "repo_bpb_8k",
                    "after 8k against the first 2k",
                    log_x=False,
                ),
            ],
        ),
        text(n, "long", "long context."),
        *unstable(e, p, s),
        wr.H1(text="Throughput and cost"),
        grid(runset(e, p, "every run", every), speed()),
        wr.H1(text="Results"),
        text(n, "eras", "each era's milestone under both regimes."),
        text(n, "results", "each change alone."),
        text(n, "axes", "each axis's best change."),
        wr.H2(text="The analysis's tables"),
        wr.P(
            text="From the newest analysis run: "
            + "; ".join(f"{k}: {v}" for k, v in TABLES.items() if k in s.tables)
            + "."
        ),
        grid(runset(e, p, ANALYSIS, analysis), results(s), columns=1),
        wr.H2(text="Caveats"),
        text(n, "caveats", "the caveats."),
    ]
    made = wr.Report(entity=e, project=p, title=TITLE, blocks=blocks, width="fluid")
    made.description = "Architecture ablations from a 2020 baseline, one at a time."
    for old in api.reports(f"{e}/{p}"):
        if old.display_name == TITLE:
            made.id = old.id
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="Make or update the W&B dashboard.")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--entity", help="default: the API key's")
    parser.add_argument("--notes", type=Path, help="the report's text, ## sections")
    parser.add_argument("--keep", type=Path, help="JSONL of the runs a name keeps")
    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    entity = args.entity or api.default_entity or parser.error("pass --entity")
    found = study(api, f"{entity}/{args.project}", kept(args.keep))
    for name, f in found.files.items():
        print(
            f"{name}: variants on axes.{f.variant}, rungs on axes.{f.rung}: {f.rungs}"
        )
    print(
        f"hidden: {len(found.hidden)} older runs of a name; analysis: {found.analysis}"
    )
    view = workspace(entity, args.project, found).save()
    made = report(entity, args.project, found, notes(args.notes), api).save()
    print(f"workspace: {view.url}")
    print(f"report: {made.url}")


if __name__ == "__main__":
    main()
