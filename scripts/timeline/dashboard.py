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
``--notes`` fills the report's text from a Markdown file of ``## name`` sections:
``intro``, ``headline``, ``winners``, ``eras``, ``axes``, ``caveats``, and
``axis: <file>`` for each file's.
"""

import argparse
import contextlib
import json
import re
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
    "milestones": "model",
    "stacks": "model",
    "baseline": None,
}
REFERENCE = ("positions", "rope")  # the parent of an axis file's variants: file, label
# Each file's parent family, (file, label), where it is not REFERENCE.
PARENTS: dict[str, tuple[str, str | None] | None] = {
    "positions": ("baseline", None),
    "milestones": ("baseline", None),
    "stacks": ("milestones", "m2026_hybrid"),
    "baseline": None,
}
ANALYSIS = "analysis"  # the group of the runs that log the results' tables
TABLES = {
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
    starts and ends in tokens, the runs the views hide, the token scans' labels, and
    the newest analysis run."""

    files: dict[str, File] = field(default_factory=dict)
    phases: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    hidden: set[str] = field(default_factory=set)
    scans: set[str] = field(default_factory=set)
    analysis: str | None = None


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
        found.analysis = max(analyses, key=lambda r: r.created_at).id
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


def _add(found: Study, run: Any) -> None:
    """Add what `run` says of its file: its variants' axis, its rung, its scan."""
    axes = run.config.get("axes") or {}
    rung = next((k for k, v in axes.items() if re.fullmatch(r"d\d+", v)), None)
    rest = [k for k in axes if k != rung and k not in NOT_VARIANTS]
    f = found.files.setdefault(run.group, File(next(iter(rest), None)))
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


def group_is(file: str, s: Study) -> str:
    """The filter of an axis file's runs, or its parent's variant's: `PARENTS`."""
    mine = f"Metric('Group') = '{file}'"
    parent = PARENTS.get(file, REFERENCE)
    if parent is None:
        return mine
    group, label = parent
    theirs = f"Metric('Group') = '{group}'"
    variant = s.files[group].variant if group in s.files else FILES.get(group)
    if label is not None and variant:
        theirs = f"({theirs} and Config('axes.{variant}') = '{label}')"
    return f"({mine} or {theirs})"


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


def results() -> list[wr.WeavePanelSummaryTable | wr.MediaBrowser]:
    """The analysis's tables and plots."""
    tables = [wr.WeavePanelSummaryTable(table_name=t) for t in TABLES]
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
            section("Results", results()),
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
        parent = PARENTS.get(file, REFERENCE)
        beside = "" if parent is None else f", beside {'/'.join(filter(None, parent))}"
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
    blocks: list = [wr.H1(text="Stability")]
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
        *unstable(e, p, s),
        wr.H1(text="Throughput and cost"),
        grid(runset(e, p, "every run", every), speed()),
        wr.H1(text="Results"),
        text(n, "eras", "each era's milestone under both regimes."),
        text(n, "axes", "each axis's best change."),
        wr.H2(text="The analysis's tables"),
        wr.P(
            text="From the newest analysis run: "
            + "; ".join(f"{k}: {v}" for k, v in TABLES.items())
            + "."
        ),
        grid(runset(e, p, ANALYSIS, analysis), results(), columns=1),
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
