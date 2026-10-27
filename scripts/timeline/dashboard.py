"""The W&B dashboard of the timeline ablations: a saved workspace view and a report of
the project their runs log to, made or updated in place.

    uv run --with wandb-workspaces --with wandb python scripts/timeline/dashboard.py

It reads ``WANDB_API_KEY`` and ``WANDB_BASE_URL`` from the environment, and needs
none of bux. Run it again once runs exist, and as more axis files start: it builds a
section for each axis file the project has runs of (a W&B group is a file), on the
axis its variants take (the one besides the rung's and the learning rate's), and a
grid for each rung it has runs at. Before any run, it uses `FILES`.

A bux run's config has ``axes``, what it took on each axis by the axis's name
(``{"model": "rope", "d_model": "d768"}``), ``params``, ``active_params`` and its
settings under ``train`` and ``arch``. A run with an ``objective`` logs it at each
eval. The results' panels read W&B Tables that runs of the group ``analysis`` log, as
``run.log({"extrapolation": wandb.Table(dataframe=df)})``: `TABLES` names them.
"""

import argparse
import contextlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
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
NOT_VARIANTS = {"optimizer.lr", "steps"}  # axes besides a file's variants and rungs
# The axis files and the axis their variants are on, before the project has runs.
FILES = {
    **dict.fromkeys(("milestones", "positions", "attention", "norms"), "model"),
    **dict.fromkeys(("mixers", "embeddings"), "model"),
    **dict.fromkeys(("ffn", "moe"), "ffn"),
    "residuals": "residual",
    "structure": "width",
}
ANALYSIS = "analysis"  # the group of the runs that log the results' tables
TABLES = {
    "extrapolation": "Extrapolated loss at equal serving cost, unlimited data",
    "extrapolation_limited": "Extrapolated loss at equal serving cost, limited data",
    "fits": "The fits each extrapolation comes from",
}
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


@dataclass
class File:
    """An axis file's runs: the axis of its variants, the axis of its rungs, and the
    rungs it has runs at."""

    variant: str | None
    rung: str = RUNG
    rungs: list[str] = field(default_factory=list)


@dataclass
class Study:
    """What the project has runs of: its axis files, and where each rung's long phase
    starts and ends, in tokens."""

    files: dict[str, File] = field(default_factory=dict)
    phases: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)


def study(api: wandb.Api, path: str) -> Study:
    """Return what the runs at `path` are, or `FILES` before any."""
    found = Study()
    for run in api.runs(path, filters={"group": {"$ne": ANALYSIS}}):
        axes = run.config.get("axes") or {}
        rung = next((k for k, v in axes.items() if re.fullmatch(r"d\d+", v)), None)
        rest = [k for k in axes if k != rung and k not in NOT_VARIANTS]
        f = found.files.setdefault(run.group, File(next(iter(rest), None)))
        if rung is not None and axes[rung] not in f.rungs:
            f.rung = rung
            f.rungs.append(axes[rung])
            phase = _long_phase(run.config.get("train") or {})
            found.phases.setdefault(axes[rung], phase)
    if not found.files:
        found.files = {name: File(v, rungs=list(RUNGS)) for name, v in FILES.items()}
    for f in found.files.values():
        f.rungs.sort(key=lambda r: int(r[1:]))
    return found


def _long_phase(train: dict) -> tuple[float | None, float | None]:
    """Return the tokens where a run's last phase starts and where the run ends."""
    steps, rows, length = train.get("steps"), train.get("batch"), train.get("seq_len")
    phases = train.get("curriculum") or []
    if not (steps and rows and length and phases):
        return (None, None)
    at = phases[-1].get("at", 0)
    start = at * steps if 0 < at < 1 else at  # a share of the steps, or a step
    return (start * rows * length, steps * rows * length)


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


def lr_scatter(y: str = OBJECTIVE) -> wr.ScatterPlot:
    """Each run's final `y` against its learning rate."""
    return wr.ScatterPlot(
        title=f"{y} by learning rate",
        x=wr.Config(LR),
        y=wr.SummaryMetric(y),
        log_x=True,
    )


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


def size_scatter() -> wr.ScatterPlot:
    """Each run's objective against the parameters a token goes through."""
    return wr.ScatterPlot(
        title="objective by active parameters",
        x=wr.Config("active_params"),
        y=wr.SummaryMetric(OBJECTIVE),
        log_x=True,
    )


def evals() -> list[wr.LinePlot]:
    """The objective and each eval's bits per byte, by tokens."""
    return [curve(m, m) for m in (OBJECTIVE, *EVALS)]


def training() -> list[wr.LinePlot]:
    """The training loss smoothed and raw, its z-loss and the gradient's norm."""
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


def speed() -> list[wr.LinePlot | wr.BarPlot]:
    """Throughput and memory, by run and over training."""
    return [
        bars(["tokens_per_s"], "tokens/s"),
        bars(["mfu"], "MFU"),
        curve("tokens_per_s", "tokens/s over training"),
        curve("time/step_ms", "step time, ms"),
        bars(["mem/peak_gib"], "peak memory per GPU, GiB"),
        curve(["time/data_ms", "time/wait_ms"], "host: data and wait, ms per step"),
    ]


def results() -> list[wr.WeavePanelSummaryTable | wr.MarkdownPanel]:
    """The analysis's tables, and a note on where they come from."""
    note = wr.MarkdownPanel(
        markdown=f"Tables that the runs of group `{ANALYSIS}` log: "
        + ", ".join(f"`{t}`" for t in TABLES)
    )
    return [wr.WeavePanelSummaryTable(table_name=t) for t in TABLES] + [note]


# --- The workspace ----------------------------------------------------------------------


def section(name: str, panels: list, *, is_open: bool = False) -> ws.Section:
    """A workspace section of `panels`, three a row, all on one page."""
    layout = ws.SectionLayoutSettings(columns=3, rows=max(2, -(-len(panels) // 3)))
    return ws.Section(name=name, panels=panels, is_open=is_open, layout_settings=layout)


def workspace(entity: str, project: str) -> ws.Workspace:
    """The saved view: every run, best objective first, with the panels by topic."""
    leaderboard = [
        bars([OBJECTIVE], "objective, best first"),
        bars(EVALS, "code, repo, web bpb"),
        size_scatter(),
        coordinates(),
        wr.RunComparer(diff_only="split"),
    ]
    long = [
        bars(POSITIONS, "repo_bpb by position"),
        curve(list(POSITIONS), "by position"),
    ]
    view = ws.Workspace(
        entity=entity,
        project=project,
        name="timeline",
        sections=[
            section("Leaderboard", leaderboard, is_open=True),
            section("Evals", evals(), is_open=True),
            section("Long context", long),
            section("Training", training()),
            section("Stability", stability()),
            section(
                "LR sensitivity", [lr_scatter(), lr_scatter("repo_bpb"), coordinates()]
            ),
            section("Throughput", speed()),
            section("Results", results()),
        ],
        settings=ws.WorkspaceSettings(x_axis=X, max_runs=100),
        runset_settings=ws.RunsetSettings(order=BEST, pinned_columns=list(COLUMNS)),
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


def grid(runs: wr.Runset, panels: list) -> wr.PanelGrid:
    """A grid of `panels`, over `runs`, three a row."""
    for i, panel in enumerate(panels):
        panel.layout = wr.Layout(x=8 * (i % 3), y=6 * (i // 3), w=8, h=6)
    return wr.PanelGrid(runsets=[runs], panels=panels)


def of_file(file: str, f: File | None = None, rung: str | None = None) -> str:
    """The filter of an axis file's runs, at `rung` of `f`'s rungs if given."""
    kept = f"Metric('Group') = '{file}'"
    if f is None or rung is None:
        return kept
    return kept + f" and Config('axes.{f.rung}') = '{rung}'"


def ablation(e: str, p: str, file: str, rung: str | None, s: Study) -> wr.PanelGrid:
    """The grid of one axis file at one rung: a line per variant, the least over its
    learning rates at each point, shaded to their range, and its long context."""
    f = s.files[file]
    group = [f"config.axes.{f.variant}"] if f.variant else []
    columns = [f"config:axes.value.{f.variant}", *COLUMNS] if f.variant else COLUMNS
    name = f"{file} {rung or ''}".strip()
    runs = runset(
        e, p, name, of_file(file, f, rung), groupby=group, pinned_columns=list(columns)
    )
    best = {"groupby_aggfunc": "min", "groupby_rangefunc": "minmax", "aggregate": True}
    least = {"groupby_aggfunc": "min", "aggregate": True}
    start, end = s.phases.get(rung or "", (None, None))
    return grid(
        runs,
        [
            curve(OBJECTIVE, "objective", **best),
            curve("repo_bpb", "repo_bpb, rows of 16k", **best),
            bars([OBJECTIVE, *EVALS], "final, best lr", **least),
            bars(POSITIONS, "repo_bpb by position", **least),
            curve(
                "repo_bpb", "repo_bpb in the 16k phase", range_x=(start, end), **best
            ),
            lr_scatter(),
            bars(["tokens_per_s", "mfu"], "tokens/s and MFU", aggregate=True),
        ],
    )


def per_file(e: str, p: str, s: Study) -> list:
    """A collapsed heading per axis file, with a grid per rung."""
    blocks: list = [wr.H1(text="Per axis")]
    for file, f in s.files.items():
        note = f"A line per variant (config axes.{f.variant}): the least over its lrs."
        inner: list = [wr.P(text=note)]
        for rung in f.rungs or [None]:
            inner += [wr.H3(text=rung or "every rung"), ablation(e, p, file, rung, s)]
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
        runs = runset(e, p, file, of_file(file))
        blocks.append(wr.H2(text=file, collapsed_blocks=[grid(runs, panels)]))
    return blocks


def placeholder(text: str) -> wr.CalloutBlock:
    """A block for the coordinator to fill."""
    return wr.CalloutBlock(text=f"To fill: {text}")


def report(e: str, p: str, s: Study, api: wandb.Api) -> wr.Report:
    """The report: the leaderboard, each axis file, long context, stability, speed and
    the results; an existing report of the same title is updated."""
    every = f"Metric('Group') != '{ANALYSIS}'"
    by_file = ["Group", f"config.axes.{RUNG}"]
    leader = runset(e, p, "best of each file and rung", every, groupby=by_file)
    best = {"groupby_aggfunc": "min", "aggregate": True}
    blocks = [
        wr.TableOfContents(),
        placeholder("the question, the baseline, the recipe and how to read this."),
        wr.H1(text="Leaderboard"),
        wr.P(text="Each axis file at each rung: its best run's objective."),
        grid(leader, [bars([OBJECTIVE], "best objective", **best)]),
        wr.P(text="Every run, best objective first."),
        grid(
            runset(e, p, "every run", every),
            [bars([OBJECTIVE], "objective"), bars(EVALS, "evals"), coordinates()],
        ),
        *per_file(e, p, s),
        wr.H1(text="Long context"),
        wr.P(
            text="repo_bpb reads rows of 16k from the start; the last tenth trains on them."
        ),
        grid(
            runset(e, p, "every run", every),
            [
                bars(POSITIONS, "repo_bpb by position"),
                curve(list(POSITIONS), "by position"),
            ],
        ),
        *unstable(e, p, s),
        wr.H1(text="Throughput and cost"),
        grid(runset(e, p, "every run", every), speed()),
        wr.H1(text="Results"),
        placeholder("extrapolated loss at equal serving cost, unlimited data."),
        placeholder("the same with limited data: every architecture the same tokens."),
        grid(runset(e, p, ANALYSIS, f"Metric('Group') = '{ANALYSIS}'"), results()),
        placeholder("what changed the most, what did not matter, what to train next."),
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
    args = parser.parse_args()
    api = wandb.Api()
    entity = args.entity or api.default_entity
    found = study(api, f"{entity}/{args.project}")
    for name, f in found.files.items():
        print(
            f"{name}: variants on axes.{f.variant}, rungs on axes.{f.rung}: {f.rungs}"
        )
    view = workspace(entity, args.project).save()
    made = report(entity, args.project, found, api).save()
    print(f"workspace: {view.url}")
    print(f"report: {made.url}")


if __name__ == "__main__":
    main()
