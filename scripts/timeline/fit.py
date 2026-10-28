"""Fit the timeline's ablations and extrapolate every family to a target of equal
agentic serving cost.

It reads runs (run directories, or files of RESULT lines: fit_inputs.py) and keeps
one point per family, rung and token budget, at its best learning rate, its seeds
averaged. The parameters come from compiling each family's arm at its rung
(fit_counts.py, cached). The root's law L(N, D) = E + A N^-alpha + B D^-beta (the
baseline's, or --root's) fits its ladder and its token scan; every other family is
fit relative to its parent, read from its file (fit_family.py), and says its
effective-parameter multiplier at each rung (fit_law.py). Each target, from serving.py's
matches, is predicted under unlimited data (by training GPU-hours on B300s) and
limited data (130B, 1T and 10T tokens), with the leave-one-rung-out range and a
noise bootstrap (fit_analysis.py). Tables, a JSON of everything and plots go to
--out; --wandb logs the tables to W&B.

    python scripts/timeline/fit.py results.txt --targets serving-table.json --out fit
    python scripts/timeline/fit.py results.txt --match B1b --mfu mfu.json --wandb
    python scripts/timeline/fit.py results.txt --match B1b --parents p.json --share B,beta
    python scripts/timeline/fit.py results.jsonl --root positions/rope --drop-scan baseline \\
        --fix-e objective=0.3204,code_bpb=0.2808,repo_bpb=0.2448,web_bpb=0.3832
    python scripts/timeline/fit.py --selftest --out /tmp/fit-selftest
"""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

import fit_analysis
import fit_family
import fit_inputs
import fit_plots
import fit_points
import fit_report
import fit_targets
from fit_counts import FOLDER, Counts, find, notes

HERE = Path(__file__).resolve().parent
MIN_DOF = 2  # replicates' degrees of freedom to trust their noise over --noise


ARGV = sys.argv[1:]  # the fit's arguments, as the report repeats them


def command() -> str:
    """How this fit was run."""
    return shlex.join(["python", "scripts/timeline/fit.py", *ARGV])


def main() -> None:
    a = parser().parse_args()
    if a.selftest:
        import fit_selftest

        sys.exit(fit_selftest.run(a.out, a.reps))
    if not a.results:
        sys.exit("give a file of RESULT lines, or --selftest")
    report = run(a)
    print(report["markdown"])


def run(a: argparse.Namespace) -> dict:
    """Read, fit and write; return what was written."""
    runs, short = fit_points.cut_short(fit_inputs.read(a.results))
    counts = Counts(a.cache or a.out / "counts.json", tuple(a.files))
    parents = family_parents(a, runs, counts)
    if not a.keep_base:
        runs = fit_family.alias_bases(runs, counts, parents, a.root)
    points, diverged = fit_points.monotone(
        fit_points.points(runs, a.pick, not a.no_average_seeds)
    )
    points = attach(points, counts, a.rung_flops)
    if not any(p.family == a.root for p in points):
        sys.exit(
            f"no points of the root family {a.root!r}; families: {sorted({p.family for p in points})}"
        )
    targets = build_targets(a, points, counts, parents)
    metrics = (
        a.metrics.split(",") if a.metrics else fit_points.metrics_of(points, a.root)
    )
    noise = {m: fit_points.noise(runs, m) for m in metrics}
    analyses = [
        fit_analysis.analyze(points, m, parents, targets, options(a, noise[m], m))
        for m in metrics
    ]
    tables = tables_of(analyses, points, metrics, targets, a)
    tables["lr"] = tables["learning_rates"] = fit_points.lr_vertices(runs)
    tables["long_context"] = fit_report.long_context(analyses)
    tables["long_summary"] = fit_report.long_summary(tables["long_context"])
    tables["unlimited_summary"] = fit_report.unlimited_summary(analyses, targets)
    tables["limited_summary"] = fit_report.limited_summary(analyses)
    facts = {
        "noise": noise,
        "diverged": diverged,
        "points": points,
        "analyses": analyses,
        "short": short,
    }
    out = write(a, analyses, targets, tables, facts)
    if a.wandb:
        log_wandb(tables, out["plots"], a.project, a.out)
    return out


def family_parents(a: argparse.Namespace, runs: list, counts: Counts) -> dict[str, str]:
    """Each family's parent: read from its file's modules (by its kind where the
    file uses two), --parents over them."""
    first = {}
    for r in sorted(runs, key=lambda r: r.width):
        first.setdefault(r.family, r)

    def kind_of(f: str) -> str | None:
        r = first.get(f)
        return fit_family.kind(counts.at(r.stem, r.arms, r.width)) if r else None

    found = fit_family.infer_parents(set(first), tuple(a.files), a.root, kind_of)
    return found | (json.loads(a.parents.read_text()) if a.parents else {})


def options(
    a: argparse.Namespace, noise: tuple | None, metric: str
) -> fit_analysis.Options:
    """The fit's options for a metric: its bootstrap noise the replicates', if they
    have `MIN_DOF` degrees of freedom, else --noise."""
    measured = noise is not None and noise[1] >= MIN_DOF
    sigma, source = (
        (noise[0], f"replicates, {noise[1]} dof") if measured else (a.noise, "--noise")
    )
    share = tuple(a.share.split(",")) if a.share else ()
    e = (
        dict(kv.split("=") for kv in a.fix_e.split(",")).get(metric)
        if a.fix_e
        else None
    )
    drop = tuple(a.drop_scan.split(",")) if a.drop_scan else ()
    return fit_analysis.Options(
        a.root, not a.no_joint, sigma, a.reps, a.seed, share, source,
        None if e is None else float(e), drop,
    )  # fmt: skip


def attach(
    points: list[fit_points.Point], counts: Counts, flops: bool = False
) -> list[fit_points.Point]:
    """`points` with their parameter counts (and FLOPs per token, if `flops`);
    those whose model does not compile dropped, with a warning."""
    kept = []
    for p in points:
        p.counts = counts.at(p.stem, p.arms, p.width)
        if flops and "error" not in p.counts:
            p.counts = {**p.counts, **counts.rung_flops(p.stem, p.arms, p.width)}
        if "error" in p.counts:
            print(
                f"warning: {p.family} d{p.width}: {p.counts['error']}", file=sys.stderr
            )
        else:
            kept.append(p)
    return kept


def build_targets(
    a: argparse.Namespace, points: list, counts: Counts, parents: dict
) -> dict:
    """Each family's target: from --targets, or serving.py's match at --match; a
    family with no target of its own takes that of one with its architecture."""
    raw = fit_targets.load(a.targets, a.budget) if a.targets else {}
    first: dict[str, fit_points.Point] = {}
    for p in sorted(points, key=lambda p: p.width):
        first.setdefault(p.family, p)
    if a.match:
        sys.path.insert(0, str(HERE))
        for f, p in first.items():
            file, run = find(p.stem, tuple(a.files)), counts.sized(p.stem, p.arms, 1024)
            if f not in raw and file and run:
                raw[f] = fit_targets.match(file, run.arch, a.match, counts.table)
                counts.save()
    if a.targets:
        sys.path.insert(0, str(HERE))
        for f, p in first.items():
            if f not in raw and (file := find(p.stem, tuple(a.files))):
                arch = lambda p=p: served_arch(counts, p)
                raw[f] = fit_targets.by_architecture(raw, file, arch) or {}
    raw = {f: r for f, r in raw.items() if r.get("n_active")}
    same = {(first[f].width, first[f].counts["arch"]): f for f in raw if f in first}
    for f, p in first.items():
        if f not in raw and (g := same.get((p.width, p.counts["arch"]))):
            raw[f] = raw[g]
    measured = fit_targets.mfus(a.mfu)
    kinds = {f: fit_family.kind(p.counts) for f, p in first.items()}
    out = {}
    for f, r in raw.items():
        if f in first:
            entry, source = fit_family.mfu_source(f, measured, parents, kinds)
            out[f] = make_target(r, first[f], counts, entry, source)
    return out


def served_arch(counts: Counts, p: fit_points.Point) -> Any:
    """`p`'s architecture at ``d_model = 1024``, by which serving.py's arm is found.

    Raises:
        LookupError: If its file has no such arm.
    """
    run = counts.sized(p.stem, p.arms, 1024)
    if run is None:
        raise LookupError(f"{p.family}: no arm at d_model = 1024")
    return run.arch


def make_target(
    raw: dict, p: fit_points.Point, counts: Counts, measured: dict | None, source: str
) -> fit_targets.Target:
    """A target, its FLOPs per token from the compiler where its row has none: its
    FLOPs per active parameter at the nearest valid width, times its own."""
    flops = raw.get("flops")
    if not flops and raw.get("d_model"):
        c = counts.flops(p.stem, p.arms, raw["d_model"])
        flops = c["flops"] / c["active"] * raw["n_active"] if "flops" in c else math.nan
    sparse = fit_family.kind(p.counts) != "dense"
    return fit_targets.target(raw, flops or math.nan, sparse, measured, source)


def tables_of(
    analyses: list,
    points: list,
    metrics: list[str],
    targets: dict,
    a: argparse.Namespace,
) -> dict:
    """The tables, as rows: the leaderboard of each metric, and the rest."""
    first: dict[str, fit_points.Point] = {}
    for p in points:
        first.setdefault(p.family, p)
    info = {f: notes(p.stem, p.arms, tuple(a.files)) for f, p in first.items()}
    out = {
        f"leaderboard_{x.metric}": fit_report.leaderboard(x, info, a.root)
        for x in analyses
    }
    out["leaderboard"] = out.get("leaderboard_objective", [])
    out["multipliers"] = fit_report.multipliers(analyses)
    out["extrapolation_unlimited"] = fit_report.unlimited(analyses, targets)
    out["extrapolation_limited"] = fit_report.limited(analyses)
    out["laws"] = fit_report.laws(analyses)
    out["points"] = fit_report.points_table(points, metrics)
    return out


def write(
    a: argparse.Namespace, analyses: list, targets: dict, tables: dict, facts: dict
) -> dict:
    """Write report.md, fit.json and the plots to --out."""
    a.out.mkdir(parents=True, exist_ok=True)
    plots = []
    first = next((x for x in analyses if x.laws), None)
    wanted = a.plot.split(",") if a.plot else None
    # one family per colour slot in every plot: chosen once, on the first metric
    fams = fit_plots.chosen(first, targets, a.root, wanted) if first else []
    for x in analyses:
        if x.metric in a.plot_metrics.split(",") and x.laws:
            plots += fit_plots.ladder(x, fams, targets, a.out)
            if a.root in targets:
                plots += fit_plots.hours(x, fams, targets, a.root, a.out)
            plots += fit_plots.bars(x, a.root, a.out)
    text = markdown(a, tables, targets, facts)
    (a.out / "report.md").write_text(text)
    everything = {
        "assumptions": assumptions(a, targets, facts),
        "targets": {f: fit_targets.as_dict(t) for f, t in targets.items()},
        "metrics": {x.metric: as_json(x) for x in analyses},
        "tables": tables,
    }
    (a.out / "fit.json").write_text(json.dumps(everything, indent=1, default=_json))
    return {"markdown": text, "plots": plots, "json": a.out / "fit.json"}


def as_json(x: fit_analysis.Analysis) -> dict:
    def paths(d: dict) -> dict:
        return {
            f: {"/".join(map(str, k)): v for k, v in r.items()} for f, r in d.items()
        }

    return {
        "parents": {f: fam.parent for f, fam in x.fams.items()},
        "own_terms": {f: fam.free for f, fam in x.fams.items()},
        "quantities": x.q,
        "leave_one_rung_out": paths(x.loro),
        "bootstrap_5_95": paths(x.boot),
    }


def _json(v: Any) -> Any:
    """numpy's numbers as Python's; anything else as its text."""
    return v.item() if hasattr(v, "item") else str(v)


def assumptions(a: argparse.Namespace, targets: dict, facts: dict) -> list[str]:
    """What the numbers rest on, said beside them: `facts` has each metric's
    replicate noise and analysis, the points dropped as diverged and those kept."""
    points = facts["points"]
    fit = (
        "all families fit at once"
        if not a.no_joint
        else "the root fit alone, then each family with its parent held"
    )
    out = [
        f"N is the parameters a token goes through besides the embeddings; points at the best of their learning rates ({a.pick}), "
        + ("its lr chosen on its first runs, then every run at that lr averaged (the seeds)." if not a.no_average_seeds else "each of its first runs only: seeds and base arms only measure the noise."),
        f"E is shared by every family; {fit}" + (f"; {a.share} pooled: only the root has them of its own." if a.share else "."),
        noise_line(facts["analyses"], a.reps),
        f"Targets: {a.targets or ('serving.py match at ' + a.match if a.match else 'none')} (budget {a.budget}).",
        (f"GPU-hours = tokens x FLOPs/token / (MFU x {fit_targets.PEAK:.3g} x 3600); MFU measured, else a measured kin's "
         "(an MoE or hybrid of its file), else its nearest measured ancestor's (the (a) table's 'MFU from')."),
    ]  # fmt: skip
    if a.fix_e:
        out.append(f"E held (not fit) for {a.fix_e}; the other metrics fit it.")
    if a.drop_scan:
        out.append(
            f"Token scans left out for {a.drop_scan}: they take their parent's data term."
        )
    if short := facts.get("short"):
        names = ", ".join(
            sorted(
                f"{r.family} d{r.width} t{r.per_param} ({r.tokens / 1e9:.3g}B tokens)"
                for r in short
            )
        )
        out.append(
            f"Left out as cut short (under {fit_points.SHORT:.0%} of their rung's tokens): {names}."
        )
    if gone := facts["diverged"]:
        names = ", ".join(sorted(f"{p.family} d{p.width} t{p.per_param}" for p in gone))
        out.append(
            f"Dropped as diverged (worse than a smaller rung or fewer tokens): {names}."
        )
    if edges := sorted({f"{p.family} d{p.width}" for p in points if p.edge}):
        out.append(
            f"Best lr at the edge of its scan (scan further): {', '.join(edges)}."
        )
    if placeholder := sorted(
        f for f, t in targets.items() if t.mfu_from == "placeholder"
    ):
        out.append(
            f"PLACEHOLDER MFU ({fit_targets.MFU_DENSE} dense, {fit_targets.MFU_SPARSE} MoE or hybrid), "
            f"no measured family above it, for: {', '.join(placeholder)}."
        )
    return out


def noise_line(analyses: list, reps: int) -> str:
    """Each metric's noise per run, where it comes from, and the verdict's rule."""
    each = ", ".join(f"{x.metric} {x.sigma:.2%} ({x.sigma_from})" for x in analyses)
    return (
        f"Noise per run: {each}. The bootstrap refits {reps} times with each point off by it "
        f"over the root of the runs it averages; 'no measurable difference' means "
        f"|delta| < {fit_analysis.VERDICT} sigmas of the delta at every rung, "
        f"sigma x sqrt(1/n + 1/n_parent) for points of n and n_parent runs."
    )


def markdown(a: argparse.Namespace, tables: dict, targets: dict, facts: dict) -> str:
    parts = [
        "# Timeline fit\n",
        f"Command, from the repository's root: `{command()}`\n",
        *(f"- {s}" for s in assumptions(a, targets, facts)),
        "",
    ]
    for key, title in (
        ("leaderboard_objective", "One-change leaderboard (objective)"),
        ("leaderboard_repo_bpb", "Leaderboard (repositories, 16k)"),
    ):
        if key in tables:
            parts += [
                f"## {title}\n",
                fit_report.markdown(tables[key], fit_report.LEADERBOARD),
            ]
    parts += [
        "## (a) Unlimited data: GPU-hours to the reference target's 1T-token loss (vs: the parent's, or the root's; every metric in fit.json)\n",
        fit_report.markdown(tables["unlimited_summary"], fit_report.UNLIMITED_SUMMARY),
    ]
    parts += [
        "## Learning rates (vertex of a parabola in log lr, trials within 3% of the best)\n",
        fit_report.markdown(tables["learning_rates"], fit_report.LEARNING_RATES),
    ]
    parts += [
        "## Long context: repositories after 8k tokens against their first tokens (means over the rungs)\n",
        fit_report.markdown(tables["long_summary"], fit_report.LONG_SUMMARY),
    ]
    parts += [
        "## (b) Limited data: loss on the same tokens (objective; every metric in fit.json)\n",
        fit_report.markdown(tables["limited_summary"], fit_report.LIMITED_SUMMARY),
    ]
    for key, title in (
        ("multipliers", "Effective-parameter multipliers at each rung"),
        ("laws", "Laws"),
        ("points", "Points"),
    ):
        parts += [f"## {title}\n", fit_report.markdown(tables[key])]
    return "\n".join(parts)


def log_wandb(tables: dict, plots: list[Path], project: str, out: Path) -> None:
    """Log the tables as W&B Tables, and the plots, in a run named ``analysis``
    whose local files go under `out`."""
    import wandb

    run = wandb.init(
        project=project,
        name="analysis",
        group="analysis",
        job_type="analysis",
        dir=str(out),
    )
    for name in (
        "leaderboard",
        "extrapolation_unlimited",
        "extrapolation_limited",
        "multipliers",
        "laws",
        "points",
        "lr",
        "long_context",
    ):
        rows = tables.get(name, [])
        columns = list(dict.fromkeys(k for r in rows for k in r))
        data = [[_cell(r.get(c)) for c in columns] for r in rows]
        run.log({name: wandb.Table(columns=columns, data=data)})
    for path in plots:
        if "-dark" not in path.stem:
            run.log({f"plots/{path.stem}": wandb.Image(str(path))})
    run.finish()


def _cell(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v if isinstance(v, int | float | str | bool) or v is None else str(v)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "results",
        nargs="*",
        type=Path,
        help="run directories, folders of them, or files of RESULT lines",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("timeline-fit"),
        help="where the tables, JSON and plots go",
    )
    p.add_argument(
        "--cache", type=Path, help="the counts' JSON cache (default: OUT/counts.json)"
    )
    p.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=[FOLDER],
        help="folders of the families' .bux files",
    )
    p.add_argument(
        "--root",
        default=fit_points.ROOT,
        help="the family every other one descends from",
    )
    p.add_argument(
        "--parents", type=Path, help='JSON {"family": "parent"}; default: the root'
    )
    p.add_argument(
        "--keep-base",
        action="store_true",
        help="keep axis files' base arms apart from the root",
    )
    p.add_argument(
        "--pick", choices=("min", "vertex"), default="min", help="a point's best lr"
    )
    p.add_argument(
        "--no-average-seeds",
        action="store_true",
        help="a point is its first runs only; its seeds only measure the noise",
    )
    p.add_argument(
        "--metrics",
        help="comma-separated; default: the four and repo_bpb_* the root has",
    )
    p.add_argument(
        "--targets", type=Path, help="targets JSON, or serving.py's serving-table.json"
    )
    p.add_argument(
        "--budget", default="B1b", help="the serving budget of a serving-table.json"
    )
    p.add_argument(
        "--match", help="match targets with serving.py at this budget (B1a, B1b, B2b)"
    )
    p.add_argument(
        "--mfu",
        type=Path,
        help="JSON of measured training MFU or tok/s/GPU at target scale",
    )
    p.add_argument(
        "--rung-flops",
        action="store_true",
        help="count each rung's FLOPs per token too",
    )
    p.add_argument(
        "--no-joint",
        action="store_true",
        help="fit the root alone, then each family with its parent held",
    )
    p.add_argument(
        "--share",
        help="terms only the root has of its own, as alpha,B,beta: pooled over every family",
    )
    p.add_argument(
        "--fix-e",
        help="hold E, as objective=0.3204,code_bpb=0.2808 (metrics not named: fit)",
    )
    p.add_argument(
        "--drop-scan",
        help="comma-separated families whose token scans are left out: they take their parent's data term",
    )
    p.add_argument(
        "--noise", type=float, default=0.002, help="the bootstrap's relative noise"
    )
    p.add_argument("--reps", type=int, default=100, help="bootstrap refits (0: none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", help="comma-separated families to plot beside the root")
    p.add_argument("--plot-metrics", default="objective,repo_bpb")
    p.add_argument(
        "--wandb", action="store_true", help="log the tables to W&B (env: WANDB_*)"
    )
    p.add_argument("--project", default="bux-timeline")
    p.add_argument(
        "--selftest", action="store_true", help="fit synthetic runs of known laws"
    )
    return p


if __name__ == "__main__":
    main()
