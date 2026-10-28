"""Fit the timeline's ablations and extrapolate every family to a target of equal
agentic serving cost.

It reads RESULT lines (one per finished run, see fit_points.py) and keeps one point
per family, rung and token budget, at its best learning rate. The parameters come
from compiling each family's arm at its rung (fit_counts.py, cached). The baseline's
law L(N, D) = E + A N^-alpha + B D^-beta fits its ladder and its token scan; every
other family is fit relative to its parent (fit_law.py), and says its
effective-parameter multiplier at each rung. Each target, from serving.py's
matches, is predicted under unlimited data (by training GPU-hours on B300s) and
limited data (130B, 1T and 10T tokens), with the leave-one-rung-out range and a
noise bootstrap (fit_analysis.py). Tables, a JSON of everything and plots go to
--out; --wandb logs the tables to W&B.

    python scripts/timeline/fit.py results.txt --targets serving-table.json --out fit
    python scripts/timeline/fit.py results.txt --match B1b --mfu mfu.json --wandb
    python scripts/timeline/fit.py results.txt --match B1b --parents p.json --share B,beta
    python scripts/timeline/fit.py --selftest --out /tmp/fit-selftest
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import fit_analysis
import fit_plots
import fit_points
import fit_report
import fit_targets
from fit_counts import FOLDER, Counts, find, notes

HERE = Path(__file__).resolve().parent


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
    runs = fit_points.read(a.results)
    counts = Counts(a.cache or a.out / "counts.json", tuple(a.files))
    parents = json.loads(a.parents.read_text()) if a.parents else {}
    if not a.keep_base:
        runs = alias_bases(runs, counts, parents, a.root)
    points, diverged = fit_points.monotone(fit_points.points(runs, a.pick))
    points = attach(points, counts, a.rung_flops)
    if not any(p.family == a.root for p in points):
        sys.exit(
            f"no points of the root family {a.root!r}; families: {sorted({p.family for p in points})}"
        )
    targets = build_targets(a, points, counts)
    metrics = (
        a.metrics.split(",") if a.metrics else fit_points.metrics_of(points, a.root)
    )
    share = tuple(a.share.split(",")) if a.share else ()
    o = fit_analysis.Options(a.root, not a.no_joint, a.noise, a.reps, a.seed, share)
    analyses = [fit_analysis.analyze(points, m, parents, targets, o) for m in metrics]
    tables = tables_of(analyses, points, metrics, targets, a)
    facts = {"noise": fit_points.noise(runs), "diverged": diverged, "points": points}
    out = write(a, analyses, targets, tables, facts)
    if a.wandb:
        log_wandb(tables, out["plots"], a.project)
    return out


def alias_bases(runs: list, counts: Counts, parents: dict, root: str) -> list:
    """`runs` with each axis file's ``base`` arm as the run it repeats: the parent
    of the axis' other arms (the root if --parents names none), where the two have
    one architecture at that rung."""
    for r in runs:
        if r.arms != ("base",):
            continue
        axis = [v for f, v in parents.items() if f.startswith(f"{r.stem}/")]
        parent = max(set(axis), key=axis.count) if axis else root
        stem, _, arm = parent.partition("/")
        arms = (arm,) if arm else ()
        mine, theirs = (
            counts.at(r.stem, r.arms, r.width),
            counts.at(stem, arms, r.width),
        )
        if mine.get("arch") and mine.get("arch") == theirs.get("arch"):
            r.stem, r.arms = stem, arms
    return runs


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


def build_targets(a: argparse.Namespace, points: list, counts: Counts) -> dict:
    """Each family's target: from --targets, or serving.py's match at --match; a
    family with no target of its own takes that of one with its architecture."""
    raw = fit_targets.load(a.targets, a.budget) if a.targets else {}
    first: dict[str, fit_points.Point] = {}
    for p in sorted(points, key=lambda p: p.width):
        first.setdefault(p.family, p)
    if a.match:
        sys.path.insert(0, str(HERE))
        for f, p in first.items():
            if f not in raw and (file := find(p.stem, tuple(a.files))):
                arch = lambda p=p: counts.sized(p.stem, p.arms, 1024).arch
                raw[f] = fit_targets.match(file, f, arch, a.match, counts.table)
                counts.save()
    if a.targets:
        sys.path.insert(0, str(HERE))
        for f, p in first.items():
            if f not in raw and (file := find(p.stem, tuple(a.files))):
                arch = lambda p=p: counts.sized(p.stem, p.arms, 1024).arch
                raw[f] = fit_targets.by_architecture(raw, file, arch) or {}
    raw = {f: r for f, r in raw.items() if r.get("n_active")}
    same = {(first[f].width, first[f].counts["arch"]): f for f in raw if f in first}
    for f, p in first.items():
        if f not in raw and (g := same.get((p.width, p.counts["arch"]))):
            raw[f] = raw[g]
    measured = fit_targets.mfus(a.mfu)
    return {
        f: make_target(r, first[f], counts, measured.get(f))
        for f, r in raw.items()
        if f in first
    }


def make_target(
    raw: dict, p: fit_points.Point, counts: Counts, measured: dict | None
) -> fit_targets.Target:
    """A target, its FLOPs per token from the compiler where its row has none: its
    FLOPs per active parameter at the nearest valid width, times its own."""
    flops = raw.get("flops")
    if not flops and raw.get("d_model"):
        c = counts.flops(p.stem, p.arms, raw["d_model"])
        flops = c["flops"] / c["active"] * raw["n_active"] if "flops" in c else math.nan
    sparse = p.counts.get("moe") or p.counts.get("hybrid")
    return fit_targets.target(raw, flops or math.nan, bool(sparse), measured)


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
    for x in analyses:
        if x.metric in a.plot_metrics.split(",") and x.laws:
            fams = fit_plots.chosen(
                x, targets, a.root, a.plot.split(",") if a.plot else None
            )
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
    """What the numbers rest on, said beside them: `facts` has the replicates'
    noise, the points dropped as diverged and the points kept."""
    noise, points = facts["noise"], facts["points"]
    placeholder = sorted(f for f, t in targets.items() if not t.mfu_measured)
    out = [
        f"N is the parameters a token goes through besides the embeddings; points at the best of their learning rates ({a.pick}).",
        f"E is shared by every family; {'all families fit at once' if not a.no_joint else 'the root fit alone, then each family with its parent held'}"
        + (
            f"; {a.share} pooled: only the root has them of its own."
            if a.share
            else "."
        ),
        f"Bootstrap: {a.reps} refits with each point off by a relative noise of {a.noise:g}"
        + (
            f"; replicate runs measured {noise[0]:.2%} ({noise[1]} degrees of freedom)."
            if noise
            else "."
        ),
        f"Targets: {a.targets or ('serving.py match at ' + a.match if a.match else 'none')} (budget {a.budget}).",
        f"GPU-hours = tokens x FLOPs/token / (MFU x {fit_targets.PEAK:.3g} x 3600).",
    ]
    if gone := facts["diverged"]:
        names = ", ".join(sorted(f"{p.family} d{p.width} t{p.per_param}" for p in gone))
        out.append(
            f"Dropped as diverged (worse than a smaller rung or fewer tokens): {names}."
        )
    if edges := sorted({f"{p.family} d{p.width}" for p in points if p.edge}):
        out.append(
            f"Best lr at the edge of its scan (scan further): {', '.join(edges)}."
        )
    if placeholder:
        out.append(
            f"PLACEHOLDER MFU ({fit_targets.MFU_DENSE} dense, {fit_targets.MFU_SPARSE} MoE or hybrid), "
            f"no measured throughput, for: {', '.join(placeholder)}."
        )
    return out


def markdown(a: argparse.Namespace, tables: dict, targets: dict, facts: dict) -> str:
    parts = [
        "# Timeline fit\n",
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
        "## (a) Unlimited data: GPU-hours to the reference target's losses (vs: the parent's, or the root's)\n",
        fit_report.markdown(tables["extrapolation_unlimited"], fit_report.UNLIMITED),
    ]
    parts += [
        "## (b) Limited data: loss on the same tokens\n",
        fit_report.markdown(
            tables["extrapolation_limited"], fit_report.LIMITED_COLUMNS
        ),
    ]
    for key, title in (
        ("multipliers", "Effective-parameter multipliers at each rung"),
        ("laws", "Laws"),
        ("points", "Points"),
    ):
        parts += [f"## {title}\n", fit_report.markdown(tables[key])]
    return "\n".join(parts)


def log_wandb(tables: dict, plots: list[Path], project: str) -> None:
    """Log the tables as W&B Tables, and the plots, in a run named ``analysis``."""
    import wandb

    run = wandb.init(
        project=project, name="analysis", group="analysis", job_type="analysis"
    )
    for name in (
        "leaderboard",
        "extrapolation_unlimited",
        "extrapolation_limited",
        "multipliers",
        "laws",
        "points",
    ):
        rows = tables[name]
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
    p.add_argument("results", nargs="*", type=Path, help="files of RESULT lines")
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
