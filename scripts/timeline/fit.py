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
import fit_best
import fit_family
import fit_inputs
import fit_plots
import fit_points
import fit_report
import fit_targets
import numpy as np
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
    if a.fix_e and a.e_free_variant:
        free_variant(a)


def free_variant(a: argparse.Namespace) -> dict:
    """The fit again with E fit (--fix-e dropped) into OUT/e-free/, without W&B:
    the E-free variant readers take beside the main fit."""
    global ARGV
    main = ARGV
    ARGV = e_free_argv(main, a.out / "e-free")
    try:
        variant = argparse.Namespace(
            **{
                **vars(a),
                "fix_e": None,
                "out": a.out / "e-free",
                "wandb": False,
                "far_end": True,
            }
        )
        return run(variant)
    finally:
        ARGV = main


def e_free_argv(argv: list[str], out: Path) -> list[str]:
    """`argv` without --fix-e and --wandb, writing to `out`."""
    out_argv, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
        elif x in ("--fix-e", "--out"):
            skip = True
            if x == "--out":
                out_argv += ["--out", str(out)]
        elif x.startswith("--fix-e=") or x == "--wandb":
            continue
        elif x.startswith("--out="):
            out_argv.append(f"--out={out}")
        else:
            out_argv.append(x)
    return out_argv + (
        []
        if "--out" in out_argv or any(x.startswith("--out=") for x in out_argv)
        else ["--out", str(out)]
    )


def run(a: argparse.Namespace) -> dict:
    """Read, fit and write; return what was written."""
    runs, short = fit_points.cut_short(fit_inputs.read(a.results))
    runs, wild = fit_points.diverged_seeds(runs)
    excluded = excluded_families(a)
    runs = [r for r in runs if r.family not in excluded]
    counts = Counts(a.cache or a.out / "counts.json", tuple(a.files))
    parents = family_parents(a, runs, counts)
    if not a.keep_base:
        runs = fit_family.alias_bases(runs, counts, parents, a.root)
    blocks = {m: fit_points.seed_noise(runs, m) for m in metric_names(runs)}
    on_seed0 = runs if a.no_seed_adjust else fit_points.seed_adjusted(runs, blocks)
    points, diverged = fit_points.monotone(
        fit_points.points(on_seed0, a.pick, not a.no_average_seeds)
    )
    points = attach(points, counts, a.rung_flops)
    if not any(p.family == a.root for p in points):
        sys.exit(
            f"no points of the root family {a.root!r}; families: {sorted({p.family for p in points})}"
        )
    targets = build_targets(a, points, counts, parents)
    other = [
        p
        for p in (fit_targets.precisions(a.targets) if a.targets else [])
        if p != serving_precision(a)
    ]
    others = {p: build_targets(a, points, counts, parents, p) for p in other}
    metrics = (
        a.metrics.split(",") if a.metrics else fit_points.metrics_of(points, a.root)
    )
    noise = {m: blocks.get(m) or fit_points.seed_noise(runs, m) for m in metrics}
    analyses = [
        fit_analysis.analyze(points, m, parents, targets, options(a, noise[m], m))
        for m in metrics
    ]
    tables = tables_of(analyses, points, metrics, targets, a)
    tables["best"] = best_rows(a, analyses, points, parents, targets, noise, others)
    tables["lr"] = tables["learning_rates"] = fit_points.lr_vertices(runs)
    longctx = fit_targets.load_longctx(a.targets_longctx) if a.targets_longctx else {}
    main = next((x for x in analyses if x.metric == "objective"), None)
    tables["best_longctx"] = (
        fit_best.longctx_rows(main, longctx, (a.candidates or "").split(","))
        if longctx and main
        else []
    )
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
        "longctx": longctx,
        "others": others,
        "wild": wild,
        "excluded": excluded,
        "objective": next(
            (x for x in analyses if x.metric == "objective"), analyses[0]
        ),
    }
    out = write(a, analyses, targets, tables, facts)
    if a.wandb:
        log_wandb(tables, out["plots"], a.project, a.out)
    return out


def best_rows(
    a: argparse.Namespace,
    analyses: list,
    points: list,
    parents: dict,
    targets: dict,
    noise: dict,
    others: dict | None = None,
) -> list[dict]:
    """The candidates' rows of "Today's best architecture", the objective's, at
    the serving precision of `targets` and at each of `others`', with the E range's
    other end fit with E free if --fix-e holds it."""
    main = next((x for x in analyses if x.metric == "objective"), None)
    if not a.candidates or main is None:
        return []
    free = bounded = None
    if a.fix_e:
        o = options(a, noise["objective"], "objective")
        free = fit_analysis.analyze(
            points, "objective", parents, targets, fit_best.free_e(o)
        )
        bounded = fit_analysis.analyze(
            points, "objective", parents, targets, fit_best.bounded_e(o)
        )
    cands = a.candidates.split(",")
    line = fit_best.e_line(free, bounded)
    rows = [
        r | {"serve": serving_precision(a), "e_line": line}
        for r in fit_best.rows(main, free, cands, targets, points, bounded)
    ]
    for p, t in (others or {}).items():
        rows += [
            r | {"serve": p, "e_line": line}
            for r in fit_best.rows(main, free, cands, t, points, bounded)
        ]
    return rows


def serving_precision(a: argparse.Namespace) -> str:
    """The serving precision of the targets: --serve, else the file's default."""
    if a.serve:
        return a.serve
    return fit_targets.precisions(a.targets)[0] if a.targets else "bf16"


def metric_names(runs: list) -> list[str]:
    """The metrics a fit may read: the four and the repositories' by position."""
    keys = {k for r in runs for k in r.metrics}
    return [
        k for k in sorted(keys) if k in fit_points.METRICS or k.startswith("repo_bpb_")
    ]


def excluded_families(a: argparse.Namespace) -> dict[str, str]:
    """--exclude's families, each with its reason: ``residuals/mhc=collapsed init``."""
    out = {}
    for item in (a.exclude or "").split(","):
        if item.strip():
            f, _, why = item.partition("=")
            out[f.strip()] = why.strip()
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
    a: argparse.Namespace, noise: dict | None, metric: str
) -> fit_analysis.Options:
    """The fit's options for a metric: its noise the replicates' (a run's own, and
    the seeds' shared shifts), if they have `MIN_DOF` degrees of freedom, else
    --noise for a run's own and none shared."""
    measured = noise is not None and noise["dof"] >= MIN_DOF
    sigma, seed_sigma = (
        (noise["sigma"], noise["seed_sigma"]) if measured else (a.noise, 0.0)
    )
    source = f"replicates, {noise['dof']} dof" if measured else "--noise"
    share = tuple(a.share.split(",")) if a.share else ()
    e = (
        dict(kv.split("=") for kv in a.fix_e.split(",")).get(metric)
        if a.fix_e
        else None
    )
    drop = tuple(a.drop_scan.split(",")) if a.drop_scan else ()
    return fit_analysis.Options(
        root=a.root, joint=not a.no_joint, sigma=sigma, reps=a.reps, seed=a.seed, share=share,
        sigma_from=source, e=None if e is None else float(e), drop_scan=drop, seed_sigma=seed_sigma,
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
    a: argparse.Namespace,
    points: list,
    counts: Counts,
    parents: dict,
    serve: str | None = None,
) -> dict:
    """Each family's target: from --targets at `serve`'s precision (--serve's by
    default), or serving.py's match at --match; a family with no target of its own
    takes that of one with its architecture."""
    raw = (
        fit_targets.load(a.targets, a.budget, serve or serving_precision(a))
        if a.targets
        else {}
    )
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
    """A target, its FLOPs per token from the compiler where its row has none (its
    FLOPs per active parameter at the nearest valid width, times its own), how its
    size was had (analytic where nothing says), and its size at its lower-bound
    cost if it has one."""
    t = _target(raw, p, counts, measured, source)
    t.cost_from = raw.get("cost_from", "analytic")
    t.reason = raw.get("reason")
    if raw.get("lower"):
        t.lower = _target(raw["lower"], p, counts, measured, source)
    return t


def _target(
    raw: dict, p: fit_points.Point, counts: Counts, measured: dict | None, source: str
) -> fit_targets.Target:
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
        *(["THE FAR END OF THE E RANGE, not a variant to quote: E fit, unidentified (→ 0); the main fit holds E at 0.3204."] if getattr(a, "far_end", False) else []),
        f"N is the parameters a token goes through besides the embeddings; points at the best of their learning rates ({a.pick}), "
        + ("its lr chosen on its first runs, then every run at that lr averaged (the seeds)." if not a.no_average_seeds else "each of its first runs only: seeds and base arms only measure the noise."),
        f"E is shared by every family; {fit}" + (f"; {a.share} pooled: only the root has them of its own." if a.share else "."),
        noise_line(facts["analyses"], a.reps),
        f"Targets: {a.targets or 'none'}" + (f", served in {serving_precision(a)}" if a.targets else "") + (f"; the families it lacks: serving.py's analytic match at {a.match}." if a.match else "."),
        (f"GPU-hours = tokens x FLOPs/token / (MFU x {fit_targets.PEAK:.3g} x 3600); MFU measured, else a measured kin's "
         "(an MoE or hybrid of its file), else its nearest measured ancestor's (the (a) table's 'MFU from')."),
    ]  # fmt: skip
    if a.fix_e:
        out.append(f"E held (not fit) for {a.fix_e}; the other metrics fit it.")
    if a.drop_scan:
        out.append(
            f"Token scans left out for {a.drop_scan}: they take their parent's data term."
        )
    if not a.no_seed_adjust and (b := facts["noise"].get("objective")):
        shifts = ", ".join(
            f"seed {s} {100 * (e - b['effects'][0]):+.2f}%"
            for s, e in sorted(b["effects"].items())
            if s
        )
        out.append(
            f"Points are on seed 0's scale: a run's metrics divided by its seed's shift against seed 0's (objective: {shifts})."
        )
    if excluded := facts.get("excluded"):
        out.append(
            "Left out of every table: "
            + ", ".join(f"{f} ({why})" if why else f for f, why in excluded.items())
            + "."
        )
    if wild := facts.get("wild"):
        names = ", ".join(sorted(f"{r.family} d{r.width} seed {r.seed}" for r in wild))
        out.append(
            f"Seeds left out as diverged (over {fit_points.DIVERGED - 1:.0%} above their point's best run): {names}."
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
    out += cost_lines(targets)
    out += budget_lines(a, targets, facts.get("others") or {})
    if placeholder := sorted(
        f for f, t in targets.items() if t.mfu_from == "placeholder"
    ):
        out.append(
            f"PLACEHOLDER MFU ({fit_targets.MFU_DENSE} dense, {fit_targets.MFU_SPARSE} MoE or hybrid), "
            f"no measured family above it, for: {', '.join(placeholder)}."
        )
    return out


def budget_lines(a: argparse.Namespace, targets: dict, others: dict) -> list[str]:
    """The serving budgets at each precision, and how the targets move between them."""
    found = fit_targets.budgets(a.targets) if a.targets else {}
    if len(found) < 2:
        return []
    main = serving_precision(a)
    text = (
        "Serving budgets of the budget model (old_2022 at d_model 4096), sessions/h per GPU: "
        + ", ".join(f"{p} {b:.0f}" for p, b in found.items())
    )
    for p, t in others.items():
        both = [
            targets[f].n_active / t[f].n_active
            for f in targets
            if f in t and " only (" not in targets[f].cost_from
        ]
        if both and p in found and main in found:
            text += (
                f". The budget model serves {found[main] / found[p]:.2f}× faster in {main} than in {p}, more than "
                f"the families gain, so the {main} targets are smaller: N active ×{float(np.median(both)):.2f} theirs in {p} (median)"
            )
    return [text + "."]


def cost_lines(targets: dict) -> list[str]:
    """How the targets' sizes were had, where not measured outright, and why."""
    say = {
        "analytic": "Sized by the analytic serving model (no measured match)",
        "measured, DeepSeek dims": "Sized at the DeepSeek-dims point, the architecture's cost (as trained it serves slower)",
        "measured upper bound of cost": "Cost measured with fused residual kernels, an upper bound: the 'lower cost' columns give the other end",
        "kernel-limited, anchored": "KERNEL-LIMITED (the plugin's kernels, not the architecture, set the timing): extrapolated at the anchored size",
        "kernel-limited, analytic": "KERNEL-LIMITED: extrapolated at the analytic size",
    }  # fmt: skip
    groups: dict[str, list[str]] = {}
    for f, t in sorted(targets.items()):
        if t.cost_from in say:
            groups.setdefault(t.cost_from, []).append(f)
    out = [f"{say[k]}: {', '.join(v)}." for k, v in groups.items()]
    fallback = sorted(f for f, t in targets.items() if " only (" in t.cost_from)
    if fallback:
        out.append(
            f"Served at the other precision, with none measured at this one: {', '.join(fallback)}."
        )
    return out + [
        f"Why kernel-limited: {r}"
        for r in sorted(
            {t.reason for t in targets.values() if t.reason and "kernel" in t.cost_from}
        )
    ]


def converged(analyses: list, reps: int) -> str:
    """How many of the bootstrap's refits converged, per metric, where any didn't."""
    short = [f"{x.metric} {x.boot_refits}" for x in analyses if x.boot_refits < reps]
    return "all converged" if not short else "converged: " + ", ".join(short)


def noise_line(analyses: list, reps: int) -> str:
    """Each metric's noise, where it comes from, and the verdict's rule."""
    each = ", ".join(
        f"{x.metric} {x.sigma:.2%} own and {x.seed_sigma:.2%} by seed ({x.sigma_from})"
        for x in analyses
    )
    return (
        f"Noise of a run, seeds as blocks (a seed shifts every family's run alike): {each}. "
        f"The bootstrap refits {reps} times ({converged(analyses, reps)}) with each point off by its runs' own noise over the root "
        f"of their number and by its seeds' shifts, drawn once a refit for every family. "
        f"'No measurable difference' means |delta| < {fit_analysis.VERDICT} sigmas of the delta at every "
        f"rung: own sigma x sqrt(1/n + 1/n_parent) for points on the same seeds, the seeds' spread added otherwise."
    )


def markdown(a: argparse.Namespace, tables: dict, targets: dict, facts: dict) -> str:
    parts = [
        "# Timeline fit\n",
        f"Command, from the repository's root: `{command()}`\n",
        *(f"- {s}" for s in assumptions(a, targets, facts)),
        "",
    ]
    if tables.get("best"):
        parts += ["## Today's best architecture\n", fit_best.verdict(tables["best"], targets) + "\n",
                  fit_report.markdown(tables["best"], fit_best.COLUMNS),
                  *(f"- {line}" for line in fit_best.scale_lines(facts["objective"], a.candidates.split(","), facts["points"], tables["learning_rates"])), "",
                  *([f"- {tables['best'][0]['e_line']}"] if tables["best"][0].get("e_line") else []),
                  ("Against each reference's target of equal measured serving cost. The main numbers hold E at 0.3204; "
                  "the range spans E held at 0.3204 … E fit (unidentified, → 0), E fit within [0.25, 0.35] between them, "
                  "and the serving cost's lower bound where it is measured as an upper one.\n")]  # fmt: skip
    if tables.get("best_longctx"):
        ctx = facts["longctx"]
        parts += ["## Today's best at equal serving cost by context\n",
                  ("The same laws (objective, E held at 0.3204), at each context's FP8 targets of equal serving cost "
                  "against the budget model at that context; each family's predicted objective on 1T tokens.\n"),
                  fit_report.markdown(tables["best_longctx"], fit_best.longctx_columns(ctx)),
                  *(f"- {line}" for line in fit_best.longctx_notes(ctx, a.longctx_preliminary)), ""]  # fmt: skip
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
        "best",
        "best_longctx",
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
    if (isinstance(v, float) and not math.isfinite(v)) or v == "—":
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
        "--targets",
        type=Path,
        help="targets JSON (targets-measured.json's flags read), or serving.py's serving-table.json",
    )
    p.add_argument(
        "--serve",
        choices=fit_targets.PRECISIONS,
        help="the targets' serving precision; default: fp8 once the file has it",
    )
    p.add_argument(
        "--targets-longctx",
        type=Path,
        help="targets-measured-longctx.json: the candidates' and camps' 1T objective at each context's targets",
    )
    p.add_argument(
        "--longctx-preliminary",
        action="store_true",
        help="mark the long-context table preliminary (its targets not all in)",
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
        "--candidates", help='comma-separated families for "Today\'s best architecture"'
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
        "--no-seed-adjust",
        action="store_true",
        help="average seeds as they are, not on seed 0's scale",
    )
    p.add_argument(
        "--exclude",
        help="families left out of every table, each with its reason: fam=why,fam2=why",
    )
    p.add_argument(
        "--e-free-variant",
        action="store_true",
        help="with --fix-e, also write the fit with E free to OUT/e-free/: the E range's far end, not a variant to quote",
    )
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
