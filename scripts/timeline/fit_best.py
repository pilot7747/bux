"""The report's "Today's best architecture": the candidates against their parent and
the 2020 baseline, per parameter (their multiplier at their rungs) and at equal
measured serving cost under both regimes, with the two stated uncertainties: the
serving-cost range (where a target's cost is measured as an upper bound) and the E
range (E held, against E fit).
"""

import math
from dataclasses import replace

import numpy as np
from fit_analysis import LIMITED, Analysis, Options, at_target, tokens_label
from fit_law import NO_DATA
from fit_points import ROOT

COLUMNS = [
    ("candidate", "candidate", "s"), ("vs", "vs", "s"), ("served", "serve", "s"), ("rungs", "rungs", "s"),
    ("paired Δ by rung %", "paired", "s"),
    ("× per parameter", "mult", ".2f"), ("(a) speedup to vs's 1T loss", "speedup", ".3g"),
    ("(a) Δ at vs's 1T GPU-h %", "delta_hours", "+.2f"),
    *((f"(b) Δ {d} %", f"delta_{d}", "+.2f") for d in map(tokens_label, LIMITED)),
    ("(b) Δ 1T, E fit %", "delta_1T_efree", "+.2f"),
    ("(b) Δ 1T, lower cost %", "delta_1T_lower", "+.2f"),
    ("(b) Δ 1T range %", "range_1T", "s"),
]  # fmt: skip


def rows(
    a: Analysis,
    free: Analysis | None,
    candidates: list[str],
    targets: dict,
    points: list,
) -> list[dict]:
    """A row per candidate and reference (its parent, then the baseline)."""
    out = []
    for f in (c for c in candidates if c in a.laws and c in targets):
        for ref in dict.fromkeys([a.fams[f].parent, ROOT]):
            if ref in a.laws and ref in targets:
                out.append(
                    row(a, free, f, ref, targets)
                    | {"paired": paired_text(points, f, ref, a.metric)}
                )
    return out


def paired(points: list, f: str, ref: str, metric: str) -> dict[int, tuple]:
    """`f`'s delta against `ref` at each rung, paired by seed: the mean over the seeds
    both have of each seed's delta, its standard error (none from one seed), and
    the seeds. Seeds shift every family alike, so a pair cancels its seed's shift."""
    by = {(p.family, p.width): p for p in points if p.ladder}
    out = {}
    for (fam, w), p in sorted(by.items()):
        q = by.get((ref, w))
        if fam != f or q is None:
            continue
        seeds = [
            s
            for s in p.by_seed
            if s in q.by_seed and metric in p.by_seed[s] and metric in q.by_seed[s]
        ]
        d = [100 * (p.by_seed[s][metric] / q.by_seed[s][metric] - 1) for s in seeds]
        if d:
            se = (
                float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan
            )
            out[w] = (float(np.mean(d)), se, tuple(seeds))
    return out


def paired_text(points: list, f: str, ref: str, metric: str) -> str:
    """The paired deltas by rung, as ``d512 -2.61 ± 0.21 (seeds 0,1,2)``."""
    parts = []
    for w, (d, se, seeds) in paired(points, f, ref, metric).items():
        err = f" ± {se:.2f}" if math.isfinite(se) else ""
        parts.append(
            f"d{w} {d:+.2f}{err} (seed{'s' if len(seeds) > 1 else ''} {','.join(map(str, seeds))})"
        )
    return "; ".join(parts)


def row(a: Analysis, free: Analysis | None, f: str, ref: str, targets: dict) -> dict:
    """One candidate against one reference."""
    out = {"candidate": f, "vs": ref} | per_parameter(a, f, ref)
    out |= serving(a.laws, f, ref, targets)
    if (lower := targets[f].lower) is not None:
        out["delta_1T_lower"] = serving(a.laws, f, ref, {**targets, f: lower})[
            "delta_1T"
        ]
    if free is not None and f in free.laws and ref in free.laws:
        out["delta_1T_efree"] = serving(free.laws, f, ref, targets)["delta_1T"]
    ends = [
        out[k] for k in ("delta_1T", "delta_1T_lower", "delta_1T_efree") if k in out
    ]
    out["range_1T"] = f"[{min(ends):+.2f}, {max(ends):+.2f}]"
    return out


def per_parameter(a: Analysis, f: str, ref: str) -> dict:
    """`f`'s effective-parameter multiplier over `ref`'s law at its rungs, and its
    rungs with the runs each averages."""
    fam, law = a.fams[f], a.laws[ref]
    lad = np.flatnonzero(fam.ladder)
    mults = [law.n_for(fam.y[i], fam.d[i]) / fam.n[i] for i in lad]
    finite = [m for m in mults if 0 < m < math.inf]
    runs = fam.count if fam.count is not None else np.ones(len(fam.y))
    widths = "/".join(
        f"{int(fam.width[i])}×{int(runs[i])}"
        for i in sorted(lad, key=lambda i: fam.width[i])
    )
    return {
        "rungs": widths,
        "mult": float(np.exp(np.mean(np.log(finite)))) if finite else math.nan,
    }


def scale_lines(
    a: Analysis, candidates: list[str], points: list, lr_rows: list[dict]
) -> list[str]:
    """Whether each candidate's gain over its parent shrinks from rung to rung
    (paired by seed), how its fit treats it, and the learning rates it rests on."""
    out = []
    for f in (c for c in candidates if c in a.q):
        parent = a.fams[f].parent
        rungs = paired(points, f, parent, a.metric)
        if len(rungs) > 1:
            steps = " → ".join(f"{d:+.2f}% at d{w}" for w, (d, _, _) in rungs.items())
            own = (
                "alpha too"
                if "alpha" in a.fams[f].free
                else f"A only, {parent}'s alpha and data term"
            )
            out.append(
                f"{f} against {parent}, paired by seed: {steps}. Its fit has {own}, so one multiplier at "
                f"every scale (×{a.q[f].get('mult_target', math.nan):.2f} at the target): a Δ% that shrinks "
                f"while the multiplier holds is the A term's share of the loss falling; a multiplier that "
                f"falls too is averaged over the rungs, not followed."
            )
    return out + [lr_line(candidates, points, lr_rows, a)]


def lr_line(
    candidates: list[str], points: list, lr_rows: list[dict], a: Analysis
) -> str:
    """The caveat on learning rates: every family keeps its d512 vertex at every width."""
    fams = [f for f in candidates if f in a.fams]
    root = next((f for f, fam in a.fams.items() if fam.parent == NO_DATA), ROOT)
    fams += [a.fams[fams[0]].parent, root] if fams else []
    by = {
        f: " ".join(
            f"d{p.width} {p.lr:.3g}"
            for p in sorted(points, key=lambda p: p.width)
            if p.family == f and p.ladder
        )
        for f in fams
    }
    fragile = {
        r["family"]: r
        for r in lr_rows
        if r["rung"] == 512 and math.isfinite(r["above"]) and r["above"] < 20
    }
    worst = sorted(fragile.values(), key=lambda r: -r["above"])[:5]
    text = (
        "Learning rates: every family keeps its d512 vertex at every width ("
        + "; ".join(f"{f}: {v}" for f, v in by.items())
        + ")"
    )
    if worst:
        ref = fragile.get(root)
        text += (
            "; the most lr-fragile at d512, one lr step above their best: "
            + ", ".join(f"{r['family']} +{r['above']:.1f}%" for r in worst)
        )
        text += f" (against {root}'s +{ref['above']:.1f}%)" if ref else ""
    return text + ", so a fragile arm may sit off its optimum on the wider rungs."


def serving(laws: dict, f: str, ref: str, targets: dict) -> dict:
    """`f`'s target against `ref`'s at equal serving cost, under both regimes."""
    q = at_target(f, ref, laws, targets, ref)
    unl = q["unlimited"]["1T"]
    out = {"speedup": unl["speedup"], "delta_hours": unl["delta_equal_hours"]}
    return out | {
        f"delta_{d}": q["limited"][d]["delta_parent"]
        for d in map(tokens_label, LIMITED)
    }


def verdict(table: list[dict], targets: dict) -> str:
    """One sentence: the best candidate against its parent and the baseline, at the
    main serving precision, the others' 1T delta beside it."""
    if not table:
        return ""
    main = table[0].get("serve")
    rows = [r for r in table if r.get("serve") == main]
    parent = [r for r in rows if r["vs"] != ROOT] or rows
    best = min(parent, key=lambda r: r["delta_1T"])
    base = next(
        (r for r in rows if r["candidate"] == best["candidate"] and r["vs"] == ROOT),
        None,
    )
    alt = [
        r
        for r in table
        if r.get("serve") != main
        and (r["candidate"], r["vs"]) == (best["candidate"], best["vs"])
    ]
    t = targets[best["candidate"]]
    served = f" served in {main}" if main else ""
    text = (
        f"Today's best architecture is {best['candidate']}: at equal measured serving cost{served} it "
        f"predicts {best['delta_1T']:+.1f}% on the objective against {best['vs']} on 1T tokens "
        f"(range {best['range_1T']} over the serving cost and E"
        + "".join(f"; {r['serve']}: {r['delta_1T']:+.1f}%" for r in alt)
        + ")"
    )
    if base is not None:
        text += f", {base['delta_1T']:+.1f}% against the 2020 baseline"
    return text + (
        f", and per parameter it is ×{best['mult']:.2f} {best['vs']} at its rungs "
        f"({best['rungs']}); its MFU is {t.mfu:.2f} ({t.mfu_from})."
    )


def free_e(options: Options) -> Options:
    """The options of the E range's other end: E fit, no bootstrap."""
    return replace(options, e=None, reps=0)
