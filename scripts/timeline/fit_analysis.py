"""What the fits say, per metric: each family's effective-parameter multiplier over
its parent at its rungs (the N at which the parent's law reaches the family's loss,
over the family's N), and its target's predicted loss under unlimited data (by
training GPU-hours) and under limited data (the same tokens for every target), each
with the range of its leave-one-rung-out fits and its noise bootstrap's.
"""

import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from fit_law import (
    NO_DATA,
    Family,
    Law,
    bootstrap,
    fit_metric,
    leave_one_rung_out,
    topological,
)
from fit_points import Point
from fit_targets import Target

LIMITED = (130e9, 1e12, 10e12)  # tokens of the limited-data regime
REFERENCES = ("20tpp", "1T")  # the root target at 20 tokens a parameter, and at 1T


@dataclass
class Options:
    """How to fit and how far to look for uncertainty."""

    root: str = "baseline"
    joint: bool = True  # E jointly with the families that have every term
    sigma: float = 0.002  # the bootstrap's relative noise of a point
    reps: int = 100
    seed: int = 0


@dataclass
class Analysis:
    """One metric's laws and what they say."""

    metric: str
    fams: dict[str, Family]
    laws: dict[str, Law]
    q: dict[str, dict]
    loro: dict[str, dict] = field(default_factory=dict)
    boot: dict[str, dict] = field(default_factory=dict)


def families(points: list[Point], metric: str, parents: dict, root: str) -> dict:
    """Each family's points with `metric` and a parameter count."""
    by: dict[str, list[Point]] = defaultdict(list)
    for p in points:
        if metric in p.metrics and math.isfinite(p.n):
            by[p.family].append(p)
    out = {}
    for f, ps in by.items():
        arrays = [np.array(v, float) for v in zip(*[(p.n, p.tokens, p.metrics[metric], p.width, p.per_param) for p in ps], strict=True)]  # fmt: skip
        out[f] = Family(*arrays, NO_DATA if f == root else parents.get(f, root))
    kept = topological(out, root) if root in out else []
    if dropped := sorted(set(out) - set(kept)):
        print(
            f"warning: {metric}: no parent's points for {', '.join(dropped)}",
            file=sys.stderr,
        )
    return {f: out[f] for f in kept}


def analyze(
    points: list[Point], metric: str, parents: dict, targets: dict, o: Options
) -> Analysis:
    """Fit `metric`'s laws, and say what they say with their uncertainty."""
    fams = families(points, metric, parents, o.root)
    laws = fit_metric(fams, o.root, o.joint)
    a = Analysis(metric, fams, laws, quantities(fams, laws, targets, o.root))
    for f in fams:
        others = leave_one_rung_out(fams, f, laws, o.root, o.joint)
        qs = [one(f, fams, {**laws, f: law}, targets, o.root) for law in others]
        a.loro[f] = spread(qs, 0, 100)
    if o.reps:
        reps = bootstrap(fams, laws, o.root, o.joint, o.sigma, o.reps, o.seed)
        qs = [quantities(fb, lb, targets, o.root) for fb, lb in reps]
        a.boot = {f: spread([q[f] for q in qs if f in q], 5, 95) for f in fams}
    return a


def quantities(fams: dict, laws: dict, targets: dict, root: str) -> dict[str, dict]:
    return {f: one(f, fams, laws, targets, root) for f in laws}


def one(f: str, fams: dict, laws: dict, targets: dict[str, Target], root: str) -> dict:
    """What family `f`'s points and law say."""
    rows = rungs(f, fams, laws)
    q = {"law": asdict(laws[f]), "rungs": rows, **trend(rows)}
    if f in targets and root in targets:
        q |= at_target(f, fams[f].parent, laws, targets, root)
    return q


def rungs(f: str, fams: dict[str, Family], laws: dict[str, Law]) -> dict[int, dict]:
    """At each of `f`'s rungs: its loss, against its parent's, and its multiplier
    from its loss (``mult``) and from its law (``mult_law``)."""
    fam, parent = fams[f], fams[f].parent
    rows: dict[int, dict] = {}
    for i in np.flatnonzero(fam.ladder):
        w, n, d, y = int(fam.width[i]), fam.n[i], fam.d[i], fam.y[i]
        row = {"loss": float(y), "n": float(n), "d": float(d)}
        if parent != NO_DATA:
            p = laws[parent]
            row["mult"] = p.n_for(y, d) / n
            row["mult_law"] = p.n_for(float(laws[f].loss(n, d)), d) / n
            pf = fams[parent]
            if (j := np.flatnonzero(pf.ladder & (pf.width == w))).size:
                row["delta"] = pct(y, pf.y[j[0]])
        rows[w] = row
    return rows


def trend(rows: dict[int, dict]) -> dict:
    """The multipliers' geometric mean over the rungs, and their slope against N in
    log-log: above 0, the gain grows with scale."""
    pairs = [
        (r["n"], r["mult"]) for r in rows.values() if 0 < r.get("mult", 0) < math.inf
    ]
    if not pairs:
        return {}
    n, m = np.log([p[0] for p in pairs]), np.log([p[1] for p in pairs])
    out = {"mult": float(np.exp(m.mean()))}
    if len(pairs) > 1:
        out["mult_slope"] = float(np.polyfit(n, m, 1)[0])
    return out


def at_target(f: str, parent: str, laws: dict, targets: dict, root: str) -> dict:
    """`f`'s target under limited data (each of `LIMITED` tokens) and unlimited data
    (the GPU-hours to reach the root target's loss of each of `REFERENCES`, and the
    loss in the GPU-hours the root target takes to reach it), against the root's."""
    t, r = targets[f], targets[root]
    law, base = laws[f], laws[root]
    d20 = 20 * r.n_active
    out: dict = {"limited": {}, "unlimited": {}}
    for d in LIMITED:
        loss = float(law.loss(t.n_active, d))
        row = {"loss": loss, "delta": pct(loss, base.loss(r.n_active, d))}
        if parent not in (root, NO_DATA) and parent in targets:
            row["delta_parent"] = pct(
                loss, laws[parent].loss(targets[parent].n_active, d)
            )
        out["limited"][tokens_label(d)] = row
    for ref, d in zip(REFERENCES, (d20, 1e12), strict=True):
        out["unlimited"][ref] = unlimited(
            law, t, float(base.loss(r.n_active, d)), r.hours(d)
        )
    floor = base.asymptote(r.n_active)
    out["asymptote"] = {
        "loss": law.asymptote(t.n_active),
        "delta": pct(law.asymptote(t.n_active), floor),
    }
    if parent != NO_DATA:
        out["mult_target"] = (
            laws[parent].n_for(float(law.loss(t.n_active, d20)), d20) / t.n_active
        )
    return out


def unlimited(law: Law, t: Target, loss: float, hours: float) -> dict:
    """The GPU-hours `t` takes to reach `loss`, its speedup over `hours`, and its
    loss in `hours`."""
    tokens = law.d_for(loss, t.n_active)
    own = t.hours(tokens)
    equal = float(law.loss(t.n_active, t.tokens(hours)))
    return {
        "loss_ref": loss,
        "hours_root": hours,
        "tokens": tokens,
        "hours": own,
        "speedup": hours / own if own > 0 else math.inf,
        "loss_equal_hours": equal,
        "delta_equal_hours": pct(equal, loss),
    }


def pct(a: float, b: float) -> float:
    return 100.0 * (float(a) / float(b) - 1.0)


def tokens_label(d: float) -> str:
    return f"{d / 1e12:g}T" if d >= 1e12 else f"{d / 1e9:g}B"


def flatten(q: dict, prefix: tuple = ()) -> dict[tuple, float]:
    """The numbers of a nested dict, by their path."""
    out = {}
    for k, v in q.items():
        if isinstance(v, dict):
            out |= flatten(v, (*prefix, k))
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[(*prefix, k)] = float(v)
    return out


def spread(qs: list[dict], lo: float, hi: float) -> dict[tuple, tuple[float, float]]:
    """The `lo` and `hi` percentiles of each number of `qs`, by its path."""
    values: dict[tuple, list[float]] = defaultdict(list)
    for q in qs:
        for path, v in flatten(q).items():
            values[path].append(v)
    return {
        p: (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
        for p, v in values.items()
        if all(math.isfinite(x) for x in v)
    }
