"""Fit the fractional factorial on the late-2026 hybrid (factorial.bux): the main
effects of each metric and the 2-factor interactions the design tells apart, by
least squares over its runs, with standard errors from the spread of the seeds.

An arm's label gives its levels: ``r1v0k1f0p1s0`` is hyper-connections (R: 0 plain,
1 hc, 2 AttnRes) with the key-value conv (K) and RoPE (P) on. Effects are in percent:
the fit is of 100 ln(metric), each column centered, so a main effect is the average
change from switching its factor on. For each metric it prints the effects with their
t (|t| > 2 is significant), the interactions with what they are aliased with, and
then, from the main effects and the significant interactions, the predicted best of
the 96 combinations, its change against the hybrid (every factor off), and the
combination of the significant gains alone.

    python scripts/timeline/factorial.py results.jsonl runs/ [--reuse] [--extra]
    python scripts/timeline/factorial.py --selftest

``--reuse`` fills arms and seeds the factorial has not run from runs of the same
config in milestones.bux and stacks.bux; ``--extra`` adds stacks.bux's other arms,
points off the design. Both keep only runs at the factorial's learning rate.
"""

import argparse
import itertools
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import fit_inputs
import numpy as np
from fit_points import classify, number, parse_name

FACTORS = ("V", "K", "F", "P", "S")
R_LEVELS = ("plain", "hc", "attnres")
LABEL = re.compile(r"r([012])v([01])k([01])f([01])p([01])s([01])")
BASE = "r0v0k0f0p0s0"
METRICS = ("objective", "repo_bpb", "repo_bpb_8k")
LR = 2e-3
#: Runs of other files whose config is an arm's (bux plan gives them its run id).
SAME = {("milestones", "m2026_hybrid"): BASE, ("stacks", "vres"): "r0v1k0f0p0s0"}
#: stacks.bux's arms that are points of the factorial's space off its design.
EXTRA = {
    ("stacks", "kvconv"): "r0v0k1f0p0s0",
    ("stacks", "rope"): "r0v0k0f0p1s0",
    ("stacks", "best_nope"): "r0v1k1f0p0s0",
    ("stacks", "best"): "r0v1k1f0p1s0",
}
#: The interactions to try, likeliest first; one is kept if the design tells it
#: from the effects before it.
PAIRS = (
    ("V", "K"), ("V", "P"), ("K", "P"), ("P", "S"), ("R", "V"), ("V", "S"),
    ("V", "F"), ("K", "S"), ("F", "P"), ("K", "F"), ("F", "S"), ("R", "K"),
    ("R", "P"), ("R", "S"), ("R", "F"),
)  # fmt: skip
TOLERANCE = 0.3  # a column is new if this much of its norm is outside the others'


def levels(label: str) -> dict[str, int]:
    """The 0/1 indicators of an arm: hc, attnres and each binary factor."""
    m = LABEL.fullmatch(label)
    if m is None:
        raise ValueError(f"not a factorial arm: {label}")
    r, *b = map(int, m.groups())
    return {"hc": int(r == 1), "attnres": int(r == 2), **dict(zip(FACTORS, b))}


def label_of(r: int, bits: tuple[int, ...]) -> str:
    """The arm's label from R's level and the binary factors' levels."""
    return f"r{r}" + "".join(f"{f.lower()}{b}" for f, b in zip(FACTORS, bits))


def every_arm() -> list[str]:
    """The 96 combinations of the six factors."""
    return [
        label_of(r, b) for r in range(3) for b in itertools.product((0, 1), repeat=5)
    ]


def main_terms() -> list[str]:
    return ["hc", "attnres", *FACTORS]


def pair_terms(pair: tuple[str, str]) -> list[str]:
    """An interaction's columns: R's has one for hc and one for AttnRes."""
    a, b = pair
    return [f"{r}×{b}" for r in ("hc", "attnres")] if a == "R" else [f"{a}×{b}"]


# Reading the runs.


def rows(paths: list[Path]) -> list[dict]:
    """The RESULT rows of files and the rows of run directories under `paths`."""
    out: list[dict] = []
    for path in paths:
        if path.is_dir():
            out += [fit_inputs.run_row(d) for d in fit_inputs.run_dirs(path)]
        else:
            out += fit_inputs.result_rows(path)
    return out


def observation(row: dict, aliases: dict) -> tuple[str, int, int, dict] | None:
    """(arm, seed, rank, metrics) of a finished d512 run of the factorial, or of a
    run `aliases` maps to an arm (rank 1: the factorial's own runs win), else None."""
    if row.get("state") not in ("done", None):
        return None
    stem, names = parse_name(row.get("name") or row["dir"])
    width, per_param, settings, arms = classify(names)
    if width != 512 or per_param != 20 or len(arms) != 1:
        return None
    seed = int(settings.pop("seed", 0))
    scan = settings.pop("optimizer.lr", None)
    lr = number(row.get("lr")) or number(scan)
    if settings or (lr is not None and not math.isclose(lr, LR)):
        return None
    if stem == "factorial":
        if scan is not None or not LABEL.fullmatch(arms[0]):
            return None  # an lr scan of an arm, not one of its seeds
        return arms[0], seed, 0, row
    arm = aliases.get((stem, arms[0]))
    return (arm, seed, 1, row) if arm else None


def collect(paths: list[Path], reuse: bool, extra: bool) -> dict[tuple[str, int], dict]:
    """The metrics of each (arm, seed): the factorial's run, else a reused one."""
    aliases = (SAME if reuse else {}) | (EXTRA if extra else {})
    best: dict[tuple[str, int], tuple[int, dict]] = {}
    for row in rows(paths):
        if (obs := observation(row, aliases)) is None:
            continue
        arm, seed, rank, metrics = obs
        if (arm, seed) not in best or rank <= best[arm, seed][0]:
            best[arm, seed] = (rank, metrics)
    return {k: m for k, (_, m) in sorted(best.items())}


# The fit.


class Design:
    """The model matrix of some terms over the observed arms, columns centered on
    the observations' means, so it codes any other arm the same way."""

    def __init__(self, arms: list[str], terms: list[str]) -> None:
        self.terms = terms
        raw = np.array([[self.raw(a, t) for t in terms] for a in arms], dtype=float)
        self.center = raw.mean(axis=0) if len(arms) else np.zeros(len(terms))
        self.x = self.code(arms)

    @staticmethod
    def raw(arm: str, term: str) -> float:
        lv = levels(arm)
        return float(np.prod([lv[f] for f in term.split("×")]))

    def code(self, arms: list[str]) -> np.ndarray:
        raw = np.array(
            [[self.raw(a, t) for t in self.terms] for a in arms], dtype=float
        )
        return np.hstack([np.ones((len(arms), 1)), raw - self.center])


def outside(c: np.ndarray, x: np.ndarray) -> float:
    """The fraction of `c`'s norm outside the span of `x`'s columns."""
    fit, *_ = np.linalg.lstsq(x, c, rcond=None)
    return float(np.linalg.norm(c - x @ fit) / max(np.linalg.norm(c), 1e-12))


def estimable(arms: list[str]) -> tuple[list[str], dict[str, list[tuple[str, float]]]]:
    """The interaction columns the design tells apart, in `PAIRS`' order after the
    main effects, and for each kept one the dropped ones it carries (alias
    coefficients of at least 0.25)."""
    kept, dropped = [], []
    for pair in PAIRS:
        for term in pair_terms(pair):
            x = Design(arms, main_terms() + kept + [term]).x
            new = outside(x[:, -1], x[:, :-1]) > TOLERANCE
            (kept if new else dropped).append(term)
    x = Design(arms, main_terms() + kept).x
    aliases: dict[str, list[tuple[str, float]]] = {t: [] for t in kept}
    for term in dropped:
        coef, *_ = np.linalg.lstsq(x, Design(arms, [term]).x[:, 1], rcond=None)
        for t, v in zip(kept, coef[1 + len(main_terms()) :]):
            if abs(v) >= 0.25:
                aliases[t].append((term, float(v)))
    return kept, aliases


class Fit:
    """Least squares of 100 ln(metric) on some terms; the variance of a run from the
    seeds' spread around their arm's mean (the pure error)."""

    def __init__(self, obs: list[tuple[str, float]], terms: list[str]) -> None:
        arms = [a for a, _ in obs]
        y = np.array([100 * math.log(v) for _, v in obs])
        self.design = Design(arms, terms)
        x = self.design.x
        self.beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        self.s2, self.dof = pure_error(arms, y)
        if self.dof == 0:  # no seeds: the residuals' variance instead
            self.dof = max(len(y) - np.linalg.matrix_rank(x), 1)
            self.s2 = float(np.sum((y - x @ self.beta) ** 2) / self.dof)
        self.cov = self.s2 * np.linalg.pinv(x.T @ x)

    def effects(self) -> list[tuple[str, float, float]]:
        """(term, estimate, standard error) of every term."""
        se = np.sqrt(np.diag(self.cov))
        return [
            (t, self.beta[i + 1], se[i + 1]) for i, t in enumerate(self.design.terms)
        ]

    def delta(self, arm: str, against: str = BASE) -> tuple[float, float]:
        """`arm`'s predicted change against `against`, in percent, and its error."""
        c = (self.design.code([arm]) - self.design.code([against]))[0]
        return float(c @ self.beta), float(math.sqrt(max(c @ self.cov @ c, 0.0)))


def pure_error(arms: list[str], y: np.ndarray) -> tuple[float, int]:
    """The pooled variance of the runs around their arm's mean, and its dof."""
    groups: dict[str, list[float]] = {}
    for a, v in zip(arms, y):
        groups.setdefault(a, []).append(v)
    ss = sum(float(np.sum((np.array(g) - np.mean(g)) ** 2)) for g in groups.values())
    dof = len(y) - len(groups)
    return (ss / dof if dof else 0.0), dof


def t_of(est: float, se: float) -> float:
    return est / se if se > 0 else (math.inf if est else 0.0)


# The report.


def describe(arm: str) -> str:
    lv = levels(arm)
    r = R_LEVELS[1] if lv["hc"] else R_LEVELS[2] if lv["attnres"] else R_LEVELS[0]
    on = [f for f in FACTORS if lv[f]]
    return f"{arm} ({r}{', ' + ' '.join(on) if on else ''})"


def table(fit: Fit, notes: dict | None = None) -> list[str]:
    lines = [f"  {'term':14} {'effect %':>9} {'se':>6} {'t':>6}"]
    for term, est, se in fit.effects():
        t = t_of(est, se)
        mark = " *" if abs(t) > 2 else ""
        alias = ""
        if notes and notes.get(term):
            alias = "   carries " + ", ".join(f"{a} ({c:+.2f})" for a, c in notes[term])
        lines.append(f"  {term:14} {est:+9.2f} {se:6.2f} {t:+6.1f}{mark}{alias}")
    return lines


def significant_gains(fit: Fit) -> str:
    """The hybrid with each factor whose main effect is a significant gain on, and
    R at its best level if that is one."""
    eff = {t: (e, s) for t, e, s in fit.effects()}
    gain = {t: e < 0 and abs(t_of(e, s)) > 2 for t, (e, s) in eff.items()}
    rs = [r for r in ("hc", "attnres") if gain[r]]
    r = 1 + ("hc", "attnres").index(min(rs, key=lambda x: eff[x][0])) if rs else 0
    return label_of(r, tuple(int(gain[f]) for f in FACTORS))


def measured(obs: list[tuple[str, float]]) -> list[str]:
    """Each arm's runs and their mean's change against the hybrid's, in percent."""
    by: dict[str, list[float]] = {}
    for a, v in obs:
        by.setdefault(a, []).append(100 * math.log(v))
    base = np.mean(by[BASE]) if BASE in by else math.nan
    return ["Measured:"] + [
        f"  {describe(a):40} {len(v)} runs {np.mean(v) - base:+6.2f}%"
        for a, v in sorted(by.items())
    ]


def report(metric: str, data: dict[tuple[str, int], dict]) -> list[str]:
    obs = [(a, m[metric]) for (a, _), m in data.items() if m.get(metric, 0) > 0]
    arms = sorted({a for a, _ in obs})
    if len(arms) < 8:
        return [f"## {metric}: {len(arms)} arms, too few to fit"]
    main = Fit(obs, main_terms())
    kept, notes = estimable(arms)
    full = Fit(obs, main_terms() + kept)
    sig = [t for t, e, s in full.effects() if t in kept and abs(t_of(e, s)) > 2]
    final = Fit(obs, main_terms() + sig)
    noise = f"a run's noise {math.sqrt(main.s2):.2f}% ({main.dof} dof)"
    lines = [
        f"## {metric}: {len(obs)} runs of {len(arms)} arms; {noise}",
        *measured(obs),
        "Main effects:",
        *table(main),
        "With the interactions the design tells apart:",
        *table(full, notes),
        f"Predicted from the main effects{' and ' + ', '.join(sig) if sig else ''}:",
    ]
    ranked = sorted(every_arm(), key=lambda a: final.delta(a)[0])
    for arm in ranked[:5]:
        d, se = final.delta(arm)
        lines.append(f"  {describe(arm):40} {d:+6.2f} ± {se:.2f}% against the hybrid")
    gains = significant_gains(final)
    d, se = final.delta(gains)
    lines += [
        f"  best: {describe(ranked[0])}",
        f"  significant gains alone: {describe(gains)} {d:+.2f} ± {se:.2f}%",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "paths", nargs="*", type=Path, help="RESULT files or run directories"
    )
    p.add_argument("--metrics", default=",".join(METRICS))
    p.add_argument("--reuse", action="store_true", help="runs of the same configs")
    p.add_argument("--extra", action="store_true", help="stacks.bux's other arms")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest()
    data = collect(args.paths, args.reuse, args.extra)
    print(f"{len(data)} runs: " + ", ".join(sorted({a for a, _ in data})))
    for metric in args.metrics.split(","):
        print("\n".join(report(metric, data)) + "\n")
    return 0


# The self-test.

TRUTH = {"hc": -2.0, "attnres": -1.2, "V": -0.5, "K": -0.6, "F": 0.0, "P": 0.3}
TRUTH_PAIRS = {"V×K": 0.4}
DESIGN = [
    "r0v0k0f0p0s0", "r0v1k1f0p0s1", "r0v1k0f1p1s0", "r0v0k1f1p1s1",
    "r2v0k0f0p1s1", "r2v1k1f0p1s0", "r2v1k0f1p0s1", "r2v0k1f1p0s0",
    "r1v1k0f0p1s1", "r1v0k1f0p1s0", "r1v0k0f1p0s1", "r1v1k1f1p0s0",
    "r0v1k0f0p0s0", "r0v0k1f0p0s1", "r0v0k0f1p1s0", "r0v1k1f1p1s1",
]  # fmt: skip


def truth(arm: str) -> float:
    """The self-test's change of an arm against the hybrid, in percent."""
    total = sum(v * Design.raw(arm, t) for t, v in (TRUTH | TRUTH_PAIRS).items())
    return float(total)


def synthetic(path: Path, noise: float, rng: np.random.Generator) -> None:
    """RESULT lines of the design's 48 runs; the hybrid's as milestones.bux's runs."""
    lines = []
    for arm, seed in itertools.product(DESIGN, range(3)):
        v = {m: math.exp(math.log(base) + (truth(arm) + noise * rng.normal()) / 100)
             for m, base in (("objective", 0.68), ("repo_bpb", 0.53), ("repo_bpb_8k", 0.50))}  # fmt: skip
        if arm == BASE:
            label = "optimizer.lr=2e-3" if seed == 0 else f"seed={seed}"
            name = f"milestones[m2026_hybrid, d512, {label}]"
        else:
            name = f"factorial[{arm}, d512, seed={seed}]"
        row = {"dir": name, "name": name, "state": "done", "lr": LR} | v
        lines.append("RESULT " + json.dumps(row))
    path.write_text("\n".join(lines) + "\n")


def selftest() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.txt"
        synthetic(path, 0.0, rng)
        assert len(collect([path], False, False)) == 45, "the hybrid's runs are reused"
        exact = collect([path], True, False)
        assert len(exact) == 48
        obs = [(a, m["objective"]) for (a, _), m in exact.items()]
        kept, _ = estimable(DESIGN)
        assert "V×K" in kept, kept
        fit = Fit(obs, main_terms() + kept)
        worst = max(abs(fit.delta(a)[0] - truth(a)) for a in every_arm())
        assert worst < 1e-6, f"noiseless fit off by {worst}"
        exact = Fit(obs, main_terms() + ["V×K"]).effects()
        synthetic(path, 0.9, rng)
        noisy = collect([path], True, False)
        obs = [(a, m["objective"]) for (a, _), m in noisy.items()]
        fit = Fit(obs, main_terms() + ["V×K"])
        assert 0.6 < math.sqrt(fit.s2) < 1.2, fit.s2
        for (term, est, se), (_, want, _) in zip(fit.effects(), exact):
            assert abs(est - want) < 4 * se, (term, est, se, want)
        print("\n".join(report("objective", noisy)))
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
