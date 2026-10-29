"""Turn the timeline's runs into points: one per family, rung and token budget, at
its best learning rate (runs are read by fit_inputs.py).

A run's name gives its family (the file's stem and its arm labels), its rung (a
label ``d512``), its token budget (a label ``t40``: 40 tokens per baseline
parameter; none is the rung's 20) and the settings a scan or a seed added
(``optimizer.lr=2e-3``, ``seed=1``). A name is the run's own
(``positions[rope, d768]``) or its directory's (``positions_rope_d768-<id>``).
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

ROOT = "baseline"
METRICS = ("objective", "code_bpb", "repo_bpb", "web_bpb")
OBJECTIVE = {"code_bpb": 0.5, "repo_bpb": 0.3, "web_bpb": 0.2}
#: The tokens of each rung at 20 tokens per baseline parameter (rungs.bux), for
#: lines that do not say their budget.
RUNG_TOKENS = {512: 503e6, 768: 1.7e9, 1024: 4.03e9, 1280: 7.86e9}
LADDER = 20  # tokens per baseline parameter of a rung's own budget
NAME = re.compile(r"^(?P<stem>[^\[\]]+?)(?:\[(?P<labels>[^\]]*)\])?(?:-[0-9a-f]{6,})?$")
#: A run's directory as bux names it (``bux.train.trainer.directory``): its name
#: with what a path cannot hold as "_", then its id. The stem has no "_"; the rung
#: is the first "_d<width>" after the arm that only labels follow.
LABEL = r"(?:t\d+|[A-Za-z][\w.]*=[^_]*)"
SLUG = re.compile(
    rf"^(?P<stem>[A-Za-z0-9.+=-]+?)(?:_(?P<arm>.+?))?_(?P<rung>d\d+)"
    rf"(?P<rest>(?:_{LABEL})*)$"
)
SI = {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
#: Keys of a RESULT line that are not final metrics.
SETTINGS = {
    "dir",
    "name",
    "state",
    "lr",
    "budget",
    "tokens",
    "steps",
    "d_model",
    "step",
    "batch",
}
SHORT = 0.97  # a run that trained on less of its rung's budget was cut short


@dataclass
class Run:
    """A finished run: where it sits, and its final metrics."""

    stem: str
    arms: tuple[str, ...]
    width: int
    per_param: int
    lr: float | None
    tokens: float
    metrics: dict[str, float]
    replicate: bool = False  # a seed's run, or a base arm's: another run of a point
    seed: int = 0  # its training seed: seeds act alike across families (a block)

    @property
    def family(self) -> str:
        return family_key(self.stem, self.arms)


@dataclass
class Point:
    """A family at a rung and a token budget, at its best learning rate."""

    family: str
    stem: str
    arms: tuple[str, ...]
    width: int
    per_param: int
    tokens: float
    metrics: dict[str, float]
    lr: float | None = None
    lrs: tuple[float, ...] = ()
    edge: bool = False  # the best learning rate is the scan's highest or lowest
    replicates: int = 1
    counts: dict = field(default_factory=dict)  # total, nonembed, active (fit_counts)
    by_seed: dict = field(default_factory=dict)  # each seed's metrics at its lr

    @property
    def n(self) -> float:
        """The parameters a token goes through besides the embeddings."""
        return self.counts.get("active", math.nan)

    @property
    def ladder(self) -> bool:
        return self.per_param == LADDER


def family_key(stem: str, arms: tuple[str, ...]) -> str:
    """``baseline``, ``positions/rope``: a file's stem and its run's arm labels,
    joined by "_" as a run's directory joins them."""
    return f"{stem}/{'_'.join(arms)}" if arms else stem


def number(value: Any) -> float | None:
    """A number from a RESULT field: 2e-3, 503000000 or "503M"."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    m = re.fullmatch(r"\s*([0-9.]+(?:e[+-]?\d+)?)\s*([kKMBGT]?)\s*", str(value))
    return float(m.group(1)) * SI[m.group(2)] if m else None


def parse_name(name: str) -> tuple[str, list[str]]:
    """The stem and the labels of a run's name (``positions[rope, d768]``) or of
    its directory (``positions_rope_d768-6d3db7e680``)."""
    name = name.strip()
    if "[" not in name and (m := SLUG.match(re.sub(r"-[0-9a-f]{10}$", "", name))):
        rest = re.findall(rf"_({LABEL})", m.group("rest"))
        return m.group("stem"), [
            *filter(None, [m.group("arm"), m.group("rung")]),
            *rest,
        ]
    m = NAME.match(name)
    if m is None:
        return name, []
    labels = [s.strip() for s in (m.group("labels") or "").split(",") if s.strip()]
    return m.group("stem"), labels


def classify(labels: list[str]) -> tuple[int | None, int, dict[str, str], tuple]:
    """A run's rung width, tokens per baseline parameter, settings and arm labels."""
    width, per_param, settings, arms = None, LADDER, {}, []
    for label in labels:
        if m := re.fullmatch(r"d(\d+)", label):
            width = int(m.group(1))
        elif m := re.fullmatch(r"t(\d+)", label):
            per_param = int(m.group(1))
        elif "=" in label:
            key, value = label.split("=", 1)
            settings[key.strip()] = value.strip()
        else:
            arms.append(label)
    return width, per_param, settings, tuple(arms)


def parse_row(row: dict) -> Run | None:
    """A RESULT row's run, or None if it did not finish or has no rung."""
    if row.get("state") not in ("done", None):
        return None
    stem, labels = parse_name(row.get("name") or row["dir"])
    width, per_param, settings, arms = classify(labels)
    width = width or int(number(row.get("d_model")) or 0)
    if not width:
        return None
    lr = number(row.get("lr")) or number(settings.get("optimizer.lr"))
    tokens = (
        number(row.get("tokens"))
        or number(row.get("budget"))
        or RUNG_TOKENS.get(width, math.nan) * (per_param / LADDER)
    )
    seed = int(settings.get("seed", "0"))
    return Run(
        stem, arms, width, per_param, lr, tokens, final_metrics(row), seed != 0, seed
    )


def cut_short(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    """The runs kept, and those that trained on less than `SHORT` of their rung's
    budget (rungs.bux's, by tokens per baseline parameter): they stopped early."""

    def short(r: Run) -> bool:
        want = RUNG_TOKENS.get(r.width, math.nan) * r.per_param / LADDER
        return r.tokens < SHORT * want

    return [r for r in runs if not short(r)], [r for r in runs if short(r)]


def final_metrics(row: dict) -> dict[str, float]:
    """The finite metrics of a RESULT line, with the objective made from its terms."""
    out = {}
    for key, value in row.items():
        v = number(value) if key not in SETTINGS else None
        if v is not None and math.isfinite(v):
            out[key] = v
    if all(m in out for m in OBJECTIVE):
        out["objective"] = sum(w * out[m] for m, w in OBJECTIVE.items())
    return out


def points(runs: list[Run], pick: str = "min", average: bool = True) -> list[Point]:
    """One point per family, rung and token budget. Its best learning rate is
    chosen on its first runs (replicates only where an lr has none), by the
    objective (`pick` "min"), or at the vertex of a parabola in log lr through the
    best and its neighbours ("vertex"); its value is the mean of every run at that
    lr if `average` (the seeds run at the best lr), else of its first runs."""
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if "objective" in r.metrics:
            groups[(r.family, r.width, r.per_param)][r.lr].append(r)
    return [_best(by_lr, pick, average) for _, by_lr in sorted(groups.items())]


def firsts(runs: list[Run]) -> list[Run]:
    """At each family, rung, budget and lr, the runs that are not replicates, or
    the replicates where none is."""
    have = {(r.family, r.width, r.per_param, r.lr) for r in runs if not r.replicate}
    return [
        r
        for r in runs
        if not r.replicate or (r.family, r.width, r.per_param, r.lr) not in have
    ]


def _best(by_lr: dict[float | None, list[Run]], pick: str, average: bool) -> Point:
    lrs = sorted(by_lr, key=lambda x: x or 0.0)
    chosen = {lr: firsts(rs) for lr, rs in by_lr.items()}
    means = {lr: _mean([r.metrics for r in rs]) for lr, rs in chosen.items()}
    best = min(lrs, key=lambda lr: means[lr]["objective"])
    i = lrs.index(best)
    used = by_lr[best] if average else chosen[best]
    metrics = _mean([r.metrics for r in used])
    known = [lr for lr in lrs if lr is not None]
    if pick == "vertex" and 0 < i < len(lrs) - 1 and len(known) == len(lrs):
        metrics = _vertex(known[i - 1 : i + 2], means | {best: metrics})
    first = used[0]
    return Point(
        family=first.family,
        stem=first.stem,
        arms=first.arms,
        width=first.width,
        per_param=first.per_param,
        tokens=float(np.median([r.tokens for rs in by_lr.values() for r in rs])),
        metrics=metrics,
        lr=best,
        lrs=tuple(known),
        edge=len(lrs) > 1 and i in (0, len(lrs) - 1),
        replicates=len(used),
        by_seed=_by_seed(used),
    )


def _by_seed(runs: list[Run]) -> dict[int, dict[str, float]]:
    """The runs' metrics by seed, a seed's runs averaged."""
    seeds: dict[int, list[dict]] = defaultdict(list)
    for r in runs:
        seeds[r.seed].append(r.metrics)
    return {s: _mean(ms) for s, ms in sorted(seeds.items())}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(r) for r in rows))
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _vertex(lrs: list[float], means: dict) -> dict[str, float]:
    """Each metric's parabola in log lr through three learning rates, at the vertex
    of the objective's."""
    x = np.log(lrs)
    c2, c1, _ = np.polyfit(x, [means[lr]["objective"] for lr in lrs], 2)
    at = -c1 / (2 * c2) if c2 > 0 else x[1]
    keys = set.intersection(*(set(means[lr]) for lr in lrs))
    return {
        k: float(np.polyval(np.polyfit(x, [means[lr][k] for lr in lrs], 2), at))
        for k in keys
    }


DIVERGED = 1.05  # a run this much worse than its point's best run diverged


def diverged_seeds(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    """The runs kept, and those more than `DIVERGED` times their point's best run
    (same family, rung, budget and lr) on the objective: a seed that diverged."""
    best: dict[tuple, float] = {}
    for r in runs:
        if "objective" in r.metrics:
            k = (r.family, r.width, r.per_param, r.lr)
            best[k] = min(best.get(k, math.inf), r.metrics["objective"])

    def bad(r: Run) -> bool:
        k = (r.family, r.width, r.per_param, r.lr)
        return "objective" in r.metrics and r.metrics["objective"] > DIVERGED * best[k]

    return [r for r in runs if not bad(r)], [r for r in runs if bad(r)]


def noise(runs: list[Run], metric: str = "objective") -> tuple[float, int] | None:
    """The relative noise of `metric` between runs at the same point and learning
    rate, seeds' blocks left in, and its degrees of freedom."""
    block = seed_noise(runs, metric)
    return (
        (math.hypot(block["sigma"], block["seed_sigma"]), block["dof"])
        if block
        else None
    )


def seed_noise(runs: list[Run], metric: str = "objective") -> dict | None:
    """The noise of `metric` between runs at one point and lr, seeds as blocks: a
    seed shifts every family's run alike (its ``effects``, log, mean 0), the rest is
    the run's own (``sigma``). ``seed_sigma`` is the spread of the seeds' shifts,
    their estimates' own noise taken out."""
    groups: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if metric in r.metrics:
            groups[(r.family, r.width, r.per_param, r.lr)][r.seed].append(
                math.log(r.metrics[metric])
            )
    cells = [
        {s: float(np.mean(v)) for s, v in g.items()}
        for g in groups.values()
        if len(g) > 1
    ]
    if not cells:
        return None
    effects, means = _blocks(cells)
    resid = [
        y - means[i] - effects[s] for i, c in enumerate(cells) for s, y in c.items()
    ]
    dof = len(resid) - len(cells) - (len(effects) - 1)
    if dof < 1:
        return None
    sigma = math.sqrt(sum(r * r for r in resid) / dof)
    per_seed = np.mean([sum(s in c for c in cells) for s in effects])
    spread = float(np.var(list(effects.values()), ddof=1)) if len(effects) > 1 else 0.0
    seed_sigma = math.sqrt(max(spread - sigma**2 / per_seed, 0.0))
    return {"sigma": sigma, "seed_sigma": seed_sigma, "effects": effects, "dof": dof}


def seed_adjusted(runs: list[Run], blocks: dict[str, dict | None]) -> list[Run]:
    """`runs` on seed 0's scale: each metric divided by its seed's shift against
    seed 0's (`blocks`, by metric, `seed_noise`'s), so a point averaged over seeds
    0-2 and one of seed 0 alone are alike."""
    out = []
    for r in runs:
        metrics = dict(r.metrics)
        for m, b in blocks.items():
            e = (b or {}).get("effects", {})
            if m in metrics and r.seed in e and 0 in e:
                metrics[m] *= math.exp(-(e[r.seed] - e[0]))
        if all(k in metrics for k in OBJECTIVE):
            metrics["objective"] = sum(w * metrics[k] for k, w in OBJECTIVE.items())
        out.append(replace(r, metrics=metrics))
    return out


def _blocks(cells: list[dict[int, float]]) -> tuple[dict[int, float], list[float]]:
    """Seed shifts and cell means of an additive model, fit by alternation."""
    seeds = sorted({s for c in cells for s in c})
    effects = dict.fromkeys(seeds, 0.0)
    means = [0.0] * len(cells)
    for _ in range(50):
        means = [float(np.mean([y - effects[s] for s, y in c.items()])) for c in cells]
        effects = {
            s: float(np.mean([c[s] - means[i] for i, c in enumerate(cells) if s in c]))
            for s in seeds
        }
        shift = float(np.mean(list(effects.values())))
        effects = {s: e - shift for s, e in effects.items()}
    return effects, means


def metrics_of(ps: list[Point], root: str = ROOT) -> list[str]:
    """The metrics to fit: the four, and the repository's by position, as far as the
    root family's points (or, with none, any point) have them."""
    mine = [set(p.metrics) for p in ps if p.family == root] or [
        set(p.metrics) for p in ps
    ]
    have = set.intersection(*mine) if mine else set()
    extra = sorted(k for k in have if re.fullmatch(r"repo_bpb_\w+", k))
    return [m for m in METRICS if m in have] + extra


def monotone(ps: list[Point], slack: float = 0.005) -> tuple[list[Point], list[Point]]:
    """The points kept, and those dropped as diverged: a point whose objective is
    over `slack` above that of the same family at a smaller rung with as many tokens
    per parameter, or at the same rung on fewer tokens."""
    dropped = {id(p) for p in ps if _worse(p, ps, slack)}
    return [p for p in ps if id(p) not in dropped], [p for p in ps if id(p) in dropped]


def _worse(p: Point, ps: list[Point], slack: float) -> bool:
    smaller = [
        q.metrics["objective"]
        for q in ps
        if q.family == p.family
        and (
            (q.per_param == p.per_param and q.width < p.width)
            or (q.width == p.width and q.per_param < p.per_param)
        )
    ]
    return bool(smaller) and p.metrics["objective"] > (1 + slack) * min(smaller)


NEAR = 1.03  # the trials a learning rate's parabola fits: within 3% of the best


def lr_vertices(runs: list[Run]) -> list[dict]:
    """Each family's best learning rate at each rung it scanned: the vertex of a
    parabola in log lr through its trials within `NEAR` of the best (or through the
    best and its two neighbours), and whether the best is at the scan's edge."""
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for r in firsts(runs):
        if r.per_param == LADDER and r.lr and "objective" in r.metrics:
            groups[(r.family, r.width)][r.lr].append(r.metrics["objective"])
    return [
        _vertex_row(f, w, by_lr)
        for (f, w), by_lr in sorted(groups.items())
        if len(by_lr) > 1
    ]


def _vertex_row(family: str, width: int, by_lr: dict) -> dict:
    lrs = sorted(by_lr)
    x, y = np.log(lrs), np.array([np.mean(by_lr[lr]) for lr in lrs])
    i = int(np.argmin(y))
    near = y <= NEAR * y.min()
    fit = (
        near
        if near.sum() >= 3
        else (np.abs(np.arange(len(lrs)) - i) <= 1 if 0 < i < len(lrs) - 1 else None)
    )
    vertex = lrs[i]
    if fit is not None:
        c2, c1, _ = np.polyfit(x[fit], y[fit], 2)
        vertex = math.exp(-c1 / (2 * c2)) if c2 > 0 else vertex
    return {"family": family, "rung": width, "trials": len(lrs), "best": lrs[i], "vertex": vertex,
            "edge": i in (0, len(lrs) - 1), "lrs": " ".join(f"{lr:g}" for lr in lrs),
            "above": 100 * (y[i + 1] / y[i] - 1) if i + 1 < len(lrs) else math.nan}  # fmt: skip
