"""Read the timeline's RESULT lines into points: one per family, rung and token
budget, at its best learning rate.

A RESULT line is ``RESULT {"dir": "positions[rope, d768]-<id>", "state": "done",
"lr": ..., "budget": ..., <final metrics>}``. The run's name gives its family (the
file's stem and its arm labels), its rung (a label ``d512``), its token budget (a
label ``t40``: 40 tokens per baseline parameter; none is the rung's 20) and the
settings a scan added (``optimizer.lr=2e-3``).
"""

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
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
SI = {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
#: Keys of a RESULT line that are not final metrics.
SETTINGS = {"dir", "state", "lr", "budget", "steps", "d_model", "step", "batch"}


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

    @property
    def n(self) -> float:
        """The parameters a token goes through besides the embeddings."""
        return self.counts.get("active", math.nan)

    @property
    def ladder(self) -> bool:
        return self.per_param == LADDER


def family_key(stem: str, arms: tuple[str, ...]) -> str:
    """``baseline``, ``positions/rope``: a file's stem and its run's arm labels."""
    return "/".join((stem, "+".join(arms))) if arms else stem


def number(value: Any) -> float | None:
    """A number from a RESULT field: 2e-3, 503000000 or "503M"."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    m = re.fullmatch(r"\s*([0-9.]+(?:e[+-]?\d+)?)\s*([kKMBGT]?)\s*", str(value))
    return float(m.group(1)) * SI[m.group(2)] if m else None


def parse_name(name: str) -> tuple[str, list[str]]:
    """The stem and the labels of a run's directory name."""
    m = NAME.match(name.strip())
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


def read(paths: list[Path], alias_base: bool = True) -> list[Run]:
    """Every finished run of the RESULT lines in `paths`."""
    runs = []
    for path in paths:
        for line in path.read_text().splitlines():
            if (at := line.find("RESULT {")) >= 0:
                row = json.JSONDecoder().raw_decode(line, at + len("RESULT "))[0]
                run = parse_row(row, alias_base)
                if run is not None:
                    runs.append(run)
    return runs


def parse_row(row: dict, alias_base: bool = True) -> Run | None:
    """A RESULT line's run, or None if it did not finish or has no rung."""
    if row.get("state") not in ("done", None):
        return None
    stem, labels = parse_name(row["dir"])
    width, per_param, settings, arms = classify(labels)
    width = width or int(number(row.get("d_model")) or 0)
    if not width:
        return None
    if alias_base and arms == ("base",):  # an axis file's base arm is the baseline
        stem, arms = ROOT, ()
    lr = number(row.get("lr")) or number(settings.get("optimizer.lr"))
    tokens = number(row.get("budget")) or RUNG_TOKENS.get(width, math.nan) * (
        per_param / LADDER
    )
    return Run(stem, arms, width, per_param, lr, tokens, final_metrics(row))


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


def points(runs: list[Run], pick: str = "min") -> list[Point]:
    """One point per family, rung and token budget: its runs at each learning rate
    averaged, then the best learning rate by the objective (`pick` "min"), or the
    vertex of a parabola in log lr through the best and its neighbours ("vertex")."""
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if "objective" in r.metrics:
            groups[(r.family, r.width, r.per_param)][r.lr].append(r)
    return [_best(by_lr, pick) for _, by_lr in sorted(groups.items())]


def _best(by_lr: dict[float | None, list[Run]], pick: str) -> Point:
    lrs = sorted(by_lr, key=lambda x: x or 0.0)
    means = {lr: _mean([r.metrics for r in rs]) for lr, rs in by_lr.items()}
    best = min(lrs, key=lambda lr: means[lr]["objective"])
    i = lrs.index(best)
    first = by_lr[best][0]
    metrics = means[best]
    if pick == "vertex" and 0 < i < len(lrs) - 1 and None not in lrs:
        metrics = _vertex([lrs[i - 1], best, lrs[i + 1]], means)
    return Point(
        family=first.family,
        stem=first.stem,
        arms=first.arms,
        width=first.width,
        per_param=first.per_param,
        tokens=float(np.median([r.tokens for rs in by_lr.values() for r in rs])),
        metrics=metrics,
        lr=best,
        lrs=tuple(lr for lr in lrs if lr is not None),
        edge=len(lrs) > 1 and i in (0, len(lrs) - 1),
        replicates=len(by_lr[best]),
    )


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


def noise(runs: list[Run], metric: str = "objective") -> tuple[float, int] | None:
    """The relative standard deviation of `metric` between runs at the same point
    and learning rate, pooled, and how many such pairs of runs there were."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in runs:
        if metric in r.metrics:
            groups[(r.family, r.width, r.per_param, r.lr)].append(r.metrics[metric])
    devs = [np.log(v) - np.mean(np.log(v)) for v in groups.values() if len(v) > 1]
    if not devs:
        return None
    dof = sum(len(d) - 1 for d in devs)
    return float(math.sqrt(sum(float(np.sum(d**2)) for d in devs) / dof)), dof


def metrics_of(ps: list[Point], root: str = ROOT) -> list[str]:
    """The metrics to fit: the four, and the repository's by position, as far as the
    root family's points have them."""
    have = set.intersection(*(set(p.metrics) for p in ps if p.family == root))
    extra = sorted(k for k in have if re.fullmatch(r"repo_bpb_\w+", k))
    return [m for m in METRICS if m in have] + extra
