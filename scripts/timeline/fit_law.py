"""Scaling laws L(N, D) = E + A N^-alpha + B D^-beta of families of points, fit by the
Huber loss of log L (Hoffmann et al., 2022).

The root family (the baseline) has every term of its own. Every other family has
its parent's terms but those its points can tell apart: A always, alpha with three
rungs or more, B and beta with a token scan. E is the data's, shared by every
family: it is fit jointly with the root and the families that have every term of
their own (the milestones' full ladders), and the rest are fit with it held.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

SHAPES = ("A", "alpha", "B", "beta")
LOGGED = {"A", "B"}  # fit as their logarithms
DELTA = 1e-3  # Huber's threshold on log L, as Hoffmann et al.'s
NO_DATA = ""  # the root's parent where it has no token scan: no D term


@dataclass(frozen=True)
class Law:
    """L(N, D) = E + A N^-alpha + B D^-beta."""

    E: float
    A: float
    alpha: float
    B: float
    beta: float

    def loss(self, n: float | np.ndarray, d: float | np.ndarray) -> np.ndarray:
        return (
            self.E
            + self.A * np.power(n, -self.alpha)
            + self.B * np.power(d, -self.beta)
        )

    def asymptote(self, n: float) -> float:
        """The loss at `n` parameters on unlimited data."""
        return self.E + self.A * n**-self.alpha

    def n_for(self, loss: float, d: float) -> float:
        """The parameters at which this law reaches `loss` on `d` tokens."""
        rest = loss - self.E - self.B * d**-self.beta
        return (rest / self.A) ** (-1 / self.alpha) if rest > 0 else math.inf

    def d_for(self, loss: float, n: float) -> float:
        """The tokens on which `n` parameters reach `loss`."""
        rest = loss - self.asymptote(n)
        if self.B <= 0:
            return 0.0 if rest >= 0 else math.inf
        return (rest / self.B) ** (-1 / self.beta) if rest > 0 else math.inf


FLAT = Law(0.0, 1.0, 0.3, 0.0, 1.0)  # NO_DATA's terms


@dataclass
class Family:
    """A family's points for one metric, its parent and the terms of its own."""

    n: np.ndarray
    d: np.ndarray
    y: np.ndarray
    width: np.ndarray
    per_param: np.ndarray
    parent: str  # NO_DATA for the root
    free: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.free = own_terms(self.width, self.per_param, self.parent == NO_DATA)

    @property
    def ladder(self) -> np.ndarray:
        return self.per_param == 20

    def without(self, i: int) -> "Family":
        """This family without its i-th point."""
        keep = np.arange(len(self.y)) != i
        return self._take(keep, self.y[keep])

    def noisy(self, rng: np.random.Generator, sigma: float) -> "Family":
        """This family with each point's loss off by a relative noise of `sigma`."""
        y = self.y * np.exp(sigma * rng.standard_normal(len(self.y)))
        return self._take(np.ones(len(self.y), bool), y)

    def _take(self, keep: np.ndarray, y: np.ndarray) -> "Family":
        n, d, w, t = (v[keep] for v in (self.n, self.d, self.width, self.per_param))
        return Family(n, d, y, w, t, self.parent)


def own_terms(width: np.ndarray, per_param: np.ndarray, root: bool) -> tuple[str, ...]:
    """The terms a family's points tell apart from its parent's."""
    rungs = len(set(width[per_param == 20]))
    scan = any(len(set(per_param[width == w])) > 1 for w in set(width))
    names = {"A"} | ({"alpha"} if rungs >= 3 or root else set())
    names |= {"B", "beta"} if scan else set()
    return tuple(s for s in SHAPES if s in names)


def fit_metric(
    fams: dict[str, Family], root: str, joint: bool = True, start: dict | None = None
) -> dict[str, Law]:
    """Every family's law: the root's group jointly, E with it, then the others
    in order, each with its parent's law and E held."""
    order = topological(fams, root)
    group = [root]
    for f in order[1:] if joint else ():
        if set(fams[f].free) == set(SHAPES) and fams[f].parent in group:
            group.append(f)
    starts = [start] if start else root_starts(fams, group)
    laws = fit_group(fams, group, {NO_DATA: FLAT}, None, starts)
    for f in order:
        if f not in laws:
            laws[f] = fit_one(fams, f, laws, start)
    return laws


def topological(fams: dict[str, Family], root: str) -> list[str]:
    """The families that descend from `root`, each after its parent."""
    order, frontier = [root], [root]
    while frontier:
        kids = sorted(f for f in fams if fams[f].parent in frontier and f not in order)
        order += kids
        frontier = kids
    return order


def fit_one(
    fams: dict[str, Family], f: str, laws: dict[str, Law], start: dict | None = None
) -> Law:
    """Family `f`'s law, its parent's and E held."""
    parent = laws[fams[f].parent]
    guesses = [start[f]] if start and f in start else [
        replace(parent, A=parent.A * s) for s in (1.0, 0.5, 2.0)
    ]  # fmt: skip
    starts = [{f: g} for g in guesses]
    return fit_group(fams, [f], {fams[f].parent: parent}, parent.E, starts)[f]


def fit_group(
    fams: dict[str, Family],
    group: list[str],
    fixed: dict[str, Law],
    e: float | None,
    starts: list[dict[str, Law]],
) -> dict[str, Law]:
    """The laws of `group` that fit its points best from any of `starts`: E too
    unless `e` holds it, each family's own terms, the rest from its parent."""
    floor = min(float(fams[f].y.min()) for f in group)

    def unpack(x: np.ndarray) -> dict[str, Law]:
        values = iter(x[1:] if e is None else x)
        laws: dict[str, Law] = {}
        E = math.exp(x[0]) if e is None else e
        for f in group:
            parent = laws.get(fams[f].parent) or fixed[fams[f].parent]
            own = {s: _decode(s, next(values)) for s in fams[f].free}
            terms = {s: own.get(s, getattr(parent, s)) for s in SHAPES}
            laws[f] = Law(E, **terms)
        return laws

    def residuals(x: np.ndarray) -> np.ndarray:
        laws = unpack(x)
        return np.concatenate(
            [
                np.log(laws[f].loss(fams[f].n, fams[f].d)) - np.log(fams[f].y)
                for f in group
            ]
        )

    lo, hi = _bounds(fams, group, e is None, floor)
    sols = [
        least_squares(residuals, x0, loss="huber", f_scale=DELTA, bounds=(lo, hi))
        for x0 in (
            np.clip(_encode(fams, group, s, e is None), lo + 1e-9, hi - 1e-9)
            for s in starts
        )
    ]
    return unpack(min(sols, key=lambda sol: sol.cost).x)


def _decode(name: str, v: float) -> float:
    return math.exp(v) if name in LOGGED else float(v)


def _encode(
    fams: dict, group: list[str], start: dict[str, Law], with_e: bool
) -> np.ndarray:
    first = start[group[0]]
    x = [math.log(first.E)] if with_e else []
    for f in group:
        law = start.get(f) or start.get(fams[f].parent) or first
        x += [
            math.log(getattr(law, s)) if s in LOGGED else getattr(law, s)
            for s in fams[f].free
        ]
    return np.array(x, float)


def _bounds(fams: dict, group: list[str], with_e: bool, floor: float) -> tuple:
    lo = [math.log(1e-4)] if with_e else []
    hi = [math.log(floor * 0.999)] if with_e else []
    for f in group:
        for s in fams[f].free:
            lo.append(-40.0 if s in LOGGED else 0.01)
            hi.append(40.0 if s in LOGGED else 3.0)
    return np.array(lo), np.array(hi)


def root_starts(fams: dict[str, Family], group: list[str]) -> list[dict[str, Law]]:
    """A grid of starts for the root's group: E a share of the lowest loss, the
    exponents at a few values, A and B sharing the rest at the median point."""
    r = fams[group[0]]
    n, d, y = (float(np.median(v)) for v in (r.n, r.d, r.y))
    floor = float(min(fams[f].y.min() for f in group))
    starts = []
    for share in (0.3, 0.6, 0.85):
        for alpha in (0.15, 0.3, 0.5):
            for beta in (0.2, 0.4):
                E = share * floor
                law = Law(
                    E, 0.8 * (y - E) * n**alpha, alpha, 0.2 * (y - E) * d**beta, beta
                )
                starts.append({f: law for f in group})
    return starts


def leave_one_rung_out(
    fams: dict[str, Family], f: str, laws: dict[str, Law], root: str, joint: bool
) -> list[Law]:
    """`f`'s laws fit without each of its rungs in turn (its other points kept), its
    parent's held; the root's with E refit."""
    out = []
    for i in np.flatnonzero(fams[f].ladder):
        fewer = fams[f].without(int(i))
        if len(set(fewer.width[fewer.ladder])) < (2 if f == root else 1):
            continue
        if f == root:
            out.append(fit_metric({**fams, f: fewer}, root, joint, laws)[f])
        else:
            out.append(fit_one({**fams, f: fewer}, f, laws))
    return out


def bootstrap(
    fams: dict[str, Family],
    laws: dict[str, Law],
    root: str,
    joint: bool,
    sigma: float,
    reps: int,
    seed: int = 0,
) -> list[tuple[dict[str, Family], dict[str, Law]]]:
    """`reps` refits of every family with each point's loss off by a relative
    noise of `sigma`, each from the full fit."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        noisy = {f: fam.noisy(rng, sigma) for f, fam in fams.items()}
        out.append((noisy, fit_metric(noisy, root, joint, laws)))
    return out
