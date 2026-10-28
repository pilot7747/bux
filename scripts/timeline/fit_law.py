"""Scaling laws L(N, D) = E + A N^-alpha + B D^-beta of families of points, fit by the
Huber loss of log L (Hoffmann et al., 2022).

The root family has every term of its own. Every other family has its parent's
terms but those its points can tell apart: A always, alpha with three rungs or
more, B and beta with a token scan (``share`` takes some of them from the parent
whatever the points). E is the data's, shared by every family. All families are fit
at once, so a term many families share rests on all of their points; the staged
fit (the root alone, then each family with its parent held) is its start.
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
        return _root(rest / self.A, self.alpha) if rest > 0 else math.inf

    def d_for(self, loss: float, n: float) -> float:
        """The tokens on which `n` parameters reach `loss`."""
        rest = loss - self.asymptote(n)
        if self.B <= 0:
            return 0.0 if rest >= 0 else math.inf
        return _root(rest / self.B, self.beta) if rest > 0 else math.inf


def _root(x: float, power: float) -> float:
    """x^(-1/power), inf where it overflows."""
    log = -math.log(x) / power
    return math.exp(log) if log < 700 else math.inf


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
    share: tuple[str, ...] = ()  # terms it takes from its parent whatever its points
    extra: tuple[str, ...] = ()  # terms of its own its children's points tell apart
    count: np.ndarray | None = None  # the runs each point averages (1 if None)
    seeds: list | None = None  # each point's seeds (seed 0 alone if None)
    free: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        root = self.parent == NO_DATA
        own = set(own_terms(self.width, self.per_param, root)) | set(self.extra)
        self.free = tuple(
            t for t in SHAPES if t in own and (root or t not in self.share)
        )

    @property
    def ladder(self) -> np.ndarray:
        return self.per_param == 20

    def without(self, i: int) -> "Family":
        """This family without its i-th point."""
        keep = np.arange(len(self.y)) != i
        return self._take(keep, self.y[keep])

    def noisy(
        self, rng: np.random.Generator, sigma: float, shifts: dict | None = None
    ) -> "Family":
        """This family with each point's loss off by its runs' own noise, `sigma`
        over the root of the runs it averages, and by the mean of its seeds'
        `shifts` (log), which every family's runs of a seed share."""
        count = np.ones(len(self.y)) if self.count is None else self.count
        seeds = self.seeds or [(0,)] * len(self.y)
        block = np.array(
            [np.mean([(shifts or {}).get(s, 0.0) for s in ss]) for ss in seeds]
        )
        y = self.y * np.exp(
            block + sigma / np.sqrt(count) * rng.standard_normal(len(self.y))
        )
        return self._take(np.ones(len(self.y), bool), y)

    def _take(self, keep: np.ndarray, y: np.ndarray) -> "Family":
        n, d, w, t = (v[keep] for v in (self.n, self.d, self.width, self.per_param))
        count = None if self.count is None else self.count[keep]
        seeds = (
            None
            if self.seeds is None
            else [s for s, k in zip(self.seeds, keep, strict=True) if k]
        )
        return Family(n, d, y, w, t, self.parent, self.share, self.extra, count, seeds)


def own_terms(width: np.ndarray, per_param: np.ndarray, root: bool) -> tuple[str, ...]:
    """The terms a family's points tell apart from its parent's."""
    rungs = len(set(width[per_param == 20]))
    scan = any(len(set(per_param[width == w])) > 1 for w in set(width))
    names = {"A"} | ({"alpha"} if rungs >= 3 or root else set())
    names |= {"B", "beta"} if scan else set()
    return tuple(s for s in SHAPES if s in names)


def fit_metric(
    fams: dict[str, Family],
    root: str,
    joint: bool = True,
    start: dict | None = None,
    e: float | None = None,
) -> dict[str, Law]:
    """Every family's law. From `start`, or from the staged fits and a grid of
    the root's: jointly, E (unless `e` holds it), the root's terms and every
    family's own at once (the terms a family shares then rest on every family that
    shares them); else staged."""
    order = topological(fams, root)
    if not joint:
        return staged(fams, order, start, e)
    starts = [start] if start else [staged(fams, order, None, e)]
    if not start:
        grid = root_starts(fams, [root], e)
        starts += [spread(fams, order, s[root]) for s in grid]
    return fit_group(fams, order, {NO_DATA: FLAT}, e, starts)


def staged(
    fams: dict[str, Family],
    order: list[str],
    start: dict | None = None,
    e: float | None = None,
) -> dict:
    """The root's law alone, then each family's with its parent's and E held."""
    root = order[0]
    starts = [start] if start else root_starts(fams, [root], e)
    laws = fit_group(fams, [root], {NO_DATA: FLAT}, e, starts)
    for f in order[1:]:
        laws[f] = fit_one(fams, f, laws, start)
    return laws


def spread(fams: dict[str, Family], order: list[str], law: Law) -> dict[str, Law]:
    """A start for every family from one law: its terms, each family's A what its
    points need beside them."""
    out = {}
    for f in order:
        fam = fams[f]
        rest = (fam.y - law.E - law.B * fam.d**-law.beta) * fam.n**law.alpha
        out[f] = replace(
            law, A=float(np.median(rest)) if np.median(rest) > 0 else law.A
        )
    return out


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
    with_e = e is None
    src, owners = _sources(fams, group, fixed, with_e)

    def unpack(x: np.ndarray) -> dict[str, Law]:
        E = math.exp(x[0]) if e is None else e
        return {
            f: Law(E, **{t: _term(t, c, x) for t, c in src[f].items()}) for f in group
        }

    def residuals(x: np.ndarray) -> np.ndarray:
        laws = unpack(x)
        return np.concatenate(
            [
                np.log(laws[f].loss(fams[f].n, fams[f].d)) - np.log(fams[f].y)
                for f in group
            ]
        )

    def jacobian(x: np.ndarray) -> np.ndarray:
        laws = unpack(x)
        return np.concatenate(
            [_rows(fams[f], laws[f], src[f], len(x), with_e) for f in group]
        )

    floor = min(float(fams[f].y.min()) for f in group)
    lo, hi = _bounds(owners, with_e, floor)
    sols = [
        least_squares(
            residuals, x0, jac=jacobian, loss="huber", f_scale=DELTA, bounds=(lo, hi)
        )
        for x0 in (
            np.clip(_encode(owners, s, with_e, fams, group), lo + 1e-9, hi - 1e-9)
            for s in starts
        )
    ]
    return unpack(min(sols, key=lambda sol: sol.cost).x)


def _rows(fam: Family, law: Law, src: dict, cols: int, with_e: bool) -> np.ndarray:
    """The Jacobian's rows of a family's residuals, log L - log y, in the fit's
    columns: logs of E, A and B, and the exponents."""
    a, b = law.A * fam.n**-law.alpha, law.B * fam.d**-law.beta
    loss = law.E + a + b
    parts = {"A": a, "alpha": -a * np.log(fam.n), "B": b, "beta": -b * np.log(fam.d)}
    rows = np.zeros((len(fam.y), cols))
    if with_e:
        rows[:, 0] = law.E / loss
    for t, c in src.items():
        if isinstance(c, int):
            rows[:, c] = parts[t] / loss
    return rows


def _sources(
    fams: dict, group: list[str], fixed: dict[str, Law], with_e: bool
) -> tuple:
    """For each family and term, the column of x that holds it or the held law it
    comes from; and the family and term of each column."""
    src: dict[str, dict] = {}
    owners: list[tuple[str, str]] = []
    for f in group:
        parent = fams[f].parent
        src[f] = {}
        for t in SHAPES:
            if t in fams[f].free:
                src[f][t] = len(owners) + int(with_e)
                owners.append((f, t))
            else:
                src[f][t] = src[parent][t] if parent in src else fixed[parent]
    return src, owners


def _term(name: str, c: int | Law, x: np.ndarray) -> float:
    return _decode(name, x[c]) if isinstance(c, int) else getattr(c, name)


def _decode(name: str, v: float) -> float:
    return math.exp(v) if name in LOGGED else float(v)


def _encode(
    owners: list, start: dict[str, Law], with_e: bool, fams: dict, group: list[str]
) -> np.ndarray:
    first = start.get(group[0]) or next(iter(start.values()))
    x = [math.log(first.E)] if with_e else []
    for f, t in owners:
        law = start.get(f) or start.get(fams[f].parent) or first
        x.append(math.log(getattr(law, t)) if t in LOGGED else getattr(law, t))
    return np.array(x, float)


def _bounds(owners: list, with_e: bool, floor: float) -> tuple:
    lo = [math.log(1e-4)] if with_e else []
    hi = [math.log(floor * 0.999)] if with_e else []
    for _, t in owners:
        lo.append(-40.0 if t in LOGGED else 0.01)
        hi.append(40.0 if t in LOGGED else 3.0)
    return np.array(lo), np.array(hi)


def root_starts(
    fams: dict[str, Family], group: list[str], e: float | None = None
) -> list[dict[str, Law]]:
    """A grid of starts for the root's group: E a share of the lowest loss (or
    `e`), the exponents at a few values, A and B sharing the rest at the median
    point."""
    r = fams[group[0]]
    n, d, y = (float(np.median(v)) for v in (r.n, r.d, r.y))
    floor = float(min(fams[f].y.min() for f in group))
    starts = []
    for E in [share * floor for share in (0.3, 0.6, 0.85)] if e is None else [e]:
        for alpha in (0.15, 0.3, 0.5):
            for beta in (0.2, 0.4):
                law = Law(
                    E, 0.8 * (y - E) * n**alpha, alpha, 0.2 * (y - E) * d**beta, beta
                )
                starts.append({f: law for f in group})
    return starts


def leave_one_rung_out(
    fams: dict[str, Family],
    f: str,
    laws: dict[str, Law],
    root: str,
    joint: bool,
    e: float | None = None,
) -> list[Law]:
    """`f`'s laws fit without each of its rungs in turn (its other points kept), its
    parent's held; the root's with E refit."""
    out = []
    for i in np.flatnonzero(fams[f].ladder):
        fewer = fams[f].without(int(i))
        if len(set(fewer.width[fewer.ladder])) < (2 if f == root else 1):
            continue
        if f == root:
            out.append(fit_metric({**fams, f: fewer}, root, joint, laws, e)[f])
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
    e: float | None = None,
    seed_sigma: float = 0.0,
) -> list[tuple[dict[str, Family], dict[str, Law]]]:
    """`reps` refits of every family, each from the full fit, with each point off
    by its runs' noise (`sigma`) and by its seeds' shifts, drawn once per refit
    for every family (`seed_sigma`): seeds act alike across families."""
    rng = np.random.default_rng(seed)
    seeds = sorted(
        {s for fam in fams.values() for ss in (fam.seeds or [(0,)]) for s in ss}
    )
    out = []
    for _ in range(reps):
        shifts = {s: seed_sigma * rng.standard_normal() for s in seeds}
        noisy = {f: fam.noisy(rng, sigma, shifts) for f, fam in fams.items()}
        out.append((noisy, fit_metric(noisy, root, joint, laws, e)))
    return out
