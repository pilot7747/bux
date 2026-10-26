"""The tree-structured Parzen estimator (TPE), as Optuna's ``TPESampler`` has it.

The trials so far are split into the best of them, a share ``gamma``, and the rest,
and a Parzen estimator is fit to the points of each: l(x) to the best's, g(x) to the
rest's. Of ``candidates`` points drawn from l, the one of greatest l(x) / g(x) is
tried, which is the one of most expected improvement (Bergstra et al., 2011,
"Algorithms for Hyper-Parameter Optimization").

An estimator is a mixture of a kernel at each point and one wide kernel, the prior,
as Optuna's defaults build it; Watanabe (2023), "Tree-Structured Parzen Estimator:
Understanding Its Algorithm Components and Their Roles for Better Empirical
Performance" (arXiv:2304.11127), weighs each of their parts. In a number's
coordinate a kernel is a Gaussian truncated to [0, 1], as wide as the farther of its
point's neighbours but no narrower than 1 / min(100, n + 2), the "magic clip"; the
prior's is at the middle, as wide as the whole. In a choice's, a kernel puts most of
its mass on its point's option. With ``multivariate`` (Falkner et al., 2018, "BOHB")
a kernel is a product over the dimensions, which sees how they interact; else each
dimension is a TPE of its own.

Trials still training count among the rest, at their points: Optuna's constant liar,
so that trials at once differ. Failed trials count among the rest too.
"""

import math
from typing import NamedTuple

import numpy as np
from jaxtyping import Float
from scipy.special import logsumexp
from scipy.stats import truncnorm

from bux.search.methods import History, Point, generator, methods, quasi_random
from bux.search.space import Space
from bux.train import config as tcfg

#: Optuna's 25: the most trials that are the best, and the newest trials that an
#: estimator weighs in full.
NEWEST = 25


class Parzen(NamedTuple):
    """A Parzen estimator over a space's coordinates: a mixture of kernels, one at
    each point it was fit to and the prior last, each a product over the dimensions."""

    weights: Float[np.ndarray, " k"]  #: Each kernel's weight; they sum to 1.
    mu: Float[np.ndarray, "k p"]  #: Each kernel's center in each number's coordinate.
    sigma: Float[
        np.ndarray, "k p"
    ]  #: Its width there, before the truncation to [0, 1].
    #: Each kernel's probability of each option, a matrix for each choice.
    probs: tuple[Float[np.ndarray, "k _"], ...]


def _numbers(sizes: tuple[int, ...]) -> list[int]:
    """Return the columns of the numbers among dimensions of `sizes` (`Space.sizes`)."""
    return [i for i, k in enumerate(sizes) if not k]


def _choices(sizes: tuple[int, ...]) -> list[int]:
    """Return the columns of the choices among dimensions of `sizes`."""
    return [i for i, k in enumerate(sizes) if k]


def split(
    history: History, gamma: float
) -> tuple[Float[np.ndarray, "b d"], Float[np.ndarray, "a d"]]:
    """Return the points of the best trials of `history` and those of the rest.

    Args:
        history: The search so far.
        gamma: The share of the trials with an objective that are the best.

    Returns:
        The points of the ceil(gamma * n) best of the n trials with an objective, at
        most 25 and only those whose objective is a number; and the others', the
        failed among them, then the points of the trials still training.
    """
    values = np.where(np.isfinite(history.values), history.values, np.inf)
    finite = int(np.isfinite(values).sum())
    n_below = min(math.ceil(gamma * len(values)), NEWEST, finite)
    best = np.zeros(len(values), dtype=bool)
    best[np.argsort(values, kind="stable")[:n_below]] = True
    return history.points[best], np.concatenate(
        [history.points[~best], history.pending]
    )


def _weights(n: int) -> Float[np.ndarray, " n"]:
    """Return the weights of an estimator's `n` points, oldest first, as Optuna's
    ``default_weights``: 1 for the newest 25, and up a ramp from 1 / n for the older."""
    ramp = np.linspace(1 / n, 1, n - NEWEST) if n > NEWEST else np.ones(0)
    return np.concatenate([ramp, np.ones(min(n, NEWEST))])


def _widths(mu: Float[np.ndarray, "n p"]) -> Float[np.ndarray, "n p"]:
    """Return each point's kernel's width in each coordinate: the farther of its
    neighbours, the bounds 0 and 1 among them but for the outermost points, clipped to
    [1 / min(100, n + 2), 1]."""
    n, p = mu.shape
    order = np.argsort(mu, axis=0)
    edges = np.vstack([np.zeros(p), np.take_along_axis(mu, order, 0), np.ones(p)])
    gaps = np.diff(edges, axis=0)
    sigma = np.maximum(gaps[:-1], gaps[1:])
    if n >= 2:  # the outermost take their inner neighbour's gap, not the bound's
        sigma[0], sigma[-1] = gaps[1], gaps[-2]
    widths = np.empty_like(sigma)
    np.put_along_axis(widths, order, sigma, axis=0)
    return np.clip(widths, 1 / min(100, n + 2), 1.0)


def _probs(options: Float[np.ndarray, " n"], k: int) -> Float[np.ndarray, "n+1 _"]:
    """Return each kernel's probability of each of a choice's `k` options: every
    kernel puts the prior's weight over the n + 1 kernels on each option, and a
    point's 1 more on its own, normalized; so the prior's is uniform."""
    n = len(options)
    probs = np.full((n + 1, k), 1 / (n + 1))
    probs[np.arange(n), np.rint(options).astype(int)] += 1
    return probs / probs.sum(axis=1, keepdims=True)


def fit(points: Float[np.ndarray, "n d"], sizes: tuple[int, ...]) -> Parzen:
    """Return the Parzen estimator of `points` in a space of `sizes` (`Space.sizes`),
    a kernel at each and the prior, which weighs as much as the newest point."""
    numbers = _numbers(sizes)
    mu = np.clip(points[:, numbers], 0.0, 1.0)
    weights = np.append(_weights(len(points)), 1.0)
    return Parzen(
        weights=weights / weights.sum(),
        mu=np.vstack([mu, np.full(len(numbers), 0.5)]),
        sigma=np.vstack([_widths(mu), np.ones(len(numbers))]),
        probs=tuple(_probs(points[:, i], sizes[i]) for i in _choices(sizes)),
    )


def sample(
    f: Parzen, sizes: tuple[int, ...], n: int, rng: np.random.Generator
) -> Float[np.ndarray, "n d"]:
    """Return `n` points drawn from `f`, each from a kernel drawn by its weight."""
    kernels = rng.choice(len(f.weights), size=n, p=f.weights)
    mu, sigma = f.mu[kernels], f.sigma[kernels]
    x = np.empty((n, len(sizes)))
    q = rng.random(mu.shape)
    u = truncnorm.ppf(q, -mu / sigma, (1 - mu) / sigma, mu, sigma)
    x[:, _numbers(sizes)] = np.clip(u, 0.0, 1.0)
    for i, probs in zip(_choices(sizes), f.probs, strict=True):
        cdf = np.cumsum(probs[kernels], axis=1)[:, :-1]
        x[:, i] = (cdf < rng.random((n, 1))).sum(axis=1)
    return x


def log_pdf(
    f: Parzen, x: Float[np.ndarray, "n d"], sizes: tuple[int, ...]
) -> Float[np.ndarray, " n"]:
    """Return the log of `f`'s density at each of the points `x`."""
    a, b = -f.mu / f.sigma, (1 - f.mu) / f.sigma
    numbers = x[:, None, _numbers(sizes)]
    log_k = truncnorm.logpdf(numbers, a, b, f.mu, f.sigma).sum(axis=-1)
    for i, probs in zip(_choices(sizes), f.probs, strict=True):
        log_k += np.log(probs[:, x[:, i].astype(int)]).T
    return logsumexp(log_k + np.log(f.weights), axis=1)


def best_of(
    below: Float[np.ndarray, "b d"],
    above: Float[np.ndarray, "a d"],
    sizes: tuple[int, ...],
    candidates: int,
    rng: np.random.Generator,
) -> Point:
    """Return the point of most log l(x) - log g(x), l the estimator of the best
    trials' points `below` and g the rest's, `above`, of `candidates` drawn from l."""
    good, bad = fit(below, sizes), fit(above, sizes)
    x = sample(good, sizes, candidates, rng)
    return x[np.argmax(log_pdf(good, x, sizes) - log_pdf(bad, x, sizes))]


@methods.register("tpe")
def _tpe(c: tcfg.TPESearch, space: Space, history: History) -> Point:
    """Of `candidates` points drawn from a density fit to the best trials, the one
    most likely under it relative to a density fit to the rest: the tree-structured
    Parzen estimator, as Optuna's (`bux.search.tpe`). The first `startup` points are
    quasi-random, as are any before a trial has an objective that is a number."""
    if history.proposed < c.startup or not np.isfinite(history.values).any():
        return quasi_random(space, history.seed, history.proposed)
    below, above = split(history, c.gamma)
    rng, sizes = generator(history), space.sizes
    if c.multivariate:
        return best_of(below, above, sizes, c.candidates, rng)
    return np.array(
        [
            best_of(below[:, [i]], above[:, [i]], (k,), c.candidates, rng)[0]
            for i, k in enumerate(sizes)
        ]
    )
