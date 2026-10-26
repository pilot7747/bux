"""Bayesian optimization with a Gaussian process: `bux.train.config.GPSearch`'s method.

After its startup trials, a search fits a GP to the objectives it has and proposes
the point that maximizes an acquisition under it (Snoek et al., 2012, "Practical
Bayesian Optimization of Machine Learning Algorithms"), with the choices of BoTorch
since 0.12, Vizier and Optuna's ``GPSampler``:

- The kernel is a Matérn-5/2 with a length scale for each dimension, times the
  signal's variance, plus Gaussian noise. A number's distance is along its coordinate
  in [0, 1]; a choice's is 0 to the same option and 1 to another, as Optuna and
  Vizier compare categories.
- The objectives are standardized. A failed trial, not a number, is worse than the
  worst by half their range and a deviation, and all are centred again, Vizier's
  rule (Song et al., 2024, "The Vizier Gaussian Process Bandit Algorithm"): so the
  GP learns to keep away from where trials diverge.
- The hyperparameters are their posterior's mode, L-BFGS-B's best from the priors'
  mode and from a few of their samples. The length scales' priors are
  LogNormal(√2 + log(d) / 2, √3), from 0.025 (Hvarfner et al., 2024, "Vanilla
  Bayesian Optimization Performs Great in High Dimensions"), the noise's
  LogNormal(-4, 1), from 1e-4, as BoTorch's; the signal's is LogNormal(0, 1).
- The acquisition is the log of the expected improvement on the least of the GP's
  means at the trials, which one lucky trial does not set, computed so that it
  neither underflows nor flattens far from the best (Ament et al., 2023,
  "Unexpected Improvements to Expected Improvement for Bayesian Optimization"); or
  the lower confidence bound. Trials still training count as ended at the GP's mean
  for them, the kriging believer (Ginsbourger et al., 2010, "Kriging is well-suited
  to parallelize optimization"): which keeps the mean and takes the uncertainty
  there away, so that trials at once differ.
- It is maximized as Optuna maximizes it: at quasi-random points and around the best
  trials, then by local searches from the best of them, L-BFGS-B over the numbers
  and a sweep over each choice's options in turn.

It is numpy and scipy only: the process that proposes trials starts no JAX backend.
"""

import contextlib
import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.linalg import LinAlgError, cho_solve, cholesky
from scipy.optimize import minimize
from scipy.special import erfcx, log_ndtr, ndtr
from scipy.stats import qmc

from bux.search.methods import History, Point, generator, methods, quasi_random
from bux.search.space import Space
from bux.train import config as tcfg

SQRT2, SQRT3, SQRT5 = math.sqrt(2), math.sqrt(3), math.sqrt(5)
SQRT_2PI = math.sqrt(2 * math.pi)
FAR = 1e6  # past -FAR, log_h's tail is its leading term, as BoTorch's
#: Bounds of the hyperparameters, the variances in the standardized objectives' units:
#: each length scale's, from BoTorch's least, the signal's variance's, the noise's.
LENGTH, SIGNAL, NOISE = (0.025, 1e3), (1e-3, 1e2), (1e-4, 10.0)
RESTARTS = 4  # samples of the priors the fit starts from, besides their mode
JITTERS = (0.0, 1e-9, 1e-7, 1e-5, 1e-3)  # of the mean variance, until it factors
VARIANCE = 1e-12  # the least posterior variance, which a deviation divides by
CANDIDATES = 2048  # quasi-random points the acquisition is first evaluated at
ELITES, NEIGHBOURS = 5, 64  # the best trials, and the perturbations of each
STEPS = (0.01, 0.3)  # the range of a perturbation's scale, log-uniformly
STARTS = 10  # the best candidates a local search starts from
ROUNDS = 3  # L-BFGS-B's and the choices' sweeps a local search alternates, at most


@dataclass(frozen=True)
class GP:
    """A Gaussian process conditioned on labels at points.

    Attributes:
        x: The points, in coordinates (`bux.search.space`).
        y: Their labels.
        choice: Which coordinates are choices: their distance is 0 or 1.
        theta: The log hyperparameters: each length scale, the signal's variance
            and the noise's.
        chol: The lower Cholesky factor of the labels' covariance.
        alpha: The covariance's inverse times the labels.
    """

    x: Float[np.ndarray, "n d"]
    y: Float[np.ndarray, " n"]
    choice: Bool[np.ndarray, " d"]
    theta: Float[np.ndarray, " p"]
    chol: Float[np.ndarray, "n n"]
    alpha: Float[np.ndarray, " n"]


def labels(values: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
    """Return the objectives as the GP sees them: standardized, each one that is not a
    number worse than the worst by half their range and a deviation, all centred."""
    finite = np.isfinite(values)
    z = values[finite] - values[finite].mean()
    z /= z.std() if z.std() > 0 else 1.0
    y = np.full(values.shape, 1.5 * z.max() - 0.5 * z.min() + 1)
    y[finite] = z
    return y - y.mean()


def squared_distances(
    a: Float[np.ndarray, "n d"],
    b: Float[np.ndarray, "m d"],
    choice: Bool[np.ndarray, " d"],
) -> Float[np.ndarray, "n m d"]:
    """Return the squared distance from each of `a` to each of `b` in each coordinate:
    for a choice, 0 to the same option and 1 to another."""
    diff = a[:, None, :] - b[None, :, :]
    return np.where(choice, diff != 0, diff**2)


def matern(
    r2: Float[np.ndarray, "*s"],
) -> tuple[Float[np.ndarray, "*s"], Float[np.ndarray, "*s"]]:
    """Return the Matérn-5/2 kernel at the squared scaled distances `r2`, and its
    slope: minus twice its derivative in `r2`."""
    r = SQRT5 * np.sqrt(r2)
    e = np.exp(-r)
    return (1 + r + r**2 / 3) * e, 5 / 3 * (1 + r) * e


def _unpack(
    theta: Float[np.ndarray, " p"],
) -> tuple[Float[np.ndarray, " d"], float, float]:
    """Return the length scales, the signal's variance and the noise's of `theta`."""
    e = np.exp(theta)
    return e[:-2], float(e[-2]), float(e[-1])


def priors(dims: int) -> tuple[Float[np.ndarray, " p"], Float[np.ndarray, " p"]]:
    """Return the mean and deviation of each log hyperparameter's normal prior."""
    loc = np.r_[np.full(dims, SQRT2 + math.log(dims) / 2), 0.0, -4.0]
    return loc, np.r_[np.full(dims, SQRT3), 1.0, 1.0]


def _log_prior(theta: Float[np.ndarray, " p"]) -> tuple[float, Float[np.ndarray, " p"]]:
    """Return the log density of the hyperparameters, log-normal, at exp(`theta`), up to
    a constant, and its gradient in `theta`."""
    loc, scale = priors(len(theta) - 2)
    u = (theta - loc) / scale
    return float(-(u**2 / 2 + theta).sum()), -u / scale - 1


def _cholesky(k: Float[np.ndarray, "n n"]) -> Float[np.ndarray, "n n"]:
    """Return the lower Cholesky factor of the covariance `k`, with as little jitter on
    its diagonal as rounding needs, as duplicate points do."""
    eye = np.eye(len(k)) * np.diag(k).mean()
    for jitter in JITTERS[:-1]:
        with contextlib.suppress(LinAlgError):
            return cholesky(k + jitter * eye, lower=True)
    return cholesky(k + JITTERS[-1] * eye, lower=True)


def _loss(
    theta: Float[np.ndarray, " p"],
    d2: Float[np.ndarray, "n n d"],
    y: Float[np.ndarray, " n"],
) -> tuple[float, Float[np.ndarray, " p"]]:
    """Return minus the log posterior density of the hyperparameters `theta` given the
    labels `y` at squared distances `d2`, up to a constant, and its gradient."""
    lengths, signal, noise = _unpack(theta)
    k, slope = matern(d2 @ lengths**-2)
    chol = _cholesky(signal * k + noise * np.eye(len(y)))
    alpha = cho_solve((chol, True), y)
    w = np.outer(alpha, alpha) - cho_solve((chol, True), np.eye(len(y)))
    grad = np.r_[
        np.einsum("ab,abi->i", w * signal * slope, d2) / lengths**2,
        (w * signal * k).sum(),
        noise * np.trace(w),
    ]
    prior, dprior = _log_prior(theta)
    fit = y @ alpha / 2 + np.log(np.diag(chol)).sum()
    return float(fit - prior), -grad / 2 - dprior


def condition(
    x: Float[np.ndarray, "n d"],
    y: Float[np.ndarray, " n"],
    choice: Bool[np.ndarray, " d"],
    theta: Float[np.ndarray, " p"],
) -> GP:
    """Return the GP of hyperparameters `theta` conditioned on labels `y` at `x`."""
    lengths, signal, noise = _unpack(theta)
    k, _ = matern(squared_distances(x, x, choice) @ lengths**-2)
    chol = _cholesky(signal * k + noise * np.eye(len(y)))
    return GP(x, y, choice, theta, chol, cho_solve((chol, True), y))


def fit(
    x: Float[np.ndarray, "n d"],
    y: Float[np.ndarray, " n"],
    choice: Bool[np.ndarray, " d"],
    rng: np.random.Generator,
) -> GP:
    """Return the GP of labels `y` at `x` whose hyperparameters are their posterior's
    mode: the best that L-BFGS-B finds from the priors' mode and `RESTARTS` samples."""
    loc, scale = priors(x.shape[1])
    bounds = np.log([LENGTH] * x.shape[1] + [SIGNAL, NOISE])
    starts = np.vstack([loc - scale**2, rng.normal(loc, scale, (RESTARTS, len(loc)))])
    d2 = squared_distances(x, x, choice)
    runs = [
        minimize(_loss, s, (d2, y), "L-BFGS-B", jac=True, bounds=bounds)
        for s in np.clip(starts, bounds[:, 0], bounds[:, 1])
    ]
    best = min(runs, key=lambda r: r.fun if np.isfinite(r.fun) else np.inf)
    return condition(x, y, choice, best.x)


def _cross(
    gp: GP, x: Float[np.ndarray, "m d"]
) -> tuple[Float[np.ndarray, "m n"], Float[np.ndarray, "m n d"]]:
    """Return the GP's kernel from each of `x` to each of its points, and its gradient
    in `x`'s numbers."""
    lengths, signal, _ = _unpack(gp.theta)
    k, slope = matern(squared_distances(x, gp.x, gp.choice) @ lengths**-2)
    diff = (x[:, None, :] - gp.x[None, :, :]) * ~gp.choice
    return signal * k, -signal * slope[..., None] * diff / lengths**2


def posterior(
    gp: GP, x: Float[np.ndarray, "m d"]
) -> tuple[
    Float[np.ndarray, " m"],
    Float[np.ndarray, " m"],
    Float[np.ndarray, "m d"],
    Float[np.ndarray, "m d"],
]:
    """Return the GP's mean and deviation of the objective, without the noise, at each
    of `x`, and their gradients in `x`'s numbers."""
    k, dk = _cross(gp, x)
    w = cho_solve((gp.chol, True), k.T)
    variance = _unpack(gp.theta)[1] - np.einsum("mn,nm->m", k, w)
    std = np.sqrt(np.maximum(variance, VARIANCE))
    dvariance = -2 * np.einsum("mnd,nm->md", dk, w)
    dstd = np.where((variance > VARIANCE)[:, None], dvariance / (2 * std[:, None]), 0.0)
    return k @ gp.alpha, std, np.einsum("mnd,n->md", dk, gp.alpha), dstd


def believe(gp: GP, pending: Float[np.ndarray, "m d"]) -> GP:
    """Return `gp` conditioned on the trials still training at `pending` as if they had
    ended at its mean for them, the kriging believer: its mean stays as it is, and its
    uncertainty there goes."""
    if not len(pending):
        return gp
    mean = posterior(gp, pending)[0]
    return condition(np.vstack([gp.x, pending]), np.r_[gp.y, mean], gp.choice, gp.theta)


def _log1mexp(a: Float[np.ndarray, " m"]) -> Float[np.ndarray, " m"]:
    """Return log(1 - exp(`a`)) for `a` < 0, accurate near 0 and far from it."""
    return np.where(a > -math.log(2), np.log(-np.expm1(a)), np.log1p(-np.exp(a)))


def log_h(z: Float[np.ndarray, " m"]) -> Float[np.ndarray, " m"]:
    """Return log(φ(z) + z Φ(z)), the log of a standard normal's expected improvement
    past -`z`, finite and increasing for any `z`, as Ament et al.'s (2023) equation 9
    and BoTorch's ``_log_ei_helper`` compute it."""
    high, low = np.maximum(z, -1.0), np.clip(z, -FAR, -1.0)
    direct = np.log(np.exp(-(high**2) / 2) / SQRT_2PI + high * ndtr(high))
    tail = _log1mexp(np.log(erfcx(-low / SQRT2) * -low) + math.log(math.pi / 2) / 2)
    far = -2 * np.log(-np.minimum(z, -1.0))
    log_pdf = -(z**2) / 2 - math.log(SQRT_2PI)
    return np.where(z > -1, direct, log_pdf + np.where(z > -FAR, tail, far))


def _acquire(
    c: tcfg.GPSearch,
    mean: Float[np.ndarray, " m"],
    std: Float[np.ndarray, " m"],
    best: float,
) -> tuple[Float[np.ndarray, " m"], Float[np.ndarray, " m"], Float[np.ndarray, " m"]]:
    """Return `c`'s acquisition at the posterior's `mean` and `std`, and its derivatives
    in them: log EI on `best`, or minus the lower confidence bound."""
    if c.acquisition == "ucb":
        return c.beta * std - mean, np.full_like(mean, -1.0), np.full_like(std, c.beta)
    z = (best - mean) / std
    log_ei = log_h(z)
    slope = np.exp(log_ndtr(z) - log_ei)  # d log_h / dz
    return log_ei + np.log(std), -slope / std, (1 - slope * z) / std


@dataclass(frozen=True)
class Acquisition:
    """What a proposal maximizes: `c`'s acquisition under `gp`, on `best`.

    Attributes:
        c: The search's method, which says the acquisition.
        gp: The GP of the trials, those still training believed.
        best: The label to improve on: the least of the GP's means at its points.
    """

    c: tcfg.GPSearch
    gp: GP
    best: float

    def __call__(
        self, x: Float[np.ndarray, "m d"]
    ) -> tuple[Float[np.ndarray, " m"], Float[np.ndarray, "m d"]]:
        """Return the acquisition at each of `x`, and its gradient in their numbers."""
        mean, std, dmean, dstd = posterior(self.gp, x)
        a, da_mean, da_std = _acquire(self.c, mean, std, self.best)
        return a, da_mean[:, None] * dmean + da_std[:, None] * dstd


def perturb(
    x: Float[np.ndarray, "n d"], sizes: Int[np.ndarray, " d"], rng: np.random.Generator
) -> Float[np.ndarray, "n d"]:
    """Return each of `x` moved at random: its numbers by a normal step, of a scale
    log-uniform in `STEPS`, and each choice, with chance 1 / d, to any option."""
    numeric = sizes == 0
    scale = np.exp(rng.uniform(*np.log(STEPS), (len(x), 1)))
    moved = np.where(numeric, np.clip(x + scale * rng.normal(size=x.shape), 0, 1), x)
    other = np.floor(rng.random(x.shape) * sizes)
    flip = ~numeric & (rng.random(x.shape) < 1 / x.shape[1])
    return np.where(flip, other, moved)


def candidates(
    space: Space, history: History, rng: np.random.Generator
) -> Float[np.ndarray, "c d"]:
    """Return the points the acquisition is first evaluated at: `CANDIDATES`
    quasi-random ones, the `ELITES` best trials and `NEIGHBOURS` perturbations of each."""
    spread = space.from_unit(qmc.Sobol(len(space.names), rng=rng).random(CANDIDATES))
    finite = np.isfinite(history.values)
    order = np.argsort(history.values[finite], kind="stable")
    elites = history.points[finite][order[:ELITES]]
    near = perturb(np.repeat(elites, NEIGHBOURS, 0), np.array(space.sizes), rng)
    return np.vstack([spread, elites, near])


def _refine(
    acquisition: Acquisition, x: Point, numeric: Bool[np.ndarray, " d"]
) -> Point:
    """Return `x` with its numbers where L-BFGS-B from them maximizes the acquisition
    within [0, 1], and its choices as they are."""
    if not numeric.any():
        return x
    point = x.copy()

    def loss(u: Float[np.ndarray, " k"]) -> tuple[float, Float[np.ndarray, " k"]]:
        point[numeric] = u
        a, grad = acquisition(point[None])
        return -float(a[0]), -grad[0, numeric]

    bounds = [(0.0, 1.0)] * int(numeric.sum())
    point[numeric] = minimize(
        loss, x[numeric], jac=True, method="L-BFGS-B", bounds=bounds
    ).x
    return point


def _sweep(acquisition: Acquisition, x: Point, sizes: Int[np.ndarray, " d"]) -> Point:
    """Return `x` with each choice in turn at its option of most acquisition."""
    x = x.copy()
    for i in np.flatnonzero(sizes):
        options = np.repeat(x[None], sizes[i], 0)
        options[:, i] = np.arange(sizes[i])
        x[i] = float(np.argmax(acquisition(options)[0]))
    return x


def local_search(
    acquisition: Acquisition, x: Point, sizes: Int[np.ndarray, " d"]
) -> tuple[Point, float]:
    """Return where a local search from `x` ends, and the acquisition there: L-BFGS-B
    over the numbers and a sweep over the choices, in turn, until the sweep moves none."""
    for _ in range(ROUNDS):
        x = _refine(acquisition, x, sizes == 0)
        swept = _sweep(acquisition, x, sizes)
        if np.array_equal(swept, x):
            break
        x = swept
    return x, float(acquisition(x[None])[0][0])


def maximize(
    acquisition: Acquisition, x: Float[np.ndarray, "c d"], sizes: Int[np.ndarray, " d"]
) -> Point:
    """Return the point of most acquisition that local searches from the `STARTS` best
    of the candidates `x` end at."""
    order = np.argsort(-acquisition(x)[0], kind="stable")
    ends = [local_search(acquisition, start, sizes) for start in x[order[:STARTS]]]
    return max(ends, key=lambda end: end[1])[0]


@methods.register("gp")
def _gp(c: tcfg.GPSearch, space: Space, history: History) -> Point:
    """Bayesian optimization with a Gaussian process: after `startup` quasi-random
    trials, the point of most log expected improvement (or least lower confidence
    bound) under a Matérn-5/2 GP of the trials, those still training believed to end
    at its mean (`bux.search.gp`)."""
    if history.proposed < c.startup or np.isfinite(history.values).sum() < 2:
        return quasi_random(space, history.seed, history.proposed)
    rng = generator(history)
    sizes = np.array(space.sizes)
    gp = fit(history.points, labels(history.values), sizes > 0, rng)
    gp = believe(gp, history.pending)
    acquisition = Acquisition(c, gp, float(posterior(gp, gp.x)[0].min()))
    return maximize(acquisition, candidates(space, history, rng), sizes)
