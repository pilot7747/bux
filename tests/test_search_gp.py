"""Bayesian optimization with a Gaussian process (bux.search.gp): its proposals, the GP
it fits, log EI, and what it finds against random search."""

import numpy as np
import pytest
from jaxtyping import Float
from scipy.optimize import approx_fprime
from scipy.special import ndtr
from scipy.stats import qmc

from bux.search import gp
from bux.search.methods import History, propose, quasi_random
from bux.search.space import Space
from bux.train import config as tcfg

U = tcfg.UniformConfig(low=0, high=1)
CHOICE = tcfg.ChoiceConfig(options=("a", "b", "c"))
PLANE = Space(("x", "y"), (U, U))
MIXED = Space(("x", "y", "c"), (U, U, CHOICE))
METHOD = tcfg.GPSearch()


def bowl(x: Float[np.ndarray, " d"]) -> float:
    """A shifted quadratic in the numbers, plus an offset for the choice, if any."""
    offset = [0.3, 0.0, 0.6][int(x[2])] if len(x) > 2 else 0.0
    return float((x[0] - 0.3) ** 2 + 2 * (x[1] - 0.7) ** 2 + offset)


def hartmann3(x: Float[np.ndarray, " 3"]) -> float:
    """Hartmann's 3-dimensional function, whose least is -3.86278."""
    a = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    p = [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    weights = np.array([1.0, 1.2, 3.0, 3.2])
    return float(-weights @ np.exp(-(a * (x - 1e-4 * np.array(p)) ** 2).sum(1)))


def history(space: Space, n: int, f=bowl, seed: int = 0) -> History:
    """The history of `n` random trials of `f` over `space`."""
    points = space.from_unit(np.random.default_rng(seed).random((n, len(space.names))))
    values = np.array([f(x) for x in points])
    return History(points, values, np.zeros((0, len(space.names))), n, seed)


def in_space(space: Space, x: Float[np.ndarray, " d"]) -> bool:
    """Whether `x` is a point of `space`: its numbers in [0, 1], its choices options."""
    sizes = np.array(space.sizes)
    numbers = x[sizes == 0]
    options = x[sizes > 0]
    return (
        x.shape == (len(space.names),)
        and bool(np.all((numbers >= 0) & (numbers <= 1)))
        and bool(np.all((options == np.round(options)) & (options >= 0)))
        and bool(np.all(options < sizes[sizes > 0]))
    )


def search(method: tcfg.SearchMethod, space: Space, f, trials: int, seed: int) -> float:
    """Return the least objective of `trials` trials of `f` that `method` proposes, one
    after another."""
    d = len(space.names)
    points, values = np.zeros((0, d)), np.zeros(0)
    for i in range(trials):
        x = propose(method, space, History(points, values, np.zeros((0, d)), i, seed))
        points, values = np.vstack([points, x]), np.r_[values, f(x)]
    return float(values.min())


def test_a_history_proposes_the_same_point_every_time():
    h = history(MIXED, 8)
    assert np.array_equal(propose(METHOD, MIXED, h), propose(METHOD, MIXED, h))


@pytest.mark.parametrize("acquisition", ["log_ei", "ucb"])
@pytest.mark.parametrize("seed", range(3))
def test_a_proposal_is_in_the_space(acquisition, seed):
    c = METHOD.model_copy(update={"acquisition": acquisition})
    assert in_space(MIXED, propose(c, MIXED, history(MIXED, 8, seed=seed)))


def test_the_first_trials_are_quasi_random():
    h = history(MIXED, 4)
    assert np.array_equal(propose(METHOD, MIXED, h), quasi_random(MIXED, 0, 4))
    failed = h._replace(values=np.r_[np.nan, np.nan, np.nan, 1.0], proposed=9)
    assert np.array_equal(propose(METHOD, MIXED, failed), quasi_random(MIXED, 0, 9))


def test_the_gp_predicts_a_smooth_function_it_saw():
    def f(x: Float[np.ndarray, "n 2"]) -> Float[np.ndarray, " n"]:
        return np.sin(3 * x[:, 0]) + np.cos(5 * x[:, 1]) * x[:, 0]

    seen = qmc.Sobol(2, rng=0).random(32)
    fitted = gp.fit(seen, f(seen), np.zeros(2, bool), np.random.default_rng(0))
    unseen = np.random.default_rng(1).random((200, 2))
    mean, std, _, _ = gp.posterior(fitted, unseen)
    error = np.abs(mean - f(unseen))  # of a function from -0.3 to 1.9
    assert np.sqrt(np.mean(error**2)) < 0.02 and np.all(error < 3 * std)


def test_the_fits_gradient_is_its_loss_derivative():
    h = history(MIXED, 12)
    d2 = gp.squared_distances(h.points, h.points, np.array([False, False, True]))
    theta = np.log([0.3, 0.5, 1.5, 0.8, 0.01])
    _, grad = gp._loss(theta, d2, gp.labels(h.values))
    numeric = approx_fprime(theta, lambda t: gp._loss(t, d2, gp.labels(h.values))[0])
    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("acquisition", ["log_ei", "ucb"])
def test_the_acquisitions_gradient_is_its_derivative(acquisition):
    h = history(MIXED, 10)
    fitted = gp.fit(
        h.points,
        gp.labels(h.values),
        np.array([False, False, True]),
        np.random.default_rng(0),
    )
    c = METHOD.model_copy(update={"acquisition": acquisition})
    acquire = gp.Acquisition(c, fitted, float(fitted.y.min()))
    x = np.array([0.4, 0.6, 1.0])
    grad = acquire(x[None])[1][0]
    numeric = approx_fprime(x, lambda x: float(acquire(x[None])[0][0]), 1e-7)
    assert np.allclose(grad[:2], numeric[:2], rtol=1e-4, atol=1e-5) and grad[2] == 0


def test_log_ei_is_finite_and_increasing_where_ei_underflows():
    z = np.linspace(-1e3, 5, 20_001)
    log_ei = gp.log_h(z)
    assert np.all(np.isfinite(log_ei)) and np.all(np.diff(log_ei) > 0)
    plain = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi) + z * ndtr(z)
    assert np.all(plain[z < -40] == 0)  # underflows
    accurate = z > -5
    assert np.allclose(log_ei[accurate], np.log(plain[accurate]), rtol=1e-9)
    assert np.isfinite(gp.log_h(np.array([-1e7, -1e12]))).all()


def at_once(h: History, believed: bool) -> list[Float[np.ndarray, " 2"]]:
    """Four proposals at once after `h`, each with those before it still training if
    `believed`, else with none."""
    proposals = []
    for i in range(4):
        pending = np.array(proposals).reshape(i, 2)[: i if believed else 0]
        proposals.append(
            propose(METHOD, PLANE, h._replace(pending=pending, proposed=8 + i))
        )
    return proposals


def least_gap(points: list[Float[np.ndarray, " 2"]]) -> float:
    """The least distance between two of `points`."""
    return min(np.linalg.norm(a - b) for i, a in enumerate(points) for b in points[:i])


def test_trials_still_training_make_proposals_at_once_differ():
    h = history(PLANE, 8)
    assert least_gap(at_once(h, believed=False)) < 1e-3
    assert least_gap(at_once(h, believed=True)) > 0.05


def test_a_failed_trial_counts_as_worse_than_the_worst():
    y = gp.labels(np.array([1.0, 2.0, np.nan, 3.0, np.inf]))
    assert np.isfinite(y).all() and np.isclose(y.mean(), 0)
    assert y[2] == y[4] > y[3] > y[1] > y[0]
    assert np.isclose(y[2] - y[3], 0.5 * (y[3] - y[0]) + 1)  # in deviations


def test_the_gp_expects_failures_where_trials_failed():
    h = history(PLANE, 12)
    values = np.where(h.points[:, 0] > 0.7, np.nan, h.values)  # diverged past 0.7
    fitted = gp.fit(
        h.points, gp.labels(values), np.zeros(2, bool), np.random.default_rng(0)
    )
    mean = gp.posterior(fitted, h.points)[0]
    assert mean[np.isnan(values)].min() > mean[~np.isnan(values)].max()
    assert np.isfinite(propose(METHOD, PLANE, h._replace(values=values))).all()


LINE = Space(("x",), (U,))
CHOICES = Space(("c", "e"), (CHOICE, tcfg.ChoiceConfig(options=("p", "q"))))
TWICE = History(
    np.array([[0.5, 0.5, 1.0]] * 3 + [[0.2, 0.9, 0.0]] * 3),
    np.array([1.0, 1.1, 0.9, 2.0, 2.0, 2.0]),
    np.array([[0.5, 0.5, 1.0]]),
    7,
)
DEGENERATE = {
    "equal objectives": (MIXED, history(MIXED, 8)._replace(values=np.full(8, 2.5))),
    "the same points again": (MIXED, TWICE),
    "mostly failed": (
        MIXED,
        history(MIXED, 8)._replace(values=np.r_[np.full(6, np.nan), 1.0, 2.0]),
    ),
    "one number": (LINE, history(LINE, 6, f=lambda x: float((x[0] - 0.3) ** 2))),
    "only choices": (CHOICES, history(CHOICES, 6, f=lambda x: float(x[0] + x[1]))),
}


@pytest.mark.parametrize("space, h", DEGENERATE.values(), ids=DEGENERATE.keys())
def test_a_degenerate_history_still_proposes(space, h):
    assert in_space(space, propose(METHOD, space, h))


@pytest.mark.parametrize(
    "space, f, trials, least",
    [(MIXED, bowl, 16, 0.0), (Space(tuple("xyz"), (U, U, U)), hartmann3, 20, -3.86278)],
    ids=["bowl and a choice", "hartmann3"],
)
def test_the_gp_finds_lower_minima_than_random_search(space, f, trials, least):
    seeds = range(10)
    found = np.array([search(METHOD, space, f, trials, s) for s in seeds]) - least
    random = np.array([search(tcfg.RandomSearch(), space, f, trials, s) for s in seeds])
    random -= least
    assert np.median(found) < np.median(random) / 10
    assert (found < random).sum() >= 9
