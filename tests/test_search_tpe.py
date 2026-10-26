"""The tree-structured Parzen estimator (bux.search.tpe): its densities against
Optuna's, what it proposes for a history, and that it searches better than random."""

import math

import numpy as np
import pytest
from jaxtyping import Float

from bux.search import tpe
from bux.search.methods import History, propose, quasi_random
from bux.search.space import Space
from bux.train import config as tcfg

SPACE = Space(
    ("optimizer.lr", "optimizer.weight_decay", "ffn.activation"),
    (
        tcfg.LogUniformConfig(low=1e-4, high=1e-1),
        tcfg.UniformConfig(low=0, high=0.2),
        tcfg.ChoiceConfig(options=("swish", "gelu", "relu")),
    ),
)
TPE = tcfg.TPESearch()


def _loss(values: dict[str, float]) -> float:
    """A loss least at lr = 3e-3, weight decay 0.05 and gelu, quadratic in log lr."""
    lr, decay = values["optimizer.lr"], values["optimizer.weight_decay"]
    activation = (0.3, 0.0, 0.6)[int(values["ffn.activation"])]
    return math.log10(lr / 3e-3) ** 2 + 25 * (decay - 0.05) ** 2 + activation


def _history(
    n: int, seed: int = 0, pending: int = 0
) -> tuple[History, Float[np.ndarray, "t 3"]]:
    """A history of `n` random trials of `_loss` and `pending` still training, and
    the points of all of them, those training last."""
    rng = np.random.default_rng(seed)
    points = SPACE.from_unit(rng.random((n + pending, 3)))
    values = np.array([_loss(SPACE.decode(x)) for x in points[:n]])
    return History(points[:n], values, points[n:], n + pending, seed), points


def _search(method: tcfg.SearchMethod, seed: int, trials: int) -> float:
    """The least loss of `trials` trials of `method`, one after another, each trained
    at the values its point decodes to."""
    points, values = np.zeros((0, 3)), np.zeros(0)
    for i in range(trials):
        x = propose(method, SPACE, History(points, values, np.zeros((0, 3)), i, seed))
        trained = SPACE.decode(x)
        points = np.vstack([points, SPACE.encode(trained)])
        values = np.append(values, _loss(trained))
    return float(values.min())


def test_the_densities_are_optunas():
    # Optuna 5.0.0's _ParzenEstimator at these points, with TPESampler's defaults: at
    # 30 points the older weigh less, and the choice's kernels share the prior.
    x = np.array([[0.1, 0.9, 0.0], [0.5, 0.5, 1.0], [0.95, 0.02, 2.0]])
    optuna = {
        3: [-1.47007582, -0.8827053, -2.19912183],
        30: [-1.66994334, -2.98659264, 0.28469943],
    }
    for n, expected in optuna.items():
        points = np.random.default_rng(0).random((n, 3))
        points[:, 2] = np.floor(points[:, 2] * 3)
        f = tpe.fit(points, (0, 0, 3))
        np.testing.assert_allclose(tpe.log_pdf(f, x, (0, 0, 3)), expected, rtol=1e-7)


def test_a_history_proposes_the_same_point_every_time():
    history, _ = _history(20, pending=3)
    for c in (TPE, tcfg.TPESearch(multivariate=False)):
        np.testing.assert_array_equal(
            propose(c, SPACE, history), propose(c, SPACE, history)
        )


@pytest.mark.parametrize("multivariate", [True, False])
def test_a_proposal_is_in_bounds_and_an_option_of_each_choice(multivariate):
    c = tcfg.TPESearch(multivariate=multivariate)
    for seed in range(20):
        history, _ = _history(12 + seed, seed, pending=seed % 4)
        x = propose(c, SPACE, history)
        assert np.all((0 <= x[:2]) & (x[:2] <= 1))
        assert x[2] in (0, 1, 2)


def test_the_first_trials_are_quasi_random():
    history, _ = _history(TPE.startup - 1)
    point = quasi_random(SPACE, history.seed, history.proposed)
    np.testing.assert_array_equal(propose(TPE, SPACE, history), point)
    after = history._replace(proposed=TPE.startup)
    assert not np.array_equal(
        propose(TPE, SPACE, after), quasi_random(SPACE, history.seed, TPE.startup)
    )


def test_trials_still_training_count_among_the_rest():
    history, points = _history(20, pending=2)
    best, rest = tpe.split(history, TPE.gamma)
    assert len(best) == 2 and len(rest) == 20
    np.testing.assert_array_equal(rest[-2:], points[-2:])
    assert history.values.min() == min(_loss(SPACE.decode(x)) for x in best)


def test_trials_at_once_differ():
    # The same candidates (the same generator), but one is already training.
    history, _ = _history(20)
    first = propose(TPE, SPACE, history)
    second = propose(TPE, SPACE, history._replace(pending=first[None]))
    assert not np.allclose(first, second)


def test_failed_trials_count_among_the_rest_and_never_as_the_best():
    history, _ = _history(20)
    failed = np.full(20, np.nan)
    failed[[3, 11]] = history.values[[3, 11]]
    best, rest = tpe.split(history._replace(values=failed), 0.5)
    np.testing.assert_array_equal(best, history.points[[3, 11]])
    assert len(rest) == 18
    x = propose(TPE, SPACE, history._replace(values=failed))
    assert np.isfinite(x).all()


def test_a_search_of_failed_trials_only_stays_quasi_random():
    history, _ = _history(15)
    failed = history._replace(values=np.full(15, np.nan))
    point = quasi_random(SPACE, failed.seed, failed.proposed)
    np.testing.assert_array_equal(propose(TPE, SPACE, failed), point)


def test_tpe_finds_a_lower_loss_than_random_search():
    # Over 10 searches of 30 trials, TPE's least losses are ~10x lower than random's
    # (medians 0.0055 and 0.070, geometric means 0.0047 and 0.045).
    seeds = range(10)
    found = np.array([_search(TPE, seed, 30) for seed in seeds])
    random = np.array([_search(tcfg.RandomSearch(), seed, 30) for seed in seeds])
    assert np.median(found) < np.median(random) / 4
    assert np.log(found).mean() < np.log(random).mean() - np.log(4)
