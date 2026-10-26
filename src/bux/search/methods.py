"""How a search proposes its next trial: its method, found by the type of its config.

A method's builder gets its config (`bux.train.config.SearchMethod`), the `Space` and
the `History` of the search so far, and returns the next point's coordinates
(`bux.search.space`). Random and quasi-random search are here; TPE is in
`bux.search.tpe` and the Gaussian process in `bux.search.gp`. A builder registered
under a taken name replaces bux's own::

    @methods.register("random")
    def _(c: RandomSearch, space: Space, history: History) -> Point:
        return space.from_unit(generator(history).random(len(space.names)))

A new method needs its config in the ``SearchMethod`` union of
`bux.train.config` too, where the compiler finds it.
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Float
from scipy.stats import qmc

from bux.search.space import Space
from bux.train import config as tcfg
from bux.train.registry import Registry

#: A point of a space, in its coordinates (`bux.search.space`).
Point = Float[np.ndarray, " d"]


class History(NamedTuple):
    """What a search knows when it proposes a trial.

    Attributes:
        points: The coordinates of the trials it has an objective of.
        values: Their objectives, the lower the better; NaN for a trial that failed
            or whose objective is not a number, as a run that diverged.
        pending: The coordinates of the trials still training.
        proposed: How many trials it proposed before this one, all of them.
        seed: The search's seed (``search.seed``).
    """

    points: Float[np.ndarray, "n d"]
    values: Float[np.ndarray, " n"]
    pending: Float[np.ndarray, "m d"]
    proposed: int
    seed: int = 0


#: The methods, by the ``type`` of their config. A builder gets the config, the
#: `Space` and the `History`, and returns the next `Point`.
methods = Registry[Point]("search method", "bux.search.methods.methods")


def propose(config: tcfg.SearchMethod, space: Space, history: History) -> Point:
    """Return the next point of the search that `config`'s method proposes.

    Args:
        config: The search's method.
        space: Its dimensions.
        history: What it knows so far.

    Returns:
        The coordinates of the next trial, which `Space.decode` turns into values.

    Raises:
        NotImplementedError: If no method is registered under the config's type.
    """
    return methods.build(config, space, history)


def generator(history: History) -> np.random.Generator:
    """Return the random generator of the trial `history` is before, seeded by the
    search's seed and the trials proposed so far: so a search launched again
    proposes what it would have."""
    return np.random.default_rng([history.seed, history.proposed])


def quasi_random(space: Space, seed: int, index: int) -> Point:
    """Return point `index`, from 0, of a scrambled Halton sequence that `seed`
    scrambles: what `halton` proposes, and the model-based methods at first. It
    extends a point at a time, as trials that start whenever a GPU frees up need."""
    return _nth(qmc.Halton(len(space.names), scramble=True, rng=seed), space, index)


def _nth(engine: qmc.QMCEngine, space: Space, index: int) -> Point:
    """Return point `index` of `engine`'s sequence, as coordinates of `space`."""
    if index:
        engine.fast_forward(index)
    return space.from_unit(engine.random(1)[0])


@methods.register("random")
def _random(c: tcfg.RandomSearch, space: Space, history: History) -> Point:
    """A point drawn uniformly at random, every dimension on its own scale."""
    return space.from_unit(generator(history).random(len(space.names)))


@methods.register("halton")
def _halton(c: tcfg.HaltonSearch, space: Space, history: History) -> Point:
    """The next point of a scrambled Halton sequence (`quasi_random`), which the
    search's seed scrambles."""
    return quasi_random(space, history.seed, history.proposed)


@methods.register("sobol")
def _sobol(c: tcfg.SobolSearch, space: Space, history: History) -> Point:
    """The next point of a scrambled Sobol sequence, which the search's seed
    scrambles."""
    engine = qmc.Sobol(len(space.names), scramble=True, rng=history.seed)
    return _nth(engine, space, history.proposed)
