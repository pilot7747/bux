"""A tail EMA of the weights, which a run ends on (``ema``, modded-nanogpt's TailEMA).

`tracked` is an optax transformation that keeps the average in the optimizer's
state, in float32, sharded as the parameters are, so that a checkpoint keeps it and
a run resumes inside its window. It seeds the average with the weights after the
first step of the window, and moves it toward every later step's by the rate
``2 / (window + 1)``. `Ship`, a hook, ends the run on ``last + blend (average -
last)``, before the last step's eval and checkpoint read the weights::

    tx = beside(optimizer(...), config)  # optax.chain(tx, tracked(...))
    trainer = Trainer(model, tx, loss)
    trainer.fit(data, steps, hooks=[log, Ship(config.ema, config.steps), *evals])
"""

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from bux.train.config import EMAConfig, TrainConfig
from bux.train.hooks import Hook
from bux.train.objective import Metrics

if TYPE_CHECKING:
    from bux.train.trainer import Trainer


class EMAState(NamedTuple):
    """The average and the steps taken, in the optimizer's state."""

    count: Int[Array, ""]  # the updates applied so far
    average: PyTree[Float[Array, "*_"]]  # of the weights, in float32


def first(config: EMAConfig, steps: int) -> int:
    """Return the step whose weights seed the average, of a run of `steps`."""
    return max(steps - config.steps(steps) + 1, 1)


def tracked(config: EMAConfig, steps: int) -> optax.GradientTransformation:
    """Return the transformation that keeps the average of `config` for a run of
    `steps`, beside the optimizer's: it changes no update, and reads the weights
    each one makes.

    Args:
        config: The run's ``ema`` setting.
        steps: The run's steps.
    """
    start, rate = first(config, steps), 2 / (config.steps(steps) + 1)

    def init(params: PyTree[Float[Array, "*_"]]) -> EMAState:
        zeros = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
        return EMAState(jnp.zeros([], jnp.int32), zeros)

    def update(
        updates: PyTree[Float[Array, "*_"]],
        state: EMAState,
        params: PyTree[Float[Array, "*_"]] | None = None,
    ) -> tuple[PyTree[Float[Array, "*_"]], EMAState]:
        assert params is not None, "the average reads the weights"
        step = state.count + 1  # the step these updates make
        weights = optax.apply_updates(params, updates)

        def averaged(
            a: Float[Array, "*s"], w: Float[Array, "*s"]
        ) -> Float[Array, "*s"]:
            w = w.astype(jnp.float32)
            moved = jnp.where(step == start, w, a + rate * (w - a))
            return jnp.where(step >= start, moved, a)

        average = jax.tree.map(averaged, state.average, weights)
        return updates, EMAState(step, average)

    return optax.GradientTransformation(init, update)  # ty: ignore[invalid-argument-type]


def beside(
    tx: optax.GradientTransformation, train: TrainConfig
) -> optax.GradientTransformation:
    """Return `tx`, and `tracked` beside it where the run `train` has an ``ema``."""
    if train.ema is None:
        return tx
    return optax.chain(tx, tracked(train.ema, train.steps))


def average_of(opt_state: PyTree) -> EMAState | None:
    """Return the average's state in an optimizer's state, or None for none."""
    found = jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, EMAState))
    return next((x for x in found if isinstance(x, EMAState)), None)


class Ship(Hook):
    """Set the weights to the blend of `tracked`'s average and the last step's,
    after the run's last step: a hook before the evals and the checkpoint, so that
    they read it. A run whose window has not begun keeps its weights.

    Args:
        config: The run's ``ema`` setting.
        steps: The run's steps.
    """

    def __init__(self, config: EMAConfig, steps: int):
        self.blend, self.first = config.blend, first(config, steps)

    def due(self, step: int, steps: int) -> bool:
        return step == steps

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> None:
        state = average_of(trainer.optimizer.opt_state)
        assert state is not None, "the optimizer keeps no average: chain `tracked`"
        if int(state.count[...]) < self.first:
            return
        weights = nnx.as_pure(nnx.state(trainer.model, nnx.Param))
        average = nnx.as_pure(state.average)

        def blended(w: Float[Array, "*s"], a: Float[Array, "*s"]) -> Float[Array, "*s"]:
            return (w + self.blend * (a - w.astype(jnp.float32))).astype(w.dtype)

        nnx.update(trainer.model, jax.tree.map(blended, weights, average))
