"""Hooks: what the training loop does between steps, every so many of them.

A hook runs after every `every` steps and after the last. What it returns, if
anything, is a report that goes to every sink (``bux.train.sinks``). Another hook is
a subclass:

    class WeightNorm(Hook):
        every = 100

        def __call__(self, trainer: Trainer, metrics: Metrics) -> Report:
            params = nnx.state(trainer.model, nnx.Param)
            return {"weight_norm": float(optax.tree.norm(params))}

    train(run, hooks=[WeightNorm()])

Hooks run in order, so the run's own (log, eval, checkpoint) see a step first.
"""

import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import orbax.checkpoint as ocp

from bux.train.data import Batch
from bux.train.optim import Schedule
from bux.train.sinks import Report

if TYPE_CHECKING:
    from bux.train.trainer import Trainer

Metrics = dict[str, jax.Array]  # what a step measured, still on the device


class Hook:
    every: int = 1

    def due(self, step: int, steps: int) -> bool:
        """Whether the hook runs after `step` of a run of `steps`."""
        return step % self.every == 0 or step == steps

    def start(self, trainer: Trainer) -> None:
        """Before the first step."""

    def __call__(self, trainer: Trainer, metrics: Metrics) -> Report | None:
        """After a step that it is due at, with what the step measured."""
        raise NotImplementedError

    def close(self) -> None:
        """After the last step, or when training stops."""


class Log(Hook):
    """The step's metrics, with the learning rate, the tokens seen and the tokens a
    second since the last log."""

    def __init__(self, *, every: int, lr: Schedule, tokens: int):
        self.every, self.lr, self.tokens = every, lr, tokens  # tokens a step
        self.clock, self.since = time.perf_counter(), 0

    def start(self, trainer: Trainer) -> None:
        self.clock, self.since = time.perf_counter(), trainer.step

    def __call__(self, trainer: Trainer, metrics: Metrics) -> Report:
        report = {k: float(v) for k, v in metrics.items()}  # waits for the step
        now = time.perf_counter()
        seen = (trainer.step - self.since) * self.tokens
        report |= {
            "lr": float(self.lr(trainer.step - 1)),
            "tokens": trainer.step * self.tokens,
            "tokens_per_s": seen / (now - self.clock),
        }
        self.clock, self.since = now, trainer.step
        return report


class Eval(Hook):
    """Held-out loss: the first head's, over the first `batches` batches of `data`,
    the same ones every time."""

    def __init__(
        self, data: Callable[[], Iterator[Batch]], *, every: int, batches: int
    ):
        self.data, self.every, self.batches = data, every, batches

    def __call__(self, trainer: Trainer, metrics: Metrics) -> Report:
        held_out = itertools.islice(self.data(), self.batches)
        return {"eval_loss": trainer.evaluate(held_out)}


class Checkpoint(Hook):
    """The trainer's state, saved under `folder` at its step; the `keep` most recent
    checkpoints stay."""

    def __init__(self, folder: Path, *, every: int, keep: int):
        self.every = every
        self.manager = ocp.CheckpointManager(
            folder.resolve(), options=ocp.CheckpointManagerOptions(max_to_keep=keep)
        )

    def restore(self, trainer: Trainer) -> bool:
        """Load the latest checkpoint into `trainer`, if there is one."""
        if (latest := self.manager.latest_step()) is None:
            return False
        args = ocp.args.StandardRestore(trainer.state())
        trainer.load(self.manager.restore(latest, args=args), step=latest)
        return True

    def __call__(self, trainer: Trainer, metrics: Metrics) -> None:
        self.manager.save(trainer.step, args=ocp.args.StandardSave(trainer.state()))

    def close(self) -> None:
        self.manager.wait_until_finished()
        self.manager.close()
