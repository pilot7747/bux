"""Hooks: what the training loop does between steps, every so many of them.

A hook runs after every `every` steps and after the last. What it returns, if
anything, is scalars that go to every sink (``bux.train.sinks``). Another hook is a
subclass:

    class WeightNorm(Hook):
        every = 100

        def __call__(self, trainer: Trainer, metrics: Metrics) -> Scalars:
            params = nnx.state(trainer.model, nnx.Param)
            return {"weight_norm": float(optax.tree.norm(params))}

    train(run, hooks=[WeightNorm()])

Hooks run in order, so the run's own (log, eval, checkpoint) see a step first. When
training stops early (``Trainer.stopping``, which the launcher sets at SIGTERM), every
hook's `stop` runs at the step it stopped at, and the checkpoint hook saves it.
"""

import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import orbax.checkpoint as ocp

from bux.train.data import Batch
from bux.train.objective import Metrics
from bux.train.optim import Schedule
from bux.train.sinks import Scalars

if TYPE_CHECKING:
    # Quoted where used: jaxtyping, which checks annotations as the tests run, reads
    # them as a module loads, and trainer.py imports this one.
    from bux.train.trainer import Trainer


class Hook:
    every: int = 1

    def due(self, step: int, steps: int) -> bool:
        """Whether the hook runs after `step` of a run of `steps`."""
        return step % self.every == 0 or step == steps

    def start(self, trainer: "Trainer") -> None:
        """Before the first step."""

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars | None:
        """After a step that it is due at, with what the step measured."""
        raise NotImplementedError

    def stop(self, trainer: "Trainer") -> None:
        """When training stops before the last step, at the step it stopped at."""

    def close(self) -> None:
        """After the last step, or when training stops."""


class Log(Hook):
    """The step's metrics, with its learning rate, the tokens seen and the tokens per
    second since the last log. Optax counts from 0: the step just taken is
    `trainer.step - 1`."""

    def __init__(self, *, every: int, lr: Schedule, tokens: int):
        self.every, self.lr, self.tokens = every, lr, tokens  # tokens per step
        self.clock, self.since = time.perf_counter(), 0

    def start(self, trainer: "Trainer") -> None:
        self.clock, self.since = time.perf_counter(), trainer.step

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars:
        scalars = {k: float(v) for k, v in metrics.items()}  # waits for the step
        now = time.perf_counter()
        seen = (trainer.step - self.since) * self.tokens
        scalars |= {
            "lr": float(self.lr(trainer.step - 1)),
            "tokens": trainer.step * self.tokens,
            "tokens_per_s": seen / (now - self.clock),
        }
        self.clock, self.since = now, trainer.step
        return scalars


class Eval(Hook):
    """Held-out loss: the first head's, over the first `batches` batches of `data`,
    the same ones every time."""

    def __init__(
        self, data: Callable[[], Iterator[Batch]], *, every: int, batches: int
    ):
        self.data, self.every, self.batches = data, every, batches

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars:
        held_out = itertools.islice(self.data(), self.batches)
        return {"eval_loss": trainer.evaluate(held_out)}


class Checkpoint(Hook):
    """The trainer's state, saved under `folder` at its step, and at the step training
    stops at if it stops early; the `keep` most recent checkpoints stay."""

    def __init__(self, folder: Path, *, every: int, keep: int):
        self.every = every
        self.manager = ocp.CheckpointManager(
            folder.resolve(),
            options=ocp.CheckpointManagerOptions(max_to_keep=keep),
            item_handlers=ocp.StandardCheckpointHandler(),  # so it reads metadata
        )

    def restore(self, trainer: "Trainer") -> bool:
        """Load the latest checkpoint into `trainer`, if there is one."""
        if (latest := self.manager.latest_step()) is None:
            return False
        state = trainer.state()
        _fits(self.manager.item_metadata(latest), state, latest)
        args = ocp.args.StandardRestore(state)
        trainer.load(self.manager.restore(latest, args=args), step=latest)
        return True

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> None:
        self.manager.save(trainer.step, args=ocp.args.StandardSave(trainer.state()))

    def stop(self, trainer: "Trainer") -> None:
        self(trainer, {})  # unless the step is saved already, which orbax skips

    def close(self) -> None:
        self.manager.wait_until_finished()
        self.manager.close()


def _fits(saved: object, state: object, step: int) -> None:
    """Raise unless the checkpoint of `step`, whose metadata is `saved`, holds the
    variables of `state`. `methods.repeat` is not in a run's id, but it changes them:
    `scan` stacks a repeat's passes into one."""
    on_disk, mine = _paths(saved), _paths(state)
    if on_disk == mine:
        return
    raise ValueError(
        f"the checkpoint of step {step} does not hold this model's state: it has "
        f"{min(on_disk - mine, default='nothing else')}, the model "
        f"{min(mine - on_disk, default='nothing else')}. A run saved with "
        "methods(repeat=unroll) resumes with unroll alone, and one with scan with scan"
    )


def _paths(tree: object) -> set[str]:
    """The path of every leaf of `tree`, as a checkpoint keeps it."""

    def key(k: object) -> str:
        return str(next(getattr(k, a) for a in ("key", "idx", "name") if hasattr(k, a)))

    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return {"/".join(map(key, path)) for path, _ in leaves}
