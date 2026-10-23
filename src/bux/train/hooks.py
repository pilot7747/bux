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

Hooks run in order, so the run's own (log, eval, checkpoint, profile) see a step
first. When training stops early (``Trainer.stopping``, which the launcher sets at
SIGTERM), every hook's `stop` runs at the step it stopped at, and the checkpoint hook
saves it.
"""

import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import orbax.checkpoint as ocp
from jaxtyping import Array, Shaped

from bux.train import trace
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
    second since the last log; and `mfu`, the model FLOPs a second over the devices'
    peak, where `flops` (a step's) and `peak` (the devices' a second) are known.
    Optax counts from 0: the step just taken is `trainer.step - 1`."""

    def __init__(
        self,
        *,
        every: int,
        lr: Schedule,
        tokens: int,
        flops: float | None = None,
        peak: float | None = None,
    ):
        self.every, self.lr, self.tokens = every, lr, tokens  # tokens per step
        self.ideal = flops / peak if flops and peak else None  # a step at the peak, s
        self.clock, self.since = time.perf_counter(), 0

    def start(self, trainer: "Trainer") -> None:
        self.clock, self.since = time.perf_counter(), trainer.step

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars:
        scalars = {k: float(v) for k, v in metrics.items()}  # waits for the step
        now = time.perf_counter()
        steps, seconds = trainer.step - self.since, now - self.clock
        scalars |= {
            "lr": float(self.lr(trainer.step - 1)),
            "tokens": trainer.step * self.tokens,
            "tokens_per_s": steps * self.tokens / seconds,
        }
        if self.ideal is not None:
            scalars["mfu"] = steps * self.ideal / seconds
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
        """Load the latest checkpoint into `trainer`, if there is one, in the memory of
        the state it replaces: the steps, which donate it, then find the rest of the
        memory in one piece, not on either side of it."""
        if (latest := self.manager.latest_step()) is None:
            return False
        state = trainer.state()
        _fits(self.manager.item_metadata(latest), state, latest)
        shapes = jax.tree.map(_shape, state)
        jax.tree.map(lambda x: x.delete(), state)
        args = ocp.args.StandardRestore(shapes)
        trainer.load(self.manager.restore(latest, args=args), step=latest)
        return True

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> None:
        self.manager.save(trainer.step, args=ocp.args.StandardSave(trainer.state()))

    def stop(self, trainer: "Trainer") -> None:
        self(trainer, {})  # unless the step is saved already, which orbax skips

    def close(self) -> None:
        self.manager.wait_until_finished()
        self.manager.close()


class Profile(Hook):
    """A jax.profiler trace, into `folder`, of the `steps` steps after step `start`.
    It waits for the step before it starts and stops, so the trace holds whole steps.

    On GPUs it then reports what a step of them spent in collectives, `comm_ms`, and in
    collectives with no compute beside them, `exposed_comm_ms` (``bux.train.trace``)."""

    def __init__(self, folder: Path, *, start: int, steps: int):
        self.folder, self.first, self.last = folder, start, start + steps
        self.tracing = False

    def due(self, step: int, steps: int) -> bool:
        return step in (self.first, self.last)

    def start(self, trainer: "Trainer") -> None:
        self._at(trainer.step)  # step 0, or the step a run resumes at

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars | None:
        jax.block_until_ready(metrics)
        self._at(trainer.step)
        if trainer.step != self.last or (file := trace.latest(self.folder)) is None:
            return None
        return trace.comm_ms(file, self.last - self.first)

    def _at(self, step: int) -> None:
        if step == self.first:
            jax.profiler.start_trace(self.folder)
            self.tracing = True
        else:
            self.close()

    def close(self) -> None:
        if self.tracing:
            jax.profiler.stop_trace()
            self.tracing = False


def _shape(x: Shaped[Array, "..."]) -> jax.ShapeDtypeStruct:
    """`x`'s shape, dtype and sharding: a typed key's too, which orbax's own drops."""
    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)


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
