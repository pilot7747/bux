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
import random
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.experimental import multihost_utils
from jaxtyping import Array, Shaped

from bux import parallel
from bux.lang.printer import value_to_source
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
    `peak`, where the peak is known and the trainer counts its FLOPs (`work`).
    Optax counts from 0: the step just taken is `trainer.step - 1`."""

    def __init__(
        self, *, every: int, lr: Schedule, tokens: int, peak: float | None = None
    ):
        self.every, self.lr, self.tokens = every, lr, tokens  # tokens per step
        self.peak = peak
        self.clock, self.since, self.flops = time.perf_counter(), 0, 0.0

    def start(self, trainer: "Trainer") -> None:
        self.clock, self.since = time.perf_counter(), trainer.step
        self.flops = trainer.flops

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars:
        scalars = {k: float(v) for k, v in metrics.items()}  # waits for the step
        now = time.perf_counter()
        steps, seconds = trainer.step - self.since, now - self.clock
        scalars |= {
            "lr": float(self.lr(trainer.step - 1)),
            "tokens": trainer.step * self.tokens,
            "tokens_per_s": steps * self.tokens / seconds,
        }
        if self.peak and trainer.work is not None:
            scalars["mfu"] = (trainer.flops - self.flops) / seconds / self.peak
        self.clock, self.since, self.flops = now, trainer.step, trainer.flops
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
    stops at if it stops early; the `keep` most recent checkpoints stay.

    A checkpoint also keeps how the run in force lays out the state (``layout``), and
    a run resumes from one only with its arrays and their layout, else it fails before
    it reads one. Every process of a job saves its part, so `folder` must be one they
    all see: one that is not fails at once, where orbax would wait at the first save
    for the parts it cannot see until it timed out, twenty minutes later."""

    def __init__(self, folder: Path, *, every: int, keep: int):
        self.every, self.layout = every, layout()
        _check_shared(folder.resolve())
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
        if misfit := self._misfit(latest, state):
            raise ValueError(
                f"the checkpoint of step {latest} does not hold this run's state: "
                f"{misfit}. {_RESUMES_WITH}"
            )
        shapes = jax.tree.map(_shape, state)
        jax.tree.map(lambda x: x.delete(), state)
        args = ocp.args.StandardRestore(shapes)
        trainer.load(self.manager.restore(latest, args=args), step=latest)
        return True

    def _misfit(self, step: int, state: nnx.State) -> str | None:
        """Say how the checkpoint of `step` differs from `state` and its layout, if it
        does."""
        saved = self.manager.metadata(step).custom_metadata
        arrays = _arrays(self.manager.item_metadata(step))
        return _changed(saved, self.layout) or _differs(arrays, _arrays(state))

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> None:
        state = ocp.args.StandardSave(trainer.state())
        self.manager.save(trainer.step, args=state, custom_metadata=self.layout)

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


# The methods that a run's id leaves out but that change the arrays of its state.
_RESUMES_WITH = (
    "A run resumes with the methods it was saved with where they lay out its state: "
    "repeat (unroll or scan), a circular pipeline's rounds and an uneven one's stages"
)


def layout() -> dict[str, list[int] | None]:
    """Return how the run in force lays out its state where the arrays' paths and
    shapes do not say: the passes of an uneven pipeline's stages, which spread its
    layers over them as no other split does (``pipeline.spread``)."""
    on, split = parallel.active(), None
    if on is not None and parallel.pipeline.pipelined():
        split = on.methods.stages
    if split is not None and len(set(split)) == 1:
        split = None  # even, as a pipeline lays out its layers without a split
    return {"methods.stages": None if split is None else list(split)}


def _changed(
    saved: Mapping[str, object] | None, now: Mapping[str, object]
) -> str | None:
    """Say which setting of the layout `now` was another in the `saved` one, if any."""
    for name, value in now.items():
        if (was := (saved or {}).get(name)) != value:
            was, value = value_to_source(was), value_to_source(value)
            return f"it was saved with {name} = {was}, and this run has {value}"
    return None


# An array as a checkpoint keeps it: its dtype and shape.
Stored = tuple[str, tuple[int, ...]]


def _differs(saved: dict[str, Stored], mine: dict[str, Stored]) -> str | None:
    """Say how the arrays of a checkpoint, `saved`, differ from the run's, `mine`, if
    they do."""
    if saved.keys() != mine.keys():
        extra = min(saved.keys() - mine.keys(), default="nothing else")
        missing = min(mine.keys() - saved.keys(), default="nothing else")
        return f"it has {extra}, and this run {missing}"
    for path in sorted(mine):
        if saved[path] != mine[path]:
            was, now = _typed(saved[path]), _typed(mine[path])
            return f"its {path} is {was}, and this run's {now}"
    return None


def _typed(array: Stored) -> str:
    dtype, shape = array
    return f"{dtype}{list(shape)}"


def _arrays(tree: object) -> dict[str, Stored]:
    """The dtype and shape of every array of `tree`, by its path, as a checkpoint
    keeps them: a typed key as its key's data."""

    def stored(x: Any) -> Stored:  # an array, or a checkpoint's metadata of one
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.eval_shape(jax.random.key_data, x)
        return str(x.dtype), tuple(x.shape)

    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return {_named(path): stored(x) for path, x in leaves}


def _named(path: tuple) -> str:
    """Return a pytree path as its keys, joined by /: a name of a checkpoint's array."""
    return jax.tree_util.keystr(path, simple=True, separator="/")


def _check_shared(folder: Path) -> None:
    """Raise unless the job's processes all see `folder` as one folder, which a node's
    own disk is not: each writes a file into it, and each counts them."""
    if (n := jax.process_count()) == 1:
        return
    folder.mkdir(parents=True, exist_ok=True)
    token = int(multihost_utils.broadcast_one_to_all(np.int32(random.getrandbits(31))))
    mine = folder / f".process-{token}-{jax.process_index()}"
    mine.touch()
    multihost_utils.sync_global_devices("each process wrote its file")
    seen = _seen(folder, f".process-{token}-*", n)
    fewest = int(multihost_utils.process_allgather(np.int32(seen)).min())
    mine.unlink()
    if fewest < n:
        raise ValueError(
            f"the {n} processes of this job do not share {folder}: one sees {fewest} "
            "of the files they wrote there. Each saves its part of a checkpoint into "
            "it, so put the run's directory (--out) where every node sees it"
        )


def _seen(folder: Path, pattern: str, n: int, wait: float = 10.0) -> int:
    """Return how many files in `folder` match `pattern`, once there are `n` or after
    `wait` seconds: a network filesystem may show another node's file a little late."""
    deadline = time.monotonic() + wait
    while len(found := list(folder.glob(pattern))) < n and time.monotonic() < deadline:
        time.sleep(0.5)
    return len(found)
