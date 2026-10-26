"""Hooks: what the training loop does between steps, every so many of them.

A hook runs after every `Hook.every` steps and after the last. What it returns, if
anything, is scalars that go to every sink (`bux.train.sinks`), or a function that
gives them (`Later`), which the loop calls once it has launched the next step: the
log reads a step's metrics so, which waits for that step as the next one runs, so
that the GPU never waits for the host to launch a step. Another hook is a subclass::

    class WeightNorm(Hook):
        every = 100

        def __call__(self, trainer: Trainer, metrics: Metrics) -> Scalars:
            params = nnx.state(trainer.model, nnx.Param)
            return {"weight_norm": float(optax.tree.norm(params))}

    train(run, hooks=[WeightNorm()])

A run's own hooks are `Log`, an `Eval` for each of its evals, `Checkpoint` and
`Profile`, made from its ``log``, ``eval``, ``checkpoint`` and ``profile`` settings.
Hooks run in order, so the run's own see a step first. When training stops early (``Trainer.stopping``,
which the launcher sets at SIGTERM), every hook's `Hook.stop` runs at the step it
stopped at, and the checkpoint hook saves it.
"""

import functools
import itertools
import random
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.experimental import multihost_utils
from jaxtyping import Array, Int32, Shaped

from bux import parallel
from bux.lang.printer import value_to_source
from bux.train import trace
from bux.train.data import Batch
from bux.train.objective import Metrics
from bux.train.optim import Schedule
from bux.train.sinks import Scalars

#: A function that returns a hook's scalars, which the loop calls once the next step
#: is launched. It may wait for the step that made them.
Later = Callable[[], Scalars]

if TYPE_CHECKING:
    # Quoted where used: jaxtyping, which checks annotations as the tests run, reads
    # them as a module loads, and trainer.py imports this one.
    from bux.train.trainer import Trainer


class Hook:
    """What the loop does after every `every` steps and after the last.

    A subclass sets `every` and overrides `__call__`, and any of `start`, `stop` and
    `close` it needs; `due` decides when it runs.

    Attributes:
        every: The steps between its runs.
    """

    every: int = 1

    def due(self, step: int, steps: int) -> bool:
        """Return whether the hook runs after `step` of a run of `steps`: every
        `every` steps, and after the last."""
        return step % self.every == 0 or step == steps

    def start(self, trainer: "Trainer") -> None:
        """Before the first step, or the step a run resumes at."""

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars | Later | None:
        """Run after a step that it is due at.

        Args:
            trainer: The trainer, its `step` the steps taken.
            metrics: What the step measured, still on the device: reading one waits
                for the step.

        Returns:
            Scalars for the sinks, a `Later` that gives them, or None for nothing.
        """
        raise NotImplementedError

    def stop(self, trainer: "Trainer") -> None:
        """When training stops before the last step, at the step it stopped at."""

    def close(self) -> None:
        """After the last step, or when training stops."""


class Log(Hook):
    """The step's metrics, with its learning rate `lr`, the tokens seen, `tokens`, and
    `tokens_per_s` since the last log; and `mfu`, the model FLOPs a second over the
    devices' `peak`, where the peak is known and the trainer counts its FLOPs
    (``Trainer.work``). Optax counts from 0: the step just taken is
    ``trainer.step - 1``.

    Args:
        every: The steps between logs.
        lr: The learning rate at each step, from 0 (`bux.train.optim.lr_schedule`).
        tokens: The tokens a step takes: ``batch * seq_len``.
        peak: The FLOPs a second of all the run's devices at their peak, or None
            where it is not known, which leaves `mfu` out.
    """

    def __init__(
        self, *, every: int, lr: Schedule, tokens: int, peak: float | None = None
    ):
        self.every, self.lr, self.tokens = every, lr, tokens  # tokens per step
        self.peak = peak
        self.clock, self.since, self.flops = time.perf_counter(), 0, 0.0

    def start(self, trainer: "Trainer") -> None:
        """Start the clock and the counts at the trainer's step."""
        self.clock, self.since = time.perf_counter(), trainer.step
        self.flops = trainer.flops

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Later:
        """Return the step's scalars, to read once the next step is launched. Its
        learning rate is computed now, on the GPU behind this step: computed as it is
        read, it would queue behind the next step, and reading it wait for that one
        too, leaving the GPU nothing to run as the host launches the one after."""
        lr = self.lr(trainer.step - 1)
        counted = trainer.work is not None
        return functools.partial(
            self._read, metrics | {"lr": lr}, trainer.step, trainer.flops, counted
        )

    def _read(
        self, metrics: Metrics, step: int, flops: float, counted: bool
    ) -> Scalars:
        """Return the scalars of `step`, which waits for it; `flops` the trainer's
        count then, which `counted` says it keeps."""
        scalars = {k: float(v) for k, v in metrics.items()}
        now = time.perf_counter()
        steps, seconds = step - self.since, now - self.clock
        scalars |= {
            "tokens": step * self.tokens,
            "tokens_per_s": steps * self.tokens / seconds,
        }
        if self.peak and counted:
            scalars["mfu"] = (flops - self.flops) / seconds / self.peak
        self.clock, self.since, self.flops = now, step, flops
        return scalars


class Eval(Hook):
    """Held-out metrics over the first `batches` batches of `data`, the same ones
    every time (``Trainer.evaluate``): the first head's loss, as ``<name>_loss``, and
    given each token id's bytes of text, its bits per byte, ``<name>_bpb``.

    Args:
        data: A function that returns the held-out batches from the first, each time
            it is called.
        every: The steps between evaluations.
        batches: The batches an evaluation reads.
        name: What the metrics' names start with.
        token_bytes: The bytes of text each id of the model's vocabulary stands for
            (``Tokenizer.token_bytes``); None leaves out bits per byte.

    Example::

        held_out = functools.partial(batches, eval_data, config, vocab, held_out=True)
        train(run, hooks=[Eval(held_out, every=500, batches=20, name="val")])
    """

    def __init__(
        self,
        data: Callable[[], Iterator[Batch]],
        *,
        every: int,
        batches: int,
        name: str = "eval",
        token_bytes: Int32[np.ndarray, " vocab"] | None = None,
    ):
        self.data, self.every, self.batches = data, every, batches
        self.name, self.token_bytes = name, token_bytes

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars:
        """Return the trainer's means on the batches, each under the eval's name:
        ``{"eval_loss": ..., "eval_bpb": ...}``."""
        held_out = itertools.islice(self.data(), self.batches)
        means = trainer.evaluate(held_out, self.token_bytes)
        return {f"{self.name}_{k}": v for k, v in means.items()}


class Checkpoint(Hook):
    """The trainer's state, saved under `folder` at its step, and at the step training
    stops at if it stops early; the `keep` most recent checkpoints stay.

    A checkpoint also keeps how the run in force lays out the state (``layout``), and
    a run resumes from one only with its arrays and their layout, else it fails before
    it reads one. Every process of a job saves its part, so `folder` must be one they
    all see: one that is not fails at once, where orbax would wait at the first save
    for the parts it cannot see until it timed out, twenty minutes later.

    A save that a crash left half written is deleted as a run starts, and the run saves
    again only past its step: orbax would save that step in a folder of the same name,
    which another node of many may still see as the one deleted, fail to write into,
    and hang the save.

    Args:
        folder: Where the checkpoints go, a folder each, by step.
        every: The steps between saves.
        keep: The most recent checkpoints to keep.

    Raises:
        ValueError: In a job of many processes, if they do not all see `folder` as
            one folder.

    Example::

        checkpoint = Checkpoint(Path("runs/mine/checkpoints"), every=1000, keep=2)
        checkpoint.restore(trainer)  # if there is one
        trainer.fit(data, steps, hooks=[checkpoint])
    """

    def __init__(self, folder: Path, *, every: int, keep: int):
        self.every, self.layout = every, layout()
        _check_shared(folder.resolve())
        self.unsaved_until = _drop_unfinished(folder.resolve())
        self.manager = ocp.CheckpointManager(
            folder.resolve(),
            options=ocp.CheckpointManagerOptions(max_to_keep=keep),
            item_handlers=ocp.StandardCheckpointHandler(),  # so it reads metadata
        )

    def restore(self, trainer: "Trainer") -> bool:
        """Load the latest checkpoint into `trainer`, if there is one, in the memory of
        the state it replaces: the steps, which donate it, then find the rest of the
        memory in one piece, not on either side of it.

        Returns:
            Whether there was a checkpoint to load; `trainer` is at its step if so.

        Raises:
            ValueError: If the checkpoint holds other arrays than the trainer's state,
                or was saved with another layout of it (`layout`).
        """
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
        """Start saving the trainer's state at its step, in the background."""
        if trainer.step <= self.unsaved_until:
            return
        state = ocp.args.StandardSave(trainer.state())
        self.manager.save(trainer.step, args=state, custom_metadata=self.layout)

    def stop(self, trainer: "Trainer") -> None:
        """Save the step training stopped at."""
        self(trainer, {})  # unless the step is saved already, which orbax skips

    def close(self) -> None:
        """Wait for the saves to finish."""
        self.manager.wait_until_finished()
        self.manager.close()


class Profile(Hook):
    """A jax.profiler trace, into `folder`, of the `steps` steps after step `start`.
    It waits for the step before it starts and stops, so the trace holds whole steps.

    On GPUs it then reports what a step of them spent in collectives, `comm_ms`, in
    collectives with no compute beside them, `exposed_comm_ms`, and running nothing,
    `idle_ms` (`bux.train.trace`).

    Args:
        folder: Where jax.profiler writes the trace.
        start: The step after which the trace starts.
        steps: The steps it traces.
    """

    def __init__(self, folder: Path, *, start: int, steps: int):
        self.folder, self.first, self.last = folder, start, start + steps
        self.tracing = False

    def due(self, step: int, steps: int) -> bool:
        """Return whether `step` is the one the trace starts or stops after."""
        return step in (self.first, self.last)

    def start(self, trainer: "Trainer") -> None:
        """Start the trace if it starts at the trainer's step."""
        self._at(trainer.step)  # step 0, or the step a run resumes at

    def __call__(self, trainer: "Trainer", metrics: Metrics) -> Scalars | None:
        """Start or stop the trace once the step is done; after the last step traced,
        return what the steps spent in communication, on GPUs."""
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
        """Stop the trace if it is running."""
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


def _drop_unfinished(folder: Path) -> int:
    """Delete the saves that a crash left half written, on the first process, and
    return the last of their steps, -1 if there are none, on every process once it
    has."""
    last = -1
    if jax.process_index() == 0 and folder.is_dir():
        for name in ocp.path.step.tmp_checkpoints(folder):
            shutil.rmtree(folder / name)
            last = max(last, int(name.split(".")[0]))  # 20.orbax-checkpoint-tmp
    if jax.process_count() > 1:
        last = int(multihost_utils.broadcast_one_to_all(np.int32(last)))
    return last


def _seen(folder: Path, pattern: str, n: int, wait: float = 10.0) -> int:
    """Return how many files in `folder` match `pattern`, once there are `n` or after
    `wait` seconds: a network filesystem may show another node's file a little late."""
    deadline = time.monotonic() + wait
    while len(found := list(folder.glob(pattern))) < n and time.monotonic() < deadline:
        time.sleep(0.5)
    return len(found)
