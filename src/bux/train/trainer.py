"""Trains a compiled Run with the reference Flax NNX backend, on the devices its mesh
takes (`bux.parallel`).

A run's directory, ``<out>/<name>-<id>``, holds::

    run.bux        the run as one self-contained file, which `bux train` takes as is
    run.json       its IR
    metrics.jsonl  a line of metrics every `log.every` steps, and at every eval
    histograms.jsonl  with observe(histograms=true), the histograms of each step
                   that observes the model
    checkpoints/   with a `checkpoint` setting; a run there resumes from the latest
    profile/       with a `profile` setting, a trace or report of its steps
    tensorboard/   with a tensorboard tracker, its event files
    wandb/         with a wandb tracker, its files, and wandb_id, the W&B run's id

Its name says which alternatives it took, and its id hashes what it computes, so a
changed setting gets a fresh directory and an unchanged one finds its checkpoints.

`train` puts a run together from blocks that each work alone: a `Trainer` (a
model, its optimizer and its objective), the run's batches, hooks for its log, eval
and checkpoints, and sinks for what the hooks return. A script can swap any of
them::

    trainer = Trainer(model, optax.adamw(1e-3), objective(run.train.loss, run.arch))
    trainer.fit(batches(...), steps=1000, hooks=[...], sinks=[Console("mine", 1000)])
"""

import contextlib
import functools
import math
import re
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int32, Shaped

from bux import kernels, parallel, passes
from bux.architecture import Model, Scanned, num_params
from bux.architecture import observe as observing
from bux.architecture.common.segments import Segments
from bux.architecture.logits import HeadOutput
from bux.architecture.precision import matmuls
from bux.architecture.registry import Loss, Stream
from bux.architecture.repeats import pipelined, remat
from bux.architecture.repeats.repeat import pipelined_repeat, scanned_variables
from bux.data import tokenizer
from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.lang.printer import config_to_source, run_to_source
from bux.parallel.gather import Axes
from bux.train import (
    curriculum,
    fitting,
    flops,
    interleave,
    observe,
    reorder,
    stages,
    trackers,
    xla,
)
from bux.train.config import (
    EvalConfig,
    PrecisionConfig,
    TrainConfig,
    evals,
    evaluated,
)
from bux.train.data import Batch, batches, prefetch, training_batches
from bux.train.hooks import Checkpoint, Eval, Hook, Later, Log, Profile
from bux.train.objective import (
    Metrics,
    Objective,
    Ratio,
    held_out,
    objective,
    reported,
    target_mask,
)
from bux.train.optim import lr_multiplier, lr_schedule, optimizer
from bux.train.sinks import (
    JSONL,
    Console,
    LatestMetrics,
    Scalars,
    Sink,
    echo,
    is_main_process,
)


def forward(model: Model, batch: Batch) -> list[HeadOutput]:
    """Return every head's output on `batch`, each document of a row on its own.

    Not a training forward (`training_forward`): layers report no losses and update
    no state of their own, as an MoE does its router bias.

    Args:
        model: The model.
        batch: Its rows; their positions say where documents start.

    Returns:
        Each head's output in execution order. Its ``logits()`` are ``b t vocab``
        (rows, positions, the vocabulary).

    Example::

        logits = forward(model, plain(ids))[0].logits()
    """
    return model.heads(jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids))


def training_forward(model: Model, batch: Batch) -> tuple[list[HeadOutput], list[Loss]]:
    """Return `forward`'s outputs in a training step, and the losses layers report."""
    tokens, positions = jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids)
    return model.training_heads(tokens, positions)


def cast[M: nnx.Module](model: M, dtype: DTypeLike) -> M:
    """Return a copy of `model` with its floating-point parameters in `dtype` and the
    rest of its state as it is. Gradients through it reach the parameters in their
    own dtype. A scanned repeat's stay as they are, and each pass casts its own as it
    runs (``Scanned.compute``)."""
    stacked = nnx.All(nnx.Param, scanned_variables(model))
    graphdef, kept, params, buffers = nnx.split(model, stacked, nnx.Param, ...)
    copy = nnx.merge(graphdef, kept, _cast_params(params, dtype), buffers)
    for _, node in nnx.iter_graph(copy):
        if isinstance(node, Scanned):
            node.compute = jnp.dtype(dtype)
    return copy


@runtime_checkable
class StepEndUpdate(Protocol):
    """A module that learns from the batch other than by gradients, as an MoE moves
    its router bias. A training step sets `update_at_step_end`: its microbatches'
    forwards only count what they see, and `step_end_update` applies it once after
    them. Outside a step, a training forward applies it at once."""

    update_at_step_end: bool

    def step_end_update(self) -> None:
        """Apply what the step's forwards counted, once, after them all."""
        ...


def apply_step_end_updates(model: nnx.Module) -> None:
    """Call every module's `step_end_update`, once per step, after all its forwards."""
    for _, module in nnx.iter_modules(model):
        if isinstance(module, StepEndUpdate):
            module.step_end_update()


def token_shares(batch: Batch, n: int, over: Axes = ()) -> Float[Array, " n"]:
    """Return each of `n` microbatches' share of the batch's target tokens (the first
    head's), or an even share if it has none. In a shard_map manual over the axes
    `over`, of the targets of every device's rows."""
    counts = parallel.microbatches(target_mask(batch), n).sum((1, 2))
    num_targets = jax.lax.psum(counts.sum(), over) if over else counts.sum()
    even = 1 / (n * _size(over))
    return jnp.where(num_targets > 0, counts / jnp.maximum(num_targets, 1), even)


def rows_independent(model: nnx.Module) -> bool:
    """Return whether `model` computes each row of a batch apart from the others: no
    module of it learns from the whole batch (``StepEndUpdate``), as an MoE's router
    does, whose balance loss reads it too."""
    return not any(isinstance(m, StepEndUpdate) for _, m in nnx.iter_modules(model))


def _size(axes: Axes) -> int:
    on = parallel.active()
    return 1 if on is None else on.size(*axes)


def _scaled(metrics: Metrics, share: float | Float[Array, ""]) -> Metrics:
    return {k: v * share for k, v in metrics.items()}


def _shard_state(state: nnx.State) -> nnx.State:
    """Constrain `state` to the shardings of its variables, on a mesh."""
    if parallel.active() is None:
        return state
    return jax.lax.with_sharding_constraint(state, parallel.shardings(state))


def _cast_params(params: nnx.State, dtype: DTypeLike) -> nnx.State:
    """Return `params` with their floating-point arrays in `dtype`."""
    return jax.tree.map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )


def _flat(state: nnx.State) -> dict[tuple, Shaped[Array, "..."]]:
    """Return the arrays of `state` by their paths."""
    return {p: v.get_value() for p, v in nnx.to_flat_state(state)}


def _filled(state: nnx.State, values: dict[tuple, Shaped[Array, "..."]]) -> nnx.State:
    """Return `state` holding `values`, by their paths."""
    flat = nnx.to_flat_state(state)
    return nnx.from_flat_state([(p, v.replace(values[p])) for p, v in flat])


def _by_spec(
    f: Callable[[P, Shaped[Array, "*s"]], Shaped[Array, "*w"]],
    specs: nnx.State,
    state: nnx.State,
) -> nnx.State:
    """Apply `f` to each variable's value of `state` and its spec in `specs`."""
    return jax.tree.map(lambda s, v: v.replace(f(s, v.get_value())), specs, state)


def _is_variable(x: object) -> bool:
    return isinstance(x, nnx.Variable)


def _variables(state: nnx.State) -> list[nnx.Variable]:
    return jax.tree.leaves(state, is_leaf=_is_variable)


# A variable's value: an array, or a pytree of them, as optax's MaskedNode is of none.
Value = Any


class _Split:
    """The model's and the optimizer's state, split once for the training steps: the
    structure of their graph, and its variables, whose values a step reads and
    replaces. nnx.jit walks the graph at every call, from it to its arrays and back:
    on a B300 node's host 20 ms a step of the 1.3B and 70 of Qwen3-30B-A3B, which the
    GPU waits for where XLA's launch of a step returns only near its end. The graph's
    own variables hold each step's state, so the model and the optimizer are always
    up to date."""

    def __init__(self, nodes: tuple[Model, nnx.Optimizer]):
        self.graphdef, state = nnx.split(nodes)
        self.variables: list[nnx.Variable] = _variables(state)
        self.structure = jax.tree.structure(state, is_leaf=_is_variable)

    def values(self) -> list[Value]:
        """Return the variables' values, as a step takes them: raw, as nnx.split
        gives them (bux sets no hooks on a variable's value), which reads and
        writes 4x quicker than through the hooks."""
        return [variable.get_raw_value() for variable in self.variables]

    def take(self, values: Sequence[Value]) -> None:
        """Have the variables hold `values`, the state a step leaves."""
        for variable, value in zip(self.variables, values, strict=True):
            variable.set_raw_value(value)

    def merged(self, values: Sequence[Value]) -> tuple[Model, nnx.Optimizer]:
        """Return a model and an optimizer of their own variables, which hold
        `values`: inside a step, as it traces."""
        own = [v.replace(x) for v, x in zip(self.variables, values, strict=True)]
        return nnx.merge(self.graphdef, jax.tree.unflatten(self.structure, own))

    def values_of(self, nodes: tuple[Model, nnx.Optimizer]) -> list[Value]:
        """Return the values of the model's and the optimizer's variables at a step's
        end, in the order of `values`. Raise if the step changed which they are."""
        state = nnx.state(nodes)
        if jax.tree.structure(state, is_leaf=_is_variable) != self.structure:
            raise ValueError(
                "a training step changes the values of the model's and the "
                "optimizer's variables, but not which variables they hold"
            )
        return [variable.get_raw_value() for variable in _variables(state)]


class _Micro(NamedTuple):
    """A pipeline's microbatches, each leaf (microbatches, rows, ...): the rows, and
    each row's share of the batch's target tokens."""

    batch: Batch
    shares: Float[Array, "m n"]


def _row_shares(batch: Batch) -> Float[Array, " b"]:
    """Return each row's share of the batch's target tokens (the first head's)."""
    counts = target_mask(batch).sum(1)
    return counts / jnp.maximum(counts.sum(), 1)


class _Cut:
    """A model cut at its pipelined repeat, for a pipeline that schedules its own
    backward (``parallel.scheduled``): the repeat's layers' parameters and state
    (``layers``), the parameters outside the repeat (`others`) and the rest of their
    state (`kept`), and the model's forward cut there, a microbatch at a time
    (``parts``). The cut model holds stand-ins for the repeat's variables, which it
    does not run (``pipelined.stream_into``, ``pipelined.leaving``)."""

    def __init__(self, model: Model, objective: Objective, all_microbatches: int):
        self.path, scanned = pipelined_repeat(model)
        self.objective = objective
        self.all_microbatches = all_microbatches  # of every device
        _check_outside(model, self._inside)
        split = nnx.split(
            model, nnx.All(nnx.Param, self._inside), self._inside, nnx.Param, ...
        )
        self.graphdef, self.piped, self.rest, self.others, self.kept = split
        self.stand_ins = parallel.pipeline.stand_ins((self.piped, self.rest))
        graphdef = nnx.graphdef(scanned.layers)
        self.stage = pipelined.Stage.of(graphdef, scanned.times, True, None)
        # The names of what the layers before the repeat observe, and those after it,
        # once `first` and `last` have traced.
        self.before: list[str] = []
        self.after: list[str] = []

    def _inside(self, path: tuple, _: object) -> bool:
        return path[: len(self.path)] == self.path

    def layers(self) -> tuple[nnx.State, nnx.State]:
        """Return the repeat's layers' parameters and other state, each leaf (rounds,
        layers, ...) with one round."""
        at = (*self.path, "layers")
        return _under(self.piped, at, 1), _under(self.rest, at, 1)

    def joined(self, layers: nnx.State, outside: nnx.State) -> nnx.State:
        """Return the model's state from the repeat's `layers`, by (rounds, layers,
        ...), and the rest of the model's."""
        inner = parallel.pipeline.first(layers)  # of the one round
        return nnx.merge_state(_prefixed((*self.path, "layers"), inner), outside)

    def parts(self) -> "parallel.scheduled.Parts":
        return parallel.scheduled.Parts(
            self.first,
            self.staged,
            self.last,
            lambda: self.stage.reports,
            self.observed,
        )

    def model(self, others: nnx.State, kept: nnx.State) -> Model:
        """Return the cut model of `others` and `kept`, its variables new ones of the
        trace it runs in: a forward changes them, as dropout counts its masks."""
        return nnx.merge(self.graphdef, *self.stand_ins, others, kept, copy=True)

    def _outside(self, model: Model) -> nnx.State:
        """Return the state of `model`'s layers outside the repeat, but parameters."""
        return nnx.state(model, nnx.All(nnx.Not(nnx.Param), nnx.Not(self._inside)))

    def first(
        self, others: nnx.State, kept: nnx.State, rows: _Micro
    ) -> tuple[Stream, nnx.State]:
        """Run a microbatch up to the repeat: the stream there, and the state of the
        layers before it as they leave it."""
        model, before = self.model(others, kept), []
        x = pipelined.stream_into(lambda: training_forward(model, rows.batch), before)
        self.before = [a.name for a in before if observing.is_observed(a.name)]
        return x, self._outside(model)

    def staged(
        self, params: nnx.State, rest: nnx.State, x: Stream, rows: _Micro
    ) -> tuple[Stream, nnx.State, Float[Array, "l k"]]:
        """Run a stage's share of the repeat on a microbatch, its documents apart."""
        positions, tokens = rows.batch.position_ids, rows.batch.input_ids
        local = pipelined.Rows(Segments.of(jnp.asarray(positions)), jnp.asarray(tokens))
        return self.stage(params, rest, x, local)

    def last(
        self, others: nnx.State, kept: nnx.State, rows: _Micro, y: Stream
    ) -> tuple[Float[Array, ""], Metrics, nnx.State]:
        """Run a microbatch from the stream leaving the repeat, `y`, to its share of
        the step's loss and that of its metrics, and the state of the layers outside
        the repeat as they leave it: the forward runs those before it too."""
        model = self.model(others, kept)
        with pipelined.leaving(y):
            heads, aux_losses = training_forward(model, rows.batch)
        per_token, per_row = self.objective(heads, rows.batch, reported(aux_losses))
        observed = [a for a in aux_losses if observing.is_observed(a.name)]
        self.after = [a.name for a in observed]
        share = rows.shares.sum()
        loss = share * per_token.loss + per_row.loss / self.all_microbatches
        metrics = _scaled(per_token.metrics, share)
        rows_metrics = per_row.metrics | self._numbered(observed, "after")
        metrics |= _scaled(rows_metrics, 1 / self.all_microbatches)
        return loss, metrics, self._outside(model)

    def observed(self, values: Float[Array, "n k"], names: list[str]) -> Metrics:
        """Return what the repeat's layers observed (``Parts.observed``), named as the
        whole model numbers them, a pass's slot of every stage a row of `values`: of
        the slots the stages run, not those an uneven one skips."""
        runs = parallel.pipeline.layers_run(self.stage.shares).reshape(-1)
        losses = [
            Loss(name, 0.0, v)
            for row, kept in zip(values, runs, strict=True)
            if kept
            for name, v in zip(names, row, strict=True)
        ]
        return self._numbered(losses, "repeat")

    def _numbered(self, losses: list[Loss], part: str) -> Metrics:
        """Return what the layers of `part` of the model observed, `losses`, named as
        a forward of the whole model numbers them (`bux.train.observe.numbered`).
        Nothing gets the values of the layers before the repeat, whose forward stops
        there; they are numbered all the same."""
        per_pass = [n for n, _ in self.stage.reports if observing.is_observed(n)]
        parts = {
            "before": self.before,
            "repeat": per_pass * sum(self.stage.shares),
            "after": self.after,
        }
        return observe.numbered(parts, part, losses)


def _check_outside(model: nnx.Module, inside: Callable[[tuple, object], bool]) -> None:
    """Raise if a layer outside the pipelined repeat learns from the whole batch: a
    scheduled pipeline runs those a microbatch at a time, on one stage."""
    on = parallel.active()
    method = on.methods.pipeline if on else "?"
    for path, module in nnx.iter_modules(model):
        if isinstance(module, StepEndUpdate) and not inside(path, module):
            raise ValueError(
                f"methods.pipeline = {method!r} runs the "
                "layers outside the pipelined repeat a microbatch at a time, and "
                f"{type(module).__name__} at {'.'.join(map(str, path))} learns from the "
                "whole batch: put it in the repeat, or use methods(pipeline=gpipe)"
            )


def _under(state: nnx.State, path: tuple, rounds: int) -> nnx.State:
    """Return the part of `state` under `path`, each leaf with `rounds` in front: none
    if it has nothing there."""
    for key in path:
        if key not in state:
            return nnx.State({})
        state = state[key]
    return jax.tree.map(lambda a: a.reshape(rounds, *a.shape), state)


def _prefixed(path: tuple, state: nnx.State) -> nnx.State:
    """Return `state` as the part under `path` of a larger one."""
    flat = nnx.to_flat_state(state)
    return nnx.from_flat_state([((*path, *p), v) for p, v in flat])


class Trainer:
    """A model, its optimizer, which applies `tx`, and the objective it minimizes;
    `step` counts the steps taken.

    `train_step` and `eval_step` are compiled once, so a subclass that overrides them
    changes what a step does (keeps an average of the weights, skips a spike) and
    keeps the loop, hooks and checkpoints. They get the model and the optimizer as
    arguments, and what `train_step` changes of them is the step's new state. The
    training step takes their variables' values, which the trainer keeps split:
    between steps a hook may change those values, as `load` does, but not which
    variables the model and the optimizer hold.

    Both compute in `compute`, the model's parameters cast to it inside the loss, so
    that the parameters, their gradients and the optimizer's state stay as they are
    (float32); and the projections inside blocks multiply as the precision `matmuls`
    says, or in the format it names (`bux.architecture.precision.matmuls`). At the
    step of a `switch`, the steps take up its precision, and compile again; so they
    do at a batch of a new shape, as a curriculum's phase or a batch ramp makes, and
    under remat `auto` choose their policy again, for a longer row may not fit under
    the one before's.

    Made inside `bux.parallel.use`, it puts its state on the shardings the layout
    gives it, and every step keeps it there. A step that runs its layers more than
    once, in `microbatches` or a pipeline's ticks, gathers its parameters once with
    `gather` ``"step"``, or as each layer runs with ``"layer"``. With `accumulate`
    ``"unreduced"`` its microbatches' gradients add up on each device and reduce
    once, in `reduce` (`bux.parallel.gather`), a pipeline's in float32; with
    ``"reduced"`` they reduce each microbatch; ``"auto"`` adds them up unreduced
    where the rows' axes span processes, and where that computes the same. With
    `interleave`, each microbatch's forward runs beside the backward of the one
    before, a piece of the model at a time (`bux.train.interleave`).

    With `donate`, a training step donates the model's and the optimizer's arrays, so
    that their new state takes the old one's memory: an array read from them before a
    step is gone after it, and one to keep is a copy (``np.asarray``).

    The training step compiles with XLA's `compiler_options` (`bux.train.xla`), and
    with its gradients' `reductions` started ``"early"`` (`bux.train.reorder`) or
    where XLA puts them (``"xla"``).

    Given the `work` of a step at each shape of batch (`bux.train.flops.Work`),
    `flops` adds up the model FLOPs of the steps taken, each at the keys its batch's
    queries see, for the log's MFU.

    Every `observe_every` steps, the step is one compiled apart that also measures the
    model by layer (`bux.train.observe`): its forward, gradient, parameters and
    update, and with `histograms`, histograms of the stream and of the gradient.
    The others run as they would without it. `timers` keeps the seconds the host
    spends in each part of the loop (`Timers`).

    Args:
        model: The model to train.
        tx: The optax update rule (`bux.train.optim.optimizer`).
        objective: What a step minimizes (`bux.train.objective.objective`).
        compute: The dtype the forward and backward compute in.
        matmuls: The precision whose formats the projections multiply in, or a
            format's name for all of them.
        switch: The step at which the steps take up another precision, and that
            precision (`bux.architecture.precision.matmuls.late`); None for none.
        microbatches: The microbatches a step accumulates its gradients over; 1
            under a pipeline, which takes them through its stages.
        interleave: Run each microbatch's forward beside the one before's backward.
        gather: ``"step"`` or ``"layer"``: when a step gathers what FSDP splits.
        accumulate: ``"auto"``, ``"unreduced"`` or ``"reduced"``: how microbatches'
            gradients add up.
        reduce: The dtype gradients are summed over the rows in; None is `compute`.
        donate: Donate the state's arrays to each training step.
        compiler_options: XLA's options for the training step (`bux.train.xla`).
        reductions: ``"early"`` or ``"xla"``: where the gradients' reductions start.
        work: A step's model FLOPs on a batch of so many rows of so many tokens
            (`step_flops`), for the log's MFU; None does not count them.
        observe_every: The steps between those that observe the model; None for
            none.
        histograms: Whether those steps take histograms too.

    Attributes:
        model: The model, whose variables hold the latest step's state.
        optimizer: The `flax.nnx.Optimizer` that applies `tx` to the parameters.
        step: The steps taken.
        flops: The model FLOPs of the steps taken, where `work` is given.
        stopping: Set it, from a signal handler say, to stop `fit` after this step.
        timers: The seconds the host spent in each part of the loop, since the log
            last read them.

    Raises:
        ValueError: If `interleave` is asked for under a pipeline, or `accumulate`
            ``"unreduced"`` or a `reduce` other than `compute` where they cannot
            compute the same as reducing each microbatch.

    Example::

        class SkipSpikes(Trainer):
            def train_step(self, model, optimizer, batch):
                grads, metrics = self.gradients(model, batch)
                norm = optax.tree.norm(grads)
                grads = jax.tree.map(lambda g: jnp.where(norm < 10, g, 0), grads)
                optimizer.update(model, grads)
                return metrics | {"grad_norm": norm}

        trainer = SkipSpikes(model, tx, objective(run.train.loss, run.arch))
        trainer.fit(data, steps=1000, sinks=[Console("mine", 1000)])
    """

    def __init__(
        self,
        model: Model,
        tx: optax.GradientTransformation,
        objective: Objective,
        *,
        compute: DTypeLike = jnp.float32,
        matmuls: PrecisionConfig | str = "compute",
        switch: tuple[int, PrecisionConfig] | None = None,
        microbatches: int = 1,
        interleave: bool = False,
        gather: str = "step",
        accumulate: str = "auto",
        reduce: DTypeLike | None = None,
        donate: bool = False,
        compiler_options: xla.Options | None = None,
        reductions: str = "early",
        work: Callable[[int, int], flops.Work] | None = None,
        observe_every: int | None = None,
        histograms: bool = False,
    ):
        if interleave and parallel.pipeline.pipelined():
            raise ValueError(
                "methods(interleave=true) runs a step's microbatches side by side, "
                "and a pipeline schedules its own: keep mesh.stage = 1"
            )
        self.model, self.objective = model, objective
        self.compute, self.microbatches = jnp.dtype(compute), microbatches
        self.interleave = interleave
        self.matmuls, self.switch, self.gather = matmuls, switch, gather
        self.reduce, self.accumulate = jnp.dtype(reduce or compute), accumulate
        self.reduces_once = self._reduces_once(model)
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        parallel.shard_state(self.model, self.optimizer)
        self.step = 0
        self.work, self.flops = work, 0.0
        self.observe_every, self.histograms = observe_every, histograms
        self.layers = observe.layers(model) if observe_every else {}
        self.timers = Timers()
        self.stopping = False  # set, by a signal handler say, to stop after this step
        self._donate = donate
        self._compiler_options = compiler_options or {}
        self._reductions = reductions
        self._shape: tuple[int, ...] | None = None  # of the batches the steps took
        self._compile()

    def _compile(self) -> None:
        """Make the compiled steps, which read `matmuls` as they trace. Under remat
        `auto`, the training step chooses its policy as it first compiles
        (``fitting``)."""
        passes.use(reorder.NAME, self._reductions == "early")
        self._split = _Split((self.model, self.optimizer))
        jit = fitting.Fitting if remat.setting() == "auto" else xla.jit
        self._train = jit(
            self._on_values(self._train_step_on_shards),
            self._compiler_options,
            donate_argnums=0 if self._donate else (),
        )
        self._observed: Any = None  # compiled at the first step that observes
        self._launched: set[bool] = set()  # which steps have compiled: observed or not
        self._eval = nnx.jit(self.eval_step)

    def observes(self, step: int) -> bool:
        """Return whether the step that takes the trainer to `step` observes the
        model (`bux.train.observe`): every `observe_every` steps."""
        every = self.observe_every
        return every is not None and step % every == 0

    def _observed_step(self) -> Any:
        """Return the compiled step that observes, made at its first call: under remat
        `auto`, under the policy the other step chose, or one it chooses itself."""
        if self._observed is None:
            step = self._on_values(self._observe_on_shards)
            chosen = getattr(self._train, "policy", None)
            if chosen is not None:
                step = _with_policy(chosen, step)
            auto = remat.setting() == "auto" and chosen is None
            self._observed = (fitting.Fitting if auto else xla.jit)(
                step, self._compiler_options, donate_argnums=0 if self._donate else ()
            )
        return self._observed

    def _observe_on_shards(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        """Take `train_step` as a step that observes does: with its forward observed,
        and what it measures of the gradient, the parameters and the update, by
        layer (`bux.train.observe.params`)."""
        before = nnx.to_pure_dict(nnx.state(model, nnx.Param))
        with observe.observing(self.histograms) as recorded:
            optimizer.tx = recorded.wrap(optimizer.tx)
            metrics = self._train_step_on_shards(model, optimizer, batch)
        grads, updates = recorded.grads, recorded.updates
        measured = observe.params(self.layers, before, grads, updates, self.histograms)
        return metrics | measured

    def _on_values(self, step: Callable[..., Metrics]) -> Callable[..., Any]:
        """Return `step`, a function of the model, the optimizer and a batch, as one
        of their variables' values (``_Split``) and the batch, which returns their
        new values and the step's metrics."""
        split = self._split

        @functools.wraps(step)
        def on_values(values: list[Value], batch: Batch) -> tuple[list[Value], Metrics]:
            model, optimizer = split.merged(values)
            metrics = step(model, optimizer, batch)
            return split.values_of((model, optimizer)), metrics

        return on_values

    def _reduces_once(self, model: Model) -> bool:
        """Return whether a step adds up its gradients unreduced (``_unreduced``, or a
        pipeline's ``gather.apart``): where `accumulate` or a `reduce` other than
        `compute` asks for it, and by "auto" where the step runs its layers more than
        once, the rows' axes span processes (``gather.rows_span_processes``) and that
        computes the same. Raise where it is asked for but would not."""
        in_float32 = self.reduce != self.compute
        asked = self.accumulate == "unreduced" or in_float32
        runs = self.microbatches > 1 or parallel.pipeline.pipelined()
        across = parallel.gather.rows_span_processes()
        auto = self.accumulate == "auto" and runs and across
        if not (asked or auto) or not parallel.gather.rows():
            return False  # with no axis splitting the rows, nothing reduces over them
        why = self._why_not_unreduced(model)
        if why is not None and asked:
            what = (
                f"precision(reduce={self.reduce.name}) reduces a step's gradients once"
                if in_float32
                else "methods(accumulate=unreduced) adds up each device's gradients"
            )
            raise ValueError(f"{what}, which {why}")
        return why is None

    def _why_not_unreduced(self, model: Model) -> str | None:
        """Return why a step could not add up its gradients unreduced, or None."""
        if self.accumulate == "reduced":
            return "methods(accumulate=reduced) reduces as the backward makes them"
        if self.gather != "step":
            return "needs the parameters gathered once a step: methods(gather=step)"
        scheduled = parallel.scheduled.active() is not None  # sums all in float32
        if (
            self.reduce != self.compute
            and parallel.pipeline.pipelined()
            and not scheduled
        ):
            return "a pipeline's embedding and head do not: they reduce in the backward"
        if not rows_independent(model):
            return (
                "a layer that learns from the whole batch, as the MoE's router does, "
                "cannot: methods(accumulate=reduced)"
            )
        return None

    def _train_step_on_shards(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        metrics = self.train_step(model, optimizer, batch)
        parallel.shard_state(model, optimizer)
        return metrics

    def train_step(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        """Take one optimizer step on `batch`, and return its metrics with
        `grad_norm`, the gradients' norm before clipping.

        It is traced and compiled once, inside the training step; override it to
        change what a step does.

        Args:
            model: The model, its parameters the step's to change.
            optimizer: Its optimizer, whose update is the step's new state.
            batch: The step's rows, on the devices.

        Returns:
            The step's metrics, still on the device. They are the objective's
            terms, the losses layers report, and `grad_norm`.
        """
        grads, metrics = self.gradients(model, batch)
        optimizer.update(model, grads)
        return metrics | {"grad_norm": optax.tree.norm(grads)}

    def gradients(self, model: Model, batch: Batch) -> tuple[nnx.State, Metrics]:
        """Return the loss's gradient on `batch` and its metrics, summed over the
        `microbatches`. The forwards update the model's buffers (its state that is not
        a parameter) in turn, and then the step-end updates run.

        For a `train_step` of a subclass: it takes the gradient as the trainer's own
        step does, by microbatches, a pipeline, or neither.

        Returns:
            The gradient of each parameter, sharded as the parameter is, and the
            metrics of the whole batch.
        """
        graphdef, params, buffers = nnx.split(model, nnx.Param, ...)
        if parallel.scheduled.active() is not None:
            grads, metrics, buffers = self._scheduled(graphdef, params, buffers, batch)
        elif self.reduces_once and not parallel.pipeline.pipelined():
            grads, metrics, buffers = self._unreduced(graphdef, params, buffers, batch)
        elif self.microbatches > 1:
            grads, metrics, buffers = self._reduced(graphdef, params, buffers, batch)
        else:
            grads, metrics, buffers = self._whole(graphdef, params, buffers, batch)
        nnx.update(model, buffers)
        apply_step_end_updates(model)
        observe.record_grads(grads)
        return grads, metrics

    def _whole(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return `_grads` of the whole batch at once. A pipeline that gathers once a
        step takes the parameters whole over FSDP's axes, in float32: its ticks add up
        their gradients in the dtype of what they read. One that reduces them once
        runs its stages manual over the rows' axes (``gather.apart``)."""
        if self.gather == "step" and parallel.pipeline.pipelined():
            params = parallel.gather.whole(params)
        apart = parallel.gather.rows() if self.reduces_once else ()
        with parallel.gather.apart(apart):
            grads, metrics, buffers = self._grads(
                graphdef, params, buffers, batch, token_share=1.0, row_share=1.0
            )
        return _shard_state(grads), metrics, buffers

    def _scheduled(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return `_grads` of the whole batch from a pipeline that schedules its own
        backward (``parallel.scheduled``), with the parameters cast to `compute` once,
        and with `gather` "step" gathered once too (``_gathered``). It runs the model
        a microbatch at a time, cut at the pipelined repeat (``_Cut``): the part
        before the repeat on the first stage, and the part after it, to the loss, on
        the last."""
        method = parallel.scheduled.active()
        assert method is not None
        n = method.table().microbatches
        apart = parallel.gather.rows() if self.reduces_once else ()
        if self.gather == "step":
            used = self._gathered(params)
        else:  # cast on their shards: nothing in the pipeline casts them
            used = _shard_state(_cast_params(params, self.compute))
        model = nnx.merge(graphdef, used, buffers)
        model.set_attributes(update_at_step_end=True, raise_if_not_found=False)
        cut = _Cut(model, self.objective, all_microbatches=n * _size(apart))
        shares = parallel.microbatches(_row_shares(batch), n)
        rows = _Micro(parallel.microbatches(batch, n), shares)
        tokens = jnp.asarray(batch.input_ids)
        with parallel.gather.apart(apart), matmuls.use(self.matmuls, tokens):
            out = method.gradients(
                cut.parts(), *cut.layers(), cut.others, cut.kept, rows
            )
        grads = cut.joined(out.params, out.others)
        return _shard_state(grads), out.metrics, cut.joined(out.rest, out.kept)

    def _gathered(self, params: nnx.State) -> nnx.State:
        """Return `params` as the microbatches use them: with `gather` "step", cast to
        `compute` on each device's shards and then gathered whole, once a step."""
        if self.gather != "step":
            return params
        return parallel.gather.whole(_shard_state(_cast_params(params, self.compute)))

    def _reduced(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return `_accumulate`'s sums, each microbatch's gradient reduced over the
        batch axes as the backward makes it, and added to its shards."""
        zeros = _shard_state(jax.tree.map(jnp.zeros_like, params))
        shares = token_shares(batch, self.microbatches)
        used, n = self._gathered(params), self.microbatches
        return self._accumulate(graphdef, used, buffers, batch, zeros, shares, n)

    def _unreduced(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return `_accumulate`'s sums, computed in a shard_map manual over the axes
        that split the rows: each device gathers its shards of the parameters once,
        in `compute`, adds up its own rows' gradients over the microbatches, and the
        sums reduce once, in `reduce`, each device keeping its shards."""
        over = parallel.gather.rows()
        splits = parallel.gather.splits(params, over)
        all_microbatches = self.microbatches * _size(over)

        def gathered(spec: P, x: Shaped[Array, "*s"]) -> Shaped[Array, "*w"]:
            return parallel.gather.gather(spec, x.astype(self.compute))

        def reduced(spec: P, g: Float[Array, "*w"]) -> Float[Array, "*s"]:
            summed = parallel.gather.scatter(spec, over, g, self.reduce)
            return summed.astype(g.dtype)

        def on_device(
            params: nnx.State, buffers: nnx.State, batch: Batch
        ) -> tuple[nnx.State, Metrics, nnx.State]:
            whole = _by_spec(gathered, splits, params)
            zeros = jax.tree.map(
                lambda w, p: jnp.zeros(w.shape, p.dtype), whole, params
            )
            shares = token_shares(batch, self.microbatches, over)
            own = parallel.gather.own_streams(buffers, over)
            grads, metrics, after = self._accumulate(
                graphdef, whole, own, batch, zeros, shares, all_microbatches
            )
            grads = _by_spec(reduced, splits, grads)
            after = parallel.gather.keys_of(buffers, after)
            return grads, jax.lax.psum(metrics, over), after

        return jax.shard_map(
            on_device,
            in_specs=(splits, P(), P(over)),
            out_specs=(splits, P(), P()),
            axis_names=set(over),
            check_vma=False,  # each device's gradients are its own, summed by hand
        )(params, buffers, batch)

    def _accumulate(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
        zeros: nnx.State,
        token_shares: Float[Array, " n"],
        all_microbatches: int,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Sum `_grads` over the microbatches of `batch` in a scan, into `zeros`, each
        one's per-token part scaled by its share of the batch's targets and its
        per-row part by 1 / `all_microbatches`, of every device: so the sums are
        the batch's, even when microbatches have unequal numbers of targets. With
        `interleave`, each one's forward runs beside the one before's backward."""
        if self.interleave and self.microbatches > 1:
            return self._interleaved(
                graphdef, params, buffers, batch, zeros, token_shares, all_microbatches
            )
        row_share = 1 / all_microbatches

        def add_microbatch(
            carry: tuple[nnx.State, nnx.State],
            xs: tuple[Batch, Float[Array, ""]],
        ) -> tuple[tuple[nnx.State, nnx.State], Metrics]:
            grad_sum, buffers = carry
            microbatch, token_share = xs
            grads, metrics, buffers = self._grads(
                graphdef, params, buffers, microbatch, token_share, row_share
            )
            return (parallel.gather.added(grad_sum, grads), buffers), metrics

        xs = (parallel.microbatches(batch, self.microbatches), token_shares)
        (grads, buffers), metrics = jax.lax.scan(add_microbatch, (zeros, buffers), xs)
        return grads, {k: v.sum() for k, v in metrics.items()}, buffers

    def _interleaved(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
        zeros: nnx.State,
        token_shares: Float[Array, " n"],
        all_microbatches: int,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return `_accumulate`'s sums, each microbatch's forward run beside the
        backward of the one before, a piece of the model at a time
        (``interleave``)."""
        model = nnx.merge(graphdef, params, buffers)
        hooks = interleave.Hooks(self._training_module, self._loss)
        step = interleave.Step(model, hooks, self.matmuls, 1 / all_microbatches)
        micro = parallel.microbatches(batch, self.microbatches)
        grads, metrics, after = step.gradients(
            _flat(params), _flat(buffers), _flat(zeros), micro, token_shares
        )
        return _filled(zeros, grads), metrics, _filled(buffers, after)

    def _grads(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
        token_share: float | Float[Array, ""],
        row_share: float,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return the loss's gradient on `batch`, computed in `compute`, its metrics,
        and the buffers after the forward, each part of the objective scaled by its
        share."""

        def loss_fn(params: nnx.State) -> tuple[Float[Array, ""], tuple[Metrics, Any]]:
            model = self._training_module(graphdef, params, buffers)
            with matmuls.use(self.matmuls, jnp.asarray(batch.input_ids)):
                heads, aux_losses = training_forward(model, batch)
            loss, metrics = self._loss(heads, batch, aux_losses, token_share, row_share)
            return loss, (metrics, nnx.state(model, nnx.Not(nnx.Param)))

        grads, (metrics, buffers) = jax.grad(loss_fn, has_aux=True)(params)
        return grads, metrics, buffers

    def _training_module[M: nnx.Module](
        self, graphdef: nnx.GraphDef[M], params: nnx.State, buffers: nnx.State
    ) -> M:
        """Return the module of `graphdef` and the states, for a training step's
        forward: its parameters cast to `compute`, its step-end updates kept for the
        step's end."""
        module = cast(nnx.merge(graphdef, params, buffers, copy=True), self.compute)
        module.set_attributes(update_at_step_end=True, raise_if_not_found=False)
        return module

    def _loss(
        self,
        heads: Sequence[HeadOutput],
        batch: Batch,
        aux_losses: Sequence[Loss],
        token_share: float | Float[Array, ""],
        row_share: float,
    ) -> tuple[Float[Array, ""], Metrics]:
        """Return a microbatch's share of the step's loss, and of its metrics: the
        objective's terms by its share of the batch's targets, and the losses layers
        report by its share of rows."""
        per_token, per_row = self.objective(heads, batch, aux_losses)
        loss = token_share * per_token.loss + row_share * per_row.loss
        metrics = _scaled(per_token.metrics, token_share)
        return loss, metrics | _scaled(per_row.metrics, row_share)

    def eval_step(
        self,
        model: Model,
        batch: Batch,
        nbytes: Int32[np.ndarray | Array, "b t"] | None = None,
    ) -> dict[str, Ratio]:
        """Return what an eval measures of the first head on `batch`, each metric a
        sum and what it is a mean of (`bux.train.objective.held_out`): its loss, and
        given each token's bytes of text, `nbytes`, its bits per byte. It computes in
        `compute`, its projections multiplied as the precision's `eval` says.

        It is compiled once, by `evaluate`; override it to change what an evaluation
        measures: every name it returns is a metric."""
        with matmuls.use(matmuls.evaluating(self.matmuls)):
            heads = forward(cast(model, self.compute), batch)
        return held_out(heads[0], batch, nbytes, self.objective.offset)

    def train_on_batch(self, batch: Batch) -> Metrics:
        """Take one training step on `batch`, and return its metrics.

        It puts the batch on the devices, runs the compiled step, which compiles at
        its first call, at a `switch` and at a batch of a new shape, and counts the
        step and its FLOPs. Every `observe_every` steps it runs the step that
        observes instead. It returns once the step is launched: reading a metric
        waits for it. `timers` counts the time it took as ``launch``, or as
        ``compile`` where the step compiled.

        Args:
            batch: The step's rows, on the host (`bux.train.data.batches`).

        Returns:
            The step's metrics (`train_step`), still on the device.
        """
        self._switch_if_due()
        self._compile_if_reshaped(batch.input_ids.shape)
        observes = self.observes(self.step + 1)
        step = self._observed_step() if observes else self._train
        on_devices = parallel.shard_batch(batch)
        part = "launch" if observes in self._launched else "compile"
        with self.timers(part):
            values, metrics = step(self._split.values(), on_devices)
        self._launched.add(observes)
        self._split.take(values)
        self.step += 1
        if self.work is not None:
            work = self.work(*batch.position_ids.shape)
            self.flops += work.of(rows_read(batch.position_ids))
        return metrics

    def lower(self, batch: Batch) -> jax.stages.Lowered:
        """Return the training step lowered for the trainer's state and `batch`: its
        ``compile().memory_analysis()`` says what a step holds, its ``as_text()``
        the program."""
        return self._train.lower(self._split.values(), batch)

    def _switch_if_due(self) -> None:
        if self.switch is not None and self.step >= self.switch[0]:
            self.matmuls, self.switch = self.switch[1], None
            self._compile()

    def _compile_if_reshaped(self, shape: tuple[int, ...]) -> None:
        """Make the steps again for batches of a new `shape`: under remat `auto`, so
        that they choose the policy that fits it."""
        if self._shape is not None and shape != self._shape:
            self._compile()
        self._shape = shape

    def evaluate(
        self,
        batches: Iterable[Batch],
        token_bytes: Int32[np.ndarray, " vocab"] | None = None,
    ) -> dict[str, float]:
        """Return the mean of each metric of `eval_step` over `batches`, in eval mode:
        the first head's `loss`, and given each token id's bytes of text, its bits
        per byte, `bpb`. A mean weighs every target alike, whichever batch it is in.
        Every batch's step is launched before any is read, so the GPU runs one as the
        host launches the next.

        Args:
            batches: The held-out batches, on the host; a finite number of them.
            token_bytes: The bytes of text each id of the model's vocabulary stands
                for (``Tokenizer.token_bytes``); None leaves `bpb` out.

        Returns:
            Each metric's mean, by name; NaN for a mean of nothing.

        Example::

            held_out = batches(eval_data, run.train, vocab, held_out=True)
            trainer.evaluate(itertools.islice(held_out, 10))  # {"loss": 3.21}
        """
        self.model.eval()
        try:
            sums = [
                self._eval(
                    self.model, *parallel.shard_batch(_with_bytes(b, token_bytes))
                )
                for b in batches
            ]
        finally:
            self.model.train()
        return _means(sums)

    def state(self) -> nnx.State:
        """What a checkpoint keeps: the model's state and the optimizer's."""
        return nnx.state((self.model, self.optimizer))

    def load(self, state: nnx.State | Mapping[str, Any], step: int) -> None:
        """Take up `state`, from `state()` or a checkpoint, at `step`.

        Args:
            state: The model's and the optimizer's state, as `state` returns it.
            step: The steps taken at that state.
        """
        nnx.update((self.model, self.optimizer), state)
        self.step = step

    def fit(
        self,
        data: Iterator[Batch],
        steps: int,
        hooks: Sequence[Hook] = (),
        sinks: Sequence[Sink] = (),
    ) -> None:
        """Train on `data` to step `steps`, or until `stopping` is set, after which
        every hook's `stop` runs; the next batches load as a step runs
        (`bux.train.data.prefetch`). After every step, the hooks due run in order, and
        what they return goes to every sink once the next step is launched
        (`bux.train.hooks.Later`). All of them close at the end, or when training
        stops. `timers` counts the host's time waiting for batches (``data``), in
        hooks (``hooks``), and reading what they return, which waits for the step
        before (``wait``).

        Args:
            data: The batches, on the host, from the trainer's step on.
            steps: The step to train to: a resumed trainer takes ``steps - step``.
            hooks: What runs between steps, in order (`bux.train.hooks`).
            sinks: Where what the hooks return goes (`bux.train.sinks`).

        Example::

            trainer.fit(
                batches(run.train.data, run.train, vocab, start=trainer.step),
                steps=1000,
                hooks=[Log(every=10, lr=lambda step: 1e-3, tokens=batch * seq_len)],
                sinks=[Console("mine", 1000)],
            )
        """
        with contextlib.ExitStack() as stack:
            for block in (*sinks, *hooks):
                stack.callback(block.close)
            for hook in hooks:
                hook.start(self)
            loaded = stack.enter_context(contextlib.closing(prefetch(data)))
            reported: list[tuple[int, Scalars | Later]] = []
            while self.step < steps and not self.stopping:
                with self.timers("data"):
                    batch = next(loaded)
                metrics = self.train_on_batch(batch)
                with self.timers("wait"):
                    _write(reported, sinks)  # the step before's, as this one runs
                with self.timers("hooks"):
                    reported = [
                        (self.step, out)
                        for hook in hooks
                        if hook.due(self.step, steps) and (out := hook(self, metrics))
                    ]
            _write(reported, sinks)
            if self.step < steps:
                for hook in hooks:
                    hook.stop(self)


def _with_bytes(
    batch: Batch, token_bytes: Int32[np.ndarray, " vocab"] | None
) -> tuple[Batch, Int32[np.ndarray, "b t"] | None]:
    """Return `batch`, and the bytes of text each of its tokens stands for."""
    return batch, None if token_bytes is None else token_bytes[batch.input_ids]


def _means(sums: Sequence[Mapping[str, Ratio]]) -> dict[str, float]:
    """Return each metric's sums added up over the batches, the first over the
    second: a mean over all of them. Reading them waits for each batch's step."""
    means = {}
    for name in sums[0]:
        total, count = (sum(float(s[name][i]) for s in sums) for i in (0, 1))
        means[name] = total / count if count else math.nan
    return means


def _write(reported: list[tuple[int, Scalars | Later]], sinks: Sequence[Sink]) -> None:
    """Give every sink what the hooks reported at each step, read now where it is
    `Later`: the histograms whose bins it holds apart (`Sink.histograms`)."""
    for step, out in reported:
        scalars, histograms = observe.histograms_of(
            out if isinstance(out, Mapping) else out()
        )
        for sink in sinks:
            sink.write(step, scalars)
            if histograms:
                sink.histograms(step, histograms)


def _with_policy(policy: str, step: Callable[..., Any]) -> Callable[..., Any]:
    """Return `step`, traced under the remat `policy`, by its own name."""

    @functools.wraps(step)
    def under(*args: Any) -> Any:
        with remat.trying(policy):
            return step(*args)

    return under


class Timers:
    """The seconds the host spends in each part of the training loop, by name, since
    they were last read: ``with trainer.timers("data"): ...``."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str) -> Generator[None]:
        """Count the block's time as `name`'s."""
        start = time.perf_counter()
        try:
            yield
        finally:
            spent = time.perf_counter() - start
            self.seconds[name] = self.seconds.get(name, 0.0) + spent

    def read(self) -> dict[str, float]:
        """Return the seconds by name since the last read, and count from zero."""
        seconds, self.seconds = self.seconds, {}
        return seconds


def directory(run: ir.Run, out: str | Path) -> Path:
    """Return the run's directory under `out`: ``<name>-<id>``, its name with what a
    path cannot hold as ``_``."""
    slug = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", run.name).strip("_")
    return Path(out) / f"{slug}-{run.id}"


def train(
    run: ir.Run,
    out: str | Path = "runs",
    *,
    cache: str | Path = DEFAULT_CACHE,
    echo: Callable[[str], None] = echo,
    hooks: Sequence[Hook] = (),
    sinks: Sequence[Sink] = (),
) -> dict[str, float]:
    """Train `run` to its last step, writing to its directory under `out`, after
    preparing the datasets it reads under `cache`; return the latest value of every
    metric. `hooks` run after the run's own, and `sinks` get what all of them return,
    as the console and metrics.jsonl do.

    It is what ``bux train`` runs for each run of a file. It writes run.bux and
    run.json into the run's directory (`directory`), builds the model on the run's
    mesh, resumes from the latest checkpoint there if the run has a ``checkpoint``
    setting, and trains with the run's own hooks: its log, evals, checkpoint and
    profile; every ``observe.every`` steps with a step that observes the model
    (`bux.train.observe`). Its sinks are the console, metrics.jsonl (and
    histograms.jsonl) and the latest metrics, then the ``log.trackers``
    (`bux.train.trackers`).

    Args:
        run: A compiled run (`bux.lang.compile_runs`).
        out: The folder of runs' directories.
        cache: Where prepared datasets go and are found (`bux.data.pipeline`).
        echo: What prints the run's lines: what it trains, what it resumed from,
            and its log; each flushed as it is printed, by default.
        hooks: More hooks, after the run's own (`bux.train.hooks`).
        sinks: More sinks, after the console, metrics.jsonl and the latest metrics
            (`bux.train.sinks`).

    Returns:
        The latest value of every metric the hooks reported, by name, as `loss`,
        `eval_loss`, `eval_bpb` and `tokens_per_s`.

    Raises:
        ValueError: If the model has no vocabulary, the batch does not split over the
            mesh and the microbatches, or a setting cannot run as asked.

    Example::

        from bux.lang import compile_runs
        from bux.train.trainer import train

        source = open("examples/experiments/smoke.bux").read()
        [run] = compile_runs(source, filename="smoke.bux", overrides=["steps=10"])
        metrics = train(run, out="runs")
    """
    config, vocab = run.train, run.arch.vocab_size
    if vocab is None:
        raise ValueError("a model without a vocabulary cannot be trained on tokens")
    for job in jobs(config):
        prepare(job, cache, echo=echo)
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (folder / "run.bux").write_text(run_to_source(run), encoding="utf-8")
        (folder / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )

    multiplier = lr_multiplier(config)
    lr, peak = lr_schedule(config.optimizer.lr, multiplier), _peak(run)
    stretches = curriculum.stretches(config)
    tokens = functools.partial(curriculum.tokens, stretches)
    log = Log(every=config.log.every, lr=lr, tokens=tokens, peak=peak)
    work = functools.cache(functools.partial(step_flops, run)) if peak else None
    methods = stages.balanced(run)
    if methods.stages != config.methods.stages:
        echo(f"{run.name}: methods.stages = {methods.stages}")
    with parallel.use(config.mesh, config.layout, methods):
        _check_shapes(stretches, config)
        _check_cpu_bfloat16(run)
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(config.seed)))
        parallel.methods.prepare(run)
        precision = config.precision
        tx = optimizer(
            config.optimizer, multiplier, model, config.clip, precision.moments
        )
        trainer = Trainer(
            model,
            tx,
            objective(config.loss, run.arch),
            compute=precision.compute,
            matmuls=precision,
            switch=matmuls.late(precision, config.steps),
            microbatches=_accumulated(config.methods.microbatches),
            interleave=config.methods.interleave,
            gather=config.methods.gather,
            accumulate=config.methods.accumulate,
            reduce=_reduce_dtype(config.precision),
            donate=config.methods.donate,
            compiler_options=xla.options(config.compiler),
            reductions=config.compiler.reductions,
            work=work,
            observe_every=None if config.observe is None else config.observe.every,
            histograms=config.observe is not None and config.observe.histograms,
        )
        make_batches = functools.partial(
            batches, config=config, vocab_size=vocab, cache=cache
        )
        run_hooks = _hooks(run, trainer, folder, log, make_batches, echo)
        for line in _about(run, folder, trainer):
            echo(f"{run.name}: {line}")
        latest, start = LatestMetrics(), trainer.step
        histograms = folder / "histograms.jsonl"
        trainer.fit(
            training_batches(config, vocab, cache=cache, start=start),
            config.steps,
            hooks=[*run_hooks, *hooks],
            sinks=[
                Console(run.name, config.steps, echo, _formats(config)),
                JSONL(folder / "metrics.jsonl", start=start, histograms=histograms),
                latest,
                *trackers.build(run, folder, start),
                *sinks,
            ],
        )
    return latest.metrics


def _about(run: ir.Run, folder: Path, trainer: Trainer) -> list[str]:
    """Return the lines that say what a run trains, before it does: where it writes,
    its model, its steps and tokens, its phases, and what it logs and saves."""
    t = run.train
    devices = parallel.mesh(t.mesh).devices
    params = _si(num_params(trainer.model))
    how = f"{config_to_source(t.optimizer)}, {config_to_source(t.schedule)}"
    tokens = _si(curriculum.tokens(curriculum.stretches(t), t.steps))
    shape = "" if curriculum.varies(t) else f" × {t.batch} × {t.seq_len} tokens"
    lines = [
        f"{folder}, on {devices.size} × {devices.flat[0].device_kind}",
        f"{params} parameters, {t.steps:,} steps{shape} = {tokens} tokens, {how}",
        *curriculum.lines(t),
    ]
    watched = [f"log every {t.log.every}"]
    if t.observe is not None:
        more = " with histograms" if t.observe.histograms else ""
        watched.append(f"observe every {t.observe.every}{more}")
    watched += [f"{e.name} every {e.every}{_eval_shape(t, e)}" for e in evals(t)]
    if t.checkpoint is not None:
        watched.append(f"checkpoint every {t.checkpoint.every}")
    watched += [config_to_source(c) for c in t.log.trackers]
    return [*lines, ", ".join(watched)]


def _si(n: float) -> str:
    """Return `n` with an SI suffix, as ``bux plan`` shows it: 1.3B, 25.6k."""
    for suffix, scale in (("", 1), ("k", 1e3), ("M", 1e6), ("B", 1e9)):
        if "e" not in (shown := f"{n / scale:.3g}"):
            return shown + suffix
    return f"{n / 1e12:.3g}T"


def _check_cpu_bfloat16(run: ir.Run) -> None:
    """Raise where XLA:CPU would abort: it cannot compile a bfloat16 psum that a
    shard_map's transpose makes, as the MoE's methods but `dense`, the pipelines and
    sparse attention split over `context` do on more than one device."""
    on, config = parallel.active(), run.train
    if config.precision.compute != "bfloat16" or jax.default_backend() != "cpu":
        return
    if on is None or on.mesh.size == 1:
        return
    types = {layer.config.type for layer in run.arch.layers()}
    if "moe" in types and config.methods.moe != "dense":
        what = f"an MoE by methods(moe={config.methods.moe})"
    elif parallel.pipeline.pipelined():
        what = "a pipeline"
    elif types & {"indexer", "csa"} and on.size("context") > 1:
        what = "sparse attention split over context"
    else:
        return
    raise ValueError(
        f"XLA:CPU cannot train {what} in bfloat16 on more than one device: set "
        "precision(compute=float32), or methods(moe=dense) for an MoE"
    )


def _eval_shape(t: TrainConfig, e: EvalConfig) -> str:
    """Return the shape of eval `e`'s batches, ``" of 16 × 16384 tokens"``, where it
    is its own; else nothing."""
    shape = evaluated(t, e)
    return "" if shape is t else f" of {shape.batch} × {shape.seq_len} tokens"


def _check_shapes(stretches: Sequence[curriculum.Stretch], config: TrainConfig) -> None:
    """Raise unless every shape of batch the run takes, and each eval of a shape of
    its own, splits over the mesh and the microbatches (``parallel.check``), saying
    from which step, or which eval, where it is not the first's."""
    for s in stretches:
        try:
            parallel.check(s.config.batch, s.config.seq_len)
        except ValueError as e:
            raise ValueError(
                f"from step {s.start}, {e}" if s.start else str(e)
            ) from None
    for e in evals(config):
        shape = evaluated(config, e)
        if shape is config:
            continue
        try:
            parallel.check(shape.batch, shape.seq_len)
        except ValueError as err:
            raise ValueError(f"eval {e.name}: {err}") from None


def _reduce_dtype(precision: PrecisionConfig) -> str:
    """Return the dtype `precision` sums gradients over the batch's rows in."""
    return precision.compute if precision.reduce == "compute" else precision.reduce


def _accumulated(microbatches: int) -> int:
    """Return how many microbatches a step accumulates gradients over: one under a
    pipeline, which takes them through its stages instead."""
    return 1 if parallel.pipeline.pipelined() else microbatches


def rows_read(
    positions: Int32[np.ndarray | Array, "b t"],
) -> Int32[np.ndarray | Array, "n t"]:
    """Return the rows of a batch's `positions` that this process read: in a job of
    many, each reads its own and the others stay 0 (``data.batches``), which would
    count as documents of a token each."""
    return positions[parallel.local_rows(*positions.shape)]


def step_flops(
    run: ir.Run, rows: int | None = None, seq_len: int | None = None
) -> flops.Work:
    """Return the model FLOPs of a training step of `run`: its loss's gradient on a
    batch, counted from its program (`bux.train.flops.count`) on one device, outside
    any mesh, with remat and kernels off, which would add work or hide it.

    Nothing runs: the step is traced on shapes alone. Attention counts only the keys
    each query sees, so the FLOPs of a step depend on its batch's documents:
    ``step_flops(run).of(batch.position_ids)``.

    Args:
        run: A compiled run.
        rows: The batch's rows; None: the run's ``batch``.
        seq_len: The tokens of each; None: the run's ``seq_len``.

    Returns:
        The step's FLOPs, those of every batch and attention's by its mask.

    Example::

        work = step_flops(run)
        work.one_document(run.train.seq_len)  # a step's FLOPs, a document a row
    """
    t = run.train
    rows, length = rows or t.batch, seq_len or t.seq_len
    loss = objective(t.loss, run.arch)

    def gradient(
        params: nnx.State, rest: nnx.State, ids: Int32[Array, "b t"]
    ) -> nnx.State:
        positions = jnp.arange(length, dtype=jnp.int32)
        batch = Batch(ids, jnp.broadcast_to(positions, ids.shape), ids >= 0)

        def f(params: nnx.State) -> Float[Array, ""]:
            model = nnx.merge(graphdef, params, rest, copy=True)
            model = cast(model, t.precision.compute)
            model.set_attributes(update_at_step_end=True, raise_if_not_found=False)
            heads, losses = training_forward(model, batch)
            per_token, per_row = loss(heads, batch, losses)
            return per_token.loss + per_row.loss

        return jax.grad(f)(params)

    with remat.off(), kernels.off():
        abstract = nnx.eval_shape(lambda: Model(run.arch, rngs=nnx.Rngs(t.seed)))
        graphdef, params, rest = nnx.split(abstract, nnx.Param, ...)
        ids = jax.ShapeDtypeStruct((rows, length), jnp.int32)
        return flops.count(gradient, params, rest, ids)


def _peak(run: ir.Run) -> float | None:
    """Return the FLOPs a second of the run's devices at their peak, if known."""
    devices = parallel.mesh(run.train.mesh).devices
    each = flops.peak(devices.flat[0].device_kind)
    return each and each * devices.size


def _hooks(
    run: ir.Run,
    trainer: Trainer,
    folder: Path,
    log: Log,
    make_batches: Callable[..., Iterator[Batch]],
    echo: Callable[[str], None],
) -> list[Hook]:
    """Return the run's own hooks: its `log`, evals, checkpoint, which resumes `trainer`
    from the latest checkpoint in `folder` if there is one, and profile."""
    config = run.train
    hooks: list[Hook] = [log]
    sets = evals(config)
    token_bytes = _token_bytes(config, run.arch.vocab_size) if sets else None
    for e in sets:
        shape = evaluated(config, e)
        data = functools.partial(make_batches, e.data, config=shape, held_out=True)
        hooks.append(
            Eval(
                data,
                every=e.every,
                batches=e.batches,
                name=e.name,
                token_bytes=token_bytes,
            )
        )
    if (saves := config.checkpoint) is not None:
        checkpoint = Checkpoint(
            folder / "checkpoints", every=saves.every, keep=saves.keep
        )
        start = time.perf_counter()
        if checkpoint.restore(trainer):
            seconds = time.perf_counter() - start
            echo(
                f"{run.name}: restored in {seconds:.1f}s, resuming at step {trainer.step}"
            )
        hooks.append(checkpoint)
    if (trace := config.profile) is not None:
        profile = Profile(
            folder / "profile", start=trace.start, steps=trace.steps, tool=trace.tool
        )
        hooks.append(profile)
    return hooks


def _token_bytes(
    config: TrainConfig, vocab_size: int | None
) -> Int32[np.ndarray, " vocab"] | None:
    """Return the bytes of text each id of the model's vocabulary stands for, by the
    run's tokenizer, 0 past its ids; None without a tokenizer."""
    if config.tokenizer is None:
        return None
    lengths = tokenizer.load(config.tokenizer).token_bytes()
    return np.pad(lengths, (0, max(0, (vocab_size or 0) - len(lengths))))


def _formats(config: TrainConfig) -> dict[str, str]:
    """Return the metrics the console shows, `Console.FORMATS` and each eval's."""
    formats = dict(Console.FORMATS)
    for e in evals(config):
        formats[f"{e.name}_loss"] = f"{e.name} loss {{:.4f}}"
        formats[f"{e.name}_bpb"] = f"{e.name} bpb {{:.4f}}"
    return formats
