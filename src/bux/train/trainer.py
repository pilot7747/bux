"""Trains a compiled Run with the reference Flax NNX backend, on the devices its mesh
takes (``bux.parallel``).

A run's directory, ``<out>/<name>-<id>``, holds:

    run.bux        the run as one self-contained file, which `bux train` takes as is
    run.json       its IR
    metrics.jsonl  a line of metrics every `log.every` steps, and at every eval
    checkpoints/   with a `checkpoint` setting; a run there resumes from the latest
    profile/       with a `profile` setting, a trace of its steps

Its name says which alternatives it took, and its id hashes what it computes, so a
changed setting gets a fresh directory and an unchanged one finds its checkpoints.

``train`` puts a run together from blocks that each work alone: a ``Trainer`` (a
model, its optimizer and its objective), the run's batches, hooks for its log, eval
and checkpoints, and sinks for what the hooks return. A script can swap any of them:

    trainer = Trainer(model, optax.adamw(1e-3), objective(run.train.loss, run.arch))
    trainer.fit(batches(...), steps=1000, hooks=[...], sinks=[Console("mine", 1000)])
"""

import contextlib
import functools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int32, Shaped

from bux import kernels, parallel
from bux.architecture import Model, Scanned
from bux.architecture.common.segments import Segments
from bux.architecture.logits import HeadOutput
from bux.architecture.precision import matmuls
from bux.architecture.registry import Loss, Stream
from bux.architecture.repeats import pipelined, remat
from bux.architecture.repeats.repeat import scanned_variables
from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.lang.printer import run_to_source
from bux.parallel.gather import Axes
from bux.train import fitting, flops, stages, xla
from bux.train.config import PrecisionConfig
from bux.train.data import Batch, batches, prefetch
from bux.train.hooks import Checkpoint, Eval, Hook, Log, Profile
from bux.train.objective import Metrics, Objective, objective, target_mask
from bux.train.optim import Schedule, lr_multiplier, lr_schedule, optimizer
from bux.train.sinks import JSONL, Console, LatestMetrics, Sink, is_main_process


def forward(model: Model, batch: Batch) -> list[HeadOutput]:
    """Return every head's output on `batch`, each document of a row on its own."""
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
    params = jax.tree.map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    copy = nnx.merge(graphdef, kept, params, buffers)
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

    def step_end_update(self) -> None: ...


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
    return jax.tree.map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )


def _add(total: Float[Array, "*s"], part: Float[Array, "*s"]) -> Float[Array, "*s"]:
    return total + part.astype(total.dtype)


def _by_spec(
    f: Callable[[P, Shaped[Array, "*s"]], Shaped[Array, "*w"]],
    specs: nnx.State,
    state: nnx.State,
) -> nnx.State:
    """Apply `f` to each variable's value of `state` and its spec in `specs`."""
    return jax.tree.map(lambda s, v: v.replace(f(s, v.get_value())), specs, state)


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

    def __init__(self, model: Model, objective: Objective, rows: int):
        [(self.path, scanned)] = [
            (p, m)
            for p, m in nnx.iter_graph(model)
            if isinstance(m, Scanned) and m.rounds
        ]
        self.objective, self.rows = objective, rows  # the microbatches of every device
        _check_outside(model, self._inside)
        split = nnx.split(
            model, nnx.All(nnx.Param, self._inside), self._inside, nnx.Param, ...
        )
        self.graphdef, self.piped, self.rest, self.others, self.kept = split
        self.stand_ins = jax.tree.map(_stand_in, (self.piped, self.rest))
        graphdef = nnx.graphdef(scanned.layers)
        self.stage = pipelined.Stage.of(graphdef, scanned.times, True, None)

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
        inner = jax.tree.map(lambda a: a[0], layers)
        return nnx.merge_state(_prefixed((*self.path, "layers"), inner), outside)

    def parts(self) -> "parallel.scheduled.Parts":
        return parallel.scheduled.Parts(
            self.first, self.staged, self.last, lambda: self.stage.reports
        )

    def model(self, others: nnx.State, kept: nnx.State) -> Model:
        return nnx.merge(self.graphdef, *self.stand_ins, others, kept)

    def first(
        self, others: nnx.State, kept: nnx.State, rows: _Micro
    ) -> tuple[Stream, nnx.State]:
        """Run a microbatch up to the repeat: the stream there, and the state of the
        layers before it as they leave it."""
        model = self.model(others, kept)
        x = pipelined.stream_into(lambda: training_forward(model, rows.batch))
        return x, nnx.state(model, nnx.All(nnx.Not(nnx.Param), nnx.Not(self._inside)))

    def staged(
        self, params: nnx.State, rest: nnx.State, x: Stream, rows: _Micro
    ) -> tuple[Stream, nnx.State, Float[Array, "l k"]]:
        """Run a stage's share of the repeat on a microbatch, its documents apart."""
        positions, tokens = rows.batch.position_ids, rows.batch.input_ids
        local = pipelined.Rows(Segments.of(jnp.asarray(positions)), jnp.asarray(tokens))
        return self.stage(params, rest, x, local)

    def last(
        self, others: nnx.State, kept: nnx.State, rows: _Micro, y: Stream
    ) -> tuple[Float[Array, ""], Metrics]:
        """Run a microbatch from the stream leaving the repeat, `y`, to its share of
        the step's loss, and that of its metrics."""
        with pipelined.leaving(y):
            heads, aux_losses = training_forward(self.model(others, kept), rows.batch)
        per_token, per_row = self.objective(heads, rows.batch, aux_losses)
        share = rows.shares.sum()
        loss = share * per_token.loss + per_row.loss / self.rows
        metrics = _scaled(per_token.metrics, share)
        return loss, metrics | _scaled(per_row.metrics, 1 / self.rows)


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


def _stand_in(a: Shaped[Array, "..."]) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(a.shape, a.dtype)


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
    arguments, which is how nnx.jit sees them change.

    Both compute in `compute`, the model's parameters cast to it inside the loss, so
    that the parameters, their gradients and the optimizer's state stay as they are
    (float32); and the projections inside blocks multiply as the precision `matmuls`
    says, or in the format it names (``bux.architecture.precision.matmuls``). At the
    step of a `switch`, the steps take up its precision, and compile again.

    Made inside ``parallel.use``, it puts its state on the shardings the layout gives
    it, and every step keeps it there. A step that runs its layers more than once, in
    `microbatches` or a pipeline's ticks, gathers its parameters once with `gather`
    "step", or as each layer runs with "layer". With `accumulate` "unreduced" its
    microbatches' gradients add up on each device and reduce once, in `reduce`
    (``parallel.gather``), a pipeline's in float32; with "reduced" they reduce each
    microbatch; "auto" adds them up unreduced where the rows' axes span processes,
    and where that computes the same, as `_why_not_unreduced` says.

    With `donate`, a training step donates the model's and the optimizer's arrays, so
    that their new state takes the old one's memory: an array read from them before a
    step is gone after it, and one to keep is a copy (``np.asarray``).

    The training step compiles with XLA's `compiler_options` (``bux.train.xla``).
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
        gather: str = "step",
        accumulate: str = "auto",
        reduce: DTypeLike | None = None,
        donate: bool = False,
        compiler_options: xla.Options | None = None,
    ):
        self.model, self.objective = model, objective
        self.compute, self.microbatches = jnp.dtype(compute), microbatches
        self.matmuls, self.switch, self.gather = matmuls, switch, gather
        self.reduce, self.accumulate = jnp.dtype(reduce or compute), accumulate
        self.reduces_once = self._reduces_once(model)
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        parallel.shard_state(self.model, self.optimizer)
        self.step = 0
        self.stopping = False  # set, by a signal handler say, to stop after this step
        self._donated = ("model", "optimizer") if donate else ()
        self._compiler_options = compiler_options or {}
        self._compile()

    def _compile(self) -> None:
        """Make the compiled steps, which read `matmuls` as they trace. Under remat
        `auto`, the training step chooses its policy as it first compiles
        (``fitting``)."""
        jit = fitting.Fitting if remat.setting() == "auto" else xla.jit
        self._train = jit(
            self._train_step_on_shards,
            self._compiler_options,
            donate_argnames=self._donated,
        )
        self._eval = nnx.jit(self.eval_step)

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
        `grad_norm`, the gradients' norm before clipping."""
        grads, metrics = self.gradients(model, batch)
        optimizer.update(model, grads)
        return metrics | {"grad_norm": optax.tree.norm(grads)}

    def gradients(self, model: Model, batch: Batch) -> tuple[nnx.State, Metrics]:
        """Return the loss's gradient on `batch` and its metrics, summed over the
        `microbatches`. The forwards update the model's buffers (its state that is not
        a parameter) in turn, and then the step-end updates run."""
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
        backward (``parallel.scheduled``), with the parameters cast and gathered once
        (``_gathered``). It runs the model a microbatch at a time, cut at the
        pipelined repeat (``_Cut``): the part before the repeat on the first stage,
        and the part after it, to the loss, on the last."""
        method = parallel.scheduled.active()
        assert method is not None
        n = method.table().microbatches
        apart = parallel.gather.rows() if self.reduces_once else ()
        model = nnx.merge(graphdef, self._gathered(params), buffers)
        model.set_attributes(update_at_step_end=True, raise_if_not_found=False)
        cut = _Cut(model, self.objective, rows=n * _size(apart))
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
        rows = self.microbatches * _size(over)

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
                graphdef, whole, own, batch, zeros, shares, rows
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
        rows: int,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Sum `_grads` over the microbatches of `batch` in a scan, into `zeros`, each
        one's per-token part scaled by its share of the batch's targets and its
        per-row part by 1 / `rows`, the microbatches of every device: so the sums are
        the batch's, even when microbatches have unequal numbers of targets."""

        def add_microbatch(
            carry: tuple[nnx.State, nnx.State],
            xs: tuple[Batch, Float[Array, ""]],
        ) -> tuple[tuple[nnx.State, nnx.State], Metrics]:
            grad_sum, buffers = carry
            microbatch, token_share = xs
            grads, metrics, buffers = self._grads(
                graphdef, params, buffers, microbatch, token_share, row_share=1 / rows
            )
            return (jax.tree.map(_add, grad_sum, grads), buffers), metrics

        xs = (parallel.microbatches(batch, self.microbatches), token_shares)
        (grads, buffers), metrics = jax.lax.scan(add_microbatch, (zeros, buffers), xs)
        return grads, {k: v.sum() for k, v in metrics.items()}, buffers

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
            model = cast(nnx.merge(graphdef, params, buffers, copy=True), self.compute)
            model.set_attributes(update_at_step_end=True, raise_if_not_found=False)
            with matmuls.use(self.matmuls, jnp.asarray(batch.input_ids)):
                heads, aux_losses = training_forward(model, batch)
            per_token, per_row = self.objective(heads, batch, aux_losses)
            loss = token_share * per_token.loss + row_share * per_row.loss
            metrics = _scaled(per_token.metrics, token_share)
            metrics |= _scaled(per_row.metrics, row_share)
            return loss, (metrics, nnx.state(model, nnx.Not(nnx.Param)))

        grads, (metrics, buffers) = jax.grad(loss_fn, has_aux=True)(params)
        return grads, metrics, buffers

    def eval_step(self, model: Model, batch: Batch) -> Float[Array, ""]:
        """Return the first head's loss on `batch`, computed in `compute`."""
        with matmuls.use(matmuls.evaluating(self.matmuls)):
            heads = forward(cast(model, self.compute), batch)
        per_token, _ = self.objective(heads, batch)
        return per_token.metrics["loss"]

    def train_on_batch(self, batch: Batch) -> Metrics:
        """Take one training step on `batch`, and return its metrics."""
        self._switch_if_due()
        metrics = self._train(self.model, self.optimizer, parallel.shard_batch(batch))
        self.step += 1
        return metrics

    def _switch_if_due(self) -> None:
        if self.switch is not None and self.step >= self.switch[0]:
            self.matmuls, self.switch = self.switch[1], None
            self._compile()

    def evaluate(self, batches: Iterable[Batch]) -> float:
        """Return the mean of `eval_step` over `batches`, in eval mode."""
        self.model.eval()
        try:
            losses = [
                float(self._eval(self.model, parallel.shard_batch(b))) for b in batches
            ]
        finally:
            self.model.train()
        return sum(losses) / len(losses)

    def state(self) -> nnx.State:
        """What a checkpoint keeps: the model's state and the optimizer's."""
        return nnx.state((self.model, self.optimizer))

    def load(self, state: nnx.State | Mapping[str, Any], step: int) -> None:
        """Take up `state`, from `state()` or a checkpoint, at `step`."""
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
        every hook's `stop` runs; the next batches load as a step runs (``prefetch``).
        After every step, the hooks due run in order, and what they return goes to
        every sink. All of them close at the end, or when training stops."""
        with contextlib.ExitStack() as stack:
            for block in (*sinks, *hooks):
                stack.callback(block.close)
            for hook in hooks:
                hook.start(self)
            loaded = stack.enter_context(contextlib.closing(prefetch(data)))
            while self.step < steps and not self.stopping:
                metrics = self.train_on_batch(next(loaded))
                for hook in hooks:
                    if hook.due(self.step, steps) and (scalars := hook(self, metrics)):
                        for sink in sinks:
                            sink.write(self.step, scalars)
            if self.step < steps:
                for hook in hooks:
                    hook.stop(self)


def directory(run: ir.Run, out: str | Path) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", run.name).strip("_")
    return Path(out) / f"{slug}-{run.id}"


def train(
    run: ir.Run,
    out: str | Path = "runs",
    *,
    cache: str | Path = DEFAULT_CACHE,
    echo: Callable[[str], None] = print,
    hooks: Sequence[Hook] = (),
    sinks: Sequence[Sink] = (),
) -> dict[str, float]:
    """Train `run` to its last step, writing to its directory under `out`, after
    preparing the datasets it reads under `cache`; return the latest value of every
    metric. `hooks` run after the run's own, and `sinks` get what all of them return,
    as the console and metrics.jsonl do."""
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

    multiplier = lr_multiplier(config.schedule, config.steps)
    log = _log(run, lr_schedule(config.optimizer.lr, multiplier))
    methods = stages.balanced(run)
    if methods.stages != config.methods.stages:
        echo(f"{run.name}: methods.stages = {methods.stages}, by their work")
    with parallel.use(config.mesh, config.layout, methods):
        parallel.check(config.batch, config.seq_len)
        _check_cpu_bfloat16(run)
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(config.seed)))
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
            gather=config.methods.gather,
            accumulate=config.methods.accumulate,
            reduce=_reduce_dtype(config.precision),
            donate=config.methods.donate,
            compiler_options=xla.options(config.compiler),
        )
        make_batches = functools.partial(
            batches, config=config, vocab_size=vocab, cache=cache
        )
        run_hooks = _hooks(run, trainer, folder, log, make_batches, echo)
        latest = LatestMetrics()
        trainer.fit(
            make_batches(config.data, start=trainer.step),
            config.steps,
            hooks=[*run_hooks, *hooks],
            sinks=[
                Console(run.name, config.steps, echo),
                JSONL(folder / "metrics.jsonl", append=trainer.step > 0),
                latest,
                *sinks,
            ],
        )
    return latest.metrics


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


def _reduce_dtype(precision: PrecisionConfig) -> str:
    """Return the dtype `precision` sums gradients over the batch's rows in."""
    return precision.compute if precision.reduce == "compute" else precision.reduce


def _accumulated(microbatches: int) -> int:
    """Return how many microbatches a step accumulates gradients over: one under a
    pipeline, which takes them through its stages instead."""
    return 1 if parallel.pipeline.pipelined() else microbatches


def step_flops(run: ir.Run) -> float:
    """Return the model FLOPs of a training step of `run`: its loss's gradient on a
    batch, counted from its program (``flops.count``) on one device, outside any mesh,
    with remat and kernels off, which would add work or hide it."""
    t = run.train
    loss = objective(t.loss, run.arch)

    def gradient(
        params: nnx.State, rest: nnx.State, ids: Int32[Array, "b t"]
    ) -> nnx.State:
        positions = jnp.arange(t.seq_len, dtype=jnp.int32)
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
        ids = jax.ShapeDtypeStruct((t.batch, t.seq_len), jnp.int32)
        return flops.count(gradient, params, rest, ids)


def _log(run: ir.Run, lr: Schedule) -> Log:
    """Return the run's log, with its MFU where the peak of its devices is known."""
    t = run.train
    devices = parallel.mesh(t.mesh).devices
    each = flops.peak(devices.flat[0].device_kind)
    most = each and each * devices.size
    work = step_flops(run) if most else None
    return Log(
        every=t.log.every, lr=lr, tokens=t.batch * t.seq_len, flops=work, peak=most
    )


def _hooks(
    run: ir.Run,
    trainer: Trainer,
    folder: Path,
    log: Log,
    make_batches: Callable[..., Iterator[Batch]],
    echo: Callable[[str], None],
) -> list[Hook]:
    """Return the run's own hooks: its `log`, eval, checkpoint, which resumes `trainer`
    from the latest checkpoint in `folder` if there is one, and profile."""
    config = run.train
    hooks: list[Hook] = [log]
    if (evals := config.eval) is not None:
        held_out = functools.partial(make_batches, evals.data, held_out=True)
        hooks.append(Eval(held_out, every=evals.every, batches=evals.batches))
    if (saves := config.checkpoint) is not None:
        checkpoint = Checkpoint(
            folder / "checkpoints", every=saves.every, keep=saves.keep
        )
        if checkpoint.restore(trainer):
            echo(f"{run.name}: resuming at step {trainer.step}")
        hooks.append(checkpoint)
    if (trace := config.profile) is not None:
        hooks.append(Profile(folder / "profile", start=trace.start, steps=trace.steps))
    return hooks
