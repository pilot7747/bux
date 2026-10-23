"""Pipelines that take each microbatch's backward through the stages by hand: the
methods ``1f1b`` and ``zbh1`` of `methods.pipeline`, at the ticks of their tables
(bux.parallel.schedule).

gpipe and circular run the forward alone, and JAX's transpose of their scan of ticks
is the backward: every microbatch's residuals stay until it. Here a stage runs an op
in each slot of a tick, and keeps each microbatch's residuals only from its forward
to its backward. The forward is a ``jax.vjp``, whose residuals are a pytree of
arrays: a few microbatches' of them sit in a buffer (``Held``), and the backward
rebuilds the vjp from its slot. The parameters are not saved with each, as they do
not change in a step (``saveable_args``).

A backward needs the gradient of the loss as soon as the last stage has run its
microbatch forward, so these pipelines take the whole model, cut at the pipelined
repeat (``Parts``): the first stage runs the part before the repeat (the embedding)
on each microbatch, and the last stage the part after it (the head and the loss). A
training step asks for the ``gradients`` (``active``); a forward alone, as an eval's,
runs gpipe's.

    a lax.scan over the table's ticks; at each, every stage runs its ops slot by slot
    (a lax.switch), then sends its forward's output on and its backward's input
    gradient back, a ppermute each way

The stages' gradients add up in float32 on their own devices, and those of the parts
before and after the repeat on the first and last stage, summed over the stages as
they leave. Where the step computes its rows apart (``gather.apart``), each device
runs its own rows, and every gradient sums over the rows' axes once, at the end.
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.api import VJP, NotNeeded, NotSaveable
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PyTree, Shaped

from bux.parallel import gather, methods, pipeline, schedule, split
from bux.parallel.pipeline import AXIS, Out, Stage, _at, _by_round, _by_stage, _put

Metrics = dict[str, Float[Array, ""]]
Stream = Shaped[Array, "n t *r"]  # one microbatch's
Vjp = Callable[..., Any]  # a jax.vjp function: a pytree of its residuals


class Parts(NamedTuple):
    """A model cut at its pipelined repeat, as functions of arrays alone, each run on
    one microbatch's `rows`: `first` up to the repeat, ``(others, kept, rows) -> (x,
    kept)``; `stage`, a stage's share of the repeat (``pipeline.Stage``), whose
    losses' names and weights `reports` gives once it has traced; and `last`, from the
    repeat to the loss, ``(others, kept, rows, y) -> (loss, metrics)``. `others` are
    the parameters outside the repeat, and `kept` the rest of their state, which the
    part before the repeat may change, as dropout counts its masks."""

    first: Callable[[PyTree, PyTree, PyTree], tuple[Stream, PyTree]]
    stage: Stage
    last: Callable[[PyTree, PyTree, PyTree, Stream], tuple[Float[Array, ""], Metrics]]
    reports: Callable[[], list[tuple[str, float]]]


class Gradients(NamedTuple):
    """A step of a scheduled pipeline: the gradients of the stages' parameters, each
    leaf (rounds, layers, ...) as they came, and of the parameters outside the repeat;
    the layers' state and the rest of the model's as the forwards leave them; and
    the metrics, summed over the microbatches, with the means of the losses the
    layers report."""

    params: PyTree
    others: PyTree
    rest: PyTree
    kept: PyTree
    metrics: Metrics


class Scheduled:
    """A pipeline method run by the table ``schedule.TABLES[name]``."""

    def __init__(self, name: str):
        self.name = name

    def __call__(
        self,
        stage: Stage,
        params: PyTree,
        rest: PyTree,
        x: Shaped[Array, "b t *r"],
        rows: PyTree,
    ) -> Out:
        """Run the forward alone, as gpipe does."""
        return pipeline.gpipe(stage, params, rest, x, rows)

    def table(self) -> schedule.Table:
        """Return the table of the mesh's stages and the run's microbatches."""
        on = pipeline._on()
        return schedule.TABLES[self.name](on.size(AXIS), on.methods.microbatches)

    def gradients(
        self,
        parts: Parts,
        params: PyTree,
        rest: PyTree,
        others: PyTree,
        kept: PyTree,
        rows: PyTree,
    ) -> Gradients:
        """Return a step's gradients, of the loss summed over the microbatches of
        `rows`, each leaf (microbatches, ...), with respect to the stages' `params`
        and the `others`. `params` and `rest` are the layers' parameters and other
        state, each leaf (rounds, layers, ...)."""
        table, apart = self.table(), gather.rows_apart()
        by_stage = P(AXIS)
        by_rows = jax.tree.map(
            lambda a: P(None, apart or None) if a.ndim > 1 else P(), rows
        )
        grads, outside, rest, kept, metrics = jax.shard_map(
            functools.partial(_device, table, parts, apart),
            in_specs=(by_stage, by_stage, P(), P(), by_rows),
            out_specs=(by_stage, P(), by_stage, by_stage, P()),
            axis_names={AXIS, *apart},
            check_vma=False,  # else nested shard_maps' gradients fail to lower
        )(
            _by_stage(params, table.stages),
            _by_stage(rest, table.stages),
            others,
            kept,
            rows,
        )
        kept = jax.tree.map(lambda a: a[0], kept)  # as the first stage leaves it
        return Gradients(_by_round(grads), outside, _by_round(rest), kept, metrics)


def active() -> Scheduled | None:
    """Return the run's pipeline method if it schedules its own backward."""
    found = methods.method("pipeline")
    return found if isinstance(found, Scheduled) and pipeline.pipelined() else None


for _name in schedule.TABLES:
    methods.pipeline.register(_name)(Scheduled(_name))


class Held(NamedTuple):
    """What a stage holds between ticks. Each buffer has a slot more than the table
    needs, which takes the writes of the ticks that have nothing to write."""

    inputs: Shaped[Array, "k n t *r"]  # the stream the stage before sent
    cotangents: Shaped[Array, "k n t *r"]  # the gradient the stage after sent
    residuals: list[Shaped[Array, "k ..."]]  # each forward's vjp, until its backward
    firsts: list[Shaped[Array, "k ..."]]  # the vjp of the part before, on stage 0
    weights: list[Shaped[Array, "k ..."]]  # what a split backward leaves W
    rest: PyTree  # the layers' state
    kept: PyTree  # the rest of the model's, which the part before changes
    losses: Float[Array, "l k"]  # what the layers reported, summed over microbatches
    grads: PyTree  # of the stage's parameters, in float32
    others: PyTree  # of the parameters outside the repeat, in float32
    metrics: Metrics
    out: Stream  # the forward's output, sent on at the tick's end
    back: Stream  # the backward's input gradient, sent back


class Shapes(NamedTuple):
    """What each part leaves, traced once on microbatch 0."""

    x: jax.ShapeDtypeStruct
    first: VJP
    stage: VJP
    losses: jax.ShapeDtypeStruct
    metrics: dict[str, jax.ShapeDtypeStruct]


def _device(
    table: schedule.Table,
    parts: Parts,
    apart: tuple[str, ...],
    params: PyTree,
    rest: PyTree,
    others: PyTree,
    kept: PyTree,
    rows: PyTree,
) -> tuple[PyTree, PyTree, PyTree, PyTree, Metrics]:
    """Run one stage's ticks of `table` on its devices."""
    before = _at(_at(rest, 0), 0)  # this stage's, of the one round
    ticks = Ticks(table, parts, apart, _at(_at(params, 0), 0), others, kept, rows)
    own = gather.own_streams
    held = ticks.start(own(before, apart), own(kept, apart))
    held, _ = jax.lax.scan(ticks.tick, held, ticks.xs())
    return ticks.finish(held, before, kept)


class Ticks:
    """A table's ticks on one stage's devices: the ops a slot runs (``forward``,
    ``backward``), and the sends at the end of a tick (``tick``)."""

    def __init__(
        self,
        table: schedule.Table,
        parts: Parts,
        apart: tuple[str, ...],
        params: PyTree,
        others: PyTree,
        kept: PyTree,
        rows: PyTree,
    ):
        self.table, self.parts, self.apart = table, parts, apart
        self.params, self.others, self.kept, self.rows = params, others, kept, rows
        self.index, self.last = jax.lax.axis_index(AXIS), table.stages - 1
        slots = table.slots()
        self.depth = {name: int(s.max()) + 1 for name, s in slots.items()}
        self.slots = {name: jnp.asarray(s) for name, s in slots.items()}
        self.rows_apart = int(np.prod([jax.lax.axis_size(a) for a in apart]))
        self.shapes: Shapes | None = None
        self.split: split.Split | None = None

    # --- The parts, on one microbatch ---------------------------------------------

    def embedded(self, kept: PyTree, rows: PyTree) -> tuple[Stream, Vjp, PyTree]:
        """Run the part before the repeat: the stream, the vjp that takes its
        gradient back to the parameters outside the repeat, and their other state
        after."""

        def first(others: PyTree) -> tuple[Stream, PyTree]:
            return self.parts.first(others, kept, rows)

        return jax.vjp(first, self.others, has_aux=True, saveable_args=(False,))

    def staged(
        self, rest: PyTree, rows: PyTree, x: Stream
    ) -> tuple[Stream, Float[Array, "l k"], Vjp, PyTree]:
        """Run this stage's share of the repeat: its output, the losses its layers
        report, the vjp that takes both back to its parameters and its input, and the
        layers' state after."""

        def run(params: PyTree, x: Stream) -> tuple[tuple[Stream, Array], PyTree]:
            y, after, losses = self.parts.stage(params, rest, x, rows)
            return (y, losses), after

        (y, losses), vjp, after = jax.vjp(
            run, self.params, x, has_aux=True, saveable_args=(False, True)
        )
        return y, losses, vjp, after

    def scored(self, rows: PyTree, y: Stream) -> tuple[Metrics, PyTree, Stream]:
        """Run the part after the repeat on the stage's output, to the loss: its
        metrics, and the loss's gradients with respect to the parameters outside the
        repeat and to `y`."""

        def loss(others: PyTree, y: Stream) -> tuple[Float[Array, ""], Metrics]:
            return self.parts.last(others, self.kept, rows, y)

        grad = jax.grad(loss, argnums=(0, 1), has_aux=True)
        (d_others, dy), metrics = grad(self.others, y)
        return metrics, d_others, dy

    def trace(self, rest: PyTree) -> Shapes:
        """Trace the parts on microbatch 0, for the shapes of what they leave."""

        def parts(rest: PyTree, rows: PyTree) -> tuple:
            x, first, _ = self.embedded(self.kept, rows)
            y, losses, vjp, _ = self.staged(rest, rows, x)
            metrics, _, _ = self.scored(rows, y)
            return x, first, vjp, losses, metrics

        return Shapes(*jax.eval_shape(parts, rest, _at(self.rows, 0)))

    # --- What a stage holds -------------------------------------------------------

    def start(self, rest: PyTree, kept: PyTree) -> Held:
        """Return what a stage holds before its first tick, with the layers' `rest`
        and the rest of the model's state, `kept`."""
        shapes = self.shapes = self.trace(rest)
        self.split = self._split() if self.table.split else None
        saved = self.split.shapes if self.split else []

        def buffer(a: jax.ShapeDtypeStruct, name: str) -> Shaped[Array, "k ..."]:
            return jnp.zeros((self.depth[name] + 1, *a.shape), a.dtype)

        def zeros(tree: PyTree) -> PyTree:
            return jax.tree.map(lambda a: jnp.zeros(a.shape, jnp.float32), tree)

        stream = jnp.zeros(shapes.x.shape, shapes.x.dtype)
        return Held(
            inputs=buffer(shapes.x, "inputs"),
            cotangents=buffer(shapes.x, "cotangents"),
            residuals=[buffer(a, "residuals") for a in jax.tree.leaves(shapes.stage)],
            firsts=[buffer(a, "residuals") for a in jax.tree.leaves(shapes.first)],
            weights=[buffer(a, "weights") for a in saved],
            rest=rest,
            kept=kept,
            losses=zeros(shapes.losses),
            grads=zeros(self.params),
            others=zeros(self.others),
            metrics=zeros(shapes.metrics),
            out=stream,
            back=stream,
        )

    def slot(self, buffer: str, j: Int[Array, ""]) -> Int[Array, ""]:
        """Return the slot of `buffer` that holds microbatch `j` on this stage, or
        the spare last one where `j` is -1."""
        found = self.slots[buffer][self.index, jnp.maximum(j, 0)]
        return jnp.where(j >= 0, found, self.depth[buffer])

    # --- The ops ------------------------------------------------------------------

    def forward(self, held: Held, j: Int[Array, ""]) -> Held:
        """Run microbatch `j` forward through this stage, from the embedding on the
        first stage, and keep its residuals. The last stage scores it too."""
        rows = _at(self.rows, j)
        x, firsts, kept = jax.lax.cond(
            self.index == 0, self._embed, self._receive, held, j, rows
        )
        y, losses, vjp, rest = self.staged(held.rest, rows, x)
        slot = self.slot("residuals", j)
        held = held._replace(
            residuals=_put(held.residuals, slot, jax.tree.leaves(vjp)),
            firsts=_put(held.firsts, slot, firsts),
            rest=rest,
            kept=kept,
            losses=held.losses + losses,
            out=y,
        )
        return jax.lax.cond(self.index == self.last, self._score, _keep, held, j, y)

    def _embed(
        self, held: Held, j: Int[Array, ""], rows: PyTree
    ) -> tuple[Stream, list[Array], PyTree]:
        x, vjp, kept = self.embedded(held.kept, rows)
        return x, jax.tree.leaves(vjp), kept

    def _receive(
        self, held: Held, j: Int[Array, ""], rows: PyTree
    ) -> tuple[Stream, list[Array], PyTree]:
        assert self.shapes is not None
        empty = [
            jnp.zeros(a.shape, a.dtype) for a in jax.tree.leaves(self.shapes.first)
        ]
        return _at(held.inputs, self.slot("inputs", j)), empty, held.kept

    def _score(self, held: Held, j: Int[Array, ""], y: Stream) -> Held:
        metrics, d_others, dy = self.scored(_at(self.rows, j), y)
        return held._replace(
            cotangents=_put(held.cotangents, self.slot("cotangents", j), dy),
            others=_added(held.others, d_others),
            metrics=_added(held.metrics, metrics),
        )

    def backward(self, held: Held, j: Int[Array, ""]) -> Held:
        """Take microbatch `j`'s gradient back through this stage, from its slot of
        residuals, and on the first stage on to the embedding. Where the table splits
        the backward, take the input's gradient alone, and keep what the weights'
        needs for ``weights``."""
        slot = self.slot("residuals", j)
        residuals = _at(held.residuals, slot)
        dy = _at(held.cotangents, self.slot("cotangents", j))
        if self.split is None:
            d_params, dx = self._vjp(residuals)((dy, self._losses_cotangent()))
            held = held._replace(grads=_added(held.grads, d_params), back=dx)
        else:
            args = [
                *jax.tree.leaves(self.params),
                *residuals,
                dy,
                self._losses_cotangent(),
            ]
            (dx,), saved = self.split.inputs(*args)
            weights = _put(held.weights, self.slot("weights", j), saved)
            held = held._replace(weights=weights, back=dx)
        return jax.lax.cond(self.index == 0, self._unembed, _keep, held, slot, dx)

    def weights(self, held: Held, j: Int[Array, ""]) -> Held:
        """Take microbatch `j`'s gradient on to this stage's weights, from what its
        backward kept."""
        assert self.split is not None
        saved = _at(held.weights, self.slot("weights", j))
        flat = self.split.weights(*jax.tree.leaves(self.params), saved)
        d_params = jax.tree.unflatten(jax.tree.structure(self.params), flat)
        return held._replace(grads=_added(held.grads, d_params))

    def _vjp(self, residuals: list[Array]) -> Vjp:
        """Return this stage's vjp, rebuilt from a slot's `residuals`."""
        assert self.shapes is not None
        return _rebuilt(self.shapes.stage, residuals, self.params)

    def _split(self) -> split.Split:
        """Return this stage's backward split in two: the input's gradient and what
        the weights' needs, then the weights' (``split.split``)."""
        shapes = self.shapes
        assert shapes is not None
        params = jax.tree.leaves(self.params)
        residuals = jax.tree.leaves(shapes.stage)
        tree = jax.tree.structure(self.params)

        def backward(*args: Array) -> list[Array]:
            p, (*r, dy, dl) = args[: len(params)], args[len(params) :]
            vjp = _rebuilt(shapes.stage, r, jax.tree.unflatten(tree, p))
            d_params, dx = vjp((dy, dl))
            return [*jax.tree.leaves(d_params), dx]

        x, losses = shapes.x, shapes.losses
        free = [_shape(a) for a in params]
        rest = [*map(_shape, residuals), _shape(x), _shape(losses)]
        return split.split(backward, free, rest, weights=len(params))

    def _unembed(self, held: Held, slot: Int[Array, ""], dx: Stream) -> Held:
        assert self.shapes is not None
        vjp = _rebuilt(self.shapes.first, _at(held.firsts, slot), self.others)
        (d_others,) = vjp(dx)
        return held._replace(others=_added(held.others, d_others))

    def _losses_cotangent(self) -> Float[Array, "l k"]:
        """Return the gradient of the step's loss with respect to the losses this
        stage's layers report for a microbatch: their weights over the microbatches
        of every device, as the loss takes their mean."""
        assert self.shapes is not None
        weights = np.array([w for _, w in self.parts.reports()], np.float32)
        weights /= self.table.microbatches * self.rows_apart
        return jnp.broadcast_to(jnp.asarray(weights), self.shapes.losses.shape)

    # --- Ticks --------------------------------------------------------------------

    def xs(self) -> tuple[Int[Array, "t ..."], ...]:
        """Return what the ticks read, tick by tick: each slot's op and microbatch,
        and the microbatch each stage receives from the stage before and after."""
        kinds, micros = self.table.codes()
        got = [self.table.received(k) for k in (schedule.FORWARD, schedule.BACKWARD)]
        return tuple(jnp.asarray(a) for a in (kinds, micros, *got))

    def tick(self, held: Held, xs: tuple[Int[Array, "..."], ...]) -> tuple[Held, None]:
        """Run a tick: this stage's op in each slot, then the sends both ways."""
        kinds, micros, got_forward, got_backward = xs
        ops = [
            _idle,
            self.forward,
            self.backward,
            self.weights if self.split else _idle,
        ]
        for k in range(self.table.width):
            op, j = kinds[self.index, k], micros[self.index, k]
            held = jax.lax.switch(op, ops, held, j)
        out, back = _send(held.out, 1), _send(held.back, -1)
        got = self.slot("inputs", got_forward[self.index])
        inputs = _put(held.inputs, got, out)
        got = self.slot("cotangents", got_backward[self.index])
        cotangents = _put(held.cotangents, got, back)
        return held._replace(inputs=inputs, cotangents=cotangents), None

    def finish(
        self, held: Held, before: PyTree, kept: PyTree
    ) -> tuple[PyTree, PyTree, PyTree, PyTree, Metrics]:
        """Return the gradients, the state and the metrics as the shard_map gives
        them out: the parts outside the repeat summed over the stages, and everything
        over the rows' axes, where each device ran its own rows."""
        rest = gather.keys_of(before, held.rest)
        kept = gather.keys_of(kept, held.kept)
        others = jax.lax.psum(held.others, AXIS)
        metrics = jax.lax.psum(held.metrics, AXIS)
        reported = jax.lax.psum(self._reported(held.losses), AXIS)
        grads = held.grads
        if self.apart:
            grads, others, metrics = jax.lax.psum((grads, others, metrics), self.apart)
            reported = jax.lax.pmean(reported, self.apart)
        lift = functools.partial(jax.tree.map, lambda a: a[None, None])
        kept = jax.tree.map(lambda a: a[None], kept)
        return lift(grads), others, lift(rest), kept, _merged(metrics, reported)

    def _reported(self, losses: Float[Array, "l k"]) -> Metrics:
        """Return the means over microbatches of the losses this stage's layers
        report, each name's summed over its layers."""
        means: Metrics = {}
        for i, (name, _) in enumerate(self.parts.reports()):
            total = losses[:, i].sum() / self.table.microbatches
            means[name] = means.get(name, jnp.zeros((), jnp.float32)) + total
        return means


def _idle(held: Held, j: Int[Array, ""]) -> Held:
    return held


def _keep(held: Held, *_: object) -> Held:
    return held


def _added(total: PyTree, part: PyTree) -> PyTree:
    """Add `part` to the float32 `total`, leaf by leaf."""
    return jax.tree.map(lambda a, b: a + b.astype(a.dtype), total, part)


def _merged(metrics: Metrics, more: Metrics) -> Metrics:
    """Return `metrics` and `more`, a name both have summed."""
    return metrics | {k: v + metrics[k] if k in metrics else v for k, v in more.items()}


def _rebuilt(shapes: Vjp, leaves: list[Array], unsaved: PyTree) -> VJP:
    """Return the vjp of the structure `shapes` from its residual `leaves`, with its
    first primal, which it did not save, put back as `unsaved`."""
    vjp = jax.tree.unflatten(jax.tree.structure(shapes), leaves)

    def sentinel(v: object) -> bool:
        return isinstance(v, NotSaveable | NotNeeded)

    held, tree = jax.tree.flatten(vjp.args_res[0], is_leaf=sentinel)
    values = [
        u if isinstance(h, NotSaveable) else h
        for h, u in zip(held, jax.tree.leaves(unsaved), strict=True)
    ]
    return vjp.replace(args_res=[jax.tree.unflatten(tree, values), *vjp.args_res[1:]])


def _shape(a: Array | jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(a.shape, a.dtype)


def _send(y: Stream, step: int) -> Stream:
    """Send each stage's `y` to the stage `step` after it, round the ring."""
    n = jax.lax.axis_size(AXIS)
    return jax.lax.ppermute(y, AXIS, [(i, (i + step) % n) for i in range(n)])
