"""Pipelines that run each microbatch's backward themselves, at the ticks of a table
(`bux.parallel.schedule`): the methods ``1f1b`` and ``zbh1`` of ``methods.pipeline``.

gpipe's backward is JAX's transpose of its scan of ticks, so a stage keeps every
microbatch's residuals until the step ends. Here a stage's forward is a `jax.vjp`,
whose residuals sit in a slot of a buffer (`Held`) only until the backward rebuilds
the vjp from them. A backward needs the loss's gradient, so these pipelines take the
whole model, cut at the pipelined repeat (`Parts`): the first stage runs the part
before the repeat, and the last stage the part after it, to the loss. On each
stage::

    a lax.scan over the table's ticks; at each tick,
        each lane:  F, B or W on the lane's microbatch, each kind under a lax.cond
        then:       a ppermute sends F's output on, and one sends B's input gradient back

The gradients add up in float32 on each stage. Those of the parts outside the repeat
sum over the stages as they leave, and where each device runs its own rows
(`bux.parallel.gather.apart`), every gradient sums over the rows' axes once, at the
end. A forward alone, an eval's, runs gpipe's::

    mesh    = mesh(stage=4)
    methods = methods(pipeline=zbh1, microbatches=8)
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PyTree, Shaped

from bux import jax_internals
from bux.architecture import observe
from bux.parallel import gather, methods, pipeline, schedule, split
from bux.parallel.pipeline import (
    AXIS,
    Out,
    Stage,
    add_axes,
    at,
    join_stages,
    put,
    send,
    split_by_stage,
    stand_ins,
)
from bux.parallel.schedule import BACKWARD, CODES, FORWARD, WEIGHTS

SENT = ("inputs", "cotangents")  # the buffers what the stages send arrives in
Metrics = dict[str, Float[Array, ""]]
Stream = Shaped[Array, "n t *r"]  # one microbatch's
Losses = Float[Array, "l k"]  # what a stage's layers report: (layers, reports)
Vjp = Callable[..., Any]  # a jax.vjp function: a pytree of its residuals
Leaf = Shaped[Array, "..."]  # one of a vjp's residuals, or a gradient


class Parts(NamedTuple):
    """A model cut at its pipelined repeat, each part a function of arrays alone, run
    on one microbatch's `rows`::

        first(others, kept, rows) -> (x, kept)                 up to the repeat
        stage(params, rest, x, rows) -> (y, rest, losses)       a stage's share of it
        last(others, kept, rows, y) -> (loss, metrics, kept)   from the repeat on

    `others` are the parameters outside the repeat and `kept` the rest of their state,
    which `first` and `last` may change, as dropout counts its masks: `last` runs the
    forward from the start, the layers before the repeat too. `reports` gives the names
    and weights of the losses the stage's layers report, once `stage` has traced;
    `observed`, given the values the repeat's layers observed, a pass's slot of every
    stage a row, and their names, those values by their metrics' names
    (``bux.architecture.observe``)."""

    first: Callable[[PyTree, PyTree, PyTree], tuple[Stream, PyTree]]
    stage: Stage
    last: Callable[
        [PyTree, PyTree, PyTree, Stream], tuple[Float[Array, ""], Metrics, PyTree]
    ]
    reports: Callable[[], list[tuple[str, float]]]
    observed: Callable[[Float[Array, "n k"], list[str]], Metrics] | None = None


class Gradients(NamedTuple):
    """What a step of a scheduled pipeline returns: the gradients of the stages'
    parameters, each leaf (rounds, layers, ...), and of the parameters outside the
    repeat; the state of the layers and of the rest of the model as the forwards
    leave them; and the metrics, summed over the microbatches, with the means of the
    losses the layers report."""

    params: PyTree
    others: PyTree
    rest: PyTree
    kept: PyTree
    metrics: Metrics


class Scheduled:
    """A pipeline that runs each microbatch's backward itself, at the ticks of the
    table ``schedule.TABLES[name]``: ``1f1b``, a forward and a whole backward in two
    lanes, each as soon as it can run; or ``zbh1`` (ZB-H1), the backward split so
    that each weight gradient fills a tick the stage would idle. The first stage also
    runs the model up to the pipelined repeat, and the last the rest of it to the
    loss.

    Args:
        name: The table's name in `~bux.parallel.schedule.TABLES`.
    """

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
            split_by_stage(params, table.stages),
            split_by_stage(rest, table.stages),
            others,
            kept,
            rows,
        )
        kept = jax.tree.map(lambda a: a[-1], kept)  # as the last stage leaves it
        return Gradients(join_stages(grads), outside, join_stages(rest), kept, metrics)


def active() -> Scheduled | None:
    """Return the run's pipeline method if it schedules its own backward."""
    found = methods.method("pipeline")
    return found if isinstance(found, Scheduled) and pipeline.pipelined() else None


for _name in schedule.TABLES:
    methods.pipeline.register(_name)(Scheduled(_name))


class Held(NamedTuple):
    """What a stage holds from tick to tick: buffers of a slot per microbatch held
    (``schedule.Table.buffer_slots``), the state as the forwards leave it, and sums."""

    inputs: Shaped[Array, "k n t *r"]  # the stream the stage before sent
    cotangents: Shaped[Array, "k n t *r"]  # the gradient the stage after sent
    residuals: list[Shaped[Array, "k ..."]]  # each forward's vjp, until its backward
    first_residuals: list[Shaped[Array, "k ..."]]  # the first part's, on stage 0
    saved: list[Shaped[Array, "k ..."]]  # what a split backward saves for W
    rest: PyTree  # the layers' state
    # The rest of the model's: on the first stage as the part before the repeat
    # leaves it, on the last as the part after it does, which runs the whole forward.
    kept: PyTree
    losses: Losses  # what the layers reported, summed over microbatches
    d_params: PyTree  # the gradient of the stage's parameters, in float32
    d_others: PyTree  # of the parameters outside the repeat, in float32
    metrics: Metrics
    to_next: Stream  # the forward's output, sent on at the tick's end
    to_previous: Stream  # the backward's input gradient, sent back


class Shapes(NamedTuple):
    """What each part leaves, traced once on microbatch 0: the stream it enters the
    repeat with, the vjps of the first part and of the stage (their residuals' shapes
    and structure), the losses the stage's layers report, and the metrics."""

    x: jax.ShapeDtypeStruct
    first: jax_internals.VJP
    stage: jax_internals.VJP
    losses: jax.ShapeDtypeStruct
    metrics: dict[str, jax.ShapeDtypeStruct]


class TickInputs(NamedTuple):
    """What a tick reads, each by stage: the code and microbatch of each lane's op
    (``schedule.CODES``), and the microbatch that arrives from the stage before and
    from the stage after (-1 for none)."""

    codes: Int[Array, "s k"]
    microbatches: Int[Array, "s k"]
    from_before: Int[Array, " s"]
    from_after: Int[Array, " s"]


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
    layers = at(at(rest, 0), 0)  # this stage's, of the one round
    params = at(at(params, 0), 0)
    ticks = Ticks(table, parts, apart, params, layers, others, kept, rows)
    own = gather.own_streams
    held = ticks.start(own(layers, apart), own(kept, apart))
    held, _ = jax.lax.scan(ticks.tick, held, ticks.tick_inputs())
    return ticks.finish(held, layers, kept)


class Ticks:
    """A table's ticks on one stage's devices: the model's parts on a microbatch, the
    ops a lane runs (``forward``, ``backward``, ``weights``), and a tick (``tick``)."""

    def __init__(
        self,
        table: schedule.Table,
        parts: Parts,
        apart: tuple[str, ...],
        params: PyTree,
        rest: PyTree,
        others: PyTree,
        kept: PyTree,
        rows: PyTree,
    ):
        """Take this stage's `params` and its layers' state `rest`, the parameters
        outside the repeat, `others`, and the rest of their state, `kept`, and every
        microbatch's `rows`."""
        self.table, self.parts, self.apart = table, parts, apart
        self.params, self.others, self.kept, self.rows = params, others, kept, rows
        self.index, self.last = jax.lax.axis_index(AXIS), table.stages - 1
        slots = table.buffer_slots()
        self.depth = {name: int(s.max()) + 1 for name, s in slots.items()}
        self.slots = {name: jnp.asarray(s) for name, s in slots.items()}
        self.row_shards = int(np.prod([jax.lax.axis_size(a) for a in apart]))
        self.shapes = self.trace(rest)
        self.split = self._split_backward() if table.split else None

    # --- The parts, on one microbatch ---------------------------------------------

    def first_with_vjp(self, kept: PyTree, rows: PyTree) -> tuple[Stream, Vjp, PyTree]:
        """Run the part before the repeat: the stream it leaves, the vjp that takes
        its gradient back to the parameters outside the repeat, and the rest of their
        state after."""

        def first(others: PyTree) -> tuple[Stream, PyTree]:
            return self.parts.first(others, kept, rows)

        return jax.vjp(first, self.others, has_aux=True, saveable_args=(False,))

    def stage_with_vjp(
        self, rest: PyTree, rows: PyTree, x: Stream
    ) -> tuple[Stream, Losses, Vjp, PyTree]:
        """Run this stage's share of the repeat: its output, the losses its layers
        report, the vjp that takes both back to its parameters and its input, and the
        layers' state after."""

        def run(params: PyTree, x: Stream) -> tuple[tuple[Stream, Losses], PyTree]:
            y, after, losses = self.parts.stage(params, rest, x, rows)
            return (y, losses), after

        (y, losses), vjp, after = jax.vjp(
            run, self.params, x, has_aux=True, saveable_args=(False, True)
        )
        return y, losses, vjp, after

    def loss_gradients(
        self, kept: PyTree, rows: PyTree, y: Stream
    ) -> tuple[Metrics, PyTree, Stream, PyTree]:
        """Run the part after the repeat on the stage's output, to the loss, from the
        rest of the model's state `kept`: its metrics, the loss's gradients with
        respect to the parameters outside the repeat and to `y`, and that state
        after."""

        def loss(
            others: PyTree, y: Stream
        ) -> tuple[Float[Array, ""], tuple[Metrics, PyTree]]:
            loss, metrics, after = self.parts.last(others, kept, rows, y)
            return loss, (metrics, after)

        grad = jax.grad(loss, argnums=(0, 1), has_aux=True)
        (d_others, dy), (metrics, after) = grad(self.others, y)
        return metrics, d_others, dy, after

    def trace(self, rest: PyTree) -> Shapes:
        """Trace the parts on microbatch 0, for the shapes of what they leave."""

        def parts(rest: PyTree, rows: PyTree) -> tuple:
            x, first, _ = self.first_with_vjp(self.kept, rows)
            y, losses, stage, _ = self.stage_with_vjp(rest, rows, x)
            metrics, *_ = self.loss_gradients(self.kept, rows, y)
            return x, first, stage, losses, metrics

        return Shapes(*jax.eval_shape(parts, rest, at(self.rows, 0)))

    # --- What a stage holds -------------------------------------------------------

    def start(self, rest: PyTree, kept: PyTree) -> Held:
        """Return what a stage holds before its first tick: empty buffers, zero sums,
        the layers' state `rest`, and the rest of the model's, `kept`."""
        shapes = self.shapes
        saved = self.split.saved_shapes if self.split else []
        stream = jnp.zeros(shapes.x.shape, shapes.x.dtype)
        return Held(
            inputs=self._buffer("inputs", shapes.x),
            cotangents=self._buffer("cotangents", shapes.x),
            residuals=self._buffers("residuals", jax.tree.leaves(shapes.stage)),
            first_residuals=self._buffers("residuals", jax.tree.leaves(shapes.first)),
            saved=self._buffers("saved", saved),
            rest=rest,
            kept=kept,
            losses=_float32_zeros(shapes.losses),
            d_params=_float32_zeros(self.params),
            d_others=_float32_zeros(self.others),
            metrics=_float32_zeros(shapes.metrics),
            to_next=stream,
            to_previous=stream,
        )

    def _buffer(self, name: str, a: jax.ShapeDtypeStruct) -> Shaped[Array, "k ..."]:
        """Return an empty buffer `name` of arrays like `a`, a slot for each
        microbatch held at once. A buffer that takes what the stages send has one
        more, spare, for what a tick sends that no microbatch's slot awaits."""
        spare = name in SENT
        return jnp.zeros((self.depth[name] + spare, *a.shape), a.dtype)

    def _buffers(
        self, name: str, leaves: list[jax.ShapeDtypeStruct]
    ) -> list[Shaped[Array, "k ..."]]:
        return [self._buffer(name, a) for a in leaves]

    def slot(self, buffer: str, j: Int[Array, ""]) -> Int[Array, ""]:
        """Return the slot of `buffer` that holds microbatch `j` on this stage, or
        the spare one where `j` is -1."""
        found = self.slots[buffer][self.index, jnp.maximum(j, 0)]
        return jnp.where(j >= 0, found, self.depth[buffer])

    # --- The ops ------------------------------------------------------------------

    def forward(self, held: Held, j: Int[Array, ""]) -> Held:
        """Run microbatch `j` forward through this stage, and keep its residuals. The
        first stage runs the part before the repeat first, and the last stage the
        part after it, to the loss, whose gradient it keeps for its backward."""
        rows = at(self.rows, j)
        x, first_residuals, kept = jax.lax.cond(
            self.index == 0,
            self._from_first_part,
            self._from_stage_before,
            held,
            j,
            rows,
        )
        y, losses, vjp, rest = self.stage_with_vjp(held.rest, rows, x)
        slot = self.slot("residuals", j)
        held = held._replace(
            residuals=put(held.residuals, slot, jax.tree.leaves(vjp)),
            first_residuals=put(held.first_residuals, slot, first_residuals),
            rest=rest,
            kept=kept,
            losses=held.losses + losses,
            to_next=y,
        )
        return jax.lax.cond(
            self.index == self.last, self._to_the_loss, _unchanged, held, j, y
        )

    def _from_first_part(
        self, held: Held, j: Int[Array, ""], rows: PyTree
    ) -> tuple[Stream, list[Leaf], PyTree]:
        """Return the stream as the part before the repeat leaves it, that part's
        residuals, and the rest of the model's state after it."""
        x, vjp, kept = self.first_with_vjp(held.kept, rows)
        return x, jax.tree.leaves(vjp), kept

    def _from_stage_before(
        self, held: Held, j: Int[Array, ""], rows: PyTree
    ) -> tuple[Stream, list[Leaf], PyTree]:
        """Return the stream the stage before sent, and zeros for the residuals of
        the part before the repeat, which this stage does not run."""
        leaves = jax.tree.leaves(self.shapes.first)
        empty = [jnp.zeros(a.shape, a.dtype) for a in leaves]
        return at(held.inputs, self.slot("inputs", j)), empty, held.kept

    def _to_the_loss(self, held: Held, j: Int[Array, ""], y: Stream) -> Held:
        """Run the part after the repeat on `y`, and keep the loss's gradient with
        respect to `y` for this stage's backward, and the state it leaves."""
        rows = at(self.rows, j)
        metrics, d_others, dy, kept = self.loss_gradients(held.kept, rows, y)
        return held._replace(
            cotangents=put(held.cotangents, self.slot("cotangents", j), dy),
            kept=kept,
            d_others=gather.added(held.d_others, d_others),
            metrics=gather.added(held.metrics, metrics),
        )

    def backward(self, held: Held, j: Int[Array, ""]) -> Held:
        """Take microbatch `j`'s gradient back through this stage, from its slot of
        residuals, and on the first stage on through the part before the repeat.
        Where the table splits the backward, take the input's gradient alone (`B`),
        and keep what the weights' (`W`) needs."""
        slot = self.slot("residuals", j)
        residuals = at(held.residuals, slot)
        dy = at(held.cotangents, self.slot("cotangents", j))
        if self.split is None:
            held = self._whole_backward(held, residuals, dy)
        else:
            held = self._input_backward(held, j, residuals, dy)
        return jax.lax.cond(
            self.index == 0,
            self._through_first_part,
            _unchanged,
            held,
            slot,
            held.to_previous,
        )

    def _whole_backward(self, held: Held, residuals: list[Leaf], dy: Stream) -> Held:
        """Take `dy` back to this stage's parameters and its input."""
        vjp = self._vjp(residuals)
        d_params, dx = vjp((dy, self._losses_cotangent()))
        return held._replace(
            d_params=gather.added(held.d_params, d_params), to_previous=dx
        )

    def _input_backward(
        self, held: Held, j: Int[Array, ""], residuals: list[Leaf], dy: Stream
    ) -> Held:
        """Take `dy` back to this stage's input alone, and keep what the gradient of
        its weights needs in microbatch `j`'s slot of `saved`."""
        assert self.split is not None
        params = jax.tree.leaves(self.params)
        cotangents = [dy, self._losses_cotangent()]
        (dx,), saved = self.split.inputs(*params, *residuals, *cotangents)
        saved = put(held.saved, self.slot("saved", j), saved)
        return held._replace(saved=saved, to_previous=dx)

    def _through_first_part(self, held: Held, slot: Int[Array, ""], dx: Stream) -> Held:
        """Take `dx` on back through the part before the repeat, to the parameters
        outside it, from its residuals in `slot`."""
        residuals = at(held.first_residuals, slot)
        vjp = jax_internals.rebuild_vjp(self.shapes.first, residuals, self.others)
        (d_others,) = vjp(dx)
        return held._replace(d_others=gather.added(held.d_others, d_others))

    def weights(self, held: Held, j: Int[Array, ""]) -> Held:
        """Take microbatch `j`'s gradient on to this stage's weights, from what its
        backward saved."""
        assert self.split is not None
        saved = at(held.saved, self.slot("saved", j))
        residuals = at(held.residuals, self.slot("residuals", j))
        params = jax.tree.leaves(self.params)
        flat = self.split.weights(*params, *residuals, saved)
        d_params = jax.tree.unflatten(jax.tree.structure(self.params), flat)
        return held._replace(d_params=gather.added(held.d_params, d_params))

    def _vjp(self, residuals: list[Leaf]) -> Vjp:
        """Return this stage's vjp, rebuilt from a slot's `residuals`."""
        return jax_internals.rebuild_vjp(self.shapes.stage, residuals, self.params)

    def _split_backward(self) -> split.Split:
        """Return this stage's backward split in two (``split.split``): `B`, the
        input's gradient and what the weights' needs, and `W`, the weights'
        gradient, which takes the parameters and the residuals again."""
        shapes = self.shapes
        params = jax.tree.leaves(self.params)
        residuals = jax.tree.leaves(shapes.stage)
        tree = jax.tree.structure(self.params)

        def backward(*args: Leaf) -> list[Leaf]:  # params, residuals, dy, dl
            p, (*r, dy, dl) = args[: len(params)], args[len(params) :]
            vjp = jax_internals.rebuild_vjp(
                shapes.stage, r, jax.tree.unflatten(tree, p)
            )
            d_params, dx = vjp((dy, dl))
            return [*jax.tree.leaves(d_params), dx]

        shared = stand_ins([*params, *residuals])
        cotangents = stand_ins([shapes.x, shapes.losses])
        return split.split(backward, shared, cotangents, weight_grads=len(params))

    def _losses_cotangent(self) -> Losses:
        """Return the gradient of the step's loss with respect to the losses this
        stage's layers report for a microbatch: their weights over the microbatches
        of every device, as the loss takes their mean."""
        weights = np.array([w for _, w in self.parts.reports()], np.float32)
        weights /= self.table.microbatches * self.row_shards
        return jnp.broadcast_to(jnp.asarray(weights), self.shapes.losses.shape)

    # --- Ticks --------------------------------------------------------------------

    def tick_inputs(self) -> TickInputs:
        """Return what the ticks read, each leaf (ticks, ...)."""
        codes, microbatches = self.table.as_arrays()
        from_before = self.table.arrivals(FORWARD)
        from_after = self.table.arrivals(BACKWARD)
        arrays = (codes, microbatches, from_before, from_after)
        return TickInputs(*(jnp.asarray(a) for a in arrays))

    def tick(self, held: Held, now: TickInputs) -> tuple[Held, None]:
        """Run a tick: this stage's op in each lane, then the sends both ways."""
        for lane in range(self.table.lanes):
            code, j = now.codes[self.index, lane], now.microbatches[self.index, lane]
            held = self._run_lane(held, lane, code, j)
        return self._send_and_receive(held, now), None

    def _run_lane(
        self, held: Held, lane: int, code: Int[Array, ""], j: Int[Array, ""]
    ) -> Held:
        """Run the op of `code` on microbatch `j`, by a cond for each kind of op
        `lane` runs: in one switch of them all, XLA would copy the buffers one op
        writes and another reads, a tick at a time."""
        ops = {FORWARD: self.forward, BACKWARD: self.backward, WEIGHTS: self.weights}
        for kind in self.table.kinds_in(lane):
            held = jax.lax.cond(code == CODES[kind], ops[kind], _unchanged, held, j)
        return held

    def _send_and_receive(self, held: Held, now: TickInputs) -> Held:
        """Send the forward's output on and the backward's input gradient back, and
        put what arrives in the slot of the microbatch it belongs to."""
        to_next, to_previous = send(held.to_next, 1), send(held.to_previous, -1)
        got = self.slot("inputs", now.from_before[self.index])
        inputs = put(held.inputs, got, to_next)
        got = self.slot("cotangents", now.from_after[self.index])
        cotangents = put(held.cotangents, got, to_previous)
        return held._replace(inputs=inputs, cotangents=cotangents)

    def finish(
        self, held: Held, rest: PyTree, kept: PyTree
    ) -> tuple[PyTree, PyTree, PyTree, PyTree, Metrics]:
        """Return the gradients, the state and the metrics as the shard_map gives
        them out: the parts outside the repeat summed over the stages, and everything
        over the rows' axes, where each device ran its own rows. `rest` and `kept`
        are the state as it came in, whose RNG keys it keeps."""
        rest = gather.keys_of(rest, held.rest)
        kept = gather.keys_of(kept, held.kept)
        d_others = jax.lax.psum(held.d_others, AXIS)
        metrics = jax.lax.psum(held.metrics, AXIS)
        reported = jax.lax.psum(self._reported(held.losses), AXIS)
        reported |= self._observed(held.losses)
        d_params = held.d_params
        if self.apart:
            summed = jax.lax.psum((d_params, d_others, metrics), self.apart)
            d_params, d_others, metrics = summed
            reported = jax.lax.pmean(reported, self.apart)
        kept = add_axes(kept)  # stage
        d_params, rest = add_axes((d_params, rest), 2)  # stage, round
        return d_params, d_others, rest, kept, _merged(metrics, reported)

    def _reported(self, losses: Losses) -> Metrics:
        """Return the means over microbatches of the losses this stage's layers
        report, each name's summed over its layers. What they observe is left out:
        `_observed` names them."""
        means: Metrics = {}
        for i, (name, _) in enumerate(self.parts.reports()):
            if observe.is_observed(name):
                continue
            total = losses[:, i].sum() / self.table.microbatches
            means[name] = means.get(name, jnp.zeros((), jnp.float32)) + total
        return means

    def _observed(self, losses: Losses) -> Metrics:
        """Return the means over microbatches of what the repeat's layers observed,
        every stage's, by their metrics' names (``Parts.observed``)."""
        reports = self.parts.reports()
        at = [i for i, (name, _) in enumerate(reports) if observe.is_observed(name)]
        if not at or self.parts.observed is None:
            return {}
        mine = losses[:, np.asarray(at)] / self.table.microbatches
        every = jax.lax.all_gather(mine, AXIS).reshape(-1, len(at))
        return self.parts.observed(every, [reports[i][0] for i in at])


def _unchanged(held: Held, *_: object) -> Held:
    return held


def _float32_zeros(tree: PyTree) -> PyTree:
    return jax.tree.map(lambda a: jnp.zeros(a.shape, jnp.float32), tree)


def _merged(metrics: Metrics, more: Metrics) -> Metrics:
    """Return `metrics` and `more`, a name both have summed."""
    return metrics | {k: v + metrics[k] if k in metrics else v for k, v in more.items()}
