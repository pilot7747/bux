"""Run a scanned repeat's layers as a pipeline over the mesh's stages.

With `mesh.stage` above 1, the longest scanned repeat outside any other splits its
stacked layers over `stage` (``Scanned.pipe``), and ``run`` runs them by the run's
pipeline method (bux.parallel.pipeline). A stage gets the stream and the rows' tokens
and documents, split into microbatches with it, and nothing else: a pass that reads a
tap from before the repeat, or leaves one changed for after it, is an error.

A pipeline that schedules its own backward runs the rest of the model a microbatch at
a time, the part before the repeat on the first stage and the part after it on the
last. Both are the model's own forward, cut at the repeat: ``stream_into`` runs it up
to the repeat and returns the stream there, and under ``leaving`` the repeat gives
back a stream it is handed, without running its layers.
"""

import contextlib
import contextvars
from collections.abc import Callable, Generator
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import parallel
from bux.architecture.common.segments import Segments
from bux.architecture.registry import Loss, Stream, Taps
from bux.architecture.repeats import pure, remat, stacked, tracing
from bux.parallel import pipeline


class Rows(NamedTuple):
    """What a pipeline splits into microbatches with the stream."""

    segments: Segments | None
    tokens: Int[Array, "b t"] | None


class _Reached(Exception):
    """Raised at the pipelined repeat by a forward that stops there."""

    def __init__(self, x: Stream):
        super().__init__("the forward reached the pipelined repeat")
        self.x = x


# The stream the pipelined repeat gives back without running, or None to stop there.
_CUT = contextvars.ContextVar[tuple[Stream | None] | None](
    "pipelined_cut", default=None
)


def stream_into(forward: Callable[[], object]) -> Stream:
    """Run `forward`, a model's, up to its pipelined repeat, and return the stream
    that enters the repeat."""
    token = _CUT.set((None,))
    try:
        forward()
    except _Reached as reached:
        return reached.x
    finally:
        _CUT.reset(token)
    raise ValueError("the forward has no pipelined repeat to stop at")


@contextlib.contextmanager
def leaving(y: Stream) -> Generator[None]:
    """Run forwards in which the pipelined repeat gives back `y` without running its
    layers, and the losses reported before it are dropped: the forward up to the
    repeat counts them (``stream_into``)."""
    token = _CUT.set((y,))
    try:
        yield
    finally:
        _CUT.reset(token)


def _cut(x: Stream, taps: Taps) -> Stream:
    """Stop the forward at the repeat, or give back the stream `leaving` set."""
    (y,) = _CUT.get() or (None,)
    if y is None:
        raise _Reached(x)
    taps.losses.clear()
    return y


def run(
    layers: nnx.Module,
    passes: int,
    rounds: int,
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None = None,
) -> Stream:
    """Run `layers`, stacked by round, by the run's pipeline method: each stage runs its
    share of a round's `passes` on one microbatch at a time (``Stage``), each pass cast
    to `compute`, if given. The losses they report go into `taps`, each the mean over
    microbatches. Cut there (``stream_into``, ``leaving``), it runs none of them."""
    graphdef, params, rest = nnx.split(layers, nnx.Param, ...)
    one = jax.tree.map(lambda a: _one_pass(a, rounds), nnx.merge_state(params, rest))
    _check(rounds * passes, graphdef, one, x, taps, compute)
    if _CUT.get() is not None:
        return _cut(x, taps)
    if rounds == 1:
        params, rest = jax.tree.map(lambda a: a[None], (params, rest))
    stage = Stage.of(graphdef, passes, taps.training, compute)
    rows = Rows(taps.segments, taps.tokens)
    y, rest, losses = parallel.method("pipeline")(stage, params, rest, x, rows)
    nnx.update(layers, rest if rounds > 1 else jax.tree.map(lambda a: a[0], rest))
    stage.report(taps, losses)
    return y


class Stage:
    """A stage's share of a round, run on one microbatch as a function of arrays alone:
    ``(params, rest, x, rows) -> (y, rest, losses)``. Stage s runs `shares[s]` of
    the round's passes, and holds as many as the most any runs, `layers`: it passes
    the stream through the others (``pipeline.spread``). As it traces, `reports`
    keeps the names and weights of the losses."""

    def __init__(
        self,
        graphdef: nnx.GraphDef,
        shares: tuple[int, ...],
        training: bool,
        compute: DTypeLike | None,
    ):
        self.graphdef, self.shares, self.layers = graphdef, shares, max(shares)
        self.training, self.compute = training, compute
        self.reports: list[tuple[str, float]] = []

    @classmethod
    def of(
        cls,
        graphdef: nnx.GraphDef,
        passes: int,
        training: bool,
        compute: DTypeLike | None,
    ) -> "Stage":
        """Return the stage of a round of `passes`, split over the stages as the run in
        force says (``pipeline.shares``)."""
        return cls(graphdef, pipeline.shares(passes), training, compute)

    def __call__(
        self, params: nnx.State, rest: nnx.State, x: Stream, rows: Rows
    ) -> tuple[Stream, nnx.State, Float[Array, "l n"]]:
        local = _stage_taps(rows, self.training)
        state = nnx.merge_state(params, rest)
        y, rest = self._run(state, x, local)
        losses = local.losses
        self.reports = [
            (a.name, a.weight) for a in losses[: len(losses) // self.layers]
        ]
        if not losses:
            return y, rest, jnp.zeros((self.layers, 0))
        return y, rest, jnp.stack([a.value for a in losses]).reshape(self.layers, -1)

    def _run(self, state: nnx.State, x: Stream, taps: Taps) -> tuple[Stream, nnx.State]:
        """Run the stage's layers in runs that trace alike (``tracing.runs``): those
        every stage runs by a scan, and those some stage skips by a cond each
        (``_skipping``). The run's `remat(first=n)` counts them from the stage's
        first."""
        every, rests = min(self.shares), []
        for run in tracing.runs(self.layers, [every, *remat.cuts()]):
            part = stacked.passes(state, run)
            with tracing.passes(run):
                if run.start < every:
                    x, rest = stacked.run(
                        self.graphdef, part, len(run), x, taps, self.compute
                    )
                else:
                    on = self._this_stage_runs()[run.start : run.stop]
                    x, rest = _skipping(self.graphdef, part, on, x, taps, self.compute)
            rests.append(rest)
        return x, stacked.joined(rests)

    def _this_stage_runs(self) -> Bool[Array, " l"]:
        """Return which of its layers the stage on this device runs."""
        table = jnp.asarray(pipeline.layers_run(self.shares))
        return table[jax.lax.axis_index(pipeline.AXIS)]

    def report(self, taps: Taps, losses: Float[Array, "r n k"]) -> None:
        """Add each layer's losses, of values `losses`, to `taps`."""
        if not self.reports:
            return
        for values in losses.reshape(-1, losses.shape[-1]):
            taps.losses.extend(
                Loss(name, weight, v)
                for (name, weight), v in zip(self.reports, values, strict=True)
            )


def _skipping(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    on: Bool[Array, " l"],
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None,
) -> tuple[Stream, nnx.State]:
    """Run the stacked layers of `state` by a scan, as ``stacked.run`` does, each that
    `on` marks: the others pass the stream through, and report losses of 0. A
    pipelined pass leaves the taps as it found them, so the scan carries the stream
    alone."""
    call = pure.Pure(graphdef, taps)
    layer = stacked.scanned_pass(call, compute)

    def run_layer(
        s: nnx.State, x: Stream
    ) -> tuple[Stream, nnx.State, list[Float[Array, ""]]]:
        y, rest, _, losses = layer(s, x, pure.NOTHING)
        return y, rest, losses

    _, _, reported = jax.eval_shape(run_layer, stacked.at(state, 0), x)

    def skip(
        s: nnx.State, x: Stream
    ) -> tuple[Stream, nnx.State, list[Float[Array, ""]]]:
        zeros = [jnp.zeros(a.shape, a.dtype) for a in reported]
        return x, nnx.split_state(s, nnx.Param, ...)[1], zeros

    def step(x: Stream, xs: tuple[nnx.State, Bool[Array, ""]]) -> tuple[Stream, tuple]:
        s, runs = xs
        y, rest, losses = jax.lax.cond(runs, run_layer, skip, s, x)
        return y, (rest, losses)

    y, (rest, losses) = jax.lax.scan(step, x, (state, on))
    pure.apply(
        taps, pure.NOTHING, call, [[v[i] for v in losses] for i in range(len(on))]
    )
    return y, rest


def _stage_taps(rows: Rows, training: bool) -> Taps:
    """Return the taps a stage starts from: its rows', collecting losses if `training`."""
    return Taps(tokens=rows.tokens, segments=rows.segments, training=training)


def _one_pass(
    a: Shaped[Array, "..."] | jax.ShapeDtypeStruct, rounds: int
) -> jax.ShapeDtypeStruct:
    """Return the shape of one pass's slice of a stacked variable, by `rounds`, or of
    a stand-in for it."""
    return jax.ShapeDtypeStruct(a.shape[1 if rounds == 1 else 2 :], a.dtype)


def _check(
    passes: int,
    graphdef: nnx.GraphDef,
    layer: nnx.State,
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None,
) -> None:
    """Raise if a pass of the pipelined repeat of `passes`, of the shapes `layer`,
    reads a tap from before the repeat or leaves one changed for after it."""
    if crossing := _crossing(graphdef, layer, x, taps, compute):
        raise ValueError(
            f"mesh.stage = {pipeline.stages()} pipelines the repeat of {passes} "
            f"passes, but they take {', '.join(crossing)} from before it, or leave "
            "it changed for after it: a pipeline sends its stages the stream alone. "
            "Read each tap in the pass that writes it, or train with mesh.stage = 1"
        )


def _crossing(
    graphdef: nnx.GraphDef,
    layer: nnx.State,
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None,
) -> list[str]:
    """Return what a pass of `layer` would need to cross the stage boundary: the tap it
    reads from before the repeat; else the taps it leaves changed, and the stream if
    the pass changes its shape."""
    local = _stage_taps(Rows(taps.segments, taps.tokens), taps.training)
    try:
        with remat.off():
            y = jax.eval_shape(
                lambda s, x: nnx.merge(graphdef, stacked.cast(s, compute))(x, local),
                layer,
                x,
            )
    except KeyError as e:  # a tap from before the repeat
        if e.args and e.args[0] in taps:
            return [e.args[0]]
        raise
    changed = [k for k, v in local.items() if not pure.same(taps.get(k, []), v)]
    changed += [
        name
        for name, ran in [
            ("nextn", local.nextn_runs > 0),
            ("the heads' outputs", bool(local.heads)),
            ("the single-pass read", local.pending_read is not None),
        ]
        if ran
    ]
    stream = pure.signature(y) != pure.signature(x)
    return sorted(changed) + (["the stream"] if stream else [])
