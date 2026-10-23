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
import numpy as np
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import parallel
from bux.architecture import pure, remat, stacked
from bux.architecture.registry import Loss, Stream, Taps
from bux.architecture.segments import Segments
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
    first = jax.tree.map(lambda a: _layer_of(a, rounds), nnx.merge_state(params, rest))
    _check(rounds * passes, graphdef, first, x, taps, compute)
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
    """A stage's share of a round, `layers` layers, run on one microbatch as a function
    of arrays alone: ``(params, rest, x, rows) -> (y, rest, losses)``. As it traces,
    `reports` keeps the names and weights of the losses. Where `runs` is given, stage
    s runs its layers `runs[s]` marks, and passes the stream through the others."""

    def __init__(
        self,
        graphdef: nnx.GraphDef,
        layers: int,
        training: bool,
        compute: DTypeLike | None,
        runs: Bool[np.ndarray, "s l"] | None = None,
    ):
        self.graphdef, self.layers, self.training = graphdef, layers, training
        self.compute, self.runs = compute, runs
        self.reports: list[tuple[str, float]] = []

    @classmethod
    def of(
        cls,
        graphdef: nnx.GraphDef,
        passes: int,
        training: bool,
        compute: DTypeLike | None,
    ) -> "Stage":
        """Return the stage of a round of `passes`, split as the run in force says."""
        split = pipeline.uneven()
        runs = None if split is None else pipeline.layers_run(split)
        return cls(graphdef, passes // pipeline.stages(), training, compute, runs)

    def __call__(
        self, params: nnx.State, rest: nnx.State, x: Stream, rows: Rows
    ) -> tuple[Stream, nnx.State, Float[Array, "l n"]]:
        local = _stage_taps(rows, self.training)
        state = nnx.merge_state(params, rest)
        if self.runs is None:
            y, rest = stacked.run(
                self.graphdef, state, self.layers, x, local, self.compute
            )
        else:
            runs = jnp.asarray(self.runs)[jax.lax.axis_index(pipeline.AXIS)]
            y, rest = _skipping(self.graphdef, state, runs, x, local, self.compute)
        losses = local.losses
        self.reports = [
            (a.name, a.weight) for a in losses[: len(losses) // self.layers]
        ]
        if not losses:
            return y, rest, jnp.zeros((self.layers, 0))
        return y, rest, jnp.stack([a.value for a in losses]).reshape(self.layers, -1)

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
    runs: Bool[Array, " l"],
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None,
) -> tuple[Stream, nnx.State]:
    """Run the stacked layers of `state` by a scan, as ``stacked.run`` does, each that
    `runs` marks: the others pass the stream through, and report losses of 0. A
    pipelined pass leaves the taps as it found them, so the scan carries the stream
    alone."""
    call = pure.Pure(graphdef, taps)

    def run(s: nnx.State, x: Stream) -> tuple[Stream, nnx.State, list[Array]]:
        y, rest, _, losses = call(stacked.cast(s, compute), x, pure.NOTHING)
        return y, rest, losses

    layer = remat.checkpoint(run, prevent_cse=False, scanned=True)
    shapes = jax.eval_shape(layer, jax.tree.map(lambda a: a[0], state), x)

    def skip(s: nnx.State, x: Stream) -> tuple[Stream, nnx.State, list[Array]]:
        zeros = [jnp.zeros(a.shape, a.dtype) for a in shapes[2]]
        return x, nnx.split_state(s, nnx.Param, ...)[1], zeros

    def step(x: Stream, xs: tuple[nnx.State, Array]) -> tuple[Stream, tuple]:
        s, on = xs
        y, rest, losses = jax.lax.cond(on, layer, skip, s, x)
        return y, (rest, losses)

    y, (rest, losses) = jax.lax.scan(step, x, (state, runs))
    pure.apply(
        taps, pure.NOTHING, call, [[v[i] for v in losses] for i in range(len(runs))]
    )
    return y, rest


def _stage_taps(rows: Rows, training: bool) -> Taps:
    """Return the taps a stage starts from: its rows', collecting losses if `training`."""
    return Taps(tokens=rows.tokens, segments=rows.segments, training=training)


def _layer_of(
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
