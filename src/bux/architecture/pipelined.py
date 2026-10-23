"""Run a scanned repeat's layers as a pipeline over the mesh's stages.

With `mesh.stage` above 1, the longest scanned repeat outside any other splits its
stacked layers over `stage` (``Scanned.pipe``), and ``run`` runs them by the run's
pipeline method (bux.parallel.pipeline). A stage gets the stream and the rows' tokens
and documents, split into microbatches with it, and nothing else: a pass that reads a
tap from before the repeat, or leaves one changed for after it, is an error.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int

from bux import parallel
from bux.architecture import pure, remat, stacked
from bux.architecture.registry import Loss, Stream, Taps
from bux.architecture.segments import Segments
from bux.parallel import pipeline


class Rows(NamedTuple):
    """What a pipeline splits into microbatches with the stream."""

    segments: Segments | None
    tokens: Int[Array, "b t"] | None


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
    microbatches."""
    graphdef, params, rest = nnx.split(layers, nnx.Param, ...)
    if rounds == 1:
        params, rest = jax.tree.map(lambda a: a[None], (params, rest))
    state = nnx.merge_state(params, rest)
    _check(rounds * passes, graphdef, state, x, taps, compute)
    stage = Stage(graphdef, passes // pipeline.stages(), taps.training, compute)
    rows = Rows(taps.segments, taps.tokens)
    y, rest, losses = parallel.method("pipeline")(stage, params, rest, x, rows)
    nnx.update(layers, rest if rounds > 1 else jax.tree.map(lambda a: a[0], rest))
    stage.report(taps, losses)
    return y


class Stage:
    """A stage's share of a round, `layers` layers, run on one microbatch as a function
    of arrays alone: ``(params, rest, x, rows) -> (y, rest, losses)``. As it traces,
    `reports` keeps the names and weights of the losses."""

    def __init__(
        self,
        graphdef: nnx.GraphDef,
        layers: int,
        training: bool,
        compute: DTypeLike | None,
    ):
        self.graphdef, self.layers, self.training = graphdef, layers, training
        self.compute = compute
        self.reports: list[tuple[str, float]] = []

    def __call__(
        self, params: nnx.State, rest: nnx.State, x: Stream, rows: Rows
    ) -> tuple[Stream, nnx.State, Float[Array, "l n"]]:
        local = _stage_taps(rows, self.training)
        state = nnx.merge_state(params, rest)
        y, rest = stacked.run(self.graphdef, state, self.layers, x, local, self.compute)
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


def _stage_taps(rows: Rows, training: bool) -> Taps:
    """Return the taps a stage starts from: its rows', collecting losses if `training`."""
    return Taps(tokens=rows.tokens, segments=rows.segments, training=training)


def _check(
    passes: int,
    graphdef: nnx.GraphDef,
    state: nnx.State,
    x: Stream,
    taps: Taps,
    compute: DTypeLike | None,
) -> None:
    """Raise if a pass of the pipelined repeat of `passes`, of stacked `state`, reads
    a tap from before the repeat or leaves one changed for after it."""
    first = stacked.cast(jax.tree.map(lambda a: a[0, 0], state), compute)
    if crossing := _crossing(graphdef, first, x, taps):
        raise ValueError(
            f"mesh.stage = {pipeline.stages()} pipelines the repeat of {passes} "
            f"passes, but they take {', '.join(crossing)} from before it, or leave "
            "it changed for after it: a pipeline sends its stages the stream alone. "
            "Read each tap in the pass that writes it, or train with mesh.stage = 1"
        )


def _crossing(
    graphdef: nnx.GraphDef, layer: nnx.State, x: Stream, taps: Taps
) -> list[str]:
    """Return what a pass of `layer` would need to cross the stage boundary: the tap it
    reads from before the repeat; else the taps it leaves changed, and the stream if
    the pass changes its shape."""
    local = _stage_taps(Rows(taps.segments, taps.tokens), taps.training)
    try:
        with remat.off():
            y = jax.eval_shape(lambda s, x: nnx.merge(graphdef, s)(x, local), layer, x)
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
