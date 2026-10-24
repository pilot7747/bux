"""The split of a pipeline's passes over its stages, where the run leaves
`methods.stages` to bux: the one that balances the stages' work, which a run's
checkpoints keep.

Under 1f1b and zbh1 the first stage also runs the model up to the pipelined repeat
(the embedding), and the last runs the rest of it to the loss (the head), so with as
many passes on each, the last stage is the slowest and the others wait for it. The
work is a row's forward in FLOPs (``flops.count``), in each part of the model cut at
the repeat as a schedule cuts it (``pipelined.stream_into``, ``pipelined.leaving``),
and ``pipeline.balanced`` picks the split whose slowest stage does the least. Under
gpipe every stage runs the rest of the model, so its stages split the passes evenly.
"""

import contextlib
import functools
from collections.abc import Callable, Generator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AbstractMesh, AxisType
from jaxtyping import Float

from bux import kernels, parallel
from bux.architecture import Model, Scanned
from bux.architecture.common.segments import Segments
from bux.architecture.registry import Stream, Taps
from bux.architecture.repeats import pipelined, remat
from bux.architecture.repeats.repeat import pipelined_repeat
from bux.lang import ir
from bux.parallel import pipeline, schedule
from bux.train import flops
from bux.train.config import MethodsConfig
from bux.train.data import Batch, plain
from bux.train.objective import Objective, objective


def balanced(run: ir.Run) -> MethodsConfig:
    """Return the run's methods, with the stages that balance their work where it
    pipelines in one round and does not give them."""
    t = run.train
    methods, n = t.methods, t.mesh.stage
    if n == 1 or methods.stages is not None or methods.rounds > 1:
        return methods
    passes, work = _work(run)
    if methods.pipeline in schedule.TABLES:
        split = pipeline.balanced(passes, n, work)
    else:
        split = pipeline.even(passes, n)
    return methods.model_copy(update={"stages": list(split)})


def _work(run: ir.Run) -> tuple[int, pipeline.Work]:
    """Return the passes of the run's pipelined repeat, and the FLOPs of a row's
    forward in each part of its model: counted on its stages alone, from shapes, with
    remat and kernels off, which would add work or hide it."""
    t = run.train
    loss = objective(t.loss, run.arch)
    row = _row(t.seq_len)
    with _on_stages(run), remat.off(), kernels.off():
        model = nnx.eval_shape(lambda: Model(run.arch, rngs=nnx.Rngs(t.seed)))
        _, repeat = pipelined_repeat(model)
        graphdef, state = nnx.split(model)
        before = functools.partial(_before, graphdef)
        after = functools.partial(_after, graphdef, loss)
        run_pass, one = _one_pass(repeat)
        x = jax.eval_shape(before, state, row)
        up_to = _of_one_document(t.seq_len, before, state, row)
        to_loss = _of_one_document(t.seq_len, after, state, row, x)  # and up_to
        each = _of_one_document(t.seq_len, run_pass, one, row, x)
    return repeat.times, pipeline.Work(up_to, each, to_loss - up_to)


def _of_one_document(t: int, f: Callable[..., Any], *args: Any) -> float:
    """Return the FLOPs of `f` on a row of `t` tokens, all of one document."""
    return flops.count(f, *args).one_document(t)


@contextlib.contextmanager
def _on_stages(run: ir.Run) -> Generator[None]:
    """Put the run's layout and methods in force on a mesh of its stages alone,
    without devices: the repeat pipelines, and the rest of the model runs whole."""
    t = run.train
    shape = tuple(t.mesh.stage if a == pipeline.AXIS else 1 for a in parallel.AXES)
    auto = (AxisType.Auto,) * len(parallel.AXES)
    grid = AbstractMesh(shape, parallel.AXES, axis_types=auto)
    with parallel.use(grid, t.layout, t.methods):
        yield


def _row(seq_len: int) -> Batch:
    """Return a batch of one row of `seq_len` tokens, which the counts trace."""
    return plain(np.zeros((1, seq_len), np.int32))


def _before(graphdef: nnx.GraphDef, state: nnx.State, row: Batch) -> Stream:
    """Return the stream that enters the pipelined repeat: the first stage's part."""
    model = nnx.merge(graphdef, state)
    return pipelined.stream_into(
        lambda: model.training_heads(row.input_ids, row.position_ids)
    )


def _after(
    graphdef: nnx.GraphDef, loss: Objective, state: nnx.State, row: Batch, y: Stream
) -> Float[jax.Array, ""]:
    """Return the loss from the stream `y` that leaves the pipelined repeat: the last
    stage's part, which traces the first stage's too."""
    model = nnx.merge(graphdef, state)
    with pipelined.leaving(y):
        heads, reported = model.training_heads(row.input_ids, row.position_ids)
    per_token, per_row = loss(heads, row, reported)
    return per_token.loss + per_row.loss


def _one_pass(repeat: Scanned) -> tuple[functools.partial, nnx.State]:
    """Return a function that runs a pass of the pipelined `repeat`, `(state, row, x)
    -> y`, and the shapes of a pass's state."""
    graphdef, stacked = nnx.split(repeat.layers)
    return functools.partial(_pass, graphdef), pipeline.stand_ins(stacked, 1)


def _pass(graphdef: nnx.GraphDef, state: nnx.State, row: Batch, x: Stream) -> Stream:
    """Return the stream after a pass of state `state` on `x`, as a stage runs it."""
    tokens, positions = jnp.asarray(row.input_ids), jnp.asarray(row.position_ids)
    taps = Taps(tokens=tokens, segments=Segments.of(positions), training=True)
    return nnx.merge(graphdef, state)(x, taps)
