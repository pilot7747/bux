"""The split of a scheduled pipeline's passes over its stages, where the run leaves
`methods.stages` to bux: the one that balances the stages' work.

Under 1f1b and zbh1 the first stage also runs the model up to the pipelined repeat
(the embedding), and the last runs the rest of it to the loss (the head), so with as
many passes on each, the last stage is the slowest and the others wait for it. The
work is a row's forward in FLOPs (``flops.count``), in each part of the model cut at
the repeat as a schedule cuts it (``pipelined.stream_into``, ``pipelined.leaving``),
and ``pipeline.balanced`` picks the split whose slowest stage does the least.
"""

import contextlib
import functools
from collections.abc import Generator

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
from bux.lang import ir
from bux.parallel import pipeline, schedule
from bux.train import flops
from bux.train.config import MethodsConfig
from bux.train.data import Batch, plain
from bux.train.objective import Objective, objective


def balanced(run: ir.Run) -> MethodsConfig:
    """Return the run's methods, with the stages that balance their work where it
    pipelines by a schedule and does not give them."""
    t = run.train
    methods = t.methods
    scheduled = methods.pipeline in schedule.TABLES
    if t.mesh.stage == 1 or methods.stages is not None or not scheduled:
        return methods
    passes, work = _work(run)
    split = pipeline.balanced(passes, t.mesh.stage, work)
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
        [repeat] = [m for _, m in nnx.iter_graph(model) if _pipelined(m)]
        graphdef, state = nnx.split(model)
        before = functools.partial(_before, graphdef)
        after = functools.partial(_after, graphdef, loss)
        run_pass, one = _first_pass(repeat)
        x = jax.eval_shape(before, state, row)
        up_to = flops.count(before, state, row)
        to_loss = flops.count(after, state, row, x)
        each = flops.count(run_pass, one, row, x)
    return repeat.times, pipeline.Work(up_to, each, to_loss - up_to)


@contextlib.contextmanager
def _on_stages(run: ir.Run) -> Generator[None]:
    """Put the run's layout and methods in force on a mesh of its stages alone,
    without devices: the repeat pipelines, and the rest of the model runs whole."""
    t = run.train
    shape = tuple(t.mesh.stage if a == pipeline.AXIS else 1 for a in parallel.AXES)
    auto = (AxisType.Auto,) * len(parallel.AXES)
    with parallel.use(
        AbstractMesh(shape, parallel.AXES, axis_types=auto), t.layout, t.methods
    ):
        yield


def _pipelined(module: object) -> bool:
    return isinstance(module, Scanned) and module.rounds > 0


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


def _first_pass(repeat: Scanned) -> tuple[functools.partial, nnx.State]:
    """Return a function that runs a pass of the pipelined `repeat`, `(state, row, x)
    -> y`, and the shapes of a pass's state."""
    graphdef, stacked = nnx.split(repeat.layers)
    one = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), stacked)
    return functools.partial(_pass, graphdef), one


def _pass(graphdef: nnx.GraphDef, state: nnx.State, row: Batch, x: Stream) -> Stream:
    tokens, positions = jnp.asarray(row.input_ids), jnp.asarray(row.position_ids)
    taps = Taps(tokens=tokens, segments=Segments.of(positions), training=True)
    return nnx.merge(graphdef, state)(x, taps)
