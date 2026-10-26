"""Parallelism: one mesh of six axes, a layout that says which of them each named
dimension splits over, and the methods that run the parts that communicate.

.. code-block:: bux

    mesh    = mesh(fsdp=8, tensor=2)              # data takes the devices left
    layout  = layout(stream=[context, tensor])    # sequence parallelism
    methods = methods(attention=ring)

Layers don't shard anything themselves: they name the dimensions of their parameters
(`~bux.parallel.layout.named`) and of a few activations
(`~bux.parallel.layout.shard`), and the compiler propagates the rest. Every
parameter's first whole dimension also splits over ``fsdp``. Outside
`~bux.parallel.settings.use` all of this does nothing, so one device runs as if it
were not there.

The package re-exports its modules' public names, as ``parallel.use``:

- `bux.parallel.mesh`, the device grid: ``AXES``, ``mesh``, ``active``.
- `bux.parallel.settings`, the run's settings in force: ``use``, ``check``.
- `bux.parallel.layout`, names resolved to mesh axes: ``named``, ``spec``,
  ``splits``, ``shard``, ``shard_stream``, ``axes_of``.
- `bux.parallel.place`, where state and batches live: ``shardings``, ``build``,
  ``shard_state``, ``shard_batch``, ``local_rows``, ``microbatches``.
- `bux.parallel.methods`, one registry per kind of method: ``method``.
- `bux.parallel.gather`, a step's parameters gathered once and its gradients
  reduced once: ``whole``, ``rows``, ``splits``, ``gather``, ``scatter``, ``apart``.

The methods themselves are in `bux.parallel.context` and `bux.parallel.ring`
(attention), `bux.parallel.experts` and `bux.parallel.nccl_ep` (the MoE),
`bux.parallel.recurrence`, `bux.parallel.tensor`, and `bux.parallel.pipeline` and
`bux.parallel.scheduled` (pipelines). The parallelism guide (:doc:`/parallelism/index`)
explains them.

Example::

    from flax import nnx

    from bux import parallel
    from bux.architecture import Model
    from bux.train.config import MeshConfig

    # run: a compiled run, e.g. bux.lang.compile_runs(source)[0]
    with parallel.use(MeshConfig(fsdp=8)):
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
"""

from bux.parallel import gather, methods
from bux.parallel.layout import (
    DIMENSIONS,
    NAMES,
    Name,
    axes_of,
    named,
    shard,
    shard_stream,
    spec,
    splits,
)
from bux.parallel.mesh import AXES, active, mesh, sizes
from bux.parallel.methods import method
from bux.parallel.place import (
    build,
    local_rows,
    microbatches,
    shard_batch,
    shard_state,
    shardings,
)
from bux.parallel.settings import check, use

# isort: split
# Imported last, so that the names above exist when their methods register themselves.
from bux.parallel import (
    context,
    experts,
    flash,
    nccl_ep,
    pipeline,
    recurrence,
    ring,
    scheduled,
    tensor,
)

__all__ = [
    "AXES",
    "DIMENSIONS",
    "NAMES",
    "Name",
    "active",
    "axes_of",
    "build",
    "check",
    "context",
    "experts",
    "flash",
    "gather",
    "local_rows",
    "mesh",
    "method",
    "methods",
    "microbatches",
    "named",
    "nccl_ep",
    "pipeline",
    "recurrence",
    "ring",
    "scheduled",
    "shard",
    "shard_batch",
    "shard_state",
    "shard_stream",
    "shardings",
    "sizes",
    "spec",
    "splits",
    "tensor",
    "use",
]
