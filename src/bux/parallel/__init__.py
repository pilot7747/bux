"""Parallelism: one mesh of six axes, a layout that says which of them each named
dimension splits over, and the methods that run the parts that communicate.

    mesh    = mesh(fsdp=8, tensor=2)              # data takes the devices left
    layout  = layout(stream=[context, tensor])    # sequence parallelism
    methods = methods(attention=ring)

Layers don't shard anything themselves: they name the dimensions of their parameters
(``named``) and of a few activations (``shard``), and the compiler propagates the
rest. Every parameter's largest whole dimension also splits over `fsdp`. Outside
``use`` all of this does nothing, so one device runs as if it were not there.

    mesh      the device grid                     AXES, mesh, active
    settings  the run's settings in force         use, check
    layout    names, resolved to mesh axes        named, spec, shard, shard_stream
    place     where state and batches live        shardings, build, shard_state,
                                                  shard_batch, microbatches
    methods   one registry per kind of method     method

docs/parallelism.md explains them.
"""

from bux.parallel import methods
from bux.parallel.layout import (
    DIMENSIONS,
    NAMES,
    Name,
    named,
    shard,
    shard_stream,
    spec,
)
from bux.parallel.mesh import AXES, active, mesh, sizes
from bux.parallel.methods import method
from bux.parallel.place import build, microbatches, shard_batch, shard_state, shardings
from bux.parallel.settings import check, use

# isort: split
# Imported last, so that the names above exist when their methods register themselves.
from bux.parallel import context, experts, pipeline, recurrence

__all__ = [
    "AXES",
    "DIMENSIONS",
    "NAMES",
    "Name",
    "active",
    "build",
    "check",
    "context",
    "experts",
    "mesh",
    "method",
    "methods",
    "microbatches",
    "named",
    "pipeline",
    "recurrence",
    "shard",
    "shard_batch",
    "shard_state",
    "shard_stream",
    "shardings",
    "sizes",
    "spec",
    "use",
]
