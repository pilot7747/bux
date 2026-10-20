"""Parallelism: one mesh of six axes, a layout that says which of them each named
dimension splits over, and the methods that compute what communicates.

    mesh    = mesh(fsdp=8, tensor=2)              # data takes the devices left
    layout  = layout(stream=[context, tensor])    # sequence parallelism
    methods = methods(attention=ring)

Layers stay unaware of sharding: they name the dimensions of their parameters
(``named``) and of a few activations (``shard``), and the compiler propagates the
rest. Every parameter's largest whole dimension also splits over `fsdp`. Outside
``use`` all of it does nothing, so one device runs as if none of it were there.

    mesh      the device grid                     AXES, mesh, active
    settings  a run's settings in force           use, check
    layout    names, resolved to mesh axes        named, spec, shard, stream
    place     where state and batches live        shardings, build, place, rows
    methods   a registry per kind of method       method

docs/parallelism.md explains them.
"""

from bux.parallel import methods
from bux.parallel.layout import DIMENSIONS, NAMES, Name, named, shard, spec, stream
from bux.parallel.mesh import AXES, active, mesh, sizes
from bux.parallel.methods import method
from bux.parallel.place import build, place, rows, shardings
from bux.parallel.settings import check, use

# isort: split
# Last, so that what they import from the modules above is there: their methods
# register themselves.
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
    "named",
    "pipeline",
    "place",
    "recurrence",
    "rows",
    "shard",
    "shardings",
    "sizes",
    "spec",
    "stream",
    "use",
]
