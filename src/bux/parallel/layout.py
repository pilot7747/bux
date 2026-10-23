"""Dimension names, and the mesh axes the run's layout splits each over.

A layer names the dimensions of a parameter where it creates it (``named``, or
``nnx.Param(value, names=(...))``), as bux's own metadata, which Flax ignores, and
those of an activation with ``shard``. Outside ``use`` names do nothing, so one
device runs as if there were none.
"""

from collections.abc import Sequence

import jax
from flax import nnx
from jax.core import Tracer
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped

from bux.parallel.mesh import Active, active
from bux.train.config import LayoutConfig

NAMES = "names"  # the metadata key that holds a parameter's dimension names
LAYERS = "layers"  # the stacked dimension of a scanned repeat
# Every field of the layout that names a dimension.
DIMENSIONS = tuple(n for n in LayoutConfig.model_fields if n not in ("type", "zero"))

Split = tuple[str, ...]  # the mesh axes one dimension splits over; () keeps it whole
# A dimension's name: a string, None for no name, or (name, count) for a dimension
# that flattens `count` of them, such as ("kv_heads", 2) for 2 kv heads of 64 dims.
# Such a dimension splits only at the boundaries between them.
Name = str | tuple[str, int] | None
LECUN = nnx.initializers.lecun_normal()  # nnx.Linear's kernel_init


def axes(layout: LayoutConfig, name: Name) -> Split:
    """Return the mesh axes that `layout` gives the dimension `name`."""
    if isinstance(name, tuple):
        name = name[0]
    if name is None:
        return ()
    if name not in DIMENSIONS:
        raise ValueError(f"'{name}' is not a dimension; the names are {DIMENSIONS}")
    a = getattr(layout, name)
    return () if a is None else (a,) if isinstance(a, str) else tuple(a)


def resolve(names: Sequence[Name], shape: tuple[int, ...], on: Active) -> list[Split]:
    """Return the mesh axes that each dimension of an array splits over.

    For each dimension, take the axes of its name, minus axes of size 1, axes an
    earlier dimension already uses, and axes that a shard_map being traced handles
    manually. If those axes don't divide the dimension's size (or its count), use none.
    """
    if len(names) != len(shape):
        raise ValueError(f"{len(names)} names for {len(shape)} dimensions: {names}")
    used = set(jax.sharding.get_abstract_mesh().manual_axes)
    splits = []
    for name, size in zip(names, shape):
        split = tuple(
            a for a in axes(on.layout, name) if on.size(a) > 1 and a not in used
        )
        if (name[1] if isinstance(name, tuple) else size) % on.size(*split):
            split = ()
        used.update(split)
        splits.append(split)
    return splits


def fsdp(
    splits: list[Split], shape: tuple[int, ...], on: Active, names: Sequence[Name]
) -> list[Split]:
    """Split the largest whole dimension that `fsdp` divides over `fsdp`, unless one
    already is. Never the stacked `layers` of a scanned repeat: a scan reads one layer
    at a time."""
    n = on.size("fsdp")
    if n == 1 or any("fsdp" in s for s in splits):
        return splits
    whole = [
        i
        for i, s in enumerate(splits)
        if not s and shape[i] % n == 0 and names[i] != LAYERS
    ]
    if not whole:
        return splits
    i = max(whole, key=lambda i: shape[i])
    return [*splits[:i], ("fsdp",), *splits[i + 1 :]]


def partition(splits: list[Split]) -> P:
    return P(*(None if not s else s[0] if len(s) == 1 else s for s in splits))


def spec(names: Sequence[Name], shape: tuple[int, ...]) -> P:
    """Return the partition spec of an array of `shape` whose dimensions are `names`."""
    on = active()
    if on is None:
        return P()
    return partition(resolve(names, shape, on))


def splits(names: Sequence[Name], shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return how many ways each dimension of an array of `shape` named `names` splits."""
    on = active()
    if on is None:
        return (1,) * len(shape)
    return tuple(on.size(*split) for split in resolve(names, shape, on))


def shard(x: Shaped[Array, "*s"], *names: Name) -> Shaped[Array, "*s"]:
    """Constrain `x` to split as the `names` of its dimensions say.

    Outside ``use`` it returns `x` as it is. Called eagerly, it moves `x` to the mesh.
    """
    if (on := active()) is None:
        return x
    split = partition(resolve(names, x.shape, on))
    if not isinstance(x, Tracer):
        return jax.lax.with_sharding_constraint(x, NamedSharding(on.mesh, split))
    # A bare spec is of the mesh being traced, which knows a shard_map's manual axes.
    return jax.lax.with_sharding_constraint(x, split)


def shard_stream(x: Shaped[Array, "b t *r"]) -> Shaped[Array, "b t *r"]:
    """Shard the residual stream: rows over `batch`, positions over `stream`."""
    return shard(x, "batch", "stream", *(None,) * (x.ndim - 2))


def named(
    *names: Name,
    init: nnx.Initializer = LECUN,
) -> nnx.Initializer:
    """Wrap `init` to name the dimensions of its parameters, for ``kernel_init=``."""
    for name in names:
        axes(LayoutConfig(), name)  # an unknown name fails here, with or without a mesh
    return nnx.with_metadata(init, names=names)  # under NAMES
