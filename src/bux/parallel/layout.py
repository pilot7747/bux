"""Names: what each dimension of a parameter or an activation means, resolved to the
mesh axes the run's layout splits it over.

A layer names the dimensions of a parameter where it makes it, as metadata of its
own (``named``, or ``nnx.Param(value, names=(...))``), which Flax leaves alone; and
of an activation where it knows what they mean (``shard``). Outside ``use`` names do
nothing, so one device runs as if there were none.
"""

from collections.abc import Sequence

import jax
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped

from bux.parallel.mesh import Active, active
from bux.train.config import LayoutConfig

NAMES = "names"  # the metadata key a parameter's names are under
LAYERS = "layers"  # the stacked dimension of a scanned repeat
# The names a dimension can have: every field of the layout that is a dimension.
DIMENSIONS = tuple(n for n in LayoutConfig.model_fields if n not in ("type", "zero"))

Split = tuple[str, ...]  # the mesh axes one dimension splits over, () for whole
# What a dimension is: a name, None for nothing, or a name and a count for a dimension
# that flattens `count` of them, ("kv_heads", 2) for 2 kv heads of 64 dims. That one
# splits only between them.
Name = str | tuple[str, int] | None
LECUN = nnx.initializers.lecun_normal()  # nnx.Linear's kernel_init


def axes(layout: LayoutConfig, name: Name) -> Split:
    """The axes `layout` gives the dimension `name`."""
    if isinstance(name, tuple):
        name = name[0]
    if name is None:
        return ()
    if name not in DIMENSIONS:
        raise ValueError(f"'{name}' is not a dimension; the names are {DIMENSIONS}")
    a = getattr(layout, name)
    return () if a is None else (a,) if isinstance(a, str) else tuple(a)


def resolve(names: Sequence[Name], shape: tuple[int, ...], on: Active) -> list[Split]:
    """Dimension by dimension, its name's axes, less those of size 1 and those an
    earlier dimension took: all of them if their sizes divide it (or its count), else
    none."""
    if len(names) != len(shape):
        raise ValueError(f"{len(names)} names for {len(shape)} dimensions: {names}")
    taken: set[str] = set()
    splits = []
    for name, n in zip(names, shape):
        split = tuple(
            a for a in axes(on.layout, name) if on.size(a) > 1 and a not in taken
        )
        if (name[1] if isinstance(name, tuple) else n) % on.size(*split):
            split = ()
        taken.update(split)
        splits.append(split)
    return splits


def fsdp(
    splits: list[Split], shape: tuple[int, ...], on: Active, names: Sequence[Name]
) -> list[Split]:
    """`splits` with the largest whole dimension that `fsdp` divides split over it,
    unless a dimension already is. A scanned repeat's stacked `layers` never is: a
    scan takes a layer at a time."""
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
    """The partition spec of an array of `shape` whose dimensions are `names`."""
    on = active()
    if on is None:
        return P()
    return partition(resolve(names, shape, on))


def shard(x: Shaped[Array, "*s"], *names: Name) -> Shaped[Array, "*s"]:
    """`x`, constrained to split as its dimensions' `names` say; as it is outside
    ``use``."""
    if active() is None:
        return x
    return jax.lax.with_sharding_constraint(x, spec(names, x.shape))


def stream(x: Shaped[Array, "b t *r"]) -> Shaped[Array, "b t *r"]:
    """The residual stream between branches: rows over `batch`, positions over
    `stream`, the rest whole."""
    return shard(x, "batch", "stream", *(None,) * (x.ndim - 2))


def named(
    *names: Name,
    init: nnx.Initializer = LECUN,
) -> nnx.Initializer:
    """`init`, making parameters whose dimensions are `names`: for `kernel_init=`."""
    for name in names:
        axes(LayoutConfig(), name)  # an unknown name fails here, with or without a mesh
    return nnx.with_metadata(init, names=names)  # under NAMES
