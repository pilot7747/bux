"""Dimension names, and the mesh axes the run's layout splits each over.

A layer names the dimensions of a parameter where it creates it (`named`, or
``nnx.Param(value, names=(...))``), as bux's own metadata, which Flax ignores, and
those of an activation with `shard`. Outside `~bux.parallel.settings.use` names do
nothing, so one device runs as if there were none::

    self.up = nnx.Linear(d, h, kernel_init=named("embed", "mlp"), rngs=rngs)
    y = shard(self.up(x), "batch", "seq", "mlp")

The names are the fields of ``layout`` (`DIMENSIONS`): ``batch``, ``seq``,
``stream``, ``embed``, ``vocab``, ``heads``, ``kv_heads``, ``mlp``, ``experts``,
``latent``, ``layers`` and ``stages``.
"""

from collections.abc import Collection, Sequence

import jax
from flax import nnx
from jax.core import Tracer
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped

from bux.parallel.mesh import Active, active, manual_axes
from bux.train.config import LayoutConfig

#: The metadata key that holds a parameter's dimension names.
NAMES = "names"
LAYERS = "layers"  # the stacked dimension of a scanned repeat
#: Every field of the layout that names a dimension, the names a layer may use.
DIMENSIONS = tuple(
    n for n in LayoutConfig.model_fields if n not in ("type", "fsdp", "zero")
)

Split = tuple[str, ...]  # the mesh axes one dimension splits over; () keeps it whole
Entry = str | tuple[str, ...] | None  # one dimension of a PartitionSpec
#: A dimension's name, None for no name, or ``(name, count)`` for a dimension that
#: flattens `count` of them, such as ``("kv_heads", 2)`` for 2 kv heads of 64 dims.
#: Such a dimension splits only at the boundaries between them.
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
    return as_split(getattr(layout, name))


def as_split(entry: Entry) -> Split:
    """Return the mesh axes that one entry of a partition spec, or a dimension's
    field of the layout, names: none, one or several."""
    if entry is None:
        return ()
    return (entry,) if isinstance(entry, str) else tuple(entry)


def axes_of(spec: P) -> Split:
    """Return every mesh axis that `spec` splits over, in order."""
    return tuple(a for entry in spec for a in as_split(entry))


def split_axes(on: Active, name: Name, skip: Collection[str] = ()) -> Split:
    """Return the axes of `on`'s mesh that split the dimension `name`: those its
    layout gives it, but those of size 1 and those in `skip`."""
    return tuple(a for a in axes(on.layout, name) if on.size(a) > 1 and a not in skip)


def resolve(names: Sequence[Name], shape: tuple[int, ...], on: Active) -> list[Split]:
    """Return the mesh axes that each dimension of an array splits over.

    For each dimension, take the axes of its name, minus axes of size 1, axes an
    earlier dimension already uses, and axes that a shard_map being traced handles
    manually. If those axes don't divide the dimension's size (or its count), use none.
    """
    if len(names) != len(shape):
        raise ValueError(f"{len(names)} names for {len(shape)} dimensions: {names}")
    used = set(manual_axes())
    splits = []
    for name, size in zip(names, shape):
        split = split_axes(on, name, skip=used)
        if (name[1] if isinstance(name, tuple) else size) % on.size(*split):
            split = ()
        used.update(split)
        splits.append(split)
    return splits


def fsdp(
    splits: list[Split], shape: tuple[int, ...], on: Active, names: Sequence[Name]
) -> list[Split]:
    """Split the first whole dimension over FSDP's axes (`layout.fsdp`) that no
    dimension splits over yet, or if none divides by them all, over as many of them as
    one does, from the first. The first, so that each shard is whole rows of the
    array, which XLA gathers and scatters without a transpose: an expert stack's
    experts, a kernel's inputs. Never the stacked ``layers`` of a scanned repeat: a
    scan reads one layer at a time. And one that its own axes leave whole only if no
    other is free."""
    taken = {a for s in splits for a in s}
    over = [a for a in as_split(on.layout.fsdp) if on.size(a) > 1 and a not in taken]
    for k in range(len(over), 0, -1):
        i = _first_whole(splits, shape, names, on, on.size(*over[:k]))
        if i is not None:
            return [*splits[:i], tuple(over[:k]), *splits[i + 1 :]]
    return splits


def _first_whole(
    splits: list[Split],
    shape: tuple[int, ...],
    names: Sequence[Name],
    on: Active,
    n: int,
) -> int | None:
    """The first dimension that nothing splits and `n` divides, but `layers`: of
    those, first one whose name no axis of `on`'s mesh is for. Where its own axes do
    not divide a dimension, as tensor=4 does not CSA's 2 output groups, the layer's
    activations keep it whole too, and XLA, which would split them as the parameter
    is to compute with it, rematerializes them."""
    whole = [
        i
        for i, s in enumerate(splits)
        if not s and shape[i] % n == 0 and names[i] != LAYERS
    ]
    free = [i for i in whole if not split_axes(on, names[i])]
    return (free or whole or [None])[0]


def partition(splits: list[Split]) -> P:
    """Return the partition spec of dimensions that split over `splits`."""
    return P(*(None if not s else s[0] if len(s) == 1 else s for s in splits))


def spec(names: Sequence[Name], shape: tuple[int, ...]) -> P:
    """Return the partition spec of an array of `shape` whose dimensions are `names`.

    Each dimension splits over the axes the layout gives its name, but those of size
    1, those an earlier dimension took, and those a shard_map being traced maps; and
    over none if they do not divide it. Outside `~bux.parallel.settings.use` it is
    ``P()``. Parameters split further by FSDP: `~bux.parallel.place.shardings`.

    Example::

        >>> with parallel.use(MeshConfig(tensor=4)):  # on 4 devices
        ...     spec(("embed", "mlp"), (1024, 4096))
        PartitionSpec(None, 'tensor')
    """
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
    """Constrain `x` to split as the `names` of its dimensions say (`spec`).

    Outside `~bux.parallel.settings.use` it returns `x` as it is. Called eagerly, it
    moves `x` to the mesh. Inside a shard_map it splits over the axes left free.

    Args:
        x: An activation.
        *names: A name for each of its dimensions, None for one that stays whole.

    Example::

        logits = shard(logits, "batch", "heads", "seq", None)
    """
    if (on := active()) is None:
        return x
    split = partition(resolve(names, x.shape, on))
    if not isinstance(x, Tracer) or not manual_axes():
        # So also a gradient's tracer of an array that is not on the mesh yet.
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
    """Wrap `init` to name the dimensions of its parameters, for ``kernel_init=``.

    Args:
        *names: A name for each dimension of the parameter (`Name`).
        init: The initializer to wrap: LeCun normal, as `flax.nnx.Linear`'s.

    Raises:
        ValueError: If a name is not one of `DIMENSIONS`, with or without a mesh.

    Example::

        nnx.Linear(d_model, heads * head_dim, kernel_init=named("embed", "heads"))
    """
    for name in names:
        axes(LayoutConfig(), name)  # an unknown name fails here, with or without a mesh
    return nnx.with_metadata(init, names=names)  # under NAMES
