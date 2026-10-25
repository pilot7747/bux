"""A training step's parameters gathered once, and its gradients reduced once
(`bux.train.trainer`): ``methods(gather=step)`` and ``methods(accumulate=...)``.

Every axis is Auto, and XLA cannot move a collective out of a loop: a step of n
microbatches would gather each layer's parameters as it runs, n times forward and n
back, and reduce the gradients n times. `whole` gathers them once, before the
loop. Gradients reduced once need an array that each device holds a part of the sum
of, which JAX types only on Explicit or Manual axes (``P(unreduced=...)``). So a step
whose rows are independent runs in a shard_map manual over the axes that split them
(`rows`): there each device gathers its shard of every parameter (`gather`), sums
its own rows' gradients over the microbatches, and sums the sums over the devices
once (`scatter`), each device keeping its shard. A pipeline's step does the same in
its own shard_map, made manual over those axes too (`apart`).

- `whole`: the parameters as a step uses them, whole over FSDP's axes.
- `rows`: the axes that split the batch's rows, manual in the shard_map.
- `rows_span_processes`: whether they cross hosts, where reducing once pays.
- `splits`: each parameter's split over them.
- `gather`: a device's shard of a parameter, gathered whole.
- `scatter`: a device's part of a gradient, summed over them, as a shard.
- `added`: a microbatch's gradients added into the step's sums.
- `own_streams`: each device's own RNG keys, for its own rows' masks.
- `apart`: says the step computes its rows apart, which a pipeline reads.
"""

import contextlib
import contextvars
from collections.abc import Generator

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, PyTree, Shaped

from bux.parallel.layout import as_split, axes_of, partition, split_axes
from bux.parallel.mesh import active, manual_axes
from bux.parallel.place import shardings

Axes = tuple[str, ...]  # mesh axes, by name


def whole(params: nnx.State) -> nnx.State:
    """Constrain `params` to how a step uses them: split by their names alone, whole
    over FSDP's axes. Each device then holds every parameter it computes with."""
    if active() is None:
        return params
    return jax.lax.with_sharding_constraint(params, shardings(params, gathered=True))


def rows() -> Axes:
    """Return the axes of the run's mesh that split a batch's rows, as
    ``layout.batch`` gives them, but those of size 1 and those a shard_map being
    traced maps."""
    on = active()
    return () if on is None else split_axes(on, "batch", skip=manual_axes())


def spans_processes(processes: Int[np.ndarray, "..."], axes: tuple[int, ...]) -> bool:
    """Return whether the devices along the mesh dimensions `axes` belong to more than
    one process, given each device's process index on the mesh's grid: whether a
    reduction over them goes between hosts."""
    along = [processes.shape[a] for a in axes]
    moved = np.moveaxis(processes, axes, range(-len(axes), 0))
    groups = moved.reshape(-1, int(np.prod(along)))
    return bool((groups != groups[:, :1]).any())


def rows_span_processes() -> bool:
    """Return whether the axes that split the rows (`rows`) span more than one
    process of the run's mesh. Only then does a step's one reduction beat one a
    microbatch: between hosts the collectives go through the NICs, while over NVLink
    XLA hides a microbatch's behind the backward."""
    on = active()
    if on is None or not isinstance(on.mesh, Mesh):
        return False  # an AbstractMesh has no devices
    names = list(on.mesh.axis_names)
    processes = np.vectorize(lambda d: d.process_index, otypes=[int])(on.mesh.devices)
    return spans_processes(processes, tuple(names.index(a) for a in rows()))


def splits(params: nnx.State, over: Axes) -> nnx.State:
    """Return each parameter's spec on the axes `over`: how it splits over them."""

    def kept(spec: P) -> P:
        return partition([tuple(a for a in as_split(e) if a in over) for e in spec])

    return jax.tree.map(lambda s: kept(s.spec), shardings(params))


def gather(spec: P, x: Shaped[Array, "*s"]) -> Shaped[Array, "*w"]:
    """Gather a device's shard `x` of a parameter whole, in a shard_map manual over
    the axes of its `spec`."""
    for dim, split in enumerate(spec):
        if split is not None:
            x = jax.lax.all_gather(x, split, axis=dim, tiled=True)
    return x


def scatter(
    spec: P, over: Axes, g: Float[Array, "*w"], dtype: DTypeLike
) -> Float[Array, "*s"]:
    """Sum a device's part `g` of a gradient over the devices of `over` in `dtype`,
    keeping the shard that `spec` gives this device: a reduce-scatter over the axes
    of `spec`, and an all-reduce over the rest."""
    g = _summable(g.astype(dtype))
    for dim, along in enumerate(spec):
        if along is not None:
            g = jax.lax.psum_scatter(g, along, scatter_dimension=dim, tiled=True)
    rest = tuple(a for a in over if a not in axes_of(spec))
    return (jax.lax.psum(g, rest) if rest else g).astype(dtype)


def added(total: PyTree, part: PyTree) -> PyTree:
    """Return `total` with `part` added, leaf by leaf, in `total`'s dtypes: a
    microbatch's gradients or metrics added into the step's sums."""
    return jax.tree.map(lambda a, b: a + b.astype(a.dtype), total, part)


def _summable(g: Float[Array, "*s"]) -> Float[Array, "*s"]:
    """Return `g` as XLA sums it: on a CPU in float32, as XLA:CPU sums a bfloat16
    all-reduce, and aborts on one in a shard_map (the parallelism guide)."""
    return g.astype(jnp.float32) if jax.default_backend() == "cpu" else g


def own_streams(state: nnx.State, over: Axes) -> nnx.State:
    """Fold this device's index into every RNG key of `state`, in a shard_map manual
    over `over`: so that each device draws masks for its own rows, not the same ones
    as every other device."""
    if not over:
        return state
    index = jax.lax.axis_index(over)
    fold_in = jax.vmap(jax.random.fold_in, (0, None))  # a scanned repeat's, stacked

    def fold(v: nnx.Variable) -> nnx.Variable:
        if not isinstance(v, nnx.RngKey):
            return v
        keys = v.get_value()
        return v.replace(fold_in(keys.reshape(-1), index).reshape(keys.shape))

    return jax.tree.map(fold, state, is_leaf=_is_variable)


def keys_of(before: nnx.State, after: nnx.State) -> nnx.State:
    """Return `after` with the RNG keys of `before`: a forward moves only the counts."""

    def pick(new: nnx.Variable, old: nnx.Variable) -> nnx.Variable:
        return old if isinstance(old, nnx.RngKey) else new

    return jax.tree.map(pick, after, before, is_leaf=_is_variable)


def _is_variable(x: object) -> bool:
    return isinstance(x, nnx.Variable)


APART = contextvars.ContextVar[Axes]("bux.parallel.gather.apart", default=())


@contextlib.contextmanager
def apart(over: Axes) -> Generator[None]:
    """Say that the step being traced computes each row apart from the others over
    the axes `over`: a pipeline then runs its stages manual over them too, so that
    its ticks' gradients stay each device's own, summed once as they leave it."""
    token = APART.set(over)
    try:
        yield
    finally:
        APART.reset(token)


def rows_apart() -> Axes:
    """Return the axes `apart` put in force, but those a shard_map being traced maps."""
    manual = manual_axes()
    return tuple(a for a in APART.get() if a not in manual)
