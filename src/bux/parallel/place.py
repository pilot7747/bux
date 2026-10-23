"""Where state and batches live: the sharding of every variable, a model created
already sharded, and batches split by rows and into microbatches."""

from collections.abc import Callable

import jax
import numpy as np
from flax import nnx
from jax.core import Tracer
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int, Shaped

from bux.parallel.layout import NAMES, fsdp, partition, resolve, shard
from bux.parallel.mesh import Active, active


def _spec(path: tuple, x: object, on: Active, gathered: bool) -> P:
    """Return a variable's spec: its names resolved, then FSDP for the optimizer's
    state, and for parameters unless the layout's `zero` is 1 or they are `gathered`."""
    if not isinstance(x, nnx.Variable) or not hasattr(x.get_value(), "shape"):
        return P()  # optax's masked nodes, say
    shape = tuple(x.shape)
    names = x.get_metadata().get(NAMES) or (None,) * len(shape)
    try:
        splits = resolve(names, shape, on)
    except ValueError as e:
        raise ValueError(f"{jax.tree_util.keystr(path)}: {e}") from None
    use_fsdp = isinstance(x, nnx.OptState) or (
        isinstance(x, nnx.Param) and on.layout.zero == 3 and not gathered
    )
    return partition(fsdp(splits, shape, on, names) if use_fsdp else splits)


def _variable(x: object) -> bool:
    return isinstance(x, nnx.Variable)


def shardings(state: nnx.State, gathered: bool = False) -> nnx.State:
    """Return the NamedSharding of every variable of `state` on the mesh of `use`; or
    with `gathered`, of the parameters as a step uses them: split by their names
    alone, whole over FSDP's axes."""
    on = active()
    if on is None:
        raise RuntimeError("shardings are of the mesh `bux.parallel.use` sets")
    return jax.tree_util.tree_map_with_path(
        lambda p, x: NamedSharding(on.mesh, _spec(p, x, on, gathered)),
        state,
        is_leaf=_variable,
    )


def build[M: nnx.Module](make: Callable[[], M]) -> M:
    """Return `make()`'s module with every parameter created on its shards, so that
    none is ever whole on one device."""
    if active() is None:
        return make()
    graphdef, abstract = nnx.split(nnx.eval_shape(make))
    init = jax.jit(lambda: nnx.split(make())[1], out_shardings=shardings(abstract))
    return nnx.merge(graphdef, init())


def shard_state(*nodes: nnx.Pytree) -> None:
    """Shard the state of `nodes` by its `shardings`: move it there eagerly, and keep
    it there inside a step.

    Eagerly, what is there already stays in its buffers: a step that donates them
    then finds the rest of the memory in one piece, not on either side of a copy."""
    if active() is None:
        return
    state = nnx.state(nodes)
    traced = any(isinstance(x, Tracer) for x in jax.tree.leaves(state))
    put = jax.lax.with_sharding_constraint if traced else jax.device_put
    nnx.update(nodes, put(state, shardings(state)))


def shard_batch[T](batch: T) -> T:
    """Put every (b, t) host array of `batch` on the devices: rows over `batch`,
    positions over `seq`.

    In a job of many processes, each puts the parts its own devices hold, of a batch
    whose other rows it need not have (``local_rows``).
    """
    on = active()
    if on is None:
        return jax.device_put(batch)

    def put(x: Shaped[np.ndarray, "b ..."]) -> Shaped[Array, "b ..."]:
        sharding = _batch_sharding(x.shape, on)
        return jax.make_array_from_callback(x.shape, sharding, lambda part: x[part])

    return jax.tree.map(put, batch)


def _batch_sharding(shape: tuple[int, ...], on: Active) -> NamedSharding:
    names = ("batch", "seq", *(None,) * len(shape))[: len(shape)]
    return NamedSharding(on.mesh, partition(resolve(names, shape, on)))


def local_rows(rows: int, seq_len: int) -> Int[np.ndarray, " n"]:
    """Return the rows of a batch of `rows` that this process's devices hold, as
    ``shard_batch`` splits it, in order: every row on one process or outside ``use``."""
    on = active()
    if on is None or jax.process_count() == 1 or not isinstance(on.mesh, Mesh):
        return np.arange(rows)
    shape = (rows, seq_len)
    held = _batch_sharding(shape, on).addressable_devices_indices_map(shape).values()
    mine = {r for part in held if part is not None for r in range(rows)[part[0]]}
    return np.array(sorted(mine), int)


def microbatches[T](batch: T, n: int) -> T:
    """Split every (b, ...) array of `batch` into `n` microbatches, (n, b / n, ...).

    Microbatch i takes rows i, i + n, i + 2n and so on, so that every device's rows
    split evenly among them and none moves. Each shards as ``shard_batch`` does.
    """

    def split(x: Shaped[Array, "b ..."]) -> Shaped[Array, "n m ..."]:
        b = x.shape[0]
        if b % n:
            raise ValueError(f"{b} rows do not split into {n} microbatches")
        y = x.reshape(b // n, n, *x.shape[1:]).swapaxes(0, 1)
        return shard(y, *(None, "batch", "seq", *(None,) * x.ndim)[: y.ndim])

    return jax.tree.map(split, batch)
