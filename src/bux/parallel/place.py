"""Where state and batches live: the sharding of every variable, a model born
sharded, and batches split by rows."""

from collections.abc import Callable

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped

from bux.parallel.layout import NAMES, fsdp, partition, resolve
from bux.parallel.mesh import Active, active


def _spec(path: tuple, x: object, on: Active) -> P:
    """A variable's spec: its names resolved, then FSDP, for parameters unless the
    layout's `zero` is 1, and for the optimizer's state. Anything else is whole."""
    if not isinstance(x, nnx.Variable) or not hasattr(x.get_value(), "shape"):
        return P()  # optax's masked nodes, say
    shape = tuple(x.shape)
    names = x.get_metadata().get(NAMES) or (None,) * len(shape)
    try:
        splits = resolve(names, shape, on)
    except ValueError as e:
        raise ValueError(f"{jax.tree_util.keystr(path)}: {e}") from None
    sharded = isinstance(x, nnx.OptState) or (
        isinstance(x, nnx.Param) and on.layout.zero == 3
    )
    return partition(fsdp(splits, shape, on, names) if sharded else splits)


def _variable(x: object) -> bool:
    return isinstance(x, nnx.Variable)


def shardings(state: nnx.State) -> nnx.State:
    """A NamedSharding for every variable of `state`, as `use` puts them."""
    on = active()
    if on is None:
        raise RuntimeError("shardings are of the mesh `bux.parallel.use` sets")
    return jax.tree_util.tree_map_with_path(
        lambda p, x: NamedSharding(on.mesh, _spec(p, x, on)), state, is_leaf=_variable
    )


def build[M: nnx.Module](make: Callable[[], M]) -> M:
    """The module `make` makes, every parameter born on its shards, so that none is
    ever whole on one device; with the values `make()` gives."""
    if active() is None:
        return make()
    graphdef, abstract = nnx.split(nnx.eval_shape(make))
    born = jax.jit(lambda: nnx.split(make())[1], out_shardings=shardings(abstract))
    return nnx.merge(graphdef, born())


def place(*nodes: nnx.Pytree) -> None:
    """Move the state of `nodes` to its `shardings`; in a step, keep it there."""
    if active() is None:
        return
    state = nnx.state(nodes)
    nnx.update(nodes, jax.lax.with_sharding_constraint(state, shardings(state)))


def rows[T](batch: T) -> T:
    """Every array of `batch`, (b, t) and on the host, on the devices: rows over
    `batch` and positions over `seq`. Of a job of many processes, each has the whole
    batch, and puts the parts its own devices hold."""
    on = active()
    if on is None:
        return jax.device_put(batch)

    def put(x: Shaped[np.ndarray, "b ..."]) -> Shaped[Array, "b ..."]:
        names = ("batch", "seq", *(None,) * x.ndim)[: x.ndim]
        split = NamedSharding(on.mesh, partition(resolve(names, x.shape, on)))
        return jax.make_array_from_callback(x.shape, split, lambda part: x[part])

    return jax.tree.map(put, batch)
