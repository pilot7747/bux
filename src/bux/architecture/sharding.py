"""The sharding constraints and the dense layer that several layers share.

Each constraint only says how an activation splits over the mesh (``bux.parallel``);
outside ``parallel.use`` it returns its input unchanged.
"""

from flax import nnx
from jaxtyping import Array, Float, Shaped

from bux import parallel


def linear(
    n_in: int,
    n_out: int,
    names: tuple[parallel.Name, parallel.Name],
    bias: bool,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    """Make an `nnx.Linear` whose kernel's dimensions are `names`, its bias's the last."""
    return nnx.Linear(
        n_in,
        n_out,
        use_bias=bias,
        kernel_init=parallel.named(*names),
        bias_init=parallel.named(names[1], init=nnx.initializers.zeros),
        rngs=rngs,
    )


def shard_layer_input(x: Float[Array, "b t d"]) -> Float[Array, "b t d"]:
    """Split a layer's input over `seq`, as inside a layer, rather than as the stream:
    under sequence parallelism, this gathers it before the projections."""
    return parallel.shard(x, "batch", "seq", None)


def shard_heads(
    x: Shaped[Array, "b t h *e"], name: parallel.Name = "heads"
) -> Shaped[Array, "b t h *e"]:
    """Split something of every head at every position by head, as `name` says."""
    return parallel.shard(x, "batch", "seq", name, *(None,) * (x.ndim - 3))
