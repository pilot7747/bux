"""The sharding constraints and the dense layers that several layers share.

Each constraint only says how an activation splits over the mesh (``bux.parallel``);
outside ``parallel.use`` it returns its input unchanged.
"""

from typing import Any

import jax
from flax import nnx
from jaxtyping import Array, Float, Shaped

from bux import parallel
from bux.architecture import matmuls


def projection(
    n_in: int, n_out: int, role: str | None = None, **kwargs: Any
) -> nnx.Linear:
    """Make an `nnx.Linear` of a block, which multiplies in the format the run's
    precision gives its `role` (``matmuls``)."""
    return nnx.Linear(n_in, n_out, dot_general=matmuls.role(role), **kwargs)


def linear(
    n_in: int,
    n_out: int,
    names: tuple[parallel.Name, parallel.Name],
    bias: bool,
    rngs: nnx.Rngs,
    in_compute: bool = False,
    role: str | None = None,
) -> nnx.Linear:
    """Make a `projection` of `role` whose kernel's dimensions are `names`, its bias's
    the last; or `in_compute`, an `nnx.Linear` that multiplies in the dtype the run
    computes in."""
    return nnx.Linear(
        n_in,
        n_out,
        use_bias=bias,
        kernel_init=parallel.named(*names),
        bias_init=parallel.named(names[1], init=nnx.initializers.zeros),
        dot_general=jax.lax.dot_general if in_compute else matmuls.role(role),
        rngs=rngs,
    )


def shard_layer_input(x: Float[Array, "b t d"]) -> Float[Array, "b t d"]:
    """Split a layer's input over `seq`, as inside a layer, rather than as the stream:
    under sequence parallelism, this gathers it before the projections."""
    return parallel.shard(x, "batch", "seq", None)


def project_input(
    x: Float[Array, "b t d"], *linears: nnx.Linear
) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
    """Return a layer's input as `shard_layer_input` splits it, and its projection by
    each of `linears`, whose kernels split by columns: by the run's `methods.tensor`,
    which may gather the input a chunk at a time beside the matmuls."""
    return parallel.tensor.project_in(x, linears)


def project_output(
    h: Float[Array, "b t f"], linear: nnx.Linear
) -> Float[Array, "b t d"]:
    """Return a layer's output projection, whose kernel splits by rows, by the run's
    `methods.tensor`, which may reduce its partial sums a chunk at a time."""
    return parallel.tensor.project_out(h, linear)


def shard_heads(
    x: Shaped[Array, "b t h *e"], name: parallel.Name = "heads"
) -> Shaped[Array, "b t h *e"]:
    """Split something of every head at every position by head, as `name` says."""
    return parallel.shard(x, "batch", "seq", name, *(None,) * (x.ndim - 3))
