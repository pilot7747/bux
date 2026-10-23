"""A training step's parameters gathered once, and its gradients reduced once
(bux.train.trainer).

Every axis is Auto, and XLA cannot move a collective out of a loop: a step of n
microbatches would gather each layer's parameters as it runs, n times forward and n
back, and reduce the gradients n times. ``whole`` gathers them once, before the
loop. Gradients reduced once need an array that each device holds a part of the sum
of, which JAX types only on Explicit or Manual axes (``P(unreduced=...)``). So a step
whose rows are independent runs in a shard_map manual over the axes that split them
(``rows``): there each device gathers its shard of every parameter (``gather``), sums
its own rows' gradients over the microbatches, and sums the sums over the devices
once (``scatter``), each device keeping its shard.

    whole     the parameters as a step uses them, whole over FSDP's axes
    rows      the axes that split the batch's rows, which the shard_map is manual over
    splits    each parameter's split over them
    gather    a device's shard of a parameter, gathered whole
    scatter   a device's part of a gradient, summed over the devices, as a shard
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Shaped

from bux.parallel.layout import axes
from bux.parallel.mesh import active
from bux.parallel.place import shardings

Axes = tuple[str, ...]


def whole(params: nnx.State) -> nnx.State:
    """Constrain `params` to how a step uses them: split by their names alone, whole
    over FSDP's axes. Each device then holds every parameter it computes with."""
    if active() is None:
        return params
    return jax.lax.with_sharding_constraint(params, shardings(params, gathered=True))


def rows() -> Axes:
    """Return the axes of the run's mesh that split a batch's rows, as `layout.batch`
    gives them, but those of size 1 and those a shard_map being traced maps."""
    on = active()
    if on is None:
        return ()
    manual = set(jax.sharding.get_abstract_mesh().manual_axes)
    return tuple(
        a for a in axes(on.layout, "batch") if on.size(a) > 1 and a not in manual
    )


def splits(params: nnx.State, over: Axes) -> nnx.State:
    """Return each parameter's spec on the axes `over`: how it splits over them."""

    def kept(s: str | tuple[str, ...] | None) -> str | tuple[str, ...] | None:
        names = () if s is None else (s,) if isinstance(s, str) else s
        names = tuple(a for a in names if a in over)
        return None if not names else names[0] if len(names) == 1 else names

    return jax.tree.map(lambda s: P(*map(kept, s.spec)), shardings(params))


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
    split = set()
    for dim, along in enumerate(spec):
        if along is not None:
            g = jax.lax.psum_scatter(g, along, scatter_dimension=dim, tiled=True)
            split |= {along} if isinstance(along, str) else set(along)
    rest = tuple(a for a in over if a not in split)
    return (jax.lax.psum(g, rest) if rest else g).astype(dtype)


def _summable(g: Float[Array, "*s"]) -> Float[Array, "*s"]:
    """Return `g` as XLA sums it: on a CPU in float32, as XLA:CPU sums a bfloat16
    all-reduce, and aborts on one in a shard_map (docs/parallelism.md)."""
    return g.astype(jnp.float32) if jax.default_backend() == "cpu" else g
