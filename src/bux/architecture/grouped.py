"""Grouped matmuls, each group of rows by its own matrix, as an MoE's experts run:
`jax.lax.ragged_dot`, by kernels on a GPU.

The forward and the gradient of the rows run by tokamax's Mosaic kernel where its
extra is installed (Linux), its fastest on Blackwell and Hopper. The gradient of the
matrices contracts the ragged dimension, which that kernel cannot: tokamax's Triton
kernel takes it under jit, where jax-triton works on jax 0.12, and `contract`, a
Pallas kernel here, where it does not (eagerly). XLA's own, off a GPU or at shapes
they do not take, is 27 times slower on a B300 at an MoE's shapes.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jaxtyping import Array, Float, Int

try:  # tokamax's kernels, where its extra is installed (Linux)
    import tokamax
except ImportError:
    tokamax = None

ROWS = ("mosaic", "xla")  # tokamax's kernels for x @ w, fastest first
RAGGED_CONTRACTING = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)
BLOCK_ROWS = 64  # rows of a group that `contract` takes at a time
TILES = (128, 64, 32, 16)  # the tiles of dw it tries, largest first


def ragged_dot(
    x: Float[Array, "m d"], w: Float[Array, "g d f"], sizes: Int[Array, " g"]
) -> Float[Array, "m f"]:
    """Return the first sizes[0] rows of x times w[0], the next sizes[1] times w[1],
    and so on; rows past them come out 0."""
    if tokamax is None and jax.default_backend() != "gpu":
        return jax.lax.ragged_dot(x, w, sizes)
    return _kernels(*_vary_alike(x, w), sizes)


def _vary_alike(
    x: Float[Array, "m d"], w: Float[Array, "g d f"]
) -> tuple[Float[Array, "m d"], Float[Array, "g d f"]]:
    """Return x and w, each cast to vary over every mesh axis either does, in a
    shard_map that types what varies (an MoE's tokens over theirs, its experts over
    `expert`): then the VJP's gradients vary as its inputs, and autodiff sums them
    over the other axes, outside it."""
    either = _varies(x) | _varies(w)

    def cast(a: Float[Array, "*a"]) -> Float[Array, "*a"]:
        more = tuple(either - _varies(a))
        return jax.lax.pcast(a, more, to="varying") if more else a

    return cast(x), cast(w)


def _varies(a: Float[Array, "*a"]) -> frozenset[str]:
    """Return the mesh axes `a` varies over, in a shard_map that types them."""
    return frozenset(jax.typeof(a).manual_axis_type.varying)


@jax.custom_vjp
def _kernels(
    x: Float[Array, "m d"], w: Float[Array, "g d f"], sizes: Int[Array, " g"]
) -> Float[Array, "m f"]:
    return _grouped_dot(x, w, sizes)


def _grouped_dot(
    x: Float[Array, "m d"], w: Float[Array, "g d f"], sizes: Int[Array, " g"]
) -> Float[Array, "m f"]:
    """Run `ragged_dot` by tokamax's kernels, zeroing the rows past the groups, or by
    XLA."""
    if tokamax is None:
        return jax.lax.ragged_dot(x, w, sizes)
    varies = jax.typeof(x).manual_axis_type  # in a shard_map, what the output does
    out = tokamax.ragged_dot(x, w, sizes, implementation=ROWS, manual_axis_type=varies)
    return jnp.where((jnp.arange(x.shape[0]) < sizes.sum())[:, None], out, 0)


# What the backward takes of the forward: its inputs.
Saved = tuple[Float[Array, "m d"], Float[Array, "g d f"], Int[Array, " g"]]


def _forward(
    x: Float[Array, "m d"], w: Float[Array, "g d f"], sizes: Int[Array, " g"]
) -> tuple[Float[Array, "m f"], Saved]:
    return _grouped_dot(x, w, sizes), (x, w, sizes)


def _backward(
    saved: Saved, dy: Float[Array, "m f"]
) -> tuple[Float[Array, "m d"], Float[Array, "g d f"], None]:
    x, w, sizes = saved
    dx = _grouped_dot(dy, jnp.swapaxes(w, 1, 2), sizes).astype(x.dtype)
    dw = _weight_grad(x, dy, sizes).astype(w.dtype)
    return _vary_like(dx, x), _vary_like(dw, w), None


def _vary_like(
    grad: Float[Array, "*a"], like: Float[Array, "*a"]
) -> Float[Array, "*a"]:
    """Return the gradient `grad` cast to vary as `like` does: a kernel's outputs need
    not say what they vary over, which a shard_map that types it asks of a VJP's."""
    more = tuple(_varies(like) - _varies(grad))
    return jax.lax.pcast(grad, more, to="varying") if more else grad


def _weight_grad(
    x: Float[Array, "m d"], dy: Float[Array, "m f"], sizes: Int[Array, " g"]
) -> Float[Array, "g d f"]:
    """Return dW, x^T dy per group: by tokamax's Triton kernel, 1.7 times as fast as
    `contract` on a B300, where it runs."""
    if tokamax is not None and jax.default_backend() == "gpu":
        try:
            return tokamax.ragged_dot_general(
                x,
                dy,
                sizes,
                RAGGED_CONTRACTING,
                manual_axis_type=jax.typeof(x).manual_axis_type,
                implementation="triton",
            )
        except NotImplementedError, TypeError:  # TypeError: jax-triton, eagerly
            pass
    return contract(x, dy, sizes)


_kernels.defvjp(_forward, _backward)


def contract(
    x: Float[Array, "m d"], y: Float[Array, "m f"], sizes: Int[Array, " g"]
) -> Float[Array, "g d f"]:
    """Return x^T y for each group of rows, as `sizes` groups them: the gradient of
    `ragged_dot`'s matrices. By a Pallas kernel on a GPU, one program per tile of a
    group's matrix; by XLA elsewhere."""
    td, tf = _tile(x.shape[1]), _tile(y.shape[1])
    if jax.default_backend() != "gpu" or td is None or tf is None:
        return jax.lax.ragged_dot_general(x, y, sizes, RAGGED_CONTRACTING)
    return _contract(x, y, sizes, td, tf)


def _tile(n: int) -> int | None:
    """Return the largest of TILES that divides n, if any."""
    return next((t for t in TILES if n % t == 0), None)


def _contract(
    x: Float[Array, "m d"],
    y: Float[Array, "m f"],
    sizes: Int[Array, " g"],
    td: int,
    tf: int,
    interpret: bool = False,
) -> Float[Array, "g d f"]:
    """Run `contract`'s kernel: one program per (td, tf) tile of a group's matrix,
    which sums the products of the group's rows, BLOCK_ROWS at a time, in float32."""
    from jax.experimental.pallas import triton as plgpu

    (m, d), f, groups = x.shape, y.shape[1], sizes.shape[0]
    ends = jnp.cumsum(sizes).astype(jnp.int32)
    bounds = jnp.stack([ends - sizes.astype(jnp.int32), ends], -1)
    rows = BLOCK_ROWS

    # Pallas refs, which index and load as arrays do not.
    def kernel(bounds_ref: Any, x_ref: Any, y_ref: Any, out_ref: Any) -> None:
        g = pl.program_id(0)
        start, end = bounds_ref[g, 0], bounds_ref[g, 1]

        def block(
            i: Int[Array, ""], acc: Float[Array, "td tf"]
        ) -> Float[Array, "td tf"]:
            at = i * rows + jnp.arange(rows)
            keep = ((at >= start) & (at < end))[:, None]
            span = pl.ds(i * rows, rows)
            xs = plgpu.load(x_ref.at[span, :], mask=keep, other=0.0)
            ys = plgpu.load(y_ref.at[span, :], mask=keep, other=0.0)
            return acc + plgpu.dot(xs.T, ys)

        zero = jnp.zeros((td, tf), jnp.float32)
        acc = jax.lax.fori_loop(start // rows, pl.cdiv(end, rows), block, zero)
        out_ref[...] = acc.astype(out_ref.dtype)

    return pl.pallas_call(
        kernel,
        grid=(groups, d // td, f // tf),
        in_specs=[
            pl.BlockSpec((groups, 2), lambda g, i, j: (0, 0)),
            pl.BlockSpec((m, td), lambda g, i, j: (0, i)),
            pl.BlockSpec((m, tf), lambda g, i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((None, td, tf), lambda g, i, j: (g, i, j)),
        out_shape=jax.ShapeDtypeStruct((groups, d, f), x.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
        interpret=interpret,
        name=f"contract_{math.prod((td, tf))}",
    )(bounds, x, y)
