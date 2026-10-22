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
    x: Float[Array, "m d"], w: Float[Array, "e d f"], sizes: Int[Array, " e"]
) -> Float[Array, "m f"]:
    """The first sizes[0] rows of x by w[0], the next sizes[1] by w[1], and so on;
    rows past them come out 0."""
    if tokamax is None and jax.default_backend() != "gpu":
        return jax.lax.ragged_dot(x, w, sizes)
    return _kernels(*_alike(x, w), sizes)


def _alike(
    x: Float[Array, "m d"], w: Float[Array, "e d f"]
) -> tuple[Float[Array, "m d"], Float[Array, "e d f"]]:
    """x and w, each made to vary over the mesh axes either does, in a shard_map that
    types what varies (an MoE's tokens over theirs, its experts over `expert`): so
    the VJP's gradients vary as its inputs, and autodiff sums them over the axes the
    inputs did not vary over, outside it."""
    either = _varies(x) | _varies(w)

    def cast(a: Float[Array, "*a"]) -> Float[Array, "*a"]:
        more = tuple(either - _varies(a))
        return jax.lax.pcast(a, more, to="varying") if more else a

    return cast(x), cast(w)


def _varies(a: Float[Array, "*a"]) -> frozenset[str]:
    """The mesh axes `a` varies over, in a shard_map that types them."""
    return frozenset(jax.typeof(a).manual_axis_type.varying)


@jax.custom_vjp
def _kernels(
    x: Float[Array, "m d"], w: Float[Array, "e d f"], sizes: Int[Array, " e"]
) -> Float[Array, "m f"]:
    return _rows(x, w, sizes)


def _rows(
    x: Float[Array, "m d"], w: Float[Array, "e d f"], sizes: Int[Array, " e"]
) -> Float[Array, "m f"]:
    """`ragged_dot` by tokamax's kernels, which leave the rows past the groups as
    they find them, or by XLA."""
    if tokamax is None:
        return jax.lax.ragged_dot(x, w, sizes)
    varies = jax.typeof(x).manual_axis_type  # in a shard_map, what the output does
    out = tokamax.ragged_dot(x, w, sizes, implementation=ROWS, manual_axis_type=varies)
    return jnp.where((jnp.arange(x.shape[0]) < sizes.sum())[:, None], out, 0)


# What the backward takes of the forward: its inputs.
Saved = tuple[Float[Array, "m d"], Float[Array, "e d f"], Int[Array, " e"]]


def _forward(
    x: Float[Array, "m d"], w: Float[Array, "e d f"], sizes: Int[Array, " e"]
) -> tuple[Float[Array, "m f"], Saved]:
    return _rows(x, w, sizes), (x, w, sizes)


def _backward(
    saved: Saved, dy: Float[Array, "m f"]
) -> tuple[Float[Array, "m d"], Float[Array, "e d f"], None]:
    x, w, sizes = saved
    dx = _rows(dy, jnp.swapaxes(w, 1, 2), sizes).astype(x.dtype)
    dw = _matrices(x, dy, sizes).astype(w.dtype)
    return _typed(dx, x), _typed(dw, w), None


def _typed(g: Float[Array, "*a"], like: Float[Array, "*a"]) -> Float[Array, "*a"]:
    """The gradient `g`, typed to vary as `like` does: a kernel's outputs need not say
    what they vary over, which a shard_map that types it asks of a VJP's."""
    more = tuple(_varies(like) - _varies(g))
    return jax.lax.pcast(g, more, to="varying") if more else g


def _matrices(
    x: Float[Array, "m d"], dy: Float[Array, "m f"], sizes: Int[Array, " e"]
) -> Float[Array, "e d f"]:
    """The gradient of the matrices: by tokamax's Triton kernel, 1.7 times as fast
    as `contract` on a B300, where it runs."""
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
    x: Float[Array, "m d"], y: Float[Array, "m f"], sizes: Int[Array, " e"]
) -> Float[Array, "e d f"]:
    """x^T y a group of rows at a time, as `sizes` groups them: the gradient of
    `ragged_dot`'s matrices. By a Pallas kernel on a GPU, a program a tile of a
    group's matrix; by XLA elsewhere."""
    td, tf = _tile(x.shape[1]), _tile(y.shape[1])
    if jax.default_backend() != "gpu" or td is None or tf is None:
        return jax.lax.ragged_dot_general(x, y, sizes, RAGGED_CONTRACTING)
    return _contract(x, y, sizes, td, tf)


def _tile(n: int) -> int | None:
    """The largest of TILES that divides n, if any."""
    return next((t for t in TILES if n % t == 0), None)


def _contract(
    x: Float[Array, "m d"],
    y: Float[Array, "m f"],
    sizes: Int[Array, " e"],
    td: int,
    tf: int,
    interpret: bool = False,
) -> Float[Array, "e d f"]:
    """`contract`'s kernel: a program a (td, tf) tile of a group's matrix, which
    sums the products of the group's rows, BLOCK_ROWS at a time, in float32."""
    from jax.experimental.pallas import triton as plgpu

    (m, d), f, e = x.shape, y.shape[1], sizes.shape[0]
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
        grid=(e, d // td, f // tf),
        in_specs=[
            pl.BlockSpec((e, 2), lambda g, i, j: (0, 0)),
            pl.BlockSpec((m, td), lambda g, i, j: (0, i)),
            pl.BlockSpec((m, tf), lambda g, i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((None, td, tf), lambda g, i, j: (g, i, j)),
        out_shape=jax.ShapeDtypeStruct((e, d, f), x.dtype),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
        interpret=interpret,
        name=f"contract_{math.prod((td, tf))}",
    )(bounds, x, y)
