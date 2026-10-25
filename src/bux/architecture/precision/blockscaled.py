"""Block-scaled GEMMs: a projection ``y = x W`` and its gradients, each GEMM
multiplying operands quantized in blocks along what it contracts, a scale a block.
cuDNN's block-scaled GEMM multiplies them on a Blackwell GPU; elsewhere XLA dequantizes
and multiplies, which computes the same values.

With x (m, k), W (k, n) and dy (m, n), each GEMM is ``a b^T``, a and b both in blocks
along their rows, which it contracts. A name says along what: ``x_k`` is x in blocks
along k, ``xt_m`` is x^T in blocks along m::

    fprop   y  = x W      x_k  wt_k^T
    dgrad   dx = dy W^T   dy_n w_n^T
    wgrad   dW = x^T dy   xt_m dyt_m^T

So a `Recipe` quantizes each of x, W and dy twice: along its rows, and along its
columns as its transpose's rows. The forward keeps ``xt_m`` and ``w_n`` for the
backward. Only the GEMMs in ``passes`` multiply in the format, the others their
operands as they are (`project`). MXFP8's recipes are here (`MXFP8`, `MXFP8_2D`),
NVFP4's in `~bux.architecture.precision.nvfp4`;
`~bux.architecture.precision.matmuls` registers a format of each.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Shaped, UInt8, UInt32

from bux import jax_internals, kernels
from bux.parallel.mesh import manual_axes
from bux.train.config import PASSES, Pass

E4M3_MAX = 448.0  # float8 E4M3's largest value


class Blocks(NamedTuple):
    """An (r, c) array in a block-scaled format, in blocks along its rows: its values
    times their block's scale, times the tensor's."""

    #: The values, in the format's dtype (E4M3, E2M1).
    values: Shaped[Array, "r c"]
    #: The scales, one a block (E8M0, E4M3).
    scales: Shaped[Array, "r n"]
    #: The tensor's scale; None for 1.
    tensor_scale: Float[Array, ""] | None


# How a recipe quantizes an operand a, given a seed for what rounds stochastically: a
# and a^T, each in blocks along its rows.
Twice = Callable[[Float[Array, "r c"], UInt32[Array, ""]], tuple[Blocks, Blocks]]


class Recipe(NamedTuple):
    """How a block-scaled format quantizes x, W and dy, each twice: itself and its
    transpose, each in blocks along its rows.

    Example:
        MXFP8 for x and dy, and W's scales per 32 x 32 tile, as `MXFP8_2D` is::

            Recipe(MX_BLOCK, x=mx_twice, w=mx_tiles_twice, dy=mx_twice)
    """

    #: The values a block holds, which each of a GEMM's dimensions must be whole
    #: blocks of.
    block: int
    #: How it quantizes x, the input.
    x: Twice
    #: How it quantizes W, the weight.
    w: Twice
    #: How it quantizes dy, the output's gradient.
    dy: Twice


def project(
    x: Float[Array, "*lead k"],
    w: Float[Array, "k n"],
    recipe: Recipe,
    out: DTypeLike,
    seed: UInt32[Array, ""],
    passes: tuple[Pass, ...] = PASSES,
) -> Float[Array, "*lead n"]:
    """Return ``x W`` in `out`, its GEMMs named in `passes` as `recipe` quantizes
    them, the others multiplying their operands as they are.

    Args:
        x: the input, (..., k): its leading dimensions flatten into m rows.
        w: the weight, (k, n).
        recipe: how the format quantizes x, W and dy.
        out: the dtype of y.
        seed: what a recipe that rounds stochastically draws its noise from.
        passes: the GEMMs that multiply in the format: of ``fprop``, ``dgrad`` and
            ``wgrad``.

    Returns:
        y, (..., n). Its gradients reach x in its dtype and W in its own.
    """
    dtypes = Dtypes(x.dtype, w.dtype, jnp.dtype(out))
    y = _project(x.reshape(-1, x.shape[-1]), w, seed, recipe, dtypes, passes)
    return y.reshape(*x.shape[:-1], w.shape[1])


class Dtypes(NamedTuple):
    """The dtypes of x, W and y, which dx, dW and y take."""

    x: np.dtype
    w: np.dtype
    y: np.dtype


# An operand of a GEMM: in the format, or as it is where its GEMM is not in `passes`.
Operand = Blocks | Float[Array, "_ _"]
Saved = tuple[Operand, Operand, UInt32[Array, ""]]  # xt_m, w_n and the seed


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _project(
    x: Float[Array, "m k"],
    w: Float[Array, "k n"],
    seed: UInt32[Array, ""],
    recipe: Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
) -> Float[Array, "m n"]:
    """Return `project` of a matrix x, with its own gradients."""
    return _forward(x, w, seed, recipe, dtypes, passes)[0]


def _forward(
    x: Float[Array, "m k"],
    w: Float[Array, "k n"],
    seed: UInt32[Array, ""],
    recipe: Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
) -> tuple[Float[Array, "m n"], Saved]:
    """Return y = x W, by fprop, and what the backward's GEMMs take of x and W."""
    fprop, dgrad, wgrad = (p in passes for p in PASSES)
    x_k, xt_m = _quantized(recipe.x, x, seed, rows=fprop, columns=wgrad)
    w_n, wt_k = _quantized(recipe.w, w, seed, rows=dgrad, columns=fprop)
    y = _times(x_k, wt_k, dtypes.y)
    return y, (xt_m, w_n, seed)


def _backward(
    recipe: Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
    saved: Saved,
    dy: Float[Array, "m n"],
) -> tuple[Float[Array, "m k"], Float[Array, "k n"], None]:
    """Return dx = dy W^T, by dgrad, and dW = x^T dy, by wgrad; the seed has none."""
    xt_m, w_n, seed = saved
    _, dgrad, wgrad = (p in passes for p in PASSES)
    dy_n, dyt_m = _quantized(recipe.dy, dy, seed, rows=dgrad, columns=wgrad)
    dx = _times(dy_n, w_n, dtypes.x)
    dw = _times(xt_m, dyt_m, dtypes.w)
    return dx, dw, None


_project.defvjp(_forward, _backward)


def _quantized(
    quantize: Twice,
    a: Float[Array, "r c"],
    seed: UInt32[Array, ""],
    rows: bool,
    columns: bool,
) -> tuple[Operand, Operand]:
    """Return a and a^T, each in the format where its GEMM multiplies in it (`rows`
    for a, `columns` for a^T), else as it is."""
    if not rows and not columns:
        return a, a.T
    along_rows, along_columns = quantize(a, seed)
    return along_rows if rows else a, along_columns if columns else a.T


def _times(a: Operand, b: Operand, out: DTypeLike) -> Float[Array, "m n"]:
    """Return a b^T in `out`: by the block-scaled GEMM where both are in the format,
    else as they are."""
    if isinstance(a, Blocks) and isinstance(b, Blocks):
        return gemm(a, b, out)
    assert not isinstance(a, Blocks) and not isinstance(b, Blocks)
    return jnp.matmul(a, b.T, preferred_element_type=out)


def gemm(a: Blocks, b: Blocks, out: DTypeLike) -> Float[Array, "m n"]:
    """Return ``a b^T`` in `out`: the GEMM of their values by their block scales, then
    times their tensor scales. That rescales the GEMM's output in `out`, where XLA
    fuses it into what reads it, rather than in a float32 copy."""
    y = scaled_matmul(a, b, out)
    if a.tensor_scale is None or b.tensor_scale is None:
        return y
    return (y.astype(jnp.float32) * (a.tensor_scale * b.tensor_scale)).astype(out)


def scaled_matmul(a: Blocks, b: Blocks, out: DTypeLike) -> Float[Array, "m n"]:
    """Return ``a b^T`` of their values times their block scales, in `out`: by
    `jax.nn.scaled_matmul`, cuDNN's block-scaled GEMM on a Blackwell GPU, which
    elsewhere dequantizes and multiplies. In a shard_map, by its primitive, which
    jax's custom partitioning cannot place there."""
    args = (a.values[None], b.values[None], a.scales[None], b.scales[None])
    if manual_axes():  # in a shard_map
        return jax_internals.scaled_matmul_unpartitioned(*args, out)[0]
    return jax.nn.scaled_matmul(*args, preferred_element_type=out)[0]


def kept_quantized(blocks: Blocks) -> Blocks:
    """Return `blocks` as they are, behind an optimization barrier: without a native
    GEMM, XLA would fold the casts to the format and back away, and multiply the
    values unquantized."""
    values, scales = jax.lax.optimization_barrier((blocks.values, blocks.scales))
    return Blocks(values, scales, blocks.tensor_scale)


def dequantized(blocks: Blocks) -> Float[Array, "r c"]:
    """Return the values `blocks` hold, in float32: each times its block's scale and
    the tensor's."""
    block = blocks.values.shape[-1] // blocks.scales.shape[-1]
    scales = jnp.repeat(blocks.scales.astype(jnp.float32), block, axis=-1)
    values = blocks.values.astype(jnp.float32) * scales
    return values if blocks.tensor_scale is None else values * blocks.tensor_scale


# --- Quantizing kernels: what MXFP8's and NVFP4's share -------------------------------
# A Pallas kernel runs a program per tile of x, which writes its part of x's copy and of
# x^T's at once. Custom partitioning runs it on each device's part of x.

Shardings = tuple[NamedSharding, ...]


def largest_tile(n: int, tiles: tuple[int, ...]) -> int:
    """Return the largest of `tiles` that divides n."""
    return next(t for t in tiles if n % t == 0)


def as_bytes(x: Shaped[Array, "*s"]) -> UInt8[Array, "*s"]:
    """Return the bits of a format of a byte, E4M3 values or E4M3 and E8M0 scales, as
    a kernel stores them."""
    return jax.lax.bitcast_convert_type(x, jnp.uint8)


def partitioned_unless_manual(
    kernel: Callable[..., Any], partitioned: Callable[..., Any]
) -> Callable[..., Any]:
    """Return `partitioned`, which runs `kernel` on each device's part of what it
    takes; inside a shard_map, where each device holds its part already and custom
    partitioning cannot go, `kernel` itself."""
    return kernel if manual_axes() else partitioned


def split_of(spec: P) -> tuple[Any, Any]:
    """Return what an (r, c) array's rows and columns split over: a mesh axis,
    several, or None."""
    rows, columns = (*spec, None, None)[:2]
    return rows, columns


def copies_shardings(mesh: Mesh, spec: P) -> Shardings:
    """Return how a kernel's copies split, of x split by `spec`: x's values and scales
    as x does, and x^T's transposed."""
    r, c = split_of(spec)
    rows, cols = NamedSharding(mesh, P(r, c)), NamedSharding(mesh, P(c, r))
    return rows, rows, cols, cols


# --- MXFP8 ----------------------------------------------------------------------------
# OCP's MX format: E4M3 values in blocks of 32, a power-of-two scale (E8M0) a block.

MX_BLOCK = 32


def mx_twice(
    x: Float[Array, "r c"], seed: UInt32[Array, ""] | None = None
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in MXFP8, each in blocks of 32 along its rows: by one pass of
    a kernel over x on a GPU, and by XLA elsewhere. MXFP8 rounds to nearest, and needs
    no seed. `MXFP8`'s quantizer of each operand (a `Twice`)."""
    if not kernels.gpu():
        return mx_quantize(x), mx_quantize(x.T)
    return mx_kernel_twice(x)


def mx_tiles_twice(
    w: Float[Array, "r c"], seed: UInt32[Array, ""] | None = None
) -> tuple[Blocks, Blocks]:
    """Return W and W^T in MXFP8 with a scale per 32 x 32 tile, so that y = x W and
    dx = dy W^T multiply the same values (TorchTitan's weights). By XLA: a weight is
    small next to the activations."""
    values, exponents = _in_e4m3(w.astype(jnp.float32), "tiles")
    rows = jnp.repeat(exponents, MX_BLOCK, axis=0)
    columns = jnp.repeat(exponents.T, MX_BLOCK, axis=0)
    return _mx_blocks(values, rows), _mx_blocks(values.T, columns)


#: Transformer Engine's MXFP8, E4M3 for all three GEMMs, gradients too.
MXFP8 = Recipe(MX_BLOCK, x=mx_twice, w=mx_twice, dy=mx_twice)
#: MXFP8 with W's scales per 32 x 32 tile, the same for y and dx (TorchTitan's).
MXFP8_2D = Recipe(MX_BLOCK, x=mx_twice, w=mx_tiles_twice, dy=mx_twice)


def mx_quantize(x: Float[Array, "r c"]) -> Blocks:
    """Return x in MXFP8, in blocks of 32 along its rows, by XLA."""
    return _mx_blocks(*_in_e4m3(x.astype(jnp.float32), "rows"))


def _mx_blocks(values: Shaped[Array, "r c"], exponents: UInt8[Array, "r n"]) -> Blocks:
    """Return E4M3 values and their scales' biased exponents as the GEMM takes them."""
    scales = jax.lax.bitcast_convert_type(exponents, jnp.float8_e8m0fnu)
    return kept_quantized(Blocks(values, scales, None))


def _in_e4m3(
    x: Float[Array, "r c"], blocks: Literal["rows", "columns", "tiles"]
) -> tuple[Shaped[Array, "r c"], UInt8[Array, "p q"]]:
    """Return x's E4M3 values, and the biased exponents of their scales (E8M0's
    bits), in blocks of 32 along its `rows`, down its `columns` (x^T's, untransposed)
    or in 32 x 32 `tiles`. A block's scale is the least power of two that takes its
    largest magnitude to at most 448: found in float32, and applied in x's dtype, in
    which scaling by a power of two is as exact as in float32."""
    (r, c), b = x.shape, MX_BLOCK
    shape, axes = {
        "rows": ((r, c // b, b), 2),
        "columns": ((r // b, b, c), 1),
        "tiles": ((r // b, b, c // b, b), (1, 3)),
    }[blocks]
    blocked = x.reshape(shape)
    largest = jnp.abs(blocked).max(axes).astype(jnp.float32)
    exponent = _ceil_log2(largest / E4M3_MAX)
    down = _power_of_two(-exponent).astype(x.dtype)
    scaled = blocked * jnp.expand_dims(down, axes)
    values = jnp.clip(scaled, -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)
    return values.reshape(r, c), (exponent + 127).astype(jnp.uint8)


def _ceil_log2(x: Float[Array, "*s"]) -> Int[Array, "*s"]:
    """Return the least e with 2^e >= x, for x >= 0, between -126 and 126: of x's
    float32 bits, exactly."""
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    exponent = (bits >> 23).astype(jnp.int32) + ((bits & 0x7FFFFF) != 0)
    return jnp.clip(exponent - 127, -126, 126)


def _power_of_two(e: Int[Array, "*s"]) -> Float[Array, "*s"]:
    """Return 2^e, exactly, for integers e between -126 and 127."""
    return jax.lax.bitcast_convert_type((e + 127).astype(jnp.uint32) << 23, jnp.float32)


# MXFP8 as bytes: what an exchange sends a row, or a Triton kernel takes.


def mx_from_bytes(
    values: UInt8[Array, "r c"], exponents: UInt8[Array, "r n"]
) -> Blocks:
    """Return MXFP8 blocks of their bytes: E4M3 values, and their scales' biased
    exponents (E8M0's bits)."""
    as_e4m3 = jax.lax.bitcast_convert_type(values, jnp.float8_e4m3fn)
    as_e8m0 = jax.lax.bitcast_convert_type(exponents, jnp.float8_e8m0fnu)
    return Blocks(as_e4m3, as_e8m0, None)


def mx_packed(blocks: Blocks) -> UInt8[Array, "r e"]:
    """Return each row of MXFP8 `blocks` as one row of bytes, c + c / 32 of them: its
    E4M3 values, then its scales."""
    return jnp.concatenate([as_bytes(blocks.values), as_bytes(blocks.scales)], axis=-1)


def mx_unpacked(packed: UInt8[Array, "r e"], c: int) -> Blocks:
    """Return the MXFP8 blocks of rows of `c` values that `mx_packed` made."""
    return mx_from_bytes(packed[:, :c], packed[:, c:])


# The kernel's tiles, of rows and of columns, largest first, and its warps: 64 x 128
# on 2 is the fastest on a B300, 4.3-4.7 TB/s at Qwen3-30B-A3B's projections, 1.3x
# the same tiles on 4 warps scaling in float32.
ROWS, COLUMNS = (64, 32), (128, 64, 32)
WARPS = 2
# x's E4M3 values and exponents, then x^T's.
MxArrays = tuple[
    Shaped[Array, "r c"], UInt8[Array, "r n"], UInt8[Array, "c r"], UInt8[Array, "c q"]
]


def mx_kernel_twice(x: Float[Array, "r c"]) -> tuple[Blocks, Blocks]:
    """Return x and x^T in MXFP8 by one pass of the kernel, each device on its part
    of x; off a GPU, the kernel interpreted."""
    quantize = partitioned_unless_manual(_mx_kernel, _mx_partitioned)
    rows, row_exponents, cols, col_exponents = quantize(x)
    cols = jax.lax.bitcast_convert_type(cols, jnp.float8_e4m3fn)
    return _mx_blocks(rows, row_exponents), _mx_blocks(cols, col_exponents)


def _mx_kernel(x: Float[Array, "r c"]) -> MxArrays:
    """Quantize x and x^T, a program per tile of x."""
    from jax.experimental.pallas import triton as plgpu

    (r, c), b = x.shape, MX_BLOCK
    tr, tc = largest_tile(r, ROWS), largest_tile(c, COLUMNS)
    shapes = [((r, c), jnp.float8_e4m3fn), ((r, c // b), jnp.uint8)]
    shapes += [((c, r), jnp.uint8), ((c, r // b), jnp.uint8)]
    return pl.pallas_call(
        _mx_quantize_tile,
        grid=(r // tr, c // tc),
        in_specs=[pl.BlockSpec((tr, tc), lambda i, j: (i, j))],
        out_specs=[
            pl.BlockSpec((tr, tc), lambda i, j: (i, j)),
            pl.BlockSpec((tr, tc // b), lambda i, j: (i, j)),
            pl.BlockSpec((tc, tr), lambda i, j: (j, i)),
            pl.BlockSpec((tc, tr // b), lambda i, j: (j, i)),
        ],
        out_shape=[jax.ShapeDtypeStruct(*s) for s in shapes],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1),
        interpret=kernels.interpreted(),
        name=f"mx_twice_{tr}x{tc}",
    )(x)


# Pallas refs, which index and load as arrays do not.
def _mx_quantize_tile(
    x_ref: Any,
    rows_ref: Any,
    row_exponents_ref: Any,
    cols_ref: Any,
    col_exponents_ref: Any,
) -> None:
    """Quantize a tile of x into its part of x's copy and of x^T's, in x's dtype."""
    tile = x_ref[...]
    rows_ref[...], row_exponents_ref[...] = _in_e4m3(tile, "rows")
    values, exponents = _in_e4m3(tile, "columns")
    # x^T's values transpose as bytes, half of x's bfloat16 tile.
    cols_ref[...] = as_bytes(values).T
    col_exponents_ref[...] = exponents.T


@custom_partitioning
def _mx_partitioned(x: Float[Array, "r c"]) -> MxArrays:
    """Return `_mx_kernel`'s copies, each device running it on its part of x."""
    return _mx_kernel(x)


def _mx_infer(mesh: Mesh, shapes: tuple[Any, ...], out: Any) -> Shardings:
    """Return how the copies split, of x as it splits."""
    return copies_shardings(mesh, shapes[0].sharding.spec)


def _mx_partition(
    mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., MxArrays], Shardings, Shardings]:
    """Each device runs the kernel on its part of x: the blocks are whole on every
    device (``matmuls.blocks_whole``)."""
    spec = shapes[0].sharding.spec
    arg = NamedSharding(mesh, P(*split_of(spec)))
    return mesh, _mx_kernel, copies_shardings(mesh, spec), (arg,)


_mx_partitioned.def_partition(
    infer_sharding_from_operands=_mx_infer,
    partition=_mx_partition,
    sharding_rule="r c -> r c, r c, c r, c r",
)
