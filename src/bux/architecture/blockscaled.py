"""Block-scaled GEMMs: a projection y = x W and its gradients, each operand of each of
the three GEMMs quantized along what that GEMM contracts, in blocks with a scale each,
then multiplied by cuDNN's block-scaled GEMM on a Blackwell GPU, and elsewhere by XLA,
which dequantizes and computes the same values.

A `Recipe` quantizes each of x, W and dy twice, along its rows and along its columns:
x along k for y = x W and along m for dW = x^T dy, W along n for dx = dy W^T and along
k for y, dy along n for dx and along m for dW. The forward keeps x's and W's second
copies for the backward, in the format, rather than x and W. ``matmuls`` registers a
format per recipe: `mxfp8` and `nvfp4`. A seed, which the trainer draws from each
batch, seeds what rounds stochastically.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.cudnn.scaled_matmul_stablehlo import _scaled_matmul_p
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Shaped, UInt8, UInt32

from bux import kernels

E4M3, E2M1 = 448.0, 6.0  # the largest float8 E4M3 and float4 E2M1


class Blocks(NamedTuple):
    """An (r, c) array as a format holds it: its `values`, a scale per block of a row,
    and a `scale` of the whole, or None for 1."""

    values: Shaped[Array, "r c"]
    scales: Shaped[Array, "r n"]
    scale: Float[Array, ""] | None


# a, seed -> a and a^T, each in blocks along its rows.
Twice = Callable[[Float[Array, "r c"], UInt32[Array, ""]], tuple[Blocks, Blocks]]


class Recipe(NamedTuple):
    """How a format quantizes x, W and dy, each twice (`Twice`); each GEMM's
    dimensions are whole `block`s."""

    block: int
    x: Twice
    w: Twice
    dy: Twice


def project(
    x: Float[Array, "*lead k"],
    w: Float[Array, "k n"],
    recipe: Recipe,
    out: DTypeLike,
    seed: UInt32[Array, ""],
) -> Float[Array, "*lead n"]:
    """Return x W in `out`, its three GEMMs as `recipe` quantizes them."""
    dtypes = (x.dtype, w.dtype, jnp.dtype(out))
    y = _project(x.reshape(-1, x.shape[-1]), w, seed, recipe, dtypes)
    return y.reshape(*x.shape[:-1], w.shape[1])


Dtypes = tuple[np.dtype, np.dtype, np.dtype]  # x's, W's and y's
Saved = tuple[Blocks, Blocks, UInt32[Array, ""]]  # x^T along m, W along n, the seed


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _project(
    x: Float[Array, "m k"],
    w: Float[Array, "k n"],
    seed: UInt32[Array, ""],
    recipe: Recipe,
    dtypes: Dtypes,
) -> Float[Array, "m n"]:
    return _forward(x, w, seed, recipe, dtypes)[0]


def _forward(
    x: Float[Array, "m k"],
    w: Float[Array, "k n"],
    seed: UInt32[Array, ""],
    recipe: Recipe,
    dtypes: Dtypes,
) -> tuple[Float[Array, "m n"], Saved]:
    x_k, x_m = recipe.x(x, seed)
    w_n, w_k = recipe.w(w, seed)
    return gemm(x_k, w_k, dtypes[2]), (x_m, w_n, seed)


def _backward(
    recipe: Recipe, dtypes: Dtypes, saved: Saved, dy: Float[Array, "m n"]
) -> tuple[Float[Array, "m k"], Float[Array, "k n"], None]:
    x_m, w_n, seed = saved
    dy_n, dy_m = recipe.dy(dy, seed)
    return gemm(dy_n, w_n, dtypes[0]), gemm(x_m, dy_m, dtypes[1]), None


_project.defvjp(_forward, _backward)


def gemm(a: Blocks, b: Blocks, out: DTypeLike) -> Float[Array, "m n"]:
    """Return a b^T in `out`, of their values, block scales and scales."""
    if a.scale is None or b.scale is None:
        return scaled_matmul(a, b, out)
    return (scaled_matmul(a, b, jnp.float32) * (a.scale * b.scale)).astype(out)


def scaled_matmul(a: Blocks, b: Blocks, out: DTypeLike) -> Float[Array, "m n"]:
    """Return a b^T of their values and block scales, in `out`. In a shard_map, by the
    GEMM itself, which jax's custom partitioning cannot place there."""
    args = (a.values[None], b.values[None], a.scales[None], b.scales[None])
    if _manual():
        return _scaled_matmul_p.bind(*args, preferred_element_type=jnp.dtype(out))[0][0]
    return jax.nn.scaled_matmul(*args, preferred_element_type=out)[0]


def _manual() -> bool:
    """Whether this traces inside a shard_map."""
    return bool(jax.sharding.get_abstract_mesh().manual_axes)


def _keep(blocks: Blocks) -> Blocks:
    """Return `blocks` as they are: without a native GEMM, XLA would otherwise fold the
    casts to the format and back away, and multiply the values unquantized."""
    values, scales = jax.lax.optimization_barrier((blocks.values, blocks.scales))
    return Blocks(values, scales, blocks.scale)


# --- MXFP8 --------------------------------------------------------------------------

MX_BLOCK = 32
# The kernel's tiles, of rows and of columns, largest first: 64 x 128 is the fastest
# on a B300, 3.2 TB/s at an MLP's shapes, 1.4x 128 x 128.
ROWS, COLUMNS = (64, 32), (128, 64, 32)
WARPS = 4


def mx_twice(
    x: Float[Array, "r c"], seed: UInt32[Array, ""] | None = None
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in MXFP8: by one pass of a kernel over x on a GPU, and by XLA
    elsewhere. MXFP8 rounds to nearest, and needs no seed."""
    if not kernels.gpu():
        return mx_quantize(x), mx_quantize(x.T)
    return mx_kernel_twice(x)


def mx_kernel_twice(x: Float[Array, "r c"]) -> tuple[Blocks, Blocks]:
    """Return x and x^T in MXFP8 by the kernel, each device on its part of x; off a
    GPU, the kernel interpreted."""
    kernel = _mx_kernel if _manual() else _mx_partitioned
    rows, row_scales, cols, col_scales = kernel(x)
    cols = jax.lax.bitcast_convert_type(cols, jnp.float8_e4m3fn)
    return _mx_blocks(rows, row_scales), _mx_blocks(cols, col_scales)


def mx_quantize(x: Float[Array, "r c"]) -> Blocks:
    """Return x in MXFP8: E4M3 values and, per 32 of a row, a power-of-two scale
    (E8M0), the least that takes the block's largest value to at most 448."""
    return _mx_blocks(*_mx_tile(x.astype(jnp.float32)))


def _mx_blocks(values: Shaped[Array, "r c"], exponents: UInt8[Array, "r n"]) -> Blocks:
    scales = jax.lax.bitcast_convert_type(exponents, jnp.float8_e8m0fnu)
    return _keep(Blocks(values, scales, None))


def _mx_tile(
    x: Float[Array, "r c"], down: bool = False
) -> tuple[Shaped[Array, "r c"], UInt8[Array, "p q"]]:
    """Return x's E4M3 values, and its scales' biased exponents (E8M0's bits), in
    blocks along its rows, or `down` its columns: x^T's, untransposed."""
    (r, c), b = x.shape, MX_BLOCK
    shape, axis = ((r // b, b, c), 1) if down else ((r, c // b, b), 2)
    blocks = x.reshape(shape)
    exponent = _exponent_above(jnp.abs(blocks).max(axis) / E4M3)
    values = blocks * jnp.expand_dims(_power_of_two(-exponent), axis)
    values = jnp.clip(values, -E4M3, E4M3).astype(jnp.float8_e4m3fn).reshape(r, c)
    return values, (exponent + 127).astype(jnp.uint8)


def _exponent_above(x: Float[Array, "*s"]) -> Int[Array, "*s"]:
    """Return the least e with 2^e >= x, for x >= 0, between -126 and 126."""
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    exponent = (bits >> 23).astype(jnp.int32) + ((bits & 0x7FFFFF) != 0)
    return jnp.clip(exponent - 127, -126, 126)


def _power_of_two(e: Int[Array, "*s"]) -> Float[Array, "*s"]:
    """Return 2^e, exactly, for integers e between -126 and 127."""
    return jax.lax.bitcast_convert_type((e + 127).astype(jnp.uint32) << 23, jnp.float32)


MxArrays = tuple[
    Shaped[Array, "r c"], UInt8[Array, "r n"], Shaped[Array, "c r"], UInt8[Array, "c q"]
]


def _mx_kernel(x: Float[Array, "r c"]) -> MxArrays:
    """Run `_mx_tile` on x and on x^T, a program per tile of x: x's values and
    exponents, then x^T's."""
    from jax.experimental.pallas import triton as plgpu

    (r, c), b = x.shape, MX_BLOCK
    tr, tc = _tile(r, ROWS), _tile(c, COLUMNS)

    # Pallas refs, which index and load as arrays do not.
    # x^T's values transpose as bytes, a quarter of what x's tile would be.
    def kernel(x_ref: Any, rows: Any, row_exps: Any, cols: Any, col_exps: Any) -> None:
        tile = x_ref[...].astype(jnp.float32)
        rows[...], row_exps[...] = _mx_tile(tile)
        values, exponents = _mx_tile(tile, down=True)
        cols[...] = jax.lax.bitcast_convert_type(values, jnp.uint8).T
        col_exps[...] = exponents.T

    shapes = [((r, c), jnp.float8_e4m3fn), ((r, c // b), jnp.uint8)]
    shapes += [((c, r), jnp.uint8), ((c, r // b), jnp.uint8)]
    return pl.pallas_call(
        kernel,
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
        interpret=jax.default_backend() != "gpu",
        name=f"mx_twice_{tr}x{tc}",
    )(x)


def _tile(n: int, tiles: tuple[int, ...]) -> int:
    """Return the largest of `tiles` that divides n, a whole number of blocks."""
    return next(t for t in tiles if n % t == 0)


@custom_partitioning
def _mx_partitioned(x: Float[Array, "r c"]) -> MxArrays:
    return _mx_kernel(x)


Shardings = tuple[NamedSharding, ...]


def _mx_shardings(mesh: Mesh, spec: P) -> Shardings:
    """x's values and exponents split as x is, and x^T's transposed."""
    r, c = (*spec, None, None)[:2]
    rows, cols = NamedSharding(mesh, P(r, c)), NamedSharding(mesh, P(c, r))
    return rows, rows, cols, cols


def _mx_infer(mesh: Mesh, shapes: tuple[Any, ...], out: Any) -> Shardings:
    return _mx_shardings(mesh, shapes[0].sharding.spec)


def _mx_partition(
    mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., MxArrays], Shardings, Shardings]:
    """Each device runs the kernel on its part of x: the blocks are whole on every
    device (``matmuls.blocks_whole``)."""
    spec = shapes[0].sharding.spec
    arg = NamedSharding(mesh, P(*(*spec, None, None)[:2]))
    return mesh, _mx_kernel, _mx_shardings(mesh, spec), (arg,)


_mx_partitioned.def_partition(
    infer_sharding_from_operands=_mx_infer,
    partition=_mx_partition,
    sharding_rule="r c -> r c, r c, c r, c r",
)

MXFP8 = Recipe(MX_BLOCK, mx_twice, mx_twice, mx_twice)


# --- NVFP4 --------------------------------------------------------------------------

NV_BLOCK = 16
# A random Hadamard transform of 16 along the rows dW contracts: orthogonal, so it
# cancels in x^T dy, and it spreads each block's outliers over the block. Sylvester's
# Hadamard matrix, (-1)^popcount(i & j), its rows' signs drawn once.
_I = np.arange(NV_BLOCK)
_SYLVESTER = (-1.0) ** np.bitwise_count(_I[:, None] & _I[None, :])
_SIGNS = np.random.default_rng(0).choice([-1.0, 1.0], NV_BLOCK)
HADAMARD = (_SIGNS[:, None] * _SYLVESTER / 4).astype(np.float32)


def nv_twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4 (`nv_quantize`). x^T multiplies along x's rows, which
    dW contracts unless `tiles` (W), so it is transformed first."""
    quantize = partial(nv_quantize, seed=seed, tiles=tiles, stochastic=stochastic)
    return quantize(x), quantize(x.T, hadamard=not tiles)


def nv_quantize(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
    hadamard: bool = False,
) -> Blocks:
    """Return x in NVFP4: E2M1 values, an E4M3 scale per 16 of a row, or per 16 x 16
    `tiles` (the same for x and x^T), and a float32 scale of the whole, which takes
    its largest value to 6 x 448. `stochastic` rounds the values up or down, with
    odds that leave them unbiased, by noise from `seed`; `hadamard` first transforms
    each 16 of a row."""
    r, c = x.shape
    x = x.astype(jnp.float32)
    if hadamard:
        x = (x.reshape(r, c // NV_BLOCK, NV_BLOCK) @ HADAMARD).reshape(r, c)
    amax = jnp.abs(x).max()
    scale = jnp.maximum(amax, jnp.finfo(jnp.float32).tiny) / (E2M1 * E4M3)
    blocks = (x / scale).reshape(r, c // NV_BLOCK, NV_BLOCK)
    scales = _tile_amax(blocks) if tiles else jnp.abs(blocks).max(-1)
    scales = jnp.minimum(scales / E2M1, E4M3).astype(jnp.float8_e4m3fn)
    wide = scales.astype(jnp.float32)
    values = blocks / jnp.where(wide > 0, wide, 1)[..., None]
    if stochastic:
        values = _round_stochastically(values, _noise(values.shape, seed))
    values = jnp.clip(values, -E2M1, E2M1).astype(jnp.float4_e2m1fn).reshape(r, c)
    return _keep(Blocks(values, scales, scale))


def _tile_amax(blocks: Float[Array, "r n b"]) -> Float[Array, "r n"]:
    """Return the largest magnitude of each 16 x 16 tile, for each of its rows."""
    r, n, b = blocks.shape
    tiles = jnp.abs(blocks).reshape(r // b, b, n, b).max((1, 3))
    return jnp.repeat(tiles, b, axis=0)


def _round_stochastically(
    v: Float[Array, "*s"], u: Float[Array, "*s"]
) -> Float[Array, "*s"]:
    """Return v at one of the two E2M1 values around it, the upper with odds of how
    far v is above the lower, given uniform noise u in [0, 1)."""
    size = jnp.abs(v)
    step = jnp.where(size < 2, 0.5, jnp.where(size < 4, 1.0, 2.0))
    return jnp.sign(v) * jnp.minimum(jnp.floor(size / step + u) * step, E2M1)


def _noise(
    shape: tuple[int, ...], seed: UInt32[Array, ""] | None
) -> Float[Array, "*s"]:
    """Return uniform noise in [0, 1), a hash of each element's index and `seed`: the
    same on any mesh."""
    i = jax.lax.iota(jnp.uint32, int(np.prod(shape))).reshape(shape)
    salt = mix(jnp.uint32(0) if seed is None else seed)
    return (mix(i * jnp.uint32(0x9E3779B1) + salt) >> 8).astype(jnp.float32) * 2.0**-24


def mix(h: UInt32[Array, "*s"]) -> UInt32[Array, "*s"]:
    """Return a hash of h: murmur3's finalizer."""
    h = (h ^ (h >> 16)) * jnp.uint32(0x85EBCA6B)
    h = (h ^ (h >> 13)) * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


NVFP4 = Recipe(
    NV_BLOCK,
    x=nv_twice,
    w=partial(nv_twice, tiles=True),
    dy=partial(nv_twice, stochastic=True),
)
