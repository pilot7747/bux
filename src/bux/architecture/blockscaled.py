"""Block-scaled GEMMs: a projection y = x W and its gradients, each operand of each of
the three GEMMs quantized along what that GEMM contracts, in blocks with a scale each,
then multiplied by cuDNN's block-scaled GEMM on a Blackwell GPU, and elsewhere by XLA,
which dequantizes and computes the same values.

A `Recipe` quantizes each of x, W and dy twice, along its rows and along its columns:
x along k for y = x W and along m for dW = x^T dy, W along n for dx = dy W^T and along
k for y, dy along n for dx and along m for dW. The forward keeps x's and W's second
copies for the backward, in the format, rather than x and W. Only the GEMMs named in
`passes` multiply in the format, the others in the dtype of their operands, which the
forward then keeps as they are. ``matmuls`` registers a format per recipe: `mxfp8`,
`mxfp8_2d` and `nvfp4` (``nvfp4``). A seed, which the trainer draws from each batch,
seeds what rounds stochastically.
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


Pass = str  # "fprop", "dgrad" or "wgrad": y's GEMM, dx's and dW's
PASSES: tuple[Pass, ...] = ("fprop", "dgrad", "wgrad")


def project(
    x: Float[Array, "*lead k"],
    w: Float[Array, "k n"],
    recipe: Recipe,
    out: DTypeLike,
    seed: UInt32[Array, ""],
    passes: tuple[Pass, ...] = PASSES,
) -> Float[Array, "*lead n"]:
    """Return x W in `out`, its GEMMs named in `passes` as `recipe` quantizes them."""
    dtypes = (x.dtype, w.dtype, jnp.dtype(out))
    y = _project(x.reshape(-1, x.shape[-1]), w, seed, recipe, dtypes, passes)
    return y.reshape(*x.shape[:-1], w.shape[1])


Dtypes = tuple[np.dtype, np.dtype, np.dtype]  # x's, W's and y's
# An operand of a GEMM: in the format, or as it is where its pass multiplies in compute.
Operand = Blocks | Float[Array, "_ _"]
Saved = tuple[Operand, Operand, UInt32[Array, ""]]  # x^T along m, W along n, the seed


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _project(
    x: Float[Array, "m k"],
    w: Float[Array, "k n"],
    seed: UInt32[Array, ""],
    recipe: Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
) -> Float[Array, "m n"]:
    return _forward(x, w, seed, recipe, dtypes, passes)[0]


def _forward(
    x: Float[Array, "m k"],
    w: Float[Array, "k n"],
    seed: UInt32[Array, ""],
    recipe: Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
) -> tuple[Float[Array, "m n"], Saved]:
    fprop, dgrad, wgrad = (p in passes for p in PASSES)
    x_k, x_m = _twice(recipe.x, x, seed, rows=fprop, columns=wgrad)
    w_n, w_k = _twice(recipe.w, w, seed, rows=dgrad, columns=fprop)
    return times(x_k, w_k, dtypes[2]), (x_m, w_n, seed)


def _backward(
    recipe: Recipe,
    dtypes: Dtypes,
    passes: tuple[Pass, ...],
    saved: Saved,
    dy: Float[Array, "m n"],
) -> tuple[Float[Array, "m k"], Float[Array, "k n"], None]:
    x_m, w_n, seed = saved
    _, dgrad, wgrad = (p in passes for p in PASSES)
    dy_n, dy_m = _twice(recipe.dy, dy, seed, rows=dgrad, columns=wgrad)
    return times(dy_n, w_n, dtypes[0]), times(x_m, dy_m, dtypes[1]), None


_project.defvjp(_forward, _backward)


def _twice(
    quantize: Twice,
    a: Float[Array, "r c"],
    seed: UInt32[Array, ""],
    rows: bool,
    columns: bool,
) -> tuple[Operand, Operand]:
    """Return a and a^T, each in the format where a pass multiplies it in the format
    (`rows` for a, `columns` for a^T), else as it is."""
    if not rows and not columns:
        return a, a.T
    along_rows, along_columns = quantize(a, seed)
    return along_rows if rows else a, along_columns if columns else a.T


def times(a: Operand, b: Operand, out: DTypeLike) -> Float[Array, "m n"]:
    """Return a b^T in `out`: by the block-scaled GEMM where both are in the format,
    else as they are."""
    if isinstance(a, Blocks) and isinstance(b, Blocks):
        return gemm(a, b, out)
    assert not isinstance(a, Blocks) and not isinstance(b, Blocks)
    return jnp.matmul(a, b.T, preferred_element_type=out)


def gemm(a: Blocks, b: Blocks, out: DTypeLike) -> Float[Array, "m n"]:
    """Return a b^T in `out`, of their values, block scales and scales. The scales'
    product rescales the GEMM's output in `out`, where XLA fuses it into what reads
    it, rather than in a float32 copy."""
    y = scaled_matmul(a, b, out)
    if a.scale is None or b.scale is None:
        return y
    return (y.astype(jnp.float32) * (a.scale * b.scale)).astype(out)


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


def mx_tiles_twice(
    w: Float[Array, "r c"], seed: UInt32[Array, ""] | None = None
) -> tuple[Blocks, Blocks]:
    """Return W and W^T in MXFP8 with a scale per 32 x 32 tile, so that y = x W and
    dx = dy W^T multiply the same values (TorchTitan's weights). By XLA: a weight is
    small next to the activations."""
    values, exponents = _mx_square(w.astype(jnp.float32))
    rows = jnp.repeat(exponents, MX_BLOCK, axis=0)
    columns = jnp.repeat(exponents.T, MX_BLOCK, axis=0)
    return _mx_blocks(values, rows), _mx_blocks(values.T, columns)


def _mx_square(
    x: Float[Array, "r c"],
) -> tuple[Shaped[Array, "r c"], UInt8[Array, "p q"]]:
    """Return x's E4M3 values, and the biased exponents of a power-of-two scale per
    32 x 32 tile, the least that takes the tile's largest value to at most 448."""
    (r, c), b = x.shape, MX_BLOCK
    tiles = x.reshape(r // b, b, c // b, b)
    exponent = _exponent_above(jnp.abs(tiles).max((1, 3)) / E4M3)
    values = tiles * _power_of_two(-exponent)[:, None, :, None]
    values = jnp.clip(values, -E4M3, E4M3).astype(jnp.float8_e4m3fn).reshape(r, c)
    return values, (exponent + 127).astype(jnp.uint8)


MXFP8 = Recipe(MX_BLOCK, mx_twice, mx_twice, mx_twice)
MXFP8_2D = Recipe(MX_BLOCK, mx_twice, mx_tiles_twice, mx_twice)


# --- NVFP4 --------------------------------------------------------------------------
# By XLA: the reference of NVFP4's kernels (``nvfp4``), and its format off a GPU.

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
    """Return x and x^T in NVFP4 (`nv_quantize`), by XLA. x^T multiplies along x's
    rows, which dW contracts unless `tiles` (W), so it is transformed first."""
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
    x = x.astype(jnp.float32)
    if hadamard:
        x = transform_rows(x)
    encode = nv_encode(jnp.abs(x).max())
    # On a GPU, XLA folds the cast to E4M3 and back away, and would scale the values
    # by scales other than those the GEMM multiplies them by.
    scales = jax.lax.optimization_barrier(nv_scales(x, encode, tiles))
    values = nv_values(x, scales, 1 / encode)
    if stochastic:
        values = round_stochastically(values, _noise(values.shape, seed))
    values = jnp.clip(values, -E2M1, E2M1).astype(jnp.float4_e2m1fn)
    return _keep(Blocks(values, scales, 1 / encode))


def transform_rows(x: Float[Array, "r c"]) -> Float[Array, "r c"]:
    """Return x with each 16 of a row through the random Hadamard transform."""
    r, c = x.shape
    blocks = x.reshape(r, c // NV_BLOCK, NV_BLOCK)
    return jnp.matmul(blocks, HADAMARD, precision="highest").reshape(r, c)


def nv_encode(amax: Float[Array, "*s"]) -> Float[Array, "*s"]:
    """Return the global scale that takes `amax` to 6 x 448, Transformer Engine's; 1
    for 0."""
    encode = jnp.minimum(E2M1 * E4M3 / amax, jnp.finfo(jnp.float32).max)
    return jnp.where((amax == 0) | (encode == 0), 1.0, encode)


# The kernels quantize their tiles by nv_scales and nv_values, as XLA does the whole.


def nv_scales(
    x: Float[Array, "r c"], encode: Float[Array, ""], tiles: bool = False
) -> Shaped[Array, "r n"]:
    """Return an E4M3 scale per 16 of x's rows, or per 16 x 16 `tiles` for each of
    their rows: what takes the block's largest magnitude to 6 under the global scale
    `encode`, at most 448."""
    (r, c), b = x.shape, NV_BLOCK
    amax = jnp.abs(x.reshape(r, c // b, b)).max(2)
    if tiles:
        amax = each_row(amax.reshape(r // b, b, c // b).max(1))
    scale = jnp.minimum(amax * (encode * (1 / E2M1)), E4M3)
    return scale.astype(jnp.float8_e4m3fn)


def each_row(tiles: Shaped[Array, "p q"]) -> Shaped[Array, "r q"]:
    """Return a 16 x 16 tile's value for each of its 16 rows."""
    p, q = tiles.shape
    rows = jnp.broadcast_to(tiles[:, None, :], (p, NV_BLOCK, q))
    return rows.reshape(p * NV_BLOCK, q)


def nv_values(
    x: Float[Array, "r c"], scales: Shaped[Array, "r n"], decode: Float[Array, ""]
) -> Float[Array, "r c"]:
    """Return x's values in E2M1's range: each 16 of a row times 1 / (its scale x
    `decode`), at most float32's largest, for a scale of 0."""
    (r, c), b = x.shape, NV_BLOCK
    inverse = 1 / (scales.astype(jnp.float32) * decode)
    inverse = jnp.minimum(inverse, jnp.finfo(jnp.float32).max)
    return (x.reshape(r, c // b, b) * inverse[:, :, None]).reshape(r, c)


def round_stochastically(
    v: Float[Array, "*s"], u: Float[Array, "*s"]
) -> Float[Array, "*s"]:
    """Return v at one of the two E2M1 values around it, given noise u in [0, 1) in
    8 bits: the upper where u and how far v is above the lower, in steps between them
    and rounded up to 8 bits, add up to 1, as Blackwell's stochastic conversion
    rounds. So the upper's odds are how far v is above the lower, to 1/256 of a step."""
    size = jnp.abs(v)
    step = jnp.where(size < 2, 0.5, jnp.where(size < 4, 1.0, 2.0))
    steps = jnp.where(size < 2, 2.0, jnp.where(size < 4, 1.0, 0.5))  # 1 / step, exactly
    lower = jnp.floor(size * steps)
    above = jnp.ceil((size * steps - lower) * 256) * 2.0**-8
    rounded = jnp.where(above + u >= 1, lower + 1, lower) * step
    return jnp.sign(v) * jnp.minimum(rounded, E2M1)


def _noise(
    shape: tuple[int, ...], seed: UInt32[Array, ""] | None
) -> Float[Array, "*s"]:
    """Return uniform noise in [0, 1), of a hash of each element's index and `seed`:
    the same on any mesh."""
    i = jax.lax.iota(jnp.uint32, int(np.prod(shape))).reshape(shape)
    return uniform(i, salt(seed))


def salt(seed: UInt32[Array, ""] | None) -> UInt32[Array, ""]:
    """Return what `quad_bits` hashes with each index, of `seed` (0 for None)."""
    return mix(jnp.uint32(0) if seed is None else seed)


def uniform(index: UInt32[Array, "*s"], salt: UInt32[Array, ""]) -> Float[Array, "*s"]:
    """Return noise in [0, 1) for each `index`, in 8 bits: the k-th byte of its
    group of 4's random bits (`quad_bits`) for the group's k-th, as Blackwell's
    stochastic conversion of 4 to E2M1 takes 8 bits each."""
    shift = (index & 3) * 8
    byte = (quad_bits(index >> 2, salt) >> shift) & 0xFF
    return byte.astype(jnp.float32) * 2.0**-8


def quad_bits(
    quad: UInt32[Array, "*s"], salt: UInt32[Array, ""]
) -> UInt32[Array, "*s"]:
    """Return the random bits of the group of 4 elements numbered `quad`: a hash of it
    and `salt`."""
    return mix(quad * jnp.uint32(0x9E3779B1) + salt)


def mix(h: UInt32[Array, "*s"]) -> UInt32[Array, "*s"]:
    """Return a hash of h: murmur3's finalizer."""
    h = (h ^ (h >> 16)) * jnp.uint32(0x85EBCA6B)
    h = (h ^ (h >> 13)) * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)
