"""NVFP4, as NVIDIA's pretraining recipe uses it: E2M1 values in blocks of 16 along a
row, an E4M3 scale a block and a float32 scale a tensor. W quantizes in 16 x 16 tiles,
so that W and W^T hold the same values; dy rounds stochastically; and x^T and dy^T,
which dW multiplies, go through a random Hadamard transform of each 16 first.

`twice` quantizes an operand and its transpose (``blockscaled.Twice``): on a GPU by two
passes of Pallas kernels (`kernel_twice`), the first for the largest magnitudes, which
set the tensor scales, the second writing both copies, rounded by Blackwell's E2M1
conversions; elsewhere by XLA (`xla_twice`), their reference. Off a GPU the kernels run
interpreted, and quantize as XLA does, bit for bit.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Shaped, UInt8, UInt32

from bux import kernels
from bux.architecture.precision import blockscaled
from bux.architecture.precision.blockscaled import (
    E4M3_MAX,
    Blocks,
    Recipe,
    Shardings,
    as_bytes,
)
from bux.parallel.layout import axes_of

BLOCK = 16
E2M1_MAX = 6.0  # float4 E2M1's largest value


def twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4: by the kernels on a GPU, and by XLA elsewhere."""
    if not kernels.gpu():
        return xla_twice(x, seed, tiles, stochastic)
    return kernel_twice(x, seed, tiles, stochastic)


# NVIDIA's pretraining recipe: W in 16 x 16 tiles, dy rounded stochastically, and the
# operands of dW transformed.
NVFP4 = Recipe(
    BLOCK,
    x=twice,
    w=partial(twice, tiles=True),
    dy=partial(twice, stochastic=True),
)


# --- By XLA ---------------------------------------------------------------------------
# The kernels quantize each tile by `block_scales` and `scaled_values`, as XLA does the
# whole.

# A random Hadamard transform of 16 along the rows dW contracts: orthogonal, so it
# cancels in x^T dy, and it spreads each block's outliers over the block. Sylvester's
# Hadamard matrix, (-1)^popcount(i & j), its rows' signs drawn once.
_I = np.arange(BLOCK)
_SYLVESTER = (-1.0) ** np.bitwise_count(_I[:, None] & _I[None, :])
_SIGNS = np.random.default_rng(0).choice([-1.0, 1.0], BLOCK)
HADAMARD = (_SIGNS[:, None] * _SYLVESTER / 4).astype(np.float32)


def xla_twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4, by XLA. x^T is an operand of dW, so it goes
    through the Hadamard transform first; but W^T (`tiles`) is y's, and does not."""
    copy = partial(quantize, seed=seed, tiles=tiles, stochastic=stochastic)
    return copy(x), copy(x.T, hadamard=not tiles)


def quantize(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
    hadamard: bool = False,
) -> Blocks:
    """Return x in NVFP4, by XLA: E2M1 values, an E4M3 scale per 16 of a row, or per
    16 x 16 `tiles` (the same for x and x^T), and a float32 scale that takes x's
    largest magnitude to 6 x 448. `stochastic` rounds each value up or down, unbiased,
    by noise of `seed`; `hadamard` first transforms each 16 of a row."""
    x = x.astype(jnp.float32)
    if hadamard:
        x = hadamard_rows(x)
    encode = encode_scale(jnp.abs(x).max())
    # On a GPU, XLA folds the cast to E4M3 and back away, and would scale the values
    # by other scales than those the GEMM multiplies them by.
    scales = jax.lax.optimization_barrier(block_scales(x, encode, tiles))
    values = scaled_values(x, scales, 1 / encode)
    if stochastic:
        values = round_stochastically(values, _noise(values.shape, seed))
    values = jnp.clip(values, -E2M1_MAX, E2M1_MAX).astype(jnp.float4_e2m1fn)
    return blockscaled.kept_quantized(Blocks(values, scales, 1 / encode))


def hadamard_rows(x: Float[Array, "r c"]) -> Float[Array, "r c"]:
    """Return x with each 16 of a row through the random Hadamard transform."""
    r, c = x.shape
    blocks = x.reshape(r, c // BLOCK, BLOCK)
    return jnp.matmul(blocks, HADAMARD, precision="highest").reshape(r, c)


def encode_scale(amax: Float[Array, "*s"]) -> Float[Array, "*s"]:
    """Return the tensor scale that takes `amax` to 6 x 448, Transformer Engine's; 1
    for 0. Its inverse, the decode scale, dequantizes."""
    encode = jnp.minimum(E2M1_MAX * E4M3_MAX / amax, jnp.finfo(jnp.float32).max)
    return jnp.where((amax == 0) | (encode == 0), 1.0, encode)


def block_scales(
    x: Float[Array, "r c"], encode: Float[Array, ""], tiles: bool = False
) -> Shaped[Array, "r n"]:
    """Return an E4M3 scale per 16 of x's rows, or per 16 x 16 `tiles` for each of
    their rows: what takes the block's largest magnitude to 6 under the tensor scale
    `encode`, at most 448."""
    (r, c), b = x.shape, BLOCK
    amax = jnp.abs(x.reshape(r, c // b, b)).max(2)
    if tiles:
        amax = jnp.repeat(amax.reshape(r // b, b, c // b).max(1), b, axis=0)
    scale = jnp.minimum(amax * (encode * (1 / E2M1_MAX)), E4M3_MAX)
    return scale.astype(jnp.float8_e4m3fn)


def scaled_values(
    x: Float[Array, "r c"], scales: Shaped[Array, "r n"], decode: Float[Array, ""]
) -> Float[Array, "r c"]:
    """Return x's values in E2M1's range: each 16 of a row times 1 / (its scale x
    `decode`), at most float32's largest, for a scale of 0."""
    (r, c), b = x.shape, BLOCK
    inverse = 1 / (scales.astype(jnp.float32) * decode)
    inverse = jnp.minimum(inverse, jnp.finfo(jnp.float32).max)
    return (x.reshape(r, c // b, b) * inverse[:, :, None]).reshape(r, c)


# --- Stochastic rounding --------------------------------------------------------------
# Each quad, 4 consecutive elements of a row, takes 32 random bits: a hash of the
# quad's place in the whole array and the seed, so the same on any mesh. Its k-th
# element rounds by byte k, as Blackwell's cvt.rs takes 8 random bits a value.


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
    return jnp.sign(v) * jnp.minimum(rounded, E2M1_MAX)


def _noise(
    shape: tuple[int, ...], seed: UInt32[Array, ""] | None
) -> Float[Array, "*s"]:
    """Return noise in [0, 1) for each element of an array of `shape`, of its index
    and `seed`."""
    i = jax.lax.iota(jnp.uint32, int(np.prod(shape))).reshape(shape)
    return _uniform(i, salt_of(seed))


def _uniform(index: UInt32[Array, "*s"], salt: UInt32[Array, ""]) -> Float[Array, "*s"]:
    """Return noise in [0, 1) for each `index`, in 8 bits: the k-th byte of its quad's
    random bits for the quad's k-th element."""
    shift = (index & 3) * 8
    byte = (quad_bits(index >> 2, salt) >> shift) & 0xFF
    return byte.astype(jnp.float32) * 2.0**-8


def salt_of(seed: UInt32[Array, ""] | None) -> UInt32[Array, ""]:
    """Return what `quad_bits` hashes with each quad: a hash of `seed` (0 for None)."""
    return mix(jnp.uint32(0) if seed is None else seed)


def quad_bits(
    quad: UInt32[Array, "*s"], salt: UInt32[Array, ""]
) -> UInt32[Array, "*s"]:
    """Return the random bits of the quad numbered `quad`: a hash of it and `salt`."""
    return mix(quad * jnp.uint32(0x9E3779B1) + salt)


def mix(h: UInt32[Array, "*s"]) -> UInt32[Array, "*s"]:
    """Return a hash of h: murmur3's finalizer."""
    h = (h ^ (h >> 16)) * jnp.uint32(0x85EBCA6B)
    h = (h ^ (h >> 13)) * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


# --- By the kernels -------------------------------------------------------------------
# On a B300 the kernels round x and W bit for bit as XLA does there. Against XLA on a
# CPU 0.15% of the values are a step apart: a GPU's division is at times an ulp from
# the CPU's, and the scales' few bits put many values on the midpoints between E2M1's.

# The kernels' tiles, of rows and of columns, largest first, each a whole number of
# blocks: 64 x 128 is the fastest on a B300.
ROWS, COLUMNS = (64, 32, 16), (128, 64, 32, 16)
WARPS = 4
# The random Hadamard transform of the largest tile's rows, 16 at a time (HADAMARD on
# the diagonal); a smaller tile takes its top left corner.
TILE_HADAMARD = np.kron(np.eye(ROWS[0] // BLOCK), HADAMARD).astype(np.float32)


class Kind(NamedTuple):
    """Which operand a kernel quantizes: W, in 16 x 16 `tiles`; dy, rounded
    `stochastic`ally; else x."""

    tiles: bool
    stochastic: bool


def kernel_twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4 as `xla_twice` does, by two passes of the kernels,
    each device on its part of x; off a GPU, the kernels interpreted."""
    hadamard = jnp.asarray(TILE_HADAMARD, x.dtype)  # 1/4s, exact in bfloat16
    encode = encode_scale(_amaxes(x, hadamard, tiles))  # x's, then x^T's
    decode = 1 / encode
    salt = salt_of(seed).reshape(1)
    kind = Kind(tiles, stochastic)
    quantize = blockscaled.on_each_device(_quantize, _quantize_partitioned)
    rows, row_scales, cols, col_scales = quantize(
        x, hadamard, encode, decode, salt, kind
    )
    return _blocks(rows, row_scales, decode[0]), _blocks(cols, col_scales, decode[1])


def _amaxes(
    x: Float[Array, "r c"], hadamard: Float[Array, "t t"], tiles: bool
) -> Float[Array, "2"]:
    """Return the largest magnitude of x and of x^T transformed, by the first pass;
    W's, which is not transformed, by XLA, as a weight is small next to the
    activations."""
    if tiles:
        return jnp.abs(x).max().astype(jnp.float32).repeat(2)
    amaxes = blockscaled.on_each_device(_amaxes_by_kernel, _amaxes_partitioned)
    return amaxes(x, hadamard)


def _blocks(
    packed: UInt8[Array, "r h"], scales: UInt8[Array, "r n"], decode: Float[Array, ""]
) -> Blocks:
    """Return a copy as the GEMM takes it: its bytes as E2M1 and E4M3, as they are."""
    r, h = packed.shape
    values = jax.lax.bitcast_convert_type(packed, jnp.float4_e2m1fn).reshape(r, 2 * h)
    scales = jax.lax.bitcast_convert_type(scales, jnp.float8_e4m3fn)
    return Blocks(values, scales, decode)


# --- The second pass: the quantize kernel ---------------------------------------------

# x's E2M1 values two a byte and its scales' bits, then x^T's.
NvArrays = tuple[
    UInt8[Array, "r h"], UInt8[Array, "r n"], UInt8[Array, "c g"], UInt8[Array, "c q"]
]


def _quantize(
    x: Float[Array, "r c"],
    hadamard: Float[Array, "t t"],
    encode: Float[Array, "2"],
    decode: Float[Array, "2"],
    salt: UInt32[Array, "1"],
    kind: Kind,
    start: UInt32[Array, "2"] | None = None,
    shape: tuple[int, int] | None = None,
) -> NvArrays:
    """Quantize x and x^T, a program per tile of x, by x's tensor scales and then
    x^T's in `encode` and `decode`. x is the part at `start` of an array of `shape`,
    the whole if None: the noise hashes each element's place in the whole."""
    from jax.experimental.pallas import triton as plgpu

    (r, c), b = x.shape, BLOCK
    tr, tc = blockscaled.largest_tile(r, ROWS), blockscaled.largest_tile(c, COLUMNS)
    start = jnp.zeros(2, jnp.uint32) if start is None else start
    kernel = partial(_quantize_tile, shape=shape or (r, c), kind=kind)
    scalars = pl.BlockSpec(None, lambda i, j: (0,))
    shapes = [(r, c // 2), (r, c // b), (c, r // 2), (c, r // b)]
    return pl.pallas_call(
        kernel,
        grid=(r // tr, c // tc),
        in_specs=[
            pl.BlockSpec((tr, tc), lambda i, j: (i, j)),
            pl.BlockSpec((tr, tr), lambda i, j: (0, 0)),
            *[scalars] * 4,
        ],
        out_specs=[
            pl.BlockSpec((tr, tc // 2), lambda i, j: (i, j)),
            pl.BlockSpec((tr, tc // b), lambda i, j: (i, j)),
            pl.BlockSpec((tc, tr // 2), lambda i, j: (j, i)),
            pl.BlockSpec((tc, tr // b), lambda i, j: (j, i)),
        ],
        out_shape=[jax.ShapeDtypeStruct(s, jnp.uint8) for s in shapes],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1),
        interpret=kernels.interpreted(),
        name=f"nv_quantize_{tr}x{tc}",
    )(x, hadamard, encode, decode, salt, start)


# Pallas refs, which index and load as arrays do not.
def _quantize_tile(
    x_ref: Any,
    hadamard_ref: Any,
    encode_ref: Any,
    decode_ref: Any,
    salt_ref: Any,
    start_ref: Any,
    rows_ref: Any,
    row_scales_ref: Any,
    cols_ref: Any,
    col_scales_ref: Any,
    shape: tuple[int, int],
    kind: Kind,
) -> None:
    """Quantize a tile of x into its part of x's copy and of x^T's."""
    tile, salt = x_ref[...], salt_ref[0]
    (r, c), (tr, tc) = shape, tile.shape
    row = start_ref[0] + pl.program_id(0).astype(jnp.uint32) * tr  # the tile's first
    column = start_ref[1] + pl.program_id(1).astype(jnp.uint32) * tc
    x = tile.astype(jnp.float32)
    scales = block_scales(x, encode_ref[0], kind.tiles)
    values = scaled_values(x, scales, decode_ref[0])
    bits = _tile_bits(row, column, (tr, tc), c, salt) if kind.stochastic else None
    rows_ref[...], row_scales_ref[...] = _to_e2m1(values, bits), as_bytes(scales)
    if kind.tiles:  # W^T: the same values, transposed
        cols_ref[...] = _to_e2m1(values.T)
        col_scales_ref[...] = _tiles_transposed(as_bytes(scales))
        return
    xt = _hadamard_transposed(tile, hadamard_ref[...])  # x^T's tile, transformed
    scales = block_scales(xt, encode_ref[1])
    values = scaled_values(xt, scales, decode_ref[1])
    bits = _tile_bits(column, row, (tc, tr), r, salt) if kind.stochastic else None
    cols_ref[...], col_scales_ref[...] = _to_e2m1(values, bits), as_bytes(scales)


def _tile_bits(
    first_row: UInt32[Array, ""],
    first_column: UInt32[Array, ""],
    shape: tuple[int, int],
    width: int,
    salt: UInt32[Array, ""],
) -> UInt32[Array, "n q"]:
    """Return `quad_bits` for each quad of the (n, m) elements from (`first_row`,
    `first_column`) on of an array `width` wide."""
    n, m = shape
    rows = first_row + jax.lax.broadcasted_iota(jnp.uint32, (n, 1), 0)
    quads = first_column // 4 + jax.lax.broadcasted_iota(jnp.uint32, (1, m // 4), 1)
    return quad_bits(rows * jnp.uint32(width // 4) + quads, salt)


def _hadamard_transposed(
    x: Shaped[Array, "r c"], hadamard: Shaped[Array, "r r"]
) -> Float[Array, "c r"]:
    """Return `hadamard_rows(x.T)`: x's columns times `hadamard`, a transform for
    each 16 of x's rows."""
    columns = (((0,), (0,)), ((), ()))
    return jax.lax.dot_general(x, hadamard, columns, preferred_element_type=jnp.float32)


def _tiles_transposed(scales: UInt8[Array, "r q"]) -> UInt8[Array, "c p"]:
    """Return W^T's scales for each of its rows, of W's for each of W's rows: those of
    the same 16 x 16 tiles."""
    r, q = scales.shape
    tiles = scales.reshape(r // BLOCK, BLOCK, q).max(1)  # a tile's rows agree
    return jnp.repeat(tiles.T, BLOCK, axis=0)


# --- Rounding to E2M1 in a kernel -----------------------------------------------------


def _to_e2m1(
    values: Float[Array, "r c"], bits: UInt32[Array, "r q"] | None = None
) -> UInt8[Array, "r h"]:
    """Return values as E2M1, at most 6 in size, two a byte with the first in the low
    4 bits, as XLA lays E2M1 out: rounded to nearest even, or stochastically by
    `bits`, 32 a quad. On a GPU by Blackwell's conversions; interpreted, by their
    arithmetic."""
    if kernels.interpreted():
        if bits is not None:
            values = round_stochastically(values, _noise_of_bits(bits))
        return _pack(_e2m1_codes(values))
    if bits is None:
        return _cvt_rn(values)
    return _cvt_rs(values, bits)


def _noise_of_bits(bits: UInt32[Array, "r q"]) -> Float[Array, "r c"]:
    """Return each element's noise in [0, 1): its byte of its quad's `bits`, as
    `_uniform` does."""
    r, q = bits.shape
    shifts = jax.lax.broadcasted_iota(jnp.uint32, (r, q, 4), 2) * 8
    each = (bits[:, :, None] >> shifts) & 0xFF
    return each.reshape(r, 4 * q).astype(jnp.float32) * 2.0**-8


# Halfway between E2M1's magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4 and 6.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _e2m1_codes(v: Float[Array, "*s"]) -> UInt8[Array, "*s"]:
    """Return the 4 bits of the E2M1 value nearest each of v, at most 6 in size: its
    sign, then how many midpoints its magnitude is past. A midpoint rounds to the even
    code, as float casts do: down after an even one, up after an odd one."""
    size = jnp.abs(v)
    past = [size > m if i % 2 == 0 else size >= m for i, m in enumerate(_MIDPOINTS)]
    magnitude = sum(p.astype(jnp.uint8) for p in past)
    sign = (jax.lax.bitcast_convert_type(v, jnp.uint32) >> 31).astype(jnp.uint8)
    return (sign << 3) | magnitude


def _pack(codes: UInt8[Array, "r c"]) -> UInt8[Array, "r h"]:
    """Return the codes two a byte, the first in the low 4 bits."""
    r, c = codes.shape
    first, second = jax.lax.unstack(codes.reshape(r, c // 2, 2), axis=2)
    return first | (second << 4)


def _cvt_rn(values: Float[Array, "r c"]) -> UInt8[Array, "r h"]:
    """Return values as E2M1 rounded to nearest even, by Blackwell's cvt.rn, which
    puts its first operand in the high 4 bits: a byte a pair."""
    from jax.experimental.pallas import triton as plgpu

    r, c = values.shape
    first, second = jax.lax.unstack(values.reshape(r, c // 2, 2), axis=2)
    [pairs] = plgpu.elementwise_inline_asm(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; mov.b32 $0, {t, t, t, t}; }",
        args=[first, second],
        constraints="=r,f,f",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(first.shape, jnp.int32)],
    )
    return (pairs & 0xFF).astype(jnp.uint8)


# On a B300, cvt.rs spends rbits' byte 2 on its last operand, byte 0 bit-reversed on
# the one before, byte 3 on the second and byte 1 reversed on the first: brev and prmt
# rearrange `bits` so that the k-th value of a quad spends its byte k.
_CVT_RS = (
    "{ .reg .b32 reversed, spent; .reg .b16 t;"
    " brev.b32 reversed, $5;"
    " prmt.b32 spent, $5, reversed, 0x2046;"
    " cvt.rs.satfinite.e2m1x4.f32 t, {$4, $3, $2, $1}, spent;"
    " mov.b32 $0, {t, t}; }"
)


def _cvt_rs(
    values: Float[Array, "r c"], bits: UInt32[Array, "r q"]
) -> UInt8[Array, "r h"]:
    """Return values as E2M1 rounded stochastically, by Blackwell's cvt.rs, a quad by
    its 32 random bits: two bytes a quad."""
    from jax.experimental.pallas import triton as plgpu

    r, c = values.shape
    even, odd = jax.lax.unstack(values.reshape(r, c // 4, 2, 2), axis=3)
    first, third = jax.lax.unstack(even, axis=2)
    second, fourth = jax.lax.unstack(odd, axis=2)
    [both] = plgpu.elementwise_inline_asm(
        _CVT_RS,
        args=[first, second, third, fourth, bits.astype(jnp.int32)],
        constraints="=r,f,f,f,f,r",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(first.shape, jnp.int32)],
    )
    low, high = (both & 0xFF).astype(jnp.uint8), ((both >> 8) & 0xFF).astype(jnp.uint8)
    return jnp.stack([low, high], axis=-1).reshape(r, c // 2)


# --- The first pass: the amax kernel --------------------------------------------------


def _amaxes_by_kernel(
    x: Float[Array, "r c"], hadamard: Float[Array, "t t"]
) -> Float[Array, "2"]:
    """Return the largest magnitude of x and of x^T transformed, of a program's for
    each tile."""
    from jax.experimental.pallas import triton as plgpu

    r, c = x.shape
    tr, tc = blockscaled.largest_tile(r, ROWS), blockscaled.largest_tile(c, COLUMNS)
    each = jax.ShapeDtypeStruct((r // tr, c // tc), jnp.float32)
    rows, columns = pl.pallas_call(
        _amax_tile,
        grid=(r // tr, c // tc),
        in_specs=[
            pl.BlockSpec((tr, tc), lambda i, j: (i, j)),
            pl.BlockSpec((tr, tr), lambda i, j: (0, 0)),
        ],
        out_specs=[pl.BlockSpec((1, 1), lambda i, j: (i, j))] * 2,
        out_shape=[each, each],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1),
        interpret=kernels.interpreted(),
        name=f"nv_amax_{tr}x{tc}",
    )(x, hadamard)
    return jnp.stack([rows.max(), columns.max()])


def _amax_tile(x_ref: Any, hadamard_ref: Any, rows_ref: Any, cols_ref: Any) -> None:
    """Write a tile's largest magnitude, and its transpose's, transformed."""
    tile = x_ref[...]
    transformed = _hadamard_transposed(tile, hadamard_ref[...])
    rows_ref[...] = jnp.abs(tile.astype(jnp.float32)).max().reshape(1, 1)
    cols_ref[...] = jnp.abs(transformed).max().reshape(1, 1)


# --- On a mesh: each device on its part of x ------------------------------------------


def _offset(axes: Any, size: int) -> UInt32[Array, ""]:
    """Return where this device's part starts along a dimension split over `axes`
    into parts of `size`."""
    if axes is None:
        return jnp.uint32(0)
    return jax.lax.axis_index(axes).astype(jnp.uint32) * jnp.uint32(size)


@custom_partitioning
def _amaxes_partitioned(
    x: Float[Array, "r c"], hadamard: Float[Array, "t t"]
) -> Float[Array, "2"]:
    return _amaxes_by_kernel(x, hadamard)


def _amaxes_infer(mesh: Mesh, shapes: tuple[Any, ...], out: Any) -> NamedSharding:
    return NamedSharding(mesh, P())


def _amaxes_partition(
    mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., Float[Array, "2"]], NamedSharding, Shardings]:
    """Each device runs the kernel on its part of x, and the largest of theirs is
    every device's."""
    spec = P(*blockscaled.split_of(shapes[0].sharding.spec))
    split = axes_of(spec)

    def amaxes(
        x: Float[Array, "r c"], hadamard: Float[Array, "t t"]
    ) -> Float[Array, "2"]:
        mine = _amaxes_by_kernel(x, hadamard)
        return jax.lax.pmax(mine, split) if split else mine

    everywhere = NamedSharding(mesh, P())
    return mesh, amaxes, everywhere, (NamedSharding(mesh, spec), everywhere)


_amaxes_partitioned.def_partition(
    infer_sharding_from_operands=_amaxes_infer,
    partition=_amaxes_partition,
    sharding_rule="r c, h k -> s",
)


@partial(custom_partitioning, static_argnums=(5,))
def _quantize_partitioned(
    x: Float[Array, "r c"],
    hadamard: Float[Array, "t t"],
    encode: Float[Array, "2"],
    decode: Float[Array, "2"],
    salt: UInt32[Array, "1"],
    kind: Kind,
) -> NvArrays:
    return _quantize(x, hadamard, encode, decode, salt, kind)


def _quantize_infer(
    kind: Kind, mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> Shardings:
    return blockscaled.copies_shardings(mesh, shapes[0].sharding.spec)


def _quantize_partition(
    kind: Kind, mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., NvArrays], Shardings, Shardings]:
    """Each device runs the kernel on its part of x, the blocks whole on every device
    (``matmuls.blocks_whole``), its noise that of the elements' places in the whole."""
    whole = shapes[0].shape
    r, c = blockscaled.split_of(shapes[0].sharding.spec)

    def quantize(
        x: Float[Array, "r c"],
        hadamard: Float[Array, "t t"],
        encode: Float[Array, "2"],
        decode: Float[Array, "2"],
        salt: UInt32[Array, "1"],
    ) -> NvArrays:
        start = jnp.stack([_offset(r, x.shape[0]), _offset(c, x.shape[1])])
        return _quantize(x, hadamard, encode, decode, salt, kind, start, whole)

    everywhere = NamedSharding(mesh, P())
    args = (NamedSharding(mesh, P(r, c)), *[everywhere] * 4)
    return mesh, quantize, blockscaled.copies_shardings(mesh, P(r, c)), args


_quantize_partitioned.def_partition(
    infer_sharding_from_operands=_quantize_infer,
    partition=_quantize_partition,
    sharding_rule="r c, h k, e, d, s -> r c, r c, c r, c r",
)
