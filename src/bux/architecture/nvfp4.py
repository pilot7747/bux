"""NVFP4's quantizers, by Pallas kernels on a GPU and elsewhere by XLA
(``blockscaled.nv_twice``, their reference): each of x, W and dy and its transpose,
which the three GEMMs of y = x W multiply, in two passes over it.

The first pass finds x's largest magnitude and that of x^T through the random
Hadamard transform, which set the two copies' global scales, as Transformer Engine's
NVFP4 recipe scales them: per tensor, from this step's values. The second quantizes
each tile of x into both copies at once, by the same functions of a tile as XLA
(``blockscaled.nv_scales``, ``nv_values``), and rounds by Blackwell's conversions to
E2M1, which round, saturate and pack two values a byte in one instruction: to nearest
even, or stochastically by 8 random bits a value. A weight's 16 x 16 tiles are the
same for W and W^T, untransformed, and its largest magnitude is XLA's: a weight is
small next to the activations.

Off a GPU the kernels run interpreted, the conversions by their arithmetic, and
quantize as XLA does, bit for bit. On one Triton divides within 2 ulps, not exactly,
and a value within as much of a midpoint between E2M1's values may round the other
way.
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
from bux.architecture import blockscaled
from bux.architecture.blockscaled import HADAMARD, NV_BLOCK, Blocks, Recipe

# The kernels' tiles, of rows and of columns, largest first, each a whole number of
# blocks: 64 x 128 is the fastest on a B300.
ROWS, COLUMNS = (64, 32, 16), (128, 64, 32, 16)
WARPS = 4
# The random Hadamard transform of each 16 of a tile's rows, of the largest.
HADAMARDS = np.kron(np.eye(ROWS[0] // NV_BLOCK), HADAMARD).astype(np.float32)


def twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4, as ``blockscaled.nv_twice``: by the kernels on a
    GPU, and by XLA elsewhere."""
    if not kernels.gpu():
        return blockscaled.nv_twice(x, seed, tiles, stochastic)
    return kernel_twice(x, seed, tiles, stochastic)


class How(NamedTuple):
    """What a kernel quantizes: W's 16 x 16 `tiles`, or dy, rounded `stochastic`ally;
    else x."""

    tiles: bool
    stochastic: bool


def kernel_twice(
    x: Float[Array, "r c"],
    seed: UInt32[Array, ""] | None = None,
    tiles: bool = False,
    stochastic: bool = False,
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in NVFP4 by the kernels, each device on its part of x; off a
    GPU, the kernels interpreted."""
    hadamards = jnp.asarray(HADAMARDS, x.dtype)  # 1/4s, exact in bfloat16
    encode = blockscaled.nv_encode(_amaxes(x, hadamards, tiles))
    decode = 1 / encode
    salt = blockscaled.salt(seed).reshape(1)
    quantize = _quantize if _manual() else _quantize_partitioned
    rows, row_scales, cols, col_scales = quantize(
        x, hadamards, encode, decode, salt, How(tiles, stochastic)
    )
    return _blocks(rows, row_scales, decode[0]), _blocks(cols, col_scales, decode[1])


def _amaxes(
    x: Float[Array, "r c"], hadamards: Float[Array, "t t"], tiles: bool
) -> Float[Array, "2"]:
    """Return the largest magnitude of x and of x^T transformed, which W's tiles are
    not."""
    if tiles:
        return jnp.abs(x).max().astype(jnp.float32).repeat(2)
    if _manual():
        return _amax_kernels(x, hadamards)
    return _amaxes_partitioned(x, hadamards)


def _blocks(
    packed: UInt8[Array, "r h"], scales: UInt8[Array, "r n"], decode: Float[Array, ""]
) -> Blocks:
    """Return a copy as the GEMM takes it: its bytes as E2M1 and E4M3, as they are."""
    r, h = packed.shape
    values = jax.lax.bitcast_convert_type(packed, jnp.float4_e2m1fn).reshape(r, 2 * h)
    scales = jax.lax.bitcast_convert_type(scales, jnp.float8_e4m3fn)
    return Blocks(values, scales, decode)


def _manual() -> bool:
    """Whether this traces inside a shard_map."""
    return bool(jax.sharding.get_abstract_mesh().manual_axes)


def _interpreted() -> bool:
    """Whether the kernels run interpreted, by XLA: off a GPU."""
    return jax.default_backend() != "gpu"


# --- Rounding to E2M1 -----------------------------------------------------------------


def _rounded(
    values: Float[Array, "r c"], bits: UInt32[Array, "r q"] | None = None
) -> UInt8[Array, "r h"]:
    """Return values as E2M1, at most 6 in size, two a byte with the first in the low
    4 bits, as XLA lays E2M1 out: rounded to nearest even, or stochastically by
    `bits`, 32 for each 4 values (``blockscaled.uniform``). On a GPU by Blackwell's
    conversions; interpreted by their arithmetic."""
    if _interpreted():
        if bits is not None:
            values = blockscaled.round_stochastically(values, _spread(bits))
        return _pack(_e2m1(values))
    if bits is None:
        return _nearest(values)
    return _stochastic(values, bits)


def _spread(bits: UInt32[Array, "r q"]) -> Float[Array, "r c"]:
    """Return each of 4 values' noise in [0, 1): its byte of its group's `bits`."""
    r, q = bits.shape
    shifts = jax.lax.broadcasted_iota(jnp.uint32, (r, q, 4), 2) * 8
    each = (bits[:, :, None] >> shifts) & 0xFF
    return each.reshape(r, 4 * q).astype(jnp.float32) * 2.0**-8


# Halfway between E2M1's magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4 and 6.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _e2m1(v: Float[Array, "*s"]) -> UInt8[Array, "*s"]:
    """Return the 4 bits of the E2M1 value nearest each of v, at most 6 in size: its
    sign, then the number of magnitudes it is past the midpoints to. A midpoint
    rounds to the even code, as float casts do: down after an even one, up after an
    odd one."""
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


def _nearest(values: Float[Array, "r c"]) -> UInt8[Array, "r h"]:
    """Return values as E2M1 by Blackwell's cvt.rn, which puts its first operand in
    the high 4 bits: a byte a pair."""
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
# rearrange `bits` so that the k-th value of a group spends its byte k.
_CVT_RS = (
    "{ .reg .b32 reversed, spent; .reg .b16 t;"
    " brev.b32 reversed, $5;"
    " prmt.b32 spent, $5, reversed, 0x2046;"
    " cvt.rs.satfinite.e2m1x4.f32 t, {$4, $3, $2, $1}, spent;"
    " mov.b32 $0, {t, t}; }"
)


def _stochastic(
    values: Float[Array, "r c"], bits: UInt32[Array, "r q"]
) -> UInt8[Array, "r h"]:
    """Return values as E2M1 by Blackwell's cvt.rs, a group of 4 by its 32 random
    bits: two bytes a group."""
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


# --- The math of a tile -------------------------------------------------------------


def _quad_bits(
    first_row: UInt32[Array, ""],
    first_column: UInt32[Array, ""],
    shape: tuple[int, int],
    width: int,
    salt: UInt32[Array, ""],
) -> UInt32[Array, "n q"]:
    """Return ``blockscaled.uniform``'s random bits for each group of 4 of the (n, m)
    elements from (`first_row`, `first_column`) on of an array `width` wide."""
    n, m = shape
    rows = first_row + jax.lax.broadcasted_iota(jnp.uint32, (n, 1), 0)
    quads = first_column // 4 + jax.lax.broadcasted_iota(jnp.uint32, (1, m // 4), 1)
    return blockscaled.quad_bits(rows * jnp.uint32(width // 4) + quads, salt)


def _transformed(
    x: Shaped[Array, "r c"], hadamards: Shaped[Array, "r r"]
) -> Float[Array, "c r"]:
    """Return x^T with each 16 of a row through the random Hadamard transform, as
    ``blockscaled.transform_rows``: x's columns by `hadamards`, a transform for each 16
    of x's rows."""
    columns = (((0,), (0,)), ((), ()))
    return jax.lax.dot_general(
        x, hadamards, columns, preferred_element_type=jnp.float32
    )


def _bytes(scales: Shaped[Array, "r n"]) -> UInt8[Array, "r n"]:
    return jax.lax.bitcast_convert_type(scales, jnp.uint8)


def _tiles_transposed(scales: UInt8[Array, "r q"]) -> UInt8[Array, "c p"]:
    """Return W^T's scales for each of its rows, of W's for each of W's rows: those of
    the same 16 x 16 tiles."""
    r, q = scales.shape
    tiles = scales.reshape(r // NV_BLOCK, NV_BLOCK, q).max(1)  # a tile's rows agree
    return blockscaled.each_row(tiles.T)


# --- The quantize kernel ------------------------------------------------------------

NvArrays = tuple[
    UInt8[Array, "r h"], UInt8[Array, "r n"], UInt8[Array, "c g"], UInt8[Array, "c q"]
]


def _quantize(
    x: Float[Array, "r c"],
    hadamards: Float[Array, "t t"],
    encode: Float[Array, "2"],
    decode: Float[Array, "2"],
    salt: UInt32[Array, "1"],
    how: How,
    start: UInt32[Array, "2"] | None = None,
    shape: tuple[int, int] | None = None,
) -> NvArrays:
    """Quantize x and x^T, a program per tile of x: x's E2M1 values two a byte and the
    bits of its scales, then x^T's. Its global scales are `encode`'s and `decode`'s
    first and second. x is the part of an array of `shape` at `start`, where the noise
    hashes each element's index, the whole array if None."""
    from jax.experimental.pallas import triton as plgpu

    (r, c), b = x.shape, NV_BLOCK
    tr, tc = _tile(r, ROWS), _tile(c, COLUMNS)
    start = jnp.zeros(2, jnp.uint32) if start is None else start
    kernel = partial(_quantize_kernel, shape=shape or (r, c), how=how)
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
        interpret=_interpreted(),
        name=f"nv_quantize_{tr}x{tc}",
    )(x, hadamards, encode, decode, start, salt)


# Pallas refs, which index and load as arrays do not.
def _quantize_kernel(
    x_ref: Any,
    hadamards_ref: Any,
    encode_ref: Any,
    decode_ref: Any,
    start_ref: Any,
    salt_ref: Any,
    rows_ref: Any,
    row_scales_ref: Any,
    cols_ref: Any,
    col_scales_ref: Any,
    shape: tuple[int, int],
    how: How,
) -> None:
    tile, salt = x_ref[...], salt_ref[0]
    (r, c), (tr, tc) = shape, tile.shape
    row = start_ref[0] + pl.program_id(0).astype(jnp.uint32) * tr  # the tile's first
    column = start_ref[1] + pl.program_id(1).astype(jnp.uint32) * tc
    x = tile.astype(jnp.float32)
    scales = blockscaled.nv_scales(x, encode_ref[0], how.tiles)
    values = blockscaled.nv_values(x, scales, decode_ref[0])
    bits = _quad_bits(row, column, (tr, tc), c, salt) if how.stochastic else None
    rows_ref[...], row_scales_ref[...] = _rounded(values, bits), _bytes(scales)
    if how.tiles:  # W^T: the same values, transposed
        cols_ref[...] = _rounded(values.T)
        col_scales_ref[...] = _tiles_transposed(_bytes(scales))
        return
    x = _transformed(tile, hadamards_ref[...])  # x^T's tile, transformed
    scales = blockscaled.nv_scales(x, encode_ref[1])
    values = blockscaled.nv_values(x, scales, decode_ref[1])
    bits = _quad_bits(column, row, (tc, tr), r, salt) if how.stochastic else None
    cols_ref[...], col_scales_ref[...] = _rounded(values, bits), _bytes(scales)


def _tile(n: int, tiles: tuple[int, ...]) -> int:
    """Return the largest of `tiles` that divides n."""
    return next(t for t in tiles if n % t == 0)


# --- The amax kernel ----------------------------------------------------------------


def _amax_kernels(
    x: Float[Array, "r c"], hadamards: Float[Array, "t t"]
) -> Float[Array, "2"]:
    """Return the largest magnitude of x and of x^T transformed, of a kernel's for
    each tile."""
    from jax.experimental.pallas import triton as plgpu

    r, c = x.shape
    tr, tc = _tile(r, ROWS), _tile(c, COLUMNS)
    each = jax.ShapeDtypeStruct((r // tr, c // tc), jnp.float32)
    rows, columns = pl.pallas_call(
        _amax_kernel,
        grid=(r // tr, c // tc),
        in_specs=[
            pl.BlockSpec((tr, tc), lambda i, j: (i, j)),
            pl.BlockSpec((tr, tr), lambda i, j: (0, 0)),
        ],
        out_specs=[pl.BlockSpec((1, 1), lambda i, j: (i, j))] * 2,
        out_shape=[each, each],
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1),
        interpret=_interpreted(),
        name=f"nv_amax_{tr}x{tc}",
    )(x, hadamards)
    return jnp.stack([rows.max(), columns.max()])


def _amax_kernel(x_ref: Any, hadamards_ref: Any, rows_ref: Any, cols_ref: Any) -> None:
    tile = x_ref[...]
    transformed = _transformed(tile, hadamards_ref[...])
    rows_ref[...] = jnp.abs(tile.astype(jnp.float32)).max().reshape(1, 1)
    cols_ref[...] = jnp.abs(transformed).max().reshape(1, 1)


# --- On a mesh: each device on its part of x -----------------------------------------

Shardings = tuple[NamedSharding, ...]


def _axes(spec: P) -> tuple[Any, Any]:
    """Return what x's rows and columns split over: a mesh axis, several, or None."""
    rows, columns = (*spec, None, None)[:2]
    return rows, columns


def _offset(axes: Any, size: int) -> UInt32[Array, ""]:
    """Return where this device's part starts along a dimension split over `axes`
    into parts of `size`."""
    if axes is None:
        return jnp.uint32(0)
    return jax.lax.axis_index(axes).astype(jnp.uint32) * jnp.uint32(size)


def _names(spec: P) -> tuple[str, ...]:
    """Return every mesh axis that `spec` splits over."""
    names = []
    for axes in spec:
        names += [] if axes is None else [axes] if isinstance(axes, str) else list(axes)
    return tuple(names)


@custom_partitioning
def _amaxes_partitioned(
    x: Float[Array, "r c"], hadamards: Float[Array, "t t"]
) -> Float[Array, "2"]:
    return _amax_kernels(x, hadamards)


def _amaxes_infer(mesh: Mesh, shapes: tuple[Any, ...], out: Any) -> NamedSharding:
    return NamedSharding(mesh, P())


def _amaxes_partition(
    mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., Float[Array, "2"]], NamedSharding, Shardings]:
    """Each device runs the kernel on its part of x, and the largest of theirs is
    every device's."""
    spec = P(*_axes(shapes[0].sharding.spec))
    split = _names(spec)

    def amaxes(
        x: Float[Array, "r c"], hadamards: Float[Array, "t t"]
    ) -> Float[Array, "2"]:
        mine = _amax_kernels(x, hadamards)
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
    hadamards: Float[Array, "t t"],
    encode: Float[Array, "2"],
    decode: Float[Array, "2"],
    salt: UInt32[Array, "1"],
    how: How,
) -> NvArrays:
    return _quantize(x, hadamards, encode, decode, salt, how)


def _nv_shardings(mesh: Mesh, spec: P) -> Shardings:
    """x's codes and scales split as x is, and x^T's transposed."""
    r, c = _axes(spec)
    rows, cols = NamedSharding(mesh, P(r, c)), NamedSharding(mesh, P(c, r))
    return rows, rows, cols, cols


def _quantize_infer(
    how: How, mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> Shardings:
    return _nv_shardings(mesh, shapes[0].sharding.spec)


def _quantize_partition(
    how: How, mesh: Mesh, shapes: tuple[Any, ...], out: Any
) -> tuple[Mesh, Callable[..., NvArrays], Shardings, Shardings]:
    """Each device runs the kernel on its part of x, the blocks whole on every device
    (``matmuls.blocks_whole``), its noise that of the elements' places in the whole."""
    whole, (r, c) = shapes[0].shape, _axes(shapes[0].sharding.spec)

    def quantize(
        x: Float[Array, "r c"],
        hadamards: Float[Array, "t t"],
        encode: Float[Array, "2"],
        decode: Float[Array, "2"],
        salt: UInt32[Array, "1"],
    ) -> NvArrays:
        start = jnp.stack([_offset(r, x.shape[0]), _offset(c, x.shape[1])])
        return _quantize(x, hadamards, encode, decode, salt, how, start, whole)

    everywhere = NamedSharding(mesh, P())
    args = (NamedSharding(mesh, P(r, c)), *[everywhere] * 4)
    return mesh, quantize, _nv_shardings(mesh, P(r, c)), args


_quantize_partitioned.def_partition(
    infer_sharding_from_operands=_quantize_infer,
    partition=_quantize_partition,
    sharding_rule="r c, h k, e, d, s -> r c, r c, c r, c r",
)


# NVFP4: E2M1 values, an E4M3 scale per 16 and a float32 scale per tensor, W in 16 x 16
# tiles, dy rounded stochastically, and x and dy transformed for dW (NVIDIA's
# pretraining recipe).
NVFP4 = Recipe(
    NV_BLOCK,
    x=twice,
    w=partial(twice, tiles=True),
    dy=partial(twice, stochastic=True),
)
