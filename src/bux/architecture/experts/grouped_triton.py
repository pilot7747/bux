"""Triton's quantizer of ``grouped_blockscaled``: x and x^T in MXFP8, E4M3 values with
a power-of-two scale per 32, in one pass over x, straight into the layout that
``grouped_mxfp8``'s kernels load (`grouped_mxfp8.Tiled`): each copy padded by 0 to
whole tiles of 128 along what it is in blocks along, and each tile's scales in their
bytes of tensor memory's 512-byte tiles. x is a matrix, or a stack of them, (g, r, c),
as each expert's W is; its rows from `filled` on quantize as 0. Given an index, the
rows are x[index], each read from x where it lies: an MoE's picks, from its tokens.

A program takes TILE_ROWS rows by 128 columns of x, in x's dtype: x's blocks along
its rows, and x^T's down x's columns, so that only the E4M3 bytes of x^T transpose.
The scaling, by a power of two, is exact in bfloat16 as in float32.
"""

from typing import no_type_check

import jax
import jax.numpy as jnp
import jax_triton
import triton
import triton.language as tl
from jaxtyping import Array, Float, Int, UInt8

from bux import jax_internals
from bux.architecture.experts.grouped_mxfp8 import LANES, Tiled
from bux.architecture.precision.blockscaled import MX_BLOCK

jax_internals.keep_triton_calls_whole()  # jax-triton's own rule fails on jax 0.12

# Rows of x a program takes, and its warps: 5.2 TB/s on a B300 at an MoE's shapes, 1.3x
# 4 warps and 1.7x 128 rows; rows gathered by index, 5.8 TB/s on one warp, 1.1x two.
TILE_ROWS, WARPS, GATHERED_WARPS = 32, 2, 1


def quantized_twice(
    x: Float[Array, "*g r c"],
    filled: Int[Array, ""],
    columns: bool = True,
    index: Int[Array, " m"] | None = None,
) -> tuple[Tiled, Tiled | None]:
    """Return x along its rows and, if `columns`, x^T along its rows, in MXFP8 as
    `blockscaled.mx_twice` quantizes them, laid out as `grouped_mxfp8.Tiled`: of x
    (r, c), values (r, c') and (c, r'), c' and r' whole tiles of 128. Given `index`,
    of x[index] (m, c), a matrix of x's rows."""
    stack = x if x.ndim == 3 else x[None]
    g, r, c = stack.shape if index is None else (1, *index.shape, x.shape[-1])
    rt, ct = -(-r // LANES), -(-c // LANES)
    shapes = [((g, r, ct * LANES), jnp.uint8), ((g, rt, ct, 32, 16), jnp.uint8)]
    if columns:
        shapes += [((g, c, rt * LANES), jnp.uint8), ((g, ct, rt, 32, 16), jnp.uint8)]
    values, scales, *transposed = jax_triton.triton_call(
        stack,
        jnp.zeros(1, jnp.int32) if index is None else index.astype(jnp.int32),
        filled.astype(jnp.int32).reshape(1),
        kernel=_quantize_kernel if columns else _quantize_rows_kernel,
        out_type=[jax.ShapeDtypeStruct(*s) for s in shapes],
        grid=(g, rt * (LANES // TILE_ROWS), ct),  # every row of each tile of 128
        name="bux_mx_twice" if columns else "bux_mx_rows",
        num_warps=WARPS if index is None else GATHERED_WARPS,
        num_stages=1,
        r=r,
        c=c,
        tile_r=TILE_ROWS,
        gathered=index is not None,
    )
    ndim = x.ndim if index is None else 2
    rows = _tiled(values, scales, ndim)
    return rows, _tiled(*transposed, ndim) if columns else None


def _tiled(
    values: UInt8[Array, "g r c"], scales: UInt8[Array, "g p q 32 16"], ndim: int
) -> Tiled:
    """Return the kernel's bytes as `Tiled`, without the stack's axis if x had none."""
    as_e4m3 = jax.lax.bitcast_convert_type(values, jnp.float8_e4m3fn)
    as_e8m0 = jax.lax.bitcast_convert_type(scales, jnp.float8_e8m0fnu)
    return Tiled(as_e4m3, as_e8m0) if ndim == 3 else Tiled(as_e4m3[0], as_e8m0[0])


# Triton's kernels: what they take is a tl.tensor or a pointer, or a constant; and
# they may name no global but Triton's. Triton compiles each from its source in this
# module's globals, so no type checker may wrap one, as the tests' import hook would.


@triton.jit
@no_type_check
def _quantize_kernel(
    x: tl.tensor,
    index: tl.tensor,
    rows: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    t_values: tl.tensor,
    t_scales: tl.tensor,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    gathered: tl.constexpr,
) -> None:
    """A tile of x in E4M3 in blocks of 32 along its rows, and of x^T along its
    columns, x's rows from `rows` on as 0."""
    tile, rs, cs = _tile_of(x, index, rows, r, c, tile_r, gathered)
    _store_along_rows(tile, values, scales, rs, cs, r, c, tile_r)
    _store_along_columns(tile, t_values, t_scales, rs, cs, r, c, tile_r)


@triton.jit
@no_type_check
def _quantize_rows_kernel(
    x: tl.tensor,
    index: tl.tensor,
    rows: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    gathered: tl.constexpr,
) -> None:
    """A tile of x in E4M3 in blocks of 32 along its rows, x's rows from `rows` on as
    0."""
    tile, rs, cs = _tile_of(x, index, rows, r, c, tile_r, gathered)
    _store_along_rows(tile, values, scales, rs, cs, r, c, tile_r)


@triton.jit
@no_type_check
def _tile_of(
    x: tl.tensor,
    index: tl.tensor,
    rows: tl.tensor,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    gathered: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    """Return this program's tile of x[g], or of x[index] if `gathered`, 0 past it and
    from `rows` on, and the tile's rows and columns."""
    rs = tl.program_id(1) * tile_r + tl.arange(0, tile_r)
    cs = tl.program_id(2) * 128 + tl.arange(0, 128)
    kept = (rs < r) & (rs < tl.load(rows))
    if gathered:
        lying = tl.load(index + rs, mask=rs < r, other=0).to(tl.int64)
        at = x + lying[:, None] * c
    else:
        at = x + tl.program_id(0).to(tl.int64) * r * c + rs[:, None].to(tl.int64) * c
    tile = tl.load(at + cs[None, :], mask=kept[:, None] & (cs < c)[None, :], other=0.0)
    return tile, rs, cs


@triton.jit
@no_type_check
def _store_along_rows(
    tile: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    rs: tl.tensor,
    cs: tl.tensor,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
) -> None:
    """Store the tile in E4M3 in blocks of 32 along its rows, of x's (r, c'), and
    their scales in their tile of tensor memory's layout."""
    v, e = _in_e4m3(tile.reshape(tile_r, 4, 32), 2)
    width: tl.constexpr = (c + 127) // 128 * 128
    g = tl.program_id(0).to(tl.int64)
    at = values + g * r * width + rs[:, None].to(tl.int64) * width + cs[None, :]
    tl.store(at, v.reshape(tile_r, 128), mask=(rs < r)[:, None])
    i, j = tl.program_id(1) * tile_r // 128, tl.program_id(2)
    tile_of_scales = (g * ((r + 127) // 128) + i) * ((c + 127) // 128) + j
    _store_scales(scales, tile_of_scales, rs % 128, 0, e)


@triton.jit
@no_type_check
def _store_along_columns(
    tile: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    rs: tl.tensor,
    cs: tl.tensor,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
) -> None:
    """Store the tile's transpose in E4M3 in blocks of 32 along its rows, of x^T's
    (c, r'), and their scales: blocks down x's columns, only their bytes transposed."""
    v, e = _in_e4m3(tile.reshape(tile_r // 32, 32, 128), 1)
    height: tl.constexpr = (r + 127) // 128 * 128
    g = tl.program_id(0).to(tl.int64)
    at = values + g * c * height + cs[:, None].to(tl.int64) * height + rs[None, :]
    tl.store(at, tl.trans(v.reshape(tile_r, 128)), mask=(cs < c)[:, None])
    i, j = tl.program_id(1) * tile_r // 128, tl.program_id(2)
    tile_of_scales = (g * ((c + 127) // 128) + j) * ((r + 127) // 128) + i
    first = (tl.program_id(1) * tile_r % 128) // 32  # this tile's first block of 4
    _store_scales(scales, tile_of_scales, tl.arange(0, 128), first, tl.trans(e))


@triton.jit
@no_type_check
def _store_scales(
    scales: tl.tensor,
    tile: tl.tensor,
    lines: tl.tensor,
    first: tl.tensor,
    exponents: tl.tensor,
) -> None:
    """Store the exponents of `lines` (rows of 128) and their blocks from `first` (of
    4) in their bytes of 512-byte tile `tile`: row i's 4 at (i % 32) * 16 + (i // 32) * 4."""
    blocks = first + tl.arange(0, exponents.shape[1])
    at = (lines[:, None] % 32) * 16 + (lines[:, None] // 32) * 4 + blocks[None, :]
    tl.store(scales + tile * 512 + at, exponents)


@triton.jit
@no_type_check
def _in_e4m3(blocks: tl.tensor, axis: tl.constexpr) -> tuple[tl.tensor, tl.tensor]:
    """Return blocks in E4M3 along `axis`, each times the least power of two that
    takes its largest value to at most 448, and those powers' biased exponents (E8M0's
    bits), as ``blockscaled._in_e4m3`` computes them."""
    over = tl.max(tl.abs(blocks), axis=axis).to(tl.float32) / 448.0
    bits = over.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23).to(tl.int32) + ((bits & 0x7FFFFF) != 0).to(tl.int32) - 127
    exponent = tl.minimum(tl.maximum(exponent, -126), 126)
    down = ((127 - exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scaled = blocks * tl.expand_dims(down.to(blocks.dtype), axis)
    values = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    return values.to(tl.uint8, bitcast=True), (exponent + 127).to(tl.uint8)


assert MX_BLOCK == 32  # the kernels' scales, a 32nd of their values' offsets
