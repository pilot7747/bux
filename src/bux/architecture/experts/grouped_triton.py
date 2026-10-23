"""Triton's quantizer of ``grouped_blockscaled``: x and x^T in MXFP8, E4M3 values with
a power-of-two scale per 32, in one pass over x (`quantized_twice`), x's rows past the
groups as 0. The experts' GEMMs multiply them by ``grouped_mxfp8``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_triton
import triton
import triton.language as tl
from jaxtyping import Array, Float, Int, UInt8

from bux import jax_internals
from bux.architecture.precision.blockscaled import MX_BLOCK, Blocks

jax_internals.keep_triton_calls_whole()  # jax-triton's own rule fails on jax 0.12


class Tiles(NamedTuple):
    """The quantizer's tiles, of rows and columns, and how it runs."""

    m: int
    n: int
    warps: int
    stages: int


QUANTIZE = Tiles(m=64, n=128, warps=4, stages=1)  # the most of each


def quantized_twice(
    x: Float[Array, "r c"],
    filled: Int[Array, ""],
    columns: bool = True,
    most: Tiles = QUANTIZE,
) -> tuple[Blocks, Blocks | None]:
    """Return x and, if `columns`, x^T in MXFP8, as `blockscaled.mx_twice` does, x's
    rows from `filled` on as 0: one pass over x, a tile a program, the largest of 32
    times a power of two up to `most`'s that x's sides are whole tiles of."""
    r, c = x.shape
    tiles = most._replace(m=_tile(r, most.m), n=_tile(c, most.n))
    shapes = [((r, c), jnp.uint8), ((r, c // 32), jnp.uint8)]
    shapes += [((c, r), jnp.uint8), ((c, r // 32), jnp.uint8)] if columns else []
    values, scales, *transposed = jax_triton.triton_call(
        x,
        filled.astype(jnp.int32).reshape(1),
        kernel=_quantize_kernel if columns else _quantize_rows_kernel,
        out_type=[jax.ShapeDtypeStruct(*s) for s in shapes],
        grid=(r // tiles.m, c // tiles.n),
        name="bux_mx_twice" if columns else "bux_mx_rows",
        num_warps=tiles.warps,
        num_stages=tiles.stages,
        **({"r": r} if columns else {}),
        c=c,
        tile_r=tiles.m,
        tile_c=tiles.n,
    )
    return _blocks(values, scales), _blocks(*transposed) if columns else None


def _tile(n: int, most: int) -> int:
    """Return the largest tile, 32 times a power of two up to `most`, that n is whole
    tiles of."""
    tile = most
    while n % tile:
        tile //= 2
    assert tile >= 32, f"{n} is not whole blocks of 32"
    return tile


def _blocks(values: UInt8[Array, "r c"], exponents: UInt8[Array, "r n"]) -> Blocks:
    as_e4m3 = jax.lax.bitcast_convert_type(values, jnp.float8_e4m3fn)
    as_e8m0 = jax.lax.bitcast_convert_type(exponents, jnp.float8_e8m0fnu)
    return Blocks(as_e4m3, as_e8m0, None)


# Triton's kernels: what they take is a tl.tensor or a pointer, or a constant; and
# they may name no global but Triton's.


@triton.jit
def _in_e4m3(blocks: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    """Return blocks (a, b, 32) of float32 in E4M3, each times the least power of two
    that takes its largest value to at most 448, and those powers' biased exponents
    (E8M0's bits), as ``blockscaled._in_e4m3`` computes them."""
    over = tl.max(tl.abs(blocks), axis=2) / 448.0
    bits = over.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23).to(tl.int32) + ((bits & 0x7FFFFF) != 0).to(tl.int32) - 127
    exponent = tl.minimum(tl.maximum(exponent, -126), 126)
    down = ((127 - exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    values = tl.clamp(blocks * down[:, :, None], -448.0, 448.0).to(tl.float8e4nv)
    return values.to(tl.uint8, bitcast=True), (exponent + 127).to(tl.uint8)


@triton.jit
def _quantize_kernel(
    x: tl.tensor,
    rows: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    t_values: tl.tensor,
    t_scales: tl.tensor,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    tile_c: tl.constexpr,
) -> None:
    """A tile of x in E4M3 in blocks of 32 along its rows, and of x^T along its
    columns, x's rows from `rows` on as 0."""
    tile, rs, cs = _tile_of(x, rows, c, tile_r, tile_c)
    _store_rows(tile, values, scales, rs, cs, c, tile_r, tile_c)
    v, e = _in_e4m3(tl.trans(tile).reshape(tile_c, tile_r // 32, 32))
    tl.store(t_values + cs[:, None] * r + rs[None, :], v.reshape(tile_c, tile_r))
    blocks = tl.program_id(0) * (tile_r // 32) + tl.arange(0, tile_r // 32)
    tl.store(t_scales + cs[:, None] * (r // 32) + blocks[None, :], e)


@triton.jit
def _quantize_rows_kernel(
    x: tl.tensor,
    rows: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    tile_c: tl.constexpr,
) -> None:
    """A tile of x in E4M3 in blocks of 32 along its rows, x's rows from `rows` on as
    0."""
    tile, rs, cs = _tile_of(x, rows, c, tile_r, tile_c)
    _store_rows(tile, values, scales, rs, cs, c, tile_r, tile_c)


@triton.jit
def _tile_of(
    x: tl.tensor,
    rows: tl.tensor,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    tile_c: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    """Return this program's tile of x in float32, its rows from `rows` on as 0, and
    the tile's rows and columns."""
    rs = (tl.program_id(0) * tile_r + tl.arange(0, tile_r)).to(tl.int64)
    cs = (tl.program_id(1) * tile_c + tl.arange(0, tile_c)).to(tl.int64)
    tile = tl.load(x + rs[:, None] * c + cs[None, :]).to(tl.float32)
    return tl.where((rs < tl.load(rows))[:, None], tile, 0.0), rs, cs


@triton.jit
def _store_rows(
    tile: tl.tensor,
    values: tl.tensor,
    scales: tl.tensor,
    rs: tl.tensor,
    cs: tl.tensor,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    tile_c: tl.constexpr,
) -> None:
    """Store the tile in E4M3 in blocks of 32 along its rows, and their scales."""
    v, e = _in_e4m3(tile.reshape(tile_r, tile_c // 32, 32))
    tl.store(values + rs[:, None] * c + cs[None, :], v.reshape(tile_r, tile_c))
    blocks = tl.program_id(1) * (tile_c // 32) + tl.arange(0, tile_c // 32)
    tl.store(scales + rs[:, None] * (c // 32) + blocks[None, :], e)


assert MX_BLOCK == 32  # the kernels' scales, a 32nd of their values' offsets
