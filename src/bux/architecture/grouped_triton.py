"""Triton's kernels of ``grouped_blockscaled``: grouped GEMMs of E4M3 values with a
power-of-two scale per 32 of what they contract, by `tl.dot_scaled`, which Triton
lowers to Blackwell's block-scaled MMA.

`rows` multiplies each group's rows by its matrix: a program per tile of rows that a
group has rows in (a tile two groups share runs once for each) and per tile of
columns, a row tile's column tiles one after another, so that its rows stay in L2.
`contract` sums the products of each group's rows of two matrices, dW: a program per
tile of each group's output.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax_triton
import triton
import triton.language as tl
from jax._src.interpreters import partial_eval
from jax.extend.core import JaxprEqn
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Shaped, UInt8

from bux.architecture.blockscaled import MX_BLOCK, Blocks


def _keeps_its_inputs(
    used: list[bool], live: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn]:
    """Dead-code elimination's rule of a Triton call: keep it whole. jax-triton 0.4.1
    registers one of jax 0.11's signature, which jax 0.12's calls with one more."""
    return [True] * len(eqn.invars), eqn


partial_eval.dce_rules[jax_triton.triton_lib.triton_kernel_call_p] = _keeps_its_inputs


class Tiles(NamedTuple):
    """A kernel's tiles, of rows, columns and what it contracts, and how it runs."""

    m: int
    n: int
    k: int
    warps: int
    stages: int


ROWS = Tiles(m=128, n=128, k=128, warps=4, stages=4)
CONTRACT = Tiles(m=128, n=128, k=128, warps=4, stages=4)


def rows(
    a: Blocks,
    b: Blocks,
    at: Int[Array, "t 3"],
    groups: int,
    stacked: str,
    out: DTypeLike,
    tiles: Tiles = ROWS,
) -> Float[Array, "m n"]:
    """Return a b[g]^T for each tile of rows `at` (``grouped_blockscaled.tiles_of``,
    of tiles.m rows) of group g, of b's (n, k) matrices stacked by rows, (g n, k), or
    by columns, (n, g k); the rows in no tile as they were."""
    m, k = a.values.shape
    if stacked == "rows":
        n, group, row = b.values.shape[0] // groups, b.values.shape[0] // groups * k, k
    else:
        n, group, row = b.values.shape[0], k, groups * k
    return jax_triton.triton_call(
        *map(_bytes, (a.values, a.scales, b.values, b.scales)),
        at,
        kernel=_rows,
        out_type=jax.ShapeDtypeStruct((m, n), out),
        grid=(-(-n // tiles.n), at.shape[0]),
        name="bux_grouped_rows",
        num_warps=tiles.warps,
        num_stages=tiles.stages,
        m=m,
        k=k,
        n=n,
        group=group,
        row=row,
        tile_m=tiles.m,
        tile_n=tiles.n,
        tile_k=tiles.k,
    )


def contract(
    a: Blocks,
    b: Blocks,
    sizes: Int[Array, " g"],
    out: DTypeLike,
    tiles: Tiles = CONTRACT,
) -> Float[Array, "g d f"]:
    """Return a[:, rows of g] b[:, rows of g]^T for each group g: of a (d, m) and
    b (f, m), each in blocks along the rows m."""
    (d, m), f, g = a.values.shape, b.values.shape[0], sizes.shape[0]
    ends = jnp.cumsum(sizes)
    bounds = jnp.stack([ends - sizes, ends], -1).astype(jnp.int32)
    return jax_triton.triton_call(
        *map(_bytes, (a.values, a.scales, b.values, b.scales)),
        bounds,
        kernel=_contract,
        out_type=jax.ShapeDtypeStruct((g, d, f), out),
        grid=(-(-d // tiles.m), -(-f // tiles.n), g),
        name="bux_grouped_contract",
        num_warps=tiles.warps,
        num_stages=tiles.stages,
        m=m,
        d=d,
        f=f,
        tile_m=tiles.m,
        tile_n=tiles.n,
        tile_k=tiles.k,
    )


QUANTIZE = Tiles(m=64, n=128, k=0, warps=4, stages=1)  # the most of each


def quantized_twice(
    x: Float[Array, "r c"], rows: Int[Array, ""], most: Tiles = QUANTIZE
) -> tuple[Blocks, Blocks]:
    """Return x and x^T in MXFP8, as `blockscaled.mx_twice` does, x's rows from
    `rows` on as 0: one pass over x, a tile a program, the largest of 32 times a
    power of two up to `most`'s that x's sides are whole tiles of."""
    r, c = x.shape
    tiles = most._replace(m=_tile(r, most.m), n=_tile(c, most.n))
    shapes = [((r, c), jnp.uint8), ((r, c // 32), jnp.uint8)]
    shapes += [((c, r), jnp.uint8), ((c, r // 32), jnp.uint8)]
    values, scales, t_values, t_scales = jax_triton.triton_call(
        x,
        rows.astype(jnp.int32).reshape(1),
        kernel=_quantize,
        out_type=[jax.ShapeDtypeStruct(*s) for s in shapes],
        grid=(r // tiles.m, c // tiles.n),
        name="bux_mx_twice",
        num_warps=tiles.warps,
        num_stages=tiles.stages,
        r=r,
        c=c,
        tile_r=tiles.m,
        tile_c=tiles.n,
    )
    return _blocks(values, scales), _blocks(t_values, t_scales)


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


def _bytes(x: Shaped[Array, "*s"]) -> UInt8[Array, "*s"]:
    """Return x's bytes: E4M3 values or E8M0 scales, which jax-triton passes so."""
    return jax.lax.bitcast_convert_type(x, jnp.uint8)


@triton.jit
def _in_e4m3(blocks: Any) -> tuple[Any, Any]:
    """Return blocks (a, b, 32) of float32 in E4M3, each times the least power of two
    that takes its largest value to at most 448, and those powers' biased exponents
    (E8M0's bits), as ``blockscaled._mx_tile`` computes them."""
    over = tl.max(tl.abs(blocks), axis=2) / 448.0
    bits = over.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23).to(tl.int32) + ((bits & 0x7FFFFF) != 0).to(tl.int32) - 127
    exponent = tl.minimum(tl.maximum(exponent, -126), 126)
    down = ((127 - exponent).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    values = tl.clamp(blocks * down[:, :, None], -448.0, 448.0).to(tl.float8e4nv)
    return values.to(tl.uint8, bitcast=True), (exponent + 127).to(tl.uint8)


@triton.jit
def _quantize(
    x: Any,
    rows: Any,
    values: Any,
    scales: Any,
    t_values: Any,
    t_scales: Any,
    r: tl.constexpr,
    c: tl.constexpr,
    tile_r: tl.constexpr,
    tile_c: tl.constexpr,
) -> None:
    """A tile of x in E4M3 in blocks of 32 along its rows, and of x^T along its
    columns, x's rows from `rows` on as 0."""
    i, j = tl.program_id(0), tl.program_id(1)
    rs = (i * tile_r + tl.arange(0, tile_r)).to(tl.int64)
    cs = (j * tile_c + tl.arange(0, tile_c)).to(tl.int64)
    tile = tl.load(x + rs[:, None] * c + cs[None, :]).to(tl.float32)
    tile = tl.where((rs < tl.load(rows))[:, None], tile, 0.0)
    v, e = _in_e4m3(tile.reshape(tile_r, tile_c // 32, 32))
    tl.store(values + rs[:, None] * c + cs[None, :], v.reshape(tile_r, tile_c))
    blocks_c = j * (tile_c // 32) + tl.arange(0, tile_c // 32)
    tl.store(scales + rs[:, None] * (c // 32) + blocks_c[None, :], e)
    v, e = _in_e4m3(tl.trans(tile).reshape(tile_c, tile_r // 32, 32))
    tl.store(t_values + cs[:, None] * r + rs[None, :], v.reshape(tile_c, tile_r))
    blocks_r = i * (tile_r // 32) + tl.arange(0, tile_r // 32)
    tl.store(t_scales + cs[:, None] * (r // 32) + blocks_r[None, :], e)


# Triton's pointers, which index and load as arrays do not, and its constants.
@triton.jit
def _rows(
    a: Any,
    a_scales: Any,
    b: Any,
    b_scales: Any,
    at: Any,
    out: Any,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    group: tl.constexpr,
    row: tl.constexpr,
    tile_m: tl.constexpr,
    tile_n: tl.constexpr,
    tile_k: tl.constexpr,
) -> None:
    """out[r] = a[r] b[g]^T for the rows r in [lo, hi) of tile `at`, of group g, and
    tile_n of the columns: a (m, k), b[g] (n, k) at b + g group, a row every `row`;
    a scale per 32 along k, at a 32nd of the same offsets."""
    column, tile = tl.program_id(0), tl.program_id(1)
    g = tl.load(at + 3 * tile).to(tl.int64)
    lo = tl.load(at + 3 * tile + 1)
    hi = tl.load(at + 3 * tile + 2)
    if lo >= hi:
        return
    rs = (lo // tile_m * tile_m + tl.arange(0, tile_m)).to(tl.int64)
    cs = (column * tile_n + tl.arange(0, tile_n)).to(tl.int64)
    across, blocks = tl.arange(0, tile_k), tl.arange(0, tile_k // 32)
    r_in, c_in = (rs < m)[:, None], (cs < n)[:, None]
    a_rows, a_scale_rows = a + rs[:, None] * k, a_scales + rs[:, None] * (k // 32)
    b_rows = b + g * group + cs[:, None] * row
    b_scale_rows = b_scales + g * (group // 32) + cs[:, None] * (row // 32)
    acc = tl.zeros((tile_m, tile_n), tl.float32)
    for start in range(0, k, tile_k):
        ks, ss = start + across, start // 32 + blocks
        k_in, s_in = (ks < k)[None, :], (ss < k // 32)[None, :]
        x = tl.load(a_rows + ks[None, :], mask=r_in & k_in, other=0)
        x_scales = tl.load(a_scale_rows + ss[None, :], mask=r_in & s_in, other=0)
        w = tl.load(b_rows + ks[None, :], mask=c_in & k_in, other=0)
        w_scales = tl.load(b_scale_rows + ss[None, :], mask=c_in & s_in, other=0)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w.T, w_scales, "e4m3", acc=acc)
    mine = ((rs >= lo) & (rs < hi))[:, None] & (cs < n)[None, :]
    y = acc.to(out.dtype.element_ty)
    tl.store(out + rs[:, None] * n + cs[None, :], y, mask=mine)


@triton.jit
def _contract(
    a: Any,
    a_scales: Any,
    b: Any,
    b_scales: Any,
    bounds: Any,
    out: Any,
    m: tl.constexpr,
    d: tl.constexpr,
    f: tl.constexpr,
    tile_m: tl.constexpr,
    tile_n: tl.constexpr,
    tile_k: tl.constexpr,
) -> None:
    """out[g] = a[:, lo:hi] b[:, lo:hi]^T for group g's rows [lo, hi), a tile of it:
    a (d, m) and b (f, m), a scale per 32 along m; the rows of a block outside the
    group count as 0."""
    i, j, g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    lo = tl.load(bounds + 2 * g)
    hi = tl.load(bounds + 2 * g + 1)
    ds = (i * tile_m + tl.arange(0, tile_m)).to(tl.int64)
    fs = (j * tile_n + tl.arange(0, tile_n)).to(tl.int64)
    across, blocks = tl.arange(0, tile_k), tl.arange(0, tile_k // 32)
    d_in, f_in = (ds < d)[:, None], (fs < f)[:, None]
    acc = tl.zeros((tile_m, tile_n), tl.float32)
    for start in range(lo // tile_k * tile_k, hi, tile_k):
        ks, ss = start + across, start // 32 + blocks
        k_in, s_in = ((ks >= lo) & (ks < hi))[None, :], (ss < m // 32)[None, :]
        x = tl.load(a + ds[:, None] * m + ks[None, :], mask=d_in & k_in, other=0)
        x_scales = tl.load(
            a_scales + ds[:, None] * (m // 32) + ss[None, :], mask=d_in & s_in, other=0
        )
        y = tl.load(b + fs[:, None] * m + ks[None, :], mask=f_in & k_in, other=0)
        y_scales = tl.load(
            b_scales + fs[:, None] * (m // 32) + ss[None, :], mask=f_in & s_in, other=0
        )
        acc = tl.dot_scaled(x, x_scales, "e4m3", y.T, y_scales, "e4m3", acc=acc)
    at = out + g.to(tl.int64) * d * f + ds[:, None] * f + fs[None, :]
    tl.store(at, acc.to(out.dtype.element_ty), mask=d_in & (fs < f)[None, :])


assert MX_BLOCK == 32  # the kernels' scales, a 32nd of their values' offsets
