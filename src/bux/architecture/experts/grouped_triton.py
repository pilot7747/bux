"""Triton's kernels of ``grouped_blockscaled``: grouped GEMMs of E4M3 values with a
power-of-two scale per 32 of what they contract, by `tl.dot_scaled`, which Triton
lowers to Blackwell's block-scaled MMA.

`rows` multiplies each group's rows by its matrix: a program per tile of rows that a
group has rows in (a tile two groups share runs once for each) and per tile of
columns, a row tile's column tiles one after another, so that its rows stay in L2.
`contract` sums the products of each group's rows of two matrices, dW: a program per
tile of each group's output. Both load their values by TMA, which Triton pipelines on
Blackwell where it does not plain loads, and their scales, rows of a few bytes, by
plain loads. `quantized_twice` quantizes x and x^T in one pass.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_triton
import triton
import triton.language as tl
from jax._src.interpreters import partial_eval
from jax.extend.core import JaxprEqn
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Shaped, UInt8

from bux.architecture.precision.blockscaled import MX_BLOCK, Blocks


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


# The fastest tried at the baseline MoE's shapes on a B300 (docs/parallelism.md).
ROWS = Tiles(m=128, n=128, k=256, warps=8, stages=2)
CONTRACT = Tiles(m=128, n=256, k=128, warps=8, stages=3)


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
    if stacked == "rows":  # group g's matrix n rows below group g - 1's
        n, below, right = b.values.shape[0] // groups, b.values.shape[0] // groups, 0
    else:  # k columns to the right
        n, below, right = b.values.shape[0], 0, k
    return jax_triton.triton_call(
        _described(a.values, tiles.m, tiles.k),
        _bytes(a.scales),
        _described(b.values, tiles.n, tiles.k),
        _bytes(b.scales),
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
        scales_row=b.scales.shape[1],
        below=below,
        right=right,
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
    values = (_bytes(a.values), _bytes(b.values))
    return jax_triton.triton_call(
        _described(values[0], tiles.m, tiles.k),
        values[0],
        _bytes(a.scales),
        _described(values[1], tiles.n, tiles.k),
        values[1],
        _bytes(b.scales),
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


def _described(
    values: Shaped[Array, "r c"], rows: int, columns: int
) -> jax_triton.TensorDescriptor:
    """Return a TMA descriptor of `values`' bytes, a (rows, columns) tile a load, and
    0 past its edges: Triton pipelines its loads, where it does not plain ones."""
    return jax_triton.TensorDescriptor.from_array(_bytes(values), (rows, columns))


QUANTIZE = Tiles(m=64, n=128, k=0, warps=4, stages=1)  # the most of each


def quantized_twice(
    x: Float[Array, "r c"],
    rows: Int[Array, ""],
    columns: bool = True,
    most: Tiles = QUANTIZE,
) -> tuple[Blocks, Blocks | None]:
    """Return x and, if `columns`, x^T in MXFP8, as `blockscaled.mx_twice` does, x's
    rows from `rows` on as 0: one pass over x, a tile a program, the largest of 32
    times a power of two up to `most`'s that x's sides are whole tiles of."""
    r, c = x.shape
    tiles = most._replace(m=_tile(r, most.m), n=_tile(c, most.n))
    shapes = [((r, c), jnp.uint8), ((r, c // 32), jnp.uint8)]
    shapes += [((c, r), jnp.uint8), ((c, r // 32), jnp.uint8)] if columns else []
    values, scales, *transposed = jax_triton.triton_call(
        x,
        rows.astype(jnp.int32).reshape(1),
        kernel=_quantize if columns else _quantize_rows,
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


def _bytes(x: Shaped[Array, "*s"]) -> UInt8[Array, "*s"]:
    """Return x's bytes: E4M3 values or E8M0 scales, which jax-triton passes so."""
    return jax.lax.bitcast_convert_type(x, jnp.uint8)


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
def _quantize(
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
def _quantize_rows(
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


# Triton's kernels: what they take is a tl.tensor, a pointer or a TMA descriptor, or
# a constant; and they may name no global but Triton's.
@triton.jit
def _rows(
    a: tl.tensor,
    a_scales: tl.tensor,
    b: tl.tensor,
    b_scales: tl.tensor,
    at: tl.tensor,
    out: tl.tensor,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
    scales_row: tl.constexpr,
    below: tl.constexpr,
    right: tl.constexpr,
    tile_m: tl.constexpr,
    tile_n: tl.constexpr,
    tile_k: tl.constexpr,
) -> None:
    """out[r] = a[r] b[g]^T for the rows r in [lo, hi) of tile `at`, of group g, and
    tile_n of the columns. a (m, k) and b by TMA, b[g] (n, k) `below` rows and `right`
    columns of b on from b[g - 1]; a scale per 32 along k, scales_row a row of b's.
    What a tile reads past b[g]'s k multiplies a's 0, and past its n is not kept."""
    column, tile = tl.program_id(0), tl.program_id(1)
    g = tl.load(at + 3 * tile)
    lo = tl.load(at + 3 * tile + 1)
    hi = tl.load(at + 3 * tile + 2)
    if lo >= hi:
        return
    first = lo // tile_m * tile_m
    rs = (first + tl.arange(0, tile_m)).to(tl.int64)
    cs = (column * tile_n + tl.arange(0, tile_n)).to(tl.int64)
    b_row, b_column = g * below + column * tile_n, g * right
    blocks = tl.arange(0, tile_k // 32)
    r_in, c_in = (rs < m)[:, None], (cs < n)[:, None]
    a_scale_rows = a_scales + rs[:, None] * (k // 32)
    b_scale_rows = b_scales + (g * below + cs[:, None]) * scales_row + g * right // 32
    acc = tl.zeros((tile_m, tile_n), tl.float32)
    for start in range(0, k, tile_k):
        ss = start // 32 + blocks
        s_in = (ss < k // 32)[None, :]
        x = a.load([first, start])
        w = b.load([b_row, b_column + start])
        x_scales = tl.load(a_scale_rows + ss[None, :], mask=r_in & s_in, other=0)
        w_scales = tl.load(b_scale_rows + ss[None, :], mask=c_in & s_in, other=0)
        acc = tl.dot_scaled(x, x_scales, "e4m3", w.T, w_scales, "e4m3", acc=acc)
    mine = ((rs >= lo) & (rs < hi))[:, None] & (cs < n)[None, :]
    y = acc.to(out.dtype.element_ty)
    tl.store(out + rs[:, None] * n + cs[None, :], y, mask=mine)


@triton.jit
def _contract(
    a: tl.tensor,
    a_values: tl.tensor,
    a_scales: tl.tensor,
    b: tl.tensor,
    b_values: tl.tensor,
    b_scales: tl.tensor,
    bounds: tl.tensor,
    out: tl.tensor,
    m: tl.constexpr,
    d: tl.constexpr,
    f: tl.constexpr,
    tile_m: tl.constexpr,
    tile_n: tl.constexpr,
    tile_k: tl.constexpr,
) -> None:
    """out[g] = a[:, lo:hi] b[:, lo:hi]^T for group g's rows [lo, hi), a tile of it:
    a (d, m) and b (f, m), a scale per 32 along m; the tiles of m wholly inside the
    group by TMA, and the one it starts in and the one it ends in by plain loads that
    take the rows outside it as 0."""
    i, j, g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    lo = tl.load(bounds + 2 * g)
    hi = tl.load(bounds + 2 * g + 1)
    ds = (i * tile_m + tl.arange(0, tile_m)).to(tl.int64)
    fs = (j * tile_n + tl.arange(0, tile_n)).to(tl.int64)
    d_in, f_in = (ds < d)[:, None], (fs < f)[:, None]
    acc = tl.zeros((tile_m, tile_n), tl.float32)
    whole_lo = (lo + tile_k - 1) // tile_k * tile_k
    whole_hi = tl.maximum(hi // tile_k * tile_k, whole_lo)
    if lo < whole_lo:
        acc = _contract_part(
            acc,
            a_values,
            a_scales,
            b_values,
            b_scales,
            ds,
            fs,
            lo // tile_k * tile_k,
            lo,
            hi,
            m,
            d,
            f,
            tile_k,
        )
    for start in range(whole_lo, whole_hi, tile_k):
        ss = start // 32 + tl.arange(0, tile_k // 32)
        x = a.load([i * tile_m, start])
        y = b.load([j * tile_n, start])
        x_at, y_at = ds[:, None] * (m // 32), fs[:, None] * (m // 32)
        x_scales = tl.load(a_scales + x_at + ss[None, :], mask=d_in, other=0)
        y_scales = tl.load(b_scales + y_at + ss[None, :], mask=f_in, other=0)
        acc = tl.dot_scaled(x, x_scales, "e4m3", y.T, y_scales, "e4m3", acc=acc)
    if whole_hi < hi:
        acc = _contract_part(
            acc,
            a_values,
            a_scales,
            b_values,
            b_scales,
            ds,
            fs,
            whole_hi,
            lo,
            hi,
            m,
            d,
            f,
            tile_k,
        )
    at = out + g.to(tl.int64) * d * f + ds[:, None] * f + fs[None, :]
    tl.store(at, acc.to(out.dtype.element_ty), mask=d_in & (fs < f)[None, :])


@triton.jit
def _contract_part(
    acc: tl.tensor,
    a: tl.tensor,
    a_scales: tl.tensor,
    b: tl.tensor,
    b_scales: tl.tensor,
    ds: tl.tensor,
    fs: tl.tensor,
    start: tl.tensor,
    lo: tl.tensor,
    hi: tl.tensor,
    m: tl.constexpr,
    d: tl.constexpr,
    f: tl.constexpr,
    tile_k: tl.constexpr,
) -> tl.tensor:
    """Return acc plus the tile of m from `start` of a b^T, its rows outside
    [lo, hi) as 0: a tile that a group shares with another."""
    ks = start + tl.arange(0, tile_k)
    ss = start // 32 + tl.arange(0, tile_k // 32)
    d_in, f_in = (ds < d)[:, None], (fs < f)[:, None]
    k_in, s_in = ((ks >= lo) & (ks < hi))[None, :], (ss < m // 32)[None, :]
    x = tl.load(a + ds[:, None] * m + ks[None, :], mask=d_in & k_in, other=0)
    x_scales = tl.load(
        a_scales + ds[:, None] * (m // 32) + ss[None, :], mask=d_in & s_in, other=0
    )
    y = tl.load(b + fs[:, None] * m + ks[None, :], mask=f_in & k_in, other=0)
    y_scales = tl.load(
        b_scales + fs[:, None] * (m // 32) + ss[None, :], mask=f_in & s_in, other=0
    )
    return tl.dot_scaled(x, x_scales, "e4m3", y.T, y_scales, "e4m3", acc=acc)


assert MX_BLOCK == 32  # the kernels' scales, a 32nd of their values' offsets
