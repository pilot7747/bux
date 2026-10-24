"""Grouped MXFP8 GEMMs for an MoE's experts on Blackwell, by Mosaic GPU: E4M3 values
with an E8M0 scale per 32 of what they contract, multiplied by tcgen05's block-scaled
MMA, the scales brought by TMA into shared memory as tensor memory lays them out, and
copied on into it by `tcgen05.cp`.

`rows` multiplies each group's rows by its matrix, x W[g] and dy W[g]^T; `contract`
sums the products of each group's rows of two matrices, dW[g] = x^T dy over them.
Both run a program on each SM, which takes tiles of the output one after another:

    rows      128 rows that a group has rows in, by `Tiles.n` columns: a tile two
              groups share runs once for each, which stores its own rows
    contract  128 rows of dW[g] by `Tiles.n` columns, over the group's rows in tiles
              of 128: the first and last, which it may share, from copies of them
              with the other groups' rows as 0

Three roles share each tile's steps along what it contracts (k):

    warp 0 (load):     TMA a, b and their scales into a ring of `stages` buffers
    warp 1 (multiply): copy the scales into tensor memory, MMA into the accumulator
    warpgroup 1:       the accumulator into the output's dtype, and to memory by TMA

The accumulator has two buffers in tensor memory, so the store of one tile overlaps
the MMAs of the next.
"""

import functools
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, Shaped

from bux import kernels
from bux.architecture.precision.blockscaled import MX_BLOCK, Blocks

E8M0 = jnp.float8_e8m0fnu
LANES = 128  # tensor memory's rows: a tile's, and a tile of scales'
SCALE_TILE = 4  # scales to a row of a tile of scales: 128 values of k
STORE = 128  # bytes of a row the store takes at a time: the widest swizzle


class Tiles(NamedTuple):
    """A kernel's tiles: of rows on each SM, of columns, and of what it contracts
    (k); how many steps along k its ring of buffers holds; and on how many SMs a
    tile runs. On 2, tcgen05's MMA of a pair of SMs: each holds its m rows of the
    tile and half of b's n, and multiplies them by the other's half too."""

    m: int
    n: int
    k: int
    stages: int
    sms: int = 1

    @property
    def height(self) -> int:
        """The rows of a tile, over its SMs."""
        return self.m * self.sms


ROWS = Tiles(m=LANES, n=128, k=128, stages=5)
CONTRACT = Tiles(m=LANES, n=128, k=LANES, stages=5)  # k: a tile of rows

# How each group's (n, k) matrix stacks: by rows, (g n, k); by columns, (n, g k).
Stacked = Literal["rows", "columns"]


def runs() -> bool:
    """Whether these kernels run here: on a Blackwell GPU (sm_100 on), outside
    ``kernels.off``."""
    if not kernels.gpu():
        return False
    return float(getattr(jax.devices()[0], "compute_capability", 0)) >= 10


# --- The operands, as the kernels load them ---------------------------------------


def rows(
    a: Blocks,
    b: Blocks,
    sizes: Int[Array, " g"],
    stacked: Stacked,
    out: DTypeLike,
    tiles: Tiles = ROWS,
) -> Float[Array, "m n"]:
    """Return a b[g]^T for each group g's rows of a, the first sizes[0] rows group
    0's and so on, of b's (n, k) matrices `stacked`; the rows past the groups as 0."""
    (m, k), groups = a.values.shape, sizes.shape[0]
    matrices = _matrices(b.values, groups, stacked)
    n = matrices.shape[1 if stacked == "rows" else 0]
    values, matrices = _whole_tiles_of_k(a.values), _whole_tiles_of_k(matrices)
    scales = (in_tensor_memory(a.scales), _matrix_scales(b.scales, groups, stacked))
    past = jnp.maximum(m - sizes.sum(), 0)[None].astype(sizes.dtype)
    at = tiles_of(jnp.concatenate([sizes, past]), m, tiles.height)
    kernel = _rows_kernel(m, n, k, at.shape[0], groups, stacked, jnp.dtype(out), tiles)
    return kernel(values, matrices, *scales, at[:, 0], at[:, 1], at[:, 2])


def tiles_of(sizes: Int[Array, " g"], m: int, tile: int) -> Int[Array, "t 3"]:
    """Return, for each tile of `tile` rows of m that a group has rows in, the group
    and where its rows there start and end; then (0, 0, 0), as many as make the most
    there can be: one a tile of m, and one more a group but the first."""
    g = sizes.shape[0]
    ends = jnp.cumsum(sizes)
    starts = ends - sizes
    first = starts // tile
    count = jnp.where(sizes > 0, -(-ends // tile) - first, 0)
    before = jnp.cumsum(count) - count
    at = jnp.arange(-(-m // tile) + g - 1)
    group = jnp.minimum(jnp.searchsorted(before + count, at, side="right"), g - 1)
    index = first[group] + at - before[group]
    lo = jnp.maximum(starts[group], index * tile)
    hi = jnp.minimum(ends[group], (index + 1) * tile)
    real = at < count.sum()
    spans = [group, jnp.where(real, lo, 0), jnp.where(real, hi, 0)]
    return jnp.stack(spans, -1).astype(jnp.int32)


def contract(
    a: Blocks,
    b: Blocks,
    sizes: Int[Array, " g"],
    out: DTypeLike,
    tiles: Tiles = CONTRACT,
) -> Float[Array, "g d f"]:
    """Return a[:, rows of g] b[:, rows of g]^T for each group g: of a (d, m) and
    b (f, m), each in blocks along the rows m, a whole number of tiles of them."""
    (d, m), f, g = a.values.shape, b.values.shape[0], sizes.shape[0]
    assert m % tiles.k == 0, f"{m} rows are not whole tiles of {tiles.k}"
    ends = jnp.cumsum(sizes).astype(jnp.int32)
    starts = ends - sizes.astype(jnp.int32)
    edges = (_edges(a.values, starts, ends), _edges(b.values, starts, ends))
    scales = (in_tensor_memory(a.scales), in_tensor_memory(b.scales))
    kernel = _contract_kernel(d, f, g, jnp.dtype(out), tiles)
    return kernel(a.values, b.values, *edges, *scales, starts, ends)


def in_tensor_memory(scales: Shaped[Array, "r c"]) -> Shaped[Array, "p q 32 16"]:
    """Return (r, c) scales, a row's per 32 of k, laid out as tcgen05's MMA takes
    them in tensor memory, so that TMA loads a tile's at a time: a 512-byte tile for
    each 128 rows by 4 scales (128 of k), row i's 4 at bytes (i % 32) * 16 +
    (i // 32) * 4. Padded with 0 to whole tiles: 2^-127 times values of 0."""
    r, c = scales.shape
    p, q = -(-r // LANES), -(-c // SCALE_TILE)
    padded = jnp.pad(scales, ((0, p * LANES - r), (0, q * SCALE_TILE - c)))
    tiles = padded.reshape(p, 4, 32, q, SCALE_TILE).transpose(0, 3, 2, 1, 4)
    return tiles.reshape(p, q, 32, 16)


def _matrices(
    values: Shaped[Array, "r c"], groups: int, stacked: Stacked
) -> Shaped[Array, "..."]:
    """Return the stacked matrices' values as a view TMA loads a group's from, each
    tile of k as long as the group's own and 0 past it: (g, n, k) by rows, (n, g, k)
    by columns."""
    r, c = values.shape
    if stacked == "rows":
        return values.reshape(groups, r // groups, c)
    return values.reshape(r, groups, c // groups)


def _whole_tiles_of_k(values: Shaped[Array, "*r k"]) -> Shaped[Array, "*r q"]:
    """Return the values with k padded by 0 to whole tiles of 128, which TMA loads
    in swizzled rows of 128 bytes: 0 times whatever the padded scales say."""
    pad = -values.shape[-1] % LANES
    return jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)]) if pad else values


def _matrix_scales(
    scales: Shaped[Array, "r c"], groups: int, stacked: Stacked
) -> Shaped[Array, "g p q 32 16"]:
    """Return each group's scales as tensor memory takes them (`in_tensor_memory`)."""
    r, c = scales.shape
    if stacked == "rows":
        each = scales.reshape(groups, r // groups, c)
    else:
        each = scales.reshape(r, groups, c // groups).transpose(1, 0, 2)
    return jax.vmap(in_tensor_memory)(each)


def _edges(
    values: Shaped[Array, "r m"], starts: Int[Array, " g"], ends: Int[Array, " g"]
) -> Shaped[Array, "g 2 r 128"]:
    """Return, for each group, the tiles of 128 of m its rows [start, end) begin
    and end in, the rows outside them as 0."""

    def edge(at: Int[Array, ""], start: Int[Array, ""], end: Int[Array, ""]) -> Any:
        first = at // LANES * LANES
        tile = lax.dynamic_slice_in_dim(values, first, LANES, axis=1)
        m = first + jnp.arange(LANES)
        return jnp.where((m >= start) & (m < end), tile, jnp.zeros((), tile.dtype))

    begin = jax.vmap(edge)(starts, starts, ends)
    finish = jax.vmap(edge)(jnp.maximum(ends - 1, starts), starts, ends)
    return jnp.stack([begin, finish], 1)


# --- The kernels ---------------------------------------------------------------------


class _Place(NamedTuple):
    """Where a tile of the output is: whether it runs, its group, the group's rows
    [start, end), the tile's row and column (in tiles), and its steps along k."""

    runs: Bool[Array, ""]
    group: Int[Array, ""]
    start: Int[Array, ""]
    end: Int[Array, ""]
    row: Int[Array, ""]
    column: Int[Array, ""]
    steps: Int[Array, ""]


class _Tile(NamedTuple):
    """A tile as a program runs it: its `_Place`, how many tiles and steps the
    program ran before it, which of the tile's SMs this is, and the kernel's
    tiles."""

    runs: Bool[Array, ""]
    group: Int[Array, ""]
    start: Int[Array, ""]
    end: Int[Array, ""]
    row: Int[Array, ""]
    column: Int[Array, ""]
    steps: Int[Array, ""]
    done: Int[Array, ""]
    stepped: Int[Array, ""]
    sm: Int[Array, ""]
    tiles: Tiles

    @property
    def first(self) -> Int[Array, ""]:
        """The first of this SM's rows of the tile."""
        return self.row * self.tiles.height + self.sm * self.tiles.m

    @property
    def acc(self) -> Int[Array, ""]:
        """Which of the two accumulators this tile's MMAs go into."""
        return lax.rem(self.done, 2)

    def slot(self, i: Int[Array, ""]) -> Int[Array, ""]:
        """Which buffer of the ring step i of this tile loads into."""
        return lax.rem(self.stepped + i, self.tiles.stages)


class _Buffers(NamedTuple):
    """What a program keeps on chip: the ring of loaded tiles and scales in shared
    memory, the store's buffers, the accumulators and scales in tensor memory, and
    the barriers between the roles."""

    a: Any
    b: Any
    a_scales: Any
    b_scales: Any
    out: Any
    acc: Any
    a_scales_tm: Any
    b_scales_tm: Any
    loaded: Any
    consumed: Any
    multiplied: Any
    stored: Any


# A step's copies into the ring (tile, step, slot, buffers) and a tile's store.
Load = Callable[[_Tile, Int[Array, ""], Int[Array, ""], _Buffers], None]
Store = Callable[[_Tile, _Buffers], None]


@functools.cache
def _rows_kernel(
    m: int,
    n: int,
    k: int,
    count: int,
    groups: int,
    stacked: Stacked,
    out: jnp.dtype,
    tiles: Tiles,
) -> Any:
    """Return the kernel of `rows` for these shapes: a jax function of a's values,
    b's, their scales, and each row tile's group, start and end; a tile of the rows
    past the groups, of group `groups`, stores 0."""
    columns = -(-n // tiles.n)

    def body(*refs: Any) -> None:
        a, b, a_scales, b_scales, group_of, starts, ends, y, *scratch = refs

        def place(index: Int[Array, ""]) -> _Place:
            t, column = plgpu.planar_snake(index, (count, columns), 1, 1)
            group, start, end = group_of[t], starts[t], ends[t]
            steps = jnp.where(group < groups, -(-k // tiles.k), 0)
            row = start // tiles.height
            return _Place(start < end, group, start, end, row, column, steps)

        load = functools.partial(
            _load_rows, a=a, b=b, a_scales=a_scales, b_scales=b_scales, stacked=stacked
        )
        store = functools.partial(_store_rows, y=y)
        _each_tile(place, load, store, _Buffers(*scratch), tiles)

    name = f"bux_mxfp8_rows_{stacked}"
    return _launch(body, (m, n), out, count * columns, tiles, name)


@functools.cache
def _contract_kernel(d: int, f: int, groups: int, out: jnp.dtype, tiles: Tiles) -> Any:
    """Return the kernel of `contract` for these shapes: a jax function of a's and
    b's values, their edges, their scales, and where each group's rows start and
    end."""
    row_tiles, columns = -(-d // tiles.height), -(-f // tiles.n)

    def body(*refs: Any) -> None:
        a, b, a_edges, b_edges, a_scales, b_scales, starts, ends, y, *scratch = refs

        def place(index: Int[Array, ""]) -> _Place:
            group = lax.div(index, row_tiles * columns)
            within = lax.rem(index, row_tiles * columns)
            row, column = plgpu.planar_snake(within, (row_tiles, columns), 1, 1)
            start, end = starts[group], ends[group]
            steps = jnp.where(start < end, (end - 1) // LANES - start // LANES + 1, 0)
            return _Place(jnp.bool_(True), group, start, end, row, column, steps)

        load = functools.partial(
            _load_contract,
            a=a,
            b=b,
            a_edges=a_edges,
            b_edges=b_edges,
            a_scales=a_scales,
            b_scales=b_scales,
        )
        store = functools.partial(_store_tile, y=y)
        _each_tile(place, load, store, _Buffers(*scratch), tiles)

    count = groups * row_tiles * columns
    return _launch(body, (groups, d, f), out, count, tiles, "bux_mxfp8_contract")


def _launch(
    body: Callable[..., None],
    shape: tuple[int, ...],
    out: jnp.dtype,
    count: int,
    tiles: Tiles,
    name: str,
) -> Any:
    """Return `body` as a kernel of `count` tiles over two warpgroups on each of
    `tiles.sms` SMs, with the buffers of `tiles` (`_buffers`)."""
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct(shape, out),
        scratch_types=tuple(_buffers(tiles, shape[-1], out)),
        grid=(count,),
        grid_names=("tiles",),
        cluster=(tiles.sms,),
        cluster_names=("pair",),
        num_threads=2,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(unsafe_no_auto_barriers=True),
        kernel_name=name,
    )


def _buffers(tiles: Tiles, n: int, out: jnp.dtype) -> _Buffers:
    """Return the buffers of a kernel of `tiles` with n columns of `out`."""
    e4m3 = jnp.float8_e4m3fn
    s, k_tiles, n_tiles = tiles.stages, tiles.k // LANES, -(-tiles.n // LANES)
    pair = tiles.sms > 1  # tensor memory that the pair's MMA reads on both SMs
    scales = functools.partial(
        plgpu.TMEM, layout=plgpu.TMEMLayout.SCALES_LAYOUT, collective=pair
    )
    return _Buffers(
        a=_swizzled((s, tiles.m, tiles.k), e4m3),
        b=_swizzled((s, tiles.n // tiles.sms, tiles.k), e4m3),
        a_scales=plgpu.SMEM((s, tiles.m // LANES, k_tiles, 32, 16), E8M0),
        b_scales=plgpu.SMEM((s, n_tiles, k_tiles, 32, 16), E8M0),
        out=_swizzled((2, tiles.m, _store_width(n, out)), out, rows=1),
        acc=plgpu.TMEM((tiles.m, 2 * tiles.n), jnp.float32, collective=pair),
        a_scales_tm=scales((tiles.m, tiles.k // MX_BLOCK), E8M0),
        b_scales_tm=scales((n_tiles * LANES, tiles.k // MX_BLOCK), E8M0),
        loaded=plgpu.Barrier(num_arrivals=4, num_barriers=s),
        consumed=plgpu.Barrier(num_barriers=s, orders_tensor_core=True),
        multiplied=plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
        stored=_barrier_of_both(tiles.sms, num_barriers=2, orders_tensor_core=True),
    )


def _store_width(n: int, out: jnp.dtype) -> int:
    """Return how many columns of `out` the store takes at a time: STORE bytes of
    them, or half or a quarter of that, the most that n is whole of."""
    width = STORE // out.itemsize
    while n % width:
        width //= 2
    return width


def _barrier_of_both(sms: int, **kwargs: Any) -> Any:
    """Return a barrier that the stores of each of a tile's SMs arrive at."""
    if sms == 1:
        return plgpu.Barrier(**kwargs)
    return plgpu.ClusterBarrier(collective_axes=("pair",), **kwargs)


def _swizzled(shape: tuple[int, ...], dtype: DTypeLike, rows: int = 8) -> Any:
    """Return shared memory for tiles of `shape`, in tiles of `rows` by up to 128
    bytes, swizzled as TMA and the MMA read them."""
    bits = jnp.dtype(dtype).itemsize * 8
    swizzle = plgpu.find_swizzle(shape[-1] * bits)
    # Concrete transforms, which ty takes for their abstract Protocol.
    tiling = plgpu.TilingTransform((rows, swizzle * 8 // bits))  # ty: ignore
    swizzled = plgpu.SwizzleTransform(swizzle)  # ty: ignore
    return plgpu.SMEM(shape, dtype, transforms=(tiling, swizzled))


# --- The roles -----------------------------------------------------------------------


def _each_tile(
    place: Callable[[Int[Array, ""]], _Place],
    load: Load,
    store: Store,
    buffers: _Buffers,
    tiles: Tiles,
) -> None:
    """Run the three roles on each tile this program takes, `place` saying where the
    i-th is: warpgroup 0's warps 0 and 1 load and multiply (on the first of the
    tile's SMs for both), warpgroup 1 stores."""
    start = (jnp.int32(0), jnp.int32(0))  # tiles and steps done
    sm = lax.axis_index("pair")

    @plgpu.dynamic_scheduling_loop(
        ("tiles",), thread_axis="wg", cluster_axes=("pair",), init_carry=start
    )
    def _(loop: plgpu.NDLoopInfo, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        tile = _Tile(*place(loop.index[0]), *carry, sm, tiles)
        wg = lax.axis_index("wg")

        @pl.when(tile.runs & (wg == 0))
        def _() -> None:
            @plgpu.warp_map
            def _(warp: Int[Array, ""]) -> None:
                pl.when(warp == 0)(lambda: _load(tile, load, buffers))
                pl.when((warp == 1) & (sm == 0))(lambda: _multiply(tile, buffers))

        pl.when(tile.runs & (wg == 1))(lambda: store(tile, buffers))
        ran = tile.runs.astype(jnp.int32)
        return tile.done + ran, tile.stepped + ran * tile.steps


def _load(tile: _Tile, load: Load, buffers: _Buffers) -> None:
    """For each step along k, once the MMA of the step that last used its buffer
    in the ring has read it, load the step's tiles and scales into it."""

    @pl.loop(0, tile.steps)
    def _(i: Int[Array, ""]) -> None:
        slot = tile.slot(i)

        @pl.when(tile.stepped + i >= tile.tiles.stages)
        @jax.named_scope("wait for a free buffer")
        def _() -> None:
            plgpu.barrier_wait(buffers.consumed.at[slot])

        with jax.named_scope("load"):
            load(tile, i, slot, buffers)


def _load_rows(
    tile: _Tile,
    i: Int[Array, ""],
    slot: Int[Array, ""],
    buffers: _Buffers,
    a: Any,
    b: Any,
    a_scales: Any,
    b_scales: Any,
    stacked: Stacked,
) -> None:
    """Copy step i of a tile of `rows`: its rows of a, its group's matrix's tile of
    columns, and their scales, each a tile of k."""
    t = tile.tiles
    rows, columns, row_tiles, column_tiles = _spans(tile)
    ks, k_tiles = pl.ds(i * t.k, t.k), pl.ds(i * (t.k // LANES), t.k // LANES)
    if stacked == "rows":
        matrix = b.at[tile.group, columns, ks]
    else:
        matrix = b.at[columns, tile.group, ks]
    _copy(a.at[rows, ks], buffers.a, slot, buffers, t)
    _copy(matrix, buffers.b, slot, buffers, t)
    _copy(a_scales.at[row_tiles, k_tiles], buffers.a_scales, slot, buffers, t)
    _copy(
        b_scales.at[tile.group, column_tiles, k_tiles],
        buffers.b_scales,
        slot,
        buffers,
        t,
        whole=True,
    )


def _load_contract(
    tile: _Tile,
    i: Int[Array, ""],
    slot: Int[Array, ""],
    buffers: _Buffers,
    a: Any,
    b: Any,
    a_edges: Any,
    b_edges: Any,
    a_scales: Any,
    b_scales: Any,
) -> None:
    """Copy step i of a tile of `contract`: its rows of a and b in the group's
    i-th tile of m (the first and last from their edges, the other groups' rows
    as 0), and their scales."""
    t = tile.tiles
    rows, columns, row_tiles, column_tiles = _spans(tile)
    k_tile = tile.start // LANES + i
    inside = (i > 0) & (i < tile.steps - 1)
    which = jnp.where(i == 0, 0, 1)

    @pl.when(inside)
    def _() -> None:
        ks = pl.ds(k_tile * LANES, LANES)
        _copy(a.at[rows, ks], buffers.a, slot, buffers, t)
        _copy(b.at[columns, ks], buffers.b, slot, buffers, t)

    @pl.when((i == 0) | (i == tile.steps - 1))
    def _() -> None:
        _copy(a_edges.at[tile.group, which, rows], buffers.a, slot, buffers, t)
        _copy(b_edges.at[tile.group, which, columns], buffers.b, slot, buffers, t)

    k_tiles = pl.ds(k_tile, 1)
    _copy(a_scales.at[row_tiles, k_tiles], buffers.a_scales, slot, buffers, t)
    _copy(
        b_scales.at[column_tiles, k_tiles],
        buffers.b_scales,
        slot,
        buffers,
        t,
        whole=True,
    )


def _spans(tile: _Tile) -> tuple[pl.Slice, pl.Slice, pl.Slice, pl.Slice]:
    """Return the tile's rows and columns, and their tiles of scales: over both of
    its SMs, whose copies split them."""
    t = tile.tiles
    rows, columns = pl.ds(tile.row * t.height, t.height), pl.ds(tile.column * t.n, t.n)
    row_tiles = pl.ds(tile.row * (t.height // LANES), t.height // LANES)
    column_tiles = pl.ds(tile.column * (t.n // LANES), -(-t.n // LANES))
    return rows, columns, row_tiles, column_tiles


def _copy(
    source: Any,
    ring: Any,
    slot: Int[Array, ""],
    buffers: _Buffers,
    tiles: Tiles,
    whole: bool = False,
) -> None:
    """Copy `source` by TMA into buffer `slot` of `ring`, counted by its barrier: on
    a pair of SMs, the first half of it to the first and the second to the second,
    or the `whole` of it to each, counted by the first's barrier."""
    if tiles.sms == 1:
        plgpu.copy_gmem_to_smem(source, ring.at[slot], buffers.loaded.at[slot])
        return
    split = (
        plgpu.CopyPartition.REPLICATED if whole else plgpu.CopyPartition.PARTITIONED(0)
    )
    plgpu.copy_gmem_to_smem(
        source,
        ring.at[slot],
        buffers.loaded.at[slot],
        collective_axes="pair",
        leader_tracked=split,
    )


def _multiply(tile: _Tile, buffers: _Buffers) -> None:
    """Once the store has emptied this tile's accumulator, for each step along k:
    copy its scales into tensor memory and MMA its tiles into the accumulator. The
    copies and MMAs of one thread run in order, so one buffer of scales serves
    every step. Then signal the store."""
    t = tile.tiles
    acc = buffers.acc.at[:, pl.ds(tile.acc * t.n, t.n)]
    pair = "pair" if t.sms > 1 else None

    @pl.when(tile.done > 1)
    @jax.named_scope("wait for the store")
    def _() -> None:
        plgpu.barrier_wait(buffers.stored.at[tile.acc])

    @pl.loop(0, tile.steps)
    @jax.named_scope("multiply")
    def _(i: Int[Array, ""]) -> None:
        slot = tile.slot(i)
        with jax.named_scope("wait for the loads"):
            plgpu.barrier_wait(buffers.loaded.at[slot])
        plgpu.async_copy_scales_to_tmem(
            buffers.a_scales.at[slot], buffers.a_scales_tm, pair
        )
        plgpu.async_copy_scales_to_tmem(
            buffers.b_scales.at[slot], buffers.b_scales_tm, pair
        )
        plgpu.tcgen05_mma(
            acc,
            buffers.a.at[slot],
            buffers.b.at[slot].T,
            buffers.consumed.at[slot],
            a_scale=buffers.a_scales_tm,
            b_scale=buffers.b_scales_tm,
            accumulate=i > 0,
            collective_axis=pair,
        )

    plgpu.tcgen05_commit_arrive(buffers.multiplied.at[tile.acc], collective_axis=pair)


def _store_rows(tile: _Tile, buffers: _Buffers, y: Any) -> None:
    """Store this SM's rows of the tile that are in [start, end) to y: as 0 in a
    tile of the rows past the groups."""
    start = jnp.maximum(tile.start, tile.first)
    count = jnp.maximum(jnp.minimum(tile.end, tile.first + tile.tiles.m) - start, 0)

    def part(staged: Any, columns: pl.Slice) -> None:
        offset, size = start - tile.first, tile.tiles.m
        while size:  # by TMA copies of 128, 64, ... 1 rows
            _store_some(staged, y, tile.first, offset, size, count & size, columns)
            offset, size = offset + (count & size), size // 2
        plgpu.commit_smem_to_gmem_group()

    _store(tile, buffers, part, zero=tile.steps == 0)


def _store_some(
    staged: Any,
    y: Any,
    first: Int[Array, ""],
    offset: Int[Array, ""],
    size: int,
    taken: Int[Array, ""],
    columns: pl.Slice,
) -> None:
    """Copy `size` rows from `offset` of the staged tile to y's from first + offset,
    if they are `taken`."""

    @pl.when(taken != 0)
    def _() -> None:
        rows = pl.ds(first + offset, size)
        plgpu.copy_smem_to_gmem(
            staged.at[pl.ds(offset, size)], y.at[rows, columns], commit_group=False
        )


def _store_tile(tile: _Tile, buffers: _Buffers, y: Any) -> None:
    """Store the tile of the accumulator to y[group], as 0 where the group has no
    rows."""

    def part(staged: Any, columns: pl.Slice) -> None:
        rows = pl.ds(tile.first, tile.tiles.m)
        plgpu.copy_smem_to_gmem(staged, y.at[tile.group, rows, columns])

    _store(tile, buffers, part, zero=tile.steps == 0)


def _store(
    tile: _Tile,
    buffers: _Buffers,
    part: Callable[[Any, pl.Slice], None],
    zero: Bool[Array, ""] | bool = False,
) -> None:
    """Once the MMAs are done: the accumulator, as many columns at a time as the
    store's buffers hold, into the output's dtype in shared memory (or 0 where
    `zero`), and `part` of it to memory. The accumulator is free for the MMAs again
    once it is read."""
    t, width = tile.tiles, buffers.out.shape[-1]
    with jax.named_scope("wait for the MMAs"):
        plgpu.barrier_wait(buffers.multiplied.at[tile.acc])
    for e in range(t.n // width):
        staged = buffers.out.at[e % 2]
        held = pl.ds(tile.acc * t.n + e * width, width)
        acc = plgpu.async_load_tmem(
            buffers.acc.at[:, held], layout=plgpu.Layout.TCGEN05
        )
        if zero is not False:
            acc = jnp.where(jnp.broadcast_to(zero, acc.shape), jnp.zeros_like(acc), acc)
        staged[...] = acc.astype(staged.dtype)
        plgpu.commit_smem()
        part(staged, pl.ds(tile.column * t.n + e * width, width))
        plgpu.wait_smem_to_gmem(1, wait_read_only=True)
    plgpu.wait_load_tmem()
    plgpu.barrier_arrive(buffers.stored.at[tile.acc])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
