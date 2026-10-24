"""Grouped MXFP8 GEMMs by Mosaic GPU (bux.architecture.experts.grouped_mxfp8): the
operands laid out as the kernels take them and each group's edge tiles, on the CPU;
the kernels lowered for a Blackwell GPU, anywhere; and on a Blackwell GPU, the
kernels against XLA's dequantize-and-multiply, with empty groups, tiny ones and rows
past them, and Triton's quantizer (grouped_triton) against XLA's layout of XLA's
quantization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int

from bux import kernels
from bux.architecture.experts import grouped
from bux.architecture.experts import grouped_mxfp8 as gm
from bux.architecture.precision import blockscaled
from bux.architecture.precision.blockscaled import Blocks

M = 1024
# Empty groups, a group of one row, groups inside a tile of 128 and across them, and
# rows past the groups.
SIZES = jnp.array([0, 130, 1, 3, 256, 0, 300, 100], jnp.int32)
E4M3, E8M0 = jnp.float8_e4m3fn, jnp.float8_e8m0fnu
blackwell = pytest.mark.skipif(
    not kernels.gpu() or "B" not in jax.devices()[0].device_kind,
    reason="tcgen05's block-scaled MMA: a Blackwell GPU",
)


def test_the_tiles_of_rows_cover_each_groups_rows_once():
    tiles = np.asarray(gm.tiles_of(SIZES, M, 128))
    group_of = np.searchsorted(np.cumsum(SIZES), np.arange(M), side="right")
    covered = np.full(M, -1)
    for group, lo, hi in tiles:
        assert lo == hi or lo // 128 == (hi - 1) // 128  # inside one tile
        assert (covered[lo:hi] == -1).all() and (group_of[lo:hi] == group).all()
        covered[lo:hi] = group
    assert (covered >= 0).sum() == SIZES.sum()
    assert len(tiles) == -(-M // 128) + len(SIZES) - 1


def test_the_scales_lie_as_tensor_memory_takes_them():
    r, c = 256, 8
    scales = (jnp.arange(r * c) % 251).astype(jnp.uint8).reshape(r, c)
    i, j = np.meshgrid(np.arange(r), np.arange(c), indexing="ij")
    tiles = np.asarray(gm.in_tensor_memory(scales))
    got = tiles[i // 128, j // 4, i % 32, i % 128 // 32 * 4 + j % 4]
    np.testing.assert_array_equal(got, scales)


def test_padding_k_adds_zeros_to_whole_tiles():
    values = jax.random.normal(jax.random.key(0), (3, 5, 704)).astype(E4M3)
    padded = gm._whole_tiles_of_k(values)
    assert padded.shape == (3, 5, 768) and padded.dtype == values.dtype
    np.testing.assert_array_equal(
        padded[..., :704].astype(jnp.float32), values.astype(jnp.float32)
    )
    assert (padded[..., 704:].astype(jnp.float32) == 0).all()


def test_the_scales_pad_to_whole_tiles_as_a_block_of_zeros_is_scaled():
    scales = jnp.ones((100, 22), E8M0)  # 2^0, a byte of 127 each
    tiles = blockscaled.as_bytes(gm.in_tensor_memory(scales))
    assert tiles.shape == (1, 6, 32, 16)
    padding = 128 * 24 - 100 * 22  # bytes of 1, 2^-126: E8M0 pads a 0 as NaN
    assert int(tiles.astype(jnp.int32).sum()) == 100 * 22 * 127 + padding


def test_an_operand_laid_out_for_the_kernels_pads_k_and_each_matrix_s_scales():
    w = _mxfp8((3, 64, 704), 5)  # each of 3 matrices (n, k), k not whole tiles
    tiled = gm.tiled(w)
    assert tiled.values.shape == (3, 64, 768)
    assert tiled.scales.shape == (3, 1, 6, 32, 16)
    assert (blockscaled.as_bytes(tiled.values[..., 704:]) == 0).all()
    np.testing.assert_array_equal(
        blockscaled.as_bytes(tiled.scales[1]),
        blockscaled.as_bytes(gm.in_tensor_memory(w.scales[1])),
    )


def test_a_groups_edges_are_its_first_and_last_tiles_with_the_rest_as_zero():
    values = jax.random.normal(jax.random.key(0), (16, M)).astype(E4M3)
    ends = jnp.cumsum(SIZES)
    starts = ends - SIZES
    edges = np.asarray(gm._edges(values, starts, ends).astype(jnp.float32))
    values = values.astype(jnp.float32)
    for g, (lo, hi) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        for e, at in enumerate([lo, max(hi - 1, lo)]):
            first = at // 128 * 128
            m = first + np.arange(128)
            tile = np.asarray(values)[:, first : first + 128]
            np.testing.assert_array_equal(
                edges[g, e], np.where((m >= lo) & (m < hi), tile, 0)
            )


def _shapes(shape: tuple[int, ...]) -> gm.Tiled:
    """The shapes and dtypes of a matrix, or a stack of them, laid out for the
    kernels in MXFP8."""
    of = jax.ShapeDtypeStruct(shape, jnp.float32)
    return jax.eval_shape(lambda x: gm.tiled(_quantized(x)), of)


def _quantized(x: Float[Array, "*g r c"]) -> Blocks:
    """x in MXFP8 along its rows, each matrix of a stack on its own."""
    if x.ndim == 3:
        return jax.vmap(blockscaled.mx_quantize)(x)
    return blockscaled.mx_quantize(x)


def _mxfp8(shape: tuple[int, ...], seed: int) -> Blocks:
    return _quantized(jax.random.normal(jax.random.key(seed), shape))


# The baseline MoE's GEMMs (d 2048, f 704): x W and dy W^T, with W[g]^T (n, k); dW.
ROWS = [(2048, 1408), (704, 2048), (2048, 704)]
CONTRACTS = [(2048, 1408), (704, 2048)]


@pytest.mark.parametrize("k, n", ROWS)
def test_the_rows_kernel_lowers_for_a_blackwell_gpu(k, n):
    def f(a: gm.Tiled, b: gm.Tiled, sizes: Int[Array, " g"]) -> Float[Array, "m n"]:
        return gm.rows(a, b, sizes, jnp.bfloat16)

    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    export = jax.export.export(jax.jit(f), platforms=["cuda"])
    export(_shapes((M, k)), _shapes((8, n, k)), sizes)


@pytest.mark.parametrize("d, f", CONTRACTS)
def test_the_contract_kernel_lowers_for_a_blackwell_gpu(d, f):
    def fn(a: gm.Tiled, b: gm.Tiled, sizes: Int[Array, " g"]) -> Float[Array, "g d f"]:
        return gm.contract(a, b, sizes, jnp.bfloat16)

    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    export = jax.export.export(jax.jit(fn), platforms=["cuda"])
    export(_shapes((d, M)), _shapes((f, M)), sizes)


@blackwell
@pytest.mark.parametrize("k, n", [(256, 384), (704, 256)])
def test_the_rows_kernel_multiplies_as_xla_dequantized(k, n):
    a, b = _mxfp8((M, k), 0), _mxfp8((SIZES.shape[0], n, k), 1)
    got = gm.rows(gm.tiled(a), gm.tiled(b), SIZES, jnp.float32)
    each = jnp.swapaxes(jax.vmap(blockscaled.dequantized)(b), 1, 2)
    want = jax.lax.ragged_dot(blockscaled.dequantized(a), each, SIZES)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
    assert (np.asarray(got)[int(SIZES.sum()) :] == 0).all()


@blackwell
@pytest.mark.parametrize("d, f", [(256, 384), (704, 256)])
def test_the_contract_kernel_sums_as_xla_dequantized(d, f):
    a, b = _mxfp8((d, M), 2), _mxfp8((f, M), 3)
    got = gm.contract(gm.tiled(a), gm.tiled(b), SIZES, jnp.float32)
    rows = (blockscaled.dequantized(a).T, blockscaled.dequantized(b).T)
    want = jax.lax.ragged_dot_general(*rows, SIZES, grouped.RAGGED_CONTRACTING)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
    assert (np.asarray(got)[np.asarray(SIZES) == 0] == 0).all()


@blackwell
@pytest.mark.parametrize("shape, filled", [((M, 704), 900), ((3, 704, 2048), 704)])
def test_the_quantizer_writes_what_xla_lays_out_for_the_kernels(shape, filled):
    from bux.architecture.experts import grouped_triton

    x = jax.random.normal(jax.random.key(4), shape, jnp.bfloat16)
    rows, columns = grouped_triton.quantized_twice(x, jnp.int32(filled))
    assert columns is not None
    kept = jnp.where((jnp.arange(shape[-2]) < filled)[:, None], x, 0)
    transposed = jnp.swapaxes(kept, -1, -2)
    for got, want in [(rows, _quantized(kept)), (columns, _quantized(transposed))]:
        want = gm.tiled(want)
        for a, b in zip(got, want, strict=True):
            np.testing.assert_array_equal(
                blockscaled.as_bytes(a), blockscaled.as_bytes(b)
            )


def test_the_quantizer_s_kernels_are_left_as_written_for_triton():
    """Triton compiles a kernel from its source in its module's globals: the import
    hook of these tests' type checks must not wrap one in a function of its own."""
    runtime = pytest.importorskip("triton.runtime")
    from bux.architecture.experts import grouped_triton

    found = [
        f for f in vars(grouped_triton).values() if isinstance(f, runtime.JITFunction)
    ]
    assert found
    assert all(f.fn.__globals__ is vars(grouped_triton) for f in found)
