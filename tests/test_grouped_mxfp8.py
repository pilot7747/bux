"""Grouped MXFP8 GEMMs by Mosaic GPU (bux.architecture.experts.grouped_mxfp8): the
scales as tensor memory takes them and each group's edge tiles, on the CPU; the
kernels lowered for a Blackwell GPU, anywhere; and on a Blackwell GPU, the kernels
against XLA's dequantize-and-multiply, with empty groups, tiny ones and rows past
them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int

from bux import kernels
from bux.architecture.experts import grouped, grouped_blockscaled
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


def test_the_scales_pad_to_whole_tiles_with_bytes_of_zero_not_nan():
    scales = jnp.ones((100, 22), E8M0)  # 2^0, a byte of 127 each
    tiles = blockscaled.as_bytes(gm.in_tensor_memory(scales))
    assert tiles.shape == (1, 6, 32, 16)
    assert int(tiles.astype(jnp.int32).sum()) == 100 * 22 * 127


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


def _blocks(shape: tuple[int, int]) -> Blocks:
    """The shapes and dtypes of an (r, c) array in MXFP8."""
    return jax.eval_shape(
        blockscaled.mx_quantize, jax.ShapeDtypeStruct(shape, jnp.float32)
    )


# The baseline MoE's GEMMs (d 2048, f 704): x W and dy W^T, and dW.
ROWS = [("columns", 2048, 1408), ("columns", 704, 2048), ("rows", 2048, 704)]
CONTRACTS = [(2048, 1408), (704, 2048)]


@pytest.mark.parametrize("stacked, k, n", ROWS)
def test_the_rows_kernel_lowers_for_a_blackwell_gpu(stacked, k, n):
    g = 8
    b = _blocks((g * n, k) if stacked == "rows" else (n, g * k))

    def f(a: Blocks, b: Blocks, sizes: Int[Array, " g"]) -> Float[Array, "m n"]:
        return gm.rows(a, b, sizes, stacked, jnp.bfloat16)

    sizes = jax.ShapeDtypeStruct((g,), jnp.int32)
    jax.export.export(jax.jit(f), platforms=["cuda"])(_blocks((M, k)), b, sizes)


@pytest.mark.parametrize("d, f", CONTRACTS)
def test_the_contract_kernel_lowers_for_a_blackwell_gpu(d, f):
    def fn(a: Blocks, b: Blocks, sizes: Int[Array, " g"]) -> Float[Array, "g d f"]:
        return gm.contract(a, b, sizes, jnp.bfloat16)

    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    jax.export.export(jax.jit(fn), platforms=["cuda"])(
        _blocks((d, M)), _blocks((f, M)), sizes
    )


def _mxfp8(shape: tuple[int, int], seed: int) -> Blocks:
    return blockscaled.mx_quantize(jax.random.normal(jax.random.key(seed), shape))


@blackwell
@pytest.mark.parametrize(
    "stacked, k, n", [("columns", 256, 384), ("rows", 256, 384), ("columns", 704, 256)]
)
def test_the_rows_kernel_multiplies_as_xla_dequantized(stacked, k, n):
    g = SIZES.shape[0]
    a = _mxfp8((M, k), 0)
    b = _mxfp8((g * n, k) if stacked == "rows" else (n, g * k), 1)
    got = gm.rows(a, b, SIZES, stacked, jnp.float32)
    matrices = grouped_blockscaled._stacked(blockscaled.dequantized(b), g, stacked)
    want = jax.lax.ragged_dot(blockscaled.dequantized(a), matrices, SIZES)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
    assert (np.asarray(got)[int(SIZES.sum()) :] == 0).all()


@blackwell
@pytest.mark.parametrize("d, f", [(256, 384), (704, 256)])
def test_the_contract_kernel_sums_as_xla_dequantized(d, f):
    a, b = _mxfp8((d, M), 2), _mxfp8((f, M), 3)
    got = gm.contract(a, b, SIZES, jnp.float32)
    rows = (blockscaled.dequantized(a).T, blockscaled.dequantized(b).T)
    want = jax.lax.ragged_dot_general(*rows, SIZES, grouped.RAGGED_CONTRACTING)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
    assert (np.asarray(got)[np.asarray(SIZES) == 0] == 0).all()
