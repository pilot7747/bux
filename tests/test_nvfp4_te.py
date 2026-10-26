"""NVFP4's quantizers by Transformer Engine's kernels (bux.architecture.precision.nvfp4_te)
off a GPU: when they are taken, how a mesh's parts fit their tiles, the noise each part
draws, and their plumbing on the 8 CPU devices the tests have, by a stand-in for TE's
primitives that quantizes as XLA does. scripts/gpu_smoke/nvfp4.py checks TE's own on a
B300."""

import enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bux import kernels, parallel
from bux.architecture.precision import blockscaled, nvfp4, nvfp4_te
from bux.train import config as tcfg


def test_off_a_gpu_xla_quantizes():
    x = jnp.ones((64, 128), jnp.bfloat16)
    assert not nvfp4_te.takes(x)
    rows, cols = nvfp4.twice(x)
    want = nvfp4.xla_twice(x)
    assert jnp.array_equal(rows.values, want[0].values)
    assert jnp.array_equal(cols.values, want[1].values)


@pytest.fixture
def on_a_b300(monkeypatch):
    """TE's primitives by the stand-in, as on a B300 with TE installed."""
    monkeypatch.setattr(nvfp4_te, "_te", lambda: STAND_IN)
    monkeypatch.setattr(kernels, "gpu", lambda: True)


@pytest.mark.parametrize(
    ("shape", "dtype", "tiles", "taken"),
    [
        ((8192, 2048), jnp.bfloat16, False, True),
        ((4096, 2048), jnp.bfloat16, False, False),  # too few rows
        ((64, 128), jnp.bfloat16, True, True),  # W, of any rows
        ((8192, 128), jnp.float32, False, False),
        ((8224, 128), jnp.bfloat16, False, False),  # not whole tiles
        ((8192, 64), jnp.bfloat16, False, False),
    ],
)
def test_te_takes_bfloat16_in_whole_tiles_of_enough_rows(
    on_a_b300, shape, dtype, tiles, taken
):
    assert nvfp4_te.takes(jnp.zeros(shape, dtype), tiles) == taken


def test_te_takes_enough_rows_for_each_device_of_the_mesh(on_a_b300):
    x = jnp.zeros((8 * nvfp4_te.ROWS, 128), jnp.bfloat16)
    with parallel.use(tcfg.MeshConfig(data=4, tensor=2)):
        assert nvfp4_te.takes(x)
        assert not nvfp4_te.takes(x[: 4 * nvfp4_te.ROWS])


def test_twice_goes_to_te_where_it_takes_x_else_to_the_pallas_kernels(
    on_a_b300, monkeypatch
):
    te, pallas = (
        nvfp4.xla_twice(jnp.ones((16, 16))),
        nvfp4.xla_twice(jnp.ones((16, 16))),
    )
    monkeypatch.setattr(nvfp4_te, "twice", lambda *args: te)
    monkeypatch.setattr(nvfp4, "kernel_twice", lambda *args: pallas)
    assert nvfp4.twice(jnp.zeros((8192, 128), jnp.bfloat16)) is te
    assert nvfp4.twice(jnp.zeros((64, 128), jnp.bfloat16), tiles=True) is te
    assert nvfp4.twice(jnp.zeros((64, 128), jnp.bfloat16)) is pallas
    assert nvfp4.twice(jnp.zeros((8192, 128), jnp.float32)) is pallas


def test_te_transforms_by_bux_s_hadamard_matrix():
    # TE builds its transform from the mask: row i's sign negative where bit i is set.
    signs = [-1.0 if nvfp4_te.SIGN_MASK >> i & 1 else 1.0 for i in range(16)]
    i = np.arange(16)
    sylvester = (-1.0) ** np.bitwise_count(i[:, None] & i[None, :])
    assert np.array_equal(np.diag(signs) @ sylvester / 4, nvfp4.HADAMARD)


@pytest.mark.parametrize(
    ("spec", "shape", "fitting"),
    [
        (P("data", "tensor"), (512, 512), P("data", "tensor")),
        (P("data", "tensor"), (128, 384), P(None, None)),
        (P(("data", "tensor"), None), (512, 128), P(("data", "tensor"), None)),
        (P(("data", "tensor"), None), (256, 128), P(None, None)),
        (P(None, "data"), (64, 512), P(None, "data")),
    ],
)
def test_a_split_whose_parts_are_not_whole_tiles_stays_whole(spec, shape, fitting):
    mesh = jax.make_mesh((4, 2), ("data", "tensor"))
    assert nvfp4_te._in_tiles(spec, shape, mesh) == fitting


def test_each_part_draws_its_own_noise():
    salt = jnp.array([7], jnp.uint32)
    starts = ([0, 0], [64, 0], [0, 128], [64, 128])
    states = {
        tuple(nvfp4_te._noise(salt, jnp.array(start, jnp.uint32)).tolist())
        for start in starts
    }
    assert len(states) == len(starts)


# Interpreted off a GPU: x's largest magnitudes, the largest over every device's part,
# then both copies, by the stand-in, on any split.
@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize(
    ("spec", "shape"),
    [
        (P(), (128, 256)),
        (P("data", None), (256, 128)),
        (P(None, "data"), (64, 512)),
        (P("data", "tensor"), (512, 256)),
        (P("data", "tensor"), (128, 384)),  # parts not whole tiles: whole on each
    ],
)
def test_te_s_copies_are_xla_s_on_any_split(on_a_b300, spec, shape, tiles):
    x = jax.random.normal(jax.random.key(0), shape) * 2
    x = x.at[3, 5].set(40.0).astype(jnp.bfloat16)  # a tensor scale of one device's
    with parallel.use(tcfg.MeshConfig(data=4, tensor=2)) as mesh:
        split = jax.device_put(x, NamedSharding(mesh, spec))
        ours = jax.jit(lambda x: nvfp4_te.twice(x, tiles=tiles))(split)
    for got, want in zip(ours, nvfp4.xla_twice(x, tiles=tiles), strict=True):
        assert got.values.shape == want.values.shape
        assert got.scales.shape == want.scales.shape
        assert jnp.array_equal(
            blockscaled.dequantized(got), blockscaled.dequantized(want)
        )


# --- A stand-in for TE's primitives, by XLA ------------------------------------------
# Each binds as TE's does and quantizes by bux's XLA functions, on one device's part:
# the largest magnitudes of x and of x^T transformed; the cast, its scales padded to
# whole tiles of 128 x 4 as TE's are.


class Layout(enum.Enum):
    ROWWISE_COLWISE = "both"


class Mode(enum.Enum):
    NVFP4_1D_SCALING = "blocks"
    NVFP4_2D_SCALING = "tiles"


class Amax:
    @staticmethod
    def bind(x, **params: Any):
        x = x.astype(jnp.float32)
        hadamard = _hadamard(params["rht_matrix_random_sign_mask_t"])
        transformed = _transformed(x, hadamard)
        return jnp.abs(x).max().reshape(1), jnp.abs(transformed).max().reshape(1)


class Cast:
    @staticmethod
    def bind(x, scale, amax, noise, transformed, matrix, **params: Any):
        assert params["q_layout"] is Layout.ROWWISE_COLWISE
        tiles = params["scaling_mode"] == "tiles"
        x = x.astype(jnp.float32)
        rows = _cast(x, amax[0], tiles)
        if tiles:
            cols = _cast(x.T, amax[0], tiles=True)
        else:
            hadamard = matrix.astype(jnp.float32)
            cols = _cast(_transformed(x, hadamard), transformed[0], tiles=False)
        return rows[0], cols[0], rows[1], cols[1], amax, None, None


# Custom partitioning takes no constants: the stand-in builds bux's transform of TE's
# sign mask by ops.
def _hadamard(mask):
    i = jax.lax.iota(jnp.int32, 16)
    sylvester = 1 - 2 * (jax.lax.population_count(i[:, None] & i[None, :]) & 1)
    signs = 1 - 2 * ((mask >> i) & 1)
    return (signs[:, None] * sylvester).astype(jnp.float32) / 4


def _transformed(x, hadamard):
    """x^T, each 16 of its rows through `hadamard`, as nvfp4.hadamard_rows does."""
    r, c = x.shape
    blocks = x.T.reshape(c, r // 16, 16)
    return jnp.matmul(blocks, hadamard, precision="highest").reshape(c, r)


# As nvfp4.quantize does, by the tensor scale of `amax`, but in Python floats: custom
# partitioning takes no constants, which numpy's float32 scalars would be.
F32_MAX = float(jnp.finfo(jnp.float32).max)


def _cast(x, amax, tiles):
    encode = jnp.minimum(6 * 448 / amax, F32_MAX)
    encode = jnp.where((amax == 0) | (encode == 0), 1.0, encode)
    scales = nvfp4.block_scales(x, encode, tiles)
    inverse = jnp.minimum(1 / (scales.astype(jnp.float32) * (1 / encode)), F32_MAX)
    blocks = x.reshape(*scales.shape, 16) * inverse[:, :, None]
    values = jnp.clip(blocks.reshape(x.shape), -6, 6).astype(jnp.float4_e2m1fn)
    (r, n), raw = scales.shape, blockscaled.as_bytes(scales)
    padded = jnp.pad(raw, ((0, -r % 128), (0, -n % 4)))
    return values, jax.lax.bitcast_convert_type(padded, jnp.float8_e4m3fn)


STAND_IN = nvfp4_te.Te(Amax, Cast, None, Layout, Mode)
