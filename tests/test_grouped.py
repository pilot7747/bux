"""Grouped matmuls (bux.architecture.grouped): the Pallas kernel of the matrices'
gradient, interpreted on the CPU, and the custom VJP that runs the kernels, against
`jax.lax.ragged_dot` and its own gradients; and in MXFP8 (grouped_blockscaled), by XLA
on the CPU, against ragged_dot of the operands quantized as each GEMM contracts them."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from bux.architecture import blockscaled, grouped, grouped_blockscaled
from bux.train import config as tcfg

M, D, F = 300, 32, 48
X = jax.random.normal(jax.random.key(0), (M, D))
Y = jax.random.normal(jax.random.key(1), (M, F))
W = jax.random.normal(jax.random.key(2), (5, D, F))
# An empty group, groups that start and end inside the kernel's blocks of rows, and
# rows past the groups.
SIZES = jnp.array([70, 0, 100, 3, 77], jnp.int32)


def test_the_kernel_sums_each_groups_rows():
    want = jax.lax.ragged_dot_general(X, Y, SIZES, grouped.RAGGED_CONTRACTING)
    got = grouped._contract(X, Y, SIZES, 16, 16, interpret=True)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


def test_the_vjp_is_ragged_dots():
    def loss(f):
        return lambda x, w: jnp.sum(jnp.sin(f(x, w, SIZES)))

    out = grouped._kernels(X, W, SIZES)
    np.testing.assert_allclose(out, jax.lax.ragged_dot(X, W, SIZES), atol=1e-5)
    assert bool((out[int(SIZES.sum()) :] == 0).all())
    got = jax.grad(loss(grouped._kernels), (0, 1))(X, W)
    want = jax.grad(loss(jax.lax.ragged_dot), (0, 1))(X, W)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


# --- In MXFP8 (bux.architecture.grouped_blockscaled) -----------------------------------

W32 = jax.random.normal(jax.random.key(3), (5, D, 64))  # whole blocks of 32
G32 = jax.random.normal(jax.random.key(4), (M, 64))
SEED = jnp.uint32(0)


def _mxfp8(a: Float[Array, "r c"]) -> Float[Array, "r c"]:
    """a in MXFP8 along its rows, and back."""
    return grouped_blockscaled.dequantized(blockscaled.mx_quantize(a))


def _in_groups(a: Float[Array, "m c"]) -> Float[Array, "m c"]:
    return jnp.where((jnp.arange(M) < SIZES.sum())[:, None], a, 0)


def _ragged(passes=blockscaled.PASSES):
    def f(x: Float[Array, "m d"], w: Float[Array, "g d f"]) -> Float[Array, "m f"]:
        recipe = blockscaled.MXFP8
        return grouped_blockscaled.ragged_dot(x, w, SIZES, recipe, SEED, passes)

    return f


def test_the_tiles_of_rows_cover_each_groups_rows_once():
    tiles = np.asarray(grouped_blockscaled.tiles_of(SIZES, M, 32))
    group_of = np.searchsorted(np.cumsum(SIZES), np.arange(M), side="right")
    covered = np.full(M, -1)
    for group, lo, hi in tiles:
        assert lo == hi or lo // 32 == (hi - 1) // 32  # inside one tile
        assert (covered[lo:hi] == -1).all() and (group_of[lo:hi] == group).all()
        covered[lo:hi] = group
    assert (covered >= 0).sum() == SIZES.sum()
    assert len(tiles) == -(-M // 32) + len(SIZES) - 1


def test_mxfp8_multiplies_each_operand_as_mxfp8_holds_it_along_what_it_contracts():
    y, vjp = jax.vjp(_ragged(), X, W32)
    dx, dw = vjp(G32)
    x, g = _in_groups(X), _in_groups(G32)
    w_t = _mxfp8(jnp.swapaxes(W32, 1, 2).reshape(-1, D)).reshape(5, 64, D)
    w = _mxfp8(W32.reshape(-1, 64)).reshape(5, D, 64)
    want_y = jax.lax.ragged_dot(_mxfp8(x), jnp.swapaxes(w_t, 1, 2), SIZES)
    want_dx = jax.lax.ragged_dot(_mxfp8(g), jnp.swapaxes(w, 1, 2), SIZES)
    # In blocks along the rows, which dW contracts: whole blocks, as 0 past them.
    x_m, g_m = (_mxfp8(jnp.pad(a, ((0, -M % 32), (0, 0))).T).T[:M] for a in (x, g))
    want_dw = jax.lax.ragged_dot_general(x_m, g_m, SIZES, grouped.RAGGED_CONTRACTING)
    np.testing.assert_allclose(y, want_y, atol=1e-5)
    np.testing.assert_allclose(dx, want_dx, atol=1e-5)
    np.testing.assert_allclose(dw, want_dw, rtol=1e-5, atol=1e-4)
    assert not np.allclose(y, jax.lax.ragged_dot(X, W32, SIZES), atol=1e-3)
    past = int(SIZES.sum())
    assert (y[past:] == 0).all() and (dx[past:] == 0).all()


def test_rows_past_the_groups_add_nothing_to_the_matrices_gradient():
    def dw(x: Float[Array, "m d"]) -> Float[Array, "g d f"]:
        return jax.vjp(lambda w: _ragged()(x, w), W32)[1](G32)[0]

    garbage = X.at[int(SIZES.sum()) :].set(1e30)
    np.testing.assert_array_equal(dw(garbage), dw(X))


def test_mxfp8_multiplies_only_the_passes_named_in_it():
    def outputs(f):
        y, vjp = jax.vjp(f, X, W32)
        return [y, *vjp(G32)]

    exact = outputs(lambda x, w: jax.lax.ragged_dot(x, w, SIZES))
    for y, e in zip(outputs(_ragged(())), exact, strict=True):
        np.testing.assert_allclose(y, e, rtol=1e-5, atol=1e-4)
    every = outputs(_ragged())
    for i, name in enumerate(blockscaled.PASSES):
        others = outputs(_ragged(tuple(p for p in blockscaled.PASSES if p != name)))
        assert not np.allclose(every[i], exact[i], atol=1e-3)
        np.testing.assert_allclose(others[i], exact[i], rtol=1e-4, atol=1e-3)


def test_the_grouped_formats_are_those_the_config_allows():
    assert {"compute", *grouped_blockscaled.RECIPES} == set(tcfg.GROUPED)
