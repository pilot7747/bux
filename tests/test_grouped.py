"""Grouped matmuls (bux.architecture.grouped): the Pallas kernel of the matrices'
gradient, interpreted on the CPU, and the custom VJP that runs the kernels, against
`jax.lax.ragged_dot` and its own gradients."""

import jax
import jax.numpy as jnp
import numpy as np

from bux.architecture import grouped

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
