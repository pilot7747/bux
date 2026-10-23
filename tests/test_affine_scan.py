"""The chain of affine maps (bux.architecture.recurrent.affine_scan): its kernel,
interpreted on the CPU, and the kernel's VJP, against a scan."""

import jax
import jax.numpy as jnp
import numpy as np

from bux.architecture.recurrent import affine_scan

Z, N, K, V = 3, 5, 16, 64
A = jax.random.normal(jax.random.key(0), (Z, N, K, K)) / 4
B = jax.random.normal(jax.random.key(1), (Z, N, K, V))
WEIGHTS = jax.random.normal(jax.random.key(2), (Z, N, K, V))


def _reference(A, B):
    return affine_scan._reference(A, B, jnp.zeros_like(B[:, 0]))


def _loss(f):
    def loss(A, B):
        before, last = f(A, B)
        return jnp.sum(before * WEIGHTS) + jnp.sum(jnp.sin(last))

    return loss


def test_the_kernel_carries_each_state_to_the_next():
    for got, want in zip(
        affine_scan._kernel_states(A, B), _reference(A, B), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_its_gradient_runs_back_along_the_chain():
    got = jax.grad(_loss(affine_scan._kernel_states), (0, 1))(A, B)
    want = jax.grad(_loss(_reference), (0, 1))(A, B)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
