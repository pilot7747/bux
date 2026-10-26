"""bux.softmax against jax's and an exact reference: the same values and gradients,
but where float32 rounds the largest probability to 1, gradients that hold and sum to
0, as in Barron et al.'s worked example."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from bux.softmax import log_softmax, softmax

E = np.exp(-18.0)  # e^-m at the example's margin, 18, where float32's 1 + 2e^-m is 1
FUNCTIONS = {
    "softmax": (softmax, jax.nn.softmax),
    "log": (log_softmax, jax.nn.log_softmax),
}


def test_the_papers_example_keeps_what_jax_rounds_off():
    z = jnp.array([9.0, -9.0, -9.0])
    np.testing.assert_allclose(log_softmax(z), [-2 * E, -18, -18], rtol=1e-6)
    assert jax.nn.log_softmax(z)[0] == 0
    grad = jax.jit(jax.grad(lambda z: softmax(z)[0]))(z)
    np.testing.assert_allclose(grad, [2 * E, -E, -E], rtol=1e-6)
    assert jax.grad(lambda z: jax.nn.softmax(z)[0])(z)[0] == 0


Rows = Float[np.ndarray, "64 n"]


def _rows(margin: float, n: int = 32) -> tuple[Rows, Rows]:
    """Logits with a largest of up to `margin` over the rest, and cotangents."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, n))
    x[np.arange(64), rng.integers(0, n, 64)] += rng.uniform(0, margin, 64)
    return x.astype(np.float32), rng.normal(size=(64, n)).astype(np.float32)


def _exact_vjp(x: Rows, g: Rows, log: bool) -> Rows:
    """The VJP in float64 by sums over pairs, which take no 1 - p: ``p_k sum_j p_j (g_k
    - g_j)`` for the softmax, ``sum_j p_j g_k - p_k g_j`` for the log-softmax."""
    x, g = x.astype(np.float64), g.astype(np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    if log:
        return (p[:, None, :] * g[:, :, None] - p[:, :, None] * g[:, None, :]).sum(-1)
    return p * ((g[:, :, None] - g[:, None, :]) * p[:, None, :]).sum(-1)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_gradients_are_exact_and_sum_to_0_past_float64s_cliff(name):
    fixed, _ = FUNCTIONS[name]
    x, g = _rows(margin=40)
    (dx,) = jax.vjp(jax.jit(fixed), jnp.asarray(x))[1](jnp.asarray(g))
    exact = _exact_vjp(x, g, name == "log")
    scale = np.abs(exact).max(-1, keepdims=True)
    dx = np.asarray(dx, np.float64)
    assert (np.abs(dx - exact) / scale).max() < 1e-5
    if name == "softmax":
        assert (np.abs(dx.sum(-1, keepdims=True)) / scale).max() < 1e-5


def test_jaxs_softmax_gradient_does_not_sum_to_0_there():
    x, g = _rows(margin=40)
    (dx,) = jax.vjp(jax.nn.softmax, jnp.asarray(x))[1](jnp.asarray(g))
    scale = np.abs(_exact_vjp(x, g, False)).max(-1)
    assert (np.abs(np.asarray(dx, np.float64).sum(-1)) / scale).max() > 0.1


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_away_from_the_cliff_and_at_ties_they_are_jaxs(name, axis):
    fixed, of_jax = FUNCTIONS[name]
    x, g = (jnp.asarray(a.reshape(8, 8, 32)) for a in _rows(margin=3))
    x = x.at[:, :, 1].set(x[:, :, 0])
    ours, ours_vjp = jax.vjp(lambda x: fixed(x, axis), x)
    theirs, theirs_vjp = jax.vjp(lambda x: of_jax(x, axis), x)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ours_vjp(g)[0], theirs_vjp(g)[0], rtol=1e-4, atol=1e-6)


def test_the_softmax_is_jaxs_to_the_bit_and_keeps_bfloat16():
    x = jnp.asarray(_rows(margin=30)[0])
    assert jnp.array_equal(softmax(x), jax.nn.softmax(x))
    x = x.astype(jnp.bfloat16)
    assert softmax(x).dtype == log_softmax(x).dtype == jnp.bfloat16
    assert jax.grad(lambda x: softmax(x)[:, 0].sum())(x).dtype == jnp.bfloat16


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_masked_key_gets_nothing_and_a_row_of_them_is_uniform(name):
    fixed, _ = FUNCTIONS[name]

    def probs(x: Float[Array, "2 3"]) -> Float[Array, "2 3"]:
        return jnp.exp(fixed(x)) if name == "log" else fixed(x)

    x = jnp.array([[1.0, -jnp.inf, 0.0], [jnp.finfo(jnp.float32).min] * 3])
    np.testing.assert_allclose(probs(x)[1], [1 / 3] * 3, rtol=1e-6)
    grad = jax.grad(lambda x: probs(x)[:, 0].sum())(x)
    assert grad[0, 1] == 0 and jnp.isfinite(grad).all()
