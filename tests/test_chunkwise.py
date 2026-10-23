"""The recurrence of linear attention by chunks (bux.architecture.recurrent.chunkwise)
against the reference, a position at a time: outputs, end states, transitions and reads,
and gradients, over chunks and blocks smaller than the documents and larger."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from bux.architecture import Segments
from bux.architecture.recurrent import chunkwise
from bux.architecture.recurrent.linear_attention import stepwise

B, T, H, K, V = 2, 19, 2, 4, 3
POSITIONS = np.asarray([[*range(5), *range(9), *range(5)], [*range(T)]], np.int32)
CONTINUES = (Segments.of(jnp.asarray(POSITIONS)).within > 0).astype(jnp.float32)


def _normal(seed: int, *shape: int) -> Float[Array, "..."]:
    return jax.random.normal(jax.random.key(seed), shape)


def _delta_rule(decay: int, strong: bool) -> tuple:
    """`_inputs` as the delta rule makes them: a = -w k, and no b, which is the key
    it writes at."""
    q, k, v, g, c = _inputs(decay, False, strong)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    step = jax.nn.sigmoid(_normal(6, B, T, H, 1)) * k
    return q, step, v, g, c, -jnp.exp(g) * k


def _inputs(decay: int, rank_one: bool, strong: bool) -> tuple:
    """q, k, v, log decay, continues, and a and b if `rank_one`; the decays up to
    -40 a position if `strong`, which no exponent may overflow from."""
    top = 40.0 if strong else 0.7
    g = -jax.random.uniform(jax.random.key(3), (B, T, H, decay), maxval=top)
    xs = (_normal(0, B, T, H, K), _normal(1, B, T, H, K), _normal(2, B, T, H, V), g)
    ab = (_normal(4, B, T, H, K) / 3, _normal(5, B, T, H, K) / 3) if rank_one else ()
    return (*xs, CONTINUES, *ab)


CASES = [
    pytest.param(decay, rank_one, strong, id=f"{name}-{kind}{'-strong' * strong}")
    for decay, name in ((1, "head"), (K, "channel"))
    for rank_one, kind in ((False, "decay"), (True, "rank_one"))
    for strong in (False, True)
]


@pytest.mark.parametrize(("decay", "rank_one", "strong"), CASES)
@pytest.mark.parametrize(("size", "sub"), [(4, 2), (8, 4), (64, 16)])
def test_chunks_compute_what_positions_do(decay, rank_one, strong, size, sub):
    xs = _inputs(decay, rank_one, strong)
    got = chunkwise.chunk(xs, True, size, sub)
    want = stepwise(xs, (), True)
    for name in ("out", "state", "transition", "reads"):
        a, b = getattr(got, name), getattr(want, name)
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5, err_msg=name)


@pytest.mark.parametrize(("decay", "rank_one", "strong"), CASES)
def test_chunks_differentiate_as_positions_do(decay, rank_one, strong):
    xs = _inputs(decay, rank_one, strong)
    weights = jnp.cos(jnp.arange(B * T * H * V)).reshape(B, T, H, V)

    def loss(f, *xs):
        return (f(xs).out * weights).sum()

    # Not of continues, which says where documents start: a constant of the data.
    learned = tuple(i + 1 for i in range(len(xs)) if i != 4)
    forms = (
        lambda x: chunkwise.chunk(x, False, 8, 4),
        lambda x: stepwise(x, (), False),
    )
    grads = [jax.grad(loss, argnums=learned)(f, *xs) for f in forms]
    for i, (a, b) in enumerate(zip(*grads, strict=True)):
        assert bool(jnp.isfinite(a).all()), i
        np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5, err_msg=str(i))


@pytest.mark.parametrize("strong", [False, True])
@pytest.mark.parametrize(("size", "sub"), [(4, 2), (64, 16)])
def test_q_and_k_of_a_group_are_its_heads(strong, size, sub):
    """With one decay a head, one q and k for both heads, as Mamba-2's C and B."""
    q, k, v, g, continues = _inputs(1, False, strong)
    q, k = q[:, :, :1], k[:, :, :1]
    got = chunkwise.chunk((q, k, v, g, continues), True, size, sub)
    shared = (jnp.repeat(q, H, 2), jnp.repeat(k, H, 2), v, g, continues)
    want = stepwise(shared, (), True)
    for name in ("out", "state", "transition", "reads"):
        a, b = getattr(got, name), getattr(want, name)
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5, err_msg=name)


def test_the_gpu_smoke_script_compares_every_form_on_cpu(monkeypatch):
    path = Path(__file__).parent.parent / "scripts" / "gpu_smoke" / "recurrence.py"
    spec = importlib.util.spec_from_file_location("gpu_smoke_recurrence", path)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setattr(smoke, "T", 150)  # chunks of 64 and a part of one
    for case in smoke.CASES:
        assert max(smoke.errors(case)) < smoke.TOLERANCE, case
    for case in smoke.KERNELS:  # the chunks themselves, where no kernel runs
        assert max(smoke.kernel_errors(case)) < smoke.KERNEL_TOLERANCE, case


@pytest.mark.parametrize("c", [8, 64])
def test_the_inverse_differentiates_as_its_solve_does(c):
    """(I - L)^-1 by its own backward, T^T dT T^T of T alone, against jax's of the
    solve, row by row and by halves, below the diagonal: the solve reads zeros
    above it too, where L has no entries."""
    L = jnp.tril(_normal(0, B, 3, H, c, c) / c, -1)
    weights = _normal(1, B, 3, H, c, c)

    def loss(f, L):
        return (f(L) * weights).sum()

    got = jax.grad(loss, argnums=1)(chunkwise._inverse, L)
    want = jax.grad(loss, argnums=1)(chunkwise._halves, L)
    np.testing.assert_allclose(got, jnp.tril(want, -1), atol=1e-5, rtol=1e-5)


def test_keys_alike_with_steps_near_one_stay_finite():
    """Keys nearly equal, as a trained model's can be, with steps beta near 1: the
    powers of L grow as 64^n here, though the inverse does not."""
    t, k = 128, 16
    base = _normal(0, 1, 1, 1, k)
    keys = base + 0.01 * _normal(1, B, t, H, k)
    keys = keys / jnp.linalg.norm(keys, axis=-1, keepdims=True)
    g = -0.01 * jnp.ones((B, t, H, 1))
    beta = 0.95
    xs = (
        _normal(2, B, t, H, k),
        beta * keys,
        _normal(3, B, t, H, V),
        g,
        jnp.ones((B, t)),
        -jnp.exp(g) * keys,
        beta * keys,
    )
    got, want = chunkwise.chunk(xs, True), stepwise(xs, (), True)
    assert bool(jnp.isfinite(got.out).all())
    scale = float(jnp.abs(want.out).max())
    np.testing.assert_allclose(got.out, want.out, atol=1e-4 * scale)


@pytest.mark.parametrize("decay", [1, K])
@pytest.mark.parametrize("strong", [False, True])
@pytest.mark.parametrize(("size", "sub"), [(4, 2), (64, 16)])
def test_the_delta_rule_where_b_is_k(decay, strong, size, sub):
    xs = _delta_rule(decay, strong)
    got, want = chunkwise.chunk(xs, True, size, sub), stepwise(xs, (), True)
    for name in ("out", "state", "transition", "reads"):
        a, b = getattr(got, name), getattr(want, name)
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5, err_msg=name)
    weights = jnp.cos(jnp.arange(B * T * H * V)).reshape(B, T, H, V)

    def loss(f, q, k, v, g, a):
        return (f((q, k, v, g, CONTINUES, a)).out * weights).sum()

    forms = (
        lambda x: chunkwise.chunk(x, False, size, sub),
        lambda x: stepwise(x, (), False),
    )
    args = (xs[0], xs[1], xs[2], xs[3], xs[5])
    grads = [jax.grad(loss, argnums=(1, 2, 3, 4, 5))(f, *args) for f in forms]
    for i, (a, b) in enumerate(zip(*grads, strict=True)):
        np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-5, err_msg=str(i))
