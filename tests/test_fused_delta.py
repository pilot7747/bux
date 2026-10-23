"""The delta rule by cuDNN's kernels (bux.architecture.fused_delta), with stand-ins for
the kernels, which the CPU lacks: bux's own delta rule on the packed tokens, a sequence
a document as cu_seqlens says. So what is checked is the packing: rows end to end,
documents' bounds, rows of more documents than cu_seqlens has room for, and which
gates KDA's kernel takes."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bux.architecture import Segments, fused_delta
from bux.architecture.linear_attention import _chunked

B, T, H, K = 2, 64, 2, 8


def _kernel(q, k, v, g, beta, cu, *, scale):
    """kimi_delta_attention on THD arrays: each sequence of cu its own document."""
    n = q.shape[0]
    starts = jnp.zeros(n, bool).at[cu[:-1]].set(True, mode="drop")
    first = jax.lax.cummax(jnp.where(starts, jnp.arange(n), 0))
    docs = Segments.of((jnp.arange(n) - first)[None].astype(jnp.int32))
    out = _chunked(q[None] * scale, k[None], v[None], g[None], beta[None], docs)
    return out[0], None


def _gdn(q, k, v, g, beta, cu):
    """gated_delta_net, by the stand-in of KDA's kernel with one decay a head."""
    return _kernel(q, k, v, g[..., None], beta, cu, scale=1.0)[0]


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(
        fused_delta, "frost", types.SimpleNamespace(kimi_delta_attention=_kernel)
    )
    monkeypatch.setattr(fused_delta, "gated_delta_net", _gdn)


def _inputs(decay: int = K, low: float = -5.0):
    """q, k, v, the log decay, of `decay` channels in [low, 0], and beta."""
    ks = jax.random.split(jax.random.key(0), 5)
    q, k, v = (jax.random.normal(key, (B, T, H, K)) for key in ks[:3])
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    g = low * jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H, decay)))
    return q, k, v, g, jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H)))


def _documents(every: int) -> Segments:
    return Segments.of(
        jnp.broadcast_to(jnp.arange(T) % every, (B, T)).astype(jnp.int32)
    )


# One document a row, a few, and many; room for them all, and for 2 (the default at
# T = 64), which only the first fits: the rest go by the fallback.
@pytest.mark.parametrize("decay", [1, K], ids=["gdn", "kda"])
@pytest.mark.parametrize("every", [T, 20, 3])
@pytest.mark.parametrize("room", [4096 * 64, fused_delta.DOCS])
def test_packed_rows_give_what_rows_do(stand_in, monkeypatch, decay, every, room):
    monkeypatch.setattr(fused_delta, "DOCS", room)
    docs, xs = _documents(every), _inputs(decay)
    got = fused_delta.delta_rule(*xs, docs, -5.0, _chunked)
    np.testing.assert_allclose(got, _chunked(*xs, docs), atol=1e-5)


def _overflows(q, k, v, g, beta, cu, *, scale):
    """KDA's kernel as it fails: nan wherever a head's gate falls below -5."""
    out, _ = _kernel(q, k, v, g, beta, cu, scale=scale)
    return jnp.where((g < fused_delta.FLOOR).any(-1)[..., None], jnp.nan, out), None


@pytest.fixture
def overflowing(stand_in, monkeypatch):
    kernel = types.SimpleNamespace(kimi_delta_attention=_overflows)
    monkeypatch.setattr(fused_delta, "frost", kernel)


def _steep(pairs) -> tuple:
    """`_inputs` of gates in [-3, 0], but at -12 in the (row, head) `pairs`."""
    q, k, v, g, beta = _inputs(K, low=-3.0)
    for row, head in pairs:
        g = g.at[row, :, head].set(-12.0)
    return q, k, v, g, beta


def _loss(f, docs):
    weights = jnp.cos(jnp.arange(B * T * H * K)).reshape(B, T, H, K)
    return lambda *xs: (f(*xs, docs) * weights).sum()


# No steep pair, one in a slot of one, one in two slots, and two in two.
@pytest.mark.parametrize(
    ("pairs", "share"),
    [((), 1 / 4), (((1, 0),), 1 / 4), (((1, 0),), 1 / 2), (((1, 0), (0, 1)), 1 / 2)],
)
def test_kda_runs_the_pairs_too_steep_for_its_kernel_by_chunks(
    overflowing, monkeypatch, pairs, share
):
    """Unbounded gates: the kernel runs the pairs it takes, the chunks the rest, and
    slots for steep pairs left over add nothing, to the outputs or the gradients."""
    monkeypatch.setattr(fused_delta, "STEEP", share)
    docs, xs = _documents(20), _steep(pairs)

    def fused(*xs):
        return fused_delta.delta_rule(*xs, None, _chunked)

    got, want = fused(*xs, docs), _chunked(*xs, docs)
    np.testing.assert_allclose(got, want, atol=1e-5)
    grads = [jax.grad(_loss(f, docs), argnums=range(5))(*xs) for f in (fused, _chunked)]
    for a, b in zip(*grads, strict=True):
        assert bool(jnp.isfinite(a).all())
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_kda_runs_by_chunks_alone_past_its_share_of_steep_pairs(overflowing):
    docs, xs = _documents(T), _steep([(0, 0), (1, 1)])
    got = fused_delta.delta_rule(*xs, docs, None, _chunked)
    np.testing.assert_allclose(got, _chunked(*xs, docs), atol=1e-5)


@pytest.mark.parametrize(
    ("gate", "steep"),
    [(-5.0, False), (-5.1, True), (-0.1, False)],
    ids=["at-the-floor", "below-it", "shallow"],
)
def test_a_pair_is_steep_where_16_positions_decay_by_more_than_exp_minus_80(
    gate, steep
):
    g = jnp.full((B, T, H, K), gate)
    assert bool(fused_delta._steep(g).all()) is steep


def test_one_steep_position_makes_its_pair_steep_alone():
    g = jnp.full((B, T, H, K), -0.1).at[1, 40, 0, 3].set(-100.0)
    np.testing.assert_array_equal(
        fused_delta._steep(g), [[False, False], [True, False]]
    )
