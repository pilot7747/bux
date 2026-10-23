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


def test_kda_runs_steep_gates_without_the_kernel(stand_in, monkeypatch):
    """A gate below the kernel's -5 at every position of a tile, unbounded: the
    fallback runs it, and the kernel, here a stand-in that gives nan, does not."""

    def overflows(q, k, v, g, beta, cu, *, scale):
        return jnp.full(v.shape, jnp.nan), None

    monkeypatch.setattr(
        fused_delta, "frost", types.SimpleNamespace(kimi_delta_attention=overflows)
    )
    docs, xs = _documents(T), _inputs(K, low=-12.0)
    got = fused_delta.delta_rule(*xs, docs, None, _chunked)
    np.testing.assert_allclose(got, _chunked(*xs, docs), atol=1e-5)


def test_kda_runs_shallow_unbounded_gates_by_the_kernel(stand_in):
    docs, xs = _documents(20), _inputs(K, low=-3.0)
    got = fused_delta.delta_rule(*xs, docs, None, _chunked)
    np.testing.assert_allclose(got, _chunked(*xs, docs), atol=1e-5)


@pytest.mark.parametrize(
    ("gate", "fits"),
    [(-5.0, True), (-5.1, False), (-0.1, True)],
    ids=["at-the-floor", "below-it", "shallow"],
)
def test_tiles_fit_the_kernel_where_16_positions_decay_by_at_most_exp_minus_80(
    gate, fits
):
    g = jnp.full((B, T, H, K), gate)
    assert bool(fused_delta._shallow(g)) is fits


def test_one_steep_position_in_a_tile_is_too_steep():
    g = jnp.full((B, T, H, K), -0.1).at[1, 40, 0, 3].set(-100.0)
    assert not bool(fused_delta._shallow(g))
