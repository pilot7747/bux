"""The delta rule by cuDNN's kernel (bux.architecture.fused_delta), with a stand-in for
the kernel, which the CPU lacks: bux's own delta rule on the packed tokens, a sequence
a document as cu_seqlens says. So what is checked is the packing: rows end to end,
documents' bounds, and rows of more documents than cu_seqlens has room for."""

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


@pytest.fixture
def stand_in(monkeypatch):
    kernel = types.SimpleNamespace(kimi_delta_attention=_kernel)
    monkeypatch.setattr(fused_delta, "frost", kernel)


def _inputs():
    ks = jax.random.split(jax.random.key(0), 5)
    q, k, v = (jax.random.normal(key, (B, T, H, K)) for key in ks[:3])
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    g = -5 * jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H, K)))
    return q, k, v, g, jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H)))


# One document a row, a few, and many; room for them all, and for 2 (the default at
# T = 64), which only the first fits: the rest go by the fallback.
@pytest.mark.parametrize("every", [T, 20, 3])
@pytest.mark.parametrize("room", [4096 * 64, fused_delta.DOCS])
def test_packed_rows_give_what_rows_do(stand_in, monkeypatch, every, room):
    monkeypatch.setattr(fused_delta, "DOCS", room)
    docs = Segments.of(
        jnp.broadcast_to(jnp.arange(T) % every, (B, T)).astype(jnp.int32)
    )
    xs = _inputs()
    got = fused_delta.delta_rule(*xs, docs, _chunked)
    np.testing.assert_allclose(got, _chunked(*xs, docs), atol=1e-5)
