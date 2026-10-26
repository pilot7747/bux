"""Decayed linear attention (GLA, Lightning) by cuDNN's GDN-2 kernel
(bux.architecture.recurrent.fused_delta.linear), with a stand-in for the kernel, which
the CPU lacks: GDN-2's recurrence by bux's own chunks on the packed tokens, a sequence
a document as cu_seqlens says. So what is checked is what bux gives the kernel: beta 0
and w 1, a decay per head spread over the channels, the packing, rows of more documents
than cu_seqlens has room for, what the backward pass keeps, and where it runs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bux.architecture import Segments
from bux.architecture.recurrent import fused_delta, linear_attention
from bux.architecture.recurrent.linear_attention import recurrence

B, T, H, K = 2, 64, 2, 8


@pytest.fixture(autouse=True)
def full_precision():
    """Float32's matmuls at their full precision on a GPU too, whose default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


def _gdn2(q, k, v, g, beta, w, cu):
    """GDN-2 on THD arrays, ``S_t = (I - k (beta k)^T) Diag(e^g) S + k (w v)^T``, each
    sequence of cu its own document: bux's rank-one term, a = -beta k e^g and b = k."""
    n = q.shape[0]
    starts = jnp.zeros(n, bool).at[cu[:-1]].set(True, mode="drop")
    first = jax.lax.cummax(jnp.where(starts, jnp.arange(n), 0))
    docs = Segments.of((jnp.arange(n) - first)[None].astype(jnp.int32))
    a = -beta * k * jnp.exp(g)
    return recurrence(q[None], k[None], (w * v)[None], g[None], docs, a[None], k[None])[
        0
    ]


def _graph_call(op, results, *xs):
    """The kernel's forward and backward graphs, by the stand-in."""
    if op == "gdn2":
        return (_gdn2(*xs),)
    *ins, d_out = xs
    return jax.vjp(lambda *x: _gdn2(*x, ins[-1]), *ins[:6])[1](d_out)


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(fused_delta, "_graph_call", _graph_call)


# What the kernel falls back on: the chunks, recomputed, as linear_attention passes.
def _recomputed(*xs):
    return recurrence(*xs, states="recompute")


def _inputs(decay: int):
    """q, k, v and the log decay, of `decay` channels in [-1, 0]."""
    ks = jax.random.split(jax.random.key(0), 4)
    q, k, v = (jax.random.normal(key, (B, T, H, K)) for key in ks[:3])
    g = -jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H, decay)))
    return q, k, v, g


def _documents(every: int) -> Segments:
    return Segments.of(
        jnp.broadcast_to(jnp.arange(T) % every, (B, T)).astype(jnp.int32)
    )


def _loss(f, docs):
    weights = jnp.cos(jnp.arange(B * T * H * K)).reshape(B, T, H, K)
    return lambda *xs: (f(*xs, docs) * weights).sum()


def _kernel(*xs):
    *xs, docs = xs
    return fused_delta.linear(*xs, docs, _recomputed)


# One document a row, a few, and many; room for them all, and for 2 (the default at
# T = 64), which only the first fits: the rest go by the fallback.
@pytest.mark.parametrize("decay", [K, 1], ids=["gla", "lightning"])
@pytest.mark.parametrize("every", [T, 20, 3])
@pytest.mark.parametrize("room", [4096 * 64, fused_delta.DOCS])
def test_packed_rows_give_what_rows_do(stand_in, monkeypatch, decay, every, room):
    """And their gradients, the log decay's of one a head summed over the channels
    the kernel spreads it over."""
    monkeypatch.setattr(fused_delta, "DOCS", room)
    docs, (q, k, v, g) = _documents(every), _inputs(decay)
    xs = (q, k, v, g)
    want = recurrence(q, k, v, g, docs)
    np.testing.assert_allclose(_kernel(*xs, docs), want, atol=1e-5)
    grads = [
        jax.grad(_loss(f, docs), argnums=range(4))(*xs) for f in (_kernel, recurrence)
    ]
    for a, b in zip(*grads, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("decay", [K, 1], ids=["gla", "lightning"])
def test_the_kernel_keeps_the_rows_as_they_come(stand_in, decay):
    """For the backward pass: q, k, v and the log decay, of the shapes the layer gives
    them, which the cond's two branches share."""
    docs, xs = _documents(20), _inputs(decay)

    def kept(*xs):
        return jax.tree.leaves(jax.vjp(lambda *x: _kernel(*x, docs), *xs)[1])

    rows, small = sum(x.nbytes for x in xs), 4 * B * T * 4  # and where documents start
    kept_bytes = sum(x.size * x.dtype.itemsize for x in jax.eval_shape(kept, *xs))
    assert kept_bytes <= rows + small


def test_decayed_runs_by_the_kernel_where_it_runs(stand_in, monkeypatch):
    """And by the chunks elsewhere: the same values either way."""
    docs, xs = _documents(20), _inputs(K)
    calls, linear = [], fused_delta.linear
    monkeypatch.setattr(fused_delta, "linear", lambda *a: calls.append(a) or linear(*a))
    want = linear_attention.decayed(*xs, docs)
    assert not calls
    monkeypatch.setattr(fused_delta, "linear_runs", lambda *shapes: True)
    np.testing.assert_allclose(linear_attention.decayed(*xs, docs), want, atol=1e-5)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("dims", "dtype", "gdn2", "runs"),
    [
        ((128, 128), jnp.bfloat16, True, True),
        ((64, 128), jnp.bfloat16, True, True),
        ((128, 128), jnp.float16, True, False),
        ((128, 128), jnp.float32, True, False),
        ((32, 128), jnp.bfloat16, True, False),
        ((128, 128), jnp.bfloat16, False, False),
    ],
    ids=["bfloat16", "dk-64", "float16", "float32", "dk-32", "release"],
)
def test_the_kernel_runs_bfloat16_heads_of_64_or_128_where_cudnn_has_gdn2(
    monkeypatch, dims, dtype, gdn2, runs
):
    monkeypatch.setattr(fused_delta.kernels, "gpu", lambda: True)
    monkeypatch.setattr(fused_delta, "_sm", lambda: True)
    monkeypatch.setattr(fused_delta, "_has_gdn2", lambda: gdn2)
    assert fused_delta.linear_runs(*dims, dtype) is runs


def test_no_kernel_runs_off_a_gpu():
    assert not fused_delta.linear_runs(128, 128, jnp.bfloat16)
