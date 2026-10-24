"""The delta rule by cuDNN's kernels (bux.architecture.recurrent.fused_delta), with
stand-ins for the kernels, which the CPU lacks: bux's own delta rule on the packed
tokens, a sequence a document as cu_seqlens says. So what is checked is the packing:
rows end to end, documents' bounds, rows of more documents than cu_seqlens has room for,
what the backward pass keeps, and which gates the kernels take."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bux.architecture import Segments
from bux.architecture.recurrent import fused_delta
from bux.architecture.recurrent.linear_attention import _chunked

B, T, H, K = 2, 64, 2, 8
# What the kernels fall back on: the chunks, recomputed, as linear_attention passes.
_recomputed = functools.partial(_chunked, states="recompute")


@pytest.fixture(autouse=True)
def full_precision():
    """Float32's matmuls at their full precision on a GPU too, whose default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


def _run(q, k, v, g, beta, cu):
    """The kernels on THD arrays, each sequence of cu its own document: GDN's for
    gates of one decay a head (n, h), else KDA's."""
    n = q.shape[0]
    starts = jnp.zeros(n, bool).at[cu[:-1]].set(True, mode="drop")
    first = jax.lax.cummax(jnp.where(starts, jnp.arange(n), 0))
    docs = Segments.of((jnp.arange(n) - first)[None].astype(jnp.int32))
    g = g[..., None] if g.ndim == 2 else g
    return _chunked(q[None], k[None], v[None], g[None], beta[None], docs)[0]


def _grads_of(run):
    """The kernels' backward passes, of the stand-in `run`."""

    def grads(*tokens):
        *xs, d_out = tokens
        return jax.vjp(lambda *x: run(*x, xs[-1]), *xs[:5])[1](d_out)

    return grads


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(fused_delta, "_run", _run)
    monkeypatch.setattr(fused_delta, "_grads", _grads_of(_run))


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
    """And their gradients: the kernels' backward passes, packed as the forward."""
    monkeypatch.setattr(fused_delta, "DOCS", room)
    docs, xs = _documents(every), _inputs(decay)

    def fused(*xs):
        return fused_delta.delta_rule(*xs, _recomputed)

    np.testing.assert_allclose(fused(*xs, docs), _chunked(*xs, docs), atol=1e-5)
    grads = [jax.grad(_loss(f, docs), argnums=range(5))(*xs) for f in (fused, _chunked)]
    for a, b in zip(*grads, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("decay", [1, K], ids=["gdn", "kda"])
def test_the_kernels_keep_the_rows_as_the_chunks_do(stand_in, decay):
    """For the backward pass: q, k, v, the log decay and beta, as they come, which the
    cond's two branches share, and the chunks' inverses, of the branch not taken."""
    docs, xs = _documents(20), _inputs(decay)

    def kept(*xs):
        rule = lambda *x: fused_delta.delta_rule(*x, docs, _recomputed)
        return jax.tree.leaves(jax.vjp(rule, *xs)[1])

    inverses, rows = B * H * T * 64 * 4, sum(x.nbytes for x in xs)
    small = 4 * B * T * 4  # where documents continue, and so on
    kept_bytes = sum(x.size * x.dtype.itemsize for x in jax.eval_shape(kept, *xs))
    assert kept_bytes <= rows + inverses + small


def _loss(f, docs):
    weights = jnp.cos(jnp.arange(B * T * H * K)).reshape(B, T, H, K)
    return lambda *xs: (f(*xs, docs) * weights).sum()


@pytest.mark.parametrize(
    ("decays", "floor", "dtype", "runs"),
    [
        (1, None, jnp.bfloat16, True),
        (128, -5.0, jnp.bfloat16, True),
        (128, -1.0, jnp.bfloat16, True),
        (128, None, jnp.bfloat16, False),
        (128, -6.0, jnp.bfloat16, False),
        (1, None, jnp.float32, False),
    ],
    ids=["gdn", "kda-bounded", "kda-shallow", "kda", "kda-steep", "float32"],
)
def test_a_kernel_runs_one_decay_a_head_or_one_a_channel_bounded_at_minus_5(
    monkeypatch, decays, floor, dtype, runs
):
    monkeypatch.setattr(fused_delta.kernels, "gpu", lambda: True)
    monkeypatch.setattr(fused_delta, "_sm", lambda: True)
    monkeypatch.setattr(fused_delta, "frost", object())
    assert fused_delta.runs(128, 128, dtype, decays, floor) is runs
