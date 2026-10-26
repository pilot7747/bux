"""Gated DeltaNet-2 (linear_attention.gated_delta_2 and GatedDeltaNet2): its chunks
against a scan over positions as the paper's code steps it, KDA at b = w = beta, cuDNN's
GDN-2 kernel by a stand-in, which the CPU lacks (the scan on the packed tokens, a
sequence a document as cu_seqlens says), where it runs and where it falls back, and the
layer's gates, parameters and where they start."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bux import config as cfg
from bux import parallel
from bux.architecture import Segments, Taps
from bux.architecture.recurrent import fused_delta, linear_attention
from bux.architecture.recurrent.linear_attention import (
    GatedDeltaNet2,
    LowRank,
    delta_rule,
    gated_delta_2,
    l2norm,
)
from bux.train import config as tcfg

B, T, H, K, V = 2, 80, 2, 8, 4  # two chunks of 64, the second padded
D = 32  # the layer's width


@pytest.fixture(autouse=True)
def full_precision():
    """Float32's matmuls at their full precision on a GPU too, whose default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


def stepwise(q, k, v, g, erase, write, docs):
    """GDN-2 a position at a time, as the paper's code steps it: decay the state, read
    it along b * k, and write w * v less what it read at k; the state zero at each
    document's start."""

    def step(S, at):
        q, k, v, g, b, w, starts = at
        S = jnp.where(starts[:, None, None, None], 0, S) * jnp.exp(g)[..., None]
        read = jnp.einsum("bhk,bhkv->bhv", b * k, S)
        S = S + k[..., None] * (w * v - read)[..., None, :]
        return S, jnp.einsum("bhk,bhkv->bhv", q, S)

    xs = (q, k, v, g, erase, write, docs.within == 0)
    zero = jnp.zeros((*k.shape[:1], *k.shape[2:], v.shape[-1]))
    _, out = jax.lax.scan(step, zero, tuple(jnp.moveaxis(x, 1, 0) for x in xs))
    return jnp.moveaxis(out, 0, 1)


def _inputs(low: float = -1.0):
    """q, scaled, and k, normalized; v; log decays in [low, 0]; and b and w."""
    ks = jax.random.split(jax.random.key(0), 6)
    q, k = (l2norm(jax.random.normal(key, (B, T, H, K))) for key in ks[:2])
    v = jax.random.normal(ks[2], (B, T, H, V))
    g = low * jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H, K)))
    erase = jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H, K)))
    write = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H, V)))
    return q * K**-0.5, k, v, g, erase, write


def _documents(every: int) -> Segments:
    return Segments.of(
        jnp.broadcast_to(jnp.arange(T) % every, (B, T)).astype(jnp.int32)
    )


def _loss(f, docs):
    weights = jnp.cos(jnp.arange(B * T * H * V)).reshape(B, T, H, V)
    return lambda *xs: (f(*xs, docs) * weights).sum()


def _same(f, g, xs, docs):
    """Assert that f and g give the same outputs, and gradients of every input."""
    np.testing.assert_allclose(f(*xs, docs), g(*xs, docs), atol=1e-5)
    argnums = tuple(range(len(xs)))
    grads = [jax.grad(_loss(h, docs), argnums=argnums)(*xs) for h in (f, g)]
    for a, b in zip(*grads, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-4)


# Gates to -20 a position decay a block of 16 positions past exp(-83), as cuDNN's
# kernel has its exact path for; an erase gate in (0, 2), negative eigenvalues.
@pytest.mark.parametrize("low", [-1.0, -20.0], ids=["mild", "steep"])
@pytest.mark.parametrize("scale", [1.0, 2.0], ids=["erase", "negative-eigenvalues"])
def test_the_chunks_are_the_papers_recurrence(low, scale):
    """In rows of documents that cross the chunks' bounds."""
    q, k, v, g, erase, write = _inputs(low)
    _same(gated_delta_2, stepwise, (q, k, v, g, scale * erase, write), _documents(30))


def test_at_b_and_w_beta_it_is_kda():
    """Both gates beta in every channel: KDA's delta rule, and beta's gradient the sum
    of theirs."""
    q, k, v, g, _, _ = _inputs()
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.key(1), (B, T, H)))

    def gdn2(q, k, v, g, beta, docs):
        erase = jnp.broadcast_to(beta[..., None], k.shape)
        write = jnp.broadcast_to(beta[..., None], v.shape)
        return gated_delta_2(q, k, v, g, erase, write, docs)

    _same(gdn2, delta_rule, (q, k, v, g, beta), _documents(30))


# --- cuDNN's GDN-2 kernel, by a stand-in ------------------------------------------------


def _kernel(q, k, v, g, erase, write, cu):
    """cuDNN's GDN-2 graph on THD arrays, each sequence of cu its own document."""
    n = q.shape[0]
    starts = jnp.zeros(n, bool).at[cu[:-1]].set(True, mode="drop")
    first = jax.lax.cummax(jnp.where(starts, jnp.arange(n), 0))
    docs = Segments.of((jnp.arange(n) - first)[None].astype(jnp.int32))
    return stepwise(*(x[None] for x in (q, k, v, g, erase, write)), docs)[0]


def _graph_call(op, results, *xs):
    """The kernel's forward and backward graphs, by the stand-in."""
    if op == "gdn2":
        return (_kernel(*xs),)
    *ins, d_out = xs
    return jax.vjp(lambda *x: _kernel(*x, ins[-1]), *ins[:6])[1](d_out)


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(fused_delta, "_graph_call", _graph_call)


# What the kernel falls back on: the chunks, recomputed, as gated_delta_2 passes.
_recomputed = functools.partial(linear_attention._gated_chunked, states="recompute")


def _by_kernel(*xs):
    *xs, docs = xs
    return fused_delta.gated(*xs, docs, _recomputed)


# One document a row, a few, and many; room for them all, and for 2 (the default at
# T = 80), which only the first fits: the rest go by the fallback.
@pytest.mark.parametrize("every", [T, 20, 3])
@pytest.mark.parametrize("room", [4096 * 64, fused_delta.DOCS])
def test_packed_rows_give_what_rows_do(stand_in, monkeypatch, every, room):
    """And the gradients of every input, b's and w's too."""
    monkeypatch.setattr(fused_delta, "DOCS", room)
    _same(_by_kernel, linear_attention._gated_chunked, _inputs(), _documents(every))


def _ungated(q, k, v, g, docs):
    """Decayed linear attention by the kernel: no erase gate, and no write gate."""
    return fused_delta.linear(q, k, v, g, docs, linear_attention.recurrence)


def _outputs_and_grads(f, xs, docs):
    grads = jax.grad(_loss(f, docs), argnums=tuple(range(len(xs))))(*xs)
    return f(*xs, docs), grads


@pytest.mark.parametrize("gated", [True, False], ids=["gdn2", "decayed"])
def test_on_a_mesh_each_device_runs_its_rows_and_heads(stand_in, gated):
    """With the gates, and without them, as decayed linear attention runs it."""
    f, xs, docs = _by_kernel if gated else _ungated, _inputs(), _documents(T)
    xs = xs if gated else xs[:4]
    want = _outputs_and_grads(f, xs, docs)
    with parallel.use(tcfg.MeshConfig(data=2, tensor=2)):
        got = jax.jit(lambda *xs: _outputs_and_grads(f, xs, docs))(*xs)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_the_kernel_keeps_the_rows_as_they_come(stand_in):
    """For the backward pass: q, k, v, the log decay and the gates, of the shapes the
    layer gives them, which the cond's two branches share; and the inverse of each
    chunk of 64 that the chunks it falls back on keep."""
    docs, xs = _documents(20), _inputs()

    def kept(*xs):
        return jax.tree.leaves(jax.vjp(lambda *x: _by_kernel(*x, docs), *xs)[1])

    rows, inverses = sum(x.nbytes for x in xs), B * H * 2 * 64 * 64 * 4
    small = 4 * B * T * 4  # where documents continue, and so on
    kept_bytes = sum(x.size * x.dtype.itemsize for x in jax.eval_shape(kept, *xs))
    assert kept_bytes <= rows + inverses + small


@pytest.mark.parametrize(
    ("dims", "dtype", "floor", "runs"),
    [
        ((128, 128), jnp.bfloat16, -5.0, True),
        ((64, 128), jnp.bfloat16, -1.0, True),
        ((128, 128), jnp.bfloat16, -6.0, False),
        ((128, 128), jnp.bfloat16, None, False),
        ((128, 128), jnp.float16, -5.0, False),
        ((128, 128), jnp.float32, -5.0, False),
        ((32, 128), jnp.bfloat16, -5.0, False),
    ],
    ids=["bounded", "shallow", "steep", "unbounded", "float16", "float32", "dk-32"],
)
def test_the_kernel_runs_bfloat16_gates_bounded_at_minus_5(
    monkeypatch, dims, dtype, floor, runs
):
    monkeypatch.setattr(fused_delta.kernels, "gpu", lambda: True)
    monkeypatch.setattr(fused_delta, "_sm", lambda: True)
    monkeypatch.setattr(fused_delta, "_has_gdn2", lambda: True)
    assert fused_delta.gated_runs(*dims, dtype, floor) is runs


def test_no_kernel_runs_off_a_gpu():
    assert not fused_delta.gated_runs(128, 128, jnp.bfloat16, -5.0)


def _layer(**settings) -> GatedDeltaNet2:
    c = cfg.GatedDeltaNet2Config(num_heads=H, head_dim=K, v_head_dim=V, **settings)
    return GatedDeltaNet2(c, D, nnx.Rngs(0))


@pytest.mark.parametrize(("bound", "calls"), [(-5.0, 1), (None, 0)])
def test_the_layer_takes_the_kernel_bounded(stand_in, monkeypatch, bound, calls):
    """Its gate bounded at -5, where the kernel runs, and by the chunks unbounded:
    the same values either way."""
    layer, x = (
        _layer(gate_lower_bound=bound),
        jax.random.normal(jax.random.key(0), (B, T, D)),
    )
    want = layer(x, Taps())
    seen, gated = [], fused_delta.gated
    monkeypatch.setattr(fused_delta, "gated", lambda *a: seen.append(a) or gated(*a))
    monkeypatch.setattr(fused_delta, "linear_runs", lambda *shapes: True)
    np.testing.assert_allclose(layer(x, Taps()), want, atol=1e-5)
    assert len(seen) == calls


# --- The layer --------------------------------------------------------------------------


def test_the_gates_are_a_qk_heads_but_the_write_gate():
    """The log decay and b of a q/k head, shared by its value heads; w a value
    head's. b and w in (0, 1), the log decay at most 0."""
    c = cfg.GatedDeltaNet2Config(num_heads=2, num_v_heads=4, head_dim=8, v_head_dim=4)
    layer = GatedDeltaNet2(c, D, nnx.Rngs(0))
    g, erase, write = layer._gates(jax.random.normal(jax.random.key(0), (2, 6, D)))
    assert g.shape == erase.shape == (2, 6, 4, 8) and write.shape == (2, 6, 4, 4)
    for shared in (g, erase):
        np.testing.assert_array_equal(shared[:, :, 0], shared[:, :, 1])
        assert not np.allclose(shared[:, :, 1], shared[:, :, 2])
    assert not np.allclose(write[:, :, 0], write[:, :, 1])
    assert bool((g <= 0).all())
    assert all(bool(((x > 0) & (x < 1)).all()) for x in (erase, write))


def test_the_parameters_start_as_the_papers():
    """Every projection from Xavier's uniform at a gain of 2^-2.5, the decay's and the
    output gate's of rank v_head_dim, the gate's bias at 0, A_log at log U(1, 16) a
    q/k head and dt_bias a key channel."""
    c = cfg.GatedDeltaNet2Config(num_heads=2, num_v_heads=4, head_dim=16, v_head_dim=8)
    layer = GatedDeltaNet2(c, 64, nnx.Rngs(0))
    assert isinstance(layer.gate, LowRank) and layer.gate.bias is not None
    assert layer.f.down.shape == layer.gate.down.shape == (64, 8)
    assert (layer.gate.bias[...] == 0).all()
    assert layer.A_log.shape == (2,) and layer.dt_bias.shape == (32,)
    assert bool(
        ((jnp.exp(layer.A_log[...]) >= 1) & (jnp.exp(layer.A_log[...]) <= 16)).all()
    )
    kernels = [
        p.kernel for p in (layer.q, layer.k, layer.v, layer.erase, layer.write, layer.o)
    ]
    for w in [*kernels, layer.f.down, layer.f.up, layer.gate.down, layer.gate.up]:
        limit = 2**-2.5 * np.sqrt(6 / sum(w.shape))
        assert 0.9 * limit < float(jnp.abs(w[...]).max()) <= limit


def test_a_lower_bound_bounds_the_decay_and_starts_A_log_at_0():
    """Of inputs large enough to reach toward the bound past the small projections."""
    layer = _layer(gate_lower_bound=-5)
    g, _, _ = layer._gates(1000 * jax.random.normal(jax.random.key(0), (2, 6, D)))
    assert bool(((g >= -5) & (g <= 0)).all()) and float(g.min()) < -4.5
    assert (layer.A_log[...] == 0).all()


def test_negative_eigenvalues_double_the_erase_gate():
    x = jax.random.normal(jax.random.key(0), (2, 6, D))
    _, erase, _ = _layer()._gates(x)
    _, doubled, _ = _layer(negative_eigenvalues=True)._gates(x)
    np.testing.assert_allclose(doubled, 2 * erase)
