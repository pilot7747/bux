"""The delta rule with a decay per key channel by bux's Pallas kernels
(bux.architecture.recurrent.pallas_delta), interpreted on the CPU: the outputs and
every gradient against the chunks and the scan over positions, at any gate, in packed
documents and on a mesh; what the backward pass keeps; and where the kernels run."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import kept_bytes, outputs_and_grads

from bux import parallel
from bux.architecture import Segments
from bux.architecture.recurrent import fused_delta, linear_attention, pallas_delta
from bux.architecture.recurrent.linear_attention import _chunked
from bux.train import config as tcfg

B, H, K = 2, 2, 64


@pytest.fixture(autouse=True)
def full_precision():
    """Float32's matmuls at their full precision on a GPU too, whose default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


def _inputs(t: int, low: float, spike: float | None = None):
    """q, k, v, the log decay of every channel in [low, 0], spiking to `spike` every
    16 positions if given, and beta."""
    keys = jax.random.split(jax.random.key(0), 5)
    q, k, v = (jax.random.normal(key, (B, t, H, K)) for key in keys[:3])
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    g = low * jax.nn.sigmoid(jax.random.normal(keys[3], (B, t, H, K)))
    if spike is not None:
        g = g.at[:, 5::16].set(spike)
    return q, k, v, g, jax.nn.sigmoid(jax.random.normal(keys[4], (B, t, H)))


def _documents(t: int, every: int) -> Segments:
    return Segments.of(
        jnp.broadcast_to(jnp.arange(t) % every, (B, t)).astype(jnp.int32)
    )


def _kernels(*xs):
    return pallas_delta.delta_rule(*xs, _chunked)


def _stepwise(q, k, v, g, beta, docs):
    continues = (docs.within > 0).astype(jnp.float32)
    xs = (q, beta[..., None] * k, v, g, continues, -jnp.exp(g) * k)
    return linear_attention.stepwise(xs, (), False).out


def _outputs_and_grads(f, xs, docs):
    return outputs_and_grads(lambda *xs: f(*xs, docs), *xs)


def _assert_close(got, want, tolerance):
    """Each array within `tolerance` of its norm."""
    for a, b in zip(got, want, strict=True):
        assert np.linalg.norm(a - b) <= tolerance * np.linalg.norm(b)


# One document a row, a few, and many; a row of whole chunks and one padded to them;
# gates that decay a chunk's channel by up to exp(-320), far past exp(-90).
@pytest.mark.parametrize("every", [1000, 20, 3])
@pytest.mark.parametrize("t", [128, 100])
@pytest.mark.parametrize("low", [-1.0, -20.0])
def test_the_kernels_give_what_the_chunks_do(every, t, low):
    xs, docs = _inputs(t, low), _documents(t, every)
    got = _outputs_and_grads(_kernels, xs, docs)
    _assert_close(got, _outputs_and_grads(_chunked, xs, docs), 1e-5)


def test_a_spike_of_minus_ten_thousand_starts_the_state_again():
    """In every channel, as a document start would."""
    xs, docs = _inputs(128, -1.0, spike=-1e4), _documents(128, 7)
    got = _outputs_and_grads(_kernels, xs, docs)
    _assert_close(got, _outputs_and_grads(_chunked, xs, docs), 1e-4)


def _kept_bytes(xs, docs, states):
    """What the backward pass keeps, in bytes, by the run's `methods.states`."""
    return kept_bytes(lambda *x: _kernels(*x, docs), *xs, states=states)


def test_recomputed_the_backward_pass_keeps_the_inputs_alone():
    xs, docs = _inputs(128, -1.0), _documents(128, 20)
    inputs = sum(x.nbytes for x in xs) + B * 128 * 4  # and the documents
    assert _kept_bytes(xs, docs, "recompute") <= inputs


def test_kept_the_backward_pass_keeps_the_chunks_parts_and_states_too():
    """Each chunk's W, Y, Q and K, its M, KK and T of 64 a position, D, and the state
    before it; and gives what recomputing them gives."""
    t, xs, docs = 128, _inputs(128, -1.0), _documents(128, 20)
    inputs = sum(x.nbytes for x in xs) + B * t * 4
    parts = (4 * K + 3 * 64) * B * t * H * 4 + 2 * B * H * K * 4
    states = B * H * 2 * K * K * 4
    assert _kept_bytes(xs, docs, "keep") <= inputs + parts + states
    grads = []
    for states in ("keep", "recompute"):
        methods = tcfg.MethodsConfig(states=states)
        with parallel.use(tcfg.MeshConfig(data=1), methods=methods):
            grads.append(
                jax.jit(lambda *xs: _outputs_and_grads(_kernels, xs, docs))(*xs)
            )
    _assert_close(*grads, 1e-6)


def test_on_a_mesh_each_device_runs_its_rows_and_heads():
    xs, docs = _inputs(128, -5.0), _documents(128, 20)
    want = _outputs_and_grads(_kernels, xs, docs)
    with parallel.use(tcfg.MeshConfig(data=2, tensor=2)):
        got = jax.jit(lambda *xs: _outputs_and_grads(_kernels, xs, docs))(*xs)
    _assert_close(got, want, 1e-6)


@pytest.mark.parametrize(
    ("dk", "decays", "dtype", "recurrence", "runs"),
    [
        (128, 128, jnp.bfloat16, "scan", True),
        (64, 64, jnp.float16, "scan", True),
        (128, 1, jnp.bfloat16, "scan", False),
        (128, 128, jnp.float32, "scan", False),
        (256, 256, jnp.bfloat16, "scan", False),
        (128, 128, jnp.bfloat16, "pass_state", False),
    ],
    ids=["kda", "kda-f16", "gdn", "float32", "wide", "pass_state"],
)
def test_the_kernels_run_a_decay_per_channel_of_16_bits(
    monkeypatch, dk, decays, dtype, recurrence, runs
):
    monkeypatch.setattr(pallas_delta.kernels, "gpu", lambda: True)
    on = tcfg.MethodsConfig(recurrence=recurrence)
    with parallel.use(tcfg.MeshConfig(data=1), methods=on):
        assert pallas_delta.runs(dk, dk, dtype, decays) is runs


@pytest.mark.parametrize(
    ("floor", "b300", "by"),
    [
        (None, True, "pallas_delta"),
        (-5.0, True, "fused_delta"),
        (-6.0, True, "pallas_delta"),
        (-5.0, False, "pallas_delta"),
    ],
    ids=["unbounded", "bounded", "steep", "bounded-elsewhere"],
)
def test_kda_runs_by_cudnn_where_bounded_on_a_b300_else_by_bux(
    monkeypatch, floor, b300, by
):
    monkeypatch.setattr(pallas_delta.kernels, "gpu", lambda: True)
    monkeypatch.setattr(fused_delta, "frost", object())
    monkeypatch.setattr(fused_delta, "_sm", lambda: b300)
    chosen = linear_attention._kernels(128, 128, jnp.bfloat16, 128, floor)
    assert chosen is not None and chosen.__name__.endswith(by)
