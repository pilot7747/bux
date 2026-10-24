"""QK-norm then RoPE by bux's Pallas kernels (bux.architecture.attention.norm_rope),
interpreted on the CPU: the output and the gradients of x and the norm's scale against
the norm and `rotate` as they are, at the tokens' indices and at documents'
positions, zero-centered too, and on a mesh; and where the kernels run."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import outputs_and_grads
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture.attention import norm_rope
from bux.architecture.attention.rope import rotate
from bux.architecture.layers.norm import Norm
from bux.train import config as tcfg

ROPE = cfg.RoPEConfig(base=1e6)
RMS = cfg.NormalizationConfig(type="rmsnorm")


def _inputs(h: int, d: int, dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 2)
    x = 3 * jax.random.normal(keys[0], (2, 16, h, d))
    gain = 1 + 0.1 * jax.random.normal(keys[1], (d,))
    return x.astype(dtype), gain.astype(dtype)


def _norm(config: cfg.NormalizationConfig, scale: Float[Array, " d"]) -> Norm:
    """Return a norm of `config` whose scale parameter is `scale`."""
    norm = Norm(config, scale.shape[0])
    norm.scale = nnx.Param(scale)
    return norm


def _kernels(config: cfg.NormalizationConfig, positions=None):
    def run(x, scale):
        b, t = x.shape[:2]
        at = jnp.broadcast_to(jnp.arange(t), (b, t)) if positions is None else positions
        gain = 1 + scale if config.zero_centered else scale
        return norm_rope._fused(x, gain, at, ROPE, config.eps)

    return run


def _as_they_are(config: cfg.NormalizationConfig, positions=None):
    def run(x, scale):
        return rotate(_norm(config, scale)(x), ROPE, positions)

    return run


def _error(a, b):
    return np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32))


def _assert_close(got, want, tolerance):
    for a, b in zip(got, want, strict=True):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.linalg.norm(a - b) <= tolerance * np.linalg.norm(b)


def _in_float32(f, x, scale):
    """f's outputs and gradients of x and the scale, all in float32."""
    f32 = jnp.float32
    return outputs_and_grads(f, x.astype(f32), scale.astype(f32))


@pytest.mark.parametrize(("h", "d"), [(4, 64), (8, 32), (1, 128)])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_the_kernels_give_what_the_norm_and_rotate_do_in_float32(h, d, zero_centered):
    """To bfloat16's rounding, the scale's gradient too, which XLA's sum in bfloat16
    takes 2 to 7 times further (`test_the_kernels_round_less_than_xla`)."""
    config = RMS.model_copy(update={"zero_centered": zero_centered})
    x, gain = _inputs(h, d)
    scale = gain - 1 if zero_centered else gain
    got = outputs_and_grads(_kernels(config), x, scale)
    _assert_close(got, _in_float32(_as_they_are(config), x, scale), 5e-3)


def test_at_documents_positions():
    x, gain = _inputs(4, 64)
    positions = jnp.broadcast_to(jnp.arange(16) % 5, (2, 16))
    got = outputs_and_grads(_kernels(RMS, positions), x, gain)
    _assert_close(got, _in_float32(_as_they_are(RMS, positions), x, gain), 5e-3)


def test_the_kernels_round_less_than_xla():
    """The kernels' backward rounds its float32 once, where XLA rounds the
    rotation's gradient and the norm's between them, and sums the scale's in
    bfloat16."""
    x, gain = _inputs(4, 64)
    exact = _in_float32(_as_they_are(RMS), x, gain)
    kernel = outputs_and_grads(_kernels(RMS), x, gain)
    xla = outputs_and_grads(_as_they_are(RMS), x, gain)
    for k, a, e in zip(kernel[1:], xla[1:], exact[1:], strict=True):
        assert _error(k, e) < _error(a, e)


def test_on_a_mesh_each_device_runs_its_rows_and_heads(monkeypatch):
    """Bit for bit but the gain's gradient, which devices sum in bfloat16."""
    monkeypatch.setattr(norm_rope.kernels, "gpu", lambda: True)
    x, gain = _inputs(4, 64)

    def fused(x, scale):
        norm = _norm(RMS, scale)
        return norm_rope.norm_and_rotate(x, norm, ROPE, None, "heads")

    want = outputs_and_grads(_kernels(RMS), x, gain)
    with parallel.use(tcfg.MeshConfig(data=2, tensor=2)):
        assert norm_rope.takes(x, _norm(RMS, gain), ROPE, "heads")
        got = jax.jit(lambda x, gain: outputs_and_grads(fused, x, gain))(x, gain)
    _assert_close(got[:2], want[:2], 1e-6)
    _assert_close(got[2:], want[2:], 1e-2)


@pytest.mark.parametrize(
    ("change", "runs"),
    [
        ({}, True),
        ({"dtype": jnp.float32}, False),
        ({"h": 6}, False),
        ({"layout": "interleaved"}, False),
        ({"fraction": 0.5}, False),
        ({"norm": "layernorm"}, False),
    ],
    ids=["qwen3", "float32", "six-heads", "interleaved", "partial", "layernorm"],
)
def test_the_kernels_run_rmsnorm_and_rope_by_halves_of_16_bits(
    monkeypatch, change, runs
):
    monkeypatch.setattr(norm_rope.kernels, "gpu", lambda: True)
    x, gain = _inputs(change.get("h", 4), 64, change.get("dtype", jnp.bfloat16))
    norm = _norm(cfg.NormalizationConfig(type=change.get("norm", "rmsnorm")), gain)
    rope = ROPE.model_copy(
        update={k: change[k] for k in ("layout", "fraction") if k in change}
    )
    assert norm_rope.takes(x, norm, rope, "heads") == runs
