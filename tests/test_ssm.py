"""Mamba, Mamba-2 and LFM2's short convolution, against their references' math."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture import CausalConv, Model, Segments, Taps, num_params
from bux.architecture.graph import ResidualBranch, Sequence
from bux.architecture.ssm import Mamba, Mamba2, ShortConv
from bux.lang import compile
from bux.train import config as tcfg

D, V = 24, 64

LAYERS = {
    "mamba": "mamba(state_size=4)",
    "mamba_options": "mamba(state_size=3, expand=1, dt_rank=5, conv_size=3)",
    "mamba2": "mamba2(state_size=4, head_dim=8)",
    "mamba2_groups": "mamba2(state_size=4, head_dim=4, num_groups=3, conv_size=3)",
    "short_conv": "short_conv",
    "short_conv_bias": "short_conv(conv_size=2, bias=true)",
}


def source(mixer: str) -> str:
    return f"""
d_model = {D}
vocab_size = {V}
norm = rmsnorm
ffn = gated_mlp(intermediate_size=32, activation=swish)
model = embed [norm {mixer}] [norm ffn] norm head
"""


def model(mixer: str) -> Model:
    return Model(compile(source(mixer)), rngs=nnx.Rngs(0))


def mixer(m: Model) -> nnx.Module:
    """The mixer of the model's one block."""
    assert isinstance(m.graph, Sequence)
    block = m.graph.items[1]
    assert isinstance(block, ResidualBranch) and isinstance(block.body, Sequence)
    return block.body.items[1]


def tokens(seed: int = 0, length: int = 8) -> Int[Array, "b t"]:
    return jax.random.randint(jax.random.key(seed), (2, length), 0, V)


@pytest.mark.parametrize("name", LAYERS)
def test_runs_in_a_model_and_every_parameter_learns(name):
    m, t = model(LAYERS[name]), tokens()
    logits = nnx.jit(lambda m, t: m(t))(m, t)
    assert logits.shape == (2, 8, V) and bool(jnp.isfinite(logits).all())

    def loss(m: Model) -> Float[Array, ""]:
        return -jnp.take_along_axis(jax.nn.log_softmax(m(t)), t[..., None], -1).mean()

    grads = jax.tree.leaves(nnx.jit(nnx.grad(loss))(m))
    assert all(bool(jnp.isfinite(g).all()) for g in grads)
    x = jax.random.normal(jax.random.key(1), (2, 8, D))
    grads = jax.tree.leaves(nnx.grad(lambda m: jnp.square(m(x)).mean())(mixer(m)))
    assert all(bool((g != 0).any()) for g in grads)


@pytest.mark.parametrize("name", LAYERS)
def test_is_causal(name):
    m, t = model(LAYERS[name]), tokens()
    changed = t.at[:, 5:].set((t[:, 5:] + 1) % V)
    assert jnp.allclose(m(t)[:, :5], m(changed)[:, :5], atol=1e-5)
    assert not jnp.allclose(m(t)[:, 5:], m(changed)[:, 5:])


@pytest.mark.parametrize("name", LAYERS)
def test_documents_packed_in_a_row_see_only_themselves(name):
    m = model(LAYERS[name])
    a, b = tokens(0, 5), tokens(1, 7)
    positions = jnp.concatenate([jnp.arange(5), jnp.arange(7)])[None].repeat(2, 0)
    [packed] = m.outputs(jnp.concatenate([a, b], axis=1), positions)
    assert jnp.allclose(packed[:, :5], m(a), atol=1e-5)
    assert jnp.allclose(packed[:, 5:], m(b), atol=1e-5)
    # Without the positions, the second document sees the first.
    assert not jnp.allclose(m(jnp.concatenate([a, b], axis=1))[:, 5:], m(b), atol=1e-3)


def test_parameter_counts_follow_the_references():
    def count(layer: str) -> int:
        return num_params(mixer(model(layer)))

    # JambaMambaMixer: in_proj, conv1d (with a bias), x_proj, dt_proj (with a bias),
    # A_log, D, out_proj and the norms of the time step, B and C.
    for layer, e, n, r, k in [
        (LAYERS["mamba"], 2 * D, 4, math.ceil(D / 16), 4),
        (LAYERS["mamba_options"], D, 3, 5, 3),
    ]:
        expected = (
            D * 2 * e
            + (k * e + e)
            + e * (r + 2 * n)
            + (r * e + e)
            + e * n
            + e
            + e * D
            + (r + 2 * n)
        )
        assert count(layer) == expected
    # Mamba2Mixer: in_proj to z, x, B, C and dt; conv1d over x, B and C (with a bias);
    # dt_bias, A_log and D a head; the gated norm; out_proj.
    for layer, e, n, g, k, p in [
        (LAYERS["mamba2"], 2 * D, 4, 1, 4, 8),
        (LAYERS["mamba2_groups"], 2 * D, 4, 3, 3, 4),
    ]:
        h, conv = e // p, e + 2 * g * n
        expected = D * (e + conv + h) + (k * conv + conv) + 3 * h + e + e * D
        assert count(layer) == expected
    # Lfm2ShortConv: in_proj to B, C and x, the conv, out_proj; bias on all three.
    assert count(LAYERS["short_conv"]) == 3 * D * D + 3 * D + D * D
    assert count(LAYERS["short_conv_bias"]) == (
        (3 * D * D + 3 * D) + (2 * D + D) + (D * D + D)
    )


def test_initialization_follows_mamba():
    c = cfg.MambaConfig(dt_min=0.01, dt_max=0.05)
    m = Mamba(c, D, nnx.Rngs(0))
    dt = jax.nn.softplus(m.dt_bias[...])
    assert bool(((dt >= 0.01 * 0.999) & (dt <= 0.05 * 1.001)).all())
    assert jnp.allclose(jnp.exp(m.A_log[...]), jnp.arange(1, 17)[None].repeat(2 * D, 0))
    bound = m.rank**-0.5
    assert bool((jnp.abs(m.dt_proj[...]) <= bound).all())
    assert (m.D[...] == 1).all()
    m2 = Mamba2(cfg.Mamba2Config(head_dim=8), D, nnx.Rngs(0))
    dt = jax.nn.softplus(m2.dt_bias[...])
    assert bool(((dt >= 1e-3 * 0.999) & (dt <= 0.1 * 1.001)).all())
    a = jnp.exp(m2.A_log[...])
    assert a.shape == (m2.heads,) and bool(((a >= 1) & (a <= 16)).all())


def test_mamba2_heads_must_fill_the_width_and_groups():
    with pytest.raises(ValueError, match="heads"):
        Mamba2(cfg.Mamba2Config(head_dim=7), D, nnx.Rngs(0))
    with pytest.raises(ValueError, match="groups"):
        Mamba2(cfg.Mamba2Config(head_dim=8, num_groups=4), D, nnx.Rngs(0))


# Naive references, a position (and a head) at a time, in float64, on one document.
Doc = Float[np.ndarray, "t d"]


def _np(v: nnx.Param | None) -> Float[np.ndarray, "*shape"]:
    assert v is not None
    return np.asarray(v[...], np.float64)


def _silu(v: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
    return v / (1 + np.exp(-v))


def _softplus(v: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
    return np.log1p(np.exp(v))


def _rms(
    v: Float[np.ndarray, "*lead c"], weight: Float[np.ndarray, "*w c"], eps: float
) -> Float[np.ndarray, "*lead c"]:
    return v / np.sqrt((v**2).mean(-1, keepdims=True) + eps) * weight


def _conv(module: CausalConv, u: Float[np.ndarray, "t c"]) -> Float[np.ndarray, "t c"]:
    """Depthwise causal convolution: out[i] = bias + sum_j w[k - 1 - j] u[i - j]."""
    w = _np(module.weight)
    k = len(w)
    out = np.stack(
        [
            sum(w[k - 1 - j] * u[i - j] for j in range(k) if i >= j)
            for i in range(len(u))
        ]
    )
    return out if module.bias is None else out + _np(module.bias)


def _linear(
    module: nnx.Linear, v: Float[np.ndarray, "t c"]
) -> Float[np.ndarray, "t f"]:
    out = v @ _np(module.kernel)
    return out if module.bias is None else out + _np(module.bias)


def naive_mamba(m: Mamba, x: Doc) -> Doc:
    """JambaMambaMixer's slow path, one document of shape (t, d)."""
    u, z = np.split(_linear(m.in_proj, x), 2, -1)
    u = _silu(_conv(m.conv, u))
    r, n = m.rank, m.state_size
    dbc = _linear(m.x_proj, u)
    dt = _rms(dbc[:, :r], _np(m.dt_norm.scale), 1e-6) @ _np(m.dt_proj)
    dt = _softplus(dt + _np(m.dt_bias))
    B = _rms(dbc[:, r : r + n], _np(m.b_norm.scale), 1e-6)
    C = _rms(dbc[:, r + n :], _np(m.c_norm.scale), 1e-6)
    A = -np.exp(_np(m.A_log))  # (e, n)
    state, ys = np.zeros_like(A), []
    for i in range(len(x)):
        state = np.exp(dt[i][:, None] * A) * state + (dt[i] * u[i])[:, None] * B[i]
        ys.append(state @ C[i] + _np(m.D) * u[i])
    return _linear(m.out_proj, np.stack(ys) * _silu(z))


def naive_mamba2(m: Mamba2, x: Doc) -> Doc:
    """Mamba-2's recurrence, a head at a time, one document of shape (t, d)."""
    e, h, p, g, n = m.inner, m.heads, m.head_dim, m.groups, m.state_size
    t, gn = len(x), g * n
    proj = _linear(m.in_proj, x)
    z, ubc, dt = proj[:, :e], proj[:, e : 2 * e + 2 * gn], proj[:, 2 * e + 2 * gn :]
    ubc = _silu(_conv(m.conv, ubc))
    u = ubc[:, :e].reshape(t, h, p)
    B, C = ubc[:, e : e + gn].reshape(t, g, n), ubc[:, e + gn :].reshape(t, g, n)
    dt, A, skip = _softplus(dt + _np(m.dt_bias)), -np.exp(_np(m.A_log)), _np(m.D)
    state, y = np.zeros((h, p, n)), np.zeros((t, h, p))
    for i in range(t):
        for j in range(h):
            group = j // (h // g)
            update = dt[i, j] * np.outer(u[i, j], B[i, group])
            state[j] = np.exp(dt[i, j] * A[j]) * state[j] + update
            y[i, j] = state[j] @ C[i, group] + skip[j] * u[i, j]
    # Gated, then normalized over each group of e / g channels.
    y = (y.reshape(t, e) * _silu(z)).reshape(t, g, -1)
    y = _rms(y, np.ones(1), 1e-5).reshape(t, e) * _np(m.norm.scale).reshape(e)
    return _linear(m.out_proj, y)


def naive_short_conv(m: ShortConv, x: Doc) -> Doc:
    """Lfm2ShortConv, one document of shape (t, d)."""
    B, C, u = np.split(_linear(m.in_proj, x), 3, -1)
    return _linear(m.out_proj, C * _conv(m.conv, B * u))


def naive(layer: nnx.Module, x: Doc) -> Doc:
    match layer:
        case Mamba():
            return naive_mamba(layer, x)
        case Mamba2():
            return naive_mamba2(layer, x)
        case ShortConv():
            return naive_short_conv(layer, x)
    raise AssertionError(layer)


@pytest.mark.parametrize("name", LAYERS)
def test_matches_a_naive_loop_over_positions(name):
    layer = mixer(model(LAYERS[name]))
    # Move every parameter off its initial value, so that ones and zeros hide nothing.
    params = nnx.state(layer, nnx.Param)
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(1), len(leaves))
    noise = [v + 0.2 * jax.random.normal(k, v.shape) for v, k in zip(leaves, keys)]
    nnx.update(layer, jax.tree.unflatten(tree, noise))
    x = jax.random.normal(jax.random.key(2), (2, 7, D))
    y = layer(x)
    for row in range(2):
        expected = naive(layer, np.asarray(x[row], np.float64))
        assert np.allclose(y[row], expected, atol=1e-4)
    # Two documents in a row, 3 and 4 tokens long, each from a zero state.
    docs = Segments.of(jnp.asarray([[0, 1, 2, 0, 1, 2, 3]] * 2))
    y = layer(x, Taps(segments=docs))
    for row in range(2):
        first, second = (
            np.asarray(x[row, s], np.float64) for s in (slice(3), slice(3, 7))
        )
        expected = np.concatenate([naive(layer, first), naive(layer, second)])
        assert np.allclose(y[row], expected, atol=1e-4)


# --- Split over devices ----------------------------------------------------------------

# Tensor parallelism 4 ways with FSDP, and 2 ways with sequence parallelism.
SPLITS = [
    {"mesh": "mesh(fsdp=2, tensor=4)"},
    {"mesh": "mesh(fsdp=4, tensor=2)", "layout": "layout(stream=tensor)"},
]
# At 1e-2, these small runs are chaotic enough that rounding grows past 1e-5.
STABLE = "adamw(lr=1e-3)"


@pytest.mark.parametrize("name", LAYERS)
def test_trains_as_on_one_device_split_over_channels(losses, involuntary, name):
    one = losses(source(LAYERS[name]), mesh="mesh(data=1)", optimizer=STABLE)
    involuntary()
    for split in SPLITS:
        split_losses = losses(source(LAYERS[name]), optimizer=STABLE, **split)
        assert np.allclose(split_losses, one, atol=1e-5)
    assert involuntary() == 0


def _spec(v: nnx.Variable | None) -> P:
    """Where `v` is, a name or None for every dimension."""
    assert v is not None
    x = v[...]
    spec = tuple(x.sharding.spec)
    return P(*spec, *(None,) * (x.ndim - len(spec)))


def test_channels_split_over_mlp_and_heads_but_fused_projections_stay_whole():
    with parallel.use(tcfg.MeshConfig(tensor=2)):
        mamba = mixer(parallel.build(lambda: model(LAYERS["mamba"])))
        mamba2 = mixer(parallel.build(lambda: model(LAYERS["mamba2"])))
    assert isinstance(mamba, Mamba) and isinstance(mamba2, Mamba2)
    assert _spec(mamba.conv.weight) == _spec(mamba.dt_proj) == P(None, "tensor")
    assert _spec(mamba.A_log) == _spec(mamba.x_proj.kernel) == P("tensor", None)
    assert _spec(mamba.D) == _spec(mamba.dt_bias) == P("tensor")
    assert _spec(mamba.out_proj.kernel) == _spec(mamba2.out_proj.kernel)
    assert _spec(mamba2.out_proj.kernel) == P("tensor", None)
    assert _spec(mamba2.A_log) == _spec(mamba2.D) == P("tensor")
    # u and the gate side by side, and Mamba-2's convolved u, B and C
    whole = [mamba.in_proj.kernel, mamba2.in_proj.kernel, mamba2.conv.weight]
    assert all(_spec(p) == P(None, None) for p in whole)
