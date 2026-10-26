"""The experts' MLP by cuDNN's fused kernels (bux.architecture.experts.grouped_glu and
grouped_cudnn). On the CPU, with stand-ins for the kernels that compute what they do by
XLA from the operands as quantized: the layout of each expert's rows from a multiple
of 256, the GLU's halves, which activations the kernels fuse, and the MLP's forward and
gradients against bux's path without it, within MXFP8's rounding, for each of them and
for hidden units padded to whole tiles. On a Blackwell GPU, the kernels against the
stand-ins, and the whole against that path."""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import kernels, parallel
from bux.architecture import Taps
from bux.architecture.experts import grouped, grouped_cudnn, grouped_glu, moe
from bux.architecture.experts import grouped_mxfp8 as gm
from bux.architecture.experts.grouped_cudnn import Columns, Epilogue
from bux.architecture.experts.grouped_mxfp8 import Tiled
from bux.architecture.experts.moe import MoE
from bux.architecture.precision import blockscaled, matmuls
from bux.train import config as tcfg

SWIGLU = cfg.MLPConfig(type="gated_mlp", intermediate_size=256, activation="swish")
RELU2 = cfg.MLPConfig(type="mlp", intermediate_size=256, activation="relu2")
# gpt-oss's experts but their biases, DeepSeek-V4's clamped SwiGLU, Kimi K3's SiTU.
GEGLU = SWIGLU.model_copy(
    update={"activation": "quick_gelu", "limit": 2.0, "up_offset": 1.0}
)
CLAMPED = SWIGLU.model_copy(update={"limit": 1.0})
SITU = SWIGLU.model_copy(update={"activation": "situ", "up_softcap": 2.0})
FFNS = {"swiglu": SWIGLU, "relu2": RELU2, "geglu": GEGLU, "clamped": CLAMPED}
FFNS["situ"] = SITU
# What the kernels fuse of each.
EPILOGUES = {
    "swiglu": grouped_cudnn.SWIGLU,
    "relu2": grouped_cudnn.SRELU,
    "geglu": Epilogue("geglu", alpha=1.702, limit=2.0, offset=1.0),
    "clamped": Epilogue("geglu", alpha=1.0, limit=1.0),
    "situ": Epilogue("situglu", cap=2.0),
}
D, F, E, K = 256, 256, 4, 2
blackwell = pytest.mark.skipif(
    not grouped_cudnn.runs() or grouped_glu.grouped_triton is None,
    reason="cuDNN's CuTe DSL kernels: a Blackwell GPU",
)


# --- Stand-ins: what the kernels compute, by XLA -----------------------------------


def _dequantized(a: Tiled) -> Float[Array, "*g r k"]:
    """Return the values a Tiled operand holds, of a matrix or a stack of them."""
    if a.values.ndim == 3:
        return jax.vmap(lambda v, s: _dequantized(Tiled(v, s)))(*a)
    p, q = a.scales.shape[:2]
    tiles = blockscaled.as_bytes(a.scales).reshape(p, q, 32, 4, 4)
    exponents = tiles.transpose(0, 3, 2, 1, 4).reshape(p * 128, q * 4)
    r, k = a.values.shape
    scale = jnp.exp2(exponents[:r, : k // 32].astype(jnp.float32) - 127)
    return a.values.astype(jnp.float32) * jnp.repeat(scale, 32, axis=1)


def _tiled(x: Float[Array, "*g r k"]) -> Tiled:
    """Return x along its rows, as XLA quantizes it, laid out as the kernels take it."""
    quantize = lambda m: gm.tiled(blockscaled.mx_quantize(m))
    for _ in range(x.ndim - 2):
        quantize = jax.vmap(quantize)
    return quantize(x)


def _columns(x: Float[Array, "m n"]) -> Columns:
    """Return x down its columns as the epilogues write it."""
    t = _tiled(x.T)
    return Columns(t.values.T[: x.shape[0]], t.scales)


def _pairs_of(halves: Int[Array, " n"]) -> Int[Array, " n"]:
    """Where columns of halves lie among pairs: `halves_of`'s inverse."""
    return jnp.argsort(grouped_cudnn.halves_of(jnp.arange(halves.shape[0])))[halves]


def _quantized_twice(x, filled, columns=True, index=None, halves=False, padded=None):
    if index is not None:
        x = jnp.where((index >= 0)[:, None], x[jnp.maximum(index, 0)], 0)
    x = jnp.where((jnp.arange(x.shape[-2]) < filled)[:, None], x, 0)
    if halves:
        x = x[..., _pairs_of(jnp.arange(x.shape[-1]))]
    if padded is not None:
        more = [(0, n - m) for n, m in zip(padded, x.shape[-2:], strict=True)]
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 2) + more)
    return _tiled(x), _tiled(jnp.swapaxes(x, -1, -2)) if columns else None


def _transposed(values, pairs=False):
    if pairs:
        values = values[:, grouped_cudnn.halves_of(jnp.arange(values.shape[1]))]
    return values.T


def _sizes(ends: Int[Array, " g"]) -> Int[Array, " g"]:
    return jnp.diff(ends, prepend=0)


def _times(a: Tiled, b: Tiled, ends: Int[Array, " g"]) -> Float[Array, "m n"]:
    each = jnp.swapaxes(_dequantized(b), 1, 2)
    return jax.lax.ragged_dot(_dequantized(a), each, _sizes(ends))


def _gate_and_up(product: Float[Array, "m n"]) -> tuple[Float[Array, "m f"], ...]:
    m, n = product.shape
    halves = product.reshape(m, n // 64, 2, 32)
    return halves[:, :, 0].reshape(m, -1), halves[:, :, 1].reshape(m, -1)


def _activation(
    epilogue: Epilogue, product: Float[Array, "m n"]
) -> Float[Array, "m f"]:
    """The epilogue's activation of the product, as cuDNN's kernels define it."""
    if not epilogue.gated:
        return jnp.square(jax.nn.relu(product))
    gate, up = _gate_and_up(product)
    if epilogue.name == "swiglu":
        return up * jax.nn.silu(gate)
    if epilogue.name == "situglu":
        situ = 4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate)
        return situ * epilogue.cap * jnp.tanh(up / epilogue.cap)
    # Clamped as cuDNN's dGLU differentiates it: a value at the limit passes its
    # gradient whole, where jnp.minimum and jnp.clip would split it.
    limit = epilogue.limit
    gate = jnp.where(gate > limit, limit, gate)
    up = jnp.where(up > limit, limit, jnp.where(up < -limit, -limit, up))
    return gate * jax.nn.sigmoid(epilogue.alpha * gate) * (up + epilogue.offset)


def _activated(x, w, ends, weights, epilogue):
    product = _times(x, w, ends)
    h = _activation(epilogue, product) * weights[:, None]
    return grouped_cudnn.Hidden(product.astype(jnp.bfloat16), _tiled(h), _columns(h))


def _activation_grad(dy, w, product, ends, weights, epilogue):
    dh = _times(dy, w, ends)
    activate = functools.partial(_activation, epilogue)
    h, vjp = jax.vjp(activate, product.astype(jnp.float32))
    (dp,) = vjp(dh * weights[:, None])
    dweights = (dh * h).sum(-1)
    return grouped_cudnn.HiddenGrad(_tiled(dp), _columns(dp), dweights)


def _rows(a, b, ends):
    return _times(a, b, ends).astype(jnp.bfloat16)


def _packed_rows(a, b, sizes, out, *, starts, height):
    """bux's rows kernel, each group's rows of a from `starts`, its output packed."""
    m = a.values.shape[0]
    spans = jnp.diff(jnp.append(starts, m))  # each group's rows of a, to the next's
    y = jax.lax.ragged_dot(_dequantized(a), jnp.swapaxes(_dequantized(b), 1, 2), spans)
    r = jnp.arange(height)
    ends = jnp.cumsum(sizes)
    group = jnp.minimum(jnp.searchsorted(ends, r, side="right"), sizes.shape[0] - 1)
    at = starts[group] + r - (ends - sizes)[group]
    rows = jnp.where((r < ends[-1])[:, None], y[jnp.minimum(at, m - 1)], 0)
    return rows.astype(out)


def _contract(a, b, ends, shape=None):
    if isinstance(a, Columns):
        a = Tiled(a.values.T, a.scales)
    rows = (_dequantized(a).T, _dequantized(b).T)
    dw = jax.lax.ragged_dot_general(*rows, _sizes(ends), grouped.RAGGED_CONTRACTING)
    d, f = dw.shape[1:] if shape is None else shape
    return dw[:, :d, :f].astype(jnp.bfloat16)


@pytest.fixture
def stand_in(monkeypatch):
    triton = types.SimpleNamespace(
        quantized_twice=_quantized_twice, transposed=_transposed
    )
    monkeypatch.setattr(grouped_glu, "grouped_triton", triton)
    packing = types.SimpleNamespace(rows=_packed_rows)
    monkeypatch.setattr(grouped_glu, "grouped_mxfp8", packing)
    for name, kernel in [
        ("runs", lambda: True),
        ("activated", _activated),
        ("activation_grad", _activation_grad),
        ("rows", _rows),
        ("contract", _contract),
    ]:
        monkeypatch.setattr(grouped_cudnn, name, kernel)


# --- The layout --------------------------------------------------------------------


def test_each_experts_rows_start_at_a_multiple_of_256_and_read_their_tokens():
    sizes = jnp.array([3, 0, 300, 1], jnp.int32)
    m, filled = 320, 304  # rows past the experts' lie nowhere
    source = jnp.arange(m) // 2
    picks = jnp.arange(m).reshape(-1, 2)
    layout, at = grouped_glu.padded(sizes, source, picks)
    np.testing.assert_array_equal(layout.ends, [256, 256, 768, 1024])
    starts = [0, 256, 256, 768]
    each = np.asarray(sizes)
    want = np.concatenate([starts[e] + np.arange(s) for e, s in enumerate(each)])
    np.testing.assert_array_equal(at[:filled], want)
    assert (np.asarray(at[filled:]) == layout.source.shape[0]).all()
    reads = np.asarray(layout.source)
    np.testing.assert_array_equal(reads[want], np.asarray(source[:filled]))
    assert (np.delete(reads, want) == -1).all()
    np.testing.assert_array_equal(layout.picks, np.asarray(at)[np.asarray(picks)])
    assert layout.source.shape[0] % 256 == 0
    assert layout.source.shape[0] >= m + 3 * 255


def test_the_glu_s_halves_put_32_gates_then_their_ups_in_each_64_columns():
    halves = np.asarray(grouped_cudnn.halves_of(jnp.arange(256)))
    assert sorted(halves) == list(range(256))
    unit, gate = np.arange(256) // 2, np.arange(256) % 2
    np.testing.assert_array_equal(halves // 64, unit // 32)
    np.testing.assert_array_equal(halves % 64 < 32, gate == 1)
    np.testing.assert_array_equal(halves % 32, unit % 32)


def test_the_scales_of_rows_in_halves_move_with_their_rows_into_pairs():
    x = jax.random.normal(jax.random.key(0), (256, 512)) * jnp.exp2(
        jax.random.randint(jax.random.key(1), (256, 1), -8, 8)
    )
    moved = grouped_glu._scales_in_pairs(_tiled(x).scales)
    want = _tiled(x[grouped_cudnn.halves_of(jnp.arange(256))])
    np.testing.assert_array_equal(
        blockscaled.as_bytes(moved), blockscaled.as_bytes(want.scales)
    )


@pytest.mark.parametrize("q", [1, 2])
def test_the_weights_quantize_with_their_hidden_units_padded_by_zero(stand_in, q):
    """Nemotron 3 Nano's 1856 hidden units as 1920: a GLU's halves of 32 units in
    each 64 rows keep the real ones first."""
    w = grouped_glu.in_mxfp8(jnp.ones((2, D, 1856, q)), jnp.ones((2, 1856, D)))
    up = (w.up_fprop.values, jnp.swapaxes(w.up_dgrad.values, 1, 2))
    down = (w.down_dgrad.values, jnp.swapaxes(w.down_fprop.values, 1, 2))
    for by_unit, real in [*((u, 1856 * q) for u in up), *((d, 1856) for d in down)]:
        units = np.asarray(by_unit.astype(jnp.float32))
        assert units.shape[1] == real // 1856 * 1920
        assert units[:, :real].all() and not units[:, real:].any()


# --- Which FFNs the kernels fuse -----------------------------------------------------


@pytest.mark.parametrize("name", list(FFNS))
def test_the_kernels_fuse_the_activations_cudnn_has(name):
    assert grouped_glu.epilogue(FFNS[name]) == EPILOGUES[name]


@pytest.mark.parametrize(
    ("ffn", "want"),
    [
        (SWIGLU.model_copy(update={"up_offset": 1.0}), Epilogue("geglu", offset=1.0)),
        (SWIGLU.model_copy(update={"activation": "gelu"}), None),
        (SWIGLU.model_copy(update={"up_softcap": 2.0}), None),
        (SITU.model_copy(update={"up_softcap": None}), None),
        (RELU2.model_copy(update={"limit": 2.0}), None),
        (RELU2.model_copy(update={"activation": "gelu"}), None),
    ],
)
def test_the_kernels_fuse_no_other_activation(ffn, want):
    assert grouped_glu.epilogue(ffn) == want


def test_the_fused_mlp_runs_in_mxfp8_alone_with_no_bias_and_whole_blocks(stand_in):
    up = jnp.zeros((E, D, F, 2))
    passes = blockscaled.PASSES
    for ffn in FFNS.values():
        assert grouped_glu.runs(blockscaled.MXFP8, passes, ffn, up)
    assert not grouped_glu.runs(blockscaled.MXFP8_2D, passes, SWIGLU, up)
    assert not grouped_glu.runs(blockscaled.MXFP8, passes[1:], SWIGLU, up)
    biased = GEGLU.model_copy(update={"bias": True})
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, biased, up)
    gelu = SWIGLU.model_copy(update={"activation": "gelu"})
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, gelu, up)
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, SWIGLU, up[:, :96])
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, SWIGLU, up[:, :, :240])
    # Nemotron 3 Nano's experts, d 2688 and 1856 hidden units, padded to 1920.
    nano = jnp.zeros((128, 2688, 1856, 1))
    assert grouped_glu.runs(blockscaled.MXFP8, passes, RELU2, nano)


# --- The MLP, against bux's path without it -------------------------------------------


def _sorted_case(ffn: cfg.MLPConfig = SWIGLU, tokens: int = 64, f: int = F):
    ks = jax.random.split(jax.random.key(2), 5)
    logits = jax.random.normal(ks[0], (tokens, E))
    top, idx = jax.lax.top_k(logits, K)
    weights = jax.nn.softmax(top)
    z = jax.random.normal(ks[1], (tokens, D))
    p = 2 if ffn.type == "gated_mlp" else 1
    up = jax.random.normal(ks[2], (E, D, f, p)) / D**0.5
    down = jax.random.normal(ks[3], (E, f, D)) / f**0.5
    dy = jax.random.normal(ks[4], (tokens, D))

    def run(precision, z, weights, up, down):
        with matmuls.use(precision):
            kernels = moe.Kernels(up, down)
            return moe.routed(z[None], idx[None], weights[None], kernels, ffn)[0]

    return run, (z, weights, up, down), dy


def _rows_case(ffn: cfg.MLPConfig = SWIGLU, rows: int = 96, filled: int = 80, f=F):
    ks = jax.random.split(jax.random.key(3), 5)
    sizes = jnp.array([30, 0, 41, 9], jnp.int32)
    assert int(sizes.sum()) == filled
    x = jax.random.normal(ks[0], (rows, D))
    weights = jax.random.uniform(ks[1], (rows,))
    p = 2 if ffn.type == "gated_mlp" else 1
    up = jax.random.normal(ks[2], (E, D, f, p)) / D**0.5
    down = jax.random.normal(ks[3], (E, f, D)) / f**0.5
    dy = jax.random.normal(ks[4], (rows, D))

    def run(precision, x, weights, up, down):
        with matmuls.use(precision):
            return moe.experts(x, sizes, weights, moe.Kernels(up, down), ffn)

    return run, (x, weights, up, down), dy


def _with_grads(run, precision, xs, dy):
    y, vjp = jax.vjp(functools.partial(run, precision), *xs)
    return [np.asarray(a, np.float32) for a in (y, *vjp(dy))]


def _error(got, want):
    return np.linalg.norm(got - want) / np.linalg.norm(want)


def _close(got, want, tolerance):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert np.isfinite(got).all()
    assert _error(got, want) < tolerance, _error(got, want)


def _fused_against_before(case, monkeypatch):
    """The fused MLP's output and gradients as near bfloat16's as bux's path without
    it, and near that path's: the two round to MXFP8 at their own places, so they lie
    as far apart as each from bfloat16's (half again, at these few values; farther
    under a clamp, whose edge their roundings cross apart)."""
    run, xs, dy = case
    fused = _with_grads(run, "mxfp8", xs, dy)
    monkeypatch.setattr(grouped_glu, "runs", lambda *a, **k: False)
    before = _with_grads(run, "mxfp8", xs, dy)
    exact = _with_grads(run, "compute", xs, dy)
    for got, was, want in zip(fused, before, exact, strict=True):
        assert got.shape == want.shape
        assert np.isfinite(got).all()
        assert _error(got, want) < 1.25 * _error(was, want) + 1e-3
        assert _error(got, was) < max(0.06, 1.5 * _error(was, want))


@pytest.mark.parametrize("case", [_sorted_case, _rows_case], ids=["sorted", "rows"])
@pytest.mark.parametrize("name", list(FFNS))
def test_the_fused_mlp_computes_what_the_experts_did_within_mxfp8(
    stand_in, monkeypatch, case, name
):
    _fused_against_before(case(FFNS[name]), monkeypatch)


@pytest.mark.parametrize("name", ["relu2", "geglu"])
def test_hidden_units_not_whole_tiles_take_their_gradients_unpadded(
    stand_in, monkeypatch, name
):
    case = _sorted_case(FFNS[name], f=192)
    _fused_against_before(case, monkeypatch)


def test_rows_past_the_experts_come_out_zero_and_take_no_gradient(stand_in):
    run, xs, dy = _rows_case()
    y, dx, dweights, *_ = _with_grads(run, "mxfp8", xs, dy)
    assert (np.asarray(y[80:]) == 0).all()
    assert (np.asarray(dx[80:]) == 0).all() and (np.asarray(dweights[80:]) == 0).all()


def _layer_outputs(mesh: dict[str, int], ffn: cfg.MLPConfig):
    """An MoE layer's output and gradients by `sorted` in MXFP8, on `mesh`."""
    x = jax.random.normal(jax.random.key(8), (8, 16, D))
    probe = jax.random.normal(jax.random.key(9), x.shape)
    config = cfg.MoEConfig(num_experts=8, top_k=2, expert=ffn)
    with parallel.use(tcfg.MeshConfig.model_validate(mesh)), matmuls.use("mxfp8"):
        layer = parallel.build(lambda: MoE(config, D, nnx.Rngs(0)))
        loss = lambda e, x: (e(x, Taps()) * probe).sum()
        y = nnx.jit(lambda e, x: e(x, Taps()))(layer, x)
        grads = nnx.jit(nnx.grad(loss, argnums=(0, 1)))(layer, x)
        return jax.device_get((y, grads))


@pytest.mark.skipif(jax.device_count() < 8, reason="8 devices, as the CPU fakes")
@pytest.mark.parametrize("name", ["swiglu", "relu2"])
def test_each_device_quantizes_its_own_shard_of_the_experts_as_one_does_them_all(
    stand_in, monkeypatch, name
):
    gathered = []
    real = grouped_glu.gathered
    monkeypatch.setattr(
        grouped_glu, "gathered", lambda *a: gathered.append(a[2]) or real(*a)
    )
    ffn = FFNS[name].model_copy(update={"intermediate_size": 192})  # padded to 256
    (y, grads), (y_split, grads_split) = (
        _layer_outputs({"data": 1}, ffn),
        _layer_outputs({"fsdp": 8}, ffn),
    )
    assert gathered and all(axes == ("fsdp",) for axes in gathered)
    _close(y_split, y, 1e-6)
    # Each device's gradients in bfloat16, summed over the devices.
    for got, want in zip(jax.tree.leaves(grads_split), jax.tree.leaves(grads)):
        _close(got, want, 1e-2)


# --- On a Blackwell GPU: the kernels against the stand-ins --------------------------


def _gpu_operands(m: int = 1024, groups: int = 4, n: int = 2 * F, k: int = D):
    ks = jax.random.split(jax.random.key(4), 4)
    sizes = jnp.array([256, 0, 512, 256], jnp.int32)
    ends = jnp.cumsum(sizes).astype(jnp.int32)
    x = jax.random.normal(ks[0], (m, k)).astype(jnp.bfloat16)
    w = (jax.random.normal(ks[1], (groups, n, k)) / k**0.5).astype(jnp.bfloat16)
    weights = jax.random.uniform(ks[2], (m,))
    return x, w, ends, weights


def _values(a: Tiled | Columns) -> Float[Array, "m n"]:
    """Return what an operand holds, rows first."""
    if isinstance(a, Columns):
        return _dequantized(Tiled(a.values.T, a.scales)).T
    return _dequantized(a)


def _same_values(got: Tiled | Columns, want: Tiled | Columns, live: int):
    _close(_values(got)[:live], _values(want)[:live], 0.02)


@blackwell
@pytest.mark.parametrize("name", list(EPILOGUES))
def test_each_activation_kernel_computes_what_its_stand_in_does(name):
    epilogue = EPILOGUES[name]
    n = 2 * F if epilogue.gated else F
    x, w, ends, weights = _gpu_operands(n=n)
    run = functools.partial(grouped_cudnn.activated, epilogue=epilogue)
    got = jax.jit(run)(_tiled(x), _tiled(w), ends, weights)
    want = _activated(_tiled(x), _tiled(w), ends, weights, epilogue)
    live = int(ends[-1])
    _close(got.product[:live], want.product[:live], 1e-2)
    _same_values(got.h, want.h, live)
    _same_values(got.h_columns, want.h_columns, live)


@blackwell
@pytest.mark.parametrize("name", list(EPILOGUES))
def test_each_gradient_kernel_computes_what_its_stand_in_does(name):
    epilogue = EPILOGUES[name]
    n = 2 * F if epilogue.gated else F
    x, w, ends, weights = _gpu_operands(n=n)
    product = _activated(_tiled(x), _tiled(w), ends, weights, epilogue).product
    dy = _tiled(x)
    down = _tiled(w[:, :F])  # (g, f, d): down's rows along d
    run = functools.partial(grouped_cudnn.activation_grad, epilogue=epilogue)
    got = jax.jit(run)(dy, down, product, ends, weights)
    want = _activation_grad(dy, down, product, ends, weights, epilogue)
    live = int(ends[-1])
    _same_values(got.product, want.product, live)
    _same_values(got.product_columns, want.product_columns, live)
    _close(got.weights[:live], want.weights[:live], 1e-2)


@blackwell
def test_the_plain_kernels_compute_what_their_stand_ins_do():
    x, w, ends, _ = _gpu_operands()
    a, b = _tiled(x), _tiled(w)
    live = int(ends[-1])
    _close(
        jax.jit(grouped_cudnn.rows)(a, b, ends)[:live], _rows(a, b, ends)[:live], 1e-2
    )
    dy = _tiled(jax.random.normal(jax.random.key(5), (2 * F, x.shape[0])))
    for first in (_tiled(x.T), _columns(x.astype(jnp.float32))):
        for shape in (None, (D - 64, 2 * F - 64)):  # the stores stop at the edge
            run = functools.partial(grouped_cudnn.contract, shape=shape)
            got = jax.jit(run)(first, dy, ends)
            _close(got, _contract(first, dy, ends, shape), 1e-2)


@blackwell
@pytest.mark.parametrize("halves", [False, True])
def test_triton_pads_its_copies_by_zero(halves):
    from bux.architecture.experts import grouped_triton

    w = jax.random.normal(jax.random.key(6), (3, 384, 448), jnp.bfloat16)
    shape = (512, 640)
    got = grouped_triton.quantized_twice(w, jnp.int32(384), halves=halves, padded=shape)
    want = _quantized_twice(w, 384, halves=halves, padded=shape)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(blockscaled.as_bytes(a), blockscaled.as_bytes(b))


@blackwell
def test_bux_s_rows_kernel_reads_groups_at_their_starts_and_packs_its_output():
    x, w, ends, _ = _gpu_operands()
    sizes = jnp.array([200, 0, 300, 1], jnp.int32)
    starts = ends - jnp.array([256, 0, 512, 256], jnp.int32)
    kept = jnp.zeros(x.shape[0], bool)
    for start, size in zip(starts, sizes, strict=True):
        kept = kept | (
            (jnp.arange(x.shape[0]) >= start) & (jnp.arange(x.shape[0]) < start + size)
        )
    a, b = _tiled(jnp.where(kept[:, None], x, 0)), _tiled(w)
    run = functools.partial(gm.rows, out=jnp.bfloat16, starts=starts, height=600)
    got = jax.jit(run)(a, b, sizes)
    want = _packed_rows(a, b, sizes, jnp.bfloat16, starts=starts, height=600)
    _close(got[:501], want[:501], 1e-2)
    assert (np.asarray(got[501:]) == 0).all()


@blackwell
def test_triton_reads_pairs_as_halves_and_transposes_them_back():
    from bux.architecture.experts import grouped_triton

    w = jax.random.normal(jax.random.key(6), (3, 384, 512), jnp.bfloat16)
    got = grouped_triton.quantized_twice(w, jnp.int32(384), halves=True)
    want = _quantized_twice(w, 384, halves=True)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(blockscaled.as_bytes(a), blockscaled.as_bytes(b))
    values = got[0].values[0]
    np.testing.assert_array_equal(
        blockscaled.as_bytes(grouped_triton.transposed(values, pairs=True)),
        blockscaled.as_bytes(_transposed(values, pairs=True)),
    )


@blackwell
def test_triton_reads_rows_at_minus_one_as_zero():
    from bux.architecture.experts import grouped_triton

    tokens = jax.random.normal(jax.random.key(7), (300, 512), jnp.bfloat16)
    index = jnp.where(jnp.arange(1024) % 3 == 0, -1, jnp.arange(1024) % 300)
    got = grouped_triton.quantized_twice(tokens, jnp.int32(1024), index=index)
    want = _quantized_twice(tokens, 1024, index=index)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(blockscaled.as_bytes(a), blockscaled.as_bytes(b))


@blackwell
@pytest.mark.parametrize("case", [_sorted_case, _rows_case], ids=["sorted", "rows"])
@pytest.mark.parametrize("name", list(FFNS))
def test_the_fused_kernels_compute_what_the_experts_did_within_mxfp8(
    monkeypatch, case, name
):
    assert kernels.gpu()
    _fused_against_before(case(FFNS[name]), monkeypatch)


@blackwell
@pytest.mark.parametrize("name", ["relu2", "geglu"])
def test_the_fused_kernels_take_hidden_units_not_whole_tiles(monkeypatch, name):
    case = _sorted_case(FFNS[name], f=192)
    _fused_against_before(case, monkeypatch)
