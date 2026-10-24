"""The experts' gated MLP by cuDNN's fused kernels (bux.architecture.experts.grouped_glu
and grouped_cudnn). On the CPU, with stand-ins for the kernels that compute what they
do by XLA from the operands as quantized: the layout of each expert's rows from a
multiple of 256, the GLU's halves, and the MLP's forward and gradients against bux's
path without it, within MXFP8's rounding. On a Blackwell GPU, the kernels against the
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
from bux.architecture.experts.grouped_cudnn import Columns
from bux.architecture.experts.grouped_mxfp8 import Tiled
from bux.architecture.experts.moe import MoE
from bux.architecture.precision import blockscaled, matmuls
from bux.train import config as tcfg

SWIGLU = cfg.MLPConfig(type="gated_mlp", intermediate_size=256, activation="swish")
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


def _quantized_twice(x, filled, columns=True, index=None, halves=False):
    if index is not None:
        x = jnp.where((index >= 0)[:, None], x[jnp.maximum(index, 0)], 0)
    x = jnp.where((jnp.arange(x.shape[-2]) < filled)[:, None], x, 0)
    if halves:
        x = x[..., _pairs_of(jnp.arange(x.shape[-1]))]
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


def _swiglu(x, w, ends, weights):
    product = _times(x, w, ends)
    gate, up = _gate_and_up(product)
    h = up * jax.nn.silu(gate) * weights[:, None]
    return grouped_cudnn.Glu(product.astype(jnp.bfloat16), _tiled(h), _columns(h))


def _dswiglu(dy, w, product, ends, weights):
    dh = _times(dy, w, ends)
    gate, up = _gate_and_up(product.astype(jnp.float32))
    sig = jax.nn.sigmoid(gate)
    d_up = dh * gate * sig * weights[:, None]
    d_gate = dh * up * sig * (1 + gate * (1 - sig)) * weights[:, None]
    m, f = dh.shape
    halves = jnp.stack([d_gate.reshape(m, -1, 32), d_up.reshape(m, -1, 32)], 2)
    dp = halves.reshape(m, 2 * f)
    dweights = (dh * up * gate * sig).sum(-1)
    return grouped_cudnn.GluGrad(_tiled(dp), _columns(dp), dweights)


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


def _contract(a, b, ends):
    if isinstance(a, Columns):
        a = Tiled(a.values.T, a.scales)
    rows = (_dequantized(a).T, _dequantized(b).T)
    dw = jax.lax.ragged_dot_general(*rows, _sizes(ends), grouped.RAGGED_CONTRACTING)
    return dw.astype(jnp.bfloat16)


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
        ("swiglu", _swiglu),
        ("dswiglu", _dswiglu),
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


# --- The MLP, against bux's path without it -------------------------------------------


def _sorted_case(tokens: int = 64):
    ks = jax.random.split(jax.random.key(2), 5)
    logits = jax.random.normal(ks[0], (tokens, E))
    top, idx = jax.lax.top_k(logits, K)
    weights = jax.nn.softmax(top)
    z = jax.random.normal(ks[1], (tokens, D))
    up = jax.random.normal(ks[2], (E, D, F, 2)) / D**0.5
    down = jax.random.normal(ks[3], (E, F, D)) / F**0.5
    dy = jax.random.normal(ks[4], (tokens, D))

    def run(precision, z, weights, up, down):
        with matmuls.use(precision):
            return moe.routed(z[None], idx[None], weights[None], up, down, SWIGLU)[0]

    return run, (z, weights, up, down), dy


def _rows_case(rows: int = 96, filled: int = 80):
    ks = jax.random.split(jax.random.key(3), 5)
    sizes = jnp.array([30, 0, 41, 9], jnp.int32)
    assert int(sizes.sum()) == filled
    x = jax.random.normal(ks[0], (rows, D))
    weights = jax.random.uniform(ks[1], (rows,))
    up = jax.random.normal(ks[2], (E, D, F, 2)) / D**0.5
    down = jax.random.normal(ks[3], (E, F, D)) / F**0.5
    dy = jax.random.normal(ks[4], (rows, D))

    def run(precision, x, weights, up, down):
        with matmuls.use(precision):
            return moe.experts(x, sizes, weights, up, down, SWIGLU)

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
    it, and near that path's."""
    run, xs, dy = case
    fused = _with_grads(run, "mxfp8", xs, dy)
    monkeypatch.setattr(grouped_glu, "runs", lambda *a, **k: False)
    before = _with_grads(run, "mxfp8", xs, dy)
    exact = _with_grads(run, "compute", xs, dy)
    for got, was, want in zip(fused, before, exact, strict=True):
        assert np.isfinite(got).all()
        assert _error(got, want) < 1.25 * _error(was, want) + 1e-3
        assert _error(got, was) < 0.06


@pytest.mark.parametrize("case", [_sorted_case, _rows_case], ids=["sorted", "rows"])
def test_the_fused_mlp_computes_what_the_experts_did_within_mxfp8(
    stand_in, monkeypatch, case
):
    _fused_against_before(case(), monkeypatch)


def test_rows_past_the_experts_come_out_zero_and_take_no_gradient(stand_in):
    run, xs, dy = _rows_case()
    y, dx, dweights, *_ = _with_grads(run, "mxfp8", xs, dy)
    assert (np.asarray(y[80:]) == 0).all()
    assert (np.asarray(dx[80:]) == 0).all() and (np.asarray(dweights[80:]) == 0).all()


def test_the_fused_mlp_runs_a_plain_swiglu_in_mxfp8_alone(stand_in):
    up = jnp.zeros((E, D, F, 2))
    passes = blockscaled.PASSES
    assert grouped_glu.runs(blockscaled.MXFP8, passes, SWIGLU, up)
    assert not grouped_glu.runs(blockscaled.MXFP8_2D, passes, SWIGLU, up)
    assert not grouped_glu.runs(blockscaled.MXFP8, passes[1:], SWIGLU, up)
    clamped = SWIGLU.model_copy(update={"limit": 7.0})
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, clamped, up)
    gelu = SWIGLU.model_copy(update={"activation": "gelu"})
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, gelu, up)
    assert not grouped_glu.runs(blockscaled.MXFP8, passes, SWIGLU, up[:, :128])


def _layer_outputs(mesh: dict[str, int]):
    """An MoE layer's output and gradients by `sorted` in MXFP8, on `mesh`."""
    x = jax.random.normal(jax.random.key(8), (8, 16, D))
    probe = jax.random.normal(jax.random.key(9), x.shape)
    config = cfg.MoEConfig(num_experts=8, top_k=2, expert=SWIGLU)
    with parallel.use(tcfg.MeshConfig.model_validate(mesh)), matmuls.use("mxfp8"):
        layer = parallel.build(lambda: MoE(config, D, nnx.Rngs(0)))
        loss = lambda e, x: (e(x, Taps()) * probe).sum()
        y = nnx.jit(lambda e, x: e(x, Taps()))(layer, x)
        grads = nnx.jit(nnx.grad(loss, argnums=(0, 1)))(layer, x)
        return jax.device_get((y, grads))


def test_each_device_quantizes_its_own_shard_of_the_experts_as_one_does_them_all(
    stand_in, monkeypatch
):
    gathered = []
    real = grouped_glu.gathered
    monkeypatch.setattr(
        grouped_glu, "gathered", lambda *a: gathered.append(a[2]) or real(*a)
    )
    (y, grads), (y_split, grads_split) = (
        _layer_outputs({"data": 1}),
        _layer_outputs({"fsdp": 8}),
    )
    assert gathered and all(axes == ("fsdp",) for axes in gathered)
    _close(y_split, y, 1e-6)
    # Each device's gradients in bfloat16, summed over the devices.
    for got, want in zip(jax.tree.leaves(grads_split), jax.tree.leaves(grads)):
        _close(got, want, 1e-2)


# --- On a Blackwell GPU: the kernels against the stand-ins --------------------------


def _gpu_operands(m: int = 1024, groups: int = 4):
    ks = jax.random.split(jax.random.key(4), 4)
    sizes = jnp.array([256, 0, 512, 256], jnp.int32)
    ends = jnp.cumsum(sizes).astype(jnp.int32)
    x = jax.random.normal(ks[0], (m, D)).astype(jnp.bfloat16)
    w = (jax.random.normal(ks[1], (groups, 2 * F, D)) / D**0.5).astype(jnp.bfloat16)
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
def test_the_swiglu_kernel_computes_what_its_stand_in_does():
    x, w, ends, weights = _gpu_operands()
    got = jax.jit(grouped_cudnn.swiglu)(_tiled(x), _tiled(w), ends, weights)
    want = _swiglu(_tiled(x), _tiled(w), ends, weights)
    live = int(ends[-1])
    _close(got.product[:live], want.product[:live], 1e-2)
    _same_values(got.h, want.h, live)
    _same_values(got.h_columns, want.h_columns, live)


@blackwell
def test_the_dswiglu_kernel_computes_what_its_stand_in_does():
    x, w, ends, weights = _gpu_operands()
    product = _swiglu(_tiled(x), _tiled(w), ends, weights).product
    dy = _tiled(x)
    down = _tiled(w[:, :F])  # (g, f, d): down's rows along d
    got = jax.jit(grouped_cudnn.dswiglu)(dy, down, product, ends, weights)
    want = _dswiglu(dy, down, product, ends, weights)
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
        got = jax.jit(grouped_cudnn.contract)(first, dy, ends)
        _close(got, _contract(first, dy, ends), 1e-2)


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
def test_the_fused_kernels_compute_what_the_experts_did_within_mxfp8(monkeypatch, case):
    assert kernels.gpu()
    _fused_against_before(case(), monkeypatch)
