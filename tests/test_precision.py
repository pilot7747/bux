"""Precision and microbatches: runs that compute in bfloat16 on float32 parameters, the
float32 islands inside them, and gradient accumulation against the whole batch, on one
device and split over the 8 CPU devices the tests have."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import count
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture import Model
from bux.architecture.experts import grouped_blockscaled
from bux.architecture.experts.moe import MoE
from bux.architecture.layers.norm import Norm
from bux.architecture.precision import blockscaled, matmuls, nvfp4
from bux.architecture.recurrent.linear_attention import delta_rule, l2norm, recurrence
from bux.architecture.recurrent.ssm import selective_scan
from bux.architecture.registry import Taps
from bux.architecture.repeats import tracing
from bux.lang import BuxError, compile_runs
from bux.train import config as tcfg
from bux.train.data import Batch, plain
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer, cast, forward, train

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
"""


def _with(mixer: str) -> str:
    body = "model = embed [norm a] [norm ffn] [norm a] [norm ffn] norm head"
    return PRELUDE + f"a = {mixer}\n" + body


EXPERTS = (
    "num_experts=8, top_k=2, expert=gated_mlp(intermediate_size=64, activation=swish)"
)
ARCHITECTURES = {
    "llama": _with("attn"),
    "moe": _with(f"moe({EXPERTS}, num_shared_experts=1, load_balancing=aux_free)"),
    "moe_sequence_loss": _with(f"moe({EXPERTS}, aux_loss_scope=sequence)"),
    "mla": _with("mla(num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8)"),
    "gdn": _with("gdn(num_heads=2, head_dim=16)"),
    "rwkv7": _with("rwkv7(num_heads=2, head_dim=16)"),
    "mamba": _with("mamba(state_size=4)"),
    "mamba2": _with("mamba2(state_size=4, head_dim=16)"),
}
# Every other family, for the stream's dtype.
FAMILIES = ARCHITECTURES | {
    "mqa_gated": _with(
        "gqa(num_heads=4, num_kv_heads=1, rope=rope(), gated=true, sinks=true,"
        " kv_conv_size=3, relative_bias=relbias(rank=2, extent=4))"
    ),
    "mla_indexer": _with(
        "mla(num_heads=4, kv_lora_rank=16, qk_rope_head_dim=8,"
        " indexer=indexer(num_heads=2, head_dim=8, top_k=3))"
    ),
    "csa": _with(
        "csa(num_heads=4, head_dim=16, q_lora_rank=12, o_lora_rank=6, o_groups=2,"
        " sliding_window_size=3, sinks=true, rope=rope(fraction=0.25))"
    ),
    "kda": _with("kda(num_heads=2, head_dim=8, gate_rank=4, output_gate_rank=4)"),
    "deltanet": _with("deltanet(num_heads=2, head_dim=8)"),
    "gla": _with("gla(num_heads=2, head_dim=8, gate_rank=4)"),
    "lightning": _with("lightning(num_heads=2, head_dim=8, v_head_dim=4)"),
    "short_conv": _with("short_conv"),
    "latent_moe": _with(
        f"moe({EXPERTS}, router=sigmoid, latent_size=16, latent_output_norm=norm,"
        " expert_norm=norm, shared_expert_gate=true, num_shared_experts=1,"
        " shared_expert_sink=true)"
    ),
    "engram": _with("engram(head_dim=4, table_size=50)"),
    "dyt": _with("layernorm").replace("norm = rmsnorm", "norm = dyt"),
    "mhc": "residual = mhc(expansion=2)\n"
    + _with("attn").replace("norm head", "merge norm head"),
    "nextn": PRELUDE
    + "mtp = $h! nextn [norm ffn] norm @h head\n"
    + "model = embed [norm attn] [norm ffn] norm @h head mtp^2",
    "tied": _with("attn").replace("norm head", "norm head(tied=true, softcap=30)"),
}
BF16 = "precision(compute=bfloat16)"
ONE = "mesh(data=1)"
STABLE = "adamw(lr=1e-3)"  # at 1e-2, these small runs are chaotic enough to drift


# --- Precision ------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_a_run_in_bfloat16_trains_near_float32(losses, arch):
    f32 = losses(ARCHITECTURES[arch], mesh=ONE, optimizer=STABLE)
    bf16 = losses(ARCHITECTURES[arch], mesh=ONE, optimizer=STABLE, precision=BF16)
    assert np.allclose(bf16, f32, rtol=5e-3) and not np.array_equal(bf16, f32)


# XLA:CPU fails to compile a bfloat16 psum out of a shard_map's transpose, which the
# MoE's sorted and EP methods have on more than one device: its reducer ends in a copy
# that ChangeOpDataType cannot clone. Dense runs the experts without one.
@pytest.mark.parametrize("arch", ["llama", "moe"])
def test_bfloat16_on_a_split_mesh_trains_as_on_one_device(losses, involuntary, arch):
    source = ARCHITECTURES[arch]
    given = {"optimizer": STABLE, "precision": BF16, "methods": "methods(moe=dense)"}
    one = losses(source, mesh=ONE, **given)
    involuntary()
    for mesh in ("mesh(fsdp=8)", "mesh(fsdp=2, expert=2, tensor=2)"):
        assert np.allclose(losses(source, mesh=mesh, **given), one, rtol=5e-3)
    assert involuntary() == 0


def test_parameters_gradients_and_the_optimizer_state_stay_float32():
    trainer = _trainer(ARCHITECTURES["moe"], compute=jnp.bfloat16)
    grads, _ = nnx.jit(trainer.gradients)(trainer.model, _batch())
    trainer.train_on_batch(_batch())
    params = nnx.state(trainer.model, nnx.Param)
    moments = nnx.state(trainer.optimizer, nnx.OptState)
    floats = [
        x
        for x in jax.tree.leaves((grads, params, moments))
        if jnp.issubdtype(x.dtype, jnp.floating)
    ]
    assert floats and all(x.dtype == jnp.float32 for x in floats)


def test_eval_computes_in_bfloat16_near_float32():
    f32 = _trainer(ARCHITECTURES["llama"]).evaluate([_batch()])["loss"]
    bf16 = _trainer(ARCHITECTURES["llama"], compute=jnp.bfloat16).evaluate([_batch()])
    bf16 = bf16["loss"]
    assert bf16 == pytest.approx(f32, rel=1e-2) and bf16 != f32


@pytest.mark.parametrize("family", FAMILIES)
def test_a_bfloat16_forward_keeps_the_stream_in_bfloat16(family):
    model = cast(_trainer(FAMILIES[family]).model, jnp.bfloat16)
    heads = forward(model, _batch())
    assert all(head.hidden.dtype == jnp.bfloat16 for head in heads)
    assert all(head.projection.kernel.dtype == jnp.bfloat16 for head in heads)
    assert all(head.logits().dtype == jnp.bfloat16 for head in heads)


ISLANDS = (
    "rmsnorm",
    "layernorm",
    "l2norm",
    "recurrence",
    "delta_rule",
    "selective_scan",
)


def _islands() -> dict:
    """Each float32 island, and arrays it takes."""
    x = jax.random.normal(jax.random.key(0), (2, 16, 2, 8))
    decay = -jax.nn.softplus(jax.random.normal(jax.random.key(1), x.shape)) / 8
    beta = jax.nn.sigmoid(x[..., 0])
    dt = jax.nn.softplus(x[..., 0])
    A = -jnp.arange(1.0, 3.0)[:, None]
    rms = Norm(cfg.NormalizationConfig(type="rmsnorm"), 8)
    layer = Norm(cfg.NormalizationConfig(type="layernorm"), 8)
    return {
        "rmsnorm": (rms.normalize, (x,)),
        "layernorm": (layer.normalize, (x,)),
        "l2norm": (l2norm, (x,)),
        "recurrence": (lambda *a: recurrence(*a, None), (x, x, x, decay)),
        "delta_rule": (lambda *a: delta_rule(*a, None), (x, x, x, decay, beta)),
        "selective_scan": (selective_scan, (x, dt, A, x[:, :, :1], x[:, :, :1])),
    }


@pytest.mark.parametrize("island", ISLANDS)
def test_a_float32_island_computes_bfloat16_inputs_in_float32(island):
    fn, args = _islands()[island]
    low = [a.astype(jnp.bfloat16) for a in args]
    y = fn(*low)
    assert y.dtype == jnp.bfloat16
    assert jnp.array_equal(y, fn(*(a.astype(jnp.float32) for a in low)).astype(y.dtype))


def test_the_router_picks_in_float32_what_bfloat16_would_tie():
    ffn = cfg.MLPConfig(type="gated_mlp", intermediate_size=8, activation="swish")
    c = cfg.MoEConfig(num_experts=4, top_k=1, router="sigmoid", expert=ffn)
    moe = MoE(c, 16, nnx.Rngs(0))
    # Expert 1's logit is 2^-10 above expert 0's, below bfloat16's resolution at 1.
    kernel = jnp.zeros((16, 4)).at[:, :2].set(1 / 16).at[0, 1].add(2**-10)
    moe.router.kernel[...] = kernel
    x = jnp.ones((1, 1, 16), jnp.bfloat16)
    tied = x @ kernel.astype(jnp.bfloat16)
    assert tied[0, 0, 0] == tied[0, 0, 1]
    idx, weights, _ = cast(moe, jnp.bfloat16)._route(x)
    assert idx[0, 0, 0] == 1 and weights.dtype == jnp.bfloat16


def test_precision_is_in_the_run_s_id_and_a_sweep_over_it_is_two_runs():
    f32, bf16 = compile_runs(_source("precision.compute = float32 | bfloat16"))
    assert bf16.train.precision.compute == "bfloat16"
    assert f32.id != bf16.id and f32.name != bf16.name
    [plain] = compile_runs(_source())
    assert plain.train.precision.compute == "bfloat16" and plain.id == bf16.id


# --- Matmul formats ---------------------------------------------------------------------

PROJECTION = (((2,), (0,)), ((), ()))  # x @ w, as nnx.Linear multiplies
# Each format's relative error, forward and backward, against float32 at random
# normal operands (float8 rounds to 1 part in 16, float4 to 1 in 4), and how far a few
# steps' losses may be from bfloat16's.
FORMATS = {"mxfp8": (0.02, 0.06), "mxfp8_2d": (0.02, 0.06), "nvfp4": (0.08, 0.25)}
LOSSES = {"mxfp8": 1e-2, "mxfp8_2d": 1e-2, "nvfp4": 3e-2}


def _precision(fmt: str) -> str:
    return f"precision(compute=bfloat16, matmuls={fmt})"


def _projection(
    fmt: str | Callable, shape: tuple[int, int, int], dtype=jnp.bfloat16
) -> list:
    """Return x @ w in the format `fmt`, or by a dot_general, and its gradients by x
    and w."""
    multiply = matmuls.formats.find(fmt) if isinstance(fmt, str) else fmt
    b, k, n = shape
    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(keys[0], (b, 32, k)).astype(dtype)
    w = (jax.random.normal(keys[1], (k, n)) * k**-0.5).astype(dtype)
    g = jax.random.normal(keys[2], (b, 32, n)).astype(dtype)
    y, vjp = jax.vjp(lambda x, w: multiply(x, w, PROJECTION), x, w)
    return [y, *vjp(g)]


def _error(a: Float[Array, "*s"], b: Float[Array, "*s"]) -> float:
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_format_multiplies_forward_and_backward_near_float32(fmt):
    shape, (low, high) = (4, 256, 512), FORMATS[fmt]
    exact = _projection("compute", shape, jnp.float32)
    bf16, quantized = _projection("compute", shape), _projection(fmt, shape)
    for y, q, e in zip(bf16, quantized, exact, strict=True):
        assert bool(jnp.isfinite(q).all()) and q.dtype == jnp.bfloat16
        assert _error(y, e) < 0.01 and low < _error(q, e) < high


def test_mxfp8_is_jax_s_scaled_dot_general():
    config = [jax.nn.get_scaled_dot_general_config("mxfp8")] * 3

    def jax_s(
        x: Float[Array, "b t k"], w: Float[Array, "k n"], dimensions: tuple
    ) -> Float[Array, "b t n"]:
        return jax.nn.scaled_dot_general(x, w, dimensions, jnp.bfloat16, config)

    shape = (4, 256, 512)
    for ours, theirs in zip(_projection("mxfp8", shape), _projection(jax_s, shape)):
        assert jnp.array_equal(ours, theirs)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_projection_not_in_whole_blocks_multiplies_in_compute(fmt):
    shape = (2, 40, 64)
    for y, q in zip(_projection("compute", shape), _projection(fmt, shape)):
        assert jnp.array_equal(y, q)


def test_nvfp4_tiles_quantize_w_as_its_transpose():
    w = jax.random.normal(jax.random.key(0), (64, 32))

    def dequantized(w: Float[Array, "r c"]) -> Float[Array, "r c"]:
        return blockscaled.dequantized(nvfp4.quantize(w, tiles=True))

    assert jnp.array_equal(dequantized(w), dequantized(w.T).T)
    assert 0.05 < _error(dequantized(w), w) < 0.2


def test_nvfp4_rounds_gradients_stochastically_and_unbiased():
    # 1.5 at the start of each block takes it to E2M1's 6, and 1.25 to 5, halfway
    # between its 4 and 6.
    x = jnp.full((256, 64), 1.25).at[:, ::16].set(1.5)
    rest = np.arange(64) % 16 != 0

    def values(stochastic: bool = False) -> Float[Array, "r c"]:
        q = nvfp4.quantize(x, stochastic=stochastic)
        return q.values.astype(jnp.float32)[:, rest]

    assert jnp.unique(values()).size == 1
    assert set(jnp.unique(values(stochastic=True)).tolist()) == {4.0, 6.0}
    assert float(values(stochastic=True).mean()) == pytest.approx(5.0, rel=0.01)


def test_stochastic_rounding_rounds_as_blackwells_conversion_does():
    # As a B300's cvt.rs rounded them: up where a value's fraction of a step, rounded
    # up to 8 bits, and its random byte add up to 1, so also where they fall just short.
    v = jnp.array([0.771353, -2.656003, -3.879317, 1.25, 1.25, 1.125, 1.125])
    byte = jnp.array([117, 88, 30, 127, 128, 191, 192])
    rounded = nvfp4.round_stochastically(v, byte * 2.0**-8)
    assert rounded.tolist() == [1.0, -3.0, -4.0, 1.0, 1.5, 1.0, 1.5]


def test_mxfp8_rows_pack_into_bytes_and_back():
    x = jax.random.normal(jax.random.key(0), (8, 64)) * 3
    blocks = blockscaled.mx_quantize(x)
    packed = blockscaled.mx_packed(blocks)
    assert packed.dtype == jnp.uint8 and packed.shape == (8, 64 + 2)
    back = blockscaled.mx_unpacked(packed, 64)
    assert jnp.array_equal(
        blockscaled.dequantized(back), blockscaled.dequantized(blocks)
    )


def test_the_hadamard_transform_is_orthogonal():
    h = nvfp4.HADAMARD
    assert np.allclose(h @ h.T, np.eye(16), atol=1e-6) and set(np.abs(h).flat) == {0.25}


# Interpreted off a GPU: x and x^T in one pass, each device on its part of x.
@pytest.mark.parametrize(
    "spec", [P(), P("data", None), P(None, "data"), P("data", "tensor")]
)
def test_the_mxfp8_kernel_quantizes_as_xla_does_on_any_split(spec):
    x = (jax.random.normal(jax.random.key(0), (256, 384)) * 3).astype(jnp.bfloat16)
    with parallel.use(tcfg.MeshConfig(data=4, tensor=2)) as mesh:
        split = jax.device_put(x, NamedSharding(mesh, spec))
        kernel = jax.jit(blockscaled.mx_kernel_twice)(split)
    xla = blockscaled.mx_quantize(x), blockscaled.mx_quantize(x.T)
    for ours, theirs in zip(kernel, xla, strict=True):
        for a, b in [(ours.values, theirs.values), (ours.scales, theirs.scales)]:
            assert jnp.array_equal(a.astype(jnp.float32), b.astype(jnp.float32))


# A block that its scale takes to E2M1's 6 and the midpoints between its values, which
# round to the even one.
MIDPOINTS = [12, 0.5, 1.5, 2.5, 3.5, 5, 7, 10, -0.5, -1.5, -2.5, -3.5, -5, -7, -10, 0]


# Interpreted off a GPU: x's largest magnitudes, then x and x^T, each device on its
# part of x, the noise of each element's place in the whole.
@pytest.mark.parametrize("kind", [{}, {"tiles": True}, {"stochastic": True}])
@pytest.mark.parametrize(
    "spec", [P(), P("data", None), P(None, "data"), P("data", "tensor")]
)
def test_the_nvfp4_kernels_quantize_as_xla_does_on_any_split(spec, kind):
    x = (jax.random.normal(jax.random.key(0), (256, 384)) * 2).astype(jnp.bfloat16)
    x = x.at[0, :16].set(jnp.array(MIDPOINTS, jnp.bfloat16))
    seed = jnp.uint32(7)
    with parallel.use(tcfg.MeshConfig(data=4, tensor=2)) as mesh:
        split = jax.device_put(x, NamedSharding(mesh, spec))
        kernel = jax.jit(lambda x, s: nvfp4.kernel_twice(x, s, **kind))(split, seed)
    xla = nvfp4.xla_twice(x, seed, **kind)
    for ours, theirs in zip(kernel, xla, strict=True):
        for a, b in [(ours.values, theirs.values), (ours.scales, theirs.scales)]:
            assert jnp.array_equal(a.astype(jnp.float32), b.astype(jnp.float32))
        assert ours.tensor_scale == theirs.tensor_scale


def test_block_projections_multiply_in_the_format_and_the_router_and_head_in_compute():
    model = _trainer(ARCHITECTURES["moe"]).model
    linears = [(p, m) for p, m in nnx.iter_modules(model) if isinstance(m, nnx.Linear)]
    compute = {p[-1] for p, m in linears if m.dot_general is jax.lax.dot_general}
    assert compute == {"router", "proj"}  # proj: the head's

    def run() -> list[Float[Array, "b t vocab"]]:
        with matmuls.use("mxfp8"):
            heads = forward(cast(model, jnp.bfloat16), _batch())
            return [head.logits() for head in heads]

    # The up, gate and down of 2 FFNs and of 2 MoEs' shared experts.
    assert count(jax.make_jaxpr(run)().jaxpr, "scaled_matmul_wrapper") == 12


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("arch", ["llama", "moe"])
def test_a_run_in_a_format_trains_near_bfloat16(losses, arch, fmt):
    given = {"mesh": ONE, "optimizer": STABLE}
    bf16 = losses(ARCHITECTURES[arch], precision=BF16, **given)
    quantized = losses(ARCHITECTURES[arch], precision=_precision(fmt), **given)
    assert np.allclose(quantized, bf16, rtol=LOSSES[fmt])
    assert not np.array_equal(quantized, bf16)


# In float32, the blocks are those of one device; a split mesh sums in another order,
# which moves a few values across a step of float8, so the losses agree to 1e-3.
@pytest.mark.parametrize("fmt", FORMATS)
def test_a_format_on_a_split_mesh_trains_as_on_one_device(losses, fmt):
    given = {
        "optimizer": STABLE,
        "precision": f"precision(compute=float32, matmuls={fmt})",
    }
    one = losses(ARCHITECTURES["llama"], mesh=ONE, **given)
    split = losses(ARCHITECTURES["llama"], mesh="mesh(fsdp=4, tensor=2)", **given)
    assert np.allclose(split, one, rtol=1e-3)


def test_a_pipeline_trains_in_mxfp8_as_its_stages_do_on_one_device(losses):
    source = PRELUDE + "block = [norm attn] [norm ffn]\nmodel = embed block^4 norm head"
    # In float32: XLA:CPU cannot compile a pipeline's bfloat16 psums.
    given = {
        "optimizer": STABLE,
        "precision": "precision(compute=float32, matmuls=mxfp8)",
    }
    scan = losses(source, mesh=ONE, methods="methods(repeat=scan)", **given)
    how = "methods(repeat=scan, microbatches=2)"
    piped = losses(source, mesh="mesh(stage=2, data=1)", methods=how, **given)
    assert np.allclose(piped, scan, rtol=1e-5)


def test_the_matmul_format_is_in_the_run_s_id_and_a_bfloat16_run_s_is_as_it_was():
    plain, mxfp8 = compile_runs(
        _source(f"precision = {BF16}", "precision.matmuls = compute | mxfp8", arch=OLD)
    )
    assert mxfp8.train.precision.matmuls == "mxfp8"
    assert plain.id != mxfp8.id and plain.name != mxfp8.name
    [bf16] = compile_runs(_source(f"precision = {BF16}", arch=OLD))
    assert bf16.id == plain.id == "de94a2eb32"  # as before matmuls


def test_the_reduce_dtype_is_in_the_run_s_id_and_a_run_that_leaves_it_is_as_it_was():
    plain, f32 = compile_runs(
        _source(f"precision = {BF16}", "precision.reduce = compute | float32", arch=OLD)
    )
    assert f32.train.precision.reduce == "float32"
    assert plain.id != f32.id and plain.name != f32.name
    assert plain.id == "de94a2eb32"  # as before reduce


def test_a_format_not_registered_is_an_error():
    trainer = _trainer(ARCHITECTURES["llama"], matmuls="mine")
    hint = (
        r"no 'mine' matmul format yet; "
        r"add one with @bux\.architecture\.precision\.matmuls\.formats"
    )
    with pytest.raises(NotImplementedError, match=hint):
        trainer.train_on_batch(_batch())


# --- Which projections, which branches, which passes ------------------------------------


def test_mxfp8_2d_quantizes_w_as_its_transpose():
    w = jax.random.normal(jax.random.key(0), (64, 96))
    rows, columns = map(blockscaled.dequantized, blockscaled.mx_tiles_twice(w))
    assert jnp.array_equal(rows, columns.T)
    assert 0.01 < _error(rows, w) < 0.06


def _multiplied(passes: tuple[tcfg.Pass, ...]) -> list:
    """Return x @ w and its gradients by x and w in mxfp8, only `passes` in it."""
    with matmuls.use(tcfg.PrecisionConfig(matmuls="mxfp8", passes=passes)):
        return _projection(matmuls.dot_general, (4, 256, 512))


def test_only_the_passes_named_multiply_in_the_format():
    exact, quantized = _projection("compute", (4, 256, 512)), _multiplied(tcfg.PASSES)
    for i, name in enumerate(tcfg.PASSES):
        alone = _multiplied((name,))
        others = _multiplied(tuple(p for p in tcfg.PASSES if p != name))
        assert jnp.array_equal(alone[i], quantized[i])
        assert jnp.allclose(others[i], exact[i], rtol=1e-2, atol=1e-2)
        assert not jnp.allclose(others[i], quantized[i], rtol=1e-3, atol=1e-3)
    for y, e in zip(_multiplied(()), exact, strict=True):
        assert jnp.array_equal(y, e)


# llama: 2 attention branches of q, k, v and o, and 2 FFN branches of up, gate, down.
QUANTIZED = {
    "matmuls=mxfp8": 14,
    "matmuls=mxfp8, roles=roles(kv=compute)": 10,
    "matmuls=nvfp4, roles=roles(q=compute, o=compute, mlp=mxfp8)": 10,
    "matmuls=mxfp8, first=1": 10,
    "matmuls=mxfp8, last=1": 11,
    "matmuls=mxfp8, first=1, last=0.5": 3,
    "roles=roles(mlp=mxfp8)": 6,
}


def _block_scaled(model: Model, precision: tcfg.PrecisionConfig) -> int:
    """How many projections of a forward of `model` multiply in a format."""

    def run_forward() -> list[Float[Array, "b t vocab"]]:
        with matmuls.use(precision):
            return [head.logits() for head in forward(model, _batch())]

    return count(jax.make_jaxpr(run_forward)().jaxpr, "scaled_matmul_wrapper")


@pytest.mark.parametrize("given", QUANTIZED)
def test_roles_and_branches_choose_what_multiplies_in_the_format(given):
    [run] = compile_runs(_source(f"precision = precision({given})"))
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    assert _block_scaled(model, run.train.precision) == QUANTIZED[given]


def test_the_branches_after_the_first_head_take_the_mtp_role():
    [run] = compile_runs(_source(arch=FAMILIES["nextn"]))
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    mxfp8 = tcfg.PrecisionConfig(matmuls="mxfp8")
    mtp = mxfp8.model_copy(update={"roles": tcfg.RolesConfig(mtp="compute")})
    # 4 + 3 in the main model, and nextn's projection and 3 in each MTP pass.
    assert _block_scaled(model, mxfp8) == 7 + 2 * 4
    assert _block_scaled(model, mtp) == 7


def test_every_block_projection_has_a_role():
    unnamed = set()
    for family, source in FAMILIES.items():
        [run] = compile_runs(_source(arch=source))
        model = parallel.build(lambda run=run: Model(run.arch, rngs=nnx.Rngs(0)))
        for path, m in nnx.iter_modules(model):
            if isinstance(m, nnx.Linear) and m.dot_general is matmuls.dot_general:
                unnamed.add((family, path[-1]))
    assert unnamed == {("engram", "kv")}


SCANNED = PRELUDE + "block = [norm attn] [norm ffn]\nmodel = embed block^4 norm head"
PATTERN = (
    PRELUDE
    + "A = [norm attn] [norm ffn]\nmtp = $h! nextn [norm ffn] norm @h head\n"
    + 'model = embed "AA" A^2 norm @h head mtp'
)
BRANCHES = PRELUDE + "model = embed [norm ffn]^4 norm head"  # a pass is a branch
# The residual branches outside any other: before the first head, and all.
NUMBERED = {
    "llama": (FAMILIES["llama"], 4, 4),
    "nextn": (FAMILIES["nextn"], 2, 4),
    "mhc": (FAMILIES["mhc"], 4, 4),
    "scanned": (SCANNED, 8, 8),
    "pattern": (PATTERN, 8, 9),
    "branches": (BRANCHES, 4, 4),
}


@pytest.mark.parametrize("arch", NUMBERED)
def test_branches_are_numbered_as_they_run_scanned_or_not(arch):
    source, main, every = NUMBERED[arch]
    [run] = compile_runs(_source(arch=source))
    tokens = jnp.asarray(_batch().input_ids)
    methods = tcfg.MethodsConfig(repeat="scan")
    with parallel.use(tcfg.MeshConfig(data=1), methods=methods):
        model = Model(run.arch, rngs=nnx.Rngs(0))
        with tracing.forward(model.main_branches):
            model.graph(tokens, Taps(tokens=tokens))
            [after] = tracing.next_branches(1)
    assert (after.main, after.number) == (main, every)


EDGES = [
    (SCANNED, "first=3"),
    (SCANNED, "last=2"),
    (SCANNED, "first=1, last=0.5"),
    (BRANCHES, "first=1, last=2"),
]


def _formats_traced(
    source: str, precision: str, methods: tcfg.MethodsConfig, monkeypatch
) -> dict:
    """Return the format each projection of a forward of `source` multiplies in, by
    its pass of a repeat (as remat counts them) and its role, run by `methods`."""
    [run] = compile_runs(_source(f"precision = {precision}", arch=source))
    seen, of = {}, matmuls.format_of

    def recorded(role: str | None) -> str:
        seen[tracing.current_pass(), role] = fmt = of(role)
        return fmt

    monkeypatch.setattr(matmuls, "format_of", recorded)
    with parallel.use(tcfg.MeshConfig(data=1), methods=methods):
        model = Model(run.arch, rngs=nnx.Rngs(0))
        with matmuls.use(run.train.precision):
            forward(model, _batch())
    monkeypatch.setattr(matmuls, "format_of", of)
    return seen


@pytest.mark.parametrize(("source", "given"), EDGES)
def test_a_scanned_repeat_multiplies_its_first_and_last_as_unrolled(
    losses, monkeypatch, source, given
):
    precision = f"precision(compute=float32, matmuls=mxfp8, {given})"
    how = tcfg.MethodsConfig(repeat="unroll")
    unrolled = _formats_traced(source, precision, how, monkeypatch)
    # A scan traces a run of passes that multiply alike once, as its first.
    how = tcfg.MethodsConfig(repeat="scan")
    scanned = _formats_traced(source, precision, how, monkeypatch)
    assert scanned.items() <= unrolled.items()
    assert set(unrolled.values()) == {"compute", "mxfp8"}
    # XLA fuses what a scanned pass after one in compute quantizes apart, which moves
    # mxfp8's loss by an ulp's rounding: 5e-5, where another format moves it 4e-4.
    run = {"mesh": ONE, "optimizer": STABLE, "steps": 1, "precision": precision}
    [losses_unrolled] = losses(source, **run)
    [losses_scanned] = losses(source, methods="methods(repeat=scan)", **run)
    assert losses_scanned == pytest.approx(losses_unrolled, rel=1e-4)


def test_a_scan_split_by_precision_and_by_remat_counts_its_passes_as_unrolled(
    monkeypatch,
):
    # Passes 1 to 3 multiply in mxfp8, and remat's first 3 split them again.
    precision = "precision(compute=float32, matmuls=mxfp8, first=2)"
    policy = tcfg.RematConfig(first=3)
    runs = {
        repeat: tcfg.MethodsConfig(repeat=repeat, remat=policy)
        for repeat in ("unroll", "scan")
    }
    unrolled = _formats_traced(SCANNED, precision, runs["unroll"], monkeypatch)
    scanned = _formats_traced(SCANNED, precision, runs["scan"], monkeypatch)
    assert scanned.items() <= unrolled.items()
    assert {at for at, _ in scanned} == {0, 1, 3}


def test_first_and_last_branches_inside_a_pipeline_are_an_error(losses):
    given = {
        "optimizer": STABLE,
        "precision": "precision(compute=float32, matmuls=mxfp8, first=1)",
        "mesh": "mesh(stage=2, data=1)",
        "methods": "methods(repeat=scan, microbatches=2)",
    }
    with pytest.raises(ValueError, match="fall inside a pipelined repeat"):
        losses(SCANNED, **given)


def test_the_routed_experts_in_a_format_with_no_grouped_gemm_are_an_error():
    with pytest.raises(BuxError, match="bux has no grouped GEMM in nvfp4"):
        compile_runs(_source("precision = precision(roles=roles(experts=nvfp4))"))


# The moe architecture's 2 MoEs, each a GEMM of up and gate, and one of down.
GROUPED = {
    "roles=roles(experts=mxfp8)": 4,
    "roles=roles(experts=mxfp8_2d)": 4,
    "matmuls=nvfp4, roles=roles(experts=mxfp8)": 4,
    "matmuls=mxfp8": 4,  # none is matmuls, where a grouped GEMM takes it
    "matmuls=mxfp8_2d": 4,
    "matmuls=nvfp4": 0,  # else compute
    "matmuls=mxfp8, roles=roles(experts=compute)": 0,
    "roles=roles(experts=mxfp8), first=1": 2,  # the first MoE is the first branch
}


@pytest.mark.parametrize("given", GROUPED)
def test_the_routed_experts_multiply_as_their_role_says_else_as_grouped_matmuls(
    monkeypatch, given
):
    calls, ragged_dot = [], grouped_blockscaled.ragged_dot

    def counted(*args, **kwargs):
        calls.append(args[3])  # its recipe
        return ragged_dot(*args, **kwargs)

    monkeypatch.setattr(grouped_blockscaled, "ragged_dot", counted)
    [run] = compile_runs(
        _source(f"precision = precision({given})", arch=ARCHITECTURES["moe"])
    )
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    _block_scaled(model, run.train.precision)
    assert len(calls) == GROUPED[given]


def _moe_id(precision: str) -> str:
    [run] = compile_runs(_source(f"precision = {precision}", arch=ARCHITECTURES["moe"]))
    return run.id


def test_experts_that_take_mxfp8_from_matmuls_have_the_id_of_those_that_name_it():
    """As they took compute before: a run that names their format keeps its id, and
    one whose experts now multiply in MXFP8 takes a new one, as a launch must not find
    it done."""
    mxfp8 = "precision(compute=bfloat16, matmuls=mxfp8"
    assert _moe_id(f"{mxfp8})") == _moe_id(f"{mxfp8}, roles=roles(experts=mxfp8))")
    # The MoE's ids since its weights start from the default normal.
    assert _moe_id(f"{mxfp8})") == "bdcce93100"
    assert _moe_id(f"{mxfp8}, roles=roles(experts=compute))") == "8621603114"
    assert _moe_id("precision(compute=bfloat16, matmuls=nvfp4)") == "57df881446"
    assert _moe_id(BF16) == "ed5cd2a755"
    [dense] = compile_runs(_source(f"precision = {mxfp8})", arch=OLD))
    assert dense.id == "f66305703c"  # no experts: as before


# In float32, as above; but the experts' dW quantizes in blocks of 32 of the rows each
# device has of an expert, which differ as a mesh splits the tokens.
@pytest.mark.parametrize("method", ["sorted", "all_to_all", "all_gather"])
def test_mxfp8_experts_on_a_split_mesh_train_near_one_device(losses, method):
    given = {
        "optimizer": STABLE,
        "precision": "precision(compute=float32, roles=roles(experts=mxfp8))",
        "methods": f"methods(moe={method})",
    }
    one = losses(ARCHITECTURES["moe"], mesh=ONE, **given)
    split = losses(ARCHITECTURES["moe"], mesh="mesh(fsdp=2, expert=4)", **given)
    assert np.allclose(split, one, rtol=1e-4)


def test_a_switch_takes_up_its_passes_at_its_step(losses):
    given = {"mesh": ONE, "optimizer": STABLE, "steps": 4}
    mxfp8 = losses(ARCHITECTURES["llama"], precision=_precision("mxfp8"), **given)
    late = _precision("mxfp8").replace(")", ", switch=switch(at=0.5))")
    switched = losses(ARCHITECTURES["llama"], precision=late, **given)
    bf16 = losses(ARCHITECTURES["llama"], precision=BF16, **given)
    assert switched[:2] == mxfp8[:2] and switched[2:] != mxfp8[2:]
    assert np.allclose(switched, bf16, rtol=LOSSES["mxfp8"])


def test_evaluation_in_compute_evaluates_as_compute_does():
    def evaluated(precision: tcfg.PrecisionConfig | str) -> float:
        llama = ARCHITECTURES["llama"]
        trainer = _trainer(llama, compute=jnp.bfloat16, matmuls=precision)
        return trainer.evaluate([_batch()])["loss"]

    mxfp8 = tcfg.PrecisionConfig(matmuls="mxfp8")
    in_compute = mxfp8.model_copy(update={"eval": "compute"})
    assert evaluated(in_compute) == evaluated("compute") != evaluated(mxfp8)


def test_moments_in_bfloat16_train_with_gradients_reduced_once(losses, involuntary):
    given = {
        "mesh": "mesh(fsdp=8)",
        "methods": "methods(microbatches=2, accumulate=unreduced)",
        "optimizer": STABLE,
        "batch": "16",
        "steps": 5,
    }
    once = "precision(reduce=float32)"
    f32 = losses(ARCHITECTURES["llama"], precision=once, **given)
    involuntary()
    moments = once.replace(")", ", moments=bfloat16)")
    bf16 = losses(ARCHITECTURES["llama"], precision=moments, **given)
    assert np.allclose(bf16, f32, rtol=1e-3) and bf16 != f32
    assert involuntary() == 0


def test_moments_in_bfloat16_train_near_float32(losses):
    given = {"mesh": ONE, "optimizer": STABLE, "steps": 5}
    f32 = losses(ARCHITECTURES["llama"], precision=BF16, **given)
    moments = "precision(moments=bfloat16)"
    bf16 = losses(ARCHITECTURES["llama"], precision=moments, **given)
    assert np.allclose(bf16, f32, rtol=1e-3) and bf16 != f32

    [run] = compile_runs(_source())
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    multiplier = lr_multiplier(run.train.schedule, run.train.steps)
    tx = optimizer(run.train.optimizer, multiplier, model, moments=jnp.bfloat16)
    state = nnx.state(nnx.Optimizer(model, tx, wrt=nnx.Param), nnx.OptState)
    floats = [
        x for x in jax.tree.leaves(state) if jnp.issubdtype(x.dtype, jnp.floating)
    ]
    assert floats and all(x.dtype == jnp.bfloat16 for x in floats)


RECIPES = {
    "nvidia_mxfp8": "mxfp8",
    "nvidia_nvfp4": "nvfp4",
    "deepseek_v3_fp8": "mxfp8_2d",
    "torchtitan_mxfp8": "mxfp8_2d",
    "nemotron3_nvfp4": "nvfp4",
}


@pytest.mark.parametrize("recipe", RECIPES)
def test_a_recipe_is_a_module_a_run_uses_and_sweeps(recipe):
    runs = compile_runs(f"use {recipe}\n" + _source("precision.last = 1 | 2"))
    assert [r.train.precision.last for r in runs] == [1, 2]
    assert all(r.train.precision.matmuls == RECIPES[recipe] for r in runs)
    model = parallel.build(lambda: Model(runs[0].arch, rngs=nnx.Rngs(0)))
    assert _block_scaled(model, runs[0].train.precision) > 0


def test_precision_s_new_settings_at_their_defaults_keep_the_run_s_id():
    def run_id(*settings: str) -> str:
        [run] = compile_runs(_source(*settings))
        return run.id

    plain, f32 = run_id(), run_id("precision = precision(compute=float32)")
    assert run_id("precision = precision(eval=compute)") == plain
    assert run_id("precision = precision(compute=float32, eval=compute)") == f32
    defaults = (
        "precision(roles=roles(), first=0, last=0, moments=float32, dispatch=compute)"
    )
    assert run_id(f"precision = {defaults}") == plain
    for changed in ("last=2", "moments=bfloat16", "passes=[fprop,]", "dispatch=mxfp8"):
        assert run_id(f"precision = precision({changed})") != plain


# --- Microbatches -----------------------------------------------------------------------

# A mesh, and the microbatches its batch axes leave rows for, of the 16 rows of _batch.
SPLITS = [(ONE, 4), ("mesh(fsdp=8)", 2), ("mesh(fsdp=2, expert=2, tensor=2)", 4)]
EXACT = ["llama", "moe", "moe_sequence_loss"]


@pytest.mark.parametrize(("mesh", "n"), SPLITS)
@pytest.mark.parametrize("arch", EXACT)
def test_microbatches_give_the_whole_batch_s_gradient_and_router_bias(arch, mesh, n):
    batch = _documents()
    with parallel.use(tcfg.MeshConfig.model_validate(_sizes(mesh))):
        whole = _trainer(ARCHITECTURES[arch])
        parts = _trainer(ARCHITECTURES[arch], microbatches=n)
        g1, m1 = nnx.jit(whole.gradients)(whole.model, parallel.shard_batch(batch))
        gn, mn = nnx.jit(parts.gradients)(parts.model, parallel.shard_batch(batch))
    assert jax.tree.all(
        jax.tree.map(lambda a, b: jnp.allclose(a, b, atol=2e-6), g1, gn)
    )
    assert all(np.isclose(m1[k], mn[k], atol=1e-6) for k in m1)
    for a, b in zip(_biases(whole.model), _biases(parts.model), strict=True):
        assert jnp.array_equal(a, b)
    assert arch != "moe" or any((b != 0).any() for b in _biases(parts.model))


@pytest.mark.parametrize("arch", ["llama", "moe"])
def test_a_run_in_microbatches_trains_as_the_whole_batch_split_or_not(
    losses, involuntary, arch
):
    source, batch = ARCHITECTURES[arch], "16"
    one = losses(source, mesh=ONE, batch=batch)
    involuntary()
    for mesh, n in SPLITS:
        given = {"mesh": mesh, "batch": batch, "methods": f"methods(microbatches={n})"}
        assert np.allclose(losses(source, **given), one, atol=1e-5)
    assert involuntary() == 0


def test_dropout_draws_each_microbatch_s_masks_in_turn():
    source = _with("attn").replace("activation=swish", "activation=swish, dropout=0.1")

    def draws(n: int) -> list[int]:
        """How far each RNG count moves in a step of `n` microbatches."""
        trainer = _trainer(source, microbatches=n)
        before = _counts(trainer.model)
        nnx.jit(trainer.gradients)(trainer.model, _batch())
        return [b - a for a, b in zip(before, _counts(trainer.model), strict=True)]

    once, twice = draws(1), draws(2)
    assert any(once) and twice == [2 * c for c in once]


def test_a_batch_that_does_not_split_into_microbatches_is_an_error(tmp_path):
    def run(*settings: str):
        [run] = compile_runs(_source("methods = methods(microbatches=2)", *settings))
        return run

    with pytest.raises(ValueError, match=r"batch = 9 does not split into .*= 2$"):
        train(run("batch = 9", "mesh = mesh(data=1)"), tmp_path, echo=lambda _: None)
    rows = r"batch = 8 does not split over layout.batch, data=8, 2 microbatches of 4"
    with pytest.raises(ValueError, match=rows + r" rows: make it a multiple of 16"):
        train(run("batch = 8"), tmp_path, echo=lambda _: None)


def test_microbatches_and_bfloat16_take_less_memory():
    def temp(**settings) -> int:
        trainer = _trainer(ARCHITECTURES["llama"], **settings)
        batch = parallel.shard_batch(_batch())
        step = trainer.lower(batch).compile()
        memory = step.memory_analysis()
        assert memory is not None
        return memory.temp_size_in_bytes

    f32 = temp()
    assert temp(compute=jnp.bfloat16) < 0.8 * f32
    assert temp(microbatches=4) < 0.5 * f32


# --- Helpers ---------------------------------------------------------------------------


TRAIN = [
    "data = synthetic",
    "optimizer = adamw(lr=1e-2)",
    "seq_len = 16",
    "batch = 8",
    "steps = 3",
]


# The llama as every weight matrix started before the default init was a normal, of
# the run ids before it.
OLD = "init = fan_in(embed=fan_in(distribution=gaussian))\n" + ARCHITECTURES["llama"]


def _source(*settings: str, arch: str = ARCHITECTURES["llama"]) -> str:
    given = {s.split("=")[0].strip(): s for s in [*TRAIN, *settings]}
    return arch + "\n" + "\n".join(given.values())


def _trainer(arch: str, **settings) -> Trainer:
    [run] = compile_runs(_source(arch=arch))
    t = run.train
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
    return Trainer(model, tx, objective(t.loss, run.arch), **settings)


def _batch(rows: int = 16, t: int = 16) -> Batch:
    return plain(np.random.default_rng(0).integers(0, 128, (rows, t)))


def _documents() -> Batch:
    """16 rows of documents, row r's first r tokens not targets: microbatches of
    unequal numbers of targets."""
    ids = np.random.default_rng(1).integers(0, 128, (16, 16)).astype(np.int32)
    positions = np.tile(np.r_[0:5, 0:11].astype(np.int32), (16, 1))
    mask = np.arange(16)[None, :] >= np.arange(16)[:, None]
    return Batch(ids, positions, mask)


def _sizes(mesh: str) -> dict[str, int]:
    """mesh(fsdp=2, tensor=4) as {"fsdp": 2, "tensor": 4}."""
    inside = mesh.removeprefix("mesh(").removesuffix(")")
    return {k: int(v) for k, v in (kv.split("=") for kv in inside.split(", "))}


def _counts(model: Model) -> list[int]:
    return [int(c) for c in jax.tree.leaves(nnx.state(model, nnx.RngCount))]


def _biases(model: Model) -> list[Float[Array, " e"]]:
    moes = [m for _, m in nnx.iter_modules(model) if isinstance(m, MoE)]
    return [m.router_bias[...] for m in moes if m.router_bias is not None]
