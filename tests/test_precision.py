"""Precision and microbatches: runs that compute in bfloat16 on float32 parameters, the
float32 islands inside them, and gradient accumulation against the whole batch, on one
device and split over the 8 CPU devices the tests have."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux import parallel
from bux.architecture import Model
from bux.architecture.linear_attention import delta_rule, l2norm, recurrence
from bux.architecture.moe import MoE
from bux.architecture.norm import Norm
from bux.architecture.ssm import selective_scan
from bux.lang import compile_runs
from bux.train import config as tcfg
from bux.train.data import Batch, plain
from bux.train.objective import objective
from bux.train.optim import optimizer, schedule
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
    trainer.update(_batch())
    params = nnx.state(trainer.model, nnx.Param)
    moments = nnx.state(trainer.optimizer, nnx.OptState)
    floats = [
        x
        for x in jax.tree.leaves((grads, params, moments))
        if jnp.issubdtype(x.dtype, jnp.floating)
    ]
    assert floats and all(x.dtype == jnp.float32 for x in floats)


def test_eval_computes_in_bfloat16_near_float32():
    f32 = _trainer(ARCHITECTURES["llama"]).evaluate([_batch()])
    bf16 = _trainer(ARCHITECTURES["llama"], compute=jnp.bfloat16).evaluate([_batch()])
    assert bf16 == pytest.approx(f32, rel=1e-2) and bf16 != f32


@pytest.mark.parametrize("family", FAMILIES)
def test_a_bfloat16_forward_keeps_the_stream_in_bfloat16(family):
    model = cast(_trainer(FAMILIES[family]).model, jnp.bfloat16)
    assert all(y.dtype == jnp.bfloat16 for y in forward(model, _batch()))


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
    assert plain.id == f32.id


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
        step = trainer._train.lower(trainer.model, trainer.optimizer, batch).compile()
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


def _source(*settings: str, arch: str = ARCHITECTURES["llama"]) -> str:
    given = {s.split("=")[0].strip(): s for s in [*TRAIN, *settings]}
    return arch + "\n" + "\n".join(given.values())


def _trainer(arch: str, **settings) -> Trainer:
    [run] = compile_runs(_source(arch=arch))
    t = run.train
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    tx = optimizer(t.optimizer, schedule(t.schedule, t.steps), model, t.clip)
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
