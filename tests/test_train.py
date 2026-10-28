"""The trainer: schedules, optimizers, data, losses, and runs that resume."""

import contextlib
import functools
import itertools
import json
import re
import shutil
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import corpus
from flax import nnx
from flax.nnx import graphlib
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float
from optax.contrib import MuonDimensionNumbers, scale_by_muon

from bux import parallel
from bux.__main__ import main
from bux.architecture import Model
from bux.data.pipeline import jobs
from bux.lang import compile, compile_runs
from bux.train import config as tcfg
from bux.train import flops, profilers
from bux.train import trainer as trainer_module
from bux.train.data import Batch, batches, plain, prefetch
from bux.train.hooks import Checkpoint, Hook, Log, Profile
from bux.train.objective import (
    Term,
    cross_entropy,
    held_out,
    objective,
    prediction_offsets,
    target_mask,
    z_loss,
)
from bux.train.optim import (
    lr_multiplier,
    momentum,
    muon_matrices,
    newton_schulz,
    optimizer,
    optimizers,
    orthogonalized,
    path_keys,
    schedules,
    weight_decay_mask,
    whole_matrix_spec,
)
from bux.train.sinks import LatestMetrics, Sink
from bux.train.trainer import Trainer, directory, forward, step_flops, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"

ARCH = """
d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
block = [norm attn] [norm ffn]
"""


def constant(steps: int) -> tcfg.TrainConfig:
    """A run of `steps` at a constant learning rate, for its multiplier."""
    return tcfg.TrainConfig(
        data=tcfg.SyntheticConfig(),
        optimizer=tcfg.AdamWConfig(lr=1e-3),
        seq_len=8,
        batch=2,
        steps=steps,
    )


def smoke(*overrides: str):
    """The smoke run, shortened: its warmup would not fit in a few steps."""
    settings = ("schedule.warmup = 1", *overrides)
    [run] = compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=settings)
    return run


def test_muon_trains_the_block_matrices_and_adam_the_rest():
    m = Model(
        compile(
            ARCH
            + "experts = moe(num_experts=2, top_k=1, expert=ffn)\n"
            + "model = embed [norm attn] [norm experts] norm head\n"
        ),
        rngs=nnx.Rngs(0),
    )
    found = {
        path[-2:] if path[-1] == "kernel" else path[-1:] for path in muon_matrices(m)
    }
    assert found == {
        ("q", "kernel"),
        ("k", "kernel"),
        ("v", "kernel"),
        ("o", "kernel"),
        ("router", "kernel"),
        ("up",),  # and the gate, beside it
        ("down",),
    }
    stacked = [d for p, d in muon_matrices(m).items() if p[-1] in ("up", "down")]
    assert {(d.reduction_axis, d.output_axis) for d in stacked} == {(1, 2)}


def test_weight_decay_leaves_norms_and_engram_tables_alone_whatever_their_shape():
    """The gated residual's and Engram's norms have (n, d) gains, and Engram's tables
    are matrices: none of them decays. Every other matrix does, Muon's included."""
    m = Model(
        compile(
            ARCH
            + "residual = gr(rank=4)\nmem = engram(head_dim=4, table_size=50)\n"
            + "model = embed mem [attn] [dyt ffn] head\n"
        ),
        rngs=nnx.Rngs(0),
    )
    params = nnx.state(m, nnx.Param)
    flags = jax.tree.leaves(weight_decay_mask(m)(params))
    leaves = jax.tree_util.tree_leaves_with_path(params)
    assert len(flags) == len(leaves)
    named = [
        (tuple(k.key for k in path[:-1]), x.ndim, d)
        for (path, x), d in zip(leaves, flags, strict=True)
    ]
    assert all(ndim >= 2 for _, ndim, d in named if d)
    kept = {keys[-2:] for keys, ndim, d in named if ndim >= 2 and not d}
    assert {k for _, k in kept} == {"scale", "table"}
    assert {o for o, k in kept if k == "scale"} == {
        "norm",  # the gated residual's, and its read before the head
        "q_norm",
        "k_norm",
        "conv_norm",
    }
    decays = {keys for keys, _, d in named if d}
    assert set(muon_matrices(m)) <= decays


class Valued(nnx.Module):
    """A parameter named as the attribute a Variable flattens to, as `scale`'s."""

    def __init__(self):
        self.value, self.w = nnx.Param(jnp.ones(3)), nnx.Param(jnp.ones((2, 2)))


def test_a_state_s_paths_are_its_variables_as_a_pure_tree_s_are():
    state = nnx.state(Valued(), nnx.Param)
    pure = nnx.to_pure_dict(state)
    paths = [
        [path_keys(p) for p, _ in jax.tree_util.tree_leaves_with_path(tree)]
        for tree in (state, pure)
    ]
    assert paths[0] == paths[1] == [("value",), ("w",)]


def test_weight_decay_leaves_decay_rates_mixes_and_static_maps_alone():
    """Mamba's A_log, RWKV-7's token shift and r_k, and hc's static maps are matrices,
    but of rates, mixes and maps that start away from 0, which their references never
    decay. The projections around them do."""
    rwkv7 = "rwkv7(num_heads=2, head_dim=8, decay_rank=4, a_rank=4, v_rank=4)"
    mamba2 = "mamba2(state_size=4, head_dim=8)"
    mixers = f"[norm mamba(state_size=4)] [norm {rwkv7}] [norm {mamba2}]"
    source = f"residual = hc(expansion=2)\nmodel = embed {mixers} norm head\n"
    m = Model(compile(ARCH + source), rngs=nnx.Rngs(0))
    mask = weight_decay_mask(m)(nnx.state(m, nnx.Param))
    flags = {path: v.get_value() for path, v in nnx.to_flat_state(mask)}
    kept = ("A_log", "shift", "r_k", "static_alpha", "static_beta")
    names = {path[-1] for path in flags}
    assert set(kept) <= names
    assert not any(d for path, d in flags.items() if path[-1] in kept)
    assert all(d for path, d in flags.items() if path[-2:] == ("in_proj", "kernel"))


# --- Muon split over devices: methods.orthogonalize ------------------------------------

MOE = (
    ARCH
    + "experts = moe(num_experts=8, top_k=2, expert=ffn)\n"
    + "model = embed [norm attn] [norm experts] norm head\n"
)
SPLITS = {"fsdp": {"fsdp": 8}, "fsdp_tensor": {"fsdp": 4, "tensor": 2}}


class Stack(nnx.Module):
    """A matrix, and a stack of two: all Muon's."""

    def __init__(self):
        a, b = jax.random.split(jax.random.key(0))
        self.w = nnx.Param(jax.random.normal(a, (16, 32)))
        self.e = nnx.Param(jax.random.normal(b, (2, 16, 8)))


def _muon_updates(tx: optax.GradientTransformation, params: dict, steps=3) -> list:
    state, step = tx.init(params), jax.jit(tx.update)
    found = []
    for i in range(steps):
        grads = jax.tree.map(lambda x, i=i: jnp.sin(x + i), params)
        u, state = step(grads, state, params)
        found.append(u)
    return found


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_on_one_device_is_optax_s_whichever_method_orthogonalizes(nesterov):
    stack = Stack()
    params = nnx.to_pure_dict(nnx.state(stack, nnx.Param))
    dims = {"w": MuonDimensionNumbers(0, 1), "e": MuonDimensionNumbers(1, 2)}
    theirs = scale_by_muon(nesterov=nesterov, weight_dimension_numbers=dims)
    expected = _muon_updates(theirs, params)
    for method in ("gspmd", "gather", "owner"):
        methods = tcfg.MethodsConfig(orthogonalize=method)
        one = parallel.use(tcfg.MeshConfig(data=1), methods=methods)
        for where in (contextlib.nullcontext(), one):
            with where:
                ours = optax.chain(
                    momentum(0.95, nesterov),
                    orthogonalized(
                        newton_schulz(5), {(k,): d for k, d in dims.items()}, stack
                    ),
                )
                got = _muon_updates(ours, params)
            assert jax.tree.all(jax.tree.map(jnp.array_equal, got, expected))


def _spread(x: Float[Array, "m n"]) -> float:
    """How far from 1 the singular values of `x` are."""
    return float(
        jnp.abs(jnp.linalg.svd(x.astype(jnp.float32), compute_uv=False) - 1).max()
    )


@pytest.mark.parametrize("coefficients", ["jordan", "polar_express"])
def test_newton_schulz_in_bfloat16_is_close_to_float32(coefficients):
    x = jax.random.normal(jax.random.key(0), (64, 256))
    ns, dims = newton_schulz(5, coefficients), MuonDimensionNumbers(0, 1)
    f32, bf16 = ns(x, dims), ns(x.astype(jnp.bfloat16), dims)
    assert bf16.dtype == jnp.bfloat16
    assert jnp.linalg.norm(bf16 - f32) < 0.1 * jnp.linalg.norm(f32)


def test_polar_express_orthogonalizes_closer_than_jordan_s_quintic():
    x = jax.random.normal(jax.random.key(0), (64, 256)) * jnp.logspace(-2, 0, 256)
    dims = MuonDimensionNumbers(0, 1)
    jordan, polar = (newton_schulz(5, c)(x, dims) for c in ("jordan", "polar_express"))
    assert _spread(polar) < 0.15 and _spread(jordan) > 0.3
    assert _spread(newton_schulz(8, "polar_express")(x, dims)) < 1e-3


def test_muon_runs_newton_schulz_in_its_dtype_and_keeps_the_update_s():
    stack = Stack()
    params = nnx.to_pure_dict(nnx.state(stack, nnx.Param))
    dims = {("w",): MuonDimensionNumbers(0, 1), ("e",): MuonDimensionNumbers(1, 2)}
    ns = newton_schulz(5)
    f32, bf16 = (
        _muon_updates(orthogonalized(ns, dims, stack, dtype), params, steps=1)[0]
        for dtype in (jnp.float32, jnp.bfloat16)
    )
    for a, b in zip(jax.tree.leaves(f32), jax.tree.leaves(bf16), strict=True):
        assert b.dtype == jnp.float32 and not jnp.array_equal(a, b)
        assert jnp.linalg.norm(b - a) < 0.1 * jnp.linalg.norm(a)


def test_newton_schulz_s_dtype_is_in_the_run_s_id():
    plain = smoke("optimizer = muon(lr=0.02)")
    f32 = smoke("optimizer = muon(lr=0.02, ns_dtype=float32)")
    bf16 = smoke("optimizer = muon(lr=0.02, ns_dtype=bfloat16)")
    assert plain.id == bf16.id != f32.id


# --- Gram Newton-Schulz -----------------------------------------------------------------

MATRIX = MuonDimensionNumbers(0, 1)


def _of_spectrum(spectrum: Float[Array, " m"], n: int) -> Float[Array, "m n"]:
    """An m×n matrix of singular values `spectrum`, its singular vectors at random."""
    x = jax.random.normal(jax.random.key(1), (len(spectrum), n))
    u, _, vt = jnp.linalg.svd(x, full_matrices=False)
    return (u * spectrum) @ vt


@pytest.mark.parametrize("restarts", [None, (1, 3)], ids=["2", "1,3"])
@pytest.mark.parametrize("coefficients", ["jordan", "polar_express"])
def test_gram_newton_schulz_is_the_standard_s_in_float32(coefficients, restarts):
    """On a wide matrix, a tall one, and a stack of them along their middle axis."""
    standard = newton_schulz(5, coefficients)
    gram = newton_schulz(5, coefficients, "gram", restarts)
    for shape, dims in [
        ((64, 256), MATRIX),
        ((256, 64), MATRIX),
        ((32, 3, 96), MuonDimensionNumbers(0, 2)),
    ]:
        x = jax.random.normal(jax.random.key(0), shape)
        ours, theirs = gram(x, dims), standard(x, dims)
        assert ours.shape == shape
        assert jnp.linalg.norm(ours - theirs) < 1e-4 * jnp.linalg.norm(theirs)


def test_gram_newton_schulz_iterates_16_bits_in_float16():
    """Normalized in float32 and iterated in float16, a matrix in bfloat16 comes out
    closer to float32's than the standard iteration's in bfloat16."""
    x = jax.random.normal(jax.random.key(0), (64, 256))
    standard, gram = newton_schulz(5), newton_schulz(5, algorithm="gram")
    f32, half = standard(x, MATRIX), x.astype(jnp.bfloat16)
    ours, theirs = gram(half, MATRIX), standard(half, MATRIX)
    assert ours.dtype == jnp.bfloat16
    assert re.search(
        r"\dxf16>", jax.jit(lambda x: gram(x, MATRIX)).lower(half).as_text()
    )
    error = lambda y: jnp.linalg.norm(y.astype(jnp.float32) - f32)
    assert error(ours) < error(theirs) / 2


def test_gram_newton_schulz_restarts_before_the_gram_matrix_s_rounding_grows():
    """Eight iterations of Polar Express on a spectrum half zero, rounded in 16 bits:
    restarting after the second alone, the Gram matrix's rounding around its zeros
    grows to singular values past 100; after every third, they stay as near 1 as the
    standard iteration's."""
    spectrum = jnp.concatenate([jnp.logspace(0, -2, 32), jnp.zeros(32)])
    x = _of_spectrum(spectrum, 256).astype(jnp.bfloat16)

    def largest(algorithm: str, restarts: tuple[int, ...] | None = None) -> float:
        y = newton_schulz(8, "polar_express", algorithm, restarts)(x, MATRIX)
        return float(jnp.linalg.norm(y.astype(jnp.float32), 2))

    assert largest("gram", (2,)) > 100
    assert largest("gram") < largest("standard") + 0.01


def test_gram_runs_the_standard_iteration_where_it_would_not_save_products():
    """On a square matrix, or nearly: its m³ + 2m²n an iteration is less there."""
    standard, gram = newton_schulz(5), newton_schulz(5, algorithm="gram")
    for shape in [(64, 64), (64, 80), (80, 64)]:
        x = jax.random.normal(jax.random.key(0), shape)
        assert jnp.array_equal(gram(x, MATRIX), standard(x, MATRIX))


def test_gram_newton_schulz_computes_a_third_less_on_a_matrix_four_times_as_wide():
    x = jax.random.normal(jax.random.key(0), (64, 256))

    def flops(algorithm: str) -> float:
        ns = newton_schulz(5, algorithm=algorithm)
        compiled = jax.jit(lambda x: ns(x, MATRIX)).lower(x).compile()
        return (compiled.cost_analysis() or {})["flops"]

    assert flops("gram") < 0.7 * flops("standard")


def test_newton_schulz_s_algorithm_and_restarts_are_in_the_run_s_id():
    plain = smoke("optimizer = muon(lr=0.02)")
    standard = smoke("optimizer = muon(lr=0.02, ns_algorithm=standard)")
    gram = smoke("optimizer = muon(lr=0.02, ns_algorithm=gram)")
    once = smoke("optimizer = muon(lr=0.02, ns_algorithm=gram, ns_restarts=2)")
    twice = smoke("optimizer = muon(lr=0.02, ns_algorithm=gram, ns_restarts=[2, 5])")
    assert plain.id == standard.id != gram.id != once.id != twice.id
    restarts = [r.train.optimizer for r in (once, twice)]
    assert [c.ns_restarts for c in restarts if isinstance(c, tcfg.MuonConfig)] == [
        (2,),
        (2, 5),
    ]
    with pytest.raises(NotImplementedError, match="ns_algorithms.register"):
        newton_schulz(5, algorithm="cubic")


@functools.cache
def _muon_step(
    orthogonalize: str = "gspmd",
    zero: int = 3,
    ns_dtype: str = "float32",
    algorithm: str = "standard",
    **mesh: int,
) -> tuple[list, str]:
    """The MoE's parameters after a Muon step from fixed gradients, by Newton-Schulz's
    `algorithm`, split over `mesh` (outside parallel.use with none), and the compiled
    step's HLO."""
    methods = tcfg.MethodsConfig(orthogonalize=orthogonalize)
    layout = tcfg.LayoutConfig.model_validate({"zero": zero})
    split = parallel.use(tcfg.MeshConfig.model_validate(mesh), layout, methods)
    with split if mesh else contextlib.nullcontext():
        model = parallel.build(lambda: Model(compile(MOE), rngs=nnx.Rngs(0)))
        given = {"lr": 0.02, "ns_dtype": ns_dtype, "ns_algorithm": algorithm}
        c = tcfg.MuonConfig.model_validate(given)
        lr = lr_multiplier(constant(1))
        trained = nnx.Optimizer(model, optimizer(c, lr, model), wrt=nnx.Param)
        parallel.shard_state(model, trained)
        grads = jax.tree.map(jnp.sin, nnx.state(model, nnx.Param))
        step = nnx.jit(lambda m, o, g: o.update(m, g))
        hlo = step.lower(model, trained, grads).compile().as_text() or ""
        step(model, trained, grads)
    return jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(model, nnx.Param))), hlo


ALGORITHMS = ["standard", "gram"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("orthogonalize", ["gspmd", "gather", "owner"])
@pytest.mark.parametrize("mesh", SPLITS.values(), ids=list(SPLITS))
def test_muon_updates_split_over_devices_as_on_one(orthogonalize, mesh, algorithm):
    one, _ = _muon_step(algorithm=algorithm)
    split, _ = _muon_step(orthogonalize, algorithm=algorithm, **mesh)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(split, one, strict=True))


MORE_SPLITS = {
    "data_fsdp": {"data": 2, "fsdp": 4},
    "fsdp_expert_tensor": {"fsdp": 2, "expert": 2, "tensor": 2},
    "expert": {"expert": 8},
}


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("zero", [3, 1])
@pytest.mark.parametrize("mesh", MORE_SPLITS.values(), ids=list(MORE_SPLITS))
def test_muon_owners_orthogonalize_as_gspmd_does(mesh, zero, algorithm):
    gspmd, _ = _muon_step("gspmd", zero, algorithm=algorithm, **mesh)
    owner, _ = _muon_step("owner", zero, algorithm=algorithm, **mesh)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(owner, gspmd, strict=True))


def _count(op: str, hlo: str) -> int:
    return len(re.findall(rf"\b{op}(-start)?\(", hlo))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("mesh", SPLITS.values(), ids=list(SPLITS))
def test_muon_gathering_whole_matrices_needs_no_all_reduce(mesh, algorithm):
    """gspmd runs Newton-Schulz on shards, all-reducing products over split
    dimensions; gather moves each matrix whole once and runs it there; owner moves
    each to one device, the matrices of a shape and split by one all-to-all."""
    _, gspmd = _muon_step("gspmd", algorithm=algorithm, **mesh)
    _, gather = _muon_step("gather", algorithm=algorithm, **mesh)
    _, owner = _muon_step("owner", algorithm=algorithm, **mesh)
    assert _count("all-reduce", gather) == _count("all-reduce", owner) == 0
    assert _count("all-reduce", gspmd) > 0
    assert _count("all-gather", gather) > _count("all-gather", owner)
    assert _count("all-to-all", owner) > 0


@pytest.mark.parametrize("mesh", SPLITS.values(), ids=list(SPLITS))
def test_gram_newton_schulz_on_shards_all_reduces_less_than_the_standard(mesh):
    """Under gspmd, a product by a matrix split along its longer side is an
    all-reduce: Gram's iteration takes four such products, the standard ten."""
    _, standard = _muon_step("gspmd", **mesh)
    _, gram = _muon_step("gspmd", algorithm="gram", **mesh)
    assert _count("all-reduce", gram) < _count("all-reduce", standard)


def test_muon_owners_under_zero_1_gather_less_than_gather_does():
    """The update comes split by FSDP, as the optimizer's state: owner sends its shards
    to the owners, and gathers the results, where gather gathers every matrix."""
    _, gather = _muon_step("gather", 1, fsdp=8)
    _, owner = _muon_step("owner", 1, fsdp=8)
    assert _count("all-gather", owner) + _count("all-to-all", owner) < _count(
        "all-gather", gather
    )


def test_muon_owners_get_their_matrices_in_newton_schulz_s_dtype():
    """In the program jax gives XLA, that is: XLA on CPU moves them in float32."""
    methods = tcfg.MethodsConfig(orthogonalize="owner")
    with parallel.use(tcfg.MeshConfig(fsdp=8), methods=methods):
        stack = parallel.build(Stack)
        params = nnx.to_pure_dict(nnx.state(stack, nnx.Param))
        dims = {("w",): MuonDimensionNumbers(0, 1), ("e",): MuonDimensionNumbers(1, 2)}
        tx = orthogonalized(newton_schulz(5), dims, stack, dtype=jnp.bfloat16)
        text = jax.jit(tx.update).lower(params, tx.init(params), params).as_text()
    moved = re.findall(r"all_to_all.*?: \(tensor<[\dx]*(\w+)>", text)
    assert len(moved) == 4 and set(moved) == {"bf16"}  # 2 shapes, there and back


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_muon_updates_in_bfloat16_split_over_devices_as_on_one(algorithm):
    one, _ = _muon_step(ns_dtype="bfloat16", algorithm=algorithm)
    split, _ = _muon_step("owner", 3, "bfloat16", algorithm, fsdp=4, tensor=2)
    assert all(np.allclose(a, b, atol=1e-4) for a, b in zip(split, one, strict=True))


def test_gather_holds_matrices_whole_and_a_stack_split_over_what_that_frees():
    mesh = parallel.mesh(tcfg.MeshConfig(fsdp=2, expert=2, tensor=2))
    matrix, experts = MuonDimensionNumbers(0, 1), MuonDimensionNumbers(1, 2)
    assert whole_matrix_spec(P("fsdp", "tensor"), matrix, (64, 32), mesh) == P(
        None, None
    )
    every = ("expert", "fsdp", "tensor")
    split = P("expert", "fsdp", "tensor")
    assert whole_matrix_spec(split, experts, (8, 64, 32), mesh) == P(every, None, None)
    # 4 experts do not split 8 ways: each stays whole, and the experts as they were.
    assert whole_matrix_spec(split, experts, (4, 64, 32), mesh) == P(
        "expert", None, None
    )
    assert whole_matrix_spec(P(None, "fsdp"), experts, (3, 64, 32), mesh) == P(
        None, None, None
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("orthogonalize", ["gather", "owner"])
def test_a_run_that_moves_matrices_for_muon_trains_as_on_one_device(
    losses, orthogonalize, algorithm
):
    how = f"methods(orthogonalize={orthogonalize})"
    given = f"muon(lr=0.02, ns_dtype=float32, ns_algorithm={algorithm})"
    muon = {"optimizer": given, "methods": how}
    one = losses(MOE, mesh="mesh(data=1)", **muon)
    assert np.allclose(losses(MOE, mesh="mesh(fsdp=8)", **muon), one, atol=1e-5)


def rows(data: tcfg.DataConfig, batch: int, seq_len: int) -> tcfg.TrainConfig:
    return tcfg.TrainConfig(
        data=data,
        optimizer=tcfg.AdamWConfig(lr=1e-3),
        seq_len=seq_len,
        batch=batch,
        steps=1,
    )


def test_shards_are_read_after_their_header_epoch_by_epoch(tmp_path):
    for i in range(2):
        tokens = np.arange(1000, dtype=np.uint16) + 1000 * i
        (tmp_path / f"s{i}.bin").write_bytes(b"\0" * 8 + tokens.tobytes())
    c = tcfg.TokensConfig(path=str(tmp_path / "s*.bin"), header=8)
    t = rows(c, 4, 9)
    first = next(batches(c, t, 2048, start=5))
    again = next(batches(c, t, 2048, start=5))
    assert all((a == b).all() for a, b in zip(first[:3], again[:3], strict=True))
    assert first.input_ids.shape == (4, 9)
    assert (
        np.diff(first.input_ids, axis=1) == 1
    ).all()  # windows of consecutive tokens
    # Windows start every 8 tokens, overlapping by one: 124 a shard, each once an epoch.
    epoch = np.concatenate(
        [b.input_ids for b in itertools.islice(batches(c, t, 2048), 62)]
    )
    assert sorted(epoch[:, 0].tolist()) == [
        s + 8 * w for s in (0, 1000) for w in range(124)
    ]
    with pytest.raises(ValueError, match="outside the vocabulary of 1000"):
        next(batches(c, rows(c, 64, 9), 1000))
    with pytest.raises(FileNotFoundError, match="no shards match"):
        missing = tcfg.TokensConfig(path=str(tmp_path / "x*.bin"))
        next(batches(missing, rows(missing, 1, 2), 10))


def test_a_mixture_draws_in_proportion(tmp_path):
    (tmp_path / "a.bin").write_bytes(np.zeros(100, np.uint16).tobytes())
    (tmp_path / "b.bin").write_bytes(np.ones(100, np.uint16).tobytes())
    a, b = (tcfg.TokensConfig(path=str(tmp_path / f"{n}.bin")) for n in "ab")
    mix = tcfg.MixtureConfig(
        sources=(tcfg.Weighted(weight=3, source=a), tcfg.Weighted(weight=1, source=b))
    )
    tokens = next(batches(mix, rows(mix, 4000, 4), 2)).input_ids
    assert tokens[:, 0].mean() == 0.25


def test_prefetch_keeps_the_order_raises_what_its_thread_met_and_stops_when_closed():
    def shards():
        yield from range(3)
        raise FileNotFoundError("no shards match")

    def counted():
        for i in itertools.count():
            read.append(i)
            yield i

    early = prefetch(shards())
    assert [next(early) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(FileNotFoundError, match="no shards match"):
        next(early)
    assert list(prefetch(iter("ab"))) == ["a", "b"]
    read: list[int] = []
    threads = threading.active_count()
    endless = prefetch(counted(), ahead=2)
    assert next(endless) == 0
    endless.close()
    assert len(read) <= 4  # the one taken, the two ahead, and one on its way
    assert threading.active_count() == threads


def test_a_step_puts_its_batch_on_the_devices_itself_not_the_thread_ahead():
    """An array the thread put on a GPU, as a step's temporaries held most of its
    memory, split what they left free: the next step's did not fit."""
    arch = compile(ARCH + "model = embed block norm head\n")
    given: list[type] = []

    class Watched(Trainer):
        def train_on_batch(self, batch):
            given.append(type(batch.input_ids))
            return super().train_on_batch(batch)

    loss = objective(tcfg.LossConfig(), arch)
    trainer = Watched(Model(arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
    trainer.fit(
        batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32), 2
    )
    assert given == [np.ndarray, np.ndarray]


def test_heads_predict_further_ahead_after_each_nextn():
    arch = compile(
        ARCH + "mtp = $h! nextn [norm ffn] norm @h head\n"
        "model = embed block norm @h head mtp^2\n"
    )
    assert prediction_offsets(arch) == [1, 2, 3]
    tokens = np.asarray(jax.random.randint(jax.random.key(0), (2, 8), 0, 32))
    outputs = Model(arch, rngs=nnx.Rngs(0)).outputs(jnp.asarray(tokens))
    batch = plain(tokens)
    (total, metrics), _ = objective(tcfg.LossConfig(z=1e-4), arch)(outputs, batch)
    expected = metrics["loss"] + 1e-4 * metrics["z_loss"] + 0.3 * metrics["mtp"]
    each = {f"lm_loss/t+{k}" for k in (1, 2, 3)}  # every head's, of weight 0
    assert jnp.allclose(total, expected)
    assert set(metrics) == {"loss", "z_loss", "mtp", *each}
    assert metrics["lm_loss/t+1"] == metrics["loss"]


def test_a_head_learns_only_targets_of_its_own_document_in_the_mask():
    logits = jax.random.normal(jax.random.key(0), (1, 8, 32))
    tokens = np.arange(8)[None]
    batch = plain(tokens)._replace(position_ids=np.array([[0, 1, 2, 3, 0, 1, 2, 3]]))
    ce = optax.softmax_cross_entropy_with_integer_labels(logits[:, :5], tokens[:, 3:])[
        0
    ]
    ahead = cross_entropy(logits, batch, 3)
    assert jnp.allclose(ahead, ce[jnp.array([0, 4])].mean())  # 0->3 and 4->7
    mask = batch.loss_mask.copy()
    mask[0, 7] = False
    assert jnp.allclose(cross_entropy(logits, batch._replace(loss_mask=mask), 3), ce[0])
    # The second document starts where positions stop increasing, not only at a 0.
    later = batch._replace(position_ids=np.array([[0, 1, 2, 3, 2, 3, 4, 5]]))
    assert jnp.allclose(cross_entropy(logits, later, 3), ahead)


def test_the_z_loss_is_on_the_positions_that_predict_a_target():
    logits = jax.random.normal(jax.random.key(0), (1, 8, 32))
    batch = plain(np.arange(8)[None])._replace(
        loss_mask=np.array([[True] * 5 + [False] * 3])  # padding after 5 tokens
    )
    z = jax.nn.logsumexp(logits[0], -1) ** 2
    assert jnp.allclose(
        z_loss(logits, target_mask(batch)), z[:4].mean()
    )  # 0->1 .. 3->4
    arch = compile(ARCH + "model = embed block norm head\n")
    (_, metrics), _ = objective(tcfg.LossConfig(z=1e-4), arch)([logits], batch)
    assert jnp.allclose(metrics["z_loss"], z[:4].mean())


def test_a_run_writes_its_directory_and_resumes_where_it_stopped(tmp_path):
    run = smoke(
        "steps = 6",
        "log = log(every=1)",
        "checkpoint = checkpoint(every=3)",
        "eval = eval(data=synthetic, every=3, batches=2)",
        "ffn.dropout = 0.1",  # whose random keys the checkpoint keeps
        "methods = methods(donate=true)",  # saved, then donated to the next step
    )
    whole = train(run, tmp_path, echo=lambda _: None)
    folder = directory(run, tmp_path)
    assert {p.name for p in folder.iterdir()} == {
        "run.bux",
        "run.json",
        "metrics.jsonl",
        "checkpoints",
    }
    [again] = compile_runs((folder / "run.bux").read_text())
    assert again.id == run.id
    lines = [
        json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
    ]
    assert [line["step"] for line in lines if "loss" in line] == [1, 2, 3, 4, 5, 6]
    assert "eval_loss" in whole

    # Lose the last checkpoint: the run resumes from step 3 and ends where it did,
    # its metrics logged once a step, as if it had not stopped.
    shutil.rmtree(folder / "checkpoints" / "6")
    said: list[str] = []
    resumed = train(run, tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 3")
    assert (
        resumed["loss"] == whole["loss"] and resumed["grad_norm"] == whole["grad_norm"]
    )
    again = [json.loads(line) for line in (folder / "metrics.jsonl").open()]
    assert [line["step"] for line in again] == [line["step"] for line in lines]


def test_a_checkpoint_restores_only_into_a_state_of_its_arrays(tmp_path):
    def trainer(width: int) -> Trainer:
        arch = compile(f"d_model = {width}\nvocab_size = 32\nmodel = embed head\n")
        model = Model(arch, rngs=nnx.Rngs(0))
        return Trainer(model, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))

    save = Checkpoint(tmp_path, every=1, keep=1)
    save(trainer(16), {})
    save.close()
    narrower, load = trainer(8), Checkpoint(tmp_path, every=1, keep=1)
    with pytest.raises(
        ValueError, match=re.escape("[32, 16], and this run's float32[32, 8]")
    ):
        load.restore(narrower)
    load.close()
    assert not any(x.is_deleted() for x in jax.tree.leaves(narrower.state()))


def test_a_save_a_crash_left_unfinished_is_dropped_and_its_step_not_saved(tmp_path):
    # Orbax's names for a step's save until it ends, which a crash leaves behind.
    unfinished = tmp_path / "20.orbax-checkpoint-tmp" / "default.orbax-checkpoint-tmp"
    unfinished.mkdir(parents=True)
    arch = compile("d_model = 16\nvocab_size = 32\nmodel = embed head\n")
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(1e-3), objective(tcfg.LossConfig(), arch))
    save = Checkpoint(tmp_path, every=10, keep=2)
    for step in (20, 30):
        trainer.step = step
        save(trainer, {})
    save.close()
    assert [p.name for p in tmp_path.iterdir()] == ["30"]


def test_a_profile_traces_its_steps_into_the_run_s_directory_and_keeps_its_id(
    tmp_path,
):
    run = smoke("steps = 4", "profile = profile(start=1, steps=2)")
    assert run.id == smoke("steps = 4").id
    train(run, tmp_path, echo=lambda _: None)
    traces = (directory(run, tmp_path) / "profile").rglob("*.xplane.pb")
    assert len(list(traces)) == 1
    hook = Profile(tmp_path, start=1, steps=2)
    assert [s for s in range(1, 5) if hook.due(s, 4)] == [1, 3]


def test_a_profile_by_nsys_captures_nothing_off_gpus_and_writes_no_trace(tmp_path):
    run = smoke("steps = 3", "profile = profile(start=1, steps=1, tool=nsys)")
    last = train(run, tmp_path, echo=lambda _: None)
    assert "comm_ms" not in last
    assert not list((directory(run, tmp_path) / "profile").rglob("*.xplane.pb"))


def test_nsys_and_ncu_run_a_process_under_their_command(tmp_path, monkeypatch):
    monkeypatch.delenv(profilers.VARIABLE, raising=False)
    nsys = tcfg.ProfileConfig(tool="nsys", options="--cuda-memory-usage=true")
    tool = profilers.command(nsys, tmp_path)
    assert tool[:4] == ["env", "BUX_PROFILER=nsys", "nsys", "profile"]
    assert f"--output={tmp_path / 'profile' / '%h.%p'}" in tool
    assert tool[-1] == "--cuda-memory-usage=true"
    ncu = profilers.command(tcfg.ProfileConfig(tool="ncu"), tmp_path)
    assert ncu[2:4] == ["ncu", "--profile-from-start=off"]
    assert profilers.command(tcfg.ProfileConfig(), tmp_path) == []  # jax: in it
    assert profilers.unfit(tcfg.ProfileConfig(tool="perf")) == (
        "profile: no tool 'perf'; there are jax, nsys, ncu"
    )
    monkeypatch.setenv(profilers.VARIABLE, "nsys")  # under it already
    assert profilers.command(nsys, tmp_path) == []


def test_a_run_trains_on_text_that_it_prepares_first(tmp_path, byte_tokenizer):
    text = corpus(tmp_path / "corpus")
    held = ", ".join(
        f'eval(name="{name}", data=text(path="{text}", split=[{a}, {b}]), every=2)'
        for name, a, b in (("near", 0.9, 0.95), ("far", 0.95, 1))
    )
    run = smoke(
        f'tokenizer = tokenizer(name="{byte_tokenizer.name}")',
        f'data = text(path="{text}", split=[0, 0.9])',
        f"eval = [{held}]",
        "steps = 2",
    )
    said: list[str] = []
    last = train(run, tmp_path / "runs", cache=tmp_path / "packed", echo=said.append)
    measured = ("loss", "near_loss", "near_bpb", "far_loss", "far_bpb")
    assert all(np.isfinite(last[k]) for k in measured)
    assert [s.split(":")[0] for s in said[:3]] == [j.name for j in jobs(run.train)]
    assert len(list((tmp_path / "packed").iterdir())) == 3
    assert any("near loss" in s and "near bpb" in s for s in said)


def test_an_eval_measures_the_same_on_one_device_and_split_over_eight(
    tmp_path, byte_tokenizer
):
    def measured(mesh: str) -> dict[str, float]:
        run = smoke(
            f'tokenizer = tokenizer(name="{byte_tokenizer.name}")',
            "eval = eval(data=synthetic, every=2, batches=2)",
            "steps = 2",
            f"mesh = {mesh}",
            "precision = precision(compute=float32)",
        )
        return train(run, tmp_path / str(len(mesh)), echo=lambda _: None)

    # Ids past the tokenizer's 259 stand for no text, as its special ones do.
    one, eight = measured("mesh(data=1)"), measured("mesh(data=2, tensor=2, context=2)")
    for name in ("eval_loss", "eval_bpb"):
        assert np.isclose(one[name], eight[name], rtol=1e-5)


def test_muon_runs_end_to_end(tmp_path):
    run = smoke("optimizer = muon(lr=0.02)", "steps = 3")
    last = train(run, tmp_path, echo=lambda _: None)
    assert np.isfinite(last["loss"])


def test_a_builder_registered_for_a_type_replaces_its_own(monkeypatch):
    def half(c, train):
        return lambda step: jnp.asarray(0.5)

    monkeypatch.setitem(schedules.builders, "constant", half)
    run = constant(4).model_copy(update={"schedule": tcfg.ConstantSchedule(warmup=2)})
    f = lr_multiplier(run)
    assert [float(f(s)) for s in range(4)] == [0.25, 0.5, 0.5, 0.5]
    monkeypatch.delitem(optimizers.builders, "adamw")
    hint = r"no 'adamw' optimizer yet; add one with @bux\.train\.optim\.optimizers"
    with pytest.raises(NotImplementedError, match=hint):
        optimizer(tcfg.AdamWConfig(lr=1e-3), half(None, 1), nnx.Module())


class WeightNorm(Hook):
    every = 2

    def __call__(self, trainer, metrics):
        params = nnx.state(trainer.model, nnx.Param)
        return {"weight_norm": float(optax.tree.norm(params))}


class Steps(Sink):
    def __init__(self):
        self.reports: list[tuple[int, set[str]]] = []

    def write(self, step, scalars):
        self.reports.append((step, set(scalars)))


class Kept(Hook):
    """Keeps the trainer a run makes."""

    def start(self, trainer):
        self.trainer = trainer

    def __call__(self, trainer, metrics):
        return None


def test_a_trainer_takes_terms_hooks_and_sinks_of_ones_own():
    arch = compile(ARCH + "model = embed block norm head\n")
    loss = objective(tcfg.LossConfig(), arch)
    loss.terms["logit_l2"] = Term(1e-4, lambda logits, _: jnp.mean(logits[0] ** 2))
    trainer = Trainer(Model(arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
    lr = lr_multiplier(constant(3))
    latest, steps = LatestMetrics(), Steps()
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    hooks = [Log(every=1, lr=lr, tokens=16), WeightNorm()]
    trainer.fit(data, 3, hooks=hooks, sinks=[latest, steps])
    assert trainer.step == 3
    assert [s for s, _ in steps.reports] == [1, 2, 2, 3, 3]  # log, then weight norm
    assert {"loss", "logit_l2", "grad_norm", "lr", "weight_norm"} <= set(latest.metrics)


def test_a_step_s_flops_are_counted_from_its_program():
    """The smoke run's: 6 N of its matrices, and 12 L d t of attention (PaLM's) times
    the share of the t × t logits that a causal mask keeps, (t + 1) / 2t."""
    matrices = 2 * (2 * 64 * 64 + 2 * 64 * 32 + 3 * 64 * 256) + 64 * 512
    attention = 12 * 2 * 64 * 64 * 65 / 128  # layers, heads × head_dim, positions
    assert step_flops(smoke()).one_document(64) == (6 * matrices + attention) * 8 * 64
    x, w, sizes = jnp.ones((6, 8)), jnp.ones((3, 8, 5)), jnp.array([1, 2, 3])
    grad = jax.grad(lambda x, w: jax.lax.ragged_dot(x, w, sizes).sum(), (0, 1))
    assert flops.count(grad, x, w).fixed == 3 * 2 * 6 * 8 * 5

    def scan(c):
        return jax.lax.scan(lambda c, _: (c @ x.T @ x, None), c, length=5)[0]

    assert flops.count(scan, jnp.ones((4, 8))).fixed == 5 * 2 * (2 * 4 * 6 * 8)
    depthwise = functools.partial(
        jax.lax.conv_general_dilated,
        window_strides=(1,),
        padding="VALID",
        feature_group_count=6,
    )
    ones = jnp.ones((2, 6, 10)), jnp.ones((6, 1, 3))
    assert flops.count(depthwise, *ones).fixed == 2 * (2 * 6 * 8) * 3


def test_the_log_reports_mfu_where_the_devices_peak_is_known():
    assert flops.peak(jax.devices()[0].device_kind) is None  # a CPU's
    assert flops.peak("NVIDIA B300 SXM6 AC") == 2.25e15
    arch = compile(ARCH + "model = embed block norm head\n")
    loss = objective(tcfg.LossConfig(), arch)
    work = flops.Work(3e6, {flops.Mask(): 2e6})  # 1e6 of attention a step on rows of 8
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(1e-3), loss, work=lambda rows, t: work)
    lr = lr_multiplier(constant(2))
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    latest = LatestMetrics()
    log = Log(every=1, lr=lr, tokens=16, peak=1e9)
    trainer.fit(data, 2, hooks=[log], sinks=[latest])
    steps_per_s = latest.metrics["tokens_per_s"] / 16
    flops_per_step = 3e6 + 2e6 * (8 + 1) / (2 * 8)
    assert trainer.flops == 2 * flops_per_step
    assert latest.metrics["mfu"] == pytest.approx(steps_per_s * flops_per_step / 1e9)


def test_a_pipelined_step_s_flops_are_counted_outside_its_mesh(tmp_path, monkeypatch):
    """Under it, a program of 2 stages holds a stage's share of each of its 3 ticks:
    3/4 of the repeat's FLOPs, where the log's MFU counts them all."""
    monkeypatch.setattr(trainer_module, "_peak", lambda run: 1e15)  # as a GPU's
    run = smoke(
        "steps = 1",
        "mesh = mesh(stage=2)",
        "methods = methods(repeat=scan, pipeline=gpipe, microbatches=2)",
        "precision = precision(compute=float32)",  # XLA:CPU: no bfloat16 pipelines
    )
    kept = Kept()
    train(run, tmp_path, echo=lambda _: None, hooks=[kept])
    assert kept.trainer.flops == pytest.approx(step_flops(run).one_document(64))


@pytest.mark.parametrize("donate", [False, True])
def test_a_step_that_donates_takes_the_memory_of_the_state_it_replaces(donate):
    arch = compile(ARCH + "model = embed block norm head\n")
    loss = objective(tcfg.LossConfig(), arch)
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(1e-3), loss, donate=donate)
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    batch = jax.device_put(next(data))
    step = trainer.lower(batch).compile()
    state = sum(x.nbytes for x in jax.tree.leaves(trainer.state()))
    memory = step.memory_analysis()
    assert memory is not None and memory.alias_size_in_bytes == donate * state


def _small_trainer(trainer: type[Trainer] = Trainer) -> tuple[Trainer, Batch]:
    arch = compile(ARCH + "model = embed block norm head\n")
    loss = objective(tcfg.LossConfig(), arch)
    model = Model(arch, rngs=nnx.Rngs(0))
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    return trainer(model, optax.adamw(1e-3), loss, donate=True), next(data)


def test_a_step_takes_the_state_without_walking_the_model_and_leaves_it_there(
    monkeypatch,
):
    trainer, batch = _small_trainer()
    model = trainer.model
    trainer.train_on_batch(batch)  # compiles
    walks: list[object] = []
    flatten = graphlib.flatten
    monkeypatch.setattr(
        graphlib, "flatten", lambda *a, **k: walks.append(a) or flatten(*a, **k)
    )
    trainer.train_on_batch(batch)
    assert walks == []  # where nnx.jit walked the graph at every call
    monkeypatch.undo()
    alive = [x for x in jax.tree.leaves(nnx.state(model)) if not x.is_deleted()]
    assert len(alive) == len(jax.tree.leaves(nnx.state(model)))  # not the donated


def test_a_step_starts_from_the_values_a_hook_left():
    trainer, batch = _small_trainer()
    trainer.train_on_batch(batch)
    params = nnx.state(trainer.model, nnx.Param)
    nnx.update(trainer.model, jax.tree.map(jnp.zeros_like, params))
    again, _ = _small_trainer()
    nnx.update(again.model, jax.tree.map(jnp.zeros_like, params))
    loss = trainer.train_on_batch(batch)["loss"]
    assert loss == again.train_on_batch(batch)["loss"]


def test_a_step_that_changes_which_variables_the_model_holds_is_an_error():
    class Growing(Trainer):
        def train_step(self, model, optimizer, batch):
            model.extra = nnx.BatchStat(jnp.zeros(()))
            return super().train_step(model, optimizer, batch)

    trainer, batch = _small_trainer(Growing)
    with pytest.raises(ValueError, match="not which variables they hold"):
        trainer.train_on_batch(batch)


def test_the_log_s_learning_rate_is_computed_at_its_step_not_as_it_is_read():
    trainer, batch = _small_trainer()
    asked: list[object] = []
    log = Log(
        every=1, lr=lambda step: asked.append(step) or jnp.asarray(0.1), tokens=16
    )
    log.start(trainer)
    later = log(trainer, trainer.train_on_batch(batch))
    assert asked == [0]  # before the next step is launched, then read
    assert later()["lr"] == pytest.approx(0.1)


def test_eval_launches_every_batch_before_it_reads_a_loss(monkeypatch):
    trainer, batch = _small_trainer()
    events: list[str] = []
    step, means = trainer._eval, trainer_module._means

    def launched(*args):
        events.append("launched")
        return step(*args)

    def read(sums):
        events.append("read")
        return means(sums)

    monkeypatch.setattr(trainer, "_eval", launched)
    monkeypatch.setattr(trainer_module, "_means", read)
    assert set(trainer.evaluate([batch] * 3)) == {"loss"}
    assert events == ["launched"] * 3 + ["read"]
    assert trainer.evaluate([]) == {}  # no batches, no means


def test_held_out_sums_the_loss_over_targets_and_bits_over_their_bytes():
    arch = compile(ARCH + "model = embed block norm head\n")
    model = Model(arch, rngs=nnx.Rngs(0))
    ids = np.arange(1, 17, dtype=np.int32).reshape(2, 8)
    ids[1, 3] = 0  # a special token, which stands for no text
    positions = np.array([[0, 1, 2, 3, 0, 1, 2, 3], list(range(8))], np.int32)
    mask = np.ones(ids.shape, bool)
    mask[1, 5] = False
    batch = Batch(ids, positions, mask)
    nbytes = np.where(ids == 0, 0, ids % 3 + 1).astype(np.int32)
    [head] = forward(model, batch)
    measured = held_out(head, batch, jnp.asarray(nbytes))

    logits = head.logits()[:, :-1]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:])
    targets = np.asarray(target_mask(batch))
    counted = np.where(targets, nbytes[:, 1:], 0)
    (nats, n), (bits, size) = measured["loss"], measured["bpb"]
    assert n == targets.sum() == 12  # 3 + 3 in documents of 4, 7 - 1 masked in one
    assert np.isclose(nats, (ce * targets).sum())
    assert size == counted.sum()
    assert np.isclose(bits, (ce * (counted > 0)).sum() / np.log(2))


def test_eval_weighs_every_target_alike_whichever_batch_it_is_in():
    trainer, batch = _small_trainer()
    fewer = batch._replace(loss_mask=batch.loss_mask & (np.arange(8) < 3))
    each = [trainer.evaluate([b])["loss"] for b in (batch, fewer)]
    targets = [int(target_mask(b).sum()) for b in (batch, fewer)]
    both = trainer.evaluate([batch, fewer])["loss"]
    assert targets == [14, 4]
    assert np.isclose(both, np.dot(each, targets) / sum(targets))
    assert not np.isclose(both, np.mean(each))
    vocab = np.ones(32, np.int32)
    measured = trainer.evaluate([batch, fewer], token_bytes=vocab)
    assert np.isclose(measured["bpb"], both / np.log(2))  # a byte a token


def test_a_run_reports_what_hooks_of_ones_own_return(tmp_path):
    steps = Steps()
    run = smoke("steps = 4")
    last = train(
        run, tmp_path, echo=lambda _: None, hooks=[WeightNorm()], sinks=[steps]
    )
    lines = (directory(run, tmp_path) / "metrics.jsonl").read_text().splitlines()
    assert [s for s, keys in steps.reports if "weight_norm" in keys] == [2, 4]
    assert sum("weight_norm" in json.loads(line) for line in lines) == 2
    assert "weight_norm" in last and "loss" in last


def test_a_file_that_cannot_be_read_is_an_error_not_a_traceback(tmp_path, capsys):
    missing = str(tmp_path / "missing.bux")
    for argv in (["check", missing], ["export", missing, str(tmp_path / "out")]):
        assert main(argv) == 1
        said = capsys.readouterr().err
        assert said == f"error: cannot read {missing}: No such file or directory\n"


def test_the_command_line(tmp_path, capsys):
    assert main(["check", str(SMOKE)]) == 0
    assert "50 steps × 8 × 64 tokens = 25.6k tokens" in capsys.readouterr().out
    assert main(["plan", str(SMOKE), "optimizer.lr = 1e-3 | 3e-3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("2 runs") and "smoke[optimizer.lr=3e-3]" in out
    assert main(["train", str(SMOKE), "steps=2", "--out", str(tmp_path)]) == 1
    assert "the schedule takes 10 steps (10 of warmup)" in capsys.readouterr().err
    settings = ["steps=2", "schedule.warmup=1", "--out", str(tmp_path)]
    assert main(["train", str(SMOKE), *settings]) == 0
    assert "step 2/2" in capsys.readouterr().out
    assert main(["fmt", str(SMOKE), "seed = 0 | 1"]) == 0
    assert capsys.readouterr().out.count("# smoke[seed=") == 2
    assert main(["prepare", str(SMOKE)]) == 1
    assert "reads no documents" in capsys.readouterr().err
    assert main(["check", str(SMOKE), "steps = 0"]) == 1
    assert "'steps' must be greater than 0" in capsys.readouterr().err
