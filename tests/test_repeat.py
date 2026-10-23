"""Repeats scanned (methods.repeat=scan) and rematerialization (methods.remat), against
repeats unrolled and nothing rematerialized, on one device and split over 8."""

import shutil
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax._src.ad_checkpoint import saved_residuals
from jax.extend.core import ClosedJaxpr, Jaxpr
from jax.sharding import PartitionSpec as P
from jaxtyping import Float
from optax.contrib import MuonDimensionNumbers

from bux import parallel
from bux.architecture import Model, MoE, Norm, Scanned, matmuls, remat
from bux.architecture.graph import Sequence
from bux.architecture.residual import HyperConnections
from bux.lang import compile, compile_runs
from bux.lang.printer import run_to_source
from bux.parallel import methods
from bux.parallel.context import xla
from bux.train.config import RematConfig
from bux.train.optim import muon_matrices, weight_decay_mask
from bux.train.trainer import cast, directory, train

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
block = [norm attn] [norm ffn]
"""
LLAMA = PRELUDE + "model = embed block^3 norm head"
# Jamba's pattern. Gated DeltaNet's (Qwen3-Next's) forward is the same scanned, but its
# training at lr=1e-2 drifts from itself rematerialized too: Adam's first steps
# amplify its rounding.
PATTERN = (
    PRELUDE
    + "D = [norm mamba2(state_size=4, head_dim=8)] [norm ffn]\nA = block\n"
    + 'model = embed "DDDA"^2 norm head'
)
GDN = PATTERN.replace(
    "mamba2(state_size=4, head_dim=8)", "gdn(num_heads=2, head_dim=8)"
)
HC = PRELUDE + "residual = hc(expansion=3, dynamic=true)\n" + LLAMA.split("\n")[-1]
MOE = (
    PRELUDE
    + "experts = moe(num_experts=4, top_k=2, load_balancing=aux_free, "
    + "bias_update_rate=0.01, aux_loss_coef=0.01, "
    + "expert=gated_mlp(intermediate_size=64, activation=swish))\n"
    + "model = embed ([norm attn] [norm experts])^3 norm head"
)
SCANNED = {"llama": LLAMA, "pattern": PATTERN, "hc": HC}
ONE = "mesh(data=1)"
MESHES = [ONE, "mesh(fsdp=8)", "mesh(fsdp=2, expert=2, tensor=2)"]
SCAN, UNROLL = "methods(repeat=scan)", "methods(repeat=unroll)"
# The policies that keep what layers name (bux.architecture.remat).
NAMED = ["save_attention", "save_qkv_attention", "dots_attention", "offload_attention"]
TOKENS = np.random.default_rng(0).integers(0, 128, (8, 16))
TRAIN = "data = synthetic\noptimizer = adamw(lr=1e-2)\nseq_len = 16\nbatch = 8\n"


def _run(source: str, *settings: str):
    [run] = compile_runs("\n".join([source, TRAIN, "steps = 4", *settings]))
    return run


def _use(methods: str, mesh: str = ONE):
    """`parallel.use`, with the settings `methods` and `mesh` say."""
    t = _run(LLAMA, f"methods = {methods}", f"mesh = {mesh}").train
    return parallel.use(t.mesh, t.layout, t.methods)


def _model(source: str, methods: str = SCAN, mesh: str = ONE) -> Model:
    with _use(methods, mesh):
        return parallel.build(lambda: Model(compile(source), rngs=nnx.Rngs(0)))


def _scans(model: nnx.Module) -> list[Scanned]:
    return [node for _, node in nnx.iter_graph(model) if isinstance(node, Scanned)]


def _first[T](kind: type[T], model: nnx.Module) -> T:
    return next(node for _, node in nnx.iter_graph(model) if isinstance(node, kind))


def _path(model: nnx.Module, node: object) -> tuple:
    return next(path for path, n in nnx.iter_graph(model) if n is node)


# --- Scan against unroll ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "mesh"),
    [(arch, mesh) for arch in SCANNED for mesh in MESHES] + [("gdn", ONE)],
)
def test_a_scanned_repeat_computes_what_the_unrolled_one_does(arch, mesh):
    source = (SCANNED | {"gdn": GDN})[arch]
    scanned = _logits(source, SCAN, mesh)
    assert np.allclose(scanned, _logits(source, UNROLL, mesh), atol=1e-5)


def _logits(
    source: str, methods: str, mesh: str = ONE
) -> Float[np.ndarray, "b t vocab"]:
    model = _model(source, methods, mesh)
    with _use(methods, mesh):
        return np.asarray(nnx.jit(lambda m, t: m(t))(model, TOKENS))


@pytest.mark.parametrize("mesh", MESHES)
@pytest.mark.parametrize("arch", SCANNED)
def test_a_scanned_repeat_trains_as_the_unrolled_one(losses, arch, mesh):
    scanned = losses(SCANNED[arch], mesh=mesh, methods=SCAN)
    assert np.allclose(scanned, losses(SCANNED[arch], mesh=ONE), atol=1e-5)


def test_a_scanned_repeat_stacks_the_passes_unroll_builds():
    graph, [scan] = _model(HC, UNROLL).graph, _scans(_model(HC))
    assert isinstance(graph, Sequence) and isinstance(graph.items[1], Sequence)
    passes = [dict(nnx.to_flat_state(nnx.state(p))) for p in graph.items[1].items]
    for path, x in nnx.to_flat_state(nnx.state(scan.layers)):
        assert x.get_metadata()["names"][0] == "layers"
        for i, p in enumerate(passes):
            assert jnp.array_equal(x[...][i], p[path][...])
    alpha = _first(HyperConnections, scan).static_alpha[...]
    assert not jnp.array_equal(alpha[0], alpha[1])  # hc starts by depth


def test_a_scanned_moe_reports_its_losses_and_moves_its_bias_as_unrolled(losses):
    assert _scans(_model(MOE))
    scanned = losses(MOE, steps=4, methods=SCAN)
    assert np.allclose(scanned, losses(MOE, steps=4), atol=1e-5)


@pytest.mark.parametrize("arch", ["llama", "moe"])
def test_a_scanned_repeat_casts_each_pass_as_it_runs_it(arch):
    model = _model((SCANNED | {"moe": MOE})[arch])
    by_pass = cast(model, jnp.bfloat16)
    [scan] = _scans(by_pass)
    stacked = jax.tree.leaves(nnx.state(scan.layers, nnx.Param))
    assert scan.compute == jnp.bfloat16
    assert all(x.dtype == jnp.float32 for x in stacked)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    at_once = nnx.merge(graphdef, params, rest)
    with _use(SCAN):
        y, z = (nnx.jit(lambda m, t: m(t))(m, TOKENS) for m in (by_pass, at_once))
    assert y.dtype == jnp.bfloat16 and jnp.array_equal(y, z)


@pytest.mark.parametrize("orthogonalize", ["gspmd", "gather", "owner"])
@pytest.mark.parametrize("mesh", [ONE, "mesh(fsdp=8)"])
def test_muon_trains_a_scanned_repeat_as_the_unrolled_one(losses, mesh, orthogonalize):
    muon = "muon(lr=0.02, ns_dtype=float32)"
    how = f"methods(repeat=scan, orthogonalize={orthogonalize})"
    scanned = losses(MOE, mesh=mesh, optimizer=muon, methods=how)
    assert np.allclose(scanned, losses(MOE, optimizer=muon), atol=1e-5)


# --- What scans ------------------------------------------------------------------------


def _scan_lengths(model: Model) -> list[int]:
    graphdef, state = nnx.split(model)
    tokens = jnp.asarray(TOKENS)
    with _use(SCAN):
        jaxpr = jax.make_jaxpr(lambda s: nnx.merge(graphdef, s)(tokens))(state)
    return [e.params["length"] for e in jaxpr.eqns if e.primitive.name == "scan"]


def test_a_repeat_runs_its_passes_in_one_loop():
    assert _scan_lengths(_model(LLAMA)) == [3]
    assert _scan_lengths(_model(PATTERN)) == [2]


def test_a_repeat_of_repeats_stacks_twice():
    nested = PRELUDE + "model = embed (block^2 [norm ffn])^3 norm head"
    model = _model(nested)
    [inner] = [scan for scan in _scans(model) if scan.times == 2]
    kernel = _path(model, _first(nnx.Linear, inner)) + ("kernel",)
    assert muon_matrices(model)[kernel] == MuonDimensionNumbers(2, 3)
    assert _scan_lengths(model) == [3]
    assert np.allclose(_logits(nested, SCAN), _logits(nested, UNROLL), atol=1e-5)


UNSTACKED = {
    "shared": PRELUDE + "shared sa = block\nmodel = embed (block sa)^2 norm head",
    "attnres": PRELUDE
    + "residual = attnres(block_size=2)\nmodel = embed block^2 merge norm head",
    "nextn": PRELUDE
    + "mtp = $h! nextn [norm ffn] norm @h head\n"
    + "model = embed block norm @h head mtp^2",
}


@pytest.mark.parametrize("arch", UNSTACKED)
def test_a_repeat_unrolls_where_its_passes_do_not_stack(losses, arch):
    assert not _scans(_model(UNSTACKED[arch]))
    scanned = losses(UNSTACKED[arch], methods=SCAN)
    assert np.allclose(scanned, losses(UNSTACKED[arch]), atol=1e-5)


UNET = PRELUDE + "model = embed (block @skip)^2 ([$skip! scale] block)^2 norm head"
SINGLE_PASS = (
    PRELUDE
    + "residual = mhc(expansion=2, single_pass=true)\n"
    + "model = embed block^3 merge norm head"
)


def test_a_repeat_that_pushes_to_a_tap_runs_its_layers_one_by_one(losses):
    assert len(_scans(_model(UNET))) == 2 and _scan_lengths(_model(UNET)) == []
    assert np.allclose(losses(UNET, methods=SCAN), losses(UNET), atol=1e-5)


def test_single_pass_mhc_scans_from_the_layer_after_the_one_that_starts_its_read():
    assert _scan_lengths(_model(SINGLE_PASS)) == [2]
    scanned = _logits(SINGLE_PASS, SCAN)
    assert np.allclose(scanned, _logits(SINGLE_PASS, UNROLL), atol=1e-5)


# --- The optimizer and the layout ------------------------------------------------------


def test_muon_takes_the_stacked_dimension_as_a_batch_dimension():
    model = _model(MOE)
    dims, [scan] = muon_matrices(model), _scans(model)
    up = _path(model, _first(MoE, scan)) + ("up",)
    kernel = _path(model, _first(nnx.Linear, scan)) + ("kernel",)
    assert dims[up] == MuonDimensionNumbers(2, 3)
    assert dims[kernel] == MuonDimensionNumbers(1, 2)


def test_weight_decay_leaves_stacked_norm_scales_alone():
    model = _model(LLAMA)
    [scan] = _scans(model)
    mask = dict(
        nnx.to_flat_state(weight_decay_mask(model)(nnx.state(model, nnx.Param)))
    )
    scale = _path(model, _first(Norm, scan)) + ("scale",)
    kernel = _path(model, _first(nnx.Linear, scan)) + ("kernel",)
    assert mask[scale].get_value() is False
    assert mask[kernel].get_value() is True


def test_fsdp_never_splits_the_stacked_layers():
    deep = PRELUDE + "residual = hc(expansion=2)\nmodel = embed block^8 merge norm head"
    [scan] = _scans(_model(deep, mesh="mesh(fsdp=8)"))
    with _use(SCAN, "mesh(fsdp=8)"):
        specs = parallel.shardings(nnx.state(scan.layers))
    assert all(s.spec[0] is None for s in jax.tree.leaves(specs))


def test_scanned_fsdp_needs_no_full_rematerialization(losses, involuntary):
    involuntary()
    losses(LLAMA, steps=1, mesh="mesh(fsdp=8)", methods=SCAN)
    assert involuntary() == 0


# --- Compile time and remat ------------------------------------------------------------


def _gradient(source: str, methods: str) -> tuple[Any, tuple[nnx.State, nnx.State]]:
    """The gradient of a forward of `source`, jitted, and what it takes."""
    model = _model(source, methods)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def grads(params: nnx.State, rest: nnx.State) -> nnx.State:
        m = nnx.merge(graphdef, params, rest)
        return nnx.grad(lambda m: m(jnp.asarray(TOKENS)).mean())(m)

    return jax.jit(grads), (params, rest)


def _compiled(source: str, methods: str) -> tuple[float, jax.stages.Compiled]:
    """The seconds the gradient of a forward of `source` takes to trace and compile,
    and what it compiles to."""
    grads, args = _gradient(source, methods)
    with _use(methods):
        start = time.perf_counter()
        compiled = grads.lower(*args).compile()
    return time.perf_counter() - start, compiled


def test_scanning_24_layers_compiles_faster_than_unrolling_them():
    deep = PRELUDE + "model = embed block^24 norm head"
    scanned, _ = _compiled(deep, SCAN)
    unrolled, _ = _compiled(deep, UNROLL)
    assert scanned < unrolled / 2


REMATS = ["full", "dots", "dots_no_batch", *NAMED]


@pytest.mark.parametrize("remat", REMATS)
def test_remat_trains_the_same(losses, remat):
    plain = losses(HC)
    for repeat in ("unroll", "scan"):
        rematted = losses(HC, methods=f"methods(repeat={repeat}, remat={remat})")
        assert np.allclose(rematted, plain, atol=1e-5)


VALUES = (
    PRELUDE
    + "first = [norm attn(v=@v1)] [norm ffn]\nlater = [norm attn(v=$v1)] [norm ffn]\n"
    + "model = embed first later^2 norm head"
)
# Each pass of the repeat runs a nextn and a head, rematerialized as a whole by scan.
NEXTN = (
    PRELUDE
    + "mtp = $h! nextn [norm ffn] norm @h head\n"
    + "model = embed block norm @h head mtp^2"
)
TAPS = {"moe": MOE, "values": VALUES, "single_pass": SINGLE_PASS, "nextn": NEXTN}


@pytest.mark.parametrize("arch", TAPS)
def test_a_rematerialized_branch_writes_its_taps_and_losses_as_it_would(losses, arch):
    plain = losses(TAPS[arch])
    for repeat in ("unroll", "scan"):
        how = f"methods(repeat={repeat}, remat=full)"
        assert np.allclose(losses(TAPS[arch], methods=how), plain, atol=1e-5)


def _temp(source: str, methods: str) -> int:
    _, c = _compiled(source, methods)
    memory = c.memory_analysis()
    assert memory is not None
    return memory.temp_size_in_bytes


@pytest.mark.parametrize("repeat", ["unroll", "scan"])
@pytest.mark.parametrize("remat", REMATS)
def test_remat_needs_less_temp_memory(remat, repeat):
    # Than keeping everything, as an unrolled `none` does: a scanned one keeps less
    # (`remat.unfused`). Quoted: a bare none is no name.
    keep_all = _temp(LLAMA, 'methods(repeat=unroll, remat="none")')
    assert _temp(LLAMA, f'methods(repeat={repeat}, remat="{remat}")') < keep_all


@pytest.mark.parametrize("arch", [*SCANNED, "moe"])
def test_a_scanned_repeat_needs_no_more_temp_memory_than_the_unrolled_one(arch):
    source = (SCANNED | {"moe": MOE})[arch]
    assert _temp(source, SCAN) <= _temp(source, UNROLL)


@jax.custom_vjp
def _cube_root(q: Float[jax.Array, "*s"]) -> Float[jax.Array, "*s"]:
    """A kernel whose backward takes what only its forward makes, as a flash kernel's
    takes its log-sum-exp, which no policy by products of matrices keeps."""
    return jnp.cbrt(q)


def _cube_root_forward(
    q: Float[jax.Array, "*s"],
) -> tuple[Float[jax.Array, "*s"], Float[jax.Array, "*s"]]:
    root = jnp.cbrt(q)
    return root, root


_cube_root.defvjp(_cube_root_forward, lambda root, g: (g / (3 * root**2 + 1),))


@methods.attention.register("cube_root")
def _(q, k, v, args):
    return xla(_cube_root(q), k, v, args)


@pytest.mark.parametrize(
    ("remat", "runs"), [("none", 1), ("dots", 2), *((name, 1) for name in NAMED)]
)
def test_a_policy_that_keeps_attention_runs_its_kernel_once(remat, runs):
    how = f'methods(attention=cube_root, remat="{remat}")'
    grads, args = _gradient(LLAMA, how)
    with _use(how):
        text = grads.lower(*args).as_text()
    assert text.count("stablehlo.cbrt") == 3 * runs  # 3 layers


# --- Remat as data ---------------------------------------------------------------------


def test_a_policy_is_data_that_the_language_writes_and_prints():
    how = "methods(remat=remat(save=[attention.in, mlp], offload=qkv, first=2))"
    [run] = compile_runs(f"{LLAMA}\n{TRAIN}steps = 1\nmethods = {how}")
    policy = run.train.methods.remat
    assert policy == RematConfig(
        save=("attention.in", "mlp"), offload=("qkv",), first=2
    )
    assert compile_runs(run_to_source(run))[0].train.methods == run.train.methods


def test_a_policy_as_data_keeps_what_the_preset_of_its_names_does():
    preset = _temp(LLAMA, "methods(remat=save_qkv_attention)")
    names = "methods(remat=remat(save=[attention, recurrence, qkv]))"
    assert _temp(LLAMA, names) == preset > _temp(LLAMA, "methods(remat=full)")


# Policies of every kind, rematerializing some passes and not others: the second scan
# of single-pass mhc starts from what the first leaves in the taps.
FIRST = [
    "remat(first=1)",
    "remat(save=[attention, mlp], first=2)",
    "remat(save=dots_no_batch, offload=mlp, first=2)",
]


@pytest.mark.parametrize("remat", FIRST)
@pytest.mark.parametrize("arch", ["hc", "single_pass"])
def test_a_policy_of_the_first_passes_trains_the_same(losses, arch, remat):
    source = {"hc": HC, "single_pass": SINGLE_PASS}[arch]
    plain = losses(source)
    for repeat in ("unroll", "scan"):
        rematted = losses(source, methods=f"methods(repeat={repeat}, remat={remat})")
        assert np.allclose(rematted, plain, atol=1e-5)


@pytest.mark.parametrize(("repeat", "runs"), [("unroll", 4), ("scan", 3)])
def test_a_policy_of_the_first_pass_runs_only_its_kernel_twice(repeat, runs):
    # Unrolled, the first of 3 layers runs it twice; scanned, the first pass is a scan
    # of its own, its body once in the forward and once in the backward, and the rest
    # another, once.
    how = f"methods(attention=cube_root, repeat={repeat}, remat=remat(first=1))"
    grads, args = _gradient(LLAMA, how)
    with _use(how):
        text = grads.lower(*args).as_text()
    assert text.count("stablehlo.cbrt") == runs


def _scaled_matmuls(how: str) -> int:
    """How many MXFP8 GEMMs the gradient of tanh(x W) takes, inside a shard_map
    rematerialized by `how`."""

    def project(
        x: Float[jax.Array, "m k"], w: Float[jax.Array, "k n"]
    ) -> Float[jax.Array, "m n"]:
        return jnp.tanh(matmuls.dot_general(x, w, (((1,), (0,)), ((), ()))))

    def loss(
        x: Float[jax.Array, "m k"], w: Float[jax.Array, "k n"]
    ) -> Float[jax.Array, ""]:
        rematted = remat.checkpoint(project, prevent_cse=True)
        return jax.shard_map(rematted, in_specs=P(), out_specs=P())(x, w).sum()

    x = jnp.ones((64, 64))
    with _use(f"methods(remat={how})"), matmuls.use("mxfp8", jnp.asarray(TOKENS)):
        jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(x, x)
    return _count(jaxpr.jaxpr, "scaled_matmul")


def test_a_dots_policy_keeps_an_mxfp8_gemm_inside_a_shard_map():
    # There blockscaled binds jax's inner `scaled_matmul`, which jax's dots policy
    # misses: its backward would run the forward's GEMM again for tanh's.
    assert _scaled_matmuls("dots") == _scaled_matmuls("dots_no_batch") == 3
    assert _scaled_matmuls("full") == 4


def _count(jaxpr: Jaxpr, primitive: str) -> int:
    """How many times `primitive` runs in `jaxpr`, inside the jaxprs it calls too."""
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == primitive
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                if isinstance(sub, ClosedJaxpr):
                    sub = sub.jaxpr
                if isinstance(sub, Jaxpr):
                    n += _count(sub, primitive)
    return n


MLA = "mla(num_heads=4, kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8"
SPARSE = PRELUDE + (
    f"sparse = {MLA}, indexer=indexer(num_heads=2, head_dim=8, top_k=4))\n"
    "model = embed ([norm sparse] [norm ffn])^2 norm head"
)


def _saved(source: str, how: str) -> tuple[str, list[tuple[tuple[int, ...], str]]]:
    """The gradient of a training forward of `source` under `how`, as text, and what
    its forward saves for its backward."""
    model = _model(source, f"methods(remat={how})")
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss(params: nnx.State, rest: nnx.State) -> Float[jax.Array, ""]:
        m = nnx.merge(graphdef, params, rest, copy=True)
        heads, losses = m.training_heads(jnp.asarray(TOKENS))
        return heads[0].logits().mean() + sum(loss.value for loss in losses)

    with _use(f"methods(remat={how})"):
        text = jax.jit(jax.grad(loss)).lower(params, rest).as_text()
        saved = saved_residuals(loss, params, rest)
    return text, [(getattr(aval, "shape", ()), what) for aval, what in saved]


@pytest.mark.parametrize(
    ("remat", "runs"),
    [
        ('"none"', 1),
        ("full", 2),
        ("save_attention", 1),
        ("remat(save=[qkv, attention.out, attention.index])", 1),
    ],
)
def test_a_policy_that_keeps_sparse_attention_runs_its_indexer_once(remat, runs):
    text, _ = _saved(SPARSE, remat)
    # The top-k of each layer's indexer counts its scores' bits by shifts.
    assert text.count("stablehlo.shift_left") == 2 * runs


def test_mla_keeps_its_latents_for_a_policy_that_keeps_qkv():
    # And its backward projects the heads up from them again, as DeepSeek's does.
    _, saved = _saved(SPARSE, "save_qkv_attention")
    widths = {shape[-1] for shape, what in saved if "named 'qkv'" in what}
    assert widths == {32, 16 + 8}  # the query's latent; the keys' and values' and RoPE


# --- Checkpoints -----------------------------------------------------------------------


def test_a_checkpoint_resumes_only_with_the_repeat_method_it_was_saved_with(tmp_path):
    def run(repeat: str):
        every = "checkpoint = checkpoint(every=2)"
        return _run(LLAMA, f"methods = methods(repeat={repeat})", every)

    whole = train(run("scan"), tmp_path, echo=lambda _: None)
    shutil.rmtree(directory(run("scan"), tmp_path) / "checkpoints" / "4")
    with pytest.raises(ValueError, match=r"saved with methods\(repeat=unroll\)"):
        train(run("unroll"), tmp_path, echo=lambda _: None)
    said: list[str] = []
    resumed = train(run("scan"), tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 2")
    assert resumed["loss"] == pytest.approx(whole["loss"], abs=1e-5)
