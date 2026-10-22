"""Repeats scanned (methods.repeat=scan) and rematerialization (methods.remat), against
repeats unrolled and nothing rematerialized, on one device and split over 8."""

import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Float
from optax.contrib import MuonDimensionNumbers

from bux import parallel
from bux.architecture import Model, MoE, Norm, Scanned
from bux.architecture.graph import Sequence
from bux.architecture.residual import HyperConnections
from bux.lang import compile, compile_runs
from bux.train.optim import muon_matrices, weight_decay_mask
from bux.train.trainer import directory, train

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


@pytest.mark.parametrize("orthogonalize", ["gspmd", "gather", "owner"])
@pytest.mark.parametrize("mesh", [ONE, "mesh(fsdp=8)"])
def test_muon_trains_a_scanned_repeat_as_the_unrolled_one(losses, mesh, orthogonalize):
    muon, how = "muon(lr=0.02)", f"methods(repeat=scan, orthogonalize={orthogonalize})"
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


def _compiled(source: str, methods: str) -> tuple[float, jax.stages.Compiled]:
    """The seconds the gradient of a forward of `source` takes to trace and compile,
    and what it compiles to."""
    model = _model(source, methods)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def grads(params: nnx.State, rest: nnx.State) -> nnx.State:
        m = nnx.merge(graphdef, params, rest)
        return nnx.grad(lambda m: m(jnp.asarray(TOKENS)).mean())(m)

    with _use(methods):
        start = time.perf_counter()
        compiled = jax.jit(grads).lower(params, rest).compile()
    return time.perf_counter() - start, compiled


def test_scanning_24_layers_compiles_faster_than_unrolling_them():
    deep = PRELUDE + "model = embed block^24 norm head"
    scanned, _ = _compiled(deep, SCAN)
    unrolled, _ = _compiled(deep, UNROLL)
    assert scanned < unrolled / 2


@pytest.mark.parametrize("remat", ["full", "dots", "dots_no_batch"])
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
TAPS = {"moe": MOE, "values": VALUES, "single_pass": SINGLE_PASS}


@pytest.mark.parametrize("arch", TAPS)
def test_a_rematerialized_branch_writes_its_taps_and_losses_as_it_would(losses, arch):
    plain = losses(TAPS[arch])
    for repeat in ("unroll", "scan"):
        how = f"methods(repeat={repeat}, remat=full)"
        assert np.allclose(losses(TAPS[arch], methods=how), plain, atol=1e-5)


@pytest.mark.parametrize("repeat", ["unroll", "scan"])
@pytest.mark.parametrize("remat", ["full", "dots", "dots_no_batch"])
def test_remat_needs_less_temp_memory(remat, repeat):
    def temp(remat: str) -> int:
        _, c = _compiled(LLAMA, f'methods(repeat={repeat}, remat="{remat}")')
        memory = c.memory_analysis()
        assert memory is not None
        return memory.temp_size_in_bytes

    assert temp(remat) < temp("none")  # quoted: a bare none is no name


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
