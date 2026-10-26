"""Pipelines over `stage` (methods.pipeline, mesh.stage > 1), against one device: the
longest scanned repeat split over the stages, on methods.microbatches microbatches."""

import re
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import count, equations
from flax import nnx
from jaxtyping import Float

from bux import parallel
from bux.__main__ import main
from bux.architecture import Model, MoE, Scanned
from bux.lang import compile, compile_runs
from bux.parallel import pipeline
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer, _accumulated, directory, train

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
block = [norm attn] [norm ffn]
"""
LLAMA = PRELUDE + "model = embed block^8 norm head"
HC = (
    PRELUDE
    + "residual = hc(expansion=2, dynamic=true)\nmodel = embed block^4 norm head"
)
# Balanced by a loss of each sequence, which the mean over microbatches keeps.
MOE = (
    PRELUDE
    + "experts = moe(num_experts=4, top_k=2, aux_loss_coef=0.01, aux_loss_scope=sequence,"
    + " expert=gated_mlp(intermediate_size=64, activation=swish))\n"
    + "model = embed ([norm attn] [norm experts])^4 norm head"
)
BIASED = MOE.replace(
    "top_k=2,", "top_k=2, load_balancing=aux_free, bias_update_rate=0.01,"
)
ONE, SCAN = "mesh(data=1)", "methods(repeat=scan)"
TOKENS = np.random.default_rng(0).integers(0, 128, (8, 16))
TRAIN = (
    "data = synthetic\noptimizer = adamw(lr=1e-2)\nseq_len = 16\nbatch = 8\nsteps = 4"
    "\nprecision = precision(compute=float32)"  # XLA:CPU: no bfloat16 pipelines
)


def _how(
    microbatches: int, pipeline: str = "gpipe", rounds: int = 1, **more: str
) -> str:
    """The methods of a pipeline: `more` are other fields, `remat="full"`."""
    fields = {"microbatches": microbatches, "pipeline": pipeline, "rounds": rounds}
    return (
        "methods(repeat=scan, "
        + ", ".join(f"{k}={v}" for k, v in (fields | more).items())
        + ")"
    )


def _use(mesh: str, methods: str = SCAN):
    [run] = compile_runs(f"{LLAMA}\n{TRAIN}\nmesh = {mesh}\nmethods = {methods}")
    t = run.train
    return parallel.use(t.mesh, t.layout, t.methods)


def _model(source: str, mesh: str = ONE, methods: str = SCAN) -> Model:
    with _use(mesh, methods):
        return parallel.build(lambda: Model(compile(source), rngs=nnx.Rngs(0)))


def _outputs(
    source: str, mesh: str = ONE, methods: str = SCAN
) -> tuple[Float[np.ndarray, "b t vocab"], list[Float[np.ndarray, " n"]]]:
    """The logits of `source`'s model on TOKENS, and the gradient of their mean, each
    parameter's flattened in a row."""
    model = _model(source, mesh, methods)

    def mean(m: Model) -> Float[jax.Array, ""]:
        return m(jnp.asarray(TOKENS)).mean()

    with _use(mesh, methods):
        logits = nnx.jit(lambda m: m(jnp.asarray(TOKENS)))(model)
        grads = nnx.jit(nnx.grad(mean))(model)
    flat = [np.asarray(g).reshape(-1) for g in jax.tree.leaves(nnx.to_pure_dict(grads))]
    return np.asarray(logits), flat


def _scans(model: nnx.Module) -> list[Scanned]:
    return [node for _, node in nnx.iter_graph(model) if isinstance(node, Scanned)]


def _forward(model: Model, mesh: str, methods: str, training: bool = False) -> None:
    """A forward of `model` on TOKENS, under `mesh` and `methods`: a training one,
    which moves the state its layers change, if `training`."""
    tokens = jnp.asarray(TOKENS)
    with _use(mesh, methods):
        nnx.jit(
            lambda m: m.training_heads(tokens)[0][0].logits() if training else m(tokens)
        )(model)


# --- Against one device ----------------------------------------------------------------

PIPELINES = [
    *[("gpipe", s, m, 1) for s in (2, 4) for m in (4, 8)],
    ("circular", 2, 4, 2),
    ("circular", 4, 4, 2),
    ("circular", 2, 2, 4),
]


@pytest.mark.parametrize(("method", "stages", "microbatches", "rounds"), PIPELINES)
def test_a_pipeline_computes_what_one_device_does(method, stages, microbatches, rounds):
    logits, grads = _outputs(LLAMA)
    how = _how(microbatches, method, rounds)
    piped, piped_grads = _outputs(LLAMA, f"mesh(stage={stages}, data=1)", how)
    assert np.allclose(piped, logits, atol=1e-5)
    for a, b in zip(piped_grads, grads, strict=True):
        assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize(
    ("mesh", "microbatches"),
    [
        ("mesh(stage=2, data=1)", 4),
        ("mesh(stage=4, data=1)", 8),
        ("mesh(stage=2, data=2, fsdp=2)", 4),
        ("mesh(stage=2, fsdp=2, tensor=2)", 4),
        ("mesh(stage=4, tensor=2)", 8),
        ("mesh(stage=4, data=2)", 4),
    ],
)
def test_a_pipeline_trains_as_one_device(losses, mesh, microbatches):
    one = losses(LLAMA, mesh=ONE, methods=SCAN, batch="16")
    piped = losses(LLAMA, mesh=mesh, methods=_how(microbatches), batch="16")
    assert np.allclose(piped, one, atol=1e-5)


def test_a_circular_pipeline_trains_as_one_device(losses):
    one = losses(LLAMA, mesh=ONE, methods=SCAN, batch="16")
    how = _how(4, "circular", 2)
    piped = losses(LLAMA, mesh="mesh(stage=2, fsdp=2)", methods=how, batch="16")
    assert np.allclose(piped, one, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["standard", "gram"])
@pytest.mark.parametrize("orthogonalize", ["gspmd", "gather", "owner"])
@pytest.mark.parametrize("pipeline", ["gpipe", "circular"])
def test_muon_trains_a_pipelined_repeat_as_one_device(
    losses, pipeline, orthogonalize, algorithm
):
    muon = f"muon(lr=0.02, ns_dtype=float32, ns_algorithm={algorithm})"
    rounds = 2 if pipeline == "circular" else 1
    how = _how(4, pipeline, rounds, orthogonalize=orthogonalize)
    mesh, batch = "mesh(stage=2, fsdp=2)", "16"  # 4 microbatches over data × fsdp
    piped = losses(LLAMA, mesh=mesh, optimizer=muon, methods=how, batch=batch)
    one = losses(LLAMA, mesh=ONE, optimizer=muon, methods=SCAN, batch=batch)
    assert np.allclose(piped, one, atol=1e-5)


@pytest.mark.parametrize(
    "remat", ["full", "dots", "save_attention", "remat(save=[attention], first=1)"]
)
def test_a_rematerialized_pipeline_trains_as_one_device(losses, remat):
    piped = losses(LLAMA, mesh="mesh(stage=2, data=1)", methods=_how(4, remat=remat))
    # AdamW at lr=1e-2 grows float32's rounding over the steps to ~5e-5, by any policy.
    assert np.allclose(piped, losses(LLAMA, mesh=ONE, methods=SCAN), atol=1e-4)


def test_a_pipelined_residual_algebra_trains_as_one_device(losses):
    piped = losses(HC, mesh="mesh(stage=4, tensor=2)", methods=_how(4))
    # The losses part from the second step, by float32's rounding of the first update,
    # more on x86 than on arm64: by 7.5e-5 on CI's runners.
    assert np.allclose(piped, losses(HC, mesh=ONE, methods=SCAN), atol=2e-4)


def test_a_pipelined_moe_reports_the_mean_of_its_microbatches_losses(losses):
    piped = losses(MOE, mesh="mesh(stage=2, data=1)", methods=_how(4))
    assert np.allclose(piped, losses(MOE, mesh=ONE, methods=SCAN), atol=1e-5)


def test_a_pipelined_moe_sorts_its_tokens_on_each_devices_rows(losses):
    piped = losses(BIASED, mesh="mesh(stage=2, data=2, fsdp=2)", methods=_how(1))
    assert np.allclose(piped, losses(BIASED, mesh=ONE, methods=SCAN), atol=1e-5)


def _bias(model: Model) -> Float[np.ndarray, "layers experts"]:
    [moe] = [m for _, m in nnx.iter_graph(model) if isinstance(m, MoE)]
    assert moe.router_bias is not None
    return np.asarray(moe.router_bias[...])


def test_the_router_bias_moves_as_the_microbatches_in_a_row_would_move_it():
    micro, mesh = 4, "mesh(stage=2, data=1)"
    alone = _model(BIASED)
    for m in range(micro):
        alone.training_heads(jnp.asarray(TOKENS[m::micro]))
    piped = _model(BIASED, mesh, _how(micro))
    _forward(piped, mesh, _how(micro), training=True)
    assert np.allclose(_bias(piped), _bias(alone), atol=1e-6)


def _stepped_bias(mesh: str, methods: str) -> Float[np.ndarray, "layers experts"]:
    """The router bias of BIASED's model after a training step on TOKENS."""
    [run] = compile_runs(f"{BIASED}\n{TRAIN}\nmesh = {mesh}\nmethods = {methods}")
    t = run.train
    with parallel.use(t.mesh, t.layout, t.methods):
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
        tx = optimizer(t.optimizer, lr_multiplier(t), model, t.clip)
        loss = objective(t.loss, run.arch)
        trainer = Trainer(model, tx, loss, microbatches=_accumulated(4))
        trainer.train_on_batch(plain(TOKENS))
    return _bias(trainer.model)


def test_a_training_step_moves_a_pipelined_router_bias_once_from_the_whole_batch():
    whole = _stepped_bias(ONE, SCAN)
    piped = _stepped_bias("mesh(stage=2, data=1)", _how(4))
    assert np.abs(whole).max() > 0 and np.allclose(piped, whole, atol=1e-6)


def test_dropout_draws_a_mask_a_microbatch():
    dropped = LLAMA.replace("activation=swish)", "activation=swish, dropout=0.1)")
    model = _model(dropped, "mesh(stage=2, data=1)", _how(4))
    _forward(model, "mesh(stage=2, data=1)", _how(4), training=True)
    [scan] = _scans(model)
    counts = [v for _, v in nnx.iter_graph(scan) if isinstance(v, nnx.RngCount)]
    assert counts and all(np.all(np.asarray(c[...]) == 4) for c in counts)


UNEVEN = [
    ("mesh(stage=2, data=1)", "[3, 5]", 4),
    ("mesh(stage=4, data=1)", "[1, 3, 3, 1]", 8),
]


@pytest.mark.parametrize("method", ["gpipe", "1f1b", "zbh1"])
@pytest.mark.parametrize(("mesh", "stages", "microbatches"), UNEVEN)
def test_uneven_stages_train_as_one_device(losses, method, mesh, stages, microbatches):
    how = _how(microbatches, f'"{method}"', stages=stages)
    piped = losses(LLAMA, mesh=mesh, methods=how, batch="16")
    one = losses(LLAMA, mesh=ONE, methods=SCAN, batch="16")
    assert np.allclose(piped, one, atol=1e-5)


def test_passes_that_do_not_divide_split_as_evenly_as_they_go():
    model = _model(PRELUDE + "model = embed block^7 norm head", "mesh(stage=4)", SCAN)
    [scan] = _scans(model)
    kernel = next(v for _, v in nnx.iter_graph(scan) if isinstance(v, nnx.Param))
    assert pipeline.even(7, 4) == (2, 2, 2, 1)
    assert (scan.times, kernel.shape[0]) == (7, 4 * 2)


def test_uneven_stages_hold_as_many_layers_as_the_most_any_runs():
    model = _model(LLAMA, "mesh(stage=2, fsdp=2)", _how(4, stages="[3, 5]"))
    [scan] = _scans(model)
    kernel = next(v for _, v in nnx.iter_graph(scan) if isinstance(v, nnx.Param))
    assert (scan.times, kernel.shape[0]) == (8, 2 * 5)  # passes run, and held
    with _use("mesh(stage=2, fsdp=2)", _how(4, stages="[3, 5]")):
        spec = parallel.shardings(nnx.state(scan, nnx.Param))
    assert all(s.spec[0] == "stage" for s in jax.tree.leaves(spec))


# --- What pipelines --------------------------------------------------------------------


def test_the_longest_scanned_repeat_splits_over_the_stages_and_the_rest_does_not():
    source = PRELUDE + "model = embed block^2 [norm ffn] block^4 norm head"
    model = _model(source, "mesh(stage=2, fsdp=2)", _how(2))
    short, long = _scans(model)
    with _use("mesh(stage=2, fsdp=2)", _how(2)):
        specs = parallel.shardings(nnx.state(model, nnx.Param))
    path = next(p for p, n in nnx.iter_graph(model) if n is long)
    for p, s in nnx.to_flat_state(specs):
        piped = p[: len(path)] == path
        assert ("stage" in jax.tree.leaves(tuple(s.spec))) == piped
    assert (short.rounds, long.rounds) == (0, 1)


def test_a_circular_pipeline_stacks_each_rounds_layers_by_stage():
    model = _model(LLAMA, "mesh(stage=2)", _how(4, "circular", 2))
    inner, outer = _scans(model)
    assert (outer.times, outer.rounds, inner.times) == (2, 2, 4)
    kernel = next(v for _, v in nnx.iter_graph(inner) if isinstance(v, nnx.Param))
    assert kernel.shape[:2] == (2, 4)
    assert kernel.get_metadata()["names"][:2] == ("layers", "stages")


def test_a_pipeline_runs_microbatches_plus_stages_less_one_ticks():
    graphdef, state = nnx.split(_model(LLAMA, "mesh(stage=4)", _how(8)))

    def forward(state: nnx.State) -> Float[jax.Array, "b t vocab"]:
        return nnx.merge(graphdef, state)(jnp.asarray(TOKENS))

    with _use("mesh(stage=4)", _how(8)):
        jaxpr = jax.make_jaxpr(forward)(state).jaxpr
    scans = [e for e, _ in equations(jaxpr) if e.primitive.name == "scan"]
    assert 8 + 4 - 1 in {e.params["length"] for e in scans}
    assert count(jaxpr, "ppermute")


def test_a_pipelined_run_needs_no_full_rematerialization(losses, involuntary):
    involuntary()
    losses(LLAMA, steps=1, mesh="mesh(stage=2, fsdp=4)", methods=_how(2))
    assert involuntary() == 0


def test_pipelined_says_whether_the_mesh_has_stages():
    assert not pipeline.pipelined()
    with _use("mesh(stage=2)", _how(2)):
        assert pipeline.pipelined()
    with _use("mesh(fsdp=2)"):
        assert not pipeline.pipelined()


# --- Errors ----------------------------------------------------------------------------

VALUES = (
    PRELUDE
    + "first = [norm attn(v=@v1)] [norm ffn]\nlater = [norm attn(v=$v1)] [norm ffn]\n"
    + "model = embed first later^4 norm head"
)


def test_a_tap_read_across_the_pipelined_repeat_is_an_error_that_names_it():
    model = _model(VALUES, "mesh(stage=2)", _how(2))
    with pytest.raises(ValueError, match="take v1 from before it"):
        _forward(model, "mesh(stage=2)", _how(2))


@pytest.mark.parametrize(
    ("source", "methods", "error"),
    [
        (LLAMA, "methods(repeat=unroll)", r"methods\(repeat=scan\)"),
        (PRELUDE + "model = embed block norm head", SCAN, r"methods\(repeat=scan\)"),
        (PRELUDE + "model = embed block^3 norm head", SCAN, "a pass a stage"),
        (LLAMA, _how(2, "circular", 3), "a multiple of 12"),
        (LLAMA, _how(2, "circular", 2), "a microbatch a stage"),
        (LLAMA, _how(2, stages="[1, 2, 2, 1]"), "splits 6 passes"),
    ],
    ids=[
        "unrolled",
        "no repeat",
        "too few passes",
        "uneven rounds",
        "too few microbatches",
        "wrong split",
    ],
)
def test_a_model_that_cannot_pipeline_is_an_error_that_says_why(source, methods, error):
    with pytest.raises(ValueError, match=error):
        _model(source, "mesh(stage=4)", methods)


def test_gpipe_goes_round_the_stages_once():
    model = _model(LLAMA, "mesh(stage=2)", _how(2, "gpipe", 2))
    with pytest.raises(ValueError, match=r"methods\(pipeline=circular\)"):
        _forward(model, "mesh(stage=2)", _how(2, "gpipe", 2))


# --- The bubble, and checkpoints -------------------------------------------------------


def test_the_bubble_is_the_share_of_ticks_a_stage_idles():
    assert pipeline.bubble(1, 8) == 0
    assert pipeline.bubble(4, 8) == pytest.approx(3 / 11)
    assert pipeline.bubble(4, 8, rounds=2) == pytest.approx(3 / 19)


def test_the_plan_shows_the_bubble(tmp_path, capsys):
    source = tmp_path / "piped.bux"
    source.write_text(f"{LLAMA}\n{TRAIN}\nmethods = {_how(4)}")
    sweep = "mesh = one: mesh(data=1) | piped: mesh(stage=2, data=1)"
    assert main(["plan", str(source), sweep]) == 0
    head, *rows = capsys.readouterr().out.splitlines()[1:]
    bubble, params = head.split().index("bubble"), head.split().index("params/gpu")
    one, piped = (row.split() for row in rows)
    assert (one[bubble], piped[bubble]) == ("0%", "20%")
    assert one[params] != piped[params]  # the stages split the blocks


def test_a_gpipe_checkpoint_resumes_on_one_stage(tmp_path):
    def run(mesh: str):
        every = "checkpoint = checkpoint(every=2)"
        [r] = compile_runs(
            f"{LLAMA}\n{TRAIN}\nmesh = {mesh}\nmethods = {_how(4)}\n{every}"
        )
        return r

    piped = run("mesh(stage=2, data=1)")
    whole = train(piped, tmp_path, echo=lambda _: None)
    shutil.rmtree(directory(piped, tmp_path) / "checkpoints" / "4")
    said: list[str] = []
    resumed = train(run("mesh(data=1)"), tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 2")
    assert resumed["loss"] == pytest.approx(whole["loss"], abs=1e-5)


def test_an_uneven_pipeline_resumes_with_its_split_alone(tmp_path):
    def run(mesh: str, split: str | None):
        how = _how(4, '"1f1b"', stages=split) if split else "methods(repeat=scan)"
        every = "checkpoint = checkpoint(every=2)"
        [r] = compile_runs(f"{LLAMA}\n{TRAIN}\nmesh = {mesh}\nmethods = {how}\n{every}")
        return r

    piped = run("mesh(stage=4, fsdp=2)", "[3, 2, 2, 1]")
    whole = train(piped, tmp_path, echo=lambda _: None)
    shutil.rmtree(directory(piped, tmp_path) / "checkpoints" / "4")
    # [3, 3, 1, 1] stacks as many layers as [3, 2, 2, 1], each in another place.
    for mesh, split in [("mesh(stage=4, fsdp=2)", "[3, 3, 1, 1]"), ("mesh()", None)]:
        changed = "and this run has " + (split or "none")
        with pytest.raises(ValueError, match=re.escape(changed)):
            train(run(mesh, split), tmp_path, echo=lambda _: None)
    said: list[str] = []
    resumed = train(
        run("mesh(stage=4, data=2)", "[3, 2, 2, 1]"), tmp_path, echo=said.append
    )
    assert said[0].endswith("resuming at step 2")
    assert resumed["loss"] == pytest.approx(whole["loss"], abs=1e-5)


def test_a_pipeline_resumes_on_the_stages_its_run_balanced(tmp_path):
    wide = LLAMA.replace("vocab_size = 128", "vocab_size = 4096")  # a head of 4 blocks

    def run(split: str | None):
        how = _how(4, '"1f1b"', stages=split) if split else _how(4, '"1f1b"')
        every = "checkpoint = checkpoint(every=2)"
        mesh = "mesh(stage=2, data=1)"
        [r] = compile_runs(f"{wide}\n{TRAIN}\nmesh = {mesh}\nmethods = {how}\n{every}")
        return r

    said: list[str] = []
    whole = train(run(None), tmp_path, echo=said.append)
    assert said[0].endswith("methods.stages = [6, 2]")
    shutil.rmtree(directory(run(None), tmp_path) / "checkpoints" / "4")
    with pytest.raises(
        ValueError, match=re.escape("saved with methods.stages = [6, 2]")
    ):
        train(run("[4, 4]"), tmp_path, echo=lambda _: None)
    said.clear()
    resumed = train(run(None), tmp_path, echo=said.append)
    assert said[1].endswith("resuming at step 2")
    assert resumed["loss"] == pytest.approx(whole["loss"], abs=1e-5)
