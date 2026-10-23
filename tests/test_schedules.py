"""Pipelines that schedule their own backward (methods.pipeline = 1f1b, zbh1), against
one device: their tables, the backward split in two, and the runs they train."""

from collections import Counter

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jaxtyping import Float, Shaped

from bux import parallel
from bux.__main__ import main
from bux.architecture import Model, MoE, Scanned, matmuls, pipelined
from bux.lang import compile, compile_runs
from bux.parallel import pipeline, schedule, split
from bux.parallel.scheduled import _rebuilt
from bux.train.data import plain
from bux.train.objective import objective
from bux.train.optim import lr_multiplier, optimizer
from bux.train.trainer import Trainer, _accumulated

PRELUDE = """
d_model = 64
vocab_size = 128
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)
block = [norm attn] [norm ffn]
"""
LLAMA = PRELUDE + "model = embed block^8 norm head"
EXPERTS = (
    "experts = moe(num_experts=4, top_k=2, aux_loss_coef=0.01, aux_loss_scope=sequence,"
    " expert=gated_mlp(intermediate_size=64, activation=swish))\n"
)
MOE = PRELUDE + EXPERTS + "model = embed ([norm attn] [norm experts])^4 norm head"
BIASED = MOE.replace(
    "top_k=2,", "top_k=2, load_balancing=aux_free, bias_update_rate=0.01,"
)
ONE, SCAN = "mesh(data=1)", "methods(repeat=scan)"
TOKENS = np.random.default_rng(0).integers(0, 128, (16, 16))
TRAIN = (
    "data = synthetic\noptimizer = adamw(lr=1e-2)\nseq_len = 16\nbatch = 16\nsteps = 4"
    "\nprecision = precision(compute=float32)"  # XLA:CPU: no bfloat16 pipelines
)
PIPELINES = list(schedule.TABLES)


def _how(pipeline: str, microbatches: int, **more: str) -> str:
    """The methods of a scheduled pipeline: `more` are other fields, `remat="full"`."""
    fields = {"microbatches": str(microbatches), "pipeline": f'"{pipeline}"'} | more
    return f"methods(repeat=scan, {', '.join(f'{k}={v}' for k, v in fields.items())})"


def _run(source: str, mesh: str, methods: str):
    [run] = compile_runs(f"{source}\n{TRAIN}\nmesh = {mesh}\nmethods = {methods}")
    return run


def _trainer(source: str, mesh: str, methods: str) -> Trainer:
    """A trainer of `source`'s model, made under `mesh` and `methods`."""
    run = _run(source, mesh, methods)
    t = run.train
    with parallel.use(t.mesh, t.layout, t.methods):
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
        tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
        microbatches = _accumulated(t.methods.microbatches)
        loss = objective(t.loss, run.arch)
        return Trainer(
            model, tx, loss, microbatches=microbatches, accumulate=t.methods.accumulate
        )


def _step_grads(mesh: str, methods: str) -> list[Float[np.ndarray, " n"]]:
    """The gradient a training step of LLAMA takes on TOKENS, each parameter's
    flattened in a row."""
    trainer = _trainer(LLAMA, mesh, methods)
    t = _run(LLAMA, mesh, methods).train
    with parallel.use(t.mesh, t.layout, t.methods):
        batch = parallel.shard_batch(plain(TOKENS))
        grads = nnx.jit(lambda m, b: trainer.gradients(m, b)[0])(trainer.model, batch)
    return [np.asarray(g).reshape(-1) for g in jax.tree.leaves(nnx.to_pure_dict(grads))]


# --- Tables ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize(("stages", "microbatches"), [(2, 2), (2, 8), (3, 5), (4, 8)])
def test_a_table_runs_every_op_once_after_what_it_reads(name, stages, microbatches):
    table = schedule.TABLES[name](stages, microbatches)
    when: dict[tuple[str, int, int], tuple[int, int]] = {}
    for (t, s, k), op in table._each():
        assert (op.kind, s, op.micro) not in when
        when[op.kind, s, op.micro] = (t, k)
    kinds = "FBW" if table.split else "FB"
    ops = [
        (kind, s, j)
        for kind in kinds
        for s in range(stages)
        for j in range(microbatches)
    ]
    assert set(when) == set(ops)
    for (kind, s, j), (t, k) in when.items():
        if kind == "F" and s > 0:
            assert when["F", s - 1, j][0] < t  # sent at the end of an earlier tick
        if kind == "B":
            assert when["F", s, j] < (t, k)
            assert s == stages - 1 or when["B", s + 1, j][0] < t
        if kind == "W":
            assert when["B", s, j] < (t, k)


def _slots(table: schedule.Table, buffer: str) -> int:
    return int(table.slots()[buffer].max()) + 1


def test_1f1b_holds_fewer_microbatches_than_gpipe_and_idles_as_much():
    table = schedule.one_f_one_b(4, 8)
    assert _slots(table, "residuals") == 2 * 4 - 1
    assert table.bubble() == pytest.approx(pipeline.bubble(4, 8))


def test_zbh1_holds_a_microbatch_a_stage_and_idles_a_tick_a_stage():
    table = schedule.zero_bubble_h1(4, 8)
    assert table.ticks == 3 * 8 + 4 - 1
    assert _slots(table, "residuals") == _slots(table, "weights") == 4
    assert table.bubble() == pytest.approx(3 / 27)


# --- The backward split in two --------------------------------------------------------


def _dots(f, *args) -> int:
    """The matmuls `f` runs, each as many times as the scans around it."""

    def count(jaxpr: jax.extend.core.Jaxpr, times: int) -> int:
        n = 0
        for e in jaxpr.eqns:
            n += times * (e.primitive.name == "dot_general")
            inner = times * e.params["length"] if e.primitive.name == "scan" else times
            for sub in jax.extend.core.jaxprs_in_params(e.params):
                n += count(sub, inner)
        return n

    return count(jax.make_jaxpr(f)(*args).jaxpr, 1)


SPLITS = [("none", None), ("full", None), ("none", [True, False, True, True])]


@pytest.mark.parametrize(("remat", "runs"), SPLITS, ids=["", "remat", "skipping"])
def test_a_split_backward_takes_the_whole_ones_gradients_and_each_matmul_once(
    remat, runs
):
    """Of 4 Llama blocks, a stage's (the blocks `runs` marks, by a cond each)."""
    with _use(ONE, f"methods(repeat=scan, remat={remat})"):
        model = Model(
            compile(PRELUDE + "model = embed block^4 norm head"), rngs=nnx.Rngs(0)
        )
        [scanned] = [m for _, m in nnx.iter_graph(model) if isinstance(m, Scanned)]
        graphdef, params, rest = nnx.split(scanned.layers, nnx.Param, ...)
        stage = pipelined.Stage(graphdef, 4, True, None)
        x, dy = (jax.random.normal(jax.random.key(i), (2, 16, 64)) for i in range(2))
        rows = pipelined.Rows(None, jnp.zeros((2, 16), jnp.int32))

        def forward(
            p: nnx.State, x: Float[jax.Array, "n t d"]
        ) -> Float[jax.Array, "n t d"]:
            if runs is None:
                return stage(p, rest, x, rows)[0]
            taps = pipelined._stage_taps(rows, True)
            state = nnx.merge_state(p, rest)
            return pipelined._skipping(graphdef, state, jnp.array(runs), x, taps, None)[
                0
            ]

        _, vjp = jax.vjp(forward, params, x, saveable_args=(False, True))
        flat, tree = jax.tree.leaves(params), jax.tree.structure(params)
        residuals = jax.tree.leaves(vjp)

        def backward(*args: Shaped[jax.Array, "..."]) -> list[Shaped[jax.Array, "..."]]:
            p, (*r, d) = args[: len(flat)], args[len(flat) :]
            d_params, dx = _rebuilt(vjp, r, jax.tree.unflatten(tree, p))(d)
            return [*jax.tree.leaves(d_params), dx]

        shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in [*residuals, dy]]
        free = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in flat]
        two = split.split(backward, free, shapes, weights=len(flat))
        (dx,), saved = two.inputs(*flat, *residuals, dy)
        got = [*two.weights(*flat, saved), dx]
        want = backward(*flat, *residuals, dy)
        b = _dots(two.inputs, *flat, *residuals, dy)
        w = _dots(lambda *a: two.weights(*a[:-1], a[-1]), *flat, saved)
    for a, e in zip(got, want, strict=True):
        assert np.allclose(a, e, atol=1e-6)
    assert w == 4 * 7  # a weight's gradient per matrix: q, k, v, o, gate, up, down
    assert b + w == _dots(backward, *flat, *residuals, dy)  # none of them twice


def _use(mesh: str, methods: str):
    t = _run(LLAMA, mesh, methods).train
    return parallel.use(t.mesh, t.layout, t.methods)


# --- Against one device ---------------------------------------------------------------


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize(("stages", "microbatches"), [(2, 4), (4, 8)])
def test_a_scheduled_pipeline_takes_the_gradients_one_device_does(
    name, stages, microbatches
):
    one = _step_grads(ONE, SCAN)
    mesh = f"mesh(stage={stages}, data=1)"
    piped = _step_grads(mesh, _how(name, microbatches))
    for a, b in zip(piped, one, strict=True):
        assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize(
    "mesh",
    [
        "mesh(stage=2, fsdp=2, tensor=2)",
        "mesh(stage=2, data=2, fsdp=2)",
        "mesh(stage=4, tensor=2)",
    ],
)
def test_a_scheduled_pipeline_trains_as_one_device(losses, name, mesh):
    one = losses(LLAMA, mesh=ONE, methods=SCAN, batch="16")
    piped = losses(LLAMA, mesh=mesh, methods=_how(name, 4), batch="16")
    assert np.allclose(piped, one, atol=1e-5)


@pytest.mark.parametrize("name", PIPELINES)
def test_a_scheduled_pipeline_reduces_once_where_it_runs_the_rows_apart(losses, name):
    how = _how(name, 4, accumulate="unreduced")
    piped = losses(LLAMA, mesh="mesh(stage=2, fsdp=4)", methods=how, batch="16")
    assert np.allclose(
        piped, losses(LLAMA, mesh=ONE, methods=SCAN, batch="16"), atol=1e-5
    )


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize("remat", ["full", "dots"])
def test_a_rematerialized_scheduled_pipeline_trains_as_one_device(losses, name, remat):
    how = _how(name, 4, remat=remat)
    piped = losses(LLAMA, mesh="mesh(stage=2, data=1)", methods=how, batch="16")
    one = losses(LLAMA, mesh=ONE, methods=SCAN, batch="16")
    assert np.allclose(piped, one, atol=1e-5)


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize("stages", ["", "[3, 5]"])
def test_remat_of_the_first_passes_counts_them_from_each_stage_s_first(
    losses, name, stages
):
    remat = "remat(save=[attention], first=1)"
    how = _how(name, 4, remat=remat, **({"stages": stages} if stages else {}))
    piped = losses(LLAMA, mesh="mesh(stage=2, data=1)", methods=how, batch="16")
    one = losses(LLAMA, mesh=ONE, methods=SCAN, batch="16")
    assert np.allclose(piped, one, atol=1e-5)


# The first block outside the pipelined repeat, its 2 branches mxfp8's first.
OUTSIDE = PRELUDE + "model = embed block block^8 norm head"
FIRST = "precision(compute=float32, matmuls=mxfp8, first=2)"


def _formats(name: str, monkeypatch) -> dict[int | None, set[str]]:
    """The formats a training step of OUTSIDE multiplies its projections in, by the
    branch of the model they are in, as it traces: None for the uncounted."""
    train = TRAIN.replace("precision(compute=float32)", FIRST)
    mesh, how = "mesh(stage=2, data=1)", _how(name, 4)
    [run] = compile_runs(f"{OUTSIDE}\n{train}\nmesh = {mesh}\nmethods = {how}")
    t = run.train
    seen: dict[int | None, set[str]] = {}
    of = matmuls.format_of

    def recorded(role: str | None) -> str:
        fmt = of(role)
        seen.setdefault(matmuls._BRANCH.get(), set()).add(fmt)
        return fmt

    monkeypatch.setattr(matmuls, "format_of", recorded)
    with parallel.use(t.mesh, t.layout, t.methods):
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
        tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
        trainer = Trainer(
            model, tx, objective(t.loss, run.arch), matmuls=t.precision, microbatches=1
        )
        batch = parallel.shard_batch(plain(TOKENS))
        trainer._train.lower(trainer.model, trainer.optimizer, batch)
    monkeypatch.setattr(matmuls, "format_of", of)
    return seen


@pytest.mark.parametrize("name", PIPELINES)
def test_the_first_branches_before_a_scheduled_pipeline_multiply_as_under_gpipe(
    name, monkeypatch
):
    """gpipe numbers the stages' branches, a schedule runs them uncounted: in neither
    are they first, so both multiply them in mxfp8, and the first block's in compute."""
    gpipe, scheduled = _formats("gpipe", monkeypatch), _formats(name, monkeypatch)
    assert gpipe[0] == gpipe[1] == scheduled[0] == scheduled[1] == {"compute"}
    inside = set().union(*(f for b, f in gpipe.items() if b is not None and b >= 2))
    assert inside == scheduled[None] == {"mxfp8"}


@pytest.mark.parametrize("name", PIPELINES)
@pytest.mark.parametrize("loss", ["chunks", "logits"])
def test_the_last_stage_takes_the_loss_by_either_method_as_one_device(
    losses, name, loss
):
    """By chunks, whose custom_vjp makes the head's gradient as it scores, or whole."""
    how = _how(name, 4, loss=loss)
    piped = losses(LLAMA, mesh="mesh(stage=2, data=1)", methods=how, batch="16")
    one = losses(
        LLAMA, mesh=ONE, methods=f"methods(repeat=scan, loss={loss})", batch="16"
    )
    assert np.allclose(piped, one, atol=1e-5)


@pytest.mark.parametrize("name", PIPELINES)
def test_first_branches_inside_a_scheduled_pipeline_are_an_error(losses, name):
    given = {"precision": "precision(compute=float32, matmuls=mxfp8, first=1)"}
    with pytest.raises(ValueError, match="fall inside a pipelined repeat"):
        losses(LLAMA, mesh="mesh(stage=2, data=1)", methods=_how(name, 2), **given)


@pytest.mark.parametrize("name", PIPELINES)
def test_a_scheduled_pipeline_counts_its_layers_losses_as_one_device(losses, name):
    piped = losses(MOE, mesh="mesh(stage=2, data=1)", methods=_how(name, 4))
    assert np.allclose(piped, losses(MOE, mesh=ONE, methods=SCAN), atol=1e-5)


def _bias(model: Model) -> Float[np.ndarray, "layers experts"]:
    [moe] = [m for _, m in nnx.iter_graph(model) if isinstance(m, MoE)]
    assert moe.router_bias is not None
    return np.asarray(moe.router_bias[...])


def _stepped_bias(mesh: str, methods: str) -> Float[np.ndarray, "layers experts"]:
    """The router bias of BIASED's model after a training step on TOKENS."""
    trainer = _trainer(BIASED, mesh, methods)
    t = _run(BIASED, mesh, methods).train
    with parallel.use(t.mesh, t.layout, t.methods):
        trainer.train_on_batch(plain(TOKENS))
    return _bias(trainer.model)


@pytest.mark.parametrize("name", PIPELINES)
def test_a_scheduled_step_moves_the_router_bias_once_from_the_whole_batch(name):
    whole = _stepped_bias(ONE, SCAN)
    piped = _stepped_bias("mesh(stage=2, data=1)", _how(name, 4))
    assert np.abs(whole).max() > 0 and np.allclose(piped, whole, atol=1e-6)


# --- Errors and the plan --------------------------------------------------------------


def test_a_layer_outside_the_repeat_that_learns_from_the_batch_is_an_error():
    source = PRELUDE + EXPERTS + "model = embed [norm experts] block^4 norm head"
    trainer = _trainer(source, "mesh(stage=2, data=1)", _how("1f1b", 2))
    t = _run(source, "mesh(stage=2, data=1)", _how("1f1b", 2)).train
    error = pytest.raises(ValueError, match="learns from the whole batch")
    with parallel.use(t.mesh, t.layout, t.methods), error:
        trainer.train_on_batch(plain(TOKENS))


@pytest.mark.parametrize("name", ["gpipe", *PIPELINES])
def test_only_a_scheduled_pipeline_reduces_its_gradients_in_float32(name):
    run = _run(LLAMA, "mesh(stage=2, fsdp=4)", _how(name, 4))
    t = run.train
    with parallel.use(t.mesh, t.layout, t.methods):
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
        tx = optimizer(t.optimizer, lr_multiplier(t.schedule, t.steps), model, t.clip)
        loss = objective(t.loss, run.arch)

        def make() -> Trainer:
            return Trainer(model, tx, loss, compute="bfloat16", reduce="float32")

        if name == "gpipe":
            with pytest.raises(ValueError, match="embedding and head do not"):
                make()
        else:
            assert make().reduces_once


def test_a_scheduled_pipeline_goes_round_the_stages_once():
    error = pytest.raises(ValueError, match="goes round the stages once")
    with error, _use("mesh(stage=2)", _how("1f1b", 4, rounds="2")):
        Model(compile(LLAMA), rngs=nnx.Rngs(0))


def test_the_plan_shows_each_schedules_bubble(tmp_path, capsys):
    source = tmp_path / "piped.bux"
    source.write_text(f"{LLAMA}\n{TRAIN}\nmesh = mesh(stage=4, data=1)")
    sweep = (
        "methods = gpipe: methods(repeat=scan, microbatches=8) | onef: "
        f"{_how('1f1b', 8)} | zb: {_how('zbh1', 8)}"
    )
    assert main(["plan", str(source), sweep]) == 0
    head, *rows = capsys.readouterr().out.splitlines()[1:]
    column = head.split().index("bubble")
    bubbles = Counter(row.split()[column] for row in rows)
    assert bubbles == Counter({"27%": 2, "11%": 1})
