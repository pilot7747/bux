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
from jax.sharding import PartitionSpec as P
from optax.contrib import MuonDimensionNumbers, scale_by_muon

from bux import parallel
from bux.__main__ import main
from bux.architecture import Model
from bux.data.pipeline import jobs
from bux.lang import compile, compile_runs
from bux.train import config as tcfg
from bux.train import flops
from bux.train.data import batches, plain, prefetch
from bux.train.hooks import Hook, Log, Profile
from bux.train.objective import (
    Term,
    cross_entropy,
    objective,
    prediction_offsets,
    target_mask,
    z_loss,
)
from bux.train.optim import (
    lr_multiplier,
    momentum,
    muon_matrices,
    optimizer,
    optimizers,
    orthogonalized,
    schedules,
    weight_decay_mask,
    whole_matrix_spec,
)
from bux.train.sinks import LatestMetrics, Sink
from bux.train.trainer import Trainer, directory, step_flops, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"

ARCH = """
d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
block = [norm attn] [norm ffn]
"""


def smoke(*overrides: str):
    """The smoke run, shortened: its warmup would not fit in a few steps."""
    settings = ("schedule.warmup = 1", *overrides)
    [run] = compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=settings)
    return run


def values(c: tcfg.ScheduleConfig, steps: int) -> list[float]:
    f = lr_multiplier(c, steps)
    return [round(float(f(i)), 4) for i in range(steps)]


def test_schedules_warm_up_and_reach_final_at_the_last_step():
    assert values(tcfg.ConstantSchedule(warmup=4), 6) == [0.25, 0.5, 0.75, 1, 1, 1]
    cosine = values(tcfg.CosineSchedule(warmup=2, final=0.1), 6)
    assert cosine[:2] == [0.5, 1.0] and cosine[-1] == 0.1
    assert all(a >= b for a, b in itertools.pairwise(cosine[1:]))  # after warmup
    wsd = values(tcfg.WSDSchedule(decay=0.4, shape="linear"), 10)
    assert wsd == [1, 1, 1, 1, 1, 1, 0.75, 0.5, 0.25, 0]
    sqrt = values(tcfg.WSDSchedule(decay=0.4, shape="sqrt", final=0.1), 10)
    assert sqrt[5] == 1 and sqrt[6] == round(0.1 + 0.9 * (1 - 0.25**0.5), 4)


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
        ("up",),
        ("gate",),
        ("down",),
    }
    stacked = [
        d for p, d in muon_matrices(m).items() if p[-1] in ("up", "gate", "down")
    ]
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
    for method in ("gspmd", "gather"):
        methods = tcfg.MethodsConfig(orthogonalize=method)
        one = parallel.use(tcfg.MeshConfig(data=1), methods=methods)
        for where in (contextlib.nullcontext(), one):
            with where:
                ours = optax.chain(
                    momentum(0.95, nesterov),
                    orthogonalized(5, {(k,): d for k, d in dims.items()}, stack),
                )
                got = _muon_updates(ours, params)
            assert jax.tree.all(jax.tree.map(jnp.array_equal, got, expected))


@functools.cache
def _muon_step(orthogonalize: str = "gspmd", **mesh: int) -> tuple[list, str]:
    """The MoE's parameters after a Muon step from fixed gradients, split over `mesh`
    (outside parallel.use with none), and the compiled step's HLO."""
    methods = tcfg.MethodsConfig(orthogonalize=orthogonalize)
    split = parallel.use(tcfg.MeshConfig.model_validate(mesh), methods=methods)
    with split if mesh else contextlib.nullcontext():
        model = parallel.build(lambda: Model(compile(MOE), rngs=nnx.Rngs(0)))
        c, lr = tcfg.MuonConfig(lr=0.02), lr_multiplier(tcfg.ConstantSchedule(), 1)
        trained = nnx.Optimizer(model, optimizer(c, lr, model), wrt=nnx.Param)
        parallel.shard_state(model, trained)
        grads = jax.tree.map(jnp.sin, nnx.state(model, nnx.Param))
        step = nnx.jit(lambda m, o, g: o.update(m, g))
        hlo = step.lower(model, trained, grads).compile().as_text() or ""
        step(model, trained, grads)
    return jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(model, nnx.Param))), hlo


@pytest.mark.parametrize("orthogonalize", ["gspmd", "gather"])
@pytest.mark.parametrize("mesh", SPLITS.values(), ids=list(SPLITS))
def test_muon_updates_split_over_devices_as_on_one(orthogonalize, mesh):
    one, _ = _muon_step()
    split, _ = _muon_step(orthogonalize, **mesh)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(split, one, strict=True))


def _all_reduces(hlo: str) -> int:
    return len(re.findall(r"\ball-reduce(-start)?\(", hlo))


@pytest.mark.parametrize("mesh", SPLITS.values(), ids=list(SPLITS))
def test_muon_gathering_whole_matrices_needs_no_all_reduce(mesh):
    """gspmd runs Newton-Schulz on shards, all-reducing products over split
    dimensions; gather moves each matrix whole once and runs it there."""
    _, gspmd = _muon_step("gspmd", **mesh)
    _, gather = _muon_step("gather", **mesh)
    assert _all_reduces(gather) == 0 < _all_reduces(gspmd)


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


def test_a_run_that_gathers_matrices_for_muon_trains_as_on_one_device(losses):
    muon = {"optimizer": "muon(lr=0.02)", "methods": "methods(orthogonalize=gather)"}
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
    assert all((a == b).all() for a, b in zip(first, again))
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

    early = prefetch(shards(), lambda x: 10 * x)
    assert [next(early) for _ in range(3)] == [0, 10, 20]
    with pytest.raises(FileNotFoundError, match="no shards match"):
        next(early)
    assert list(prefetch(iter("ab"), str.upper)) == ["A", "B"]
    put: list[int] = []
    threads = threading.active_count()
    endless = prefetch(itertools.count(), lambda x: put.append(x) or x, ahead=2)
    assert next(endless) == 0
    endless.close()
    assert len(put) <= 4  # the one taken, the two ahead, and one on its way
    assert threading.active_count() == threads


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
    assert jnp.allclose(total, expected) and set(metrics) == {"loss", "z_loss", "mtp"}


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

    # Lose the last checkpoint: the run resumes from step 3 and ends where it did.
    shutil.rmtree(folder / "checkpoints" / "6")
    said: list[str] = []
    resumed = train(run, tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 3")
    assert (
        resumed["loss"] == whole["loss"] and resumed["grad_norm"] == whole["grad_norm"]
    )


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


def test_a_run_trains_on_text_that_it_prepares_first(tmp_path, byte_tokenizer):
    text = corpus(tmp_path / "corpus")
    run = smoke(
        f'tokenizer = tokenizer(name="{byte_tokenizer.name}")',
        f'data = text(path="{text}", split=[0, 0.9])',
        f'eval = eval(data=text(path="{text}", split=[0.9, 1]), every=2, batches=1)',
        "steps = 2",
    )
    said: list[str] = []
    last = train(run, tmp_path / "runs", cache=tmp_path / "packed", echo=said.append)
    assert np.isfinite(last["loss"]) and np.isfinite(last["eval_loss"])
    assert [s.split(":")[0] for s in said[:2]] == [j.name for j in jobs(run.train)]
    assert len(list((tmp_path / "packed").iterdir())) == 2


def test_muon_runs_end_to_end(tmp_path):
    run = smoke("optimizer = muon(lr=0.02)", "steps = 3")
    last = train(run, tmp_path, echo=lambda _: None)
    assert np.isfinite(last["loss"])


def test_a_builder_registered_for_a_type_replaces_its_own(monkeypatch):
    def half(c, steps):
        return lambda step: jnp.asarray(0.5)

    monkeypatch.setitem(schedules.builders, "constant", half)
    assert values(tcfg.ConstantSchedule(warmup=2), 4) == [0.25, 0.5, 0.5, 0.5]
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


def test_a_trainer_takes_terms_hooks_and_sinks_of_ones_own():
    arch = compile(ARCH + "model = embed block norm head\n")
    loss = objective(tcfg.LossConfig(), arch)
    loss.terms["logit_l2"] = Term(1e-4, lambda logits, _: jnp.mean(logits[0] ** 2))
    trainer = Trainer(Model(arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
    lr = lr_multiplier(tcfg.ConstantSchedule(), 3)
    latest, steps = LatestMetrics(), Steps()
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    hooks = [Log(every=1, lr=lr, tokens=16), WeightNorm()]
    trainer.fit(data, 3, hooks=hooks, sinks=[latest, steps])
    assert trainer.step == 3
    assert [s for s, _ in steps.reports] == [1, 2, 2, 3, 3]  # log, then weight norm
    assert {"loss", "logit_l2", "grad_norm", "lr", "weight_norm"} <= set(latest.metrics)


def test_a_step_s_flops_are_counted_from_its_program():
    """The smoke run's: 6 N of its matrices, and 12 L d t of attention (PaLM's)."""
    matrices = 2 * (2 * 64 * 64 + 2 * 64 * 32 + 3 * 64 * 256) + 64 * 512
    attention = 12 * 2 * 64 * 64  # layers, heads × head_dim, positions
    assert step_flops(smoke()) == (6 * matrices + attention) * 8 * 64
    x, w, sizes = jnp.ones((6, 8)), jnp.ones((3, 8, 5)), jnp.array([1, 2, 3])
    grad = jax.grad(lambda x, w: jax.lax.ragged_dot(x, w, sizes).sum(), (0, 1))
    assert flops.count(grad, x, w) == 3 * 2 * 6 * 8 * 5

    def scan(c):
        return jax.lax.scan(lambda c, _: (c @ x.T @ x, None), c, length=5)[0]

    assert flops.count(scan, jnp.ones((4, 8))) == 5 * 2 * (2 * 4 * 6 * 8)
    depthwise = functools.partial(
        jax.lax.conv_general_dilated,
        window_strides=(1,),
        padding="VALID",
        feature_group_count=6,
    )
    ones = jnp.ones((2, 6, 10)), jnp.ones((6, 1, 3))
    assert flops.count(depthwise, *ones) == 2 * (2 * 6 * 8) * 3


def test_the_log_reports_mfu_where_the_devices_peak_is_known():
    assert flops.peak(jax.devices()[0].device_kind) is None  # a CPU's
    assert flops.peak("NVIDIA B300 SXM6 AC") == 2.25e15
    arch = compile(ARCH + "model = embed block norm head\n")
    loss = objective(tcfg.LossConfig(), arch)
    trainer = Trainer(Model(arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
    lr = lr_multiplier(tcfg.ConstantSchedule(), 2)
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    latest = LatestMetrics()
    log = Log(every=1, lr=lr, tokens=16, flops=4e6, peak=1e9)
    trainer.fit(data, 2, hooks=[log], sinks=[latest])
    steps_per_s = latest.metrics["tokens_per_s"] / 16
    assert latest.metrics["mfu"] == pytest.approx(steps_per_s * 4e6 / 1e9)


def test_a_step_takes_the_memory_of_the_state_it_replaces():
    arch = compile(ARCH + "model = embed block norm head\n")
    trainer = Trainer(
        Model(arch, rngs=nnx.Rngs(0)),
        optax.adamw(1e-3),
        objective(tcfg.LossConfig(), arch),
    )
    data = batches(tcfg.SyntheticConfig(), rows(tcfg.SyntheticConfig(), 2, 8), 32)
    batch = jax.device_put(next(data))
    step = trainer._train.lower(trainer.model, trainer.optimizer, batch).compile()
    state = sum(x.nbytes for x in jax.tree.leaves(trainer.state()))
    memory = step.memory_analysis()
    assert memory is not None and memory.alias_size_in_bytes == state


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
