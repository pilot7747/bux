"""Curricula and batch ramps: the stretches of a run's steps, the checks on them, the
batches each stretch reads, and runs that train and resume through them."""

import itertools
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import optax
import pytest
from flax import nnx

from bux import config as cfg
from bux.__main__ import main
from bux.architecture import Model
from bux.architecture.attention import rope
from bux.data.pipeline import jobs
from bux.lang import BuxError, compile, compile_runs
from bux.lang.printer import run_to_source
from bux.train import config as tcfg
from bux.train import curriculum, flops
from bux.train.curriculum import ramped, stretches
from bux.train.data import Synthetic, batches, plain, training_batches
from bux.train.objective import objective
from bux.train.trainer import Trainer, directory, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"


def smoke(*overrides: str):
    """The smoke run, 50 steps of 8 rows of 64 tokens, with `overrides`."""
    settings = ("schedule.warmup = 1", *overrides)
    [run] = compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=settings)
    return run


def shapes(t: tcfg.TrainConfig) -> list[tuple[int, int, int, int]]:
    return [(s.start, s.stop, s.config.batch, s.config.seq_len) for s in stretches(t)]


def test_each_phase_changes_what_it_sets_from_its_step_on():
    run = smoke(
        'curriculum = [phase(at=0.6, seq_len=128), phase(at=40, batch=4, data=tokens(path="x"))]'
    )
    assert shapes(run.train) == [(0, 30, 8, 64), (30, 40, 8, 128), (40, 50, 4, 128)]
    first, second, last = stretches(run.train)
    assert first.config.data == second.config.data == tcfg.SyntheticConfig()
    assert last.config.data == tcfg.TokensConfig(path="x")
    expected = 30 * 8 * 64 + 10 * 8 * 128 + 10 * 4 * 128
    assert curriculum.tokens(stretches(run.train), 50) == expected
    alone = smoke("curriculum = phase(at=0.5, seq_len=128)")  # a phase, not a list
    assert shapes(alone.train) == [(0, 25, 8, 64), (25, 50, 8, 128)]


def test_a_batch_ramp_grows_linearly_in_multiples_of_its_start():
    ramp = tcfg.RampConfig(start=64, steps=100)
    sizes = [ramped(ramp, 256, step) for step in range(101)]
    assert sorted(sizes) == sizes and set(sizes) == {64, 128, 192, 256}
    assert sizes.index(128) == 34 and sizes.index(192) == 67  # 64 + 192 s / 100
    coarse = tcfg.RampConfig(start=48, steps=100, multiple=100)
    assert {ramped(coarse, 256, step) for step in range(100)} == {48, 100, 200}
    run = smoke("batch_ramp = ramp(start=2, steps=10)")
    assert shapes(run.train) == [
        (0, 4, 2, 64),
        (4, 7, 4, 64),
        (7, 10, 6, 64),
        (10, 50, 8, 64),
    ]
    for s in stretches(run.train):
        assert {ramped(run.train.batch_ramp, 8, i) for i in range(s.start, s.stop)} == {
            s.config.batch
        }


def test_a_run_without_them_keeps_its_id_and_one_with_them_prints_back_to_its_own():
    # As every weight started before the default init was a normal.
    before = "init = fan_in(embed=fan_in(distribution=gaussian))"
    [plain_smoke] = compile_runs(
        SMOKE.read_text(), filename=str(SMOKE), overrides=(before,)
    )
    assert plain_smoke.id == "f114ac17ae"  # as before either setting was
    run = smoke(
        "batch_ramp = ramp(start=2, steps=10, multiple=4)",
        "curriculum = [phase(at=0.6, seq_len=128), phase(at=45, data=0.5 * synthetic + 0.5 * synthetic)]",
    )
    [again] = compile_runs(run_to_source(run))
    assert again.train == run.train and again.id == run.id != smoke().id


@pytest.mark.parametrize(
    ("settings", "error", "at"),
    [
        (
            ["curriculum = [phase(at=30), phase(at=20)]"],
            "phase 2 starts at step 20, not after the phase before it (step 30)",
            "[phase(at=30)",
        ),
        (
            ["curriculum = phase(at=50)"],
            "phase 1 starts at step 50, and the run has 50",
            "phase(at=50)",
        ),
        (
            ["batch_ramp = ramp(start=8, steps=5)"],
            "the batch ramp starts at 8 rows, not below the run's batch of 8",
            "ramp(start=8",
        ),
        (
            ["batch_ramp = ramp(start=2, steps=60)"],
            "the batch ramp takes 60 steps, and the run has 50",
            "ramp(start=2",
        ),
        (
            [
                "batch_ramp = ramp(start=2, steps=20)",
                "curriculum = phase(at=10, batch=4)",
            ],
            "phase 1 sets the batch at step 10, before the batch ramp is over at step 20",
            "phase(at=10",
        ),
        (
            ['curriculum = phase(at=0.5, data=text(path="x"))'],
            "text data is tokenized, so the run needs a tokenizer",
            "phase(at=0.5",
        ),
        (
            [
                "model = embed(max_positions=64) block norm head",
                "curriculum = phase(at=0.5, seq_len=128)",
            ],
            "sequences of 128 tokens are longer than the 64 positions the embedding learns",
            "phase(at=0.5",
        ),
        (
            ["curriculum = [phase(at=0.5)]"],
            "phase(...) is a training setting, not a layer",
            "phase(at=0.5)",
        ),
    ],
)
def test_phases_and_ramps_are_checked_where_they_are_written(settings, error, at):
    with pytest.raises(BuxError, match=re.escape(error)) as caught:
        smoke(*settings)
    err = caught.value
    assert err.source is not None and err.span is not None
    assert err.source[err.span.start :].startswith(at)


def test_a_source_goes_on_from_stretch_to_stretch_where_its_rows_are_as_long():
    """A ramp's batches read the rows that batches of one row would, one after
    another. A phase at another seq_len reads a stream of its own, from its first row,
    and one back at the run's goes on where the run's stopped. A run resumes at any
    step on the batches it would have read."""
    run = smoke(
        "batch_ramp = ramp(start=2, steps=10)",
        "curriculum = [phase(at=20, seq_len=32), phase(at=30, seq_len=64)]",
    )
    read = list(itertools.islice(training_batches(run.train, 512), 40))
    assert [b.input_ids.shape for b in read[18:22]] == [(8, 64)] * 2 + [(8, 32)] * 2
    rows = 4 * 2 + 3 * 4 + 3 * 6 + 10 * 8 + 10 * 8  # of 64 tokens, before step 40
    long = np.concatenate([b.input_ids for b in read[:20] + read[30:]])
    assert (long == Synthetic(512, 64, (0, 0, 0)).rows(0, rows).input_ids).all()
    short = np.concatenate([b.input_ids for b in read[20:30]])
    assert (short == Synthetic(512, 32, (0, 0, 1)).rows(0, 80).input_ids).all()
    resumed = next(training_batches(run.train, 512, start=33))
    assert all((a == b).all() for a, b in zip(resumed, read[33], strict=True))
    plain_run = smoke()
    alike = zip(
        training_batches(plain_run.train, 512),
        batches(synthetic(), plain_run.train, 512),
    )
    for a, b in itertools.islice(alike, 3):
        assert all((x == y).all() for x, y in zip(a, b, strict=True))


def synthetic() -> tcfg.SyntheticConfig:
    return tcfg.SyntheticConfig()


def test_a_phase_that_mixes_in_a_source_goes_on_with_the_one_before():
    run = smoke("curriculum = phase(at=10, data=synthetic + synthetic)")
    read = list(itertools.islice(training_batches(run.train, 512), 11))
    before = Synthetic(512, 64, (0, 0, 0)).rows(80, 4).input_ids  # after 10 × 8 rows
    added = Synthetic(512, 64, (0, 0, 1)).rows(0, 4).input_ids
    assert (read[10].input_ids == np.concatenate([before, added])).all()
    mixed = smoke(
        "data = 0.9 * synthetic + 0.1 * synthetic", "curriculum = phase(at=10, batch=2)"
    )
    with pytest.raises(ValueError, match=r"2 rows \(from step 10\) has no room"):
        training_batches(mixed.train, 512)


def test_a_phase_at_another_seq_len_prepares_a_dataset_of_its_own():
    def config(*phases: tcfg.PhaseConfig) -> tcfg.TrainConfig:
        return tcfg.TrainConfig(
            data=tcfg.TextConfig(path="docs/*.parquet"),
            tokenizer=tcfg.TokenizerConfig(name="gpt2"),
            optimizer=tcfg.AdamWConfig(lr=1e-3),
            seq_len=64,
            batch=4,
            steps=10,
            curriculum=phases,
        )

    longer = config(tcfg.PhaseConfig(at=5, seq_len=128))
    assert [job.seq_len for job in jobs(longer)] == [64, 128]
    assert [job.seq_len for job in jobs(config(tcfg.PhaseConfig(at=5, batch=8)))] == [
        64
    ]


YARN = cfg.YaRNConfig(factor=4, original_max_len=16)
LAYERS = (
    "d_model = 16\nvocab_size = 32\n"
    "full = gqa(num_heads=2, num_kv_heads=2, rope=rope(fraction=0.5))\n"
    "local = gqa(num_heads=2, num_kv_heads=2, rope=rope(), sliding_window_size=4)\n"
    "nope = gqa(num_heads=2, num_kv_heads=2)\n"
    "latent = mla(num_heads=2, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,"
    " v_head_dim=4)\n"
    "model = embed [rmsnorm full] [rmsnorm local] [rmsnorm nope] [rmsnorm latent]"
    " rmsnorm head\n"
)


def ropes_of(arch) -> list[Any]:
    return [
        layer.config.rope for layer in arch.layers() if hasattr(layer.config, "rope")
    ]


def test_a_phase_sets_the_rope_of_full_attention_from_its_step_on():
    [run] = compile_runs(
        LAYERS + "data = synthetic\noptimizer = adamw(lr=1e-3)\n"
        "seq_len = 16\nbatch = 8\nsteps = 10\n"
        "curriculum = [phase(at=4, seq_len=64, rope_scaling=yarn(factor=4,"
        " original_max_len=16)), phase(at=6, batch=4), phase(at=8, rope_base=1e5)]"
    )
    before = ropes_of(run.arch)
    assert ropes_of(run.arch_at(3)) == before and run.arch_at(0) == run.arch
    full, local, nope, latent = ropes_of(run.arch_at(4))
    assert full == before[0].model_copy(update={"scaling": YARN})  # fraction kept
    assert latent == cfg.RoPEConfig(scaling=YARN)
    assert (local, nope) == (before[1], None)  # a window, and no RoPE
    assert run.arch_at(7) == run.arch_at(4)  # as before, where a phase sets none
    full, local, _, _ = ropes_of(run.arch_at(8))
    assert (full.scaling, full.base, local.base) == (YARN, 1e5, 1e4)
    yarn = "yarn(factor=4.0, original_max_len=16)"
    assert curriculum.lines(run.train)[1:] == [
        f"phase 1, from step 4: 8 × 64 tokens, rope_scaling={yarn}",
        "phase 2, from step 6: 4 × 64 tokens",
        "phase 3, from step 8: 4 × 64 tokens, rope_base=100000.0",
    ]
    [again] = compile_runs(run_to_source(run))
    assert again.train == run.train and again.id == run.id


def test_a_trainer_computes_with_a_phase_s_rope_from_its_step_on(monkeypatch):
    """The steps and the evals from the phase's step on rotate by YaRN in its full
    attention, not in its window; an eval of the same shape compiles again for it."""
    [run] = compile_runs(
        LAYERS + "data = synthetic\noptimizer = adamw(lr=1e-3)\n"
        "seq_len = 8\nbatch = 2\nsteps = 4\n"
        "curriculum = phase(at=2, rope_scaling=yarn(factor=4, original_max_len=16))"
    )
    traced: list[object] = []
    frequencies = rope.frequencies

    def recorded(config: cfg.RoPEConfig, dims: int):
        traced.append(config.scaling)
        return frequencies(config, dims)

    monkeypatch.setattr(rope, "frequencies", recorded)
    loss = objective(tcfg.LossConfig(), run.arch)
    long = run.arch_at(2)
    model = Model(run.arch, rngs=nnx.Rngs(0))
    trainer = Trainer(
        model,
        optax.sgd(0.0),
        loss,
        remodel=[(2, lambda: Model(long, rngs=nnx.Rngs(0)))],
    )
    batch = plain(np.arange(16, dtype=np.int32).reshape(2, 8) % 32)
    each: list[set] = []
    evals: list[float] = []
    for _ in range(4):
        traced.clear()
        evals.append(trainer.evaluate([batch])["loss"])
        trainer.train_on_batch(batch)
        each.append(set(map(repr, traced)))
    yarn, plain_rope = repr(YARN), repr(None)
    # Step 0 traces the eval and the step, without YaRN; step 2 both again, with it
    # in the full attention and the latent one, not in the window.
    assert each == [{plain_rope}, set(), {plain_rope, yarn}, set()]
    assert evals[0] == evals[1] != evals[2] == evals[3]  # the step's lr is 0
    as_built = Trainer(Model(long, rngs=nnx.Rngs(0)), optax.sgd(0.0), loss)
    assert as_built.evaluate([batch])["loss"] == evals[2]


def test_a_trainer_compiles_again_for_batches_of_a_new_shape_and_counts_their_flops():
    arch = compile(
        "d_model = 16\nvocab_size = 32\n"
        "model = embed [rmsnorm gqa(num_heads=2, num_kv_heads=2)] rmsnorm head\n"
    )
    asked: list[tuple[int, int]] = []

    def work(rows: int, seq_len: int) -> flops.Work:
        asked.append((rows, seq_len))
        return flops.Work(float(rows * seq_len), {})

    loss = objective(tcfg.LossConfig(), arch)
    model = Model(arch, rngs=nnx.Rngs(0))
    trainer = Trainer(model, optax.adamw(1e-3), loss, work=work)
    compiled = []
    for shape in [(2, 8), (2, 8), (4, 16), (2, 8)]:
        trainer.train_on_batch(plain(np.ones(shape, np.int32)))
        compiled.append("compile" in trainer.timers.read())
    assert compiled == [True, False, True, True]
    assert asked == [(2, 8), (2, 8), (4, 16), (2, 8)]
    assert trainer.flops == 2 * 16 + 64 + 16


def test_a_run_trains_through_its_phases_and_resumes_where_it_stopped(tmp_path):
    run = smoke(
        "steps = 8",
        "log = log(every=1)",
        "checkpoint = checkpoint(every=3, keep=3)",
        "batch_ramp = ramp(start=2, steps=3)",
        "curriculum = phase(at=5, seq_len=32, batch=4,"
        " rope_scaling=yarn(factor=4, original_max_len=16))",
        "mesh = mesh(data=1)",
        'eval = eval(name="eval", data=synthetic, every=4, batches=1)',
    )
    said: list[str] = []
    whole = train(run, tmp_path, echo=said.append)
    assert any(
        "from step 0: 8 × 64 tokens, ramped from 2 rows over the first 3 steps in 3 sizes"
        in line
        for line in said
    )
    assert any(
        "phase 1, from step 5: 4 × 32 tokens, "
        "rope_scaling=yarn(factor=4.0, original_max_len=16)" in line
        for line in said
    )
    folder = directory(run, tmp_path)
    lines = [json.loads(line) for line in (folder / "metrics.jsonl").open()]
    logged = [line for line in lines if "loss" in line]
    each = [2 * 64, 4 * 64, 6 * 64, 8 * 64, 8 * 64, 4 * 32, 4 * 32, 4 * 32]
    assert [line["tokens"] for line in logged] == list(itertools.accumulate(each))
    assert [line["step"] for line in logged if "time/compile_s" in line] == [
        1,
        2,
        3,
        4,
        6,
    ]
    assert [line["step"] for line in lines if "eval_loss" in line] == [4, 8]

    # Lose the checkpoints after step 3: the run resumes there, before its phase, and
    # ends where it did.
    shutil.rmtree(folder / "checkpoints" / "8")
    shutil.rmtree(folder / "checkpoints" / "6")
    said.clear()
    resumed = train(run, tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 3")
    assert resumed["loss"] == whole["loss"]
    assert resumed["eval_loss"] == whole["eval_loss"]
    assert resumed["tokens"] == whole["tokens"] == sum(each)

    # And from inside it, with its RoPE.
    shutil.rmtree(folder / "checkpoints" / "8")
    said.clear()
    inside = train(run, tmp_path, echo=said.append)
    assert said[0].endswith("resuming at step 6")
    assert inside["loss"] == whole["loss"]
    assert inside["eval_loss"] == whole["eval_loss"]


def test_bux_check_says_how_many_tokens_each_phase_takes(capsys):
    phase = "curriculum = phase(at=0.5, seq_len=128)"
    assert (
        main(["check", str(SMOKE), phase, "batch_ramp = ramp(start=2, steps=10)"]) == 0
    )
    out = capsys.readouterr().out
    tokens = (4 * 2 + 3 * 4 + 3 * 6 + 15 * 8) * 64 + 25 * 8 * 128
    assert f"50 steps = {tokens / 1e3:.3g}k tokens" in out
    assert "phase 1, from step 25: 8 × 128 tokens" in out


def test_every_shape_a_run_takes_must_split_over_its_mesh(tmp_path):
    """On the tests' 8 devices, a phase's 4 rows do not split: before anything runs."""
    run = smoke("curriculum = phase(at=5, batch=4)")
    with pytest.raises(ValueError, match="from step 5, batch = 4 does not split"):
        train(run, tmp_path, echo=lambda _: None)


def test_an_eval_of_its_own_shape_prepares_its_rows_and_reads_them(tmp_path):
    held_out = tcfg.TextConfig(path="docs/*.parquet")
    t = tcfg.TrainConfig(
        data=held_out,
        tokenizer=tcfg.TokenizerConfig(name="gpt2"),
        optimizer=tcfg.AdamWConfig(lr=1e-3),
        seq_len=64,
        batch=4,
        steps=10,
        eval=(
            tcfg.EvalConfig(name="short", data=held_out),
            tcfg.EvalConfig(name="long", data=held_out, seq_len=256, batch=2),
        ),
    )
    assert [job.seq_len for job in jobs(t)] == [64, 256]
    short, long = (tcfg.evaluated(t, e) for e in tcfg.evals(t))
    assert (long.batch, long.seq_len) == (2, 256)
    assert short is t

    run = smoke(
        "steps = 4",
        'eval = [eval(name="short", data=synthetic, every=2),'
        ' eval(name="long", data=synthetic, every=2, seq_len=128, batch=8,'
        " positions=[64, 1024])]",
    )
    said: list[str] = []
    train(run, tmp_path, echo=said.append)
    assert any("short every 2, long every 2 of 8 × 128 tokens" in line for line in said)
    lines = [
        json.loads(line) for line in (directory(run, tmp_path) / "metrics.jsonl").open()
    ]
    assert [line["step"] for line in lines if "long_loss" in line] == [2, 4]
    [last] = [line for line in lines if line["step"] == 4 and "long_loss" in line]
    short, long = last["long_loss_0"], last["long_loss_64"]
    assert min(short, long) <= last["long_loss"] <= max(short, long)
    assert np.isnan(last["long_loss_1k"]) and "short_loss_0" not in last


def test_an_eval_runs_its_batch_whole_but_in_a_pipeline_s_microbatches(tmp_path):
    """On the tests' 8 devices, an eval's 8 rows split over the mesh, which a step's 2
    microbatches of 4 would not; under a pipeline, its 4 microbatches of 2 do not."""
    long = 'eval = eval(name="long", data=synthetic, every=2, seq_len=128, batch=8)'
    run = smoke("steps = 2", "batch = 16", "methods = methods(microbatches=2)", long)
    assert "long_loss" in train(run, tmp_path, echo=lambda _: None)
    piped = smoke(
        long,
        "batch = 16",
        "mesh = mesh(stage=2)",
        "methods = methods(repeat=scan, pipeline=gpipe, microbatches=4)",
        "precision = precision(compute=float32)",
    )
    with pytest.raises(ValueError, match="eval long: batch = 8 does not split over"):
        train(piped, tmp_path, echo=lambda _: None)


def test_an_eval_of_its_own_shape_must_split_over_the_mesh(tmp_path):
    """On the tests' 8 devices, an eval's 4 rows do not split: before anything runs."""
    run = smoke('eval = eval(name="long", data=synthetic, seq_len=128, batch=4)')
    with pytest.raises(ValueError, match="eval long: batch = 4 does not split"):
        train(run, tmp_path, echo=lambda _: None)
