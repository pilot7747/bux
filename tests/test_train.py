"""The trainer: schedules, optimizers, data, losses, and runs that resume."""

import itertools
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import corpus
from flax import nnx

from bux.__main__ import main
from bux.architecture import Model
from bux.data.pipeline import jobs
from bux.lang import compile, compile_runs
from bux.train import config as tcfg
from bux.train.data import batches, plain
from bux.train.optim import matrices, schedule
from bux.train.trainer import directory, lookahead, losses, train

SMOKE = Path(__file__).parent.parent / "experiments" / "smoke.bux"

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
    f = schedule(c, steps)
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
    found = {path[-2:] if path[-1] == "kernel" else path[-1:] for path in matrices(m)}
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
    stacked = [d for p, d in matrices(m).items() if p[-1] in ("up", "gate", "down")]
    assert {(d.reduction_axis, d.output_axis) for d in stacked} == {(1, 2)}


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


def test_heads_predict_further_ahead_after_each_nextn():
    arch = compile(
        ARCH + "mtp = $h! nextn [norm ffn] norm @h head\n"
        "model = embed block norm @h head mtp^2\n"
    )
    assert lookahead(arch) == [1, 2, 3]
    tokens = np.asarray(jax.random.randint(jax.random.key(0), (2, 8), 0, 32))
    outputs = Model(arch, rngs=nnx.Rngs(0)).outputs(jnp.asarray(tokens))
    batch = plain(tokens)
    total, metrics = losses(outputs, batch, lookahead(arch), tcfg.LossConfig(z=1e-4))
    expected = metrics["loss"] + 1e-4 * metrics["z_loss"] + 0.3 * metrics["mtp"]
    assert jnp.allclose(total, expected) and set(metrics) == {"loss", "z_loss", "mtp"}


def test_a_head_learns_only_targets_of_its_own_document_in_the_mask():
    logits = jax.random.normal(jax.random.key(0), (1, 8, 32))
    tokens = np.arange(8)[None]
    batch = plain(tokens)._replace(position_ids=np.array([[0, 1, 2, 3, 0, 1, 2, 3]]))
    ce = optax.softmax_cross_entropy_with_integer_labels(logits[:, :5], tokens[:, 3:])[
        0
    ]

    def loss(b):
        return losses([logits], b, [3], tcfg.LossConfig())[1]["loss"]

    assert jnp.allclose(loss(batch), ce[jnp.array([0, 4])].mean())  # 0->3 and 4->7
    mask = batch.loss_mask.copy()
    mask[0, 7] = False
    assert jnp.allclose(loss(batch._replace(loss_mask=mask)), ce[0])


def test_a_run_writes_its_directory_and_resumes_where_it_stopped(tmp_path):
    run = smoke(
        "steps = 6",
        "log = log(every=1)",
        "checkpoint = checkpoint(every=3)",
        "eval = eval(data=synthetic, every=3, batches=2)",
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
