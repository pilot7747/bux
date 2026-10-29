"""Prefix-token prediction (modded-nanogpt's record 88): each target's prefix token by
the tokenizer's bytes, its cross-entropy beside the loss, its weight's schedule over
the run's steps, and runs without it as before."""

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import byte_spec

from bux.data.tokenizer import Tokenizer
from bux.lang import compile, compile_runs
from bux.train import config as tcfg
from bux.train import trainer as trainer_module
from bux.train.data import plain
from bux.train.objective import Heads, objective, prefix_weight
from bux.train.sinks import Scalars, Sink
from bux.train.trainer import train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"
# smoke.bux, short, with an eval: as main computed it (CPU, float32).
SHORT = [
    "schedule.warmup = 1",
    "steps = 4",
    "log = log(every=1)",
    "eval = eval(data=synthetic, every=2, batch=4, batches=2)",
    "mesh = mesh(data=1)",
]


def test_a_token_s_prefix_token_is_its_longest_proper_byte_prefix_in_the_vocabulary(
    tmp_path,
):
    """As record 88's table: "abcd" to "abc", "abc" to "ab", "ab" to "a"; a byte, a
    token whose prefixes are no tokens, and a special token, to none."""
    spec = byte_spec()
    vocab = spec["model"]["vocab"]
    vocab |= {"ab": 259, "abc": 260, "abcd": 261, "xyz": 262}
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    tok = Tokenizer(tcfg.TokenizerConfig(name=str(tmp_path / "tokenizer.json")))
    table = tok.prefix_tokens()
    assert table[[259, 260, 261]].tolist() == [ord("a"), 259, 260]
    assert table[262] == ord("x")  # "xy" is no token, "x" is
    assert (table[:256] == -1).all() and (table[256:259] == -1).all()


def logits_and_batch():
    logits = jax.random.normal(jax.random.key(0), (2, 6, 8)) * 2
    ids = np.array([[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]])
    return logits, plain(ids)


def test_the_prefix_loss_is_the_cross_entropy_of_the_targets_prefix_tokens():
    """Over every target, those without a prefix token adding nothing, by the same
    softmax as the next token's."""
    logits, batch = logits_and_batch()
    table = np.array([-1, 0, -1, 2, 1, -1, 3, 5])
    heads = Heads([logits], batch)
    got = heads.prefix_sum(jnp.asarray(table), 1)
    log_p = jax.nn.log_softmax(np.asarray(logits, np.float64), -1)
    targets = batch.input_ids[:, 1:]
    nll = [
        -log_p[b, t, table[targets[b, t]]] if table[targets[b, t]] >= 0 else 0.0
        for b in range(2)
        for t in range(5)
    ]
    assert float(got) == pytest.approx(sum(float(n) for n in nll) / 10, rel=1e-5)


MTP = """
d_model = 32
vocab_size = 8
norm = rmsnorm
ffn = gated_mlp(intermediate_size=64, activation=swish)
block = [norm gqa(num_heads=2, num_kv_heads=2)] [norm ffn]
shared out = head
model = embed block norm @h out $h nextn(norm=norm) block norm out
"""


def test_it_takes_the_first_head_alone_and_each_document_s_own_targets():
    """Beside an MTP module, of the first head's logits; in packed rows, no target
    of the next document, as the loss's."""
    logits, batch = logits_and_batch()
    positions = np.array([[0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5]], np.int32)
    packed = batch._replace(position_ids=positions)
    loss = tcfg.LossConfig(prefix=tcfg.PrefixConfig())
    obj = objective(loss, compile(MTP), prefixes=np.arange(1, 9) % 8)
    ahead = jax.random.normal(jax.random.key(5), logits.shape)
    part, _ = obj([logits, ahead], packed)
    alone, _ = obj([logits, ahead * 3], packed)
    assert part.metrics["prefix_loss"] == alone.metrics["prefix_loss"]
    log_p = jax.nn.log_softmax(np.asarray(logits, np.float64), -1)
    seen = [(b, t) for b in range(2) for t in range(5) if (b, t) != (0, 2)]
    nll = [-log_p[b, t, (batch.input_ids[b, t + 1] + 1) % 8] for b, t in seen]
    got = float(part.metrics["prefix_loss"])
    assert got == pytest.approx(sum(float(n) for n in nll) / len(seen), rel=1e-5)


def test_the_weight_holds_then_falls_to_zero_as_the_record_s():
    """0.25 for the first third of the steps, to 0 over the second, then 0."""
    c = tcfg.PrefixConfig()
    shares = jnp.array([0.0, 0.3, 1 / 3, 0.5, 2 / 3, 0.9, 1.0])
    want = [0.25, 0.25, 0.25, 0.125, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(prefix_weight(c, shares), want, atol=1e-6)
    sudden = tcfg.PrefixConfig(weight=1.0, hold=0.5, decay=0.0)
    assert prefix_weight(sudden, jnp.array([0.5, 0.51])).tolist() == [1.0, 0.0]


def test_the_objective_weighs_it_at_the_step_s_share_of_the_run():
    logits, batch = logits_and_batch()
    arch = compile_runs(SMOKE.read_text(), filename=str(SMOKE))[0].arch
    loss = tcfg.LossConfig(prefix=tcfg.PrefixConfig(weight=0.5, hold=0.0, decay=1.0))
    table = np.arange(1, 9) % 8
    obj = objective(loss, arch, steps=4, prefixes=table)
    for step, weight in [(0, 0.5), (2, 0.25), (4, 0.0)]:
        with obj.at(jnp.uint32(step)):
            part, _ = obj([logits], batch)
        m = part.metrics
        assert float(m["prefix"]) == pytest.approx(weight * float(m["prefix_loss"]))
        assert float(part.loss) == pytest.approx(float(m["loss"] + m["prefix"]))


def short(*sets: str) -> list:
    return compile_runs(
        SMOKE.read_text(), filename=str(SMOKE), overrides=[*SHORT, *sets]
    )


def trained(run) -> list[tuple[int, str, float]]:
    seen: list[tuple[int, str, float]] = []

    class Seen(Sink):
        def write(self, step: int, scalars: Scalars) -> None:
            kept = {k: v for k, v in scalars.items() if "loss" in k or "prefix" in k}
            seen.extend((step, k, float(v)) for k, v in kept.items())

    train(run, tempfile.mkdtemp(), echo=lambda _: None, sinks=[Seen()])
    return seen


def test_a_run_without_it_trains_as_before():
    """smoke.bux's id, losses and eval losses, as main computed them before it
    (04f1159; CPU, float32)."""
    [run] = short()
    assert run.id == "0e4953952b"
    want = [
        (1, "loss", 6.256721496582031),
        (2, "loss", 6.250439643859863),
        (2, "eval_loss", 6.25772942437066),
        (3, "loss", 6.242168426513672),
        (4, "loss", 6.254068851470947),
        (4, "eval_loss", 6.257589431036086),
    ]
    seen = trained(run)
    assert [(s, k) for s, k, _ in seen] == [(s, k) for s, k, _ in want]
    for (_, _, got), (_, _, value) in zip(seen, want, strict=True):
        assert got == pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize(
    "methods", ["", "methods = methods(remat=full, microbatches=2)"], ids=["", "mb2"]
)
def test_a_run_with_it_weighs_it_by_its_schedule_and_evals_the_plain_loss(
    byte_tokenizer, monkeypatch, methods
):
    """Each step's prefix term is its weight at the step's share of the run times its
    cross-entropy, by microbatches too; the eval logs the next token's loss alone."""
    table = (np.arange(512) + 1) % 512  # every id has a prefix token
    monkeypatch.setattr(trainer_module, "_prefix_tokens", lambda _: table)
    tokenizer = f'tokenizer = tokenizer(name="{byte_tokenizer.name}")'
    on = "loss = loss(prefix=prefix(weight=1, hold=0.5, decay=0.25))"
    [run] = short(tokenizer, on, *[m for m in [methods] if m])
    assert run.id != short(tokenizer)[0].id
    seen = trained(run)
    by_step: dict[int, dict[str, float]] = {}
    for step, name, value in seen:
        by_step.setdefault(step, {})[name] = value
    # Steps 1 to 4 run at shares 0, 1/4, 1/2 and 3/4 of the run.
    for step, weight in [(1, 1.0), (2, 1.0), (3, 1.0), (4, 0.0)]:
        got = by_step[step]
        assert got["prefix_loss"] > 0
        assert got["prefix"] == pytest.approx(weight * got["prefix_loss"], rel=1e-5)
    evals = {k for s in by_step.values() for k in s if k.startswith("eval")}
    assert evals == {"eval_loss"}


def test_it_takes_a_tokenizer():
    with pytest.raises(ValueError, match="takes a tokenizer"):
        short("loss = loss(prefix=prefix())")
