"""Searches: dimensions and objectives in the language, the rungs of halving, a
search's trials kept in its directory, and the launcher training them."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bux.__main__ import main
from bux.lang import Space, check, compile_runs, compile_spaces, ir
from bux.lang.lsp import features
from bux.lang.printer import run_to_source
from bux.launch import Queue, search, status, summary
from bux.search import halving
from bux.search.halving import Job, Trial
from bux.search.methods import History, propose
from bux.search.search import Search, Searches, objective
from bux.search.space import Space as Coordinates
from bux.train import config as tcfg
from bux.train.trainer import directory

MODEL = """
d_model = 16
vocab_size = 32
norm = rmsnorm
attn = gqa(num_heads=2, num_kv_heads=2)
ffn = gated_mlp(intermediate_size=32, activation=swish)
block = [norm attn] [norm ffn]
model = embed block norm head(tied=true)
"""

TRAIN = """
data = synthetic
optimizer = adamw(lr=log_uniform(1e-4, 1e-2))
schedule = wsd(warmup=1)
seq_len = 8
batch = 8
steps = 9
log = log(every=9)
eval = eval(data=synthetic, every=100, batches=1)
"""

SOURCE = MODEL + TRAIN


def spaces(source: str = SOURCE, **kw) -> list[Space]:
    return compile_spaces(source, filename="tune.bux", **kw)


def error(source: str) -> str:
    report = check(source, filename="tune.bux")
    assert report.errors, "no error"
    return str(report.errors[0])


# --- Dimensions and objectives ------------------------------------------------------


def test_a_dimension_takes_its_middle_outside_a_search():
    [run] = compile_runs(SOURCE, filename="tune.bux")
    assert run.train.optimizer.lr == 1e-3 and run.name == "tune"  # geometric mean
    report = check(SOURCE + "schedule.warmup = int_uniform(1, 4)\n")
    assert [d.where for d in report.dimensions] == ["optimizer.lr", "schedule.warmup"]
    assert report.run is not None and report.run.train.schedule.warmup == 2


def test_a_trial_takes_its_values_is_named_for_them_and_prints_back_the_same():
    [space] = spaces(SOURCE + "optimizer.weight_decay = uniform(0, 0.2)\n")
    assert list(space.dimensions) == ["optimizer.lr", "optimizer.weight_decay"]
    run = space.trial({"optimizer.lr": 3.16e-4, "optimizer.weight_decay": 0.05})
    assert run.name == "tune[optimizer.lr=3.16e-4, optimizer.weight_decay=0.05]"
    assert run.train.optimizer.lr == 3.16e-4
    [again] = compile_runs(run_to_source(run), filename="again.bux")
    assert again.id == run.id
    short = space.trial({"optimizer.lr": 3e-3}, steps=3)
    assert short.name.endswith("weight_decay=0.1, steps=3]") and short.train.steps == 3
    assert short.axes == ("optimizer.lr", "optimizer.weight_decay", "steps")


def test_a_choice_picks_any_value_or_block_and_names_its_trials_by_its_option():
    source = (
        SOURCE.replace(
            "block = [norm attn] [norm ffn]",
            "block = choice(pre=[norm attn] [norm ffn], par=[norm (attn + ffn)])",
        )
        + "ffn.activation = choice(swish, gelu)\n"
    )
    [space] = spaces(source)
    run = space.trial({"block": 1, "ffn.activation": 1, "optimizer.lr": 1e-3})
    assert "block=par" in run.name and "ffn.activation=gelu" in run.name
    assert any(n.kind == "sum" for n in ir.walk(run.arch.graph))  # attn + ffn
    assert space.run.name == "tune" and "block" in space.dimensions


def test_a_dimension_inside_an_option_of_a_choice_is_one_of_the_space():
    source = SOURCE.replace(
        "optimizer = adamw(lr=log_uniform(1e-4, 1e-2))",
        "optimizer = choice(adamw(lr=log_uniform(1e-4, 1e-2)), "
        "muon(lr=log_uniform(1e-3, 0.1)))",
    )
    [space] = spaces(source)
    assert list(space.dimensions) == ["optimizer", "optimizer.lr", "optimizer.lr#2"]
    muon = space.trial({"optimizer": 1, "optimizer.lr#2": 0.02})
    assert muon.train.optimizer.type == "muon" and muon.train.optimizer.lr == 0.02
    assert (
        muon.name
        == "tune[optimizer=muon(lr=log_uniform(1e-3, 0.1)), optimizer.lr=0.02]"
    )


def test_each_run_of_a_sweep_is_a_search_of_its_own():
    found = spaces(SOURCE + "attn.num_kv_heads = mha: 2 | mqa: 1\n")
    assert [s.run.name for s in found] == ["tune[mha]", "tune[mqa]"]
    trial = found[1].trial({"optimizer.lr": 2e-3})
    assert trial.name == "tune[mqa, optimizer.lr=2e-3]"
    assert trial.axes == ("attn.num_kv_heads", "optimizer.lr")


def test_a_macros_dimension_takes_one_value_at_every_call():
    source = SOURCE.replace(
        "block = [norm attn] [norm ffn]",
        "layer(x) = [norm attn] [norm ffn(intermediate_size=int_uniform(8, 64))]\n"
        "block = layer(1) layer(2)",
    )
    [space] = spaces(source)
    assert list(space.dimensions) == ["layer.intermediate_size", "optimizer.lr"]
    run = space.trial({"layer.intermediate_size": 40, "optimizer.lr": 1e-3})
    sizes = {
        l.config.intermediate_size
        for l in run.arch.layers()
        if l.config.type == "gated_mlp"
    }
    assert sizes == {40}
    bounded = SOURCE.replace(
        "block = [norm attn] [norm ffn]",
        "layer(n) = [norm ffn(intermediate_size=int_uniform(8, n))]\n"
        "block = layer(16) layer(32)",
    )
    assert "its bounds cannot change with a macro's arguments" in error(bounded)


@pytest.mark.parametrize(
    "setting, message",
    [
        ("attn.num_kv_heads = int_uniform(1, 3)", "3 key and value heads"),
        ("optimizer.lr = log_uniform(0, 1)", "low: Input should be greater than 0"),
        ("optimizer.lr = uniform(0.1, 0.01)", "low must be less than high"),
        ("optimizer.lr = uniform(1, 2, 3)", "takes low and high in order"),
        ("ffn.activation = choice(swish)", "a choice needs two options"),
        ("ffn.activation = choice(a=swish, gelu)", "name every option"),
    ],
)
def test_check_tries_each_dimension_at_both_ends_and_every_option(setting, message):
    assert message in error(SOURCE + setting + "\n")


def test_an_objective_is_a_weighted_sum_of_metrics_and_prints_as_one():
    source = SOURCE + "w = 0.25\nobjective = w * eval_loss + (1 - w) * loss / 2 - 1\n"
    [run] = compile_runs(source, filename="tune.bux")
    goal = run.train.objective
    assert goal is not None and goal.constant == -1.0
    assert [(t.metric, t.weight) for t in goal.terms] == [
        ("eval_loss", 0.25),
        ("loss", 0.375),
    ]
    assert "objective = 0.25 * eval_loss + 0.375 * loss - 1.0" in run_to_source(run)
    assert tcfg.objective(compile_runs(SOURCE)[0].train).terms[0].metric == "eval_loss"


@pytest.mark.parametrize(
    "objective, message",
    [
        ("eval_loss * loss", "weights metrics by numbers, not by other metrics"),
        ("eval_bpb", "bits per byte need a tokenizer"),
        ("val_bpb", "no eval logs it"),
        ("2", "but this has none"),
        ("loss / eval_loss", "divides metrics by numbers"),
    ],
)
def test_an_objective_reads_only_what_the_run_logs(objective, message):
    assert message in error(SOURCE + f"objective = {objective}\n")


def test_the_editor_shows_a_dimension_and_an_objective_on_hover():
    source = SOURCE + "objective = 0.5 * eval_loss + 0.5 * loss\n"
    shown = features.hover(source, source.index("log_uniform") + 1)
    assert shown is not None and "A dimension of a search" in shown.markdown
    shown = features.hover(source, source.index("objective =") + 1)
    assert shown is not None and "0.5 * eval_loss + 0.5 * loss" in shown.markdown


def test_neither_the_search_nor_the_objective_is_in_a_runs_id():
    [run] = compile_runs(SOURCE)
    [other] = compile_runs(
        SOURCE + "objective = loss\nsearch = search(method=tpe, trials=4)\n"
    )
    assert run.id == other.id


# --- Coordinates, methods and halving -------------------------------------------------


def test_coordinates_decode_to_the_values_a_trial_takes_and_back():
    space = Coordinates(
        ("lr", "warmup", "act"),
        (
            tcfg.LogUniformConfig(low=1e-4, high=1e-2),
            tcfg.IntUniformConfig(low=1, high=8, log=True),
            tcfg.ChoiceConfig(options=("swish", "gelu", "relu")),
        ),
    )
    values = space.decode(space.from_unit(np.array([0.5, 0.999, 0.4])))
    assert values == {"lr": 1e-3, "warmup": 8, "act": 1}
    assert space.decode(space.encode(values)) == values


def test_a_value_rounds_to_its_digits_within_its_bounds():
    b2 = tcfg.UniformConfig(low=0.99, high=0.9999)
    assert [b2.value(u) for u in (0, 0.5, 0.97, 1)] == [0.99, 0.995, 0.9999, 0.9999]
    assert tcfg.LogUniformConfig(low=1.234e-4, high=1e-2).value(0) == 1.234e-4


def test_a_search_seed_is_not_negative():
    assert "greater than or equal to 0" in error(SOURCE + "search = search(seed=-1)\n")


@pytest.mark.parametrize(
    "method", [tcfg.RandomSearch(), tcfg.HaltonSearch(), tcfg.SobolSearch()]
)
def test_the_simple_methods_propose_the_same_points_again(method):
    space = Coordinates(
        ("x", "c"),
        (tcfg.UniformConfig(low=0, high=1), tcfg.ChoiceConfig(options=("a", "b"))),
    )
    empty = np.zeros((0, 2))
    points = [
        propose(method, space, History(empty, np.zeros(0), empty, i, 7))
        for i in range(8)
    ]
    again = [
        propose(method, space, History(empty, np.zeros(0), empty, i, 7))
        for i in range(8)
    ]
    assert np.array_equal(points, again)
    assert all(0 <= p[0] <= 1 and p[1] in (0, 1) for p in points)
    assert len({tuple(p) for p in points}) == 8


def finished(n: int, value: float, rung: int = 0, bracket: int = 0) -> Trial:
    return Trial(n, {}, bracket, started=[rung], results={rung: value})


def test_asha_promotes_the_best_third_of_a_rung_as_soon_as_it_has_earned_it():
    config = tcfg.HalvingConfig(eta=3, rungs=3)
    trials = [finished(0, 3.0), finished(1, 1.0)]
    assert halving.next_job(trials, config, budget=9) == Job(0)  # two: none earned
    trials.append(finished(2, 2.0))
    job = halving.next_job(trials, config, budget=9)
    assert job is not None and job.rung == 1 and job.trial is trials[1]
    trials[1].started.append(1)
    assert halving.next_job(trials, config, budget=3) is None  # until one ends
    assert (
        halving.steps(config, 900, 0) == 100 and halving.steps(config, 900, 2) is None
    )


def test_hyperband_starts_trials_at_each_bracket_in_turn_and_ranks_them_apart():
    config = tcfg.HalvingConfig(eta=2, rungs=3, brackets=2)
    trials = [finished(0, 1.0), finished(1, 2.0, rung=1, bracket=1)]
    assert halving.next_job(trials, config, budget=9) == Job(0)  # number 2: bracket 0
    trials += [finished(2, 3.0), finished(3, 0.5, rung=1, bracket=1)]
    job = halving.next_job(trials, config, budget=9)
    assert job == Job(2, trials[3])  # bracket 1's best, up from rung 1


def test_the_best_trial_is_the_least_at_the_highest_rung_with_a_number():
    trials = [finished(0, 0.1), finished(1, 0.5, rung=1), finished(2, math.nan, rung=2)]
    assert halving.best(trials) is trials[1]
    assert halving.best([finished(0, math.nan)]) is None


# --- A search's trials, on disk -------------------------------------------------------


def ended(run, out: Path, state: str = "done", **metrics: float) -> None:
    """Make `run`'s directory say it ended so, with these latest metrics."""
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "status.json").write_text(json.dumps({"state": state}))
    (folder / "metrics.jsonl").write_text(json.dumps({"step": 9, **metrics}) + "\n")
    (folder / "run.bux").write_text(run_to_source(run))


def started(search: Search) -> ir.Run:
    """Return the next run `search` trains."""
    run = search.next()
    assert run is not None
    return run


def test_a_search_keeps_its_trials_and_goes_on_where_it_stopped(tmp_path):
    source = SOURCE + "search = search(method=halton, trials=3)\n"
    [space] = spaces(source)
    first = Search(space, tmp_path)
    runs = [started(first) for _ in range(3)]
    assert first.next() is None and len({r.id for r in runs}) == 3
    ended(runs[0], tmp_path, eval_loss=2.0)
    ended(runs[1], tmp_path, "failed")
    first.ended(runs[0])
    first.ended(runs[1])
    assert first.trials[0].results == {0: 2.0} and math.isnan(
        first.trials[1].results[0]
    )

    again = Search(space, tmp_path)  # the third, unfinished, trains again first
    assert again.folder == first.folder and again.next() == runs[2]
    assert again.next() is None
    ended(runs[2], tmp_path, eval_loss=1.5)
    again.ended(runs[2])
    assert again.best() is again.trials[2]
    assert (again.folder / "best.bux").read_text() == run_to_source(runs[2])
    more = Search(spaces(source, overrides=["search.trials = 4"])[0], tmp_path)
    assert more.folder == first.folder and len(more.trials) == 3
    assert more.next() is not None and len(more.trials) == 4


def test_a_point_a_setting_rejects_fails_its_trial_without_a_run(tmp_path):
    source = SOURCE + "attn.num_kv_heads = int_uniform(1, 3)\n"
    source += "search = search(method=halton, trials=8)\n"
    said: list[str] = []
    s = Search(spaces(source)[0], tmp_path, echo=said.append)
    started = [run for _ in range(8) if (run := s.next()) is not None]
    rejected = [t for t in s.trials if t.values["attn.num_kv_heads"] == 3]
    assert rejected and len(started) == 8 - len(rejected) and len(s.trials) == 8
    assert all(math.isnan(t.results[0]) for t in rejected)
    assert len(said) == len(rejected) and "do not divide 2 query heads" in said[0]


def test_a_trial_already_done_is_read_not_trained_again(tmp_path):
    source = SOURCE + "search = search(method=halton, trials=2)\n"
    [space] = spaces(source)
    s = Search(space, tmp_path)
    run = s.next()
    assert run is not None
    ended(run, tmp_path, eval_loss=4.0)
    (s.folder / "trials.jsonl").unlink()  # another search, same point
    t = Search(space, tmp_path)
    second = t.next()
    assert second != run and t.trials[0].results == {0: 4.0}


def test_a_search_trains_at_most_its_parallel_trials_at_once(tmp_path):
    source = SOURCE + "search = search(method=halton, trials=4, parallel=2)\n"
    s = Search(spaces(source)[0], tmp_path)
    first, second = started(s), started(s)
    assert s.next() is None
    ended(first, tmp_path, eval_loss=1.0)
    s.ended(first)
    assert started(s) not in (first, second)


def test_trials_at_the_same_point_are_one_run_trained_once(tmp_path):
    source = SOURCE.replace("log_uniform(1e-4, 1e-2)", "1e-3")
    source += "ffn.activation = choice(swish, gelu)\n"
    source += "search = search(method=halton, trials=4)\n"
    s = Search(spaces(source)[0], tmp_path)
    runs = [run for _ in range(4) if (run := s.next()) is not None]
    assert len(s.trials) == 4 and len({r.id for r in runs}) == len(runs) == 2
    for run in runs:
        ended(run, tmp_path, eval_loss=1.0)
        s.ended(run)
    assert [t.results for t in s.trials] == [{0: 1.0}] * 4


def test_searches_that_propose_the_same_run_train_it_once(tmp_path):
    first, second = (
        Search(spaces(SOURCE + f"search = search(method={m}, trials=1)\n")[0], tmp_path)
        for m in ("halton", "tpe")  # tpe starts at halton's points
    )
    queue = Searches([first, second])
    taken = queue.take(["0", "1"], lambda run: 1)
    assert taken is not None and queue.take(["1"], lambda run: 1) is None
    ended(taken[0], tmp_path, eval_loss=2.0)
    queue.ended(taken[0])
    assert first.trials[0].results == second.trials[0].results == {0: 2.0}


def test_an_objective_is_read_from_the_latest_metrics(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 3.0, "a_loss": 9.0}\n{"step": 2, "loss": 2.0}\n'
    )
    goal = tcfg.ObjectiveConfig(
        terms=(
            tcfg.Term(metric="loss", weight=0.5),
            tcfg.Term(metric="a_loss", weight=1.0),
        )
    )
    assert objective(goal, tmp_path) == 10.0
    missing = tcfg.ObjectiveConfig(terms=(tcfg.Term(metric="b_loss", weight=1.0),))
    assert math.isnan(objective(missing, tmp_path))


def test_searches_take_turns_and_each_run_gets_the_gpus_it_needs(tmp_path):
    source = SOURCE + "attn.num_kv_heads = mha: 2 | mqa: 1\n"
    source += "search = search(method=halton, trials=2)\n"
    queue = Searches([Search(s, tmp_path) for s in spaces(source)])
    names = []
    while (taken := queue.take(["0", "1", "2"], lambda run: 2)) is not None:
        run, group = taken
        assert group == ["0", "1"]
        names.append(run.name.split(",")[0])
    assert names == ["tune[mha", "tune[mqa", "tune[mha", "tune[mqa"]
    assert queue.take(["0"], lambda run: 2) is None
    assert isinstance(queue, Queue)


# --- Trained --------------------------------------------------------------------------


def test_bux_train_searches_a_file_and_launched_again_trains_nothing(tmp_path, capsys):
    path = tmp_path / "tune.bux"
    path.write_text(
        SOURCE + "search = search(method=halton, trials=4, halving=halving(rungs=2))\n"
    )
    out = str(tmp_path / "runs")
    assert main(["train", str(path), "--out", out]) == 0
    printed = capsys.readouterr().out
    table = printed[printed.index("search tune:") :].splitlines()
    assert table[0].startswith("search tune: 4 of 4 trials; objective = eval_loss")
    assert table[2].split()[1] == "1/1"  # a trial promoted, the best
    assert sum(" 0/1 " in line for line in table) == 3

    assert main(["train", str(path), "--out", out]) == 0
    assert "training on" not in capsys.readouterr().out
    assert main(["results", str(path), "--out", out]) == 0
    assert capsys.readouterr().out.splitlines()[:3] == table[:3]


def test_bux_launch_trains_trials_side_by_side(tmp_path):
    source = SOURCE + "search = search(method=random, trials=3)\n"
    said: list[str] = []
    assert search(spaces(source), tmp_path, gpus=["0", "1"], echo=said.append) == 0
    assert said[0] == "searching tune on GPUs 0,1"
    folders = [p for p in tmp_path.iterdir() if p.name.startswith("tune_")]
    states = [status(f) for f in folders]
    assert len(states) == 3 and {s["state"] for s in states} == {"done"}
    assert sorted({tuple(s["gpus"]) for s in states}) == [("0",), ("1",)]
    assert said[-1].startswith("search tune: 3 of 3 trials")


def test_a_sweep_with_an_objective_shows_each_runs(tmp_path):
    runs = compile_runs(
        SOURCE.replace("log_uniform(1e-4, 1e-2)", "1e-3 | 1e-2") + "objective = loss\n"
    )
    ended(runs[0], tmp_path, loss=2.5)
    lines = summary(runs, tmp_path).splitlines()
    assert lines[1].split()[:5] == ["state", "step", "loss", "eval", "loss"]
    assert lines[2].split()[4] == "2.5000" and lines[3].split()[4] == "-"
