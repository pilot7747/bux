"""The launcher: runs recorded, stopped by signals, resumed, and swept in workers."""

import os
import signal
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

import bux.launch
from bux.__main__ import main
from bux.lang import compile_runs
from bux.launch import Stop, launch, status, summary
from bux.train.hooks import Hook
from bux.train.trainer import Trainer, directory, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"


def smoke(*overrides: str):
    settings = ("schedule.warmup = 1", "steps = 6", *overrides)
    return compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=settings)


class Signal(Hook):
    """Sends this process `signum` after step `at`, as a scheduler would."""

    def __init__(self, at, signum=signal.SIGTERM):
        self.at, self.signum = at, signum

    def due(self, step, steps):
        return step == self.at

    def __call__(self, trainer, metrics):
        os.kill(os.getpid(), self.signum)


def test_a_launch_records_its_run_and_does_not_train_it_again(tmp_path):
    [run] = smoke()
    said: list[str] = []
    assert launch([run], tmp_path, echo=said.append) == 0
    folder = directory(run, tmp_path)
    s = status(folder)
    assert s["state"] == "done" and s["step"] == 6 and s["error"] is None
    assert s["metrics"]["loss"] > 0 and s["pid"] == os.getpid()
    assert said[-1] == "smoke: done"
    metrics = (folder / "metrics.jsonl").read_text()

    said.clear()
    assert launch([run], tmp_path, echo=said.append) == 0
    assert said == ["1 of 1 run done already"]
    assert (folder / "metrics.jsonl").read_text() == metrics


def test_a_signal_stops_a_run_at_a_checkpoint_and_a_launch_resumes_it(
    tmp_path, monkeypatch
):
    [run] = smoke("checkpoint = checkpoint(every=100)")

    def interrupted(run, out, *, hooks=(), **kwargs):
        return train(run, out, hooks=[*hooks, Signal(at=3)], **kwargs)

    monkeypatch.setattr(bux.launch, "train", interrupted)
    handler = signal.getsignal(signal.SIGTERM)
    said: list[str] = []
    assert launch([run], tmp_path, echo=said.append) == 128 + signal.SIGTERM
    folder = directory(run, tmp_path)
    assert status(folder)["state"] == "stopped" and status(folder)["step"] == 3
    assert [p.name for p in (folder / "checkpoints").iterdir()] == ["3"]
    assert said[-1].startswith(
        "smoke: stopped at step 3 by SIGTERM; launching it again"
    )
    assert signal.getsignal(signal.SIGTERM) == handler  # as it was

    monkeypatch.undo()
    said.clear()
    assert launch([run], tmp_path, echo=said.append) == 0
    assert "smoke: resuming at step 3" in said
    assert status(folder)["state"] == "done" and status(folder)["step"] == 6


def test_a_signal_delivered_twice_is_one_and_a_second_stops_at_once():
    stop, trainer = Stop(), SimpleNamespace(stopping=False)
    with pytest.raises(KeyboardInterrupt):  # before training, at once
        Stop().handle(signal.SIGTERM, None)
    stop.start(typing.cast(Trainer, trainer))
    stop.handle(signal.SIGTERM, None)
    assert trainer.stopping and stop.signal == signal.SIGTERM
    stop.handle(signal.SIGTERM, None)  # from `uv run`, passing it on
    stop.since -= 2
    with pytest.raises(KeyboardInterrupt):
        stop.handle(signal.SIGINT, None)
    assert stop.signal == signal.SIGTERM


def test_a_sweep_trains_a_worker_a_run_and_a_failure_leaves_the_others(tmp_path):
    missing = tmp_path / "nowhere" / "*.bin"
    runs = smoke(f'data = ok: synthetic | missing: tokens(path="{missing}")')
    said: list[str] = []
    assert launch(runs, tmp_path, gpus=["0", "1"], echo=said.append) == 1
    assert said[0] == "training 2 runs, 2 at a time on GPUs 0,1"
    ok, failed = (status(directory(run, tmp_path)) for run in runs)
    assert ok["state"] == "done" and failed["state"] == "failed"
    assert {ok["gpu"], failed["gpu"]} == {"0", "1"}  # a GPU each
    assert failed["error"].startswith("FileNotFoundError: no shards match")
    assert ok["pid"] != os.getpid()
    log = (directory(runs[1], tmp_path) / "train.log").read_text()
    assert "Traceback" in log and "smoke[missing]: failed at step 0" in log
    assert any(line.startswith("smoke[ok] step 6/6") for line in said)
    assert any(line.startswith("smoke[missing]: failed (exit 1)") for line in said)
    table = summary(runs, tmp_path).splitlines()
    assert table[0] == "2 runs: 1 done, 1 failed"
    assert table[2].split()[:2] == ["done", "6/6"] and table[3].endswith(
        "smoke[missing]"
    )


def test_the_command_line_launches_one_run_of_a_sweep(tmp_path, capsys):
    settings = ["steps=2", "schedule.warmup=1", "seed = 0 | 1", "--out", str(tmp_path)]
    assert main(["launch", str(SMOKE), *settings, "--index", "1"]) == 0
    out = capsys.readouterr().out
    assert "smoke[seed=1] step 2/2" in out and "seed=0" not in out
    assert main(["launch", str(SMOKE), *settings, "--index", "2"]) == 1
    assert "--index 2, but the file makes 2 runs" in capsys.readouterr().err
    assert main(["train", str(SMOKE), *settings, "--only", "seed=7"]) == 1
    assert "no run's name contains 'seed=7'" in capsys.readouterr().err
