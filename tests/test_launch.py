"""The launcher: runs recorded, stopped by signals, resumed, swept in workers on groups
of GPUs, and trained by many processes together."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

import bux.launch
from bux.__main__ import _bytes, _si, abstract, main, memory
from bux.lang import compile_runs
from bux.launch import Stop, fit, launch, need, processes, status, summary
from bux.runtime import cache_dir, xla_flags
from bux.train.config import MeshConfig
from bux.train.hooks import Hook
from bux.train.trainer import Trainer, directory, train

SMOKE = Path(__file__).parent.parent / "examples" / "experiments" / "smoke.bux"
# A process of a job of many: `bux` with the arguments it gets. Gloo, which carries
# collectives between CPU processes, finds its peers by the host's name, which a laptop
# may not resolve, so here it is localhost.
JOINED = """
import sys

import jax.extend.backend
from jax._src import distributed, xla_bridge
from jax._src.lib import _jax


def cpu():
    client = distributed.global_state.client
    gloo = _jax.make_gloo_tcp_collectives(client, hostname="localhost")
    return xla_bridge.make_cpu_client(collectives=gloo)


jax.extend.backend.register_backend_factory("cpu", cpu, fail_quietly=False)
from bux.__main__ import main

sys.exit(main(sys.argv[1:]))
"""


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
    assert said[0] == "training 2 runs on GPUs 0,1"
    ok, failed = (status(directory(run, tmp_path)) for run in runs)
    assert ok["state"] == "done" and failed["state"] == "failed"
    assert sorted([ok["gpus"], failed["gpus"]]) == [["0"], ["1"]]  # a GPU each
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


# --- GPUs per run ----------------------------------------------------------------------


def test_a_run_needs_its_meshs_gpus_and_data_unset_is_all_alone_and_one_in_a_sweep():
    def gpus(alone: bool, **sizes: int) -> int:
        return need(MeshConfig.model_validate(sizes), 8, alone=alone)

    assert gpus(True) == 8 and gpus(False) == 1
    assert gpus(True, fsdp=2, tensor=2) == 8 and gpus(False, fsdp=2, tensor=2) == 4
    assert gpus(True, data=2, fsdp=2) == gpus(False, data=2, fsdp=2) == 4
    assert gpus(True, fsdp=16) == 16  # more than there are
    assert need(MeshConfig(), 0, alone=True) == 1  # on the default device


def test_the_first_run_that_fits_gets_the_first_free_gpus():
    assert fit([2, 1], ["0", "1", "2"]) == (0, ["0", "1"])
    assert fit([4, 2, 1], ["3", "5", "6"]) == (1, ["3", "5"])  # the 4 waits
    assert fit([4], ["0", "1"]) is None and fit([], ["0"]) is None


def test_a_sweep_gives_each_run_its_gpus_and_one_that_needs_too_many_fails_at_once(
    tmp_path,
):
    runs = smoke("mesh = two: mesh(fsdp=2) | one: mesh(data=1) | four: mesh(fsdp=4)")
    said: list[str] = []
    assert launch(runs, tmp_path, gpus=["0", "1", "2"], echo=said.append) == 1
    too_many = "mesh: needs 4 GPUs, but there are 3"
    assert said[:2] == [
        f"smoke[four]: failed before training: {too_many}",
        "training 2 runs on GPUs 0,1,2",
    ]
    two, one, four = (status(directory(run, tmp_path)) for run in runs)
    assert two["gpus"] == ["0", "1"] and one["gpus"] == ["2"]
    assert two["state"] == one["state"] == "done"
    assert four["state"] == "failed" and four["error"] == too_many
    assert not (directory(runs[2], tmp_path) / "train.log").exists()


def test_a_run_alone_trains_on_the_gpus_it_needs(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")  # restored after the test
    [run] = smoke("mesh = mesh(data=1, fsdp=2)")
    said: list[str] = []
    assert launch([run], tmp_path, echo=said.append) == 0
    assert status(directory(run, tmp_path))["gpus"] == ["4", "5"]
    assert "cpu (GPU 4,5) of " in said[0]


# --- XLA -------------------------------------------------------------------------------


def test_bux_s_xla_flags_go_first_and_one_already_set_keeps_its_value():
    ragged = "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl"
    assert xla_flags("") == f"{ragged}=false"
    assert xla_flags("--xla_dump_to=x") == f"{ragged}=false --xla_dump_to=x"
    assert (
        xla_flags(f"{ragged}=true --xla_dump_to=x") == f"{ragged}=true --xla_dump_to=x"
    )
    assert cache_dir({"XDG_CACHE_HOME": "/c"}) == Path("/c/bux/jax")


def test_a_launch_keeps_its_compiled_steps_for_the_next(tmp_path):
    env = os.environ | {
        "XDG_CACHE_HOME": str(tmp_path),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",  # even a smoke run's
    }
    del env["JAX_COMPILATION_CACHE_DIR"]
    settings = ["steps=1", "schedule.warmup=1", "--out", str(tmp_path / "runs")]
    command = [sys.executable, "-m", "bux", "train", str(SMOKE), *settings]
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert any("train_step" in p.name for p in (tmp_path / "bux" / "jax").iterdir())


# --- Many processes --------------------------------------------------------------------


def test_the_environment_says_how_many_processes_train_together():
    assert processes({}) == 1
    assert processes({"SLURM_STEP_NUM_TASKS": "16", "SLURM_PROCID": "3"}) == 16
    assert processes({"OMPI_COMM_WORLD_SIZE": "4"}) == 4
    assert processes({"JAX_NUM_PROCESSES": "2", "SLURM_STEP_NUM_TASKS": "8"}) == 2


def test_many_processes_train_a_run_alone_not_a_sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_NUM_PROCESSES", "2")
    said: list[str] = []
    assert launch(smoke("seed = 0 | 1"), tmp_path, echo=said.append) == 1
    assert said == ["2 processes train one run, not 2: use --index"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def joined(out: Path, settings: tuple[str, ...]) -> list[subprocess.Popen]:
    """Two processes of 4 CPU devices each, launching the smoke run under `out`
    together, each saying what it says to out/process<i>.txt."""
    out.mkdir(exist_ok=True)
    env = os.environ | {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "JAX_COORDINATOR_ADDRESS": f"localhost:{_free_port()}",
        "JAX_NUM_PROCESSES": "2",
    }
    command = [sys.executable, "-c", JOINED, "launch", str(SMOKE), *settings]
    workers = []
    for i in range(2):
        with (out / f"process{i}.txt").open("a") as log:
            workers.append(
                subprocess.Popen(
                    [*command, "--out", str(out)],
                    env=env | {"JAX_PROCESS_ID": str(i)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
    return workers


def ended(workers: list[subprocess.Popen]) -> list[int]:
    try:
        return [w.wait(timeout=300) for w in workers]
    finally:
        for w in workers:
            w.kill()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def logged(path: Path) -> int:
    """The lines of metrics.jsonl so far."""
    return path.read_text().count("\n") if path.exists() else 0


def outputs(out: Path) -> list[str]:
    return [(out / f"process{i}.txt").read_text() for i in range(2)]


def test_two_processes_train_as_one_does_and_a_signal_to_either_stops_both(tmp_path):
    settings = (
        "schedule.warmup = 1",
        "steps = 40",
        "mesh = mesh(fsdp=8)",
        "log = log(every=1)",
        "checkpoint = checkpoint(every=100)",
        "precision.compute = float32",  # bfloat16 rounds apart between splits
    )
    [run] = compile_runs(SMOKE.read_text(), filename=str(SMOKE), overrides=settings)
    out = tmp_path / "two"
    folder = directory(run, out)
    workers = joined(out, settings)
    deadline = time.monotonic() + 300
    while logged(folder / "metrics.jsonl") < 2 and time.monotonic() < deadline:
        assert all(w.poll() is None for w in workers), outputs(out)
        time.sleep(0.05)
    workers[1].send_signal(signal.SIGTERM)  # the second alone
    assert ended(workers) == [128 + signal.SIGTERM] * 2, outputs(out)
    s = status(folder)
    assert s["state"] == "stopped" and 2 <= s["step"] < 40
    assert s["pid"] == workers[0].pid  # the first process records
    assert [p.name for p in (folder / "checkpoints").iterdir()] == [str(s["step"])]

    assert ended(joined(out, settings)) == [0, 0], outputs(out)
    assert status(folder)["state"] == "done" and status(folder)["step"] == 40
    said = outputs(out)
    assert f"smoke: stopped at step {s['step']} by SIGTERM" in said[0]
    assert "smoke step 40/40" in said[0] and "step" not in said[1]
    two = lines(folder / "metrics.jsonl")
    assert [m["step"] for m in two] == list(range(1, 41))  # written once

    train(run, tmp_path / "one", echo=lambda _: None)  # one process of 8 devices
    one = lines(directory(run, tmp_path / "one") / "metrics.jsonl")
    assert [m["loss"] for m in two] == pytest.approx([m["loss"] for m in one], abs=1e-5)


# --- The plan --------------------------------------------------------------------------


def test_the_plan_says_each_runs_gpus_and_its_bytes_on_each(capsys):
    sweep = "mesh = fsdp: mesh(fsdp=8) | one: mesh(data=1)"
    assert main(["plan", str(SMOKE), sweep]) == 0
    head, fsdp, one = capsys.readouterr().out.splitlines()[1:]
    assert head.split() == [
        *("id", "params", "tokens", "tok/param", "flops/tok"),
        *("gpus", "params/gpu", "optim/gpu", "bubble", "run"),
    ]
    assert fsdp.split()[5] == "8" and one.split()[5] == "1"
    runs = smoke(sweep)
    made = abstract(runs[0])
    assert made is not None
    split, whole = memory(runs[0], made, 8), memory(runs[1], made, 1)
    assert split is not None and whole is not None
    assert split[0] * 8 == whole[0]  # every parameter of the smoke model splits
    assert split[1] == pytest.approx(whole[1] / 8, rel=1e-3)  # and its moments


def test_the_plan_shows_three_digits_never_an_exponent():
    assert _bytes(1010 * 2**20) == "0.986GiB" and _bytes(609 * 2**10) == "609KiB"
    assert _bytes(512) == "512B" and _bytes(3 * 2**40) == "3TiB"
    assert _si(999_600) == "1M" and _si(25_600) == "25.6k" and _si(512) == "512"
    assert _si(10e9) == "10B" and _si(70.6e9) == "70.6B"


def test_the_plan_sizes_muons_state_on_a_mesh(capsys):
    muon = ("optimizer = muon(lr=0.02)", "mesh = mesh(fsdp=8)")
    assert main(["plan", str(SMOKE), *muon]) == 0  # its state holds masked nodes
    row = capsys.readouterr().out.splitlines()[-1].split()
    assert row[5] == "8" and "?" not in row[4:8]
