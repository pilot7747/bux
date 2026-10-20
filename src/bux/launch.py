"""The launcher, ``bux launch``: what trains an experiment, a .bux file of one run or a
sweep of many, to its end, on a cluster as on a laptop.

    bux launch experiments/attention_ablation.bux             # all 6 runs, a GPU each
    bux launch experiments/attention_ablation.bux --index 3   # the fourth, in an array job

First it prepares the datasets the runs read, once. A run alone then trains in this
process, on the first visible GPU. A sweep trains in workers, a process a run, one on
each visible GPU at a time (CUDA_VISIBLE_DEVICES says which), or one at a time on the
default device if there are none. A worker's output goes to the console and to its
run's train.log, and a run that fails or runs out of memory leaves the others training.

A run's directory (``bux.train.trainer``) also keeps a status.json: whether the run is
running, done, failed or stopped, where, since when, at what step, and its latest
metrics. A run that is done is not trained again, and one that is not resumes from its
latest checkpoint, so launching an experiment again, after a preemption, a crash or a
fix, trains what is left of it.

SIGTERM or SIGINT stops training after the step it is at, which a checkpoint setting
saves, and a second one, a second or more later, stops it at once; a sweep's launcher
passes them on to its workers. The launcher exits with 0 when every run is done, 1 when one failed, and
128 + the signal's number when a signal stopped them.
"""

import collections
import contextlib
import datetime
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import jax

from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.train.hooks import Hook
from bux.train.sinks import Latest
from bux.train.trainer import Trainer, directory, train

SIGNALS = (signal.SIGTERM, signal.SIGINT)
# A signal sooner than this after the first is the same one, delivered twice: to this
# process and to its parent, `uv run` or `srun` say, which passes it on.
REPEAT = 1.0  # seconds


def status(folder: Path) -> dict[str, Any]:
    """What the run in `folder` last said of itself in its status.json, if anything."""
    path = folder / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record(folder: Path, **fields: Any) -> None:
    """Update the run's status.json with `fields`, all at once."""
    tmp = folder / "status.json.tmp"
    tmp.write_text(json.dumps(status(folder) | fields, indent=2), encoding="utf-8")
    tmp.replace(folder / "status.json")


def _print(line: str) -> None:
    print(line, flush=True)  # now, though stdout is a pipe, to a container's log say


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


class Stop(Hook):
    """Stops training at SIGTERM or SIGINT, after the step it is at; at a second
    signal, or at one before training starts, at once, with a KeyboardInterrupt."""

    def __init__(self) -> None:
        self.trainer: Trainer | None = None
        self.signal: int | None = None  # the first one
        self.since = 0.0

    def due(self, step: int, steps: int) -> bool:
        return False

    def start(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def handle(self, signum: int, frame: FrameType | None) -> None:
        if self.signal is None:
            self.signal, self.since = signum, time.monotonic()
            if self.trainer is None:
                raise KeyboardInterrupt
            self.trainer.stopping = True
        elif time.monotonic() - self.since >= REPEAT:
            raise KeyboardInterrupt

    @contextlib.contextmanager
    def armed(self) -> Generator["Stop"]:
        """With the signals handled."""
        previous = {s: signal.signal(s, self.handle) for s in SIGNALS}
        try:
            yield self
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def train_here(
    run: ir.Run,
    out: str | Path = "runs",
    cache: str | Path = DEFAULT_CACHE,
    *,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train `run` in this process, as `train` does, and record how it went in its
    status.json; the exit code."""
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    host, gpu = socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES")
    record(
        folder,
        state="running",
        host=host,
        pid=os.getpid(),
        gpu=gpu,
        started=_now(),
        ended=None,
        error=None,
    )
    stop, latest, error = Stop(), Latest(), None
    try:
        with stop.armed():
            kind = jax.devices()[0].device_kind + (f" (GPU {gpu})" if gpu else "")
            echo(f"{run.name}: training on {kind} of {host} with jax {jax.__version__}")
            train(run, out, cache=cache, echo=echo, hooks=[stop], sinks=[latest])
    except KeyboardInterrupt:
        stop.signal = stop.signal or signal.SIGINT
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        error = f"{type(err).__name__}: {err}"
    step = None if stop.trainer is None else stop.trainer.step
    at = "before training" if step is None else f"at step {step}"
    if error is not None:
        state, code, said = "failed", 1, f"failed {at}: {error}"
    elif stop.signal is not None and step != run.train.steps:
        state, code = "stopped", 128 + stop.signal
        again = (
            "resumes it from its latest checkpoint"
            if run.train.checkpoint is not None
            else "starts it over, as it has no checkpoint setting"
        )
        said = f"stopped {at} by {_name(stop.signal)}; launching it again {again}"
    else:
        state, code, said = "done", 0, "done"
    fields: dict[str, Any] = {"state": state, "ended": _now(), "error": error}
    if step is not None:
        fields["step"] = step
    if latest.metrics:
        fields["metrics"] = latest.metrics
    record(folder, **fields)
    echo(f"{run.name}: {said}")
    return code


def visible_gpus() -> list[str]:
    """The GPUs to give workers: those CUDA_VISIBLE_DEVICES lists, if it is set, or
    else every one that nvidia-smi finds."""
    if (listed := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
        return [g for g in listed.split(",") if g.strip() not in ("", "-1")]
    try:
        found = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        return []
    return [str(i) for i in range(sum(g.startswith("GPU ") for g in found.split("\n")))]


def _spawn(
    run: ir.Run, out: str | Path, cache: str | Path, gpu: str | None
) -> subprocess.Popen[str]:
    """A worker that trains `run` on `gpu`, or on the default device, from its
    run.json: what the launcher compiled, whatever its files say by now."""
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    record(folder, state="running", gpu=gpu, started=_now(), ended=None, error=None)
    env = os.environ | ({} if gpu is None else {"CUDA_VISIBLE_DEVICES": gpu})
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "bux.launch", str(folder), str(cache)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        start_new_session=True,  # so that a Ctrl-C reaches it once, through us
    )


def supervise(
    runs: Sequence[ir.Run],
    out: str | Path,
    cache: str | Path,
    gpus: Sequence[str | None],
    *,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train `runs` in workers, one on each of `gpus` at a time (None: the default
    device); the exit code."""
    pending, free = collections.deque(runs), list(gpus)
    active: dict[subprocess.Popen[str], tuple[ir.Run, str | None]] = {}
    ended: queue.Queue[subprocess.Popen[str]] = queue.Queue()
    lock = threading.Lock()
    signals: list[int] = []
    failed = False

    def say(line: str) -> None:
        with lock:
            echo(line)

    def relay(worker: subprocess.Popen[str], log: Path) -> None:
        assert worker.stdout is not None
        with log.open("a", encoding="utf-8") as f:
            for line in worker.stdout:
                f.write(line)
                f.flush()
                say(line.rstrip("\n"))
        worker.wait()
        ended.put(worker)

    def forward(signum: int, frame: FrameType | None) -> None:
        signals.append(signum)
        for worker in list(active):
            worker.send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in SIGNALS}
    try:
        while pending or active:
            while pending and free and not signals:
                run, gpu = pending.popleft(), free.pop(0)
                worker = _spawn(run, out, cache, gpu)
                active[worker] = run, gpu
                if signals:  # that came as it started
                    worker.send_signal(signals[0])
                log = directory(run, out) / "train.log"
                threading.Thread(target=relay, args=(worker, log), daemon=True).start()
            if not active:
                break
            worker = ended.get()
            run, gpu = active.pop(worker)
            free.append(gpu)
            folder, code = directory(run, out), worker.returncode
            how = _name(-code) if code < 0 else f"exit {code}"
            if status(folder).get("state") == "running":  # it died before it could say
                stopped = bool(signals) and code in (-signals[0], 128 + signals[0])
                error = None if stopped else how
                state = "stopped" if stopped else "failed"
                record(folder, state=state, ended=_now(), error=error)
            if status(folder)["state"] == "failed":
                failed = True
                log = folder / "train.log"
                say(f"{run.name}: failed ({how}); its output is in {log}")
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    return 128 + signals[0] if signals else int(failed)


def summary(runs: Sequence[ir.Run], out: str | Path) -> str:
    """A line a run: its state, its step, and its latest losses."""
    rows: list[tuple[str, ...]] = [("state", "step", "loss", "eval loss", "run")]
    for run in runs:
        s = status(directory(run, out))
        m = s.get("metrics") or {}
        losses = (f"{m[k]:.4f}" if k in m else "-" for k in ("loss", "eval_loss"))
        step = f"{s.get('step', 0)}/{run.train.steps}"
        rows.append((s.get("state", "pending"), step, *losses, run.name))
    states = Counter(row[0] for row in rows[1:])
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths)) + "  " + row[4]
        for row in rows
    ]
    head = ", ".join(f"{n} {state}" for state, n in states.most_common())
    return f"{len(runs)} runs: {head}\n" + "\n".join(lines)


def launch(
    runs: Sequence[ir.Run],
    out: str | Path = "runs",
    *,
    cache: str | Path = DEFAULT_CACHE,
    gpus: Sequence[str] | None = None,
    workers: int | None = None,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train what is left of `runs` under `out`, after preparing the datasets they
    read under `cache`, with `workers` processes; the exit code. A run alone trains
    in this process, on the first of `gpus` (default: every visible one), and a sweep
    in workers, one on each of them at a time."""
    todo = [run for run in runs if status(directory(run, out)).get("state") != "done"]
    if done := len(runs) - len(todo):
        echo(f"{done} of {len(runs)} run{'s' if len(runs) != 1 else ''} done already")
    if not todo:
        return 0
    stop = Stop()
    try:
        with stop.armed():
            for job in dict.fromkeys(job for run in todo for job in jobs(run.train)):
                prepare(job, cache, workers=workers, echo=echo)
    except KeyboardInterrupt:
        return 128 + (stop.signal or signal.SIGINT)
    gpus = visible_gpus() if gpus is None else list(gpus)
    if len(todo) == 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        return train_here(todo[0], out, cache, echo=echo)
    slots: list[str | None] = [*gpus] or [None]
    on = f" on GPUs {','.join(gpus)}" if gpus else ""
    echo(f"training {len(todo)} runs, {len(slots)} at a time{on}")
    code = supervise(todo, out, cache, slots, echo=echo)
    echo(summary(runs, out))
    return code


if __name__ == "__main__":  # a sweep's worker: python -m bux.launch FOLDER CACHE
    folder, cache = Path(sys.argv[1]), Path(sys.argv[2])
    run = ir.Run.model_validate_json((folder / "run.json").read_text(encoding="utf-8"))
    sys.exit(train_here(run, folder.parent, cache))
