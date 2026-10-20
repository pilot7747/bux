"""The launcher, ``bux launch``: what trains an experiment, a .bux file of one run or a
sweep of many, to its end, on a cluster as on a laptop.

    bux launch examples/experiments/attention_ablation.bux             # all 6 runs
    bux launch examples/experiments/attention_ablation.bux --index 3   # the fourth, in an array job

A run needs the GPUs of its mesh, the product of its axes' sizes, where `data` unset
is every visible GPU for a run alone and one in a sweep; one that needs more than
there are fails at once. The launcher prepares the datasets the others read, once. A
run alone then trains in this process. A sweep trains in workers, a process a run,
each on a group of free GPUs (CUDA_VISIBLE_DEVICES says which), the first pending run
that fits starting first, or one at a time on the default device if there are none. A
worker's output goes to the console and to its run's train.log, and a run that fails
or runs out of memory leaves the others training.

A run alone trains on many hosts when the environment says so: Slurm's srun, Open
MPI's mpirun, or JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES and JAX_PROCESS_ID. Its
processes join through jax.distributed, and the first writes the console, status.json
and the run's files; checkpoints are all of theirs.

A run's directory (``bux.train.trainer``) also keeps a status.json: whether the run is
running, done, failed or stopped, where, on what GPUs, since when, at what step, and
its latest metrics. A run that is done is not trained again, and one that is not
resumes from its latest checkpoint, so launching an experiment again, after a
preemption, a crash or a fix, trains what is left of it.

SIGTERM or SIGINT stops training after the step it is at, which a checkpoint setting
saves, and a second one, a second or more later, stops it at once; a sweep's launcher
passes them on to its workers, and a signal to any process of many stops them all at
the same step. The launcher exits with 0 when every run is done, 1 when one failed,
and 128 + the signal's number when a signal stopped them.
"""

import contextlib
import datetime
import json
import math
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
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from bux import parallel
from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.train.config import MeshConfig
from bux.train.hooks import Hook, Metrics
from bux.train.sinks import Latest, primary
from bux.train.trainer import Trainer, directory, train

SIGNALS = (signal.SIGTERM, signal.SIGINT)
# A signal sooner than this after the first is the same one, delivered twice: to this
# process and to its parent, `uv run` or `srun` say, which passes it on.
REPEAT = 1.0  # seconds
# What says how many processes train a run together, the first of them set: the user,
# Slurm's srun, Open MPI's mpirun.
PROCESSES = ("JAX_NUM_PROCESSES", "SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE")


def status(folder: Path) -> dict[str, Any]:
    """What the run in `folder` last said of itself in its status.json, if anything."""
    path = folder / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record(folder: Path, **fields: Any) -> None:
    """Update the run's status.json with `fields`, all at once; of a job of many
    processes, the first does."""
    if not primary():
        return
    tmp = folder / "status.json.tmp"
    tmp.write_text(json.dumps(status(folder) | fields, indent=2), encoding="utf-8")
    tmp.replace(folder / "status.json")


def _print(line: str) -> None:
    print(line, flush=True)  # now, though stdout is a pipe, to a container's log say


def _quiet(line: str) -> None:
    """What a process of many but the first says: nothing."""


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def processes(env: Mapping[str, str] = os.environ) -> int:
    """How many processes train a run together, as `env` says: 1 alone."""
    return next((int(env[v]) for v in PROCESSES if v in env), 1)


def join() -> None:
    """Join the other processes of a job through jax.distributed, if there are any:
    before anything touches a device."""
    if processes() == 1 or jax.distributed.is_initialized():
        return
    if "JAX_NUM_PROCESSES" in os.environ:  # and JAX_COORDINATOR_ADDRESS, jax reads it
        jax.distributed.initialize(
            num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
            process_id=int(os.environ["JAX_PROCESS_ID"]),
        )
    else:
        jax.distributed.initialize()  # as Slurm or Open MPI says


def _largest(signum: int) -> Int[Array, ""]:
    """The largest of every process's `signum`, the same in all of them, once the
    devices get to it."""
    grid = Mesh(np.array(jax.devices()), ("devices",))
    split = NamedSharding(grid, P("devices"))
    local = np.full(jax.local_device_count(), signum, np.int32)  # one a device
    said = jax.make_array_from_process_local_data(split, local)
    with jax.set_mesh(grid):
        return jax.jit(jnp.max, out_shardings=NamedSharding(grid, P()))(said)


class Stop(Hook):
    """Stops training at SIGTERM or SIGINT, after the step it is at; at a second
    signal, or at one before training starts, at once, with a KeyboardInterrupt. Of a
    job of many processes, a signal to any stops them all after the same step: the
    one after they all hear what each said at the step before."""

    def __init__(self) -> None:
        self.trainer: Trainer | None = None
        self.signal: int | None = None  # the first one
        self.since = 0.0
        self.many = False  # whether it is a job of many processes
        self.said: Int[Array, ""] | None = None  # by every process, a step ago

    def due(self, step: int, steps: int) -> bool:
        return self.many

    def start(self, trainer: Trainer) -> None:
        self.trainer, self.many = trainer, jax.process_count() > 1

    def __call__(self, trainer: Trainer, metrics: Metrics) -> None:
        # A step late, the devices have it by now: agreeing keeps a step in flight.
        said, self.said = self.said, _largest(self.signal or 0)
        if said is not None and (signum := int(said)):
            if self.signal is None:  # another process's
                self.signal, self.since = signum, time.monotonic()
            trainer.stopping = True

    def handle(self, signum: int, frame: FrameType | None) -> None:
        if self.signal is None:
            self.signal, self.since = signum, time.monotonic()
            if self.trainer is None:
                raise KeyboardInterrupt
            if not self.many:  # many stop at the step they agree on
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


def _devices(run: ir.Run, gpus: list[str] | None) -> str:
    """What `run` trains on: 8 × NVIDIA H100 (GPU 0,...,7) of a host, or of 2
    processes."""
    devices = jax.devices()
    n = math.prod(parallel.sizes(run.train.mesh, len(devices)).values())
    kind = devices[0].device_kind + (f" (GPU {','.join(gpus)})" if gpus else "")
    hosts = jax.process_count()
    where = f"{hosts} processes" if hosts > 1 else socket.gethostname()
    return f"{f'{n} × ' if n > 1 else ''}{kind} of {where}"


def _outcome(run: ir.Run, stop: Stop, error: str | None) -> tuple[str, int, str]:
    """How the run ended: its state, the exit code and what to say."""
    step = None if stop.trainer is None else stop.trainer.step
    at = "before training" if step is None else f"at step {step}"
    if error is not None:
        return "failed", 1, f"failed {at}: {error}"
    if stop.signal is not None and step != run.train.steps:
        again = (
            "resumes it from its latest checkpoint"
            if run.train.checkpoint is not None
            else "starts it over, as it has no checkpoint setting"
        )
        said = f"stopped {at} by {_name(stop.signal)}; launching it again {again}"
        return "stopped", 128 + stop.signal, said
    return "done", 0, "done"


def train_here(
    run: ir.Run,
    out: str | Path = "runs",
    cache: str | Path = DEFAULT_CACHE,
    *,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train `run` in this process, as `train` does, and record how it went in its
    status.json; the exit code. Of a job of many processes, which it joins, the first
    says and records it."""
    join()
    echo = echo if primary() else _quiet
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    gpus = _listed() or None
    record(
        folder,
        state="running",
        host=socket.gethostname(),
        pid=os.getpid(),
        gpus=gpus,
        started=_now(),
        ended=None,
        error=None,
    )
    stop, latest, error = Stop(), Latest(), None
    try:
        with stop.armed():
            on = _devices(run, gpus)
            echo(f"{run.name}: training on {on} with jax {jax.__version__}")
            train(run, out, cache=cache, echo=echo, hooks=[stop], sinks=[latest])
    except KeyboardInterrupt:
        stop.signal = stop.signal or signal.SIGINT
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        error = f"{type(err).__name__}: {err}"
    state, code, said = _outcome(run, stop, error)
    fields: dict[str, Any] = {"state": state, "ended": _now(), "error": error}
    if stop.trainer is not None:
        fields["step"] = stop.trainer.step
    if latest.metrics:
        fields["metrics"] = latest.metrics
    record(folder, **fields)
    echo(f"{run.name}: {said}")
    return code


def _listed() -> list[str] | None:
    """The GPUs CUDA_VISIBLE_DEVICES lists, if it is set."""
    if (listed := os.environ.get("CUDA_VISIBLE_DEVICES")) is None:
        return None
    return [g for g in listed.split(",") if g.strip() not in ("", "-1")]


def visible_gpus() -> list[str]:
    """The GPUs to give workers: those CUDA_VISIBLE_DEVICES lists, if it is set, or
    else every one that nvidia-smi finds."""
    if (listed := _listed()) is not None:
        return listed
    try:
        found = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        return []
    return [str(i) for i in range(sum(g.startswith("GPU ") for g in found.split("\n")))]


def need(c: MeshConfig, gpus: int, *, alone: bool) -> int:
    """The GPUs a run of mesh `c` trains on, of `gpus` there are: the product of its
    axes' sizes, where `data` unset is every GPU for a run alone, and one in a sweep."""
    rest = math.prod(getattr(c, a) for a in parallel.AXES[1:])
    if c.data is not None:
        return c.data * rest
    return max(gpus, rest) if alone else rest


def fit[S](needs: Sequence[int], free: Sequence[S]) -> tuple[int, list[S]] | None:
    """The first of the runs that need `needs` GPUs to fit in the `free` ones, and the
    group it gets: the first of them. None if none fits."""
    for i, n in enumerate(needs):
        if n <= len(free):
            return i, list(free[:n])
    return None


def _spawn(
    run: ir.Run, out: str | Path, cache: str | Path, gpus: list[str]
) -> subprocess.Popen[str]:
    """A worker that trains `run` on `gpus`, or on the default device if there are
    none, from its run.json: what the launcher compiled, whatever its files say by
    now."""
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    record(
        folder,
        state="running",
        gpus=gpus or None,
        started=_now(),
        ended=None,
        error=None,
    )
    env = os.environ | ({"CUDA_VISIBLE_DEVICES": ",".join(gpus)} if gpus else {})
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


def _ended(
    run: ir.Run, out: str | Path, code: int, signals: Sequence[int]
) -> str | None:
    """Record how the worker that trained `run` ended, with `code`, if it died before
    it could say; what to say if it failed."""
    folder = directory(run, out)
    how = _name(-code) if code < 0 else f"exit {code}"
    if status(folder).get("state") == "running":  # it died before it could say
        stopped = bool(signals) and code in (-signals[0], 128 + signals[0])
        state, error = ("stopped", None) if stopped else ("failed", how)
        record(folder, state=state, ended=_now(), error=error)
    if status(folder)["state"] != "failed":
        return None
    return f"{run.name}: failed ({how}); its output is in {folder / 'train.log'}"


def supervise(
    runs: Sequence[ir.Run],
    out: str | Path,
    cache: str | Path,
    gpus: Sequence[str],
    *,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train `runs` in workers, each on a group of free `gpus`, as many as its mesh
    needs and no more than there are, the first pending run that fits starting first;
    or one at a time on the default device if there are no GPUs. The exit code."""
    slots: list[str | None] = [*gpus] or [None]  # None: the default device
    needs = [need(r.train.mesh, len(gpus), alone=False) if gpus else 1 for r in runs]
    pending, free = list(zip(runs, needs)), list(slots)
    active: dict[subprocess.Popen[str], tuple[ir.Run, list[str | None]]] = {}
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
            while not signals and (found := fit([n for _, n in pending], free)):
                i, group = found
                run = pending.pop(i)[0]
                free = [s for s in free if s not in group]
                worker = _spawn(run, out, cache, [g for g in group if g is not None])
                active[worker] = run, group
                if signals:  # that came as it started
                    worker.send_signal(signals[0])
                log = directory(run, out) / "train.log"
                threading.Thread(target=relay, args=(worker, log), daemon=True).start()
            if not active:
                break
            worker = ended.get()
            run, group = active.pop(worker)
            free = [s for s in slots if s in free or s in group]
            if said := _ended(run, out, worker.returncode, signals):
                failed = True
                say(said)
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


def _fitting(
    runs: Sequence[ir.Run],
    out: str | Path,
    gpus: Sequence[str],
    echo: Callable[[str], None],
) -> list[ir.Run]:
    """The runs that need no more of `gpus` than there are; the others fail."""
    fits = []
    for run in runs:
        n = need(run.train.mesh, len(gpus), alone=len(runs) == 1)
        if gpus and n > len(gpus):
            error = f"mesh: needs {n} GPUs, but there are {len(gpus)}"
            folder = directory(run, out)
            folder.mkdir(parents=True, exist_ok=True)
            record(folder, state="failed", started=_now(), ended=_now(), error=error)
            echo(f"{run.name}: failed before training: {error}")
        else:
            fits.append(run)
    return fits


def _prepare(
    runs: Sequence[ir.Run],
    cache: str | Path,
    workers: int | None,
    echo: Callable[[str], None],
) -> int | None:
    """Prepare the datasets `runs` read, once each; the exit code if a signal stops
    it."""
    stop = Stop()
    try:
        with stop.armed():
            for job in dict.fromkeys(job for run in runs for job in jobs(run.train)):
                prepare(job, cache, workers=workers, echo=echo)
    except KeyboardInterrupt:
        return 128 + (stop.signal or signal.SIGINT)
    return None


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
    in this process, on as many of `gpus` (default: every visible one) as it needs,
    or with others on many hosts if the environment says so; a sweep in workers,
    each on a group of them."""
    todo = [run for run in runs if status(directory(run, out)).get("state") != "done"]
    if done := len(runs) - len(todo):
        echo(f"{done} of {len(runs)} run{'s' if len(runs) != 1 else ''} done already")
    if not todo:
        return 0
    if (many := processes()) > 1:
        if len(todo) > 1:
            echo(f"{many} processes train one run, not {len(todo)}: use --index")
            return 1
        join()
        echo = echo if primary() else _quiet
    # Of many processes, each has the GPUs jax.distributed gives it.
    there = [] if many > 1 else list(visible_gpus() if gpus is None else gpus)
    if not (fits := _fitting(todo, out, there, echo)):
        return 1
    if (code := _prepare(fits, cache, workers, echo)) is not None:
        return code
    if len(todo) == 1:
        if there:
            n = need(todo[0].train.mesh, len(there), alone=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(there[:n])
        return train_here(todo[0], out, cache, echo=echo)
    on = f" on GPUs {','.join(there)}" if there else ", one at a time"
    echo(f"training {len(fits)} runs{on}")
    code = supervise(fits, out, cache, there, echo=echo)
    echo(summary(runs, out))
    return code or int(len(fits) < len(todo))


if __name__ == "__main__":  # a sweep's worker: python -m bux.launch FOLDER CACHE
    folder, cache = Path(sys.argv[1]), Path(sys.argv[2])
    run = ir.Run.model_validate_json((folder / "run.json").read_text(encoding="utf-8"))
    sys.exit(train_here(run, folder.parent, cache))
