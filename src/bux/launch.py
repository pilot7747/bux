"""The launcher, ``bux launch``: train an experiment, a .bux file of one run or a
sweep of many, to its end, on a cluster or on a laptop::

    bux launch examples/experiments/attention_ablation.bux            # all 6 runs
    bux launch examples/experiments/attention_ablation.bux --index 3  # the fourth

``--index`` trains one run of the sweep, as each task of an array job does
(``--index $SLURM_ARRAY_TASK_ID``). From Python, `launch` takes the runs that
`bux.lang.compile_runs` returns.

A run needs the GPUs of its mesh: the product of its axes' sizes, where an unset
`data` is every visible GPU for a run alone and one in a sweep; a run that needs more
than there are fails at once. The launcher prepares the datasets the other runs read,
once. A run alone then trains in this process. A sweep trains one worker process per
run, each on a group of free GPUs (``CUDA_VISIBLE_DEVICES``), the first pending run
that fits first; without GPUs, one at a time on the default device. A worker's output
goes to the console and to its run's train.log, and a run that fails or runs out of
memory leaves the others training.

A run whose ``profile`` setting names nsys or ncu trains under that tool
(`bux.train.profilers`): a sweep's worker starts under it, and a run alone's process
starts again under it, with the same command, before it does anything else.

A run alone trains on many hosts when the environment says so: Slurm's srun, Open
MPI's mpirun, or ``JAX_COORDINATOR_ADDRESS``, ``JAX_NUM_PROCESSES`` and
``JAX_PROCESS_ID``. Its processes join through jax.distributed (`join`); the first
writes the console, status.json and the run's files, and all of them write
checkpoints. They keep JAX's cache only where it is one folder for all of them
(`bux.runtime.agree_on_cache`).

A run's directory (`bux.train.trainer`) also keeps a status.json: whether the run is
running, done, failed or stopped, where, on which GPUs, since when, at what step, and
its latest metrics. A run that is done is not trained again, and one that is not
resumes from its latest checkpoint, so launching an experiment again, after a
preemption, a crash or a fix, trains what is left of it.

SIGTERM or SIGINT stops training after the current step, saving a checkpoint if the
run has a checkpoint setting; a second signal, a second or more later, stops it at
once (`Stop`). A sweep's launcher passes signals on to its workers, and a signal to
any process of a run stops them all at the same step. The launcher exits with 0 when
every run is done, 1 when one failed, and 128 + the signal's number when a signal
stopped them.
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
from bux.runtime import agree_on_cache, configure
from bux.train import profilers
from bux.train.config import MeshConfig
from bux.train.hooks import Hook, Metrics
from bux.train.sinks import LatestMetrics, is_main_process
from bux.train.trainer import Trainer, directory, train

#: The signals that stop training after the current step.
SIGNALS = (signal.SIGTERM, signal.SIGINT)
#: Seconds within which a second signal is the first delivered twice, and ignored.
#:
#: A signal goes to this process and to its parent (``uv run`` or ``srun``, say),
#: which passes it on.
REPEAT = 1.0
#: The variables that give the number of processes training one run.
#:
#: The first set wins: the user's, Slurm's srun's, Open MPI's mpirun's.
PROCESSES = ("JAX_NUM_PROCESSES", "SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE")


def status(folder: Path) -> dict[str, Any]:
    """Return the run's status.json in `folder`, or {} if there is none.

    Its fields: ``state`` (``running``, ``done``, ``failed`` or ``stopped``),
    ``host``, ``pid``, ``gpus``, ``started`` and ``ended`` (UTC, ISO 8601),
    ``error``, ``step`` and the latest ``metrics``.
    """
    path = folder / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record(folder: Path, **fields: Any) -> None:
    """Update the run's status.json with `fields`, all at once. In a job of many
    processes, only the first writes it."""
    if not is_main_process():
        return
    tmp = folder / "status.json.tmp"
    tmp.write_text(json.dumps(status(folder) | fields, indent=2), encoding="utf-8")
    tmp.replace(folder / "status.json")


def _print(line: str) -> None:
    print(line, flush=True)  # now, even if stdout is a pipe, to a container's log say


def _quiet(line: str) -> None:
    """Print nothing: the echo of every process of many but the first."""


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def processes(env: Mapping[str, str] = os.environ) -> int:
    """Return how many processes train one run together, as `env` says: 1 alone."""
    return next((int(env[v]) for v in PROCESSES if v in env), 1)


def join() -> None:
    """Join the job's other processes through jax.distributed, if there are any, and
    keep JAX's cache only if it is one for all of them (`bux.runtime.agree_on_cache`).

    With ``JAX_NUM_PROCESSES`` set, the variables say all there is: its count,
    ``JAX_PROCESS_ID`` and ``JAX_COORDINATOR_ADDRESS`` (which JAX reads itself), with
    no detection of a cluster. Otherwise JAX reads Slurm's or Open MPI's environment.
    Nothing happens for a process alone, or one that has joined already.

    Call it before anything touches a device.
    """
    if processes() == 1 or jax.distributed.is_initialized():
        return
    if "JAX_NUM_PROCESSES" in os.environ:  # jax reads JAX_COORDINATOR_ADDRESS itself
        # The variables say all there is: no detection of a cluster, which on
        # Kubernetes asks its API, or warns that the client to ask it is missing.
        jax.distributed.initialize(
            num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
            process_id=int(os.environ["JAX_PROCESS_ID"]),
            cluster_detection_method="deactivate",
        )
    else:
        jax.distributed.initialize()  # from Slurm's or Open MPI's environment
    agree_on_cache()


def _largest(signum: int) -> Int[Array, ""]:
    """Return the largest `signum` of all processes, the same in each, once the
    devices compute it."""
    grid = Mesh(np.array(jax.devices()), ("devices",))
    sharding = NamedSharding(grid, P("devices"))
    local = np.full(jax.local_device_count(), signum, np.int32)  # one per device
    signums = jax.make_array_from_process_local_data(sharding, local)
    with jax.set_mesh(grid):
        return jax.jit(jnp.max, out_shardings=NamedSharding(grid, P()))(signums)


class Stop(Hook):
    """Stop training after the current step at SIGTERM or SIGINT.

    A second signal, a second or more after the first (`REPEAT`), or one before
    training starts, stops it at once with a KeyboardInterrupt. A signal to any
    process of many stops them all after the same step, once they have all seen it:
    every step, the processes take the largest signal of any of them on the devices.

    Example::

        stop = Stop()
        with stop.armed():
            train(run, hooks=[stop])
        stop.signal  # the signal that stopped it, or None

    Attributes:
        trainer: The trainer it stops, once training starts.
        signal: The first signal, this process's or another's; None until then.
        since: When the first signal came, by `time.monotonic`.
        multiprocess: Whether other processes train the run too.
        agreed: Every process's signal, the step before, as the devices hold it.
    """

    def __init__(self) -> None:
        self.trainer: Trainer | None = None
        self.signal: int | None = None
        self.since = 0.0
        self.multiprocess = False
        self.agreed: Int[Array, ""] | None = None

    def due(self, step: int, steps: int) -> bool:
        """Run every step in a job of many processes, to agree on a stop; never for
        a process alone, which its handler stops."""
        return self.multiprocess

    def start(self, trainer: Trainer) -> None:
        """Take the trainer to stop, as training starts."""
        self.trainer, self.multiprocess = trainer, jax.process_count() > 1

    def __call__(self, trainer: Trainer, metrics: Metrics) -> None:
        """After a step of many processes: stop if any of them had a signal by the
        step before, whose largest signal the devices have computed by now."""
        # Read last step's result, ready by now, so as not to stall the step in flight.
        agreed, self.agreed = self.agreed, _largest(self.signal or 0)
        if agreed is not None and (signum := int(agreed)):
            if self.signal is None:  # another process's
                self.signal, self.since = signum, time.monotonic()
            trainer.stopping = True

    def handle(self, signum: int, frame: FrameType | None) -> None:
        """The handler of `SIGNALS`: the first stops training after the step (for a
        process of many, at the step they agree on), a later one at once."""
        if self.signal is None:
            self.signal, self.since = signum, time.monotonic()
            if self.trainer is None:
                raise KeyboardInterrupt
            if not self.multiprocess:  # many stop at the step they agree on
                self.trainer.stopping = True
        elif time.monotonic() - self.since >= REPEAT:
            raise KeyboardInterrupt

    @contextlib.contextmanager
    def armed(self) -> Generator["Stop"]:
        """Handle SIGTERM and SIGINT inside the context."""
        previous = {s: signal.signal(s, self.handle) for s in SIGNALS}
        try:
            yield self
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def _devices(run: ir.Run, gpus: list[str] | None) -> str:
    """Describe what `run` trains on: 8 × NVIDIA H100 (GPU 0,...,7) of a host, or of
    2 processes."""
    devices = jax.devices()
    n = math.prod(parallel.sizes(run.train.mesh, len(devices)).values())
    kind = devices[0].device_kind + (f" (GPU {','.join(gpus)})" if gpus else "")
    hosts = jax.process_count()
    where = f"{hosts} processes" if hosts > 1 else socket.gethostname()
    return f"{f'{n} × ' if n > 1 else ''}{kind} of {where}"


def _outcome(run: ir.Run, stop: Stop, error: str | None) -> tuple[str, int, str]:
    """Return how the run ended: its state, the exit code and a message."""
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
        message = f"stopped {at} by {_name(stop.signal)}; launching it again {again}"
        return "stopped", 128 + stop.signal, message
    return "done", 0, "done"


def train_here(
    run: ir.Run,
    out: str | Path = "runs",
    cache: str | Path = DEFAULT_CACHE,
    *,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train `run` in this process, as `~bux.train.trainer.train` does, record how
    it went in its status.json, and return the exit code.

    It joins the job's other processes first, if there are any (`join`), and then
    only the first prints and records. A run alone of `launch` trains so, and so
    does each worker of a sweep.

    Args:
        run: The run.
        out: Where run directories go.
        cache: Where prepared datasets are.
        echo: Where progress lines go.

    Returns:
        0 when the run is done, 1 when it failed, 128 + the signal's number when a
        signal stopped it.
    """
    join()
    echo = echo if is_main_process() else _quiet
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
    stop, latest, error = Stop(), LatestMetrics(), None
    try:
        with stop.armed():
            devices = _devices(run, gpus)
            echo(f"{run.name}: training on {devices} with jax {jax.__version__}")
            train(run, out, cache=cache, echo=echo, hooks=[stop], sinks=[latest])
    except KeyboardInterrupt:
        stop.signal = stop.signal or signal.SIGINT
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        error = f"{type(err).__name__}: {err}"
    state, code, message = _outcome(run, stop, error)
    fields: dict[str, Any] = {"state": state, "ended": _now(), "error": error}
    if stop.trainer is not None:
        fields["step"] = stop.trainer.step
    if latest.metrics:
        fields["metrics"] = latest.metrics
    record(folder, **fields)
    echo(f"{run.name}: {message}")
    return code


def _listed() -> list[str] | None:
    """Return the GPUs that CUDA_VISIBLE_DEVICES lists, or None if it is not set."""
    if (listed := os.environ.get("CUDA_VISIBLE_DEVICES")) is None:
        return None
    return [g for g in listed.split(",") if g.strip() not in ("", "-1")]


def visible_gpus() -> list[str]:
    """Return the GPUs to give workers: those CUDA_VISIBLE_DEVICES lists if it is
    set, else every one that nvidia-smi finds."""
    if (listed := _listed()) is not None:
        return listed
    try:
        found = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        return []
    return [str(i) for i in range(sum(g.startswith("GPU ") for g in found.split("\n")))]


def need(config: MeshConfig, gpus: int, *, alone: bool) -> int:
    """Return how many of `gpus` GPUs a run with mesh `config` uses: the product of
    its axes' sizes, where an unset `data` is every GPU for a run alone, one in a sweep.
    """
    rest = math.prod(getattr(config, a) for a in parallel.AXES if a != "data")
    if config.data is not None:
        return config.data * rest
    return max(gpus, rest) if alone else rest


def fit[S](needs: Sequence[int], free: Sequence[S]) -> tuple[int, list[S]] | None:
    """Return the first run, of those that need `needs` GPUs, that fits in the `free`
    ones, and the group it gets: the first free ones. None if none fits."""
    for i, n in enumerate(needs):
        if n <= len(free):
            return i, list(free[:n])
    return None


def _spawn(
    run: ir.Run, out: str | Path, cache: str | Path, gpus: list[str]
) -> subprocess.Popen[str]:
    """Start a worker that trains `run` on `gpus`, or on the default device if there
    are none, from its run.json: the run as the launcher compiled it, whatever the
    files say by now. A run with nsys or ncu trains under it."""
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
    worker = [sys.executable, "-u", "-m", "bux.launch", str(folder), str(cache)]
    return subprocess.Popen(
        [*_profiler(run, out), *worker],
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
    """Record how the worker that trained `run` ended, with exit `code`, if it died
    before it could. Return an error message if the run failed."""
    folder = directory(run, out)
    exit_status = _name(-code) if code < 0 else f"exit {code}"
    if status(folder).get("state") == "running":  # it died before it could record
        stopped = bool(signals) and code in (-signals[0], 128 + signals[0])
        state, error = ("stopped", None) if stopped else ("failed", exit_status)
        record(folder, state=state, ended=_now(), error=error)
    if status(folder)["state"] != "failed":
        return None
    log = folder / "train.log"
    return f"{run.name}: failed ({exit_status}); its output is in {log}"


def supervise(
    runs: Sequence[ir.Run],
    out: str | Path,
    cache: str | Path,
    gpus: Sequence[str],
    *,
    echo: Callable[[str], None] = _print,
) -> int:
    """Train `runs` in workers and return the exit code.

    Each worker gets a group of free `gpus`, as many as its mesh needs, and the first
    pending run that fits starts first. Without GPUs, runs train one at a time on the
    default device. A worker is ``python -m bux.launch FOLDER CACHE``, which trains
    the run.json written in its run's directory, with ``CUDA_VISIBLE_DEVICES`` set to
    its group; its output goes to `echo` and to the run's train.log. SIGTERM and
    SIGINT are passed on to every worker, and no new one starts after them.

    Args:
        runs: The runs to train.
        out: Where run directories go.
        cache: Where prepared datasets are.
        gpus: The GPUs to share out, as ``CUDA_VISIBLE_DEVICES`` names them.
        echo: Where the workers' output and the launcher's lines go.

    Returns:
        0 when every run is done, 1 when one failed, 128 + the signal's number when a
        signal stopped them.
    """
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
                if signals:  # a signal that came as it started
                    worker.send_signal(signals[0])
                log = directory(run, out) / "train.log"
                threading.Thread(target=relay, args=(worker, log), daemon=True).start()
            if not active:
                break
            worker = ended.get()
            run, group = active.pop(worker)
            free = [s for s in slots if s in free or s in group]
            if message := _ended(run, out, worker.returncode, signals):
                failed = True
                say(message)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    return 128 + signals[0] if signals else int(failed)


def summary(runs: Sequence[ir.Run], out: str | Path) -> str:
    """Return one line per run: its state, its step, and its latest losses.

    A sweep's launch ends with it::

        3 runs: 2 done, 1 stopped
        state    step       loss    eval loss  run
        done     3814/3814  3.2141  3.2577     width_sweep[s]
        done     3814/3814  3.0836  3.1302     width_sweep[m]
        stopped  2731/3814  3.0412  3.0815     width_sweep[l]
    """
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


def _unfit(run: ir.Run, gpus: Sequence[str], *, alone: bool) -> str | None:
    """Return why `run` cannot train on `gpus`: it needs more, or its profiler cannot
    profile it; None if it can."""
    n = need(run.train.mesh, len(gpus), alone=alone)
    if gpus and n > len(gpus):
        return f"mesh: needs {n} GPUs, but there are {len(gpus)}"
    return profilers.unfit(run.train.profile, n if gpus else 1)


def _fitting(
    runs: Sequence[ir.Run],
    out: str | Path,
    gpus: Sequence[str],
    echo: Callable[[str], None],
) -> list[ir.Run]:
    """Return the runs that can train on `gpus` (`_unfit`); record the others failed."""
    fits = []
    for run in runs:
        if (error := _unfit(run, gpus, alone=len(runs) == 1)) is None:
            fits.append(run)
            continue
        folder = directory(run, out)
        folder.mkdir(parents=True, exist_ok=True)
        record(folder, state="failed", started=_now(), ended=_now(), error=error)
        echo(f"{run.name}: failed before training: {error}")
    return fits


def _profiler(run: ir.Run, out: str | Path) -> list[str]:
    """Return the command that runs `run`'s process under its profiler, if it has one
    the process does not run under yet (`bux.train.profilers.command`), and make the
    folder its report goes to."""
    tool = profilers.command(run.train.profile, folder := directory(run, out))
    if tool:
        (folder / "profile").mkdir(parents=True, exist_ok=True)
    return tool


def _rerun_under_profiler(run: ir.Run, out: str | Path, gpus: Sequence[str]) -> None:
    """Start this process again, the same command, under the profiler of `run`, a run
    alone on `gpus`, if it can train there and the process is not under it yet."""
    if _unfit(run, gpus, alone=True) is None and (tool := _profiler(run, out)):
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvp(tool[0], [*tool, sys.executable, *sys.orig_argv[1:]])


def _prepare(
    runs: Sequence[ir.Run],
    cache: str | Path,
    workers: int | None,
    echo: Callable[[str], None],
) -> int | None:
    """Prepare the datasets `runs` read, once each. Return the exit code if a signal
    stops it, else None."""
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
    """Train what is left of `runs` under `out`, and return the exit code.

    This is ``bux launch``. Runs whose status.json says ``done`` are skipped, and
    the others resume from their latest checkpoints. A run alone whose ``profile``
    names nsys or ncu starts this process again under it first. A run that needs more
    GPUs than there are, or a profiler that is not installed, is recorded as failed
    at once. Then the datasets the rest read are prepared under `cache`, once each,
    with `workers` processes. A run alone trains in this process, on as many of
    `gpus` as it needs, or on many hosts if the environment says so (`processes`); a
    sweep trains in workers (`supervise`), each on a group of the GPUs, and ends with
    a table of its runs (`summary`).

    Args:
        runs: The runs of an experiment, as `bux.lang.compile_runs` returns them.
        out: Where run directories go.
        cache: Where prepared datasets go.
        gpus: The GPUs to train on, as ``CUDA_VISIBLE_DEVICES`` names them; None
            for every visible one (`visible_gpus`).
        workers: Processes that prepare data: None for one a core.
        echo: Where progress lines go.

    Returns:
        0 when every run is done, 1 when one failed (or many processes were given
        a sweep), and 128 + the signal's number when a signal stopped them.

    Example::

        from bux.lang import compile_runs
        from bux.launch import launch

        source = open("examples/experiments/attention_ablation.bux").read()
        runs = compile_runs(source, filename="attention_ablation.bux")
        code = launch(runs, "runs")
    """
    todo = [run for run in runs if status(directory(run, out)).get("state") != "done"]
    # In a job of many processes, each has the GPUs jax.distributed gives it.
    nproc = processes()
    available = [] if nproc > 1 else list(visible_gpus() if gpus is None else gpus)
    if len(todo) == 1:  # before anything starts a backend
        _rerun_under_profiler(todo[0], out, available)
    if done := len(runs) - len(todo):
        echo(f"{done} of {len(runs)} run{'s' if len(runs) != 1 else ''} done already")
    if not todo:
        return 0
    if nproc > 1:
        if len(todo) > 1:
            echo(f"{nproc} processes train one run, not {len(todo)}: use --index")
            return 1
        join()
        echo = echo if is_main_process() else _quiet
    if not (fits := _fitting(todo, out, available, echo)):
        return 1
    if (code := _prepare(fits, cache, workers, echo)) is not None:
        return code
    if len(todo) == 1:
        if available:
            n = need(todo[0].train.mesh, len(available), alone=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available[:n])
        return train_here(todo[0], out, cache, echo=echo)
    where = f" on GPUs {','.join(available)}" if available else ", one at a time"
    echo(f"training {len(fits)} runs{where}")
    code = supervise(fits, out, cache, available, echo=echo)
    echo(summary(runs, out))
    return code or int(len(fits) < len(todo))


if __name__ == "__main__":  # a sweep's worker: python -m bux.launch FOLDER CACHE
    configure()
    folder, cache = Path(sys.argv[1]), Path(sys.argv[2])
    run = ir.Run.model_validate_json((folder / "run.json").read_text(encoding="utf-8"))
    sys.exit(train_here(run, folder.parent, cache))
