"""Benchmark a suite of runs, and compare two trees' results.

    python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out new.json
    python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out old.json --tree ../old
    python scripts/benchmark.py compare old.json new.json

`run` trains each run of a suite, a .bux file of named alternatives, in a worker
process of its own, on as many of the visible GPUs as its mesh needs: one where the
run sets no mesh, so the runs of a one-GPU suite train side by side. Of each run it
keeps the median tokens a second and MFU of the logs after the first, whose time
includes compiling, the peak memory of its busiest GPU and every logged loss, in the
JSON file `--out`; the runs' directories and their workers' output go in a folder
beside it. `--tree PATH` trains with the bux of another checkout, PATH/src, in this
one's Python environment. On many hosts (JAX_NUM_PROCESSES, Slurm or Open MPI) every
host runs the same command: the runs train one at a time on all of them, joined as
`bux launch` joins them, and the first host writes the results.

`compare` prints a table of the runs of two results files: tokens a second, old and
new, and their difference; peak memory; and the change of the loss at the last log,
each flagged past the noise. It exits with 1 if a run got slower, took more memory,
moved its loss or failed.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.experimental import multihost_utils

import bux
from bux.lang import Run, compile_runs
from bux.launch import fit, join, need, processes, visible_gpus
from bux.runtime import configure
from bux.train.sinks import Scalars, Sink, is_main_process
from bux.train.trainer import directory, train

Result = dict[str, Any]
# The variables that give a host's index among those that train a run together, as
# bux.launch.PROCESSES gives their number. Here, not in bux.launch: with --tree this
# script runs on another checkout's bux, an older one too, which has no such helper.
RANKS = ("JAX_PROCESS_ID", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK")
REGRESSIONS = {"slower", "more memory", "loss", "failed"}


def suite_runs(suite: str, settings: list[str]) -> list[Run]:
    """Return the runs of the suite in file `suite`, `settings` written at its end."""
    source = Path(suite).read_text(encoding="utf-8")
    return compile_runs(source, filename=suite, overrides=settings)


# --- A worker: one run, measured --------------------------------------------------


class Logs(Sink):
    """The scalars of every log, with its step: the writes with tokens_per_s."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def write(self, step: int, scalars: Scalars) -> None:
        if "tokens_per_s" in scalars:
            self.rows.append({"step": step, **scalars})


def peak_gib() -> float | None:
    """Return the peak memory of the busiest GPU of all hosts, in GiB; None where the
    devices do not say, as a CPU does not."""
    stats = [device.memory_stats() or {} for device in jax.local_devices()]
    here = max(s.get("peak_bytes_in_use", 0) for s in stats) / 2**30
    # In GiB, a float: JAX would gather bytes as an int32, which overflows at 2 GiB.
    everywhere = multihost_utils.process_allgather(np.float32(here))
    return float(np.max(everywhere)) or None


def measure(rows: list[dict[str, float]], seconds: float) -> Result:
    """Return what the benchmark keeps of a run that logged `rows`."""
    timed = rows[1:] or rows  # the first log's time includes compiling
    mfu = [row["mfu"] for row in timed if "mfu" in row]
    return {
        "tokens_per_s": statistics.median(row["tokens_per_s"] for row in timed),
        "mfu": statistics.median(mfu) if mfu else None,
        "peak_gib": peak_gib(),
        "steps": [int(row["step"]) for row in rows],
        "losses": [row["loss"] for row in rows],
        "seconds": seconds,
        "device": jax.devices()[0].device_kind,
        "bux": str(Path(bux.__file__).parent),
    }


def work(suite: str, index: int, settings: list[str], result: Path) -> None:
    """Train run `index` of `suite` into `result`'s folder, and write what `measure`
    keeps of it to `result`."""
    started = time.perf_counter()
    configure()
    join()
    run = suite_runs(suite, settings)[index]
    logs = Logs()
    train(run, result.parent, sinks=[logs])
    measured = measure(logs.rows, time.perf_counter() - started)
    if is_main_process():
        result.write_text(json.dumps(measured, indent=2), encoding="utf-8")


# --- Running a suite: a worker a run ----------------------------------------------


def rank() -> int:
    """Return this host's index among those that train a run together: 0 alone."""
    return next((int(os.environ[v]) for v in RANKS if v in os.environ), 0)


@dataclass
class Job:
    """Run `index` of the suite, whose worker's output and result go beside the run's
    directory, `stem`."""

    index: int
    run: Run
    stem: Path

    @property
    def log(self) -> Path:
        host = rank()
        return Path(f"{self.stem}.txt" if host == 0 else f"{self.stem}.{host}.txt")

    @property
    def result(self) -> Path:
        return Path(f"{self.stem}.json")


def outcome(job: Job, code: int) -> Result:
    """Return the result `job`'s worker wrote, or why there is none."""
    if job.result.exists():
        return json.loads(job.result.read_text(encoding="utf-8"))
    lines = job.log.read_text(encoding="utf-8").strip().splitlines() or [""]
    return {"error": f"exit {code}: {lines[-1][:300]}", "log": str(job.log)}


def describe(name: str, result: Result) -> str:
    """Return a line on how the run went."""
    if "error" in result:
        return f"{name}: failed, {result['error']}; its output is in {result['log']}"
    mfu = f", mfu {result['mfu']:.1%}" if result["mfu"] is not None else ""
    peak = f", peak {result['peak_gib']:.1f} GiB" if result["peak_gib"] else ""
    speed = f"{result['tokens_per_s']:,.0f} tok/s"
    return f"{name}: {speed}{mfu}{peak}, {result['seconds']:.0f} s"


def coordinator(i: int) -> dict[str, str]:
    """Return the coordinator of the `i`th run on many hosts: JAX_COORDINATOR_ADDRESS's
    port plus `i`, so that no host joins the last run's, which may still be closing."""
    if (address := os.environ.get("JAX_COORDINATOR_ADDRESS")) is None:
        return {}
    host, port = address.rsplit(":", 1)
    return {"JAX_COORDINATOR_ADDRESS": f"{host}:{int(port) + i}"}


def gpus_needed(job: Job, gpus: list[str]) -> int:
    """Return how many of `gpus` the run of `job` trains on: those of its mesh, where
    `data` unset is one; 1, the default device, if there are no GPUs."""
    return need(job.run.train.mesh, len(gpus), alone=False) if gpus else 1


def first_to_end(workers: list[subprocess.Popen]) -> subprocess.Popen:
    """Wait for one of `workers` to end, and return it."""
    while True:
        for worker in workers:
            if worker.poll() is not None:
                return worker
        time.sleep(1)


class Workers:
    """The worker processes that train a suite's runs, and the results they wrote."""

    def __init__(self, args: argparse.Namespace):
        self.suite, self.settings = args.suite, args.settings
        self.env = dict(os.environ)
        if args.tree:  # its bux comes before the one installed
            self.env["PYTHONPATH"] = str(Path(args.tree, "src").resolve())
        self.results: dict[str, Result] = {}

    def start(self, job: Job, more: dict[str, str]) -> subprocess.Popen:
        """Start the worker that trains `job`, with `more` in its environment."""
        job.result.unlink(missing_ok=True)  # a result is this worker's or none
        command = [__file__, "worker", self.suite, str(job.index), str(job.result)]
        with job.log.open("w", encoding="utf-8") as log:
            return subprocess.Popen(
                [sys.executable, *command, *self.settings],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env | more,
            )

    def finish(self, job: Job, code: int) -> None:
        """Keep the result of `job`, whose worker exited with `code`, and say it."""
        if rank() == 0:  # the other hosts' workers write none
            self.results[job.run.name] = outcome(job, code)
            print(describe(job.run.name, self.results[job.run.name]), flush=True)

    def side_by_side(self, jobs: list[Job]) -> None:
        """Train `jobs` on this host's GPUs, each on as many as its mesh needs, as
        many at once as fit, the first that fits first; one at a time on the default
        device if there are no GPUs."""
        gpus = visible_gpus()
        slots: list[str | None] = [*gpus] or [None]  # None: the default device
        pending = [(job, gpus_needed(job, gpus)) for job in jobs]
        free = list(slots)
        active: dict[subprocess.Popen, tuple[Job, list[str | None]]] = {}
        while pending or active:
            while found := fit([n for _, n in pending], free):
                i, group = found
                job = pending.pop(i)[0]
                free = [s for s in free if s not in group]
                visible = ",".join(g for g in group if g is not None)
                more = {"CUDA_VISIBLE_DEVICES": visible} if gpus else {}
                active[self.start(job, more)] = job, group
            if not active:  # what is left needs more GPUs than there are
                for job, n in pending:
                    job.log.write_text(f"needs {n} GPUs, but there are {len(gpus)}\n")
                    self.finish(job, 1)
                return
            worker = first_to_end(list(active))
            job, group = active.pop(worker)
            free = [s for s in slots if s in free or s in group]
            self.finish(job, worker.returncode)

    def one_at_a_time(self, jobs: list[Job]) -> None:
        """Train `jobs` one after another on all the hosts' GPUs, as every host does."""
        for i, job in enumerate(jobs):
            worker = self.start(job, coordinator(i))
            self.finish(job, worker.wait())


def run_suite(args: argparse.Namespace) -> int:
    """Train the runs of `args.suite`, write their results to `args.out`, and return
    1 if one failed, else 0."""
    folder = Path(args.out).with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    runs = suite_runs(args.suite, args.settings)
    jobs = [
        Job(i, run, directory(run, folder))
        for i, run in enumerate(runs)
        if args.only in run.name
    ]
    workers = Workers(args)
    if processes() > 1:
        workers.one_at_a_time(jobs)
    else:
        workers.side_by_side(jobs)
    if rank() != 0:
        return 0
    results = {job.run.name: workers.results[job.run.name] for job in jobs}
    written = {"suite": args.suite, "settings": args.settings, "runs": results}
    Path(args.out).write_text(json.dumps(written, indent=2), encoding="utf-8")
    return int(any("error" in result for result in results.values()))


# --- Comparing two results files ----------------------------------------------------


def load(path: str) -> dict[str, Result]:
    """Return the results of the runs in the results file at `path`, by name."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["runs"]


def change(old: Result, new: Result, key: str) -> float | None:
    """Return how much `key` grew from `old` to `new`, as a share of the old value;
    None if either lacks it."""
    if not old.get(key) or not new.get(key):
        return None
    return new[key] / old[key] - 1


def loss_change(old: Result, new: Result) -> float | None:
    """Return the change of the loss at the last log, if both logged the same steps."""
    if not old.get("steps") or old["steps"] != new.get("steps"):
        return None
    return new["losses"][-1] - old["losses"][-1]


def flags(old: Result, new: Result, noise: float, tolerance: float) -> list[str]:
    """Return what changed past `noise` from `old` to `new`, or a loss past
    `tolerance`."""
    if "error" in new:
        return ["failed"]
    found = []
    if (speed := change(old, new, "tokens_per_s")) and abs(speed) > noise:
        found.append("slower" if speed < 0 else "faster")
    if (memory := change(old, new, "peak_gib")) and abs(memory) > noise:
        found.append("more memory" if memory > 0 else "less memory")
    if (loss := loss_change(old, new)) and abs(loss) > tolerance:
        found.append("loss")
    return found


def shown(value: float | None, form: str) -> str:
    """Return `value` written in `form`, or - if there is none."""
    return "-" if value is None else form.format(value)


def speed(result: Result) -> str:
    """Return the run's tokens a second, written, or that it failed."""
    if "error" in result:
        return "failed"
    return shown(result.get("tokens_per_s"), "{:,.0f}")


def cells(name: str, old: Result, new: Result) -> list[str]:
    """Return the numbers of a run's line in the table."""
    return [
        name,
        speed(old),
        speed(new),
        shown(change(old, new, "tokens_per_s"), "{:+.1%}"),
        shown(new.get("mfu"), "{:.1%}"),
        shown(old.get("peak_gib"), "{:.1f}"),
        shown(new.get("peak_gib"), "{:.1f}"),
        shown(loss_change(old, new), "{:+.4f}"),
    ]


def table(rows: list[list[str]]) -> str:
    """Return `rows` as columns: the first and last to the left, numbers to the right."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(row: list[str]) -> str:
        numbers = [cell.rjust(w) for cell, w in zip(row[1:-1], widths[1:-1])]
        return "  ".join([row[0].ljust(widths[0]), *numbers, row[-1]]).rstrip()

    return "\n".join(map(line, rows))


def compare(args: argparse.Namespace) -> int:
    """Print the table of `args.old` against `args.new`; return 1 if a run regressed."""
    old, new = load(args.old), load(args.new)
    head = ["run", "old tok/s", "new tok/s", "diff", "new MFU", "old GiB", "new GiB"]
    rows, regressed = [[*head, "loss diff", ""]], False
    for name in dict.fromkeys([*old, *new]):
        before, after = old.get(name, {}), new.get(name, {})
        found = flags(before, after, args.noise, args.loss)
        rows.append([*cells(name, before, after), ", ".join(found)])
        regressed |= bool(REGRESSIONS.intersection(found))
    print(table(rows))
    return int(regressed)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train a suite and write its results")
    run.add_argument("suite", help="a .bux file")
    run.add_argument("settings", nargs="*", metavar="SETTING=VALUE")
    run.add_argument("--out", required=True, help="the results file, a .json")
    run.add_argument("--tree", help="train with the bux of this checkout")
    run.add_argument("--only", default="", help="only the runs whose name has this")
    pair = commands.add_parser("compare", help="print the table of two results files")
    pair.add_argument("old")
    pair.add_argument("new")
    pair.add_argument("--noise", type=float, default=0.03, help="of tok/s and memory")
    pair.add_argument("--loss", type=float, default=0.05, help="the last loss's noise")
    worker = commands.add_parser("worker", help="train one run, as run does")
    worker.add_argument("suite")
    worker.add_argument("index", type=int)
    worker.add_argument("result", type=Path)
    worker.add_argument("settings", nargs="*")
    return parser


def main() -> int:
    args = parser().parse_args()
    match args.command:
        case "run":
            return run_suite(args)
        case "compare":
            return compare(args)
        case "worker":
            work(args.suite, args.index, args.settings, args.result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
