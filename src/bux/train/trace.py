"""What a profiler's trace says of a step's communication: how long the collectives
take, how long they run with no compute beside them, which the step waits for, and
how long its GPUs run nothing at all.

A GPU's kernels are on the trace's lines of streams ("Stream #13(compute)"). A kernel
communicates if NCCL runs it, or if XLA's op that launched it (its `hlo_op`) is a
collective; CUDA's copies and memsets are neither; every other kernel computes. A
collective is exposed where no computing kernel overlaps it, on any of its GPU's
streams. Its time includes the wait for its slowest peer. A GPU is idle where none of
its streams runs anything, from its first kernel to its last: mostly the wait for the
host to launch a step, after a step the host waited for.
"""

import re
from pathlib import Path
from typing import Any, NamedTuple

from jax.profiler import ProfileData

Intervals = list[tuple[float, float]]  # (start, end) in ns

_COLLECTIVE = re.compile(
    r"nccl|all[-_]gather|all[-_]reduce|reduce[-_]scatter|all[-_]to[-_]all|"
    r"collective[-_]permute|send|recv|memcpy ?p(2|to)p"
)
_COPY = re.compile(r"memcpy ?[hd](2|to)[hd]|memset")  # CUDA's, not XLA's copy kernels


class Kernels(NamedTuple):
    """When a device's kernels ran: those that communicate, those that compute, and
    CUDA's copies."""

    comm: Intervals
    compute: Intervals
    copy: Intervals


def comm_ms(file: Path, steps: int) -> dict[str, float]:
    """Return the milliseconds a step of the `steps` traced in `file` spends in
    collectives (`comm_ms`), in collectives alone (`exposed_comm_ms`) and running
    nothing (`idle_ms`), the mean of its GPUs; nothing if it traced no GPU."""
    devices = gpu_kernels(ProfileData.from_file(str(file)))
    if not devices:
        return {}
    comm = [length(merged(d.comm)) for d in devices]
    alone = [exposed(d) for d in devices]
    nothing = [idle(d) for d in devices]
    per_step = 1e-6 / (steps * len(devices))
    return {
        "comm_ms": sum(comm) * per_step,
        "exposed_comm_ms": sum(alone) * per_step,
        "idle_ms": sum(nothing) * per_step,
    }


def latest(folder: Path, host: str | None = None) -> Path | None:
    """Return the newest trace under `folder`, which jax.profiler writes as
    ``plugins/profile/<time>/<host>.xplane.pb``: `host`'s, or any host's for None.
    Every host of a job writes its own into the run's directory, which they share."""
    pattern = f"{'*' if host is None else host}.xplane.pb"
    traces = sorted(folder.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


def gpu_kernels(data: ProfileData) -> list[Kernels]:
    """Return the kernels of every GPU in `data`."""
    gpus = [p for p in data.planes if p.name.startswith("/device:GPU:")]
    return [_kernels(plane) for plane in gpus]


def _kernels(plane: Any) -> Kernels:
    found = Kernels([], [], [])
    for line in plane.lines:
        if not line.name.startswith("Stream"):
            continue
        for event in line.events:
            op = next((v for k, v in event.stats if k == "hlo_op"), "")
            kind = _kind(event.name, op)
            getattr(found, kind).append((event.start_ns, event.end_ns))
    return found


def _kind(name: str, op: object) -> str:
    """Whether a kernel communicates, copies or computes, by its name and XLA op."""
    name = name.lower()
    if _COLLECTIVE.search(name) or _COLLECTIVE.search(str(op).lower()):
        return "comm"
    return "copy" if _COPY.search(name) else "compute"


def exposed(kernels: Kernels) -> float:
    """Return the time collectives ran with no computing kernel beside them."""
    comm, compute = merged(kernels.comm), merged(kernels.compute)
    return length(comm) - overlap(comm, compute)


def idle(kernels: Kernels) -> float:
    """Return the time from a device's first kernel to its last that it ran none."""
    busy = merged([*kernels.comm, *kernels.compute, *kernels.copy])
    return busy[-1][1] - busy[0][0] - length(busy) if busy else 0.0


def merged(intervals: Intervals) -> Intervals:
    """Return the disjoint intervals, in order, that cover what `intervals` cover."""
    out: Intervals = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def length(intervals: Intervals) -> float:
    """Return the total time of disjoint `intervals`."""
    return sum(end - start for start, end in intervals)


def overlap(a: Intervals, b: Intervals) -> float:
    """Return the time two lists of disjoint, ordered intervals share."""
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total
