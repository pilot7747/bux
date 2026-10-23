"""What a profiler's trace says of a step's communication: how long the collectives
take, and how long they run with no compute beside them, which the step waits for.

A GPU's kernels are on the trace's lines of streams ("Stream #13(compute)"). A kernel
communicates if NCCL runs it, or if XLA's op that launched it (its `hlo_op`) is a
collective; CUDA's copies and memsets are neither; every other kernel computes. A
collective is exposed where no computing kernel overlaps it, on any of its GPU's
streams. Its time includes the wait for its slowest peer.
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
    """When a device's kernels ran: those that communicate, and those that compute."""

    comm: Intervals
    compute: Intervals


def comm_ms(file: Path, steps: int) -> dict[str, float]:
    """Return the milliseconds a step of the `steps` traced in `file` spends in
    collectives (`comm_ms`) and in collectives alone (`exposed_comm_ms`), the mean of
    its GPUs; nothing if it traced no GPU."""
    devices = gpu_kernels(ProfileData.from_file(str(file)))
    if not devices:
        return {}
    comm = [length(merged(d.comm)) for d in devices]
    alone = [exposed(d) for d in devices]
    per_step = 1e-6 / (steps * len(devices))
    return {"comm_ms": sum(comm) * per_step, "exposed_comm_ms": sum(alone) * per_step}


def latest(folder: Path) -> Path | None:
    """Return the newest trace under `folder`, which jax.profiler writes."""
    traces = sorted(folder.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


def gpu_kernels(data: ProfileData) -> list[Kernels]:
    """Return the kernels of every GPU in `data`."""
    gpus = [p for p in data.planes if p.name.startswith("/device:GPU:")]
    return [_kernels(plane) for plane in gpus]


def _kernels(plane: Any) -> Kernels:
    found = Kernels([], [])
    for line in plane.lines:
        if not line.name.startswith("Stream"):
            continue
        for event in line.events:
            op = next((v for k, v in event.stats if k == "hlo_op"), "")
            kind = _kind(event.name, op)
            if kind is not None:
                getattr(found, kind).append((event.start_ns, event.end_ns))
    return found


def _kind(name: str, op: object) -> str | None:
    """Whether a kernel communicates or computes, by its name and its XLA op."""
    name = name.lower()
    if _COLLECTIVE.search(name) or _COLLECTIVE.search(str(op).lower()):
        return "comm"
    return None if _COPY.search(name) else "compute"


def exposed(kernels: Kernels) -> float:
    """Return the time collectives ran with no computing kernel beside them."""
    comm, compute = merged(kernels.comm), merged(kernels.compute)
    return length(comm) - overlap(comm, compute)


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
