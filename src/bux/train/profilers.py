"""The profilers of the ``profile`` setting: jax.profiler, which runs in the process,
and the tools a process runs under, Nsight Systems (``nsys``) and Nsight Compute
(``ncu``):

.. code-block:: bux

    profile = profile(start=20, steps=5)             # jax.profiler
    profile = profile(start=20, steps=5, tool=nsys)  # Nsight Systems

``bux launch`` runs the process that trains a run under its tool's command
(`command`): a sweep's worker, or a run alone's own process, which it starts again
under it. The tool captures nothing until the `~bux.train.hooks.Profile` hook starts
it after step ``start``, by ``cudaProfilerStart``, and stops it after ``steps``
steps, by ``cudaProfilerStop``. Its report goes to the run's ``profile/``. A
process under a tool sets ``BUX_PROFILER`` (`VARIABLE`), from which
`bux.runtime.configure` takes the tool's `XLA_FLAGS` and `VARIABLES`.

Another tool is an entry of `TOOLS`, which ``profile(tool=...)`` takes in a script
that adds it before it launches::

    nsys = profilers.TOOLS["nsys"]
    profilers.TOOLS["nsys_memory"] = [*nsys, "--cuda-memory-usage=true"]
"""

import ctypes
import functools
import os
import shlex
import shutil
from pathlib import Path

import jax

from bux.train.config import ProfileConfig

#: Set, to the tool's name, in a process that runs under its run's tool already, where
#: `command` gives nothing. Set it yourself to run under a command of your own.
VARIABLE = "BUX_PROFILER"

#: The command each tool runs a process under, by name; ``{out}`` is the report's path
#: without its suffix. Each captures from ``cudaProfilerStart`` to ``cudaProfilerStop``.
TOOLS: dict[str, list[str]] = {
    # A timeline of the GPUs' streams, CUDA's calls and NVTX's ranges; a CUDA graph's
    # kernels one by one. Training goes on after the capture.
    "nsys": [
        "nsys",
        "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-graph-trace=node",
        "--trace=cuda,nvtx,cublas,cudnn",
        "--force-overwrite=true",
        "--output={out}",
    ],
    # Each kernel's counters, from replays of it: many times slower than the kernel.
    "ncu": [
        "ncu",
        "--profile-from-start=off",
        "--target-processes=all",
        "--force-overwrite",
        "--export={out}",
    ],
}

#: XLA's flags in a process under each tool, which `bux.runtime.configure` puts before
#: those of ``XLA_FLAGS``. XLA's command buffers (CUDA graphs) are off, as jax.profiler
#: turns them off as it traces: on 8 GPUs, both tools crashed the process in the CUDA
#: calls that make and free them.
XLA_FLAGS: dict[str, tuple[str, ...]] = {
    "nsys": ("--xla_gpu_enable_command_buffer=",),
    "ncu": ("--xla_gpu_enable_command_buffer=",),
}

#: The variables a process under each tool sets, where the environment does not
#: (`bux.runtime.configure`). Under ncu, XLA takes memory as it needs it, not 95% of
#: the GPU at once: ncu saves and restores all of it before each pass of a replay.
VARIABLES: dict[str, dict[str, str]] = {
    "ncu": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
}

#: The tools that profile a process of one GPU. ncu runs one kernel at a time, which a
#: step's collectives over many GPUs cannot: on 8 B300s it hung at a step of FSDP, and
#: failed to save the memory of a data-parallel one for its replay.
ONE_GPU = {"ncu"}


def command(config: ProfileConfig | None, folder: Path) -> list[str]:
    """Return the command that runs a process of a run under its profiler, to put
    before the process's own.

    Args:
        config: The run's ``profile`` setting.
        folder: The run's directory, whose ``profile/`` takes the report: a file a
            process, named by its host and its process id.

    Returns:
        The tool's command, with the setting's ``options``, which sets `VARIABLE`;
        nothing for jax.profiler or no setting, or in a process under its tool.

    Raises:
        ValueError: If the tool is neither jax nor in `TOOLS`.
    """
    if config is None or config.tool == "jax" or VARIABLE in os.environ:
        return []
    if config.tool not in TOOLS:
        raise ValueError(unfit(config))
    out = folder / "profile" / "%h.%p"
    tool = [arg.format(out=out) for arg in TOOLS[config.tool]]
    return ["env", f"{VARIABLE}={config.tool}", *tool, *shlex.split(config.options)]


def unfit(config: ProfileConfig | None, gpus: int = 1) -> str | None:
    """Return why a run's tool cannot profile a process of `gpus` GPUs: it is unknown,
    not on PATH, or for one GPU alone (`ONE_GPU`); None if it can, or there is none."""
    if config is None or config.tool == "jax":
        return None
    if config.tool not in TOOLS:
        return f"profile: no tool {config.tool!r}; there are jax, {', '.join(TOOLS)}"
    if shutil.which(program := TOOLS[config.tool][0]) is None:
        return f"profile: {program} is not on PATH"
    if config.tool in ONE_GPU and gpus > 1:
        return f"profile: {config.tool} profiles one GPU, not {gpus}: give the run one"
    return None


def start(tool: str, folder: Path) -> None:
    """Start tracing: jax.profiler's trace into `folder`, or the capture of the tool
    the process runs under."""
    if tool == "jax":
        jax.profiler.start_trace(folder)
    else:
        _capture("cudaProfilerStart")


def stop(tool: str) -> None:
    """Stop tracing, as `start` started it."""
    if tool == "jax":
        jax.profiler.stop_trace()
    else:
        _capture("cudaProfilerStop")


def _capture(call: str) -> None:
    """Call the CUDA runtime's `call`, which starts or stops a tool's capture; off
    GPUs there is nothing to capture."""
    if jax.default_backend() != "gpu":
        return
    if code := getattr(_cudart(), call)():
        raise RuntimeError(f"{call} failed with CUDA error {code}")


@functools.cache
def _cudart() -> ctypes.CDLL:
    """Return the CUDA runtime that jax's GPU plugin loaded."""
    maps = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    if not (found := [line.split()[-1] for line in maps if "/libcudart.so" in line]):
        raise RuntimeError("profile: the process has no CUDA runtime (libcudart)")
    return ctypes.CDLL(found[0])
