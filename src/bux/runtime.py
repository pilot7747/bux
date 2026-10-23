"""What bux sets for JAX before it trains: XLA's flags, and a cache of compiled steps.

`bux train`, `bux launch` and a sweep's workers call ``configure`` before anything
starts a backend. A flag that XLA_FLAGS sets already keeps its value. Compiled
programs, and on GPUs the autotuner's picks, persist in $XDG_CACHE_HOME/bux/jax
(~/.cache/bux/jax), so a launch after the first compiles little; a
JAX_COMPILATION_CACHE_DIR that is set says where instead, and one set empty turns the
cache off.

In a job of many processes only the first writes the cache, so the others read it
only where it is one directory for all of them, on a filesystem every host mounts.
There ``compile_once`` has the first process compile each program and send the others
its binary.
"""

import os
import secrets
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

import jax
import numpy as np
from jax.experimental import multihost_utils

# GPU flags, which XLA takes on any backend and a CPU ignores.
XLA_FLAGS = (
    # Without it, ragged_all_to_all (expert parallelism's dispatch) hangs on B300s.
    "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false",
)


def _name(flag: str) -> str:
    return flag.split("=")[0]


def xla_flags(given: str, defaults: Sequence[str] = XLA_FLAGS) -> str:
    """Return the flags `given`, after those of `defaults` that they do not set."""
    named = {_name(flag) for flag in given.split()}
    ours = [flag for flag in defaults if _name(flag) not in named]
    return " ".join([*ours, given]).strip()


def cache_dir(env: Mapping[str, str] = os.environ) -> Path:
    """Return where compiled programs persist by default."""
    return Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "bux" / "jax"


def configure(env: MutableMapping[str, str] = os.environ) -> None:
    """Add bux's flags to XLA_FLAGS in `env`, and keep compiled programs in
    JAX_COMPILATION_CACHE_DIR, `cache_dir` unless it is set. Workers inherit both."""
    env["XLA_FLAGS"] = xla_flags(env.get("XLA_FLAGS", ""))
    env.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir(env)))
    jax.config.update(
        "jax_compilation_cache_dir", env["JAX_COMPILATION_CACHE_DIR"] or None
    )


def compile_once() -> None:
    """In a job of many processes, have the first compile each program and send the
    others its binary (jax_share_binary_between_hosts), if they all read one cache, or
    none. Call it once they have joined, before they compile anything.

    It sends a binary only for a program it did not find in its cache, so where each
    host has a cache of its own, the others would wait for a binary that never comes:
    there each host compiles for itself. So does a CPU, whose processes key the same
    program apart."""
    if jax.process_count() == 1 or jax.default_backend() != "gpu":
        return
    folder = jax.config.jax_compilation_cache_dir
    one = not folder or "://" in folder or shared(Path(folder))
    jax.config.update("jax_share_binary_between_hosts", one)
    if not one and jax.process_index() == 0:
        warnings.warn(
            f"JAX's cache in {folder} is this host's alone, so every other host "
            "compiles every program at every launch: a JAX_COMPILATION_CACHE_DIR on a "
            "filesystem every host mounts saves that",
            stacklevel=2,
        )


def shared(folder: Path) -> bool:
    """Whether `folder` is the same directory for every process of the job: a file
    that the first makes there, the others see. Every process calls it."""
    token = multihost_utils.broadcast_one_to_all(np.int32(secrets.randbits(31)))
    marker = f".bux-{int(token)}"
    first = jax.process_index() == 0
    if first:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / marker).touch()
    multihost_utils.sync_global_devices("bux: the cache's marker is made")
    # A listing, not a lookup: NFS may cache that a name it looked up was missing.
    seen = marker in (os.listdir(folder) if folder.is_dir() else ())
    every = multihost_utils.process_allgather(np.int32(seen))
    if first:
        (folder / marker).unlink()
    return bool(every.all())
