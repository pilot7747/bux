"""What bux sets for JAX before it trains: XLA's flags, and a cache of compiled steps.

`bux train`, `bux launch` and a sweep's workers call ``configure`` before anything
starts a backend. A flag that XLA_FLAGS sets already keeps its value. Compiled
programs, and on GPUs the autotuner's picks, persist in $XDG_CACHE_HOME/bux/jax
(~/.cache/bux/jax), so a launch after the first compiles little; a
JAX_COMPILATION_CACHE_DIR that is set says where instead, and one set empty turns the
cache off.

In a job of many processes, the processes must all find a program in the cache, or
all compile it (``agree_on_cache``), and only the first writes it: so the cache is
kept only where it is one folder for all of them, on a filesystem every host mounts.
"""

import os
import secrets
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

import jax

from bux import jax_internals

# GPU flags, which XLA takes on any backend and a CPU ignores.
XLA_FLAGS = (
    # Without it, ragged_all_to_all (expert parallelism's dispatch) hangs on B300s.
    "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false",
)
TIMEOUT_MS = 10 * 60 * 1000  # for the other processes to reach the same point


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


def agree_on_cache() -> None:
    """In a job of many processes, keep JAX's cache only if it is one folder for all
    of them; else turn it off. Call it once they have joined, before they compile.

    On GPUs XLA's autotuner splits its work among the processes that compile a program
    and waits for every share: a process that compiles a program as another loads it
    from its cache waits a day. Only the first process writes the cache, so where each
    host has one of its own, the first would load what the others compile."""
    folder = jax.config.jax_compilation_cache_dir
    me, n = jax_internals.processes()
    if n == 1 or not folder or "://" in folder:  # gs:// and the like: one for all
        return
    if not shared(Path(folder)):
        jax.config.update("jax_compilation_cache_dir", None)
        if me == 0:
            warnings.warn(
                f"JAX's cache in {folder} is this host's alone, so it is off, and "
                "every launch compiles every program: a JAX_COMPILATION_CACHE_DIR on "
                "a filesystem every host mounts keeps it",
                stacklevel=2,
            )


def shared(folder: Path) -> bool:
    """Whether `folder` is the same directory for every process of the job: a file
    that the first makes there, the others see. Every process calls it; they talk
    through jax.distributed's key-value store, so that nothing compiles."""
    client = jax_internals.key_value_store()
    me, n = jax_internals.processes()
    assert client is not None, "the processes have not joined"
    if me == 0:
        made = f".bux-{secrets.token_hex(8)}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / made).touch()
        client.key_value_set("bux/cache/marker", made)
    marker = client.blocking_key_value_get("bux/cache/marker", TIMEOUT_MS)
    # A listing, not a lookup: NFS may cache that a name it looked up was missing.
    seen = folder.is_dir() and marker in os.listdir(folder)
    client.key_value_set(f"bux/cache/seen/{me}", str(int(seen)))
    every = [
        client.blocking_key_value_get(f"bux/cache/seen/{i}", TIMEOUT_MS) == "1"
        for i in range(n)
    ]
    if me == 0:
        (folder / marker).unlink()
    return all(every)
