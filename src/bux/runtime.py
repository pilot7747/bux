"""What bux sets for JAX before it trains: XLA's flags, NCCL EP's settings, the share
of a GPU XLA's pool takes, and a cache of compiled steps.

``bux train``, ``bux launch`` and a sweep's workers call `configure` before anything
starts a backend; a script that trains calls it first too::

    from bux.runtime import configure

    configure()  # before anything touches a device
    train(run)

What it sets, each only where the environment does not already:

- ``XLA_FLAGS``: bux's flags (`XLA_FLAGS`) go before the ones given, and a flag
  given already keeps its value.
- ``NCCL_GIN_TYPE`` and ``NCCL_EP_TOKENS_PER_CHUNK``, for NCCL EP (`NCCL_DEFAULTS`).
- ``XLA_CLIENT_MEM_FRACTION``, the share of a GPU's memory that XLA's pool takes:
  95% (`POOL`), unless it or its older name ``XLA_PYTHON_CLIENT_MEM_FRACTION`` is
  set.
- Under nsys or ncu (``BUX_PROFILER``), the tool's flags and variables
  (`bux.train.profilers.XLA_FLAGS`, `bux.train.profilers.VARIABLES`).
- ``JAX_COMPILATION_CACHE_DIR``: compiled programs, and on GPUs the autotuner's
  picks, persist in ``$XDG_CACHE_HOME/bux/jax`` (``~/.cache/bux/jax``), so a launch
  after the first compiles little. A ``JAX_COMPILATION_CACHE_DIR`` that is set says
  where instead, and one set empty turns the cache off.

Workers inherit them all. In a job of many processes, the processes must all find a
program in the cache, or all compile it (`agree_on_cache`), and only the first writes
it: so the cache is kept only where it is one folder for all of them, on a filesystem
every host mounts.
"""

import os
import secrets
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

import jax

from bux import jax_internals
from bux.train import profilers

#: bux's XLA flags, which go before those of ``XLA_FLAGS``.
#:
#: GPU flags, which XLA takes on any backend and a CPU ignores. Without the first,
#: ``ragged_all_to_all`` (expert parallelism's dispatch) hangs on B300s.
XLA_FLAGS = ("--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false",)
#: NCCL EP's settings (``methods(moe=nccl_ep)``), where the environment sets none.
#:
#: GIN by a CPU proxy, and 256 tokens a chunk between nodes. On two B300 nodes,
#: GDAKI (its default) took 3x as long for an EP layer's exchanges, and chunks of
#: 64 1.3x.
NCCL_DEFAULTS = {"NCCL_GIN_TYPE": "2", "NCCL_EP_TOKENS_PER_CHUNK": "256"}
#: How long, in milliseconds, a process waits for the others to reach the same point.
TIMEOUT_MS = 10 * 60 * 1000
#: The share of a GPU's memory XLA's pool takes where the environment sets none.
#:
#: JAX's default is 75%. Beside the pool, the CUDA context, NCCL's buffers and NCCL
#: EP's took up to 9.8 GiB of a B300 (EP over two nodes), which 95% leaves room for
#: (14.3 GiB) and 97% does not.
POOL = "0.95"
#: jaxlib's two names for the pool's share; the second is its old one.
POOL_VARIABLES = ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")


def _name(flag: str) -> str:
    return flag.split("=")[0]


def xla_flags(given: str, defaults: Sequence[str] = XLA_FLAGS) -> str:
    """Return the flags `given`, after those of `defaults` that they do not set.

    Example::

        xla_flags("--xla_gpu_autotune_level=0")
        # '--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
        #  --xla_gpu_autotune_level=0'
    """
    named = {_name(flag) for flag in given.split()}
    ours = [flag for flag in defaults if _name(flag) not in named]
    return " ".join([*ours, given]).strip()


def cache_dir(env: Mapping[str, str] = os.environ) -> Path:
    """Return where compiled programs persist by default: ``bux/jax`` under
    ``$XDG_CACHE_HOME``, or under ``~/.cache`` if it is not set."""
    return Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "bux" / "jax"


def configure(env: MutableMapping[str, str] = os.environ) -> None:
    """Set JAX's and XLA's environment for training, where it is not set already.

    It adds bux's flags to ``XLA_FLAGS`` in `env`, NCCL EP's settings, the share of a
    GPU XLA's pool takes and, under nsys or ncu, the tool's settings, and keeps
    compiled programs in ``JAX_COMPILATION_CACHE_DIR``: `cache_dir` unless it is set,
    and none if it is set empty. Workers inherit them all. Call it before anything starts a backend:
    XLA reads its flags and its pool's share once.

    Args:
        env: The environment to change: this process's by default.
    """
    tool = env.get(profilers.VARIABLE, "")
    flags = (*XLA_FLAGS, *profilers.XLA_FLAGS.get(tool, ()))
    env["XLA_FLAGS"] = xla_flags(env.get("XLA_FLAGS", ""), flags)
    for name, value in {**NCCL_DEFAULTS, **profilers.VARIABLES.get(tool, {})}.items():
        env.setdefault(name, value)
    env.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir(env)))
    jax.config.update(
        "jax_compilation_cache_dir", env["JAX_COMPILATION_CACHE_DIR"] or None
    )
    if not any(name in env for name in POOL_VARIABLES):
        env[POOL_VARIABLES[0]] = POOL


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
