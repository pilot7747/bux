"""What bux sets for JAX before it trains: XLA's flags, and a cache of compiled steps.

`bux train`, `bux launch` and a sweep's workers call ``configure`` before anything
starts a backend. A flag that XLA_FLAGS sets already keeps its value. Compiled
programs, and on GPUs the autotuner's picks, persist in $XDG_CACHE_HOME/bux/jax
(~/.cache/bux/jax), so a launch after the first compiles little; a
JAX_COMPILATION_CACHE_DIR that is set says where instead, and one set empty turns the
cache off.
"""

import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

import jax

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
