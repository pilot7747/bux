"""bux's passes over XLA's HLO: a registry of them, each a function of a serialized
HloModuleProto that returns it changed, or None to leave it, at a stage of XLA's
pipeline, before its scheduler or after it.

    @passes.register("my_pass", stage=passes.AFTER_SCHEDULING)
    def changed(serialized: bytes) -> bytes | None: ...

    passes.use("my_pass")  # for every program the process compiles for GPUs from now

A pass that raises leaves the module as it was, with a warning: it runs inside XLA's
compile, which it must not fail. JAX's cache of compiled programs is keyed by the
passes in use, and by each one's `key`, which should change with what it does.
"""

import functools
import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax
from jax.extend import xla as jax_xla

from bux import jax_internals

Pass = Callable[[bytes], bytes | None]
BEFORE_SCHEDULING = jax_xla.PipelineStage.PRE_SCHEDULER
AFTER_SCHEDULING = jax_xla.PipelineStage.POST_SCHEDULER


class Registered(NamedTuple):
    run: Pass
    stage: jax_xla.PipelineStage
    key: str


PASSES: dict[str, Registered] = {}
_in_use: set[str] = set()


def register(
    name: str, *, stage: jax_xla.PipelineStage, key: str = ""
) -> Callable[[Pass], Pass]:
    """Return a decorator that registers a pass by `name`, run at `stage`."""

    def add(run: Pass) -> Pass:
        PASSES[name] = Registered(guarded(name, run), stage, key)
        return run

    return add


def guarded(name: str, run: Pass) -> Pass:
    """Return `run`, which on an exception leaves the module, with a warning."""

    @functools.wraps(run)
    def safely(serialized: bytes) -> bytes | None:
        try:
            return run(serialized)
        except Exception as e:  # noqa: BLE001 (XLA's compile must go on)
            warnings.warn(f"bux's pass {name} failed, and left XLA's HLO: {e!r}")
            return None

    return safely


def use(name: str, on: bool = True) -> None:
    """Run the pass `name` on every program the process compiles for GPUs from now
    on, or no longer; elsewhere, nothing."""
    if (name in _in_use) == on or jax.default_backend() != "gpu":
        return
    found = PASSES[name]
    where = {"name": name, "stage": found.stage, "platforms": "cuda"}
    if on:
        jax_xla.register_hlo_module_transformation(found.run, **where)
        _in_use.add(name)
    else:
        jax_xla.clear_hlo_module_transformation(**where)
        _in_use.discard(name)
    jax_internals.key_compiles_by(cache_key())


def cache_key() -> str:
    """The passes in use, and their keys, as JAX's cache keys programs by them."""
    return " ".join(sorted(f"{n}:{PASSES[n].key}" for n in _in_use))
