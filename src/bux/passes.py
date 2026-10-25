"""bux's passes over XLA's HLO: a registry of them, each a function of a serialized
HloModuleProto that returns it changed, or None to leave it, at a stage of XLA's
pipeline, before its scheduler or after it::

    @passes.register("my_pass", stage=passes.AFTER_SCHEDULING)
    def changed(serialized: bytes) -> bytes | None: ...

    passes.use("my_pass")  # for every program the process compiles for GPUs from now

A pass that raises leaves the module as it was, with a warning: it runs inside XLA's
compile, which it must not fail. JAX's cache of compiled programs is keyed by the
passes in use, and by each one's `key`, which should change with what it does.

`bux.jax_internals.hlo_module` reads a serialized module into one that a pass can
walk and change (its schedule, say), and ``as_serialized_hlo_module_proto()`` gives
it back. `bux.train.reorder` is such a pass, after the scheduler, and
`bux.parallel.nccl_ep` registers `bux.jax_internals.parts_off_the_stream`, before it.
"""

import functools
import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax
from jax.extend import xla as jax_xla

from bux import jax_internals

#: A pass, from a serialized HloModuleProto to the module changed, or None to leave it.
Pass = Callable[[bytes], bytes | None]
#: The stage of XLA's pipeline just before its scheduler.
BEFORE_SCHEDULING = jax_xla.PipelineStage.PRE_SCHEDULER
#: The stage just after XLA's scheduler, once the module's schedule is set.
AFTER_SCHEDULING = jax_xla.PipelineStage.POST_SCHEDULER


class Registered(NamedTuple):
    """A registered pass: `run`, guarded so that an exception leaves the module, with
    a warning; the `stage` of XLA's pipeline it runs at; and the `key` that JAX's
    cache keys programs by, with the pass's name."""

    run: Pass
    stage: jax_xla.PipelineStage
    key: str


#: Every registered pass, by name.
#:
#: :meta hide-value:
PASSES: dict[str, Registered] = {}
_in_use: set[str] = set()


def register(
    name: str, *, stage: jax_xla.PipelineStage, key: str = ""
) -> Callable[[Pass], Pass]:
    """Return a decorator that registers a pass by `name`, run at `stage`.

    Registering runs nothing: `use` turns a pass on.

    Args:
        name: The pass's name in `PASSES`, which `use` takes.
        stage: `BEFORE_SCHEDULING` or `AFTER_SCHEDULING`.
        key: Keys JAX's cache of compiled programs, with `name`, while the pass is
            in use: change it whenever what the pass does changes, so that no
            program compiled by the pass before is loaded.

    Returns:
        A decorator that registers the function it decorates and returns it as it
        is, so the function can still be called and tested alone.

    Example::

        @passes.register("my_pass", stage=passes.AFTER_SCHEDULING, key="1")
        def my_pass(serialized: bytes) -> bytes | None:
            module = jax_internals.hlo_module(serialized)
            ...  # change the module's schedule
            return module.as_serialized_hlo_module_proto()
    """

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
    on, or no longer; elsewhere, nothing.

    It registers the pass with XLA (``jax.extend.xla``) for CUDA and keys JAX's
    cache of compiled programs by the passes in use (`cache_key`). Calling it again
    with the same `on` does nothing. Off a GPU it does nothing at all, so a run's
    code can call it on any backend.

    Args:
        name: A pass of `PASSES`.
        on: Whether to run it from now on, or to stop.

    Raises:
        KeyError: On a GPU, if no pass is registered by `name`.

    Example::

        from bux import passes
        from bux.train import reorder

        passes.use(reorder.NAME)  # every program compiled from here is reordered
    """
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
    """Return the passes in use, and their keys, as JAX's cache keys programs by
    them: ``"name:key"``, separated by spaces, in order of name."""
    return " ".join(sorted(f"{n}:{PASSES[n].key}" for n in _in_use))
