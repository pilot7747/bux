"""`methods.remat = auto`: a training step that chooses its remat policy as it first
compiles.

It compiles the step under remat's candidates, the fastest first
(``remat.CANDIDATES``), and keeps the first that fits: what XLA's analysis of the
compiled step says a device holds at once, its arguments, outputs and temporaries,
against a share of the device's memory (``MARGIN``), for what the analysis misses.
In that order, the first that fits is the one that computes least again. The first
compiles alone: where it fits, as on most devices, it is the only compile. Where it
does not, the others all compile side by side, so that every process of a run
compiles the same programs, whose autotuning XLA splits among them. Each compile
goes into JAX's cache, so a run that starts again reads them back, and the step that
fits runs as ``xla.jit`` made it, which compiles it no more.
"""

import functools
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager as ContextManager
from contextlib import nullcontext
from typing import Any, NamedTuple

import jax

from bux import parallel
from bux.architecture.repeats import remat
from bux.train import xla

#: Of a device's memory a step may take, by XLA's analysis: on B300s it was the peak
#: to 0.1 GiB, and steps of 98% of a device trained.
MARGIN = 0.97


class Tried(NamedTuple):
    """A candidate the step compiled under: the bytes it holds on a device, None if
    XLA could not fit it, and the seconds XLA took to compile it."""

    policy: str
    bytes: int | None
    seconds: float


class Fitting:
    """A training step under remat `auto`. Its first call compiles it under the
    candidates until one fits, and runs that one, `policy`; `tried` says what each
    took, and `seconds` how long choosing took."""

    def __init__(self, step: Callable[..., Any], options: xla.Options, **jit: Any):
        self.step, self.options, self.jit_kwargs = step, options, jit
        self.chosen: Any = None  # the step under the policy that fits, of xla.jit
        self.policy: str | None = None
        self.tried: list[Tried] = []
        self.seconds = 0.0

    def __call__(self, *args: Any) -> Any:
        """Run the step on `args`, under the candidate that fits, which the first call
        chooses."""
        return self._fitted(args)(*args)

    def lower(self, *args: Any) -> Any:
        """Return the step lowered for `args`, under the candidate that fits."""
        return self._fitted(args).lower(*args)

    def _fitted(self, args: tuple) -> Any:
        """Return the step under the candidate that fits, which the first call of
        `args` chooses."""
        if self.chosen is None:
            start = time.perf_counter()
            self.policy, self.chosen = self._first_that_fits(args)
            self.seconds = time.perf_counter() - start
            _report(self.policy, self.tried, self.seconds)
        return self.chosen

    def _first_that_fits(self, args: tuple) -> tuple[str, Any]:
        """Return the first candidate whose step, compiled for `args`, fits, and that
        step; else the last candidate's. The first compiles alone, then the rest side
        by side."""
        first, *rest = remat.CANDIDATES
        for policies in ([first], rest):
            steps = self._compiled(policies, args)
            for tried in self.tried[-len(policies) :]:
                if fits(tried.bytes):
                    return tried.policy, steps[tried.policy]
        return rest[-1], steps[rest[-1]]

    def _compiled(self, policies: Sequence[str], args: tuple) -> dict[str, Any]:
        """Return the step under each of `policies`, compiled for `args`, and note in
        `tried` what each holds and how long XLA took. Tracing holds Python's lock,
        so they lower one at a time; XLA lets go of it, so they compile side by side,
        a thread each."""
        steps = {
            policy: xla.jit(self._trying(policy), self.options, **self.jit_kwargs)
            for policy in policies
        }
        lowered = [steps[policy].lower(*args) for policy in policies]
        held_here = functools.partial(self._held, here=_settings_in_force())
        with ThreadPoolExecutor(len(policies)) as threads:
            self.tried += threads.map(held_here, policies, lowered)
        return steps

    def _held(
        self, policy: str, lowered: Any, here: Callable[[], ContextManager]
    ) -> Tried:
        """Return what the step under `policy`, `lowered`, holds compiled, in a thread
        that puts the caller's settings in force again (`here`)."""
        start = time.perf_counter()
        with here():
            holds = held(lowered)
        return Tried(policy, holds, time.perf_counter() - start)

    def _trying(self, policy: str) -> Callable[..., Any]:
        """Return the step, traced under `policy`, by the step's own name, which
        profiles and JAX's cache show."""

        @functools.wraps(self.step)
        def step(*args: Any) -> Any:
            with remat.trying(policy):
                return self.step(*args)

        return step


def _settings_in_force() -> Callable[[], ContextManager]:
    """Return what puts the run's settings in force here again, in another thread.
    JAX keys its cache of compiles by the mesh in force, so a step's call finds a
    compile only if it ran under the same one."""
    on = parallel.active()
    if on is None:
        return nullcontext
    return lambda: parallel.use(on.mesh, on.layout, on.methods)


def held(lowered: Any) -> int | None:
    """Return the bytes a step of ``xla.jit``, `lowered`, holds on a device at once
    compiled, by XLA's analysis: its arguments, its outputs but those that take an
    argument's memory, its temporaries and its code. None if XLA could not fit it."""
    try:
        m = lowered.compile().memory_analysis()
    except jax.errors.JaxRuntimeError as e:
        if "RESOURCE_EXHAUSTED" not in str(e):
            raise
        return None
    outputs = m.output_size_in_bytes - m.alias_size_in_bytes
    return (
        m.argument_size_in_bytes
        + outputs
        + m.temp_size_in_bytes
        + m.generated_code_size_in_bytes
    )


def capacity() -> int | None:
    """Return the bytes a device can hold; None where it does not say, as a CPU."""
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("bytes_limit")


def fits(size: int | None) -> bool:
    """Return whether a step that holds `size` bytes on a device (``held``) fits in
    its memory, with a margin; anything fits where the device does not say how much it
    can hold."""
    limit = capacity()
    if limit is None:
        return True
    return size is not None and size <= MARGIN * limit


def _report(policy: str, tried: list[Tried], seconds: float) -> None:
    """Say, on the first process, which policy `auto` chose, what each candidate
    took, and how long choosing took."""
    if jax.process_index() != 0:
        return
    limit = capacity()
    each = ", ".join(f"{t.policy} {_gib(t.bytes)} in {t.seconds:.0f}s" for t in tried)
    room = "" if limit is None else f"; {MARGIN:.0%} of {_gib(limit)} fits"
    print(f"remat=auto: {policy} in {seconds:.0f}s ({each}{room})", flush=True)


def _gib(n: int | None) -> str:
    return "too much" if n is None else f"{n / 2**30:.1f} GiB"
