"""`methods.remat = auto`: a training step that chooses its remat policy as it first
compiles.

It compiles the step under each of remat's candidates in turn, the fastest first
(``remat.CANDIDATES``), and keeps the first that fits: what XLA's analysis of the
compiled step says a device holds at once, its arguments, outputs and temporaries,
against a share of the device's memory (``MARGIN``), for what the analysis misses.
In that order, the first that fits is the one that computes least again. Each
compile goes into JAX's cache, so a run that starts again reads them back.
"""

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
from flax import nnx

from bux.architecture.repeats import remat
from bux.train import xla

# Of a device's memory a step may take, by XLA's analysis: on B300s it was the peak
# to 0.1 GiB, and steps of 98% of a device trained.
MARGIN = 0.97


class Tried(NamedTuple):
    """A candidate the step compiled under: the bytes it holds on a device, None if
    XLA could not fit it, and the seconds its compile took."""

    policy: str
    bytes: int | None
    seconds: float


class Fitting:
    """A training step, `step(model, optimizer, batch)`, under remat `auto`. Its first
    call compiles it under each candidate until one fits, and runs that one, `policy`;
    `tried` says what each took."""

    def __init__(self, step: Callable[..., Any], options: xla.Options, **jit: Any):
        self.step, self.options, self.jit = step, options, jit
        self.chosen: xla.Compiled | None = None
        self.policy: str | None = None
        self.tried: list[Tried] = []

    def __call__(self, *args: Any) -> Any:
        return self._fitted(args)(*args)

    def lower(self, *args: Any) -> Any:
        """Return the step lowered for `args`, under the candidate that fits."""
        return self._fitted(args).jitted.lower(*args)

    def _fitted(self, args: tuple) -> xla.Compiled:
        """Return the step under the candidate that fits, which the first call of
        `args` chooses."""
        if self.chosen is None:
            self.policy, self.chosen = self._first_that_fits(args)
            _report(self.policy, self.tried)
        return self.chosen

    def _first_that_fits(self, args: tuple) -> tuple[str, xla.Compiled]:
        """Return the first candidate whose step, compiled for `args`, fits, and that
        step; else the last candidate's."""
        *firsts, last = remat.CANDIDATES
        for policy in firsts:
            step = self._compiled(policy, args)
            if fits(self.tried[-1].bytes):
                return policy, step
        return last, self._compiled(last, args)

    def _compiled(self, policy: str, args: tuple) -> xla.Compiled:
        """Return the step compiled for `args` under `policy`, and note in `tried` what
        it holds and how long its compile took."""
        step = xla.Compiled(nnx.jit(self._trying(policy), **self.jit), self.options)
        start = time.perf_counter()
        holds = held(step, args)
        self.tried.append(Tried(policy, holds, time.perf_counter() - start))
        return step

    def _trying(self, policy: str) -> Callable[..., Any]:
        """Return the step, traced under `policy`."""

        def step(model: Any, optimizer: Any, batch: Any) -> Any:
            with remat.trying(policy):
                return self.step(model, optimizer, batch)

        return step


def held(step: xla.Compiled, args: tuple) -> int | None:
    """Return the bytes `step`, compiled for `args`, holds on a device at once, by
    XLA's analysis: its arguments, its outputs but those that take an argument's
    memory, its temporaries and its code. None if XLA could not fit it."""
    try:
        m = step.compile(*args).memory_analysis()
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


def fits(held: int | None) -> bool:
    """Return whether a step that holds `held` bytes fits in a device's memory, with
    a margin; anything fits where the device does not say how much it can hold."""
    limit = capacity()
    if limit is None:
        return True
    return held is not None and held <= MARGIN * limit


def _report(policy: str, tried: list[Tried]) -> None:
    """Say, on the first process, which policy `auto` chose, and what each took."""
    if jax.process_index() != 0:
        return
    limit = capacity()
    each = ", ".join(f"{t.policy} {_gib(t.bytes)} in {t.seconds:.0f}s" for t in tried)
    room = "" if limit is None else f"; {MARGIN:.0%} of {_gib(limit)} fits"
    print(f"remat=auto: {policy} ({each}{room})", flush=True)


def _gib(n: int | None) -> str:
    return "too much" if n is None else f"{n / 2**30:.1f} GiB"
