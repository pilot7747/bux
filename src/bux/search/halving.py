"""Which trial of a search trains next, and for how long: asynchronous successive
halving (ASHA; Li et al., 2020), and Hyperband's brackets of it (Li et al., 2018).

A search with ``halving(eta=3, rungs=3)`` trains every trial for a ninth of the run's
steps first, rung 0; the best third of the trials that finished a rung train again
for three times as long, up to rung 2, the run's own steps. Whenever a worker frees
up, it takes the next trial up a rung, the highest rung first, if one has earned it,
else a new trial at the bottom: no rung waits for the others to finish (`next_job`).

With ``brackets=b``, trials take turns starting at rungs 0, 1, ..., b - 1, and each
bracket ranks its own: a bracket that starts higher trusts short runs less. Without
`halving`, a search has one rung, the run's own steps, and every trial trains it.
"""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

from bux.train import config as tcfg


@dataclass
class Trial:
    """A point of a search, and what it trained.

    Attributes:
        number: Its place among the search's trials, from 0.
        values: The value of each dimension, by name; a choice's option's index.
        bracket: The rung it started at.
        started: The rungs it started, in order.
        results: Its objective at each rung it finished; NaN where it failed.
    """

    number: int
    values: dict[str, float]
    bracket: int = 0
    started: list[int] = field(default_factory=list)
    results: dict[int, float] = field(default_factory=dict)

    @property
    def running(self) -> int | None:
        """Return the rung it is training, or None."""
        last = self.started[-1] if self.started else None
        return None if last is None or last in self.results else last

    @property
    def top(self) -> tuple[int, float] | None:
        """Return its highest rung finished and its objective there, or None."""
        if not self.results:
            return None
        rung = max(self.results)
        return rung, self.results[rung]


class Job(NamedTuple):
    """What a search trains next: a new trial at `rung`, if `trial` is None, else
    `trial` up to `rung`."""

    rung: int
    trial: Trial | None = None


def rungs(config: tcfg.HalvingConfig | None) -> int:
    """Return how many rungs a search has: one without halving."""
    return 1 if config is None else config.rungs


def steps(config: tcfg.HalvingConfig | None, run_steps: int, rung: int) -> int | None:
    """Return the steps a trial trains at `rung`: None at the top, the run's own."""
    if config is None or rung == config.rungs - 1:
        return None
    return config.steps(run_steps, rung)


def next_job(
    trials: list[Trial], config: tcfg.HalvingConfig | None, budget: int
) -> Job | None:
    """Return what to train next: a trial that earned the rung above, the highest
    first, else a new trial at the bottom of its bracket if fewer than `budget` were
    started; None if neither, until a running trial ends."""
    if config is not None:
        for rung in reversed(range(config.rungs - 1)):
            for bracket in range(min(rung + 1, config.brackets)):
                if (up := _earned(trials, bracket, rung, config.eta)) is not None:
                    return Job(rung + 1, up)
    if len(trials) < budget:
        return Job(_bracket(config, len(trials)))
    return None


def _bracket(config: tcfg.HalvingConfig | None, number: int) -> int:
    """Return the rung trial `number` starts at: its bracket, in turn."""
    return 0 if config is None else number % config.brackets


def _earned(trials: list[Trial], bracket: int, rung: int, eta: int) -> Trial | None:
    """Return a trial of `bracket` among the best ``1 / eta`` of those that finished
    `rung`, which has not started the rung above; None if there is none."""
    finished = [t for t in trials if t.bracket == bracket and rung in t.results]
    ranked = sorted(
        (t for t in finished if not math.isnan(t.results[rung])),
        key=lambda t: t.results[rung],
    )
    for t in ranked[: len(finished) // eta]:
        if rung + 1 not in t.started:
            return t
    return None


def best(trials: list[Trial]) -> Trial | None:
    """Return the trial with the least objective at the highest rung where any has
    one that is a number, or None if none has."""
    found = [
        (rung, value, t.number, t)
        for t in trials
        for rung, value in t.results.items()
        if not math.isnan(value)
    ]
    if not found:
        return None
    top = max(rung for rung, *_ in found)
    return min((v, n, t) for rung, v, n, t in found if rung == top)[2]
