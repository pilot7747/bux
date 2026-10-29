"""The stretches of a run's steps that each read batches of one shape from one data:
its ``curriculum``'s phases, cut wherever its ``batch_ramp`` grows the batch.

A stretch's settings are the run's, with the ``seq_len``, ``batch`` and ``data`` in
force during it, so what reads a run's settings reads a stretch's as they are: the
batches (`bux.train.data.training_batches`), the datasets to prepare
(`bux.data.pipeline.jobs`), the checks of the mesh. The RoPE a phase sets is the
model's (`ropes`, `bux.lang.ir.Run.arch_at`). It needs no JAX, so the compiler and
``bux check`` read it too::

    for s in stretches(run.train):
        print(s.start, s.stop, s.config.batch, s.config.seq_len)
"""

import itertools
from collections.abc import Sequence
from typing import NamedTuple

from bux.train.config import RampConfig, TrainConfig

#: The settings of the run a phase changes.
CHANGES = ("seq_len", "batch", "data")
#: The settings of its full-attention layers' RoPE a phase changes, by field of the
#: phase, and of ``RoPEConfig``.
ROPE = {"rope_scaling": "scaling", "rope_base": "base"}


class Stretch(NamedTuple):
    """Steps `start` to `stop` of a run, whose batches are `config`'s."""

    start: int  #: Its first step, from 0.
    stop: int  #: The step after its last.
    config: TrainConfig  #: The run's settings, with the seq_len, batch and data then.


def phases(t: TrainConfig) -> list[tuple[int, TrainConfig]]:
    """Return the step each phase of `t` starts at and its settings, from the run's
    own at step 0: each phase's, where it sets them, else the phase before's."""
    found = [(0, t)]
    for phase in t.curriculum:
        changed = {k: v for k in CHANGES if (v := getattr(phase, k)) is not None}
        found.append((phase.start(t.steps), found[-1][1].model_copy(update=changed)))
    return found


def ropes(t: TrainConfig) -> list[tuple[int, dict[str, object]]]:
    """Return the step each phase of `t` starts at and the fields of ``RoPEConfig``
    its full-attention layers take then (`ROPE`): each phase's, where it sets them,
    else the phase before's; none from step 0."""
    found: list[tuple[int, dict[str, object]]] = [(0, {})]
    for phase in t.curriculum:
        changed = {
            f: v for k, f in ROPE.items() if (v := getattr(phase, k)) is not None
        }
        found.append((phase.start(t.steps), found[-1][1] | changed))
    return found


def rope_at(t: TrainConfig, step: int) -> dict[str, object]:
    """Return the fields of ``RoPEConfig`` `t`'s full-attention layers take at `step`
    (`ropes`): in the step that takes a run from `step` steps to the next, and in an
    eval after `step` steps."""
    return next(rope for start, rope in reversed(ropes(t)) if start <= step)


def ramped(ramp: RampConfig, batch: int, step: int) -> int:
    """Return the rows of `step` under `ramp` up to `batch`: linear from its start,
    rounded down to a multiple, never below its start; `batch` once it is over."""
    if step >= ramp.steps:
        return batch
    grown = ramp.start + (batch - ramp.start) * step // ramp.steps
    multiple = ramp.multiple or ramp.start
    return max(ramp.start, grown // multiple * multiple)


def _ramp_cuts(ramp: RampConfig, batch: int) -> list[int]:
    """Return the steps at which `ramp` up to `batch` takes a new batch size: where
    its line reaches each multiple between its start and `batch`."""
    multiple, rise = ramp.multiple or ramp.start, batch - ramp.start
    sizes = range((ramp.start // multiple + 1) * multiple, batch, multiple)
    return [-(-(v - ramp.start) * ramp.steps // rise) for v in sizes]


def stretches(t: TrainConfig) -> tuple[Stretch, ...]:
    """Return the stretches of `t`'s steps, in order from step 0 to its last: one for
    a run with neither a curriculum nor a batch ramp.

    Raises:
        ValueError: If phases do not start in order inside the run, or the ramp does
            not grow to the run's batch (`check` says which).
    """
    check(t)
    starts = phases(t)
    cuts = {s for s, _ in starts}
    if (ramp := t.batch_ramp) is not None:
        cuts.update(_ramp_cuts(ramp, t.batch), [ramp.steps])
    bounds = [*sorted(c for c in cuts if c < t.steps), t.steps]
    found: list[Stretch] = []
    for start, stop in itertools.pairwise(bounds):
        config = next(c for s, c in reversed(starts) if s <= start)
        if ramp is not None and start < ramp.steps:
            config = config.model_copy(update={"batch": ramped(ramp, t.batch, start)})
        if found and found[-1].config == config:
            found[-1] = found[-1]._replace(stop=stop)
        else:
            found.append(Stretch(start, stop, config))
    return tuple(found)


def check(t: TrainConfig) -> None:
    """Raise unless `t`'s phases and its batch ramp fit together (`check_phases`,
    `check_ramp`)."""
    check_ramp(t)
    check_phases(t)


def check_phases(t: TrainConfig) -> None:
    """Raise unless `t`'s phases start in order inside the run, and none sets the
    batch before its ramp is over.

    Raises:
        ValueError: Saying which phase, and where it starts.
    """
    starts = [s for s, _ in phases(t)]
    for n, (before, start) in enumerate(itertools.pairwise(starts), 1):
        if start >= t.steps:
            raise ValueError(
                f"phase {n} starts at step {start}, and the run has {t.steps}"
            )
        if start <= before:
            raise ValueError(
                f"phase {n} starts at step {start}, not after the phase before it "
                f"(step {before}): phases go in the order they start"
            )
    ramp = t.batch_ramp
    for n, phase in enumerate(t.curriculum, 1):
        if ramp is not None and phase.batch is not None and starts[n] < ramp.steps:
            raise ValueError(
                f"phase {n} sets the batch at step {starts[n]}, before the batch "
                f"ramp is over at step {ramp.steps}"
            )


def check_ramp(t: TrainConfig) -> None:
    """Raise unless `t`'s batch ramp grows to its batch within its steps.

    Raises:
        ValueError: Saying what to make it.
    """
    if (ramp := t.batch_ramp) is None:
        return
    if ramp.start >= t.batch:
        raise ValueError(
            f"the batch ramp starts at {ramp.start} rows, not below the run's batch "
            f"of {t.batch}, which it grows to"
        )
    if ramp.steps > t.steps:
        raise ValueError(
            f"the batch ramp takes {ramp.steps} steps, and the run has {t.steps}"
        )


def varies(t: TrainConfig) -> bool:
    """Return whether `t`'s batches change shape or data as it trains."""
    return bool(t.curriculum) or t.batch_ramp is not None


def tokens(stretches: Sequence[Stretch], steps: int) -> int:
    """Return the tokens of a run's first `steps` steps, of its `stretches`: the last
    one's shape past its end."""
    total = 0
    for k, s in enumerate(stretches):
        stop = steps if k == len(stretches) - 1 else min(s.stop, steps)
        total += max(0, stop - s.start) * s.config.batch * s.config.seq_len
    return total


def si(n: float) -> str:
    """Return `n` with an SI suffix, as ``bux check``, ``bux plan`` and a run's start
    show it: 1.3B, 25.6k."""
    for suffix, scale in (("", 1), ("k", 1e3), ("M", 1e6), ("B", 1e9)):
        if "e" not in (shown := f"{n / scale:.3g}"):
            return shown + suffix
    return f"{n / 1e12:.3g}T"


def summary(t: TrainConfig) -> str:
    """Return the steps and tokens of `t`, as ``bux check`` and a run's start say
    them: ``50 steps × 8 × 64 tokens = 25.6k tokens``, with no shape where its batches
    change (`lines` says how)."""
    shape = "" if varies(t) else f" × {t.batch} × {t.seq_len} tokens"
    return f"{t.steps:,} steps{shape} = {si(tokens(stretches(t), t.steps))} tokens"


def lines(t: TrainConfig) -> list[str]:
    """Return a line for each phase of `t`, from its own settings at step 0, with its
    batch ramp: what ``bux check`` and a run's start say of them; none for a run whose
    batches never change.

    Example::

        lines(run.train)
        # ['from step 0: 512 × 1024 tokens, ramped from 64 rows over the first 2,000
        #   steps in 7 sizes', 'phase 1, from step 9,000: 64 × 8192 tokens,
        #   rope_scaling=yarn(factor=4.0, original_max_len=1024)']
    """
    from bux.lang.printer import value_to_source  # bux.lang imports this module

    if not varies(t):
        return []
    found, before = [], t
    for n, (start, c) in enumerate(phases(t)):
        line = f"phase {n}, from step {start:,}" if n else "from step 0"
        line += f": {c.batch} × {c.seq_len} tokens"
        if c.data != before.data:
            line += f", data={value_to_source(c.data)}"
        for k in ROPE if n else ():
            if (v := getattr(t.curriculum[n - 1], k)) is not None:
                line += f", {k}={value_to_source(v)}"
        found.append(line)
        before = c
    if (ramp := t.batch_ramp) is not None:
        sizes = len({s.config.batch for s in stretches(t) if s.start < ramp.steps})
        found[0] += (
            f", ramped from {ramp.start} rows over the first {ramp.steps:,} steps"
        )
        found[0] += f" in {sizes} sizes" if sizes > 1 else ""
    return found
