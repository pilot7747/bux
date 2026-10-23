"""Pipeline schedules as tables: what each stage runs at each tick
(bux.parallel.pipeline runs them).

A schedule gives each stage its ops in order, a queue for each slot of a tick. `F`
runs a microbatch forward through the stage, and `B` takes its gradient back through
it; where a schedule splits the backward, `B` takes the gradient of the stage's input
alone, and `W` that of its weights later. ``timed`` runs the queues in lockstep: at
each tick, each slot of each stage runs the next op of its queue if what the op reads
is there, and at the end of the tick each stage sends its forward's output on to the
next stage and its backward's input gradient back to the one before.

    1f1b  two slots: each stage runs its next forward and its next whole backward,
          each as soon as it can. Stage s holds 2(S − s) − 1 microbatches, and the
          stages idle (S − 1) / (M + S − 1) of a step, as gpipe's do.
    zbh1  one slot, the backward split: stage s runs S − s forwards, then a backward,
          the weights of the microbatch s before it, and a forward, in turn. Every
          stage holds S microbatches, and they idle S − 1 ticks of 3M + S − 1:
          ZB-H1 of Qi et al., "Zero Bubble Pipeline Parallelism" (2024).

A table also says how many microbatches a stage holds in each of its buffers at once
(``Table.slots``), and the share of a step the stages idle (``Table.bubble``).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from jaxtyping import Int

FORWARD, BACKWARD, WEIGHTS = "F", "B", "W"
KINDS = ("", FORWARD, BACKWARD, WEIGHTS)  # an op's code is its index; 0 is idle
# The buffers a stage holds microbatches in, from the op or send that fills one to
# the op that reads it.
BUFFERS = ("inputs", "cotangents", "residuals", "weights")


class Op(NamedTuple):
    kind: str  # FORWARD, BACKWARD or WEIGHTS
    micro: int


Queue = list[Op]
Slots = tuple[Op | None, ...]  # what a stage runs at a tick, slot by slot


@dataclass(frozen=True)
class Table:
    """The op each stage runs in each slot of each tick, `ops[t][s][k]`, or None; with
    `split`, `B` takes the input's gradient alone and `W` the weights'."""

    ops: tuple[tuple[Slots, ...], ...]
    split: bool

    @property
    def ticks(self) -> int:
        return len(self.ops)

    @property
    def stages(self) -> int:
        return len(self.ops[0])

    @property
    def width(self) -> int:
        """The slots of a tick."""
        return len(self.ops[0][0])

    @property
    def microbatches(self) -> int:
        return 1 + max(op.micro for op in self._all())

    def codes(self) -> tuple[Int[np.ndarray, "t s k"], Int[np.ndarray, "t s k"]]:
        """Return each slot's op kind, by its index in KINDS, and its microbatch (0
        where it idles)."""
        kinds = np.zeros((self.ticks, self.stages, self.width), np.int32)
        micros = np.zeros_like(kinds)
        for (t, s, k), op in self._each():
            kinds[t, s, k], micros[t, s, k] = KINDS.index(op.kind), op.micro
        return kinds, micros

    def received(self, kind: str) -> Int[np.ndarray, "t s"]:
        """Return the microbatch each stage receives at the end of each tick, or -1:
        the output of the stage before's forward (`F`), or the input gradient of the
        stage after's backward (`B`)."""
        got = np.full((self.ticks, self.stages), -1, np.int32)
        step = 1 if kind == FORWARD else -1
        for (t, s, _), op in self._each():
            if op.kind == kind and 0 <= s + step < self.stages:
                got[t, s + step] = op.micro
        return got

    def slots(self) -> dict[str, Int[np.ndarray, "s m"]]:
        """Return, for each of BUFFERS, the slot each stage keeps each microbatch in:
        no two microbatches share one while both are held. Every stage has as many
        slots as the stage that needs the most."""
        return {name: _colour(self._held(name), self.stages) for name in BUFFERS}

    def bubble(self) -> float:
        """Return the share of a step the stages idle, a forward, a backward of the
        input and one of the weights each taking a unit: a tick takes as long as its
        busiest stage."""
        cost = {FORWARD: 1, BACKWARD: 1 if self.split else 2, WEIGHTS: 1}
        busy = [
            max(sum(cost[op.kind] for op in slots if op) for slots in row)
            for row in self.ops
        ]
        return 1 - 3 * self.microbatches / sum(busy)

    def _each(self) -> Iterator[tuple[tuple[int, int, int], Op]]:
        for t, row in enumerate(self.ops):
            for s, slots in enumerate(row):
                for k, op in enumerate(slots):
                    if op is not None:
                        yield (t, s, k), op

    def _all(self) -> list[Op]:
        return [op for _, op in self._each()]

    def _held(self, buffer: str) -> dict[tuple[int, int], tuple[float, float]]:
        """Return when each stage starts and stops holding each microbatch in
        `buffer`, (stage, micro) -> (from, to), in ticks: an op at slot k of tick t
        happens at t + (k + 1) / (width + 1), a send at the tick's end."""
        at, end = {}, 1 - 1e-6
        for (t, s, k), op in self._each():
            at[op.kind, s, op.micro] = t + (k + 1) / (self.width + 1)
        last = self.stages - 1
        held = {}
        for s in range(self.stages):
            for j in range(self.microbatches):
                if buffer == "inputs" and s > 0:
                    held[s, j] = (int(at[FORWARD, s - 1, j]) + end, at[FORWARD, s, j])
                elif buffer == "cotangents":
                    filled = (
                        at[FORWARD, s, j]
                        if s == last
                        else int(at[BACKWARD, s + 1, j]) + end
                    )
                    held[s, j] = (filled, at[BACKWARD, s, j])
                elif buffer == "residuals":
                    held[s, j] = (at[FORWARD, s, j], at[BACKWARD, s, j])
                elif buffer == "weights" and self.split:
                    held[s, j] = (at[BACKWARD, s, j], at[WEIGHTS, s, j])
        return held


def _colour(
    held: dict[tuple[int, int], tuple[float, float]], stages: int
) -> Int[np.ndarray, "s m"]:
    """Give each held microbatch the lowest slot no other holds while it does."""
    micros = 1 + max((j for _, j in held), default=0)
    slots = np.zeros((stages, micros), np.int32)
    for s in range(stages):
        free_at: list[float] = []  # when each slot is free again
        for j in sorted((j for t, j in held if t == s), key=lambda j: held[s, j]):
            start, stop = held[s, j]
            slot = next((i for i, f in enumerate(free_at) if f < start), len(free_at))
            if slot == len(free_at):
                free_at.append(stop)
            free_at[slot] = stop
            slots[s, j] = slot
    return slots


def timed(queues: list[list[Queue]], split: bool) -> Table:
    """Run each stage's queues in lockstep ticks, and return the table they make. At
    each tick, slot k of stage s runs the next op of `queues[s][k]` once what it reads
    is there: a forward the output of the stage before's, a backward the gradient
    the stage after sent (the last stage's: its own forward), a backward of the
    weights the backward of its input. What another stage sent arrives after the
    tick it was sent at; what this stage made, in an earlier slot of the same tick."""
    stages = len(queues)
    done: dict[tuple[str, int, int], int] = {}  # (kind, stage, micro) -> tick
    nexts = [[0] * len(q) for q in queues]
    ticks: list[tuple[Slots, ...]] = []
    while any(n < len(q) for ns, qs in zip(nexts, queues) for n, q in zip(ns, qs)):
        t, row = len(ticks), []
        for s, stage in enumerate(queues):
            slots: list[Op | None] = []
            for k, queue in enumerate(stage):
                op = queue[nexts[s][k]] if nexts[s][k] < len(queue) else None
                if op is None or not _ready(op, s, t, stages, done):
                    slots.append(None)
                    continue
                done[op.kind, s, op.micro] = t
                nexts[s][k] += 1
                slots.append(op)
            row.append(tuple(slots))
        if all(op is None for slots in row for op in slots):
            raise ValueError(f"the queues deadlock at tick {t}")
        ticks.append(tuple(row))
    return Table(tuple(ticks), split)


def _ready(
    op: Op, s: int, t: int, stages: int, done: dict[tuple[str, int, int], int]
) -> bool:
    """Return whether stage `s` can run `op` at tick `t`, given what ran when."""

    def before(kind: str, stage: int) -> bool:  # sent by another stage, ticks ago
        return done.get((kind, stage, op.micro), t) < t

    def ran(kind: str) -> bool:  # run by this stage, this tick or before
        return (kind, s, op.micro) in done

    if op.kind == FORWARD:
        return s == 0 or before(FORWARD, s - 1)
    if op.kind == BACKWARD:
        return ran(FORWARD) and (s == stages - 1 or before(BACKWARD, s + 1))
    return ran(BACKWARD)


def one_f_one_b(stages: int, microbatches: int) -> Table:
    """1F1B in lockstep: two slots a tick, a stage's forwards in order and its whole
    backwards in order, each as soon as it can run."""
    forwards = [Op(FORWARD, j) for j in range(microbatches)]
    backwards = [Op(BACKWARD, j) for j in range(microbatches)]
    return timed([[forwards, backwards] for _ in range(stages)], split=False)


def zero_bubble_h1(stages: int, microbatches: int) -> Table:
    """ZB-H1: one slot a tick, and a stage's backward split in two, the weights'
    gradient of each microbatch filling a tick the stage would idle."""
    return timed([[_zbh1(s, stages, microbatches)] for s in range(stages)], split=True)


def _zbh1(s: int, stages: int, microbatches: int) -> Queue:
    """Stage s's ops: S − s forwards, then a backward, the weights of the microbatch
    s before it, and a forward, in turn, and the weights left."""
    warmup = min(stages - s, microbatches)
    ops = [Op(FORWARD, j) for j in range(warmup)]
    for j in range(microbatches):
        ops.append(Op(BACKWARD, j))
        if j >= s:
            ops.append(Op(WEIGHTS, j - s))
        if warmup + j < microbatches:
            ops.append(Op(FORWARD, warmup + j))
    lagging = range(max(microbatches - s, 0), microbatches)
    return ops + [Op(WEIGHTS, j) for j in lagging]


TABLES: dict[str, Callable[[int, int], Table]] = {
    "1f1b": one_f_one_b,
    "zbh1": zero_bubble_h1,
}
