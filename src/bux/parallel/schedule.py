"""Pipeline schedules as tables: the op each stage runs at each tick, which
bux.parallel.scheduled runs.

An op is `F`, a microbatch's forward through a stage, or `B`, its backward. Where a
schedule splits the backward, `B` takes the gradient of the stage's input alone and
`W` that of its weights, later. Each stage has a queue of ops for each lane of a
tick, and ``lockstep`` runs the queues tick by tick:

    tick t, on each stage, in each lane:  the queue's next op, if what it reads is there
    end of tick t:                        F's output goes on to the next stage,
                                          B's input gradient back to the one before

    1f1b  two lanes, F and a whole B, each as soon as it can run: stage s holds
          2(S − s) − 1 microbatches, and the stages idle as gpipe's do.
    zbh1  one lane, B split: stage s runs S − s forwards, then a B, the W of the
          microbatch s before it, and an F, in turn. Every stage holds S microbatches,
          and they idle S − 1 ticks of 3M + S − 1 (ZB-H1 of Qi et al., "Zero Bubble
          Pipeline Parallelism", 2024).

A table also gives the slot of a buffer each stage keeps each microbatch in
(``Table.buffer_slots``), and the share of a step the stages idle (``Table.bubble``).
"""

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from jaxtyping import Int

FORWARD, BACKWARD, WEIGHTS = "F", "B", "W"
KINDS = (FORWARD, BACKWARD, WEIGHTS)
CODES = {FORWARD: 1, BACKWARD: 2, WEIGHTS: 3}  # an op's kind in arrays; 0 is idle
# What a stage keeps microbatches in, each from the op or send that fills it to the
# op that reads it last: the stream the stage before sent, the gradient the stage
# after sent, a forward's residuals, and what a split `B` saves for `W`.
BUFFERS = ("inputs", "cotangents", "residuals", "saved")
ARRIVAL = 1 - 1e-6  # a send at tick t arrives at t + ARRIVAL, after the tick's ops


class Op(NamedTuple):
    kind: str  # FORWARD, BACKWARD or WEIGHTS
    microbatch: int


Queue = list[Op]
Lanes = tuple[Op | None, ...]  # what a stage runs at a tick, lane by lane
Place = tuple[int, int, int]  # (tick, stage, lane)
Key = tuple[str, int, int]  # an op of a stage: (kind, stage, microbatch)
Span = tuple[float, float]  # from when to when a buffer holds a microbatch, in ticks
Ran = dict[Key, int]  # the tick each op ran at


@dataclass(frozen=True)
class Table:
    """The op each stage runs in each lane of each tick, `ops[t][s][lane]`, or None.
    With `split`, `B` takes the input's gradient alone and `W` the weights'."""

    ops: tuple[tuple[Lanes, ...], ...]
    split: bool

    @property
    def ticks(self) -> int:
        return len(self.ops)

    @property
    def stages(self) -> int:
        return len(self.ops[0])

    @property
    def lanes(self) -> int:
        return len(self.ops[0][0])

    @property
    def microbatches(self) -> int:
        return 1 + max(op.microbatch for _, op in self.each_op())

    def each_op(self) -> Iterator[tuple[Place, Op]]:
        """Yield each op the table runs, with its tick, stage and lane."""
        for t, row in enumerate(self.ops):
            for s, lanes in enumerate(row):
                for lane, op in enumerate(lanes):
                    if op is not None:
                        yield (t, s, lane), op

    def as_arrays(self) -> tuple[Int[np.ndarray, "t s k"], Int[np.ndarray, "t s k"]]:
        """Return each lane's op as two arrays: the code of its kind (``CODES``),
        0 where the lane idles, and its microbatch."""
        codes = np.zeros((self.ticks, self.stages, self.lanes), np.int32)
        microbatches = np.zeros_like(codes)
        for place, op in self.each_op():
            codes[place], microbatches[place] = CODES[op.kind], op.microbatch
        return codes, microbatches

    def kinds_in(self, lane: int) -> list[str]:
        """Return the kinds of op `lane` runs at some tick, in the order of KINDS."""
        ran = {op.kind for (_, _, k), op in self.each_op() if k == lane}
        return [kind for kind in KINDS if kind in ran]

    def arrivals(self, kind: str) -> Int[np.ndarray, "t s"]:
        """Return the microbatch each stage receives at the end of each tick, or -1:
        the output of the stage before's forward (`F`), or the input gradient of the
        stage after's backward (`B`)."""
        got = np.full((self.ticks, self.stages), -1, np.int32)
        to = 1 if kind == FORWARD else -1
        for (t, s, _), op in self.each_op():
            if op.kind == kind and 0 <= s + to < self.stages:
                got[t, s + to] = op.microbatch
        return got

    def buffer_slots(self) -> dict[str, Int[np.ndarray, "s m"]]:
        """Return, for each of BUFFERS, the slot each stage keeps each microbatch in:
        no two microbatches share one while both are held."""
        times = self._times()
        return {name: self._slots(name, times) for name in BUFFERS}

    def bubble(self) -> float:
        """Return the share of a step the stages idle, a forward, a backward of the
        input and one of the weights taking a unit each: a tick takes as long as its
        busiest stage."""
        cost = {FORWARD: 1, BACKWARD: 1 if self.split else 2, WEIGHTS: 1}
        busy = [
            max(sum(cost[op.kind] for op in lanes if op) for lanes in row)
            for row in self.ops
        ]
        work = 3 * self.microbatches  # of each stage
        return 1 - work / sum(busy)

    def _times(self) -> dict[Key, float]:
        """Return when each op runs, in ticks: lane k of tick t at
        t + (k + 1) / (lanes + 1), so that a tick's lanes run in order."""
        return {
            (op.kind, s, op.microbatch): t + (k + 1) / (self.lanes + 1)
            for (t, s, k), op in self.each_op()
        }

    def _slots(self, buffer: str, times: dict[Key, float]) -> Int[np.ndarray, "s m"]:
        """Return the slot of `buffer` each stage keeps each microbatch in."""
        slots = np.zeros((self.stages, self.microbatches), np.int32)
        for s in range(self.stages):
            spans = {}
            for j in range(self.microbatches):
                if (span := self._span(buffer, s, j, times)) is not None:
                    spans[j] = span
            for j, slot in _lowest_free_slots(spans).items():
                slots[s, j] = slot
        return slots

    def _span(
        self, buffer: str, s: int, j: int, times: dict[Key, float]
    ) -> Span | None:
        """Return when stage `s` starts and stops holding microbatch `j` in `buffer`,
        or None if it never does."""

        def at(kind: str, stage: int) -> float:  # when the op runs
            return times[kind, stage, j]

        def sent(kind: str, stage: int) -> float:  # when what it sends arrives
            return int(at(kind, stage)) + ARRIVAL

        if buffer == "inputs":  # the first stage runs the part before the repeat
            return (sent(FORWARD, s - 1), at(FORWARD, s)) if s > 0 else None
        if buffer == "cotangents":  # the last stage takes the loss's in its forward
            filled = at(FORWARD, s) if s == self.stages - 1 else sent(BACKWARD, s + 1)
            return (filled, at(BACKWARD, s))
        if buffer == "residuals":  # until the op that reads them last
            return (at(FORWARD, s), at(WEIGHTS if self.split else BACKWARD, s))
        return (at(BACKWARD, s), at(WEIGHTS, s)) if self.split else None


def _lowest_free_slots(spans: dict[int, Span]) -> dict[int, int]:
    """Give each microbatch, in the order they come, the lowest slot free when it
    comes: as few slots as the most held at once."""
    free_at: list[float] = []  # when each slot is free again
    slots: dict[int, int] = {}
    for j in sorted(spans, key=lambda j: spans[j]):
        start, stop = spans[j]
        slot = next((i for i, free in enumerate(free_at) if free < start), len(free_at))
        if slot == len(free_at):
            free_at.append(stop)
        free_at[slot] = stop
        slots[j] = slot
    return slots


def lockstep(queues: list[list[Queue]], split: bool) -> Table:
    """Return the table of each stage's queues run in lockstep: at each tick, lane k
    of stage s runs the next op of `queues[s][k]` if it is ready (``_ready``)."""
    waiting = [[deque(queue) for queue in stage] for stage in queues]
    ran: Ran = {}
    ticks: list[tuple[Lanes, ...]] = []
    while any(queue for stage in waiting for queue in stage):
        t = len(ticks)
        row = tuple(
            _tick(lanes, s, t, len(queues), ran) for s, lanes in enumerate(waiting)
        )
        if all(op is None for lanes in row for op in lanes):
            raise ValueError(f"the queues deadlock at tick {t}")
        ticks.append(row)
    return Table(tuple(ticks), split)


def _tick(waiting: list[deque[Op]], s: int, t: int, stages: int, ran: Ran) -> Lanes:
    """Return what stage `s` runs at tick `t`, lane by lane: the next op waiting in
    the lane's queue if it is ready, which leaves the queue for `ran`."""
    lanes: list[Op | None] = []
    for queue in waiting:
        if not queue or not _ready(queue[0], s, t, stages, ran):
            lanes.append(None)
            continue
        op = queue.popleft()
        ran[op.kind, s, op.microbatch] = t
        lanes.append(op)
    return tuple(lanes)


def _ready(op: Op, s: int, t: int, stages: int, ran: Ran) -> bool:
    """Return whether stage `s` can run `op` at tick `t`: a forward once the stage
    before sent it its output; a backward once this stage ran the forward, and the
    stage after sent it the gradient (the last stage's comes of its own forward); the
    weights' once this stage ran the input's."""

    def sent(kind: str, stage: int) -> bool:  # by another stage, at an earlier tick
        return ran.get((kind, stage, op.microbatch), t) < t

    def ran_here(kind: str) -> bool:  # by this stage, in an earlier lane or tick
        return (kind, s, op.microbatch) in ran

    if op.kind == FORWARD:
        return s == 0 or sent(FORWARD, s - 1)
    if op.kind == BACKWARD:
        return ran_here(FORWARD) and (s == stages - 1 or sent(BACKWARD, s + 1))
    return ran_here(BACKWARD)


def one_f_one_b(stages: int, microbatches: int) -> Table:
    """1F1B in lockstep: two lanes, a stage's forwards in order in one and its whole
    backwards in order in the other, each as soon as it can run."""
    forwards = [Op(FORWARD, j) for j in range(microbatches)]
    backwards = [Op(BACKWARD, j) for j in range(microbatches)]
    return lockstep([[forwards, backwards] for _ in range(stages)], split=False)


def zero_bubble_h1(stages: int, microbatches: int) -> Table:
    """ZB-H1: one lane, and a stage's backward split in two, the weights' gradient of
    each microbatch filling a tick the stage would idle."""
    queues = [[_zbh1_queue(s, stages, microbatches)] for s in range(stages)]
    return lockstep(queues, split=True)


def _zbh1_queue(s: int, stages: int, microbatches: int) -> Queue:
    """Return stage s's ops: S − s forwards; then a backward, the weights' of the
    microbatch s before it, and a forward, in turn; and the weights' left."""
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
