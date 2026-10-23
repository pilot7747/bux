"""What a model's forward is tracing now: which residual branch outside any other, and
which pass of a repeat. Precision asks the first, to keep the model's first and last
branches in compute (``matmuls.format_of``), and remat the second, to rematerialize a
repeat's first passes alone (``remat.policy``).

    with tracing.forward(main):                    # a model's forward (``Model``)
        with tracing.branch():                     # a branch (``graph.run_branch``)
            ...
        for run in tracing.runs(times, cuts):      # a repeat's passes, in runs
            with tracing.passes(run, each):        # each run traced once
                ...

The branches outside any other are numbered in the order they run. A scan traces one
pass for all its passes, so it numbers their branches as its first pass's, and those
after it as if every pass had run. That is right only for passes that trace alike, so
a repeat cuts its passes into runs of them (``runs``): where precision or remat tell
them apart (``matmuls.cuts``, ``remat.cuts``), and in a pipeline's stage, where some
stage skips them (``pipelined.Stage``).
"""

import contextlib
import dataclasses
import itertools
from collections.abc import Generator, Iterable, Iterator
from contextvars import ContextVar
from typing import NamedTuple

from bux.lang import ir


class Branch(NamedTuple):
    """A residual branch outside any other: its number in a forward, and the forward's
    `main` branches, those before its first head."""

    number: int
    main: int


@dataclasses.dataclass
class Count:
    """The branches of a forward: `main` of them before its first head, and the number
    of the `next` to trace. Each of `scans` traces one pass for all its passes: the
    number of its first branch, and the branches of a pass."""

    main: int
    next: int = 0
    scans: list[tuple[int, int]] = dataclasses.field(default_factory=list)

    def peek(self) -> int:
        """Return the number of the next branch: in a scan, its first pass's."""
        n = self.next
        for first, each in reversed(self.scans):  # the innermost scan first
            n = first + (n - first) % each
        return n

    def take(self) -> int:
        """Return the number of the next branch, and count it."""
        n = self.peek()
        self.next += 1
        return n


_COUNT = ContextVar[Count | None]("branches", default=None)
_BRANCH = ContextVar[int | None]("branch", default=None)
_PASS = ContextVar[int | None]("pass", default=None)  # of the outermost repeat


@contextlib.contextmanager
def _set[T](var: ContextVar[T], value: T) -> Generator[None]:
    """Set `var` to `value` while the block runs."""
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def forward(main: int) -> contextlib.AbstractContextManager[None]:
    """Number the branches of a forward of a model with `main` of them before its
    first head (``main_branches``)."""
    return _set(_COUNT, Count(main))


@contextlib.contextmanager
def branch() -> Generator[None]:
    """Number the residual branch outside any other that traces inside."""
    count = _COUNT.get()
    with _set(_BRANCH, None if count is None else count.take()):
        yield


@contextlib.contextmanager
def passes(run: range, each: int = 0, is_branch: bool = False) -> Generator[None]:
    """Trace the block as passes `run` of a repeat, all in one trace. Remat sees them
    as the run's first pass, or inside a pass of an outer repeat as that one
    (``current_pass``); their branches, `each` a pass, number as ``branches`` says."""
    outer = _PASS.get()
    at = run.start if outer is None else outer
    with _set(_PASS, at), branches(len(run), each, is_branch):
        yield


@contextlib.contextmanager
def branches(times: int, each: int, is_branch: bool = False) -> Generator[None]:
    """Number the branches of `times` passes that one trace runs, `each` a pass: as the
    first pass's, however often it traces, and those after them as if every pass had
    run. With `is_branch`, each pass is a branch outside any other, numbered as one."""
    count = _COUNT.get()
    if count is None or each == 0:
        yield
        return
    first = count.next
    count.scans.append((first, each))
    try:
        with branch() if is_branch else contextlib.nullcontext():
            yield
    finally:
        count.scans.pop()
        count.next = first + times * each


def current_branch() -> Branch | None:
    """Return the branch outside any other that is tracing; None outside one."""
    count, number = _COUNT.get(), _BRANCH.get()
    return None if count is None or number is None else Branch(number, count.main)


def next_branches(n: int) -> list[Branch]:
    """Return the next `n` branches to trace, numbered as if each ran; none outside a
    forward."""
    if (count := _COUNT.get()) is None:
        return []
    start = count.peek()
    return [Branch(start + i, count.main) for i in range(n)]


def current_pass() -> int | None:
    """Return the pass of the outermost repeat that is tracing; None outside any."""
    return _PASS.get()


def runs(times: int, cuts: Iterable[int] = ()) -> list[range]:
    """Split `times` passes into runs, at each of `cuts` that falls among them."""
    ends = sorted({0, times, *(c for c in cuts if 0 < c < times)})
    return [range(a, b) for a, b in itertools.pairwise(ends)]


# --- Branches in the graph ------------------------------------------------------------


def main_branches(graph: ir.Node) -> int:
    """Return the residual branches outside any other that run before the first head:
    those of the model, which precision's `first` and `last` count, and not its MTP
    modules'."""
    count = 0
    for event in _events(graph):
        if event == "head":
            break
        count += 1
    return count


def branches_of(node: ir.Node) -> int:
    """Return the residual branches outside any other that one run of `node` takes,
    as an item of a sequence or the body of a repeat outside any branch."""
    return sum(event == "branch" for event in _item(node))


def _events(node: ir.Node) -> Iterator[str]:
    """Yield "branch" for each residual branch outside any other, and "head" for each
    head, in the order they run: as ``graph.branches`` finds the branches."""
    match node:
        case ir.Layer(config=config) if config.type == "head":
            yield "head"
        case ir.Seq(items=items):
            for item in items:
                yield from _item(item)
        case ir.Repeat(body=body, times=times):
            for _ in range(times):
                yield from _item(body)
        case ir.Residual():
            return  # what it holds is inside a branch
        case _:
            for child in ir.children(node):
                yield from _events(child)


def _item(node: ir.Node) -> Iterator[str]:
    """The events of an item of a sequence or a repeat's body: a branch, if it is one."""
    if isinstance(node, ir.Residual):
        yield "branch"
    else:
        yield from _events(node)
