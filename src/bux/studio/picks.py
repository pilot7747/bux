"""The run of a sweep that the page picked: its axes by id, and its trace.

The page names each axis by an id that survives edits elsewhere in the file (its
labels, or where it is and its options), and sends ``picks``, the option it takes on
each: ``{"gqa|mha|mqa": 2, "optimizer.lr=1e-3|3e-3": 1}``. An axis left out takes its
first option, and an axis inside an option is met only once that option is picked,
so `traced` traces again until the axes it meets stop changing.
"""

from collections.abc import Mapping

from bux.lang.compiler import Axis, Trace, trace
from bux.lang.syntax import Span

type Choices = dict[Span | frozenset[str], int | str]


def axis_id(axis: Axis) -> str:
    """Return how the page names `axis`: its sorted labels, ``gqa|mha|mqa``, or where
    it is and its options as written, ``optimizer.lr=1e-3|3e-3``."""
    if isinstance(axis.key, frozenset):
        return "|".join(sorted(axis.key))
    return f"{axis.where}={'|'.join(axis.options)}"


def choices(axes: list[Axis], picks: Mapping[str, int]) -> Choices:
    """Return the compiler's choices for `picks` over `axes`; ids not among them, and
    options past the last, are ignored."""
    found: Choices = {}
    for axis in axes:
        option = picks.get(axis_id(axis), 0)
        if 0 < option < len(axis.options):
            found[axis.key] = axis.pick(option)
    return found


def traced(source: str, picks: Mapping[str, int], *, path: str | None) -> Trace:
    """Return the trace of the run that `picks` selects in `source`, the file at
    `path` (None for one not saved), whose ``use`` searches from there.

    Raises:
        BuxError: If the file does not compile, as `bux.lang.compiler.trace` raises.
    """
    filename = path or "<string>"
    picked: Choices = {}
    t = trace(source, filename=filename)
    for _ in range(8):  # an axis is nested in at most a few others
        again = choices(t.axes, picks)
        if again == picked:
            break
        picked = again
        t = trace(source, choices=picked, filename=filename)
    return t
