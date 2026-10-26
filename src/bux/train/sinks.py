"""Sinks: where hooks' scalars go. Scalars are metrics by name, as numbers, that a hook
returned after a step (`bux.train.hooks`); a step that observes the model returns
histograms too (`bux.train.observe`), which reach a sink apart (`Sink.histograms`).

A run writes to three: `Console`, a line per log; `JSONL`, the run's metrics.jsonl
and histograms.jsonl; and `LatestMetrics`, the latest value of each, which
`bux.train.trainer.train` returns. Its ``log.trackers`` add more, as W&B and
TensorBoard (`bux.train.trackers`), and a tracker of one's own is a sink::

    class Printed(Sink):
        def write(self, step: int, scalars: Scalars) -> None:
            print(step, dict(scalars))

    train(run, sinks=[Printed()])

In a job of many processes (`bux.launch`), every process computes all scalars and
only the main one writes them, to the console and metrics.jsonl. A sink of your own
that writes checks `is_main_process`.
"""

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, NamedTuple, TextIO

import jax
import numpy as np
from jaxtyping import Float

Scalars = Mapping[str, float]  # metrics by name, as numbers on the host


class Histogram(NamedTuple):
    """A histogram of ``log2 |x|`` over a layer's entries: the share of them in each bin
    of `edges` (`bux.architecture.observe.histogram`)."""

    edges: Float[np.ndarray, " n+1"]  #: The bins' edges, in ``log2 |x|``.
    shares: Float[np.ndarray, " n"]  #: The share of the entries in each bin.


def is_main_process() -> bool:
    """Return whether this is process 0 of a job, or the only one, without touching a
    device."""
    return not jax.distributed.is_initialized() or jax.process_index() == 0


def echo(line: str) -> None:
    """Print `line` now, even where stdout is a pipe, as a container's log is."""
    print(line, flush=True)


class Sink:
    """Where hooks' scalars go. A subclass overrides `write`, and `histograms` and
    `close` if it needs them; the trainer calls them in order, a step's scalars once
    the next step is launched."""

    def write(self, step: int, scalars: Scalars) -> None:
        """Take what the hooks reported after `step`.

        Args:
            step: The steps taken when the hooks ran.
            scalars: Their metrics by name, already on the host.
        """
        raise NotImplementedError

    def histograms(self, step: int, histograms: Mapping[str, Histogram]) -> None:
        """Take the histograms a step that observed the model took, after `write`
        took its scalars; nothing by default.

        Args:
            step: The steps taken when it observed.
            histograms: By name: ``act/hist/attn.03``.
        """

    def close(self) -> None:
        """After the last scalars, or when training stops."""


class Console(Sink):
    """A line per write, for a terminal or a container's log (``kubectl logs``): no
    colors and no carriage returns, each line flushed as it is printed. A line shows
    the metrics it has a format for, in their order, and a log's ends with the time
    since the console was made and the time left at the latest step's pace::

        smoke step 30/50  loss 6.3513  lr 3.00e-03  grad 0.554  74,647 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00

    A step that observed the model adds a line of its worst layers (`summary`), and
    a loss or a gradient that is not finite ends its line with a warning.

    Args:
        name: What the line starts with: the run's name.
        steps: The run's steps, which the line shows the step out of.
        echo: What prints a line.
        formats: A `str.format` pattern for each metric to show, by its name, in the
            order to show them; the rest are left out.
    """

    #: The metrics a line shows by default, each with its format.
    FORMATS: ClassVar[dict[str, str]] = {
        "loss": "loss {:.4f}",
        "mtp": "mtp {:.4f}",
        "lr": "lr {:.2e}",
        "grad_norm": "grad {:.3f}",
        "tokens_per_s": "{:,.0f} tok/s",
        "mfu": "mfu {:.1%}",
        "eval_loss": "eval loss {:.4f}",
        "time/step_ms": "{:.0f} ms/step",
        "time/data_ms": "data {:.1f} ms",
        "mem/peak_gib": "mem {:.1f} GiB",
        "time/compile_s": "compiled in {:.1f}s",
    }
    #: The metrics a line warns of where they are not finite.
    FINITE = ("loss", "grad_norm", "eval_loss")

    def __init__(
        self,
        name: str,
        steps: int,
        echo: Callable[[str], None] = echo,
        formats: Mapping[str, str] = FORMATS,
    ):
        self.name, self.steps, self.echo, self.formats = name, steps, echo, formats
        self.started = time.perf_counter()

    def write(self, step: int, scalars: Scalars) -> None:
        """Print a line of the scalars it has a format for, on the main process; none
        if it has none. Then, for a step that observed the model, a line of its
        worst layers."""
        if not is_main_process():
            return
        shown = [f.format(scalars[k]) for k, f in self.formats.items() if k in scalars]
        width = len(str(self.steps))
        head = f"{self.name} step {step:>{width}}/{self.steps}  "
        if shown:
            self.echo(head + "  ".join([*shown, *self._clock(step, scalars)]))
        if (worst := summary(scalars)) is not None:
            self.echo(head + "observed  " + worst)

    def _clock(self, step: int, scalars: Scalars) -> list[str]:
        """Return the time since the start and the time left at the latest pace, for
        a log's line, and a warning of what is not finite."""
        out = []
        if "time/step_ms" in scalars:
            spent = time.perf_counter() - self.started
            left = scalars["time/step_ms"] / 1000 * (self.steps - step)
            out.append(f"{_hms(spent)} < {_hms(left)}")
        bad = [k for k in self.FINITE if k in scalars and not math.isfinite(scalars[k])]
        if bad:
            out.append(f"!! not finite: {', '.join(bad)}")
        return out


def _hms(seconds: float) -> str:
    """Return `seconds` as h:mm:ss."""
    if not math.isfinite(seconds):
        return "?"
    m, s = divmod(round(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}"


#: What the console says of a step that observed the model: each metric's range over
#: the layers, with its worst layer, by its label, its format, and whether its worst
#: is its least (an attention's entropy, which collapses) rather than its most.
SUMMARY = {
    "act/rms": ("rms", "{:.3g}", False),
    "act/max_abs": ("max|x|", "{:.3g}", False),
    "act/kurtosis": ("kurtosis", "{:.3g}", False),
    "attn/logit_max": ("logit", "{:.3g}", False),
    "attn/entropy": ("attn entropy", "{:.2f}", True),
    "router/max_load": ("max load", "{:.2f}", False),
    "router/dead": ("dead experts", "{:.0%}", False),
    "grad/norm": ("grad", "{:.2e}", False),
    "update/ratio": ("update/param", "{:.2e}", False),
}


def summary(scalars: Scalars) -> str | None:
    """Return a step's worst layers: each metric of `SUMMARY` from its least to its
    most over the layers, and its worst layer, as ``rms 0.82-41.3 (ffn.11)``; None
    if `scalars` has none."""
    parts = []
    for stat, (label, form, least) in SUMMARY.items():
        values = {k.rsplit("/", 1)[1]: v for k, v in scalars.items() if _of(k, stat)}
        if not values:
            continue
        order = sorted(values, key=lambda k: _sortable(values[k]))
        low, high = (form.format(values[k]) for k in (order[0], order[-1]))
        parts.append(f"{label} {low}-{high} ({order[0] if least else order[-1]})")
    return "  ".join(parts) or None


def _of(name: str, stat: str) -> bool:
    """Return whether `name` is the metric `stat` of a layer: ``act/rms/attn.03``."""
    return name.startswith(stat + "/") and name.count("/") == stat.count("/") + 1


def _sortable(x: float) -> float:
    return math.inf if math.isnan(x) else x


class JSONL(Sink):
    """One line of JSON per write, with its step: metrics.jsonl, the main process's;
    and given a `histograms` file, a line there for a step's histograms, each the
    shares of its bins. A run that resumes at step `start` keeps the lines up to it,
    and drops those of the steps it takes again, which a crash after its checkpoint
    left.

    Args:
        path: The file, which it starts afresh unless `start`.
        start: The step a run resumes at; 0 for a fresh run.
        histograms: The file for histograms, which it opens at the first; None
            leaves them out.
    """

    def __init__(self, path: Path, *, start: int = 0, histograms: Path | None = None):
        self.file = self.histograms_file = None
        self.start, self.histograms_path = start, histograms
        if is_main_process():
            self.file = _open(path, start)

    def write(self, step: int, scalars: Scalars) -> None:
        """Append ``{"step": step, **scalars}`` as a line, and flush it."""
        if self.file is not None:
            self.file.write(json.dumps({"step": step} | dict(scalars)) + "\n")
            self.file.flush()

    def histograms(self, step: int, histograms: Mapping[str, Histogram]) -> None:
        """Append ``{"step": step, name: shares, ...}`` to the histograms' file, the
        edges of their bins under ``"edges"``."""
        if self.histograms_path is None or not is_main_process():
            return
        if self.histograms_file is None:
            self.histograms_file = _open(self.histograms_path, self.start)
        line = {"step": step, "edges": next(iter(histograms.values())).edges.tolist()}
        line |= {name: h.shares.tolist() for name, h in histograms.items()}
        self.histograms_file.write(json.dumps(line) + "\n")
        self.histograms_file.flush()

    def close(self) -> None:
        """Close the files."""
        for f in (self.file, self.histograms_file):
            if f is not None:
                f.close()


def _open(path: Path, start: int) -> TextIO:
    """Open `path` to append to after `start`'s lines, or afresh for a fresh run."""
    if start:
        _drop_after(path, start)
    return path.open("a" if start else "w", encoding="utf-8")


def _drop_after(path: Path, step: int) -> None:
    """Drop the lines of `path` past `step`, and one a crash cut short."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    whole = [line for line in lines if line.endswith("\n")]
    kept = [line for line in whole if json.loads(line)["step"] <= step]
    if len(kept) < len(lines):
        path.write_text("".join(kept), encoding="utf-8")


class LatestMetrics(Sink):
    """The latest value of every metric written, in `metrics`: what
    `bux.train.trainer.train` returns."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def write(self, step: int, scalars: Scalars) -> None:
        """Take each of `scalars` as its metric's latest value."""
        self.metrics |= scalars
