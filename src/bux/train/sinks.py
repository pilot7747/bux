"""Sinks: where hooks' scalars go. Scalars are metrics by name, as numbers, that a hook
returned after a step (`bux.train.hooks`).

A run writes to three: `Console`, a line per log; `JSONL`, the run's metrics.jsonl;
and `LatestMetrics`, the latest value of each, which `bux.train.trainer.train`
returns. A tracker is a sink::

    class Wandb(Sink):
        def __init__(self, run: ir.Run):
            self.run = wandb.init(project="bux", name=run.name, config=run.model_dump())

        def write(self, step: int, scalars: Scalars) -> None:
            self.run.log(scalars, step=step)

        def close(self) -> None:
            self.run.finish()

    train(run, sinks=[Wandb(run)])

In a job of many processes (`bux.launch`), every process computes all scalars and
only the main one writes them, to the console and metrics.jsonl. A sink of your own
that writes checks `is_main_process`.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

import jax

Scalars = Mapping[str, float]  # metrics by name, as numbers on the host


def is_main_process() -> bool:
    """Return whether this is process 0 of a job, or the only one, without touching a
    device."""
    return not jax.distributed.is_initialized() or jax.process_index() == 0


class Sink:
    """Where hooks' scalars go. A subclass overrides `write`, and `close` if it holds
    something open; the trainer calls them in order, a step's scalars once the next
    step is launched."""

    def write(self, step: int, scalars: Scalars) -> None:
        """Take what the hooks reported after `step`.

        Args:
            step: The steps taken when the hooks ran.
            scalars: Their metrics by name, already on the host.
        """
        raise NotImplementedError

    def close(self) -> None:
        """After the last scalars, or when training stops."""


class Console(Sink):
    """One line per write, with the metrics it has a format for, in their order:
    ``smoke step 30/50  loss 6.3513  lr 3.00e-03  grad 0.554  57,163 tok/s``.

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
        "lr": "lr {:.2e}",
        "grad_norm": "grad {:.3f}",
        "tokens_per_s": "{:,.0f} tok/s",
        "mfu": "mfu {:.1%}",
        "eval_loss": "eval loss {:.4f}",
    }

    def __init__(
        self,
        name: str,
        steps: int,
        echo: Callable[[str], None] = print,
        formats: Mapping[str, str] = FORMATS,
    ):
        self.name, self.steps, self.echo, self.formats = name, steps, echo, formats

    def write(self, step: int, scalars: Scalars) -> None:
        """Print a line of the scalars it has a format for, on the main process; none
        if it has none."""
        shown = [f.format(scalars[k]) for k, f in self.formats.items() if k in scalars]
        if shown and is_main_process():
            self.echo(f"{self.name} step {step}/{self.steps}  " + "  ".join(shown))


class JSONL(Sink):
    """One line of JSON per write, with its step: metrics.jsonl, the main process's.
    A run that resumes at step `start` keeps the lines up to it, and drops those of
    the steps it takes again, which a crash after its checkpoint left.

    Args:
        path: The file, which it starts afresh unless `start`.
        start: The step a run resumes at; 0 for a fresh run.
    """

    def __init__(self, path: Path, *, start: int = 0):
        self.file = None
        if is_main_process():
            if start:
                _drop_after(path, start)
            self.file = path.open("a" if start else "w", encoding="utf-8")

    def write(self, step: int, scalars: Scalars) -> None:
        """Append ``{"step": step, **scalars}`` as a line, and flush it."""
        if self.file is not None:
            self.file.write(json.dumps({"step": step} | dict(scalars)) + "\n")
            self.file.flush()

    def close(self) -> None:
        """Close the file."""
        if self.file is not None:
            self.file.close()


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
