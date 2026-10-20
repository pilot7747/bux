"""Sinks: where reports go. A report is metrics by name, as numbers, that a hook
returned after a step (``bux.train.hooks``).

A tracker is a sink:

    class Wandb(Sink):
        def __init__(self, run: ir.Run):
            self.run = wandb.init(project="bux", name=run.name, config=run.model_dump())

        def write(self, step: int, report: Report) -> None:
            self.run.log(report, step=step)

        def close(self) -> None:
            self.run.finish()

    train(run, sinks=[Wandb(run)])

Of a job of many processes (``bux.launch``), every one computes every report, and
the first alone writes them: the console and metrics.jsonl are its. A sink of your
own that writes asks ``primary()``.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

import jax

Report = Mapping[str, float]


def primary() -> bool:
    """Whether this process writes what a job of many computes: the first, or the
    one alone, which this tells without touching a device."""
    return not jax.distributed.is_initialized() or jax.process_index() == 0


class Sink:
    def write(self, step: int, report: Report) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """After the last report, or when training stops."""


class Console(Sink):
    """A line a report, with the metrics it has a format for, in their order."""

    FORMATS: ClassVar[dict[str, str]] = {
        "loss": "loss {:.4f}",
        "lr": "lr {:.2e}",
        "grad_norm": "grad {:.3f}",
        "tokens_per_s": "{:,.0f} tok/s",
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

    def write(self, step: int, report: Report) -> None:
        shown = [f.format(report[k]) for k, f in self.formats.items() if k in report]
        if shown and primary():
            self.echo(f"{self.name} step {step}/{self.steps}  " + "  ".join(shown))


class JSONL(Sink):
    """A line of JSON a report, with its step: metrics.jsonl, the first process's."""

    def __init__(self, path: Path, *, append: bool = False):
        mode = "a" if append else "w"
        self.file = path.open(mode, encoding="utf-8") if primary() else None

    def write(self, step: int, report: Report) -> None:
        if self.file is not None:
            self.file.write(json.dumps({"step": step} | dict(report)) + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class Latest(Sink):
    """The latest value of every metric reported."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def write(self, step: int, report: Report) -> None:
        self.metrics |= report
