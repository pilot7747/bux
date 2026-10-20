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
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

Report = Mapping[str, float]


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
        if shown:
            self.echo(f"{self.name} step {step}/{self.steps}  " + "  ".join(shown))


class JSONL(Sink):
    """A line of JSON a report, with its step: metrics.jsonl."""

    def __init__(self, path: Path, *, append: bool = False):
        self.file = path.open("a" if append else "w", encoding="utf-8")

    def write(self, step: int, report: Report) -> None:
        self.file.write(json.dumps({"step": step} | dict(report)) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class Latest(Sink):
    """The latest value of every metric reported."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def write(self, step: int, report: Report) -> None:
        self.metrics |= report
