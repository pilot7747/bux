"""Sinks: where hooks' scalars go. Scalars are metrics by name, as numbers, that a hook
returned after a step (``bux.train.hooks``).

A tracker is a sink:

    class Wandb(Sink):
        def __init__(self, run: ir.Run):
            self.run = wandb.init(project="bux", name=run.name, config=run.model_dump())

        def write(self, step: int, scalars: Scalars) -> None:
            self.run.log(scalars, step=step)

        def close(self) -> None:
            self.run.finish()

    train(run, sinks=[Wandb(run)])

In a job of many processes (``bux.launch``), every process computes all scalars and
only the main one writes them, to the console and metrics.jsonl. A sink of your own
that writes checks ``is_main_process()``.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

import jax

Scalars = Mapping[str, float]


def is_main_process() -> bool:
    """Return whether this is process 0 of a job, or the only one, without touching a
    device."""
    return not jax.distributed.is_initialized() or jax.process_index() == 0


class Sink:
    def write(self, step: int, scalars: Scalars) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """After the last scalars, or when training stops."""


class Console(Sink):
    """One line per write, with the metrics it has a format for, in their order."""

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

    def write(self, step: int, scalars: Scalars) -> None:
        shown = [f.format(scalars[k]) for k, f in self.formats.items() if k in scalars]
        if shown and is_main_process():
            self.echo(f"{self.name} step {step}/{self.steps}  " + "  ".join(shown))


class JSONL(Sink):
    """One line of JSON per write, with its step: metrics.jsonl, the main process's."""

    def __init__(self, path: Path, *, append: bool = False):
        mode = "a" if append else "w"
        self.file = path.open(mode, encoding="utf-8") if is_main_process() else None

    def write(self, step: int, scalars: Scalars) -> None:
        if self.file is not None:
            self.file.write(json.dumps({"step": step} | dict(scalars)) + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class LatestMetrics(Sink):
    """The latest value of every metric written."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def write(self, step: int, scalars: Scalars) -> None:
        self.metrics |= scalars
