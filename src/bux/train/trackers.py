"""Trackers: the sinks a run's ``log.trackers`` names, each made by the builder
registered for its config's type (`trackers`): `Wandb` and `TensorBoard`. Each gets
every scalar and histogram the run's hooks report, and only a job's first process
writes. Another is a sink and a builder::

    @trackers.register("mlflow")
    def _(c: MLflowConfig, run: ir.Run, folder: Path, start: int) -> Sink:
        return MLflow(c, run)

with its config in `bux.train.config`, among the ``Tracker`` types.
"""

import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from bux.lang import ir
from bux.lang.printer import run_to_source
from bux.train import config as tcfg
from bux.train.registry import Registry
from bux.train.sinks import Histogram, Scalars, Sink, is_main_process

#: The builders of trackers, by config type: ``(config, run, folder, start) -> Sink``,
#: with the run's directory and the step it starts at, 0 or the one it resumes at.
trackers = Registry[Sink]("tracker", "bux.train.trackers.trackers")


def build(run: ir.Run, folder: Path, start: int) -> list[Sink]:
    """Return the sinks of the run's ``log.trackers``, writing for its directory
    `folder` from step `start`."""
    return [trackers.build(c, run, folder, start) for c in run.train.log.trackers]


def _module(name: str, extra: str) -> ModuleType:
    """Import `name`, or say which extra brings it."""
    try:
        return importlib.import_module(name)
    except ImportError as err:
        raise ImportError(
            f"log.trackers has {extra}, which needs the {name} package: "
            f"uv sync --extra {extra}"
        ) from err


class Wandb(Sink):
    """Weights & Biases (`bux.train.config.WandbConfig`): a W&B run for the run,
    named as it is, in a group of its file's runs, with its settings as its config.
    A run that resumes goes on with the W&B run it started, whose id its directory
    keeps in wandb_id; a fresh one starts a W&B run of its own.

    Args:
        config: The tracker's settings.
        run: The run.
        folder: The run's directory, where W&B's files go.
        start: The step the run starts at: 0, or the one it resumes at.
    """

    def __init__(self, config: tcfg.WandbConfig, run: ir.Run, folder: Path, start: int):
        self.run: Any = None
        if not is_main_process():
            return
        self.wandb = _module("wandb", "wandb")
        self.run = self.wandb.init(
            project=config.project,
            entity=config.entity,
            group=config.group or run.name.split("[")[0],
            name=run.name,
            tags=list(config.tags),
            id=_run_id(folder / "wandb_id", start, self.wandb),
            resume="allow",
            dir=str(folder),
            mode=config.mode,
            config=settings(run),
        )

    def write(self, step: int, scalars: Scalars) -> None:
        """Log `scalars` at `step`."""
        if self.run is not None:
            self.run.log(dict(scalars), step=step)

    def histograms(self, step: int, histograms: Mapping[str, Histogram]) -> None:
        """Log each histogram at `step`, as W&B's, over ``log2 |x|``."""
        if self.run is not None:
            made = {
                name: self.wandb.Histogram(np_histogram=(h.shares, h.edges))
                for name, h in histograms.items()
            }
            self.run.log(made, step=step)

    def close(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            self.run.finish()


def _run_id(path: Path, start: int, wandb: ModuleType) -> str:
    """Return the W&B run's id: the one `path` keeps for a run that resumes, else a
    new one, which `path` keeps from now on."""
    if start and path.exists():
        return path.read_text(encoding="utf-8").strip()
    made = wandb.util.generate_id()
    path.write_text(made + "\n", encoding="utf-8")
    return made


def settings(run: ir.Run) -> dict[str, Any]:
    """Return a tracker's config of `run`: its id, its sweep's choices, its training
    settings, its model's settings but its graph, and the run as one .bux file."""
    arch = run.arch.model_dump(mode="json", exclude={"graph"})
    return {
        "id": run.id,
        "choices": list(run.choices),
        "train": run.train.model_dump(mode="json"),
        "arch": arch,
        "source": run_to_source(run),
    }


class TensorBoard(Sink):
    """TensorBoard (`bux.train.config.TensorBoardConfig`): event files in `folder`,
    a scalar each metric and a histogram each histogram, over ``log2 |x|``. A run that
    resumes at `start` drops the events past it.

    Args:
        folder: Where the event files go: the run's directory's tensorboard/.
        start: The step the run starts at: 0, or the one it resumes at.
    """

    def __init__(self, folder: Path, start: int):
        self.writer: Any = None
        if is_main_process():
            tensorboard = _module("tensorboardX", "tensorboard")
            purge = start + 1 if start else None  # the steps after the checkpoint
            self.writer = tensorboard.SummaryWriter(str(folder), purge_step=purge)

    def write(self, step: int, scalars: Scalars) -> None:
        """Add each scalar at `step`, and flush them."""
        if self.writer is not None:
            for name, value in scalars.items():
                self.writer.add_scalar(name, value, step)
            self.writer.flush()

    def histograms(self, step: int, histograms: Mapping[str, Histogram]) -> None:
        """Add each histogram at `step`, its bins' shares as their counts."""
        if self.writer is None:
            return
        for name, h in histograms.items():
            mid = (h.edges[:-1] + h.edges[1:]) / 2
            self.writer.add_histogram_raw(
                name,
                min=float(h.edges[0]),
                max=float(h.edges[-1]),
                num=float(h.shares.sum()),
                sum=float((mid * h.shares).sum()),
                sum_squares=float((mid**2 * h.shares).sum()),
                bucket_limits=h.edges[1:].tolist(),
                bucket_counts=h.shares.tolist(),
                global_step=step,
            )
        self.writer.flush()

    def close(self) -> None:
        """Close the event files."""
        if self.writer is not None:
            self.writer.close()


@trackers.register("wandb")
def _(c: tcfg.WandbConfig, run: ir.Run, folder: Path, start: int) -> Sink:
    """Log to Weights & Biases (`Wandb`)."""
    return Wandb(c, run, folder, start)


@trackers.register("tensorboard")
def _(c: tcfg.TensorBoardConfig, run: ir.Run, folder: Path, start: int) -> Sink:
    """Write TensorBoard's event files under the run's tensorboard/
    (`TensorBoard`)."""
    return TensorBoard(folder / "tensorboard", start)
