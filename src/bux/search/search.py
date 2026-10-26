"""A search's trials, kept in its directory, and the runs they make for the launcher.

A search's directory, ``<out>/<name>-search-<key>``, beside its trials' runs, holds::

    trials.jsonl  a line for each trial proposed, with its values and bracket, and
                  one for each rung it started
    best.bux      the run.bux of the best trial so far, at the highest rung

Its key hashes what it searches and how: its dimensions, method, halving and seed. So
a search launched again finds its trials and goes on, one given more trials adds them,
and a search over another space is a search of its own. Each rung of a trial is a run
(`bux.lang.compiler.Space.trial`), which the launcher trains as it trains any
(`bux.launch.supervise`) and whose objective is read from its metrics.jsonl when it
ends: a run already done is read, not trained again. A trial that failed, or whose
point a setting rejects, has an objective of NaN, and stays failed when the search is
launched again; one that a signal stopped resumes.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float

from bux.lang import ir
from bux.lang.compiler import Space as Compiled
from bux.lang.syntax import BuxError
from bux.launch import Queue, status
from bux.search import halving, methods
from bux.search.halving import Trial
from bux.search.space import Space
from bux.train import config as tcfg
from bux.train.trainer import directory


def objective(config: tcfg.ObjectiveConfig, folder: Path) -> float:
    """Return the objective of the run in `folder`: its weighted sum of the latest
    value of each metric in metrics.jsonl; NaN if one is missing or not a number."""
    latest: dict[str, float] = {}
    path = folder / "metrics.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            latest |= {k: v for k, v in json.loads(line).items() if _number(v)}
    total = config.constant
    for term in config.terms:
        total += term.weight * latest.get(term.metric, math.nan)
    return total if math.isfinite(total) else math.nan


def _number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


class Search:
    """The search of one run of a file, a `bux.lang.compiler.Space`: its trials, the
    runs they train next, and what they found.

    Args:
        space: The run and its dimensions.
        out: Where run directories go; the search's own goes there too.
        echo: Where it says what went wrong with a trial.

    Attributes:
        space: The run and its dimensions.
        config: Its ``search`` setting.
        goal: Its objective (`bux.train.config.objective`).
        dims: Its dimensions as its method sees them.
        folder: Its directory.
        trials: Its trials, in the order proposed.
    """

    def __init__(
        self, space: Compiled, out: str | Path, echo: Callable[[str], None] = print
    ):
        self.space, self.out, self.echo = space, Path(out), echo
        self.config = space.run.train.search
        self.goal = tcfg.objective(space.run.train)
        self.dims = Space.of(space.dimensions)
        slug = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", space.run.name).strip("_")
        self.folder = self.out / f"{slug}-search-{self.key}"
        self.trials: list[Trial] = []
        self.runs: dict[str, tuple[Trial, int]] = {}  # by run id: its trial and rung
        self.redo: list[tuple[Trial, int]] = []  # rungs started that did not end
        self.names: dict[tuple[int, int], str] = {}  # each trial's run at each rung
        self._load()

    @property
    def key(self) -> str:
        """Ten hex digits of a hash of what it searches and how."""
        what = {
            "dimensions": {n: d.model_dump() for n, d in zip(*self._dims())},
            "method": self.config.method.model_dump(),
            "halving": self.config.halving and self.config.halving.model_dump(),
            "seed": self.config.seed,
        }
        blob = json.dumps(what, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def _dims(self) -> tuple[tuple[str, ...], tuple[tcfg.Distribution, ...]]:
        return self.dims.names, self.dims.dims

    @property
    def budget(self) -> int:
        """Return how many trials it proposes: one, the run, without dimensions."""
        return self.config.trials if self.dims.names else 1

    def best(self) -> Trial | None:
        """Return its trial with the least objective at the highest rung that has
        one that is a number; None if none has (`bux.search.halving.best`)."""
        return halving.best(self.trials)

    @property
    def top(self) -> int:
        """Return its highest rung: the run's own steps."""
        return halving.rungs(self.config.halving) - 1

    # --- Its trials, on disk ---------------------------------------------------------

    def _journal(self, **line: object) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / "trials.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def _load(self) -> None:
        """Read its trials back, and what their runs found; a rung that did not end
        trains again first."""
        path = self.folder / "trials.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        for line in map(json.loads, lines):
            if "values" in line:
                self.trials.append(
                    Trial(line["trial"], line["values"], line["bracket"])
                )
            else:
                self.trials[line["trial"]].started.append(line["rung"])
        for trial in self.trials:
            for rung in trial.started:
                if (run := self._compiled(trial, rung)) is None:
                    continue
                state = status(directory(run, self.out)).get("state")
                if state in ("done", "failed"):
                    self._finished(trial, rung, run)
                else:
                    self.redo.append((trial, rung))

    # --- What trains next ------------------------------------------------------------

    def next(self) -> ir.Run | None:
        """Return the next run to train: a rung that did not end, a trial up a rung,
        or a new trial; None if none can start until a running one ends, or the
        search is done. A run already done is read, not returned."""
        while True:
            if self.redo:
                trial, rung = self.redo.pop(0)
            elif self._full():
                return None
            elif job := halving.next_job(self.trials, self.config.halving, self.budget):
                trial, rung = job.trial or self._propose(job.rung), job.rung
                trial.started.append(rung)
                self._journal(trial=trial.number, rung=rung)
            else:
                return None
            run = self._compiled(trial, rung)
            if run is None:
                continue
            if status(directory(run, self.out)).get("state") == "done":
                self._finished(trial, rung, run)
                continue
            return run

    def later(self, run: ir.Run) -> None:
        """Put `run`, which `next` returned but could not start, first in line."""
        self.redo.insert(0, self.runs[run.id])

    def _full(self) -> bool:
        """Whether it trains as many trials at once as its ``parallel`` allows."""
        running = sum(t.running is not None for t in self.trials)
        return self.config.parallel is not None and running >= self.config.parallel

    def _propose(self, bracket: int) -> Trial:
        """Return a new trial, at the point its method proposes, and keep it."""
        values: dict[str, float] = {}
        if self.dims.names:
            point = methods.propose(self.config.method, self.dims, self.history())
            values = self.dims.decode(point)
        trial = Trial(len(self.trials), values, bracket)
        self.trials.append(trial)
        self._journal(trial=trial.number, values=values, bracket=bracket)
        return trial

    def history(self) -> methods.History:
        """Return what its method knows: the trials' objectives at the highest rung
        where enough of them have one (BOHB's rule, Falkner et al., 2018), and the
        trials training."""
        rung = self._modeled()
        done = [t for t in self.trials if rung in t.results]
        pending = [t for t in self.trials if t.running is not None]
        return methods.History(
            points=self._points(done),
            values=np.array([t.results[rung] for t in done], dtype=np.float64),
            pending=self._points(pending),
            proposed=len(self.trials),
            seed=self.config.seed,
        )

    def _modeled(self) -> int:
        """Return the highest rung with a number for at least one trial more than it
        has dimensions; else the rung with the most."""
        counts = [
            sum(not math.isnan(t.results.get(rung, math.nan)) for t in self.trials)
            for rung in range(self.top + 1)
        ]
        enough = [r for r, n in enumerate(counts) if n > len(self.dims.names)]
        return enough[-1] if enough else int(np.argmax(counts))

    def _points(self, trials: list[Trial]) -> Float[np.ndarray, "n d"]:
        rows = [self.dims.encode(t.values) for t in trials]
        return np.array(rows).reshape(len(rows), len(self.dims.names))

    def _compiled(self, trial: Trial, rung: int) -> ir.Run | None:
        """Return the run of `trial` at `rung`; None, and NaN for the rung, if a
        setting rejects its point."""
        steps = halving.steps(self.config.halving, self.space.run.train.steps, rung)
        try:
            run = self.space.trial(trial.values, steps)
        except BuxError as err:
            first = str(err).splitlines()[0].removeprefix("error: ")
            self.echo(f"{self.space.run.name}: trial {trial.number} failed: {first}")
            trial.results[rung] = math.nan
            return None
        self.runs[run.id] = trial, rung
        self.names[trial.number, rung] = run.name
        return run

    # --- What it found ---------------------------------------------------------------

    def ended(self, run: ir.Run) -> None:
        """Read what `run`, one of its trials' rungs, found, once its worker ended:
        its objective if it is done, NaN if it failed. A run a signal stopped stays
        unfinished, to resume."""
        trial, rung = self.runs[run.id]
        if status(directory(run, self.out)).get("state") in ("done", "failed"):
            self._finished(trial, rung, run)

    def _finished(self, trial: Trial, rung: int, run: ir.Run) -> None:
        folder = directory(run, self.out)
        done = status(folder).get("state") == "done"
        trial.results[rung] = objective(self.goal, folder) if done else math.nan
        if halving.best(self.trials) is trial and (folder / "run.bux").exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "best.bux").write_text(
                (folder / "run.bux").read_text(encoding="utf-8"), encoding="utf-8"
            )

    def table(self) -> str:
        """Return its line, and a line per trial, the best first: each trial's
        highest rung finished, its objective there, and its run's name."""
        best = self.best()
        head = f"search {self.space.run.name}: {len(self.trials)} of {self.budget}"
        head += f" trials{_states(self.trials)}; objective = {_written(self.goal)}"
        if best is not None and best.top is not None:
            head += f"; best {best.top[1]:.4f}, trial {best.number}"
        rows = [("trial", "rung", "objective", "run")]
        rows += [self._row(t) for t in sorted(self.trials, key=_rank)]
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        lines = [
            "  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[3] for r in rows
        ]
        return head + "\n" + "\n".join(lines)

    def _row(self, t: Trial) -> tuple[str, str, str, str]:
        if t.top is None:
            return str(t.number), "-", "-", self.names.get((t.number, t.bracket), "-")
        rung, value = t.top
        shown = "failed" if math.isnan(value) else f"{value:.4f}"
        name = self.names.get((t.number, rung), "-")
        return str(t.number), f"{rung}/{self.top}", shown, name


def _rank(t: Trial) -> tuple[float, float, int]:
    """The order of the table: the highest rung first, then the least objective."""
    if t.top is None:
        return (1.0, math.inf, t.number)
    rung, value = t.top
    return (-rung, math.inf if math.isnan(value) else value, t.number)


def _states(trials: list[Trial]) -> str:
    running = sum(t.running is not None for t in trials)
    failed = sum(t.top is not None and math.isnan(t.top[1]) for t in trials)
    said = [f"{running} training"] if running else []
    said += [f"{failed} failed"] if failed else []
    return f" ({', '.join(said)})" if said else ""


def _written(goal: tcfg.ObjectiveConfig) -> str:
    from bux.lang.printer import value_to_source

    return value_to_source(goal)


class Searches(Queue):
    """The searches of a file's runs, as a queue of runs for the launcher's workers
    (`bux.launch.supervise`): each run proposed only as it can start, the searches
    taking turns.

    Args:
        searches: The searches.
    """

    def __init__(self, searches: Sequence[Search]):
        super().__init__([])
        self.searches = list(searches)
        self.turn = 0

    def take[S](
        self, free: Sequence[S], needs: Callable[[ir.Run], int]
    ) -> tuple[ir.Run, list[S]] | None:
        """Return the next run of the first search in turn that has one to start on
        some of the `free` GPUs, and the first of them that it needs."""
        n = len(self.searches)
        for i in range(n):
            search = self.searches[(self.turn + i) % n]
            if needs(search.space.run) > len(free) or (run := search.next()) is None:
                continue
            if needs(run) > len(free):
                search.later(run)
                continue
            self.turn = (self.turn + i + 1) % n
            return run, list(free[: needs(run)])
        return None

    def ended(self, run: ir.Run) -> None:
        """Hand `run`'s end to the search it is a trial of."""
        for search in self.searches:
            if run.id in search.runs:
                search.ended(run)
                return

    def summary(self) -> str:
        """Return each search's table (`Search.table`)."""
        return "\n\n".join(search.table() for search in self.searches)
