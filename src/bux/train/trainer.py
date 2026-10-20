"""Trains a compiled Run with the reference Flax NNX backend, on the default device.

A run's directory, ``<out>/<name>-<id>``, holds:

    run.bux        the run as one self-contained file, which `bux train` takes as is
    run.json       its IR
    metrics.jsonl  a line of metrics every `log.every` steps, and at every eval
    checkpoints/   with a `checkpoint` setting; a run there resumes from the latest

Its name says which alternatives it took, and its id hashes what it computes, so a
changed setting gets a fresh directory and an unchanged one finds its checkpoints.

``train`` builds a run out of blocks, each of which works without the others: a
``Trainer`` (a model, its optimizer and its objective), the run's batches, hooks for
its log, eval and checkpoint settings, and sinks for what they report. So a script
of your own trains with its own blocks, or with the run's and one more:

    trainer = Trainer(model, optax.adamw(1e-3), objective(run.train.loss, run.arch))
    trainer.fit(batches(...), steps=1000, hooks=[...], sinks=[Console("mine", 1000)])
"""

import contextlib
import functools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float

from bux.architecture import Model
from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.lang.printer import run_to_source
from bux.train.data import Batch, batches
from bux.train.hooks import Checkpoint, Eval, Hook, Log, Metrics
from bux.train.objective import Objective, objective
from bux.train.optim import optimizer, peak, schedule
from bux.train.sinks import JSONL, Console, Latest, Sink


def forward(model: Model, batch: Batch) -> list[Float[Array, "b t vocab"]]:
    """Every head's logits on `batch`, each document in a row on its own."""
    return model.outputs(jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids))


class Trainer:
    """A model, the optimizer that trains it with `tx`, and the objective it
    minimizes, `step` steps in.

    `train_step` and `eval_step` are compiled once, so a subclass that overrides them
    changes what a step does (accumulates gradients, keeps an average of the weights,
    skips a spike) and keeps the loop, hooks and checkpoints. They get the model and
    the optimizer as arguments, which is how nnx.jit sees them change.
    """

    def __init__(
        self, model: Model, tx: optax.GradientTransformation, objective: Objective
    ):
        self.model, self.objective = model, objective
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        self.step = 0
        self.stopping = False  # set, by a signal handler say, to stop after this step
        self._train = nnx.jit(self.train_step)
        self._eval = nnx.jit(self.eval_step)

    def train_step(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        """The objective's gradient on `batch`, applied: the objective's terms, and
        the gradient's norm."""

        def loss(model: Model) -> tuple[Float[Array, ""], Metrics]:
            return self.objective(forward(model, batch), batch)

        (_, metrics), grads = nnx.value_and_grad(loss, has_aux=True)(model)
        optimizer.update(model, grads)
        return metrics | {"grad_norm": optax.tree.norm(grads)}

    def eval_step(self, model: Model, batch: Batch) -> Float[Array, ""]:
        """The first head's loss on `batch`."""
        return self.objective(forward(model, batch), batch)[1]["loss"]

    def update(self, batch: Batch) -> Metrics:
        """Take a step on `batch`."""
        metrics = self._train(self.model, self.optimizer, jax.device_put(batch))
        self.step += 1
        return metrics

    def evaluate(self, batches: Iterable[Batch]) -> float:
        """The mean of `eval_step` over `batches`, in eval mode."""
        self.model.eval()
        try:
            losses = [float(self._eval(self.model, jax.device_put(b))) for b in batches]
        finally:
            self.model.train()
        return sum(losses) / len(losses)

    def state(self) -> nnx.State:
        """What a checkpoint keeps: the model's state and the optimizer's."""
        return nnx.state((self.model, self.optimizer))

    def load(self, state: nnx.State | Mapping[str, Any], step: int) -> None:
        """Take up `state`, from `state()` or a checkpoint, at `step`."""
        nnx.update((self.model, self.optimizer), state)
        self.step = step

    def fit(
        self,
        data: Iterator[Batch],
        steps: int,
        hooks: Sequence[Hook] = (),
        sinks: Sequence[Sink] = (),
    ) -> None:
        """Train on `data` to step `steps`, or until `stopping` is set, after which
        every hook's `stop` runs. After every step, the hooks due run in order, and
        what they report goes to every sink. All of them close at the end, or when
        training stops."""
        with contextlib.ExitStack() as stack:
            for part in (*sinks, *hooks):
                stack.callback(part.close)
            for hook in hooks:
                hook.start(self)
            while self.step < steps and not self.stopping:
                metrics = self.update(next(data))
                for hook in hooks:
                    if hook.due(self.step, steps) and (report := hook(self, metrics)):
                        for sink in sinks:
                            sink.write(self.step, report)
            if self.step < steps:
                for hook in hooks:
                    hook.stop(self)


def directory(run: ir.Run, out: str | Path) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", run.name).strip("_")
    return Path(out) / f"{slug}-{run.id}"


def train(
    run: ir.Run,
    out: str | Path = "runs",
    *,
    cache: str | Path = DEFAULT_CACHE,
    echo: Callable[[str], None] = print,
    hooks: Sequence[Hook] = (),
    sinks: Sequence[Sink] = (),
) -> dict[str, float]:
    """Train `run` to its last step, writing to its directory under `out`, after
    preparing the datasets it reads under `cache`; the latest value of every metric
    reported. `hooks` run after the run's own, and `sinks` get what all of them
    report, as the console and metrics.jsonl do."""
    t, vocab = run.train, run.arch.vocab_size
    if vocab is None:
        raise ValueError("a model without a vocabulary cannot be trained on tokens")
    for job in jobs(t):
        prepare(job, cache, echo=echo)
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.bux").write_text(run_to_source(run), encoding="utf-8")
    (folder / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")

    model = Model(run.arch, rngs=nnx.Rngs(t.seed))
    multiplier = schedule(t.schedule, t.steps)
    tx = optimizer(t.optimizer, multiplier, model, t.clip)
    trainer = Trainer(model, tx, objective(t.loss, run.arch))

    lr = peak(t.optimizer.lr, multiplier)
    own: list[Hook] = [Log(every=t.log.every, lr=lr, tokens=t.batch * t.seq_len)]
    if t.eval is not None:
        held_out = functools.partial(
            batches, t.eval.data, t, vocab, cache=cache, stream=1
        )
        own.append(Eval(held_out, every=t.eval.every, batches=t.eval.batches))
    if (c := t.checkpoint) is not None:
        checkpoint = Checkpoint(folder / "checkpoints", every=c.every, keep=c.keep)
        if checkpoint.restore(trainer):
            echo(f"{run.name}: resuming at step {trainer.step}")
        own.append(checkpoint)

    latest = Latest()
    trainer.fit(
        batches(t.data, t, vocab, cache=cache, start=trainer.step),
        t.steps,
        hooks=[*own, *hooks],
        sinks=[
            Console(run.name, t.steps, echo),
            JSONL(folder / "metrics.jsonl", append=trainer.step > 0),
            latest,
            *sinks,
        ],
    )
    return latest.metrics
