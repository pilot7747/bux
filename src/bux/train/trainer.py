"""Trains a compiled Run with the reference Flax NNX backend, on the devices its mesh
takes (``bux.parallel``).

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
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from bux import parallel
from bux.architecture import Model
from bux.architecture.registry import Loss
from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.lang.printer import run_to_source
from bux.train.data import Batch, batches
from bux.train.hooks import Checkpoint, Eval, Hook, Log, Metrics
from bux.train.objective import Objective, objective, targets
from bux.train.optim import Schedule, optimizer, peak, schedule
from bux.train.sinks import JSONL, Console, Latest, Sink, primary


def forward(model: Model, batch: Batch) -> list[Float[Array, "b t vocab"]]:
    """Every head's logits on `batch`, each document in a row on its own."""
    return model.outputs(jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids))


def training_forward(
    model: Model, batch: Batch
) -> tuple[list[Float[Array, "b t vocab"]], list[Loss]]:
    """`forward` in a training step: with the losses the layers report."""
    tokens, positions = jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids)
    return model.training_outputs(tokens, positions)


def cast[M: nnx.Module](model: M, dtype: DTypeLike) -> M:
    """A copy of `model`, its floating-point parameters in `dtype` and the rest of its
    state as it is. Gradients through it reach the parameters in their own dtype."""
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    return nnx.merge(graphdef, params, rest)


@runtime_checkable
class Settles(Protocol):
    """A module that learns from a step's batch other than by gradients, as an MoE
    moves its router bias. A training step sets `deferred`: the forwards of all its
    microbatches count what they see, and `settle` applies it once, after them.
    Outside a step, a training forward applies it at once."""

    deferred: bool

    def settle(self) -> None: ...


def settle(model: nnx.Module) -> None:
    """Every module's `settle`, once a step, after its forwards."""
    for _, m in nnx.iter_modules(model):
        if isinstance(m, Settles):
            m.settle()


def shares(batch: Batch, n: int) -> Float[Array, " n"]:
    """Each of `n` microbatches' share of the targets of `batch` (the first head's),
    or 1 / n each if it has none."""
    counts = parallel.microbatches(targets(batch), n).sum((1, 2))
    total = counts.sum()
    return jnp.where(total > 0, counts / jnp.maximum(total, 1), 1 / n)


def weigh(
    total: Float[Array, ""],
    metrics: Metrics,
    reported: Sequence[Loss],
    share: Float[Array, ""],
    n: int,
) -> tuple[Float[Array, ""], Metrics]:
    """A microbatch's part of the objective of a batch of `n`: the objective's terms,
    means over targets, by the microbatch's `share` of the targets; what the layers
    report, by its share of the rows, 1 / n."""
    theirs = sum((weight * value for _, weight, value in reported), jnp.zeros(()))
    rows = {name for name, _, _ in reported}
    metrics = {k: v * (1 / n if k in rows else share) for k, v in metrics.items()}
    return share * (total - theirs) + theirs / n, metrics


def _on_shards(state: nnx.State) -> nnx.State:
    """`state`, constrained to the shardings of its variables, on a mesh."""
    if parallel.active() is None:
        return state
    return jax.lax.with_sharding_constraint(state, parallel.shardings(state))


class Trainer:
    """A model, the optimizer that trains it with `tx`, and the objective it
    minimizes, `step` steps in.

    `train_step` and `eval_step` are compiled once, so a subclass that overrides them
    changes what a step does (keeps an average of the weights, skips a spike) and
    keeps the loop, hooks and checkpoints. They get the model and the optimizer as
    arguments, which is how nnx.jit sees them change.

    Both compute in `compute`, the model's parameters cast to it inside the loss, so
    that the parameters, their gradients and the optimizer's state stay as they are
    (float32). A step takes its batch in `microbatches` (gradient accumulation): a
    scan over them, one gradient, one update.

    Made inside ``parallel.use``, it puts its state on the shardings the layout gives
    it, and every step keeps it there.
    """

    def __init__(
        self,
        model: Model,
        tx: optax.GradientTransformation,
        objective: Objective,
        *,
        compute: DTypeLike = jnp.float32,
        microbatches: int = 1,
    ):
        self.model, self.objective = model, objective
        self.compute, self.microbatches = jnp.dtype(compute), microbatches
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        parallel.place(self.model, self.optimizer)
        self.step = 0
        self.stopping = False  # set, by a signal handler say, to stop after this step
        self._train = nnx.jit(self._placed_step)
        self._eval = nnx.jit(self.eval_step)

    def _placed_step(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        metrics = self.train_step(model, optimizer, batch)
        parallel.place(model, optimizer)
        return metrics

    def train_step(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        """The objective's gradient on `batch`, applied: the objective's terms, and
        the gradient's norm."""
        grads, metrics = self.gradients(model, batch)
        optimizer.update(model, grads)
        return metrics | {"grad_norm": optax.tree.norm(grads)}

    def gradients(self, model: Model, batch: Batch) -> tuple[nnx.State, Metrics]:
        """The objective's gradient on `batch`, and its terms: the sums of
        `microbatches` parts (``weigh``), so that the loss is the batch's. The
        forwards update the model's other state in turn, and then every module
        ``settle``s."""
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        if self.microbatches == 1:
            grads, metrics, rest = self._microbatch(graphdef, params, rest, batch)
        else:
            grads, metrics, rest = self._accumulate(graphdef, params, rest, batch)
        nnx.update(model, rest)
        settle(model)
        return grads, metrics

    def _accumulate(
        self, graphdef: nnx.GraphDef, params: nnx.State, rest: nnx.State, batch: Batch
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """`_microbatch` on each of the batch's microbatches in turn, summed."""

        def one(
            carry: tuple[nnx.State, nnx.State],
            part: tuple[Batch, Float[Array, ""]],
        ) -> tuple[tuple[nnx.State, nnx.State], Metrics]:
            total, rest = carry
            grads, metrics, rest = self._microbatch(graphdef, params, rest, *part)
            return (jax.tree.map(jnp.add, total, grads), rest), metrics

        n = self.microbatches
        parts = (parallel.microbatches(batch, n), shares(batch, n))
        zeros = _on_shards(jax.tree.map(jnp.zeros_like, params))
        (grads, rest), metrics = jax.lax.scan(one, (zeros, rest), parts)
        return grads, {k: v.sum() for k, v in metrics.items()}, rest

    def _microbatch(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        rest: nnx.State,
        batch: Batch,
        share: Float[Array, ""] | None = None,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """The objective's gradient on `batch` and its terms, computed in `compute`,
        as the part of the step's that `share` of its targets makes (``weigh``); and
        the model's other state after the forward."""

        def loss(params: nnx.State) -> tuple[Float[Array, ""], tuple[Metrics, Any]]:
            model = cast(nnx.merge(graphdef, params, rest, copy=True), self.compute)
            model.set_attributes(deferred=True, raise_if_not_found=False)
            logits, reported = training_forward(model, batch)
            total, metrics = self.objective(logits, batch, reported)
            if share is not None:
                n = self.microbatches
                total, metrics = weigh(total, metrics, reported, share, n)
            return total, (metrics, nnx.state(model, nnx.Not(nnx.Param)))

        (_, (metrics, rest)), grads = jax.value_and_grad(loss, has_aux=True)(params)
        return grads, metrics, rest

    def eval_step(self, model: Model, batch: Batch) -> Float[Array, ""]:
        """The first head's loss on `batch`, computed in `compute`."""
        logits = forward(cast(model, self.compute), batch)
        return self.objective(logits, batch)[1]["loss"]

    def update(self, batch: Batch) -> Metrics:
        """Take a step on `batch`."""
        metrics = self._train(self.model, self.optimizer, parallel.rows(batch))
        self.step += 1
        return metrics

    def evaluate(self, batches: Iterable[Batch]) -> float:
        """The mean of `eval_step` over `batches`, in eval mode."""
        self.model.eval()
        try:
            losses = [float(self._eval(self.model, parallel.rows(b))) for b in batches]
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
    if primary():  # of a job of many processes, the first writes the run's files
        (folder / "run.bux").write_text(run_to_source(run), encoding="utf-8")
        (folder / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )

    with parallel.use(t.mesh, t.layout, t.methods):
        parallel.check(t.batch, t.seq_len)
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(t.seed)))
        multiplier = schedule(t.schedule, t.steps)
        tx = optimizer(t.optimizer, multiplier, model, t.clip)
        trainer = Trainer(
            model,
            tx,
            objective(t.loss, run.arch),
            compute=t.precision.compute,
            microbatches=_accumulated(t.methods.microbatches),
        )
        read = functools.partial(batches, t=t, vocab_size=vocab, cache=cache)
        lr = peak(t.optimizer.lr, multiplier)
        own = _hooks(run, trainer, folder, lr, read, echo)
        latest = Latest()
        trainer.fit(
            read(t.data, start=trainer.step),
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


def _accumulated(microbatches: int) -> int:
    """The microbatches a step accumulates gradients over: none under a pipeline,
    which takes them through its stages instead."""
    return 1 if parallel.pipeline.pipelined() else microbatches


def _hooks(
    run: ir.Run,
    trainer: Trainer,
    folder: Path,
    lr: Schedule,
    read: Callable[..., Iterator[Batch]],
    echo: Callable[[str], None],
) -> list[Hook]:
    """The run's own hooks: its log, eval and checkpoint, which resumes `trainer`
    from the latest checkpoint in `folder` if there is one. `read` makes the
    batches of a data config."""
    t = run.train
    own: list[Hook] = [Log(every=t.log.every, lr=lr, tokens=t.batch * t.seq_len)]
    if t.eval is not None:
        held_out = functools.partial(read, t.eval.data, stream=1)
        own.append(Eval(held_out, every=t.eval.every, batches=t.eval.batches))
    if (c := t.checkpoint) is not None:
        checkpoint = Checkpoint(folder / "checkpoints", every=c.every, keep=c.keep)
        if checkpoint.restore(trainer):
            echo(f"{run.name}: resuming at step {trainer.step}")
        own.append(checkpoint)
    return own
