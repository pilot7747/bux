"""Trains a compiled Run with the reference Flax NNX backend, on the devices its mesh
takes (``bux.parallel``).

A run's directory, ``<out>/<name>-<id>``, holds:

    run.bux        the run as one self-contained file, which `bux train` takes as is
    run.json       its IR
    metrics.jsonl  a line of metrics every `log.every` steps, and at every eval
    checkpoints/   with a `checkpoint` setting; a run there resumes from the latest

Its name says which alternatives it took, and its id hashes what it computes, so a
changed setting gets a fresh directory and an unchanged one finds its checkpoints.

``train`` puts a run together from blocks that each work alone: a ``Trainer`` (a
model, its optimizer and its objective), the run's batches, hooks for its log, eval
and checkpoints, and sinks for what the hooks return. A script can swap any of them:

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
from bux.train.hooks import Checkpoint, Eval, Hook, Log
from bux.train.objective import Metrics, Objective, objective, target_mask
from bux.train.optim import Schedule, lr_multiplier, lr_schedule, optimizer
from bux.train.sinks import JSONL, Console, LatestMetrics, Sink, is_main_process


def forward(model: Model, batch: Batch) -> list[Float[Array, "b t vocab"]]:
    """Return every head's logits on `batch`, each document of a row on its own."""
    return model.outputs(jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids))


def training_forward(
    model: Model, batch: Batch
) -> tuple[list[Float[Array, "b t vocab"]], list[Loss]]:
    """Return `forward`'s logits in a training step, and the losses layers report."""
    tokens, positions = jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids)
    return model.training_outputs(tokens, positions)


def cast[M: nnx.Module](model: M, dtype: DTypeLike) -> M:
    """Return a copy of `model` with its floating-point parameters in `dtype` and the
    rest of its state as it is. Gradients through it reach the parameters in their
    own dtype."""
    graphdef, params, buffers = nnx.split(model, nnx.Param, ...)
    params = jax.tree.map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )
    return nnx.merge(graphdef, params, buffers)


@runtime_checkable
class StepEndUpdate(Protocol):
    """A module that learns from the batch other than by gradients, as an MoE moves
    its router bias. A training step sets `update_at_step_end`: its microbatches'
    forwards only count what they see, and `step_end_update` applies it once after
    them. Outside a step, a training forward applies it at once."""

    update_at_step_end: bool

    def step_end_update(self) -> None: ...


def apply_step_end_updates(model: nnx.Module) -> None:
    """Call every module's `step_end_update`, once per step, after all its forwards."""
    for _, module in nnx.iter_modules(model):
        if isinstance(module, StepEndUpdate):
            module.step_end_update()


def token_shares(batch: Batch, n: int) -> Float[Array, " n"]:
    """Return each of `n` microbatches' share of the batch's target tokens (the first
    head's), or 1 / n each if it has none."""
    counts = parallel.microbatches(target_mask(batch), n).sum((1, 2))
    num_targets = counts.sum()
    return jnp.where(num_targets > 0, counts / jnp.maximum(num_targets, 1), 1 / n)


def _scaled(metrics: Metrics, share: float | Float[Array, ""]) -> Metrics:
    return {k: v * share for k, v in metrics.items()}


def _shard_state(state: nnx.State) -> nnx.State:
    """Constrain `state` to the shardings of its variables, on a mesh."""
    if parallel.active() is None:
        return state
    return jax.lax.with_sharding_constraint(state, parallel.shardings(state))


class Trainer:
    """A model, its optimizer, which applies `tx`, and the objective it minimizes;
    `step` counts the steps taken.

    `train_step` and `eval_step` are compiled once, so a subclass that overrides them
    changes what a step does (keeps an average of the weights, skips a spike) and
    keeps the loop, hooks and checkpoints. They get the model and the optimizer as
    arguments, which is how nnx.jit sees them change.

    Both compute in `compute`, the model's parameters cast to it inside the loss, so
    that the parameters, their gradients and the optimizer's state stay as they are
    (float32).

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
        self._train = nnx.jit(self._train_step_on_shards)
        self._eval = nnx.jit(self.eval_step)

    def _train_step_on_shards(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        metrics = self.train_step(model, optimizer, batch)
        parallel.place(model, optimizer)
        return metrics

    def train_step(
        self, model: Model, optimizer: nnx.Optimizer, batch: Batch
    ) -> Metrics:
        """Take one optimizer step on `batch`, and return its metrics with
        `grad_norm`, the gradients' norm before clipping."""
        grads, metrics = self.gradients(model, batch)
        optimizer.update(model, grads)
        return metrics | {"grad_norm": optax.tree.norm(grads)}

    def gradients(self, model: Model, batch: Batch) -> tuple[nnx.State, Metrics]:
        """Return the loss's gradient on `batch` and its metrics, summed over the
        `microbatches`. The forwards update the model's buffers (its state that is not
        a parameter) in turn, and then the step-end updates run."""
        graphdef, params, buffers = nnx.split(model, nnx.Param, ...)
        if self.microbatches == 1:
            grads, metrics, buffers = self._grads(
                graphdef, params, buffers, batch, token_share=1.0, row_share=1.0
            )
        else:
            grads, metrics, buffers = self._accumulate(graphdef, params, buffers, batch)
        nnx.update(model, buffers)
        apply_step_end_updates(model)
        return grads, metrics

    def _accumulate(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Sum `_grads` over the microbatches of `batch` in a scan, each one's
        per-token part scaled by its share of the batch's targets and its per-row part
        by 1 / n: so the sums are the batch's, even when microbatches have unequal
        numbers of targets."""
        n = self.microbatches

        def add_microbatch(
            carry: tuple[nnx.State, nnx.State],
            xs: tuple[Batch, Float[Array, ""]],
        ) -> tuple[tuple[nnx.State, nnx.State], Metrics]:
            grad_sum, buffers = carry
            microbatch, token_share = xs
            grads, metrics, buffers = self._grads(
                graphdef, params, buffers, microbatch, token_share, row_share=1 / n
            )
            return (jax.tree.map(jnp.add, grad_sum, grads), buffers), metrics

        xs = (parallel.microbatches(batch, n), token_shares(batch, n))
        zeros = _shard_state(jax.tree.map(jnp.zeros_like, params))
        (grads, buffers), metrics = jax.lax.scan(add_microbatch, (zeros, buffers), xs)
        return grads, {k: v.sum() for k, v in metrics.items()}, buffers

    def _grads(
        self,
        graphdef: nnx.GraphDef,
        params: nnx.State,
        buffers: nnx.State,
        batch: Batch,
        token_share: float | Float[Array, ""],
        row_share: float,
    ) -> tuple[nnx.State, Metrics, nnx.State]:
        """Return the loss's gradient on `batch`, computed in `compute`, its metrics,
        and the buffers after the forward, each part of the objective scaled by its
        share."""

        def loss_fn(params: nnx.State) -> tuple[Float[Array, ""], tuple[Metrics, Any]]:
            model = cast(nnx.merge(graphdef, params, buffers, copy=True), self.compute)
            model.set_attributes(update_at_step_end=True, raise_if_not_found=False)
            logits, aux_losses = training_forward(model, batch)
            per_token, per_row = self.objective(logits, batch, aux_losses)
            loss = token_share * per_token.loss + row_share * per_row.loss
            metrics = _scaled(per_token.metrics, token_share)
            metrics |= _scaled(per_row.metrics, row_share)
            return loss, (metrics, nnx.state(model, nnx.Not(nnx.Param)))

        grads, (metrics, buffers) = jax.grad(loss_fn, has_aux=True)(params)
        return grads, metrics, buffers

    def eval_step(self, model: Model, batch: Batch) -> Float[Array, ""]:
        """Return the first head's loss on `batch`, computed in `compute`."""
        logits = forward(cast(model, self.compute), batch)
        per_token, _ = self.objective(logits, batch)
        return per_token.metrics["loss"]

    def train_on_batch(self, batch: Batch) -> Metrics:
        """Take one training step on `batch`, and return its metrics."""
        metrics = self._train(self.model, self.optimizer, parallel.rows(batch))
        self.step += 1
        return metrics

    def evaluate(self, batches: Iterable[Batch]) -> float:
        """Return the mean of `eval_step` over `batches`, in eval mode."""
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
        what they return goes to every sink. All of them close at the end, or when
        training stops."""
        with contextlib.ExitStack() as stack:
            for block in (*sinks, *hooks):
                stack.callback(block.close)
            for hook in hooks:
                hook.start(self)
            while self.step < steps and not self.stopping:
                metrics = self.train_on_batch(next(data))
                for hook in hooks:
                    if hook.due(self.step, steps) and (scalars := hook(self, metrics)):
                        for sink in sinks:
                            sink.write(self.step, scalars)
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
    preparing the datasets it reads under `cache`; return the latest value of every
    metric. `hooks` run after the run's own, and `sinks` get what all of them return,
    as the console and metrics.jsonl do."""
    config, vocab = run.train, run.arch.vocab_size
    if vocab is None:
        raise ValueError("a model without a vocabulary cannot be trained on tokens")
    for job in jobs(config):
        prepare(job, cache, echo=echo)
    folder = directory(run, out)
    folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (folder / "run.bux").write_text(run_to_source(run), encoding="utf-8")
        (folder / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )

    with parallel.use(config.mesh, config.layout, config.methods):
        parallel.check(config.batch, config.seq_len)
        model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(config.seed)))
        multiplier = lr_multiplier(config.schedule, config.steps)
        tx = optimizer(config.optimizer, multiplier, model, config.clip)
        trainer = Trainer(
            model,
            tx,
            objective(config.loss, run.arch),
            compute=config.precision.compute,
            microbatches=_accumulated(config.methods.microbatches),
        )
        make_batches = functools.partial(
            batches, config=config, vocab_size=vocab, cache=cache
        )
        lr = lr_schedule(config.optimizer.lr, multiplier)
        run_hooks = _hooks(run, trainer, folder, lr, make_batches, echo)
        latest = LatestMetrics()
        trainer.fit(
            make_batches(config.data, start=trainer.step),
            config.steps,
            hooks=[*run_hooks, *hooks],
            sinks=[
                Console(run.name, config.steps, echo),
                JSONL(folder / "metrics.jsonl", append=trainer.step > 0),
                latest,
                *sinks,
            ],
        )
    return latest.metrics


def _accumulated(microbatches: int) -> int:
    """Return how many microbatches a step accumulates gradients over: one under a
    pipeline, which takes them through its stages instead."""
    return 1 if parallel.pipeline.pipelined() else microbatches


def _hooks(
    run: ir.Run,
    trainer: Trainer,
    folder: Path,
    lr: Schedule,
    make_batches: Callable[..., Iterator[Batch]],
    echo: Callable[[str], None],
) -> list[Hook]:
    """Return the run's own hooks: its log, eval and checkpoint, which resumes
    `trainer` from the latest checkpoint in `folder` if there is one."""
    config = run.train
    tokens = config.batch * config.seq_len
    hooks: list[Hook] = [Log(every=config.log.every, lr=lr, tokens=tokens)]
    if (evals := config.eval) is not None:
        held_out = functools.partial(make_batches, evals.data, held_out=True)
        hooks.append(Eval(held_out, every=evals.every, batches=evals.batches))
    if (saves := config.checkpoint) is not None:
        checkpoint = Checkpoint(
            folder / "checkpoints", every=saves.every, keep=saves.keep
        )
        if checkpoint.restore(trainer):
            echo(f"{run.name}: resuming at step {trainer.step}")
        hooks.append(checkpoint)
    return hooks
