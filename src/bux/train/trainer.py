"""Trains a compiled Run with the reference Flax NNX backend, on the default device.

A run's directory, ``<out>/<name>-<id>``, holds:

    run.bux        the run as one self-contained file, which `bux train` takes as is
    run.json       its IR
    metrics.jsonl  a line of metrics every `log.every` steps, and at every eval
    checkpoints/   with a `checkpoint` setting; a run there resumes from the latest

Its name says which alternatives it took, and its id hashes what it computes, so a
changed setting gets a fresh directory and an unchanged one finds its checkpoints.
"""

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from bux.architecture import Model
from bux.data.pipeline import DEFAULT_CACHE, jobs, prepare
from bux.lang import ir
from bux.lang.printer import run_to_source
from bux.train import config as tcfg
from bux.train.data import Batch, batches
from bux.train.optim import optimizer, schedule


def directory(run: ir.Run, out: str | Path) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", run.name).strip("_")
    return Path(out) / f"{slug}-{run.id}"


def lookahead(arch: ir.Architecture) -> list[int]:
    """How far ahead each head predicts, in execution order: 1 + the nextn layers before it."""
    ahead, nextn = [], 0
    for layer in arch.layers():
        if layer.config.type == "nextn":
            nextn += 1
        elif layer.config.type == "head":
            ahead.append(1 + nextn)
    return ahead or [1]


def _cross_entropy(logits: jax.Array, batch: Batch, ahead: int) -> jax.Array:
    """The mean loss of logits over positions t that predict a target, token t + ahead,
    of their own document: at least `ahead` tokens into it. A position with nothing
    that far ahead in its row, or in its document, is left out."""
    targets = batch.input_ids[:, ahead:]
    weight = batch.loss_mask[:, ahead:] & (batch.position_ids[:, ahead:] >= ahead)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, : targets.shape[1]].astype(jnp.float32), targets
    )
    return jnp.sum(loss * weight) / jnp.maximum(jnp.sum(weight), 1)


def losses(
    outputs: list[jax.Array], batch: Batch, ahead: list[int], c: tcfg.LossConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The loss to minimize, and its terms."""
    metrics = {"loss": _cross_entropy(outputs[0], batch, ahead[0])}
    total = metrics["loss"]
    if c.z:
        log_z = jax.nn.logsumexp(outputs[0].astype(jnp.float32), axis=-1)
        metrics["z_loss"] = jnp.mean(log_z**2)
        total = total + c.z * metrics["z_loss"]
    if len(outputs) > 1:
        metrics["mtp"] = jnp.mean(
            jnp.stack(
                [_cross_entropy(o, batch, a) for o, a in zip(outputs[1:], ahead[1:])]
            )
        )
        total = total + c.mtp * metrics["mtp"]
    return total, metrics


def train(
    run: ir.Run,
    out: str | Path = "runs",
    *,
    cache: str | Path = DEFAULT_CACHE,
    echo: Callable[[str], None] = print,
) -> dict[str, float]:
    """Train `run` to its last step, writing to its directory under `out`, after
    preparing the datasets it reads under `cache`; the last metrics logged."""
    t = run.train
    vocab = run.arch.vocab_size
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
    opt = nnx.Optimizer(
        model, optimizer(t.optimizer, multiplier, model, t.clip), wrt=nnx.Param
    )
    start = 0
    manager = None
    if t.checkpoint is not None:
        manager = ocp.CheckpointManager(
            (folder / "checkpoints").resolve(),
            options=ocp.CheckpointManagerOptions(max_to_keep=t.checkpoint.keep),
        )
        if (latest := manager.latest_step()) is not None:
            target = nnx.state((model, opt))
            state = manager.restore(latest, args=ocp.args.StandardRestore(target))
            nnx.update((model, opt), state)
            start = latest
            echo(f"{run.name}: resuming at step {start}")
    ahead = lookahead(run.arch)

    def outputs(model: Model, batch: Batch) -> list[jax.Array]:
        return model.outputs(
            jnp.asarray(batch.input_ids), jnp.asarray(batch.position_ids)
        )

    @nnx.jit
    def step(model: Model, opt: nnx.Optimizer, batch: Batch):
        def loss(model: Model):
            return losses(outputs(model, batch), batch, ahead, t.loss)

        (_, metrics), grads = nnx.value_and_grad(loss, has_aux=True)(model)
        opt.update(model, grads)
        return metrics | {"grad_norm": optax.tree.norm(grads)}

    @nnx.jit
    def evaluate(model: Model, batch: Batch) -> jax.Array:
        return losses(outputs(model, batch), batch, ahead, t.loss)[1]["loss"]

    log = (folder / "metrics.jsonl").open("a" if start else "w", encoding="utf-8")
    data = batches(t.data, t, vocab, cache=cache, start=start)
    last: dict[str, float] = {}
    clock, since = time.perf_counter(), start
    try:
        for i in range(start, t.steps):
            metrics = step(model, opt, jax.device_put(next(data)))
            done = i + 1
            if done % t.log.every == 0 or done == t.steps:
                elapsed = time.perf_counter() - clock
                last = {k: float(v) for k, v in metrics.items()}
                last |= {
                    "lr": float(t.optimizer.lr * multiplier(i)),
                    "tokens": done * t.batch * t.seq_len,
                    "tokens_per_s": (done - since) * t.batch * t.seq_len / elapsed,
                }
                clock, since = time.perf_counter(), done
                log.write(json.dumps({"step": done} | last) + "\n")
                log.flush()
                echo(
                    f"{run.name} step {done}/{t.steps}  loss {last['loss']:.4f}  "
                    f"lr {last['lr']:.2e}  grad {last['grad_norm']:.3f}  "
                    f"{last['tokens_per_s']:,.0f} tok/s"
                )
            if t.eval is not None and (done % t.eval.every == 0 or done == t.steps):
                model.eval()
                held_out = batches(t.eval.data, t, vocab, cache=cache, stream=1)
                value = (
                    sum(
                        float(evaluate(model, jax.device_put(next(held_out))))
                        for _ in range(t.eval.batches)
                    )
                    / t.eval.batches
                )
                model.train()
                last["eval_loss"] = value
                log.write(json.dumps({"step": done, "eval_loss": value}) + "\n")
                log.flush()
                echo(f"{run.name} step {done}/{t.steps}  eval loss {value:.4f}")
            saves = t.checkpoint is not None and (
                done % t.checkpoint.every == 0 or done == t.steps
            )
            if manager is not None and saves:
                state = nnx.state((model, opt))
                manager.save(done, args=ocp.args.StandardSave(state))
    finally:
        log.close()
        if manager is not None:
            manager.wait_until_finished()
            manager.close()
    return last
