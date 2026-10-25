# Extending the trainer

{py:func}`bux.train.trainer.train` is a few lines that put blocks together, and each
block works without the others. So changing one part of training means replacing one
block: a builder in a registry, a hook, a sink, a term of the objective, or a step of
the {py:class}`~bux.train.trainer.Trainer`. This page shows each, with examples that
run on the smoke run.

```python
import optax
from flax import nnx

from bux.lang import compile_runs
from bux.train.hooks import Hook
from bux.train.trainer import train

path = "examples/experiments/smoke.bux"
[run] = compile_runs(open(path).read(), filename=path)


class WeightNorm(Hook):
    every = 10

    def __call__(self, trainer, metrics):
        params = nnx.state(trainer.model, nnx.Param)
        return {"weight_norm": float(optax.tree.norm(params))}


train(run, hooks=[WeightNorm()])  # metrics.jsonl gets weight_norm every 10 steps
```

## The trainer's blocks

| Block | In | What it is | To change it |
|-------|----|------------|--------------|
| sources | {py:mod}`bux.train.data` | an endless stream of rows, a source per data config type | register a builder in {py:data}`~bux.train.data.sources` |
| schedules | {py:mod}`bux.train.optim` | the curve after the warmup, per schedule type | register a builder in {py:data}`~bux.train.optim.schedules` |
| optimizers | {py:mod}`bux.train.optim` | an optax transformation, per optimizer type | register a builder in {py:data}`~bux.train.optim.optimizers` |
| objective | {py:mod}`bux.train.objective` | the loss: named, weighted terms, each a metric | add a {py:class}`~bux.train.objective.Term` |
| `Trainer` | {py:mod}`bux.train.trainer` | the model, its optimizer and objective; the compiled steps | override {py:meth}`~bux.train.trainer.Trainer.train_step` |
| hooks | {py:mod}`bux.train.hooks` | what happens every `every` steps: log, eval, save, profile | `train(run, hooks=[...])` |
| sinks | {py:mod}`bux.train.sinks` | where what hooks report goes: the console, `metrics.jsonl` | `train(run, sinks=[...])` |

## Registries

A config's `type` finds its builder in a registry ({py:class}`bux.train.registry.Registry`),
as a layer's does. Registering under a taken name replaces the builder, so a variant
of an optimizer is a function, not a fork:

```python
import optax

from bux.train.optim import lr_schedule, optimizers, weight_decay_mask


@optimizers.register("adamw")
def _(c, multiplier, model):  # its config, the schedule's multiplier, the model
    return optax.adamw(
        lr_schedule(c.lr, multiplier),
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=weight_decay_mask(model),
    )
```

| Registry | A builder gets | And returns |
|----------|----------------|-------------|
| {py:data}`bux.train.optim.optimizers` | the config, the schedule's multiplier (a function of the step), the model | an optax transformation |
| {py:data}`bux.train.optim.schedules` | the config, the run's steps | the curve after the warmup, a function of the step |
| {py:data}`bux.train.optim.ns_coefficients` | the number of Newton-Schulz iterations | an array of `(a, b, c)`, one, or one an iteration |
| {py:data}`bux.train.data.sources` | the config, the run's settings, the vocabulary's size, the cache, a seed | a {py:class}`~bux.train.data.Source`: `rows(start, n)` of its endless stream |
| `bux.parallel.methods.loss` | a head's output, its targets, their weights, whether to sum log Z² | the head's `Sums` ([Loss methods](#loss-methods)) |
| `bux.architecture.precision.matmuls.formats` | `jax.lax.dot_general`'s arguments | the product, in the format ([Precision](precision.md)) |

A new optimizer, schedule or source also needs its config in `bux/train/config.py`,
where the compiler and the editor find it: a pydantic class with a `type` tag, in the
union of its setting (`OptimizerConfig`, say) and in `Setting`, the union of the
configs a file calls by name. Newton-Schulz's coefficients, a loss method and a matmul
format are names in a string field (`muon.ns_coefficients`, `methods.loss`,
`precision.matmuls`), and need no config. The names each
registry holds are listed in [Registries](../reference/language/registries.md).

## Hooks

A {py:class}`~bux.train.hooks.Hook` runs after every `every` steps and after the
last. The scalars it returns, metrics by name, go to every sink:

| Method | When it runs |
|--------|--------------|
| `due(step, steps)` | after each step: whether the hook runs now. By default every `every` steps and at the last |
| `start(trainer)` | before the first step |
| `__call__(trainer, metrics)` | after a step it is due at, with the step's metrics, still on the device; returns scalars, a function that gives them, or `None` |
| `stop(trainer)` | when training stops before the last step, at the step it stopped at |
| `close()` | after the last step, or when training stops |

Hooks run in order, the run's own first: {py:class}`~bux.train.hooks.Log`,
{py:class}`~bux.train.hooks.Eval`, {py:class}`~bux.train.hooks.Checkpoint` and
{py:class}`~bux.train.hooks.Profile`, then those given to `train`. Each hook's
scalars reach the sinks as one write of their own, at the step it ran.

**Read later.** What a hook returns goes to the sinks once the next step is launched.
A hook that only reads the step's metrics returns a function that gives its scalars,
a {py:data}`~bux.train.hooks.Later`, which the loop calls then. The log does, so that
reading a loss waits for its step as the next one runs, not the GPU for the host to
launch that one. The log also computes its learning rate as it runs, on the GPU behind
its step: one computed as it was read queued behind the next step, and reading it
waited for that one too, which left the GPU nothing to run as the host launched the
one after.

**Stopping.** When training stops early, because something set `trainer.stopping`, as
the launcher does at SIGTERM, every hook's `stop` runs, and the checkpoint hook saves
the step.

## Sinks

A {py:class}`~bux.train.sinks.Sink` is where scalars go: the console
({py:class}`~bux.train.sinks.Console`) and `metrics.jsonl`
({py:class}`~bux.train.sinks.JSONL`) are sinks, and so is a tracker:

```python
import wandb

from bux.train.sinks import Sink


class Wandb(Sink):
    def __init__(self, run):
        self.run = wandb.init(project="bux", name=run.name, config=run.model_dump())

    def write(self, step, scalars):
        self.run.log(scalars, step=step)

    def close(self):
        self.run.finish()


train(run, hooks=[WeightNorm()], sinks=[Wandb(run)])
```

In a job of many processes every process computes all scalars, and only the first
writes them. A sink of your own that writes checks
{py:func}`~bux.train.sinks.is_main_process`.

## A Trainer in a script

A {py:class}`~bux.train.trainer.Trainer` takes a model, an optax transformation and
an objective, so a script or a notebook can train with blocks of its own and keep the
rest. With `run` the smoke run of the first example:

```python
from bux.architecture import Model
from bux.train.data import batches
from bux.train.hooks import Log
from bux.train.objective import objective
from bux.train.sinks import Console
from bux.train.trainer import Trainer

model = Model(run.arch, rngs=nnx.Rngs(0))
trainer = Trainer(model, optax.adamw(1e-3), objective(run.train.loss, run.arch))
tokens = run.train.batch * run.train.seq_len
trainer.fit(
    batches(run.train.data, run.train, run.arch.vocab_size),
    steps=20,
    hooks=[Log(every=10, lr=lambda step: 1e-3, tokens=tokens)],
    sinks=[Console("mine", 20)],
)
```

```text
mine step 10/20  loss 6.6509  lr 1.00e-03  grad 1.681  12,024 tok/s
mine step 20/20  loss 6.6194  lr 1.00e-03  grad 1.573  108,656 tok/s
```

{py:meth}`~bux.train.trainer.Trainer.fit` trains to step `steps`, or until
`trainer.stopping` is set. {py:func}`~bux.train.data.batches` yields the run's
batches from batch `start` on, so a script that resumes reads what it would have.
`Trainer(..., compute=, matmuls=, microbatches=, donate=, ...)` takes what the run's
`precision` and `methods` give `train`; made inside `bux.parallel.use`, it puts its
state on the mesh ([In a script](../parallelism/index.md#in-a-script)).

## Objective terms

The objective ({py:class}`bux.train.objective.Objective`) is a weighted sum of named
terms, and each term is also a metric under its name. Another term is a name, a weight
and a function of the heads' logits and the batch:

```python
import jax.numpy as jnp

from bux.train.objective import Term, objective

loss = objective(run.train.loss, run.arch)
loss.terms["logit_l2"] = Term(1e-4, lambda logits, batch: jnp.mean(logits[0] ** 2))
trainer = Trainer(Model(run.arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
```

The log then has `logit_l2` beside `loss`.

**Heads, not logits.** The model gives the objective its heads' outputs, not their
logits: `Model.heads` (`training_heads` in a training step) returns each head's input
and its projection, a {py:class}`~bux.architecture.logits.HeadOutput`. The default
terms, `loss`, `z_loss` and `mtp`, are means over the targets of the cross-entropy,
log Z minus the target's logit, and of log Z squared, which `heads.sums(i, offset, z)`
gives for head `i` by the run's `methods.loss`, without the logits.

A term of your own gets `logits`, a sequence that makes a head's logits once a term
reads them, once for every term that does: `logits[0] ** 2` above makes the first
head's whole, by either method. `Model.outputs` returns every head's logits at once.

## Loss methods

`methods.loss` says how the loss takes a head's logits:

| Method | How |
|--------|-----|
| `chunks` | the matmul, the logsumexp and the target's logit of at most 2 GiB of float32 logits at once, and their gradients by the kernel and the stream with them, the kernel's added up in float32: the backward only scales those. The default |
| `logits` | one matmul for every position, the logits whole in `compute` |

At a vocabulary of 128K, the float32 logits of 8 rows of 4096 tokens take 16 GB, and
their gradient as much again. A sum over positions is linear in each position's term,
so its gradient is known in the forward but for one factor, its cotangent, which the
backward multiplies in. By chunks, a head multiplies by its vocabulary three times, as
by whole logits, and keeps the gradients from the forward to the backward in
`compute`, the kernel's and the stream's. Each sum keeps its own, so a z-loss costs a
second matmul by the kernel and one by the stream.

Another method is a function of a head's output, the targets, their weights and
whether to sum log Z squared, that returns their `Sums`
({py:mod}`bux.architecture.logits`):

```python
from bux.parallel import methods


@methods.loss.register("mine")
def _(head, targets, weights, z): ...  # returns a bux.architecture.logits.Sums
```

### Measurements

On a B300, for the dense 1.3B of 16 Llama blocks of width 2048 (vocabulary 128256,
`bfloat16`, 8 rows of 4096 tokens), against the whole logits, and against the chunks
before they took their gradients in the forward, whose backward made each chunk's
logits again:

| Head, `methods.loss` | tok/s | Peak memory | Largest batch |
|----------------------|-------|-------------|---------------|
| untied, `logits` | 138k | 70.6 GiB | |
| untied, chunks, before | 125k | 52.1 GiB | |
| untied, `chunks` | 139k | 51.1 GiB | 56 rows, 146k |
| tied, `logits` | 141k | 67.5 GiB | |
| tied, `chunks` | 139k | 48.1 GiB | |

On 8 B300s under `fsdp=8`, 8 rows each, `chunks` trains at 1.071M tokens a second in
38.9 GiB, and `logits` at 1.064M in 58.5: each device adds its share of the head's
gradient up alone, and the mesh reduces it once. The GDN hybrid and the MoE of
[Repeats and remat](../parallelism/remat.md) train 6% and 5% faster by `chunks` than
they did by chunks before.

## Losses from a layer

A layer adds a term of its own from its forward, in one line,
{py:func}`bux.architecture.registry.report`:

```python
report(taps, "my_loss", weight, value)
```

The objective adds `weight × value` to the training loss, and logs the values, summed
over the layers that report the name, under that name. An MoE's `balance_loss` and a
sparse attention indexer's `indexer_loss` are reported so. Eval leaves them out, and a
name that is also a term of the objective is an error. How to write a layer is in
[Adding a layer](../architectures/adding-a-layer.md).

## Overriding a step

{py:meth}`~bux.train.trainer.Trainer.train_step` and
{py:meth}`~bux.train.trainer.Trainer.eval_step` are what the trainer compiles, once. A
subclass that overrides them changes what a step does (averaging the weights,
skipping a spike) and keeps the loop, the hooks and the checkpoints:

```python
class WithParamNorm(Trainer):
    def train_step(self, model, optimizer, batch):
        metrics = super().train_step(model, optimizer, batch)
        return metrics | {"param_norm": optax.tree.norm(nnx.state(model, nnx.Param))}
```

`train_step` gets a model and an optimizer made in the step from the values of the
trainer's variables, and its changes to them are the step's new state, which the
trainer's model and optimizer hold after it. So a hook may change the values of their
variables between steps (`trainer.load`, `nnx.update`), but not which variables they
hold: a step that does raises
`a training step changes the values of the model's and the optimizer's variables, but not which variables they hold`.

## Microbatches

Gradient accumulation is built in: `methods(microbatches=n)` takes a step's batch in
`n` microbatches. On a mesh it gathers the parameters once a step and, where the rows
are independent and span hosts, reduces the gradients once
([Microbatches](../parallelism/microbatches.md)). `Trainer(..., microbatches=,
gather=, accumulate=, reduce=)` takes the same settings.

## Donation

With `Trainer(..., donate=True)`, as a run's `methods.donate` makes it by default, a
training step donates the model's and the optimizer's arrays, so that their new state
takes the old one's memory. An array read from `trainer.model` before a step is gone
after it, so a hook that keeps one keeps a copy (`np.asarray`).

The 1.3B's step, its loss by logits, peaks at 70.6 GiB instead of 81.3 on one GPU
(51.1 by chunks, the default), and it trains 36 rows a GPU, which run out of memory
without ([Donation](../parallelism/microbatches.md#donation-methodsdonate)).

## Prefetch

`fit` reads its data through {py:func}`~bux.train.data.prefetch`, a thread that loads
the next two batches on the host while a step runs. A run that resumes still reads
from its step on, and an error reading a batch is raised at that batch.

Each batch goes to the devices just before its step, from the training thread. One put
there from the prefetch thread, while a step's temporaries held most of a GPU, landed
below them and split what they left free, and the next step's no longer fit.

## The host between steps

Between two steps the host does little. A training step takes the model's and the
optimizer's arrays as a list, of variables the trainer split once from their graph
(`_Split` in `bux/train/trainer.py`), where `nnx.jit` walked the graph at every call,
from it to its arrays and back: 20 ms a step of the 1.3B, 70 of Qwen3-30B-A3B, on a
B300 node's host. The list takes 18 ms of the 30B's, most of it Python freeing the
arrays of the step before.

That is time the GPU waits for, where XLA's launch of a step returns only when the GPU
is nearly through it, its queue of kernels full:

- Qwen3-30B-A3B's step, of thousands of kernels and collectives, returns 100 to 170 ms
  before its step ends, and the host then has that long to launch the next.
- The 1.3B's, its kernels in CUDA graphs (XLA's command buffers), returns in 10 ms, and
  the host runs up to two steps ahead; but under the profiler, which turns command
  buffers off, its launch too returns at its step's end.

Python's cycle collector, which walked what `nnx.jit`'s walks made, no longer stops
the host for a full collection a few steps in (340 ms on the 1.3B, 980 on the 30B).
The GPU now waits for the host only after a step the host waits for, at an eval or a
checkpoint.

### Measurements

On 8 B300s, against the tree before:

| Run | Before | After |
|-----|--------|-------|
| 1.3B (one-node suite's `fsdp`), log every 5 steps | 1.224M tok/s | 1.245M tok/s |
| the same, profiled, logging every step: GPU 0 idle | 57.9 ms a step | 2.2 ms a step |
| the same, profiled, log every 5 steps: GPU 0 idle | 25.8 ms a step | 2.1 ms a step |
| 1.3B in 8 microbatches (`fsdp_m8`) | 1.012M tok/s | 1.054M tok/s |
| Qwen3-30B-A3B, megatron.bux's `sorted` | 42.0k tok/s a GPU | 42.3k |
| the same, logging every step | 6.30 s a step | 6.24 s |
| the one-GPU suite's 8 runs, a B300 each | | +1.2% to +7.2% |

With GEMM autotuning off and XLA's deterministic ops, both 1.3B runs train to the same
losses as before, to the bit.
