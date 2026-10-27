# Extending the trainer

{py:func}`bux.train.trainer.train` is a few lines that put blocks together, and each
block works without the others. To change one part of training, you replace one
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
| trackers | {py:mod}`bux.train.trackers` | the sinks `log.trackers` names: W&B, TensorBoard | register a builder in {py:data}`~bux.train.trackers.trackers` |

## Registries

A config's `type` finds its builder in a registry
({py:class}`bux.train.registry.Registry`), as a layer's does. Registering under a
taken name replaces the builder, so a variant of an optimizer is a function, not a
fork:

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
| {py:data}`bux.train.optim.schedules` | the config, the run's training settings (its steps, batch and peak `lr`) | the curve after the warmup, a function of the step |
| {py:data}`bux.train.optim.ns_coefficients` | the number of Newton-Schulz iterations | an array of `(a, b, c)`, one, or one an iteration |
| {py:data}`bux.train.optim.ns_algorithms` | the number of iterations, their `(a, b, c)`, the iterations to restart after (or none) | a function of a matrix, or a stack of them as its `MuonDimensionNumbers` say, that orthogonalizes each |
| {py:data}`bux.train.data.sources` | the config, the run's settings, the vocabulary's size, the cache, a seed | a {py:class}`~bux.train.data.Source`: `rows(start, n)` of its endless stream |
| {py:data}`bux.train.trackers.trackers` | the config, the run, its directory, the step it starts at | a {py:class}`~bux.train.sinks.Sink` ([Sinks](#sinks)) |
| `bux.parallel.methods.loss` | a head's output, its targets, their weights, whether to sum log Z² | the head's `Sums` ([Loss methods](#loss-methods)) |
| `bux.architecture.precision.matmuls.formats` | `jax.lax.dot_general`'s arguments | the product, in the format ([Precision](precision.md)) |

A new optimizer, schedule or source also needs its config in `bux/train/config.py`,
where the compiler and the editor find it. That is a pydantic class with a `type`
tag, added to the union of its setting (`OptimizerConfig`, say) and to `Setting`, the
union of the configs a file calls by name. Newton-Schulz's coefficients and
algorithm, a loss method and a matmul format are names in a string field
(`muon.ns_coefficients`, `muon.ns_algorithm`, `methods.loss`, `precision.matmuls`),
so they need no config. [Registries](../reference/language/registries.md) lists the
names each registry holds.

## Hooks

A {py:class}`~bux.train.hooks.Hook` runs every `every` steps and after the last.
The scalars it returns, metrics by name, go to every sink:

| Method | When it runs |
|--------|--------------|
| `due(step, steps)` | after each step: whether the hook runs now. By default every `every` steps and at the last |
| `start(trainer)` | before the first step |
| `__call__(trainer, metrics)` | after a step it is due at, with the step's metrics, still on the device; returns scalars, a function that gives them, or `None` |
| `stop(trainer)` | when training stops before the last step, at the step it stopped at |
| `close()` | after the last step, or when training stops |

Hooks run in order, the run's own first: {py:class}`~bux.train.hooks.Log`, an
{py:class}`~bux.train.hooks.Eval` for each eval, {py:class}`~bux.train.hooks.Checkpoint`
and {py:class}`~bux.train.hooks.Profile`, then those given to `train`. Each hook's
scalars reach the sinks as a separate write, at the step it ran.

**Read later.** What a hook returns goes to the sinks once the next step is launched.
A hook that only reads the step's metrics should return a function that gives its
scalars, a {py:data}`~bux.train.hooks.Later`, which the loop calls then. The log works
this way:

- Reading a loss waits for its step, but by then the next step is running, so the GPU
  never waits for the host to launch it.
- The log computes its learning rate on the device when it runs, right behind its
  step. Computed when read, it would queue behind the next step, and reading it would
  hold the host until that step finished, leaving the GPU idle while the host
  launched the one after.

A hook that reads a value off the device inside `__call__`, as `WeightNorm` above
does, makes the GPU wait at each step it runs at.

**Stopping.** When training stops early, because something set `trainer.stopping`
(as the launcher does at SIGTERM), every hook's `stop` runs, and the checkpoint hook
saves the step.

## Sinks

A {py:class}`~bux.train.sinks.Sink` is where scalars go: the console
({py:class}`~bux.train.sinks.Console`), `metrics.jsonl`
({py:class}`~bux.train.sinks.JSONL`) and the trackers
({py:class}`~bux.train.trackers.Wandb`, {py:class}`~bux.train.trackers.TensorBoard`)
are sinks. `write` takes a step's scalars, and `histograms` the histograms of a step
that observed the model ([Observing a run](observing.md)):

```python
from bux.train.sinks import Sink


class Printed(Sink):
    def write(self, step, scalars):
        print(step, scalars.get("loss"))

    def histograms(self, step, histograms):  # by name: shares of log2 |x| bins
        print(step, sorted(histograms))


train(run, hooks=[WeightNorm()], sinks=[Printed()])
```

A tracker that `log.trackers` names is a sink plus a builder registered for its
config's type in {py:data}`~bux.train.trackers.trackers`. Its config, with a `type`
tag, goes in the `Tracker` union of `bux/train/config.py`:

```python
from bux.train.trackers import trackers


@trackers.register("mlflow")
def _(c, run, folder, start):  # its config, the run, its directory, the first step
    return MLflow(c, run)
```

In a job of many processes, every process computes all scalars, and only the first
writes them. A sink of your own that writes should check
{py:func}`~bux.train.sinks.is_main_process`.

## Timers

`trainer.timers` keeps the seconds the host spends in each part of the loop, by
name, which the log reports per step as `time/<name>_ms`
([Observing a run](observing.md#what-every-log-has)). The loop counts `data`,
`launch`, `compile`, `wait` and `hooks`, and a hook of your own can count its own
parts:

```python
class Slow(Hook):
    every = 10

    def __call__(self, trainer, metrics):
        with trainer.timers("slow"):  # the log's time/slow_ms
            sum(range(10**6))
```

## A Trainer in a script

A {py:class}`~bux.train.trainer.Trainer` takes a model, an optax transformation and
an objective, so a script or a notebook can swap in its own blocks and keep the rest.
With `run` the smoke run of the first example:

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
mine step 10/20  loss 6.2466  lr 1.00e-03  grad 0.622  9,229 tok/s  ...
mine step 20/20  loss 6.2498  lr 1.00e-03  grad 0.623  62,206 tok/s  ...
```

{py:meth}`~bux.train.trainer.Trainer.fit` trains to step `steps`, or until
`trainer.stopping` is set.

- {py:func}`~bux.train.data.batches` yields the run's batches from batch `start` on,
  so a script that resumes reads what it would have read.
- {py:func}`~bux.train.data.training_batches` also follows the run's curriculum and
  batch ramp, and the trainer compiles again at each new shape
  ([Curriculum](curriculum.md)).
- `Log` takes the tokens of a step, or a function of a step count that gives the
  tokens of that many steps ({py:func}`bux.train.curriculum.tokens`).
- `Trainer(..., compute=, matmuls=, microbatches=, donate=, ...)` takes what the run's
  `precision` and `methods` give `train`. Made inside `bux.parallel.use`, it puts its
  state on the mesh ([In a script](../parallelism/index.md#in-a-script)).

## Objective terms

The objective ({py:class}`bux.train.objective.Objective`) is a weighted sum of named
terms, and each term is also a metric under its name. A new term is a name, a weight,
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
terms, `loss`, `z_loss` and `mtp`, are means over the targets of the cross-entropy
(log Z minus the target's logit) and of log Z squared. `heads.sums(i, offset, z)`
gives them for head `i` by the run's `methods.loss`, without making the logits.

A term of your own gets `logits`, a sequence that makes a head's logits when a term
first reads them, once for all the terms that do: `logits[0] ** 2` above makes the
first head's logits whole, under either method. `Model.outputs` returns every head's
logits at once.

## Loss methods

`methods.loss` says how the loss takes a head's logits:

| Method | How |
|--------|-----|
| `chunks` | the default: the matmul, the logsumexp of the other logits and the target's logit, for at most 2 GiB of float32 logits at a time, with their gradients by the kernel and by the stream. The kernel's gradients add up in float32, and the backward only scales them |
| `logits` | one matmul for every position, the logits whole in `compute` |

At a vocabulary of 128K, the float32 logits of 8 rows of 4096 tokens take 16 GB, and
their gradient as much again. `chunks` never makes them all:

- A sum over positions is linear in each position's term, so its gradient is known in
  the forward pass up to one factor, its cotangent, which the backward multiplies in.
- So each chunk computes its gradients by the kernel and by the stream in the forward
  pass, and keeps them in `compute` for the backward. A head still multiplies by its
  vocabulary three times, as with whole logits.
- Each sum keeps its own gradients, so a z-loss costs a second matmul by the kernel
  and one by the stream.

Another method is a function of a head's output, the targets, their weights, and
whether to sum log Z squared, that returns their `Sums`
({py:mod}`bux.architecture.logits`):

```python
from bux.parallel import methods


@methods.loss.register("mine")
def _(head, targets, weights, z): ...  # returns a bux.architecture.logits.Sums
```

### Measurements

On one B300 (2026-10-23), for the dense 1.3B of 16 Llama blocks of width 2048
(vocabulary 128256, bfloat16, 8 rows of 4096 tokens):

| Head, `methods.loss` | tok/s | Peak memory | Largest batch |
|----------------------|-------|-------------|---------------|
| untied, `logits` | 138k | 70.6 GiB | |
| untied, `chunks` | 139k | 51.1 GiB | 56 rows, 146k |
| tied, `logits` | 141k | 67.5 GiB | |
| tied, `chunks` | 139k | 48.1 GiB | |

On 8 B300s under `fsdp=8`, 8 rows each, `chunks` trains at 1.071M tokens/s in
38.9 GiB, and `logits` at 1.064M in 58.5 GiB: each device sums its share of the
head's gradient alone, and the mesh reduces it once.

## Losses from a layer

A layer adds its own term to the loss from its forward, in one line,
{py:func}`bux.architecture.registry.report`:

```python
report(taps, "my_loss", weight, value)
```

The objective adds `weight × value` to the training loss, and logs the values,
summed over the layers that report the name, under that name. An MoE's
`balance_loss` and a sparse attention indexer's `indexer_loss` are reported this way.
Eval leaves them out, and a name that is also a term of the objective is an error.
How to write a layer is in [Adding a layer](../architectures/adding-a-layer.md).

## Observing from a layer

A layer observes a value of its forward for the steps a run observes, in one line,
{py:func}`bux.architecture.observe.stat`:

```python
from bux.architecture import observe

observe.stat(taps, "mine/gate_mean", gate.mean())
```

On the steps that do not observe, it does nothing and compiles to no code. On those
that do, it is logged by layer, `mine/gate_mean/attn.03`, with no gradient and no
weight in the loss. An MoE's `router/*` and an attention's `attn/*` are observed this
way ([Observing a run](observing.md)). A value leaves the forward as a reported loss
of weight 0 would, so it gets out of scans, remat and pipelines as losses do.

## Overriding a step

{py:meth}`~bux.train.trainer.Trainer.train_step` and
{py:meth}`~bux.train.trainer.Trainer.eval_step` are what the trainer compiles, once.
A subclass that overrides them changes what a step does (averaging the weights,
skipping a spike) and keeps the loop, the hooks and the checkpoints:

```python
class WithParamNorm(Trainer):
    def train_step(self, model, optimizer, batch):
        metrics = super().train_step(model, optimizer, batch)
        return metrics | {"param_norm": optax.tree.norm(nnx.state(model, nnx.Param))}
```

`train_step` gets a model and an optimizer rebuilt inside the step from the values
of the trainer's variables. Its changes to them are the step's new state, which the
trainer's model and optimizer hold afterwards. So a hook may change the values of
their variables between steps (`trainer.load`, `nnx.update`), but not which variables
they hold. A step that does raises
`a training step changes the values of the model's and the optimizer's variables, but not which variables they hold`.

`eval_step` returns each metric as a pair: a sum over its batch, and the count that
sum is a mean of, `{"loss": (nats, targets), "bpb": (bits, bytes)}`
({py:func}`~bux.train.objective.held_out`).
{py:meth}`~bux.train.trainer.Trainer.evaluate` divides the sums over all the batches,
so an eval's mean weighs every target alike. A metric of your own is one more pair:
an accuracy is `(correct, targets)`, which the eval hook logs under the eval's name,
`eval_accuracy`.

## Microbatches

Gradient accumulation is built in: `methods(microbatches=n)` takes a step's batch in
`n` microbatches. On a mesh it gathers the parameters once per step and, where the
rows are independent and span hosts, reduces the gradients once
([Microbatches](../parallelism/microbatches.md)). `Trainer(..., microbatches=,
gather=, accumulate=, reduce=)` takes the same settings.

## Donation

With `Trainer(..., donate=True)`, which a run's `methods.donate` sets by default, a
training step donates the model's and the optimizer's arrays, so their new state
reuses the old state's memory. An array read from `trainer.model` before a step is
gone after it, so a hook that keeps one must keep a copy (`np.asarray`).

On one B300 (2026-10-23), donation lowers the peak memory of the 1.3B from 81.3 to
70.6 GiB with its loss by `logits` (51.1 GiB by `chunks`, the default), and lets it
train 36 rows per GPU, which run out of memory without it
([Donation](../parallelism/microbatches.md#donation-methodsdonate)).

## Prefetch

`fit` reads its data through {py:func}`~bux.train.data.prefetch`, a thread that loads
the next two batches on the host while a step runs. A run that resumes still reads
from its step on, and an error reading a batch is raised at that batch.

Each batch goes to the devices just before its step, from the training thread. A
batch put there from the prefetch thread, while a step's temporaries filled most of
a GPU, would fragment the free memory so that the next step no longer fit.

## The host between steps

Between two steps the host does little, so the GPU rarely waits for it:

- A training step takes the model's and the optimizer's arrays as a flat list of
  variables, split from their graph once (`_Split` in `bux/train/trainer.py`).
  Walking the graph at every call, as `nnx.jit` does, took 20 ms per step of the 1.3B
  and 70 ms of Qwen3-30B-A3B on a B300 node's host.
- XLA's launch of a step returns only when the GPU is nearly through it, with its
  queue of kernels full, so the host's time between launches is time the GPU may sit
  idle. Qwen3-30B-A3B's step, of thousands of kernels and collectives, returns 100 to
  170 ms before it ends, and the host has that long to launch the next. The step of
  the 1.3B, its kernels in CUDA graphs (XLA's command buffers), returns in 10 ms, and
  the host runs up to two steps ahead. Under the profiler, which turns command
  buffers off, its launch too returns only at the step's end.

The GPU waits for the host only after a step that the host itself waits for: at an
eval, a checkpoint, or the start and end of a trace. A logged loss makes no step
wait: on 8 B300s (2026-10-25), profiled while logging every step, GPU 0 idles 2.2 ms
per step of the 1.3B.
