# Custom training

In this tutorial you change how bux trains, from a Python script, without forking its
trainer: you replace an optimizer through its registry, add a hook that measures the
weights and a sink that writes a CSV, then add a term to the loss and drive a
`Trainer` yourself. The same code runs in a notebook.

`train(run)` is a few lines that put independent blocks together, so each change
replaces one block and keeps the rest:

| block      | where                    | what it is                                      | to change it                   |
|------------|--------------------------|-------------------------------------------------|--------------------------------|
| optimizers | `bux/train/optim.py`     | an optax transformation per optimizer type      | `@optimizers.register(...)`    |
| schedules  | `bux/train/optim.py`     | the curve after the warmup, per schedule type   | `@schedules.register(...)`     |
| sources    | `bux/train/data.py`      | an endless stream of rows, per data type        | `@sources.register(...)`       |
| objective  | `bux/train/objective.py` | the loss: named, weighted terms, each a metric  | add a `Term`                   |
| `Trainer`  | `bux/train/trainer.py`   | model, optimizer, objective; the compiled steps | override `train_step`          |
| hooks      | `bux/train/hooks.py`     | what happens every so many steps: log, eval, save | `train(run, hooks=[...])`    |
| sinks      | `bux/train/sinks.py`     | where metrics go: the console, `metrics.jsonl`  | `train(run, sinks=[...])`      |

It uses the stories run of [Your own data](your-own-data.md), whose data is prepared
already. Run the scripts from the checkout's root with `uv run python script.py`.

## 1. Compile a run in Python

The command line is a thin layer over two functions: `bux.lang.compile_runs` turns a
file into one `Run` per combination of its alternatives, and
`bux.train.trainer.train` trains one. A script that calls them trains exactly what
`bux train` would:

```python
from pathlib import Path

from bux.lang import compile_runs
from bux.train.trainer import train

[run] = compile_runs(Path("mine/stories.bux").read_text(), filename="mine/stories.bux")
train(run)
```

`filename` lets `use` find modules next to the file, and errors point into it. Before
it trains, `bux train` also calls `bux.runtime.configure()`, which adds bux's XLA
flags and keeps compiled steps in JAX's cache; call it first in a script to do the
same. Code you register before `train` takes effect in this process; the `bux`
command does not import your script.

## 2. Replace an optimizer

A config's `type` finds its builder in a registry, and registering under a taken name
replaces the builder. So a variant of an optimizer is a function. An optimizer's
builder gets its config, the schedule's multiplier and the model, and returns an
optax transformation. Here is bux's `adamw` with Nesterov momentum, NAdamW:

```python
import optax

from bux.train.optim import lr_schedule, optimizers, weight_decay_mask


@optimizers.register("adamw")  # replaces bux's own adamw, for this process
def nadamw(c, multiplier, model):
    return optax.adamw(
        lr_schedule(c.lr, multiplier),
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=weight_decay_mask(model),
        nesterov=True,
    )
```

- `lr_schedule(c.lr, multiplier)` is the peak learning rate scaled by the run's
  schedule, warmup included.
- `weight_decay_mask(model)` decays matrices only, not norm scales, biases or other
  vectors, as bux's AdamW does.
- The run's `clip` and `precision.moments` still wrap whatever you return.

A new name, say `lion`, also needs its config in `bux/train/config.py`, in the
optimizer union, where the compiler and the editor find it.

:::{warning}
A builder is code, not a setting, so it is not in the run's id: the stories run
trained by NAdamW has the id of the one trained by AdamW. Give it another directory
(`train(run, "runs-custom")`), or it would resume the other's checkpoints.
:::

## 3. Add a hook

A hook runs after every `every` steps and after the last, and returns scalars,
metrics by name, that go to every sink. This one measures the global norm of the
parameters every 50 steps:

```python
import optax
from flax import nnx

from bux.train.hooks import Hook


class WeightNorm(Hook):
    every = 50

    def __call__(self, trainer, metrics):
        params = nnx.state(trainer.model, nnx.Param)
        return {"weight_norm": float(optax.tree.norm(params))}
```

A hook gets the `Trainer` and the step's metrics, still on the device. It may also
define `start`, before the first step, `stop`, when training stops early (the
checkpoint hook saves there), and `close`. The run's own hooks (log, eval,
checkpoint, profile) run first.

:::{tip}
`float(...)` waits for the step that made the value. bux's own log instead returns a
function that reads its scalars (`hooks.Later`), which the loop calls once the next
step is launched, so the device never waits for the host.
:::

## 4. Add a sink

A sink is where scalars go: the console and `metrics.jsonl` are sinks, and so is a
tracker. This one writes a CSV of every scalar:

```python
import csv

from bux.train.sinks import Sink, is_main_process


class CSV(Sink):
    def __init__(self, path):
        self.file = open(path, "w", newline="") if is_main_process() else None
        self.rows = csv.writer(self.file) if self.file else None

    def write(self, step, scalars):
        if self.rows:
            for name, value in scalars.items():
                self.rows.writerow([step, name, value])

    def close(self):
        if self.file:
            self.file.close()
```

In a job of many processes, every process computes every scalar, and a sink that
writes checks `is_main_process()` so that only the first does.

## 5. Train with all three

Put the three pieces of steps 2 to 4 in `custom.py`, and end it with:

```python
source = Path("mine/stories.bux").read_text()
[run] = compile_runs(source, filename="mine/stories.bux")
latest = train(run, "runs-custom", hooks=[WeightNorm()], sinks=[CSV("metrics.csv")])
print({name: round(value, 4) for name, value in latest.items() if "/" not in name})
```

```console
$ uv run python custom.py
remat=auto: none in 1s (none 0.5 GiB in 1s)
stories step  25/200  loss 3.1163  lr 3.00e-03  grad 2.072  5,624 tok/s  ...
stories step  50/200  loss 1.0522  lr 3.00e-03  grad 0.586  6,326 tok/s  ...
stories step  50/200  eval loss 1.0030  eval bpb 0.3387
...
stories step 200/200  loss 0.7165  lr 0.00e+00  grad 0.273  8,445 tok/s  ...
stories step 200/200  eval loss 0.7284  eval bpb 0.2446
{'grad_norm': 0.2731, 'loss': 0.7165, 'lr': 0.0, 'tokens': 204800, 'tokens_per_s': 8445.3161, 'eval_bpb': 0.2446, 'eval_loss': 0.7284, 'weight_norm': 234.0505}
$ grep weight_norm metrics.csv
50,weight_norm,188.64195251464844
100,weight_norm,220.77194213867188
150,weight_norm,227.19708251953125
200,weight_norm,234.0504608154297
```

`train` returns the latest value of every metric. The console shows the metrics it
has a format for, so `weight_norm` goes to the CSV and to `metrics.jsonl`, not to the
screen; `Console(..., formats=...)` adds formats (step 6).

## 6. Add a loss term, and drive a Trainer

`train(run)` builds the objective from the run's `loss` setting. To add a term to it,
build the blocks yourself. A `Trainer` takes a model, an optax transformation and an
objective; `fit` runs the loop with any hooks and sinks. This script trains the
stories model with AdamW at a constant rate and a small penalty on the logits' size;
save it as `loop.py`:

```python
"""Train the stories model with a Trainer of our own: AdamW at a constant rate, and a
penalty on the logits' size added to the loss."""

from pathlib import Path

import jax.numpy as jnp
import optax
from flax import nnx

from bux.architecture import Model
from bux.data.tokenizer import load
from bux.lang import compile_runs
from bux.train.data import batches
from bux.train.hooks import Log
from bux.train.objective import Term, objective
from bux.train.sinks import Console
from bux.train.trainer import Trainer

[run] = compile_runs(Path("mine/stories.bux").read_text(), filename="mine/stories.bux")
t = run.train

loss = objective(t.loss, run.arch)
loss.terms["logit_l2"] = Term(1e-4, lambda logits, batch: jnp.mean(logits[0] ** 2))

trainer = Trainer(Model(run.arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
trainer.fit(
    batches(t.data, t, run.arch.vocab_size),
    steps=100,
    hooks=[Log(every=25, lr=lambda step: 1e-3, tokens=t.batch * t.seq_len)],
    sinks=[Console("mine", 100, formats=Console.FORMATS | {"logit_l2": "l2 {:.1f}"})],
)
held_out = batches(t.eval.data, t, run.arch.vocab_size, held_out=True)
token_bytes = load(t.tokenizer).token_bytes()  # each token's bytes, for bits per byte
print(trainer.evaluate((next(held_out) for _ in range(4)), token_bytes))
```

```console
$ uv run python loop.py
mine step  25/100  loss 5.2276  lr 1.00e-03  grad 2.457  5,890 tok/s  ...  l2 6.7  ...
mine step  50/100  loss 2.5528  lr 1.00e-03  grad 2.645  6,793 tok/s  ...  l2 19.2  ...
mine step  75/100  loss 1.3986  lr 1.00e-03  grad 0.767  6,746 tok/s  ...  l2 19.1  ...
mine step 100/100  loss 0.9594  lr 1.00e-03  grad 0.457  7,083 tok/s  ...  l2 19.9  ...
{'bpb': 0.3196472106765284, 'loss': 0.9508468984628653}
```

What the pieces are:

- `objective(t.loss, run.arch)` is the run's loss as named, weighted terms: `loss`,
  the first head's cross-entropy, and `z_loss` and `mtp` where the run has them.
  A `Term` is a weight and a function of every head's logits and the batch; its value
  is also logged under its name, here `logit_l2`. `logits[0]` makes the first head's
  logits whole, only because a term reads them.
- `batches(data, config, vocab_size)` streams the run's batches; `held_out=True`
  draws them apart from training's.
- `trainer.evaluate(batches, token_bytes)` returns the first head's mean loss over
  the batches' targets and, given each token's bytes of text
  (`Tokenizer.token_bytes`), its bits per byte.
- `Log` turns the step's metrics into the log's scalars, with the learning rate it is
  given and the tokens a step; `Console` prints them.
- The `Trainer` computes in float32 unless you pass `compute=jnp.bfloat16`, and does
  nothing you do not give it: no clipping unless your optax chain clips, and no
  checkpoints or eval unless you pass their hooks (`bux.train.hooks.Checkpoint`,
  `Eval`).

A layer can add a term from its own forward as well:
`report(taps, "my_loss", weight, value)` in `bux.architecture` adds `weight * value`
to the training loss and logs the value, summed over the layers that report it.

:::{note}
`train_step` and `eval_step` are what the `Trainer` compiles, once. A subclass that
overrides `train_step` changes what a step does (keep an average of the weights, skip
a spike) and keeps the loop, the hooks and checkpoints
([Extending](../training/extending.md)).
:::

## 7. In a notebook

The `dev` dependency group has JupyterLab, so a notebook runs in bux's environment
with nothing to install:

```sh
uv run jupyter lab
```

Every snippet here runs in a cell as it is. A notebook's working directory is the
notebook's folder, so either start JupyterLab from the checkout's root, or give
`compile_runs` and `batches` paths that reach `mine/` and `data/packed/` from there
(`batches(..., cache="../data/packed")`).

## What you learned

| to change            | write                                                    |
|----------------------|----------------------------------------------------------|
| an optimizer         | `@optimizers.register("adamw")` over a builder           |
| what happens between steps | a `Hook`, passed as `train(run, hooks=[...])`      |
| where metrics go     | a `Sink`, passed as `train(run, sinks=[...])`            |
| the loss             | `objective(...).terms[name] = Term(weight, fn)`, with a `Trainer` |
| the loop's blocks    | `Trainer(model, tx, objective).fit(batches, steps, hooks, sinks)` |

[Extending the trainer](../training/extending.md) documents every block, and the
[Python API](../reference/python/index.md) every function used here. Next,
[scale out](scale-out.md) splits a run over many devices.
