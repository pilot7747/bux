# Settings

A run's training settings are top-level definitions with reserved names, next to
`d_model` and `model`. This page lists every one, says which are required, and shows
the checks bux runs on them before anything trains. The
[language reference](../reference/language/training.md) has every field of every
setting, generated from the code.

```bux
tokenizer = tokenizer(name="gpt2")
data      = 0.8 * text(path="data/web/*.parquet") + 0.2 * text(path="data/code/*.parquet")
packing   = best_fit
optimizer = muon(lr=0.02)
schedule  = wsd(warmup=1k, decay=0.2)
seq_len   = 2048
batch     = 256
steps     = 10B // (batch * seq_len)
```

A setting's value is a config called by name, as a layer is (`muon(lr=0.02)`), or a
number. Settings are definitions, so arithmetic, reads of other settings and the
number suffixes `k`, `M`, `B` and `T` work on them: `steps = 10B // (batch * seq_len)`
([Values](../language/values.md)). A file changes one field of a setting it got from
another file with an amendment, `optimizer.lr = 3e-3` ([Modules](../language/modules.md)).

Each config's fields are pydantic fields in `bux/train/config.py`, and the editor
shows them on hover and completes them, as it does for layers
([Editor support](../language/tooling.md)).

## Every setting

| Setting | Takes | Default | What it is |
|---------|-------|---------|------------|
| `data` | {bux}`text`, {bux}`chat`, {bux}`tokens`, {bux}`synthetic`, or a mixture of them, `0.7 * a + 0.3 * b` | required | the rows training reads ([Data sources](../data/sources.md)) |
| `tokenizer` | {bux}`tokenizer`, or `none` | `none` | the tokenizer of `text` and `chat` documents; required with them |
| `packing` | {bux}`concat`, {bux}`greedy` or {bux}`best_fit` | `best_fit` | how documents become rows ([Packing](../data/packing.md)) |
| `optimizer` | {bux}`adamw` or {bux}`muon` | required | the update rule ([Optimizers](optimizers.md)) |
| `schedule` | {bux}`constant`, {bux}`cosine` or {bux}`wsd` | `constant` | what multiplies the learning rate at each step ([Schedules](optimizers.md#schedules)) |
| `seq_len` | an integer, at least 2 | required | tokens a row |
| `batch` | a positive integer | required | rows a step |
| `steps` | a positive integer | required | optimizer steps |
| `seed` | an integer | `0` | the model's initial weights, the order of the data, and every random stream |
| `clip` | a number, or `none` | `1.0` | the global norm gradients are clipped to ([Clipping](optimizers.md#clipping)) |
| `loss` | {bux}`loss` | `loss` | the weights of the z-loss and of multi-token prediction ([The loss](optimizers.md#the-loss)) |
| `eval` | {bux}`eval`, or `none` | `none` | a held-out loss every so many steps ([Eval](running.md#eval)) |
| `checkpoint` | {bux}`checkpoint`, or `none` | `none` | saves to resume from ([Checkpoints](running.md#checkpoints)) |
| `log` | {bux}`log` | `log` | the metrics every `every` steps, 10 by default ([The log](running.md#the-log)) |
| `profile` | {bux}`profile`, or `none` | `none` | a trace of a few steps ([Profiling](../performance/profiling.md)) |
| `precision` | {bux}`precision` | `precision` | the dtype a run computes in, and the formats its matmuls take ([Precision](precision.md)) |
| `mesh` | {bux}`mesh` | `mesh` | the device grid; unset, every device on `data` ([Parallelism](../parallelism/index.md)) |
| `layout` | {bux}`layout` | `layout` | which mesh axes each named dimension splits over ([Layout](../parallelism/layout.md)) |
| `methods` | {bux}`methods` | `methods` | how each part that communicates is computed, remat and microbatches ([Parallelism](../parallelism/index.md)) |
| `compiler` | {bux}`compiler` | `compiler` | XLA's options for the step, and where its gradients' reductions start ([XLA's options](../performance/compiler.md)) |

The architecture's own settings, `model`, `d_model`, `vocab_size` and `residual`, are
in [Top-level settings](../reference/language/top-level.md).

:::{note}
`mesh`, `layout`, `methods` and `compiler` change how a run computes, not what, and
`log`, `eval`, `checkpoint` and `profile` only watch it. None of the eight is in the
run's id, so a run resumes under other values of them
([Run ids](running.md#run-ids)).
:::

## What is required

A run needs five settings: `data`, `optimizer`, `seq_len`, `batch` and `steps`. Its
model needs `model`, `d_model`, and `vocab_size` when it has an `embed` or a `head`.
Every other setting has a default.

A file is a **run** once it sets any training setting, and then it needs all five.
This `missing.bux` sets two:

```bux
d_model    = 64
vocab_size = 512
model = embed [rmsnorm gqa(num_heads=4, num_kv_heads=2)]^2 rmsnorm head

data      = synthetic
optimizer = adamw(lr=3e-3)
```

```console
$ bux check missing.bux
error: this file sets how to train, so it is a run, and a run needs seq_len, batch and steps
hint: seq_len = 1024
```

A file with no `model` is a **module**: a piece that runs build on by name, such as
the recipe `examples/experiments/recipes/fineweb_wsd.bux`, which sets the data, the
optimizer and the schedule but has no model. `bux check` checks a module on its own,
its settings against their types, and says what it defines:

```console
$ bux check examples/experiments/recipes/fineweb_wsd.bux
a module: it defines tokenizer, web, code, data, packing, optimizer, schedule, loss, seq_len, batch, budget, steps, eval, checkpoint, and no model
```

A run uses it, `use fineweb_wsd`, next to an architecture
([Modules](../language/modules.md)). `budget` there is no setting: it is a definition
of the recipe's own, which its `steps` reads.

## Checks

Before anything runs, the compiler checks each setting against its type, where it is
written, and that the settings fit together. `bux check`, `bux plan`, `bux train`,
`bux launch` and the editor all run the same checks, and every error points at the
line that caused it, in whichever file wrote it.

### Types and ranges

Each setting takes the configs its row above lists, and each field of a config its
type:

```console
$ bux check examples/experiments/smoke.bux 'optimizer = cosine'
error: 'optimizer' must be adamw(...) or muon(...), found a cosine config
 --> <command line>:1:13
  |
1 | optimizer = cosine
  |             ^^^^^^
$ bux check examples/experiments/smoke.bux 'clip = true'
error: 'clip' must be a number or none, found a boolean
 --> <command line>:1:8
  |
1 | clip = true
  |        ^^^^
$ bux check examples/experiments/smoke.bux batch=0
error: 'batch' must be greater than 0, found 0
 --> <command line>:1:7
  |
1 | batch=0
  |       ^
$ bux check examples/experiments/smoke.bux 'schedule = wsd(decay=1.5)'
error: wsd: decay: Input should be less than or equal to 1
 --> <command line>:1:16
  |
1 | schedule = wsd(decay=1.5)
  |                ^^^^^^^^^
```

A field a config does not have is an error, with the nearest name it does have:

```console
$ bux check examples/experiments/smoke.bux 'optimizer = adamw(lr=3e-3, weight_dacay=0.1)'
error: adamw has no setting 'weight_dacay'
 --> <command line>:1:28
  |
1 | optimizer = adamw(lr=3e-3, weight_dacay=0.1)
  |                            ^^^^^^^^^^^^^^^^
  = hint: did you mean 'weight_decay'?
```

Some fields check their values further:

| Field | Its check |
|-------|-----------|
| `split` of `text` and `chat` | `[start, end]`, with `0 <= start < end <= 1` |
| `roles.experts` of `precision` | `compute`, `mxfp8` or `mxfp8_2d`: the formats a grouped GEMM takes |
| `options` of `compiler` | each option written `--name=value` |
| `switch.at` of `precision` | strictly between 0 and 1 |
| `decay` of `wsd` | between 0 and 1 |

```console
$ bux check examples/experiments/smoke.bux 'precision = precision(matmuls=nvfp4, roles=roles(experts=nvfp4))'
error: roles: experts: Value error, the routed experts multiply in compute, mxfp8, mxfp8_2d: bux has no grouped GEMM in nvfp4
 --> <command line>:1:50
  |
1 | precision = precision(matmuls=nvfp4, roles=roles(experts=nvfp4))
  |                                                  ^^^^^^^^^^^^^
```

### Settings that must fit together

The warmup and the cooldown fit in the steps. The error points into the module that
wrote the schedule:

```console
$ bux check examples/experiments/attention_ablation.bux steps=50
error: the schedule takes 1010 steps (1000 of warmup and 10 of cooldown), but the run has 50
  --> examples/experiments/recipes/fineweb_wsd.bux:19:13
   |
19 | schedule  = wsd(warmup=1k, decay=0.2, shape=sqrt)
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

`text` and `chat` documents, in `data` or in `eval`, need a tokenizer:

```console
$ bux check notok.bux
error: text data is tokenized, so the run needs a tokenizer
 --> notok.bux:5:13
  |
5 | data      = text(path="data/web/*.parquet")
  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  = hint: tokenizer = tokenizer(name="gpt2")
```

Rows fit in the positions a learned positional embedding has:

```console
$ bux check positions.bux
error: sequences of 2048 tokens are longer than the 1024 positions the embedding learns
 --> positions.bux:7:13
  |
7 | seq_len   = 2048
  |             ^^^^
  = hint: seq_len = 1024, or embed(max_positions=2048)
```

In a sweep, `bux check` and the editor check every alternative once, with the others
at their first ([Sweeps](../language/sweeps.md)).

### Checks when a run starts

A few checks need the devices or the data, so they run when training starts, before
the model is built or the step compiles:

| What | The error says |
|------|----------------|
| The mesh fits the devices | `mesh: the axes other than data take 16 devices, which do not divide the 8 there are; set mesh.data` |
| The batch splits into `methods.microbatches`, then over the batch axes, and `seq_len` over the axes of `seq` and `stream` | `batch = 12 does not split over layout.batch, fsdp=8: make it a multiple of 8, or change mesh or layout` |
| Every method, format and builder a run names is registered | `the trainer has no 'fp8' matmul format yet; add one with @bux.architecture.precision.matmuls.formats.register('fp8')` |
| XLA on a CPU can compile the run in bfloat16 | `XLA:CPU cannot train an MoE by methods(moe=sorted) in bfloat16 on more than one device: set precision(compute=float32), or methods(moe=dense) for an MoE` |
| Every source of a mixture has a row in each batch | `a batch of 8 rows has no room for a source of weight 0.001` |
| A prepared dataset's tokens fit the model's vocabulary | `data/packed/text-... is in 50257 tokens, more than the model's vocab_size of 32000` |
| A glob of documents or shards matches files | `no files match 'data/web/*.parquet'` |

`bux launch` checks the mesh of every run against the GPUs it has before it starts
any, and records a run that needs more as failed
([Launching](launching.md#gpus-and-workers)).

## What a run is in Python

`bux.lang.compile_runs(source)` returns a {py:class}`bux.lang.ir.Run` for every run a
file makes: its {py:class}`bux.lang.ir.Architecture`, its
{py:class}`bux.train.config.TrainConfig` (one field per setting on this page), its
name, and the alternatives it took. All of it is plain pydantic data, and
{py:func}`bux.train.trainer.train` trains it:

```python
from bux.lang import compile_runs
from bux.train.trainer import train

path = "examples/experiments/smoke.bux"
for run in compile_runs(open(path).read(), filename=path):
    print(run.name, run.id, run.train.optimizer)
    train(run)
```

The `filename` lets `use` find modules next to the file ([Modules](../language/modules.md)).
