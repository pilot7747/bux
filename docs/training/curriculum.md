# Curriculum and batch ramps

What a run reads can change as it trains. A `curriculum` is a list of phases, each of
which takes rows of another `seq_len`, another `batch` or other `data` from a step
on: a sequence-length ramp-up, a long-context extension at the end, an annealing mix.
A `batch_ramp` grows the batch over the first steps, as GPT-3's and DeepSeek-V3's do.

```bux
seq_len    = 4096
batch      = 1024
batch_ramp = ramp(start=256, steps=2k)
steps      = 100k
curriculum = [phase(at=0.8, data=0.5 * web + 0.5 * math),
              phase(at=0.95, seq_len=32768, batch=128, data=long)]
```

| Setting | Takes | Default |
|---------|-------|---------|
| `curriculum` | a {bux}`phase`, or a list of them in the order they start | `[]`, none |
| `batch_ramp` | a {bux}`ramp`, or `none` | `none` |

Both are in the run's id, so a run with either is a run of its own; a run with
neither keeps the id it had before they existed ([Run ids](running.md#run-ids)).

## Phases

```bux
curriculum = phase(at=0.9, seq_len=8192, batch=128)
```

A {bux}`phase` starts at step `at`, and from then on sets what it names; what it
leaves out stays as the phase before had it, and before the first phase the run's own
settings are in force.

| Field | Default | What it is |
|-------|---------|------------|
| `at` | required | the step it starts at, or a share of the steps below 1, rounded: `0.9` starts the last tenth |
| `seq_len` | `none` | tokens per row; none: as before |
| `batch` | `none` | rows per step; none: as before |
| `data` | `none` | a source or a mixture; none: as before |

A list of phases goes in the order they start. A phase may be written alone; a list
of one is `[phase(...),]`, since `[phase(...)]` is a residual branch
([Values](../language/values.md#values)).

**A sequence-length ramp-up** is phases of longer rows. With fewer rows a step, a step
keeps its tokens, as in `examples/experiments/curriculum.bux`:

```bux
seq_len    = 64
batch      = 16
batch_ramp = ramp(start=4, steps=20)
steps      = 80
curriculum = [phase(at=0.5, seq_len=128, batch=8),
              phase(at=0.75, seq_len=256, batch=4)]
```

```console
$ bux check examples/experiments/curriculum.bux
d_model=64 vocab_size=512 layers=11
rmsnorm×5  gqa×2  gated_mlp×2  embed×1  head×1
80 steps = 71.4k tokens
from step 0: 16 × 64 tokens, ramped from 4 rows over the first 20 steps in 3 sizes
phase 1, from step 40: 8 × 128 tokens
phase 2, from step 60: 4 × 256 tokens
optimizer=adamw(lr=0.003) schedule=wsd(warmup=10)
data=synthetic()
```

A gradual sequence-length warmup, a new length every few steps, would compile as
many steps and prepare as many datasets as it has lengths (below), so bux takes it
as phases, a few lengths each a stretch of the run.

## Batch ramps

```bux
batch      = 15360
batch_ramp = ramp(start=3072, steps=30k)
```

A {bux}`ramp` grows the batch from `start` rows at the first step to the run's
`batch` at step `steps`: linearly, rounded down to a multiple of `multiple`
(`start` unless set), and never below `start`. The ramp above takes 3072, 6144,
9216 and 12288 rows, each for a quarter of its 30k steps, then 15360.

| Field | Default | What it is |
|-------|---------|------------|
| `start` | required | rows per step at the first step |
| `steps` | required | the steps it takes to reach the run's batch |
| `multiple` | `none` | of the batch sizes it takes; none: `start` |

Every batch size compiles the step once more, so a coarse `multiple` compiles fewer.
The ramp leaves the learning rate to the schedule, whatever the batch: one that warms
up over the ramp is `schedule = wsd(warmup=30k)`.

## What each phase reads

Each stretch of steps of one shape and one data reads its batches as a run of its own
would ({py:func}`bux.train.data.training_batches`), from where the stretches before
it left each source:

- a source read at the same `seq_len` goes on from the row it stopped at, so a ramp,
  a phase that only changes the batch, and a phase that mixes in a new source repeat
  no row of it;
- a source at another `seq_len` is a dataset of its own, which starts from its first
  row: `text` and `chat` documents are packed once for each `seq_len`, and `bux
  prepare` and `bux train` prepare every phase's
  ([The pipeline](../data/pipeline.md));
- a run that resumes at step `i` reads what it would have had it never stopped
  ([Determinism](../data/sources.md#determinism)).

An eval reads rows of the run's own `seq_len` and `batch`, whichever phase it is in,
so its loss compares across phases; one that sets its own reads those
([Long rows](running.md#long-rows)), as an eval of the long rows a long-context phase
trains on.

## Compiles

The step compiles again wherever the shape of its batch changes. Under `methods(remat=auto)` it chooses its policy again, as a longer row may not fit
under the one before's ([Remat](../parallelism/remat.md)). The console's `compiled in`
says how long each took, in the log after it:

```console
$ bux train examples/experiments/curriculum.bux
curriculum: runs/curriculum-ad5785cd6b, on 1 × cpu
curriculum: 156k parameters, 80 steps = 71.4k tokens, adamw(lr=0.003), wsd(warmup=10)
curriculum: from step 0: 16 × 64 tokens, ramped from 4 rows over the first 20 steps in 3 sizes
curriculum: phase 1, from step 40: 8 × 128 tokens
curriculum: phase 2, from step 60: 4 × 256 tokens
curriculum: log every 10
remat=auto: none in 1s (none 0.0 GiB in 0s)
remat=auto: none in 0s (none 0.0 GiB in 0s)
curriculum step 10/80  loss 6.2552  lr 3.00e-03  grad 0.621  2,853 tok/s  117 ms/step  data 0.1 ms  compiled in 1.1s  0:00:01 < 0:00:08
...
curriculum step 80/80  loss 6.2519  lr 0.00e+00  grad 0.221  41,252 tok/s  25 ms/step  data 0.5 ms  0:00:05 < 0:00:00
```

The log's `tokens` counts each step's own batch, so it is the tokens trained on
through every phase, and `tokens_per_s` the tokens of the steps since the last log
over their time. The MFU counts each step's FLOPs at its own shape
([MFU](running.md#mfu)), and `bux plan`'s FLOPs a token average over the phases'
`seq_len` by their tokens.

## Checks

The compiler checks that the phases and the ramp fit the run, and points at the
setting that does not:

```console
$ bux check examples/experiments/curriculum.bux 'batch_ramp = ramp(start=4, steps=60)'
error: phase 1 sets the batch at step 40, before the batch ramp is over at step 60
  --> examples/experiments/curriculum.bux:24:14
   |
24 | curriculum = [phase(at=0.5, seq_len=128, batch=8),
   |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

| What | The error says |
|------|----------------|
| Phases start in order, inside the run | `phase 2 starts at step 30, not after the phase before it (step 40): phases go in the order they start` |
| The ramp starts below the batch it grows to | `the batch ramp starts at 16 rows, not below the run's batch of 16, which it grows to` |
| The ramp ends inside the run | `the batch ramp takes 100 steps, and the run has 80` |
| No phase sets the batch while the ramp grows it | `phase 1 sets the batch at step 40, before the batch ramp is over at step 60` |
| A phase's `text` and `chat` data have a tokenizer, and its rows fit a learned positional embedding | as for the run's own ([Checks](settings.md#checks)) |

When the run starts, every shape it takes must split over the mesh and the
microbatches, and every source of each phase's mixture have a row in its batch; an
error after the first stretch says from which step:
`from step 40, batch = 8 does not split over layout.batch, data=16: make it a multiple of 16, or change mesh or layout`.

## In Python

{py:func}`bux.train.curriculum.stretches` cuts a run's steps into the stretches of one
shape and one data, each with the run's settings as they are then:

```python
from bux.train.curriculum import stretches

for s in stretches(run.train):
    print(s.start, s.stop, s.config.batch, s.config.seq_len)
```

A {py:class}`~bux.train.trainer.Trainer` in a script trains on
{py:func}`~bux.train.data.training_batches` to follow them, and compiles again at
each new shape on its own ([A Trainer in a script](extending.md#a-trainer-in-a-script)).
