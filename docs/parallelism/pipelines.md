(parallelism-pipelines)=
# Pipelines

Pipeline parallelism (PP) splits a model's layers over the `stage` axis: each stage
holds a share of consecutive layers, and microbatches flow from stage to stage. A
pipeline sends little, one microbatch's stream at a time, so it suits the slowest
links. It pays in its bubble, the share of a step its stages idle, and in its
microbatches' smaller GEMMs.

Which to choose:

- **Prefer FSDP where it works** (HSDP across nodes). A pipeline of the 1.3B trains
  47% slower per GPU than FSDP on one node of B300s, and 42% slower on four nodes.
- **Across slow links**, put one stage on each node, as `pp_70b.bux` does in place of
  FSDP across nodes. A pipeline keeps its rate per GPU from one node to eight
  ([Across nodes](#across-nodes-one-stage-per-node)).
- **Keep the default schedule**, `1f1b` on balanced stages. On one node it trains the
  1.3B 8% faster than `gpipe` on 2 stages in half its memory, 22% faster on 4 stages
  in a fifth less, and as fast on 8 in a seventh less
  ([The defaults against gpipe](#the-defaults-against-gpipe)).
- **Give it microbatches.** With S stages and M microbatches, a stage idles
  (S − 1) / (M + S − 1) of a step.

bux pipelines one repeat of the model: the longest scanned repeat outside any other,
as `block^80` under `methods(repeat=scan)`.

## Quick start

Set `mesh.stage`, scan the repeat, and give the step microbatches. From
`examples/experiments/parallel/pp_70b.bux`, Llama 3 70B on 4 nodes of 8 GPUs, a
pipeline across the nodes and TP inside each:

```bux
use llama3_70b
use fineweb_wsd

mesh      = mesh(stage=4, tensor=8)
layout    = layout(stream=tensor)
precision = precision(compute=bfloat16)
methods   = methods(repeat=scan, remat=full, pipeline=circular, rounds=2,
                    microbatches=16)
optimizer.lr = 1.5e-4
```

Each node holds 20 of the 80 blocks, in two rounds of 10 (a circular pipeline), and a
step's 256 rows go through in 16 microbatches. A stage idles for
(S − 1) / (R·M + S − 1) = 3 / 35 of a step.

The other schedules and splits, one line each (a file sets `methods` once):

```bux
mesh    = mesh(stage=4, fsdp=2)
methods = methods(repeat=scan, microbatches=8)                        # 1f1b, stages balanced
methods = methods(repeat=scan, microbatches=8, pipeline=gpipe)
methods = methods(repeat=scan, microbatches=8, pipeline=circular, rounds=2)
methods = methods(repeat=scan, microbatches=8, pipeline="zbh1")       # zero bubble
methods = methods(repeat=scan, microbatches=8, stages=[5, 4, 4, 3])
```

:::{note}
Quote the schedules' names that start with a digit, `pipeline="1f1b"`: the DSL reads
`1f1b` as a number.
:::

## Options

| Setting | Values | Default | What it does |
|---|---|---|---|
| `mesh.stage` | a size | `1` | the stages; above 1, the longest scanned repeat is pipelined |
| `methods.pipeline` | `1f1b`, `zbh1`, `gpipe`, `circular` | `1f1b` | the schedule |
| `methods.microbatches` | a count | `1` | the microbatches the batch's rows split into, which the pipeline sends through its stages |
| `methods.rounds` | a count | `1` | the times a circular pipeline sends a microbatch round the stages |
| `methods.stages` | a list of passes, one per stage | none | each stage's passes of the pipelined repeat, first to last. None: under `1f1b` and `zbh1` the split that balances the stages' FLOPs; otherwise as even as it goes |
| `methods.repeat` | `scan` | `unroll` | must be `scan`: only a scanned repeat pipelines |
| `layout.stages` | axes | `stage` | what the stacked passes of a pipelined repeat split over |

These are fields of {bux}`mesh`, {bux}`methods` and {bux}`layout`. The schedules, with
S stages, M microbatches and R rounds:

| Method | Each microbatch | A stage holds | A stage idles |
|---|---|---|---|
| `gpipe` | goes through the stages once | M + S − 1 ticks | (S − 1) / (M + S − 1) |
| `circular` | goes round them `methods.rounds` times, each stage holding a share of every round's layers | R·M + S − 1 ticks | (S − 1) / (R·M + S − 1) |
| `1f1b` | as in gpipe, and its backward runs as soon as it can; the default | 2S − 1 | (S − 1) / (M + S − 1) |
| `zbh1` | as in gpipe, its backward split in two, the weights' part where a stage would idle | S, and S of the weights' | (S − 1) / (3M + S − 1) |

What a stage holds is the residuals of that many ticks or microbatches. The bubble of
`zbh1` counts a forward, a backward of the input and one of the weights as a tick
each.

## Constraints and errors

Each of these is an error that says what to change:

- **No scanned repeat.** With `mesh.stage` above 1 and no scanned repeat: repeat the
  blocks (`block^8`) and scan them, `methods(repeat=scan)`.
- **Too few passes.** A repeat of fewer passes than stages: give it at least one pass
  per stage.
- **A split that does not fit.** `methods.stages` must give every stage a pass and sum
  to the repeat's passes, in one round: it splits one round, not `methods.rounds`.
- **Rounds.** Only `circular` goes round the stages more than once: `methods.rounds`
  above 1 under `gpipe`, `1f1b` or `zbh1` says to use `methods(pipeline=circular)`.
  A circular pipeline takes at least one microbatch per stage, and the repeat's passes
  must be a multiple of `stage × rounds`.
- **Taps.** A stage gets the stream and the taps of its rows (the tokens, the
  documents), split with it. A pass that reads a tap written before the repeat
  (`$v1`), or leaves one changed after it (single-pass mhc's read, a U-net's skips),
  is an error that names the tap: a pipeline sends its stages the stream alone.
  Sending taps along is not built yet.
- **A layer outside the repeat that learns from the whole batch**, as an MoE's router
  does, under `1f1b` or `zbh1`: these run the layers outside the repeat one microbatch
  at a time, on one stage. Put it in the repeat, or use `methods(pipeline=gpipe)`.
- **Precision's `first` and `last` branches** multiply in `compute` before and after
  the pipelined repeat, and are an error inside it.
- **`precision(reduce=float32)` under `gpipe` and `circular`**, whose embedding and
  head run outside the pipeline and reduce in the backward. Under `1f1b` and `zbh1`,
  where every gradient sums in float32 as it leaves, it is no error.
- **`methods(interleave=true)`**, which runs a step's microbatches side by side: a
  pipeline schedules its own ([Side by side](microbatches.md#side-by-side-methodsinterleave)).
- **Batches.** The batch's rows must split into `methods.microbatches`, and each
  microbatch over the batch axes ([GPUs, hosts and the plan](index.md#gpus-hosts-and-the-plan)).

With one stage nothing pipelines, and a model's MTP modules run on the last stage,
with its head.

## The bubble in `bux plan`

`bux plan` shows the share of a step a stage idles, the bubble, without any GPU:

```console
$ bux plan examples/experiments/smoke.bux 'methods = methods(repeat=scan, microbatches=4)' \
    'mesh = one: mesh(data=1) | pp: mesh(stage=2, data=1)'
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
5a4f0f7d85  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[one]
5a4f0f7d85  156k    25.6k   0.2        984k       2     369KiB      738KiB     20%     smoke[pp]
```

Two stages and four microbatches under `1f1b` idle (S − 1) / (M + S − 1) = 1 / 5 of a
step.

## How it works

The pipelined repeat is the longest scanned repeat outside any other (the first of
several as long). Its stacked layers split over `stage` by a dimension named `stages`
of the layout ([Names](layout.md#names)), one share of consecutive passes per stage,
as `methods.stages` gives them or as a run balances them
([Stages](#stages-methodsstages)). `stage` comes first in the mesh
([The mesh](index.md#the-mesh)), because a pipeline's sends need the least bandwidth,
and because XLA nests a `shard_map` inside a pipeline's only over the axes after
`stage`.

The rest of the model, the embedding and the head too, does not split over `stage`:

- under `gpipe` and `circular` every stage computes it;
- under `1f1b` and `zbh1` the first stage runs the embedding and the last the head
  ([Schedules](#pipelines-schedules)).

### Ticks

The batch's rows split into `methods.microbatches` microbatches, row `i·M + j` into
microbatch `j`, so each splits over the batch axes as the rows do. A method runs them
in a `jax.shard_map` manual over `stage` alone, by a `lax.scan` over `R·M + S − 1`
ticks:

1. At each tick, every stage runs its share of a round on a microbatch
   (`stacked.run`).
2. A `ppermute` sends its output on to the next stage, the last's to the first.
3. The first stage takes each microbatch from its own buffer, where the last stage's
   outputs come back. There a circular round starts again, and from there the stream
   leaves, summed over the stages to every one.

### What a pipeline computes

The passes run once per microbatch, as if the microbatches went through the model one
after another, so with one microbatch a pipeline computes what one device does:

- **The losses a layer reports**, such as the MoE's balance loss, are the mean over
  microbatches of each one's: the whole batch's for a per-sequence loss
  (`aux_loss_scope=sequence`), close to it for a per-batch one.
- **The router bias** moves once per step, from the whole batch's picks, as under
  gradient accumulation: in a training step the MoE has `update_at_step_end`, and the
  pipeline carries its `router_load` through the ticks. A training forward outside a
  step moves it once per microbatch, each from that microbatch's load.
- **Dropout** draws a fresh mask for every microbatch, and every step. Outside the
  repeat it does so under `1f1b` and `zbh1`, which run the rest of the model one
  microbatch at a time; `gpipe` and `circular` draw one mask there for the whole
  batch.

### Gathering and reducing

The microbatches are the pipeline's: the trainer asks `pipeline.pipelined()`, and a
step does not also accumulate gradients over them.

- **Gathered once.** It gathers the parameters FSDP splits once per step, before the
  ticks (`methods.gather = step`,
  [Gathered once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
  in float32, since the ticks add up a stage's gradients in the dtype of what they
  read. `gather=layer` gathers each layer's as it runs, every tick, both ways.
- **Reduced once.** Where the rows are independent and their axes span processes,
  `accumulate=auto` (or `accumulate=unreduced` anywhere) reduces the gradients once
  too. The pipeline's `shard_map` is then manual over the axes that split the rows as
  well as over `stage` (`gather.apart`). A stage runs only its device's rows, each
  tick's gradients stay its own, and the `shard_map`'s transpose sums them over those
  axes once, as they leave it, in float32.
- **Otherwise** (`accumulate=reduced`, or an MoE) they reduce every tick, as the
  backward makes them.

### Resuming on another pipeline

- On stages of equal passes, the stacked state keeps the shapes and the order of
  `repeat=scan`'s, so a run resumes on another number of stages, or on none.
- On uneven stages, as a run balances them under `1f1b`, it resumes on the same split
  only ([Stages](#stages-methodsstages)).
- `circular` stacks each round's layers as a separate repeat, and resumes as circular
  with as many rounds.

(pipelines-schedules)=
## Schedules: `1f1b`, `zbh1`

```bux
methods = methods(repeat=scan, microbatches=8, pipeline="1f1b")
methods = methods(repeat=scan, microbatches=8, pipeline="zbh1")
```

`gpipe` and `circular` run the forward alone, and JAX's transpose of their scan of
ticks is the backward, so a stage holds every tick's residuals until the backward.
`1f1b` and `zbh1` take each microbatch's backward by hand (`bux.parallel.scheduled`),
at the ticks of a table (`bux.parallel.schedule`), and a stage keeps a microbatch's
residuals only from its forward to its backward:

- The forward is a `jax.vjp`, whose residuals are a pytree of arrays. A few
  microbatches' residuals sit in a buffer, and the backward rebuilds the vjp from its
  slot, with the stage's parameters put back rather than saved with each.
- A table gives each stage its ops in order, a queue per lane of a tick. At each tick
  a lane runs the next op of its queue once what the op reads is there.
- At the tick's end a stage sends its forward's output on, and its backward's input
  gradient back.
- Each kind of op runs under its own `lax.cond`. In one `lax.switch` of them all, XLA
  copied every buffer that one op writes and another reads, every tick: `1f1b` on
  `stage=2, fsdp=4` ran at 534k tokens/s with a peak of 40.4 GiB, against 673k and
  24.6 GiB with a cond each.

The two tables:

- **`1f1b`**: two lanes per tick, a forward and a whole backward, each as soon as it
  can run. Stage s holds 2(S − s) − 1 microbatches, not M + S − 1 ticks, and the
  stages idle as gpipe's do: as lockstep ticks they cannot keep the S − s of an
  asynchronous 1F1B without idling longer.
- **`zbh1`**: one lane, and the backward split in two. `B` takes the input's gradient,
  on the critical path, and `W` the weights' later, in a tick the stage would idle.
  Stage s runs S − s forwards, then in turn a backward, the weights of the microbatch
  s before it, and a forward (ZB-H1 of Qi et al., "Zero Bubble Pipeline
  Parallelism"). Every stage holds S microbatches' residuals, each until its `W`, and
  S of what `B` leaves `W`, and the stages idle S − 1 ticks of 3M + S − 1.

### Splitting a backward: `bux.parallel.split`

`bux.parallel.split` splits a backward by its equations. Those that lead to the
input's gradient run in `B`, and hand the values they read to those that lead only to
the weights':

- a scan of the backward, the one over a stage's layers, splits into a scan of each
  part, the first stacking what each pass saves for the second;
- a cond splits into a cond of each;
- calls of jaxprs (jit, remat) are inlined;
- any other equation, such as an MoE's shard_map, goes whole to `B`.

No equation runs twice: of 4 Llama blocks' backward, `W` runs the 28 matmuls of the
weights' gradients and `B` the rest, which `tests/test_schedules.py` checks against
the whole backward, with remat too.

### The whole model, cut at the repeat

A backward needs the gradient of the loss as soon as the last stage has a
microbatch's forward, so these schedules take the whole model, cut at the pipelined
repeat (`trainer._Cut`):

- the model's own forward, one microbatch at a time, up to the repeat on the first
  stage (`pipelined.stream_into`);
- and from the repeat to the loss on the last (`pipelined.leaving`), where the stage's
  layers stand in by their shapes.

So the embedding runs on the first stage alone, and the head and the objective on the
last, each once per microbatch, where gpipe runs them on every stage for the whole
batch. What follows from that:

- Their gradients add up in float32 on their stage and sum over the stages as they
  leave, and the stages' own add up in float32 on their devices. That is why
  `precision(reduce=float32)` is no error here.
- The losses the stages' layers report enter the loss by their weights, and their
  means its metrics; the router bias moves once per step, as under gpipe.
- The state of the layers outside the repeat, such as dropout's count of its masks,
  leaves the step as the last stage's forwards leave it: they run the model from
  the start.
- Gathered once and reduced once work as for gpipe: the parameters come in cast to
  `compute`, and gathered too under `gather=step`, and where the rows are
  independent each device runs its own and every gradient sums over the rows' axes
  once, at the end.
- As under gpipe, `remat(first=n)` counts a stage's passes from its own first
  ([Remat](remat.md)).
- The loss takes its logits by either method of `methods.loss`.
- A forward alone, as an eval's, runs gpipe's schedule: `1f1b` is the schedule of a
  training step's backward.

(pipelines-stages)=
## Stages: `methods.stages`

```bux
mesh    = mesh(stage=4, fsdp=2)
methods = methods(repeat=scan, microbatches=8, stages=[5, 4, 4, 3])
```

`methods.stages` gives each stage its passes of the pipelined repeat, first to last.

### The balanced split, the default

Left out, under `1f1b` and `zbh1` a run splits the passes so its stages do as much
work each as they can (`bux.train.stages`). The first stage also runs the model up to
the repeat (the embedding), and the last the model after it, to the loss (the head).
So the run:

1. counts the FLOPs of a row's forward (`bux.train.flops`) in a pass, before the
   repeat and after it, on the model cut at the repeat as the schedule cuts it;
2. takes the split whose slowest stage does the least; of those as slow, the most
   even; and then the one heaviest at the front (`pipeline.balanced`).

The embedding is a gather, no FLOPs. The 1.3B's head, at a vocabulary of 128k and rows
of 4096, is 4.7 passes' work, so its 16 passes split `[10, 6]` over 2 stages,
`[5, 5, 5, 1]` over 4 and `[3, 2, 2, 2, 2, 2, 2, 1]` over 8. `bux train` says the
split it takes, and `bux plan` counts the stages' parameters by it. Under `gpipe` and
`circular`, where every stage runs the rest of the model, the passes split as evenly
as they go.

### Uneven stages

A sharding splits the stacked layers evenly, so each stage holds as many as the most
any stage runs (`pipeline.spread`): its own, then copies of its first, which it skips
by a `lax.cond`. The copies are parameters too, with their optimizer state, and their
gradient is 0. A run resumes on the same split only, and on another it fails at once,
since its checkpoints keep the split ([Resuming](../training/launching.md#resuming)).

## Measurements

The 1.3B dense model of
[Gathered once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)
(16 Llama blocks, a vocabulary of 128k), 64 rows of 4096 per step in bfloat16,
`repeat=scan`, on B300s, 2026-10-23 to 2026-10-24.

### On one node of 8 B300s

8 microbatches, `loss=logits`, tokens/s and the peak memory of a GPU (2026-10-23):

| mesh | `methods.pipeline` | tok/s | peak |
|---|---|---|---|
| `stage=2, fsdp=4` | `gpipe` | 625k | 97.8 GiB |
| | `circular`, `rounds=2` | 643k | 95.2 GiB |
| | `"1f1b"` | 673k | 24.6 GiB |
| | `"1f1b"`, `stages=[10, 6]` | 697k | 33.6 GiB |
| | `"zbh1"` | 594k | 28.8 GiB |
| `stage=4, fsdp=2` | `gpipe` | 508k | 117.9 GiB |
| | `circular`, `rounds=2` | 536k | 111.1 GiB |
| | `"1f1b"` | 507k | 49.5 GiB |
| | `"1f1b"`, `stages=[5, 5, 5, 1]` | 613k | 60.6 GiB |
| | `"zbh1"` | 467k | 53.8 GiB |

**The head makes the last stage the slowest.** At a vocabulary of 128k the head is as
much work as 4.7 blocks' forward, so with as many blocks as the others the last stage
is the slowest, and the others wait for it. In a profile of `1f1b` on `stage=2`, the
first stage's GPUs spend 124 ms of a 400 ms step waiting in a send, and the last
stage's spend 202 ms in GEMMs, against the first's 137. Fewer blocks on the last stage
take that back: on 4 stages the head is nearly a stage's work, and
`stages=[5, 5, 5, 1]` runs 21% faster than gpipe in half its memory.

**More microbatches widen the gap.** At 16 microbatches:

- on `stage=4`, 652k tokens/s at 32.0 GiB, against gpipe's 521k at 117.8 GiB;
- on `stage=2, stages=[10, 6]`, 651k at 16.9 GiB, against 578k at 107.1 GiB.

**With the loss by chunks**, the default, gpipe holds less and the gap stays:

- on `stage=2`, 624k at 60.0 GiB, against `stages=[10, 6]`'s 679k at 30.8 GiB;
- on `stage=4`, 484k at 65.0 GiB, against `stages=[5, 5, 5, 1]`'s 596k at 52.7 GiB.

**`zbh1` is slower here than `1f1b`**, and no faster on uneven stages: `B` writes out
what `W` reads, and `W` reads it back. On `stage=2` a step spends 48 ms of 461 in
copies, against 25 of 400 under `1f1b`; at 8 microbatches the bubble it saves is
smaller than that.

- Splitting the head's backward too, its input's gradient into the last stage's `B`
  and its weights' into its `W`, balanced the stages but ran no faster (543k on
  `stage=2, stages=[10, 6]`, 482k on `stage=4, stages=[5, 5, 5, 1]`). It held the
  head's residuals, a microbatch's logits, in a slot of each stage, since every stage
  runs one program (113 GiB on 4 stages). So the head's backward stays whole, in the
  last stage's forward.
- Reduced once (`accumulate=unreduced`), `1f1b` runs as fast (676k on `stage=2`) and
  `zbh1` slower (467k).
- With GEMM autotuning off and deterministic ops, `gpipe`, `1f1b` and `zbh1`, on even
  stages and uneven, give the same loss at the first step, and within 7e-4 at the
  tenth, on a loss falling 0.07 per step.

### Splits around the balanced one

The 1.3B on one node of 8 B300s under `1f1b`, 8 microbatches, the loss by chunks
(2026-10-24):

| mesh | `methods.stages` | tok/s | peak |
|---|---|---|---|
| `stage=2, fsdp=4` | `[11, 5]` | 641k | 32.5 GiB |
| | `[10, 6]`, as balanced | 682k | 30.8 GiB |
| | `[9, 7]` | 682k | 29.7 GiB |
| `stage=4, fsdp=2` | `[5, 5, 4, 2]` | 540k | 52.6 GiB |
| | `[4, 5, 5, 2]` | 548k | 52.6 GiB |
| | `[5, 5, 5, 1]`, as balanced | 601k | 52.7 GiB |

The balanced split is as fast as the best hand-tuned one on both (on 2 stages tied
with `[9, 7]`). Its FLOPs count attention's: without them the head would be 5.6
passes' work, and 2 stages would take `[11, 5]`, 6% slower.

### The defaults against gpipe

The defaults against `gpipe` and against `1f1b` on even stages, by the one-node
suite's pipeline (`examples/benchmarks/one_node.bux`, 8 microbatches, 2026-10-24):

| mesh | `gpipe` | `1f1b`, even | `1f1b`, balanced: the default |
|---|---|---|---|
| `stage=2, fsdp=4` | 629k, 60.0 GiB | 652k, 22.2 GiB | `[10, 6]`: 682k, 30.8 GiB |
| `stage=4, fsdp=2` | 491k, 65.0 GiB | 494k, 40.8 GiB | `[5, 5, 5, 1]`: 598k, 52.7 GiB |
| `stage=8` | 320k, 83.7 GiB | 333k, 50.1 GiB | `[3, 2, 2, 2, 2, 2, 2, 1]`: 320k, 71.1 GiB |

On 8 stages, 8 microbatches barely fill the pipeline, whose stages idle 7 ticks of 15:
the slowest stage is in fewer ticks than the stages around it, and the even split is
4% faster there. With more microbatches the balanced split pays again:

- at 16, 364k against even stages' 358k and gpipe's 346k;
- at 32, 396k against 373k and 356k.

### Across nodes, one stage per node

The four-node suite's pipelines (`examples/benchmarks/four_nodes.bux`),
`stage=4, fsdp=8` on 32 GPUs in 16 microbatches of 2 rows per GPU, against no
pipeline on the same GPUs (2026-10-24):

| `methods` | tok/s | per GPU | peak |
|---|---|---|---|
| `pipeline="gpipe"`, even | 1,865k | 58.3k | 59.0 GiB |
| `1f1b`, `stages=[4, 4, 4, 4]` | 1,962k | 61.3k | 21.1 GiB |
| `1f1b`, `stages=[5, 4, 4, 3]` | 2,003k | 62.6k | 29.4 GiB |
| `1f1b`, balanced, `[5, 5, 5, 1]`: the default | 2,438k | 76.2k | 28.4 GiB |
| no pipeline, `fsdp=32` | 4,235k | 132.3k | 37.6 GiB |

- The balanced split is 24% faster than even stages, as on one node (21%).
- A GPU of the pipeline trains at 76k tokens/s on four nodes and on eight, as on one
  node's `stage=4, fsdp=2` (75k): the sends between the nodes cost nothing that shows
  ([Four nodes and eight](multi-node.md#four-nodes-and-eight)).
- What a pipeline loses against FSDP, 42% here and 47% on one node, is in its bubble,
  3 ticks of 19 idle, and in its microbatches' GEMMs, of 2 rows per GPU.

::::{dropdown} On 8 CPU devices
For a Llama of 8 blocks (`d_model=64`, 16 rows of 16 tokens), 2026-10-21: how far the
losses of three AdamW steps are from one device's at most, and the collectives in the compiled
gradient of a forward, those in the loop counted once, so once per tick:

| mesh | methods | losses | collective-permute | all-reduce | all-gather |
|---|---|---|---|---|---|
| `stage=2` | `microbatches=4` | 7.2e-6 | 2 | 2 | 0 |
| `stage=4` | `microbatches=8` | 1.8e-5 | 2 | 2 | 0 |
| `stage=4` | `microbatches=8, pipeline=circular, rounds=2` | 1.8e-5 | 2 | 2 | 0 |
| `stage=2, data=2, fsdp=2` | `microbatches=4` | 2.9e-6 | 3 | 1 | 21 |
| `data=2, fsdp=4` | | 6.2e-6 | 0 | 0 | 22 |

One ppermute per tick each way, and an all-reduce of the stream as it leaves, and of
its gradient as it comes back; no parameter of a stage leaves it.
::::

## Design notes

::::{dropdown} Why not MaxText's vmap over the stages
This is not MaxText's vmap over the stages with `spmd_axis_name="stage"`: on JAX 0.12
such a vmap cannot batch a shard_map manual over some of the axes, which the MoE's and
context parallelism's methods are, inside the layers.

Nested in a shard_map over `stage` they run, on three conditions the pipeline keeps:

- the outer one does not check its values' varying axes (`check_vma=False`), else the
  gradient of a nested shard_map fails to lower;
- `stage` comes first in the mesh, because XLA aborts on a nested shard_map manual
  over an axis before the outer one's;
- the pipeline takes a round's layers by dynamic slices, because XLA fails to
  partition over Auto axes the scatter that a gather's gradient is.
::::

## Not yet: DualPipe

DualPipe (DeepSeek-V3) and its half, DualPipeV, send microbatches through the stages
both ways, each device holding two chunks of the layers, and run one microbatch's
forward and another's backward in one op. An MoE's all-to-all in one then hides behind
the other's compute, which `1f1b` and `zbh1` do not do. It fits the executor of
`1f1b` and `zbh1`: a V-shaped table of ops on two chunks (device s holding chunks s
and 2S − 1 − s), the layers stacked by chunk as `circular` stacks rounds, four
ppermutes per tick, and an op `FB` that runs a forward and a backward in one branch.
The embedding and the head would then both sit on device 0, at the two ends of the V.
