(parallelism-microbatches)=
# Microbatches

`methods.microbatches = n` takes each step's batch in n microbatches: gradient
accumulation. The step runs the microbatches one after another, adds up their
gradients in float32 and makes one update, so it holds the activations of one
microbatch instead of the whole batch's. Use it when a step's batch does not fit on
your devices, or when a pipeline needs microbatches to keep its stages busy
([Pipelines](pipelines.md)).

Microbatches change how a run computes, not what, so like the rest of {bux}`methods`
they are not in the run's id. Three more methods shape the step around them: `gather`
and `accumulate` say how often the parameters are gathered and the gradients reduced,
and `interleave` runs one microbatch's forward beside the backward of the one before.

## Quick start

From `examples/experiments/parallel/fsdp_8b.bux`, Llama 3 8B over a node of 8 GPUs,
each step's 256 rows taken 64 at a time:

```bux
use llama3_8b
use fineweb_wsd

mesh      = mesh(fsdp=8)
precision = precision(compute=bfloat16)
methods   = methods(repeat=scan, remat=dots, microbatches=4)
```

How the step gathers and reduces around its microbatches, one line per choice (a file
sets `methods` once):

```bux
methods = methods(microbatches=8)                                    # reduced once across hosts
methods = methods(microbatches=8, accumulate=unreduced)              # reduced once
methods = methods(microbatches=8, accumulate=reduced)                # reduced each microbatch
methods = methods(microbatches=8, gather=layer, accumulate=reduced)  # ZeRO-3
methods = methods(microbatches=8, interleave=true)                   # side by side
```

## Options

| Setting | Values | Default | What it does |
|---|---|---|---|
| `methods.microbatches` | a count | `1` | the microbatches each step's batch splits into |
| `methods.gather` | `step`, `layer` | `step` | `step`: cast and gather the parameters FSDP splits once per step, before the microbatches. `layer`: gather as each layer runs (ZeRO-3), holding one layer's parameters at a time |
| `methods.accumulate` | `auto`, `unreduced`, `reduced` | `auto` | `unreduced`: add up each device's own gradients and reduce them once per step. `reduced`: reduce each microbatch's gradients as the backward makes them. `auto`: unreduced where the rows' axes span more than one process and no layer learns from the whole batch |
| `methods.interleave` | `true`, `false` | `false` | each microbatch's forward beside the backward of the one before, a piece of the model at a time |
| `methods.donate` | `true`, `false` | `true` | a step's new state takes the old state's memory |

`precision.reduce` (`compute` or `float32`) sets the dtype the gradients are reduced
over the rows in; see [Gradient reductions](precision.md#gradient-reductions-precisionreduce).

## Constraints and errors

All of these are errors before anything trains:

- `batch` must split into `microbatches`, and each microbatch's rows over the batch
  axes (`layout.batch`). `parallel.check` says so before anything is built, as
  `batch = <b> does not split into methods.microbatches = <n>`, or as
  `batch = <b> does not split over layout.batch, ...: make it a multiple of ...`.
- `accumulate=unreduced`, or `precision(reduce=float32)`, where adding up unreduced
  would not compute the same. The error says why:
  - `gather=layer`: it needs the parameters gathered once per step, `gather=step`;
  - a layer that learns from the whole batch, as the MoE's router does: use
    `accumulate=reduced`;
  - `precision(reduce=float32)` under a `gpipe` or `circular` pipeline, whose
    embedding and head reduce in the backward.
- `interleave=true` with a pipeline (`mesh.stage` above 1), which schedules its own
  microbatches: keep `mesh.stage = 1`.
- `interleave=true` on a model whose items pass each other more than the stream, as a
  tap or an MTP head does, or that does not end in a single head.

:::{note}
Where no axis of more than one device splits the rows, nothing reduces over them,
and `accumulate` has nothing to choose.
:::

## What the step computes

The microbatches are a `lax.scan` inside the train step. Microbatch i is rows i,
i + n, i + 2n and so on (`parallel.microbatches`), so every device's rows split evenly
among the microbatches and none moves. Pipeline parallelism splits its batch the same
way, through the same function.

The step computes the whole batch's results:

- **The objective's terms**, means over targets, are weighted by each microbatch's
  share of the batch's targets (`trainer.token_shares`). So the loss and the z-loss
  are the batch's, even when a loss mask leaves microbatches with unequal numbers of
  targets. `mtp`'s heads are weighted by the first head's targets.
- **What layers report** is weighted by each microbatch's share of rows, 1 / n: a
  balance loss over sequences (`aux_loss_scope=sequence`) and the indexer's loss are
  the batch's. A balance loss over the batch is the mean of the microbatches' own,
  each over its rows: its f_e and P_e are a microbatch's means, not the batch's.
- **The router bias** moves once per step, from the whole batch's picks. The trainer
  sets `update_at_step_end` on every module that has it. The MoE's training forwards
  then route with the bias where the step found it and add their picks to
  `router_load`, and `step_end_update` moves the bias after the last microbatch. A
  layer of your own that learns from the batch outside of gradients does the same
  (`trainer.StepEndUpdate`). Called outside a step, a training forward moves the bias
  at once, and a shared MoE moves once per step, from the picks of all its uses.
- **Dropout** draws each microbatch's masks in turn from the run's RNG stream: the
  same distribution as the whole batch's, not the same masks, and the stream's count
  advances n times per step. A resumed run draws the same masks.

`tests/test_precision.py` checks the gradients, the metrics and the router bias
against the whole batch's, on one device and on `fsdp=8` and
`fsdp=2, expert=2, tensor=2`. Apart from dropout's masks and a balance loss over the
batch, microbatches change how a run computes, not what, so they keep its id.

(microbatches-gather-accumulate)=
## Gathered once, reduced once: `methods.gather`, `methods.accumulate`

Every axis is Auto, and XLA does not move a collective out of a loop. Left to itself,
a step of n microbatches under FSDP gathers each layer's parameters as it runs: n
times forward and, in a scan of layers, n times backward. It also reduce-scatters the
gradients n times. That is what ZeRO-3 does, and what
`methods(gather=layer, accumulate=reduced)` still does. `bux.parallel.gather` does
better, in two parts.

### Gathered once: `gather=step`, the default

Before the microbatches, each device casts its shards of the parameters to `compute`
and gathers them (`gather.whole`: split by their names alone, whole over FSDP's axes),
and every microbatch uses that copy. A pipeline's step does the same before its ticks,
in float32, because its ticks add up the gradients in the dtype of what they read.

### Reduced once: `accumulate=unreduced`

The gradients add up unreduced, each device's own, and reduce once at the end:

1. JAX types an array that each device holds part of the sum of,
   `P(unreduced={...})`, only on Explicit or Manual axes. So the step runs in a
   `shard_map` manual over the axes that split the rows (`gather.rows`: those of
   `layout.batch`, `data`, `fsdp` and `expert` by default, that have more than one
   device).
2. Each device gathers its shards by `all_gather`, and sums the gradients of its rows'
   microbatches into a float32 copy of every parameter.
3. A `psum_scatter` keeps its shards of their sum (`gather.scatter`), in
   `precision.reduce`.

Each device folds its index into the RNG keys, so dropout draws masks for its own
rows. The other axes stay Auto inside, and their reductions (over `tensor`, `context`)
happen every microbatch.

### `accumulate=auto`, the default

Inside that `shard_map` a layer sees only its device's rows, so a layer that learns
from the whole batch would learn from a part (`trainer.rows_independent`): the MoE's
router, its bias and its balance loss over the batch. `accumulate=auto` adds up
unreduced where all of these hold:

- the step has microbatches, or a pipeline;
- the devices along the rows' axes span more than one process
  (`gather.rows_span_processes`), so their reductions go between hosts;
- its rows are independent;
- the parameters are gathered once.

Otherwise each microbatch's gradient reduces onto its shards as the backward makes it,
which is faster over NVLink
([below](#gathered-once-reduced-once-one-node-of-8-b300s)). `accumulate=unreduced`
where it would not compute the same is an error that says why.

### The cost: memory

Both parts cost memory: the gathered copy of the parameters in `compute`, and, when
unreduced, a float32 sum of every parameter's gradient on each device, held through
the step. `layout(zero=1)` keeps the parameters whole between steps too. With
`gather=step` it saves no collective, and holds float32 copies where the step holds
copies in `compute`.

What each part saves, on one node and through the NICs, is in the
[measurements](#gathered-once-reduced-once-collectives).

(microbatches-interleave)=
## Side by side: `methods.interleave`

```bux
methods = methods(microbatches=8, interleave=true)
```

When microbatches run in a row, each collective waits on the compute just before it.
Whatever the layer's data flow leaves nothing to run beside stays exposed: FSDP's
gathers, TP's reductions, EP's exchanges. With `interleave`, each microbatch's
forward runs beside the backward of the one before, a piece of the model at a time,
as Megatron's combined 1F1B does without a pipeline (`bux.train.interleave`):

```text
forward of microbatch 0
a lax.scan, for i in 0 .. n-2:
    tick t, for t in 0 .. r-1:  piece t's forward of i+1,  piece r-1-t's backward of i
    the head's loss of i+1, and its gradient
backward of microbatch n-1
```

:::{warning}
`interleave` is off by default. On 8 B300s it hides NCCL EP's exchange behind the
other microbatch's compute, but elsewhere its copies cost more than it hides. A model
that fills a GPU in a row does not fit interleaved: it holds half a microbatch more
({ref}`Measurements <measurements-interleaved-on-a-node>`).
:::

### How it works

- **The pieces** are the items of the model's graph in a row, with each pass of an
  unrolled repeat a separate piece (`embed`, 48 blocks, `norm`, and the head, whose
  forward ends in the loss). A scanned repeat is one piece.
- **A piece's forward is a `jax.vjp`**. Where the run's `remat` keeps everything
  (`none`), it keeps what the backward reads except what elementwise ops make, as a
  scanned pass does (`remat.unfused`). In a step in a row XLA fuses those ops into the
  backward ops that read them, but the loop carries whatever a piece's vjp keeps.
  Under any other policy, each branch inside is rematerialized by that policy. The
  scan carries each piece's residuals, as separate leaves, to its backward in the next
  iteration.
- **Each tick ends in an optimization barrier** on the stream, its gradient and the
  gradients of the tick before. So XLA's scheduler puts one half's collectives beside
  the other half's compute, reduces a tick's gradients beside the next, and runs no
  tick early. XLA:GPU keeps the barriers until it has scheduled the step.
- **It computes what the loop computes**: the gradients, the metrics, the router's
  bias and the RNG streams. `tests/test_interleave.py` checks this on FSDP, EP and TP
  layouts, to float32's rounding (XLA:CPU rounds a microbatch inside a loop to other
  bits than outside one).
- **NCCL EP in any order.** Each of NCCL EP's backward calls prepares its own route
  from the picks the forward saved, so two microbatches' calls run in any order, and
  every run below trains to the loop's losses
  ([TransformerEngine's NCCL EP](experts.md#transformerengines-nccl-ep)).
- **Its program** is about twice the loop's, with a forward and a backward on each
  side of the loop, and compiles in about twice the time.

### What the carry costs

A carried leaf is written where it was read:

- A top piece's residuals are read by the previous microbatch's backward in a tick
  before its forward makes the new ones, so XLA writes them in place.
- A bottom piece's new residuals are made before the old ones are read, so XLA keeps
  them aside and copies them in at the loop's end. That is a copy of half a
  microbatch's residuals, and the peak holds them beside the other half: 25 GiB more
  than the loop on a 12-layer cut of `megatron.bux`.

(microbatches-donate)=
## Donation: `methods.donate`

A step donates the model's and the optimizer's arrays, so its new state takes the old
state's memory where XLA can write it in place. `methods(donate=false)` keeps them.

```bux
methods = methods(donate=false)
```

More fits donated. The 1.3B trains 36 rows per GPU and the MoE 12, which run out of
memory without donation. The MoE with a checkpoint every 10 steps trains, where
without donation it runs out of memory at the step after its first save
([Donation on one B300](#donation-on-one-b300)).

:::{note}
Donation changes how XLA rounds a step a little (the gradient's norm by 7e-6 at the
dense model's first step on a GPU, the loss by 2e-6 at zamba2's third on a CPU): runs
compared bit for bit must donate alike. An array read from `trainer.model` before a
step is gone after it. A checkpoint is safe, because orbax copies the state to the
host before `save` returns.
:::

Either way, the trainer keeps the state in the buffers it was made in
(`parallel.shard_state` moves only what is elsewhere), and a resumed run loads its
checkpoint into the memory of the state it replaces
([Resuming](../training/launching.md#resuming)).

## Measurements

On B300s, except the memory on one CPU device.

### Gathered once, reduced once: collectives

2026-10-23. Unreduced, a step of 8 microbatches moves what a step of 1 does. The
counts are from the compiled step of `fsdp=8`, each collective counted as many times
as the loops around it run (`tests/test_gather.py`). The model is the benchmarks'
dense 1.3B (`examples/benchmarks/baseline.bux`), 64 rows of 4096 tokens. On one node
`accumulate=auto` reduces each microbatch (that row's loss is by chunks):

| methods | all-gathers | reduce-scatters |
|---|---|---|
| `microbatches=1` | 66, 2.68 GiB | 9, 2.68 GiB |
| `microbatches=8, gather=layer, accumulate=reduced` | 528, 21.5 GiB | 72, 21.5 GiB |
| `microbatches=8`, on one node | 66, 2.68 GiB | 72, 25.9 GiB |
| `microbatches=8, accumulate=unreduced` | 66, 2.68 GiB | 9, 2.68 GiB |
| `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 1048, 34.1 GiB | 152, 21.5 GiB |
| `repeat=scan, microbatches=8, accumulate=unreduced` | 10, 2.68 GiB | 7, 2.68 GiB |

### Gathered once, reduced once: one node of 8 B300s

2026-10-23. Tokens per second and the peak memory per GPU for the same run, 20 steps,
the loss by logits. (By chunks, the default, `microbatches=1` trains at 1122k in
38.9 GiB.)

| mesh, methods | tok/s | peak |
|---|---|---|
| `fsdp=8`, `microbatches=1` | 1147k | 58.5 GiB |
| `fsdp=8`, `microbatches=8, gather=layer, accumulate=reduced` | 817k | 11.9 GiB |
| `fsdp=8`, `microbatches=8` (reduced each microbatch) | 925k | 11.6 GiB |
| `fsdp=8`, `microbatches=8, accumulate=unreduced` | 907k | 15.7 GiB |
| `fsdp=8`, `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 719k | 10.2 GiB |
| `fsdp=8`, `repeat=scan, microbatches=8, accumulate=unreduced` | 803k | 15.7 GiB |
| `fsdp=2, tensor=4`, `layout(stream=tensor)`, `microbatches=8, gather=layer, accumulate=reduced` | 647k | 11.9 GiB |
| `fsdp=2, tensor=4`, `layout(stream=tensor)`, `microbatches=8, accumulate=unreduced` | 640k | 12.9 GiB |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 660k | 88.3 GiB |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8` (reduced each tick) | 640k | 97.8 GiB |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, accumulate=unreduced` | 644k | 97.8 GiB |

Over NVLink, gathering once is the gain. XLA overlaps each microbatch's
reduce-scatter with the backward, while adding up unreduced reads and writes a
float32 copy of every gradient each microbatch.

### Gathered once, reduced once: through the NICs

2026-10-23. Between nodes, the collectives are the cost. Here NCCL is kept off NVLink
and shared memory (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`, and XLA's one-shot
all-reduce off), so every collective goes through the NICs, as between nodes:

| mesh, methods | tok/s |
|---|---|
| `fsdp=8`, `microbatches=1` | 1030k |
| `fsdp=8`, `microbatches=1`, `precision(reduce=float32)` | 976k |
| `fsdp=8`, `microbatches=8, gather=layer, accumulate=reduced` | 358k |
| `fsdp=8`, `microbatches=8, accumulate=reduced` | 500k |
| `fsdp=8`, `microbatches=8, accumulate=unreduced` | 738k |
| `fsdp=8`, `microbatches=8`, `precision(reduce=float32)` | 699k |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 464k |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, accumulate=reduced` | 521k |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, accumulate=unreduced` | 566k |

So `gather=step` is the default, and `accumulate=auto` adds up unreduced only where
the rows' axes span processes, as between nodes. With deterministic GEMMs, the three
dense runs of 8 microbatches on one node end their 10 steps at 11.86792, 11.86793
(`reduced`) and 11.86804 (`unreduced`).

(measurements-interleaved-on-a-node)=
### Side by side: interleaved on a node

2026-10-24. Tokens per second per GPU and the peak memory, in a row and interleaved,
on a node of 8 B300s:

| run | in a row | interleaved |
|---|---|---|
| `megatron.bux`'s `nccl_ep`, `moe_stream=true`, 12 layers, 8 microbatches of 8 rows per GPU | 115.6k, 63.7 GiB | **120.3k**, 93.1 GiB |
| the same, NCCL EP on the compute stream | 115.7k, 64.5 GiB | 112.3k, 87.0 GiB |
| its `sorted`, 12 layers, 8 microbatches | 123.8k, 62.0 GiB | 118.0k, 87.5 GiB |
| `one_node.bux`'s `tp` (`repeat=unroll`, `remat=none`), 16 rows in 4 microbatches | 33.2k, 18.0 GiB | 33.5k, 19.3 GiB |
| the same, two collectives in flight | | 34.0k, 18.1 GiB |
| its `fsdp`, the dense 1.3B at one row per GPU per microbatch, 32 rows in 4 | 111.2k, 11.5 GiB | 93.1k, 12.6 GiB |

- **NCCL EP with `moe_stream`** is where interleave pays. NCCL EP's calls run on a
  separate stream, started as early as their inputs allow and waited for as late as
  their results are needed
  ([On a stream of its own](experts.md#on-a-stream-of-its-own-methodsmoe_stream)).
  The other microbatch's piece gives the scheduler compute to put beside them. At 4
  microbatches of the 12 layers, GPU 0's step takes 1111 ms against 1152 in a row:
  199 ms of its collectives run beside compute and 277 ms stay exposed, against 30
  and 360, though the bottom half's copies add 38 ms. In a row, `moe_stream` gains
  nothing, because no compute is left beside the exchange.
- **All 48 layers at Megatron's batch**, which the loop trains at 33.7-34.1k in
  234 GiB (`remat=none`), do not fit interleaved at `remat=none` (291 GiB) or at
  `dots_attention`. At `save_attention` they train at 30.7k in 193.4 GiB, against
  the loop's 29.5k in 203.8 GiB there: interleave gains, but not enough to pay for
  the rematerialization ([Remat](remat.md)).
- **Elsewhere on a node the overlap is real but small.** With the parameters gathered
  once per step (`gather=step`), the dense FSDP run has only each microbatch's
  reduce-scatters to hide, and the bottom half's copies cost more.

::::{dropdown} Details: where the overlap goes, and designs that did worse

On a node:

- TP's step on GPU 0 left 39 ms of its collectives exposed, from 47, and 35 ms
  unrolled, at 35.3k per GPU against the loop's 34.1k.
- The MoE's FSDP gathers took 50 ms less of a step, which slower reduce-scatters gave
  back.
- Tying the halves together one tick at a time beats XLA's own order. In slots, with
  the barrier on the slots and the gradients alone, so that each half's chain runs as
  XLA likes, TP trained at 31.2k in 24.9 GiB, against 32.2k tied, and the MoE at
  104.6k in 93.8 GiB, against 107.3k.
- XLA runs one collective at a time
  (`xla_gpu_experimental_parallel_collective_overlap_limit`). At 2, interleave gains,
  and the MoE's loop loses 4.5% ([XLA's options](../performance/compiler.md)).
- `sorted`'s 48-layer arm at Megatron's batch, which the loop trains at 35.8k in
  230 GiB, does not fit interleaved either: in slots it asked for 261 GiB, and the
  carry holds half a microbatch more.

Two other ways to carry the residuals did worse than the loop's carry:

- **Slots of one buffer**, which each forward takes from the backward before it (a
  residual put in by a dynamic-update-slice and taken out by a dynamic slice), hold
  one microbatch's residuals. But XLA copies every residual a kernel made into its
  slot, and out again where a kernel reads it: 160 ms of a 1050 ms step of 12 of
  those layers in 4 microbatches. cuBLAS also chose slower GEMMs for TP's slices.
  XLA's dynamic-slice fusion, with offsets from a table
  (`--xla_gpu_experimental_enable_dynamic_slice_table_offsets=true`), makes the same
  copies as memcpys.
- **No carry at all**, the ticks of every microbatch unrolled in the step, copies
  nothing. But the program is n times the loop's, and XLA held 176 GiB for the 12
  layers in 8 microbatches (124.6k tokens per second per GPU, against 126.7k for the
  loop).
::::

### Memory on one CPU device

2026-10-21. The compiled train step's temporaries on one device, for 4 Llama blocks of
width 256, `seq_len = 256`, `batch = 32` (`compiled.memory_analysis()`):

| `compute` | 1 microbatch | 4 microbatches |
|---|---|---|
| `float32` | 2032 MiB | 472 MiB |
| `bfloat16` | 1128 MiB | 225 MiB |

What bfloat16 does to the loss is in
[bfloat16 against float32 on a CPU](precision.md#bfloat16-against-float32-on-a-cpu),
and where float32 stays in [Float32 islands](precision.md#float32-islands).

### Donation on one B300

2026-10-23. On one B300 (bfloat16, 8 × 4096 tokens), the peak and the largest free
piece of the GPU's memory after 20 steps, in GiB, and tokens per second:

| model | kept: peak, free | donated: peak, free | tok/s, kept / donated |
|---|---|---|---|
| Llama-style 1.3B | 81.3, 172.1 | 70.6, 186.5 | 143.6k / 142.9k |
| MoE, 64 experts, top 8 (5.2B) | 163.9, 85.5 | 153.5, 143.1 | 103.8k / 106.2k |
| Gated DeltaNet hybrid | 148.4, 169.8 | 140.4, 185.3 | 92.1k / 91.9k |
| KDA, `gate_lower_bound=-5` | 183.4, 169.7 | 176.2, 185.3 | 110.9k / 114.4k |
