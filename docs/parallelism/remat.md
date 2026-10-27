(parallelism-remat)=
# Repeats and remat

Two settings of {bux}`methods` trade memory, speed and compile time:

- **`methods.repeat`** says how `f^n` builds its `n` passes: one module per pass, run
  in a row (`unroll`, the default), or one stacked module that `jax.lax.scan` runs
  (`scan`). A scan compiles one pass for all of them, and a pipeline needs one.
- **`methods.remat`** says what the forward pass keeps for the backward pass, which
  computes the rest again (rematerialization, or activation checkpointing). The
  default, `auto`, keeps everything when the step fits in the device's memory, and
  otherwise picks the cheapest preset that fits.

Reach for them when a step runs out of memory, when you want more rows per GPU, or
when a deep model compiles slowly.

## Quick start

From `examples/experiments/parallel/fsdp_8b.bux`, Llama 3 8B on a node of 8 GPUs, its
32 blocks scanned and rematerialized:

```bux
use llama3_8b
use fineweb_wsd

mesh      = mesh(fsdp=8)
precision = precision(compute=bfloat16)
methods   = methods(repeat=scan, remat=dots, microbatches=4)
```

A policy can also be written as data. This one keeps attention's kernel inputs and
outputs and the MLPs' up and gate projections, in the first 8 passes of each repeat
only:

```bux
methods = methods(remat=remat(save=[attention, mlp], first=8))
```

## Options

| Setting | Values | Default | What it does |
|---|---|---|---|
| `methods.repeat` | `unroll`, `scan`, or your own | `unroll` | how `f^n` builds and runs its passes |
| `methods.remat` | a preset's name, {bux}`remat` data, or your own | `auto` | what the forward keeps for the backward |

A {bux}`remat` policy as data has three fields:

| Field | Default | What it is |
|---|---|---|
| `save` | `[]` | the names it keeps on the device |
| `offload` | `[]` | the names it moves to the host's memory until the backward reads them |
| `first` | none | how many passes of each repeat it rematerializes, from the first (of each stage's share, when pipelined); the rest keep everything. None: every pass |

`save` and `offload` take the names that layers give what they make:

| Name | What it is |
|---|---|
| `attention.in` | softmax attention's q, k and v, as its kernel takes them (after RoPE) |
| `attention.out` | its output, and whatever else its backward takes: the log-sum-exp |
| `attention.index` | the keys sparse attention's indexer keeps for each query |
| `recurrence.in` | a recurrent mixer's inputs, as its kernel takes them: linear attention, the SSMs, RWKV-7 |
| `recurrence.out` | its output, and whatever else its backward takes: the chunks' states, their inverses with `methods(states=recompute)`, nothing more with cuDNN's kernels |
| `qkv` | attention's q, k and v projections; MLA's latents |
| `mlp` | an MLP's up and gate projections |
| `experts` | the MoE's grouped products |
| `dots` | every product of matrices, MXFP8's and the experts' too |
| `dots_no_batch` | those without batch dimensions |

A name covers those under it, so `attention` is all three of attention's names. These
name what layers make, not definitions: `save=[mlp]` is not a layer type, and
`save=[attention]` is a list of one name.

A preset is a policy by name (`bux.architecture.repeats.remat.PRESETS`):

| Preset | What the forward keeps | As data |
|---|---|---|
| `auto` | the default: the first of `none`, `dots_attention`, `dots`, `save_attention` and `full` whose training step fits ([below](#rematauto-the-default)) | |
| `none` | everything, except in a scanned pass ([below](#how-a-scan-runs)) | |
| `full` | nothing but the input of each branch or pass | `remat()` |
| `dots` | the products of matrices | `remat(save=dots)` |
| `dots_no_batch` | the products of matrices without batch dimensions | `remat(save=dots_no_batch)` |
| `save_attention` | the mixers' kernels' inputs and outputs | `remat(save=[attention, recurrence])` |
| `save_qkv_attention` | those, and attention's projections | `remat(save=[attention, recurrence, qkv])` |
| `dots_attention` | `dots_no_batch`'s, the mixers' and the experts' | `remat(save=[dots_no_batch, attention, recurrence, experts])` |
| `offload_attention` | the mixers', in the host's memory | `remat(offload=[attention, recurrence])` |

:::{tip}
Leave `remat` at `auto` unless you want a specific trade. `none` is the fastest
wherever it fits, and `auto` picks it there; a policy buys rows per GPU
([Measurements](#policies-on-a-b300)).
:::

## Constraints and errors

- **A repeat stays unrolled under `scan`** where its passes do not stack:
  - their graphdefs or the shapes of their variables differ (Engram's tables);
  - they share a module or a variable: a `shared` definition, or nextn's embedding;
  - they widen the stream as they go, as attnres does.
- **A checkpoint resumes only with the `repeat` it was saved with.** A scanned
  repeat's state has a different structure from the unrolled one's, but the run's id
  is the same, because `methods` is not in it. With the other `repeat`, the restore
  fails at once and says so.
- **`none` is the null value in a `.bux` file**, so a sweep writes the method as a
  string:

  ```bux
  methods.remat = "none" | full
  ```

- **`auto` does not tune.** `remat(first=n)` and policies of names are not among its
  candidates: it keeps a run from running out of memory at the least cost among the
  presets.

## Repeats: `methods.repeat`

`f^n` builds `n` passes of `f`.

- **`unroll`**, the default: each pass is a separate module, and they run in a row.
- **`scan`**: the passes are built the same way, so every init that depends on depth
  (hc's, vpgate's, MiniMax's) is as unrolled. Then they are stacked into one
  `Scanned` module: each variable of the first pass holds every pass's values, along a
  new leading dimension named `layers`, and `jax.lax.scan` runs them.

A scan carries the stream and whatever taps a pass replaces, such as single-pass
mhc's read. It collects the losses each pass reports (the MoE's balance loss) and the
state it changes (the router bias).

Before it scans, a trial pass, traced but not run, checks that the shapes of the
stream and of the taps come out as they went in:

- If they do not after the first pass, the first runs alone and the scan starts at
  the second: single-pass mhc's first pass starts its read.
- If they never do, as when every pass pushes to a tap (a U-net's `(block @skip)^6`),
  the passes run one by one, from the same stacked state.

The `layers` dimension is a name of the layout, `layers` (none by default), which
FSDP never takes: a scan reads one layer at a time ([Names](layout.md#names)). Weight
decay and Muon count dimensions from after it, so a stacked norm scale is still a
vector, and a stacked kernel a batch of matrices.

With more than one `stage`, the longest scanned repeat is the one a pipeline splits
([Pipelines](pipelines.md)).

## Remat: `methods.remat`

`remat` recomputes in the backward pass what the forward does not keep. A policy keeps
what layers name (`jax.ad_checkpoint.checkpoint_name`), as MaxText's and Megatron's
selective recomputation do, and the backward computes the rest again.

What rematerializes as a unit:

- Each residual branch outside any other is rematerialized as a whole, with the read
  and write of the residual algebra.
- A scanned pass is rematerialized as one unit.
- Nothing inside either is rematerialized again.
- With `first=n`, only the first n passes of each repeat are. Unrolled, the rest keep
  everything; scanned, they run in a separate scan, as under `none`.

### Kernels and their names

What a kernel's backward takes is not a product of matrices. Flash attention's
backward takes its output and log-sum-exp, so under `dots` the backward runs the
kernel's forward again.

`remat.named(name, f, *args)` names f's inputs `name.in`, and its output and the rest
of what its backward takes `name.out`, so a policy that keeps the name runs none of f
twice. Attention's methods, sparse attention and the recurrent mixers run through it.

- Sparse attention's indexer names the keys it keeps, so its backward neither picks
  them again nor runs the flash forward again.
- MLA names its latents, not the heads' projections up from them. As in DeepSeek's
  and Megatron's recomputation (`mla_up_proj`), a policy that keeps `qkv` projects
  them up again.
- `dots` keeps MXFP8's GEMMs too, also inside a `shard_map`, where jax's own policy
  misses them.

:::{note}
jax's own `save_only_these_names` matches names whole, so it takes `attention.out`,
not `attention`.
:::

(parallelism-remat-auto)=
## `remat=auto`, the default

`auto` chooses a training step's policy when the step first compiles
(`bux.train.fitting`):

1. It compiles the step under `none`, `dots_attention`, `dots`, `save_attention` and
   `full` (`remat.CANDIDATES`), fastest first, as they train in the
   [measurements](#policies-on-a-b300).
2. It keeps the first that fits. A step fits when what XLA's analysis of the compiled
   step says a device holds at once (its arguments, outputs, temporaries and code) is
   at most 97% of what the device can hold. That limit is its `bytes_limit`: 95% of a
   GPU, which bux gives XLA's pool
   ([Memory](../performance/tuning.md#batch-microbatches-and-memory)).
3. Where none fits, it takes the last, `full`. Where the device does not say what it
   can hold, as a CPU does not, everything fits, and `auto` is `none`.

In that order, the first that fits is the one that computes least again, so `auto`
times none of them, which would cost steps of each. On B300s, XLA's analysis matched
the measured peak to 0.1 GiB in every run, and a step of 196.7 of 200.8 GiB trained.

How it compiles:

- Where `none` fits, `auto` compiles once and is `none`.
- Where it does not, the other four compile side by side. Each is traced and lowered
  in turn, which holds Python's lock, then XLA compiles them at once, a thread each.
- The candidates compile all or none, so every process of a run compiles the same
  programs, whose autotuning they share.
- JAX's cache keeps each for the run's next launch.
- A forward outside a training step keeps everything.

Compiling side by side costs the losers' tracing and lowering, about a minute each on
a large model. On 8 B300s, `megatron.bux`'s `nccl_ep` arm with 75% of a GPU took
`save_attention` in 699 s, where compiling one candidate after another took 966 s.
XLA compiled each of the four in 218-243 s, side by side
([where a compile goes](../performance/tuning.md#compile-time-and-the-compile-cache)).
Where the second candidate fits, side by side is slower: `sorted` with 95% took
`dots_attention` in 592 s, where `none` and then it would take about 400 s. Name a
policy to compile only one.

The step says what it took, with XLA's seconds for each candidate:

```text
remat=auto: save_attention in 699s (none 265.8 GiB in 144s, dots_attention 273.5 GiB
in 243s, dots 249.1 GiB in 230s, save_attention 108.4 GiB in 218s, full 74.2 GiB in
219s; 97% of 200.8 GiB fits)
```

## Offloading

`offload` moves what it names to the host's memory until the backward reads it.

- **Unrolled, it saves nothing and costs a third of the speed.** XLA schedules each
  layer's copy to the host inside the backward, just before the copy back, so the
  device keeps the value from the forward on, as if saved, and the round trip adds a
  buffer. The compiled gradient's temp memory is 17.7 GiB against `save_attention`'s
  13.4, and 1.0 GiB goes to the host. The latency-hiding scheduler compiles the same
  program.
- **Scanned, it works as MaxText's does.** The scan stacks each pass's values into
  buffers in the host's memory, 6.0 GiB of them, and the device peaks at 22.3 GiB
  against `save_attention`'s 30.1, at 84k tokens per second against 121k (90k with
  `--xla_gpu_enable_pipelined_host_offloading=true`).

`full`, unrolled, peaks at 24.5 GiB at 120k: on a B300, computing the values again is
cheaper than the copies.

## How a scan runs

**What a scanned pass keeps.** A scan keeps every value its backward reads, stacked by
pass. Unrolled, XLA fuses the elementwise ones into the ops that read them and keeps
none: a norm's float32 copy of the stream, an MLP's activation, a token's copy for
each of its experts. So under `none`, a scanned pass keeps everything except what
elementwise ops and ops that only move or retype data make (`remat.CHEAP`), and its
backward computes those again (`remat.unfused`).

**The cast of the parameters.** A training step does not cast the stacked parameters
to its compute dtype, which would keep a copy of them all, and of their gradients.
Each pass casts its own as it runs (`Scanned.compute`).

**Running a stack.** `Scanned.layers` holds the stacked passes, and
`stacked.run(graphdef, state, times, x, taps, compute)` runs any stack of them, such
as a pipeline stage's share.

**Runs of passes that trace alike.** A scan traces one pass for all it runs, so a
repeat first cuts its passes into runs that trace alike (`tracing.runs`):

- where precision's first and last branches start or stop (`matmuls.cuts`);
- where `remat(first=n)` stops (`remat.cuts`);
- in a pipeline's stage, where some stage skips them.

Each run traces inside `tracing.passes(run, each)`, which tells remat the pass
(`tracing.current_pass`) and precision the branch (`tracing.current_branch`). A new
reason for passes to trace differently is another cut.

## Writing your own policy

A new policy is a function, registered under a name for `methods.remat`:

```python
import jax

from bux.architecture.repeats import remat
from bux.parallel import methods

methods.remat.register("save_mlp")(remat.keeping(save=("attention", "mlp")))
methods.remat.register("offload_dots")(
    jax.checkpoint_policies.offload_dot_with_no_batch_dims("device", "pinned_host")
)
```

`methods = methods(remat=save_mlp)` then picks it.

## Measurements

### Scanned and unrolled on a B300

2026-10-23. A 1B dense model (`A^16`), a Gated DeltaNet hybrid (`"DDDA"^4`) and a
64-expert MoE (`E^16`), in bfloat16 on 8 rows of 4096 tokens: the temp memory of the
compiled train step, and tokens per second.

| Model | Memory: unrolled | Memory: scanned | tok/s: unrolled | tok/s: scanned |
|---|---|---|---|---|
| dense | 56.2 GiB | 47.3 GiB | 145k | 138k |
| GDN | 124.9 GiB | 86.7 GiB | 94k | 78k |
| MoE | 95.8 GiB | 80.0 GiB | 106k | 91k |
| dense, `fsdp=8` | | | 592k | 670k |

- **On one GPU, a scan is slower.** It copies each value it keeps into its stack and
  back out, where unrolled layers read it where it was made. The more it keeps, the
  more it copies: the GDN hybrid's float32 chunk states.
- **On eight GPUs at one row each, it is faster**, and it compiles 2.3 times as fast.
- With `compiler(preset=maxtext)`, the dense scan trains 56% faster than unrolled
  there, but from two rows per GPU it is 3-4% slower, so `unroll` stays the default
  ([XLA's options](../performance/compiler.md)).

### Compile time and memory on one CPU

The gradient of a 24-layer Llama's forward (`d_model=64`, 8 rows of 128 tokens), and
its first training step, which compiles it:

| Repeat | Trace + compile | First step, AdamW | First step, Muon |
|---|---|---|---|
| `unroll` | 1.2 s | 2.9 s | 6.9 s |
| `scan` | 0.3 s | 0.8 s | 0.9 s |

The temp memory of the same gradient by policy:

| Remat | Temp memory, unrolled | Scanned |
|---|---|---|
| `none` | 327 MiB | 142 MiB |
| `full` | 22 MiB | 23 MiB |
| `dots` | 132 MiB | 138 MiB |
| `dots_no_batch` | 79 MiB | 82 MiB |
| `save_attention` | 142 MiB | 142 MiB |
| `save_qkv_attention` | 142 MiB | 142 MiB |
| `dots_attention` | 187 MiB | 193 MiB |
| `offload_attention` | 142 MiB | 142 MiB |

On a CPU, attention runs by `xla`, whose backward takes the t by t probabilities, so
what `save_attention` keeps of it is more than `dots` keeps of the whole layer; and a
CPU's host memory is its device memory. On a GPU, flash attention's backward takes
only q, k, v, the output and the log-sum-exp.

### Policies on a B300

2026-10-23. The dense 1.3B of 16 Llama blocks of width 2048 (vocabulary 128256), a
Gated DeltaNet hybrid (`"DDDA"^4`) and a 64-expert MoE (`E^16`, top 8), in bfloat16 on
8 rows of 4096 tokens, the loss by chunks. Tokens per second and the peak memory:

| Remat | Dense | GDN | MoE |
|---|---|---|---|
| `none` | 139k, 51.1 GiB | 89.1k, 120.9 GiB | 101k, 134.0 GiB |
| `dots_attention` | 137k, 41.6 | 76.9k, 106.9 | 97.2k, 108.4 |
| `remat(save=[attention, mlp])` | 135k, 41.6 | | |
| `remat(save=[attention, recurrence.in, mlp])` | | 74.9k, 41.0 | |
| `remat(save=[attention, experts])` | | | 95.9k, 108.3 |
| `dots` | 128k, 39.5 | 74.0k, 75.7 | 93.7k, 106.4 |
| `save_attention` | 126k, 30.6 | 73.4k, 89.9 | 87.7k, 81.7 |
| `remat(save=[attention, recurrence.in])` | | 71.9k, 31.4 | |
| `full` | 120k, 24.5 | 72.7k, 30.3 | 84.2k, 75.8 |
| `remat(first=8)`, `first=2` for GDN: half the passes | 127k, 37.8 | 79.3k, 73.6 | 90.1k, 104.6 |
| `remat(save=[attention], first=8)` | 133k, 40.9 | | |
| `offload_attention` | 82.1k, 39.5 | | |

What the table says:

- `none` is the fastest; a policy buys rows, and `auto` takes the first policy that
  fits, in the order of [`remat=auto`](#rematauto-the-default): the one that
  recomputes least.
- `dots` runs flash attention's forward twice, and `save_attention` the matmuls of
  every layer. `dots_attention` runs neither: on the dense model it is 2% slower than
  `none` in 19% less memory, as is `remat(save=[attention, mlp])`.
- A GDN layer's recurrence by chunks keeps their float32 states (`recurrence.out`),
  most of what `save_attention` keeps there. Keeping only its inputs
  (`recurrence.in`, whose backward runs the chunked forward again) with the MLPs'
  projections, `remat(save=[attention, recurrence.in, mlp])`, is faster than
  `save_attention` (74.9k against 73.4k) in less than half its memory (41.0 GiB
  against 89.9).
- In a Mamba-2 hybrid (`"MMMA"^4`) too, `recurrence.in` in place of `recurrence`
  costs 2-3% for 59-68% less memory (`save_attention` 78.5k in 99.6 GiB, against
  76.4k in 31.7). In a KDA hybrid by cuDNN's kernel, which costs more to run again,
  it costs 7-11%.
- `first` gives the points between. Half the passes in full is as fast as `dots` in a
  little less memory, and slower than the policy of names it would match: those keep
  a layer's costly parts, where `first` computes the whole layer again.

:::{note}
Since these were measured, GDN runs by cuDNN's kernel, which keeps only its inputs
([Context parallelism for recurrent mixers](recurrence.md)).
:::

### More rows per GPU on a B300

2026-10-23. The same models with more rows per GPU, tokens per second and the peak
memory:

| Remat | Dense | GDN | MoE |
|---|---|---|---|
| `none` | 56 rows: 146k, 191.2 GiB | 12 rows: 91.0k, 166.2 GiB | 10 rows: 105k, 149.8 GiB |
| `dots_attention` | 48: 147k, 151.6; 64: 145k, 196.7 | | 12: 102k, 127.5 |
| `remat(save=[attention, recurrence.in, mlp])` | | 32: 79.7k, 110.3 | |
| `save_attention` | 96: 138k, 161.7 | | 16: 93.8k, 95.8 |
| `full` | 128: 126k, 113.2 | 32: 76.5k, 67.3 | 16: 89.6k, 83.8 |

### `auto` on one B300

2026-10-24. The one-GPU suite's models (`examples/benchmarks/one_gpu.bux`) at more
rows than they take there, 15 steps. Tokens per second and the peak memory:

| Model, rows | `none` | `auto` | `auto`'s compiles |
|---|---|---|---|
| dense, 8 | 138.7k, 51.1 GiB | `none` | 27 s |
| dense, 64 | 142.9k, 192.6 GiB | `none` (`dots`: 137.1k, 180.2 GiB) | 35 s |
| dense, 96 | out of memory | `save_attention`: 132.5k, 161.7 GiB (`dots_attention`, `dots`: out of memory) | 4, 121 s |
| dense, 128 | out of memory | `full`: 123.6k, 113.2 GiB (`save_attention`, `dots`: out of memory) | 5, 148 s |
| MoE, 16 | 106.7k, 193.3 GiB | `none` (`dots_attention`: 102.3k, 146.6 GiB) | 55 s |
| GDN, 32 | 131.3k, 187.9 GiB | `none` (`dots_attention`: 130.3k, 149.9 GiB) | 40 s |

`none` stays the fastest wherever it fits, even near the device's limit. With only
90% of a device for the step, `auto` took `dots` for the dense model at 64 rows, 5.6%
slower than `none` there, and `dots_attention` for the MoE at 16 rows and GDN at 32.
