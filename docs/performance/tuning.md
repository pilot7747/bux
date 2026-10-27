# Tuning guide

Use this guide when a run trains slower than you expect or runs out of memory. It
starts with a workflow for measuring a change, then walks through the levers in the
order to try them: for each, the setting, when it helps, what it costs, and where it
is measured. It ends with a [troubleshooting table](#troubleshooting).

Every number here is from B300s, in October 2026, unless it says otherwise.

:::{tip}
bux's defaults are the fastest settings measured on the benchmarks' models: kernels on,
`remat=auto`, the loss by chunks, donation, parameters gathered once per step, and
gradient reductions started early. Start from them, and change one thing at a time.
:::

## The short version

1. **Fill the GPU.** Give each GPU as many rows as fit; under FSDP, at least two
   ([Batch](#batch-microbatches-and-memory)).
2. **Multiply in MXFP8** on Blackwell GPUs: `precision(matmuls=mxfp8)` trains a dense
   1.3B 1.18x as fast and an MoE 1.10x ([Precision](#precision-formats-and-recipes)).
3. **Split by FSDP within a node and HSDP across nodes**: `mesh(fsdp=8)`,
   `mesh(data=4, fsdp=8)` ([Choosing a mesh](#choosing-a-mesh)).
4. **Sort an MoE's experts under FSDP** (`methods(moe=sorted)`, the default), unless
   gathering its experts costs more than sending its tokens ([MoE methods](#moe-methods)).
5. **Profile** whatever is still slow ([Profiling](profiling.md)).

## The workflow

### 1. Measure

Every log line has `tokens_per_s`, the tokens per second since the last log. On a GPU
whose peak bux knows, it also has `mfu`, the model's FLOPs per second over that peak
([MFU and FLOPs](profiling.md#mfu-and-flops)). Read them after the first log, whose
time includes compiling.

`bux plan` shows, with no GPU, each run's FLOPs per token and the bytes of its
parameters and optimizer state on each GPU: what a step needs before its activations.

### 2. Profile

```bux
profile = profile(start=20, steps=5)
```

At step 25 the log then has `comm_ms`, `exposed_comm_ms` and `idle_ms`: the time a step
spent in collectives, in collectives with nothing computing beside them, and running
nothing at all. [Profiling](profiling.md#finding-what-a-step-waits-for) maps each to a
lever. For NVIDIA's own timeline or a kernel's counters, `tool=nsys` or `tool=ncu`
takes an Nsight report instead
([Nsight Systems and Nsight Compute](profiling.md#nsight-systems-and-nsight-compute)).

### 3. Change one thing

Write the change as a sweep, so the runs differ in that one setting:

```bux
methods.remat = keep: "none" | da: dots_attention | sa: save_attention
```

`bux launch` trains the three side by side, one per GPU, and each log has its speed.
Settings that change how a run computes but not what (`mesh`, `layout`, `methods`,
`compiler`) keep its id, so the three share one id and differ by name.

### 4. Compare on the same node

Nodes differ by a few percent. To be sure of a change, train it against the tree
before it on the same node, with `scripts/benchmark.py`:

```sh
python scripts/benchmark.py run my_suite.bux --tree ../before --out old.json
python scripts/benchmark.py run my_suite.bux --out new.json
python scripts/benchmark.py compare old.json new.json
```

A change within 3% is noise. [Benchmarks](benchmarks.md#comparing-two-trees) has the
details, the suites bux itself is measured by, and how to pin XLA's autotuning before
you compare losses.

## The levers

In the order to try them. The first cost nothing but a setting; the later ones trade
memory, compile time or what a run computes for speed.

### Kernels

On a GPU, bux runs attention, the recurrent mixers, the experts' GEMMs, the MoE's
dispatch and the low-precision quantizers by kernels, 1.4x to 27x as fast as XLA's own
program for the same computation. They are on by default, and each falls back to XLA
by itself where it cannot run.

- **Set**: nothing. `BUX_KERNELS=off` turns them all off, to debug one.
- **Helps**: always. Check that yours run, because a layer outside a kernel's
  conditions falls back. A delta rule with heads of 96 dims runs by chunks; KDA with a
  `gate_lower_bound` below -5 runs by bux's slower Pallas kernels; attention with a
  soft cap off a Blackwell GPU runs by tokamax's kernel, since cuDNN's takes no soft
  cap.
- **Costs**: nothing. None of them is in a run's id.
- **Measured**: [GPU kernels](kernels.md), which also lists when each falls back.

### Precision formats and recipes

On Blackwell GPUs, the projections inside blocks multiply faster in MXFP8 or NVFP4.
The embedding, the head, the router and softmax attention's own matmuls stay in
`compute`.

```bux
precision = precision(matmuls=mxfp8)    # every projection inside a block, the experts too
use nvidia_nvfp4                        # or a recipe: a module of precision settings
```

- **Helps**: GEMM-bound runs on B200s and B300s. In `mxfp8` the dense 1.3B trains
  1.18x as fast on one GPU and 1.16x on `fsdp=8`, and the MoE 1.10x in 15 GiB less. In
  `nvfp4` the 1.3B trains 1.22x as fast.
- **Costs**: a format changes what a run computes, so it is part of the run's id. Over
  3000 steps of FineWeb-Edu, the 1.3B in `mxfp8` ends within the noise of bfloat16
  (+0.12% and -0.02% at two seeds, which differ by 1.6%), and in NVFP4 about half a
  percent above it (`nvidia_nvfp4`: +0.48% and +0.56%).
- **Narrow projections gain least.** In MXFP8, k's and v's GEMMs, half as wide, take 7%
  more time than in bfloat16. Leaving them in bfloat16 (`roles=roles(kv=compute)`) is
  slower still. In NVFP4, q, k, v and o are slower than in MXFP8.
- **Where it does not pay**: `ring` attention on `context=2` trained 13% slower in
  `mxfp8` than in bfloat16, because its projections' tokens split along positions.
  Under TP, use `methods(tensor=chunked)` in MXFP8 ([Overlap](#overlap-of-communication)).
- **Also**: `moments=bfloat16` keeps the optimizer's moments in bfloat16 (66.3 to 61.6
  GiB on the 1.3B in MXFP8). `dispatch=mxfp8` sends an MoE's tokens to their experts in
  MXFP8 (`ragged_all_to_all`: 539k to 556k tokens/s).
- **Recipes**: `nvidia_mxfp8`, `nvidia_nvfp4`, `nemotron3_nvfp4`, `deepseek_v3_fp8` and
  `torchtitan_mxfp8`, in `examples/experiments/recipes/`.
- **Measured**: [Low-precision matmuls](kernels.md#low-precision-matmuls); the settings
  and their losses are in [Precision](../training/precision.md).

(tuning-memory)=

### Batch, microbatches and memory

A step gets faster as each GPU computes more rows, until they no longer fit.

- **Rows per GPU** are `batch` divided by the GPUs the batch splits over. On one GPU
  they matter little: the 1.3B trains at 139k tokens/s at 8 rows (51.1 GiB) and 146k
  at 56 (191.2 GiB). Under FSDP they matter a lot, because a step of one row per GPU
  waits for FSDP's gathers: on `fsdp=8` the 1.3B trains at 443k tokens/s at 1 row per
  GPU, 909k at 2 and 1.11M at 8 ([Scanned layers under
  FSDP](compiler.md#scanned-layers-under-fsdp)).
- **Microbatches** (`methods(microbatches=n)`) take a step's batch in n parts, a
  gradient accumulation inside the step: the same step in less memory, but slower. On
  `fsdp=8` at 64 rows per step, the 1.3B trains at 1.245M tokens/s whole and 1.054M in
  8 microbatches (`one_node.bux`'s `fsdp` and `fsdp_m8`). Use them when the batch you
  need does not fit, not to save memory you do not need.
- **Gathered once** (`methods.gather = step`, the default): a step of microbatches casts
  and gathers the parameters FSDP splits once, not once per microbatch. Set
  `gather=layer` (ZeRO-3) only when a whole copy of the model does not fit beside a
  microbatch's activations, as for Qwen3-30B-A3B's `sorted` run.
- **Reduced once** (`methods.accumulate = auto`, the default): where the rows span
  hosts, the microbatches' gradients add up unreduced and reduce once per step, unless
  a layer learns from the whole batch, as an MoE's router does. Through the NICs this
  is 1.5x as fast as reducing every microbatch.

XLA's memory pool takes 95% of a GPU by default (`runtime.POOL`), where JAX takes 75%,
unless `XLA_CLIENT_MEM_FRACTION` or its old name, `XLA_PYTHON_CLIENT_MEM_FRACTION`, is
set. The rest is for what allocates outside the pool: the CUDA context, NCCL's buffers
and NCCL EP's. On B300s (268.6 GiB), at most:

| Run (`megatron.bux`'s Qwen3-30B-A3B, or the dense 1.3B) | Outside the pool, per GPU |
|---------------------------------------------------------|---------------------------|
| `sorted`, `fsdp=8` | 1.9 GiB |
| `nccl_ep`, `expert=8` | 6.5 GiB |
| the 1.3B, `fsdp=16` over two nodes | 2.2 GiB |
| `nccl_ep`, `expert=16` over two nodes | 9.8 GiB |

At 95%, a B300's pool is 254 GiB and 14.3 GiB stay outside it. At 97% only 8.9 GiB
stay, too little for NCCL EP over two nodes, whose buffers grow with a device's tokens
and the model's width.

`remat=auto` fits the step to the pool, so a larger pool means less recomputation: at
95%, Qwen3-30B-A3B's `sorted` and `nccl_ep` runs at 8 rows of 4096 per GPU both fit
`remat=none`, each in one compile.

**Measured**: [Microbatches](../parallelism/microbatches.md#measurements).

(tuning-remat)=

### Remat

`methods.remat` says what the forward pass keeps for the backward; the backward
computes the rest again. The default, `auto`, compiles the step under `none` first.
Where that does not fit, it compiles `dots_attention`, `dots`, `save_attention` and
`full` side by side, and keeps the first of them, in that order, whose step fits 97% of
the pool: the one that computes least again.

```bux
methods = methods(remat=save_attention)                          # a preset by name
methods = methods(remat=remat(save=[attention, mlp], first=8))   # or a policy as data
```

- **Helps**: setting it by hand rarely beats `auto`. Set one to fit more rows at a
  known cost, or to skip `auto`'s extra compiles.
- **Costs**: speed, for memory. The dense 1.3B on one GPU trains at 139k tokens/s in
  51.1 GiB under `none`, and at 120k in 24.5 GiB under `full`.
- **Recurrent mixers**: `methods(states=recompute)` computes a chunked recurrence's
  states again in the backward, and lets the Mamba-2 hybrid fit 32 rows instead of 16
  ([The chunks' states](../parallelism/recurrence.md#the-chunks-states-methodsstates)).
  `remat(save=[attention, recurrence.in])` keeps only each mixer's inputs.
- **Offloading** to the host's memory does not pay on a B300: computing the values
  again is cheaper.
- **Measured**: [Remat](../parallelism/remat.md#measurements), every policy on the
  benchmarks' models.

### Donation

A step donates the model's and the optimizer's arrays, so its new state reuses the old
one's memory (`methods.donate = true`, the default).

- **Helps**: memory. With the loss by whole logits, the 1.3B's step on one GPU peaks
  at 70.6 GiB instead of 81.3, and fits 36 rows per GPU, which run out of memory
  without donation.
- **Costs**: an array read from `trainer.model` before a step is gone after it, so a
  hook that keeps one must keep a copy. Donation also changes how XLA rounds a step a
  little (the gradient's norm by 7e-6 at the dense model's first step): runs you
  compare bit for bit must donate alike.
- **Set** `methods(donate=false)` only to debug such a hook.
- **Measured**: [Donation](../parallelism/microbatches.md#donation-methodsdonate).

### The loss by chunks

`methods.loss = chunks`, the default, keeps the head's logits to at most 2 GiB of
float32 at a time, and takes each chunk's gradients in the forward pass. At a
vocabulary of 128K, the float32 logits of 8 rows of 4096 tokens take 16 GB, and their
gradient as much again.

- **Helps**: memory, at no cost in speed. The 1.3B on one GPU peaks at 51.1 GiB by
  chunks against 70.6 by whole logits, at 139k tokens/s against 138k, and fits 56
  rows. On `fsdp=8` it peaks at 38.9 GiB against 58.5.
- **Costs**: nothing. `methods(loss=logits)` is there to compare.
- **Measured**: [Loss methods](../training/extending.md#loss-methods).

(tuning-mesh)=

### Choosing a mesh

A run splits over its GPUs by `mesh`, a size for each of six axes. Start from FSDP, and
add an axis only when FSDP alone cannot fit the model or its rows:

| Your run | Mesh | Measured |
|----------|------|----------|
| parameters and optimizer state fit a node's GPUs | `mesh(fsdp=8)` | the 1.3B at 1.245M tok/s; FSDP keeps 97% of one GPU's rate |
| the same, over several nodes | `mesh(data=nodes, fsdp=8)`: HSDP | 96-97% of one node's rate per GPU, on 4 and on 8 nodes |
| layers too wide for FSDP alone | `mesh(tensor=8)` within a node, `layout(stream=tensor)`, FSDP or HSDP across | 4 of Llama 3 70B's layers at 219-222k tok/s on a node |
| an MoE | `methods(moe=sorted)` under FSDP or HSDP | the baseline's MoE at 831k tok/s on a node, 1.59M on two |
| an MoE whose experts should not be gathered | `mesh(expert=8)` within a node, `data` across | `nccl_ep` at 615k tok/s on a node, 1.19M on two |
| long rows | `mesh(context=8)`, `methods.attention` at `auto` | rows of 128k: `ulysses` 841k tok/s on packed documents, 213k on one document |
| a model too large for a node, where FSDP across nodes would gather every layer between them | one pipeline stage per node: `mesh(stage=4, fsdp=8)`, `methods(repeat=scan, microbatches=16)` | 76k tok/s/GPU on 4 nodes and on 8: 54% of one node's FSDP per GPU, 58% of FSDP over the same 4 nodes |

- **FSDP over many nodes** gathers every layer across them, forward and backward, and
  loses more as they grow: 98% of a node's rate on two nodes, 94% on four, 87% on
  eight. HSDP gathers within a node and adds the gradients up across nodes once per
  step.
- **FSDP takes the other axes too.** `layout.fsdp` is `(fsdp, context, expert)` by
  default, so a dense parameter and its state are not copied over `context` and
  `expert`. On `fsdp=4, context=2` the 1.3B's parameters take 611 MiB per GPU rather
  than 1.19 GiB, and it trains at 903k tokens/s rather than 886k.
- **TP** waits for its collectives at the edges of its matmuls, and is slower than FSDP
  wherever FSDP fits. Keep `tensor` within a node; the mesh does so by filling a node
  from `tensor` outward.
- **Context parallelism.** `auto` takes `ulysses` where `context` divides a device's
  heads, else `ring`. Keep it: at 128k, `ulysses` is 2.3x `ring` on packed rows and
  1.4x on rows of one document.
- **Pipelines** lose their bubble and their microbatches' small GEMMs, not the sends
  between nodes: on four nodes, a GPU of the pipeline trains as fast as on one node's
  `stage=4, fsdp=2`. `1f1b` on stages balanced by FLOPs is the default, 21-24% faster
  than even stages.
- **Measured**: [Parallelism](../parallelism/index.md), and its pages on
  [multi-node](../parallelism/multi-node.md), [attention](../parallelism/attention.md),
  [tensor](../parallelism/tensor.md), [experts](../parallelism/experts.md) and
  [pipelines](../parallelism/pipelines.md).

(tuning-moe)=

### MoE methods

`methods.moe` says how an MoE runs its picked experts. `sorted` sorts each device's
picks by expert under FSDP, with every expert's weights gathered onto every device. The
expert-parallel methods split the experts over `expert` and send each pick to its
expert's device.

```bux
methods = methods(moe=sorted)                                         # the default
mesh    = mesh(expert=8)
methods = methods(moe=nccl_ep, capacity=1.25)                         # EP by NCCL EP
methods = methods(moe=all_to_all, capacity=1.25, moe_chunks=2)        # or by XLA's
```

For the baseline's MoE (64 experts, top 8, `d_model` 2048) at 32k tokens per GPU and a
capacity of 1.25, `sorted` is the fastest:

| Method | One node | Two nodes |
|--------|----------|-----------|
| `sorted` | 831k tok/s, `fsdp=8` | 1.59M, HSDP |
| `nccl_ep` | 615k, `expert=8` | 1.19M with EP in each node and `data` across; 773k on `expert=16` |
| `all_to_all`, 2 chunks | 586k, `expert=8` | |
| `ragged_all_to_all` | 544k, `expert=8` | |

At Qwen3-30B-A3B's Megatron workload on 8 GPUs, `sorted` trains at 42.4k tok/s/GPU and
`nccl_ep` at 34.9k. Every method's numbers are in
[Experts](../parallelism/experts.md#one-node-and-two).

- **`sorted`** wins at these shapes: its FSDP gathers 0.5 GB of experts per layer,
  which hide behind compute, while an exchange moves 1.1 to 1.3 GB of picks each way,
  four times per layer. EP would win with an exchange near NVLink's 900 GB/s; NCCL EP's
  kernels move 330 to 370 GB/s.
- **`capacity`** bounds the rows a device of the ragged methods takes, or that
  `all_to_all` sends each peer, at that many times its share, and drops the rest. A
  dropped pick adds nothing to its token, and the step logs the share dropped as
  `dropped`: on random tokens, a capacity of 1.25 drops 4-10% of the picks per layer at
  first. A capacity that drops changes what a run computes, though it stays out of its
  id. Without a capacity (dropless), a device takes everything every peer could send it.
- **`moe_chunks`** cuts each device's tokens into slices whose exchanges overlap the
  other slices' experts. Only `all_to_all` gains (3%), and each chunk adds a grouped
  GEMM per peer per layer to compile: 2 chunks compile in 5 minutes, 4 in more than 8.
- **`precision(dispatch=mxfp8)`** sends the tokens in MXFP8: `ragged_all_to_all` at
  556k against 539k. It is part of the run's id, and an error with `nccl_ep`, which
  sends bfloat16 only.
- **`nccl_ep`** needs Transformer Engine 2.18 or later (the Docker image has 2.20.2),
  and starts once per process before the step traces, 25 to 60 s between two nodes.
  Across nodes it is 2.4x `ragged_all_to_all`, but at 63% of a node's rate per GPU:
  keep EP within a node and `data` across nodes.
- **`moe_stream=true`** runs NCCL EP's kernels on their own stream. One layer's forward
  and backward take 8.07 ms against 9.39, but a step of microbatches one after another
  gains nothing (34.1k either way), and in 2 chunks loses 3%. It is off by default; it
  pays with `interleave` ([below](#overlap-of-communication)).
- **Keep `expert` within a node** for XLA's methods: across hosts, XLA splits a ragged
  all-to-all into one within each host and an all-gather of the tokens across them.
- **`moe(random_logits=true)`** routes by random logits, as Megatron's benchmarks do.
  It is for benchmarks only, since it changes what a run learns.
- **Measured**: [Experts](../parallelism/experts.md).

(tuning-overlap)=

### Overlap of communication

When `exposed_comm_ms` is a large share of a step, its collectives wait for compute
that could run beside them. Try these in order:

1. **Reductions early** (`compiler(reductions=early)`, the default): each weight
   gradient's reduction starts beside the next layer's backward pass, not after the
   last one. Qwen3-30B-A3B at Megatron's batch gains 5%, the 1.3B in 8 microbatches
   6.2%. Set `reductions=xla` to see what it does for your run
   ({ref}`Compiler options <compiler-reductions>`).
2. **TP's collectives in chunks** (`methods(tensor=chunked)`): each chunk's all-gather
   and reduce-scatter run beside another chunk's matmul. In MXFP8, 4 of Llama 3 70B's
   layers train 5% faster (263k against 250k tokens/s); in bfloat16 about 1%; on
   `fsdp=2, tensor=4`, 3% slower.
3. **A compiler preset** (`compiler(preset=...)`): XLA's defaults are the fastest for
   every run measured except one. A scanned model under FSDP at one row per GPU trains
   56% faster with `methods(repeat=scan)` and `compiler(preset=maxtext)` (48% on two
   nodes). `lhs` and `o1` also schedule `ragged_all_to_all`, which XLA's defaults do
   not, though at one row per GPU it trained no faster for it
   ([The presets on one node](compiler.md#the-presets-on-one-node)).
4. **Microbatches side by side** (`methods(interleave=true)`): each microbatch's forward
   runs beside the backward of the one before, as in Megatron's combined 1F1B.
   - It hides NCCL EP's exchange: `nccl_ep` on 12 of Qwen3-30B-A3B's layers trains 4%
     faster with it and `moe_stream`.
   - It holds half a microbatch more (25-30 GiB there), copies it across the loop, and
     compiles in twice the time. The dense 1.3B under FSDP trains slower with it
     (93.1k against 111.2k tok/s/GPU).
   - It is off by default, and an error with a pipeline or with a model whose items
     pass each other more than the stream (a tap, an MTP head)
     ([Side by side](../parallelism/microbatches.md#side-by-side-methodsinterleave)).

XLA's own flags did not help where these did: PGLE, two collectives in flight, fewer
NCCL channels, NCCL's symmetric buffers
([What does not help](compiler.md#what-does-not-help)).

(tuning-host)=

### The host between steps

A GPU idles when it waits for the host to launch its next step. bux launches each step
before it reads the last one's metrics, so the GPU waits only after a step the host
itself must wait for: an eval, a checkpoint, the start and stop of a profile's
trace, or a step that compiled.

- **Logging** reads a step's metrics once the next step is launched, and computes its
  learning rate on the GPU behind its own step. Logging every step, the 1.3B's GPU 0
  idles 2.2 ms per step, against 2.1 ms logging every 5: log as often as you like.
- **A step that compiled**, the first and any of a new batch shape or precision, is
  waited for. A step that cannot run then raises at once, by its own name; read
  later, its error would name the next program to read its outputs, the log's
  `jit__stacked`.
- **An eval, a checkpoint, or a trace's start and stop** waits for its step. The
  launch of a large step returns only when the GPU is nearly through it (100 to 170 ms
  before the end, for Qwen3-30B-A3B), so after one the GPU waits for the next launch:
  145 to 210 ms for Qwen3-30B-A3B's 48 layers on 8 B300s. Space out `eval(every=...)`
  and `checkpoint(every=...)` (1000 steps by default).
- **The data**: a thread loads the next two batches while a step runs
  ({py:func}`bux.train.data.prefetch`), and each goes to the devices just before its
  step.

**Measured**: [Extending the trainer](../training/extending.md).

(tuning-compile)=

### Compile time and the compile cache

Compiled programs, and on GPUs the autotuner's picks, persist in JAX's cache:
`$XDG_CACHE_HOME/bux/jax` (`~/.cache/bux/jax`) by default, so every launch after the
first compiles little.

```sh
export JAX_COMPILATION_CACHE_DIR=/shared/bux-cache   # one folder for every host
export JAX_COMPILATION_CACHE_DIR=                    # or no cache at all
```

- **On many hosts**, the processes must all find a program in the cache or all compile
  it, so the cache must be one folder on a filesystem every host mounts. bux checks
  this when the processes join. Where it does not hold, bux turns the cache off and
  says so, and every launch then compiles everything. Warm, a launch of the 1.3B on two
  nodes reaches its fifth step in 42 s, against 78 s cold ([The compile
  cache](../parallelism/multi-node.md#the-compile-cache)).
- **Where a compile goes.** Compiling Qwen3-30B-A3B's 48-layer `sorted` step
  (`remat=none`) takes about 230 s on a B300's host:

  | Stage | Seconds |
  |-------|---------|
  | tracing the step | 38 |
  | lowering it to MLIR | 62, 88% of it the MXFP8 experts' Mosaic GPU kernels, lowered once per call (0.36 s each) although every layer calls them at the same shapes |
  | XLA | 132: 89 in its HLO passes, 16 in LLVM and ptxas for its 286 kernels |

  Creating the model also compiles its initializers, once per run: 79 s. cuDNN's fused
  experts add 20 to 110 s, because CuTe DSL compiles its kernels in every process.
- **`remat=auto`** compiles once where `none` fits. Where it does not, the other four
  candidates compile side by side, which can take minutes for a large model
  ([Remat](../parallelism/remat.md)). Name a policy to compile just one.
- **Scanned layers** (`methods(repeat=scan)`) compile one layer for all of them: 2.3x
  as fast on 8 GPUs. Unrolled layers, the default, train faster from two rows per GPU.
- **`moe_chunks`** and **`interleave`** multiply what XLA compiles (above).

(troubleshooting)=

## Troubleshooting

| Symptom | Likely cause | What to set |
|---------|--------------|-------------|
| Out of memory at the first step | the step's activations | leave `methods.remat` at `auto`; fewer rows per GPU, or `methods(microbatches=n)`; `precision(matmuls=mxfp8)`; for recurrent mixers, `methods(states=recompute)` |
| Out of memory in `jit__stacked`, the log's program | a step before the log ran out of memory, after the first | as for the first step |
| Out of memory at the first step with `nccl_ep` | `remat=auto` estimates the step without NCCL EP's own buffers | name a `methods.remat` policy that keeps less |
| Out of memory with many microbatches under FSDP | the whole model gathered once per step beside a microbatch | `methods(gather=layer)` |
| Out of memory at the step after the first checkpoint | the old state kept beside the new | `methods.donate` at `true`, the default |
| NCCL or NCCL EP fails to allocate, over nodes | too little memory outside XLA's pool | `XLA_CLIENT_MEM_FRACTION` at 0.95, the default; not 0.97 |
| Low MFU on one GPU | a kernel fell back, or bfloat16 GEMMs | check [when a kernel falls back](kernels.md#when-a-kernel-falls-back); `precision(matmuls=mxfp8)` on Blackwell |
| Low MFU under FSDP at one or two rows per GPU | FSDP's gathers exposed | more rows per GPU; or `methods(repeat=scan)` with `compiler(preset=maxtext)` |
| High `exposed_comm_ms` with microbatches | gradient reductions at the end of each backward | `compiler(reductions=early)`, the default |
| High `exposed_comm_ms` under TP | TP's collectives at its matmuls' edges | `methods(tensor=chunked)` in MXFP8; `tensor` within a node; FSDP where it fits |
| High `exposed_comm_ms` under EP | the token exchanges | `methods(moe=sorted)` under FSDP; `nccl_ep`; `moe_chunks=2` with `all_to_all`; `precision(dispatch=mxfp8)` |
| High `exposed_comm_ms` on many nodes | FSDP's gathers across nodes | HSDP: `mesh(data=nodes, fsdp=8)` |
| Collectives between nodes at about 1 GB/s | NCCL over TCP sockets, not InfiniBand | install libibverbs and its mlx5 provider (the Docker image has them); `NCCL_DEBUG=INFO` should say `Using network IB` |
| High `idle_ms` | the host: evals, checkpoints, reading data | larger `eval.every` and `checkpoint.every`; a faster data source |
| A ragged all-to-all hangs | NCCL's barrier on jax 0.12 | `bux train` and `bux launch` set the flag that avoids it; a script must call `bux.runtime.configure()` first ([Compiler options](compiler.md#the-flags-bux-sets-in-every-process)) |
| A launch on many hosts hangs while compiling | one host loads from its cache what the others compile | `JAX_COMPILATION_CACHE_DIR` on a filesystem every host mounts |
| Slow compiles | no warm cache; `remat=auto`'s candidates; many unrolled layers | a cache every launch reads; a named `methods.remat`; `methods(repeat=scan)` |
| Two runs of one tree end at different losses | XLA's GEMM autotuning, and adds in no fixed order | `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"` on both |
