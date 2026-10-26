# Tuning guide

This guide is for when a run trains slower than you expect, or runs out of memory, and
you want to know what to change. It gives a workflow for measuring a change, then the
levers in the order to try them: for each, the setting, when it helps, what it costs,
and the page with its measurements. It ends with a
[troubleshooting table](#troubleshooting).

Every number here is from B300s, in October 2026, unless it says otherwise.

:::{tip}
bux's defaults are the fastest settings measured for the benchmarks' models: kernels
on, `remat=auto`, the loss by chunks, donation, parameters gathered once a step, and
gradient reductions started early. Start from them, and change one thing at a time.
:::

## The short version

1. **Fill the GPU.** Give each GPU as many rows as fit; under FSDP, at least two
   ([Batch](#batch-microbatches-and-memory)).
2. **Multiply in MXFP8** on Blackwell GPUs: `precision(matmuls=mxfp8)`, 1.18x for a
   dense 1.3B and 1.10x for an MoE ([Precision](#precision-formats-and-recipes)).
3. **Split by FSDP in a node, and HSDP across nodes**: `mesh(fsdp=8)`,
   `mesh(data=4, fsdp=8)` ([Choosing a mesh](#choosing-a-mesh)).
4. **Sort an MoE's experts under FSDP** (`methods(moe=sorted)`, the default) unless
   its experts' gathers cost more than sending its tokens
   ([MoE methods](#moe-methods)).
5. **Profile** what is still slow ([Profiling](profiling.md)).

## The workflow

### 1. Measure

Every log line has `tokens_per_s`, the tokens a second since the last log, and, on a
GPU whose peak bux knows, `mfu`, the model's FLOPs a second over that peak
([MFU and FLOPs](profiling.md#mfu-and-flops)). Read them after the first log, whose
time includes compiling.

`bux plan` shows, with no GPU, each run's FLOPs a token and the bytes of its
parameters and optimizer state on each GPU: what a step needs before its activations.

### 2. Profile

```bux
profile = profile(start=20, steps=5)
```

At step 25 the log then has `comm_ms`, `exposed_comm_ms` and `idle_ms`: what a step
spent in collectives, in collectives with nothing computing beside them, and running
nothing. [Profiling](profiling.md#finding-what-a-step-waits-for) says which lever each
points to. For NVIDIA's own timeline, or a kernel's counters, `tool=nsys` or
`tool=ncu` takes Nsight's report instead
([Nsight Systems and Nsight Compute](profiling.md#nsight-systems-and-nsight-compute)).

### 3. Change one thing

Write the change as a sweep, so the runs differ in that one setting:

```bux
methods.remat = keep: "none" | da: dots_attention | sa: save_attention
```

`bux launch` trains the three side by side, a GPU each, and each one's log has its
speed. Settings that change how a run computes, not what (`mesh`, `layout`, `methods`,
`compiler`), keep its id, so the arms share one and are told apart by their names.

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

In the order to try them. The first ones cost nothing but a setting; the later ones
trade memory, compile time or what a run computes for speed.

### Kernels

On a GPU, bux runs attention, the recurrent mixers, the experts' GEMMs, the MoE's
dispatch and the low-precision quantizers by kernels: 1.4x to 27x XLA's own program
for the same computation. They are on by default, and each falls back to XLA by itself
where it cannot run.

- **Set**: nothing. `BUX_KERNELS=off` turns them all off, to debug one.
- **Helps**: always. Check that yours run: a layer outside a kernel's conditions
  falls back. A delta rule with heads of 96 dims runs by chunks, KDA with a
  `gate_lower_bound` below -5 by bux's slower Pallas kernels, and attention with a
  soft cap off a Blackwell GPU by tokamax's kernel, which cuDNN's does not replace.
- **Costs**: nothing; none of them is in a run's id.
- **Measured**: [Kernels](kernels.md), which also lists when each falls back.

### Precision formats and recipes

On Blackwell GPUs the projections inside blocks multiply faster in MXFP8 or NVFP4.
The embedding, the head, the router and softmax attention's own matmuls stay in
`compute`.

```bux
precision = precision(matmuls=mxfp8)    # every projection inside a block, the experts too
use nvidia_nvfp4                        # or a recipe, a module of precision settings
```

- **Helps**: GEMM-bound runs on B200s and B300s. In `mxfp8` the dense 1.3B trains
  1.18x as fast on one GPU and 1.16x on `fsdp=8`, and the MoE 1.10x in 15 GiB less;
  in `nvfp4` the 1.3B trains 1.22x as fast
  ([Low-precision matmuls](kernels.md#low-precision-matmuls) has the tables).
- **Costs**: a format changes what a run computes, so it is in the run's id. 3000
  steps of FineWeb-Edu on the 1.3B end within the noise of bfloat16 in `mxfp8`
  (+0.12% and -0.02% at two seeds, which differ by 1.6%), and about half a percent
  above it in NVFP4 (`nvidia_nvfp4`: +0.48% and +0.56%).
- **Narrow projections gain least.** In MXFP8, k's and v's GEMMs, half as wide, take
  7% more time than in bfloat16. Leaving them in bfloat16 (`roles=roles(kv=compute)`)
  is slower still. In NVFP4, q, k, v and o are slower than in MXFP8.
- **Where it does not pay**: `ring` attention on `context=2` trained 13% slower in
  `mxfp8` than in bfloat16, as its projections' tokens split along positions. Under
  TP, use `methods(tensor=chunked)` in MXFP8 ([Overlap](#overlap-of-communication)).
- **Also**: `moments=bfloat16` keeps the optimizer's moments in bfloat16 (66.3 to
  61.6 GiB on the 1.3B in MXFP8); `dispatch=mxfp8` sends an MoE's tokens to their
  experts in MXFP8 (`ragged_all_to_all`: 539k to 556k).
- **Recipes**: `nvidia_mxfp8`, `nvidia_nvfp4`, `nemotron3_nvfp4`, `deepseek_v3_fp8`
  and `torchtitan_mxfp8`, in `examples/experiments/recipes/`.
- **Measured**: [Low-precision matmuls](kernels.md#low-precision-matmuls); the settings
  and their losses in [Precision](../training/precision.md).

(tuning-memory)=

### Batch, microbatches and memory

A step's speed grows with the rows each GPU computes, until they no longer fit.

- **Rows a GPU** are `batch` over the GPUs the batch splits over. On one GPU they
  matter little: the 1.3B trains at 139k tokens a second at 8 rows (51.1 GiB) and 146k
  at 56 (191.2 GiB). Under FSDP they matter a lot, as a step at a row a GPU waits for
  FSDP's gathers: on `fsdp=8` the 1.3B trains at 443k tokens a second at a row a GPU,
  909k at 2 and 1.11M at 8 ([Scanned layers under
  FSDP](compiler.md#scanned-layers-under-fsdp)).
- **Microbatches** (`methods(microbatches=n)`) take a step's batch in n parts, a
  gradient accumulation inside the step: the same step in less memory, but slower. The
  1.3B on `fsdp=8` at 64 rows a step trains at 1.245M tokens a second whole and 1.054M
  in 8 microbatches (one_node.bux's `fsdp` and `fsdp_m8`). Use them when the batch
  you need does not fit, not to save memory you do not need.
- **Gathered once** (`methods.gather = step`, the default): a step of microbatches
  casts and gathers the parameters FSDP splits once, not once a microbatch. Set
  `gather=layer` (ZeRO-3) only when a whole copy of the model does not fit beside a
  microbatch's activations, as Qwen3-30B-A3B's `sorted` arm does.
- **Reduced once** (`methods.accumulate = auto`, the default): where the rows span
  hosts, the gradients add up unreduced and reduce once a step. Through the NICs, 8
  microbatches trained at 738k tokens a second so, against 500k reduced each
  microbatch.

XLA's memory pool takes 95% of a GPU by default (`runtime.POOL`), where JAX takes 75%,
unless `XLA_CLIENT_MEM_FRACTION` or its old name, `XLA_PYTHON_CLIENT_MEM_FRACTION`,
says otherwise. The rest is for what allocates beside the pool: the CUDA context,
NCCL's buffers and NCCL EP's, which on B300s (268.6 GiB) took, a GPU at most:

| run (`megatron.bux`, Qwen3-30B-A3B; the dense 1.3B) | beside the pool |
|------------------------------------------------------|-----------------|
| `sorted`, `fsdp=8`                                   | 1.9 GiB         |
| `nccl_ep`, `expert=8`                                | 6.5 GiB         |
| the 1.3B, `fsdp=16` over two nodes                   | 2.2 GiB         |
| `nccl_ep`, `expert=16` over two nodes                | 9.8 GiB         |

At 95% a B300's pool is 254 GiB, and 14.3 GiB stay beside it. At 97% 8.9 GiB stay,
too little for NCCL EP over two nodes, whose buffers grow with a device's tokens and
the model's width.

`remat=auto` fits the step to the pool. Before the MoE's weights moved before `down`,
Qwen3-30B-A3B's `sorted` arm took `dots` in JAX's pool of 75% (210k tokens a second,
180 GiB). Now it takes `none` at 95% (295k, 215 GiB), as does its `nccl_ep` arm (246k,
211 GiB), each in one compile.

**Measured**: [Microbatches](../parallelism/microbatches.md#measurements).

(tuning-remat)=

### Remat

`methods.remat` says what the forward pass keeps for the backward; the backward
computes the rest again. The default, `auto`, compiles the step under `none` first,
and where that does not fit, under `dots_attention`, `dots`, `save_attention` and
`full` side by side; it keeps the fastest whose step fits 97% of the pool.

```bux
methods = methods(remat=save_attention)                     # a preset by name
methods = methods(remat=remat(save=[attention, mlp], first=8))   # or a policy as data
```

- **Helps**: setting it by hand rarely beats `auto`, which takes the policy that
  computes least again among those that fit. Set one to fit more rows at a known
  cost, or to skip `auto`'s extra compiles.
- **Costs**: speed, for memory. The dense 1.3B on one GPU, 8 rows of 4096, trains at
  139k tokens a second in 51.1 GiB under `none`, 137k in 41.6 under `dots_attention`
  and 120k in 24.5 under `full`
  ([Policies on a B300](../parallelism/remat.md#policies-on-a-b300) has every
  policy).
- **Recurrent mixers**: `methods(states=recompute)` makes the backward of a recurrence
  by chunks compute its chunks' states again, in a fourteenth of the memory and a fifth
  more time. The Mamba-2 hybrid then fits 32 rows instead of 16, at 4-9% fewer tokens a
  second. Keeping only a mixer's inputs, `remat(save=[attention, recurrence.in])`,
  costs a GDN or Mamba-2 hybrid 2-3% for 59-68% less memory.
- **Offloading** to the host's memory saves nothing unrolled, and costs a third of the
  speed: computing the values again is cheaper on a B300.
- **Measured**: [Remat](../parallelism/remat.md#measurements).

### Donation

A step donates the model's and the optimizer's arrays, so that its new state takes the
old one's memory (`methods.donate = true`, the default).

- **Helps**: memory. The 1.3B's step on one GPU, its loss by logits, peaks at 70.6 GiB
  instead of 81.3 (51.1 by chunks, the default), and trains 36 rows a GPU, which run
  out of memory without it. The MoE's peaks at 153.5 GiB instead of 163.9, trains 12
  rows, and survives a checkpoint every 10 steps.
- **Costs**: an array read from `trainer.model` before a step is gone after it, so a
  hook that keeps one keeps a copy. Donation changes how XLA rounds a step a little
  (the gradient's norm by 7e-6 at the dense model's first step): runs compared bit for
  bit donate alike.
- **Set** `methods(donate=false)` only to debug such a hook.
- **Measured**: [Donation](../parallelism/microbatches.md#donation-methodsdonate).

### The loss by chunks

`methods.loss = chunks`, the default, makes the head's logits at most 2 GiB of float32
at a time, and takes each chunk's gradients in the forward. At a vocabulary of 128K,
the float32 logits of 8 rows of 4096 tokens take 16 GB, and their gradient as much
again.

- **Helps**: memory, at no cost in speed. The 1.3B on one GPU peaks at 51.1 GiB by
  chunks against 70.6 by whole logits, at 139k tokens a second against 138k, and fits
  56 rows. On `fsdp=8` it peaks at 38.9 GiB against 58.5. The GDN hybrid and the MoE
  train 6% and 5% faster by it than by the chunks before it.
- **Costs**: nothing. `methods(loss=logits)` is there to compare.
- **Measured**: [Loss methods](../training/extending.md#loss-methods).

(tuning-mesh)=

### Choosing a mesh

A run splits over its GPUs by `mesh`, a size for each of six axes. Start from FSDP and
add an axis only when FSDP alone cannot fit the model or its rows:

| your run | mesh | measured |
|----------|------|----------|
| a model whose parameters and optimizer state fit a node's GPUs | `mesh(fsdp=8)` | the 1.3B at 1.245M tok/s; FSDP keeps 97% of a GPU's rate alone |
| the same over nodes | `mesh(data=nodes, fsdp=8)`: HSDP | 96-97% of a node's rate a GPU on 4 and on 8 nodes |
| layers too wide for FSDP alone | `mesh(tensor=8)` inside a node, `layout(stream=tensor)`, FSDP or HSDP across | 4 of Llama 3 70B's layers at 219-222k on a node |
| an MoE | `methods(moe=sorted)` under FSDP or HSDP | the baseline MoE at 831k on a node, 1,586k on two |
| an MoE whose experts should not be gathered | `mesh(expert=8)` inside a node, `data` across | `nccl_ep` at 615k on a node, 1,192k on two |
| long rows | `mesh(context=8)`, `methods.attention` at `auto`; `ring` for long packed rows | 128k tokens a row: `ring` 346k on packed documents, `ulysses` 197k on one document |
| a model too large for a node, where FSDP across nodes would gather every layer between them | a pipeline of a stage a node: `mesh(stage=4, fsdp=8)`, `methods(repeat=scan, microbatches=16)` | 76k tok/s a GPU on 4 nodes and on 8: 54% of one node's FSDP a GPU, 58% of FSDP's over the 4 nodes |

- **FSDP over many nodes** gathers every layer across them, forward and backward, and
  loses more as they grow: 98% of a node's rate on two, 94% on four, 87% on eight.
  HSDP gathers within a node and adds the gradients up across nodes once a step.
- **FSDP takes the other axes too.** `layout.fsdp` is `(fsdp, context, expert)` by
  default, so a dense parameter and its state are not copied over `context` and
  `expert`: on `fsdp=4, context=2` the 1.3B's parameters take 611 MiB a GPU, from
  1.19 GiB, at 903k tokens a second, from 886k.
- **TP** waits for its collectives at the edges of its matmuls, and is slower than
  FSDP where FSDP fits: keep `tensor` inside a node, which the mesh does by filling a
  node from `tensor` outward.
- **Context parallelism.** `auto` takes `ulysses` where `context` divides a device's
  heads, else `ring`. It cannot tell packed rows from rows of one document before a
  batch comes, so give a run of long packed rows `methods(attention=ring)`: 1.5x
  `ulysses` at 128k.
- **Pipelines** lose their bubble and their microbatches' small GEMMs, not the sends
  between nodes: on four nodes a GPU of the pipeline trains as fast as on one node's
  `stage=4, fsdp=2`. `1f1b` on stages balanced by FLOPs is the default, 21-24% faster
  than even stages.
- **Measured**: [Parallelism](../parallelism/index.md), and its pages on
  [multi-node](../parallelism/multi-node.md), [attention](../parallelism/attention.md),
  [tensor](../parallelism/tensor.md), [experts](../parallelism/experts.md) and
  [pipelines](../parallelism/pipelines.md).

(tuning-moe)=

### MoE methods

`methods.moe` says how an MoE runs its picked experts: `sorted` sorts each device's
picks by expert under FSDP, every expert's kernels gathered onto every device; the
expert parallel methods split the experts over `expert` and send each pick to its
expert's device.

```bux
methods = methods(moe=sorted)                                         # the default
mesh    = mesh(expert=8)
methods = methods(moe=nccl_ep, capacity=1.25)                         # EP by NCCL EP
methods = methods(moe=all_to_all, capacity=1.25, moe_chunks=2)        # or by XLA's
```

For the baseline's MoE (64 experts, top 8, `d_model` 2048) at 32k tokens a GPU and a
capacity of 1.25, `sorted` is the fastest:

- on a node, 831k tokens a second under `fsdp=8`, against 615k by `nccl_ep`, 586k by
  `all_to_all` in 2 chunks and 544k by `ragged_all_to_all` on `expert=8`;
- on two nodes, 1,586k under HSDP, against 1,192k by `nccl_ep` with EP in each node
  and `data` across them, and 773k by `nccl_ep` on `expert=16`.

At Qwen3-30B-A3B's Megatron workload on 8 GPUs, `sorted` trains at 42.4k tokens a
second a GPU and `nccl_ep` at 34.9k. Every method's numbers are in
[Experts](../parallelism/experts.md#one-node-and-two).

- **`sorted`** is the fastest at these shapes: its FSDP gathers 0.5 GB of experts a
  layer, which hide behind compute, where an exchange moves 1.1 to 1.3 GB of picks
  each way, four times a layer. EP would win with an exchange near NVLink's 900 GB/s;
  NCCL EP's kernels move 330 to 370 GB/s.
- **`capacity`** bounds the rows a device of the ragged methods takes, or that
  `all_to_all` sends each peer, at that many times its share, and drops the rest. A
  dropped pick adds nothing to its token, and the step logs the share dropped as
  `dropped`: on random tokens a capacity of 1.25 drops 4-10% of the picks a layer at
  first. A capacity that drops changes what a run computes, though it stays out of its
  id. Dropless, a device takes what every peer could send it.
- **`moe_chunks`** cuts each device's tokens into slices whose exchanges overlap the
  other slices' experts. Only `all_to_all` gains (3%), and each chunk is a grouped
  GEMM a peer a layer more to compile: 2 chunks compile in 5 minutes, 4 in more than 8.
- **`precision(dispatch=mxfp8)`** sends the tokens in MXFP8: `ragged_all_to_all` at
  556k against 539k. It is in the run's id, and an error with `nccl_ep`, which sends
  bfloat16 only.
- **`nccl_ep`** needs TransformerEngine (2.18 on; the Docker image has 2.20.2), and
  starts once a process before the step traces, 25 to 60 s between two nodes. Across
  nodes it is 2.4x `ragged_all_to_all`, but at 63% of a node's rate a GPU: keep EP in
  a node and `data` across nodes.
- **`moe_stream=true`** runs NCCL EP's kernels on a stream of their own. One layer's
  forward and backward take 8.07 ms against 9.39, but a step of microbatches one after
  another gains nothing (34.1k either way), and in 2 chunks loses 3%. It is off by
  default, for `interleave` ([below](#overlap-of-communication)).
- **Keep `expert` inside a node** for XLA's methods: across hosts, XLA splits a ragged
  all-to-all into one within each host and an all-gather of the tokens across them.
- **`moe(random_logits=true)`** routes by random logits, as Megatron's benchmarks do:
  for benchmarks only, as it changes what a run learns.
- **Measured**: [Experts](../parallelism/experts.md).

(tuning-overlap)=

### Overlap of communication

When `exposed_comm_ms` is a large share of a step, its collectives wait for compute
that could run beside them. In the order to try:

1. **Reductions early** (`compiler(reductions=early)`, the default): each weight
   gradient's reduction starts beside the next layer's backward pass, not after the
   last. Qwen3-30B-A3B at Megatron's batch +5%, the 1.3B in 8 microbatches +6.2%.
   Set `reductions=xla` to see what it does for your run
   ({ref}`XLA's options <compiler-reductions>`).
2. **TP's collectives in chunks** (`methods(tensor=chunked)`): each chunk's
   all-gather and reduce-scatter beside another chunk's matmul. In MXFP8, 4 of Llama 3
   70B's layers train 5% faster (263k against 250k); in bfloat16 about 1%, and on
   `fsdp=2, tensor=4` 3% slower.
3. **A compiler preset** (`compiler(preset=...)`): XLA's defaults are the fastest for
   every run measured but one. A scanned model under FSDP at a row a GPU trains 56%
   faster with `methods(repeat=scan)` and `compiler(preset=maxtext)` (48% on two
   nodes). `lhs` and `o1` also schedule `ragged_all_to_all`, which XLA's defaults do
   not, though at a row a GPU it trained no faster for it
   ([The presets on one node](compiler.md#the-presets-on-one-node)).
4. **Microbatches side by side** (`methods(interleave=true)`): each microbatch's
   forward beside the backward of the one before, as Megatron's combined 1F1B.
   - It hides NCCL EP's exchange: `nccl_ep` on 12 of Qwen3-30B-A3B's layers trains 4%
     faster with it and `moe_stream`.
   - It holds half a microbatch more (25-30 GiB there), copies it across the loop, and
     compiles in twice the time. The dense 1.3B under FSDP trains slower by it (93.1k
     against 111.2k a GPU).
   - It is off by default, and an error with a pipeline or a model whose items pass
     each other more than the stream (a tap, an MTP head)
     ([Side by side](../parallelism/microbatches.md#side-by-side-methodsinterleave)).

XLA's own flags did not help where these did: PGLE, two collectives in flight, fewer
NCCL channels, NCCL's symmetric buffers ([What XLA's own flags
did](compiler.md#what-xlas-own-flags-did)).

(tuning-host)=

### The host between steps

A GPU idles when it waits for the host to launch its next step. Since the host fix
(`cc9e425`), the GPU waits only after a step the host itself waits for: an eval, a
checkpoint, or a profile's trace as it starts and stops.

- **The log** reads a step's metrics once the next step is launched, and computes its
  learning rate behind its own step. Logging every step, the 1.3B's GPU 0 idles
  2.2 ms a step, against 2.1 ms logging every 5: log as often as you like.
- **An eval, a checkpoint or a trace's start and stop** waits for its step. The launch
  of a large step returns only when the GPU is nearly through it (Qwen3-30B-A3B's 100
  to 170 ms before its end), so after one the GPU waits for the next launch: 145 to
  210 ms for Qwen3-30B-A3B's 48 layers on 8 B300s. Space out `eval(every=...)` and
  `checkpoint(every=...)` (1000 steps by default).
- **The data**: a thread loads the next two batches as a step runs (`prefetch`), and
  each goes to the devices just before its step.

Before the fix, nnx.jit walked the model's and the optimizer's graph at every call:
20 ms a step of the 1.3B, 70 of Qwen3-30B-A3B, which the GPU waited for. The 1.3B
under FSDP gained 1.7% (1.224M to 1.245M), in 8 microbatches 4.2%, and the one-GPU
suite 1.2% to 7.2%. **Measured**: [Extending the trainer](../training/extending.md).

(tuning-compile)=

### Compile time and the compile cache

Compiled programs, and on GPUs the autotuner's picks, persist in JAX's cache:
`$XDG_CACHE_HOME/bux/jax` (`~/.cache/bux/jax`) by default, so a launch after the first
compiles little.

```sh
export JAX_COMPILATION_CACHE_DIR=/shared/bux-cache   # one folder for every host
export JAX_COMPILATION_CACHE_DIR=                    # or no cache at all
```

- **On many hosts** the processes must all find a program in the cache, or all
  compile it, so the cache must be one folder on a filesystem every host mounts. When
  the processes join, bux checks that it is. Where it is not, bux turns the cache off
  and says so, and every launch then compiles everything, together. Warm, a launch of
  the 1.3B on two nodes reaches its fifth step in 42 s, against 78 cold; the train
  step loads in 0.3 s where it compiles in 16 s
  ([The compile cache](../parallelism/multi-node.md#the-compile-cache)).
- **Where a compile goes.** A compile of Qwen3-30B-A3B's `sorted` 48-layer step
  (`remat=none`) takes about 230 s on a B300's host:

  | stage | seconds |
  |-------|---------|
  | tracing the step | 38 |
  | lowering it to MLIR | 62, 88% of it the MXFP8 experts' Mosaic GPU kernels, lowered once a call (0.36 s each), though every layer calls them at the same shapes |
  | XLA | 132: 89 in its HLO passes, 16 in LLVM and ptxas for its 286 kernels |

  Creating the model compiles its initializers besides, once a run: 79 s. cuDNN's
  fused experts add 20 to 110 s, as CuTe DSL compiles its kernels in every process.
- **`remat=auto`** compiles once where `none` fits. Where it does not, the other four
  candidates compile side by side: the dense 1.3B at 96 rows took 121 s, and
  Qwen3-30B-A3B's `nccl_ep` arm in JAX's 75% pool 699 s. Name a policy to compile one.
- **Scanned layers** (`methods(repeat=scan)`) compile one layer for all: 2.3 times as
  fast on 8 GPUs. Unrolled, the default, trains faster from two rows a GPU.
- **`moe_chunks`** and **`interleave`** multiply what XLA compiles (above).

(troubleshooting)=

## Troubleshooting

| symptom | likely cause | what to set |
|---------|--------------|-------------|
| Out of memory at the first step | the step's activations | leave `methods.remat` at `auto`; fewer rows a GPU, or `methods(microbatches=n)`; `precision(matmuls=mxfp8)`; for recurrent mixers `methods(states=recompute)` |
| Out of memory, many microbatches under FSDP | the whole model gathered once a step beside a microbatch | `methods(gather=layer)` |
| Out of memory at the step after the first checkpoint | the old state kept beside the new | `methods.donate` at `true`, the default |
| NCCL or NCCL EP fails to allocate, over nodes | too little memory beside XLA's pool | `XLA_CLIENT_MEM_FRACTION` at 0.95, the default; not 0.97 |
| Low MFU on one GPU | a kernel fell back, or bfloat16 GEMMs | check [when a kernel falls back](kernels.md#when-a-kernel-falls-back); `precision(matmuls=mxfp8)` on Blackwell |
| Low MFU under FSDP at one or two rows a GPU | FSDP's gathers exposed | more rows a GPU; or `methods(repeat=scan)` with `compiler(preset=maxtext)` |
| High `exposed_comm_ms` with microbatches | gradient reductions at the end of each backward | `compiler(reductions=early)`, the default |
| High `exposed_comm_ms` under TP | TP's collectives at its matmuls' edges | `methods(tensor=chunked)` in MXFP8; `tensor` inside a node; FSDP where it fits |
| High `exposed_comm_ms` under EP | the token exchanges | `methods(moe=sorted)` under FSDP; `nccl_ep`; `moe_chunks=2` with `all_to_all`; `precision(dispatch=mxfp8)` |
| High `exposed_comm_ms` on many nodes | FSDP's gathers across nodes | HSDP: `mesh(data=nodes, fsdp=8)` |
| Collectives between nodes at about 1 GB/s | NCCL over TCP sockets, not InfiniBand | install libibverbs and its mlx5 provider (the Docker image has them); `NCCL_DEBUG=INFO` says `Using network IB` |
| High `idle_ms` | the host: evals, checkpoints, reading data | larger `eval.every` and `checkpoint.every`; a faster source |
| A ragged all-to-all hangs | NCCL's barrier on jax 0.12 | `bux train` and `bux launch` set the flag; a script calls `bux.runtime.configure()` first ([XLA's options](compiler.md#the-flags-bux-sets-in-every-process)) |
| A launch on many hosts hangs as it compiles | one host loads from its cache what the others compile | `JAX_COMPILATION_CACHE_DIR` on a filesystem every host mounts |
| Slow compiles | no warm cache; `remat=auto`'s candidates; many unrolled layers | a cache every launch reads; a named `methods.remat`; `methods(repeat=scan)` |
| Two runs of one tree end at different losses | XLA's GEMM autotuning, and adds in no fixed order | `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"` on both |
