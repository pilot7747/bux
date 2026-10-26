# Profiling

This page covers how to see where a training step's time goes: the {bux}`profile`
setting, the traces it takes (jax.profiler's, Nsight Systems' and Nsight Compute's),
the three numbers bux reads from jax.profiler's (`comm_ms`, `exposed_comm_ms` and
`idle_ms`), how bux counts a model's FLOPs for `mfu`, and how to tell from all of
these what a step waits for.

Profile a run when its `tokens_per_s` or `mfu` is lower than you expect, before you
change anything ([the tuning workflow](tuning.md#the-workflow)).

## Usage

```bux
profile = profile(start=20, steps=5)             # jax.profiler
profile = profile(start=20, steps=5, tool=nsys)  # Nsight Systems
profile = profile(start=20, steps=1, tool=ncu)   # Nsight Compute
```

| field   | what it is | default |
|---------|------------|---------|
| `start` | the step after which the trace starts | `10` |
| `steps` | how many steps it traces | `5` |
| `tool` | what traces them: `jax`, `nsys` or `ncu` | `jax` |
| `options` | more flags for nsys or ncu | `""` |

The run traces the 5 steps after step 20 into `profile/` in its directory. It waits
for the step before it starts and before it stops, so the trace holds whole steps.
Profiling at other steps, or with another tool, keeps the run's id.

| tool | what it records | in `profile/` | open it with |
|------|-----------------|---------------|--------------|
| `jax` | the host's threads, every GPU's streams and XLA's ops; and [`comm_ms`, `exposed_comm_ms`, `idle_ms`](#profiling-metrics) | `plugins/profile/<time>/<host>.xplane.pb` | xprof, TensorBoard |
| `nsys` | a timeline of every GPU's kernels, CUDA's calls and NVTX's ranges | `<host>.<pid>.nsys-rep` | Nsight Systems, `nsys stats` |
| `ncu` | each kernel's counters: its throughput, occupancy, stalls | `<host>.<pid>.ncu-repz` | Nsight Compute, `ncu --import` |

jax.profiler runs in the training process. nsys and ncu run it:
[Nsight Systems and Nsight Compute](#nsight-systems-and-nsight-compute) says how.

Open jax.profiler's trace in xprof, or in TensorBoard's profile plugin:

```sh
tensorboard --logdir runs/<run>/profile
```

Leave the first steps out of the trace: the first compiles the step, and `remat=auto`
may compile several candidates ([Remat](tuning.md#remat)).

(profiling-metrics)=

## Communication and idle time

On GPUs, the log has three more numbers at the trace's last step (step 25 above): what
a step of the traced ones spent

| metric | in |
|--------|----|
| `comm_ms` | collectives |
| `exposed_comm_ms` | collectives with no kernel computing beside them on any stream: the communication the step waits for |
| `idle_ms` | running no kernel at all |

Each is the mean, a step, over the GPUs the process traced.

### How bux computes them

{py:mod}`bux.train.trace` reads the newest `*.xplane.pb` under `profile/`
({py:func}`bux.train.trace.latest`), and {py:func}`bux.train.trace.comm_ms` does the
rest:

1. It takes each GPU's kernels from the trace's lines of streams
   (`Stream #13(compute)` and the like).
2. It sorts each kernel into one of three kinds. A kernel communicates if NCCL runs
   it, or if XLA's op that launched it (its `hlo_op`) is a collective. CUDA's copies
   and memsets neither communicate nor compute. Every other kernel computes.
3. It merges each kind's intervals on every stream of a GPU into disjoint ones.
4. `comm_ms` is the length of the communicating intervals; `exposed_comm_ms` is that
   less what overlaps a computing interval; `idle_ms` is the time from the GPU's first
   kernel to its last when none of its streams runs anything.

The same functions read a trace from a script:

```python
from pathlib import Path

from bux.train import trace

file = trace.latest(Path("runs/my-run/profile"))
print(trace.comm_ms(file, steps=5))  # comm_ms, exposed_comm_ms, idle_ms
```

{py:func}`bux.train.trace.gpu_kernels` returns every GPU's intervals of each kind,
for an analysis of your own.

### Reading them

- **A collective's time includes the wait for its slowest peer.** On unevenly loaded
  devices, such as the ring's on packed rows, `exposed_comm_ms` swings between runs of
  one speed ([Measurements](compiler.md#measurements)).
- **Most of `idle_ms` is the host.** After a step the host waits for (the trace's own
  start and stop, an eval, a checkpoint), a GPU waits for the next step's launch: 145
  to 210 ms for Qwen3-30B-A3B's 48 layers on 8 B300s, where it waits 25 ms a step
  between kernels. Logging every step, the 1.3B's GPU 0 idles 2.2 ms a step, from 57.9
  before the host fix ([The host between steps](tuning.md#the-host-between-steps)).
- **Under the profiler, a launch returns at its step's end.** XLA's launch of a large
  step returns only when the GPU is nearly through it. A small model's step, its
  kernels in CUDA graphs (XLA's command buffers), returns in 10 ms, and the host runs
  up to two steps ahead. The profiler turns command buffers off, so under it the small
  model's launch too returns at its step's end.
- **Command buffers.** jax.profiler runs XLA's command buffers a kernel at a time as
  it traces. Kept whole (`--xla_enable_command_buffers_during_profiling=true`), the
  step's idle and exposed time were the same.
- **Large steps lose events.** At Megatron's batch of 512 rows, jax's profiler lost
  most of a step's events of megatron.bux's `nccl_ep` arm.

## Reading a trace

The trace viewer shows each GPU's streams as lines of kernels, beside the host's
threads. Three things answer most questions:

- **Gaps on every stream at once** are `idle_ms`: the GPU waits for the host. Look at
  the host's threads at that time: a checkpoint, an eval, a batch being read.
- **NCCL kernels with nothing beside them** are `exposed_comm_ms`. A kernel's `hlo_op`
  names XLA's op, and so the collective: an FSDP all-gather, a gradient's
  reduce-scatter, an expert parallelism exchange.
- **The longest computing kernels** say what a step spends its time in: attention,
  the experts' GEMMs, a quantizer, a copy XLA inserted.

## Nsight Systems and Nsight Compute

`tool=nsys` takes Nsight Systems' timeline of the steps, and `tool=ncu` Nsight
Compute's counters of each kernel in them. Both are NVIDIA's profilers, which run the
training process; `bux launch` starts it under them:

- **A run alone** starts its process again under the tool, with the same command,
  before it does anything else. On many hosts each process does, and writes a report
  of its own.
- **A sweep's runs** each start their worker under it.

The tool records nothing until step `start` is done. The `profile` hook then calls
CUDA's `cudaProfilerStart`, and `cudaProfilerStop` after `steps` steps. The commands
are {py:data}`bux.train.profilers.TOOLS`:

```sh
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --trace=cuda,nvtx,cublas,cudnn \
    --force-overwrite=true --output=runs/<run>/profile/%h.%p  <the command>
ncu --profile-from-start=off --target-processes=all --force-overwrite \
    --export=runs/<run>/profile/%h.%p  <the command>
```

`options` adds flags of the tool's own after them:

```bux
profile = profile(start=20, steps=5, tool=nsys, options="--cuda-memory-usage=true")
profile = profile(start=20, steps=1, tool=ncu, options="--launch-count=20")
```

A process under either tool sets `BUX_PROFILER` to its name, and bux then changes two
things in it ({py:data}`bux.train.profilers.XLA_FLAGS`,
{py:data}`~bux.train.profilers.VARIABLES`), each unless your environment sets it:

- **XLA runs no command buffers** (`--xla_gpu_enable_command_buffer=`, before your
  `XLA_FLAGS`), as jax.profiler turns them off while it traces: `one_node.bux`'s
  `fsdp` arm trains 1.5% slower without them. With them, both tools crashed
  processes of 8 GPUs in CUDA's calls that make and free the CUDA graphs that
  command buffers are: nsys 3 of 4 runs of that arm, at exit or as they trained
  (exit 139), and ncu all 3 runs of 8 GPUs.
- **Under ncu, XLA takes memory as it needs it** (`XLA_PYTHON_CLIENT_PREALLOCATE=false`),
  not 95% of each GPU at once. ncu saves and restores that memory around each pass of
  each kernel it replays: with all of it taken, a kernel of the smoke run took 3
  minutes.

### What nsys records

On B300s, with Nsight Systems 2026.5 and jax 0.12 (2026-10-26):

- **Every GPU of the process, from one capture.** Three steps of `one_node.bux`'s
  `fsdp` arm on 8 B300s hold 1627 kernels a GPU a step, NCCL's, cuDNN's and cuBLAS's
  by name.
- **XLA's NVTX ranges.** Each run of a program is an `XlaModule` range named by the
  step (`XlaModule:#hlo_module=jit__train_step_on_shards,...`), each op in it a
  `Thunk` range with its `hlo_op`, and cuBLAS and NCCL add their calls'.
  jax.profiler's annotations are no NVTX ranges, so they do not show.
- **A traced step is nearly as fast.** A traced step of that run took 228 ms, and
  223 ms untraced. Without command buffers, which bux turns off under nsys, the run
  trains at 1.175M tokens a second, against 1.193M with them.
- **The capture's start and stop take time.** On that run, the start took about 23 s,
  and the stop 75 to 90 s, while nsys wrote its report of about 20 MB:
  `cudaProfilerStop` returns once it is written. The log's `tokens_per_s` reads low
  at the two logs around the traced steps (55k and 17k tokens a second, from 1.17M).
- **A signal reaches the run through nsys.** SIGTERM to the launcher stops the run
  after its step, as without nsys, and the launcher exits with 143.

`nsys stats --report cuda_gpu_kern_sum <report>` lists the kernels by their time,
and `nsys export --type sqlite <report>` writes the report as a database to query.
The three that took longest in that run, a GPU a step:

| kernel | what it is | ms |
|--------|------------|----|
| `ncclDevKernel_AllGather_RING` | FSDP's parameters' all-gathers | 33.8 |
| `cudnn_generated_fort_native_sdpa_...` | cuDNN's attention, a kernel a layer | 21.0 |
| `ncclDevKernel_ReduceScatter_bf16_RING` | FSDP's gradients' reduce-scatter | 18.9 |

A collective's time includes its wait for the slowest GPU. cuBLAS's GEMMs
(`nvjet_sm103_...`) come in many shapes, each a line of its own.

### What ncu records

On B300s, with Nsight Compute 2026.3 and jax 0.12 (2026-10-26):

- **Each kernel, one at a time.** ncu replays each kernel it profiles, a pass for
  each group of its counters (10 for its default set), and restores the memory the
  kernel may write before each pass. A step of the smoke run on a B300, 174
  kernels, took under two minutes.
- **Name the kernels.** A step of a large model runs thousands of kernels:
  `options="--kernel-name=regex:nvjet --launch-count=4"` profiles the first 4 of
  cuBLAS's GEMMs (`nvjet_...`), and `steps=1` keeps to one step.
- **One GPU a process.** ncu runs one kernel at a time, and a step's collectives
  over many GPUs need theirs to run together: on 8 B300s, ncu hung at a step of
  `one_node.bux`'s `fsdp` arm, and failed to save a data-parallel step's memory for
  its replay. So a launch records a run of more GPUs under ncu as failed, at once
  (`profile: ncu profiles one GPU, not 8: give the run one`): give it one with
  `mesh = mesh(data=1)`, its other axes 1.
- **Its report** is compressed (`.ncu-repz`).
  `ncu --import <report> --print-summary per-kernel` prints each kernel's counters,
  and Nsight Compute opens it.

### Running them yourself

- **The tool must be on `PATH`.** The [Docker image](../get-started/installation.md#docker)
  has both. A launch records a run whose tool it cannot find as failed, at once:
  `smoke: failed before training: profile: nsys is not on PATH`.
- **`bux train` runs under no tool.** Run it under one yourself, with `BUX_PROFILER`
  set for bux's changes, and the `profile` hook starts and stops the capture as it
  does under `bux launch`:
  `BUX_PROFILER=nsys nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop bux train run.bux`.
- **A command of your own.** With `BUX_PROFILER` set to the tool, `bux launch` starts
  no tool of its own:
  `BUX_PROFILER=nsys nsys profile --capture-range=cudaProfilerApi <flags> bux launch run.bux`.
- **Another tool** is an entry of {py:data}`~bux.train.profilers.TOOLS`, in a script
  that adds it before it launches ({py:mod}`bux.train.profilers`).

(mfu-and-flops)=

## MFU and FLOPs

On a GPU whose peak bux knows, the log has `mfu`: the model's FLOPs a second over the
peak of the mesh's GPUs. The peaks are dense bfloat16, from NVIDIA's datasheets (the
SXM parts, `flops.PEAK`):

| GPU | peak, dense bfloat16 |
|-----|----------------------|
| B300, B200 | 2.25 PFLOP/s |
| H200, H100 | 989 TFLOP/s |
| A100 | 312 TFLOP/s |

The peak is bfloat16's whatever the run's `precision`, so a run in MXFP8 or NVFP4 can
read well above a bfloat16 run's MFU. The rest of the log is in
[Running](../training/running.md).

### How bux counts FLOPs

bux counts the model's FLOPs once, before training, from the program of its loss's
gradient on a batch, traced on shapes alone with remat and kernels off
(`trainer.step_flops`, {py:mod}`bux.train.flops`):

- 2 · m · n · k for every matrix product and convolution, times the length of every
  scan around it; the larger branch of a cond; a while loop's body once.
- Each product once: what a backward pass computes again (jax.checkpoint's, the logits
  of a flash backward) counts none.
- With kernels off, since a kernel (a Pallas call, a cuDNN call) hides its FLOPs.

So a new layer counts with no formula of its own.

### Attention counts the keys each query sees

Attention counts only the keys each query sees, as Megatron's MFU does and a flash
kernel computes. Its products over the keys of a row are counted apart (inside
`flops.attention`), and each step counts them at the keys its batch's queries see,
from the batch's positions (`flops.Work.of`, on the host): those of the query's own
document up to itself, within the window, and of those no more than a sparse
attention's top_k.

| attention | counts |
|-----------|--------|
| causal, rows of one document | (t + 1) / 2t of its t × t logits |
| causal, packed documents | each document's own triangle |
| a window | the keys within it |
| DSA | its top_k keys a query in the attention, the indexer's loss and its target, with the indexer's ranking at every key it may pick |
| CSA's latents | whole |

On many hosts a step counts the keys of the rows its host reads. PaLM's MFU counts
attention whole, which made bux's 10% higher at 4096 tokens a row.

### FLOPs a token in the plan

`bux plan` shows the FLOPs a token on rows of one document, with no GPU: 6.91B for a
Llama-style 1.3B at 4096 tokens a row, 6 N of its matrices and 0.8B of attention. The
one-GPU benchmark suite:

```console
$ bux plan examples/benchmarks/one_gpu.bux
8 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
3325455d32  1.28B   983k    0.0        6.91B      1     4.77GiB     9.54GiB    0%      one_gpu[dense]
ef0e9714f9  1.38B   983k    0.0        7.07B      1     5.15GiB     10.3GiB    0%      one_gpu[gdn]
55623706de  1.39B   983k    0.0        7.11B      1     5.17GiB     10.3GiB    0%      one_gpu[kda]
78132651eb  1.39B   983k    0.0        7.11B      1     5.17GiB     10.3GiB    0%      one_gpu[kdab]
ffb822d7e6  5.16B   983k    0.0        6.92B      1     19.2GiB     38.4GiB    0%      one_gpu[moe]
f882fa43f8  1.48B   983k    0.0        7.66B      1     5.53GiB     11.1GiB    0%      one_gpu[mamba2]
22a58b5a6d  1.28B   983k    0.0        6.91B      1     4.77GiB     9.54GiB    0%      one_gpu[mxfp8]
5fb3d8d3cb  1.28B   983k    0.0        6.91B      1     4.77GiB     9.54GiB    0%      one_gpu[nvfp4]
```

Multiply the FLOPs a token by `tokens_per_s` and divide by the peak of all the run's
GPUs for the MFU the log reports on rows of one document.

## Finding what a step waits for

1. **Read the log.** `tokens_per_s` gives the step's time: the tokens of a step
   (`batch × seq_len`) over it. `mfu` says how far it is from the GPUs' peak.
2. **Profile a few steps** and read the three numbers against the step's time.
3. **Find the largest share**, and go to its lever in the
   [tuning guide](tuning.md):

| the step's time is mostly | it waits for | try |
|---------------------------|--------------|-----|
| `idle_ms` | the host | fewer evals and checkpoints, the data's reading ([The host between steps](tuning.md#the-host-between-steps)) |
| `exposed_comm_ms` | its collectives | the mesh, the MoE's method, overlap ([Overlap](tuning.md#overlap-of-communication)) |
| neither | compute | kernels, precision, rows a GPU ([The levers](tuning.md#the-levers)) |

### An example: Qwen3-30B-A3B on 8 B300s

`megatron.bux`'s `sorted` arm, profiled on GPU 0 at 8 rows of 4096 a GPU, before the
wave that took it to 90% of Megatron-Bridge's speed (1309 ms a step of 32k tokens;
Megatron takes about 800):

| what | ms | why |
|------|----|-----|
| attention, tokamax | 294 | the forward twice (remat); the backward at 0.69 PFLOP/s |
| experts' MXFP8 quantizer | 157 | 10 calls a layer, 4 of them the remat's (each at 5.1-5.8 TB/s: not slow); every GPU quantizes all 128 experts' weights |
| experts' GEMMs | 169 | 2.1 PFLOP/s: fine |
| transposes | ~96 | FSDP split the expert stacks along `embed`, not `experts`; up and gate concatenated every step |
| the picks' rows | 67 | 262144 × 2048 rows gathered in bfloat16 before the quantizer, and back |
| the router's weights | ~45 | they multiply each pick's 2048 outputs, which the backward then keeps: 1 GiB a layer |
| FSDP's collectives | 373 | 303 hidden, 69 exposed, and 72 idle |

The step was compute: most of it was neither idle nor exposed. Each row of the table
became a change, and the step fell to 884 ms: the experts' GEMMs 179, attention 138,
their quantizer 100, FSDP's collectives 281 of which 62 exposed, and 52 idle.

Later, before the second round of that work, a step of one microbatch of 8 rows a
GPU took 898 ms under `sorted`: 768 computing, 61 of collectives alone, 69 idle.
Under `nccl_ep` it took 1093: 688 computing, 326 of collectives alone, 78 idle. Its
exchange, alone for a third of its step, is what bounds expert parallelism there
([Benchmarks](benchmarks.md#against-megatron-bridge)).
