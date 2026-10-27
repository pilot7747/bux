# Profiling

Profiling shows where a training step's time goes. One setting, {bux}`profile`, traces
a few steps by jax.profiler, Nsight Systems or Nsight Compute. On GPUs, bux reads
jax.profiler's trace into three numbers in the log, `comm_ms`, `exposed_comm_ms` and
`idle_ms`, which say whether a step waits for compute, for its collectives or for the
host. This page also explains how bux counts a model's FLOPs for `mfu`.

Profile a run when its `tokens_per_s` or `mfu` is lower than you expect, before you
change anything ([the tuning workflow](tuning.md#the-workflow)).

## Usage

```bux
profile = profile(start=20, steps=5)             # jax.profiler
profile = profile(start=20, steps=5, tool=nsys)  # Nsight Systems
profile = profile(start=20, steps=1, tool=ncu)   # Nsight Compute
```

| Field | What it is | Default |
|-------|------------|---------|
| `start` | the step after which the trace starts | `10` |
| `steps` | how many steps it traces | `5` |
| `tool` | what traces them: `jax`, `nsys` or `ncu` | `jax` |
| `options` | more flags for nsys or ncu | `""` |

The first line traces the 5 steps after step 20 into `profile/` in the run directory.
The run waits for the step before it starts and before it stops the trace, so the
trace holds whole steps. Profiling at other steps, or with another tool, keeps the
run's id.

| Tool | What it records | In `profile/` | Open it with |
|------|-----------------|---------------|--------------|
| `jax` | the host's threads, every GPU's streams and XLA's ops; and [`comm_ms`, `exposed_comm_ms`, `idle_ms`](#profiling-metrics) | `plugins/profile/<time>/<host>.xplane.pb` | xprof, TensorBoard |
| `nsys` | a timeline of every GPU's kernels, CUDA's calls and NVTX ranges | `<host>.<pid>.nsys-rep` | Nsight Systems, `nsys stats` |
| `ncu` | each kernel's counters: its throughput, occupancy, stalls | `<host>.<pid>.ncu-repz` | Nsight Compute, `ncu --import` |

jax.profiler runs inside the training process; nsys and ncu run the process
([Nsight Systems and Nsight Compute](#nsight-systems-and-nsight-compute)). Open
jax.profiler's trace in xprof, or in TensorBoard's profile plugin:

```sh
tensorboard --logdir runs/<run>/profile
```

Leave the first steps out of the trace: the first compiles the step, and `remat=auto`
may compile several candidates ([Remat](tuning.md#remat)).

(profiling-metrics)=

## Communication and idle time

On GPUs, the log at the trace's last step (step 25 above) has three more numbers, each
the mean per step over the traced steps and the GPUs the process traced:

| Metric | Time a step spent |
|--------|-------------------|
| `comm_ms` | in collectives |
| `exposed_comm_ms` | in collectives with no kernel computing beside them on any stream: the communication the step waits for |
| `idle_ms` | running no kernel at all |

### Reading them

- **A collective's time includes the wait for its slowest peer.** On unevenly loaded
  devices, such as the ring's on packed rows, `exposed_comm_ms` swings between runs of
  the same speed ([Measurements](compiler.md#measurements)).
- **Most of `idle_ms` is the host.** After a step the host waits for (the trace's own
  start and stop, an eval, a checkpoint), a GPU waits for the next step's launch: 145
  to 210 ms for Qwen3-30B-A3B's 48 layers on 8 B300s, against 25 ms per step between
  kernels otherwise. Logging every step, the 1.3B's GPU 0 idles 2.2 ms per step
  ([The host between steps](tuning.md#the-host-between-steps)).
- **Under the profiler, every launch returns at its step's end.** XLA's launch of a
  large step returns only when the GPU is nearly through it. A small model's step,
  its kernels in CUDA graphs (XLA's command buffers), returns in 10 ms, and the host
  runs up to two steps ahead. The profiler turns command buffers off, so under it a
  small model's launch also returns at its step's end.
- **Command buffers** run a kernel at a time while jax.profiler traces. Kept whole
  (`--xla_enable_command_buffers_during_profiling=true`), the step's idle and exposed
  time were the same.
- **Very large steps lose events.** At Megatron's batch of 512 rows, jax.profiler lost
  most of a step's events for `megatron.bux`'s `nccl_ep` run.

### How bux computes them

{py:mod}`bux.train.trace` reads the newest `*.xplane.pb` of the process's host under
`profile/` ({py:func}`bux.train.trace.latest`), where every host of a job writes its
own, and {py:func}`bux.train.trace.comm_ms` does the rest:

1. It takes each GPU's kernels from the trace's stream lines (`Stream #13(compute)`
   and the like).
2. It sorts each kernel into one of three kinds. A kernel communicates if NCCL runs it,
   or if the XLA op that launched it (its `hlo_op`) is a collective. CUDA's copies and
   memsets neither communicate nor compute. Every other kernel computes.
3. It merges each kind's intervals over every stream of a GPU into disjoint ones.
4. `comm_ms` is the length of the communicating intervals; `exposed_comm_ms` is that
   minus what overlaps a computing interval; `idle_ms` is the time from the GPU's first
   kernel to its last when none of its streams runs anything.

The same functions read a trace from a script:

```python
from pathlib import Path

from bux.train import trace

file = trace.latest(Path("runs/my-run/profile"))  # any host's; host="node-0", its
print(trace.comm_ms(file, steps=5))  # comm_ms, exposed_comm_ms, idle_ms
```

{py:func}`bux.train.trace.gpu_kernels` returns every GPU's intervals of each kind, for
your own analysis.

## Finding what a step waits for

1. **Read the log.** `tokens_per_s` gives the step's time: a step's tokens
   (`batch × seq_len`) divided by it. `mfu` says how far it is from the GPUs' peak.
2. **Profile a few steps**, and read the three numbers against the step's time.
3. **Find the largest share**, and go to its lever in the [tuning guide](tuning.md):

| The step's time is mostly | It waits for | Try |
|---------------------------|--------------|-----|
| `idle_ms` | the host | fewer evals and checkpoints, faster data reading ([The host between steps](tuning.md#the-host-between-steps)) |
| `exposed_comm_ms` | its collectives | the mesh, the MoE's method, overlap ([Overlap](tuning.md#overlap-of-communication)) |
| neither | compute | kernels, precision, more rows per GPU ([The levers](tuning.md#the-levers)) |

### An example: Qwen3-30B-A3B on 8 B300s

`megatron.bux`'s two runs, profiled on GPU 0 at one microbatch of 8 rows of 4096 per
GPU, in October 2026:

| Run | Step | Computing | Collectives alone | Idle | Bound by |
|-----|------|-----------|-------------------|------|-----------|
| `sorted`, `fsdp=8` | 898 ms | 768 ms | 61 ms | 69 ms | compute |
| `nccl_ep`, `expert=8` | 1093 ms | 688 ms | 326 ms | 78 ms | its token exchange |

`sorted` is compute-bound, so its levers are kernels and precision. `nccl_ep` waits for
its exchange for a third of its step, which is what bounds expert parallelism at this
workload: its levers are overlap (`methods(interleave=true)`) or `sorted` itself
([Against Megatron-Bridge](benchmarks.md#against-megatron-bridge)).

## Reading a trace

The trace viewer shows each GPU's streams as lines of kernels, beside the host's
threads. Three things answer most questions:

- **Gaps on every stream at once** are `idle_ms`: the GPU waits for the host. Look at
  the host's threads at that time: a checkpoint, an eval, a batch being read.
- **NCCL kernels with nothing beside them** are `exposed_comm_ms`. A kernel's `hlo_op`
  names the XLA op, and so the collective: an FSDP all-gather, a gradient's
  reduce-scatter, an expert-parallel exchange.
- **The longest computing kernels** say what a step spends its time in: attention,
  the experts' GEMMs, a quantizer, a copy XLA inserted.

## Nsight Systems and Nsight Compute

`tool=nsys` takes Nsight Systems' timeline of the traced steps, and `tool=ncu` Nsight
Compute's counters for each kernel in them. Both tools run the training process, so
`bux launch` starts it under them:

- **A single run** starts its process again under the tool, with the same command,
  before it does anything else. On many hosts each process does, and writes its own
  report.
- **A sweep's runs** each start their worker under the tool.

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

`options` adds the tool's own flags after these:

```bux
profile = profile(start=20, steps=5, tool=nsys, options="--cuda-memory-usage=true")
profile = profile(start=20, steps=1, tool=ncu, options="--launch-count=20")
```

A process under either tool has `BUX_PROFILER` set to the tool's name, and bux changes
two things in it ({py:data}`bux.train.profilers.XLA_FLAGS`,
{py:data}`~bux.train.profilers.VARIABLES`), each unless your environment sets it:

- **XLA runs no command buffers** (`--xla_gpu_enable_command_buffer=`, placed before
  your `XLA_FLAGS`), as jax.profiler does while it traces. With command buffers, both
  tools crashed processes of 8 GPUs in the CUDA calls that create and free CUDA graphs:
  nsys in 3 of 4 runs of `one_node.bux`'s `fsdp` run (exit 139), and ncu in all 3 runs.
  Without them that run trains 1.5% slower.
- **Under ncu, XLA allocates memory as it needs it**
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`), rather than taking 95% of each GPU at once.
  ncu saves and restores that memory around each pass of each kernel it replays: with
  all of it taken, one kernel of the smoke run took 3 minutes.

### Nsight Systems

On B300s, with Nsight Systems 2026.5 and jax 0.12 (2026-10-26):

- **One capture covers every GPU of the process.** Three steps of `one_node.bux`'s
  `fsdp` run on 8 B300s hold 1627 kernels per GPU per step, NCCL's, cuDNN's and
  cuBLAS's by name.
- **XLA's NVTX ranges** mark each run of a program (an `XlaModule` range named after
  the step, `XlaModule:#hlo_module=jit__train_step_on_shards,...`) and each op in it (a
  `Thunk` range with its `hlo_op`); cuBLAS and NCCL add ranges for their calls.
  jax.profiler's annotations are not NVTX ranges, so they do not show.
- **A traced step is nearly as fast**: 228 ms against 223 ms untraced.
- **The capture's start and stop take time**: about 23 s to start and 75 to 90 s to
  stop while nsys writes its report (about 20 MB). `cudaProfilerStop` returns once it
  is written, so `tokens_per_s` reads low at the two logs around the traced steps (55k
  and 17k, against 1.17M).
- **Signals reach the run through nsys.** SIGTERM to the launcher stops the run after
  its step, as without nsys, and the launcher exits with 143.

`nsys stats --report cuda_gpu_kern_sum <report>` lists the kernels by time, and
`nsys export --type sqlite <report>` writes the report as a database to query. The
three longest in that run, per GPU per step:

| Kernel | What it is | ms |
|--------|------------|----|
| `ncclDevKernel_AllGather_RING` | FSDP's all-gathers of the parameters | 33.8 |
| `cudnn_generated_fort_native_sdpa_...` | cuDNN's attention, one kernel per layer | 21.0 |
| `ncclDevKernel_ReduceScatter_bf16_RING` | FSDP's reduce-scatter of the gradients | 18.9 |

A collective's time includes its wait for the slowest GPU. cuBLAS's GEMMs
(`nvjet_sm103_...`) come in many shapes, each on its own line.

### Nsight Compute

On B300s, with Nsight Compute 2026.3 and jax 0.12 (2026-10-26):

- **ncu profiles one GPU per process.** It runs one kernel at a time, while a step's
  collectives need all their GPUs to run together: on 8 B300s, ncu hung at a step of
  `one_node.bux`'s `fsdp` run. So a launch fails a run of more GPUs under ncu at once
  (`profile: ncu profiles one GPU, not 8: give the run one`). Give it one with
  `mesh = mesh(data=1)` and its other axes at 1.
- **Each kernel is replayed**, once per group of its counters (10 for the default
  set), with the memory the kernel may write restored before each pass. A step of the
  smoke run on a B300, 174 kernels, took under two minutes.
- **Name the kernels.** A large model's step runs thousands of kernels:
  `options="--kernel-name=regex:nvjet --launch-count=4"` profiles the first 4 of
  cuBLAS's GEMMs (`nvjet_...`), and `steps=1` keeps it to one step.
- **The report** is compressed (`.ncu-repz`).
  `ncu --import <report> --print-summary per-kernel` prints each kernel's counters, and
  Nsight Compute opens it.

### Running them yourself

- **The tool must be on `PATH`.** The [Docker image](../get-started/installation.md#docker)
  has both. A launch fails a run whose tool it cannot find, at once:
  `smoke: failed before training: profile: nsys is not on PATH`.
- **`bux train` runs under no tool.** Run it under one yourself, with `BUX_PROFILER`
  set so that bux makes its changes, and the `profile` hook starts and stops the
  capture as it does under `bux launch`:
  `BUX_PROFILER=nsys nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop bux train run.bux`.
- **A command of your own.** With `BUX_PROFILER` set to the tool, `bux launch` starts
  no tool of its own:
  `BUX_PROFILER=nsys nsys profile --capture-range=cudaProfilerApi <flags> bux launch run.bux`.
- **Another tool** is an entry of {py:data}`~bux.train.profilers.TOOLS`, added by a
  script before it launches ({py:mod}`bux.train.profilers`).

(mfu-and-flops)=

## MFU and FLOPs

On a GPU whose peak bux knows, the log has `mfu`: the model's FLOPs per second over the
peak of the mesh's GPUs. The peaks are dense bfloat16, from NVIDIA's datasheets for the
SXM parts (`flops.PEAK`):

| GPU | Peak, dense bfloat16 |
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
- Each product once: what a backward pass computes again (jax.checkpoint's, the
  logits of a flash backward) does not count.
- With kernels off, since a kernel (a Pallas call, a cuDNN call) hides its FLOPs.

So a new layer is counted with no formula of its own.

### Attention counts the keys each query sees

Attention counts only the keys each query sees, as Megatron's MFU does and as a flash
kernel computes. Its products over a row's keys are counted separately (inside
`flops.attention`), and each step counts them at the keys its batch's queries see,
from the batch's positions (`flops.Work.of`, on the host): those of the query's own
document up to itself, within the window, and no more than a sparse attention's
top_k.

| Attention | Counts |
|-----------|--------|
| causal, rows of one document | (t + 1) / 2t of its t × t logits |
| causal, packed documents | each document's own triangle |
| a window | the keys within it |
| DSA | its top_k keys per query in the attention, the indexer's loss and its target, with the indexer's ranking at every key it may pick |
| CSA's latents | whole |

On many hosts, a step counts the keys of the rows its host reads. PaLM's MFU counts
attention whole, which would make bux's 10% higher at 4096 tokens per row.

### FLOPs per token in the plan

`bux plan` shows the FLOPs per token on rows of one document, with no GPU: 6.91B for a
Llama-style 1.3B at 4096 tokens per row, 6N for its matrices plus 0.8B of attention.
The one-GPU benchmark suite:

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

Multiply the FLOPs per token by `tokens_per_s` and divide by the peak of all the run's
GPUs: that is the MFU the log reports on rows of one document.
