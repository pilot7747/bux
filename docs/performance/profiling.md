# Profiling

This page covers how to see where a training step's time goes: the {bux}`profile`
setting, the trace it writes, the three numbers bux reads from that trace
(`comm_ms`, `exposed_comm_ms` and `idle_ms`), how bux counts a model's FLOPs for
`mfu`, and how to tell from all of these what a step waits for.

Profile a run when its `tokens_per_s` or `mfu` is lower than you expect, before you
change anything ([the tuning workflow](tuning.md#the-workflow)).

## Usage

```bux
profile = profile(start=20, steps=5)
```

| field   | what it is | default |
|---------|------------|---------|
| `start` | the step after which the trace starts | `10` |
| `steps` | how many steps it traces | `5` |

The run traces the 5 steps after step 20 with `jax.profiler` into `profile/` in its
directory. It waits for the step before it starts and before it stops, so the trace
holds whole steps. Profiling at other steps keeps the run's id.

Open the trace in xprof, or in TensorBoard's profile plugin:

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
- **Under the profiler, a launch returns at its step's end.** XLA's launch of a
  step returns only when the GPU is nearly through it. A small model's step, its
  kernels in CUDA graphs (XLA's command buffers), returns in 10 ms, and the host runs
  up to two steps ahead; under the profiler, which turns command buffers off, its
  launch too returns at its step's end.
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

`bux plan` shows the FLOPs a token on rows of one document, with no GPU: 6.9G for a
Llama-style 1.3B at 4096 tokens a row, 6 N of its matrices and 0.8G of attention. The
one-GPU benchmark suite:

```console
$ bux plan examples/benchmarks/one_gpu.bux
8 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
41870be720  1.28B   983k    0.0        6.91B      1     4.77GiB     9.54GiB    0%      one_gpu[dense]
a045f7c793  1.38B   983k    0.0        7.07B      1     5.15GiB     10.3GiB    0%      one_gpu[gdn]
6e5cb35f5a  1.39B   983k    0.0        7.11B      1     5.17GiB     10.3GiB    0%      one_gpu[kda]
23163abafa  1.39B   983k    0.0        7.11B      1     5.17GiB     10.3GiB    0%      one_gpu[kdab]
3371d4b4f4  5.16B   983k    0.0        6.92B      1     19.2GiB     38.4GiB    0%      one_gpu[moe]
936c7c9d30  1.48B   983k    0.0        7.66B      1     5.53GiB     11.1GiB    0%      one_gpu[mamba2]
7b9acf4aa2  1.28B   983k    0.0        6.91B      1     4.77GiB     9.54GiB    0%      one_gpu[mxfp8]
4f66d8a674  1.28B   983k    0.0        6.91B      1     4.77GiB     9.54GiB    0%      one_gpu[nvfp4]
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
