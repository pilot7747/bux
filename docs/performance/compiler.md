# XLA's options and bux's passes

The {bux}`compiler` setting says how XLA compiles a run's training step: a preset of
XLA's options, options of your own over it, and where the step's gradient reductions
start. This page covers the setting, its presets, the flags bux sets for every
process, the pass that starts reductions early, and {py:mod}`bux.passes`, the
registry of bux's passes over XLA's HLO.

Read it when a profile shows exposed communication
([Profiling](profiling.md#communication-and-idle-time)), when you train a
scanned model under FSDP at about a row a GPU, or when you want to try an XLA flag
on one run of a sweep.

:::{tip}
The defaults, XLA's own options (`preset=default`) and the reductions started early
(`reductions=early`), are the fastest measured, or within the noise of the fastest,
for every run below but one. The exception is a scanned model under FSDP at a row a
GPU, which trains up to 56% faster with `compiler(preset=maxtext)`
([below](#scanned-layers-under-fsdp)).
:::

## Usage

```bux
compiler = compiler(preset=o1)
compiler = compiler(options="--xla_gpu_threshold_for_windowed_einsum_mib=0")
compiler.preset = default | o1 | lhs | maxtext       # a sweep
compiler = compiler(reductions=xla)                  # where XLA puts them
```

| field        | what it is | default |
|--------------|------------|---------|
| `preset`     | a name in {py:mod}`bux.train.xla`'s `PRESETS`: `default`, `o1`, `lhs`, `maxtext`, `collective_matmul` | `default` |
| `options`    | options over the preset's, as XLA_FLAGS writes them: `"--name=value --name=value"` | `""` |
| `reductions` | where the gradients' reductions start: `early`, each as its gradient is computed, or `xla`, where XLA's scheduler puts them | `early` |

The training step compiles with its preset's options, then the setting's own `options`
over them:

- The options are XLA's debug options, and jax's `optimization_level` and
  `memory_fitting_level`.
- `true` and `false` become booleans, which XLA takes from a string only as `True` or
  `False`. Anything else goes as it is, a number or an enum's name.
- An option that is not `--name=value` is an error where it is written.
- The options are the step's alone (`bux/train/xla.py`), and go over XLA_FLAGS. A
  run's step compiles with its own options even in a process that compiled another
  run's with others.

Like `mesh`, `layout` and `methods`, `compiler` changes how a run computes, not what,
and is not in its id: a sweep over presets is several runs of one id, told apart by
their names.

## Presets

| preset              | what it sets                                                     |
|---------------------|------------------------------------------------------------------|
| `default`           | nothing: XLA's defaults                                           |
| `o1`                | `optimization_level=O1`: collective pipelining, double-buffered loops, and the latency hiding scheduler for any collective |
| `lhs`               | the latency hiding scheduler, for any collective (`xla_gpu_enable_latency_hiding_scheduler`) |
| `maxtext`           | MaxText's GPU recipe: the scheduler, combine thresholds of 128, 128 and 64 MiB (all-reduce, all-gather, reduce-scatter), pipelined collectives, double-buffered loops, a high-priority stream for collectives, Triton GEMMs, command buffers and XLA's rematerialization off |
| `collective_matmul` | XLA's windowed einsum (`xla_gpu_threshold_for_windowed_einsum_mib=0`): tensor parallelism's all-gathers and reduce-scatters as loops of collective-permutes beside their shares of the matmul |

Another preset is an entry of `bux.train.xla.PRESETS`, a dict of option names (without
the leading `--`) to values:

```python
from bux.train import xla

xla.PRESETS["mine"] = {"xla_gpu_enable_latency_hiding_scheduler": "true"}
```

and `compiler(preset=mine)` takes it in a script that registers it before it trains.

XLA (in jax 0.12 and at its head) runs the latency hiding scheduler on Hopper and
Blackwell by itself, but only on a step whose collectives its cost model knows: not
one with `ragged_all_to_all` (`methods(moe=ragged_all_to_all)`), which `lhs` and `o1`
schedule. Host offload does not stop it.

## The flags bux sets in every process

`bux train`, `bux launch` and a sweep's workers call {py:func}`bux.runtime.configure`
before anything starts a backend; a script calls it itself, before it touches a
device. It sets, where the environment does not:

| what | value | why |
|------|-------|-----|
| XLA_FLAGS | `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` | without it, NCCL's ragged all-to-all (expert parallelism's dispatch) hangs on jax 0.12 |
| `NCCL_GIN_TYPE`, `NCCL_EP_TOKENS_PER_CHUNK` | `2`, `256` | NCCL EP between nodes: GIN by a CPU proxy, 256 tokens a chunk. On two B300 nodes GDAKI, NCCL's default, took 3x as long for an EP layer's exchanges, and chunks of 64 1.3x ([One layer's exchange](../parallelism/experts.md#one-layers-exchange)) |
| `XLA_CLIENT_MEM_FRACTION` | `0.95` | XLA's memory pool ([Memory](tuning.md#batch-microbatches-and-memory)) |
| `JAX_COMPILATION_CACHE_DIR` | `~/.cache/bux/jax` | the compile cache ([Compile time](tuning.md#compile-time-and-the-compile-cache)) |

A flag that XLA_FLAGS sets already keeps its value.

The flags that usually speed up collectives and scheduling changed nothing past the
noise (1%) on the 1.3B on 1 and 8 GPUs, nor on the MoE on `mesh(fsdp=8)`: the
latency-hiding scheduler, pipelined all-gathers, all-reduces and reduce-scatters,
combine thresholds of 128 MiB, double-buffered while loops, command buffers off, Triton
GEMMs off. FSDP over 8 GPUs gets 97% of 8 times one GPU's tokens already. Other splits
wait longer for their collectives, and the presets try XLA's options on them
([Measurements](#measurements)).

(compiler-reductions)=

## Where the gradients' reductions start: `reductions`

XLA's scheduler, with one collective in flight, places:

- a collective's done just before its result is first read;
- each collective's start just above the done before it;
- what feeds only that start just above it.

A gradient's reduction is read where the step adds it up, after the last layer's
backward pass (a microbatch's sum, the optimizer's update). So XLA puts every weight
gradient's GEMM there, each just before its reduction, and runs the reductions one
after another beside nothing but those GEMMs. Qwen3-30B-A3B's `sorted` arm
(`megatron.bux`, Megatron's batch of 8 microbatches, `fsdp=8`) waited 95 ms a
microbatch, 11% of its step, for 96 reduce-scatters at the end of each backward pass.

`compiler(reductions=early)`, the default, runs a pass over XLA's schedule
({py:mod}`bux.train.reorder`):

1. It finds each reduction's start: a reduce-scatter or an all-reduce.
2. It moves the start up, with the ops that compute for it alone (its GEMM, its
   operands' quantizers, the bitcasts and converts between), to just after what they
   read.
3. It waits for the reduction just before the start of the second reduction after it
   (`reorder.LAG = 2`), so that two are in flight at once.

What communicates, copies to or from the host, or runs other computations by its own
order (a while loop, a conditional) never moves. `reductions=xla` leaves XLA's
schedule as it is.

The pass changes the order of a program's ops, not what they compute: with GEMM
autotuning off and XLA's deterministic ops, its loss after 12 steps is within 0.002 of
XLA's schedule's, where two runs of one tree without them end 0.02 apart.

### What it gained

On a node of 8 B300s, measured on two such nodes, the arm trains at 41.7k to 42.1k
tokens a second a GPU with `early`, against 39.7k to 40.0k with `xla`, in the same
214.5 GiB. Its
profiled step's exposed collectives drop from 1029 ms to 726 of 6.8 s. Most of what is
left is a microbatch's last reductions, which only its next microbatch could run
beside.

| run | gain from `early`, same memory |
|-----|--------------------------------|
| megatron.bux's `sorted` | 5% (40.0k to 42.1k tok/s a GPU) |
| megatron.bux's `nccl_ep` | 0.8% |
| one_node.bux's `fsdp` | 1.1% |
| one_node.bux's `fsdp_m8` (8 microbatches) | 6.2% |
| one_node.bux's `moe` | 0.4% |
| one_node.bux's `moe_ep`, `tp` | 1.7% |
| one_node.bux's `pp` | none |
| the one-GPU suite | none |

Before the experts' kernels quantized their weights before the gather, the arm gained
5.7%, from 37.3k to 39.4k, at lags 2, 3 and 4 alike, and 3% less at lag 1.

### What XLA's own flags did

XLA's own flags left the reductions where they were. On the same arm before the pass
(`remat=none`, tokens a second a GPU, on one node but for the last two rows):

| XLA's flags | tok/s a GPU |
|-------------|-------------|
| XLA's defaults | 37.1k |
| PGLE: JAX's, 3 profiled steps, the 85th percentile | 23.8k |
| two collectives in flight (`--xla_gpu_experimental_parallel_collective_overlap_limit=2`) | 37.2k |
| the same, each started as early as it can (`--xla_gpu_experimental_collective_start_as_early_as_possible=true`) | 37.2k |
| 16 NCCL channels a collective, where XLA takes 32 of a B300's 148 SMs (`--xla_gpu_nccl_collective_max_nchannels=16`) | 35.7k |
| NCCL's symmetric buffers, or its registered ones | fail to compile: cuDNN's MXFP8 GEMMs then miss their graphs |

With 16 channels the GEMMs beside the collectives ran 3% faster and the collectives
took half again as long; with the pass, 16 and 24 channels were 1.6-6.5% slower too.

## bux's passes: `bux.passes`

The reductions' pass is an entry of `bux.passes.PASSES`, bux's passes over XLA's HLO.
Each is a function of a serialized `HloModuleProto` that returns it changed, or `None`
to leave it, run at a stage of XLA's pipeline: before its scheduler
(`passes.BEFORE_SCHEDULING`) or after it (`passes.AFTER_SCHEDULING`).

```python
from bux import passes


@passes.register("my_pass", stage=passes.AFTER_SCHEDULING, key="v1")
def changed(serialized: bytes) -> bytes | None: ...


passes.use("my_pass")  # every program the process compiles for GPUs from now on
passes.use("my_pass", on=False)  # and no longer
```

- {py:func}`bux.passes.use` turns a pass on for every program the process compiles for
  GPUs from then on; on other backends it does nothing.
- JAX's cache of compiled programs is keyed by the passes in use and by each one's
  `key`, which should change with what the pass does
  ({py:func}`bux.passes.cache_key`).
- A pass that raises leaves the module as it was, with a warning: it runs inside XLA's
  compile, which it must not fail.

| pass | stage | turned on by |
|------|-------|--------------|
| `bux_early_reductions` ({py:mod}`bux.train.reorder`) | after scheduling | `compiler(reductions=early)`, the default |
| `nccl_ep_parts_off_the_stream` | before scheduling | `methods(moe=nccl_ep, moe_stream=true)` ([MoE methods](tuning.md#moe-methods)) |

(compiler-measurements)=

## Measurements

All on B300s, bfloat16, in October 2026.

### The presets on one node

On one node of 8 B300s, rows of 4096 tokens but for the ring's, tokens a second, and
the exposed communication of a step in ms of its time (the log's `exposed_comm_ms`,
profiled):

| run | `default` | `o1` | `lhs` | `maxtext` |
|-----|-----------|------|-------|-----------|
| TP=8 and sequence parallelism: 4 of Llama 3 70B's layers, scanned, `remat=full`, 8 rows | **218k**, 34 of 150 | 211k, 31 | 218k, 34 | 216k, 35 |
| EP=8 by `all_to_all`: 8 layers of 64 experts, top 8, d = 2048, a row a GPU | 158k, 62 of 208 | 157k | 155k | 159k |
| EP=8 by `ragged_all_to_all`, as above | 45k, 618 of 733 | 45k | 45k | 45k |
| the 1.3B, `fsdp=8`, 8 rows a GPU | **1.14M**, 3.5 of 230 | 1.13M | | 1.09M, 9.0 |
| the same in 8 microbatches | **842k**, 57 of 311 | 830k, 64 | 836k | 789k, 74 |
| the same scanned (`repeat=scan`), a row a GPU | 662k, 11 of 50 | 682k, 8 | 661k | **718k**, 10 |
| `ring` over `context=8`, rows of 32k, one document, the 218M of [context parallelism](../parallelism/attention.md) | **183k**, 137 of 1436 | 167k | 182k | |
| the same on packed documents | 223k, 128 of 1177 | 210k | **225k**, 23 | |

- **TP.** `collective_matmul` takes TP to 93k: its 325 collective-permutes a step,
  0.84 ms each, take 274 ms (XLA's windowed einsum; over memcpy,
  `--xla_gpu_use_memcpy_local_p2p`, the same). TP waits for its reduce-scatters (17 ms
  a step) and all-gathers (15 ms) at the boundaries of the matmuls they feed, which no
  schedule moves. `methods(tensor=chunked)` moves them beside the matmuls, a chunk at
  a time, and hides a quarter of the wait, which the matmuls beside give back
  ([Tensor parallelism](../parallelism/tensor.md)).
- **EP.** `ragged_all_to_all` waits 610 ms a step for its 96 exchanges (of a slice a
  peer, before they went by expert and in pieces:
  [Experts](../parallelism/experts.md)). With 4 rows a GPU its step runs out of memory
  under `default` (a temporary of 97 GiB) and fits under the others, at 55k against
  `all_to_all`'s 176k.
- **FSDP.** Its collectives take 14% of the step and hide all but 1.5% of it.
- **Uneven loads.** A collective's time includes the wait for its slowest peer, so on
  unevenly loaded devices, such as the ring's on packed rows, `exposed_comm_ms` swings
  between runs of one speed.

### Scanned layers under FSDP

Only scanned layers gain, under FSDP at a row a GPU: `maxtext` trains the scanned
1.3B at 719k (three runs, against 656-662k), faster than the same unrolled (623k, and
572k under `maxtext`). Of its options, the pipelining and double buffering of `o1`
give 3%, and cuBLAS's GEMMs where XLA would take Triton's inside the loop
(`--xla_gpu_enable_triton_gemm=false`) 5%; the combine thresholds, XLA's
rematerialization and command buffers nothing. With TP the scanned layers gain
nothing.

Scanned with `maxtext` against the defaults, unrolled with XLA's options, by the
benchmarks' FSDP runs (`one_node.bux`, `two_nodes.bux`, [Benchmarks](benchmarks.md)),
20 steps, by rows of 4096 a GPU:

| run | rows a GPU | `default`, unrolled | `maxtext`, scanned |
|-----|------------|---------------------|--------------------|
| the 1.3B, `fsdp=8` | 1 | 443k | 691k (+56%) |
| | 2 | 909k | 870k (−4.3%) |
| | 4 | 1.03M | 994k (−3.9%) |
| | 8 | 1.11M | 1.08M (−3.4%) |
| the MoE, `fsdp=8` | 8 | 815k | 713k (−12.5%) |
| the 1.3B, `fsdp=16`, two nodes | 1 | 699k | 1.03M (+48%) |
| | 8 | 2.18M | 2.12M (−2.8%) |

On one GPU, the one-GPU suite at 8 rows, scanned with `maxtext` is slower for every
model: the dense one by 6%, in MXFP8 and NVFP4 by 7-8%, the GDN hybrid by 7%, the
Mamba-2 and KDA hybrids by 11%, bounded KDA by 23% and the MoE by 17%; and over
`fsdp=8` the GDN hybrid by 23%.

At a row a GPU an unrolled step waits for FSDP's gathers, and takes as long as one of
two rows; the scan's loop, its collectives pipelined and its buffers doubled, hides
them. From two rows a GPU the compute hides them anyway, and what the scan copies in
and out of its stacks costs more than the loop saves. So the defaults stay: unrolled,
with XLA's options. FSDP at a row a GPU, as a model too large for more trains, is
worth `methods(repeat=scan)` with `compiler(preset=maxtext)`:

```bux
methods  = methods(repeat=scan)
compiler = compiler(preset=maxtext)
```
