# Compiler options and passes

The {bux}`compiler` setting says how XLA compiles a run's training step: a preset of
XLA's options, options of your own on top, and where the step's gradient reductions
start. You rarely need to touch it. The defaults are the fastest measured, or within
noise of it, for every run on this page except one: a scanned model under FSDP at one
row per GPU, which trains up to 56% faster with `compiler(preset=maxtext)`
([Scanned layers under FSDP](#scanned-layers-under-fsdp)).

Come here when a profile shows exposed communication
([Profiling](profiling.md#communication-and-idle-time)), when you train a scanned
model under FSDP at about one row per GPU, or when you want to try an XLA flag on one
run of a sweep. The page also covers the flags bux sets in every process, and
{py:mod}`bux.passes`, the registry of bux's own passes over XLA's HLO.

## Usage

```bux
compiler = compiler(preset=o1)
compiler = compiler(options="--xla_gpu_threshold_for_windowed_einsum_mib=0")
compiler.preset = default | o1 | lhs | maxtext       # a sweep
compiler = compiler(reductions=xla)                  # where XLA puts them
```

| Field | What it is | Default |
|-------|------------|---------|
| `preset` | a name in {py:mod}`bux.train.xla`'s `PRESETS`: `default`, `o1`, `lhs`, `maxtext`, `collective_matmul` | `default` |
| `options` | options on top of the preset's, written as XLA_FLAGS writes them: `"--name=value --name=value"` | `""` |
| `reductions` | where the gradients' reductions start: `early`, each as soon as its gradient is computed, or `xla`, where XLA's scheduler puts them | `early` |

The training step compiles with its preset's options, then the setting's own `options`
on top:

- The options are XLA's debug options, plus jax's `optimization_level` and
  `memory_fitting_level`.
- `true` and `false` become booleans, since XLA takes a boolean from a string only as
  `True` or `False`. Anything else passes through as written: a number, or an enum's
  name.
- An option that is not `--name=value` is an error at the line that wrote it.
- The options apply to the step alone (`bux/train/xla.py`), on top of XLA_FLAGS. A run's
  step compiles with its own options even in a process that compiled another run's
  step with others.

Like `mesh`, `layout` and `methods`, `compiler` changes how a run computes, not what,
so it is not part of the run's id: a sweep over presets is several runs with one id,
told apart by their names.

## Presets

| Preset | What it sets |
|--------|--------------|
| `default` | nothing: XLA's defaults |
| `o1` | `optimization_level=O1`: collective pipelining, double-buffered loops, and the latency hiding scheduler for any collective |
| `lhs` | the latency hiding scheduler, for any collective (`xla_gpu_enable_latency_hiding_scheduler`) |
| `maxtext` | MaxText's GPU recipe: the scheduler; combine thresholds of 128, 128 and 64 MiB (all-reduce, all-gather, reduce-scatter); pipelined collectives; double-buffered loops; a high-priority stream for collectives; Triton GEMMs, command buffers and XLA's rematerialization off |
| `collective_matmul` | XLA's windowed einsum (`xla_gpu_threshold_for_windowed_einsum_mib=0`): tensor parallelism's all-gathers and reduce-scatters as loops of collective-permutes beside their shares of the matmul |

XLA runs the latency hiding scheduler on Hopper and Blackwell by itself, but only on a
step whose collectives its cost model knows. `ragged_all_to_all`
(`methods(moe=ragged_all_to_all)`) is not one of them: `lhs` and `o1` schedule it.

To add a preset, add an entry to `bux.train.xla.PRESETS`, a dict of option names
(without the leading `--`) to values, in a script before it trains:

```python
from bux.train import xla

xla.PRESETS["mine"] = {"xla_gpu_enable_latency_hiding_scheduler": "true"}
```

Then `compiler(preset=mine)` takes it.

(compiler-reductions)=

## Early gradient reductions: `reductions`

XLA's scheduler starts each collective just before its result is first read. A
gradient's reduction is read where the step adds the gradients up, after the last
layer's backward pass. So XLA puts every weight gradient's GEMM and reduction there,
and runs the reductions one after another with nothing but those GEMMs beside them.
Qwen3-30B-A3B's `sorted` run (`megatron.bux`: Megatron's batch in 8 microbatches,
`fsdp=8`) waited 95 ms per microbatch, 11% of its step, for 96 reduce-scatters at the
end of each backward pass.

`compiler(reductions=early)`, the default, moves each reduction up to where its
gradient is computed, so it runs beside the next layer's backward pass.
`reductions=xla` leaves XLA's schedule as it is.

| Run | Gain from `early`, in the same memory |
|-----|---------------------------------------|
| `megatron.bux`'s `sorted` | 5% (40.0k to 42.1k tok/s/GPU) |
| `megatron.bux`'s `nccl_ep` | 0.8% |
| `one_node.bux`'s `fsdp` | 1.1% |
| `one_node.bux`'s `fsdp_m8` (8 microbatches) | 6.2% |
| `one_node.bux`'s `moe` | 0.4% |
| `one_node.bux`'s `moe_ep`, `tp` | 1.7% |
| `one_node.bux`'s `pp` | none |
| the one-GPU suite | none |

Measured on 8 B300s, on two such nodes, the `sorted` run trains at 41.7k to 42.1k
tok/s/GPU with `early` and 39.7k to 40.0k with `xla`, in the same 214.5 GiB. The
exposed collectives of its profiled step drop from 1029 ms to 726 of 6.8 s. Most of
what remains is each microbatch's last reductions, which only the next microbatch could
run beside.

The pass changes the order of a program's ops, not what they compute. With GEMM
autotuning off and XLA's deterministic ops, the loss after 12 steps is within 0.002 of
XLA's schedule's, where two runs of one tree without them end 0.02 apart.

### How it works

The pass ({py:mod}`bux.train.reorder`) runs over XLA's schedule:

1. It finds each reduction's start: a reduce-scatter or an all-reduce.
2. It moves the start up, together with the ops that compute for it alone (its GEMM,
   its operands' quantizers, the bitcasts and converts between), to just after what
   they read.
3. It waits for each reduction just before the start of the second reduction after it
   (`reorder.LAG = 2`), so that two are in flight at once. A lag of 1 gained 3% less;
   lags of 2 to 4 gained the same.

Nothing that communicates, copies to or from the host, or runs other computations in
its own order (a while loop, a conditional) ever moves.

## The flags bux sets in every process

`bux train`, `bux launch` and a sweep's workers call {py:func}`bux.runtime.configure`
before anything starts a backend. A script that trains must call it itself, before it
touches a device. It sets, wherever the environment does not:

| What | Value | Why |
|------|-------|-----|
| XLA_FLAGS | `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` | without it, NCCL's ragged all-to-all (expert parallelism's dispatch) hangs on jax 0.12 |
| `NCCL_GIN_TYPE`, `NCCL_EP_TOKENS_PER_CHUNK` | `2`, `256` | NCCL EP between nodes: GIN by a CPU proxy, 256 tokens per chunk. On two B300 nodes, GDAKI (NCCL's default) took 3x as long for an EP layer's exchanges, and chunks of 64 took 1.3x ([One layer's exchange](../parallelism/experts.md#one-layers-exchange)) |
| `XLA_CLIENT_MEM_FRACTION` | `0.95` | XLA's memory pool ([Memory](tuning.md#batch-microbatches-and-memory)) |
| `JAX_COMPILATION_CACHE_DIR` | `~/.cache/bux/jax` | the compile cache ([Compile time](tuning.md#compile-time-and-the-compile-cache)) |

A flag that XLA_FLAGS already sets keeps its value.

## bux's passes: `bux.passes`

The early reductions are an entry of `bux.passes.PASSES`, bux's registry of passes over
XLA's HLO. Each pass is a function of a serialized `HloModuleProto` that returns it
changed, or `None` to leave it alone, and runs at a stage of XLA's pipeline: before its
scheduler (`passes.BEFORE_SCHEDULING`) or after it (`passes.AFTER_SCHEDULING`).

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
  `key`, which should change whenever what the pass does changes
  ({py:func}`bux.passes.cache_key`).
- A pass that raises leaves the module as it was, with a warning: it runs inside XLA's
  compile, which it must not fail.

| Pass | Stage | Turned on by |
|------|-------|--------------|
| `bux_early_reductions` ({py:mod}`bux.train.reorder`) | after scheduling | `compiler(reductions=early)`, the default |
| `nccl_ep_parts_off_the_stream` | before scheduling | `methods(moe=nccl_ep, moe_stream=true)` ([MoE methods](tuning.md#moe-methods)) |

(compiler-measurements)=

## Measurements

All on B300s, in bfloat16, in October 2026.

### The presets on one node

On one node of 8 B300s, with rows of 4096 tokens except the ring's: tokens/s, and the
exposed communication of a profiled step in ms against the step's time (the log's
`exposed_comm_ms`):

| Run | `default` | `o1` | `lhs` | `maxtext` |
|-----|-----------|------|-------|-----------|
| TP=8 with sequence parallelism: 4 of Llama 3 70B's layers, scanned, `remat=full`, 8 rows | **218k**, 34 of 150 | 211k, 31 | 218k, 34 | 216k, 35 |
| EP=8 by `all_to_all`: 8 layers of 64 experts, top 8, d = 2048, 1 row per GPU | 158k, 62 of 208 | 157k | 155k | 159k |
| EP=8 by `ragged_all_to_all`, as above | 45k, 618 of 733 | 45k | 45k | 45k |
| the 1.3B, `fsdp=8`, 8 rows per GPU | **1.14M**, 3.5 of 230 | 1.13M | | 1.09M, 9.0 |
| the same in 8 microbatches | **842k**, 57 of 311 | 830k, 64 | 836k | 789k, 74 |
| the same scanned (`repeat=scan`), 1 row per GPU | 662k, 11 of 50 | 682k, 8 | 661k | **718k**, 10 |
| `ring` over `context=8`, rows of 32k of one document, the 218M of [context parallelism](../parallelism/attention.md) | **183k**, 137 of 1436 | 167k | 182k | |
| the same on packed documents | 223k, 128 of 1177 | 210k | **225k**, 23 | |

- **TP** waits for its reduce-scatters (17 ms per step) and all-gathers (15 ms) at the
  boundaries of the matmuls they feed, which no schedule moves.
  `methods(tensor=chunked)` moves them beside the matmuls, one chunk at a time, and
  hides a quarter of the wait, which the matmuls beside them give back
  ([Tensor parallelism](../parallelism/tensor.md)).
- **EP.** `ragged_all_to_all` waits 610 ms per step for its 96 exchanges
  ([Experts](../parallelism/experts.md)). With 4 rows per GPU its step runs out of
  memory under `default` (a temporary of 97 GiB) and fits under the others, at 55k
  against 176k for `all_to_all`.
- **FSDP.** Its collectives take 14% of the step and hide all but 1.5% of it.
- **Uneven loads.** A collective's time includes the wait for its slowest peer, so on
  unevenly loaded devices, such as the ring's on packed rows, `exposed_comm_ms` swings
  between runs of the same speed.

### Scanned layers under FSDP

FSDP at one row per GPU, as a model too large for more rows trains, is worth scanned
layers with MaxText's preset:

```bux
methods  = methods(repeat=scan)
compiler = compiler(preset=maxtext)
```

At one row per GPU, an unrolled step waits for FSDP's gathers and takes as long as a
step of two rows. The scan's loop, with its collectives pipelined and its buffers
doubled, hides them. From two rows per GPU the compute hides them anyway, and what the
scan copies in and out of its stacks costs more than the loop saves. So the defaults
stay unrolled, with XLA's options.

The benchmarks' FSDP runs (`one_node.bux`, `two_nodes.bux`, [Benchmarks](benchmarks.md)),
20 steps, by rows of 4096 per GPU:

| Run | Rows per GPU | `default`, unrolled | `maxtext`, scanned |
|-----|--------------|---------------------|--------------------|
| the 1.3B, `fsdp=8` | 1 | 443k tok/s | 691k (+56%) |
| | 2 | 909k | 870k (−4.3%) |
| | 4 | 1.03M | 994k (−3.9%) |
| | 8 | 1.11M | 1.08M (−3.4%) |
| the MoE, `fsdp=8` | 8 | 815k | 713k (−12.5%) |
| the 1.3B, `fsdp=16`, two nodes | 1 | 699k | 1.03M (+48%) |
| | 8 | 2.18M | 2.12M (−2.8%) |

Only scanned layers gain. Under FSDP at one row per GPU, `maxtext` trains the scanned
1.3B at 719k tokens/s (three runs, against 656-662k under `default`), faster than the
same model unrolled (623k, and 572k under `maxtext`). Of its options, `o1`'s
pipelining and double buffering give 3%, and cuBLAS's GEMMs inside the loop where XLA
would take Triton's (`--xla_gpu_enable_triton_gemm=false`) give 5%. The combine
thresholds, XLA's rematerialization and command buffers give nothing. With TP, scanned
layers gain nothing.

On one GPU, at the one-GPU suite's 8 rows, scanned with `maxtext` is slower for every
model: the dense one by 6%, in MXFP8 and NVFP4 by 7-8%, the GDN hybrid by 7%, the
Mamba-2 and KDA hybrids by 11%, bounded KDA by 23% and the MoE by 17%. Over `fsdp=8`,
the GDN hybrid is 23% slower.

### What does not help

- **XLA's usual collective and scheduling flags.** On the 1.3B on 1 and 8 GPUs, and on
  the MoE on `mesh(fsdp=8)`, none moved the speed past noise (1%): the latency hiding
  scheduler, pipelined all-gathers, all-reduces and reduce-scatters, combine thresholds
  of 128 MiB, double-buffered while loops, command buffers off, Triton GEMMs off. FSDP
  over 8 GPUs already gets 97% of eight times one GPU's tokens.
- **`collective_matmul` for TP**: 93k tokens/s against 218k. Its 325
  collective-permutes per step, 0.84 ms each, take 274 ms.
- **PGLE** (JAX's, 3 profiled steps, the 85th percentile): Qwen3-30B-A3B's `sorted` run
  fell from 37.1k to 23.8k tok/s/GPU.
- **Two collectives in flight**
  (`--xla_gpu_experimental_parallel_collective_overlap_limit=2`), with or without
  `--xla_gpu_experimental_collective_start_as_early_as_possible=true`: 37.2k, against
  37.1k. Neither moves the reductions; the `reductions` pass does.
- **Fewer NCCL channels.** XLA takes 32 of a B300's 148 SMs per collective. With 16
  (`--xla_gpu_nccl_collective_max_nchannels=16`), the GEMMs beside the collectives ran
  3% faster but the collectives took half again as long: 35.7k. With the `reductions`
  pass, 16 and 24 channels were 1.6-6.5% slower too.
- **NCCL's symmetric or registered buffers** fail to compile: cuDNN's MXFP8 GEMMs then
  miss their graphs.
