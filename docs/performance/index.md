# Performance

This section shows how fast bux trains on GPUs, how to make a slow run faster or an
out-of-memory run fit, and how to prove that a change to bux is not slower. Most runs
need none of it: the defaults are tuned on B300s, and the numbers below use them for
everything their workloads do not set.

## Headline numbers

All on NVIDIA B300s (nodes of 8, InfiniBand between nodes), in October 2026.
[Benchmarks](benchmarks.md) has each workload, its exact date and how to reproduce it.

**Against Megatron-Bridge**, at the workloads of its published B300 performance
summary (26.06): its global batch, microbatches and row length, every layer in MXFP8.

| Model | GPUs | bux, tok/s/GPU | Megatron-Bridge, tok/s/GPU | bux / Megatron-Bridge |
|-------|------|----------------|----------------------------|-----------------------|
| Qwen3-30B-A3B | 8 | **42,391** | 40,769 | **104%** |
| Llama 3 8B | 8 | 32,749 | 29,521 (B200) | 111% |
| gpt-oss-120b | 64 | 16,581 | 18,534 | 89% |
| Nemotron 3 Nano 30B-A3B | 8 | 31,912 | 37,316 | 86% |
| Nemotron 3 Super 120B-A12B | 64 | 5,317 | 7,867 | 68% |

Megatron-Bridge publishes no B300 number for Llama 3 8B, so its B200 number stands in:
the two GPUs have the same dense FP8 peak.

**A Llama-style 1.3B**, rows of 4096 tokens, 8 rows per GPU:

| Setup | GPUs | Precision | tok/s/GPU | Notes |
|-------|------|-----------|-----------|-------|
| One GPU | 1 | bfloat16 | 142.6k | 44% MFU |
| One GPU | 1 | MXFP8 | 164.8k | 51% MFU |
| One GPU | 1 | NVFP4 | 181.8k | 56% MFU |
| FSDP in a node, `mesh(fsdp=8)` | 8 | bfloat16 | 141k | 1.13M tok/s in all |
| HSDP over 8 nodes, `mesh(data=8, fsdp=8)` | 64 | bfloat16 | 135.4k | 96.4% of one node's rate per GPU |

All five rows are from 2026-10-24, so they compare with each other; changes merged
the next day made them faster still (FSDP in a node: 1.245M tok/s,
[One node](benchmarks.md#one-node)). MFU is against bfloat16's peak in every row, so
the low-precision rows can read above the bfloat16 one. Behind these numbers are bux's GPU kernels: attention runs 7x and an
MoE's experts 27x as fast as XLA's own program for the same computation.

## What is here

The pages in the order you are likely to need them:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Tuning guide
:link: tuning
:link-type: doc

Start here when a run is slow or out of memory. A workflow for measuring a change,
then each lever in the order to try it, and a troubleshooting table.
:::

:::{grid-item-card} {octicon}`pulse` Profiling
:link: profiling
:link-type: doc

Traces by jax.profiler, Nsight Systems or Nsight Compute; `comm_ms`,
`exposed_comm_ms` and `idle_ms`; how MFU counts FLOPs; and how to find what a step
waits for.
:::

:::{grid-item-card} {octicon}`cpu` GPU kernels
:link: kernels
:link-type: doc

Every kernel bux runs on a GPU, its speedup over XLA, when it falls back, and how to
turn them all off to debug one.
:::

:::{grid-item-card} {octicon}`gear` Compiler options
:link: compiler
:link-type: doc

The `compiler` setting: XLA's presets, flags of your own, and the pass that starts
each gradient's reduction early.
:::

:::{grid-item-card} {octicon}`graph` Benchmarks
:link: benchmarks
:link-type: doc

The suites from one GPU to eight nodes, how to compare two trees with
`scripts/benchmark.py`, and every result, against Megatron-Bridge too.
:::
::::

## Where else to look

- [Parallelism](../parallelism/index.md): each way a run splits over GPUs, with its
  measurements. The [tuning guide](tuning.md#choosing-a-mesh) says which to pick.
- [Precision](../training/precision.md): the formats a run multiplies in, and what
  they do to its loss.
- [Running](../training/running.md): the metrics in the log, `tokens_per_s` and `mfu`
  among them.

```{toctree}
:hidden:
:maxdepth: 2

tuning
profiling
kernels
compiler
benchmarks
```
