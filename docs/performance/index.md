# Performance

How fast bux trains on GPUs, how to make a run faster, and how to measure it. bux's
defaults are tuned on B300s: most runs need no setting from this section to train at
the speeds below. Come here when a run is slower than you expect, runs out of memory,
or when you change bux itself and need to show the change is not slower.

## Headline numbers

All on NVIDIA B300s, in October 2026, at each run's defaults but the precision a row
names ([Benchmarks](benchmarks.md) has the workloads, dates and commits):

| run | bux | against |
|-----|-----|---------|
| Qwen3-30B-A3B at Megatron-Bridge's own workload (512 rows of 4096 in 8 microbatches, MXFP8), 8 GPUs | **42,391** tokens a second a GPU | Megatron-Bridge's 40,769: **104%** |
| Megatron-Bridge's other workloads: Nemotron 3 Nano (8 GPUs), gpt-oss-120b and Nemotron 3 Super (64), Llama 3 8B (8) | 86%, 89%, 62-68% of Megatron-Bridge's | Llama 3 8B: 111% of its B200 number ([details](benchmarks.md#megatron-bridges-other-workloads)) |
| a Llama-style 1.3B, one GPU, bfloat16 | 142k tokens a second, 44% MFU | 167k in MXFP8 (1.18x), 179k in NVFP4 (1.22x against 147k, a later tree's bfloat16) |
| the same 1.3B, `mesh(fsdp=8)` on one node | 1.245M tokens a second | FSDP keeps 97% of a GPU's rate alone |
| the same 1.3B, HSDP on eight nodes (64 GPUs) | 8,664k tokens a second | 96.4% a GPU of one node's |
| attention, and the MoE's experts, by bux's kernels | 7x and 27x | XLA's own program |

## What is here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Tuning guide
:link: tuning
:link-type: doc

The workflow for a faster run, then each lever in the order to try it: kernels,
precision, batch and memory, remat, the mesh, MoE methods, overlap, the host and
compiles. Ends with a troubleshooting table.
:::

:::{grid-item-card} {octicon}`cpu` GPU kernels
:link: kernels
:link-type: doc

Every kernel bux runs on a GPU, what it is against XLA, when it falls back, and how
to turn them all off to debug one.
:::

:::{grid-item-card} {octicon}`gear` XLA's options and bux's passes
:link: compiler
:link-type: doc

The `compiler` setting: XLA's presets, options of your own, and the pass that starts
each gradient's reduction early.
:::

:::{grid-item-card} {octicon}`pulse` Profiling
:link: profiling
:link-type: doc

Traces, `comm_ms`, `exposed_comm_ms` and `idle_ms`, how bux counts FLOPs for MFU, and
how to find what a step waits for.
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
  measurements. The [tuning guide](tuning.md#choosing-a-mesh) says which to take.
- [Precision](../training/precision.md): the formats a run multiplies in, and what
  they do to its loss.
- [Running](../training/running.md): the log's metrics, `tokens_per_s` and `mfu`
  among them.

```{toctree}
:hidden:
:maxdepth: 2

tuning
kernels
compiler
profiling
benchmarks
```
