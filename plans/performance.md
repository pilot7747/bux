# Performance: the plan past one node

On one node bux is near the state of the art: dense trains at 49% MFU on a B300,
MXFP8 gives 1.18x, and fsdp=8 scales at 97%. What is missing shows up across nodes,
in MoE, in TP and in pipelines. This plan closes those gaps in three waves.

## How to run it

Each work package (WP) runs as its own agent in its own worktree, branched from
`sota`. The WPs of a wave own disjoint files, and they merge into `sota` in the order
listed. A WP that needs a change in a file it doesn't own makes the smallest change
it can and says so in its report.

| Wave | WPs |
|------|-----|
| 1 | N nodes, O overlap, E experts, R remat and the loss, G gather once, P precision |
| 2 | Kernels: E2 grouped MXFP8 experts and NCCL-EP, P2 a fused NVFP4 quantize, O2 TP collective matmul, L linear-attention residuals and gate-free GDN/KDA |
| 3 | G2 pipeline schedules: 1F1B, zero-bubble, DualPipe-style |

## Rules

- **Readable first.** A student should be able to skim the implementation. Use short
  functions with names that say what they do, and keep each step on its own line.
  Comment why, never what. A clever trick goes behind a well-named function.
- **Measured, not guessed.** Every speed claim comes with tokens/s, MFU or peak memory
  from a B300, against the tree before the change. Pin GEMM autotuning off before you
  call a loss difference a regression.
- **The DSL is the interface.** A new knob is a field of a config in
  `bux/train/config.py`, or a method in a registry, documented in `docs/`.
- **Tests on 8 fake CPU devices**, against one device, as `docs/parallelism.md` does.

## WP N: nodes

**Owns:** `parallel/mesh.py`, `train/data.py`, `runtime.py`, `launch.py`, and
`parallel/layout.py` for the FSDP axes.

- A hybrid mesh when devices span more than one slice or process. Fill a node from the
  innermost axis (`tensor`) outward; the outer axes span nodes.
- A compile cache that every host reads. Today every host but the first recompiles,
  because only process 0 writes the cache, and each host keeps its own.
- Each process builds only its own rows, and no permutation of the whole corpus.
- FSDP over several axes: dense parameters and their state shard over `expert` and
  `context` too, not only `fsdp`.
- **Done when** a dense run trains on 2 nodes as FSDP and as HSDP (`data` across
  nodes), with its scaling against one node.

## WP O: overlap

**Owns:** `train/hooks.py` (the profile hook), `train/flops.py`, `__main__.py`
(`bux plan`), `parallel/ring.py`, and a setting for XLA's options, which the train
step's `jit` takes.

- XLA options as a run setting: a preset with `O1` (collective pipelining, double
  buffering and the latency hiding scheduler) and the windowed-einsum threshold for TP.
- An `exposed_comm_ms` metric from the profiler's trace, and the collective bytes per
  mesh axis in `bux plan`.
- MFU that counts only the attention a causal or document mask computes.
- Ring attention computes its local block while the K/V gather is in flight.
- **Done when** a sweep of the options on TP, EP and FSDP with microbatches says which
  one to make the default.

## WP E: experts

**Owns:** `parallel/experts.py`, `architecture/moe.py`, `architecture/grouped.py`.

- A benchmark at a real shape: E=64, top-8, 16-32k tokens per GPU, every EP method,
  and XLA's ragged all-to-all and ragged-dot flags.
- A ragged exchange bounded by a capacity and laid out by expert: each expert's rows
  arrive together, so the receiver doesn't sort again. Buffers are `lax.empty`, and
  a counter reports dropped tokens.
- A token goes once to each device it is routed to, not once per expert it picks.
- `remat=dots` saves the expert GEMMs' outputs.
- **Done when** EP fits at 32k tokens per GPU, and its speed is measured against
  `sorted` without EP.

## WP R: remat and the loss

**Owns:** `architecture/remat.py`, `architecture/logits.py`, `train/objective.py`,
the remat names in `attention.py`, `sparse.py` and `mla.py`.

- Fixes: MXFP8's GEMMs in a `shard_map` count as dots; sparse attention takes
  `remat.named`; MLA keeps its latents and recomputes the up-projections.
- A policy as data: what to save, what to offload, and which layers it applies to
  (`remat(save=[attention, mlp], first=4)`).
- The loss by chunks with eager gradients: chunk memory at full speed.
- **Done when** each policy's memory and speed are measured on the dense, GDN and MoE
  runs.

## WP G: gather once

**Owns:** `train/trainer.py`, `parallel/pipeline.py`.

- Microbatches gather and cast the parameters once a step, not once per microbatch,
  and reduce gradients once, at the end (`unreduced` shardings).
- A pipeline gathers its stage's parameters once a step, not once per tick.
- The dtype that gradients are reduced in, as a setting (`float32` like Megatron).
- **Done when** a run with 8 microbatches has the collectives of a run with 1, and
  trains to the same loss.

## WP P: precision

**Owns:** `architecture/matmuls.py`, `architecture/blockscaled.py`,
`PrecisionConfig`, `train/optim.py` (moment dtypes), `examples/recipes/`.

- Which matmuls are quantized: by role (experts, head, router, attention
  projections), by layer (the first and last N stay in compute), and by pass (fprop,
  dgrad, wgrad).
- Recipes as modules that `use` brings in: NVIDIA's MXFP8 and NVFP4, DeepSeek-V3's.
- The optimizer's moments in bfloat16; evaluation in compute; a late switch to
  compute for the last part of training.
- Keep `ir.Run.id` stable: a field at its default stays out of the id.
- **Done when** the NVFP4 recipe trains a dense run to the bf16 loss within the
  recipe's published gap.
