# GPU kernels

What XLA compiles slowly on a GPU, bux runs by a kernel there: cuDNN's, tokamax's,
cuDNN-frontend's, or a Pallas kernel of its own. On a CPU the same computation runs by
XLA. This page lists the kernels, says when each one falls back to XLA, and shows how
to turn them all off to debug one.

You need nothing to turn them on: every kernel runs by default on a GPU it supports.
None of them is in a run's id: with or without them a run computes the same values, up
to rounding, and keeps its id.

## Usage

Every kernel asks {py:func}`bux.kernels.gpu` whether it may run, and falls back to XLA
by itself where it cannot. To run a whole training with no kernel at all, set
`BUX_KERNELS=off`:

```sh
BUX_KERNELS=off bux train examples/experiments/smoke.bux
```

In Python, trace inside {py:func}`bux.kernels.off`:

```python
from bux import kernels

with kernels.off():
    loss = step(model, batch)  # every layer traces by XLA here
```

Turn kernels off to find out whether a kernel is the cause of a wrong value or a crash,
or to count the FLOPs a kernel hides: bux's own FLOPs count traces the step with
kernels off ([Profiling](profiling.md#mfu-and-flops)).

:::{note}
A Pallas kernel called off a GPU, as the tests call them, runs interpreted, by XLA
({py:func}`bux.kernels.interpreted`). On a mesh, which XLA cannot split a kernel over,
{py:func}`bux.kernels.per_device` runs a kernel on each device's part of its inputs.
:::

(kernels-table)=

## The kernels

The speedups are against XLA's own program for the same computation, on a B300.

| what | by | on a B300, against XLA |
|------|----|------------------------|
| softmax attention | `methods.attention = auto`, the default: cuDNN's causal kernels in rows of one document, cuDNN's fused attention in rows of 8192 or more and with a window, else tokamax's flash attention (sinks, a soft cap) | 7x, and no t × t logits |
| QK-norm then RoPE | `norm_rope`: a Pallas kernel a pass, forward and backward, for an RMSNorm with a scale and RoPE by halves (Qwen3's, OLMo 2's) | 1.4x forward, 1.9x backward (Qwen3-30B-A3B's queries in megatron.bux's step: 0.12 and 0.24 ms a layer) |
| sparse attention (DSA, QSA, CSA) | `sparse`: tokamax's flash attention with the kept keys as a mask; the indexer's scores, its loss's picks and target by Pallas kernels (`scoring`), its top-k by counting | 2.6x a chunk of queries at a time, and no t × t floats |
| linear attention, RWKV-7 | `chunkwise`: chunks of 64 positions, the state across them by a Pallas kernel (`affine_scan`); `methods(states=recompute)` keeps a fourteenth of the memory for the backward | 14-17x a scan over positions |
| Mamba-2 | `chunkwise`, B and C a group's, the state across chunks by a matmul; `methods(states=recompute)` keeps a sixteenth of the memory for the backward | 25x a scan over positions (2 × 4096 tokens, rematerialized: at 8 × 4096 the scan needs 199 GiB) |
| a short conv's backward: `conv`, `short_conv`, Mamba's, Mamba-2's, and the q, k and v of GDN, KDA and DeltaNet | `pallas_conv` for q, k and v, laid out channels last: a Pallas kernel, the gradients of x, the weights and the bias in one pass over x and dy. The others by XLA, every gradient from dy shifted ahead as in the kernel | 2x on q, k and v; 4x on Mamba-2's, by XLA |
| Gated DeltaNet, DeltaNet | cuDNN's fused GDN kernel (`fused_delta`), by cuDNN's graph from a `buffer_callback`: any gate | 5x `chunkwise` |
| KDA with a `gate_lower_bound` of -5 or more | cuDNN's fused KDA kernel (`fused_delta`) | 10x `chunkwise` |
| KDA otherwise | bux's Pallas kernels (`pallas_delta`): any gate, each pair's decay through a position between them | 2.6x `chunkwise` |
| MoE experts | `grouped.ragged_dot`: tokamax's Mosaic and Triton kernels | 27x `jax.lax.ragged_dot` |
| MoE experts in MXFP8 | `grouped_mxfp8`: Mosaic GPU kernels of tcgen05's block-scaled MMA, the scales by TMA into tensor memory ([Experts](../parallelism/experts.md)) | 1.5-2.4x tokamax's bfloat16 GEMMs |
| MoE experts' SwiGLU in MXFP8 | `grouped_glu`: cuDNN's CuTe DSL grouped GEMMs, SwiGLU and its backward with their quantizations in the epilogues, each GPU's shard of the kernels quantized before FSDP gathers it ([Experts](../parallelism/experts.md)) | 1.13-1.17x `grouped_mxfp8`'s layer |
| MoE dispatch | `token_rows`, `token_sums`: gathers where XLA would scatter | 4x |
| MXFP8's quantizer | `blockscaled`: a Pallas kernel that quantizes an operand both ways in one pass | 4.3-4.7 TB/s at Qwen3-30B-A3B's projections |
| NVFP4's quantizers | `nvfp4`: Pallas kernels, a pass for the largest magnitudes and one that writes both copies, rounded by Blackwell's E2M1 conversions | 3-5x |

Where they live:

| kernel | module |
|--------|--------|
| flash attention, cuDNN's and tokamax's | {py:mod}`bux.parallel.flash` |
| `norm_rope` | {py:mod}`bux.architecture.attention.norm_rope` |
| `sparse`, `scoring` | {py:mod}`bux.architecture.attention.sparse`, {py:mod}`bux.architecture.attention.scoring` |
| `chunkwise`, `affine_scan` | {py:mod}`bux.architecture.recurrent.chunkwise`, {py:mod}`bux.architecture.recurrent.affine_scan` |
| `fused_delta`, `pallas_delta` | {py:mod}`bux.architecture.recurrent.fused_delta`, {py:mod}`bux.architecture.recurrent.pallas_delta` |
| `pallas_conv`, and XLA's form | {py:mod}`bux.architecture.layers.pallas_conv`, {py:mod}`bux.architecture.layers.conv` |
| `grouped`, `grouped_mxfp8`, `grouped_glu` | {py:mod}`bux.architecture.experts.grouped`, {py:mod}`bux.architecture.experts.grouped_mxfp8`, {py:mod}`bux.architecture.experts.grouped_glu` |
| `token_rows`, `token_sums` | {py:mod}`bux.architecture.experts.moe` |
| `blockscaled`, `nvfp4` | {py:mod}`bux.architecture.precision.blockscaled`, {py:mod}`bux.architecture.precision.nvfp4` |

## When a kernel falls back

Each kernel checks its own conditions as it traces, and where one fails, the layer runs
by XLA instead, with the same values up to rounding. Outside a GPU, and inside
`kernels.off()`, every kernel falls back.

| kernel | runs where | else |
|--------|------------|------|
| cuDNN's attention, any | a GPU of compute capability 9 or more with cuDNN; bfloat16 or float16; key and value head dims equal, up to 256, in steps of 8; no soft cap | tokamax's |
| cuDNN's causal kernels (`flash.CUDNN`) | a row is one document (a `lax.cond` asks, as every run's rows have ids); no window. Sinks fold in by the logsumexp | cuDNN's packed layout, or tokamax's |
| cuDNN's packed layout, rows of documents | rows of 8192 or more, or a window; no sinks; no more documents a row than its room (`context.documents_bound`: up to t / 256) | tokamax's, by a `lax.cond` |
| cuDNN's attention backward | XLA's deterministic ops off (under them XLA builds it no graph on a B300) | tokamax's backward |
| tokamax's flash attention | Linux, the `cuda` extra; bfloat16 or float16 | in float32 tokamax's Triton kernel; else XLA's t × t logits (`xla`) |
| `norm_rope` | an RMSNorm with a scale and no bias, RoPE by halves of every dim; 16-bit inputs; a head dim that is a power of two from 32 to 256, and a power of two of heads on each device; outside any `shard_map` | the norm and `rotate` by XLA |
| `fused_delta` | cudnn-frontend installed; a B200 or B300 (SM 10.0, 10.3); heads of 64 or 128 dims; bfloat16 or float16; `methods.recurrence = scan`; one decay a head (GDN, DeltaNet) or KDA with `gate_lower_bound` of -5 or more | `pallas_delta` for KDA, else `chunkwise` |
| `pallas_delta` | KDA (a decay per key channel); heads of 64 or 128 dims; bfloat16 or float16; `methods.recurrence = scan` | `chunkwise` |
| `pallas_conv` | q, k and v side by side (`CausalConv.side_by_side`); each device's positions, and the channels, in blocks of a power of two, 16 or more; a mesh that splits nothing but the rows | XLA's form of the backward; off a GPU, autodiff's |
| `affine_scan` | a state of k × v with k a power of two of 16 or more, and v a multiple of 32 | a scan over the chunks by XLA |
| `grouped.ragged_dot` | tokamax installed | `jax.lax.ragged_dot` |
| `grouped_mxfp8` | a Blackwell GPU (compute capability 10 or more); the experts' role in `mxfp8` or `mxfp8_2d` | XLA dequantizes and multiplies |
| `grouped_glu` | a Blackwell GPU with cudnn-frontend; a SwiGLU FFN (`gated_mlp` of `swish`, no `limit`, `up_softcap`, `up_offset`, `bias` or `expert_norm`); `mxfp8` in every pass; widths in whole tiles of 256 (Qwen3-30B-A3B's 2048 and 768, not the baseline's 704) | `grouped_mxfp8` |
| `blockscaled`'s and `nvfp4`'s quantizers | a GPU | XLA quantizes |
| the block-scaled GEMMs (cuDNN's, `__cudnn$blockScaledDot`) | a Blackwell GPU | XLA dequantizes and multiplies, which computes the same values |

`fused_delta` and `pallas_delta` take every position of a row: under
`methods.recurrence = pass_state`, which splits them over `context`, the delta rule
runs by chunks.

## Debugging with kernels off

1. Run the failing step with `BUX_KERNELS=off`. If the problem goes away, a kernel
   causes it.
2. Run the GPU smoke test of the kernel's family, which checks its kernels against
   XLA: `scripts/gpu_smoke/attention.py`, `experts.py`, `grouped_mxfp8.py`,
   `recurrence.py` or `sparse.py`.
3. To compare two trees' losses, pin XLA's autotuning and deterministic ops
   ([Benchmarks](benchmarks.md#noise-and-determinism)): under XLA's deterministic
   ops, cuDNN's attention backward is tokamax's.

:::{warning}
`BUX_KERNELS=off` is for debugging, not training: attention then holds t × t logits,
and recurrent mixers run by chunks or by a scan over positions, several times slower
(the table above).
:::

## Measurements

Every number on this page is from B300s, in October 2026. The suites that measure
them are in [Benchmarks](benchmarks.md).

### Tokens a second by model

Tokens a second on B300s, 8 × 4096 tokens a GPU, `precision(compute=bfloat16)`, and
the log's `mfu` (of the attention a causal mask keeps):

| model (d = 2048, 16 layers) | 1 GPU | 8, `mesh(fsdp=8)` |
|-----------------------------|-------|-------------------|
| Llama-style, 1.3B           | 144k, 44% MFU | 1.12M, 43% |
| Gated DeltaNet hybrid, `"DDDA"` | 117k, 37% | 818k |
| Mamba-2 hybrid, `"MMMA"`, 1.5B (Nemotron-H's `mamba2`) | 97k | 741k |
| KDA hybrid, `"KKKA"`        | 96k, 30% |                |
| KDA with `gate_lower_bound=-5` | 127k, 40% | 926k     |
| MoE, 64 experts, top 8      | 102k, 31% | 834k          |

### The delta rule's kernels

The hybrids of the one-GPU suite, 8 rows of 4096 a GPU, by chunks before their
kernels and by the kernels now:

| hybrid | by chunks | by its kernels | rows that fit now |
|--------|-----------|----------------|-------------------|
| GDN, one GPU | 93k, 121 GiB | 71 GiB | 24: 136k, 169 GiB |
| GDN, 8 GPUs | 715k, 108 GiB | 58 GiB | |
| KDA, `gate_lower_bound=-5` (cuDNN's kernel) | 110k, 173 GiB | 91 GiB | 16: 133k, 163 GiB |
| KDA, unbounded (bux's kernels) | 60.5k, 122 GiB | 96.5k, 88 GiB | |

Their losses over 30 steps, GEMM autotuning off, are within 1.7e-4 of the chunks' for
GDN, and the same for KDA. On 8 GPUs the GDN kernel's callbacks run one at a time, at
87% of 8 GPUs alone. KDA's kernel, which cudnn-frontend calls from JAX itself, runs at
91%. Each layer's times are in
[The delta rule on a B300](../parallelism/recurrence.md#the-delta-rule-on-a-b300).

### Short convolutions

A short conv's backward pass, one layer's, at the one-GPU suite's shapes (8 rows of
4096 in packed documents, 4 taps), with x laid out as XLA lays out each layer's:

| conv | autodiff, before | XLA's form | `pallas_conv` |
|------|------------------|------------|---------------|
| Mamba-2's, 6144 channels, laid out first | 1.65 ms | 0.41 ms | 0.42 ms, and 0.39 ms of transposes |
| GDN's q, 2048 channels, laid out last | 0.31 ms | 0.29 ms | 0.16 ms |

Autodiff takes the weights' gradient from x shifted back, a sum over positions XLA
runs at a quarter of the speed of the same sum from dy shifted ahead where the
channels come first, as in Mamba-2's layer. Where they come last, as in q, k and v,
XLA runs either sum slowly, and the kernel reads x and dy once, at 2.6-2.9 TB/s. So
on a GPU `side_by_side`'s parts go to the kernel, which takes x channels last, and
every other conv to XLA's form, but in rows without documents, where autodiff is as
fast. Off a GPU autodiff runs. Both forms' gradients are within bfloat16's rounding
of autodiff's in float32 (`scripts/gpu_smoke/conv.py`).

The hybrids of the one-GPU suite, 60 steps, before and after on the same node, each
the mean of two runs with the halves of the node swapped:

| hybrid | before | after | the conv's backward, ms a step |
|--------|--------|-------|--------------------------------|
| Mamba-2, `"MMMA"` | 95.7k | 99.8k (+4.3%) | 24.8 → 6.6, by XLA's form |
| GDN, `"DDDA"` | 132.3k | 135.8k (+2.7%) | 14.7 → 6.8 |
| KDA, `"KKKA"` | 98.2k | 99.8k (+1.5%) | 11.9 → 6.0 |
| KDA, `gate_lower_bound=-5` | 131.4k | 134.2k (+2.1%) | |

Peak memory is the same, and the losses after 60 steps are within 0.003 of each
other, as two runs of the same tree are.

### QK-norm and RoPE

The 1.3B with Qwen3's QK-norm, in MXFP8, trains 3000 steps of FineWeb-Edu by the norm
and RoPE's kernels 3% faster than by XLA's fusions (130k tokens a second against 126k,
GEMM autotuning off). Its held-out loss is within 0.04% of theirs at three seeds,
which differ by 1.9%.

### Softmax attention

On rows of one document (random tokens, `remat=none`), on one GPU, the 1.3B trains at
155k tok/s by cuDNN's causal kernels and at 148k by tokamax's.

On packed documents (`best_fit` of documents of 16 to 4000 words, GPT-2 tokens), on one
GPU, the 1.3B trains:

| rows | kernel | tok/s |
|------|--------|-------|
| 4096 | tokamax's | 145k |
| 16384 | cuDNN's, which skips every block no query sees | 139k |
| 16384 | tokamax's, whose backward does every block of a row | 109k |
| 4096, with sinks | tokamax's | 142k, where `auto` took `xla`, whose t × t logits did not fit |

On random tokens the two rows of 16384 train at 111k and 101k. Each kernel's times, the
same in training on 8 GPUs, and rows of 32k and 128k over 8 GPUs under context
parallelism, are in [Attention](../parallelism/attention.md#measurements).

### Sparse attention

Sparse attention trains at long context. A DeepSeek-V3.2-style 1B (MLA of 16 heads,
DSA's indexer of 64 heads and top 2048 in all 16 layers, dense MLPs) on one B300, in
bfloat16, against the same model by all t × t logits, as bux attended before `sparse`:

| tokens a row × rows | t × t logits | `sparse` |
|---------------------|--------------|----------|
| 4096 × 2            | 27.0k tok/s, 85.6 GiB | 28.6k, 31.0 GiB |
| 4096 × 8            | out of memory | 36.8k, 64.6 GiB |
| 16384 × 1           | out of memory | 14.3k, 42.5 GiB |
| 32768 × 1           | out of memory | 9.9k, 85.6 GiB |
| 32768 × 1, packed documents | out of memory | 11.7k, 85.6 GiB |
| 65536 × 1           | out of memory | 6.1k, 198 GiB |

::::{dropdown} How the indexer's loss got fast

The indexer's loss took 60% of a step at 32k: XLA's gather of each query's 2048
picks, their keys' gradient added by a scatter, and the attention's probabilities,
which the backward pass took again. It takes the probabilities once now, and on a GPU
scores the picks by Pallas kernels (`scoring.picked`): a program a query that stops at
its last pick, whose backward pass adds the keys' gradient atomically (relaxed:
Pallas's acquire-release adds took 1.6x the time).

Before, the model trained at 10.0k tokens a second at 16k, 7.1k at 32k, 5.8k on packed
documents (2.0x now: most queries of a short document pick few keys) and 4.4k at 64k.
The same model with GQA's 16 heads of 128 for MLA trained at 8.26k at 32k (11.6k now,
its probabilities by a kernel too). The loss's probabilities, kept for the backward
pass, take 9 GiB more at 32k.

The log's MFU, which counts the keys each query sees, is 5.5% at 32k: the indexer's
ranking of every key, its loss and its target cost far more than the attention to
2048 keys.

Under `methods(remat=save_attention)` the backward neither selects the keys again nor
runs the flash forward again ([Remat](../parallelism/remat.md)): at 32k, 7.27k tokens
a second in 39.8 GiB, as fast as no remat (7.11k, 77.9 GiB), where it ran at 5.85k
before; at 16k, 10.2k against 8.65k. MLA keeps its latents for `save_qkv_attention`,
not the heads' projections up from them: at 8 rows of 4096, 34.1k in 41.2 GiB, against
30.1k in 51.5 before.
::::

(kernels-matmul-formats)=

### Low-precision matmuls

Unlike the kernels above, a low-precision matmul format changes what a run computes,
so it is a setting, `precision(matmuls=...)`, and in the run's id
([Precision](../training/precision.md)). Its GEMMs and quantizers are kernels all the
same:

- **MXFP8.** cuDNN's MXFP8 GEMM runs at 1.75x bfloat16's rate on a B300 (2.7-3.0
  PFLOP/s at the model's shapes). Quantizing each operand twice, in one pass of a
  Pallas kernel (4.3-4.7 TB/s at Qwen3-30B-A3B's projections, scaling in bfloat16),
  gives some of it back. An MLP projection's forward and backward take 1.3-1.35x less
  time, q's and o's 1.1x, and k's and v's, half as wide, 7% more.
- **NVFP4.** cuDNN's NVFP4 GEMM runs at 4.8-5.5 PFLOP/s at the MLP's shapes, 1.7-1.9x
  MXFP8's. Its output is bfloat16, which the two tensors' scales then rescale in what
  reads it (cuDNN's own global scale is no faster, and takes it in bfloat16).
  NVFP4's quantizers are two passes of Pallas kernels over each operand
  ({py:mod}`bux.architecture.precision.nvfp4`):

  1. the largest magnitude of the operand, and that of its transpose through the
     Hadamard transform;
  2. both copies at once, each value rounded, saturated and packed by Blackwell's
     `cvt.rn` and `cvt.rs` (stochastic, 8 random bits a value).

  An MLP projection's forward and backward then take 1.07-1.1x less time than
  MXFP8's, but q's and o's take 1.1x more and k's and v's 1.25x. There the second
  pass, ~20 more small kernels a projection (scales for cuDNN's layout, the global
  scales) and the transform's copy, at 2-2.5 TB/s, cost more than their GEMMs save.

| `matmuls`, 1.3B Llama-style | 1 GPU | 8, `mesh(fsdp=8)` |
|-----------------------------|-------|-------------------|
| `compute` (bfloat16)        | 142k  | 1.12M             |
| `mxfp8`                     | 167k (1.18x) | 1.30M (1.16x) |
| `nvfp4`                     | 179k (1.22x), bfloat16 147k and `mxfp8` 169k beside it |   |

On 300 steps of packed documents, autotuning pinned, `mxfp8`'s loss is bfloat16's to
within the noise between seeds.

Under `mxfp8` the MoE's routed experts multiply in MXFP8 too, by the grouped MXFP8
GEMMs (`roles.experts`), and its other matmuls are few. The benchmarks' MoE, 64
experts of 704, top 8, 8 rows of 4096 a GPU:

| `precision`, the MoE | 1 GPU | 8, `mesh(fsdp=8)` |
|----------------------|-------|-------------------|
| `compute` (bfloat16) | 105k, 134.0 GiB | 824k, 83.8 GiB |
| `matmuls=mxfp8, roles=roles(experts=compute)`, the experts in bfloat16 as before | 108k, 133.9 GiB | 837k, 83.5 GiB |
| `matmuls=mxfp8` | 116k (1.10x), 119.4 GiB | 909k (1.10x), 69.1 GiB |

Which projections multiply in `mxfp8`, on one B300
([Precision](../training/precision.md) says what each setting means):

| `precision(matmuls=mxfp8, ...)`, 1.3B | tok/s | peak memory |
|---------------------------------------|-------|-------------|
| `compute` (bfloat16)                  | 144k  | 70.6 GiB    |
| `mxfp8`                               | 167k  | 66.3 GiB    |
| `mxfp8_2d`                            | 165k  | 66.3 GiB    |
| `roles=roles(kv=compute)`             | 164k  | 68.4 GiB    |
| `first=2, last=2`, of 32 branches     | 161k  | 66.9 GiB    |
| `passes=[fprop, dgrad]`               | 152k  | 69.9 GiB    |
| `moments=bfloat16`                    | 169k  | 61.6 GiB    |
| `nvfp4, first=2, last=8`              | 167k  | 46.1 GiB    |
| `nvfp4`                               | 179k  | 43.8 GiB    |

- **The `nvfp4` rows** are of a later tree, which takes the loss by chunks
  (`methods.loss`); the other rows take it by logits. In that tree bfloat16 trains at
  147k in 51.1 GiB, and `mxfp8` at 169k in 46.9.
- **NVIDIA's recipe**, its first 2 and last 8 branches in bfloat16, is as fast as
  `mxfp8`; NVFP4 everywhere is 1.06x it.
- **k and v in bfloat16 are slower, not faster.** q still quantizes x, the copy all
  three shared, so k's and v's GEMMs run at bfloat16's rate, and x is kept for them
  too.
- **Over 8 GPUs**, `mesh(fsdp=8)`, `mxfp8` and `mxfp8_2d` both train at 1.34M
  (bfloat16: 1.14M). Scanned, `first=2, last=2` trains at 1.20M, where `mxfp8` alone
  gives 1.22M.
