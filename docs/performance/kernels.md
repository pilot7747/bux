# GPU kernels

On a GPU, bux runs the computations XLA compiles slowly by kernels: cuDNN's,
tokamax's, cuDNN-frontend's, Transformer Engine's, or Pallas kernels of its own. You
set nothing to get them. Each runs by default on a GPU it supports and falls back to
XLA by itself where it cannot run. None is in a run's id: with or without them, a run
computes the same values up to rounding. On a CPU every computation runs by XLA.

This page lists the kernels and their speedups, says when each one falls back, and
shows how to turn them all off to debug one. Its measurements show what the kernels buy
whole models.

## Turning kernels off

Every kernel asks {py:func}`bux.kernels.gpu` whether it may run. To train with no
kernel at all, set `BUX_KERNELS=off`:

```sh
BUX_KERNELS=off bux train examples/experiments/smoke.bux
```

In Python, trace inside {py:func}`bux.kernels.off`:

```python
from bux import kernels

with kernels.off():
    loss = step(model, batch)  # every layer traces by XLA here
```

Turn kernels off to find out whether a kernel causes a wrong value or a crash, or to
count the FLOPs a kernel hides: bux's own FLOPs count traces the step with kernels off
([Profiling](profiling.md#mfu-and-flops)).

:::{note}
A Pallas kernel called off a GPU, as the tests call them, runs interpreted, by XLA
({py:func}`bux.kernels.interpreted`). On a mesh, since XLA cannot split a kernel,
{py:func}`bux.kernels.per_device` runs a kernel on each device's part of its inputs.
:::

(kernels-table)=

## The kernels

Speedups are against XLA's own program for the same computation, on a B300.

| What | Kernel | Speedup on a B300 |
|------|--------|-------------------|
| softmax attention | `methods.attention = auto`, the default, picks cuDNN's causal kernels for rows of one document, and FlashAttention-4's (CuTe DSL) for rows of documents, windows, sinks, a soft cap and MLA's head dims. Where neither runs: cuDNN's fused attention for rows of 8192 or more and for windows, else tokamax's flash attention | 7x, and no t × t logits |
| QK-norm then RoPE | `norm_rope`: one Pallas pass each way for a per-head RMSNorm with a scale and RoPE by halves (Qwen3's) | 1.4x forward, 1.9x backward (Qwen3-30B-A3B's queries: 0.12 and 0.24 ms per layer) |
| sparse attention (DSA, QSA, CSA, MSA) | `sparse`: tokamax's flash attention with the kept keys as a mask, a group at a time for MSA's picks. The indexer's scores and its loss's target by Pallas kernels (`scoring`); its top-k and its loss's scores of the picks by cuDNN's DSA kernels (`indexer_cudnn`). MSA's scores, top-k and loss run by XLA | 2.6x XLA's chunks of queries, and no t × t floats. At 32k: the indexer's selection 3.6x, its loss's picks 5.2x, a DSA 1B's step 1.8x |
| CSA attention (DeepSeek-V4) | `sparse_cudnn`: cuDNN's DSA kernels (CuTe DSL), forward and backward, with each query's window, kept latents and sinks in one softmax; the indexer's target at its picks | 15x at 4k tokens, 55x at 16k; the target 6x and 31x |
| linear attention, RWKV-7 | `chunkwise`: chunks of 64 positions, the state across them by a Pallas kernel (`affine_scan`). `methods(states=recompute)` keeps a fourteenth of the memory for the backward | 14-17x a scan over positions |
| Mamba-2 | `chunkwise`, with B and C per group and the state across chunks by a matmul. `methods(states=recompute)` keeps a sixteenth of the memory for the backward | 25x a scan over positions (2 × 4096 tokens, rematerialized; at 8 × 4096 the scan needs 199 GiB) |
| a short conv's backward: `conv`, `short_conv`, Mamba's, Mamba-2's, and the q, k and v of GDN, KDA and DeltaNet | `pallas_conv` for q, k and v, laid out channels last: one Pallas pass over x and dy for the gradients of x, the weights and the bias. Every other conv by XLA, each gradient taken from dy shifted ahead as in the kernel | 2x on q, k and v; 4x on Mamba-2's, by XLA |
| Gated DeltaNet, DeltaNet | `fused_delta`: cuDNN's fused GDN kernel, through cuDNN's graph from a `buffer_callback`; any gate | 5x `chunkwise` |
| KDA with a `gate_lower_bound` of -5 or more | `fused_delta`: cuDNN's fused KDA kernel | 10x `chunkwise` |
| KDA, any other gate | `pallas_delta`: bux's Pallas kernels, which take each pair's decay through a position between them | 2.6x `chunkwise` |
| GLA, Lightning | `fused_delta`: cuDNN's fused GDN-2 kernel, through a `buffer_callback` (its delta rule at a step of 0 is decayed linear attention) | 4.7x `chunkwise` (GLA), 1.6x (Lightning) |
| Gated DeltaNet-2 with a `gate_lower_bound` of -5 or more | `fused_delta`: cuDNN's fused GDN-2 kernel, its erase gate over the key channels and its write gate over the value channels | 8.4x `chunkwise` |
| MoE experts | `grouped.ragged_dot`: tokamax's Mosaic and Triton kernels | 27x `jax.lax.ragged_dot` |
| MoE experts in MXFP8 | `grouped_mxfp8`: Mosaic GPU kernels of tcgen05's block-scaled MMA, the scales loaded by TMA into tensor memory ([Experts](../parallelism/experts.md)) | 1.5-2.4x tokamax's bfloat16 GEMMs |
| MoE experts' activation in MXFP8 | `grouped_glu`: cuDNN's CuTe DSL grouped GEMMs with the activation (ReLU^2, SwiGLU, a clamped GeGLU, SiTU-GLU), its backward and their quantizations in the epilogues; each GPU's shard of the weights quantized before FSDP gathers it ([Experts](../parallelism/experts.md#activations-fused-by-cudnn)) | 1.15-1.31x `grouped_mxfp8`'s layer at the experts of Qwen3-30B-A3B and Nemotron 3, 1.04-1.15x at those of gpt-oss and DeepSeek-V4-Flash |
| MoE dispatch | `token_rows`, `token_sums`: gathers where XLA would scatter | 4x |
| MXFP8 quantizer | `blockscaled`: a Pallas kernel that quantizes an operand both ways in one pass | 4.3-4.7 TB/s at Qwen3-30B-A3B's projections |
| NVFP4 quantizers | `nvfp4_te`: Transformer Engine's kernels, a pass for the largest magnitudes, then its NVFP4 cast with the transpose's Hadamard GEMM fused in. Else `nvfp4`: Pallas kernels, a pass for the largest magnitudes and one that writes both copies, rounded by Blackwell's E2M1 conversions | 3-5x; TE's 1.25-1.5x bux's Pallas kernels |

Where they live:

| Kernel | Module |
|--------|--------|
| flash attention, cuDNN's and tokamax's | {py:mod}`bux.parallel.flash` |
| FlashAttention-4's | {py:mod}`bux.parallel.fa4` |
| `norm_rope` | {py:mod}`bux.architecture.attention.norm_rope` |
| `sparse`, `scoring`, `indexer_cudnn` | {py:mod}`bux.architecture.attention.sparse`, {py:mod}`bux.architecture.attention.scoring`, {py:mod}`bux.architecture.attention.indexer_cudnn` |
| `sparse_cudnn` | {py:mod}`bux.architecture.attention.sparse_cudnn` |
| `chunkwise`, `affine_scan` | {py:mod}`bux.architecture.recurrent.chunkwise`, {py:mod}`bux.architecture.recurrent.affine_scan` |
| `fused_delta`, `pallas_delta` | {py:mod}`bux.architecture.recurrent.fused_delta`, {py:mod}`bux.architecture.recurrent.pallas_delta` |
| `pallas_conv`, and XLA's form | {py:mod}`bux.architecture.layers.pallas_conv`, {py:mod}`bux.architecture.layers.conv` |
| `grouped`, `grouped_mxfp8`, `grouped_glu` | {py:mod}`bux.architecture.experts.grouped`, {py:mod}`bux.architecture.experts.grouped_mxfp8`, {py:mod}`bux.architecture.experts.grouped_glu` |
| `token_rows`, `token_sums` | {py:mod}`bux.architecture.experts.moe` |
| `blockscaled`, `nvfp4`, `nvfp4_te` | {py:mod}`bux.architecture.precision.blockscaled`, {py:mod}`bux.architecture.precision.nvfp4`, {py:mod}`bux.architecture.precision.nvfp4_te` |

## When a kernel falls back

Each kernel checks its conditions as it traces. Where one fails, the layer runs by XLA
instead, with the same values up to rounding. Off a GPU, and inside `kernels.off()`,
every kernel falls back.

| Kernel | Runs when | Else |
|--------|-----------|------|
| FlashAttention-4's (`fa4`) | a Blackwell GPU (SM 10.x) with flash-attn-4 (the `cuda` extra); bfloat16 or float16; key and value head dims equal, up to 128, in steps of 8, or 192 and 128 (MLA); whole rows (under `context`, `ulysses` gives it them) | cuDNN's, else tokamax's |
| FlashAttention-4's varlen layout, for rows of documents | no more documents in a device's rows than its room (`fa4.room`: 16 per row of 4096) | tokamax's, by a `lax.cond` |
| cuDNN's attention, any | a GPU of compute capability 9 or more, with cuDNN; bfloat16 or float16; key and value head dims equal, up to 256, in steps of 8; no soft cap | FlashAttention-4's, else tokamax's |
| cuDNN's causal kernels (`flash.CUDNN`) | each row is one document (a `lax.cond` checks, since every run's rows carry document ids); no window. Sinks fold in by the logsumexp | FlashAttention-4's, cuDNN's packed layout, or tokamax's |
| cuDNN's packed layout, for rows of documents | rows of 8192 or more, or a window; no sinks; no more documents per row than its room (`context.documents_bound`: up to t / 256) | tokamax's, by a `lax.cond` |
| cuDNN's attention backward | XLA's deterministic ops off (under them, XLA builds no graph for it on a B300) | tokamax's backward |
| tokamax's flash attention | Linux, the `cuda` extra; bfloat16 or float16 | in float32, tokamax's Triton kernel; else XLA's t × t logits (`xla`) |
| `norm_rope` | an RMSNorm with a scale and no bias, then RoPE by halves of every dim; 16-bit inputs; a head dim that is a power of two from 32 to 256, and a power of two of heads on each device; outside any `shard_map` | the norm and `rotate` by XLA |
| `fused_delta` | cudnn-frontend installed; a B200 or B300 (SM 10.0, 10.3); heads of 64 or 128 dims; bfloat16 or float16; `methods.recurrence = scan`; one decay per head (GDN, DeltaNet), or KDA with a `gate_lower_bound` of -5 or more | `pallas_delta` for KDA, else `chunkwise` |
| `pallas_delta` | KDA (a decay per key channel); heads of 64 or 128 dims; bfloat16 or float16; `methods.recurrence = scan` | `chunkwise` |
| `pallas_conv` | q, k and v side by side (`CausalConv.side_by_side`); each device's positions, and the channels, in blocks of a power of two, 16 or more; a mesh that splits nothing but the rows | XLA's form of the backward; off a GPU, autodiff's |
| `fused_delta`'s GDN-2, for GLA and Lightning | cudnn-frontend 1.32 or later; a B200 or B300; heads of 64 or 128 dims; bfloat16; `methods.recurrence = scan` | `chunkwise` |
| `fused_delta`'s GDN-2, for Gated DeltaNet-2 | as for GLA and Lightning, and a `gate_lower_bound` of -5 or more | `chunkwise` |
| `indexer_cudnn`'s top-k | a B200 or B300 with cudnn-frontend; up to 2048 picks per query | XLA counts the k-th highest score's bits |
| `indexer_cudnn`'s loss | a B200 or B300 with cudnn-frontend; an indexer of 32 or 64 heads of 128 dims in bfloat16 (DSA's 64), its picks in multiples of 128 up to 2048 | `scoring`'s Pallas kernels, for heads and dims that are powers of two; else XLA |
| `affine_scan` | a state of k × v, with k a power of two of 16 or more and v a multiple of 32 | a scan over the chunks by XLA |
| `grouped.ragged_dot` | tokamax installed | `jax.lax.ragged_dot` |
| `grouped_mxfp8` | a Blackwell GPU (compute capability 10 or more); the experts' role in `mxfp8` or `mxfp8_2d` | XLA dequantizes and multiplies |
| `sparse_cudnn` | a B200 or B300 (SM 10.0, 10.3) with cudnn-frontend; `methods.attention = auto`; a CSA layer, its latents 512 wide; bfloat16 or float16; no soft cap; a window (`sliding_window_size`) and latents kept by an indexer one by one (not `block_size > 1`); at most 2^16 queries per device. Any number of heads: 64 per call, the last call padded. The indexer's target: bfloat16, its heads not split over devices, at most 128 | `sparse`, which runs 512 dims by XLA, and on a GPU under `auto` a window-only layer too; elsewhere a window-only layer runs by `methods.attention` |
| `grouped_glu` | a Blackwell GPU with cudnn-frontend; an activation cuDNN fuses: `mlp` of `relu2`, `gated_mlp` of `swish` or `quick_gelu` (with or without `limit` and `up_offset`), or of `situ` with an `up_softcap`; no `bias` or `expert_norm`; `mxfp8` in every pass; a width of whole blocks of 64, and hidden units in multiples of 32 (padded to whole tiles of 128) | `grouped_mxfp8` |
| `nvfp4_te` | a B200 or B300 with Transformer Engine's JAX bindings (bux's image); bfloat16 operands of whole tiles of 64 × 128 on each device (a split whose parts are not keeps the dimension whole); for x and dy, 8192 rows or more per device (`nvfp4_te.ROWS`) | `nvfp4`'s Pallas kernels |
| `blockscaled`'s MXFP8 quantizer | a GPU of compute capability 8.9 or more (Ada, Hopper, Blackwell), for Triton's E4M3 | XLA quantizes |
| `nvfp4`'s quantizers | a Blackwell GPU (compute capability 10 or more), for its E2M1 conversions (`cvt.rn`, `cvt.rs`) | XLA quantizes |
| the block-scaled GEMMs (cuDNN's, `__cudnn$blockScaledDot`) | a Blackwell GPU | XLA dequantizes and multiplies, which computes the same values |

`fused_delta` and `pallas_delta` take every position of a row. Under
`methods.recurrence = pass_state`, which splits rows over `context`, the delta rule
runs by chunks, and so do GLA and Lightning.

The `cuda` extra asks for cudnn-frontend 1.32 or later, the first release with GDN-2's
kernel and the DSA indexer's deterministic top-k.

:::{warning}
GDN-2's kernel factors each decay through a tile of 16 positions. Where a tile decays
a channel past $e^{-83}$ (past $e^{-9.7}$ in float16, so the kernel takes bfloat16
only), it runs the whole call again by an exact path: the same values, many times
slower. GLA at a gate of -20 per position took 45 ms per layer instead of 3.7. GLA's
own gate, $\log \sigma(f) / 16$, gets there only with $f$ near -83 through a whole
tile, and Lightning's decays never do. An unbounded Gated DeltaNet-2 does, which is why
it runs by chunks ([Gated DeltaNet-2's kernel](#gated-deltanet-2s-kernel)).
:::

## Debugging with kernels off

1. Run the failing step with `BUX_KERNELS=off`. If the problem goes away, a kernel
   causes it.
2. Run the GPU smoke test of the kernel's family, which checks its kernels against
   XLA: `scripts/gpu_smoke/attention.py`, `conv.py`, `experts.py`,
   `grouped_mxfp8.py`, `nvfp4.py`, `recurrence.py` or `sparse.py`.
3. To compare two trees' losses, pin XLA's autotuning and deterministic ops
   ([Benchmarks](benchmarks.md#noise-and-determinism)). Under XLA's deterministic
   ops, cuDNN's attention backward runs as tokamax's.

:::{warning}
`BUX_KERNELS=off` is for debugging, not training: attention then holds t × t logits,
and recurrent mixers run by chunks or by a scan over positions, several times slower
(the table above).
:::

## Measurements

All on one B300, 8 rows of 4096 random tokens, in bfloat16, in October 2026, unless a
section says otherwise. The table above gives each kernel's speedup alone; these
sections show what the kernels buy a whole model, and check that its loss does not
move. Each benchmark model's tokens/s is in [Benchmarks](benchmarks.md#one-gpu).

### The delta rule's kernels

The one-GPU suite's hybrids, by chunks and by their kernels:

| Hybrid | Kernel | By chunks | By the kernel | Rows that fit by the kernel |
|--------|--------|-----------|---------------|-----------------------------|
| GDN, one GPU | cuDNN's GDN | 93k tok/s, 121 GiB | 71 GiB | 24: 136k tok/s, 169 GiB |
| GDN, 8 GPUs | cuDNN's GDN | 715k tok/s, 108 GiB | 58 GiB | |
| KDA, `gate_lower_bound=-5` | cuDNN's KDA | 110k tok/s, 173 GiB | 91 GiB | 16: 133k tok/s, 163 GiB |
| KDA, unbounded | bux's Pallas kernels | 60.5k tok/s, 122 GiB | 96.5k tok/s, 88 GiB | |

By their kernels, the GDN and bounded KDA hybrids train at 120.9k and 127.3k tokens/s
in the one-GPU suite. Over 30 steps with GEMM autotuning off, the GDN kernel's losses
are within 1.7e-4 of the chunks', and the same for KDA. On 8 GPUs, the GDN kernel's
callbacks run one GPU at a time, at 87% of eight times one GPU's rate; KDA's kernel,
which cudnn-frontend calls from JAX itself, runs at 91%. Each layer's times are in
[The delta rule on a B300](../parallelism/recurrence.md#the-delta-rule-on-a-b300).

### GLA's and Lightning's kernel

The baseline's hybrids of GLA (`"GGGA"^4`) and Lightning (`"LLLA"^4`), 16 heads of 128,
by chunks and by cuDNN's GDN-2 kernel, on the same node, on 2026-10-26:

| Hybrid | By chunks | By the GDN-2 kernel | A layer's forward and backward |
|--------|-----------|---------------------|--------------------------------|
| GLA | 85.4k tok/s, 83.5 GiB | 130.7k tok/s, 66.9 GiB | 17.4 → 3.7 ms |
| GLA, `fsdp=8` on a node | 661k tok/s, 74.3 GiB | 920k tok/s, 57.1 GiB | |
| Lightning | 135.2k tok/s, 73.5 GiB | 142.2k tok/s, 61.4 GiB | 5.5 → 3.5 ms |

On a node, GDN-2's callbacks run one GPU at a time, as GDN's do: GLA trains at 88% of
eight times one GPU's rate.

RWKV-7 has no kernel yet. Its chunks take a pair's decay by one matmul, since its decay
is bounded, and its hybrid (`"RRRA"^4`) trains at 87.4k tokens/s
([RWKV-7's chunks](../parallelism/recurrence.md#rwkv-7s-chunks)).

::::{dropdown} Accuracy of the GDN-2 kernel for GLA and Lightning

- Over 30 steps, the last loss by the kernel matches the chunks' to 0.004 (GLA) and
  0.0002 (Lightning).
- On FineWeb-Edu (3000 steps of 8 × 4096 tokens, GEMM autotuning off, XLA's
  deterministic ops on), the GLA hybrid's held-out loss by the kernel is within
  0.007 of the chunks' at each of seeds 0, 1 and 2, where the seeds spread over
  0.055.
- At 2 × 4096 tokens in packed documents, the kernel's outputs and gradients are within
  4·10⁻³ of the chunks' in float32 (as a share of each one's norm), at GLA's own gate,
  gates down to -20 per position, spikes of -10⁴, and Lightning's decays. The log
  decays' gradient is within 7·10⁻³ at gates down to -5. The kernel keeps its parts in
  bfloat16, as cuDNN's GDN kernel does.
::::

### Gated DeltaNet-2's kernel

The baseline's hybrids of Gated DeltaNet-2 (`"GGGA"^4`, 16 heads of 128) and of KDA,
100 steps each on the same node, on 2026-10-26, by chunks
(`methods.recurrence = pass_state`) and as bux runs them by default:

| Hybrid | By chunks | By default | The recurrence's forward and backward |
|--------|-----------|------------|---------------------------------------|
| GDN-2, `gate_lower_bound=-5` | 59.6k tok/s, 125.3 GiB | 125.3k tok/s, 95.0 GiB (cuDNN's GDN-2) | 29.8 → 3.6 ms |
| GDN-2, unbounded (the paper's) | 58.5k tok/s, 125.3 GiB | by chunks | 29.8 ms |
| KDA, `gate_lower_bound=-5` | 61.5k tok/s, 122.0 GiB | 131.8k tok/s, 91.4 GiB (cuDNN's KDA) | 28.6 → 3.2 ms |
| KDA, unbounded | 61.0k tok/s, 122.0 GiB | 99.0k tok/s, 88.0 GiB (bux's Pallas kernels) | 28.6 → 11.2 ms |

A GDN-2 layer has two more projections of 2048 × 2048 than a KDA layer, for its gates,
so its hybrid trains 5% slower than KDA's by their kernels and 3-4% slower by chunks.

The paper's unbounded gate runs by chunks, because the kernel would get slower as the
model learns. On random tokens, whose gates stay mild, the unbounded hybrid trained at
121.3k tokens/s by the kernel. On FineWeb-Edu its gates soon decayed some tile of 16
positions past $e^{-83}$, where the kernel reruns the whole call by its exact path, and
from step 500 it trained slower than by chunks (median tokens/s over each span of
steps, seed 0; seed 1 within 4%):

| Unbounded GDN-2 hybrid | Steps 20-200 | 200-500 | 500-1000 | 1000-2000 |
|------------------------|--------------|---------|----------|-----------|
| by chunks | 57.3k | 57.3k | 57.2k | 57.0k |
| by the kernel | 121.4k | 62.8k | 50.9k | 48.0k |

By the kernel, the recurrence takes 3.6 ms at mild gates; by its exact path, 7.0 ms at
KDA's gate as it starts and 44.9 ms at gates of -20 per position.

The paper's hybrid itself (`gated_deltanet_2.bux`: 9 GDN-2 and 9 sliding-window layers,
1.45B parameters with the embedding and the head) trains at 66.0k tokens/s by chunks,
in 118.5 GiB. Bounded at -5, it trains at 111.8k by the kernel, in 98.7 GiB, and at
64.9k by chunks.

::::{dropdown} Accuracy of the GDN-2 kernel for Gated DeltaNet-2

At 2 × 4096 tokens in packed documents, the kernel's outputs and gradients are within
7.2·10⁻³ of the chunks' in float32 (as a share of each one's norm): at gates down to
-5, the layer's own, heads of 64, more documents than its room, erase gates up to 2,
and, by its exact path, at unbounded and steep gates and spikes of -10⁴.

On FineWeb-Edu (2000 steps of 8 × 4096 tokens), the held-out losses:

| Hybrid | Seed 0 | Seed 1 |
|--------|--------|--------|
| GDN-2, unbounded, by chunks | 4.0781 | 4.0223 |
| GDN-2, unbounded, by the kernel's exact path | 4.0766 | 4.0216 |
| GDN-2, `gate_lower_bound=-5`, by the kernel | 4.0889 | |
| KDA, unbounded | 4.1032 | 4.0388 |
| KDA, `gate_lower_bound=-5` | 4.1023 | |

The kernel's losses are the chunks' to 0.0015, where the seeds spread over 0.056.
GDN-2's are 0.017 and 0.025 below KDA's, with 100M more parameters: the paper matches
the two in parameters, and this comparison does not.
::::

### Short convolutions

The one-GPU suite's hybrids over 60 steps, with the conv's backward by autodiff and by
bux's (`pallas_conv` for q, k and v, XLA's form for the rest), on 2026-10-26, each
the mean of two runs with the halves of a node swapped:

| Hybrid | Autodiff | bux's backward | The conv's backward per step |
|--------|----------|----------------|------------------------------|
| Mamba-2, `"MMMA"` | 95.7k tok/s | 99.8k (+4.3%) | 24.8 → 6.6 ms, by XLA's form |
| GDN, `"DDDA"` | 132.3k | 135.8k (+2.7%) | 14.7 → 6.8 ms |
| KDA, `"KKKA"` | 98.2k | 99.8k (+1.5%) | 11.9 → 6.0 ms |
| KDA, `gate_lower_bound=-5` | 131.4k | 134.2k (+2.1%) | |

Peak memory is the same, and the losses after 60 steps are within 0.003 of each other,
as two runs of one tree are.

::::{dropdown} Why there are two forms of the backward

One layer's backward at the suite's shapes (8 rows of 4096 in packed documents, 4
taps), with x laid out as XLA lays it out in each layer:

| Conv | Autodiff | XLA's form | `pallas_conv` |
|------|----------|------------|---------------|
| Mamba-2's, 6144 channels, laid out first | 1.65 ms | 0.41 ms | 0.42 ms, plus 0.39 ms of transposes |
| GDN's q, 2048 channels, laid out last | 0.31 ms | 0.29 ms | 0.16 ms |

Autodiff takes the weights' gradient from x shifted back. Where the channels come
first, as in Mamba-2's layer, XLA runs that sum at a quarter of the speed of the same
sum from dy shifted ahead, which is XLA's form. Where they come last, as in q, k and v,
XLA runs either sum slowly, and the kernel reads x and dy once, at 2.6-2.9 TB/s. So on
a GPU, `side_by_side`'s parts go to the kernel and every other conv to XLA's form,
except in rows without documents, where autodiff is as fast. Both forms' gradients are
within bfloat16's rounding of autodiff's in float32 (`scripts/gpu_smoke/conv.py`).
::::

### QK-norm and RoPE

The 1.3B with Qwen3's QK-norm, in MXFP8, trains 3000 steps of FineWeb-Edu 3% faster by
the `norm_rope` kernels than by XLA's fusions (130k tokens/s against 126k, GEMM
autotuning off). Its held-out loss is within 0.04% of XLA's at three seeds, which
differ by 1.9%.

### Softmax attention

On rows of one document (`remat=none`), the 1.3B trains at 155k tokens/s by cuDNN's
causal kernels and at 148k by tokamax's.

On packed documents, FlashAttention-4's kernels skip every block no query sees, forward
and backward, and they take sinks, a soft cap and MLA's head dims. Without them, cuDNN's
packed layout also skips those blocks, but tokamax's backward computes every block of a
row. On packed documents (`best_fit` of documents of 16 to 4000 words, GPT-2 tokens),
30 steps each on the same B300, on 2026-10-26:

| Run | Without FlashAttention-4 | By FlashAttention-4 | Peak memory |
|-----|--------------------------|---------------------|-------------|
| the 1.3B, rows of 4096 | 143k tok/s, tokamax's | 151k (+5%) | 51.1 GiB either way |
| the 1.3B with a sink per head, rows of 4096 | 144k, tokamax's | 150k (+4%) | 51.1 GiB either way |
| the 1.3B, rows of 16384 | 123k, cuDNN's packed layout, tokamax's past its room | 149k (+21%) | 58.5 → 51.1 GiB |
| MLA (16 heads of 192 and 128), rows of 4096 | 132k, tokamax's | 138k (+5%) | 59.8 GiB either way |
| gpt-oss's attention (64 heads, 8 kv heads of 64, sinks, every other layer in a window of 128), rows of 4096 | 101k, tokamax's | 127k (+26%) | 57.6 → 55.5 GiB |

On rows of one document, the MLA model trains at 131k by tokamax's kernels and 135k by
FlashAttention-4's (+4%), and gpt-oss's attention at 112k and 127k (+14%), its windows
by FlashAttention-4 and the rest by cuDNN. The 1.3B runs by cuDNN either way, at 152k
and 150k, the same within noise. In rows of 16384 of one document, the 1.3B trains at
111k by cuDNN's kernels and 101k by tokamax's. Each kernel's times, training on 8 GPUs,
and rows of 32k and 128k over 8 GPUs under context parallelism are in
[Attention](../parallelism/attention.md#measurements).

### Sparse attention

Sparse attention is what makes long context trainable. A DeepSeek-V3.2-style 1B (MLA
with 16 heads, DSA's indexer of 64 heads and top 2048 in all 16 layers, dense MLPs),
by all t × t logits and by `sparse`, its indexer by bux's Pallas kernels and XLA
(where cuDNN's DSA kernels cannot run) or by cuDNN's (the default on a B200 or B300),
on 2026-10-26 (the t × t column on 2026-10-23):

| Tokens per row × rows | All t × t logits | `sparse`, indexer by Pallas and XLA | `sparse`, indexer by cuDNN |
|-----------------------|------------------|-------------------------------------|----------------------------|
| 4096 × 2 | 27.0k tok/s, 85.6 GiB | 35.6k, 26.1 GiB | 54.2k, 26.4 GiB |
| 4096 × 8 | out of memory | 38.7k, 64.5 GiB | 56.4k, 63.3 GiB |
| 16384 × 1 | out of memory | 15.0k, 42.5 GiB | 29.1k, 42.5 GiB |
| 32768 × 1 | out of memory | 9.99k, 85.7 GiB | 18.0k, 80.5 GiB |
| 32768 × 1, packed documents | out of memory | 12.0k, 85.7 GiB | 20.5k, 80.5 GiB |
| 32768 × 1, GQA's 16 heads of 128 instead of MLA | | 11.7k, 77.4 GiB | 24.4k, 73.8 GiB |
| 65536 × 1 | out of memory | 6.13k, 198 GiB | 10.4k, 187 GiB |

At 32k a step takes 1.83 s on the GPU, against 3.31 s with the indexer by Pallas and
XLA. The log's MFU, which counts the keys each query sees, is 10.0% there: the
indexer's ranking of every key and the loss's target cost far more than the attention
to 2048 keys.

Under `methods(remat=save_attention)`, the backward neither selects the keys again nor
runs the flash forward again ([Remat](../parallelism/remat.md)): at 32k it trains at
13.2k tokens/s in 68.8 GiB, against 18.0k without remat. Under
`save_qkv_attention`, MLA keeps its latents rather than the heads' projections up from
them: at 8 rows of 4096, the model trains at 34.1k tokens/s in 41.2 GiB.

::::{dropdown} The indexer's kernels, one layer at 32k

The indexer's parts of a layer at 32k, one row, by Pallas and XLA and by cuDNN's DSA
kernels (`indexer_cudnn`):

| Part of a layer | Pallas and XLA | cuDNN's kernels |
|-----------------|----------------|-----------------|
| the indexer's selection: its scores (Pallas, 12.2 ms) and top-k | 58.4 ms | 16.1 ms |
| the same on packed documents | 58.5 ms | 14.2 ms |
| its loss's scores of each query's 2048 picks, forward and backward | 55.0 ms | 10.6 ms |
| the same on packed documents | 38.2 ms | 9.9 ms |

- **Top-k.** XLA's top-k counts the k-th highest score's bits in 32 passes over every
  row and scatters the kept slots into place: 46 ms of a layer's 58 at 32k, where
  cuDNN's radix kernel takes 2.8. A key a query does not see enters it as one of the
  lowest floats, not as -inf, because the kernel picks among ties at the k-th one at a
  time: a short document's rows, which see fewer keys than they keep, tied at -inf
  took 5x one document's time. A query whose span holds no more slots than it keeps
  keeps all it sees, without the kernel.
- **The loss.** Its backward added each pick's share of the keys' gradient atomically,
  48 ms of its 55. cuDNN's indexer backward gathers a block of 128 picks' keys by TMA,
  runs its GEMMs on tensor cores, and adds the keys' gradient by bulk reductions, in
  8.8 ms. Its gradient of the scores rounds to bfloat16 before the GEMMs (4e-3 of the
  largest gradient, against the float32 Pallas kernels').
- **Chunks of picks.** The loss takes 2^23 picks at a time, 8 chunks of a row of 32k
  (`indexer.LOSS_PICKS`). In one chunk, XLA kept 10 GiB more alive at 4096 × 8 and 5
  GiB more at 32k, for 2-3% less time.
- **The Pallas kernels** (`scoring.picked`, on GPUs other than a B200 or B300) run a
  program per query that stops at its last pick, and add the keys' gradient
  atomically, with relaxed adds: Pallas's acquire-release adds took 1.6x the time.
- **Accuracy.** On FineWeb-Edu, a 0.4B DSA model that keeps 512 of 4096 keys reaches
  the same held-out loss after 2000 steps (65M tokens) by cuDNN's kernels as by the
  Pallas kernels, within 0.0024 at each of 4 seeds, which spread over 0.068.
::::

### CSA by cuDNN's DSA kernels

DeepSeek-V4.1-Flash's CSA attention, forward and backward, on 2026-10-26: 64 heads of
a 512-wide latent, each query's window of 128 latents and its top 512 compressed ones
(a latent every two tokens), and sinks, on rows of a dozen documents. XLA's chunks of
queries (`sparse`) compute every slot of a row; cuDNN's kernels gather only the 640 a
query sees. The indexer's target runs by XLA at every latent, and by the kernels at
the picks:

| Tokens | Attention by XLA | by cuDNN | Target by XLA | by cuDNN |
|--------|------------------|----------|---------------|----------|
| 4096 | 35.4 ms | 2.44 ms (15x) | 4.18 ms | 0.72 ms (6x) |
| 16384 | 543 ms | 9.86 ms (55x) | 51.7 ms | 1.68 ms (31x) |
| 32768 | | 15.8 ms | | 2.48 ms |

NVIDIA's own times for the two kernels alone on a GB300 are 1.82, 7.05 and 14.2 ms
(cudnn-frontend's `benchmark/dsa`).

A whole layer, `d_model` 5120, forward and backward in bfloat16, with its compression,
indexer and the indexer's loss (V4.1's `pairs`), and a window-only layer:

| Tokens | `pairs` by XLA | `pairs` by cuDNN | Window-only by cuDNN |
|--------|----------------|------------------|----------------------|
| 4 × 4096 | 166 ms, 15.6 GiB | 24.7 ms, 4.4 GiB | 12.7 ms, 4.2 GiB |
| 16384 | 613 ms, 8.1 GiB | 35.3 ms, 4.4 GiB | 12.5 ms |
| 32768 | 2408 ms, 15.4 GiB | 100 ms, 8.8 GiB | 24.8 ms |

At 32k, a `pairs` layer's 100 ms are mostly its indexer's: the attention and the target
take 18.

DeepSeek-V4.1-Flash shrunk to one B300 (its attention, width, vocabulary and mHC
stream; 8 of its 40 layers, one of each kind; 16 of its 384 experts; no Engram), in
bfloat16 with `remat=none`, trains by the kernels at 31.7k tokens/s at 4096 × 4 (166
GiB), 27.8k at 16384 × 1 (166 GiB) and 26.3k at 32768 × 1 (234 GiB). Without its
window-only layer it trains at 31.3k at 16k and 28.6k at 32k, against 3.69k and 1.95k
by XLA's chunks. At 16k, the step's attention takes 65 ms of 579, and its experts'
grouped GEMMs 136.

A ~0.5B V4-style model (16 heads of 512, a window of 128 and the top 512, three of four
layers reusing the last one's latents and picks) trained 2000 steps of FineWeb-Edu (8 ×
4096 tokens per step, GEMM autotuning off, deterministic ops) at 57.2k tokens/s in 54.8
GiB by the kernels, against 25.0k in 79.2 GiB by XLA (`methods.attention = xla`). Its
held-out loss at four seeds is within 0.0011 of XLA's (4.4136 against 4.4140 on
average), where the seeds spread over 0.088.

(kernels-matmul-formats)=

### Low-precision matmuls

A low-precision matmul format changes what a run computes, so unlike the kernels above
it is a setting, `precision(matmuls=...)`, and part of the run's id
([Precision](../training/precision.md)). Its GEMMs and quantizers are kernels all the
same.

- **MXFP8.** cuDNN's MXFP8 GEMM runs at 1.75x bfloat16's rate on a B300 (2.7-3.0
  PFLOP/s at the model's shapes). Quantizing each operand twice, in one Pallas pass
  (4.3-4.7 TB/s at Qwen3-30B-A3B's projections, scaling in bfloat16), gives some of
  that back. An MLP projection's forward and backward take 1.3-1.35x less time, q's and
  o's 1.1x less, and k's and v's, half as wide, 7% more.
- **NVFP4.** cuDNN's NVFP4 GEMM runs at 4.8-5.5 PFLOP/s at the MLP's shapes, 1.7-1.9x
  MXFP8's. Its output is bfloat16, which the two tensors' scales then rescale in
  whatever reads it. The quantizers are Transformer Engine's kernels where TE is
  installed: one pass finds the largest magnitude of the operand and of its transpose
  through the Hadamard transform, then one cast writes both copies, the transpose's by
  a Hadamard GEMM fused with the cast. Elsewhere, and for x and dy of fewer than 8192
  rows per device, two passes of bux's Pallas kernels do the same, rounding by
  Blackwell's `cvt.rn` and `cvt.rs` (stochastic, 8 random bits per value). By TE's
  kernels, an MLP projection's forward and backward take 1.17x less time than in
  MXFP8, but q's and o's take 1.08x more and k's and v's 1.25x more: about 20 small
  kernels per projection (scales for cuDNN's layout, the global scales) cost more there
  than the GEMMs save.

The dense 1.3B, 8 rows of 4096 per GPU, on 2026-10-23:

| `matmuls` | 1 GPU | 8 GPUs, `mesh(fsdp=8)` |
|-----------|-------|------------------------|
| `compute` (bfloat16) | 142k tok/s | 1.12M |
| `mxfp8` | 167k (1.18x) | 1.30M (1.16x) |
| `nvfp4` | 179k (1.22x) | |

The `nvfp4` row was measured separately, beside bfloat16 at 147k and `mxfp8` at 169k;
its 1.22x is against that 147k. On 300 steps of packed documents, with autotuning
pinned, the loss in `mxfp8` is bfloat16's to within the noise between seeds.

Under `mxfp8`, an MoE's routed experts multiply in MXFP8 too, by the grouped MXFP8
GEMMs (`roles.experts`), and its other matmuls are few. The benchmarks' MoE, 64
experts of 704, top 8, on 2026-10-24:

| `precision`, the MoE | 1 GPU | 8 GPUs, `mesh(fsdp=8)` |
|----------------------|-------|------------------------|
| `compute` (bfloat16) | 105k tok/s, 134.0 GiB | 824k, 83.8 GiB |
| `matmuls=mxfp8, roles=roles(experts=compute)`: the experts in bfloat16 | 108k, 133.9 GiB | 837k, 83.5 GiB |
| `matmuls=mxfp8` | 116k (1.10x), 119.4 GiB | 909k (1.10x), 69.1 GiB |

Which projections multiply in a low-precision format, the 1.3B on one B300, on
2026-10-23 ([Precision](../training/precision.md) says what each setting means):

| `precision(matmuls=mxfp8, ...)` | tok/s | Peak memory |
|---------------------------------|-------|-------------|
| `compute` (bfloat16) | 144k | 70.6 GiB |
| `mxfp8` | 167k | 66.3 GiB |
| `mxfp8_2d` | 165k | 66.3 GiB |
| `roles=roles(kv=compute)` | 164k | 68.4 GiB |
| `first=2, last=2`, of 32 branches | 161k | 66.9 GiB |
| `passes=[fprop, dgrad]` | 152k | 69.9 GiB |
| `moments=bfloat16` | 169k | 61.6 GiB |
| `nvfp4, first=2, last=8` | 167k | 46.1 GiB |
| `nvfp4` | 179k | 43.8 GiB |

- **The `nvfp4` rows** were measured with the loss by chunks (`methods.loss`), the
  others with whole logits. With chunks, bfloat16 trains at 147k in 51.1 GiB, and
  `mxfp8` at 169k in 46.9.
- **NVIDIA's recipe**, its first 2 and last 8 branches in bfloat16, is as fast as
  `mxfp8`; NVFP4 everywhere is 1.06x it.
- **k and v in bfloat16 are slower, not faster.** q still quantizes x, the copy all
  three share, so k's and v's GEMMs run at bfloat16's rate, and x is kept for them too.
- **Over 8 GPUs**, `mesh(fsdp=8)`, `mxfp8` and `mxfp8_2d` both train at 1.34M
  (bfloat16: 1.14M). Scanned, `first=2, last=2` trains at 1.20M, where `mxfp8` alone
  gives 1.22M.

What does not help:

- **A dense SwiGLU fused with the MXFP8 quantizer.** Backward, one Pallas pass that
  writes both gradients in MXFP8 took 0.63 ms at the 1.3B's shapes, against 0.62 ms
  for XLA's fusion plus the quantizer. Forward it could save at most 0.15 ms per layer,
  1% of the step. bux leaves it to XLA.
- **cuDNN's own global scale for NVFP4** is no faster, and takes the scale in bfloat16.
- **TE's NVFP4 GEMM** (`te_gemm_v2_ffi`, cuBLASLt) runs the same nvjet kernels as
  cuDNN's at the same speed, and its scales' swizzles are no faster than XLA's, so bux
  keeps cuDNN's (measured GEMM by GEMM, not in a step).
- **TE's NVFP4 quantizers below 8192 rows per device.** Each cast is a few more graph
  nodes, which a step of few tokens waits on: at a row of 4096 one GPU trained 2%
  slower by them, and 8 GPUs under FSDP 23% slower, waiting on each other at each
  operand's largest magnitude. So there `nvfp4` quantizes x and dy by the Pallas
  kernels (`nvfp4_te.ROWS`), and trains as fast as before (129k on one GPU, 825k on
  8).

::::{dropdown} NVFP4 quantizers: Transformer Engine against Pallas (2026-10-26)

TE 2.20.2 against bux's Pallas kernels, on the same B300s. Each operand of the 1.3B's
projections, 32768 rows, both copies with the largest magnitudes, timed by CUPTI:

| Operand | TE | Pallas |
|---------|----|--------|
| x, 2048 columns | 98 µs | 129 µs (1.31x) |
| x, 5632 | 223 µs | 330 µs (1.48x) |
| x, 1024 | 46 µs | 68 µs (1.48x) |
| dy, 2048, rounded stochastically | 118 µs | 147 µs (1.25x) |
| dy, 5632 | 271 µs | 376 µs (1.39x) |

A projection's forward and backward at 32768 rows takes 0.910 ms by TE against 1.052
by Pallas for the MLP's up and gate, 0.881 against 1.021 for its down, 0.474 against
0.540 for q and o, and 0.345 against 0.391 for k and v (MXFP8: 1.069, 1.002, 0.437,
0.275), in 25 kernels against 33. The one-GPU suite's `nvfp4` run quantizes in 28.4 ms
of a 168 ms step, against 35.7 of 172. The 1.3B in `nvfp4`, in the same memory either
way:

| Setup | Pallas | TE |
|-------|--------|----|
| 1 GPU, 8 rows of 4096 | 201k tok/s | 206k (1.02x) |
| 8 GPUs, `mesh(fsdp=8)`, 8 rows per GPU | 1.52M | 1.56M (1.03x) |
| 8 GPUs, 4 rows per GPU | 1.38M | 1.42M (1.02x) |
| 8 GPUs, 2 rows per GPU | 1.16M | 1.17M (1.01x) |

The one-GPU rows are the means of ten pairs of 60-step runs, the two kernels on the
same GPUs in turn.
::::
