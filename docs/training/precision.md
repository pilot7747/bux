# Precision

The `precision` setting says what a run computes in, bfloat16 or float32, and which
of its matmuls multiply in a low-precision format, MXFP8 or NVFP4. You pick the format
by kind of projection, by residual branch, by GEMM, and by step. Use this page to
train in FP8 or FP4, to choose a published recipe, and to see what each costs in
loss. How each format quantizes, how gradients are reduced and how it all splits over
devices is in the parallelism guide
([Precision and matmul formats](../parallelism/precision.md)).

```bux
precision = precision(compute=bfloat16, matmuls=mxfp8,
                      roles=roles(kv=compute), first=2, last=2)
```

Here every projection inside a block multiplies in MXFP8, except attention's keys and
values and the first two and last two residual branches of the model, which stay in
bfloat16. Parameters stay float32 and are cast to `compute` inside each step.

`precision` changes what a run computes, so it is in the run's id, except for its
`eval` field ([Run ids](running.md#run-ids)).

## The settings

| Field | Default | What it says |
|-------|---------|--------------|
| `compute` | `bfloat16` | the dtype the forward and backward compute in: `bfloat16` or `float32` (float16 would need loss scaling) |
| `matmuls` | `compute` | the format the projections inside blocks multiply in: `compute`, `mxfp8`, `mxfp8_2d` or `nvfp4` |
| `roles` | `roles` | a format for a kind of projection ([Roles](#roles)) |
| `first`, `last` | `0` | the residual branches at the model's start and end that stay in `compute` ([First and last](#first-and-last)) |
| `passes` | `[fprop, dgrad, wgrad]` | the GEMMs of each projection that multiply in the format ([Passes](#passes)) |
| `switch` | `none` | other `passes` from a share of the steps on ([Switch](#switch)) |
| `eval` | `train` | evaluation multiplies as training does, or in `compute` |
| `moments` | `float32` | the dtype the optimizer's moments are kept in between steps |
| `reduce` | `compute` | the dtype gradients are summed over the batch's rows in: `compute`, or `float32` as Megatron's are |
| `dispatch` | `compute` | the MoE's exchanges send tokens to their experts in `compute`, or `mxfp8` |

Parameters, their gradients and the optimizer's moments stay float32 (the moments
unless `moments=bfloat16`), and so do a few parts of every step, such as norm
statistics and the loss ([Float32 islands](../parallelism/precision.md#float32-islands)).

## Matmul formats

`matmuls` names a format of {py:mod}`bux.architecture.precision.matmuls`:

| Format | What it multiplies in |
|--------|-----------------------|
| `compute` | the run's `compute` dtype: the default |
| `mxfp8` | E4M3 values with a power-of-two scale per 32, the gradients too, as in Transformer Engine's MXFP8 recipe |
| `mxfp8_2d` | as `mxfp8`, but W's scales per 32 x 32 tile, the same for y and dx, as in TorchTitan |
| `nvfp4` | E2M1 values with an E4M3 scale per 16 and a float32 scale per tensor, W in 16 x 16 tiles, dy rounded stochastically, and a Hadamard transform for dW, as in NVIDIA's NVFP4 recipe |

On a Blackwell GPU, cuDNN's block-scaled GEMMs multiply them. Elsewhere, XLA
dequantizes and multiplies, which computes the same values, so every format also
trains on a CPU. A format is a function, so a new one is registered under its name
([Writing your own format](../parallelism/precision.md#writing-your-own-format)).

On a B200 or B300 with Transformer Engine installed (bux's Docker image has it),
`nvfp4` quantizes with TE's kernels wherever each device has at least 8192 rows of x,
as NVIDIA's recipe does. TE rounds the Hadamard transform to bfloat16 before it
quantizes, and draws its own noise for dy's stochastic rounding. So about one value
in a hundred of xᵀ lands one step away from where bux's own kernels put it, which
does not move the loss past the noise ([Quality of the formats](#quality-of-the-formats),
[Low-precision matmuls](../performance/kernels.md#low-precision-matmuls)).

The embedding, the head, the router and softmax attention's own matmuls always stay
in `compute` ([why attention does](#softmax-attention-in-fp8)).

## Roles

```bux
precision = precision(matmuls=mxfp8, roles=roles(kv=compute, o=compute))
```

`roles` gives a kind of projection its own format; a role left `none` multiplies in
`matmuls`. Each layer assigns a role to each of its projections
(`sharding.linear(..., role="kv")`):

| Role | The projections |
|------|-----------------|
| `q` | attention's queries and output gate: gqa's, mla's, csa's, the indexer's |
| `kv` | attention's keys and values |
| `o` | attention's output projection |
| `mlp` | the MLPs, the shared experts' too |
| `latent` | LatentMoE's projections to and from its latent |
| `recurrence` | the projections of the linear attentions, RWKV-7 and Mamba, except the output |
| `recurrence_out` | their output projection, Mamba's `out_proj` |
| `mtp` | nextn's projection, and every branch after the first head |
| `experts` | the routed experts, by a grouped GEMM: `compute`, `mxfp8` or `mxfp8_2d` |

`experts` left `none` follows `matmuls` where a grouped GEMM takes that format
(`mxfp8` or `mxfp8_2d`), and is `compute` otherwise, since no grouped GEMM takes
`nvfp4`. In MXFP8 the routed experts train 10-15% faster than in bfloat16 on B300s
([The experts in MXFP8](../parallelism/experts.md#the-experts-in-mxfp8)).

## First and last

```bux
precision = precision(matmuls=nvfp4, first=2, last=8)
precision = precision(matmuls=nvfp4, last=0.15)
```

`first` and `last` keep that many residual branches in `compute`: at the start of the
model, and at its end before the first head. Each is a number of branches, or a share
of them, rounded up (`last=0.15`).

A branch here is a `[...]` that is inside no other, so a Llama-style block,
`[norm attn] [norm ffn]`, counts as two. Where `first` or `last` falls inside a repeat:

- a scanned repeat (`methods(repeat=scan)`) is split into runs of passes that
  multiply alike, each scanned separately;
- a pipelined repeat must multiply alike in every pass, so keep `first` and `last` in
  the repeats before or after it.

## Passes

```bux
precision = precision(matmuls=mxfp8, passes=[fprop, dgrad])
```

A projection y = x W has three GEMMs: `fprop` computes y, `dgrad` the gradient by x,
and `wgrad` the gradient by W. `passes` lists the ones that multiply in the format.
The others multiply in `compute`, and the forward keeps their operands in `compute`.

## Switch

```bux
precision = precision(matmuls=nvfp4, first=2, last=8,
                      switch=switch(at=0.82, passes=[dgrad, wgrad]))
```

From step `round(at · steps)` on, `switch` replaces `passes` with its own, and the
step compiles again. Here fprop multiplies in `compute` for the last 18% of the
steps, as NVIDIA's NVFP4 recipe does. With `passes=[]`, every GEMM does.

## Eval, dispatch, moments and reduce

- `eval=compute` evaluates in `compute`, whatever training multiplies in. It is not
  in the run's id.
- `dispatch=mxfp8` has the MoE's exchanges send tokens to their experts in MXFP8, as
  DeepSeek-V3 does. What comes back, and the gradients, travel in `compute`
  ([FP8 dispatch](../parallelism/experts.md#fp8-dispatch-precisiondispatch)).
- `moments=bfloat16` keeps the optimizer's moments in bfloat16 between steps, as
  DeepSeek-V3's AdamW does, and updates them in float32. Rounded to nearest, a
  bfloat16 moment ignores a change smaller than 2^-9 of itself: at `b2 = 0.95`, a
  squared gradient 4% off the second moment still moves it, but at `b2 = 0.999` only
  one 3x off does.
- `reduce=float32` sums the gradients over the batch's rows in float32, once per
  step, as Megatron does. It needs a step that adds its gradients up unreduced
  ([Gradient reductions](../parallelism/precision.md#gradient-reductions-precisionreduce)).

## Recipes

A precision recipe is a module in `examples/experiments/recipes/` that sets only
`precision`. A run takes it by name next to a training recipe, and can amend it:

```bux
use llama3_8b
use fineweb_wsd
use nvidia_nvfp4
precision.last = 4 | 8
```

| Recipe | Its precision | Left out |
|--------|---------------|----------|
| `nvidia_mxfp8` | Transformer Engine's MXFP8: E4M3 everywhere, gradients too, the routed experts too | |
| `nvidia_nvfp4` | NVIDIA's NVFP4 pretraining: W in 16 x 16 tiles, dy rounded stochastically, a Hadamard transform for dW; `first=2, last=8` | NVFP4 routed experts |
| `nemotron3_nvfp4` | NVFP4, `recurrence_out` in MXFP8, and `q`, `kv`, `o`, `latent`, `mtp` and the last 15% in bfloat16 | NVFP4 routed experts, most of its matmuls |
| `deepseek_v3_fp8` | `mxfp8_2d`, the routed experts too, the tokens sent to them in MXFP8, moments in bfloat16 | groups of 128 with float32 scales, the float32 promotion, activations kept in FP8 |
| `torchtitan_mxfp8` | `mxfp8_2d`, the routed experts too | saving a linear's input in bfloat16 |

Each recipe's header comment says what it takes from the paper or library it follows,
and what it leaves out.

## Measurements

How fast each format and each choice of projections trains is in
[GPU kernels](../performance/kernels.md). This section measures what they cost in
loss.

### Quality of the formats

On one B300 per run (2026-10-23), the 1.3B Llama-style model trained 3000 steps of
8 x 4096 tokens of FineWeb-Edu (98M tokens), with GEMM autotuning off. The table gives
its held-out loss at the end, evaluated in bfloat16 (`eval=compute`), relative to
bfloat16's at the same seed. Two seeds differ by 1.6%.

| precision | seed 0 | seed 1 | tok/s |
|-----------|--------|--------|-------|
| `mxfp8` | +0.12% | -0.02% | 126k |
| `deepseek_v3_fp8` | +0.15% | -0.03% | 127k |
| `moments=bfloat16` | +0.07% | -0.01% | 116k |
| `mxfp8, first=2, last=2` | +0.09% | | 127k |
| `nvfp4` | +0.75% | | 135k |
| `nvidia_nvfp4` | +0.48% | +0.56% | 129k |
| `nvidia_nvfp4` and its switch at 0.82 | +0.46% | +0.51% | 127k |
| `nvidia_nvfp4`, `last=4` † | +0.46% | | 44k |
| `nvfp4, last=0.15`, `q`, `kv` and `o` in compute † | +0.34% | | 56k |

bfloat16's own losses were 4.0383 (seed 0) and 3.9696 (seed 1). For comparison,
NVIDIA measured a gap of 1.5% against FP8 at the end of 10T tokens on a 12B model, and
0.5% with the switch.

- NVFP4 quantized by bux's Pallas kernels here. By Transformer Engine's kernels,
  which quantize it on B300s where TE is installed, the same `nvfp4` runs
  (2026-10-26) end at 4.0678, 4.0025 and 4.0214 at seeds 0, 1 and 2, against 4.0683,
  3.9988 and 4.0260 by the Pallas kernels: at most 0.11% apart, where the seeds
  differ by 1.7%.
- † These rows quantized by XLA rather than by the Pallas kernels, hence their
  throughput. Their gaps hold either way: NVIDIA's recipe lands between +0.41% and
  +0.53% whether XLA or the kernels quantize, and a change in the last bit moves a gap
  by up to 0.15 points.

### The routed experts in MXFP8

The baseline's MoE (64 experts, top-8, 16 layers of `d_model` 2048) trained the same
3000 steps on two B300s per run (2026-10-23), its experts split over them
(`ragged_all_to_all`, no capacity). With `roles=roles(experts=mxfp8)` and
`dispatch=mxfp8`, its held-out loss is 0.13% and 0.05% below bfloat16's at seeds 0
and 1, where the seeds differ by 1.65%. With the `grouped_mxfp8` kernels it also
trains faster than in bfloat16
([The experts in MXFP8](../parallelism/experts.md#the-experts-in-mxfp8)).

## Softmax and cross-entropy at large margins

A softmax's largest probability is 1 / (1 + e^-m), where the margin m is the largest
logit minus the logsumexp of the others. Past m ≈ 16.6 in float32 (5.5 in bfloat16),
1 + e^-m rounds to 1. The cross-entropy of a target that wins by that much is then 0,
and so is its gradient by the target's logit, p - 1. The other logits' gradients are
not 0, so the gradient no longer sums to 0. Barron et al. (2026, "A Fix for Softmax
and Cross-Entropy") trace the loss spikes of Adam's slingshot effect to this, and fix
it by never adding e^-m to 1. bux applies their fix, with no setting to turn on:

- **The loss.** Each position's cross-entropy is `softplus(others - target)`, from
  the target's logit and the logsumexp of the other logits, and the target's gradient
  is `-sigmoid(others - target)`, under either `methods.loss`
  ({py:class}`~bux.architecture.logits.Scores`). The z-loss's log Z is their
  `logaddexp`.
- **Every other softmax a layer differentiates**, through {py:mod}`bux.softmax`: the
  MoE router's, CSA's gate, mHC's Sinkhorn and attnres, the indexer's KL, and
  attention by `xla`. The forward is jax's softmax, bit for bit. The backward first
  subtracts the largest entry's cotangent from every entry's, which zeroes that
  entry's term of the sum and changes nothing else. The log-softmax at the largest
  entry is `-log1p` of the others' mass.

A layer of your own takes them from there:

```python
from bux.softmax import log_softmax, softmax
```

Flash attention keeps jax's rounding: cuDNN's and tokamax's kernels, and the ring and
sparse attention built on them. Their backward recomputes P = exp(s - lse) one block
of keys at a time, and the fix needs two things they do not return: each query's
largest key, and the sum of its row of dS over every block.

**What it costs.** On B300s (2026-10-26), 0.05% to 0.42% of end-to-end throughput,
over 60 steps of 8 x 4096 random tokens on one GPU per run: -0.13% and -0.32% in two
measurements of the 1.3B dense model, -0.32% for a 5.2B MoE of 64 experts, and -0.42%,
-0.05% and -0.31% for the 1.3B with DSA, CSA or mHC as its attention or residual. The
loss of 8 x 4096 tokens, and its gradients by x and W, are the same to six digits.

::::{dropdown} Kernel timings with jax's softmax and with the fix
On one B300, 2026-10-26, each pair measured in turn:

| | jax's | with the fix |
|---|---|---|
| loss by `chunks`, `d_model` 2048, vocabulary 128K | 46.6 ms | 46.9 ms (+0.8%) |
| the same with a z-loss | 73.4 ms | 73.7 ms (+0.5%) |
| loss by `chunks`, `d_model` 4096, vocabulary 152K | 103.1 ms | 103.3 ms (+0.2%) |
| loss by `logits`, either size | | -1.8% to +0.3% |
| softmax forward and backward, (8, 16, 2048, 2048) | 2.50 ms | 2.07 ms |
| the same, (8, 4096, 2048) | 0.355 ms | 0.337 ms |
| the same, CSA's gate: 4 entries, not the last axis | 0.378 ms | 0.505 ms |
| log-softmax forward and backward, (8, 4096, 2048) | 0.371 ms | 0.387 ms |
::::

## Softmax attention in FP8

bux keeps softmax attention in `compute`. Megatron-Bridge's recipe runs it in FP8 by
cuDNN (`fp8_dot_product_attention`), which runs on B300s through jax 0.12. But no
variant that bux can run without keeping state across steps trains both faster and
as well:

- **The FP8 backward** gives dQ, dK and dV in E4M3 only, with scales that must be set
  before the kernel finds their amaxes. Transformer Engine sets them from earlier
  steps' amaxes, which bux does not keep, and a bound that is safe at any step
  overflows E4M3.
- **An FP8 forward with a bfloat16 backward diverges**: the gradients of the 1.3B passed
  1e10 and reached NaN within 650 to 1130 steps, in all 11 runs. It trains within the
  noise only if the backward takes the output and logsumexp of a second, bfloat16
  forward.
- **The FP8 forward pays off only in long rows**, and so does MXFP8 attention
  (Megatron-Bridge's B300 recipe), which XLA cannot call.

::::{dropdown} Measurements of FP8 and MXFP8 attention
On B300s with jax 0.12 and cuDNN 9.27, 2026-10-24 and 2026-10-25.

**The FP8 backward.** XLA takes no dtype but E4M3 for dQ, dK and dV beside FP8
inputs. A scale bound that is safe at any step lies up to t h / g above dK's and dV's
(a key that every query attends to), past E4M3's range. jax 0.12 also passes that
backward's scales in another order than XLA reads them, which its tests, with every
scale 1, cannot see.

**The FP8 forward with a bfloat16 backward.** q, k and v at their current scaling (a
power of two from each one's amax), the probabilities scaled by 1 and the output by
half of v's scale, then cuDNN's bfloat16 backward on q, k and v as the forward
rounded them: the gradients of the 1.3B grew past 1e10 and to NaN within 650 to 1130 steps
of FineWeb-Edu in all 11 runs (seeds 0 to 2, by cuDNN's backward or tokamax's,
subnormals kept or flushed). The backward takes the FP8 forward's output, for the
rows' sums of dO times it, and its logsumexp. With either one alone from a bfloat16
forward, it still diverged. With both from a bfloat16 forward of the rounded inputs,
it trained within the noise of bfloat16 (held-out loss at 1500 steps +0.06% and
-0.05%, where seeds differ by 0.6%), as q, k and v rounded alone did. cuDNN's FP8
backward computes those sums from its own probabilities.

**The FP8 forward's speed.** A layer of 32 heads of 4 takes 0.82 ms against
bfloat16's 0.84 in 8 rows of 4096, and 3.7 against 5.8 ms in one row of 32k. The
stable backward above adds a bfloat16 forward.

**MXFP8 attention** is E4M3 in blocks of 32 with a power-of-two scale per block,
forward and backward. cuDNN runs it at a head dim of 128 by its graph API
(`sdpa_mxfp8` and `sdpa_mxfp8_backward`), which XLA has no call for; a
`buffer_callback` would run them, as it runs the GDN kernel. Forward, q and k are in
blocks along the head dim and v along the positions; backward, q, k, v and dO are
each in blocks along what the backward's products contract. Against bfloat16's cuDNN
kernels, for a layer of 32 heads of 4:

| rows | forward | backward | both, quantized by XLA | error, of float32's |
|------|---------|----------|------------------------|---------------------|
| 8 of 4096 | 0.49 ms, 1.35x | 2.58 ms, 1.0x | 4.06 ms against 3.85 | 5% (bfloat16: 0.2%) |
| 4 of 8192 | 1.09, 1.52x | 4.89, 1.08x | 6.51 against 6.92 | |
| 1 of 32k | 3.98, 1.54x | 16.7, 1.15x | 21.2 against 25.2 | |

On a Blackwell GPU, cuDNN runs the MXFP8 backward only by its deterministic
algorithm, in two kernels. Quantizing for it and for the forward takes XLA 1.0 ms
per layer, whatever the rows. Outputs and gradients are 5-7% off float32's, and 9-13%
with q four times larger.
::::
