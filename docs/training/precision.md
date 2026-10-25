# Precision

The `precision` setting says what a run computes in, bfloat16 or float32, and which
of its matmuls multiply in a low-precision format, MXFP8 or NVFP4: by the kind of
projection, by the residual branch it is in, by the GEMM, and by the step. Use this
page to train in FP8 or FP4, or to choose a published recipe.

```bux
precision = precision(compute=bfloat16, matmuls=mxfp8,
                      roles=roles(kv=compute), first=2, last=2)
```

A run computes in `compute`. Its parameters stay float32 and are cast to `compute`
inside each step, and the projections inside blocks multiply in `matmuls`. Here every
projection inside a block multiplies in MXFP8, but attention's keys and values and the
first two and last two residual branches of the model, which stay in bfloat16.

`precision` changes what a run computes, so it is in the run's id, all but its `eval`
([Run ids](running.md#run-ids)). How each format quantizes, and where every float32
island of a run is, is in the parallelism guide
([Precision and matmul formats](../parallelism/precision.md)).

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
unless `moments=bfloat16`), and so do norm statistics, router scores and top-k,
recurrent states and the loss.

## Matmul formats

`matmuls` names a format of {py:mod}`bux.architecture.precision.matmuls`:

| Format | What it multiplies in |
|--------|-----------------------|
| `compute` | the run's `compute` dtype: the default |
| `mxfp8` | E4M3 values with a power-of-two scale per 32, the gradients too, as Transformer Engine's MXFP8 recipe |
| `mxfp8_2d` | as `mxfp8`, but W's scales per 32 x 32 tile, the same for y and dx, as TorchTitan's |
| `nvfp4` | E2M1 values with an E4M3 scale per 16 and a float32 scale per tensor, W in 16 x 16 tiles, dy rounded stochastically and a Hadamard transform for dW, as NVIDIA's NVFP4 recipe |

On a Blackwell GPU cuDNN's block-scaled GEMMs multiply them; elsewhere XLA dequantizes
and multiplies, which computes the same values, so a format trains on a CPU too. A
format is a function, so a new one is registered under its name
([Matmul formats](../parallelism/precision.md#matmul-formats)).

The embedding, the head, the router and softmax attention's own matmuls always stay in
`compute` ([why attention does](#softmax-attention-in-fp8)).

## Roles

```bux
precision = precision(matmuls=mxfp8, roles=roles(kv=compute, o=compute))
```

`roles` gives a kind of projection a format of its own; a role left `none` multiplies
in `matmuls`. Each layer gives each of its projections its role
(`sharding.linear(..., role="kv")`):

| Role | The projections |
|------|-----------------|
| `q` | attention's queries and output gate: gqa's, mla's, csa's, the indexer's |
| `kv` | attention's keys and values |
| `o` | attention's output projection |
| `mlp` | the MLPs, the shared experts' too |
| `latent` | LatentMoE's projections to and from its latent |
| `recurrence` | the linear attentions', RWKV-7's and Mamba's projections, but the output |
| `recurrence_out` | their output projection, Mamba's `out_proj` |
| `mtp` | nextn's projection, and every branch after the first head |
| `experts` | the routed experts, by a grouped GEMM: `compute`, `mxfp8` or `mxfp8_2d` |

`experts` left `none` is `matmuls` where a grouped GEMM takes it, `mxfp8` or
`mxfp8_2d`, and else `compute`, as no grouped GEMM takes `nvfp4`. In MXFP8 the routed
experts train 10-15% faster than in bfloat16 on B300s
([The experts in MXFP8](../parallelism/experts.md#the-experts-in-mxfp8)).

## First and last

```bux
precision = precision(matmuls=nvfp4, first=2, last=8)
precision = precision(matmuls=nvfp4, last=0.15)
```

`first` and `last` keep that many residual branches in `compute`, at the start of the
model and at its end before the first head. Each is a number of branches, or a share
of them, rounded up (`last=0.15`).

A branch here is a `[...]` inside no other, so a Llama-style block,
`[norm attn] [norm ffn]`, is two. A scanned repeat scans apart the passes that
multiply alike; a pipelined one must multiply alike throughout.

## Passes

```bux
precision = precision(matmuls=mxfp8, passes=[fprop, dgrad])
```

A projection y = x W has three GEMMs: `fprop`, y's, and the gradient's by x, `dgrad`,
and by W, `wgrad`. `passes` are those that multiply in the format. The others multiply
in `compute`, and the forward keeps their operands as they are.

## Switch

```bux
precision = precision(matmuls=nvfp4, first=2, last=8,
                      switch=switch(at=0.82, passes=[dgrad, wgrad]))
```

`switch` takes up other `passes` from step `round(at · steps)` on, and compiles the
steps again. Here fprop multiplies in compute for the last 18% of the steps, as
NVIDIA's NVFP4 recipe does; with `passes=[]` all of it does.

## Eval, dispatch, moments and reduce

- `eval=compute` evaluates in `compute`, whatever training multiplies in. It is not in
  the run's id.
- `dispatch=mxfp8` has the MoE's exchanges send the tokens to their experts in MXFP8,
  as DeepSeek-V3 does, and what comes back, and the gradients, in `compute`
  ([FP8 dispatch](../parallelism/experts.md#fp8-dispatch-precisiondispatch)).
- `moments=bfloat16` keeps the optimizer's moments in bfloat16 between steps, as
  DeepSeek-V3's AdamW does, and updates them in float32. Rounded to nearest, a moment
  ignores a change below 2^-9 of itself: at `b2 = 0.95`, a square gradient 4% off the
  second moment moves it, at 0.999 only one 3x off.
- `reduce=float32` sums the gradients over the batch's rows in float32, once a step,
  as Megatron does. It needs the step to add its gradients up unreduced
  ([Gradient reductions](../parallelism/precision.md#gradient-reductions-precisionreduce)).

## Recipes

A recipe is a module in `examples/experiments/recipes/` that sets `precision`, which a
run uses by name and amends:

```bux
use llama3_8b
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

Each recipe's file says in its header comment what it takes from the paper or the
library it follows, and what it leaves out.

## Measurements

How fast each format and each choice of projections trains is in
[GPU kernels](../performance/kernels.md); this section measures what they cost in
loss.

### Quality of the formats

On one B300 each, the 1.3B Llama-style model trained 3000 steps of 8 x 4096 tokens of
FineWeb-Edu (98M tokens), GEMM autotuning off. Its held-out loss at the end, in
bfloat16 (`eval=compute`), is this much above bfloat16's of the same seed, where two
seeds differ by 1.6%:

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

NVIDIA measured 1.5% at the end of 10T tokens on a 12B, against FP8, and 0.5% with the
switch. NVFP4 quantizes by its kernels here; bfloat16's losses beside them were 4.0383
and 3.9696.

The rows marked † quantized by XLA, which on a GPU folds the cast to E4M3 and back: it
scaled each block's values by its scale unrounded, and the GEMM by the rounded one, up
to 6% apart (`nvfp4.quantize` keeps them alike now). That moved nothing past the
noise: NVIDIA's recipe quantized by XLA gives +0.44% and +0.51%, with the fold +0.44%
and +0.41%, and by the kernels with a float32 GEMM output +0.50% and +0.53%. A change
at the ulp moves a gap by up to 0.15 points.

### The routed experts in MXFP8

The baseline's MoE (64 experts, top-8, 16 layers of `d_model` 2048) trained the same
3000 steps on two B300s a run, its experts split over them (`ragged_all_to_all`, no
capacity). With `roles=roles(experts=mxfp8)` and `dispatch=mxfp8` its held-out loss is
0.13% and 0.05% below bfloat16's at seeds 0 and 1, where the seeds differ by 1.65%.
It trained at 105k tokens a second against 127k by Triton's grouped GEMMs, in 75 GiB
against 89. By `grouped_mxfp8`'s kernels, which compute the same values, it trains
faster than bfloat16 ([The experts in MXFP8](../parallelism/experts.md#the-experts-in-mxfp8)).

## Softmax attention in FP8

Megatron-Bridge's recipe runs softmax attention in FP8 by cuDNN
(`fp8_dot_product_attention`); bux keeps it in `compute`. cuDNN's FP8 attention runs on
B300s through jax 0.12, but no recipe that bux can run stateless trains both faster
and as well.

**Its backward** gives dQ, dK and dV in E4M3 only (XLA takes no other dtype beside FP8
inputs), and their scales must be set before the kernel finds their amaxes.
Transformer Engine scales them by earlier steps' amaxes, which bux does not keep; a
bound safe at any step lies up to t h / g above dK's and dV's (a key that every query
attends to), past E4M3's range. jax 0.12 also passes that backward's scales in another
order than XLA reads them, which its tests, every scale 1, cannot see.

**Its forward with a backward in bfloat16 diverges.** With q, k and v at their current
scaling (a power of two from each one's amax), the probabilities by 1 and the output
by half of v's scale, then cuDNN's bfloat16 backward on q, k and v as the forward
rounded them, the 1.3B's gradients grew past 1e10 and to NaN within 650 to 1130 steps
of FineWeb-Edu in all 11 runs (seeds 0 to 2, by cuDNN's backward or tokamax's,
subnormals kept or flushed).

The backward takes the FP8 forward's output, for the rows' sums of dO times it, and
its logsumexp. Each alone, the other from bfloat16, still diverged. With both from a
bfloat16 forward of the rounded inputs it trained within the noise of bfloat16
(held-out loss at 1500 steps +0.06% and -0.05%, where seeds differ by 0.6%), as q, k
and v rounded alone did. cuDNN's FP8 backward computes those sums from its own
probabilities.

**Its forward pays only in long rows.** A layer of 32 heads of 4 takes 0.82 ms to
bfloat16's 0.84 in 8 rows of 4096, and 3.7 to 5.8 in one of 32k; the stable backward
above adds a bfloat16 forward.

**MXFP8 attention**, which Megatron-Bridge's B300 recipe runs (E4M3 in blocks of 32
with a power-of-two scale a block, forward and backward), pays only in long rows too.
cuDNN 9.27 runs it on B300s at a head dim of 128 by its graph API (`sdpa_mxfp8` and
`sdpa_mxfp8_backward`, which XLA has no call of: a `buffer_callback` would run them,
as it runs the GDN kernel). Forward, q and k are in blocks along the head dim and v
along the positions; backward, q, k, v and dO are each in blocks along what the
backward's products contract. Against bfloat16's cuDNN kernels, a layer of 32 heads of
4:

| rows | forward | backward | both, quantized by XLA | error, of float32's |
|------|---------|----------|------------------------|---------------------|
| 8 of 4096 | 0.49 ms, 1.35x | 2.58 ms, 1.0x | 4.06 ms against 3.85 | 5% (bfloat16: 0.2%) |
| 4 of 8192 | 1.09, 1.52x | 4.89, 1.08x | 6.51 against 6.92 | |
| 1 of 32k | 3.98, 1.54x | 16.7, 1.15x | 21.2 against 25.2 | |

On a Blackwell GPU cuDNN runs the MXFP8 backward by its deterministic algorithm alone,
two kernels; quantizing for it and the forward takes XLA 1.0 ms a layer whatever the
rows. Outputs and gradients are 5-7% off float32's, 9-13% with q four times larger.
