(parallelism-precision)=
# Precision and matmul formats

This page covers precision on a mesh: the dtype gradients are reduced in across
devices, how block-scaled formats (`mxfp8`, `mxfp8_2d`, `nvfp4`) keep the same blocks
on any split, what they do to FSDP's gathers and to context parallelism, and the one
CPU limit on bfloat16 collectives. It is also the reference for how bux computes in a
lower precision: which parts stay float32, and how each format multiplies.

To choose a precision or a recipe, start with [Precision](../training/precision.md)
in the training guide. It covers every field of {bux}`precision`, and what each format
does to the loss.

Unlike `mesh`, `layout` and `methods`, `precision` changes what a run computes, so it
is in the run's id: `precision.compute = float32 | bfloat16` is a sweep of two runs.

## Quick start

Llama 3 8B under FSDP on a node of 8 GPUs, computing in bfloat16, its projections in
MXFP8:

```bux
use llama3_8b
use fineweb_wsd

mesh      = mesh(fsdp=8)
precision = precision(compute=bfloat16, matmuls=mxfp8)   # in the run's id
methods   = methods(repeat=scan, remat=dots, microbatches=4)   # not in it
```

The fields of {bux}`precision` that meet the mesh:

| Setting | Values | Default | What it does on a mesh |
|---|---|---|---|
| `precision.compute` | `bfloat16`, `float32` | `bfloat16` | the dtype of the forward and backward passes, and of the parameters FSDP gathers; on a CPU, see [below](#bfloat16-psums-on-a-cpu) |
| `precision.matmuls` | `compute`, `mxfp8`, `mxfp8_2d`, `nvfp4`, or your own | `compute` | the format the projections inside blocks multiply in; FSDP gathers it ([below](#formats-on-a-mesh)) |
| `precision.reduce` | `compute`, `float32` | `compute` | the dtype gradients are summed over the batch's rows in ([below](#gradient-reductions-precisionreduce)) |
| `precision.dispatch` | `compute`, `mxfp8` | `compute` | what the MoE's exchanges send tokens in ([FP8 dispatch](experts.md#fp8-dispatch-precisiondispatch)) |
| `precision.roles.experts` | `compute`, `mxfp8`, `mxfp8_2d` | none | the routed experts' format ([The experts in MXFP8](experts.md#the-experts-in-mxfp8)) |

## Gradient reductions: `precision.reduce`

Gradients add up over microbatches in float32. Over the rows' axes they are reduced
in `precision.reduce`: `compute`, the default, or `float32`, as Megatron reduces them.

- **In `compute`**, gradients are reduced as the backward makes them: in bfloat16 for
  a weight's gradient. The exception is the head kernel's under the default
  `methods.loss` (by chunks), which adds up over the chunks in float32. Where a step
  adds up its gradients unreduced
  ([Gathered once, reduced once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
  each device's sum is rounded to `compute` before the single reduction.
- **In `float32`**, the single reduction per step sums float32: twice the bytes. On
  B300s, with every collective through the NICs, a step of the dense 1.3B trains 5%
  slower in it
  ([Through the NICs](microbatches.md#gathered-once-reduced-once-through-the-nics)).

`reduce = float32` changes what a run computes, so it is in the run's id. A run that
leaves it at `compute` keeps its id.

:::{warning}
`reduce = float32` needs the step to add up its gradients unreduced, even at one
microbatch, because XLA reduces the backward's gradients in the dtype of the matmul
that makes them. Where an axis splits the rows, it is an error:

- for a run with a layer that learns from the whole batch, such as an MoE's router;
- for a pipeline under `gpipe` or `circular`, whose embedding and head run outside the
  pipeline and reduce in the backward;
- with `methods(gather=layer)` or `methods(accumulate=reduced)`, which reduce as the
  backward makes the gradients.

`1f1b` and `zbh1` sum every gradient in float32 as it leaves, so `reduce = float32`
is no error under them. A pipeline that reduces once also sums its stages' gradients in float32 either
way, because their parameters are gathered whole in float32
([Pipelines](pipelines.md)).
:::

## bfloat16 psums on a CPU

XLA:CPU cannot compile a bfloat16 `psum` made by a `shard_map`'s transpose, and the
process aborts. On more than one CPU device, these make one:

- the MoE's `sorted` and expert-parallel methods (every method except `dense`);
- the pipelines;
- sparse attention (the indexer, CSA) split over `context`.

bux checks for these before it compiles, and the error says what to set. On a CPU,
set `precision(compute=float32)`, as `examples/experiments/parallel/laptop.bux` does,
or split an MoE by `methods(moe=dense)`. `ring`, `ulysses` and `pass_state` train in
bfloat16 on a CPU, and GPUs all-reduce bfloat16 natively.

## Formats on a mesh

### The same blocks on any mesh

On a mesh, a projection that does not split into whole blocks on every device (a
multiple of the block size times the devices) stays whole on each. So the blocks are
the same on any mesh, and so is what a run computes. `tests/test_precision.py` checks
this on `fsdp=4, tensor=2` and in a pipeline.

NVFP4's stochastic rounding draws its noise from a hash of each group of 4 elements'
index and the batch's tokens, 8 bits per element, so it is the same on any mesh too.
Each of NVFP4's kernels computes its device's part of the whole.

Transformer Engine's kernels, which quantize NVFP4 on a B200 or B300 where TE is
installed, draw Philox noise from the seed and each device's part instead. On another
mesh their noise has the same distribution, but not the same values.

### Formats under FSDP, TP and context parallelism

- **Under FSDP**, each device quantizes its part of W, so the all-gathers move float8
  values and their scales: half the bytes of bfloat16.
- **Under TP**, `methods(tensor=chunked)` overlaps the collectives in MXFP8, where
  `gspmd`'s block-scaled output projections all-reduce
  ([Tensor parallelism](tensor.md)).
- **Under context parallelism**, the blocks do not follow the split. A projection
  multiplies its tokens flattened, rows by positions, and a split along positions is
  not a split along the flattened tokens. A `ring` run on `context=2` trained 13%
  slower in `mxfp8` than in bfloat16 on B300s (2026-10-23).

## The compute dtype

Parameters, their gradients and the optimizer's state stay float32 (the optimizer's
state unless `precision.moments = bfloat16`). The train and eval steps cast the
model's floating-point parameters to `compute` inside the loss (`trainer.cast`). So
the forward and the backward compute in `compute`, and gradients reach the float32
parameters through the cast. What is not a parameter keeps its dtype: the router bias
and its counts, RNG state, Lightning's slopes.

### Float32 islands

Some parts compute in float32 from whatever they are given, and return the input's
dtype:

| Where | What is float32 |
|---|---|
| `Norm.normalize`, `l2norm` | the statistics, and the normalized values |
| the MoE's router | its logits, scores and top-k, the balance loss and the bias; the weights go back to the stream's dtype |
| `recurrence`, `delta_rule`, `selective_scan` | the recurrent state, and its decays |
| `rope.rotate` | the rotation |
| the head | its matmul's accumulation, tied or not, and its log Z and loss; by chunks, the default `methods.loss`, its logits |

Softmax attention takes its softmax in float32 under every method of
`methods.attention` ([Attention](attention.md)). Every other matmul runs in `compute`,
or in the format of `matmuls` ([below](#matmul-formats)). The stream stays in
`compute` from the embedding to the heads, which `tests/test_precision.py` checks for
every family.

## Matmul formats

`matmuls` names a format in the registry `bux.architecture.precision.matmuls`. A
block-scaled format quantizes the operands of all three GEMMs of y = x W (the
forward's and the backward's two), each along what its GEMM contracts, in blocks with
one scale each (`bux.architecture.precision.blockscaled`):

| Format | Values | A scale per | Also |
|---|---|---|---|
| `compute` | `compute` | none | the default |
| `mxfp8` | E4M3 | 32, a power of two (E8M0) | dy in E4M3 too, as in Transformer Engine's MXFP8 recipe |
| `mxfp8_2d` | E4M3 | 32 of x and dy; W's per 32 x 32 tile, quantized by XLA | y and dx see the same W, as in TorchTitan |
| `nvfp4` | E2M1 | 16 (E4M3), and the tensor (float32); W's per 16 x 16 tile, so y and dx see the same W | dy rounded stochastically; x and dy transformed for dW by a random Hadamard transform of 16, as in NVIDIA's NVFP4 pretraining recipe |

### How they multiply

- **On a Blackwell GPU**, cuDNN's block-scaled GEMM multiplies them
  (`__cudnn$blockScaledDot` in the HLO). Pallas kernels quantize each of x, W and dy
  and its transpose: for `mxfp8` in one pass, for `nvfp4` in two, the first finding
  the largest magnitudes that set its global scales
  (`bux.architecture.precision.nvfp4`).
- **Elsewhere**, XLA dequantizes and multiplies, which computes the same values, so
  the tests run the formats on a CPU.
- The forward keeps what the backward multiplies, x^T and W in the format, rather
  than x and W.

### Which matmuls take the format

Every projection made by `sharding.projection` or `sharding.linear` multiplies by
`matmuls.dot_general`, in the format for its role and its residual branch
([Roles](../training/precision.md#roles)):

- attention's q, k, v, o and gate;
- the MLPs', the shared experts' and LatentMoE's;
- MLA's, CSA's and the indexer's;
- the linear attentions', RWKV-7's and Mamba's;
- Engram's, nextn's and `linear`'s.

The routed experts multiply in `compute` unless their role says `mxfp8` or
`mxfp8_2d`, which their own grouped GEMM takes
([The experts in MXFP8](experts.md#the-experts-in-mxfp8)).

These stay in `compute`:

- the embedding and the head;
- softmax attention's own matmuls, and its relative bias;
- the router, and the shared experts' gate;
- hyper-connections and norms;
- low-rank gates and decays, and recurrent states;
- `methods(moe=dense)`'s einsums;
- a projection whose tokens, inputs or outputs are not whole blocks.

### Writing your own format

A format is a function, and the trainer enters `matmuls.use` as it traces its steps:

```python
from bux.architecture.precision import matmuls


@matmuls.formats.register("mine")
def _(lhs, rhs, dimension_numbers, **kwargs): ...  # jax.lax.dot_general's arguments
```

`matmuls.block_scaled(recipe)` makes a format from a `blockscaled.Recipe`, which says
how to quantize x, W and dy, each twice.

## Measurements

### bfloat16 against float32 on a CPU

2026-10-21, one CPU device. The loss in `float32` / `bfloat16` of two blocks of width
128 on the tests' corpus of eight words, byte by byte (`seq_len = 128`, `batch = 16`,
`adamw(lr=3e-3)`, cosine):

| Mixer | Step 10 | Step 50 | Step 200 |
|---|---|---|---|
| `gqa` | 2.2348 / 2.2389 | 0.4089 / 0.4089 | 0.4209 / 0.4213 |
| `moe` | 1.3595 / 1.3602 | 1.0057 / 1.0071 | 1.0250 / 1.0248 |
| `gdn` | 1.5034 / 1.4998 | 0.4112 / 0.4269 | 0.4205 / 0.4204 |
| `mamba2` | 0.9262 / 0.9276 | 0.4134 / 0.4129 | 0.4211 / 0.4207 |

What bfloat16 saves of a step's memory is in
[Memory on one CPU device](microbatches.md#memory-on-one-cpu-device). What each format
does to the loss at scale is in
[Quality of the formats](../training/precision.md#quality-of-the-formats).
