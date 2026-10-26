(parallelism-precision)=
# Precision and matmul formats

This page is how bux computes in a lower precision, and what it measured: which
arrays stay float32, how gradients are reduced, and how the block-scaled matmul
formats (`mxfp8`, `mxfp8_2d`, `nvfp4`) multiply. To choose a precision for a run,
start with [Precision](../training/precision.md) in the training guide, which covers
every field of {bux}`precision` from the user's side.

Unlike `mesh`, `layout` and `methods`, `precision` changes what a run computes, so it
is in the run's id: `precision.compute = float32 | bfloat16` is a sweep of two runs.

## Turn it on

Llama 3 8B under FSDP on a node of 8 GPUs, computing in bfloat16, its projections in
MXFP8:

```bux
use llama3_8b
use fineweb_wsd

mesh      = mesh(fsdp=8)
precision = precision(compute=bfloat16, matmuls=mxfp8)   # in the run's id
methods   = methods(repeat=scan, remat=dots, microbatches=4)   # not in it
```

The fields this page explains:

| Setting | Values | Default | What it does |
|---|---|---|---|
| `precision.compute` | `bfloat16`, `float32` | `bfloat16` | the dtype the forward and backward compute in; float16 would need loss scaling |
| `precision.matmuls` | a format: `compute`, `mxfp8`, `mxfp8_2d`, `nvfp4`, or your own | `compute` | the format the projections inside blocks multiply in |
| `precision.roles.experts` | `compute`, `mxfp8`, `mxfp8_2d` | none | the routed experts' format ([The experts in MXFP8](experts.md#the-experts-in-mxfp8)) |
| `precision.reduce` | `compute`, `float32` | `compute` | the dtype gradients are summed over the batch's rows in |
| `precision.dispatch` | `compute`, `mxfp8` | `compute` | what the MoE's exchanges send tokens in ([FP8 dispatch](experts.md#fp8-dispatch-precisiondispatch)) |

## Precision

Parameters, their gradients and the optimizer's state stay float32 (the optimizer's
state unless `precision.moments = bfloat16`). The train and eval steps cast the
model's floating-point parameters to `compute` inside the loss (`trainer.cast`), so
that the forward and the backward compute in it, and gradients reach the float32
parameters through the cast. What is not a parameter keeps its dtype: the router bias
and its counts, RNG state, Lightning's slopes.

### Float32 islands

Some parts compute from what they are given in float32, and give back its dtype:

| Where | What is float32 |
|---|---|
| `Norm.normalize`, `l2norm` | the statistics, and the normalized values |
| the MoE's router | its logits, scores and top-k, the balance loss and the bias; the weights go back to the stream's dtype |
| `recurrence`, `delta_rule`, `selective_scan` | the recurrent state, and its decays |
| `rope.rotate` | the rotation |
| the head | its matmul's accumulation, tied or not, and its log Z and loss; by chunks, the default `methods.loss`, its logits |

Softmax attention takes its softmax in float32, under every method of
`methods.attention` ([Attention](attention.md)). Every other matmul runs in `compute`,
or in the format of `matmuls` ([below](#matmul-formats)). The stream stays in
`compute` from the embedding to the heads, which `tests/test_precision.py` checks for
every family. What bfloat16 saves of a step's memory, and what it does to the loss, is
measured in [Memory and losses](microbatches.md#memory-and-losses).

### Gradient reductions: `precision.reduce`

Gradients add up over microbatches in float32. Over the rows' axes they are reduced
in `precision.reduce`: `compute`, the default, or `float32`, as Megatron's are.

- **In `compute`**, they are reduced as the backward makes them: in bfloat16 for a
  weight's, but the head kernel's by chunks (the default `methods.loss`), which adds
  up over the chunks in float32. Where a step adds its gradients up unreduced
  ([Gathered once, reduced once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
  each device's sum is rounded to `compute` before the one reduction.
- **In `float32`**, that reduction sums float32, twice the bytes, once a step. On
  B300s, through the NICs, a step of the dense 1.3B trains 5% slower in it.

`reduce = float32` changes what a run computes, so it is in the run's id; a run that
leaves it at `compute` keeps the id it had.

:::{warning}
`reduce = float32` needs the step to add its gradients up unreduced, also at one
microbatch: XLA reduces those of the backward in the dtype of the matmul that makes
them. So it is an error:

- for a run whose layers learn from the whole batch, such as an MoE's;
- for a pipeline's embedding and head under `gpipe` and `circular`, which run outside
  the pipeline and reduce in the backward;
- with `methods(gather=layer)` or `methods(accumulate=reduced)`, which reduce as the
  backward makes the gradients.

A pipeline that reduces once sums its stages' gradients in float32 either way, as
their parameters are whole in float32; `1f1b` and `zbh1` sum every gradient in
float32 as it leaves, so there it is no error ([Pipelines](pipelines.md)).
:::

### bfloat16 psums on a CPU

XLA:CPU fails to compile a bfloat16 `psum` that a `shard_map`'s transpose makes (its
reducer ends in a copy, which `ChangeOpDataType` cannot clone), and aborts the
process. On more than one device these make one:

- the MoE's `sorted` and expert-parallel methods (every method but `dense`);
- the pipelines;
- sparse attention (the indexer, CSA) split over `context`.

`ring`, `ulysses` and `pass_state` train in bfloat16 on a CPU. A run that would make
such a psum says so before it compiles. On a CPU, set `precision(compute=float32)`, as
`examples/experiments/parallel/laptop.bux` does, or split an MoE by
`methods(moe=dense)`. GPUs all-reduce bfloat16 natively.

## Matmul formats

`matmuls` names a format of the registry in `bux.architecture.precision.matmuls`. A
block-scaled format quantizes the operands of all three GEMMs of y = x W, the
forward's and the backward's two, each along what its GEMM contracts, in blocks with a
scale each (`bux.architecture.precision.blockscaled`):

| Format | Values | A scale per | And |
|---|---|---|---|
| `compute` | `compute` | none | the default |
| `mxfp8` | E4M3 | 32, a power of two (E8M0) | dy in E4M3 too, as Transformer Engine's MXFP8 recipe |
| `mxfp8_2d` | E4M3 | 32 of x and dy; W's per 32 x 32 tile, quantized by XLA | so that y and dx see the same W, as TorchTitan's |
| `nvfp4` | E2M1 | 16 (E4M3), and the tensor (float32); W's per 16 x 16 tile, so that y and dx see the same W | dy rounded stochastically, and x and dy transformed for dW by a random Hadamard transform of 16, as NVIDIA's NVFP4 pretraining recipe |

### How they multiply

- **On a Blackwell GPU**, cuDNN's block-scaled GEMM multiplies them
  (`__cudnn$blockScaledDot` in the HLO), and Pallas kernels quantize each of x, W and
  dy and its transpose: for `mxfp8` in one pass, for `nvfp4` in two, the first for
  the largest magnitudes that set its global scales
  (`bux.architecture.precision.nvfp4`).
- **Elsewhere**, XLA dequantizes and multiplies, which computes the same values, so
  the tests run the formats on a CPU.
- The forward keeps what the backward multiplies, x^T and W in the format, rather
  than x and W.

### Which matmuls take the format

Every projection made by `sharding.projection` or `sharding.linear` multiplies by
`matmuls.dot_general`, which runs the format in force for its role and the residual
branch it is in ([Precision](../training/precision.md)):

- attention's q, k, v, o and gate;
- the MLPs', the shared experts' and LatentMoE's;
- MLA's, CSA's and the indexer's;
- the linear attentions', RWKV-7's and Mamba's;
- Engram's, nextn's and `linear`'s.

The routed experts multiply in `compute` unless their role says `mxfp8` or
`mxfp8_2d`, which a grouped GEMM of their own takes
([The experts in MXFP8](experts.md#the-experts-in-mxfp8)).

These stay in `compute`:

- the embedding and the head;
- softmax attention's own matmuls, and its relative bias;
- the router, and the shared experts' gate;
- hyper-connections and norms;
- low-rank gates and decays, and recurrent states;
- `methods(moe=dense)`'s einsums;
- a projection whose tokens, inputs or outputs are not whole blocks.

### The same blocks on any mesh

On a mesh, a projection that does not split into whole blocks on every device (a
multiple of the block times the devices) stays whole on each. So the blocks are the
same on any mesh, and so is what a run computes, which `tests/test_precision.py`
checks on `fsdp=4, tensor=2` and in a pipeline.

NVFP4's stochastic rounding draws its noise from a hash of each group of 4 elements'
index and the batch's tokens, 8 bits an element: the same on any mesh too. NVFP4's
kernels each take the place of their device's part in the whole. Transformer Engine's
kernels, which quantize NVFP4 on a B200 or B300 where TE is installed, draw Philox's
noise of the seed and each device's part instead: alike on another mesh in its
distribution, not in its values.

### Formats under FSDP and context parallelism

- **Under FSDP** each device quantizes its part of W, so the all-gathers move float8
  values and their scales, half of bfloat16's bytes.
- **Under context parallelism** the blocks do not follow the split: a projection
  multiplies its tokens flattened, rows by positions, and a split along positions is
  not one along the flattened tokens. A `ring` run on `context=2` trained 13% slower
  in `mxfp8` than in bfloat16 on B300s.

### Writing your own format

A recipe is a function, and the trainer enters `matmuls.use` as it traces its steps:

```python
from bux.architecture.precision import matmuls


@matmuls.formats.register("mine")
def _(lhs, rhs, dimension_numbers, **kwargs): ...  # jax.lax.dot_general's arguments
```

`matmuls.block_scaled(recipe)` makes one of a `blockscaled.Recipe`: how it quantizes
x, W and dy, each twice.
