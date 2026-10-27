(parallelism-tensor)=
# Tensor parallelism

Tensor parallelism (TP) splits each layer's weights over the `tensor` axis, as
Megatron does: attention by head, an MLP by hidden unit, the embedding and the head by
the vocabulary. A device multiplies the whole input by its own columns, and the output
projection's partial sums add up over `tensor`. Use it when a layer's weights or
activations are too large for FSDP alone to train fast, typically for a 70B-class
model, and keep `tensor` inside a node, where NVLink carries its collectives.

With sequence parallelism, `layout(stream=tensor)`, the residual stream between
branches also splits by position over `tensor`, so the norms and the residual adds run
on a share of the positions and nothing is held whole between layers.

## Turn it on

Set `mesh.tensor`, and split the stream over it for sequence parallelism. From
`examples/experiments/parallel/tp_70b.bux`, Llama 3 70B on 4 nodes of 8 GPUs:

```bux
use llama3_70b
use fineweb_wsd

mesh      = mesh(fsdp=4, tensor=8)          # TP inside a node, FSDP across the nodes
layout    = layout(stream=tensor)           # sequence parallelism
precision = precision(compute=bfloat16)
methods   = methods(repeat=scan, remat=full, microbatches=8)
optimizer.lr = 1.5e-4
```

To overlap TP's collectives with the matmuls they feed, add `tensor=chunked` to the
methods, as `examples/benchmarks/one_node.bux`'s `tp` arm does:

```bux
methods = methods(repeat=scan, remat=full, microbatches=8, tensor=chunked, tensor_chunks=2)
```

## Options

The mesh and the layout say what splits; {bux}`methods` says how the collectives
at a layer's edges run.

| Setting | Values | Default | What it does |
|---|---|---|---|
| `mesh.tensor` | a size | `1` | the devices each layer's weights split over |
| `layout.stream` | axes | `context` | `tensor` (or `[context, tensor]`) adds sequence parallelism |
| `layout.heads`, `kv_heads`, `mlp`, `vocab` | axes | `tensor` | what TP splits; see [Names](layout.md#names) |
| `methods.tensor` | `gspmd`, `chunked`, or your own | `gspmd` | how the collectives meet the projections |
| `methods.tensor_chunks` | a count | `2` | the chunks of each device's rows under `chunked` |

The two methods of `methods.tensor`:

| Method | What it does |
|---|---|
| `gspmd` | The default. The compiler's collectives, at the boundary of the matmul each feeds or drains, which waits for it. |
| `chunked` | Each device's rows in `tensor_chunks` chunks, in a `shard_map` manual over every axis: chunk i + 1 all-gathers while chunk i multiplies, and chunk i reduce-scatters while chunk i + 1 multiplies. |

:::{tip}
`gspmd` stays the default: in bfloat16 `chunked` gains little, and on
`fsdp=2, tensor=4` it is 3% slower. In MXFP8 (`precision(matmuls=mxfp8)`) `chunked`
is 5% faster, so use it there ([Measurements](#on-8-b300s)).
:::

## Constraints and errors

- `tensor` should stay inside a node: the mesh puts it innermost
  ([The mesh](index.md#the-mesh)), and an axis that does not fill nodes from `tensor`
  outward is an error ([Many nodes](multi-node.md)).
- A dimension whose count `tensor` does not divide stays whole and is computed on
  every device, as GQA's 2 kv heads under `tensor=4` are
  ([How names resolve](layout.md#how-names-resolve)).
- `chunked` falls back to `gspmd`'s collectives, projection by projection:
  - where it has nothing to overlap: without sequence parallelism, for the input
    projections;
  - where a kernel splits otherwise than by `tensor`, or `tensor` splits the layer's
    rows or positions, as `layout(seq=tensor)` does;
  - where a device's share would multiply otherwise than the whole, in the format that
    projection takes in its branch (its role's, or `compute` in the `first` and `last`
    branches): in `mxfp8` and `mxfp8_2d` unless the chunk's tokens and the share are
    whole blocks, and in `nvfp4` always, whose scale is the whole operand's.
- Where `tensor_chunks` does not divide a device's rows, `chunked` takes as many chunks
  as divide both (their greatest common divisor).

## How it works

At a layer's edges, TP communicates:

- **Under sequence parallelism**, a device gathers the layer's input over `tensor`
  before the projections that split by columns, and the output projection's partial
  sums reduce-scatter back to its positions.
- **Without it**, the input is already whole on every device, and the partial sums
  all-reduce.

GQA's q, k and v (or its fused `qkv`) and the MLPs' up and gate, the shared experts'
too, project their input by `sharding.project_input`; their output projections, and
MLA's, by `sharding.project_output`. Both call the run's method of `methods.tensor`.
The other layers gather their input as `gspmd` does.

Under `chunked`, the collectives run per chunk:

```text
project_in:   gather chunk 0 | gather chunk 1    | ...
                             | multiply chunk 0  | multiply chunk 1 | ...
project_out:  multiply chunk 0 | multiply chunk 1 | ...
                               | reduce chunk 0   | reduce chunk 1   | ...
```

Each chunk's collective is NCCL's, at full NVLink speed, and an optimization barrier
ties it to the one before, so that XLA neither combines them nor runs them side by
side.

How each layer names its weights for TP is on the pages of the layers:
[softmax attention](attention.md#how-attention-splits-by-head), and
[the MLP, the embedding, the head and the recurrent mixers](layout.md#the-other-layers).

## Writing your own method

A method of `methods.tensor` is a class of two static functions:

```python
from bux.parallel import methods, tensor


@methods.tensor.register("mine")
class Mine:
    @staticmethod
    def project_in(x, linears):  # -> the input as the layer takes it, [l(x), ...]
        return tensor.GSPMD.project_in(x, linears)

    @staticmethod
    def project_out(h, linear):  # -> linear(h), split as the stream is
        return linear(h)
```

`methods = methods(tensor=mine)` then picks it.

## Measurements

### On 8 CPU devices

`chunked` gives `gspmd`'s outputs and gradients to 1e-5, in 1, 2 and 4 chunks:

- on `fsdp=2, tensor=4` and `tensor=8` with sequence parallelism;
- on `fsdp=2, context=2, tensor=2` with the stream over both;
- on `data=4, tensor=2` without sequence parallelism;
- in `mxfp8` and `mxfp8_2d` too.

With `nvfp4` for the MLPs, or for all but the first and last branches, those
projections stay `gspmd`'s, and the gradients they take from chunked branches move
some of NVFP4's values a step: they agree to 1e-3. A Llama (scanned, rematerialized,
in 2 microbatches) and an MoE (rematerialized) train as on one device, with no
involuntary rematerialization.

(tp-on-8-b300s)=
### On 8 B300s

Four of Llama 3 70B's layers (scanned, `remat=full`, 8 rows of 4096), TP=8 with
sequence parallelism. Tokens a second, and the exposed communication of a step in ms
of its time:

| `methods.tensor` | bfloat16 | `matmuls=mxfp8` |
|---|---|---|
| `gspmd` | 219k, 36.5 of 150 | 250k, 41.8 of 131 |
| `chunked`, 2 | **222k**, 28.2 of 148 | **263k**, 26.5 of 125 |
| `chunked`, 4 | 219k, 23.6 of 150 | 257k, 25.3 of 128 |

Chunks hide a quarter to a third of the wait, and the matmuls beside them give most of
it back:

- NCCL's ring kernels take 32 channels of SMs, so the GEMMs of a step take 82 ms
  beside 2 chunks' collectives and 88 beside 4's, against 79 with nothing beside
  them.
- The chunks' slices and concatenations take 2.6 to 3.4 ms.
- With fewer CTAs (`NCCL_MAX_CTAS=8`) NCCL's collectives take more than twice as long.

On `fsdp=2, tensor=4`, `chunked` is 3% slower (232k against 240k). In MXFP8 it is 5%
faster: `gspmd`'s block-scaled output projections all-reduce their partial sums
(32 ms a step), where `chunked` reduce-scatters them. So `gspmd` stays the default,
and `chunked` is TP's method in MXFP8.

### What did not gain

- The MLP by chunks from its gather to its reduce-scatter.
- Each scanned pass on two halves of its rows side by side: it hides more (20 ms
  exposed with `chunked`), at 218k.
- XLA's windowed einsum, `compiler(preset=collective_matmul)`: 93k
  ([XLA's options](../performance/compiler.md)).
- XLA's NCCL symmetric buffers,
  `compiler(options="--xla_gpu_experimental_enable_nccl_symmetric_buffers=true")`,
  make each run 1.5 to 2.7% faster (224k by `gspmd`; 227k by `chunked` with
  `NCCL_CTA_POLICY=2`; 266k by `chunked` in MXFP8), but every such run ends its 40
  steps at a loss of 12.058 rather than 12.075: NCCL's symmetric kernels reduce
  otherwise.

### Not yet: a collective matmul off the SMs

A collective matmul whose communication takes no SMs is not there yet:

- TransformerEngine's (`dense(..., collective_op_set=...)`, its userbuffers on copy
  engines) is on PyPI only as source that builds against CUDA 12 (2.20.2), and would
  bring its own GEMMs and NCCL communicators.
- jax's Pallas kernel (`pallas.ops.gpu.collective_matmul_mgpu`) sends each tile of x
  on to the next device from shared memory, by remote refs, as its GEMM multiplies
  it, but only on Hopper, and through NVSHMEM, which jax 0.12's XLA does not enable.
  On a B300 it takes a port to tcgen05 (`blackwell_matmul_mgpu`) and a reduce-scatter
  in the epilogue of the output projection.
