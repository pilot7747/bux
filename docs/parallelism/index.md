(parallelism)=
# Parallelism

A run splits over devices by three settings: the **mesh** (how many devices along each
of six axes), the **layout** (which axes each named dimension splits over) and the
**methods** (how each part that communicates runs). None of them changes what a run
computes, so none is in its id: a run resumes on another mesh, and a sweep over meshes
is still several runs, as their names differ.

```bux
use llama3_8b
use fineweb_wsd

mesh    = mesh(fsdp=8, tensor=2)              # data takes the devices the others leave
layout  = layout(stream=[context, tensor])    # sequence parallelism
methods = methods(attention=ring)
```

Any of the three can be swept, as any setting can:

```bux
mesh       = fsdp: mesh(fsdp=16) | tp: mesh(fsdp=8, tensor=2)   # the grid, as a sweep
layout.seq = context | none
```

## The three settings

| Setting | What it says | Default | Reference |
|---|---|---|---|
| `mesh` | a size per axis: `stage`, `data`, `fsdp`, `context`, `expert`, `tensor` | every device on `data` | {bux}`mesh` |
| `layout` | which axes each named dimension splits over | see [Layout](layout.md) | {bux}`layout` |
| `methods` | how each part that communicates is computed, by name | the one-device code | {bux}`methods` |

`bux.parallel` reads them; the configs are
{py:class}`~bux.train.config.MeshConfig`, {py:class}`~bux.train.config.LayoutConfig`
and {py:class}`~bux.train.config.MethodsConfig` in `bux/train/config.py`.

## Which parallelism do I need

| Your situation | Set | Page |
|---|---|---|
| The model fits on a GPU, and you want more GPUs | nothing: `data` takes every device (DDP) | [The mesh](#the-mesh) |
| The parameters and Adam's state do not fit, or you want them off each GPU | `mesh(fsdp=8)` (FSDP, ZeRO-3), or `layout(zero=1)` for ZeRO-1 | [Layout](layout.md) |
| Several nodes | `mesh(data=<nodes>, fsdp=8)` (HSDP) | [Many nodes](multi-node.md) |
| Very wide layers, as a 70B's | `mesh(tensor=8)` and `layout(stream=tensor)`, inside a node | [Tensor parallelism](tensor.md) |
| Rows too long for a GPU's memory | `mesh(context=8)`; `methods(attention=...)`, `methods(recurrence=pass_state)` | [Attention](attention.md), [Recurrent mixers](recurrence.md) |
| A mixture of experts | `methods(moe=sorted)` under FSDP, the default; or `mesh(expert=8)` and an EP method | [Experts](experts.md) |
| Too many layers, or slow links between nodes | `mesh(stage=4)`, `methods(repeat=scan, microbatches=8)` | [Pipelines](pipelines.md) |
| Activations do not fit | `methods(remat=...)` (`auto` by default), `methods(microbatches=n)` | [Remat](remat.md), [Microbatches](microbatches.md) |
| Muon on a split model | `methods(orthogonalize=owner)`, the default | [Muon](muon.md) |
| Matmuls in MXFP8 or NVFP4 | `precision(matmuls=mxfp8)` | [Precision](precision.md) |

:::{tip}
On one node of 8 B300s, at the shapes measured on these pages, FSDP is the fastest
split of the 1.3B dense model, and `sorted` under FSDP the fastest of its 64-expert
MoE. A pipeline trains 47% slower a GPU than FSDP on one node (42% on four), in its
bubble and its microbatches' smaller GEMMs. The [performance guide](../performance/tuning.md) collects the numbers.
:::

## Examples

Each file in `examples/experiments/parallel/` is a run, and `bux plan` shows its GPUs,
the bytes of its parameters and of its optimizer's state on each, and its pipeline
bubble, without any GPU.

| File | What it splits, and over what |
|---|---|
| `laptop.bux` | a tiny hybrid six ways over 8 CPU devices: DP, FSDP, TP, EP, CP, PP |
| `fsdp_8b.bux` | Llama 3 8B, FSDP over 8 GPUs, bfloat16, scanned and rematerialized |
| `mesh_sweep.bux` | a 1B Llama five ways over 8 GPUs: DDP, FSDP, ZeRO-1, HSDP, TP |
| `tp_70b.bux` | Llama 3 70B over 32 GPUs: TP and sequence parallelism, FSDP across nodes |
| `pp_70b.bux` | the same over 32 GPUs: TP, and a circular pipeline across nodes |
| `ep_moe.bux` | a 16B DeepSeek-V3-shaped MoE over 16 GPUs: EP and FSDP |
| `long_context.bux` | a Qwen3-Next-shaped hybrid on 64k-token rows over 8 GPUs: ring and `pass_state` |

`laptop.bux` trains on a laptop, its CPU split into 8 devices:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    bux train examples/experiments/parallel/laptop.bux
```

and `bux plan` shows its six runs, one id for all, as they compute the same:

```console
$ bux plan examples/experiments/parallel/laptop.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
90b55fa67f  497k    10.2k   0.0        1.74M      8     1.89MiB     3.79MiB    0%      laptop[dp]
90b55fa67f  497k    10.2k   0.0        1.74M      8     243KiB      485KiB     0%      laptop[fsdp]
90b55fa67f  497k    10.2k   0.0        1.74M      8     261KiB      521KiB     0%      laptop[tp]
90b55fa67f  497k    10.2k   0.0        1.74M      8     243KiB      485KiB     0%      laptop[ep]
90b55fa67f  497k    10.2k   0.0        1.74M      8     485KiB      970KiB     0%      laptop[cp]
90b55fa67f  497k    10.2k   0.0        1.74M      8     259KiB      517KiB     20%     laptop[pp]
```

## The mesh

One mesh, always of all six axes, in this order: the slowest links first, so that
`tensor` stays inside a node, and `stage` before `data`, as a pipeline's sends need
the least bandwidth ([Pipelines](pipelines.md) says why it must).

| Axis | What splits over it, by default | Parallelism |
|---|---|---|
| `stage` | a pipelined repeat's layers, a share a stage | pipeline (PP) |
| `data` | rows; every parameter is copied | data parallel (DDP), HSDP across nodes |
| `fsdp` | rows, and each parameter and its optimizer state | FSDP (ZeRO-3) |
| `context` | positions of each row; parameters, as one of FSDP's axes | context parallel (CP) |
| `expert` | experts; rows; parameters, as one of FSDP's axes | expert parallel (EP) |
| `tensor` | heads, hidden units and the vocabulary | tensor parallel (TP) |

An axis not set has size 1, and `data` takes the devices the others leave. Every axis
is Auto: layers do not say how anything splits, and the compiler propagates shardings
from the parameters and a few constraints ([Layout](layout.md)).

`bux train` uses every visible device; `bux launch` gives a run the GPUs its mesh
needs ([below](#gpus-hosts-and-the-plan)). These are errors before anything is built:

- a mesh that needs more devices than there are;
- axes other than `data` whose devices do not divide those there are, where `data` is
  unset (set `mesh.data`);
- a `batch` that does not split over the batch axes (and into `methods.microbatches`);
- a `seq_len` that does not split over `seq` or `stream`.

## GPUs, hosts and the plan

`bux launch` gives a run the GPUs of its mesh: the product of its axes' sizes, where
`data` unset is every visible GPU for a run alone and one in a sweep. A sweep trains
its runs side by side, each on a group of free GPUs, and a run that needs more GPUs
than there are fails at once.

A run alone also trains on many hosts, one process each, when Slurm, Open MPI or
`JAX_NUM_PROCESSES` say so ([Launching](../training/launching.md)). The mesh then
spans every process's devices, each node holding a block of its innermost axes
([Many nodes](multi-node.md)).

- Each process reads the rows of a batch that its devices hold
  (`parallel.local_rows`), as a batch depends only on the seed and the step.
- `parallel.shard_batch` puts them there with `jax.make_array_from_callback`, just
  before the step that reads it; a thread loads the next two on the host
  (`prefetch`).
- `tests/test_launch.py` trains the smoke run as two processes of 4 CPU devices each,
  through `jax.distributed` on localhost, as one process of 8 does.

`bux plan` shows what each run takes before it starts:

- its FLOPs a token ([the log](../training/running.md));
- its GPUs;
- the bytes of its parameters and of its optimizer's state on each;
- the share of a step its pipeline's stages idle, the bubble
  ([Pipelines](pipelines.md)).

The bytes come from `parallel.shardings` of the model built over a
`jax.sharding.AbstractMesh`, which `parallel.use` takes as it does a mesh, so the plan
needs no devices. They are the state's alone: a step's activations and new state come
on top.

```console
$ bux plan examples/experiments/smoke.bux 'mesh = fsdp: mesh(fsdp=8) | one: mesh(data=1)'
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
f114ac17ae  156k    25.6k   0.2        984k       8     76.2KiB     152KiB     0%      smoke[fsdp]
f114ac17ae  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[one]
```

## Methods

What communicates by hand does it inside a method: a function in a registry of
`bux.parallel.methods`, one per field of `methods`, picked by name.
`parallel.method("attention")` returns the run's method, or the default outside
`use`. A name that is not registered is an error at `use`. Each kind's page gives its
signature, and the default is registered next to the layer that calls it.

| Fields of `methods` | Default | Page |
|---|---|---|
| `attention` | `auto` | [Attention and context parallelism](attention.md) |
| `recurrence`, `states` | `scan`, `keep` | [Recurrent mixers](recurrence.md) |
| `tensor`, `tensor_chunks` | `gspmd`, `2` | [Tensor parallelism](tensor.md) |
| `moe`, `capacity`, `moe_chunks`, `moe_stream` | `sorted`, none, `1`, `false` | [Experts](experts.md) |
| `orthogonalize` | `owner` | [Muon](muon.md) |
| `repeat`, `remat` | `unroll`, `auto` | [Repeats and remat](remat.md) |
| `microbatches`, `gather`, `accumulate`, `interleave`, `donate` | `1`, `step`, `auto`, `false`, `true` | [Microbatches](microbatches.md) |
| `pipeline`, `rounds`, `stages` | `1f1b`, `1`, balanced | [Pipelines](pipelines.md) |
| `loss` | `chunks` | [The other layers](layout.md#their-activations) |

The names each registry holds are listed in the
[registries reference](../reference/language/registries.md).

A method that communicates by hand does it in a `jax.shard_map` over its own axis,
`axis_names={axis}`, the other axes staying Auto. One that sorts tokens, as the MoE's
do, is manual over every axis that splits them, since a sort in the global view
gathers every token onto every device. What reads the whole batch (MoE routing,
balance losses, the router bias) stays outside, in the global view, so its semantics
do not change.

### Writing a method

Another method is a function:

```python
from bux.parallel import methods


@methods.attention.register("mine")
def _(q, k, v, args): ...
```

and `methods = methods(attention=mine)` picks it. Registering a function under a name
bux already uses replaces bux's own.

## In a script

```python
from bux import parallel
from bux.train import stages

# The run's methods, with a pipeline's stages balanced as `train` balances them.
with parallel.use(t.mesh, t.layout, stages.balanced(run)):
    model = parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0)))
    trainer = Trainer(model, tx, loss)  # its state on its shardings
    trainer.train_on_batch(batch)  # the batch through parallel.shard_batch
```

`parallel.build` creates every parameter on its shards
([Born on their shards](layout.md#born-on-their-shards)).

## Testing

The tests run on 8 CPU devices (`tests/conftest.py` sets
`--xla_force_host_platform_device_count=8`), and compare a run split over them with
one on a single device:

- `losses(source, mesh="mesh(fsdp=8)")` trains a few steps and returns the losses;
- `involuntary()` counts the times XLA warned that it rematerialized an array in
  full, which a complete layout should never make it do.

```{toctree}
:maxdepth: 2
:hidden:

multi-node
layout
attention
tensor
recurrence
experts
muon
remat
microbatches
precision
pipelines
```
