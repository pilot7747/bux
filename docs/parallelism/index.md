(parallelism)=
# Parallelism

This section shows how to split a run over GPUs, and which split to pick. Three
settings do it: the **mesh** says how many devices lie along each of six axes, the
**layout** says which axes each named dimension splits over, and the **methods** say
how each part that communicates runs. None of them changes what a run computes, so
none is in its id: a run resumes on another mesh, and a sweep over meshes is several
runs that share one id.

```bux
use llama3_8b
use fineweb_wsd

mesh    = mesh(fsdp=8, tensor=2)              # data takes the devices the others leave
layout  = layout(stream=[context, tensor])    # sequence parallelism
methods = methods(attention=ring)
```

## Which split to pick

Start from FSDP, and add an axis only when FSDP alone cannot fit the model or its
rows. By the GPUs you have and the model you train:

| | Dense | Mixture of experts | Long rows (32k tokens and up) |
|---|---|---|---|
| **One GPU** | nothing to set | nothing to set | nothing to set: `methods.attention` picks the fastest kernel |
| **One node** | `mesh(fsdp=8)` | `mesh(fsdp=8)`, with `methods.moe = sorted`, the default | `mesh(context=8)`; `methods(attention=ring)` for very long packed rows; `methods(recurrence=pass_state)` for recurrent mixers |
| **Many nodes** | `mesh(data=<nodes>, fsdp=8)` (HSDP) | HSDP and `sorted`; or EP inside a node, `mesh(data=<nodes>, expert=8)` and `methods(moe=nccl_ep)` | `mesh(data=<nodes>, context=8)`: `context` inside a node |

With no `mesh` at all, `data` takes every device: plain data parallelism (DDP), each
GPU holding a whole copy of the model.

When that is not enough:

| Your problem | Set | Page |
|---|---|---|
| The parameters and Adam's state do not fit, or you want them off each GPU | `mesh(fsdp=8)` (FSDP, ZeRO-3), or `layout(zero=1)` for ZeRO-1 | [Layout](layout.md) |
| Very wide layers, as in a 70B model | `mesh(tensor=8)` and `layout(stream=tensor)`, inside a node | [Tensor parallelism](tensor.md) |
| Rows too long for one GPU's memory | `mesh(context=8)`, with `methods.attention` (`auto` by default) and `methods(recurrence=pass_state)` | [Attention](attention.md), [Recurrent mixers](recurrence.md) |
| An MoE whose experts should not be gathered | `mesh(expert=8)` and an EP method of `methods.moe` | [Experts](experts.md) |
| A model too large for a node, with slow links between nodes | `mesh(stage=4)`, `methods(repeat=scan, microbatches=8)` | [Pipelines](pipelines.md) |
| Activations do not fit | `methods.remat` (`auto` by default), `methods(microbatches=n)` | [Remat](remat.md), [Microbatches](microbatches.md) |
| Muon on a split model | `methods(orthogonalize=owner)`, the default | [Muon](muon.md) |
| Matmuls in MXFP8 or NVFP4 on a mesh | `precision(matmuls=mxfp8)` | [Precision](precision.md) |

:::{tip}
On one node of 8 B300s, at the shapes measured in this section, FSDP is the fastest
split of the 1.3B dense model, and `sorted` under FSDP the fastest of its 64-expert
MoE. A pipeline trains 47% slower per GPU than FSDP on one node, and 42% slower on
four, because of its bubble and its microbatches' smaller GEMMs. The {ref}`tuning guide <tuning-mesh>` has the numbers
behind each choice.
:::

## Mesh, layout and methods

Each setting answers one question: how the devices form a grid, which dimensions
split over which of its axes, and how the parts that communicate run.

| Setting | What it says | Default | Reference |
|---|---|---|---|
| `mesh` | a size per axis: `stage`, `data`, `fsdp`, `context`, `expert`, `tensor` | every device on `data` | {bux}`mesh` |
| `layout` | which axes each named dimension splits over | see [Layout](layout.md) | {bux}`layout` |
| `methods` | how each part that communicates is computed, by name | the one-device code | {bux}`methods` |

Any of the three can be swept, like any setting:

```bux
mesh       = fsdp: mesh(fsdp=16) | tp: mesh(fsdp=8, tensor=2)
layout.seq = context | none
```

`bux.parallel` reads them. Their configs are
{py:class}`~bux.train.config.MeshConfig`, {py:class}`~bux.train.config.LayoutConfig`
and {py:class}`~bux.train.config.MethodsConfig` in `bux/train/config.py`.

### The mesh

A run has one mesh, always of all six axes, in this order. The slowest links come
first, so `tensor` stays inside a node, and `stage` comes before `data` because a
pipeline's sends need the least bandwidth ([Pipelines](pipelines.md) explains why
`stage` must come first).

| Axis | What splits over it, by default | Parallelism |
|---|---|---|
| `stage` | a pipelined repeat's layers, a share per stage | pipeline (PP) |
| `data` | rows; every parameter is copied | data parallel (DDP), HSDP across nodes |
| `fsdp` | rows, and each parameter and its optimizer state | FSDP (ZeRO-3) |
| `context` | positions of each row; parameters, as one of FSDP's axes | context parallel (CP) |
| `expert` | experts; rows; parameters, as one of FSDP's axes | expert parallel (EP) |
| `tensor` | heads, hidden units and the vocabulary | tensor parallel (TP) |

An axis you do not set has size 1, and `data` takes the devices the others leave.
Every axis is Auto: layers never say how anything splits, and the compiler propagates
shardings from the parameters and a few constraints ([Layout](layout.md)).

`bux train` uses every visible device; `bux launch` gives a run the GPUs its mesh
needs ([below](#gpus-hosts-and-the-plan)). These are errors before anything is built:

- a mesh that needs more devices than there are;
- with `data` unset, axes other than `data` whose devices do not divide those there
  are (set `mesh.data`);
- a `batch` that does not split into `methods.microbatches` and then over the batch
  axes;
- a `seq_len` that does not split over `seq` or `stream`.

### The layout

Layers name the dimensions of what they make (`embed`, `heads`, `mlp`, `experts`,
...), and the layout maps each name to mesh axes. FSDP then splits each parameter's
first whole dimension over `fsdp`, `context` and `expert`. The defaults give DDP,
FSDP, TP, EP and CP wherever the mesh has those axes; change the layout for sequence
parallelism, ZeRO-1, or to split something over another axis
([Layout](layout.md)).

### Methods

Whatever communicates by hand does it inside a method: a function in a registry of
`bux.parallel.methods`, one registry per field of `methods`, picked by name.
`parallel.method("attention")` returns the run's method, or the default outside
`parallel.use`. A name that is not registered is an error at `use`. Each kind's page
gives its signature, and each default is registered next to the layer that calls it.

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

The [registries reference](../reference/language/registries.md) lists the names each
registry holds.

A method that communicates by hand does it in a `jax.shard_map` over its own axis,
`axis_names={axis}`, with the other axes left Auto. A method that sorts tokens, as the
MoE's do, is manual over every axis that splits them, since a sort in the global view
gathers every token onto every device. What reads the whole batch (MoE routing,
balance losses, the router bias) stays outside, in the global view, so its semantics
do not change.

#### Writing a method

A new method is a function:

```python
from bux.parallel import methods


@methods.attention.register("mine")
def _(q, k, v, args): ...
```

`methods = methods(attention=mine)` then picks it. Registering a function under a name
bux already uses replaces bux's own.

## The pages

Each page covers one way to split or one method, in the same order: what it is and
when to use it, a quick start, its options, constraints and errors, how it works, and
its measurements.

How the model splits:

- [Layout](layout.md): the named dimensions and the axes they split over; FSDP's
  axes, ZeRO-1 and sequence parallelism; how each layer names its parameters.
- [Many nodes](multi-node.md): one process per host, the mesh across nodes, the
  shared compile cache, and how FSDP, HSDP, EP and pipelines scale to eight nodes.
- [Tensor parallelism](tensor.md): weights split over `tensor`, sequence
  parallelism, and `methods.tensor`, which overlaps TP's collectives with its matmuls.
- [Attention and context parallelism](attention.md): attention split by head under
  TP and by position under CP (`ring`, `ulysses`), and the kernel each device runs.
- [Context parallelism for recurrent mixers](recurrence.md): `pass_state`, which
  passes the recurrent state between devices, the recurrence's chunks, and its GPU
  kernels.
- [Mixture of experts and expert parallelism](experts.md): `sorted` under FSDP, the
  EP methods (`nccl_ep` among them), capacity, the experts in MXFP8 and FP8 dispatch.
- [Pipelines](pipelines.md): layers split over `stage`; the `1f1b`, `zbh1`, `gpipe`
  and `circular` schedules, balanced stages, and the bubble.

How a step fits and runs:

- [Repeats and remat](remat.md): scanned or unrolled repeats, and what the forward
  pass keeps for the backward (`remat=auto` picks for you).
- [Microbatches](microbatches.md): gradient accumulation, parameters gathered once
  and gradients reduced once, microbatches interleaved, and donation.

How other settings meet a split:

- [Muon](muon.md): where Muon's Newton-Schulz runs on a split model.
- [Precision and matmul formats](precision.md): gradient reductions across devices,
  block-scaled formats on any mesh, and the float32 parts of a run.

## Examples

Each file in `examples/experiments/parallel/` is a run. `bux plan` shows its GPUs, the
bytes of its parameters and optimizer state on each, and its pipeline bubble, without
any GPU.

| File | What it splits, and over what |
|---|---|
| `laptop.bux` | a tiny hybrid six ways over 8 CPU devices: DP, FSDP, TP, EP, CP, PP |
| `fsdp_8b.bux` | Llama 3 8B, FSDP over 8 GPUs, bfloat16, scanned and rematerialized |
| `mesh_sweep.bux` | a 1B Llama five ways over 8 GPUs: DDP, FSDP, ZeRO-1, HSDP, TP |
| `tp_70b.bux` | Llama 3 70B over 32 GPUs: TP and sequence parallelism, FSDP across nodes |
| `pp_70b.bux` | Llama 3 70B over 32 GPUs: TP, and a circular pipeline across nodes |
| `ep_moe.bux` | a 16B DeepSeek-V3-shaped MoE over 16 GPUs: EP and FSDP |
| `long_context.bux` | a Qwen3-Next-shaped hybrid on 64k-token rows over 8 GPUs: ring and `pass_state` |

`laptop.bux` trains on a laptop, its CPU split into 8 devices:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    bux train examples/experiments/parallel/laptop.bux
```

`bux plan` shows its six runs. They share one id, because they compute the same:

```console
$ bux plan examples/experiments/parallel/laptop.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
8944944995  497k    10.2k   0.0        1.74M      8     1.89MiB     3.79MiB    0%      laptop[dp]
8944944995  497k    10.2k   0.0        1.74M      8     243KiB      485KiB     0%      laptop[fsdp]
8944944995  497k    10.2k   0.0        1.74M      8     261KiB      521KiB     0%      laptop[tp]
8944944995  497k    10.2k   0.0        1.74M      8     243KiB      485KiB     0%      laptop[ep]
8944944995  497k    10.2k   0.0        1.74M      8     485KiB      970KiB     0%      laptop[cp]
8944944995  497k    10.2k   0.0        1.74M      8     259KiB      517KiB     20%     laptop[pp]
```

The [Scale out](../tutorials/scale-out.md) tutorial walks through this file, then a
node of GPUs and many nodes.

## GPUs, hosts and the plan

`bux launch` gives a run the GPUs of its mesh: the product of its axes' sizes. An
unset `data` means every visible GPU for a run alone, and one in a sweep. A sweep
trains its runs side by side, each on a group of free GPUs; a run that needs more GPUs
than there are fails at once.

A run alone also trains on many hosts, one process each, when Slurm, Open MPI or
`JAX_NUM_PROCESSES` say so ([Launching](../training/launching.md)). The mesh then
spans every process's devices, each node holding a block of its innermost axes
([Many nodes](multi-node.md)).

- Each process reads only the rows of a batch its devices hold
  (`parallel.local_rows`); a batch depends only on the seed and the step.
- `parallel.shard_batch` puts them on the devices with
  `jax.make_array_from_callback`, just before the step that reads them. A thread
  loads the next two batches on the host (`prefetch`).
- `tests/test_launch.py` trains the smoke run as two processes of 4 CPU devices each,
  through `jax.distributed` on localhost, and checks it matches one process of 8.

`bux plan` shows what each run takes before it starts:

- its FLOPs per token ([the log](../training/running.md));
- its GPUs;
- the bytes of its parameters and of its optimizer state on each GPU;
- its bubble, the share of a step its pipeline stages idle
  ([Pipelines](pipelines.md)).

The bytes come from `parallel.shardings` of the model built over a
`jax.sharding.AbstractMesh`, which `parallel.use` accepts like a mesh, so the plan
needs no devices. They count the state alone: a step's activations and new state come
on top.

```console
$ bux plan examples/experiments/smoke.bux 'mesh = fsdp: mesh(fsdp=8) | one: mesh(data=1)'
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
5a4f0f7d85  156k    25.6k   0.2        984k       8     76.2KiB     152KiB     0%      smoke[fsdp]
5a4f0f7d85  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[one]
```

## In a script

```python
from bux import parallel
from bux.train import stages

t = run.train
# The run's methods, with a pipeline's stages balanced as `train` balances them.
with parallel.use(t.mesh, t.layout, stages.balanced(run)):
    model = parallel.build(lambda: Model(run.arch, rngs=nnx.Rngs(0)))
    trainer = Trainer(model, tx, objective)  # its state on its shardings
    trainer.train_on_batch(batch)  # the batch through parallel.shard_batch
```

`parallel.build` creates every parameter on its shards
([Born on their shards](layout.md#born-on-their-shards)). A `Trainer` made by hand
takes the run's precision and step methods as arguments (`compute=`,
`microbatches=`, `gather=`, `donate=`, ...). Its own defaults are float32 and no
donation, where `train(run)` passes the run's settings.

## Testing

The tests run on 8 CPU devices (`tests/conftest.py` sets
`--xla_force_host_platform_device_count=8`) and compare a run split over them with one
on a single device:

- `losses(source, mesh="mesh(fsdp=8)")` trains a few steps and returns the losses;
- `involuntary()` counts the times XLA warned that it rematerialized an array in
  full, which a complete layout never makes it do.

```{toctree}
:maxdepth: 2
:hidden:

layout
multi-node
tensor
attention
recurrence
experts
pipelines
remat
microbatches
muon
precision
```
