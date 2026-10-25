# Scale out

In this tutorial you split one run over many devices. On your laptop, a CPU stands in
for 8 devices, and you train a tiny hybrid model six ways: data parallel, FSDP,
tensor, expert, context and pipeline parallel. All six end on the same loss. Then you
read what changes on a node of GPUs and on many nodes: the same three settings, at
other sizes, and a launcher that runs the same command on every host.

Only the laptop part runs here. It builds on [Train on a laptop](train-on-a-laptop.md).

## 1. Three settings, none in the id

A run splits over devices by three settings:

| setting   | what it says                                                    | default                 |
|-----------|-----------------------------------------------------------------|-------------------------|
| {bux}`mesh`    | a size per axis of the device grid: `stage`, `data`, `fsdp`, `context`, `expert`, `tensor` | every device on `data` |
| {bux}`layout`  | which mesh axes each named dimension (`batch`, `heads`, `mlp`, `experts`, ...) splits over | see [Layout](../parallelism/layout.md) |
| {bux}`methods` | how each part that communicates is computed, by name             | the one-device code     |

None of them changes what the run computes, so none is in its id: a sweep over meshes
is several runs with one id, and a run resumes on another mesh. `data` takes the
devices the other axes leave.

## 2. Give your CPU eight devices

JAX sees one CPU device by default. An XLA flag makes it split the CPU into several,
which is how bux's own tests check every kind of parallelism on a laptop:

```sh
export XLA_FLAGS=--xla_force_host_platform_device_count=8
```

Without it, a mesh of 8 cannot fit:

```console
$ bux train examples/experiments/parallel/laptop.bux --only fsdp
...
ValueError: mesh: the axes other than data take 8 devices, which do not divide the 1 there are; set mesh.data
```

## 3. One model, six splits

`examples/experiments/parallel/laptop.bux` is a tiny hybrid of Gated DeltaNet,
attention and a mixture of experts, with a sweep over how it splits:

```bux
d_model    = 64
vocab_size = 512

norm    = rmsnorm
delta   = gdn(num_heads=4, head_dim=16)
attn    = gqa(num_heads=4, num_kv_heads=2, rope=rope())
experts = moe(num_experts=8, top_k=2, aux_loss_scope=sequence,
              expert=gated_mlp(intermediate_size=64, activation=swish))

layer(mixer) = [norm mixer] [norm experts]
D = layer(delta)
A = layer(attn)
model = embed "DA"^2 norm head(tied=true)

data      = synthetic
optimizer = adamw(lr=1e-3)
# XLA:CPU cannot compile the experts' and the pipeline's bfloat16 psums.
precision = precision(compute=float32)
seq_len   = 64
batch     = 16
steps     = 10

mesh    = dp:   mesh(data=8)
        | fsdp: mesh(fsdp=8)
        | tp:   mesh(fsdp=2, tensor=4)
        | ep:   mesh(fsdp=2, expert=4)
        | cp:   mesh(data=2, context=4)
        | pp:   mesh(stage=2, fsdp=4)
layout  = dp:   layout
        | fsdp: layout
        | tp:   layout(stream=tensor)
        | ep:   layout
        | cp:   layout
        | pp:   layout
methods = dp:   methods
        | fsdp: methods
        | tp:   methods
        | ep:   methods(moe=all_gather)
        | cp:   methods(attention=ring, recurrence=pass_state)
        | pp:   methods(repeat=scan, microbatches=4)
```

The three settings share their labels, so they are one axis of six runs, each taking
the mesh, layout and methods of its label:

| run    | mesh                     | what splits                                                   |
|--------|--------------------------|---------------------------------------------------------------|
| `dp`   | `data=8`                 | the batch's rows; every device holds every parameter (DDP)     |
| `fsdp` | `fsdp=8`                 | the rows, and every parameter and its optimizer state, gathered as each layer runs (FSDP, ZeRO-3) |
| `tp`   | `fsdp=2, tensor=4`       | heads and hidden units over `tensor`; with `layout(stream=tensor)`, the stream's positions too between branches (sequence parallelism) |
| `ep`   | `fsdp=2, expert=4`       | the experts, 2 on each device; `all_gather` shows each device every token |
| `cp`   | `data=2, context=4`      | each row's positions: `ring` passes keys and values round the devices, `pass_state` the DeltaNet's state along |
| `pp`   | `stage=2, fsdp=4`        | the blocks, in two stages, the batch in 4 microbatches (`1f1b`) |

## 4. Plan it

`bux plan` builds each run's model from shapes alone over an abstract mesh, so it
needs neither the devices nor the flag:

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

- Every run has the same id, `90b55fa67f`: they compute the same thing.
- `params/gpu` is what a device holds. DDP copies everything (1.89 MiB); FSDP splits
  it 8 ways (243 KiB). Under `cp`, FSDP's axes take `context` too, so the 4 context
  devices split the parameters and the 2 data copies keep them whole (485 KiB).
- `bubble` is the share of a step a pipeline's stages idle: (S - 1) / (M + S - 1) =
  1 / 5 for 2 stages and 4 microbatches.

## 5. Train it

```console
$ bux train examples/experiments/parallel/laptop.bux
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[dp] step 10/10  loss 6.7943  lr 1.00e-03  grad 2.169  5,875 tok/s
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[fsdp] step 10/10  loss 6.7943  lr 1.00e-03  grad 2.169  5,342 tok/s
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[tp] step 10/10  loss 6.7943  lr 1.00e-03  grad 2.169  5,947 tok/s
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[ep] step 10/10  loss 6.7943  lr 1.00e-03  grad 2.169  5,260 tok/s
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[cp] step 10/10  loss 6.7943  lr 1.00e-03  grad 2.169  4,653 tok/s
laptop[pp]: methods.stages = [1, 1]
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[pp] step 10/10  loss 6.7943  lr 1.00e-03  grad 2.169  5,846 tok/s
```

All six end on the same loss and gradient norm, to the digits printed. Rounding
differs between splits, and an MoE's top-k picks amplify it as a run goes on, so
longer runs drift apart slowly. `methods.stages = [1, 1]` is the split of the two
`"DA"` passes into stages that training chose, balancing the stages' FLOPs. Each run
trains into a directory of its own, as their names differ:

```console
$ ls -1d runs/laptop*
runs/laptop_cp-90b55fa67f
runs/laptop_dp-90b55fa67f
runs/laptop_ep-90b55fa67f
runs/laptop_fsdp-90b55fa67f
runs/laptop_pp-90b55fa67f
runs/laptop_tp-90b55fa67f
```

:::{note}
The run says `precision(compute=float32)` because XLA:CPU cannot compile the
bfloat16 sums an MoE's methods (but `dense`) and a pipeline make across devices. bux
stops with that advice before it would crash. On GPUs, train in bfloat16, the
default.
:::

## 6. On a node of GPUs

On GPUs the settings are the same, at real sizes. `bux launch` gives a run the GPUs
its mesh needs (`CUDA_VISIBLE_DEVICES` says which), and trains a sweep's runs side by
side on groups of free GPUs. `bux plan` shows each run's bytes a GPU before you
launch it, on any machine. Each file in `examples/experiments/parallel/` is a run:

**FSDP.** `fsdp_8b.bux` trains Llama 3 8B at its full size over 8 GPUs, in bfloat16,
its 32 blocks scanned (one compiled block) and rematerialized, each step in 4
microbatches:

```bux
use llama3_8b
use fineweb_wsd

mesh      = mesh(fsdp=8)
precision = precision(compute=bfloat16)
methods   = methods(repeat=scan, remat=dots, microbatches=4)
```

```console
$ bux plan examples/experiments/parallel/fsdp_8b.bux
1 run
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
f0e7783cf3  8.03B   10B     1.2        48.7B      8     3.74GiB     7.48GiB    0%      fsdp_8b
```

**Which split is fastest?** `mesh_sweep.bux` trains a 1B Llama five ways over 8 GPUs
with one id, so you compare their `tokens_per_s` in each run's `metrics.jsonl`:

```console
$ bux plan examples/experiments/parallel/mesh_sweep.bux
5 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
1515c7682c  1.27B   13.1M   0.0        8.02B      8     4.73GiB     9.46GiB    0%      mesh_sweep[ddp]
1515c7682c  1.27B   13.1M   0.0        8.02B      8     605MiB      1.18GiB    0%      mesh_sweep[fsdp]
1515c7682c  1.27B   13.1M   0.0        8.02B      8     4.73GiB     1.18GiB    0%      mesh_sweep[zero1]
1515c7682c  1.27B   13.1M   0.0        8.02B      8     1.18GiB     2.36GiB    0%      mesh_sweep[hsdp]
1515c7682c  1.27B   13.1M   0.0        8.02B      8     605MiB      1.18GiB    0%      mesh_sweep[tp]
```

ZeRO-1 (`layout(zero=1)`) splits only the optimizer's state, and HSDP
(`mesh(data=2, fsdp=4)`) splits within groups of 4 and copies across 2.

**Experts.** `ep_moe.bux` is a 16B DeepSeek-V3-shaped MoE over 16 GPUs (two nodes):
each GPU of a node holds 8 of the 64 experts, and FSDP spans the nodes:

```bux
mesh      = mesh(fsdp=2, expert=8)
precision = precision(compute=bfloat16)
methods   = methods(moe=all_gather, repeat=scan, remat=dots)
```

`methods.moe` picks how tokens reach their experts, by name: `sorted` (the default),
`dense`, `all_gather`, `all_to_all`, `ragged_all_to_all`, `ragged_tokens` or
`nccl_ep` ([Experts](../parallelism/experts.md)).

**A pipeline.** `pp_70b.bux` puts Llama 3 70B on 4 nodes of 8 GPUs: tensor parallel
within a node, and a circular pipeline of 4 stages across them, each node holding 20
of the 80 blocks in two rounds of 10, a step's 256 rows in 16 microbatches:

```bux
mesh      = mesh(stage=4, tensor=8)
layout    = layout(stream=tensor)
precision = precision(compute=bfloat16)
methods   = methods(repeat=scan, remat=full, pipeline=circular, rounds=2,
                    microbatches=16)
```

```console
$ bux plan examples/experiments/parallel/pp_70b.bux
1 run
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
0d0f5cc6f4  70.6B   10B     0.1        429B       32    8.95GiB     17.9GiB    9%      pp_70b
```

A stage idles (S - 1) / (R·M + S - 1) = 3 / 35 of a step. `tp_70b.bux` trains the
same model with FSDP across the nodes in place of the pipeline, and has the same id.

**Long rows.** `long_context.bux` splits each 64k-token row 8 ways over `context`:
`methods(attention=ring, recurrence=pass_state)`.

| method            | mesh                          | the example        | the guide                                |
|-------------------|-------------------------------|--------------------|------------------------------------------|
| FSDP, HSDP, ZeRO-1 | `fsdp=8`; `data=2, fsdp=4`   | `fsdp_8b.bux`, `mesh_sweep.bux` | [Layout](../parallelism/layout.md) |
| tensor            | `tensor=8`                    | `tp_70b.bux`       | [Tensor](../parallelism/tensor.md)       |
| expert            | `expert=8`                    | `ep_moe.bux`       | [Experts](../parallelism/experts.md)     |
| context           | `context=8`                   | `long_context.bux` | [Attention](../parallelism/attention.md), [Recurrence](../parallelism/recurrence.md) |
| pipeline          | `stage=4`                     | `pp_70b.bux`       | [Pipelines](../parallelism/pipelines.md) |

## 7. On many nodes

A run alone trains on many hosts when the environment says how many processes there
are: under Slurm's `srun` or Open MPI's `mpirun`, or with `JAX_COORDINATOR_ADDRESS`
(process 0's `host:port`), `JAX_NUM_PROCESSES` and `JAX_PROCESS_ID` set on each.
Every process runs the same command, and they join through `jax.distributed`. Under
Slurm and Open MPI, start a task a GPU:

```sh
srun --nodes=4 --ntasks-per-node=8 bux launch speedrun.bux   # 32 GPUs, one mesh
```

or, with a process a host and the variables set by hand:

```sh
JAX_COORDINATOR_ADDRESS=10.0.0.1:12355 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache \
    bux launch bench.bux 'mesh = mesh(data=2, fsdp=8)' --out /shared/runs
```

What to know:

- The mesh spans every process's devices, each node holding a block of the innermost
  axes, so `tensor` stays within a node. `mesh(data=2, fsdp=8)` on two nodes of 8 is
  HSDP: FSDP within a node, a copy on each.
- Each process reads only the rows of a batch that its devices hold.
- `--out` and `JAX_COMPILATION_CACHE_DIR` belong on a filesystem every host mounts:
  every process writes its part of a checkpoint, and the processes must agree on
  what the compile cache holds (bux turns off a cache that is each host's own).
- A signal to any process stops them all after the same step.
- A sweep does not train on many hosts: give each run a job of its own with
  `--index N`, as `--index $SLURM_ARRAY_TASK_ID` in an array job.

[Many nodes](../parallelism/multi-node.md) has the details and measurements, and
[Launching](../training/launching.md) the launcher.

## What you learned

| you wrote                                        | it does                                        |
|--------------------------------------------------|------------------------------------------------|
| `XLA_FLAGS=--xla_force_host_platform_device_count=8` | 8 CPU devices, for trying any split on a laptop |
| `mesh(fsdp=8)`, `mesh(data=2, fsdp=4)`          | FSDP; HSDP                                     |
| `mesh(fsdp=2, tensor=4)`, `layout(stream=tensor)` | tensor and sequence parallelism              |
| `mesh(expert=8)`, `methods(moe=...)`             | expert parallelism                             |
| `mesh(context=4)`, `methods(attention=ring, recurrence=pass_state)` | context parallelism         |
| `mesh(stage=2)`, `methods(repeat=scan, microbatches=4)` | a pipeline                              |
| `srun ... bux launch run.bux`                    | one run over many hosts                        |

The [parallelism guide](../parallelism/index.md) explains each method, and
[Performance](../performance/index.md) how to make a run fast.
