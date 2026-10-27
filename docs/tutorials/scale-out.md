# Scale out

Split one run over many devices. On your laptop, the CPU stands in for 8 devices,
and you train a tiny hybrid model six ways: data, FSDP, tensor, expert, context and
pipeline parallel. By the end you will have seen all six end on the same loss, and
you will know what changes on a node of GPUs and on many nodes: the same three
settings at other sizes, and the same command on every host.

Only the laptop part runs here. It builds on [Train on a laptop](train-on-a-laptop.md).

## 1. Three settings, none in the id

A run splits over devices by three settings:

| setting   | what it says                                                    | default                 |
|-----------|-----------------------------------------------------------------|-------------------------|
| {bux}`mesh`    | a size per axis of the device grid: `stage`, `data`, `fsdp`, `context`, `expert`, `tensor` | every device on `data` |
| {bux}`layout`  | which mesh axes each named dimension (`batch`, `heads`, `mlp`, `experts`, ...) splits over | see [Layout](../parallelism/layout.md) |
| {bux}`methods` | how each part that communicates is computed, by name             | the one-device code     |

None of them changes what the run computes, so none is part of its id. A sweep over
meshes is several runs with one id, and a run can resume on another mesh. `data`
takes whatever devices the other axes leave.

## 2. Give your CPU eight devices

JAX sees one CPU device by default. An XLA flag splits the CPU into several, which
is how bux's own tests check every kind of parallelism on a laptop:

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

The three settings share their labels, so they form one axis of six runs. Each run
takes the mesh, layout and methods of its label:

| run    | mesh                     | what splits                                                   |
|--------|--------------------------|---------------------------------------------------------------|
| `dp`   | `data=8`                 | the batch's rows; every device holds every parameter (DDP)     |
| `fsdp` | `fsdp=8`                 | the rows, and every parameter and its optimizer state, gathered as each layer runs (FSDP, ZeRO-3) |
| `tp`   | `fsdp=2, tensor=4`       | heads and hidden units over `tensor`; with `layout(stream=tensor)`, the stream's positions too between branches (sequence parallelism) |
| `ep`   | `fsdp=2, expert=4`       | the experts, 2 on each device; `all_gather` shows each device every token |
| `cp`   | `data=2, context=4`      | each row's positions: `ring` passes keys and values round the devices, `pass_state` the DeltaNet's state along |
| `pp`   | `stage=2, fsdp=4`        | the blocks, in two stages, the batch in 4 microbatches (`1f1b`) |

## 4. Plan it

`bux plan` builds each run's model from shapes alone, over an abstract mesh, so it
needs neither the devices nor the flag:

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

- Every run has the same id, `8944944995`: they compute the same thing.
- `params/gpu` is what each device holds. DDP copies everything (1.89 MiB); FSDP
  splits it 8 ways (243 KiB). Under `cp`, FSDP's axes include `context`, so the 4
  context devices split the parameters, and each of the 2 data copies holds them
  whole (485 KiB).
- `bubble` is the share of a step that a pipeline's stages sit idle:
  (S - 1) / (M + S - 1) = 1 / 5 for 2 stages and 4 microbatches.

## 5. Train it

```console
$ bux train examples/experiments/parallel/laptop.bux
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[dp] step 10/10  loss 6.2562  lr 1.00e-03  grad 0.800  5,446 tok/s  ...
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[fsdp] step 10/10  loss 6.2562  lr 1.00e-03  grad 0.800  5,143 tok/s  ...
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[tp] step 10/10  loss 6.2562  lr 1.00e-03  grad 0.800  5,736 tok/s  ...
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[ep] step 10/10  loss 6.2562  lr 1.00e-03  grad 0.800  4,872 tok/s  ...
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[cp] step 10/10  loss 6.2562  lr 1.00e-03  grad 0.800  4,467 tok/s  ...
laptop[pp]: methods.stages = [1, 1]
remat=auto: none in 1s (none 0.0 GiB in 0s)
laptop[pp] step 10/10  loss 6.2562  lr 1.00e-03  grad 0.800  5,679 tok/s  ...
```

All six end on the same loss and gradient norm, to the digits printed. Rounding
differs between splits, and an MoE's top-k choices amplify it as a run goes on, so
longer runs slowly drift apart. `methods.stages = [1, 1]` is how training split the
two `"DA"` passes into stages, balancing their FLOPs. Each run trains into its own
directory, because their names differ:

```console
$ ls -1d runs/laptop*
runs/laptop_cp-8944944995
runs/laptop_dp-8944944995
runs/laptop_ep-8944944995
runs/laptop_fsdp-8944944995
runs/laptop_pp-8944944995
runs/laptop_tp-8944944995
```

:::{note}
The run sets `precision(compute=float32)` because XLA:CPU cannot compile the
bfloat16 sums across devices that a pipeline and every MoE method except `dense`
make. bux stops with that advice before it would crash. On GPUs, train in bfloat16,
the default.
:::

## 6. On a node of GPUs

On GPUs the settings are the same, at real sizes. `bux launch` gives a run the GPUs
its mesh needs (`CUDA_VISIBLE_DEVICES` says which), and trains a sweep's runs side by
side on groups of free GPUs. `bux plan` shows each run's memory per GPU before you
launch it, on any machine. Each file in `examples/experiments/parallel/` is a run:

**FSDP.** `fsdp_8b.bux` trains Llama 3 8B at full size over 8 GPUs, in bfloat16. Its
32 blocks are scanned (one compiled block) and rematerialized, and each step runs in
4 microbatches:

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
c548b8d54b  8.03B   10B     1.2        48.7B      8     3.74GiB     7.48GiB    0%      fsdp_8b
```

**Which split is fastest?** `mesh_sweep.bux` trains a 1B Llama five ways over 8 GPUs,
all with one id, so you can compare `tokens_per_s` in each run's `metrics.jsonl`:

```console
$ bux plan examples/experiments/parallel/mesh_sweep.bux
5 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
03359bea6e  1.27B   13.1M   0.0        8.02B      8     4.73GiB     9.46GiB    0%      mesh_sweep[ddp]
03359bea6e  1.27B   13.1M   0.0        8.02B      8     605MiB      1.18GiB    0%      mesh_sweep[fsdp]
03359bea6e  1.27B   13.1M   0.0        8.02B      8     4.73GiB     1.18GiB    0%      mesh_sweep[zero1]
03359bea6e  1.27B   13.1M   0.0        8.02B      8     1.18GiB     2.36GiB    0%      mesh_sweep[hsdp]
03359bea6e  1.27B   13.1M   0.0        8.02B      8     605MiB      1.18GiB    0%      mesh_sweep[tp]
```

ZeRO-1 (`layout(zero=1)`) splits only the optimizer state. HSDP
(`mesh(data=2, fsdp=4)`) splits within groups of 4 and keeps 2 copies.

**Experts.** `ep_moe.bux` is a 16B DeepSeek-V3-shaped MoE over 16 GPUs (two nodes).
Each GPU of a node holds 8 of the 64 experts, and FSDP spans the nodes:

```bux
mesh      = mesh(fsdp=2, expert=8)
precision = precision(compute=bfloat16)
methods   = methods(moe=all_gather, repeat=scan, remat=dots)
```

`methods.moe` picks how tokens reach their experts, by name: `sorted` (the default),
`dense`, `all_gather`, `all_to_all`, `ragged_all_to_all`, `ragged_tokens` or
`nccl_ep` ([Experts](../parallelism/experts.md)).

**A pipeline.** `pp_70b.bux` puts Llama 3 70B on 4 nodes of 8 GPUs: tensor parallel
within a node, and a circular pipeline of 4 stages across the nodes. Each node holds
20 of the 80 blocks, in two rounds of 10, and a step's 256 rows go through in 16
microbatches:

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
34e7f5a8ac  70.6B   10B     0.1        429B       32    8.95GiB     17.9GiB    9%      pp_70b
```

A stage sits idle for (S - 1) / (R·M + S - 1) = 3 / 35 of a step. `tp_70b.bux` trains
the same model with FSDP across the nodes instead of the pipeline, and has the same
id.

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

A single run trains on many hosts when the environment says how many processes there
are: under Slurm's `srun` or Open MPI's `mpirun`, or with `JAX_COORDINATOR_ADDRESS`
(the `host:port` of process 0), `JAX_NUM_PROCESSES` and `JAX_PROCESS_ID` set on each
host. Every process runs the same command, and they join through `jax.distributed`.
Under Slurm and Open MPI, start one task per GPU:

```sh
srun --nodes=4 --ntasks-per-node=8 bux launch speedrun.bux   # 32 GPUs, one mesh
```

or, with one process per host and the variables set by hand:

```sh
JAX_COORDINATOR_ADDRESS=10.0.0.1:12355 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache \
    bux launch bench.bux 'mesh = mesh(data=2, fsdp=8)' --out /shared/runs
```

What to know:

- The mesh spans the devices of every process. Each node holds a block of the
  innermost axes, so `tensor` stays within a node. `mesh(data=2, fsdp=8)` on two
  nodes of 8 is HSDP: FSDP within each node, and a copy on each.
- Each process reads only the rows of a batch that its devices hold.
- `--out` and `JAX_COMPILATION_CACHE_DIR` belong on a filesystem that every host
  mounts. Every process writes its part of a checkpoint, and the processes must agree
  on what the compile cache holds (bux turns off a cache that is local to each host).
- A signal to any process stops them all after the same step.
- A sweep does not train across many hosts: give each run its own job with
  `--index N`, such as `--index $SLURM_ARRAY_TASK_ID` in an array job.

[Many nodes](../parallelism/multi-node.md) has the details and measurements, and
[Launching](../training/launching.md) covers the launcher.

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

## Next steps

- [Old against new at the same serving cost](old-vs-new.md): the advanced tutorial,
  which puts the earlier ones together on a node of 8 GPUs.
- [Parallelism](../parallelism/index.md): each method in depth.
- [Performance](../performance/index.md): how to make a run fast.
