# Parallelism

A run splits over devices by three settings. None of them changes what it computes,
so none is in its id: a run resumes on another mesh, and a sweep over meshes is still
several runs, as their names differ.

```
mesh    = mesh(fsdp=8, tensor=2)                  # data takes the devices the others leave
layout  = layout(stream=[context, tensor])        # sequence parallelism
methods = methods(attention=ring, moe=all_to_all)

mesh        = fsdp: mesh(fsdp=16) | tp: mesh(fsdp=8, tensor=2)   # the grid, as a sweep
layout.seq  = context | none
```

| setting   | what it says                                           | default                    |
|-----------|--------------------------------------------------------|----------------------------|
| `mesh`    | a size per axis: `stage`, `data`, `fsdp`, `context`, `expert`, `tensor` | every device on `data` |
| `layout`  | which axes each named dimension splits over            | see below                  |
| `methods` | how each part that communicates is computed, by name   | the one-device code        |

`bux.parallel` reads them; the configs are in `bux/train/config.py`.

## Examples

Each file in `examples/experiments/parallel/` is a run, and `bux plan` shows its GPUs,
the bytes of its parameters and of its optimizer's state on each, and its pipeline
bubble, without any GPU.

| file               | what it splits, and over what                                       |
|--------------------|----------------------------------------------------------------------|
| `laptop.bux`       | a tiny hybrid six ways over 8 CPU devices: DP, FSDP, TP, EP, CP, PP  |
| `fsdp_8b.bux`      | Llama 3 8B, FSDP over 8 GPUs, bfloat16, scanned and rematerialized  |
| `mesh_sweep.bux`   | a 1B Llama five ways over 8 GPUs: DDP, FSDP, ZeRO-1, HSDP, TP       |
| `tp_70b.bux`       | Llama 3 70B over 32 GPUs: TP and sequence parallelism, FSDP across nodes |
| `pp_70b.bux`       | the same over 32 GPUs: TP, and a circular pipeline across nodes     |
| `ep_moe.bux`       | a 16B DeepSeek-V3-shaped MoE over 16 GPUs: EP and FSDP              |
| `long_context.bux` | a Qwen3-Next-shaped hybrid on 64k-token rows over 8 GPUs: ring and `pass_state` |

```
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    bux train examples/experiments/parallel/laptop.bux
```

## The mesh

One mesh, always of all six axes, in the order `stage, data, fsdp, context, expert,
tensor`: the slowest links first, so that `tensor` stays inside a node, and `stage`
before `data`, as a pipeline's sends need the least bandwidth
([Pipelines](#pipelines-methodspipeline) says why it must). An axis not set has size 1, and `data` takes the devices the others leave. Every axis is Auto:
layers do not say how anything splits, and the compiler propagates shardings from the
parameters and a few constraints.

`bux train` uses every visible device; `bux launch` gives a run the GPUs its mesh
needs ([below](#gpus-hosts-and-the-plan)). A mesh that needs more devices than there
are, a `batch` that does not split over the batch axes, and a `seq_len` that does not
split over `seq` or `stream` are errors before anything is built.

## GPUs, hosts and the plan

`bux launch` gives a run the GPUs of its mesh: the product of its axes' sizes, where
`data` unset is every visible GPU for a run alone and one in a sweep. A sweep trains
its runs side by side, each on a group of free GPUs, and a run that needs more GPUs
than there are fails at once. A run alone also trains on many hosts, one process each,
when Slurm, Open MPI or `JAX_NUM_PROCESSES` say so
([Launching](training.md#launching)). The mesh then spans every process's devices,
each node holding a block of its innermost axes ([Many nodes](#many-nodes)). Each
process reads the rows of a batch that its devices hold (`parallel.local_rows`), as
a batch depends only on the seed and the step, and `parallel.shard_batch` puts them
there with `jax.make_array_from_callback`, just before the step that reads it (a
thread loads the next two on the host, `prefetch`). `tests/test_launch.py` trains
the smoke run as two processes of 4 CPU devices each, through `jax.distributed` on
localhost, as one process of 8 does.

`bux plan` shows what each run takes before it starts: its FLOPs a token
([the log](training.md#running)), its GPUs, the bytes of its parameters and of its
optimizer's state on each, and the share of a step its pipeline's stages idle
([below](#pipelines-methodspipeline)). The bytes come from `parallel.shardings` of
the model built over a `jax.sharding.AbstractMesh`, which `parallel.use` takes as it
does a mesh, so the plan needs no devices. They are the state's alone: a step's
activations and new state come on top.

```
$ bux plan examples/experiments/smoke.bux 'mesh = fsdp: mesh(fsdp=8) | one: mesh(data=1)'
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
bbbf3ae709  156k    25.6k   0.2        1.03M      8     76.2KiB     152KiB     0%      smoke[fsdp]
bbbf3ae709  156k    25.6k   0.2        1.03M      1     609KiB      1.19MiB    0%      smoke[one]
```

## Many nodes

```
mesh = mesh(fsdp=16)            # FSDP over two nodes of 8 GPUs
mesh = mesh(data=2, fsdp=8)     # HSDP: FSDP within a node, a copy on each
```

XLA puts the GPUs of a node, or of an NVLink domain that spans nodes, in one slice
(`slice_index`), and `jax.make_mesh` refuses devices of more than one. So
`parallel.mesh` lays them out itself (`mesh.grid`), as MaxText's hybrid mesh does: a
node holds a block of the innermost axes, filled from `tensor` outward, and the axes
outside it span nodes (`mesh.split`). On nodes of 8, `mesh(data=2, fsdp=8,
tensor=2)` puts `tensor=2 × fsdp=4` in each, and `data=2 × fsdp=2` across them. A
mesh that does not fill nodes so, `tensor=3` on nodes of 8, is an error that names the
axis. Where devices have no slice, as CPUs do, a process is a node.

Each process reads the rows of a batch that its devices hold, and leaves the others 0
(`parallel.local_rows`): no process reads the whole batch, and no source makes the
permutation of a whole corpus ([Reading](data.md#reading)).
`tests/test_nodes.py` checks that processes reading their own rows of every kind of
source make the batch one process does.

FSDP's axes take `context` and `expert` with `fsdp` (`layout.fsdp`, [How names
resolve](#how-names-resolve)), so a dense parameter and its Adam state are not copied
over them; `layout.fsdp = fsdp` takes `fsdp` alone. On 8 CPU devices a dense run and
an MoE train as one device does on `fsdp=2, context=2, expert=2` and `context=2,
expert=4`, with no rematerialization in full. On 8 B300s, for the 1.3B dense model of
[Muon](#muon-methodsorthogonalize) on `fsdp=4, context=2` and its 5.2B MoE on
`fsdp=2, expert=4` (8 rows of 4096 a GPU, bfloat16), it is a little faster, and it
halves what the dense model keeps a GPU:

| model, mesh                 | `layout.fsdp = fsdp` | the default  | parameters a GPU |
|-----------------------------|----------------------|--------------|------------------|
| dense, `fsdp=4, context=2`  | 886k tok/s           | **903k**     | 1.19 GiB → 611 MiB |
| MoE, `fsdp=2, expert=4`     | 820k tok/s           | **830k**     | 3.42 GiB → 2.40 GiB |

### The compile cache

The processes of a job must all find a program in JAX's cache, or all compile it: on
GPUs XLA's autotuner splits its work among the processes that compile a program, and
each waits for the others' shares (`xla_gpu_shard_autotuning`), so one that compiles
a program another loads waits a day. Only the first process writes the cache, so they
agree only where it is one folder for all of them, on a filesystem every host mounts:
set `JAX_COMPILATION_CACHE_DIR` there. When the processes join,
`runtime.agree_on_cache` checks that it is, a file the first makes there that the
others see, and else turns the cache off and says so: each launch then compiles
everything, together. JAX's `jax_share_binary_between_hosts` hangs the same way (the
first compiles alone and waits for the others' shares), so bux leaves it off.

### On B300s

NCCL reaches InfiniBand through libibverbs and its mlx5 provider, which the Docker
image installs. Without them it talks between nodes over TCP sockets: on two B300
nodes an all-reduce of 64 MiB a GPU across the nodes ran at a bus bandwidth of 1 GB/s,
and 25 GB/s over InfiniBand (4 and 42 GB/s over all 16 GPUs). `NCCL_DEBUG=INFO`
says which: `Using network IB`, not `Socket`.

The 1.3B dense model, 8 rows of 4096 a GPU, bfloat16, the cache on a filesystem both
nodes mount (launch: from its start to the fifth step):

| mesh                         | GPUs | tok/s     | a GPU  | of one node's | launch, cold / warm |
|------------------------------|------|-----------|--------|---------------|---------------------|
| `fsdp=8`                     | 8    | 1,133k    | 141.4k |               |                     |
| `fsdp=16`                    | 16   | 2,220k    | 138.7k | 98.1%         | 78 s / 42 s         |
| `data=2, fsdp=8` (HSDP)      | 16   | 2,238k    | 139.9k | 98.9%         | 80 s / 42 s         |

Warm, both processes load the train step from the cache (0.3 s, where it compiles in
16 s); with a cache on each node, which bux turns off, every launch is cold (76 s).

Each host runs one process, the same command, with `JAX_COORDINATOR_ADDRESS` (the
first host's `ip:port`), `JAX_NUM_PROCESSES` and `JAX_PROCESS_ID`:

```
JAX_COORDINATOR_ADDRESS=10.0.0.1:12355 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 \
JAX_COMPILATION_CACHE_DIR=/shared/bux-cache \
    bux launch bench.bux 'mesh = mesh(data=2, fsdp=8)' --out /shared/runs
```

## Names

A layer names the dimensions of a parameter where it makes it, and the layout says
which mesh axes each name splits over:

```python
from bux import parallel

self.q = nnx.Linear(d, h * k, kernel_init=parallel.named("embed", "heads"), rngs=rngs)
self.sinks = nnx.Param(jnp.zeros(h), names=("heads",))
```

The names are the fields of `layout`, a closed set; any other is an error that names
the parameter. A dimension that flattens several of what it names, as a projection's
`h * k` columns flatten `h` heads, names them with their count, `("heads", h)`: then
it splits only between them.

| name       | what it is                                  | default axes                |
|------------|---------------------------------------------|-----------------------------|
| `batch`    | rows                                        | `[data, fsdp, expert]`      |
| `seq`      | positions, inside a layer                   | `context`                   |
| `stream`   | positions of the residual stream            | `context`                   |
| `embed`    | the model's width                           | `none`                      |
| `vocab`    | the vocabulary                              | `tensor`                    |
| `heads`    | query heads, or a mixer's heads             | `tensor`                    |
| `kv_heads` | key and value heads                         | `tensor`                    |
| `mlp`      | hidden units                                | `tensor`                    |
| `experts`  | experts                                     | `expert`                    |
| `latent`   | low-rank bottlenecks: MLA, LatentMoE        | `none`                      |
| `layers`   | the stacked passes of a scanned repeat      | `none`                      |
| `stages`   | those of a pipelined one, a share per stage | `stage`                     |

Names are bux's metadata, under `names`, not Flax's `out_sharding`: Flax never acts on
them, so they cost nothing without a mesh, and `nnx.eval_shape` under one does not try
to resolve them. The optimizer's state inherits them from the parameters it tracks.

### How names resolve

1. For each dimension, take the axes of its name, minus axes of size 1 and axes an
   earlier dimension already uses. If their sizes don't divide the dimension (or its
   count), it stays whole: GQA's 2 kv heads under `tensor=4` are replicated, though
   their `2 * k` columns would split 4 ways.
2. Then FSDP: the largest whole dimension that FSDP's axes divide splits over them,
   or if none does, over as many of them as one does, from the first. FSDP's axes are
   `layout.fsdp`, those of `fsdp`, `context` and `expert` that no dimension splits
   over already: the axes that would copy a dense parameter, but `data`, whose copies
   are HSDP's. So a layer with no names at all still trains under FSDP, an expert
   kernel keeps its experts over `expert` and splits over `fsdp` and `context`, and
   names matter only for TP and EP.
3. FSDP applies to parameters and the optimizer's state, or with `layout(zero=1)` to
   the optimizer's state alone. Other state, such as RNG state, splits by its names
   alone.

`parallel.build(make)` creates a module with every parameter born on its shards, so
no parameter is ever whole on one device; its values are those `make()` gives. The
trainer moves its state to these shardings with `parallel.shard_state` when it is
created, and every step keeps it there.

## Activations

`parallel.shard_stream` constrains the residual stream after every residual branch:
rows over `batch`, positions over `stream`, the rest whole. A layer adds a constraint
where it knows what a dimension means:

```python
q = parallel.shard(q, "batch", "seq", "heads", None)
```

`shard`, `shard_stream` and `spec` do nothing outside `parallel.use`, so one device
runs as if they were not there. `parallel.shard_batch` puts batches on the devices:
every `(b, t)` array over `batch` and `seq`.

## Methods

What communicates by hand does it inside a method: a function in a registry of
`bux.parallel.methods`, one per field of `methods`, picked by name. Another method is a
function:

```python
from bux.parallel import methods


@methods.attention.register("mine")
def _(q, k, v, args): ...
```

and `methods = methods(attention=mine)` picks it. `parallel.method("attention")`
returns the run's method, or the default outside `use`. A name that is not registered
is an error at `use`. The section of each kind gives its signature, and the default
is registered next to the layer that calls it.

A method that communicates by hand does it in a `jax.shard_map` over its own axis,
`axis_names={axis}`, the other axes staying Auto; one that sorts tokens, as the MoE's
do, is manual over every axis that splits them, since a sort in the global view gathers
every token onto every device. What reads the whole batch (MoE routing, balance
losses, the router bias) stays outside, in the global view, so its semantics do not
change.

## In a script

```python
from bux import parallel

with parallel.use(t.mesh, t.layout, t.methods):
    model = parallel.build(lambda: Model(arch, rngs=nnx.Rngs(0)))
    trainer = Trainer(model, tx, loss)  # its state on its shardings
    trainer.train_on_batch(batch)  # the batch through parallel.shard_batch
```

## Testing

The tests run on 8 CPU devices (`tests/conftest.py`), and compare a run split over
them with one on a single device: `losses(source, mesh="mesh(fsdp=8)")` trains a few
steps and returns the losses. `involuntary()` counts the times XLA warned that it
rematerialized an array in full, which a complete layout should never make it do.

## Softmax attention

`gqa`, `mla`, `csa` and the indexer split by head, as Megatron does: a device projects
the input to its own heads, attends with them, and takes their rows of the output
projection, whose sums over `tensor` make the output.

| layer    | parameter                                    | names                      |
|----------|----------------------------------------------|----------------------------|
| all      | `gate`, `rel_proj`                           | `(embed, heads)`           |
| all      | `o`                                          | `(heads, embed)`           |
| all      | `sinks`                                      | `(heads,)`                 |
| GQA      | `q`                                          | `(embed, heads)`           |
| GQA      | `k`, `v`, and the channels of their convs    | `(embed, kv_heads)`        |
| GQA      | fused `qkv`, split where shards do not end   | `(embed, none)`            |
| MLA, CSA | down projections `q_a`, `kv_a`               | `(embed, latent)`          |
| MLA, CSA | up projections `q`, `kv_b`                   | `(latent, heads)`          |
| CSA      | grouped `o_a`, by group                      | `(heads, none, none)`      |
| indexer  | `q`, `weights`                               | `(latent or embed, heads)` |

Each flattened dimension names its heads with their count. What every head shares,
MQA's kv head, CSA's latents and the indexer's keys, is every device's; so are kv
heads that `tensor` does not divide, which every device computes and whose gradients
the devices that share them sum.

The input of a layer is constrained to `(batch, seq, none)`, so with
`layout(stream=tensor)` it is gathered once before the projections (sequence
parallelism); queries, keys, values and head outputs to `(batch, seq, heads, none)`;
the logits to `(batch, heads, seq, none)`, the keys whole, which with `seq` over
`context` is all-gather context parallelism.

The indexer picks its top-k by counting, the k-th highest score found bit by bit, which
splits with the rows and positions: `lax.top_k` sorts, and in the global view gathers
every row onto each device. Its scores' kernel (`bux.architecture.scoring`) is a
`shard_map` that gives a device its rows, positions and heads, and sums the heads'.

## Context parallelism: `methods.attention`

The core of softmax attention (logits, probabilities, values) is a method of
`methods.attention`, `(q, k, v, args) -> out`: queries `(b, t, h, k)`, keys and
values `(b, t, g, k)` and `(b, t, g, v)` of `g` kv heads at the queries' positions,
and the head outputs `(b, t, h, v)`. A query sees no key ahead of it; the
`AttentionArgs` (`bux.parallel.context`) carries the rest: the documents' ids
(`Segments.ids`, made in the global view before any method), the window, the scale,
the soft cap and the sinks. Every method takes its softmax in float32.

```
mesh    = mesh(context=4, tensor=2)
methods = methods(attention=ring)
```

| method    | what it does                                                         |
|-----------|----------------------------------------------------------------------|
| `auto`    | the default. On a GPU, the fastest kernel that takes the call on each device's whole rows: cuDNN's in rows of 8192 positions or more and with a window, else tokamax's. With `seq` over `context`, `ulysses` where `context` divides a device's heads, else `ring`. `xla` on a CPU, and for what only `xla` has |
| `xla`     | all t by t logits at once. With `seq` over `context` a device has its queries' rows, and every key and value: all-gather CP |
| `cudnn`   | cuDNN's fused attention (jax's) on each device's whole rows, documents in its packed layout. bfloat16 or float16, keys' and values' head dims equal and up to 256 (its packed layout gets MLA's wrong); no soft cap, no sinks, no positions over `context` |
| `tokamax` | tokamax's flash attention, its Mosaic kernels: a document and a window as each query's first key, sinks by the logsumexp. With `seq` over `context`, a device gathers every key and value and attends with its queries in one call: all-gather CP, whose backward pass does every block. Linux only (the `cuda` extra) |
| `ring`    | every key and value gathered once over `context`, and a flash call (tokamax's; jnp's on a CPU) for each device's block of them, merged by their logsumexps. A block no query sees is skipped: one ahead of them all, or before every query's first key, its document's or its window's. Without documents the queries go zigzag, device `i` holding chunks `i` and `2c-1-i` of `2c`, so that every device attends to as much. The backward pass runs each block's kernel backward with the final output and logsumexp, and reduce-scatters the blocks' gradients |
| `ulysses` | all-to-alls over `context`, from a device's positions of every head to every position of its own heads; `auto`'s kernel on those whole rows; and back. `context` must divide a device's heads; kv heads it does not divide are repeated |

What only `xla` has stays in the layer: relative bias, length scale and attention
dropout. A layer with one of them under a method other than `auto` raises
`NotImplementedError` naming it, as a method does for a feature of the
`AttentionArgs` it lacks.

Sparse attention (an indexer's top-k, CSA's compressed latents) attends by
`bux.architecture.sparse` under `xla`, `tokamax` and `auto`, and raises under the
others. Its keys come in parts (CSA's window, then its latents) that share one softmax,
merged by their logsumexp in a custom VJP. On a GPU tokamax's flash kernel runs each
part with the kept keys as a boolean mask: dense compute, exact, and t² bits of mask
forward and bytes backward. Under `xla`, off a GPU, or in `bux.kernels.off()`, XLA runs
it a chunk of queries at a time. Either way it is a `shard_map` that gives a device its
rows, queries and heads, and every key: all-gather CP. The indexer's loss takes the
attention's probabilities again from its logsumexp, a chunk of queries at a time.

`bux.parallel.flash` has the kernels. A `Kernel` attends to a block of keys: its
forward pass returns each query's logsumexp beside its output, and its backward pass
takes one. Given the output and logsumexp of a larger softmax, over more blocks or a
sink beside the keys, one block's backward pass is that softmax's gradient through
the block, as FlashAttention-2's recomputes `P = exp(s - lse)`. So `ring` merges blocks
by their logsumexps (`flash.merge`), and a sink folds into one, `y = o exp(lse - L)`
with `L = logaddexp(lse, sink)`, the backward pass run with `L` (`flash.attend`).
tokamax's kernels are one such kernel (its VJP, given the residuals `(lse, 1)`), and
`flash.JNP` the same in jnp, all t by s logits at once, which a CPU runs: with
`flash.TOKAMAX = flash.JNP` the tests check `tokamax`'s and `auto`'s paths too. A new
kernel is a `flash.Kernel(forward, backward)`.

cuDNN's attention in jax neither returns nor takes a logsumexp, so it runs whole rows
only. Its packed layout has room for a static number of documents a row, and its
backward pass takes time in proportion, and 8 bytes for each position of each head's
channels of each document's room: `context.documents_bound` makes room for up to t / 256
and 2^30 such numbers (16 for 8 rows of 4096 or one of 32k), and a `lax.cond` gives a
batch with a row of more documents to tokamax. In rows of 4096 that `lax.cond` costs a
1.3B what cuDNN gains it (145k tok/s on a B300 either way), so `auto` takes cuDNN
from rows of 8192 on: in rows of 16384 of packed documents, 1.3x tokamax's speed.

Every kernel runs in a `shard_map` manual over every axis (`context.on_devices`),
each device on its rows and heads: XLA cannot split a kernel's call, an all-to-all in
a `shard_map` of auto axes makes XLA gather every dimension they split, and cuDNN's
partitioner fails beside manual axes. Kernels need `check_vma=False`, under which the
`shard_map` sums the sinks' gradient over the devices that share them. Inside a
`shard_map`, `shard` and `spec` leave out the axes it maps, so that a layer's
constraints still hold there.

Another method is a function:

```python
from bux.parallel import methods
from bux.parallel.context import AttentionArgs


@methods.attention.register("mine")
def _(q, k, v, args: AttentionArgs):
    # args.ids (b, t) or None, args.window, args.scale, args.softcap, args.sinks (h,)
    if args.sinks is not None:
        raise NotImplementedError("methods.attention = mine has no sinks")
    return methods.attention.find("xla")(q, k, v, args)
```

On a B300, in bfloat16, 16 heads and 8 kv heads of 128 dims, a forward and backward
pass (ms):

| kernel    | causal, 8 rows of 4096 | causal, 32k | 13 documents, 32k | window 4096, 32k |
|-----------|-------|------|------|------|
| `cudnn`   | 2.82  | 12.2 | 3.3  | 5.0  |
| `tokamax` | 3.08  | 16.0 | 14.1 | 14.5 |

cuDNN skips what no query sees, forward and backward; tokamax's backward pass skips
only what is ahead of the queries, which a window as each query's first key keeps
(its own window mask does every block: 29.2 ms). cuDNN's room for documents costs:
the row of 13 takes 3.3 ms with room for 16, 3.9 for 32, 5.0 for 64 and 7.4 for 128.

On 8 B300s, `context=8`, one row, 6 calls in a jit to spread the ~2.5 ms a call on 8
GPUs takes, a forward and backward pass (ms):

| t, documents   | one GPU's tokamax / 8 | `ulysses` | `ring` | `tokamax` |
|----------------|------|----------|----------|------|
| 32k, one       | 2.06 | **2.39** | 5.24     | 5.08 |
| 128k, one      | 31.9 | **26.0** | 38.3     | 68.9 |
| 32k, of 2048   | 1.71 | **1.89** | 2.29     | 4.48 |
| 128k, of 2048  | 26.0 | 27.6     | **8.06** | 57.4 |

`ulysses` runs cuDNN, but past its room for documents (64 of them in a row of 128k)
tokamax, whose backward pass does every block; `ring` skips most blocks of packed rows
on every device. The `ring` this one replaced sent the blocks round by ppermute (~85
GB/s on a B300 node, an all-gather ~550) into an online softmax of their logits: 33.4
and 440 ms causal.

In a model, on 8 B300s: 8 layers of `d_model = 1024`, 16 heads and 8 kv heads of 128
dims (218M), rows of 32k over `mesh(context=8)`, `remat=full`, 8 microbatches,
bfloat16 (tok/s):

| `methods.attention` | one document a row | packed documents (46 a row) |
|---------------------|--------------------|-----------------------------|
| `ulysses`, `auto`   | **252k**           | **273k**                    |
| `ring`              | 175k               | 221k                        |
| `tokamax`           | 200k               | 211k                        |

With sinks, 243k by `ulysses` (tokamax's kernel), where `auto` took `xla` at 66k;
`ulysses` ran `xla` at 96k, and the `ring` it replaced 75k. In a model `ulysses` is
the faster even with 45% of the batches past cuDNN's room for documents, which
tokamax then takes, so `auto` takes it; `ring` stays the method for longer packed
rows, as at 128k above.

On 8 CPU devices, two windowed GQA layers (4 heads, 2 kv heads) with sinks and a soft
cap train on packed rows as on one device, to 1e-5 over 3 steps, by every method on
`context=2, tensor=2` and `context=4, fsdp=2`, and none makes XLA rematerialize an
array in full; on `context=8` too, every method's attention and its gradients are
`xla`'s (not `ulysses`' with 4 heads). The collectives of the compiled step of two
windowed layers (all-gathers / all-reduces / collective-permutes / all-to-alls; a
collective in a loop counts once):

| mesh                  | `xla`           | `ring`          | `ulysses`       |
|-----------------------|-----------------|-----------------|-----------------|
| `context=2, tensor=2` | 10 / 20 / 4 / 4 | 8 / 20 / 5 / 8  | 3 / 20 / 4 / 20 |
| `context=4, fsdp=2`   | 31 / 8 / 4 / 4  | 29 / 8 / 5 / 8  | 24 / 8 / 4 / 20 |
| `context=8`           | 10 / 4 / 4 / 4  | 8 / 4 / 5 / 8   |                 |

`xla` and `ring` gather each layer's keys and values whole, `ring` once forward and
once backward, where it reduce-scatters their gradients; with documents nothing goes
round. `ulysses` gathers no activation but the documents' ids, and adds 8 all-to-alls
a layer. The rest are FSDP's parameters, and int32 and boolean arrays of the
documents' starts. The temp memory of the gradient of one attention at t = 2048 on
`context=4`: `xla` 68.5 MiB, `ulysses` 81.0 MiB (whole rows of a quarter of the heads),
`ring` 26.5 MiB (every block of keys, and no logits).
`scripts/gpu_smoke/attention.py` runs every method by its GPU kernels against `xla`.

## The other layers

Dense layers split by hidden unit, as attention does by head; the embedding, the head
and the logits by the vocabulary; and the recurrent mixers by head (Mamba's by
channel), so that each device runs the recurrence of its own heads, with no
communication.

| layer                      | parameter                                       | names            |
|----------------------------|-------------------------------------------------|------------------|
| MLP                        | `up`, `gate`, their biases                      | `(embed, mlp)`   |
| MLP                        | `down`                                          | `(mlp, embed)`   |
| embedding                  | the token table                                 | `(vocab, embed)` |
| embedding                  | learned positions                               | `(none, embed)`  |
| head                       | `proj`, its bias or the tied head's             | `(embed, vocab)` |
| linear attention, RWKV-7   | `q`, `k`, `v`, `r`, `a`, `b`, the output gate   | `(embed, heads)` |
| linear attention, RWKV-7   | a low-rank projection's `down`, `up`            | `(embed, latent)`, `(latent, heads)` |
| linear attention, RWKV-7   | `o`                                             | `(heads, embed)` |
| linear attention, RWKV-7   | `A_log`, `dt_bias`, `slopes`, `k_k`, `k_a`, `r_k` | `(heads, ...)` |
| Mamba, short conv          | the conv's channels, `dt_proj`, `dt_bias`, `A_log`, `D` | `mlp`    |
| Mamba, short conv          | `x_proj`; `out_proj`                            | `(mlp, none)`; `(mlp, embed)` |
| Mamba-2                    | `dt_bias`, `A_log`, `D`; `out_proj`             | `(heads,)`; `(heads, embed)` |
| Engram                     | the table                                       | `(vocab, none)`  |

Flattened heads are named with their count, as attention's are. What is fused stays
whole, as GQA's fused `qkv` does: the SSMs' `in_proj` (u and the gate side by side),
Mamba-2's conv over u, B and C, Engram's `kv` and nextn's `proj`. Every device
computes them. Splitting such a projection in parts would keep its values under
AdamW, but not under Muon, which orthogonalizes it as one matrix. Linear attention's
one conv over q, k and v convolves each apart (`CausalConv.side_by_side`), so that
they stay on their heads. Norms, the residual algebras, RWKV-7's token shift, `scale`
and `linear` are not named: their widths mean whatever comes before them, and FSDP
alone splits them.

Constraints:

- Each mixer's input is `(batch, seq, none)`: with `layout(stream=tensor)` it is
  gathered once before the projections, not their weights. (The MLP gathers its
  input without one.) Per-head arrays are `(batch, seq, heads, ...)`, the hidden units
  `(batch, seq, mlp)`, and the SSMs' channels before `out_proj` theirs.
- The embedding gathers from its table whole but for the vocabulary, into
  `(batch, seq, none)`, then `shard_stream`. From a table split over `fsdp`, XLA
  rematerializes it in full; into positions split over `tensor`, it gathers the
  table every step. Engram gathers from its table the same way.
- The logits are `(batch, seq, vocab)`, tied heads too, and the loss reduces over the
  vocabulary where it is: the log-normalizer and the target's logit are all-reduces
  of `(b, t)`, for the loss, `z_loss` and `mtp` alike. Under `methods(loss=chunks)`
  (`bux/architecture/logits.py`) they are made a chunk of each device's positions at
  a time, `(batch, seq, none, vocab)` over the devices' shards; the head's kernel is
  gathered over `fsdp` once for all the chunks, and each device adds its share of the
  kernel's gradient up on its own, reduced once after the last chunk.
- What the embedding, nextn and Engram return is the stream, widened or not, and
  goes through `shard_stream`.

On the 8 fake devices, every family trains as on one device to 4e-6 under
`fsdp=2 × tensor=4`, `fsdp=4 × tensor=2` with sequence parallelism, `tensor=2`,
`tensor=4` and `fsdp=8`, with no involuntary rematerialization. The tests train at
`lr=1e-3`: at `1e-2` their small runs are unstable, and rounding (from `data=8` too)
grows past 1e-5 in three steps.

## Context parallelism for recurrent mixers: `methods.recurrence`

Every recurrent mixer keeps a state `S` of `k × v` per head, which each position decays,
adds to and reads its outputs from, linearly. So a chunk of positions maps the state it
starts from to the one it ends with by an affine map, its transition `S ↦ A S + B`,
and each of its outputs gains `reads_t · S`. `A` is zero across a document start.

```
mesh    = mesh(context=4, tensor=2)
methods = methods(recurrence=pass_state)
```

| method       | what it does                                                     |
|--------------|------------------------------------------------------------------|
| `scan`       | the default: every position as one chunk, which `chunkwise` runs by chunks of 64 positions (or cuDNN's kernel, below). With `seq` over `context`, XLA gathers the sequence onto every device first |
| `pass_state` | each device scans its own positions from a zero state and keeps their transition; the transitions fold, in turn, into the state each device's chunk starts from (LASP-2: XLA all-gathers them over `context`); and each device adds to its outputs what they read of that state |

| family                                | `A`, a head          | `reads_t`  |
|---------------------------------------|----------------------|------------|
| DeltaNet, Gated DeltaNet, KDA, RWKV-7 | a `k × k` matrix     | `P_tᵀ q_t` |
| GLA                                   | a diagonal of `k`    | `P_t q_t`  |
| Mamba                                 | a diagonal of `n`    | `P_t C_t`  |
| Lightning, Mamba-2                    | a scalar             | `P_t q_t`, `P_t C_t` |

`P_t` is the product of the decays (`diag(w) + b aᵀ`, or `exp(dt A)`) from the chunk's
start to position `t`, carried by the chunk's scan beside the state. For a matrix (and
k = v) that doubles the scan's work and what its backward saves; a diagonal or a scalar
costs a vector. The outputs `reads` are `(b, t, h, k)`, which for Mamba (one head per
channel) is the size of the states its backward saves.

`pass_state` is two `shard_map`s manual over every axis that splits the tokens, and
neither holds a collective: the first runs each device's chunk, the second corrects its
outputs, and the fold between them is in the global view. Inside a `shard_map` of auto
axes, a collective makes XLA gather what they split: an all-gather of the transitions
there gathered the heads over `tensor` too, so that every device folded all of them.
Where nothing splits the positions, `pass_state` is `scan`.

A method is `(recurrence) -> out`, the outputs `(b, t, h, v)` in float32. A
`Recurrence` (`bux.parallel.recurrence`) names its `family` and holds `chunk`, which
runs it over a chunk of positions from a zero state, with `xs`, the `(b, t, ...)` arrays
of positions, and `shared`, what it takes whole (Mamba's `A`). `chunk(xs, shared,
transition)` gives a `Chunk`: its outputs, the state it ends with and, with
`transition`, `A` and the reads. `recurrence` (and so `delta_rule` and RWKV-7) and
`selective_scan` run by the run's method; `scan` is in `linear_attention.py`. A method
that cannot run a family raises `NotImplementedError` naming it:

```python
from bux.parallel import methods
from bux.parallel.recurrence import Recurrence


@methods.recurrence.register("mine")
def _(r: Recurrence):
    if r.family == "the selective scan":
        raise NotImplementedError("methods.recurrence = mine has no selective scan")
    return r.chunk(r.xs, r.shared, False).out
```

Measured on 8 CPU devices, two blocks of each family under `pass_state`: on packed rows
whose documents cross the devices' chunks, the logits and gradients are one device's
(to 1e-5 of each gradient's largest entry, as `scan` on the same mesh is) on
`context=2 × tensor=2` and `context=4 × fsdp=2`; a run trains as on one device to 2.4e-6
over 3 steps on those and `context=8`; and nothing is rematerialized in full, by either
method. The all-gathers of the compiled train step
(`seq_len = 24`, 8 rows, width 32), `scan` → `pass_state`; of `scan`'s, 12 to 20 gather
a float32 array of every position, and none of `pass_state`'s does:

| family    | `context=2, tensor=2` | `context=4, fsdp=2` | `context=8` |
|-----------|-----------------------|---------------------|-------------|
| DeltaNet  | 24 → 16               | 58 → 50             | 21 → 13     |
| GDN       | 26 → 16               | 68 → 58             | 23 → 13     |
| KDA       | 26 → 16               | 72 → 62             | 23 → 13     |
| GLA       | 18 → 12               | 56 → 50             | 17 → 11     |
| Lightning | 20 → 14               | 48 → 42             | 15 → 9      |
| RWKV-7    | 32 → 22               | 91 → 81             | 23 → 13     |
| Mamba     | 20 → 14               | 84 → 76             | 17 → 11     |
| Mamba-2   | 36 → 30               | 72 → 64             | 17 → 11     |

`pass_state` all-gathers 4 small arrays per layer: the states and transitions forward,
again for the backward. Other collectives do not change, but for Mamba's and Mamba-2's
`A`, whose gradient is summed over the token axes (one all-reduce per layer on
`context=8`). The temp memory of the train step at `seq_len = 1024` (8 rows, width 64),
`scan` → `pass_state`, in MiB:

| family    | `context=4, fsdp=2` | `context=8`   |
|-----------|---------------------|---------------|
| GDN       | 26.7 → 21.2         | 41.3 → 21.3   |
| GLA       | 18.8 → 11.3         | 29.3 → 11.2   |
| Lightning | 11.3 → 9.0          | 14.6 → 8.9    |
| RWKV-7    | 26.9 → 20.2         | 41.8 → 23.1   |
| Mamba     | 136.2 → 70.6        | 248.1 → 70.8  |
| Mamba-2   | 73.1 → 40.9         | 119.4 → 40.8  |

Both meshes give a device 1024 tokens: `pass_state` holds those, `scan` every position
of its rows. On `context=2 × tensor=2` a matrix family's carried `P` costs about what
the halved sequence saves. Under `precision(compute=bfloat16)` it trains on CPU too:
what the `shard_map`s' transposes sum is float32.

A chunk runs by `chunkwise` (`bux.architecture.chunkwise`), the form of fla's chunked
kernels, under every method: matmuls within chunks of 64 positions, and a scan of the
state across them, by a Pallas kernel on a GPU (`affine_scan`). `stepwise`, a scan
over positions, is its reference. With one decay a head, the state passes across the
chunks by a matmul (SSD's state passing), and q and k may be shared by groups of heads.
Mamba-2's selective scan runs so, as decayed linear attention with `q = C` and `k = B`
of its groups, `v = dt x` and a log decay `dt A` a head, and `ssm.stepwise` is its
reference; Mamba's, a decay per channel and state dim, runs by that scan over
positions. Under `scan`, the delta rule of a gate bounded at
-5 or above (KDA's `gate_lower_bound`, DeltaNet's none) runs by cuDNN's fused KDA
kernel on a B200 or B300 instead (`fused_delta`); it overflows below -5.
`scripts/gpu_smoke/recurrence.py` checks the forms against their scans on a GPU,
whose fusions the CPU tests do not see: on a B300 one took Mamba-2's gradients for B
and C wrong, which a plain matmul in its place does not.

Not yet: a `ppermute` chain instead of the all-gather, which holds every device's
transitions (`c` matrices `k × k` per head) on each.

## Mixture of experts

The router runs in the global view: it picks each token's top-k experts, reports the
balance loss and moves the router bias. A method of `methods.moe` then runs the picked
experts, `(moe, z, idx, weights) -> y`: `z` the tokens `(b, t, d)` in the experts'
width (the latent one in a LatentMoE), `idx` and `weights` their picks `(b, t, k)`,
and `y` their weighted sum `(b, t, d)`, before `latent_output_norm`.

| method   | what it does                                                         |
|----------|----------------------------------------------------------------------|
| `sorted` | the default: sorts the picks by expert, runs each expert on its rows alone, and unsorts; drops no token |
| `dense`  | every expert on every token, weighted by the router: the reference   |

On a mesh, `sorted` sorts each device's own tokens, in a `shard_map` over the axes
that split rows and positions, with every expert's kernels gathered onto every device:
`expert` then acts as a batch axis, as it does under `dense`. Splitting the experts
themselves is expert parallelism, a method of its own.

Its pieces are functions of arrays alone, in `bux.architecture.moe`, so that a method
can run them on one device's experts inside a `shard_map`:

- `experts(x, sizes, up, gate, down, ffn, norm)`: the expert FFN of rows sorted by
  expert, `sizes[i]` of them expert i's, by `grouped.ragged_dot`: tokamax's Mosaic
  kernel on a GPU, and its Triton one for the kernels' gradient (27 times XLA's
  `jax.lax.ragged_dot` on a B300), or XLA's. Rows past `sizes.sum()` come out 0,
  before `norm`.
- `routed(z, idx, weights, up, gate, down, ffn, norm)`: sort, `experts`, unsort. A
  pick outside the experts of `up` adds nothing, so a device that holds experts
  `[i, i + n)` passes `idx - i` and its own kernels. The sort and the unsort are
  `pick_rows` and `permuted`, whose gradients gather by the inverse permutation
  where XLA's would scatter-add.

```python
from bux.architecture.moe import routed
from bux.parallel import methods


@methods.moe.register("mine")
def _(moe, z, idx, weights):
    return routed(z, idx, weights, *moe.kernels(), moe.config.expert, moe.expert_norm)
```

The MoE names its parameters: the router `(embed, None)`; `up` and `gate`
`(experts, embed, mlp)` and `down` `(experts, mlp, embed)`, with `latent` for `embed`
in a LatentMoE; `to_latent` `(embed, latent)` and `from_latent` `(latent, embed)`.
`dense` constrains its hidden units to `(batch, seq, experts, mlp)`, so that under
FSDP the weights move to the tokens rather than the tokens to the weights.

## Expert parallelism

Three methods of `methods.moe` split the experts over the axes `layout.experts` gives
them, `expert` by default: each device holds its own experts' kernels, and runs them
on the picks of every device of its group.

```
mesh    = mesh(fsdp=2, expert=4)
methods = methods(moe=all_to_all)
```

| method              | what it does                                                  |
|---------------------|---------------------------------------------------------------|
| `all_gather`        | each device gathers its group's tokens, runs its own experts on them, and a `psum_scatter` brings each token its sum back |
| `all_to_all`        | each pick goes to its expert's device and back by `jax.lax.all_to_all`, in one buffer per peer that holds every local pick, so that none drops |
| `ragged_all_to_all` | the same, one ragged slice per peer, by `jax.lax.ragged_all_to_all`: buffers hold what every peer could send, but only what they do send moves. XLA:CPU lacks it |

Each runs in a `jax.shard_map` manual over the axes that split the tokens (those of
`batch` and `seq`), not over `expert` alone: a sort of tokens split over an Auto axis
gathers them all onto every device. The kernels enter split over the experts' axes and
whole over the others, so FSDP gathers a device's own experts alone; `tensor` stays
Auto. Routing, the balance loss and the router bias run before, in the global view,
and do not change.

Where no axis splits the experts (`expert=1`, or a number of experts it does not
divide), each method is `sorted`. Experts over an axis that leaves the tokens whole,
as `layout(experts=tensor)` does, split the tokens over it too inside the method;
rows that do not split over it are an error.

A method is a body that runs on one device's tokens, and `experts.method` makes it
one of `methods.moe`:

```python
from bux.parallel import experts, methods


def mine(z, idx, weights, kernels, *, axes, ffn, norm):
    # This device's tokens and picks, and its experts' kernels: experts
    # [i * n, (i + 1) * n) for n = len(kernels[0]), i = jax.lax.axis_index(axes).
    ...


methods.moe.register("mine")(experts.method(mine))
```

`experts.ragged` takes the ragged exchange as `exchange=`, so another collective is a
function: the tests run it on CPU through one built from `all_to_all`, and
`scripts/gpu_smoke/experts.py` runs the real one, and every other method, on GPUs.

Collectives in the compiled train step of the two-MoE model of
`tests/test_experts.py` on 8 CPU devices, their number and the KiB their results take
on one device:

| mesh                       | method       | all-gather | all-to-all | reduce-scatter | all-reduce |
|----------------------------|--------------|------------|------------|----------------|------------|
| `expert=8`                 | `sorted`     | 8, 392     |            |                | 2, 678     |
|                            | `all_gather` | 10, 76     |            | 6, 8           | 2, 294     |
|                            | `all_to_all` | 2, 8       | 10, 258    |                | 2, 294     |
| `data=2, fsdp=2, expert=2` | `sorted`     | 54, 879    |            |                | 6, 808     |
|                            | `all_gather` | 56, 512    |            | 6, 8           | 8, 616     |
|                            | `all_to_all` | 48, 495    | 10, 64     |                | 8, 616     |
| `fsdp=2, expert=4`         | `sorted`     | 54, 783    |            |                | 6, 808     |
|                            | `all_gather` | 56, 433    |            | 6, 8           | 6, 520     |
|                            | `all_to_all` | 48, 399    | 10, 129    |                | 6, 520     |

`sorted` gathers every expert's kernels and all-reduces their gradients; the others
move tokens instead. Dropless with buffers of a fixed size, `all_to_all` moves
`top_k` times the rows `all_gather` does, as every peer's buffer holds every local
pick: it is the CPU sibling of `ragged_all_to_all`, which moves the picks alone. No
method rematerializes anything on these meshes, nor on `fsdp=2, expert=2, tensor=2`.

On 8 B300s (`scripts/gpu_smoke/experts.py`, a small MoE), every method computes what
one GPU does; `sorted` is the fastest there, 400k tokens a second, then `all_gather`
and `all_to_all` (335k), then `ragged_all_to_all` (270k). On jax 0.12 NCCL's ragged
all-to-all hangs unless XLA_FLAGS has
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, which
`bux train` and `bux launch` add to it, and so does `bux.runtime.configure()` in a
script that calls it before it touches a device.

## Muon: `methods.orthogonalize`

Newton-Schulz takes fifteen products of each update matrix per step. On a matrix split
over devices, most of them need an all-reduce, so how Muon reaches whole matrices is a
method:

| method   | what it does                                                                 |
|----------|------------------------------------------------------------------------------|
| `gspmd`  | Newton-Schulz on whatever shards the compiler gives each update              |
| `gather` | each update whole in a `jax.shard_map` for Newton-Schulz, and back on its parameter's shards after |
| `owner`  | each matrix whole on one device, its owner, which runs its matrices of a shape as one batch; the default |

```
methods = methods(orthogonalize=gather)
```

Under `gather`, a stack of matrices (the experts) is whole matrix by matrix, and its
stacked dimension splits over the axes that frees, where they divide it: each device
orthogonalizes its own experts. A lone matrix is whole on every device, and every
device runs its Newton-Schulz.

`owner` spreads that work, as Moonlight's and Dion's distributed Muon do. Every matrix,
a stack's one by one, has one owner. The matrices of a shape and split, from every
update of the step, are dealt out evenly over the devices, padded with zero matrices
to a multiple of them, so that each device does the same work, and the work of a shape
(m·n·min(m, n) a matrix) is split evenly. One all-to-all takes each device's shards of
them to their owners, which orthogonalize their matrices of a shape as one batched
chain of products; another brings the results back to the parameters' shards, and an
all-gather copies them over the axes that do not split the parameter (`data`, say).
Under `layout(zero=1)` an update comes split by FSDP, as the optimizer's state is, and
its matrices go back whole along FSDP's side by an all-gather, as the parameter is.

A method gets every update of a step at once, so that it can choose where each runs:

```python
from bux.parallel import methods


@methods.orthogonalize.register("mine")
def _(updates, dims, shardings, newton_schulz):
    # updates: an array per Muon parameter, in muon's ns_dtype; dims: their
    # MuonDimensionNumbers; shardings: their parameters' NamedShardings, None
    # outside parallel.use; newton_schulz(x, dims): optax's, in x's dtype.
    return [newton_schulz(x, d) for x, d in zip(updates, dims)]
```

Collectives in the compiled train step on 8 CPU devices (all-reduces / all-gathers /
all-to-alls) for the 2-block Llama and the MoE of `tests/test_parallel.py`, Newton-Schulz
in float32:

| model, mesh                         | AdamW       | Muon, `gspmd` | Muon, `gather` | Muon, `owner` |
|-------------------------------------|-------------|---------------|----------------|---------------|
| Llama, `fsdp=8`                     | 22 / 26 / 0 | 73 / 86 / 0   | 22 / 40 / 0    | 22 / 26 / 6   |
| Llama, `fsdp=4, tensor=2`           | 34 / 33 / 0 | 133 / 213 / 0 | 34 / 61 / 0    | 34 / 33 / 16  |
| MoE, `fsdp=8`                       | 24 / 31 / 0 | 65 / 181 / 0  | 24 / 46 / 6    | 24 / 31 / 8   |
| MoE, `fsdp=4, tensor=2`             | 36 / 35 / 0 | 164 / 250 / 0 | 36 / 73 / 0    | 36 / 36 / 18  |
| Llama, `fsdp=8`, `zero=1`           | 22 / 21 / 0 | 73 / 81 / 0   | 22 / 21 / 0    | 22 / 10 / 3   |
| Llama, `fsdp=4, tensor=2`, `zero=1` | 34 / 21 / 0 | 133 / 201 / 0 | 34 / 35 / 0    | 34 / 11 / 12  |

With `gather`, Muon all-reduces as often as AdamW, and adds one all-gather per matrix
and two all-to-alls per expert stack; the optimizer's update alone does no all-reduce.
With `owner`, Muon all-reduces and all-gathers as AdamW does, and adds one all-to-all
each way for every shape and split of matrix, two where a matrix splits on both sides
(`tensor`). Under `zero=1` its few all-gathers of whole matrices replace the one each
parameter takes otherwise. Step times on fake CPU devices, which share one CPU, say
nothing.

On 8 B300s, `mesh(fsdp=8)`, 64 rows of 4096 tokens a step computed in bfloat16,
thousands of tokens a second for a 1.3B dense Llama (16 layers, `d_model=2048`, 16
query heads and 8 kv heads, a gated MLP of 5632, 128k tokens) and a 5.2B MoE of its
shape (64 experts of 704, top 8), by `ns_dtype`:

| model                | AdamW | `gspmd` | `gather` | `owner` |
|----------------------|-------|---------|----------|---------|
| dense, float32       | 1,147 | 856     | 849      | 1,073   |
| dense, bfloat16      |       | 923     | 950      | 1,087   |
| dense, `repeat=scan` |       | 863     | 938      | 935     |
| MoE, float32         |       | 531     | 712      | 754     |
| MoE, bfloat16        |       |         |          | 786     |

The dense model's optimizer update alone takes 74 ms with `gspmd`, 73 with `gather`
and 12 with `owner` in float32; 48, 42 and 12 in bfloat16. With `repeat=scan` (float32
here) the stacked `layers` already split over `fsdp` under `gather`, as under `owner`.

## Repeats and remat: `methods.repeat`, `methods.remat`

```
methods = methods(repeat=scan, remat=dots)
```

`f^n` builds `n` passes of `f`. With `repeat=unroll`, the default, each is a module of
its own, in a row. With `repeat=scan`, the passes are built the same way, so every
init that depends on depth (hc's, vpgate's, MiniMax's) is as unrolled, and then
stacked into one `Scanned` module: each variable of the first pass holds every pass's,
along a new leading dimension named `layers`, and `jax.lax.scan` runs them. A repeat
stays unrolled where its passes do not stack:

- their graphdefs or the shapes of their variables differ (Engram's tables);
- they share a module or a variable: a `shared` definition, or nextn's embedding;
- they widen the stream as they go, as attnres does.

A scan carries the stream and whatever taps a pass replaces, such as single-pass
mhc's read, and collects the losses each pass reports (the MoE's balance loss) and the
state it changes (the router bias). Before it scans, a trial pass, traced but not
run, checks that the shapes of the stream and of the taps come out as they went in.
If they do not from the first pass, the first runs alone and the scan starts at the
second: single-pass mhc's first pass starts its read. If they never do, as when every
pass pushes to a tap (a U-net's `(block @skip)^6`), the passes run one by one, off the
same stacked state.

The `layers` dimension is a name of the layout, `layers` (none by default), which FSDP
never takes: a scan reads a layer at a time. Weight decay and Muon count from after
it, so a stacked norm scale is still a vector, and a stacked kernel a batch of
matrices.

The state of a scanned repeat has another structure than the unrolled one's, but the
run's id is the same, as `methods` is not in it. A checkpoint therefore resumes only
with the `repeat` it was saved with; with the other, the restore fails at once and
says so.

`remat` rematerializes in the backward pass what the forward does not keep, by a
`jax.checkpoint` policy:

| method               | what the forward keeps                                     |
|----------------------|------------------------------------------------------------|
| `none`               | everything, but in a scanned pass (below); the default     |
| `full`               | nothing but the input of each branch or pass               |
| `dots`               | the products of matrices                                   |
| `dots_no_batch`      | the products of matrices without batch dimensions          |
| `save_attention`     | the mixers' outputs, `attention` and `recurrence`          |
| `save_qkv_attention` | those, and attention's projections, `qkv`                  |
| `dots_attention`     | `dots_no_batch`'s, the mixers' outputs and `experts`       |
| `offload_attention`  | the mixers' outputs, in the host's memory                  |

Each residual branch outside any other is rematerialized as a whole, with the read
and write of the residual algebra; a scanned pass is, as one; and nothing inside
either is again. In a `.bux` file `none` is the null value, so a sweep writes the
method as a string: `methods.remat = "none" | full`.

The last four keep what layers name (`jax.ad_checkpoint.checkpoint_name`), as
MaxText's policies do:

| name         | what it is                                                           |
|--------------|----------------------------------------------------------------------|
| `attention`  | softmax attention's output, and what its backward takes              |
| `recurrence` | the same of a recurrent mixer: linear attention, the SSMs, RWKV-7    |
| `qkv`        | attention's q, k and v projections, MLA's q and kv                   |
| `mlp`        | an MLP's up and gate projections                                     |
| `experts`    | the MoE's grouped products                                           |

What a kernel's backward takes is no product of matrices: flash attention's takes its
output and log-sum-exp, so under `dots` the backward runs the kernel's forward again.
`remat.named(name, f, *args)` names f's output and everything its backward takes, so
that a policy that keeps the name runs none of f twice; attention's method and the
recurrent mixers' run through it. Another policy is a function:

```python
import jax

from bux.parallel import methods

methods.remat.register("save_mlp")(
    jax.checkpoint_policies.save_only_these_names("attention", "mlp")
)
methods.remat.register("offload")(
    jax.checkpoint_policies.offload_dot_with_no_batch_dims("device", "pinned_host")
)
```

A scan keeps every value its backward reads, stacked by pass. Unrolled, XLA fuses the
elementwise ones into the ops that read them and keeps none: a norm's float32 copy of
the stream, an MLP's activation, a token's copy for each of its experts. So under
`none` a scanned pass keeps all but what elementwise ops and those that only move or
retype data make (`remat.CHEAP`), and its backward computes those again
(`remat.unfused`). Nor does a training step cast the stacked parameters to its compute
dtype, which would keep a copy of them all, and of their gradients: each pass casts its
own as it runs (`Scanned.compute`). On a B300, for a 1B dense model (`A^16`), a Gated
DeltaNet hybrid (`"DDDA"^4`) and a 64-expert MoE (`E^16`), in bfloat16 on 8 rows of
4096 tokens, the temp memory of the compiled train step, and tokens a second:

| model         | memory: unrolled | scanned, before | scanned  | tokens: unrolled | scanned |
|---------------|------------------|-----------------|----------|------------------|---------|
| dense         | 56.2 GiB         | 90.7 GiB        | 47.3 GiB | 145k             | 138k    |
| GDN           | 124.9 GiB        | 213.8 GiB, OOM  | 86.7 GiB | 94k              | 78k     |
| MoE           | 95.8 GiB         | 131.7 GiB, OOM  | 80.0 GiB | 106k             | 91k     |
| dense, fsdp=8 |                  |                 |          | 592k             | 670k    |

("Before" kept every value its backward reads, and the cast copy.) On one GPU a scan
is still slower: it copies each value it keeps into its stack and back out, where
unrolled layers read it where it was made, and the more it keeps the more it copies:
the GDN hybrid's float32 chunk states. On eight, one row each, it is faster, and it
compiles 2.3 times as fast.

On one CPU, for the gradient of a 24-layer Llama's forward (`d_model=64`, 8 rows of
128 tokens), and for its first training step, which compiles it:

| repeat   | trace + compile | first step, AdamW | first step, Muon |
|----------|-----------------|-------------------|------------------|
| `unroll` | 1.2 s           | 2.9 s             | 6.9 s            |
| `scan`   | 0.3 s           | 0.8 s             | 0.9 s            |

| remat                | temp memory, unrolled | scanned |
|----------------------|-----------------------|---------|
| `none`               | 327 MiB               | 142 MiB |
| `full`               | 22 MiB                | 23 MiB  |
| `dots`               | 132 MiB               | 138 MiB |
| `dots_no_batch`      | 79 MiB                | 82 MiB  |
| `save_attention`     | 142 MiB               | 142 MiB |
| `save_qkv_attention` | 142 MiB               | 142 MiB |
| `dots_attention`     | 187 MiB               | 193 MiB |
| `offload_attention`  | 142 MiB               | 142 MiB |

On a CPU, attention is `xla`'s, whose backward takes the t by t probabilities: what
`save_attention` keeps of it is more than `dots` keeps of the whole layer, and a CPU's
host memory is its device's. On a GPU, flash attention's backward takes q, k, v, the
output and the log-sum-exp. On a B300, for the dense 1.3B of 16 Llama blocks of width
2048 (vocabulary 128256, 8 rows of 4096 tokens, `bfloat16`):

| remat                | tok/s   | peak memory |
|----------------------|---------|-------------|
| `none`               | 145,000 | 81.3 GiB    |
| `dots`               | 134,000 | 69.8 GiB    |
| `save_attention`     | 132,000 | 61.9 GiB    |
| `save_qkv_attention` | 130,000 | 61.9 GiB    |
| `dots_attention`     | 139,000 | 71.8 GiB    |
| `offload_attention`  | 90,000  | 62.0 GiB    |

`dots` runs flash attention's forward twice, and `save_attention` the matmuls of every
layer; `dots_attention` runs neither, 4% slower than `none` in 12% less memory.
`save_qkv_attention` keeps what `save_attention` does here: attention's backward
takes q, k and v after RoPE, so that only a QK-norm would read the projections.
Offloading moves more than it saves, at this size. The head's logits are the largest
part of the rest, which `methods(loss=chunks)` leaves out
([The trainer's blocks](training.md#the-trainers-blocks)).

`Scanned.layers` holds the stacked passes, and `stacked.run(graphdef, state, times, x,
taps)` runs any stack of them, such as a pipeline stage's share.

## Precision and microbatches

```
precision = precision(compute=float32)    # in the run's id; bfloat16 by default
methods   = methods(microbatches=4)       # gradient accumulation: not in it
```

Unlike the settings above, `precision` changes what a run computes, so it is in the
run's id: `precision.compute = float32 | bfloat16` is a sweep of two runs. `compute` is
`bfloat16`, the default, or `float32`; float16 would need loss scaling. `matmuls` is the format the
projections inside blocks multiply in, `compute` unless it says `mxfp8` or `nvfp4`.

### Precision

Parameters, their gradients and the optimizer's state stay float32. The train and eval
steps cast the model's floating-point parameters to `compute` inside the loss
(`trainer.cast`), so that the forward and the backward compute in it, and gradients
reach the float32 parameters through the cast. What is not a parameter keeps its
dtype: the router bias and its counts, RNG state, Lightning's slopes.

Float32 islands compute from what they are given in float32, and give back its dtype:

| where                                           | what is float32                                   |
|-------------------------------------------------|---------------------------------------------------|
| `Norm.normalize`, `l2norm`                      | the statistics, and the normalized values          |
| the MoE's router                                | its logits, scores and top-k, the balance loss and the bias; the weights go back to the stream's dtype |
| `recurrence`, `delta_rule`, `selective_scan`    | the recurrent state, and its decays               |
| `rope.rotate`                                   | the rotation                                      |
| the head                                        | its matmul's accumulation, tied or not, and its log Z and loss; under `methods(loss=chunks)`, its logits |

Softmax attention takes its softmax in float32, under every method of
`methods.attention`. Every other matmul runs in `compute`, or in the format of
`matmuls` (below), and the stream stays in `compute` from the embedding to the heads,
which `tests/test_precision.py` checks for every family. Gradients are reduced over the
batch axes in the dtype the backward makes them, bfloat16 for a weight's, but the head
kernel's under `methods(loss=chunks)`, which adds up over the chunks in float32; they
add up over microbatches in float32.

XLA:CPU fails to compile a bfloat16 `psum` that a `shard_map`'s transpose makes (its
reducer ends in a copy, which `ChangeOpDataType` cannot clone), and aborts the process.
The MoE's `sorted` and EP methods make one on more than one device, and so do the
pipelines; `ring`, `ulysses` and `pass_state` train in bfloat16 on CPU. A run that
would make one says so before it compiles: on a CPU, such runs set
`precision(compute=float32)`, as `laptop.bux` does, or split an MoE by
`methods(moe=dense)`. GPUs all-reduce bfloat16 natively.

### Matmul formats

```
precision = precision(compute=bfloat16, matmuls=mxfp8)
```

`matmuls` names a format of the registry in `bux.architecture.matmuls`. A block-scaled
format quantizes the operands of all three GEMMs of y = x W, the forward's and the
backward's two, each along what its GEMM contracts, in blocks with a scale each
(`bux.architecture.blockscaled`):

| format    | values  | a scale per                                   | and                                       |
|-----------|---------|-----------------------------------------------|-------------------------------------------|
| `compute` | `compute` | none                                        | the default                               |
| `mxfp8`   | E4M3    | 32, a power of two (E8M0)                     | dy in E4M3 too, as Transformer Engine's MXFP8 recipe |
| `nvfp4`   | E2M1    | 16 (E4M3), and the tensor (float32); W's per 16 x 16 tile, so that y and dx see the same W | dy rounded stochastically, and x and dy transformed for dW by a random Hadamard transform of 16, as NVIDIA's NVFP4 pretraining recipe |

On a Blackwell GPU cuDNN's block-scaled GEMM multiplies them (`__cudnn$blockScaledDot`
in the HLO), and for `mxfp8` a Pallas kernel quantizes each of x, W and dy and its
transpose in one pass. Elsewhere XLA dequantizes and multiplies, which computes the
same values, so the tests run the formats on CPU. The forward keeps what the backward
multiplies, x^T and W in the format, rather than x and W.

Every projection made by `sharding.projection` or `sharding.linear` multiplies by
`matmuls.dot_general`, which runs the format in force: attention's q, k, v, o and
gate; the MLPs', the shared experts' and LatentMoE's; MLA's, CSA's, the indexer's; the
linear attentions', RWKV-7's and Mamba's; Engram's, nextn's and `linear`'s. These stay
in `compute`: the embedding and the head, softmax attention's own matmuls and its
relative bias, the router and the shared experts' gate, hyper-connections, norms,
low-rank gates and decays, recurrent states, and the routed experts
(`grouped.ragged_dot`, and `methods(moe=dense)`'s einsums). A projection whose tokens,
inputs or outputs are not whole blocks multiplies in `compute`. On a mesh, one that
does not split into whole blocks on every device (a multiple of the block times the
devices) stays whole on each: so the blocks are the same on any mesh, and so is what a
run computes, which `tests/test_precision.py` checks on `fsdp=4, tensor=2` and in a
pipeline. NVFP4's stochastic rounding draws its noise from a hash of each element's
index and the batch's tokens, the same on any mesh too.

Under FSDP each device quantizes its part of W, so the all-gathers move float8 values
and their scales, half of bfloat16's bytes. A projection multiplies its tokens
flattened, rows by positions, and a split along positions is not one along the
flattened tokens: a `ring` run on `context=2` trained 13% slower in `mxfp8` than in
bfloat16 on B300s.

A recipe is a function, and the trainer enters `matmuls.use` as it traces its steps:

```python
from bux.architecture import matmuls


@matmuls.formats.register("mine")
def _(lhs, rhs, dimension_numbers, **kwargs): ...  # jax.lax.dot_general's arguments
```

`matmuls.block_scaled(recipe)` makes one of a `blockscaled.Recipe`: how it quantizes x,
W and dy, each twice.

### Microbatches

`methods.microbatches = n` takes each step's batch in n microbatches: a `lax.scan`
over them inside the train step, their gradients summed in float32, and one update.
Microbatch i is rows i, i + n, i + 2n and so on (`parallel.microbatches`), so that
every device's rows split evenly among them and none moves. `batch` must split into n
microbatches whose rows split over the batch axes, which `parallel.check` says before
anything is built.

What the step computes is the whole batch's:

- **The objective's terms**, means over targets, weigh by each microbatch's share of
  the batch's targets (`trainer.token_shares`), so that the loss and the z-loss are
  the batch's, also when a loss mask leaves microbatches with unequal numbers of
  targets. `mtp`'s heads weigh by the first head's targets.
- **What layers report** weighs by each microbatch's share of rows, 1 / n: a
  balance loss over sequences (`aux_loss_scope=sequence`) and the indexer's loss are
  the batch's. A balance loss over the batch is the mean of the microbatches' own,
  each over its rows: its f_e and P_e are a microbatch's means, not the batch's.
- **The router bias** moves once a step, from the whole batch's picks. The trainer
  sets `update_at_step_end` on every module that has it: the MoE's training forwards
  then route with the bias where the step found it and add their picks to
  `router_load`, and `step_end_update` moves the bias after the last microbatch. A
  layer of one's own that learns from the batch outside of gradients does the same
  (`trainer.StepEndUpdate`). Called outside a step, a training forward moves the
  bias at once; and a shared MoE moves once a step, from the picks of all its uses.
- **Dropout** draws each microbatch's masks in turn from the run's RNG stream: the
  same distribution as the whole batch's, not the same masks, and the stream's count
  advances n times a step. A resumed run draws the same.

`tests/test_precision.py` checks the gradients, the metrics and the router bias
against the whole batch's, on one device and on `fsdp=8` and `fsdp=2, expert=2,
tensor=2`. Dropout's masks and a balance loss over the batch aside, microbatches
change how a run computes, not what, so they keep its id as the rest of `methods`
does.

Each microbatch's gradient is a global array, reduced over the batch axes inside the
scan: under FSDP, a reduce-scatter a microbatch, as ZeRO-3 does. XLA:GPU has a pass
(while-loop all-reduce code motion) meant to move a data-parallel all-reduce out of
such a loop, which this has not been run on; XLA:CPU all-reduces n times a step.
`parallel.microbatches` is there for pipeline parallelism to split its batch the same
way.

### Memory and losses

The compiled train step's temporaries on one device, for 4 Llama blocks of width 256,
`seq_len = 256`, `batch = 32` (`compiled.memory_analysis()`):

| `compute`  | 1 microbatch | 4 microbatches |
|------------|--------------|----------------|
| `float32`  | 2032 MiB     | 472 MiB        |
| `bfloat16` | 1128 MiB     | 225 MiB        |

The loss in `float32` / `bfloat16` of two blocks of width 128 on the tests' corpus of
eight words, byte by byte (`seq_len = 128`, `batch = 16`, `adamw(lr=3e-3)`, cosine):

| mixer    | step 10         | step 50         | step 200        |
|----------|-----------------|-----------------|-----------------|
| `gqa`    | 2.2348 / 2.2389 | 0.4089 / 0.4089 | 0.4209 / 0.4213 |
| `moe`    | 1.3595 / 1.3602 | 1.0057 / 1.0071 | 1.0250 / 1.0248 |
| `gdn`    | 1.5034 / 1.4998 | 0.4112 / 0.4269 | 0.4205 / 0.4204 |
| `mamba2` | 0.9262 / 0.9276 | 0.4134 / 0.4129 | 0.4211 / 0.4207 |

### Donation: `methods.donate`

A step donates the model's and the optimizer's arrays, so that its new state takes
the old one's memory where XLA can write it in place; `methods(donate=false)` keeps
them. On one B300 (bf16, 8 × 4096 tokens), the peak and the largest free piece of the
GPU's memory after 20 steps, in GiB, and tokens a second:

| model                         | kept: peak, free | donated: peak, free | tok/s, kept / donated |
|-------------------------------|------------------|---------------------|-----------------------|
| Llama-style 1.3B              | 81.3, 172.1      | 70.6, 186.5         | 143.6k / 142.9k       |
| MoE, 64 experts, top 8 (5.2B) | 163.9, 85.5      | 153.5, 143.1        | 103.8k / 106.2k       |
| Gated DeltaNet hybrid         | 148.4, 169.8     | 140.4, 185.3        | 92.1k / 91.9k         |
| KDA, `gate_lower_bound=-5`    | 183.4, 169.7     | 176.2, 185.3        | 110.9k / 114.4k       |

So more fits: donated, the 1.3B trains 36 rows a GPU and the MoE 12, which run out of
memory kept, and the MoE with a checkpoint every 10 steps trains, where kept it runs
out of memory at the step after its first save. Donation changes how XLA rounds a
step a little (the gradient's norm by 7e-6 at the dense model's first step on a GPU,
the loss by 2e-6 at zamba2's third on a CPU): runs compared bitwise donate alike. An
array read from `trainer.model` before a step is gone after it; a checkpoint is safe,
as orbax copies the state to the host before `save` returns.

Either way the trainer keeps the state in the buffers it was made in
(`parallel.shard_state` moves only what is elsewhere), and a run that resumes loads
its checkpoint into the memory of the state it replaces.

## Pipelines: `methods.pipeline`

```
mesh    = mesh(stage=4, fsdp=2)
methods = methods(repeat=scan, microbatches=8)                                # gpipe
methods = methods(repeat=scan, microbatches=8, pipeline=circular, rounds=2)
```

With `mesh.stage` above 1, the longest scanned repeat outside any other (the first of
several as long) is pipelined. Its stacked layers split over `stage` by a dimension
named `stages` of the layout, one share of consecutive passes per stage. The rest of the
model, the embedding and the head too, does not split over `stage`: every stage
computes it.
A model with no scanned repeat is an error that says to repeat its blocks and scan
them, `methods(repeat=scan)`, and so is a repeat whose passes do not split evenly over
the stages.

The batch's rows split into `methods.microbatches` microbatches, row `i·M + j` into
microbatch `j`, so that each splits over the batch axes as the rows do. A method runs
them in a `jax.shard_map` manual over `stage` alone, by a `lax.scan` over `R·M + S − 1`
ticks: at each, every stage runs its share of a round on a microbatch (`stacked.run`),
and a `ppermute` sends its output on to the next stage, the last's to the first. The
first stage takes each microbatch from a buffer of its own, where the last stage's
outputs come back: there a circular round starts again, and from there the stream
leaves, summed over the stages to every one.

| method     | each microbatch                                                   | a stage idles           |
|------------|-------------------------------------------------------------------|-------------------------|
| `gpipe`    | goes through the stages once; the default                         | (S − 1) / (M + S − 1)   |
| `circular` | goes round them `methods.rounds` times, each stage holding a share of every round's layers | (S − 1) / (R·M + S − 1) |

`circular` needs at least one microbatch per stage, and only it reads `methods.rounds`.
`bux plan` shows the share of a step a stage idles, the bubble:

```
$ bux plan examples/experiments/smoke.bux 'methods = methods(repeat=scan, microbatches=4)' \
    'mesh = one: mesh(data=1) | pp: mesh(stage=2, data=1)'
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
bbbf3ae709  156k    25.6k   0.2        1.03M      1     609KiB      1.19MiB    0%      smoke[one]
bbbf3ae709  156k    25.6k   0.2        1.03M      2     369KiB      738KiB     20%     smoke[pp]
```

A stage gets the stream and the taps of its rows (the tokens, the documents), split
with it. A pass that reads a tap written before the repeat (`$v1`), or leaves one
changed after it (single-pass mhc's read, a U-net's skips), is an error that names
the tap: a pipeline sends its stages the stream alone. Sending taps along is later.

The passes run once a microbatch, as if the microbatches went through the model one
after another, so with one microbatch a pipeline computes what one device does:

- the losses a layer reports, such as the MoE's balance loss, are the mean over
  microbatches of each one's: the whole batch's for a loss of each sequence
  (`aux_loss_scope=sequence`), near it for one of the batch;
- the router bias moves once a step, from the whole batch's picks, as it does under
  gradient accumulation: in a training step the MoE has `update_at_step_end`, and the
  pipeline carries its `router_load` through the ticks it computes. A training
  forward outside a step moves it once a microbatch, each from that one's load;
- dropout draws a fresh mask for every microbatch.

The microbatches are the pipeline's: the trainer asks `pipeline.pipelined()`, and a
step does not accumulate gradients over them as well. With `gpipe` the stacked state
keeps the shapes and the order of `repeat=scan`'s, so a run resumes on another number
of stages, or on none; `circular` stacks each round's layers as a repeat of its own,
and resumes as circular with as many rounds.

Measured on 8 CPU devices, for a Llama of 8 blocks (`d_model=64`, 16 rows of 16
tokens): how far the losses of three AdamW steps are from one device's at most, and
the collectives in the compiled gradient of a forward, those in the loop once, so
once a tick:

| mesh                       | methods                                  | losses   | collective-permute | all-reduce | all-gather |
|----------------------------|------------------------------------------|----------|--------------------|------------|------------|
| `stage=2`                  | `microbatches=4`                         | 7.2e-6   | 2                  | 2          | 0          |
| `stage=4`                  | `microbatches=8`                         | 1.8e-5   | 2                  | 2          | 0          |
| `stage=4`                  | `microbatches=8, pipeline=circular, rounds=2` | 1.8e-5 | 2             | 2          | 0          |
| `stage=2, data=2, fsdp=2`  | `microbatches=4`                         | 2.9e-6   | 3                  | 1          | 21         |
| `data=2, fsdp=4`           |                                          | 6.2e-6   | 0                  | 0          | 22         |

One ppermute per tick each way, and an all-reduce of the stream as it leaves, and of its
gradient as it comes back; no parameter of a stage leaves it.

This is not MaxText's vmap over the stages with `spmd_axis_name="stage"`: on JAX 0.12
such a vmap cannot batch a shard_map manual over some of the axes, which the MoE's and
context parallelism's methods are, inside the layers ("Axes mentioned in
manual_axis_type ... should be of type Manual"). Nested in a shard_map over `stage`
they run, on three conditions the pipeline keeps: the outer one does not check its
values' varying axes (`check_vma=False`), else the gradient of a nested shard_map
fails to lower; `stage` comes first in the mesh, as XLA aborts on a nested shard_map
manual over an axis before the outer one's ("Manual axes must be sorted by mesh axis
index"); and the pipeline takes a round's layers by dynamic slices, as XLA fails to
partition over Auto axes the scatter that a gather's gradient is.
