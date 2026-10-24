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
every row onto each device. Its scores' kernel (`bux.architecture.attention.scoring`) is
a `shard_map` that gives a device its rows, positions and heads, and sums the heads'.

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
| `ring`    | every key and value gathered once over `context`, and a flash call (tokamax's; jnp's on a CPU) for each device's block of them, merged by their logsumexps, a device's own block first, as the others' are gathered. A block no query sees is skipped: one ahead of them all, or before every query's first key, its document's or its window's. Without documents the queries go zigzag, device `i` holding chunks `i` and `2c-1-i` of `2c`, so that every device attends to as much; its own block is then a gathered one. The backward pass runs each block's kernel backward with the final output and logsumexp, and reduce-scatters the blocks' gradients |
| `ulysses` | all-to-alls over `context`, from a device's positions of every head to every position of its own heads; `auto`'s kernel on those whole rows; and back. `context` must divide a device's heads; kv heads it does not divide are repeated |

What only `xla` has stays in the layer: relative bias, length scale and attention
dropout. A layer with one of them under a method other than `auto` raises
`NotImplementedError` naming it, as a method does for a feature of the
`AttentionArgs` it lacks.

Sparse attention (an indexer's top-k, CSA's compressed latents) attends by
`bux.architecture.attention.sparse` under `xla`, `tokamax` and `auto`, and raises under
the others. Its keys come in parts (CSA's window, then its latents) that share one
softmax, merged by their logsumexp in a custom VJP. On a GPU tokamax's flash kernel runs
each part with the kept keys as a boolean mask: dense compute, exact, and t² bits of
mask forward and bytes backward. Under `xla`, off a GPU, or in `bux.kernels.off()`, XLA
runs it a chunk of queries at a time. Either way it is a `shard_map` that gives a device
its rows, queries and heads, and every key: all-gather CP. The indexer's loss takes the
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
rows, as at 128k above. That `ring` attends to its own block while the others' are
gathered changes nothing past the noise in this model (223k on packed documents
against 225k gathering it first): a layer's gather is short against its attention.

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

- Each mixer's input is `(batch, seq, none)`, and so is the MLP's: with
  `layout(stream=tensor)` it is gathered once before the projections, not their
  weights ([below](#tensor-parallelism-methodstensor)). Per-head arrays are
  `(batch, seq, heads, ...)`, the hidden units `(batch, seq, mlp)`, and the SSMs'
  channels before `out_proj` theirs.
- The embedding gathers from its table whole but for the vocabulary, into
  `(batch, seq, none)`, then `shard_stream`. From a table split over `fsdp`, XLA
  rematerializes it in full; into positions split over `tensor`, it gathers the
  table every step. Engram gathers from its table the same way.
- The logits are `(batch, seq, vocab)`, tied heads too, and the loss reduces over the
  vocabulary where it is: the log-normalizer and the target's logit are all-reduces
  of `(b, t)`, for the loss, `z_loss` and `mtp` alike. By chunks, `methods.loss`'s
  default (`bux/architecture/logits.py`), they are made a chunk of each device's
  positions at a time, `(batch, seq, none, vocab)` over the devices' shards; the
  head's kernel is gathered over `fsdp` once for all the chunks, and each device adds
  its share of the kernel's gradient up on its own, reduced once after the last chunk.
- What the embedding, nextn and Engram return is the stream, widened or not, and
  goes through `shard_stream`.

On the 8 fake devices, every family trains as on one device to 4e-6 under
`fsdp=2 × tensor=4`, `fsdp=4 × tensor=2` with sequence parallelism, `tensor=2`,
`tensor=4` and `fsdp=8`, with no involuntary rematerialization. The tests train at
`lr=1e-3`: at `1e-2` their small runs are unstable, and rounding (from `data=8` too)
grows past 1e-5 in three steps.

## Tensor parallelism: `methods.tensor`

At a layer's edges, TP communicates. Under sequence parallelism a device gathers the
layer's input over `tensor` before the projections that split by columns, and the
output projection's partial sums reduce-scatter back to its positions; without it,
they all-reduce. How those collectives meet the matmuls is a method:

```
methods = methods(tensor=chunked, tensor_chunks=2)
```

| method    | what it does                                                          |
|-----------|-----------------------------------------------------------------------|
| `gspmd`   | the default: the compiler's collectives, at the boundary of the matmul each feeds or drains, which waits for it |
| `chunked` | each device's rows in `tensor_chunks` chunks (2), in a `shard_map` manual over every axis: chunk i + 1 all-gathers while chunk i multiplies, and chunk i reduce-scatters while chunk i + 1 multiplies |

GQA's q, k and v (or its fused `qkv`) and the MLPs' up and gate, the shared experts'
too, project their input by `sharding.project_input`; their output projections, and
MLA's, by `sharding.project_output`. The other layers gather their input as `gspmd`
does. Each chunk's collective is NCCL's, at full NVLink speed, and an optimization
barrier ties it to the one before, so that XLA neither combines them nor runs them
side by side. `chunked` is `gspmd` where it has nothing to overlap (no sequence
parallelism for the input projections), where a kernel splits otherwise than by
`tensor`, and where a device's share would multiply otherwise than the whole, in
the format that projection takes in its branch (its role's, or compute in the
`first` and `last`): in `mxfp8` and `mxfp8_2d` unless the chunk's tokens and the
share are whole blocks, and in `nvfp4` always, whose scale is the whole operand's.
Where `tensor_chunks` does not divide a device's rows, it takes as many chunks as
divide both. A method is a class of two functions:

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

On 8 CPU devices `chunked` gives `gspmd`'s outputs and gradients to 1e-5, in 1, 2
and 4 chunks, on `fsdp=2, tensor=4` and `tensor=8` with sequence parallelism, on
`fsdp=2, context=2, tensor=2` with the stream over both, and on `data=4, tensor=2`
without; in `mxfp8` and `mxfp8_2d` too. With `nvfp4` for the MLPs, or for all but
the first and last branches, those stay `gspmd`'s, and the gradients they take from
chunked branches move some of NVFP4's values a step: to 1e-3. A Llama (scanned,
rematerialized, in 2 microbatches) and an MoE (rematerialized) train as on one
device, with no involuntary rematerialization.

On 8 B300s, 4 of Llama 3 70B's layers (scanned, `remat=full`, 8 rows of 4096), TP=8
with sequence parallelism, tokens a second and the exposed communication of a step
in ms of its time:

| `methods.tensor`   | bfloat16           | `matmuls=mxfp8`    |
|--------------------|--------------------|--------------------|
| `gspmd`            | 219k, 36.5 of 150  | 250k, 41.8 of 131  |
| `chunked`, 2       | **222k**, 28.2 of 148 | **263k**, 26.5 of 125 |
| `chunked`, 4       | 219k, 23.6 of 150  | 257k, 25.3 of 128  |

Chunks hide a quarter to a third of the wait, and the matmuls beside them give most
of it back: NCCL's ring kernels take 32 channels of SMs, so the GEMMs of a step take
82 ms beside 2 chunks' collectives and 88 beside 4's, against 79, and the chunks'
slices and concatenations 2.6 to 3.4 ms. With fewer CTAs (`NCCL_MAX_CTAS=8`) NCCL's
collectives take more than twice as long. On `fsdp=2, tensor=4`, `chunked` is 3% slower
(232k against 240k). In MXFP8 it is 5% faster: `gspmd`'s block-scaled output
projections all-reduce their partial sums (32 ms a step), where `chunked`
reduce-scatters them. So `gspmd` stays the default, and `chunked` is TP's method in
MXFP8.

Neither did these gain: the MLP by chunks from its gather to its reduce-scatter; each
scanned pass on two halves of its rows side by side, which hides more (20 ms exposed
with `chunked`) at 218k; XLA's windowed einsum (`compiler(preset=collective_matmul)`,
93k, [XLA's options](training.md#xlas-options-compiler)). XLA's NCCL symmetric
buffers, `compiler(options="--xla_gpu_experimental_enable_nccl_symmetric_buffers=true")`,
make each 1.5 to 2.7% faster (224k by `gspmd`; 227k by `chunked` with
`NCCL_CTA_POLICY=2`; 266k by `chunked` in MXFP8), but every such run ends its 40 steps
at a loss of 12.058 rather than 12.075: NCCL's symmetric kernels reduce otherwise.

Not yet: a collective matmul whose communication takes no SMs. TransformerEngine's
(`dense(..., collective_op_set=...)`, its userbuffers on copy engines) is on PyPI only
as source that builds against CUDA 12 (2.20.2), and would bring its own GEMMs and NCCL
communicators. jax's Pallas kernel (`pallas.ops.gpu.collective_matmul_mgpu`) sends
each tile of x on to the next device from shared memory, by remote refs, as its GEMM
multiplies it, but only on Hopper, and through NVSHMEM, which jax 0.12's XLA does not
enable: on a B300 it takes a port to tcgen05 (`blackwell_matmul_mgpu`) and a
reduce-scatter in the epilogue of the output projection.

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
| `scan`       | the default: every position as one chunk, which `chunkwise` runs by chunks of 64 positions (or cuDNN's kernels, below). With `seq` over `context`, XLA gathers the sequence onto every device first |
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

A chunk runs by `chunkwise` (`bux.architecture.recurrent.chunkwise`), the form of fla's
chunked kernels, under every method: matmuls within chunks of 64 positions, and a scan
of the state across them, by a Pallas kernel on a GPU (`affine_scan`). `stepwise`, a
scan over positions, is its reference. With one decay a head, the state passes across
the chunks by a matmul (SSD's state passing), and q and k may be shared by groups of
heads. Mamba-2's selective scan runs so, as decayed linear attention with `q = C` and
`k = B` of its groups, `v = dt x` and a log decay `dt A` a head, and `ssm.stepwise` is
its reference; Mamba's, a decay per channel and state dim, runs by that scan over
positions.

What the backward pass of `recurrence` (and so `delta_rule` and RWKV-7) and of the
selective scan reads of its chunks' float32 states and maps is `methods.states`, which
one helper applies to both (`linear_attention.recomputed`): `keep`, the default, keeps
them from the forward pass; `recompute` computes them again, as fla's kernels do, from
the inputs as they come (q, k and v in bfloat16 under `precision(compute=bfloat16)`)
and each chunk's `(I - L)⁻¹`, which it keeps. That inverse's own backward, `Tᵀ dT Tᵀ`
of `T` alone, also makes `keep` faster than before. The delta rule's forward and
backward pass at 8 × 4096 tokens of 16 heads of 128, on a B300, and what it keeps:

| delta rule | GDN | KDA |
|------------|-----|-----|
| by chunks, before | 13.1 ms, 7.24 GiB | 29.6 ms, 7.99 GiB |
| `states=keep` | 11.6 ms, 6.51 GiB | 28.9 ms, 7.26 GiB |
| `states=recompute` | 14.9 ms, 0.50 GiB | 33.6 ms, 0.75 GiB |
| by its kernel (KDA bounded at -5) | 2.5 ms | 2.9 ms |

A GLA hybrid (`"GGGA"^4`), which runs by chunks alone, trains at 84.4k tokens a second
in 83.5 GiB with `keep`, and at 77.1k in 66.8 GiB with `recompute`; that fits 32 rows
(78.8k, 192.2 GiB), where `keep` fits 16 (86.2k, 149.4).

Mamba-2 has no inverse, so `recompute` keeps only the scan's inputs, x, dt, A, B and
C: a sixteenth of what `keep` keeps. Its gradients are those of `keep`, bit for bit on
CPU. The one-GPU suite's Mamba-2 hybrid (`"MMMA"^4`), rows of 4096 on a B300, in
JAX's default share of the GPU's memory:

| rows | `states=keep` | `states=recompute` |
|------|---------------|--------------------|
| 8    | 94.8k tok/s, 122.6 GiB | 90.7k, 75.5 GiB |
| 12   | 100.8k, 172.5 GiB | 91.9k, 101.4 GiB |
| 16   | 95.8k, 190.9 GiB | 91.7k, 128.1 GiB |
| 20   | out of memory | |
| 24   | | 92.6k, 182.5 GiB |
| 32   | | 92.8k, 193.5 GiB |
| 40   | | out of memory |

So `recompute` fits twice the rows, at 4 to 9% fewer tokens a second, its losses
those of `keep` to within 10⁻³.

Under `scan`, on a B200 or B300, with q, k and v of 16 bits, the delta rule runs by
cuDNN's fused kernels instead (`fused_delta`, Frost in cudnn-frontend), which keep only
its inputs for the backward pass, and so do the chunks they fall back on. One decay a
head (GDN, DeltaNet) runs by Frost's GDN kernel, which takes each pair's decay as the
exp of a difference of summed log decays: any gate, -20 a position or a single -10⁴.
cudnn-frontend gives it no JAX call, so a `buffer_callback` runs cuDNN's graph of it on
XLA's stream, one device at a time. A decay per channel (KDA) runs by Frost's KDA
kernel where `gate_lower_bound` bounds it at -5 or above. That kernel factors each
decay through the start of a tile of 16 positions, so it loses precision where a tile
decays a channel by exp(-86), and overflows at exp(-90). Unbounded, KDA runs by
`chunkwise`: within 12 steps of bux's start, its gates are that steep in 30 to 90% of
the (row, head) pairs of most layers, and a check that fell back from the kernel to
the chunks left the GPU idle 12 ms a layer, on top of the chunks' 28.
`scripts/gpu_smoke/recurrence.py` checks the forms, and Mamba-2's recomputed, against
their scans on a GPU, whose fusions the CPU tests do not see (on a B300 one took
Mamba-2's gradients for B and C wrong, which a plain matmul in its place does not),
and the kernels against the chunks.

GDN's kernel runs on each GPU of a mesh as on one, a node or two. Where it does not
apply, as under `pass_state`, the chunks run instead. The GDN hybrid (`"DDDA"^4`) at 8
rows of 4096 a GPU, on B300s, in tokens a second and the busiest GPU's peak:

| GDN hybrid | by its kernel | by the chunks (`pass_state`) |
|------------|---------------|------------------------------|
| one GPU | 127.7k, 70.6 GiB | 95.4k, 112.7 GiB |
| a node, `fsdp=8` | 815k, 58.4 GiB | 725k, 99.6 GiB |
| two nodes, `fsdp=16` | 1.58M, 58.0 GiB | |
| two nodes, `data=2, fsdp=8` | 1.60M, 58.4 GiB | 1.42M, 99.6 GiB |
| two nodes, `data=2, fsdp=4, context=2` | | 1.03M, 124.4 GiB |

Two nodes train at 97% (FSDP over 16) and 98% (HSDP) of a node's tokens a second a
GPU. The kernel's losses are the chunks' to 0.002 on a node and 0.005 on two at the
first log, and to 0.0002 at the last of 30 steps.

Not yet: a `ppermute` chain instead of the all-gather, which holds every device's
transitions (`c` matrices `k × k` per head) on each.

## Mixture of experts

The router runs in the global view: it picks each token's top-k experts, reports the
balance loss and moves the router bias. A method of `methods.moe` then runs the picked
experts, `(moe, z, idx, weights, taps) -> y`: `z` the tokens `(b, t, d)` in the
experts' width (the latent one in a LatentMoE), `idx` and `weights` their picks
`(b, t, k)`, and `y` their weighted sum `(b, t, d)`, before `latent_output_norm`.
`taps` are the forward's, through which a method reports what it measures, as the
ragged ones report `dropped` ([below](#expert-parallelism)).

| method   | what it does                                                         |
|----------|----------------------------------------------------------------------|
| `sorted` | the default: sorts the picks by expert, runs each expert on its rows alone, and unsorts; drops no token |
| `dense`  | every expert on every token, weighted by the router: the reference   |

On a mesh, `sorted` sorts each device's own tokens, in a `shard_map` over the axes
that split rows and positions, with every expert's kernels gathered onto every device:
`expert` then acts as a batch axis, as it does under `dense`. Splitting the experts
themselves is expert parallelism, a method of its own.

Its pieces are functions of arrays alone, in `bux.architecture.experts.moe`, so that a
method can run them on one device's experts inside a `shard_map`:

- `experts(x, sizes, up, gate, down, ffn, norm)`: the expert FFN of rows sorted by
  expert, `sizes[i]` of them expert i's, by `grouped.ragged_dot`: tokamax's Mosaic
  kernel on a GPU, and its Triton one for the kernels' gradient (27 times XLA's
  `jax.lax.ragged_dot` on a B300), or XLA's. Rows past `sizes.sum()` come out 0,
  before `norm`.
- `routed(z, idx, weights, up, gate, down, ffn, norm)`: `sort_by_expert`, `experts`,
  unsort. A pick outside the experts of `up` adds nothing, so a device that holds
  experts `[i, i + n)` passes `idx - i` and its own kernels. The sort and the unsort
  are `token_rows` and `permuted`, whose gradients gather by the inverse permutation
  where XLA's would scatter-add. `token_rows` also takes the first m picks alone,
  and `token_sums`, its transpose, sums each token's k picks' rows.

```python
from bux.architecture.experts.moe import routed
from bux.parallel import methods


@methods.moe.register("mine")
def _(moe, z, idx, weights, taps=None):
    return routed(z, idx, weights, *moe.kernels(), moe.config.expert, moe.expert_norm)
```

The MoE names its parameters: the router `(embed, None)`; `up` and `gate`
`(experts, embed, mlp)` and `down` `(experts, mlp, embed)`, with `latent` for `embed`
in a LatentMoE; `to_latent` `(embed, latent)` and `from_latent` `(latent, embed)`.
`dense` constrains its hidden units to `(batch, seq, experts, mlp)`, so that under
FSDP the weights move to the tokens rather than the tokens to the weights.

### The experts in MXFP8

```
precision = precision(roles=roles(experts=mxfp8))
```

The routed experts' role, `experts`, takes `compute`, `mxfp8` or `mxfp8_2d`
(docs/training.md, "Precision"), and `nvfp4` is an error. Unlike the other roles, none
is compute, not `matmuls`. On B300s MXFP8 experts train 6 to 13% faster than
bfloat16 ones (below). In a
format, `experts` runs up and gate as one GEMM, so that x quantizes once, and each
GEMM by `grouped_blockscaled.ragged_dot`: every operand of the forward's GEMM and of
the backward's two quantized along what that GEMM contracts, as a projection's are,
and each expert's W on its own. One pass of a Triton kernel quantizes x and x^T, and
zeroes the rows past the experts', which an exchange's buffer leaves as they were.
Elsewhere XLA dequantizes and multiplies, which computes the same values, so the
tests run it on CPU. Only `passes` multiply in the format, the others by `grouped`'s
kernels, as they are.

On a Blackwell GPU tcgen05's block-scaled MMA multiplies them, in the Mosaic GPU
kernels of `grouped_mxfp8`: `rows` for x W and dy W^T, and `contract` for dW. A pair
of SMs multiplies each tile, 256 rows by 128 columns, by tcgen05's MMA of two SMs,
and each pair takes tiles one after another, in bands of 8 tiles of rows, down a
band's rows first: the tiles in flight then share few rows of one operand and columns
of the other, which stay in L2. Three roles share a tile's steps along k:

| role | does |
|------|------|
| a warp | loads each step's tiles of both operands and their scales by TMA into a ring of 7 buffers, the scales laid out as tensor memory takes them, 128 rows by 4 scales a 512-byte tile |
| a warp | copies the step's scales on into tensor memory (`tcgen05.cp`) and issues its MMAs into one of two accumulators there |
| a warpgroup | stores the other accumulator in the output's dtype, so that a tile's store overlaps the next one's MMAs |

`rows` takes the 256 rows an expert has rows in, so a tile two experts share runs
once for each, which stores only its own rows; the rows past the experts it stores as
0. `contract` takes an expert's rows in tiles of 128 along what dW contracts: the
first and last of x^T, which it may share with its neighbours, from copies with their
rows as 0, so that dy's add nothing there.

Unlike a projection's, the experts' dW quantizes the rows a device has of an expert in
blocks of 32 as they lie: a block that two experts share has one scale, and each
counts the other's rows as 0. Which rows share a block, and so dW, depend on how the
mesh splits the tokens: in float32, 3 steps on `fsdp=2, expert=4` lose within 1e-4 of
one device's (`tests/test_precision.py`), not the same.

On one B300, each GEMM of the baseline MoE's experts (64 of them, `d_model` 2048,
intermediate 704, 262144 picks), in TFLOP/s of the GEMM's own kernel by CUPTI:

| GEMM | bfloat16, tokamax | MXFP8, Triton's `tl.dot_scaled` (before) | MXFP8, `grouped_mxfp8` |
|------|-------------------|-------------------------------------------|------------------------|
| x W, up and gate | 1269 | 964 | 2405 (1.9x) |
| h W, down | 1069 | 680 | 1596 (1.5x) |
| dy W^T, down | 1213 | 908 | 2230 (1.8x) |
| dh W^T, up and gate | 1266 | 901 | 2244 (1.8x) |
| dW, up and gate | 1093 | 644 | 2573 (2.4x) |
| dW, down | 1034 | 616 | 2367 (2.3x) |

and with experts of 2048, 1.7 to 2.3 times bfloat16's: 2423 to 2616 TFLOP/s against
1139 to 1414. At `all_to_all`'s GEMMs of a peer's buffer under `expert=8`, 40960 rows
of which 32768 its 8 experts', 1.3 to 2.7 times: 1271 (h W) to 2490 (dW, up and gate)
against 904 to 1213. h W contracts 704, which TMA takes in tiles of 128, so `rows` pads it
to 768 and does 9% more. Taking the tiles a row of them at a time, all its columns
before the next row, was 1.7x slower (1623 against 2725 TFLOP/s for dh W^T at k
4096): the rows and columns in flight then fell out of L2. Triton's kernels took the
scales, rows of 4 bytes too short for TMA, through registers into tensor memory at
every step of k.

Training on B300s, side by side, GEMM autotuning off and deterministic ops, the
experts in bfloat16, in MXFP8 by Triton's kernels as before, and by `grouped_mxfp8`
(tokens a second, peak memory):

| run | bfloat16 | MXFP8, Triton | MXFP8, `grouped_mxfp8` |
|-----|----------|---------------|------------------------|
| the baseline MoE, one GPU, 8 × 4096 tokens | 86.2k, 133.9 GiB | 74.2k | 91.5k (1.06x), 119.5 GiB |
| experts of 2048, 8 layers, one GPU | 90.4k, 143.2 GiB | 75.6k | 101.7k (1.12x), 129.6 GiB |
| `ragged_all_to_all`, `capacity=1.25`, `expert=8`, 32k tokens a GPU | 468k, 88.5 GiB | 417k | 517k (1.10x), 72.3 GiB |
| | | | `dispatch=mxfp8`: 515k |
| `all_to_all`, `capacity=1.25`, `expert=8`, 32k tokens a GPU | 492k, 91.4 GiB | 449k | 539k (1.10x), 78.4 GiB |
| `sorted`, `fsdp=8`, 32k tokens a GPU | 671k, 83.9 GiB | 580k | 712k (1.06x), 69.4 GiB |

Both MXFP8 kernels compute the same values: the runs' losses agree to the last digit
logged, and with bfloat16's to the rounding between them. The experts' GEMMs take
half of bfloat16's time, 3.0 ms a layer against 5.8, and the quantizer, which reads
each operand once and writes it twice, most of what they save: an MoE layer's experts
forward and backward take 6.1 ms against 7.3 in bfloat16, and 13.8 against 17.9 with
experts of 2048 (`scripts/gpu_smoke/grouped_mxfp8.py`). It keeps x^T in MXFP8 for dW
rather than x in bfloat16, so it takes 14 GiB less. It trains at bfloat16's loss: 98M
tokens of FineWeb-Edu end 0.13% and 0.05% below it, with FP8 dispatch
(docs/training.md, "Precision").

## Expert parallelism

Four methods of `methods.moe` split the experts over the axes `layout.experts` gives
them, `expert` by default: each device holds its own experts' kernels, and runs them
on the picks of every device of its group.

```
mesh    = mesh(fsdp=2, expert=4)
methods = methods(moe=ragged_all_to_all, capacity=1.25)
```

| method              | what it does                                                  |
|---------------------|---------------------------------------------------------------|
| `all_gather`        | each device gathers its group's tokens, runs its own experts on them, and a `psum_scatter` brings each token its sum back |
| `all_to_all`        | each pick goes to its expert's device and back by `jax.lax.all_to_all`, in a buffer of one size for each peer, sorted by expert: a device runs its experts on each peer's buffer as it arrives, a grouped GEMM a peer |
| `ragged_all_to_all` | each pick goes to its expert's device and back by `jax.lax.ragged_all_to_all`, a slice for each expert of each peer: a device's rows arrive sorted by its experts, which run on them as they are. XLA:CPU lacks it |
| `ragged_tokens`     | the same, but each token goes once to each device that holds any of its experts, with its picks there, and comes back as their weighted sum: a row per device, not per pick |

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

### A capacity

Dropless, a device of `all_gather` or of the ragged methods takes what every peer
could send it: `expert` times the picks it has, and its experts run on all those
rows. `methods.capacity` bounds them at that many times its own picks (its tokens
times `top_k`), about what it takes on average, and drops what does not fit: its last
experts' picks first, and of an expert's, the last peers'. A dropped pick adds nothing
to its token. A training step reports the share of the picks dropped as `dropped`,
summed over the MoE layers, as `balance_loss` is. A capacity that drops anything
changes what a run computes, unlike the rest of `methods`, though it stays out of the
run's id. `ragged_tokens` takes as many tokens as picks, or as many as could come, a
token from each device. `all_to_all` bounds each of a device's buffers, one a peer,
at `capacity` times a peer's even share of its picks (its picks over `expert`), and
drops what does not fit there, that peer's last experts' picks first; dropless, a
buffer holds every pick, so a device sends `expert` times the rows it has.

The buffers that receive are `jax.lax.empty`: the experts make 0 of the rows past
theirs, and the rows that do not come back are masked. XLA's one-shot ragged
all-to-all, on one node, copies each slice with one block of 128 threads, so each
exchange goes in 2048 pieces at least (`experts.PIECES`): on 8 B300s a slice for each
expert of each peer moved 90 GB/s, and the pieces 600 GB/s.

A method is a body that runs on one device's tokens, named as the method is in
`bux.parallel.experts`, and `experts.expert_parallel` makes it one of `methods.moe`:

```python
from bux.parallel import experts, methods


def mine(z, idx, weights, kernels, *, axes, ffn, norm):
    # This device's tokens and picks, and its experts' kernels: experts
    # [i * n, (i + 1) * n) for n = len(kernels[0]), i = jax.lax.axis_index(axes).
    # Return the tokens' sums, and the share of the group's picks it dropped.
    ...


methods.moe.register("mine")(experts.expert_parallel(mine))
```

`experts.ragged_all_to_all` and `experts.ragged_tokens` take the ragged exchange as
`exchange=`, so another collective is a function: the tests run them on CPU through
one built from `all_to_all`, and `scripts/gpu_smoke/experts.py` runs the real one, and
every other method, on GPUs.

Collectives in the compiled train step of the two-MoE model of
`tests/test_experts.py` on 8 CPU devices, their number and the KiB their results take
on one device:

| mesh                       | method       | all-gather | all-to-all | reduce-scatter | all-reduce |
|----------------------------|--------------|------------|------------|----------------|------------|
| `expert=8`                 | `sorted`     | 48, 688    |            |                | 3, 678     |
|                            | `all_gather` | 50, 372    |            | 6, 8           | 3, 294     |
|                            | `all_to_all` | 42, 304    | 10, 256    |                | 3, 294     |
| `data=2, fsdp=2, expert=2` | `sorted`     | 54, 880    |            |                | 7, 736     |
|                            | `all_gather` | 56, 513    |            | 6, 8           | 8, 544     |
|                            | `all_to_all` | 48, 496    | 10, 64     |                | 8, 544     |
| `fsdp=2, expert=4`         | `sorted`     | 54, 784    |            |                | 5, 678     |
|                            | `all_gather` | 56, 434    |            | 6, 8           | 6, 390     |
|                            | `all_to_all` | 48, 400    | 10, 128    |                | 6, 390     |

`sorted` gathers every expert's kernels and all-reduces their gradients; the others
move tokens instead. Dropless with buffers of a fixed size, `all_to_all` moves
`top_k` times the rows `all_gather` does, as every peer's buffer holds every local
pick; under a capacity it moves `capacity` times the picks, where
`ragged_all_to_all` moves the picks alone. No method rematerializes anything on these
meshes, nor on `fsdp=2, expert=2, tensor=2`.

On 8 B300s every method computes what one GPU does (`scripts/gpu_smoke/experts.py`,
in float32: 7e-7 apart at most). For the baseline's MoE (`d_model` 2048, 64 experts,
`top_k` 8, 16 layers, `bfloat16`) on 32k tokens a GPU, tokens a second and peak memory:

| method                           | mesh       | remat `none`        | remat `dots`     |
|----------------------------------|------------|---------------------|------------------|
| `sorted`                         | `fsdp=8`   | 816k, 84 GiB        | 737k, 52 GiB     |
| `all_gather`, `capacity=1.25`    | `expert=8` | 612k, 89 GiB        | 515k, 65 GiB     |
| `ragged_all_to_all`, `capacity=1.25` | `expert=8` | 550k, 88 GiB    | 515k, 78 GiB     |
| `ragged_tokens`, `capacity=1.25` | `expert=8` | 547k, 90 GiB        | 479k, 74 GiB     |
| `all_to_all`, `capacity=1.25`    | `expert=8` | 559k, 89 GiB        |                  |
| the same, `moe_chunks=2`         | `expert=8` | 574k, 89 GiB        |                  |

Before the by-expert slices, the capacity and the pieces, `all_to_all` and
`ragged_all_to_all` ran out of memory at 16k tokens a GPU, and under `remat=full` at
32k they made 72k and 25k tokens a second (now `ragged_all_to_all` dropless: 263k).
On one node at this shape, then, `sorted` is still the fastest: its FSDP gathers 0.5
GB of experts a layer, where the exchanges move 1 GB of picks each way
([below](#in-chunks-methodsmoe_chunks) says what bounds them). Under `dots`
the exchanges ran again in the backward, since their results are no products; named
for it ([below](#fp8-dispatch-precisiondispatch)), `ragged_all_to_all` trains at 515k
tokens a second in 77.5 GiB, from 452k in 72.6. XLA's ragged all-to-all leaves an allocation inside the memory pool after the first step,
so that a step whose temporaries take over half of it fits only once:
`XLA_PYTHON_CLIENT_MEM_FRACTION=.95` gives the pool more room. `ragged_tokens` moves
about 30% fewer rows when the routing is random, and half as few or fewer where a
group limit keeps each token on half the devices (`num_groups` the size of `expert`,
`top_k_groups` half of it), but it sends the picks and their weights besides, and
here gains nothing. At the start of a run on random tokens, a capacity of 1.25 drops
4-10% of the picks a layer, and 1.5 at 16k tokens 3-4%. Neither
`--xla_gpu_experimental_ragged_all_to_all_use_device_kernel` nor
`--xla_gpu_ragged_all_to_all_mode=peer` changes the speed; `=symmetric` fails in NCCL
("Window address must be suitably aligned"). Nor does XLA's grouped matmul catch up
with tokamax's: `jax.lax.ragged_dot` trains `sorted` 20 times slower, and with
`--xla_gpu_experimental_use_ragged_dot_fusion` (cuDNN) it fails to compile.

Keep `expert` inside a node for now: across hosts, XLA splits a ragged all-to-all in
an intra-host one and an all-gather of the tokens across hosts.

On jax 0.12 NCCL's ragged all-to-all hangs unless XLA_FLAGS has
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, which
`bux train` and `bux launch` add to it, and so does `bux.runtime.configure()` in a
script that calls it before it touches a device.

### In chunks: `methods.moe_chunks`

```
methods = methods(moe=all_to_all, capacity=1.25, moe_chunks=2)
```

An EP layer waits for four exchanges: the picks there and back, and the backward's
two. `methods.moe_chunks` cuts each device's tokens into that many slices, which a
method runs one after another, each on its own (`experts.in_chunks`), so that the
compiler can send one slice's picks while another slice's experts run. Where it does
not divide a device's tokens, it takes as many slices as do. A slice's experts, and
each peer's under `all_to_all`, add their part of the kernels' gradients, which add up
in float32, as one call's would.

Only `all_to_all` gains from it, on 8 B300s with jax 0.12:

- NCCL's all-to-all, which `all_to_all` sends by, runs on its own stream on a few of
  the SMs: beside an independent grouped GEMM the two take a fifth longer than the
  all-to-all alone.
- XLA's ragged all-to-all overlaps nothing. Its one-shot kernel copies in 2048 blocks
  of threads, which take every SM, so beside a GEMM the two take as long as one after
  the other, and in chunks a step is slower. Nor does XLA schedule for it: its latency
  hiding scheduler, which moves collectives beside compute, runs on Hopper and
  Blackwell only where its cost model knows every collective of a step
  (`SolLatencyEstimator::IsSupportedForModule`), and ragged-all-to-all is none of
  them. `compiler(preset=lhs)` runs it anyway, and at 32k tokens a GPU the step then
  runs out of memory, in the hole the ragged exchange leaves in the pool.
- A collective-permute a peer (in turn i each device sends the i-th after it its
  buffer and runs its experts on the one it gets) hides half its time, but seven of
  them take twice as long as one all-to-all, by NCCL or by copy engines
  (`--xla_gpu_use_memcpy_local_p2p`): 394k tokens a second.

For the baseline's MoE at 32k tokens a GPU and a capacity of 1.25, profiled, tokens a
second and ms of a step of a GPU: all its collectives, those with no compute beside
them (the log's `exposed_comm_ms`), and the experts' grouped GEMMs:

| method                                      | tokens a second | collectives | exposed | experts' GEMMs |
|---------------------------------------------|-----------------|-------------|---------|----------------|
| `sorted`, `fsdp=8`                          | 830k            | 66          | 7       | 110            |
| `ragged_all_to_all`                         | 546k            | 167         | 152     | 116            |
| `all_to_all`                                | 559k            | 165         | 143     | 117            |
| `all_to_all`, `moe_chunks=2`                | 574k            | 208         | 90      | 153            |
| `all_to_all`, 2 microbatches, unrolled      | 562k            | 219         | 98      | 155            |

The table's runs come from several nodes. Side by side on one node
(`scripts/benchmark.py`, the one-node suite's `moe` and `moe_ep`, the latter as
`methods` says), `sorted` trains at 835k tokens a second, `ragged_all_to_all` at 539k
and `all_to_all` in 2 chunks at 586k, 0.70 of `sorted`.

What bounds EP here is its exchange: four of 1.1 to 1.3 GB a layer a GPU, 150 ms of a
step at NCCL's 500 GB/s, where `sorted`'s FSDP gathers 0.5 GB of experts a layer and
reduces their gradients, 66 ms that hide behind compute. Chunks hide a third of the
exchange, but a GEMM beside NCCL's kernels runs a third slower and gives most of it
back: the experts' GEMMs take 153 ms rather than 117, and the step gains 3%. DeepSeek's
two microbatches, each one's exchanges beside the other's compute (the trainer's
microbatches unrolled rather than scanned, by hand), do no better. EP would beat
`sorted` on one node at this shape with an exchange near NVLink's 900 GB/s, on a few
SMs and without the capacity's padding, as DeepEP's and TransformerEngine's kernels
send it, hidden behind the attention as well as the experts.

Fewer NCCL channels, to leave the GEMMs more SMs, slow the exchange more
(`NCCL_MAX_P2P_NCHANNELS=8`: 349k), and a high-priority stream for collectives changes
nothing. Each chunk is a grouped GEMM a peer a layer more to compile: `moe_chunks=2`
compiles in 5 minutes here, and 4 in more than 8.

### FP8 dispatch: `precision.dispatch`

```
precision = precision(dispatch=mxfp8)
```

The exchanges send the tokens to their experts in MXFP8, as DeepSeek-V3 does: E4M3
values and a power-of-two scale per 32, d + d/32 bytes a row, 52% of bfloat16's at
d = 2048, dequantized to the compute dtype where they arrive. Each token quantizes
once, before the rows of its k picks are picked, so the quantizer reads a k-th of the
rows and the gather writes half the bytes. What comes back, and the backward's two
exchanges, go in compute; the gradient is the exchange's, as if the tokens went
unquantized (`experts._picked_in_mxfp8` for the ragged methods,
`experts._swapped_in_mxfp8` for `all_to_all`). Rows that are not whole blocks go as
they are. It changes what a run computes, so it is in the run's id.

On 8 B300s, for the baseline's MoE at 32k tokens a GPU and a capacity of 1.25 (the
one-node suite's `moe_ep`, side by side on one node), `ragged_all_to_all` trains at
556k tokens a second against 539k, and `all_to_all` in 2 chunks at 589k against 586k.
A ragged dispatch takes 17 ms a step less, and quantizing a token rather than each of
its picks costs little; `all_to_all`'s dispatch hides behind the experts already.

The exchanges name what they bring `experts.exchanged`, under `experts`, so that
`remat=dots`, which keeps `experts`, does not exchange them again in the backward.

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

`remat` rematerializes in the backward pass what the forward does not keep. A policy
keeps what layers name (`jax.ad_checkpoint.checkpoint_name`), as MaxText's and
Megatron's selective recomputation do, and the backward computes the rest again:

```
methods = methods(remat=remat(save=[attention, mlp], first=8))
```

| setting   | what it is                                                                 |
|-----------|----------------------------------------------------------------------------|
| `save`    | the names it keeps on the device                                           |
| `offload` | the names it moves to the host's memory until the backward reads them     |
| `first`   | how many passes of each repeat it rematerializes, from the first (of each stage's share, pipelined); the rest keep everything. None: every pass |

| name              | what it is                                                          |
|-------------------|---------------------------------------------------------------------|
| `attention.in`    | softmax attention's q, k and v, as its kernel takes them (after RoPE) |
| `attention.out`   | its output, and what else its backward takes: the log-sum-exp       |
| `attention.index` | the keys sparse attention's indexer keeps for each query            |
| `recurrence.in`   | a recurrent mixer's inputs, as its kernel takes them: linear attention, the SSMs, RWKV-7 |
| `recurrence.out`  | its output, and what else its backward takes: the chunks' states, their inverses with `methods(states=recompute)`, nothing more by cuDNN's kernels |
| `qkv`             | attention's q, k and v projections; MLA's latents                   |
| `mlp`             | an MLP's up and gate projections                                    |
| `experts`         | the MoE's grouped products                                          |
| `dots`            | every product of matrices, MXFP8's and the experts' too             |
| `dots_no_batch`   | those without batch dimensions                                      |

A name covers those under it, so `attention` is all three of attention's. They are
names of what layers make, not definitions: `save=[mlp]` is no layer type, and
`save=[attention]` a list of one. A preset is a policy by name:

| method               | what the forward keeps                           | as data                      |
|----------------------|--------------------------------------------------|------------------------------|
| `none`               | everything, but in a scanned pass (below); the default |                        |
| `full`               | nothing but the input of each branch or pass     | `remat()`                    |
| `dots`               | the products of matrices                         | `remat(save=dots)`           |
| `dots_no_batch`      | the products of matrices without batch dimensions | `remat(save=dots_no_batch)` |
| `save_attention`     | the mixers' kernels' inputs and outputs          | `remat(save=[attention, recurrence])` |
| `save_qkv_attention` | those, and attention's projections               | `remat(save=[attention, recurrence, qkv])` |
| `dots_attention`     | `dots_no_batch`'s, the mixers' and the experts'  | `remat(save=[dots_no_batch, attention, recurrence, experts])` |
| `offload_attention`  | the mixers', in the host's memory                | `remat(offload=[attention, recurrence])` |

Each residual branch outside any other is rematerialized as a whole, with the read
and write of the residual algebra; a scanned pass is, as one; and nothing inside
either is again. With `first=n`, the first n passes of each repeat are; unrolled, the
rest keep everything, and scanned, they run in a scan of their own, as `none`'s. In a
`.bux` file `none` is the null value, so a sweep writes the method as a string:
`methods.remat = "none" | full`.

What a kernel's backward takes is no product of matrices: flash attention's takes its
output and log-sum-exp, so under `dots` the backward runs the kernel's forward again.
`remat.named(name, f, *args)` names f's inputs `name.in` and its output and the rest
of what its backward takes `name.out`, so that a policy that keeps the name runs none
of f twice; attention's methods, sparse attention and the recurrent mixers run
through it. Sparse attention's indexer names the keys it keeps, so that its backward
neither picks them again nor runs the flash forward again. MLA names its latents, not
the heads' projections up from them: as in DeepSeek's and Megatron's
(`mla_up_proj`), a policy that keeps `qkv` projects them up again. `dots` keeps
MXFP8's GEMMs inside a `shard_map` too, where jax's own policy missed them: in a
pipeline of 2 B300s (4 microbatches of the dense 1.3B), at the same speed as when it
ran them again, 155.7k tokens a second, in 53.3 GiB rather than 42.7; `full` trains
that pipeline at 160k. Another policy is a function, which a name of `methods.remat`
gives:

```python
import jax

from bux.architecture.repeats import remat
from bux.parallel import methods

methods.remat.register("save_mlp")(remat.keeping(save=("attention", "mlp")))
methods.remat.register("offload_dots")(
    jax.checkpoint_policies.offload_dot_with_no_batch_dims("device", "pinned_host")
)
```

jax's own `save_only_these_names` matches names whole, so it takes `attention.out`,
not `attention`.

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
compiles 2.3 times as fast; with `compiler(preset=maxtext)` the dense scan trains at
719k, 15% over unrolled ([XLA's options](training.md#xlas-options-compiler)).

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
2048 (vocabulary 128256), a Gated DeltaNet hybrid (`"DDDA"^4`) and a 64-expert MoE
(`E^16`, top 8), in `bfloat16` on 8 rows of 4096 tokens, the loss by chunks, tokens a
second and the peak memory:

| remat                                        | dense        | GDN          | MoE          |
|----------------------------------------------|--------------|--------------|--------------|
| `none`                                       | 139k, 51.1 GiB | 89.1k, 120.9 GiB | 101k, 134.0 GiB |
| `dots_attention`                             | 137k, 41.6   | 76.9k, 106.9 | 97.2k, 108.4 |
| `remat(save=[attention, mlp])`               | 135k, 41.6   |              |              |
| `remat(save=[attention, recurrence.in, mlp])` |             | 74.9k, 41.0  |              |
| `remat(save=[attention, experts])`           |              |              | 95.9k, 108.3 |
| `dots`                                       | 128k, 39.5   | 74.0k, 75.7  | 93.7k, 106.4 |
| `save_attention`                             | 126k, 30.6   | 73.4k, 89.9  | 87.7k, 81.7  |
| `remat(save=[attention, recurrence.in])`     |              | 71.9k, 31.4  |              |
| `full`                                       | 120k, 24.5   | 72.7k, 30.3  | 84.2k, 75.8  |
| `remat(first=8)`, `first=2` for GDN: half the passes | 127k, 37.8 | 79.3k, 73.6 | 90.1k, 104.6 |
| `remat(save=[attention], first=8)`           | 133k, 40.9   |              |              |
| `offload_attention`                          | 82.1k, 39.5  |              |              |

`none` is the fastest, and the default; a policy buys rows. `dots` runs flash
attention's forward twice, and `save_attention` the matmuls of every layer;
`dots_attention` runs neither, 2% slower than `none` in 19% less memory on the dense
model, as does `remat(save=[attention, mlp])`. A GDN layer's recurrence by chunks
keeps their float32 states (`recurrence.out`), most of what `save_attention` keeps
there: keeping its inputs alone (`recurrence.in`), whose backward runs the chunked
forward again, is faster than `save_attention` in less than half its memory. (Since
measured, GDN runs by cuDNN's kernel, which keeps only its inputs: [Context
parallelism for recurrent mixers](#context-parallelism-for-recurrent-mixers-methodsrecurrence).)
In a Mamba-2 hybrid (`"MMMA"^4`) too, `recurrence.in` in place of `recurrence` costs 2-3% for 59-68%
less memory (`save_attention` 78.5k in 99.6 GiB, against 76.4k in 31.7); in a KDA
hybrid by cuDNN's kernel, which costs more to run again, 7-11%. `first` gives the
points between: half the passes in full is as fast as `dots` in a little less memory,
and slower than the policy of names it would match: those keep a layer's costly parts,
where `first` computes the whole layer again. With more rows a GPU:

| remat                                        | dense                | GDN                 | MoE                |
|----------------------------------------------|----------------------|---------------------|--------------------|
| `none`                                       | 56 rows: 146k, 191.2 GiB | 12 rows: 91.0k, 166.2 GiB | 10 rows: 105k, 149.8 GiB |
| `dots_attention`                             | 48: 147k, 151.6; 64: 145k, 196.7 |         | 12: 102k, 127.5    |
| `remat(save=[attention, recurrence.in, mlp])` |                     | 32: 79.7k, 110.3    |                    |
| `save_attention`                             | 96: 138k, 161.7      |                     | 16: 93.8k, 95.8    |
| `full`                                       | 128: 126k, 113.2     | 32: 76.5k, 67.3     | 16: 89.6k, 83.8    |

Offloading saves nothing unrolled, and costs a third of the speed: XLA schedules
each layer's copy to the host inside the backward, just before the copy back, so the
device keeps the value from the forward on as if saved, and the round trip adds a
buffer (the compiled gradient's temp memory is 17.7 GiB against `save_attention`'s
13.4, and 1.0 GiB goes to the host); the latency-hiding scheduler compiles the same
program. Scanned, it works as MaxText's does: the scan stacks each pass's values into
buffers in the host's memory, 6.0 GiB of them, and the device peaks at 22.3 GiB
against `save_attention`'s 30.1, at 84k tokens a second against 121k (90k with
`--xla_gpu_enable_pipelined_host_offloading=true`). `full`, unrolled, peaks at 24.5
GiB at 120k: on a B300, computing the values again is cheaper than the copies.

`Scanned.layers` holds the stacked passes, and `stacked.run(graphdef, state, times, x,
taps)` runs any stack of them, such as a pipeline stage's share. A scan traces one
pass for all it runs, so a repeat first cuts its passes into runs that trace alike
(`tracing.runs`): where precision's first and last branches start or stop
(`matmuls.cuts`), where `remat(first=n)` stops (`remat.cuts`), and in a pipeline's
stage where some stage skips them. Each run traces inside `tracing.passes(run,
each)`, which tells remat the pass (`tracing.current_pass`) and precision the branch
(`tracing.current_branch`); another reason for passes to trace apart is another cut.

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
| the head                                        | its matmul's accumulation, tied or not, and its log Z and loss; by chunks, the default `methods.loss`, its logits |

Softmax attention takes its softmax in float32, under every method of
`methods.attention`. Every other matmul runs in `compute`, or in the format of
`matmuls` (below), and the stream stays in `compute` from the embedding to the heads,
which `tests/test_precision.py` checks for every family.

Gradients add up over microbatches in float32. Over the rows' axes they are reduced in
`precision.reduce`: `compute`, the default, or `float32`, as Megatron's are. In
`compute` they are reduced as the backward makes them, bfloat16 for a weight's (but the
head kernel's by chunks, the default `methods.loss`, which adds up over the chunks in
float32); where a step adds its gradients up unreduced ([below](#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
each device's sum is rounded to it before the one reduction. In `float32` that
reduction sums float32, twice the bytes, once a step; on B300s, through the NICs, a
step of the dense 1.3B trains 5% slower in it. `float32` changes what a run
computes, so it is in the run's id, and a run that leaves `compute` keeps the id it
had. It needs the step to add its gradients up unreduced, also at one microbatch: XLA
reduces those of the backward in the dtype of the matmul that makes them, as it does
for a run whose layers learn from the whole batch, such as an MoE's, and for a
pipeline's embedding and head, which run outside it: there `float32` is an error. A
pipeline that reduces once sums its stages' gradients in float32 either way, as their
parameters are whole in float32.

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

`matmuls` names a format of the registry in `bux.architecture.precision.matmuls`. A
block-scaled format quantizes the operands of all three GEMMs of y = x W, the forward's
and the backward's two, each along what its GEMM contracts, in blocks with a scale each
(`bux.architecture.precision.blockscaled`):

| format    | values  | a scale per                                   | and                                       |
|-----------|---------|-----------------------------------------------|-------------------------------------------|
| `compute` | `compute` | none                                        | the default                               |
| `mxfp8`   | E4M3    | 32, a power of two (E8M0)                     | dy in E4M3 too, as Transformer Engine's MXFP8 recipe |
| `mxfp8_2d` | E4M3   | 32 of x and dy; W's per 32 x 32 tile, quantized by XLA | so that y and dx see the same W, as TorchTitan's |
| `nvfp4`   | E2M1    | 16 (E4M3), and the tensor (float32); W's per 16 x 16 tile, so that y and dx see the same W | dy rounded stochastically, and x and dy transformed for dW by a random Hadamard transform of 16, as NVIDIA's NVFP4 pretraining recipe |

On a Blackwell GPU cuDNN's block-scaled GEMM multiplies them (`__cudnn$blockScaledDot`
in the HLO), and Pallas kernels quantize each of x, W and dy and its transpose: for
`mxfp8` in one pass, for `nvfp4` in two, the first for the largest magnitudes that
set its global scales (`bux.architecture.precision.nvfp4`). Elsewhere XLA dequantizes
and multiplies, which computes the same values, so the tests run the formats on CPU. The
forward keeps what the backward multiplies, x^T and W in the format, rather than x and
W.

Every projection made by `sharding.projection` or `sharding.linear` multiplies by
`matmuls.dot_general`, which runs the format in force for its role and the residual
branch it is in (docs/training.md, "Precision"): attention's q, k, v, o and gate;
the MLPs', the shared experts' and LatentMoE's; MLA's, CSA's, the indexer's; the
linear attentions', RWKV-7's and Mamba's; Engram's, nextn's and `linear`'s. The routed
experts multiply in `compute` unless their role says `mxfp8` or `mxfp8_2d`, which a
grouped GEMM of their own takes ([The experts in MXFP8](#the-experts-in-mxfp8)).
These stay in `compute`: the embedding and the head, softmax attention's own matmuls and its
relative bias, the router and the shared experts' gate, hyper-connections, norms,
low-rank gates and decays, recurrent states, and `methods(moe=dense)`'s einsums. A
projection whose tokens, inputs or outputs are not whole blocks multiplies in `compute`. On a mesh, one that
does not split into whole blocks on every device (a multiple of the block times the
devices) stays whole on each: so the blocks are the same on any mesh, and so is what a
run computes, which `tests/test_precision.py` checks on `fsdp=4, tensor=2` and in a
pipeline. NVFP4's stochastic rounding draws its noise from a hash of each group of 4
elements' index and the batch's tokens, 8 bits an element, the same on any mesh too:
NVFP4's kernels each take the place of their device's part in the whole.

Under FSDP each device quantizes its part of W, so the all-gathers move float8 values
and their scales, half of bfloat16's bytes. A projection multiplies its tokens
flattened, rows by positions, and a split along positions is not one along the
flattened tokens: a `ring` run on `context=2` trained 13% slower in `mxfp8` than in
bfloat16 on B300s.

A recipe is a function, and the trainer enters `matmuls.use` as it traces its steps:

```python
from bux.architecture.precision import matmuls


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

`parallel.microbatches` is there for pipeline parallelism to split its batch the same
way.

### Gathered once, reduced once: `methods.gather`, `methods.accumulate`

```
methods = methods(microbatches=8)                                  # reduced once across hosts
methods = methods(microbatches=8, accumulate=unreduced)            # reduced once
methods = methods(microbatches=8, accumulate=reduced)              # reduced each microbatch
methods = methods(microbatches=8, gather=layer, accumulate=reduced)  # ZeRO-3
```

Every axis is Auto, and XLA does not move a collective out of a loop. Left to it, a
step of n microbatches under FSDP gathers each layer's parameters as it runs, n times
forward and, in a scan of layers, n times back, and reduce-scatters the gradients n
times: what ZeRO-3 does, and `methods(gather=layer, accumulate=reduced)` still does.
`bux.parallel.gather` does better, in two parts:

- **Gathered once** (`gather=step`, the default). Before the microbatches, each
  device casts its shards of the parameters to `compute` and gathers them
  (`gather.whole`: split by their names alone, whole over FSDP's axes), and every
  microbatch uses that copy. A pipeline's step does the same before its ticks, in
  float32, as its ticks add up the gradients in the dtype of what they read.
- **Reduced once** (`accumulate=unreduced`). The gradients add up unreduced, each
  device's own, and reduce once at the end. JAX types an array that each device holds a
  part of the sum of, `P(unreduced={...})`, only on Explicit or Manual axes, so the step
  runs in a `shard_map` manual over the axes that split the rows (`gather.rows`,
  `data` and `fsdp` by default): each device gathers its shards by `all_gather`, sums
  the gradients of its rows' microbatches into a float32 copy of every parameter, and a
  `psum_scatter` keeps its shards of their sum (`gather.scatter`), in
  `precision.reduce`. Each device folds its index into the RNG keys, so that dropout
  draws masks for its own rows. The other axes stay Auto inside, and their reductions
  (over `tensor`, `context`) happen each microbatch.

Inside that `shard_map` a layer sees its device's rows alone, so a layer that learns
from the whole batch would learn from a part (`trainer.rows_independent`): the MoE's
router, its bias and its balance loss over the batch. `accumulate=auto`, the default,
adds up unreduced where the step has microbatches or a pipeline, the devices along the
rows' axes span more than one process (`gather.rows_span_processes`: their
reductions go between hosts), its rows are independent and the parameters are
gathered once; else each microbatch's gradient reduces as the backward makes it,
onto its shards, which over NVLink is faster (below). `accumulate=unreduced` where it
would not compute the same is an error that says why.

Unreduced, a step of 8 microbatches then moves what a step of 1 does. In the compiled
step of `fsdp=8`, each collective counted as many times as the loops around it run
(`tests/test_gather.py`), for the dense 1.3B of `docs/training.md` on B300s, 64 rows
of 4096. On one node `accumulate=auto` reduces each microbatch (that row's loss by
chunks):

| methods                                        | all-gathers     | reduce-scatters |
|------------------------------------------------|-----------------|-----------------|
| `microbatches=1`                               | 66, 2.68 GiB    | 9, 2.68 GiB     |
| `microbatches=8, gather=layer, accumulate=reduced` | 528, 21.5 GiB | 72, 21.5 GiB  |
| `microbatches=8`, on one node                  | 66, 2.68 GiB    | 72, 25.9 GiB    |
| `microbatches=8, accumulate=unreduced`         | 66, 2.68 GiB    | 9, 2.68 GiB     |
| `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 1048, 34.1 GiB | 152, 21.5 GiB |
| `repeat=scan, microbatches=8, accumulate=unreduced` | 10, 2.68 GiB | 7, 2.68 GiB     |

Tokens a second and the peak memory of a GPU, on one node of 8 B300s, the same run,
20 steps, the loss by logits (by chunks, the default, `microbatches=1` takes 1122k in
38.9 GiB):

| mesh, methods                                      | tok/s | peak     |
|----------------------------------------------------|-------|----------|
| `fsdp=8`, `microbatches=1`                         | 1147k | 58.5 GiB |
| `fsdp=8`, `microbatches=8, gather=layer, accumulate=reduced` | 817k | 11.9 GiB |
| `fsdp=8`, `microbatches=8` (reduced each microbatch) | 925k | 11.6 GiB |
| `fsdp=8`, `microbatches=8, accumulate=unreduced`   | 907k  | 15.7 GiB |
| `fsdp=8`, `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 719k | 10.2 GiB |
| `fsdp=8`, `repeat=scan, microbatches=8, accumulate=unreduced` | 803k | 15.7 GiB |
| `fsdp=2, tensor=4`, `layout(stream=tensor)`, `microbatches=8, gather=layer, accumulate=reduced` | 647k | 11.9 GiB |
| `fsdp=2, tensor=4`, `layout(stream=tensor)`, `microbatches=8, accumulate=unreduced` | 640k | 12.9 GiB |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 660k | 88.3 GiB |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8` (reduced each tick) | 640k | 97.8 GiB |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, accumulate=unreduced` | 644k | 97.8 GiB |

Over NVLink, gathering once is the gain: XLA overlaps a reduce-scatter a microbatch
with the backward, and adding up unreduced reads and writes a float32 copy of every
gradient a microbatch, as each device's own. Between nodes the collectives are the
cost. With NCCL kept off NVLink and shared memory (`NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1`, and XLA's one-shot all-reduce off), every one goes through the
NICs, as between nodes:

| mesh, methods                                                      | tok/s |
|--------------------------------------------------------------------|-------|
| `fsdp=8`, `microbatches=1`                                         | 1030k |
| `fsdp=8`, `microbatches=1`, `precision(reduce=float32)`            | 976k  |
| `fsdp=8`, `microbatches=8, gather=layer, accumulate=reduced`       | 358k  |
| `fsdp=8`, `microbatches=8, accumulate=reduced`                     | 500k  |
| `fsdp=8`, `microbatches=8, accumulate=unreduced`                   | 738k  |
| `fsdp=8`, `microbatches=8`, `precision(reduce=float32)`            | 699k  |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, gather=layer, accumulate=reduced` | 464k |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, accumulate=reduced` | 521k |
| `stage=2, fsdp=4`, `repeat=scan, microbatches=8, accumulate=unreduced` | 566k |

So `gather=step` is the default, and `accumulate=auto` adds up unreduced only where the
rows' axes span processes, as between nodes. With deterministic GEMMs the three dense runs
of 8 microbatches on one node end their 10 steps at 11.86792, 11.86793 (`reduced`) and
11.86804 (`unreduced`).

The cost is memory: the gathered copy of the parameters in `compute`, and, unreduced,
a float32 sum of every parameter's gradient on each device, held through the step.
`layout(zero=1)` keeps the parameters whole between steps too; with `gather=step` it
saves no collective, and holds float32 copies where the step holds `compute`'s.

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
methods = methods(repeat=scan, microbatches=8, pipeline="zbh1")              # zero bubble
methods = methods(repeat=scan, microbatches=8, pipeline="1f1b", stages=[5, 4, 4, 3])
```

With `mesh.stage` above 1, the longest scanned repeat outside any other (the first of
several as long) is pipelined. Its stacked layers split over `stage` by a dimension
named `stages` of the layout, one share of consecutive passes per stage. The rest of the
model, the embedding and the head too, does not split over `stage`: under `gpipe` and
`circular` every stage computes it, and under `1f1b` and `zbh1` the first stage runs
the embedding and the last the head ([Schedules](#schedules-1f1b-zbh1)).
A model with no scanned repeat is an error that says to repeat its blocks and scan
them, `methods(repeat=scan)`, and so is a repeat whose passes do not split over the
stages evenly, or as `methods.stages` says ([Uneven stages](#uneven-stages-methodsstages)).

The batch's rows split into `methods.microbatches` microbatches, row `i·M + j` into
microbatch `j`, so that each splits over the batch axes as the rows do. A method runs
them in a `jax.shard_map` manual over `stage` alone, by a `lax.scan` over `R·M + S − 1`
ticks: at each, every stage runs its share of a round on a microbatch (`stacked.run`),
and a `ppermute` sends its output on to the next stage, the last's to the first. The
first stage takes each microbatch from a buffer of its own, where the last stage's
outputs come back: there a circular round starts again, and from there the stream
leaves, summed over the stages to every one.

| method     | each microbatch                                                   | a stage holds   | a stage idles           |
|------------|-------------------------------------------------------------------|-----------------|-------------------------|
| `gpipe`    | goes through the stages once; the default                         | M + S − 1 ticks | (S − 1) / (M + S − 1)   |
| `circular` | goes round them `methods.rounds` times, each stage holding a share of every round's layers | R·M + S − 1 ticks | (S − 1) / (R·M + S − 1) |
| `1f1b`     | as in gpipe, and its backward as soon as it can run               | 2S − 1          | (S − 1) / (M + S − 1)   |
| `zbh1`     | as in gpipe, its backward split in two, the weights' part where a stage would idle | S, and S of the weights' | (S − 1) / (3M + S − 1) |

What a stage holds is the residuals of as many ticks or microbatches. `circular` needs
at least one microbatch per stage, and only it reads `methods.rounds`. The bubble of
`zbh1` counts a forward, a backward of the input and one of the weights as a tick each.
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
step does not accumulate gradients over them as well. It gathers the parameters FSDP
splits once a step, before the ticks (`methods.gather = step`, [above](#gathered-once-reduced-once-methodsgather-methodsaccumulate)),
in float32, since the ticks add up a stage's gradients in the dtype of what they read;
`gather=layer` gathers each layer's as it runs, every tick, both ways. Where the rows
are independent and their axes span processes, `accumulate=auto` (or
`accumulate=unreduced` anywhere) reduces the gradients once too: the trainer says so
(`gather.apart`), and the pipeline's `shard_map` is then manual over the axes that
split the rows as well as over `stage`. A stage runs its device's rows alone, each
tick's gradients stay its own, and the `shard_map`'s transpose sums them over those
axes once, as they leave it, in float32. Else (`accumulate=reduced`, or an MoE) they
reduce every tick, as the backward makes them. With `gpipe` the stacked state
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

### Schedules: `1f1b`, `zbh1`

```
methods = methods(repeat=scan, microbatches=8, pipeline="1f1b")
methods = methods(repeat=scan, microbatches=8, pipeline="zbh1")
```

`gpipe` and `circular` run the forward alone, and JAX's transpose of their scan of
ticks is the backward, so a stage holds every tick's residuals until it. `1f1b` and
`zbh1` take each microbatch's backward by hand (`bux.parallel.scheduled`), at the
ticks of a table (`bux.parallel.schedule`), and a stage keeps a microbatch's residuals
from its forward to its backward alone. The forward is a `jax.vjp`, whose residuals
are a pytree of arrays: a few microbatches' of them sit in a buffer, and the backward
rebuilds the vjp from its slot, the stage's parameters put back, which it does not
save with each. A table gives each stage its ops in order, a queue to each lane of a
tick; at each tick a lane runs the next op of its queue once what the op reads is
there, and at the tick's end a stage sends its forward's output on and its
backward's input gradient back. The names are quoted: the DSL reads `1f1b` as a
number.

- **`1f1b`**: two lanes a tick, a forward and a whole backward, each as soon as it can
  run. Stage s holds 2(S − s) − 1 microbatches, not M + S − 1 ticks, and the stages
  idle as gpipe's do: as lockstep ticks they cannot keep the S − s of an asynchronous
  1F1B without idling longer.
- **`zbh1`**: one lane, and the backward split in two: `B` takes the input's gradient,
  on the critical path, and `W` the weights' later, in a tick the stage would idle.
  Stage s runs S − s forwards, then a backward, the weights of the microbatch s before
  it, and a forward, in turn (ZB-H1 of Qi et al., "Zero Bubble Pipeline Parallelism"):
  every stage holds S microbatches' residuals, each until its `W`, and S of what `B`
  leaves `W`, and the stages idle S − 1 ticks of 3M + S − 1.

`bux.parallel.split` splits a backward by its equations: those that lead to the
input's gradient run in `B` and hand those that lead only to the weights' the values
they read. A scan of the backward, the one over a stage's layers, splits into a scan
of each part, the first stacking what each pass saves for the second; a cond splits
into a cond of each; calls of jaxprs (jit, remat) are inlined; any other equation,
such as an MoE's shard_map, goes whole to `B`. No equation runs twice: of 4 Llama
blocks' backward, `W` runs the 28 matmuls of the weights' gradients and `B` the rest,
which `tests/test_schedules.py` checks against the whole backward, with remat too.

A backward needs the gradient of the loss as soon as the last stage has a
microbatch's forward, so these pipelines take the whole model, cut at the pipelined
repeat (`trainer._Cut`): the model's own forward, a microbatch at a time, up to the
repeat on the first stage (`pipelined.stream_into`), and from the repeat to the loss
on the last (`pipelined.leaving`), where the stage's layers stand in by their shapes.
So the embedding runs on the first stage alone, the head and the objective on the
last, each once a microbatch, where gpipe runs them on every stage for the whole
batch. Their gradients add up in float32 on their stage and sum over the stages as
they leave, and the stages' own add up in float32 on their devices. The losses the
stages' layers report enter the loss by their weights, and their means its metrics;
the router bias moves once a step, as under gpipe. A layer outside the repeat that
learns from the whole batch, as an MoE's router does, is an error: it would learn a
microbatch at a time. Gathered once and reduced once work as for gpipe: the
parameters come in cast to `compute` and gathered, and where the rows are apart each
device runs its own and every gradient sums over the rows' axes once, at the end. As
every gradient sums in float32 as it leaves, `precision(reduce=float32)` is no error
here. As under gpipe, `remat(first=n)` counts a stage's passes from its own first, and
precision's first and last branches multiply in `compute` before and after the
pipelined repeat and are an error inside it. The loss takes its logits by either
method of `methods.loss`. A forward alone, an eval's, runs gpipe's.

Each kind of op runs under a `lax.cond` of its own: in one `lax.switch` of them all,
XLA copied every buffer that one op writes and another reads in each branch, a tick
at a time, and on B300s `1f1b` on `stage=2, fsdp=4` ran at 534k tokens a second with
a peak of 40.4 GiB so, against 673k and 24.6 GiB with a cond each.

On one node of 8 B300s, the 1.3B dense model of [Gathered once](#gathered-once-reduced-once-methodsgather-methodsaccumulate)
(16 Llama blocks, a vocabulary of 128k), 64 rows of 4096 a step in bfloat16,
`repeat=scan, microbatches=8, loss=logits`, tokens a second and the peak memory of a
GPU:

| mesh             | `methods.pipeline`                 | tok/s | peak      |
|------------------|------------------------------------|-------|-----------|
| `stage=2, fsdp=4` | `gpipe`                           | 625k  | 97.8 GiB  |
|                  | `circular`, `rounds=2`             | 643k  | 95.2 GiB  |
|                  | `"1f1b"`                           | 673k  | 24.6 GiB  |
|                  | `"1f1b"`, `stages=[10, 6]`         | 697k  | 33.6 GiB  |
|                  | `"zbh1"`                           | 594k  | 28.8 GiB  |
| `stage=4, fsdp=2` | `gpipe`                           | 508k  | 117.9 GiB |
|                  | `circular`, `rounds=2`             | 536k  | 111.1 GiB |
|                  | `"1f1b"`                           | 507k  | 49.5 GiB  |
|                  | `"1f1b"`, `stages=[5, 5, 5, 1]`    | 613k  | 60.6 GiB  |
|                  | `"zbh1"`                           | 467k  | 53.8 GiB  |

The last stage runs the head, which at a vocabulary of 128k is as much work as 4.7
blocks' forward, so with as many blocks as the others it is the slowest stage, and
the others wait for it: in a profile of `1f1b` on `stage=2` the first stage's GPUs
spend 124 ms of a 400 ms step waiting in a send, and the last stage's 202 ms in
GEMMs, against the first's 137. Fewer blocks on it take that back: on 4 stages the
head is nearly a stage's work, and `stages=[5, 5, 5, 1]` runs 21% faster than gpipe
on half its memory. At 16 microbatches the gap grows: on `stage=4`, 652k tokens a
second at 32.0 GiB against gpipe's 521k at 117.8 GiB, and on `stage=2, stages=[10,
6]`, 651k at 16.9 GiB against 578k at 107.1 GiB. With the loss by chunks, the
default, gpipe holds less and the gap stays: on `stage=2`, 624k at 60.0 GiB against
`stages=[10, 6]`'s 679k at 30.8 GiB; on `stage=4`, 484k at 65.0 GiB against
`stages=[5, 5, 5, 1]`'s 596k at 52.7 GiB.

`zbh1` is slower here than `1f1b`, and no faster on uneven stages: `B` writes out
what `W` reads, and `W` reads it back, and on `stage=2` a step spends 48 ms of 461 in
copies, against 25 of 400 under `1f1b`; at 8 microbatches the bubble it saves is
smaller than that. Splitting the head's backward too, its input's gradient into the
last stage's `B` and its weights' into its `W`, balanced the stages but ran no faster
(543k on `stage=2, stages=[10, 6]`, 482k on `stage=4, stages=[5, 5, 5, 1]`), and
held the head's residuals, a microbatch's logits, in a slot of each stage, as every
stage runs one program (113 GiB on 4 stages): so the head's backward stays whole, in
the last stage's forward. Reduced once (`accumulate=unreduced`), `1f1b` runs as fast
(676k on `stage=2`) and `zbh1` slower (467k). With GEMM autotuning off
and deterministic ops, `gpipe`, `1f1b` and `zbh1`, on even stages and uneven, give
the same loss at the first step, and within 7e-4 at the tenth, on a loss falling
0.07 a step.

### Uneven stages: `methods.stages`

```
mesh    = mesh(stage=4, fsdp=2)
methods = methods(repeat=scan, microbatches=8, pipeline="1f1b", stages=[5, 4, 4, 3])
```

`methods.stages` gives each stage its passes of the pipelined repeat, first to last:
fewer on the last, which runs the head under `1f1b` and `zbh1`. A sharding splits the
stacked layers evenly, so each stage holds as many as the most any runs
(`pipeline.spread`): its own, then copies of its first, which it skips by a
`lax.cond`. The copies are parameters too, with their optimizer's state, and their
gradient is 0; a run resumes on the same split only, and on another it fails at once,
as its checkpoints keep the split ([Resuming](training.md#resuming)). The split must
give every stage its passes and sum to the repeat's, in one round, else it is an error
that says so.

### Later: DualPipe

DualPipe (DeepSeek-V3) and its half, DualPipeV, send microbatches through the stages
both ways, each device holding two chunks of the layers, and run one microbatch's
forward and another's backward in one op, so that an MoE's all-to-all in one hides
behind the other's compute. It fits the executor of `1f1b` and `zbh1`: its V is a
table of ops on two chunks, a device holding chunk s and chunk 2S − 1 − s; the
layers stack by chunk as `circular` stacks rounds, the second in the stages' reverse
order; each chunk sends its way, four ppermutes a tick; and an op `FB` runs a
forward and a backward in one branch, where XLA's scheduler can overlap them. The
embedding and the head then both sit on device 0, at the two ends of the V. It pays
where an MoE's all-to-all is exposed, which `1f1b` and `zbh1` do not hide.
