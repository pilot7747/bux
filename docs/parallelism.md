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
and `parallel.shard_batch` builds a batch with `jax.make_array_from_callback`: every
process makes all of it, as it depends only on the seed and the step, and puts the
rows its own devices hold. `tests/test_launch.py` trains the smoke run as two
processes of 4 CPU devices each, through `jax.distributed` on localhost, as one
process of 8 does.

`bux plan` shows what each run takes before it starts: its GPUs, the bytes of its
parameters and of its optimizer's state on each, and the share of a step its
pipeline's stages idle ([below](#pipelines-methodspipeline)). The bytes come from
`parallel.shardings` of the model built over a `jax.sharding.AbstractMesh`, which
`parallel.use` takes as it does a mesh, so the plan needs no devices:

```
$ bux plan examples/experiments/smoke.bux 'mesh = fsdp: mesh(fsdp=8) | one: mesh(data=1)'
2 runs
id          params  tokens  tok/param  gpus  params/gpu  optim/gpu  bubble  run
bbbf3ae709  156k    25.6k   0.2        8     76.2KiB     152KiB     0%      smoke[fsdp]
bbbf3ae709  156k    25.6k   0.2        1     609KiB      1.19MiB    0%      smoke[one]
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
2. Then FSDP: the largest whole dimension that the `fsdp` size divides splits over
   `fsdp`, unless a dimension already does. So a layer with no names at all still
   trains under FSDP; names matter only for TP and EP.
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

The indexer picks its top-k with a stable sort: `lax.top_k` gathers every row onto
each device.

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
| `auto`    | the default: `tokamax` on a GPU (7x `xla`'s speed at t = 4096 on a B300), `xla` elsewhere and for sinks, and for what only `xla` has |
| `xla`     | all t by t logits at once. With `seq` over `context` a device has its queries' rows, and every key and value: all-gather CP |
| `ring`    | each device keeps its queries, while the keys and values go round `context` a block at a time, by `ppermute`, into an online softmax; a block wholly ahead of the queries, or a window or more behind, is skipped. Its custom VJP sends them round again with their gradients, which then go home |
| `ulysses` | all-to-alls over `context`, from a device's positions of every head to every position of its own heads; `xla` on those; and back. `context` must divide a device's heads; kv heads it does not divide are repeated |
| `tokamax` | tokamax's flash attention (Mosaic, then Triton, then XLA), a document from its first token on as `k_start`, split by its `q_sharding`: all-gather CP in a kernel. No sinks; Linux only (the `cuda` extra) |

What only `xla` has stays in the layer: a mask of the keys each query keeps (an
indexer's top-k, CSA's compressed latents), relative bias, length scale and attention
dropout. A layer with one of them under a method other than `auto` raises `NotImplementedError`
naming it, as a method does for a feature of the `AttentionArgs` it lacks. The indexer's
loss reads the attention's probabilities, which only `xla` makes; it comes with the
indexer's top-k, so it is `xla`'s too.

`ring` is a `shard_map` over `context` alone. `ulysses` is manual over every axis
that splits q: an all-to-all in a `shard_map` of auto axes makes XLA gather every
dimension they split, the rows too. Inside a `shard_map`, `shard` and `spec` leave
out the axes it maps, so that a layer's constraints still hold there.

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

Measured on 8 CPU devices: two windowed GQA layers (4 heads, 2 kv heads) in a
training step on packed rows train as on one device, to 1e-6 over 3 steps, by every
method on `context=2, tensor=2`, `context=4, fsdp=2` and `context=8` (not `ulysses`
there: 4 heads), and none makes XLA rematerialize an array in full. The collectives of
the compiled step (all-gathers / all-reduces / collective-permutes / all-to-alls; a
collective in a loop counts once):

| mesh                  | `xla`        | `ring`        | `ulysses`    | `tokamax`    |
|-----------------------|--------------|---------------|--------------|--------------|
| `context=2, tensor=2` | 12 / 12 / 4 / 4 | 12 / 10 / 10 / 4 | 5 / 10 / 4 / 20 | 10 / 14 / 5 / 8 |
| `context=4, fsdp=2`   | 38 / 8 / 4 / 4  | 38 / 6 / 12 / 4  | 31 / 6 / 4 / 20 | 36 / 10 / 5 / 8 |
| `context=8`           | 11 / 4 / 4 / 4  | 4 / 2 / 12 / 4   |                 | 9 / 6 / 5 / 8   |

`xla` gathers each layer's keys and values whole; `ring` and `ulysses` gather no
activation but the documents' ids (int32), and FSDP's parameters. A layer adds 4
collective-permutes under `ring` (the blocks forward; the blocks and their gradients
backward; the gradients home), 3 on `context=2`, where the backward pass shares the
forward's one step, and 8 all-to-alls under `ulysses`. The temp memory of
the gradient of one attention at t = 2048 on `context=4`: `xla` 66.0 MiB, `ulysses`
64.5 MiB (whole rows of a quarter of the heads), `ring` 21.9 MiB (two blocks of keys).
`tokamax` here is its XLA implementation; `scripts/gpu_smoke/attention.py` runs its
kernels, and every method, on GPUs.

Not yet: zigzag load balancing (device `c - 1` of the ring attends to `c` blocks,
device 0 to one), and overlapping each block's transfer with its compute.

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
  of `(b, t)`, for the loss, `z_loss` and `mtp` alike.
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
over positions, is its reference. Under `scan`, the delta rule of a gate bounded at
-5 or above (KDA's `gate_lower_bound`, DeltaNet's none) runs by cuDNN's fused KDA
kernel on a B200 or B300 instead (`fused_delta`); it overflows below -5.

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
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`.

## Muon: `methods.orthogonalize`

Newton-Schulz takes fifteen products of each update matrix per step. On a matrix split
over devices, most of them need an all-reduce, so how Muon reaches whole matrices is a
method:

| method   | what it does                                                                 |
|----------|------------------------------------------------------------------------------|
| `gspmd`  | Newton-Schulz on whatever shards the compiler gives each update; the default |
| `gather` | each update whole in a `jax.shard_map` for Newton-Schulz, and back on its parameter's shards after |

```
methods = methods(orthogonalize=gather)
```

Under `gather`, a stack of matrices (the experts) is whole matrix by matrix, and its
stacked dimension splits over the axes that frees, where they divide it: each device
orthogonalizes its own experts. A lone matrix is whole on every device, and every
device runs its Newton-Schulz. Owner-per-matrix schemes (Moonlight, Dion), which spread
that work, are the next method.

A method gets every update of a step at once, so that it can choose where each runs:

```python
from bux.parallel import methods


@methods.orthogonalize.register("mine")
def _(updates, dims, shardings, newton_schulz):
    # updates: an array per Muon parameter; dims: their MuonDimensionNumbers;
    # shardings: their parameters' NamedShardings, None outside parallel.use;
    # newton_schulz(x, dims): optax's Newton-Schulz.
    return [newton_schulz(x, d) for x, d in zip(updates, dims)]
```

Collectives in the compiled train step on 8 CPU devices (all-reduces / all-gathers /
all-to-alls) for the 2-block Llama and the MoE of `tests/test_parallel.py`:

| model, mesh               | AdamW       | Muon, `gspmd` | Muon, `gather` |
|---------------------------|-------------|---------------|----------------|
| Llama, `fsdp=8`           | 21 / 24 / 0 | 72 / 84 / 0   | 21 / 38 / 0    |
| Llama, `fsdp=4, tensor=2` | 21 / 25 / 0 | 72 / 85 / 0   | 21 / 39 / 0    |
| MoE, `fsdp=8`             | 26 / 33 / 8 | 67 / 183 / 8  | 26 / 48 / 14   |
| MoE, `fsdp=4, tensor=2`   | 25 / 36 / 6 | 66 / 186 / 6  | 25 / 51 / 12   |

With `gather`, Muon all-reduces as often as AdamW, and adds one all-gather per matrix
and two all-to-alls per expert stack; the optimizer's update alone does no all-reduce.
Step times on fake CPU devices, which share one CPU, say nothing.

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

| method          | what the forward keeps                                     |
|-----------------|------------------------------------------------------------|
| `none`          | everything, but in a scanned pass (below); the default     |
| `full`          | nothing but the input of each branch or pass               |
| `dots`          | the products of matrices                                   |
| `dots_no_batch` | the products of matrices without batch dimensions          |

Each residual branch outside any other is rematerialized as a whole, with the read
and write of the residual algebra; a scanned pass is, as one; and nothing inside
either is again. In a `.bux` file `none` is the null value, so a sweep writes the
method as a string: `methods.remat = "none" | full`. Another policy is a function:

```python
import jax

from bux.parallel import methods

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

| remat           | temp memory, unrolled | scanned |
|-----------------|-----------------------|---------|
| `none`          | 327 MiB               | 142 MiB |
| `full`          | 22 MiB                | 23 MiB  |
| `dots`          | 132 MiB               | 138 MiB |
| `dots_no_batch` | 79 MiB                | 82 MiB  |

`Scanned.layers` holds the stacked passes, and `stacked.run(graphdef, state, times, x,
taps)` runs any stack of them, such as a pipeline stage's share.

## Precision and microbatches

```
precision = precision(compute=bfloat16)   # in the run's id
methods   = methods(microbatches=4)       # gradient accumulation: not in it
```

Unlike the settings above, `precision` changes what a run computes, so it is in the
run's id: `precision.compute = float32 | bfloat16` is a sweep of two runs. `compute` is
`float32` or `bfloat16`; float16 would need loss scaling.

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
| the objective                                   | the logits, as the loss and the z-loss take them  |

Softmax attention takes its softmax in float32, under every method of
`methods.attention`. Every other matmul runs in `compute`, and the stream stays in it from the embedding to the logits,
which `tests/test_precision.py` checks for every family. Gradients are reduced over the
batch axes in the dtype the backward makes them, bfloat16 for a weight's; they add up
over microbatches in float32.

XLA:CPU fails to compile a bfloat16 `psum` that a `shard_map`'s transpose makes (its
reducer ends in a copy, which `ChangeOpDataType` cannot clone), and aborts the process.
The MoE's `sorted` and EP methods make one on more than one device, and so do the
pipelines; `ring`, `ulysses` and `pass_state` train in bfloat16 on CPU. The tests split
bfloat16 runs with `methods(moe=dense)` and no pipeline. GPUs all-reduce bfloat16
natively, which this has not been run on.

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
id          params  tokens  tok/param  gpus  params/gpu  optim/gpu  bubble  run
bbbf3ae709  156k    25.6k   0.2        1     609KiB      1.19MiB    0%      smoke[one]
bbbf3ae709  156k    25.6k   0.2        2     369KiB      738KiB     20%     smoke[pp]
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
