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
| `mesh`    | a size per axis: `data`, `stage`, `fsdp`, `context`, `expert`, `tensor` | every device on `data` |
| `layout`  | which axes each named dimension splits over            | see below                  |
| `methods` | how each part that communicates is computed, by name   | the one-device code        |

`bux.parallel` reads them; the configs are in `bux/train/config.py`.

## The mesh

One mesh, always of all six axes, in the order `data, stage, fsdp, context, expert,
tensor`: the slowest links first, so that `tensor` stays inside a node. An axis not
set has size 1, and `data` takes the devices the others leave. Every axis is Auto:
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
than there are fails at once. A run alone also trains on many hosts, a process each,
when Slurm, Open MPI or `JAX_NUM_PROCESSES` say so
([Launching](training.md#launching)). The mesh then spans every process's devices,
and `parallel.rows` builds a batch with `jax.make_array_from_callback`: every process
makes all of it, as it depends only on the seed and the step, and puts the rows its
own devices hold. `tests/test_launch.py` trains the smoke run as two processes of 4
CPU devices each, through `jax.distributed` on localhost, as one process of 8 does.

`bux plan` shows what each run takes before it starts: its GPUs, and the bytes of its
parameters and of its optimizer's state on each. They come from `parallel.shardings`
over a `jax.sharding.AbstractMesh`, which `parallel.use` takes as it does a mesh, so
the plan needs no devices:

```
$ bux plan examples/experiments/smoke.bux 'mesh = fsdp: mesh(fsdp=8) | one: mesh(data=1)'
2 runs
id          params  tokens  tok/param  gpus  params/gpu  optim/gpu  run
bbbf3ae709  156k    25.6k   0.2        8     76.2KiB     152KiB     smoke[fsdp]
bbbf3ae709  156k    25.6k   0.2        1     609KiB      1.19MiB    smoke[one]
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

Names are bux's metadata, under `names`, not Flax's `out_sharding`: Flax never acts on
them, so they cost nothing without a mesh, and `nnx.eval_shape` under one does not try
to resolve them. The optimizer's state inherits them from the parameters it tracks.

### How names resolve

1. Dimension by dimension, each name's axes, less the axes of size 1 and those an
   earlier dimension took. If their sizes do not divide the dimension, or its count,
   it stays whole: GQA's 2 kv heads under `tensor=4` are replicated, though their
   `2 * k` columns would split 4 ways.
2. Then FSDP: the largest dimension left whole that the `fsdp` size divides splits
   over `fsdp`, unless a dimension already does. So a layer with no names at all
   trains under FSDP; names matter for TP and EP alone.
3. FSDP is for parameters and the optimizer's state, or with `layout(zero=1)` for the
   optimizer's state alone. Anything else, such as RNG state, keeps its names alone.

`parallel.build(make)` makes a module with every parameter born on its shards, so no
parameter is ever whole on one device; its values are those `make()` gives. The
trainer puts its state on these shardings when it is made, and every step keeps it
there.

## Activations

The stream is constrained at every residual branch, rows over `batch` and positions
over `stream`, the rest whole. A layer adds a constraint where it knows what a
dimension means:

```python
q = parallel.shard(q, "batch", "seq", "heads", None)
```

`shard`, `stream` and `spec` do nothing outside `parallel.use`, so one device runs as
if they were not there. Batches go to the devices by `parallel.rows`: every `(b, t)`
array over `batch` and `seq`.

## Methods

What communicates by hand does it inside a method: a function in a registry of
`bux.parallel.methods`, one per field of `methods`, picked by name. Another method is a
function:

```python
from bux.parallel import methods


@methods.attention.register("mine")
def _(q, k, v, visibility): ...
```

and `methods = methods(attention=mine)` picks it. `parallel.method("attention")` is
the run's, or the default outside `use`. A name that is not registered is an error at
`use`. The section of each kind says its signature, and the default is registered
where its layer lives.

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
    trainer.update(batch)  # the batch through parallel.rows
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
`methods.attention`, `(q, k, v, visibility) -> out`: queries `(b, t, h, k)`, keys and
values `(b, t, g, k)` and `(b, t, g, v)` of `g` kv heads at the queries' positions,
and the head outputs `(b, t, h, v)`. A query sees no key ahead of it; the
`Visibility` (`bux.parallel.context`) carries the rest: the documents' ids
(`Segments.ids`, made in the global view before any method), the window, the scale,
the soft cap and the sinks. Every method takes its softmax in float32.

```
mesh    = mesh(context=4, tensor=2)
methods = methods(attention=ring)
```

| method    | what it does                                                         |
|-----------|----------------------------------------------------------------------|
| `xla`     | the default: all t by t logits at once. With `seq` over `context` a device has its queries' rows, and every key and value: all-gather CP |
| `ring`    | each device keeps its queries, while the keys and values go round `context` a block at a time, by `ppermute`, into an online softmax; a block wholly ahead of the queries, or a window or more behind, is skipped. Its custom VJP sends them round again with their gradients, which then go home |
| `ulysses` | all-to-alls over `context`, from a device's positions of every head to every position of its own heads; `xla` on those; and back. `context` must divide a device's heads; kv heads it does not divide are repeated |
| `tokamax` | tokamax's flash attention (Mosaic, then Triton, then XLA), a document from its first token on as `k_start`, split by its `q_sharding`: all-gather CP in a kernel. No sinks; Linux only (the `cuda` extra) |

What only `xla` has stays in the layer: a mask of the keys each query keeps (an
indexer's top-k, CSA's compressed latents), relative bias, length scale and attention
dropout. A layer with one of them under another method raises `NotImplementedError`
naming it, as a method does for a feature of the visibility it lacks. The indexer's
loss reads the attention's probabilities, which only `xla` makes; it comes with the
indexer's top-k, so it is `xla`'s too.

`ring` is a `shard_map` over `context` alone. `ulysses` is manual over every axis
that splits q: an all-to-all in a `shard_map` of auto axes makes XLA gather every
dimension they split, the rows too. Inside a `shard_map`, `shard` and `spec` leave
out the axes it maps, so that a layer's constraints still hold there.

Another method is a function:

```python
from bux.parallel import methods
from bux.parallel.context import Visibility


@methods.attention.register("mine")
def _(q, k, v, seen: Visibility):
    # seen.ids (b, t) or None, seen.window, seen.scale, seen.softcap, seen.sinks (h,)
    if seen.sinks is not None:
        raise NotImplementedError("methods.attention = mine has no sinks")
    return methods.attention.find("xla")(q, k, v, seen)
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
  `(batch, seq, none)`, then `stream`. From a table split over `fsdp`, XLA
  rematerializes it in full; into positions split over `tensor`, it gathers the
  table every step. Engram gathers from its table the same way.
- The logits are `(batch, seq, vocab)`, tied heads too, and the loss reduces over the
  vocabulary where it is: the log-normalizer and the target's logit are all-reduces
  of `(b, t)`, for the loss, `z_loss` and `mtp` alike.
- What the embedding, nextn and Engram give is the stream, widened or not, and passes
  `stream`.

On the 8 fake devices, every family trains as on one device to 4e-6 under
`fsdp=2 × tensor=4`, `fsdp=4 × tensor=2` with sequence parallelism, `tensor=2`,
`tensor=4` and `fsdp=8`, with no involuntary rematerialization. The tests train at
`lr=1e-3`: at `1e-2` their small runs are unstable, and rounding (from `data=8` too)
grows past 1e-5 in three steps.

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
  expert, `sizes[i]` of them expert i's, by `jax.lax.ragged_dot`, or tokamax's where
  it is installed. Rows past `sizes.sum()` come out 0, before `norm`.
- `routed(z, idx, weights, up, gate, down, ffn, norm)`: sort, `experts`, unsort. A
  pick outside the experts of `up` adds nothing, so a device that holds experts
  `[i, i + n)` passes `idx - i` and its own kernels.

```python
from bux.architecture.moe import routed
from bux.parallel import methods


@methods.moe.register("mine")
def _(moe, z, idx, weights):
    return routed(z, idx, weights, *moe.kernels(), moe.ffn, moe.expert_norm)
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
| `all_to_all`        | each pick goes to its expert's device and back by `jax.lax.all_to_all`, in a buffer a peer that holds every local pick, so that none drops |
| `ragged_all_to_all` | the same, one ragged slice a peer, by `jax.lax.ragged_all_to_all`: buffers hold what every peer could send, but only what they do send moves. XLA:CPU lacks it |

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

## Muon: `methods.orthogonalize`

Newton-Schulz takes fifteen products of each update matrix a step. On a matrix split
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

With `gather`, Muon all-reduces as often as AdamW, and adds an all-gather a matrix
and two all-to-alls an expert stack; the optimizer's update alone does no all-reduce.
Step times on fake CPU devices, which share one CPU, say nothing.
