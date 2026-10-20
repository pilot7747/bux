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

`bux train` uses every visible device; `bux launch` still gives a run one GPU. A mesh that needs more devices than there are,
a `batch` that does not split over the batch axes, and a `seq_len` that does not split
over `seq` or `stream` are errors before anything is built.

## Names

A layer names the dimensions of a parameter where it makes it, and the layout says
which mesh axes each name splits over:

```python
from bux import parallel

self.q = nnx.Linear(d, h * k, kernel_init=parallel.named("embed", "heads"), rngs=rngs)
self.sinks = nnx.Param(jnp.zeros(h), names=("heads",))
```

The names are the fields of `layout`, a closed set; any other is an error that names
the parameter.

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
   earlier dimension took. If their sizes do not divide the dimension, it stays
   whole: GQA's 2 kv heads under `tensor=4` are replicated.
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
`axis_names={axis}`, the other axes staying Auto. What reads the whole batch (MoE
routing, balance losses, the router bias) stays outside, in the global view, so its
semantics do not change.

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
