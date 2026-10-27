(parallelism-layout)=
# Layout

The layout says which mesh axes each *named dimension* of a parameter or an activation
splits over. Layers never say how they split: a layer names the dimensions of what it
makes (`embed`, `heads`, `mlp`, ...), and {bux}`layout` maps each name to mesh axes.
FSDP then splits each parameter's first whole dimension on top.

You rarely need to change it. The defaults give data parallelism, FSDP, TP, EP and
context parallelism wherever the mesh has those axes. Change the layout for sequence
parallelism, for ZeRO-1, or to split something over another axis.

## Quick start

```bux
mesh   = mesh(fsdp=4, tensor=2)
layout = layout(stream=tensor)          # sequence parallelism
```

Two other common layouts:

- `layout(zero=1)`: FSDP splits only the optimizer's state.
- `layout(fsdp=fsdp)`: FSDP splits over `fsdp` alone, not over `fsdp`, `context` and
  `expert`.

Fields combine: `layout(stream=tensor, zero=1)`.

`examples/experiments/parallel/mesh_sweep.bux` trains the same 1B model five ways over
8 GPUs, sweeping the layout beside the mesh:

```bux
mesh   = ddp:   mesh(data=8)
       | fsdp:  mesh(fsdp=8)
       | zero1: mesh(fsdp=8)
       | hsdp:  mesh(data=2, fsdp=4)
       | tp:    mesh(fsdp=4, tensor=2)
layout = ddp:   layout
       | fsdp:  layout
       | zero1: layout(zero=1)
       | hsdp:  layout
       | tp:    layout(stream=tensor)
```

## Names

The names are the fields of {bux}`layout`, a closed set; any other name is an error
that names the parameter. Each takes one axis, a list of axes, or `none`.

| Name | What it is | Default axes |
|---|---|---|
| `batch` | rows | `[data, fsdp, expert]` |
| `seq` | positions, inside a layer | `context` |
| `stream` | positions of the residual stream, between branches | `context` |
| `embed` | the model's width | `none` |
| `vocab` | the vocabulary | `tensor` |
| `heads` | query heads, or a mixer's heads | `tensor` |
| `kv_heads` | key and value heads | `tensor` |
| `mlp` | hidden units | `tensor` |
| `experts` | experts | `expert` |
| `latent` | low-rank bottlenecks: MLA, LatentMoE | `none` |
| `layers` | the stacked passes of a scanned repeat | `none` |
| `stages` | the stacked passes of a pipelined repeat, a share per stage | `stage` |

Two more fields are not names:

| Field | Values | Default | What it does |
|---|---|---|---|
| `fsdp` | axes | `[fsdp, context, expert]` | what FSDP splits a parameter's first whole dimension over ([below](#fsdps-axes-layoutfsdp)) |
| `zero` | `1`, `3` | `3` | `1`: FSDP splits only the optimizer's state ([below](#zero-1-layoutzero)) |

A layer names a parameter's dimensions where it makes it:

```python
from bux import parallel

self.q = nnx.Linear(d, h * k, kernel_init=parallel.named("embed", "heads"), rngs=rngs)
self.sinks = nnx.Param(jnp.zeros(h), names=("heads",))
```

When a dimension flattens several of what it names, as a projection's `h * k` columns
flatten `h` heads, name it with the count, `("heads", h)`. It then splits only between
heads.

Names are bux's metadata, under `names`, not Flax's `out_sharding`:

- Flax never acts on them, so they cost nothing without a mesh;
- `nnx.eval_shape` under a mesh does not try to resolve them;
- the optimizer's state inherits them from the parameters it tracks.

## How names resolve

For each parameter, `bux.parallel.layout` turns its names into a partition spec in
three steps.

1. **Each dimension by its name.** Take the axes of its name, minus axes of size 1
   and axes an earlier dimension already uses (and, inside a `shard_map`, the axes it
   maps by hand). If their sizes do not divide the dimension (or its count), it stays
   whole. So GQA's 2 kv heads under `tensor=4` are replicated, though their `2 * k`
   columns would split 4 ways.
2. **Then FSDP.** FSDP's axes are those of `layout.fsdp` (`fsdp`, `context` and
   `expert`) that no dimension splits over already. The first whole dimension they
   divide splits over all of them. If none divides by them all, FSDP drops axes from
   the end of that list until some whole dimension divides by the rest.
   - *The first dimension*, so that each device's shard is whole rows of the array,
     which XLA gathers and scatters as they lie: an expert stack splits by expert, a
     kernel by its inputs. Splitting by the largest dimension instead put a transpose
     in every gather and reduce-scatter: 21.6 ms of a 314 ms step on 12 of
     Qwen3-30B-A3B's layers on 8 B300s, against 3.6 ms by expert.
   - *Never `layers`*, the stacked passes of a scanned repeat: a scan reads one layer
     at a time.
   - *A dimension its own axes do not divide comes last*, as when `tensor=4` does not
     divide CSA's 2 output groups. The layer's activations keep that dimension whole
     too. Had FSDP split the parameter there, XLA would split the activations to match
     and rematerialize them in full.
3. **FSDP applies to parameters and the optimizer's state**, or with `layout(zero=1)`
   to the optimizer's state alone. Other state, such as RNG state, splits by its names
   alone.

What this means in practice:

- `data` is never one of FSDP's axes: its copies are HSDP's.
- A layer with no names at all still trains under FSDP.
- An expert kernel keeps its experts over `expert` and splits over `fsdp` and
  `context`.
- Names matter only for TP and EP.

### Born on their shards

`parallel.build(make)` creates a module with every parameter born on its shards, so no
parameter is ever whole on one device; its values are those `make()` gives. The
trainer moves its state to these shardings with `parallel.shard_state` when it is
created, and every step keeps it there.

## FSDP's axes: `layout.fsdp`

By default FSDP splits over `context` and `expert` as well as `fsdp`, so a dense
parameter and its Adam state are not copied over those axes. `layout.fsdp = fsdp`
splits over `fsdp` alone.

On 8 CPU devices, a dense run and an MoE train as on one device on
`fsdp=2, context=2, expert=2` and `context=2, expert=4`, with no rematerialization in
full.

On 8 B300s (2026-10-23), the default is a little faster, and halves what the dense
model keeps per GPU. The models are the 1.3B dense model of [Muon](muon.md) on
`fsdp=4, context=2`, and its 5.2B MoE on `fsdp=2, expert=4`, at 8 rows of 4096 tokens
per GPU in bfloat16:

| Model, mesh | `layout.fsdp = fsdp` | Default | Parameters per GPU |
|---|---|---|---|
| dense, `fsdp=4, context=2` | 886k tok/s | **903k** | 1.19 GiB → 611 MiB |
| MoE, `fsdp=2, expert=4` | 820k tok/s | **830k** | 3.42 GiB → 2.40 GiB |

## ZeRO-1: `layout.zero`

`layout(zero=1)` keeps the parameters whole on every device and splits only the
optimizer's state over FSDP's axes (ZeRO-1). The default, `zero=3`, splits both.

With `methods(gather=step)`, the default, a step gathers the parameters once anyway.
So ZeRO-1 saves no collective, and it holds float32 copies where the step holds copies
in `compute` ([Gathered once, reduced once](microbatches.md#gathered-once-reduced-once-methodsgather-methodsaccumulate)).
Under Muon's `owner` method, `zero=1` changes which collectives a step makes
([Muon](muon.md)).

## Sequence parallelism

`layout(stream=tensor)` splits the residual stream's positions over `tensor` between
the residual branches, like Megatron's sequence parallelism. Each layer's input is
constrained to `(batch, seq, none)`, so it is gathered once before its projections.
The output projection's partial sums then reduce-scatter back to each device's
positions ([Tensor parallelism](tensor.md)). Under context parallelism,
`layout(stream=[context, tensor])` splits the stream over both.

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

## The other layers

Softmax attention splits by head ([How attention splits by head](attention.md#how-attention-splits-by-head)).
The other layers split the same way:

- dense layers by hidden unit;
- the embedding, the head and the logits by the vocabulary;
- the recurrent mixers by head (Mamba's by channel), so that each device runs the
  recurrence of its own heads with no communication.

| Layer | Parameter | Names |
|---|---|---|
| MLP | `up`, `gate`, their biases | `(embed, mlp)` |
| MLP | `down` | `(mlp, embed)` |
| embedding | the token table | `(vocab, embed)` |
| embedding | learned positions | `(none, embed)` |
| head | `proj`, its bias or the tied head's | `(embed, vocab)` |
| linear attention, RWKV-7 | `q`, `k`, `v`, `r`, `a`, `b`, the output gate | `(embed, heads)` |
| linear attention, RWKV-7 | a low-rank projection's `down`, `up` | `(embed, latent)`, `(latent, heads)` |
| linear attention, RWKV-7 | `o` | `(heads, embed)` |
| linear attention, RWKV-7 | `A_log`, `dt_bias`, `slopes`, `k_k`, `k_a`, `r_k` | `(heads, ...)` |
| Mamba, short conv | the conv's channels, `dt_proj`, `dt_bias`, `A_log`, `D` | `mlp` |
| Mamba, short conv | `x_proj`; `out_proj` | `(mlp, none)`; `(mlp, embed)` |
| Mamba-2 | `dt_bias`, `A_log`, `D`; `out_proj` | `(heads,)`; `(heads, embed)` |
| Engram | the table | `(vocab, none)` |

Flattened heads are named with their count, as attention's are.

Fused projections stay whole, and every device computes them: GQA's fused `qkv`, the
SSMs' `in_proj` (u and the gate side by side), Mamba-2's conv over u, B and C,
Engram's `kv` and nextn's `proj`. Splitting such a projection into parts would keep
its values under AdamW, but not under Muon, which orthogonalizes it as one matrix.
Linear attention's single conv over q, k and v convolves each separately
(`CausalConv.side_by_side`), so that they stay on their heads.

Norms, the residual algebras, RWKV-7's token shift, `scale` and `linear` are not
named: their widths mean whatever comes before them, and FSDP alone splits them.

### Their activations

- Each mixer's input is `(batch, seq, none)`, and so is the MLP's: with
  `layout(stream=tensor)` the input is gathered once before the projections, not
  their weights ([Tensor parallelism](tensor.md)). Per-head arrays are
  `(batch, seq, heads, ...)`, the hidden units `(batch, seq, mlp)`, and the SSMs'
  channels before `out_proj` are named like their parameters.
- The embedding gathers from its table, whole except for the vocabulary, into
  `(batch, seq, none)`, then `shard_stream`. From a table split over `fsdp`, XLA
  rematerializes the table in full; into positions split over `tensor`, it gathers
  the table every step. Engram gathers from its table the same way.
- The logits are `(batch, seq, vocab)`, for tied heads too, and the loss reduces over
  the vocabulary where it lies: the log-normalizer and the target's logit are
  all-reduces of `(b, t)`, for the loss, `z_loss` and `mtp` alike.
- By chunks, the default `methods.loss` (`bux/architecture/logits.py`), the logits
  are made one chunk of each device's positions at a time, `(batch, seq, none, vocab)`
  over the devices' shards. The head's kernel is gathered over `fsdp` once for all the
  chunks. Each device adds up its share of the kernel's gradient on its own, reduced
  once after the last chunk. `methods(loss=logits)` makes the logits whole.
- What the embedding, nextn and Engram return is the stream, widened or not, and it
  goes through `shard_stream`.

### Measured on 8 CPU devices

Every family trains as on one device to 4e-6, with no involuntary rematerialization,
under:

- `fsdp=2 × tensor=4`;
- `fsdp=4 × tensor=2` with sequence parallelism;
- `tensor=2` and `tensor=4`;
- `fsdp=8`.
