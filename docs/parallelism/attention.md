(parallelism-attention)=
# Attention and context parallelism

Softmax attention splits two ways. Under tensor parallelism it splits **by head**, as
Megatron does: each device projects the input to its own heads, attends with them, and
takes their rows of the output projection. Under context parallelism (CP) it splits
**by position**: the rows' positions split over the `context` axis, and a method of
`methods.attention` decides how each device's queries reach the keys and values held
by the others.

Use context parallelism when a row is too long for one device's memory or time, as for
rows of 32k to 128k tokens. Even on one device, `methods.attention` picks the kernel
attention runs by: the default, `auto`, takes the fastest one that runs the call.

## Turn it on

Split the positions over `context`, and pick a method. From
`examples/experiments/parallel/long_context.bux`, 64k-token rows over one node of 8
GPUs, with ring attention and the recurrent state passed along
([recurrent mixers](recurrence.md)):

```bux
use qwen3_next
use fineweb_wsd

d_model = 1024
delta   = gdn(num_heads=8, num_v_heads=16, head_dim=128)
attn.num_heads      = 8
experts.num_experts = 64
experts.top_k       = 8
model = embed "DDDA"^6 norm @h out mtp

seq_len = 64k
batch   = 8                                 # 512k tokens a step

mesh      = mesh(context=8)
precision = precision(compute=bfloat16)
methods   = methods(attention=ring, recurrence=pass_state, repeat=scan, remat=full,
                    microbatches=8)
```

`context` mixes with the other axes. Context and tensor parallelism together:

```bux
mesh    = mesh(context=4, tensor=2)
methods = methods(attention=ring)
```

The layout's `seq` and `stream` split over `context` by default
([Names](layout.md#names)), so setting `mesh.context` is what splits the positions;
`seq_len` must split over them, or the run is an error before anything is built.

:::{tip}
On long rows of packed documents, set `methods(attention=ring)`. At 128k tokens a row
`ring` is 1.5x `ulysses` there ([Measurements](#in-a-model-rows-of-128k)), but `auto`
cannot tell packed rows from rows of one document before a batch comes, and takes
`ulysses` wherever `context` divides a device's heads.
:::

## Options: `methods.attention`

The core of softmax attention (the logits, the probabilities and the weighted values)
is a method of `methods.attention`, a field of {bux}`methods`:

| Method | What it does |
|---|---|
| `auto` | The default. See [below](#auto-the-default). |
| `xla` | All t by t logits at once, the reference. With `seq` over `context`, a device has its queries' rows of the logits and every key and value: all-gather CP. |
| `cudnn` | cuDNN's fused attention (jax's) on each device's whole rows: rows of one document by its causal kernels (`flash.CUDNN`), documents in its packed layout. |
| `tokamax` | tokamax's flash attention, its Mosaic kernels: a document and a window as each query's first key, sinks by the logsumexp. With `seq` over `context`, a device gathers every key and value and attends with its queries in one call: all-gather CP, whose backward pass does every block. |
| `ring` | Every key and value gathered once over `context`, and a flash call for each device's block of them, merged by their logsumexps. Blocks no query sees are skipped. See [Ring](#ring). |
| `ulysses` | All-to-alls over `context`, from a device's positions of every head to every position of its own heads; `auto`'s kernel on those whole rows; and back. |

(attention-auto)=
### `auto`, the default

On a GPU, `auto` takes the fastest kernel that runs the call on each device's whole
rows:

1. cuDNN's causal kernels where each row is one document (`flash.CUDNN`), sinks too;
2. else cuDNN's packed layout, in rows of 8192 positions or more, or with a window;
3. else tokamax's kernel, which takes sinks and a soft cap;
4. in float32, tokamax's Triton kernel where there are no sinks;
5. else cuDNN where it takes the call, else `xla`.

With `seq` over `context`, `auto` runs `ulysses` where `context` divides a device's
heads, else `ring`. On a CPU, where neither kernel runs, it is `xla`, and it is `xla`
too for what only `xla` has ([below](#constraints-and-errors)).

(attention-ring)=
### `ring`

- Every key and value is gathered once over `context`. A device then makes a flash
  call (tokamax's; jnp's on a CPU) for each device's block of them, merged by their
  logsumexps.
- A device's own block comes first, from its own keys and values, while the others'
  are gathered.
- A block no query sees is skipped: one ahead of them all, or before every query's
  first key (its document's or its window's).
- Without documents or a window, the queries go zigzag: device `i` holds chunks `i`
  and `2c-1-i` of `2c`, so that every device attends to as much. Its keys and values
  stay in the positions' order, so its own block is then a gathered one.
- The backward pass runs each block's kernel backward with the final output and
  logsumexp, on the blocks the forward pass gathered, and reduce-scatters the blocks'
  gradients in the keys' dtype: in bfloat16, half the bytes of float32's.

## Constraints and errors

| Method | Takes | Raises |
|---|---|---|
| `xla` | everything | |
| `cudnn` | bfloat16 or float16; keys' and values' head dims equal, up to 256 in steps of 8 (its packed layout gets MLA's wrong) | `NotImplementedError` for a soft cap, sinks, positions over `context` (`ulysses` gives it whole rows), or a call it cannot take |
| `tokamax` | a GPU, the `cuda` extra (Linux only), bfloat16 or float16 | `NotImplementedError` otherwise |
| `ulysses` | `context` dividing a device's heads; kv heads it does not divide are repeated | `ValueError` naming both counts |
| `ring` | documents, a window, a soft cap and sinks; tokamax's kernel on a GPU, jnp's on a CPU | |

- **What only `xla` has stays in the layer:** relative bias, length scale and
  attention dropout. A layer with one of them under a method other than `xla` or
  `auto` raises `NotImplementedError` naming it, as a method does for a feature of the
  `AttentionArgs` it lacks.
- **Sparse attention** (an indexer's top-k, CSA's compressed latents) runs under `xla`,
  `tokamax` and `auto`, and raises `NotImplementedError` under the others
  ([Sparse attention](#sparse-attention)).
- Every method takes its softmax in float32.

(attention-by-head)=
## How attention splits by head

{bux}`gqa`, {bux}`mla`, {bux}`csa` and the {bux}`indexer` split by
head, as Megatron does: a device projects the input to its own heads, attends with
them, and takes their rows of the output projection, whose sums over `tensor` make the
output. Their parameters' names ([Names](layout.md#names)):

| Layer | Parameter | Names |
|---|---|---|
| all | `gate`, `rel_proj` | `(embed, heads)` |
| all | `o` | `(heads, embed)` |
| all | `sinks` | `(heads,)` |
| GQA | `q` | `(embed, heads)` |
| GQA | `k`, `v`, and the channels of their convs | `(embed, kv_heads)` |
| GQA | fused `qkv`, split where shards do not end | `(embed, none)` |
| MLA, CSA | down projections `q_a`, `kv_a` | `(embed, latent)` |
| MLA, CSA | up projections `q`, `kv_b` | `(latent, heads)`; `q` is `(embed, heads)` without a query latent |
| CSA | grouped `o_a`, by group | `(heads, none, none)` |
| indexer | `q`, `weights` | `(latent or embed, heads)` |

Each flattened dimension names its heads with their count, `("heads", h)`, so that it
splits only between heads ([How names resolve](layout.md#how-names-resolve)).

What every head shares is on every device: MQA's kv head, CSA's latents and the
indexer's keys. So are kv heads that `tensor` does not divide, which every device
computes and whose gradients the devices that share them sum.

The activations are constrained so:

| Array | Constrained to | So that |
|---|---|---|
| a layer's input | `(batch, seq, none)` | with `layout(stream=tensor)` it is gathered once before the projections: [sequence parallelism](layout.md#sequence-parallelism) |
| queries, keys, values, head outputs | `(batch, seq, heads, none)` | each device attends with its own heads |
| the logits | `(batch, heads, seq, none)` | the keys stay whole: with `seq` over `context`, that is all-gather CP |

The indexer picks its top-k by counting: it finds the k-th highest score bit by bit,
which splits with the rows and positions. `lax.top_k` sorts instead (6 times slower at
32k keys on a GPU), and in the global view it gathers every row onto each device. The
indexer's scores' kernel (`bux.architecture.attention.scoring`) is a `shard_map` that
gives a device its rows, positions and heads, and sums the heads' scores.

How TP's collectives meet the projections is [`methods.tensor`](tensor.md).

## How it works

A method of `methods.attention` is a function `(q, k, v, args) -> out`:

| Argument | Shape |
|---|---|
| `q`, the queries | `(b, t, h, dk)` |
| `k`, `v`, keys and values of `g` kv heads at the queries' positions | `(b, s, g, dk)`, `(b, s, g, dv)` |
| `out`, the head outputs | `(b, t, h, dv)` |

A query sees no key ahead of it. The `AttentionArgs` (`bux.parallel.context`) carry
the rest:

| Field | What it is |
|---|---|
| `ids` | each token's document (`Segments.ids`, made in the global view before any method), or none |
| `window` | a query attends to no key `window` or more behind it |
| `scale` | on every logit |
| `softcap` | logits to `softcap * tanh(logits / softcap)` |
| `sinks` | a per-head logit beside the keys', on no value |

(attention-sparse)=
### Sparse attention

Sparse attention (an indexer's top-k, CSA's compressed latents) attends by
`bux.architecture.attention.sparse`, under `xla`, `tokamax` and `auto`:

- Its keys come in parts (CSA's window, then its latents) that share one softmax,
  merged by their logsumexp in a custom VJP.
- On a GPU, tokamax's flash kernel runs each part with the kept keys as a boolean
  mask: dense compute, exact, and t² bits of mask forward and bytes backward.
- Under `xla`, off a GPU, or in `bux.kernels.off()`, XLA runs it a chunk of queries
  at a time.
- Either way it is a `shard_map` that gives a device its rows, queries and heads, and
  every key: all-gather CP.

The indexer's loss takes the attention's probabilities once, from its logsumexp, a
chunk of queries at a time (on a GPU by a kernel for heads of up to 128 dims). It
scores each query's picks by Pallas kernels in a `shard_map` that gives a device its
rows and queries and every key: a program a query, which stops at its last pick, and a
backward pass that adds the keys' gradient atomically.

### The flash kernels: `bux.parallel.flash`

`bux.parallel.flash` has the kernels. A `Kernel` attends to a block of keys:

- its forward pass returns each query's logsumexp beside its output;
- its backward pass takes one. Given the output and logsumexp of a larger softmax
  (over more blocks, or a sink beside the keys), one block's backward pass is that
  softmax's gradient through the block, as FlashAttention-2's recomputes
  `P = exp(s - lse)`.

So `ring` merges blocks by their logsumexps (`flash.merge`), and a sink folds into
one, `y = o exp(lse - L)` with `L = logaddexp(lse, sink)`, the backward pass run with
`L` (`flash.attend`).

| Kernel | What it is |
|---|---|
| `flash.TOKAMAX` | tokamax's kernels (its VJP, given the residuals `(lse, 1)`) |
| `flash.JNP` | the same in jnp, all t by s logits at once, which a CPU runs: with `flash.TOKAMAX = flash.JNP` the tests check `tokamax`'s and `auto`'s paths too |
| `flash.CUDNN` | cuDNN's causal kernels, for whole rows with no first keys |

A new kernel is a `flash.Kernel(forward, backward)`.

### cuDNN's kernels

cuDNN's attention in jax returns its logsumexp (`return_residual`), and its backward's
primitive takes one. So `flash.CUDNN` is a `Kernel` of cuDNN's causal kernels, for
whole rows with no first keys, which `flash.attend` folds sinks into.

The rows of a run without documents have ids too, all one document. So `auto` and
`cudnn` ask a `lax.cond` whether each row is one document, and attend by
`flash.CUDNN` where it is.

:::{note}
Under XLA's deterministic ops, XLA builds no graph for cuDNN's attention backward on a
B300 ("No serialized cuDNN graph"), so there `flash.CUDNN`'s backward is tokamax's.
:::

### Rows of documents

For rows of documents, cuDNN runs whole rows only, in its packed layout:

- The layout has room for a static number of documents a row. Its backward pass takes
  time in proportion to that room, and 8 bytes for each position of each head's
  channels of each document's room.
- `context.documents_bound` makes room for up to t / 256 documents, and up to 2^30
  such numbers: 16 for 8 rows of 4096, or for one of 32k.
- A `lax.cond` gives a batch with a row of more documents to tokamax.

In rows of 4096 that `lax.cond` costs a 1.3B what cuDNN gains it: 145k tok/s on a B300
either way. So `auto` takes cuDNN for documents from rows of 8192 on. In rows of 16384
of packed documents it trains at 1.3x tokamax's speed.

A step takes the documents' starts from their ids, and each layer from those:

- in rows of up to 8192 positions, by XLA's `cummax` and `cumsum`;
- in longer rows, by a prefix scan (`jax.lax.associative_scan`, log2(t) steps).

At 128k over `context`, XLA's `cummax` and `cumsum` take 2 and 25 ms each time. In rows
of 4096 the scan took a `1f1b` pipeline of the 1.3B 1.6% slower.

### Kernels on each device

Every kernel runs in a `shard_map` manual over every axis (`context.on_devices`), each
device on its rows and heads, because:

- XLA cannot split a kernel's call;
- an all-to-all in a `shard_map` of Auto axes makes XLA gather every dimension they
  split;
- cuDNN's partitioner fails beside manual axes.

Kernels need `check_vma=False`, under which the `shard_map` sums the sinks' gradient
over the devices that share them. Inside a `shard_map`, `shard` and `spec` leave out
the axes it maps, so that a layer's constraints still hold there.

## Writing your own method

Another method is a function, registered by name:

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

`methods = methods(attention=mine)` then picks it. A method that lacks a feature of
the `AttentionArgs` raises `NotImplementedError` naming it, as above.

## Measurements

### Kernels on a B300

A forward and backward pass on one B300, in bfloat16, 16 heads and 8 kv heads of 128
dims, in ms:

| Kernel | causal, 8 rows of 4096 | causal, 32k | 13 documents, 32k | window 4096, 32k |
|---|---|---|---|---|
| `cudnn` | 2.82 | 12.2 | 3.3 | 5.0 |
| `tokamax` | 3.08 | 16.0 | 14.1 | 14.5 |

- cuDNN skips what no query sees, forward and backward.
- tokamax's backward pass skips only what is ahead of the queries, which a window as
  each query's first key keeps. Its own window mask does every block: 29.2 ms.
- cuDNN's room for documents costs: the row of 13 takes 3.3 ms with room for 16, 3.9
  for 32, 5.0 for 64 and 7.4 for 128.

### cuDNN against tokamax in training

On 8 B300s, megatron.bux's `sorted` arm (`remat=none`, in a pool of 97% of a GPU)
trains at 34.2k tokens a second a GPU by `flash.CUDNN`, and at 31.3-31.5k by tokamax,
in 251 GiB either way. Cut to 12 blocks, it trains at:

| Attention | tok/s |
|---|---|
| `flash.CUDNN` | 925k |
| tokamax | 868k |
| cuDNN's forward, tokamax's backward | 896k |
| jax's call of cuDNN in its packed layout | 885k, in 8 GiB more: room for 8 documents a row |

A layer's attention (32 heads of 4 kv heads) takes 0.67 ms forward and 2.2 backward by
cuDNN, 1.05 and 4.5 by tokamax, whose backward is two kernels that each recompute the
probabilities.

The 1.3B (16 heads of 8 kv heads) trains at 155k tokens a second on a B300 by
`flash.CUDNN` and at 148k by tokamax, in the same 51 GiB. On FineWeb-Edu it trains at
149k and 140k, to the same held-out loss after 3000 steps: 4.041 and 4.043, where
seeds differ by 1.2%.

### Context parallelism on 8 B300s

`context=8`, one row, 6 calls in a jit to spread the ~2.5 ms a call on 8 GPUs takes; a
forward and backward pass, in ms:

| t, documents | one GPU's tokamax / 8 | `ulysses` | `ring` | `tokamax` |
|---|---|---|---|---|
| 32k, one | 2.06 | **2.39** | 5.24 | 5.08 |
| 128k, one | 31.9 | **26.0** | 38.3 | 68.9 |
| 32k, of 2048 | 1.71 | **1.89** | 2.29 | 4.48 |
| 128k, of 2048 | 26.0 | 27.6 | **8.06** | 57.4 |

- `ulysses` runs cuDNN, and tokamax past cuDNN's room for documents, whose backward
  pass does every block. In a row of 128k that room is 64 documents for a device's
  one head of 128, and 32 for two (`context.documents_bound`).
- `ring` skips most blocks of packed rows on every device.
- The `ring` this one replaced sent the blocks round by `ppermute` (~85 GB/s on a B300
  node, where an all-gather moves ~550) into an online softmax of their logits: 33.4
  and 440 ms causal.

### In a model, rows of 32k

On 8 B300s: 8 layers of `d_model = 1024`, 16 heads and 8 kv heads of 128 dims (218M),
rows of 32k over `mesh(context=8)`, `remat=full`, 8 microbatches, bfloat16, in tokens a
second:

| `methods.attention` | one document a row | packed documents (46 a row) |
|---|---|---|
| `ulysses`, `auto` | **252k** | **273k** |
| `ring` | 175k | 221k |
| `tokamax` | 200k | 211k |

- With sinks, `ulysses` trains at 243k (by tokamax's kernel), where `auto` took `xla`
  at 66k before; `ulysses` ran `xla` at 96k, and the `ring` it replaced at 75k.
- In a model `ulysses` is the faster even with 45% of the batches past cuDNN's room
  for documents, which tokamax then takes, so `auto` takes it. `ring` stays the method
  for longer packed rows, as at 128k above.
- That `ring` attends to its own block while the others' are gathered changes nothing
  past the noise in this model: 223k on packed documents against 225k gathering it
  first. A layer's gather is short against its attention.

### In a model, rows of 128k

The baseline's dense 1.3B (16 layers of `d_model = 2048`, 16 heads and 8 kv heads of
128) at 128k tokens a row over `mesh(context=8)`, a row a step, `remat=full`,
bfloat16. Packed documents of 64 to 4096 tokens (log-uniform), or one document a row.
Tokens a second, the log's MFU of the keys each query sees, and the busiest GPU's peak
memory:

| `methods.attention` | packed documents | one document a row |
|---|---|---|
| `ring` | **346k**, 12.5%, 22 GiB | 91k, 16.1%, 22 GiB |
| `ulysses`, `auto` | 227k, 8.2%, 25 GiB | **197k**, 34.9%, 25 GiB |
| `tokamax` | 123k, 4.4%, 21 GiB | |

They took 144k, 136k and 85k on packed documents before, and 68k and 125k on one
document, for two reasons:

- a step took the documents' ids, and each layer their starts, by XLA's `cumsum` and
  `cummax` of the row, which at 128k over `context` take 25 and 2 ms each time, where
  a prefix scan (`jax.lax.associative_scan`) takes log2(t) steps;
- `ring`'s backward pass gathered every block again, and reduce-scattered their
  gradients in float32.

On long packed rows `ring` is 1.5x `ulysses`, whose kernel past cuDNN's room for
documents (32 a row of 128k on a device's 2 heads) is tokamax's, which does every
block backward. `auto` cannot tell packed rows from rows of one document before a batch
comes, so give such a run `methods(attention=ring)`.

### On 8 CPU devices

Two windowed GQA layers (4 heads, 2 kv heads) with sinks and a soft cap train on
packed rows as on one device, to 1e-5 over 3 steps, by every method on
`context=2, tensor=2` and `context=4, fsdp=2`, and none makes XLA rematerialize an
array in full. On `context=8` too, every method's attention and its gradients are
`xla`'s (not `ulysses`', with 4 heads).

The collectives of the compiled step of two windowed layers: all-gathers / all-reduces
/ collective-permutes / all-to-alls, a collective in a loop counted once:

| Mesh | `xla` | `ring` | `ulysses` |
|---|---|---|---|
| `context=2, tensor=2` | 10 / 20 / 4 / 4 | 8 / 20 / 5 / 8 | 3 / 20 / 4 / 20 |
| `context=4, fsdp=2` | 31 / 8 / 4 / 4 | 29 / 8 / 5 / 8 | 24 / 8 / 4 / 20 |
| `context=8` | 10 / 4 / 4 / 4 | 8 / 4 / 5 / 8 | |

- `xla` and `ring` gather each layer's keys and values whole; `ring` once forward and
  once backward, where it reduce-scatters their gradients. With documents nothing goes
  round.
- `ulysses` gathers no activation but the documents' ids, and adds 8 all-to-alls a
  layer.
- The rest are FSDP's parameters, and int32 and boolean arrays of the documents'
  starts.

The temp memory of the gradient of one attention at t = 2048 on `context=4`:

| Method | Temp memory | Why |
|---|---|---|
| `xla` | 68.5 MiB | |
| `ulysses` | 81.0 MiB | whole rows of a quarter of the heads |
| `ring` | 26.5 MiB | every block of keys, and no logits |

`scripts/gpu_smoke/attention.py` runs every method by its GPU kernels against `xla`.

## See also

- [Context parallelism for recurrent mixers](recurrence.md), `methods.recurrence`.
- [Remat](remat.md): the names `attention.in`, `attention.out` and `attention.index`
  keep what attention's backward reads.
- The attention layers themselves: [Attention](../architectures/attention.md).
- The kernels, from the performance side: [Kernels](../performance/kernels.md).
