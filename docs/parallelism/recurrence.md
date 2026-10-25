(parallelism-recurrence)=
# Context parallelism for recurrent mixers

The recurrent mixers (linear attention, the delta rules, RWKV-7, Mamba and Mamba-2)
carry a state from position to position. When `context` splits a row's positions over
devices, a method of `methods.recurrence` decides how each device gets the state its
positions start from:

- `scan`, the default, lets XLA gather the whole sequence onto every device first;
- `pass_state` has each device run its own positions and pass on only what its chunk
  does to the state, as LASP-2 does, so no device holds a whole row.

Use `pass_state` for long rows of a hybrid model over `context`, beside
[ring or ulysses attention](attention.md) for its attention layers. On one device, or
where nothing splits the positions, the two are the same.

This page also covers how a recurrence runs on each device: by chunks of positions
(`chunkwise`), what its backward pass keeps of them (`methods.states`), and the GPU
kernels of the delta rule.

## Turn it on

Split the positions over `context`, and pass the state. From
`examples/experiments/parallel/long_context.bux`, a Qwen3-Next-shaped hybrid of Gated
DeltaNet and attention on 64k-token rows over 8 GPUs:

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

With tensor parallelism too, each device runs the recurrence of its own heads:

```bux
mesh    = mesh(context=4, tensor=2)
methods = methods(recurrence=pass_state)
```

## Options

Two fields of {bux}`methods`:

| Setting | Values | Default | What it does |
|---|---|---|---|
| `methods.recurrence` | `scan`, `pass_state`, or your own | `scan` | how a recurrence reaches the state its positions start from |
| `methods.states` | `keep`, `recompute` | `keep` | what the backward pass of a recurrence by chunks reads of its chunks' states: kept from the forward pass, or computed again from its inputs |

The methods of `methods.recurrence`:

| Method | What it does |
|---|---|
| `scan` | The default: every position as one chunk, which `chunkwise` runs by chunks of 64 positions (or cuDNN's kernels, [below](#cudnns-delta-rule-kernels)). With `seq` over `context`, XLA gathers the sequence onto every device first. |
| `pass_state` | Each device scans its own positions from a zero state and keeps their transition. The transitions fold, in turn, into the state each device's chunk starts from (LASP-2: XLA all-gathers them over `context`). Each device then adds to its outputs what they read of that state. |

The options of `methods.states`:

| Value | What the backward pass reads | Cost |
|---|---|---|
| `keep` | the chunks' float32 states and maps, kept from the forward pass | the memory of every chunk's state |
| `recompute` | the same, computed again from the inputs as they come, as fla's kernels do; it keeps only each chunk's `(I - L)⁻¹` | a fourteenth of the memory for the delta rule (a sixteenth for Mamba-2), and about a fifth more time |

:::{tip}
`recompute` buys rows: the Mamba-2 hybrid fits twice the rows a GPU under it, at 4
to 9% fewer tokens a second ([Measurements](#the-chunks-states-keep-or-recompute)).
The delta rule by cuDNN's kernels keeps only its inputs either way.
:::

## Constraints and errors

- A method that cannot run a family raises `NotImplementedError` naming it. Under
  `pass_state`, a family whose chunk gives no transition raises so.
- Where nothing splits the positions, `pass_state` is `scan`.
- The delta rule's GPU kernels (cuDNN's and bux's) run only under `scan`. Under
  `pass_state`, and wherever a mesh splits the positions, the chunks run instead.
- Under `precision(compute=bfloat16)`, `pass_state` trains on a CPU too: what the
  `shard_map`s' transposes sum is float32 ([Precision](precision.md)).

## How it works

### A chunk's transition

Every recurrent mixer keeps a state `S` of `k × v` per head, which each position
decays, adds to and reads its outputs from, linearly. So a chunk of positions maps the
state it starts from to the one it ends with by an affine map, its **transition**

$$S \mapsto A S + B,$$

and each of its outputs gains `reads_t · S`. `A` is zero across a document start.

| Family | `A`, a head | `reads_t` |
|---|---|---|
| DeltaNet, Gated DeltaNet, KDA, RWKV-7 | a `k × k` matrix | `P_tᵀ q_t` |
| GLA | a diagonal of `k` | `P_t q_t` |
| Mamba | a diagonal of `n` | `P_t C_t` |
| Lightning, Mamba-2 | a scalar | `P_t q_t`, `P_t C_t` |

`P_t` is the product of the decays (`diag(w) + b aᵀ`, or `exp(dt A)`) from the chunk's
start to position `t`, carried by the chunk's scan beside the state. What that costs:

- For a matrix (and k = v) it doubles the scan's work and what its backward saves.
- A diagonal or a scalar costs a vector.
- The outputs' `reads` are `(b, t, h, k)`, which for Mamba (one head per channel) is
  the size of the states its backward saves.

### `pass_state`: two `shard_map`s

`pass_state` is two `shard_map`s manual over every axis that splits the tokens, and
neither holds a collective:

1. the first runs each device's chunk, and returns its outputs, state, transition and
   reads;
2. the fold of the transitions, in turn, into each chunk's starting state runs between
   them, in the global view, where XLA all-gathers them over `context`;
3. the second corrects each device's outputs by what they read of that state.

Inside a `shard_map` of Auto axes, a collective makes XLA gather what they split: an
all-gather of the transitions there gathered the heads over `tensor` too, so that every
device folded all of them. Kept in the global view, the fold leaves `tensor` Auto, and
the heads stay split.

(recurrence-chunkwise)=
### Chunks: `chunkwise`

A chunk runs by `chunkwise` (`bux.architecture.recurrent.chunkwise`), the form of
fla's chunked kernels, under every method:

- matmuls within chunks of 64 positions, and a scan of the state across them, by a
  Pallas kernel on a GPU (`affine_scan`);
- `stepwise`, a scan over positions, is its reference;
- with one decay a head, the state passes across the chunks by a matmul (SSD's state
  passing), and q and k may be shared by groups of heads.

Mamba-2's selective scan runs so, as decayed linear attention with `q = C` and `k = B`
of its groups, `v = dt x` and a log decay `dt A` a head; `ssm.stepwise` is its
reference. Mamba's, with a decay per channel and state dim, runs by the scan over
positions.

`recurrence` (and so `delta_rule` and RWKV-7) and `selective_scan` run by the run's
method of `methods.recurrence`; `scan` is in `linear_attention.py`.

(recurrence-states)=
### The chunks' states: `methods.states`

What the backward pass of `recurrence` (and so `delta_rule` and RWKV-7) and of the
selective scan reads of its chunks' float32 states and maps is `methods.states`. One
helper applies it to both (`linear_attention.recomputed`):

- **`keep`**, the default, keeps them from the forward pass.
- **`recompute`** computes them again, as fla's kernels do, from the inputs as they
  come (q, k and v in bfloat16 under `precision(compute=bfloat16)`) and each chunk's
  `(I - L)⁻¹`, which it keeps.

That inverse's own backward, `Tᵀ dT Tᵀ` of `T` alone, also makes `keep` faster than
before.

Mamba-2 has no inverse, so `recompute` keeps only the scan's inputs, x, dt, A, B and
C: a sixteenth of what `keep` keeps. Its gradients are those of `keep`, bit for bit on
a CPU.

(recurrence-cudnn-kernels)=
### cuDNN's delta rule kernels

Under `scan`, on a B200 or B300, with q, k and v of 16 bits and heads of 64 or 128,
the delta rule runs by cuDNN's fused kernels instead (`fused_delta`, Frost in
cudnn-frontend). They keep only their inputs for the backward pass, and so do the
chunks they fall back on.

| Decay | Kernel | Runs |
|---|---|---|
| one a head (GDN, DeltaNet) | Frost's GDN kernel | any gate: -20 a position, or a single -10⁴. It takes each pair's decay as the exp of a difference of summed log decays |
| one a channel (KDA) | Frost's KDA kernel | where `gate_lower_bound` bounds the gate at -5 or above |

- cudnn-frontend gives the GDN kernel no JAX call, so a `buffer_callback` runs cuDNN's
  graph of it on XLA's stream, one device at a time.
- The KDA kernel factors each decay through the start of a tile of 16 positions. So it
  loses precision where a tile decays a channel by exp(-86), and overflows at
  exp(-90).

:::{warning}
Unbounded KDA gates are that steep early: within 12 steps of bux's start, they reach
it in 30 to 90% of the (row, head) pairs of most layers. So cuDNN's KDA kernel runs
only gates bounded at -5 or above (`kda(gate_lower_bound=-5)`, or a bound nearer 0).
Unbounded gates, and those bounded below -5, run by bux's own kernels, below.
:::

(recurrence-kda-kernels)=
### bux's kernels for KDA

Unbounded, or bounded below -5, on any GPU, KDA runs by bux's own Pallas kernels
(`pallas_delta`), exact at any gate, under `scan`, with q, k and v of 16 bits and heads
of 64 or 128:

- They take each pair's decay `exp(G_t - G_s)` through a position `r` between the two,
  `exp(G_t - G_r) exp(G_r - G_s)`, both factors at most 1, as a matmul.
- For blocks of 16 positions `i > j` of a chunk of 64, `r` is the end of block `j`.
  Within a block, it is the end of the first half of the 8, 4, 2 or 1 positions the
  pair spans.
- The state passes from chunk to chunk in the WY form, and documents start it again,
  so a row holds any number of them.
- The backward pass takes the gradients by hand, a few blocks at a time. With
  `methods(states=keep)`, the default, it keeps each chunk's parts and the state before
  it from the forward pass; with `recompute`, only the inputs, running the forward
  kernels again.

On a B300, at 2 × 4096 tokens in packed documents, with random gates down to -20 a
position, spikes of -10⁴ and KDA's own gate as it starts, the outputs and every
gradient are within 4·10⁻³ of the scan's over positions (as a share of each one's
norm), where the chunks in float32 are within 1.3·10⁻³: the kernels keep the parts
and states in bfloat16.

`scripts/gpu_smoke/recurrence.py` checks, on a GPU:

- the chunked forms, and Mamba-2's recomputed, against their scans, whose GPU fusions
  the CPU tests do not see: on a B300 one took Mamba-2's gradients for B and C wrong,
  which a plain matmul in its place does not;
- the kernels against the chunks.

## Writing your own method

A method of `methods.recurrence` is a function `(recurrence) -> out`, the outputs
`(b, t, h, v)` in float32. A `Recurrence` (`bux.parallel.recurrence`) names its
`family` and holds:

| Field | What it is |
|---|---|
| `chunk` | runs the recurrence over a chunk of positions from a zero state |
| `xs` | the `(b, t, ...)` arrays of positions |
| `shared` | what it takes whole, as Mamba's `A` |

`chunk(xs, shared, transition)` gives a `Chunk`: its outputs, the state it ends with
and, with `transition`, `A` and the reads. A method that cannot run a family raises
`NotImplementedError` naming it:

```python
from bux.parallel import methods
from bux.parallel.recurrence import Recurrence


@methods.recurrence.register("mine")
def _(r: Recurrence):
    if r.family == "the selective scan":
        raise NotImplementedError("methods.recurrence = mine has no selective scan")
    return r.chunk(r.xs, r.shared, False).out
```

`methods = methods(recurrence=mine)` then picks it.

## Measurements

### On 8 CPU devices

Two blocks of each family under `pass_state`, on packed rows whose documents cross the
devices' chunks:

- the logits and gradients are one device's on `context=2 × tensor=2` and
  `context=4 × fsdp=2`, to 1e-5 of each gradient's largest entry, as `scan` on the
  same mesh is;
- a run trains as on one device to 2.4e-6 over 3 steps on those meshes and on
  `context=8`;
- nothing is rematerialized in full, by either method.

The all-gathers of the compiled train step (`seq_len = 24`, 8 rows, width 32),
`scan` → `pass_state`. Of `scan`'s, 12 to 20 gather a float32 array of every
position, and none of `pass_state`'s does:

| Family | `context=2, tensor=2` | `context=4, fsdp=2` | `context=8` |
|---|---|---|---|
| DeltaNet | 24 → 16 | 58 → 50 | 21 → 13 |
| GDN | 26 → 16 | 68 → 58 | 23 → 13 |
| KDA | 26 → 16 | 72 → 62 | 23 → 13 |
| GLA | 18 → 12 | 56 → 50 | 17 → 11 |
| Lightning | 20 → 14 | 48 → 42 | 15 → 9 |
| RWKV-7 | 32 → 22 | 91 → 81 | 23 → 13 |
| Mamba | 20 → 14 | 84 → 76 | 17 → 11 |
| Mamba-2 | 36 → 30 | 72 → 64 | 17 → 11 |

`pass_state` all-gathers 4 small arrays per layer: the states and transitions forward,
and again for the backward. The other collectives do not change, but for Mamba's and
Mamba-2's `A`, whose gradient is summed over the token axes (one all-reduce per layer
on `context=8`).

The temp memory of the train step at `seq_len = 1024` (8 rows, width 64),
`scan` → `pass_state`, in MiB:

| Family | `context=4, fsdp=2` | `context=8` |
|---|---|---|
| GDN | 26.7 → 21.2 | 41.3 → 21.3 |
| GLA | 18.8 → 11.3 | 29.3 → 11.2 |
| Lightning | 11.3 → 9.0 | 14.6 → 8.9 |
| RWKV-7 | 26.9 → 20.2 | 41.8 → 23.1 |
| Mamba | 136.2 → 70.6 | 248.1 → 70.8 |
| Mamba-2 | 73.1 → 40.9 | 119.4 → 40.8 |

Both meshes give a device 1024 tokens: `pass_state` holds those, `scan` every position
of its rows. On `context=2 × tensor=2` a matrix family's carried `P` costs about what
the halved sequence saves.

### The delta rule on a B300

The delta rule's forward and backward pass at 8 × 4096 tokens of 16 heads of 128, on a
B300, and what it keeps:

| Delta rule | GDN | KDA |
|---|---|---|
| by chunks, before | 13.1 ms, 7.24 GiB | 29.6 ms, 7.99 GiB |
| `states=keep` | 11.6 ms, 6.51 GiB | 28.9 ms, 7.26 GiB |
| `states=recompute` | 14.9 ms, 0.50 GiB | 33.6 ms, 0.75 GiB |
| by its kernel (KDA bounded at -5) | 2.5 ms | 2.9 ms |
| by bux's kernels, any gate, `keep` | | 11.1 ms, 1.69 GiB |
| by bux's kernels, any gate, `recompute` | | 14.2 ms, 0.63 GiB |

("By chunks, before" is the chunks before the inverse took its own backward,
`Tᵀ dT Tᵀ` of `T` alone, which made `keep` faster.)

(recurrence-keep-or-recompute)=
### The chunks' states: keep or recompute

A GLA hybrid (`"GGGA"^4`), which runs by chunks alone, on a B300:

| `methods.states` | tok/s, memory | the rows it fits |
|---|---|---|
| `keep` | 84.4k, 83.5 GiB | 16: 86.2k, 149.4 GiB |
| `recompute` | 77.1k, 66.8 GiB | 32: 78.8k, 192.2 GiB |

The one-GPU suite's Mamba-2 hybrid (`"MMMA"^4`), rows of 4096 on a B300, in JAX's
default share of the GPU's memory:

| Rows | `states=keep` | `states=recompute` |
|---|---|---|
| 8 | 94.8k tok/s, 122.6 GiB | 90.7k, 75.5 GiB |
| 12 | 100.8k, 172.5 GiB | 91.9k, 101.4 GiB |
| 16 | 95.8k, 190.9 GiB | 91.7k, 128.1 GiB |
| 20 | out of memory | |
| 24 | | 92.6k, 182.5 GiB |
| 32 | | 92.8k, 193.5 GiB |
| 40 | | out of memory |

So `recompute` fits twice the rows, at 4 to 9% fewer tokens a second, its losses those
of `keep` to within 10⁻³.

### GDN's kernel on many GPUs

GDN's kernel runs on each GPU of a mesh as on one, on a node or two. Where it does not
apply, as under `pass_state`, the chunks run instead. The GDN hybrid (`"DDDA"^4`) at 8
rows of 4096 a GPU, on B300s, in tokens a second and the busiest GPU's peak:

| GDN hybrid | by its kernel | by the chunks (`pass_state`) |
|---|---|---|
| one GPU | 127.7k, 70.6 GiB | 95.4k, 112.7 GiB |
| a node, `fsdp=8` | 815k, 58.4 GiB | 725k, 99.6 GiB |
| two nodes, `fsdp=16` | 1.58M, 58.0 GiB | |
| two nodes, `data=2, fsdp=8` | 1.60M, 58.4 GiB | 1.42M, 99.6 GiB |
| two nodes, `data=2, fsdp=4, context=2` | | 1.03M, 124.4 GiB |

- Two nodes train at 97% (FSDP over 16) and 98% (HSDP) of a node's tokens a second a
  GPU.
- The kernel's losses are the chunks' to 0.002 on a node and 0.005 on two at the first
  log, and to 0.0002 at the last of 30 steps.

## Not yet

A `ppermute` chain instead of the all-gather of `pass_state`, which holds every
device's transitions (`c` matrices `k × k` per head) on each.

## See also

- [Attention and context parallelism](attention.md), `methods.attention`.
- [Remat](remat.md): `recurrence.in` names a mixer's inputs as its kernel takes them,
  `recurrence.out` its output and the chunks' states.
- The recurrent layers themselves: [Recurrent mixers](../architectures/recurrent.md).
- The kernels, from the performance side: [Kernels](../performance/kernels.md).
