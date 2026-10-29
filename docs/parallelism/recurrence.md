(parallelism-recurrence)=
# Context parallelism for recurrent mixers

This page covers the recurrent mixers (linear attention, the delta rules, RWKV-7, Mamba
and Mamba-2) on many devices, and the kernels they run on each. These mixers carry a
state from position to position. When `context` splits a row's positions over
devices, a method of `methods.recurrence` decides how each device gets the state its
positions start from:

- `scan`, the default, lets XLA gather the whole sequence onto every device first;
- `pass_state` has each device run its own positions and pass on only what its chunk
  does to the state, as LASP-2 does, so no device holds a whole row.

Which to choose:

- **On one device, or with no `context` axis**, set nothing. `scan` runs the GPU
  kernels wherever they apply, and `pass_state` is the same as `scan`.
- **For long rows of a hybrid model over `context`**, set
  `methods(recurrence=pass_state)`, beside [`ring` or `ulysses`](attention.md) for its
  attention layers. A device then holds only its own positions.
- **When a recurrence by chunks runs out of memory**, set `methods(states=recompute)`.
  The Mamba-2 hybrid fits twice the rows under it, at 4 to 9% fewer tokens/s.

The numbers behind these are in [Measurements](#measurements).

## Quick start

Split the positions over `context`, and pass the state:

```bux
mesh    = mesh(context=8)
methods = methods(attention=ring, recurrence=pass_state)
```

`examples/experiments/parallel/long_context.bux` trains a Qwen3-Next-shaped hybrid of
Gated DeltaNet and attention this way, on 64k-token rows over 8 GPUs; the whole file is
on [Attention and context parallelism](attention.md#quick-start).

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
| `methods.states` | `keep`, `recompute` | `keep` | what the backward of a recurrence by chunks reads of its chunks' states: kept from the forward, or computed again from its inputs |

The methods of `methods.recurrence`:

| Method | What it does |
|---|---|
| `scan` | The default: every position as one chunk, which `chunkwise` runs in chunks of 64 positions, or a GPU kernel runs whole ([GPU kernels](#gpu-kernels)). With `seq` over `context`, XLA gathers the sequence onto every device first. |
| `pass_state` | Each device scans its own positions from a zero state and keeps their transition. The transitions fold, in turn, into the state each device's chunk starts from (LASP-2: XLA all-gathers them over `context`). Each device then adds to its outputs what they read of that state. |

The values of `methods.states`:

| Value | What the backward reads | Cost |
|---|---|---|
| `keep` | the chunks' float32 states and maps, kept from the forward | the memory of every chunk's state |
| `recompute` | the same, computed again from the inputs, as fla's kernels do; it keeps only each chunk's `(I - L)⁻¹` | a fourteenth of the memory for the delta rule (a sixteenth for Mamba-2), and about a fifth more time |

:::{tip}
`recompute` buys rows: the Mamba-2 hybrid fits twice the rows per GPU under it, at 4
to 9% fewer tokens/s ([Measurements](#recurrence-keep-or-recompute)). cuDNN's kernels
keep only their inputs either way.
:::

## Constraints and errors

- A method that cannot run a family raises `NotImplementedError` naming it. Under
  `pass_state`, so does a family whose chunk gives no transition.
- Where nothing splits the positions, `pass_state` is `scan`.
- The GPU kernels run only under `scan`. Under `pass_state`, and wherever a mesh
  splits the positions, the chunks run instead, which is slower: the GDN hybrid trains
  at 725k tokens/s on a node by the chunks against 815k by its kernel.
- Under `precision(compute=bfloat16)`, `pass_state` trains on a CPU too: what the
  `shard_map`s' transposes sum is float32 ([Precision](precision.md)).

## How it works

### A chunk's transition

Every recurrent mixer keeps a state `S` of `k × v` per head, which each position
decays, adds to and reads its outputs from, linearly. So a chunk of positions maps the
state it starts from to the one it ends with by an affine map, its **transition**

$$S \mapsto A S + B,$$

and each of its outputs gains `reads_t · S`. `A` is zero across a document start.

| Family | `A`, per head | `reads_t` |
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
2. between them, in the global view, the transitions fold in turn into each chunk's
   starting state, where XLA all-gathers them over `context`;
3. the second corrects each device's outputs by what they read of that state.

The fold stays in the global view on purpose. Inside a `shard_map` of Auto axes, a
collective makes XLA gather what they split: an all-gather of the transitions there
gathered the heads over `tensor` too, so every device folded all of them. In the
global view `tensor` stays Auto, and the heads stay split.

(recurrence-chunkwise)=
### Chunks: `chunkwise`

A chunk runs by `chunkwise` (`bux.architecture.recurrent.chunkwise`), the form of
fla's chunked kernels, under every method:

- matmuls within chunks of 64 positions, and a scan of the state across them, by a
  Pallas kernel on a GPU (`affine_scan`);
- `stepwise`, a scan over positions, is its reference;
- with one decay per head, the state passes across the chunks by a matmul (SSD's
  state passing), and q and k may be shared by groups of heads;
- a decay per key channel takes each pair's decay through the end of a block of 16
  positions between them, so that no exponent is positive. One bounded near 0
  instead, as RWKV-7's (at least $-e^{-1/2}$ per position: `recurrence`'s `floor`),
  takes it through the chunk's middle position, one matmul per pair product, its
  exponents at most 20.

Mamba-2's selective scan runs this way, as decayed linear attention with `q = C` and
`k = B` of its groups, `v = dt x` and a log decay `dt A` per head; `ssm.stepwise` is
its reference. Mamba's, with a decay per channel and state dim, runs by the scan over
positions.

`recurrence` (and so `delta_rule` and RWKV-7) and `selective_scan` run by the run's
method of `methods.recurrence`; `scan` is in `linear_attention.py`.

(recurrence-states)=
### The chunks' states: `methods.states`

`methods.states` says what the backward of `recurrence` (and so `delta_rule` and
RWKV-7) and of the selective scan reads of its chunks' float32 states and maps. One
helper applies it to both (`linear_attention.recomputed`):

- **`keep`**, the default, keeps them from the forward.
- **`recompute`** computes them again, as fla's kernels do, from the inputs as they
  come (q, k and v in bfloat16 under `precision(compute=bfloat16)`) and each chunk's
  `(I - L)⁻¹`, which it keeps. That inverse has its own backward, `Tᵀ dT Tᵀ` of `T`
  alone, which speeds up `keep` too.

Mamba-2 has no inverse, so `recompute` keeps only the scan's inputs, x, dt, A, B and
C: a sixteenth of what `keep` keeps. Its gradients are those of `keep`, bit for bit on
a CPU.

## GPU kernels

Under `scan`, with the positions not split, q, k and v of 16 bits and heads of 64 or
128, the delta rule and decayed linear attention run by fused kernels instead of the
chunks:

| Family | Kernel | Where |
|---|---|---|
| Gated DeltaNet, DeltaNet | cuDNN's GDN kernel ([below](#recurrence-cudnn-kernels)) | a B200 or B300, any gate |
| KDA with `gate_lower_bound` of -5 or above | cuDNN's KDA kernel ([below](#recurrence-cudnn-kernels)) | a B200 or B300 |
| KDA otherwise | bux's Pallas kernels ([below](#recurrence-kda-kernels)) | any GPU |
| GLA, Lightning | cuDNN's GDN-2 kernel ([below](#recurrence-gla-kernel)) | a B200 or B300, bfloat16 |
| Gated DeltaNet-2 with `gate_lower_bound` of -5 or above | cuDNN's GDN-2 kernel ([below](#recurrence-gdn2-kernel)) | a B200 or B300, bfloat16 |

Every family not in the table, and every case its conditions exclude, runs by the
chunks. cuDNN's kernels keep only their inputs for the backward, and so do the chunks
they fall back on in a row of more documents than their room. bux's KDA kernels follow
`methods.states`. `scripts/gpu_smoke/recurrence.py` checks every kernel against the
chunks on a GPU (steep gates, spikes of -10⁴, heads of 64, rows of more documents than
a kernel's room), and the chunked forms against their scans, whose GPU fusions the CPU
tests do not see.

(recurrence-cudnn-kernels)=
### cuDNN's delta rule kernels

On a B200 or B300, the delta rule runs by cuDNN's fused kernels (`fused_delta`, Frost
in cudnn-frontend):

| Decay | Kernel | Runs |
|---|---|---|
| one per head (GDN, DeltaNet) | Frost's GDN kernel | any gate: -20 per position, or a single -10⁴. It takes each pair's decay as the exp of a difference of summed log decays |
| one per channel (KDA) | Frost's KDA kernel | where `gate_lower_bound` bounds the gate at -5 or above |

- cudnn-frontend gives the GDN kernel no JAX call, so a `buffer_callback` runs cuDNN's
  graph of it on XLA's stream, one device at a time.
- A graph compiles and loads its kernels the first time it runs on a device, which
  waits for the device's peers. Inside a step on several GPUs, a peer may be waiting
  at a collective that this device joins only after the callback returns, and the step
  hangs. So on a mesh of several devices, each graph first runs once on every device,
  as the step that runs it traces, while nothing else runs (`fused_delta.warm`): on a
  B300, up to 4 s for a forward graph and 7 s for a backward one, per device and
  shape (2026-10-29).
- The KDA kernel factors each decay through the start of a tile of 16 positions. So it
  loses precision where a tile decays a channel by exp(-86), and overflows at
  exp(-90).

:::{warning}
Unbounded KDA gates get that steep early: within 12 steps of the start, they reach it
in 30 to 90% of the (row, head) pairs of most layers. So cuDNN's KDA kernel runs only
gates bounded at -5 or above (`kda(gate_lower_bound=-5)`, or a bound nearer 0).
Unbounded gates, and those bounded below -5, run by bux's own kernels, below.
:::

(recurrence-kda-kernels)=
### bux's kernels for KDA

KDA's gates that are unbounded, or bounded below -5, run by bux's own Pallas kernels
(`pallas_delta`) on any GPU, exact at any gate:

- They take each pair's decay `exp(G_t - G_s)` through a position `r` between the two,
  `exp(G_t - G_r) exp(G_r - G_s)`, both factors at most 1, as a matmul.
- For blocks of 16 positions `i > j` of a chunk of 64, `r` is the end of block `j`.
  Within a block, it is the end of the first half of the 8, 4, 2 or 1 positions the
  pair spans.
- The state passes from chunk to chunk in the WY form, and documents start it again,
  so a row holds any number of them.
- The backward takes the gradients by hand, a few blocks at a time. Under
  `methods(states=keep)` it keeps each chunk's parts and the state before it from the
  forward; under `recompute`, only the inputs, running the forward kernels again.

On a B300, at 2 × 4096 tokens in packed documents, with random gates down to -20 per
position, spikes of -10⁴ and KDA's own gate at initialization, the outputs and every
gradient are within 4·10⁻³ of the scan's over positions (as a share of each one's
norm). The chunks in float32 are within 1.3·10⁻³: the kernels keep the parts and
states in bfloat16.

cuDNN's GDN-2 kernel ([below](#recurrence-gla-kernel)) could run KDA too, exact at any
gate, but it reruns a whole call by a slower exact path wherever a tile of 16
positions decays a channel past $e^{-83}$, which KDA's gates soon do. The KDA hybrid
trained at 54.6k tokens/s by it over 200 steps, and at 99.3k by bux's kernels.

(recurrence-gla-kernel)=
### GLA's and Lightning's kernel

Decayed linear attention, with no delta rule, runs by cuDNN's GDN-2 kernel on a B200
or B300, with q, k and v in bfloat16 (`fused_delta.linear`, by
`linear_attention.decayed`):

- GDN-2's state follows
  $S_t = (I - k_t (\beta_t \odot k_t)^\top)\, \mathrm{diag}(e^{g_t})\, S_{t-1} + k_t (w_t \odot v_t)^\top$,
  with $g$, $\beta$ and $w$ per channel. At $\beta = 0$ and $w = 1$ it is GLA's
  recurrence, and Lightning's with its per-head decay given to every channel.
- cudnn-frontend has it from 1.32, which the `cuda` extra requires; a run without it
  runs by the chunks.
- In a row of more documents than its room, the chunks run instead.
- It factors each decay through a tile of 16 positions. A tile that decays a channel
  past $e^{-83}$ makes it rerun the whole call by an exact path: the same values, many
  times slower. In float16 a tile past $e^{-9.7}$ would, so it takes bfloat16 only.

(recurrence-gdn2-kernel)=
### Gated DeltaNet-2's kernel

Gated DeltaNet-2 (`gdn2`) runs by the same kernel, its erase gate as GDN-2's $\beta$
and its write gate as $w$ (`fused_delta.gated`, by `linear_attention.gated_delta_2`),
with q, k and v in bfloat16 and its decay bounded at -5 or above (`gate_lower_bound`):

- Bounded so, a tile of 16 positions decays a channel by $e^{-80}$ at most, short of
  the kernel's exact path.
- The paper's unbounded gate runs by the chunks. Trained on text, such gates soon
  decay some tile past $e^{-83}$, and the kernel then runs each call twice: on
  FineWeb-Edu it trained twice as fast as the chunks for 200 steps, and 16% slower
  from step 1000 ([Kernels](../performance/kernels.md#gated-deltanet-2s-kernel)).
- The gates go to the kernel in bfloat16, and come back that way.

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

On B300s, 2026-10-21 to 2026-10-26, except where a table says CPU.

### `pass_state` on 8 CPU devices

Two blocks of each family under `pass_state`, on packed rows whose documents cross the
devices' chunks (2026-10-21):

- the logits and gradients are one device's on `context=2 × tensor=2` and
  `context=4 × fsdp=2`, to 1e-5 of each gradient's largest entry, as `scan` on the
  same mesh is;
- a run trains as on one device to 2.4e-6 over 3 steps on those meshes and on
  `context=8`;
- nothing is rematerialized in full, by either method.

The temp memory of the train step at `seq_len = 1024` (8 rows, width 64),
`scan` → `pass_state`, in MiB. Both meshes give a device 1024 tokens: `pass_state`
holds those, `scan` every position of its rows.

| Family | `context=4, fsdp=2` | `context=8` |
|---|---|---|
| GDN | 26.7 → 21.2 | 41.3 → 21.3 |
| GLA | 18.8 → 11.3 | 29.3 → 11.2 |
| Lightning | 11.3 → 9.0 | 14.6 → 8.9 |
| RWKV-7 | 26.9 → 20.2 | 41.8 → 23.1 |
| Mamba | 136.2 → 70.6 | 248.1 → 70.8 |
| Mamba-2 | 73.1 → 40.9 | 119.4 → 40.8 |

On `context=2 × tensor=2`, a matrix family's carried `P` costs about what the halved
sequence saves.

::::{dropdown} The all-gathers of the compiled step
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

`pass_state` all-gathers 4 small arrays per layer: the states and transitions
forward, and again for the backward. The other collectives do not change, except for
Mamba's and Mamba-2's `A`, whose gradient is summed over the token axes (one
all-reduce per layer on `context=8`).
::::

### The delta rule on a B300

The delta rule's forward and backward at 8 × 4096 tokens of 16 heads of 128, on a
B300, and what it keeps (2026-10-23):

| Delta rule | GDN | KDA |
|---|---|---|
| by chunks, `states=keep` | 11.6 ms, 6.51 GiB | 28.9 ms, 7.26 GiB |
| by chunks, `states=recompute` | 14.9 ms, 0.50 GiB | 33.6 ms, 0.75 GiB |
| by its cuDNN kernel (KDA bounded at -5) | 2.5 ms | 2.9 ms |
| by bux's kernels, any gate, `keep` | | 11.1 ms, 1.69 GiB |
| by bux's kernels, any gate, `recompute` | | 14.2 ms, 0.63 GiB |

GLA and Lightning, at the same shapes, a decay per channel and per head (2026-10-26):

| Decayed linear attention | GLA | Lightning |
|---|---|---|
| by chunks (`states=keep`) | 17.4 ms | 5.5 ms |
| by GDN-2's kernel | 3.7 ms | 3.5 ms |
| by GDN-2's kernel, gates of -20 per position (its exact path) | 45 ms | |

Their hybrids train 53% and 5% faster by it
([Kernels](../performance/kernels.md#glas-and-lightnings-kernel)).

Gated DeltaNet-2 at the same shapes, its gates in bfloat16, against KDA (2026-10-26):

| Gated DeltaNet-2 | forward and backward |
|---|---|
| by chunks, `states=keep` | 29.8 ms (KDA: 28.6) |
| by chunks, `states=recompute` | 34.6 ms |
| by GDN-2's kernel, gates bounded at -5 | 3.6 ms (KDA by its kernel: 3.2) |
| by GDN-2's kernel, KDA's gate at initialization (its exact path) | 7.0 ms |
| by GDN-2's kernel, gates of -20 per position (its exact path) | 44.9 ms |

Its hybrid bounded at -5 trains 2.1x as fast by the kernel as by the chunks
([Kernels](../performance/kernels.md#gated-deltanet-2s-kernel)).

(recurrence-rwkv7-chunks)=
### RWKV-7's chunks

RWKV-7's recurrence at 8 × 4096 tokens on a B300, forward and backward (2026-10-26).
Its bounded decay lets the chunks take each pair's decay through the chunk's middle,
rather than through blocks of 16 positions as for other per-channel decays
([Chunks](#recurrence-chunkwise)):

| RWKV-7 | 32 heads of 64 | 16 heads of 128 |
|---|---|---|
| through blocks of 16 | 58.3 ms | 49.6 ms |
| through the chunk's middle | 15.5 ms | 16.8 ms |

The baseline's hybrid of it (`"RRRA"^4`, 32 heads of 64, 8 rows of 4096 on one B300)
trains at 87.4k tokens/s through the chunk's middle, against 45.0k, and peaks at
142.1 GiB against 155.3. On a node (`fsdp=8`) it trains at 690k against 354k, in
129.0 GiB against 157.0. Its outputs and gradients are the scan's over positions to
10⁻⁶ in float32, and on FineWeb-Edu (1500 steps of 8 × 4096 tokens, deterministic) its
held-out loss is 4.2598 against 4.2611.

fla's Triton kernel (`chunk_rwkv7` with its lower bound, in PyTorch) takes 6.7 ms on
the same GPU, so a kernel of bux's own could take RWKV-7 further. Of the chunks' 15 ms,
XLA's fusions over their float32 pairs and maps (64 × 64 per chunk and head) take 7.5,
and TF32 GEMMs of 64 × 64 take 6.

(recurrence-keep-or-recompute)=
### The chunks' states: keep or recompute

A GLA hybrid (`"GGGA"^4`) on a B300, by chunks alone (2026-10-23), as it runs without
[its kernel](#recurrence-gla-kernel):

| `methods.states` | tok/s, memory | the rows it fits |
|---|---|---|
| `keep` | 84.4k, 83.5 GiB | 16: 86.2k, 149.4 GiB |
| `recompute` | 77.1k, 66.8 GiB | 32: 78.8k, 192.2 GiB |

The one-GPU suite's Mamba-2 hybrid (`"MMMA"^4`), rows of 4096 on a B300, in JAX's
default share of the GPU's memory (2026-10-24):

| Rows | `states=keep` | `states=recompute` |
|---|---|---|
| 8 | 94.8k tok/s, 122.6 GiB | 90.7k, 75.5 GiB |
| 12 | 100.8k, 172.5 GiB | 91.9k, 101.4 GiB |
| 16 | 95.8k, 190.9 GiB | 91.7k, 128.1 GiB |
| 20 | out of memory | |
| 24 | | 92.6k, 182.5 GiB |
| 32 | | 92.8k, 193.5 GiB |
| 40 | | out of memory |

So `recompute` fits twice the rows, at 4 to 9% fewer tokens/s, its losses those of
`keep` to within 10⁻³.

### GDN's kernel on many GPUs

GDN's kernel runs on each GPU of a mesh as on one, on a node or two. Where it does not
apply, as under `pass_state`, the chunks run instead. The GDN hybrid (`"DDDA"^4`) at 8
rows of 4096 per GPU, on B300s, in tokens/s and the busiest GPU's peak (2026-10-24):

| GDN hybrid | by its kernel | by the chunks (`pass_state`) |
|---|---|---|
| one GPU | 127.7k, 70.6 GiB | 95.4k, 112.7 GiB |
| a node, `fsdp=8` | 815k, 58.4 GiB | 725k, 99.6 GiB |
| two nodes, `fsdp=16` | 1.58M, 58.0 GiB | |
| two nodes, `data=2, fsdp=8` | 1.60M, 58.4 GiB | 1.42M, 99.6 GiB |
| two nodes, `data=2, fsdp=4, context=2` | | 1.03M, 124.4 GiB |

- Two nodes train at 97% (FSDP over 16) and 98% (HSDP) of a node's tokens/s per GPU.
- The kernel's losses are the chunks' to 0.002 on a node and 0.005 on two at the first
  log, and to 0.0002 at the last of 30 steps.

## Not yet

- A `ppermute` chain instead of the all-gather of `pass_state`, which holds every
  device's transitions (`c` matrices `k × k` per head) on each.
- A kernel for RWKV-7 ([its chunks](#recurrence-rwkv7-chunks) take 2.3x the time of
  fla's Triton kernel), and one for Lightning's per-head decay (GDN-2's takes 3.5 ms
  per layer, fla's `chunk_simple_gla` 2.1).

## See also

- [Attention and context parallelism](attention.md), `methods.attention`.
- [Remat](remat.md): `recurrence.in` names a mixer's inputs as its kernel takes them,
  `recurrence.out` its output and the chunks' states.
- The recurrent layers themselves: [Recurrent mixers](../architectures/recurrent.md).
- The kernels, from the performance side: [Kernels](../performance/kernels.md).
