# Optimizers, schedules and the loss

This page covers what turns a gradient into an update: the two optimizers, the
learning-rate schedules, gradient clipping, and the loss the gradient is of. Each is
one setting of a run.

```bux
optimizer = muon(lr=0.02)
schedule  = wsd(warmup=1k, decay=0.2, shape=sqrt)
clip      = 1.0
loss      = loss(z=1e-4)
```

| Setting | Takes | Default |
|---------|-------|---------|
| `optimizer` | {bux}`adamw` or {bux}`muon` | required |
| `schedule` | {bux}`constant`, {bux}`cosine` or {bux}`wsd` | `constant` |
| `clip` | a global norm, or `none` | `1.0` |
| `loss` | {bux}`loss` | `loss(z=0, mtp=0.3)` |

Both optimizers are optax transformations built from their configs
({py:mod}`bux.train.optim`). A variant of one is a function registered under its name
([Registries](extending.md#registries)).

## AdamW

```bux
optimizer = adamw(lr=3e-4, b1=0.9, b2=0.95, weight_decay=0.1)
```

{bux}`adamw` is AdamW (Loshchilov & Hutter, 2019), as `optax.adamw` computes it.

| Field | Default | What it is |
|-------|---------|------------|
| `lr` | required | the peak learning rate, which the schedule scales |
| `b1` | `0.9` | the first moment's decay |
| `b2` | `0.95` | the second moment's decay |
| `eps` | `1e-8` | added to the second moment's root |
| `weight_decay` | `0.1` | decoupled weight decay, times the learning rate |

**What decays.** Weight decay applies to matrices, and to stacks of them, only. It
skips every vector (biases, norm scales) and what a layer names in its
`no_weight_decay` attribute whatever its shape: norms' gains and biases, and Engram's
tables. The stacked dimension of a scanned repeat does not count, so the stacked norm
scales of `methods(repeat=scan)` are still vectors and do not decay
({py:func}`bux.train.optim.weight_decay_mask`).

## Muon

```bux
optimizer = muon(lr=0.02)
optimizer = muon(lr=0.05, consistent_rms=none, adam=adamw(lr=0.008, b1=0.8, weight_decay=0))
```

{bux}`muon` is Muon (Jordan et al., 2024): momentum, orthogonalized by Newton-Schulz
iterations, for the matrices inside the blocks, and AdamW for every other parameter.

| Field | Default | What it is |
|-------|---------|------------|
| `lr` | required | the peak learning rate, which the schedule scales |
| `momentum` | `0.95` | the momentum's decay |
| `nesterov` | `true` | Nesterov's look-ahead |
| `ns_steps` | `5` | Newton-Schulz iterations |
| `ns_coefficients` | `jordan` | the quintics of the iterations, by name |
| `ns_algorithm` | `standard` | how it iterates: `standard`, or `gram`, on each matrix's Gram matrix |
| `ns_restarts` | `none` | the iterations after which `gram` restarts; `none` is 2, 5, 8, ... |
| `ns_dtype` | `bfloat16` | the dtype Newton-Schulz runs in: `bfloat16` or `float32` |
| `weight_decay` | `0.1` | decoupled weight decay of the matrices, times the learning rate |
| `consistent_rms` | `0.2` | the RMS updates are scaled to, or `none` |
| `adam` | `none` | the AdamW of every other parameter; `none` is `adamw(lr=lr)` |

### What Muon trains

Muon trains, outside the embedding and the head:

- the kernel of every linear layer (`nnx.Linear`): attention's projections, the MLPs',
  the router's, and the rest;
- the routed experts' stacked weights, one expert at a time. In a gated MLP an
  expert's up and gate projections are stacked side by side, and Muon orthogonalizes
  the two apart;
- what a layer names in its `muon_axes` attribute, each matrix's two axes by number.

A scanned repeat's stacked dimension leads, and Muon orthogonalizes each pass's
matrix apart. Everything else trains with `adam`: the embedding, the head (whose
layers set `no_muon`), norm scales, biases, convolutions and every other parameter
that is not one of those matrices. `adam` defaults to `adamw(lr=lr)`, so it takes
Muon's learning rate and AdamW's other defaults; the schedule scales both
({py:func}`bux.train.optim.muon_matrices`).

### Scaling the updates: `consistent_rms`

An orthogonalized update has a size of its own, unlike Adam's. With
`consistent_rms=0.2`, the default, Muon scales each update to an RMS of 0.2, as
AdamW's roughly are, so that one learning rate suits both (Moonlight). With
`consistent_rms=none`, it scales by `sqrt(max(1, fan_out / fan_in))`, as Keller
Jordan's Muon does; such a run then wants a larger `lr` for Muon than for its `adam`,
as `examples/experiments/speedrun.bux` sets.

### Newton-Schulz

`ns_coefficients` names the quintic `a x + b x³ + c x⁵` of each iteration, from
{py:data}`bux.train.optim.ns_coefficients`:

| Name | Its quintics |
|------|--------------|
| `jordan` | Keller Jordan's, `(3.4445, -4.7750, 2.0315)`, the same at every iteration: optax's default |
| `polar_express` | Amsel et al.'s (2025), one for each iteration, closer to orthogonal in as many; the last one repeats past eight |

Newton-Schulz runs in `ns_dtype`, bfloat16 as in Keller Jordan's Muon, and the
updates move between devices in it. Its result is within a few percent of float32's:
on a B300, 240 steps of the 1.3B dense model on packed documents lose the same with
either, within the difference between two float32 runs.

How each matrix reaches a device whole for Newton-Schulz, when FSDP or tensor
parallelism splits it, is `methods.orthogonalize` ([Muon](../parallelism/muon.md)).

### Gram Newton-Schulz

```bux
optimizer = muon(lr=0.02, ns_algorithm=gram)
optimizer = muon(lr=0.02, ns_algorithm=gram, ns_coefficients=polar_express, ns_steps=8)
```

`ns_algorithm` names how Newton-Schulz iterates, from
{py:data}`bux.train.optim.ns_algorithms`:

| Name | How |
|------|-----|
| `standard` | optax's: the quintics of the matrix itself, three products an iteration, two of them by the matrix |
| `gram` | Gram Newton-Schulz (Zhang, Amsel, Chen and Dao, 2026): the same quintics, on the Gram matrix of each matrix's shorter side |

An iteration's quintic `a X + b (XXᵀ)X + c (XXᵀ)²X` is a polynomial of the Gram matrix
`R = XXᵀ` times X. For an m×n matrix with m ≤ n, `gram` forms R once, iterates on
m×m matrices (`Z = bR + cR²`, `Q ← Q(Z + aI)`, `R ← (Z + aI)R(Z + aI)`) and
multiplies by X only at the end, and at its restarts: `QX`, the same orthogonalized
matrix. The standard
iteration multiplies by X twice an iteration. The further a matrix is from square,
the more that saves. At 5 iterations, counting every product whole (XLA has no
symmetric products), a Llama's MLP matrices (n ≈ 2.7m) take 22% fewer multiply-adds,
a matrix four times as wide a third fewer, eight times 46%. Where it would not save
any, below n = 1.5m and on square matrices, `gram` runs the standard iteration.

Under `gspmd`, a product by a matrix split along its longer side (FSDP's) is an
all-reduce, so `gram` all-reduces less there too. Under `gather` and `owner` it runs
on whole matrices, as the standard iteration does. Nothing about how matrices move
changes ([Muon](../parallelism/muon.md)).

**Rounding.** The Gram matrix of a matrix near low rank has eigenvalues near zero,
which rounding makes a little negative. Iterated on R, these grow until they blow
up. So `gram` restarts: after each iteration in `ns_restarts` it multiplies X by
the Q so far, forms R anew and starts Q again from I. That takes two more products by
the matrix. Its authors restart five iterations once, after the second. `none`, the
default, restarts after the second and every third after it (2, 5, 8, ...).
Measured on CPU (2026-10-26) on 192×768 matrices of the authors' test spectra
(decaying to 1e-9, or half zero), in bfloat16 and at 4 to 12 iterations of either
quintic, that keeps the largest singular value within 0.07 of the standard
iteration's. A single restart after the second drifts at six iterations and blows
up from seven, past 10³ at eight. `ns_restarts=2` or `ns_restarts=[2, 4]` sets them.

In a 16-bit `ns_dtype`, `gram` normalizes in float32 and iterates in float16, as
its authors do: bfloat16's 8 bits round the Gram matrix too coarsely. Its updates
still move between devices in `ns_dtype`. Each `a` is added in float32 before the
product is rounded, as a GEMM's epilogue adds, since `Z + aI` rounded to 16 bits loses
it. The Gram matrix squares the condition number, so for an orthogonalization that
needs more digits than Muon's do, use `standard` in float32.

## Schedules

A schedule multiplies the optimizer's `lr`. Each takes a linear `warmup` of some
steps, then a curve of its own, and the multiplier at a step is their product:

| Schedule | Fields | After the warmup |
|----------|--------|------------------|
| {bux}`constant` | `warmup=0` | the peak, to the end |
| {bux}`cosine` | `warmup=0`, `final=0.1` | a cosine decay to `final` times the peak at the last step |
| {bux}`wsd` | `warmup=0`, `decay=0.2`, `final=0`, `shape=linear` | the peak, then a cooldown to `final` times the peak over the last `decay` of the steps |

### The curves exactly

Steps count from 0, the first step, to `steps - 1`, the last; write `s` for a step, `S`
for `steps` and `W` for `warmup`.

**Warmup.** The multiplier rises linearly from `1 / W` at step 0 to 1 at step `W - 1`:
`min((s + 1) / W, 1)`. With `warmup=0` there is none.

**Cosine.** From the end of the warmup, `t = clip((s + 1 - W) / (S - W), 0, 1)` runs
from 0 to 1 at the last step, and the multiplier is
`final + (1 - final) · ½ (1 + cos πt)`: the peak at the end of the warmup, `final` at
the last step.

**Warmup-stable-decay.** WSD (Hu et al., 2024) holds the peak, then cools down over
the last `C = round(decay · S)` steps. There
`t = clip((s + 1 - (S - C)) / C, 0, 1)` runs from 0 to 1 at the last step, and the
multiplier is `final + (1 - final) · f(t)`, with `f` by `shape`:

| `shape` | `f(t)` |
|---------|--------|
| `linear` | `1 - t` |
| `sqrt` | `1 - √t` (Hägele et al., 2024) |
| `cosine` | `½ (1 + cos πt)` |

So the last step of `cosine` and of `wsd` takes `final` times the peak: `wsd`'s
default `final=0` takes a last step at a learning rate of 0, which the log shows as
`lr 0.00e+00` at the end of the smoke run. Checkpoints from WSD's stable phase can be
cooled down later, to study several run lengths at once.

:::{note}
The warmup and the cooldown must fit in the steps: `bux check` fails a run whose
`warmup + round(decay · steps)` exceeds `steps`, at the line of the schedule
([Checks](settings.md#checks)).
:::

The log's `lr` is the learning rate of the step just taken. A new schedule is a
function registered under its name ([Registries](extending.md#registries)).

## Clipping

`clip = 1.0`, the default, scales the gradient down to a global norm of 1 wherever its
norm is larger, before the optimizer sees it (`optax.clip_by_global_norm`). `clip =
none` turns it off. The log's `grad_norm` is the norm before clipping.

## The loss

```bux
loss = loss(z=1e-4, mtp=0.3)
```

The objective is the cross-entropy of the model's first head, plus these terms:

| Field | Default | The term |
|-------|---------|----------|
| `z` | `0` | `z` times the z-loss, the mean of log(Z)² of the first head (PaLM: 1e-4) |
| `mtp` | `0.3` | `mtp` times the mean cross-entropy of the heads after the first (DeepSeek-V3) |

Each head predicts one token further ahead for every `nextn` before it, so a model
with heads after its first predicts several tokens at each position
([Several outputs](../language/outputs.md)). Every term is a mean over the target
tokens of the batch, not over rows, so a row of many short replies weighs no more than
a row of one long one ([Rows](../data/index.md#rows)). The cross-entropy keeps its
gradient where float32 rounds a target's probability to 1
([Softmax and cross-entropy at large margins](precision.md#softmax-and-cross-entropy-at-large-margins)).

The log has each term by name: `loss`, the first head's cross-entropy alone, then
`z_loss` with a z-loss and `mtp` with heads after the first. Layers add terms of
their own, an MoE's balance loss and a sparse attention indexer's loss, which the log
has by their names (`balance_loss`, `indexer_loss`): see
[Experts](../architectures/experts.md) and [Attention](../architectures/attention.md).
Eval leaves those out.

How the loss takes a head's logits, by chunks of positions or whole, is
`methods.loss` ([The loss's methods](extending.md#loss-methods)). A term of
your own is a function ([Objective terms](extending.md#objective-terms)).
