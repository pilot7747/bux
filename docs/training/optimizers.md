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
| `schedule` | {bux}`constant`, {bux}`cosine`, {bux}`wsd`, {bux}`linear_decay`, {bux}`multistep`, {bux}`power` or {bux}`piecewise` | `constant` |
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

## Schedules

A schedule multiplies the optimizer's `lr`. Each takes a linear `warmup`, then a
curve of its own, and the multiplier at a step is their product. `warmup` is a number
of steps, or below 1 a share of them: `warmup=0.01` is 1% of the steps.

| Schedule | Fields | After the warmup |
|----------|--------|------------------|
| {bux}`constant` | `warmup=0` | the peak, to the end |
| {bux}`cosine` | `warmup=0`, `final=0.1`, `horizon=1` | a cosine decay to `final` times the peak at the last step, or at the `horizon` of the steps |
| {bux}`wsd` | `warmup=0`, `decay=0.2`, `final=0`, `shape=linear` | the peak, then a cooldown to `final` times the peak over the last `decay` of the steps |
| {bux}`linear_decay` | `warmup=0`, `final=0` | a linear decay to `final` times the peak at the last step |
| {bux}`multistep` | `warmup=0`, `at=[0.8, 0.9]`, `factor=0.316` | the peak, times `factor` from each share of the steps in `at` on |
| {bux}`power` | `warmup=0`, `a=4`, `b=-0.51`, `decay=0.1`, `final=0`, `shape=exp` | a power law in the tokens trained, capped at the peak, then a cooldown |
| {bux}`piecewise` | `warmup=0`, `phases` | {bux}`phase` after phase: holds, decays, drops and re-warmups |

### The curves exactly

Steps count from 0, the first step, to `steps - 1`, the last; write `s` for a step, `S`
for `steps` and `W` for the steps of the warmup: `warmup`, or `round(warmup · S)`
for a share.

**Warmup.** The multiplier rises linearly from `1 / W` at step 0 to 1 at step `W - 1`:
`min((s + 1) / W, 1)`. With `warmup=0` there is none.

**Cosine.** From the end of the warmup, `t = clip((s + 1 - W) / (H - W), 0, 1)`, with
`H = round(horizon · S)`, runs from 0 to 1 at step `H - 1`, and the multiplier is
`final + (1 - final) · ½ (1 + cos πt)`: the peak at the end of the warmup, `final` at
the last step. With `horizon` above 1 the run stops before the cosine reaches
`final`, as OLMo 2's and OLMo 3's do; below 1 it holds `final` from step `H - 1` on.

**Warmup-stable-decay.** WSD (Hu et al., 2024) holds the peak, then cools down over
the last `C = round(decay · S)` steps. There
`t = clip((s + 1 - (S - C)) / C, 0, 1)` runs from 0 to 1 at the last step, and the
multiplier is `final + (1 - final) · f(t)`, with `f` by `shape`:

| `shape` | `f(t)` |
|---------|--------|
| `linear` | `1 - t` |
| `sqrt` | `1 - √t` (Hägele et al., 2024) |
| `cosine` | `½ (1 + cos πt)` |
| `exp` | `(e^-t - e^-1) / (1 - e^-1)`: it falls by e over the cooldown, moved to end at 0, as IBM's does |
| `constant` | 0 from the cooldown's first step: `final` at once |

`shape=geometric` is no `f`: the multiplier is `final^t`, falling by the same factor
at every step, as MiniCPM's exponential decay does, so its `final` must be above 0.

So the last step of `cosine` and of `wsd` takes `final` times the peak: `wsd`'s
default `final=0` takes a last step at a learning rate of 0, which the log shows as
`lr 0.00e+00` at the end of the smoke run. Checkpoints from WSD's stable phase can be
cooled down later, to study several run lengths at once.

**Linear decay.** `linear_decay` decays from the end of the warmup:
`t = clip((s + 1 - W) / (S - W), 0, 1)` and the multiplier is
`final + (1 - final) · (1 - t)`. With `final=0` it goes straight to zero, which
Bergsma et al. (2025) find better than cosine and WSD.

**Multistep.** The multiplier is `factor^k`, where `k` counts the shares `a` in `at`
with `s >= round(a · S)`. DeepSeek LLM drops to 31.6% of the peak at 80% of its
tokens and to 10% at 90%: `multistep(warmup=2k, at=[0.8, 0.9], factor=0.316)`.

**Power.** IBM's Power scheduler (Shen et al., 2024) sets the learning rate from the
tokens trained through step `s`, `n = (s + 1) · batch · seq_len`:
`p(s) = min(1, a · batch · n^b / lr)`, with `lr` the optimizer's peak. So `a` and `b`
suit any batch and number of tokens, and `lr` only caps it: the paper's is 0.02,
under µP. The warmup rises to `p(W - 1)` and holds `p` there; over the last
`C = round(decay · S)` steps, the cooldown goes from `p(S - C - 1)` to `final` times
it, as `wsd`'s goes from the peak.

**Piecewise.** Phase `k` takes the steps from `b` to `e = round(until · S)`, where the
first phase starts at `W` and every other where the one before ends. It goes from
the multiplier `m` the phase before ended at (the peak, for the first) to its `to`
(or stays at `m` with `to=none`) along its `shape`: `to + (m - to) · f(t)`, with
`t = clip((s + 1 - b) / (e - b), 0, 1)`, and `m · (to / m)^t` for `geometric`. A
phase's default `shape=constant` is at `to` from its first step. DeepSeek-V3 holds
the peak to 10T of its 14.8T tokens, decays along a cosine to 10% by 14.3T, holds
that to 14.633T and drops to 3.3% for the rest:

```bux
schedule = piecewise(warmup=2k, phases=[phase(until=10 / 14.8),
                                        phase(until=14.3 / 14.8, to=0.1, shape=cosine),
                                        phase(until=14.633 / 14.8),
                                        phase(to=0.033)])
```

:::{note}
The warmup and the cooldown must fit in the steps: `bux check` fails a run whose
warmup and cooldown, `round(decay · steps)` for `wsd` and `power`, take more than
`steps`, and a `piecewise` whose warmup ends after its first phase, at the line of
the schedule ([Checks](settings.md#checks)).
:::

### Schedules from tech reports

`examples/experiments/recipes/lr_schedules.bux` defines the pretraining schedules of
25 tech reports by name (`deepseek_v3`, `kimi_k2`, `olmo_2`, ...), with what each
leaves out. A run uses one, or sweeps them, next to its own optimizer, as
`examples/experiments/schedule_sweep.bux` does:

```bux
use lr_schedules
schedule = deepseek_v3 | kimi_k2 | olmo_2 | d2z
```

Where a report gives tokens, the recipe has shares of its run, so a shorter run keeps
the shape. As of October 2026, the reports use these shapes:

| Shape | Reports | In bux |
|-------|---------|--------|
| Cosine, to 1-10% where given | Llama 3 405B, Nemotron-H, GLM-4.5, Trinity Large, Kimi K3 | `cosine(final=0.01)` |
| Cosine cut off before its end | OLMo 2, OLMo 3 | `cosine(warmup=2k, horizon=5 / 4)` |
| WSD, linear | SmolLM2, SmolLM3, Mellum2, Marin 32B, nanochat (to 5%) | `wsd(warmup=2k, decay=0.1)` |
| WSD, `1 - √t` | Nemotron 3 Super and Ultra, Laguna XS.2, Apertus | `wsd(decay=0.2, final=0.01, shape=sqrt)` |
| WSD, cosine | Kimi K2 | `wsd(warmup=500, decay=5.5 / 15.5, final=0.1, shape=cosine)` |
| WSD, exponential | MiniCPM, halving every 5000 steps | `wsd(decay=0.1, final=0.5 ^ (0.1 * steps / 5000), shape=geometric)` |
| Linear from the warmup | Straight to zero (D2Z), OLMo's mid-training, Phi-4 (its end not given) | `linear_decay(warmup=0.1)` |
| Steps of 0.316 | DeepSeek LLM, DeepSeek-V2 | `multistep(warmup=2k, at=[0.6, 0.9])` |
| A power law in tokens | PowerLM, Granite 3.0 and 4.0 | `power(decay=0.1)` |
| Chained phases | DeepSeek-V3 and V4.1, MiMo-7B and V2-Flash, MiniMax-Text-01, GLM-5 | `piecewise(phases=[...])` |
| A re-warmup | Moonlight, Marin 8B, Trinity's context extension | a rising `phase(to=0.24, shape=linear)` |

Moonlight's, whose re-warmup is 100 steps long, is written in a run that knows its
steps:

```bux
# A cosine to 10% by 5.2T of 5.7T tokens, back up to 24% in 100 steps, then
# linearly to 0.
schedule = piecewise(warmup=2k, phases=[phase(until=5.2 / 5.7, to=0.1, shape=cosine),
                                        phase(until=5.2 / 5.7 + 100 / steps, to=0.238,
                                              shape=linear),
                                        phase(to=0, shape=linear)])
```

Not in bux: Ling 2.0's WSM, which holds the peak and merges checkpoints instead of
decaying; WSD-S's repeated cooldowns, which are as many phases; Zamba's exponential
anneal, whose time constant differs from `exp`'s; and modded-nanogpt's multipliers
that step with its batch size. A long-context stage at the final learning rate is a
run of its own at `constant`. Qwen3, Gemma 3 and 4, gpt-oss, Llama 4, Jamba, Kimi
K2.5, MiniMax-M2 and M3, Inkling and Beam state no schedule.

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
