# Optimizers, schedules and the loss

Four settings turn a gradient into an update: the optimizer, the learning-rate
schedule, gradient clipping, and the loss the gradient comes from. A fifth, `ema`,
sets the weights a run ends on ([The tail EMA](#the-tail-ema)).

```bux
optimizer = muon(lr=0.02)
schedule  = wsd(warmup=1k, decay=0.2, shape=sqrt)
clip      = 1.0
loss      = loss(z=1e-4)
```

| Setting | Takes | Default |
|---------|-------|---------|
| `optimizer` | {bux}`adamw`, {bux}`muon` or {bux}`normuon` | required |
| `schedule` | {bux}`constant`, {bux}`cosine`, {bux}`wsd`, {bux}`linear_decay`, {bux}`multistep`, {bux}`power` or {bux}`piecewise` | `constant` |
| `clip` | a global norm, or `none` | `1.0` |
| `loss` | {bux}`loss` | `loss(z=0, mtp=0.3)` |
| `ema` | {bux}`ema`, or `none` | `none` |

The optimizers are optax transformations built from their configs
({py:mod}`bux.train.optim`). A variant of either is a function registered under its
name ([Registries](extending.md#registries)).

## AdamW

```bux
optimizer = adamw(lr=3e-4, b1=0.9, b2=0.95, weight_decay=0.1)
```

{bux}`adamw` is AdamW (Loshchilov & Hutter, 2019), as `optax.adamw` computes it.

| Field | Default | What it is |
|-------|---------|------------|
| `lr` | required | the peak learning rate, which the schedule scales |
| `b1` | `0.9` | the decay of the first moment |
| `b2` | `0.95` | the decay of the second moment |
| `eps` | `1e-8` | added to the square root of the second moment |
| `weight_decay` | `0.1` | decoupled weight decay, times the learning rate |
| `cautious_weight_decay` | `false` | decay only where the update has the weight's sign ([below](#cautious-weight-decay)) |

**What decays.** Weight decay applies only to matrices and stacks of them. It skips
every vector (biases, norm scales) and anything a layer lists in its
`no_weight_decay` attribute, whatever its shape: norm gains and biases, Engram's
tables, Mamba's and Mamba-2's `A_log` (Mamba's is a matrix of decay rates), RWKV-7's
token shift and `r_k`, and hyper-connections' static maps, which start at
$[e_k, I]$ and 1 rather than at 0. The stacked dimension of a scanned repeat does not
count, so the norm scales that `methods(repeat=scan)` stacks are still vectors and
do not decay ({py:func}`bux.train.optim.weight_decay_mask`).

(cautious-weight-decay)=
### Cautious weight decay

With `cautious_weight_decay=true`, an entry decays only where the step and the decay
pull the same way: where the optimizer's update $u$ (Adam's $\hat m / (\sqrt{\hat
v} + \epsilon)$, Muon's orthogonalized update) has the weight's sign, or either is 0
(Chen et al., 2025):

$$
x \leftarrow x - \eta \left(u + \lambda\, \mathbb{I}(u \odot x \ge 0) \odot x\right).
$$

Elsewhere the step alone moves the weight. It needs no setting of its own: the same
`weight_decay` and learning rate. `adamw` and `muon` both take it, and Muon's `adam`
takes its own.

```bux
optimizer = adamw(lr=2e-3, weight_decay=0.1, cautious_weight_decay=true)
```

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
| `momentum` | `0.95` | the decay of the momentum |
| `nesterov` | `true` | Nesterov's look-ahead |
| `ns_steps` | `5` | Newton-Schulz iterations |
| `ns_coefficients` | `jordan` | the quintics of the iterations, by name |
| `ns_algorithm` | `standard` | how it iterates: `standard`, or `gram`, on each matrix's Gram matrix |
| `ns_restarts` | `none` | the iterations after which `gram` restarts; `none` is 2, 5, 8, ... |
| `ns_dtype` | `bfloat16` | the dtype Newton-Schulz runs in: `bfloat16` or `float32` |
| `weight_decay` | `0.1` | decoupled weight decay of the matrices, times the learning rate |
| `cautious_weight_decay` | `false` | decay only where the orthogonalized update has the weight's sign ([Cautious weight decay](#cautious-weight-decay)) |
| `consistent_rms` | `0.2` | the RMS updates are scaled to, or `none` |
| `adam` | `none` | the AdamW of every other parameter; `none` is `adamw(lr=lr)` |

### What Muon trains

Outside the embedding and the head, Muon trains:

- the kernel of every linear layer (`nnx.Linear`): attention's projections, the
  MLPs', the router's, and the rest;
- the routed experts' stacked weights, one expert at a time. In a gated MLP, an
  expert's up and gate projections are stacked side by side, and Muon orthogonalizes
  each separately;
- whatever a layer lists in its `muon_axes` attribute, giving each matrix's two axes
  by number.

In a scanned repeat the stacked dimension comes first, and Muon orthogonalizes each
pass's matrix separately. Everything else trains with `adam`: the embedding, the head
(whose layers set `no_muon`), norm scales, biases, convolutions, and any other
parameter that is not one of those matrices. `adam` defaults to `adamw(lr=lr)`: Muon's
learning rate with AdamW's other defaults. The schedule scales both
({py:func}`bux.train.optim.muon_matrices`).

### Scaling the updates: `consistent_rms`

An orthogonalized update has its own scale, unlike Adam's. With `consistent_rms=0.2`,
the default, Muon scales each update to an RMS of 0.2, roughly that of AdamW's
updates, so one learning rate suits both (Moonlight). With `consistent_rms=none`, it
scales by `sqrt(max(1, fan_out / fan_in))`, as Keller Jordan's Muon does. Such a run
wants a larger `lr` for Muon than for its `adam`, as
`examples/experiments/speedrun.bux` sets.

### Newton-Schulz

`ns_coefficients` names the quintic `a x + b x³ + c x⁵` of each iteration, from
{py:data}`bux.train.optim.ns_coefficients`:

| Name | Its quintics |
|------|--------------|
| `jordan` | Keller Jordan's, `(3.4445, -4.7750, 2.0315)`, the same at every iteration: optax's default |
| `polar_express` | Amsel et al. (2025): one per iteration, closer to orthogonal in the same number of iterations; the last repeats past eight |

Newton-Schulz runs in `ns_dtype`, bfloat16 by default as in Keller Jordan's Muon, and
the updates move between devices in it. Its result is within a few percent of
float32's. On a B300 (2026-10-23), 240 steps of the 1.3B dense model on packed
documents reach the same loss either way, within the difference between two float32
runs.

When FSDP or tensor parallelism splits a matrix, `methods.orthogonalize` says how it
reaches one device whole for Newton-Schulz ([Muon](../parallelism/muon.md)).

### Gram Newton-Schulz

```bux
optimizer = muon(lr=0.02, ns_algorithm=gram)
optimizer = muon(lr=0.02, ns_algorithm=gram, ns_coefficients=polar_express, ns_steps=8)
```

`ns_algorithm` names how Newton-Schulz iterates, from
{py:data}`bux.train.optim.ns_algorithms`:

| Name | How |
|------|-----|
| `standard` | optax's: the quintics of the matrix itself, three products per iteration, two of them by the matrix |
| `gram` | Gram Newton-Schulz (Zhang, Amsel, Chen and Dao, 2026): the same quintics, on the Gram matrix of each matrix's shorter side |

For an m×n matrix X with m ≤ n, an iteration's quintic `a X + b (XXᵀ)X + c (XXᵀ)²X`
is a polynomial of the Gram matrix `R = XXᵀ`, times X. `gram` forms R once, iterates
on m×m matrices (`Z = bR + cR²`, `Q ← Q(Z + aI)`, `R ← (Z + aI)R(Z + aI)`), and
multiplies by X only at the end and at its restarts, giving the same orthogonalized
`QX`. The standard iteration multiplies by X twice per iteration, so the further a
matrix is from square, the more `gram` saves. At 5 iterations, counting every product
whole (XLA has no symmetric products):

| Matrix | Fewer multiply-adds |
|--------|---------------------|
| square, or n below 1.5m | none: `gram` runs the standard iteration |
| a Llama MLP's, n ≈ 2.7m | 22% |
| n = 4m | a third |
| n = 8m | 46% |

Under `gspmd`, a product by a matrix split along its longer side (as FSDP splits it)
is an all-reduce, so `gram` also all-reduces less. Under `gather` and `owner` it runs
on whole matrices, like the standard iteration. How matrices move does not change
([Muon](../parallelism/muon.md)).

**Restarts.** The Gram matrix of a nearly low-rank matrix has eigenvalues near zero,
which rounding pushes slightly negative. Iterated on R, they grow until they blow up.
So `gram` restarts after each iteration in `ns_restarts`: it multiplies X by the Q so
far, forms R anew, and resets Q to I, at the cost of two more products by the matrix.
The default, `none`, restarts after the second iteration and every third after it
(2, 5, 8, ...); `ns_restarts=2` or `ns_restarts=[2, 4]` sets others. On the authors'
test spectra (192×768 matrices whose singular values decay to 1e-9, or are half
zero; CPU, 2026-10-26), the default keeps the largest singular value within 0.07 of
the standard iteration's, in bfloat16, at 4 to 12 iterations of either quintic. The
authors' single restart, after the second iteration, drifts at six iterations and
blows up from seven.

**Dtypes.** In a 16-bit `ns_dtype`, `gram` normalizes in float32 and iterates in
float16, as its authors do: bfloat16's 8 bits round the Gram matrix too coarsely. Its
updates still move between devices in `ns_dtype`. Each `a` is added in float32
before the product is rounded, as a GEMM epilogue would add it; rounded first, eight
iterations of Polar Express blow up. The Gram matrix squares the condition number, so
for an orthogonalization that needs more digits than Muon's, use `standard` in
float32.

**Speed.** On a B300 (2026-10-26), `gram` runs Newton-Schulz 1.05 to 1.4 times as
fast as `standard` on matrices 2.75 to 8 times as wide as tall, and up to 12% slower
at twice as wide. In bfloat16 it lands 6 to 8 times closer to float32's result than
`standard` does, and trains to the same loss. A run's step barely changes, because
under `owner` Newton-Schulz is a few milliseconds of it. The authors' 2x speedup needs
symmetric GEMM kernels with fused epilogues, which XLA does not have
([measurements](../parallelism/muon.md#gram-newton-schulz-on-b300s)).

### NorMuon

{bux}`normuon` is NorMuon (Li et al., 2025): Muon, its orthogonalized update then
divided, for each output unit of a matrix (a neuron), by the root of a running
average of the unit's mean square, and scaled back to the update's norm before. Units
whose orthogonalized rows come out larger than others then move as much as the rest.
It takes every field of `muon`, and two of its own:

| Field | Default | What it is |
|-------|---------|------------|
| `b2` | `0.95` | the decay of each neuron's mean square's average, without a bias correction |
| `eps` | `1e-10` | added to the average's root |

```bux
optimizer = normuon(lr=2e-3, weight_decay=0.1, adam=adamw(lr=2e-3, weight_decay=0.1))
```

It computes as the authors' code does: the mean square of each output's row of the
update, the update over its root, then the whole matrix back to its Frobenius norm,
before `consistent_rms` scales it. A stack of experts normalizes each expert's
matrices on their own. Its state is one float32 average per output unit of every
matrix Muon trains ({py:func}`bux.train.optim.neuron_normalized`).

On a KDA hybrid of 680M parameters at a width of 512, the line above lowered the
held-out bits per byte by 2.3% against AdamW at the same learning rate (503M tokens,
3 seeds, B300s, 2026-10-29). Half and twice that learning rate came within 0.4% of it.

## Schedules

A schedule multiplies the optimizer's `lr`. Each starts with a linear `warmup` and
then follows its own curve; the multiplier at a step is their product. `warmup` is a
number of steps or, below 1, a share of them: `warmup=0.01` is 1% of the steps.

| Schedule | Fields | After the warmup |
|----------|--------|------------------|
| {bux}`constant` | `warmup=0` | the peak, to the end |
| {bux}`cosine` | `warmup=0`, `final=0.1`, `horizon=1` | a cosine decay to `final` times the peak at the last step, or at the `horizon` of the steps |
| {bux}`wsd` | `warmup=0`, `decay=0.2`, `final=0`, `shape=linear` | the peak, then a cooldown to `final` times the peak over the last `decay` of the steps |
| {bux}`linear_decay` | `warmup=0`, `final=0` | a linear decay to `final` times the peak at the last step |
| {bux}`multistep` | `warmup=0`, `at=[0.8, 0.9]`, `factor=0.316` | the peak, times `factor` from each share of the steps in `at` on |
| {bux}`power` | `warmup=0`, `a=4`, `b=-0.51`, `decay=0.1`, `final=0`, `shape=exp` | a power law in the tokens trained, capped at the peak, then a cooldown |
| {bux}`piecewise` | `warmup=0`, `pieces` | {bux}`piece` after piece: holds, decays, drops and re-warmups |

### The curves exactly

Steps count from 0, the first step, to `steps - 1`, the last. Below, `s` is a step,
`S` is `steps`, and `W` is the number of warmup steps: `warmup`, or
`round(warmup · S)` for a share.

**Warmup.** The multiplier rises linearly from `1 / W` at step 0 to 1 at step `W - 1`:
`min((s + 1) / W, 1)`. With `warmup=0` there is none.

**Cosine.** From the end of the warmup, `t = clip((s + 1 - W) / (H - W), 0, 1)`, with
`H = round(horizon · S)`, runs from 0 to 1 at step `H - 1`, and the multiplier is
`final + (1 - final) · ½ (1 + cos πt)`: the peak at the end of the warmup, `final` at
the last step. With `horizon` above 1, the run stops before the cosine reaches
`final`, as OLMo 2 and OLMo 3 do; below 1, it holds `final` from step `H - 1` on.

**Warmup-stable-decay.** WSD (Hu et al., 2024) holds the peak, then cools down over
the last `C = round(decay · S)` steps. There, `t = clip((s + 1 - (S - C)) / C, 0, 1)`
runs from 0 to 1 at the last step, and the multiplier is `final + (1 - final) · f(t)`,
with `f` set by `shape`:

| `shape` | `f(t)` |
|---------|--------|
| `linear` | `1 - t` |
| `sqrt` | `1 - √t` (Hägele et al., 2024) |
| `cosine` | `½ (1 + cos πt)` |
| `exp` | `(e^-t - e^-1) / (1 - e^-1)`: it falls by a factor of e over the cooldown, shifted to end at 0, as IBM's does |
| `constant` | 0 from the cooldown's first step: `final` at once |

`shape=geometric` has no `f`: the multiplier is `final^t`, falling by the same factor
at every step, as MiniCPM's exponential decay does, so its `final` must be above 0.

So the last step of `cosine` and of `wsd` runs at `final` times the peak. With
`wsd`'s default `final=0`, the last step's learning rate is 0: the smoke run's log
ends with `lr 0.00e+00`. Checkpoints from WSD's stable phase can be cooled down later,
to study several run lengths at once.

**Linear decay.** `linear_decay` decays from the end of the warmup:
`t = clip((s + 1 - W) / (S - W), 0, 1)`, and the multiplier is
`final + (1 - final) · (1 - t)`. With `final=0` it goes straight to zero, which
Bergsma et al. (2025) find better than cosine and WSD.

**Multistep.** The multiplier is `factor^k`, where `k` counts the shares `a` in `at`
with `s >= round(a · S)`. DeepSeek LLM drops to 31.6% of the peak at 80% of its
tokens and to 10% at 90%: `multistep(warmup=2k, at=[0.8, 0.9], factor=0.316)`.

**Power.** IBM's Power scheduler (Shen et al., 2024) sets the learning rate from the
tokens trained through step `s`, `n = (s + 1) · batch · seq_len`:
`p(s) = min(1, a · batch · n^b / lr)`, where `lr` is the optimizer's peak. So `a` and
`b` suit any batch and any number of tokens, and `lr` only caps the result: the
paper's is 0.02, under µP. The warmup rises to `p(W - 1)` and holds `p` from there.
Over the last `C = round(decay · S)` steps, the cooldown goes from `p(S - C - 1)` to
`final` times it, as `wsd`'s goes from the peak. Under a `batch_ramp` or a
[curriculum](curriculum.md), `n` counts each step's tokens at that step's batch and
`seq_len`, and `batch` is the step's own.

**Piecewise.** Piece `k` covers the steps from `b` to `e = round(until · S)`. The
first piece starts at `W`, and every other where the one before it ends. A piece goes
from the multiplier `m` where the piece before ended (the peak, for the first) to its
`to`, or stays at `m` with `to=none`, along its `shape`: `to + (m - to) · f(t)`, with
`t = clip((s + 1 - b) / (e - b), 0, 1)`, or `m · (to / m)^t` for `geometric`. A
piece's default, `shape=constant`, is at `to` from its first step. DeepSeek-V3 holds
the peak to 10T of its 14.8T tokens, decays along a cosine to 10% by 14.3T, holds
that to 14.633T, and drops to 3.3% for the rest:

```bux
schedule = piecewise(warmup=2k, pieces=[piece(until=10 / 14.8),
                                        piece(until=14.3 / 14.8, to=0.1, shape=cosine),
                                        piece(until=14.633 / 14.8),
                                        piece(to=0.033)])
```

:::{note}
The warmup and the cooldown must fit in the steps. `bux check` fails a run whose
warmup plus cooldown (`round(decay · steps)` for `wsd` and `power`) is more than
`steps`, and a `piecewise` whose warmup ends after its first piece. The error points
at the schedule's line ([Checks](settings.md#checks)).
:::

### Schedules from tech reports

`examples/experiments/recipes/lr_schedules.bux` defines the pretraining schedules of
25 tech reports by name (`deepseek_v3`, `kimi_k2`, `olmo_2`, ...), each with a note on
what it leaves out. A run uses one, or sweeps several, next to its own optimizer, as
`examples/experiments/schedule_sweep.bux` does:

```bux
use lr_schedules
schedule = deepseek_v3 | kimi_k2 | olmo_2 | d2z
```

Where a report gives tokens, the recipe uses shares of the run, so a shorter run keeps
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
| Chained phases | DeepSeek-V3 and V4.1, MiMo-7B and V2-Flash, MiniMax-Text-01, GLM-5 | `piecewise(pieces=[...])` |
| A re-warmup | Moonlight, Marin 8B, Trinity's context extension | a rising `piece(to=0.24, shape=linear)` |

Moonlight's re-warmup is 100 steps long, so its schedule is written in a run that
knows its steps:

```bux
# A cosine to 10% by 5.2T of 5.7T tokens, back up to 24% in 100 steps, then
# linearly to 0.
schedule = piecewise(warmup=2k, pieces=[piece(until=5.2 / 5.7, to=0.1, shape=cosine),
                                        piece(until=5.2 / 5.7 + 100 / steps, to=0.238,
                                              shape=linear),
                                        piece(to=0, shape=linear)])
```

Not in bux:

- Ling 2.0's WSM, which holds the peak and merges checkpoints instead of decaying;
- WSD-S's repeated cooldowns, which would be as many pieces;
- Zamba's exponential anneal, whose time constant differs from `exp`'s;
- modded-nanogpt's multipliers, which step with its batch size.

A long-context stage at the final learning rate is a separate run at `constant`.
Qwen3, Gemma 3 and 4, gpt-oss, Llama 4, Jamba, Kimi K2.5, MiniMax-M2 and M3, Inkling
and Beam state no schedule.

The log's `lr` is the learning rate of the step just taken. A new schedule is a
function registered under its name ([Registries](extending.md#registries)).

(the-tail-ema)=
## The tail EMA

```bux
ema = ema(window=0.25, blend=0.65)
```

{bux}`ema` averages the weights over the run's last steps and ends the run on a blend
of the average and the last step's weights (modded-nanogpt's TailEMA). The average
starts at the first step of the window, as that step's weights, and every later step
moves it toward its own by $2 / (\text{window} + 1)$. After the last step the weights
become $w + \text{blend}\,(\bar w - w)$: `blend=1` ends on the average alone, `0` on
the last step's.

| Field | Default | What it is |
|-------|---------|------------|
| `window` | `0.25` | the steps it averages, the last of the run, or below 1 a share of them |
| `blend` | `0.65` | how much of the average the weights end on |

- **What reads it:** the final eval, the final checkpoint and an export of it, which
  the blend replaces the weights for. Evals before the last step see the weights as
  trained.
- **Memory:** a float32 copy of the parameters in the optimizer's state, sharded as
  they are, for the whole run.
- **Resuming:** checkpoints keep the average, so a run resumes inside its window
  and ends where it would have.
- **Schedules:** an average helps most where the steps still move the weights. A
  cooldown to 0, such as `wsd`'s, ends on steps that barely do; modded-nanogpt pairs
  its EMA with a learning rate that decays to a floor above 0.

modded-nanogpt averages over the last 298 of 1188 steps and blends 0.65 of the
average into its embedding and head, with other averages for its other weights; bux
takes one average of every parameter ({py:mod}`bux.train.ema`).

## Clipping

`clip = 1.0`, the default, scales the gradient down to a global norm of 1 whenever
its norm is larger, before the optimizer sees it (`optax.clip_by_global_norm`).
`clip = none` turns clipping off; `clip = 0`, which some frameworks take for off, is
an error, as a norm of 0 would leave the optimizer no gradient. The log's
`grad_norm` is the norm before clipping.

## The loss

```bux
loss = loss(z=1e-4, mtp=0.3)
```

The objective is the cross-entropy of the model's first head, plus these terms. A
first head with a copy pointer takes its mixture's negative log-likelihood instead,
and so do its evals ([Copy pointer](../architectures/layers.md#copy-pointer)):

| Field | Default | The term |
|-------|---------|----------|
| `z` | `0` | `z` times the z-loss, the mean of log(Z)² of the first head (PaLM: 1e-4) |
| `mtp` | `0.3` | `mtp` times the mean cross-entropy of the heads after the first (DeepSeek-V3) |

Each head predicts one token further ahead for every `nextn` before it, so a model
with heads after its first predicts several tokens at each position
([Several outputs](../language/outputs.md)). Every term is a mean over the batch's
target tokens, not over rows, so a row of many short replies weighs no more than a
row of one long one ([Rows](../data/index.md#rows)). The cross-entropy keeps its
gradient even where float32 rounds a target's probability to 1
([Softmax and cross-entropy at large margins](precision.md#softmax-and-cross-entropy-at-large-margins)).

The log names each term: `loss` is the first head's cross-entropy alone, then
`z_loss` with a z-loss, and `mtp` with heads after the first. Layers add their own
terms, such as an MoE's balance loss and a sparse attention indexer's loss, which the
log names `balance_loss` and `indexer_loss` (see [Experts](../architectures/experts.md)
and [Attention](../architectures/attention.md)). Eval leaves those out.

Whether the loss takes a head's logits in chunks of positions or whole is
`methods.loss` ([Loss methods](extending.md#loss-methods)). A term of your own is a
function ([Objective terms](extending.md#objective-terms)).
