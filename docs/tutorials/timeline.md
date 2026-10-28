(tutorial-timeline)=
# 2020 to 2026, one change at a time

What has each change to the decoder since GPT-3 been worth? Papers claim gains for
RoPE, QK-norm, SwiGLU, grouped-query attention, mixtures of experts, linear-attention
hybrids and dozens more, each on its authors' own setup. This tutorial measures them
on one setup: agentic coding with long context, at the same serving cost. It starts
from GPT-3's decoder of 2020 and changes one thing at a time. Every variant trains on
a ladder of small widths. A scaling law extrapolates it to the size that serves
agents as fast as a GPT-3 6.7B, measured in vLLM. Then the changes add up into the
milestones of each year, and a factorial screen looks for what the late-2026 hybrid
still lacks.

By the end you will have the answer. At the serving cost of a GPT-3 6.7B and on the
same 1T tokens, the late-2026 linear-attention hybrid predicts 8.7% fewer bits per
byte than the 2020 baseline. DeepSeek-V3's layout of 2024 predicts 8.0% fewer, and
the late-2026 sliding-window frontier 7.9%. With unlimited data, the mixtures of
experts gain less, because they train at 0.11 to 0.26 MFU against the dense models'
0.52 to 0.65: DeepSeek-V3 reaches the baseline's loss in 2.9 times fewer B300
GPU-hours, the hybrid in 1.7.

% Interim: every number of this table and of the paragraphs around it.

```{include} timeline/tables/eras.md
```

Each row gives the milestone's target of equal serving cost and its MFU, its measured
change against 2020 at `d768` in percent, and its effective-parameter multiplier over
the rungs. Then, at the targets: the objective on 1T tokens, its change against
2020's, and the speedup to 2020's loss on 1T tokens with unlimited data. Ranges are
in brackets ([step 6](#what-the-fit-says)).

**Today's best architecture** is the 2026 hybrid with hyper-connections, the value
residual and a key-value conv. Per parameter it is worth 1.8 times the hybrid. At
equal serving cost it predicts only 0.4% fewer bits per byte on 1T tokens than the
hybrid, 9.0% fewer than 2020 ([step 8](#8-todays-best-architecture)). Of the single
changes, two kinds win:

- **Better per parameter.** Fine-grained mixtures of experts are worth 3.7 to 4.3
  times the parameters a token goes through: 4.2% to 5.5% fewer bits per byte on 1T
  tokens. Gated attention with QK-norm, KDA and the other linear-attention mixers,
  hyper-connections and the value residual follow, at 1.2 to 1.9 times.
- **Cheaper to serve.** The changes that shrink the KV cache are no better per
  parameter, and some are worse. At equal serving cost their targets hold 3.5 to 9
  times the reference's parameters per token: on 1T tokens, multi-head latent
  attention predicts 4.3% fewer bits per byte, grouped-query attention 3.0%, sliding
  windows 2.9%.

Most of the rest make no measurable difference at these sizes: RMSNorm, no biases,
the gated MLPs, QK-norm alone, sandwich norms, soft caps, multi-token prediction,
Engram, sinks, mHC. An untied head, post-branch norms, no positions, DyT, and the
gated residual stream are worse.

Getting there takes nine steps:

1. Train on code and whole repositories, ending at 16k tokens with YaRN.
2. Size four rungs, scan each variant's learning rate, and seed it.
3. Sweep one change at a time, from the 2020 baseline or the 2021 reference.
4. Test the claims of stability.
5. Measure each family's serving cost in vLLM, and find its target of equal cost.
6. Fit the ladders and extrapolate, with unlimited and with limited data.
7. Add the changes up into each year's milestone.
8. Screen the 2026 hybrid's add-ons by a factorial.
9. Read the results and their caveats.

This tutorial builds on [Old against new at the same serving cost](old-vs-new.md):
its data, its agentic load and its fits. Do that one first. This one trains about
700 runs on nodes of 8 NVIDIA B300s ([Time and hardware](#time-and-hardware)). Every
measurement here was taken on them, on 2026-10-27 and 2026-10-28.

## 1. The recipe

Every run trains as Old against new's do: code files, whole Python repositories and
web text in rows of 2k, then the last tenth in rows of 16k. Get the data as
[step 3 of Old against new](old-vs-new.md#3-get-the-data) does, at the same paths.
`examples/experiments/recipes/timeline_code.bux` is the recipe, a module with no
model. It is `agentic_code.bux` with five changes:

```bux
long_phase = phase(at=0.9, seq_len=16384, batch=batch // 8, data=long)
yarn_phase = long_phase(rope_scaling=yarn(factor=8, original_max_len=2048))
curriculum = yarn_phase

eval = [...,
        eval(name="repo", data=repos_eval, every=2k, seq_len=16384, batch=8, batches=16,
             positions=[2048, 8192])]
```

- **The learning rate is 2e-3**, a default that each variant's scan replaces.
- **The long phase switches RoPE to YaRN.** RoPE trained on rows of 2k has seen its
  slow channels turn only as far as 2048 positions take them; in rows of 16k they
  turn eight times as far. YaRN divides the slowest frequencies by `factor`, keeps
  the fastest, and ramps between
  ([Long-context extension](../training/curriculum.md#long-context-extension)).
  Layers with a window, or without RoPE, keep theirs.
- **The repository eval logs bits per byte by position**: `repo_bpb_0` for the
  targets in a repository's first 2k tokens, `repo_bpb_2k` from 2k to 8k,
  `repo_bpb_8k` past 8k ([By position](../training/running.md#by-position)).
- **`observe(every=500)`** logs each attention's largest logit and its entropy
  ([Observing a run](../training/observing.md)).
- **Every run logs to the W&B project `bux-timeline`.** `'log.trackers = []'` on the
  command line turns that off.

## 2. Rungs, learning rates and seeds

### Rungs at the same tokens

A rung is a width, and every variant trains at the same four.
`examples/experiments/timeline/rungs.bux` zips them by their labels:

```bux
d_model = d512: 512  | d768: 768  | d1024: 1024  | d1280: 1280
budget  = d512: 503M | d768: 1.7B | d1024: 4.03B | d1280: 7.86B
batch   = d512: 48   | d768: 96   | d1024: 192   | d1280: 288
mesh    = d512: mesh(data=1) | d768: mesh(data=1) | d1024: mesh(data=2) | d1280: mesh(data=4)
```

Each budget is 20 tokens for every parameter of the 2020 baseline at that width
besides its embeddings: 25.2M at `d512`, 85.1M at `d768`, 201M at `d1024`, 394M at
`d1280`. Every variant at a rung reads the same tokens, whatever its own size, so a
rung compares variants at equal data; their sizes go into the fit (step 6). The
batch follows DeepSeek LLM's batch-size law, as in
[Old against new](old-vs-new.md#5-train-the-ladders).

### One learning rate per variant

One rate for all would favor the variants it suits, so every variant scans three
rates at `d512`, as one sweep from the command line:

```sh
bux launch examples/experiments/timeline/norms.bux --only "[qknorm, d512" \
  'optimizer.lr = 1e-3 | 2e-3 | 4e-3'
```

`--only` keeps the runs whose name contains its text. The arms that claim stability
scan 8e-3 too (step 4). The wider rungs train at the vertex of a parabola in log
learning rate, fitted through the scan's points within 3% of its best, or at the
best point where the parabola has no minimum inside the scan. `fit.py` (step 6)
prints every scan's vertex:

```{include} timeline/tables/lr.md
```

"One step up" is the loss one rate above the best: the 2020 baseline and `hc` are
the most fragile. QK-norm's vertex puts its `d768` rung at 2.26e-3:

```sh
bux launch examples/experiments/timeline/norms.bux --only "[qknorm, d768]" \
  'optimizer.lr = 2.26e-3'
```

A setting of one value adds no label, so this run is `norms[qknorm, d768]`.

### Seeds

Every arm trains two more seeds at `d512`, at its best scan rate, and the reference
four:

```sh
bux launch examples/experiments/timeline/norms.bux --only "[qknorm, d512" \
  'optimizer.lr = 2e-3' 'seed = 1 | 2'
```

A seed sets the model's init and the data's order, and every family's run of a seed
reads the same data in the same order. So a seed shifts every family alike: seed 2's
objective is about 1.3% above seed 0's in every arm. `fit.py` treats the seeds as
blocks. A run's own noise is 0.35% of the objective, and the seeds' shifts 0.79%
(199 degrees of freedom). The repository eval reads 128 rows of 16k, the fewest of
the three, and is the noisiest: 0.62% own and 2.11% by seed.

% Interim: noise numbers.

`fit.py` gives every change a verdict against its parent:

- **better** or **worse**: it moves the objective by more than twice the noise of
  the difference at some rung, in the same direction at every rung;
- **mixed**: as far, but in different directions at different rungs;
- **no measurable difference**: otherwise.

The noise of a difference counts the runs on each side, and adds the seeds' spread
where the two sides ran on different seeds.

## 3. One change at a time

### The axis files

An axis file sweeps one part of the model, a labeled arm for each way to change it.
From `examples/experiments/timeline/positions.bux`:

```bux
model = base:             model
      # 2021, RoFormer, GPT-J, GPT-NeoX: relative positions by rotation; extrapolation
      | rope:             lm(attn(rope=rope()))
      # 2024 Llama 3: base 500k; long context
      | rope_500k:        lm(attn(rope=rope(base=500000)))
      # RoPE as in `rope`, but the long phase keeps it as trained: what YaRN adds
      | rope_noyarn:      lm(attn(rope=rope()))
      ...
curriculum = base: yarn_phase | rope: yarn_phase | rope_500k: yarn_phase | ...
           | rope_noyarn: long_phase | abf: long_phase(rope_base=500000)
```

- **Each arm changes one thing.** `base` is the file's parent itself.
- **The comment above an arm** gives its era and source, what it changes, and after
  the `;` what it was claimed for. `fit.py` reads it into its leaderboard.
- **`curriculum` zips with `model` by label**, so the arms that test the extension
  get a long phase of their own.

| File | Starts from | What it sweeps |
|------|-------------|----------------|
| `positions.bux` | the 2020 baseline | RoPE and its bases and fractions, NoPE, iRoPE, a relative bias, the extension |
| `norms.bux`, `interactions.bux` | the 2021 reference | RMSNorm, QK-norm, biases, sandwich and post-branch norms, soft caps; QK-norm taken apart and paired |
| `attention.bux`, `sparse.bux` | the 2021 reference | heads, GQA, MQA, MLA, windows, gates, sinks, a key-value conv; DSA, MSA, QSA, CSA |
| `ffn.bux`, `moe.bux` | the 2021 reference | gated MLPs, squared ReLU; mixtures of experts from 8 to 256 |
| `mixers.bux` | the 2021 reference | Mamba-2, Gated DeltaNet, KDA, GDN-2, RWKV-7, in three layers of four |
| `embeddings.bux`, `residuals.bux`, `structure.bux` | the 2021 reference | untied head, MTP, Engram, value residual; hc, mHC, AttnRes, gated and variance-preserving streams; parallel blocks, deep and thin |
| `zoo.bux` | the 2021 reference | the 2025-26 zoo's open choices: window sizes and ratios, sinks, gate forms, DyT, KDA ratios |
| `milestones.bux`, `zoo_moe.bux`, `swa.bux` | the 2020 baseline; the 2026 frontier | a family for each year; the frontier's routing; the frontier with the hybrid's winners and a leaner window |
| `stacks.bux`, `stacks2.bux`, `factorial.bux`, `audit.bux` | the 2026 hybrid | its add-ons, one at a time and in a fractional factorial (step 8), and the candidate's cheaper variants |

Some changes are not arms. Zero-centered gains train as plain ones, since bux's
weight decay leaves every gain alone. ALiBi, differential attention and value
embeddings have no layer in bux.

### The 2020 baseline and the 2021 reference

`examples/experiments/timeline/gpt3_2020.bux` is GPT-3's decoder at any width: a
layer for every 64 of the width, heads of 64, LayerNorms and projections with
biases, a GELU MLP four times as wide as the stream, a tied head, and a learned
table of 16k positions:

```bux
model = embed(max_positions=16384) block^layers norm head(tied=true)
```

The table's entries past 2048 train only in the last tenth of the run, and only where
a document runs that long. The scan's runs at `d512`, one per rate:

| Learning rate | baseline: objective | `repo_bpb` | `repo_bpb_8k` | reference: objective | `repo_bpb` | `repo_bpb_8k` |
|---------------|---------------------|------------|---------------|----------------------|------------|---------------|
| 1e-3 | 0.8963 | 0.7950 | 0.7879 | 0.7495 | 0.5902 | 0.5630 |
| 2e-3 | 0.8017 | 0.6527 | 0.6321 | 0.7476 | 0.5882 | 0.5625 |
| 4e-3 | 0.8478 | 0.7898 | 0.8229 | 0.7572 | 0.5971 | 0.5708 |

The baseline's repositories jump between 0.65 and 0.80 bits per byte while its code
moves smoothly. At 4e-3 it predicts targets past 8k worse than those in a
repository's first 2k. Its token scan breaks too: on 40 tokens per parameter it
scores worse than on 20. The **2021 reference**, `gpt3_rope.bux`, is the baseline
with RoPE in place of the table, as GPT-J and GPT-NeoX had it, and
`positions.bux`'s `rope` arm. Its long context works at every rate, so every axis
besides positions starts from it. Train each arm by name: the `base` arm of each
file repeats its parent, and `fit.py` folds it into the parent's runs.

## 4. Test the stability claims

Several arms were introduced for stable training at high learning rates. Their
scans at `d512` go on to 8e-3, four times the reference's best. The scan's runs:

| Arm | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|-----|------|------|------|------|
| reference | 0.7495 | 0.7476 | 0.7572 | not run |
| `qknorm` | 0.7453 | **0.7389** | 0.7449 | 0.7489 |
| `softcap` | 0.7526 | 0.7456 | 0.7524 | 0.7829 |
| `sandwich` | 0.7560 | 0.7511 | 0.7907 | 0.8396 |
| `postnorm` | 0.8905 | 1.4260 | 1.6492 | 2.9909 |
| `gated` | 0.7420 | 0.7318 | 0.7535 | 3.0718 |
| `hc` | 0.7360 | 0.7229 | 0.7546 | 1.0702 |
| `mhc` | 0.7488 | 0.7419 | 0.7763 | 0.9165 |

- **QK-norm is flat to 8e-3**, 1.4% above its best there. At its best rate it gains
  about 1% at every rung, just under the verdict's bar at each.
- **The reference does not diverge; its attention sharpens too fast.** By step 500
  its largest attention logit (`attn/logit_max`) reaches 71 at 2e-3 and 351 at
  4e-3, and the entropy of its middle layers (`attn/entropy`) falls to 0.22 and
  0.03. It recovers within 2k to 3k steps. With QK-norm, the logits stay at 12 to
  25 and the entropy at 0.40 or more (Dehghani et al., 2023; Wortsman et al., 2023).
- **Sandwich norms and the soft cap do not touch the product of queries and keys**,
  and do not help. Post-branch norms fail at every rate, and NoPE diverges.
- **QK-norm needs its learned gains** (`interactions.bux`): without them, or with
  the attention's logits soft-capped at its range instead, its gain is gone. It adds
  to the gate and to the key-value conv, and makes post-branch norms train.

## 5. Serving cost, measured

Each variant extrapolates to its **target**: the width at which it serves the
agentic load of [Old against new](old-vs-new.md#2-measure-the-serving-cost) as fast
as the budget. A change that shrinks the KV cache buys a wider target; one that
costs more per token, a narrower one.

### The method

Each family is exported without weights and timed in vLLM 0.32 with bux's plugin,
at a few widths:

```sh
python scripts/timeline/serving.py export examples/experiments/timeline/milestones.bux \
  --arm m2024_llama3 --out exported/llama3-4608 'd_model = 4608'
CUDA_VISIBLE_DEVICES=0 python scripts/old_vs_new/agentic.py exported/llama3-4608 \
  --concurrency 8,16,32,64,128
```

- **The load** is Old against new's: sessions of 2k shared and 14k own tokens, 4
  turns of 512 output tokens with a 2k tool result between them, prefix caching,
  dummy weights, bf16, MoE on vLLM's Triton kernels.
- **A width's cost** is its best sessions per hour per GPU with a median time per
  output token of 30 ms or less.
- **The budget** is the same for GPT-3 6.7B's shape (`old_2022.bux` at
  `d_model = 4096`, 32 layers of multi-head attention): 991 sessions per hour per
  B300, at 16 sessions at once.
- **A family's target** is where its cost crosses the budget, interpolated in log
  sessions per hour against log active parameters between the two widths that
  bracket it. `scripts/timeline/sweep_fit.py` reads the timing lines, with each
  width's GPUs and parameters, and writes every target to `targets-measured.json`.

Arms that differ from a timed family only by elementwise operations or by training
(norms, activations, RoPE variants) are timed at `d_model = 2816` and take that
family's curve, scaled by their measured ratio there.

### What sets a target

- **Tensor parallelism comes in steps.** A width runs on the fewest GPUs that hold
  its weights and a session's cache, and vLLM needs the key and value heads to
  divide by them. So the 2026 families run `d_model = 3584` on 2 GPUs and 4096 on 4,
  and a family's cost per GPU falls in a step between them: the hybrid serves 3,242
  sessions per hour at 2816 on one GPU, 1,352 at 3584 on two, and 641 at 4096 on
  four. Its target lands inside a step, at 3789 on 2 GPUs.
- **The residual streams need fused kernels.** With them, hyper-connections cost the
  hybrid 16% of its sessions per GPU at 2816, and AttnRes 8%. The measured cost is
  an upper bound, since kernels can improve. Its lower bound prices the algebra as
  `serving.py`'s analytic model does: the candidate of step 8 at 3654 (11.5B
  parameters per token) as measured, 3786 (12.8B) at the model's price.
- **MLA runs at the kernel's dims.** Trained with a latent of 256 and 32 rotated
  dims, it runs on vLLM's Triton MLA, about half as fast as at DeepSeek's dims (512
  and 64) on FlashInfer's. Its target is DeepSeek's dims' match, counted in the
  trained architecture's parameters.
- **Some families are kernel-limited.** The sparse indexers (DSA, MSA, QSA) score and
  attend in PyTorch float32 over gathered keys, with a first token about 4 s late at
  2816, and CSA runs as exact PyTorch. Their timings measure today's plugin, not the
  architecture, so they take the analytic model's target.

Across 160 timed widths, measured sessions per hour come at a median of 1.04 times
the analytic model's, from 0.14 to 2.22; the farthest are the kernel-limited ones.

### The targets

% Interim: bf16 targets; the final ones are FP8.

```{include} timeline/tables/serving.md
```

Each timed width gives its best sessions per hour per GPU, at how many sessions at
once, on how many GPUs, and the analytic model's prediction: the 2024 Llama 3 at 4608
serves 963 sessions per hour at 16 at once, against the model's 987.

**The final targets are FP8**: W8A8 weights and activations and an FP8 KV cache, with
the KV cache's scales uncalibrated. Their table:

```{include} timeline/tables/serving-fp8.md
```

## 6. Fit and extrapolate

### The law

`scripts/timeline/fit.py` reads the run directories under `runs/`. A family at a
rung is one point, at its best learning rate, its seeds averaged. Runs that stopped
short of 97% of their rung's tokens are left out. Each metric of each family fits

$$L(N, D) = E + A N^{-\alpha} + B D^{-\beta},$$

where N is the parameters a token goes through besides the embeddings, the head,
Engram's tables and the MTP module (an MoE's routed experts count `top_k` of
`num_experts`), and D is the tokens.

- **The root is the 2021 reference** (`--root positions/rope`), with every term of
  its own.
- **Every other family has its parent's terms, except those its points tell
  apart**: A always, α with three rungs or more, B and β with a token scan.
  `--share B,beta` pools those over every family.
- **Parents come from each file's modules**: the reference, the baseline, or the
  2026 hybrid that the file uses.
- **E is shared by every family**: the data's entropy. `--fix-e` fixes it at the
  values Old against new's ladders fit on the same data: 0.3204 bits per byte for
  the objective. A fit with E free lands at 0.3229, so the extrapolation does not
  hinge on it.

All families fit at once, by the Huber loss of log L, as Hoffmann et al. (2022) fit
theirs. The token scans train the reference and every milestone at `d512` on 5, 10,
40 and 80 tokens per baseline parameter. The baseline's learned positions break past
20 (step 3), so `--drop-scan baseline` leaves its scan out.

### What the fit says

- **The effective-parameter multiplier** of a family is the N at which its parent's
  law reaches the family's loss, over the family's own N. Its slope against N says
  whether the gain grows with scale or fades.
- **Limited data** gives every target the same tokens, 130B, 1T and 10T, and
  compares their predicted losses.
- **Unlimited data** gives each target the GPU-hours it needs to reach its parent's
  loss on 1T tokens: tokens times FLOPs per token, over the measured MFU times a
  B300's dense bfloat16 peak, 2.25 PFLOP/s. The MFU is measured at the target's
  width in 20 steps of synthetic rows, with MXFP8 matmuls; a family without its own
  takes a measured kin's, or its nearest ancestor's.
- **Ranges** span the fits that leave out one rung and 100 refits with each point
  moved by its noise and its seeds' shifts.

```sh
python scripts/timeline/fit.py runs --targets targets-measured.json --match B1b \
  --mfu mfu.json --root positions/rope --drop-scan baseline --share B,beta \
  --fix-e objective=0.3204,code_bpb=0.2808,repo_bpb=0.2448,web_bpb=0.3832 \
  --candidates factorial/r1v1k1f0p0s0,factorial/r1v1k0f0p0s0 --out fit
```

It writes `fit/report.md`, every number in `fit/fit.json`, and the plots below.
`--match B1b` sizes any family that `targets-measured.json` lacks with the analytic
model. `--selftest` fits synthetic runs of known laws instead.

```{image} timeline/ladder-objective.png
:alt: The objective of the reference and of the leading families against the parameters a token goes through, with their rungs, fits and targets
:class: only-light
```

```{image} timeline/ladder-objective-dark.png
:alt: The objective of the reference and of the leading families against the parameters a token goes through, with their rungs, fits and targets
:class: only-dark
```

The dots are the rungs, the lines each family's fit (dashed where it extrapolates),
and the stars the targets of equal serving cost, with their objective on 130B tokens.

## 7. The milestones

Each milestone is a file, `m2020_gpt3.bux` to `m2026_hybrid.bux`, swept by
`milestones.bux`. They keep the timeline's layer for every 64 of the width, and
their mixtures of experts send a token through as much MLP as the dense ones. Their
results lead this page; read them with the unlimited-data plot:

```{image} timeline/unlimited-objective.png
:alt: The objective of the reference and of the leading families against their training GPU-hours on B300s
:class: only-light
```

```{image} timeline/unlimited-objective-dark.png
:alt: The objective of the reference and of the leading families against their training GPU-hours on B300s
:class: only-dark
```

- **With unlimited data, MFU decides as much as bits do.** Llama 3 trains at 0.65 MFU and the
  2026 families at 0.16, so Llama 3's dense target reaches 2020's loss as fast as
  theirs. Qwen3-Next at 0.11, PaLM's 39B target, and GPT-NeoX, whose parallel blocks
  cost what its RoPE gains, need more GPU-hours than 2020.
- **The dense milestones' gains shrink with width**: the baseline's learned
  positions catch up (step 9). The mixtures of experts keep most of theirs.

### The sliding-window frontier against the KDA hybrid

The two 2026 families are the consensus of two camps, scaled down. They differ in
the layers that see a bounded context: windows of 512 keys in the frontier, KDA in
the hybrid.

% Interim: the SWA arms below the first two rows are still training.

```{include} timeline/tables/swa.md
```

- **At the same width, the frontier serves 1.4 to 1.5 times as many sessions per
  GPU**, and its cache and kernels are its architecture's own.
- **The hybrid holds 1.31 times the parameters per token at a width.** At equal
  cost, the frontier gets 8% more width and 21% more parameters in all, but 5% fewer
  per token.
- **Per parameter, the frontier trails the hybrid by 1.4% to 1.9%.**
- **The frontier with the hybrid's winners, the value residual in every layer, and
  MiMo's lean windows of 128 with sinks are training.** The lean frontier serves 15%
  more than the frontier at 3584 on 2 GPUs, but the step to 4 GPUs at 4096 caps its
  target in bf16.

## 8. Today's best architecture

### The factorial screen

`examples/experiments/timeline/factorial.bux` puts six add-ons on the 2026 hybrid at
once, in a fractional factorial of 16 arms:

- **R**, the residual stream: plain, hyper-connections (`hc`), or AttnRes;
- **V**, the value residual on the global layers; **K**, a causal conv of 4 taps on
  their keys and values; **P**, partial RoPE at base 500k on them; **S**, a DSA
  indexer on them;
- **F**, ReLU² experts, 1.5 times as wide, in place of SwiGLU.

The design is a 2^(7-3) of resolution IV: no main effect is aliased with another, or
with an interaction of two of V, K, F, P and S. Every arm trains three seeds at
`d512`, at one learning rate, 2e-3, the hybrid's optimum. `factorial.py` fits the
main effects and the interactions the design tells apart, its runs paired by seed.
`--reuse` and `--extra` add the runs of the same models in `milestones.bux` and
`stacks.bux`: 64 runs of 22 arms.

```sh
python scripts/timeline/factorial.py runs --reuse --extra
```

% Interim: the factorial's effects.

```{include} timeline/tables/factorial.md
```

Paired by seed, a run's noise is 0.33% of the objective, against 0.80% unpaired.

- **Hyper-connections are the largest add-on** (t = -18.5), then AttnRes, the conv
  and the value residual. The four add up: their interactions are within the noise.
- **ReLU² experts are worse on the hybrid** (+0.68%), unlike on the reference.
  Beside `hc`, an interaction of -1.02% cancels that.
- **Partial RoPE on the global layers is slightly worse** (+0.17%, t = 2.0), and the
  DSA indexer does nothing measurable.

The candidate, the hybrid with `hc`, the value residual and the conv, measures
-2.61 ± 0.20% at `d512` over three seeds, and -2.2% at `d768` over two. Its
learning rate holds: at `d512` it scores 0.6661, 0.6607 and 0.6654 at 1e-3, 2e-3
and 4e-3.

% Interim: the d768 numbers.

### The candidate at equal serving cost

```{include} timeline/tables/best.md
```

Per parameter the candidate is worth 1.8 times the hybrid, and yet at equal serving
cost it gains only 0.3% to 0.5% on the hybrid, 9.0% on 2020. Two things shrink it:

- **The hybrid's fitted exponent is steep**: α = 0.42 to 0.49, as E is fixed or free,
  against the reference's 0.25. At the hybrid's 12.8B parameters per token, its
  parameter term A N^-α is then about 2% of the predicted loss with E fixed (0.0076
  of 0.354 bits per byte). A multiplier of 1.8 lowers that term by 1.8^0.42, about
  1.28 times: a few tenths of a percent of the loss.

% Interim: the A and B terms' shares, with E fixed and free.

- **`hc` costs serving**: the candidate's target holds 11.5B parameters per token as
  measured, against the hybrid's 12.8B.

This is fragile. The exponent rests on four rungs whose widest has 514M parameters
per token, and the candidate's multiplier on two. Runs of the candidates at `d1024`
test both: whether the multiplier holds a rung wider, and whether the exponent stays
this steep. The result holds for this hybrid's shape: 256 experts with 7 routed, KDA
in three layers of four, and the learning rate of `d512` carried to every width.
AttnRes in place of `hc` is a co-candidate: it gains less (-2.22% at `d512`, with RoPE
on the global layers) and costs half as much to serve.

## 9. Results and caveats

### Each change alone

The appendix lists every change against its parent. Beyond the summary at the top:

- **The positions lead the dense changes at the rungs, and fade.** Every RoPE arm
  with YaRN or a raised base is worth 1.9 to 2.3 times learned positions per
  parameter. The learned table catches
  up with width: the baseline is 6.7% worse than the reference at `d512`, 2.4% at
  `d1024`, and the fit puts it even at the target.
- **Linear-attention mixers gain both ways**: 1.2 to 1.9 times the parameters, and
  targets of 10B to 18B.
- **Three gains flip sign at the target**, because they shrink fast with width: the
  value residual (better at every rung, +0.2% on 1T tokens), hyper-connections alone
  on the reference (+0.3%), and RoPE at base 500k (+0.9% against the baseline).

### Long context

The report's long-context table compares each family's bits per byte past 8k tokens
with those of a repository's first 2k, against its parent, averaged over the rungs:

```{include} timeline/tables/long.md
```

- **YaRN is what makes RoPE use 16k**: without it, RoPE gains on the baseline in a
  repository's first 2k (3.9%) but hardly past 8k (0.8%).
- **Learned positions catch up with width**: 9 points behind the reference at
  `d512`, 1 at `d1280`.
- **A window as long as the training rows hurts.** `swa_2048` loses 11% past 8k:
  in the 16k phase its local layers still see 2048 keys.
- **The sparse attentions lose at long context.** Against the reference, CSA is
  9.9% worse in a repository's first 2k and 17.2% past 8k; MSA matches the reference
  in the first 2k but not beyond.
- **The reference's `d768` point sits low**: most arms' differences there are 1 to
  7 points above those at their other rungs.

```{image} timeline/ladder-repo_bpb.png
:alt: Bits per byte on whole repositories in rows of 16k against the parameters a token goes through, with the families' fits and targets
:class: only-light
```

```{image} timeline/ladder-repo_bpb-dark.png
:alt: Bits per byte on whole repositories in rows of 16k against the parameters a token goes through, with the families' fits and targets
:class: only-dark
```

### Model shape

Every family scales its depth with its width: a layer for every 64 of the width, in
heads of 64. The budget's model keeps GPT-3 6.7B's 32 layers. So a target is deep:
at its equal-cost width of 3789, the hybrid has 59 layers. Deep and thin at equal
parameters (`deep_thin`) makes no measurable difference; shallow and wide is
untested.

### Short notes

- **DyT fails at this init.** With embeddings drawn at 0.02, `tanh(αx)` reads inputs
  far below 1: DyT is 74% worse at `d512`, and 2.6% worse with the embedding scaled
  by √d.
- **KDA in seven layers of eight is slightly worse than in three of four** (×1.61
  against ×1.92 per parameter), and KDA in every other layer, ×1.65.
- **The routing choices of the frontier's experts make no measurable difference**:
  softmax, no shared expert, SMEBU, or a square-root softplus (`zoo_moe.bux`).

### Secondary metrics

:::{note}
Placeholder: the runs' stability, activation outliers and the experts' load balance,
from their observed metrics.
:::

### Caveats

- **Learning rates transfer from `d512`.** Every family trains its wider rungs at its
  `d512` vertex. The most fragile, one rate step above their best, lose 3.7% to 5.8%
  (the baseline, sandwich norms, mHC, `hc`, GeGLU) against the reference's 1.3%, so
  they may sit off their optimum at wider rungs.
- **Seeds are shared shifts.** Paired by seed, a difference is 2 to 3 times sharper
  than unpaired. Most points past `d512` are one run, on seed 0.
- **Rungs differ by family.** 17 families have four rungs (the reference, the
  baseline, four milestones and 11 arms), 17 three, 51 two, and 25 one, most of the
  factorial among them. A family with fewer than three takes its parent's exponent,
  so its gain is one multiplier at every scale.
- **E is fixed at 0.3204.** A fit with E free lands at 0.3229, and moves the
  candidates' predictions by 0.03 points, so the extrapolation does not hinge on it.
- **The targets are far.** A dense target at the reference's cost has 4.4B parameters
  per token, 50 times past `d768`; the 2026 families' have 12B. The ranges are not
  confidence intervals.
- **Serving is measured on one load, with dummy weights.** A router's choices on
  random weights need not spread as trained ones do. Kernel-limited families take
  the analytic model's targets.
- **MFU is measured on fewer layers for the largest targets**, and the one-change MoE
  arms take the reference's dense MFU, so their unlimited-data speedups are upper
  bounds.
- **Some rungs do not fit or are too slow.**
  - At 16k rows, Qwen3-Next's attention heads of 256 need
    `methods = methods(remat=full, microbatches=2)` on one GPU at `d768`, and do not
    fit at `d1024`.
  - The MoE rungs at the memory cap need `methods = methods(remat=full,
    microbatches=2)`: `remat=auto` sizes the training step, not the observed one.
  - GDN-2 and the relative bias are too slow for their rungs.

## Time and hardware

On nodes of 8 B300s:

% Coordinator: exact figures, with the factorial's and the serving sweeps'.

| Step | what | GPU-hours |
|------|------|-----------|
| 1 | the data, as in Old against new | 10 min on one node |
| 2, 3 | the scans at `d512` and their seeds | about 60 |
| 3 | `d768` for every arm | about 110 |
| 3 | `d1024` for 25 arms, the reference, the baseline and the milestones | about 150 |
| 3, 7 | `d1280`, and the milestones' ladders and token scans | about 160 |
| 5, 6 | the throughput lanes and the serving sweeps | about 20 |
| 6 | the fit, on a laptop | none |

## Next steps

- Add an arm: a labeled line in an axis file, with the comment `fit.py` reads above
  it ([Labels zip axes](../language/sweeps.md#labels-zip-axes)).
- Add a layer bux does not have, such as ALiBi
  ([Adding a layer](../architectures/adding-a-layer.md)).
- [Serving in vLLM](../export/vllm.md): the plugin that times every family here.
- [Searching hyperparameters](../training/searching.md): a search in place of the
  fixed scan.

## References

- Brown et al. (2020), *Language Models are Few-Shot Learners*,
  [arXiv:2005.14165](https://arxiv.org/abs/2005.14165): GPT-3, the baseline.
- Dehghani et al. (2023), *Scaling Vision Transformers to 22 Billion Parameters*,
  ICML, [arXiv:2302.05442](https://arxiv.org/abs/2302.05442): QK-norm.
- DeepSeek-AI (2024), *DeepSeek-V2*,
  [arXiv:2405.04434](https://arxiv.org/abs/2405.04434): multi-head latent attention.
- DeepSeek-AI (2024), *DeepSeek-V3 Technical Report*,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*,
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556): 20 tokens per parameter, and
  the fit.
- Peng et al. (2023), *YaRN: Efficient Context Window Extension of Large Language
  Models*, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).
- Qiu et al. (2025), *Gated Attention for Large Language Models*,
  [arXiv:2505.06708](https://arxiv.org/abs/2505.06708).
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864): RoPE.
- Wortsman et al. (2023), *Small-scale proxies for large-scale Transformer training
  instabilities*, [arXiv:2309.14322](https://arxiv.org/abs/2309.14322).
- Zhu et al. (2025), *Hyper-Connections*, ICLR,
  [arXiv:2409.19606](https://arxiv.org/abs/2409.19606).

The data's sources are in Old against new's references, and every arm's in the
comment above it, in its file.

## Appendix: every change

Every family against its parent: the verdict, the measured change at each rung in
percent, the multiplier per parameter, and at equal serving cost the speedup with
unlimited data and the change on 1T tokens.

% Interim: the whole table.

```{include} timeline/tables/leaderboard.md
```

Every family on 130B, 1T and 10T tokens, against the reference's target:

```{image} timeline/limited-objective.png
:alt: Every family's objective on 130B, 1T and 10T tokens against the reference's target, as bars
:class: only-light
:width: 60%
```

```{image} timeline/limited-objective-dark.png
:alt: Every family's objective on 130B, 1T and 10T tokens against the reference's target, as bars
:class: only-dark
:width: 60%
```
