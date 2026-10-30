(tutorial-timeline)=
# 2020 to 2026, one change at a time

What has each change to the decoder since GPT-3 been worth? Papers claim gains for
RoPE, QK-norm, SwiGLU, grouped-query attention, mixtures of experts, linear-attention
hybrids and dozens more, each on its authors' own setup. This tutorial measures them
on one setup: agentic coding with long context, at the same serving cost. It starts
from GPT-3's decoder of 2020 and changes one thing at a time. Every variant trains on
a ladder of small widths. A scaling law extrapolates it to the size that serves
agents as fast as a GPT-3 6.7B, measured in vLLM in FP8. Then the changes add up into
the milestones of each year, a factorial screen looks for what the late-2026 hybrid
still lacks, and the new layers of 2026 go on top of its best.

By the end you will have the answer. At the serving cost of a GPT-3 6.7B and on the
same 1T tokens, the late-2026 linear-attention hybrid predicts 8.5% fewer bits per
byte than the 2020 baseline. The mixtures of experts since 2024 land within about 0.3
points of it: Qwen3-Next's layout of 2025 predicts 8.5% fewer, DeepSeek-V3's of 2024
8.4%, and the late-2026 sliding-window frontier 8.2%. With unlimited data, the
mixtures of experts gain less, because they train at 0.11 to 0.26 MFU against the
dense models' 0.52 to 0.65: DeepSeek-V3 reaches the baseline's loss in 2.3 times
fewer B300 GPU-hours, the hybrid in 1.4.

```{include} timeline/tables/eras.md
```

Each row gives the milestone's target of equal serving cost and its MFU, its measured
change against 2020 at `d768` in percent, and its effective-parameter multiplier over
the rungs. Then, at the targets: the objective on 1T tokens, its change against
2020's, and the speedup to 2020's loss on 1T tokens with unlimited data. Ranges are
in brackets ([step 6](#what-the-fit-says)).

**Today's best architecture**, the candidate, is the 2026 hybrid with hyper-connections,
the value residual and a key-value conv. Paired by seed, it measures 2.6%, 2.2%, 1.9%
and 1.8% fewer bits per byte than the hybrid at `d512`, `d768`, `d1024` and `d1280`. Per
parameter it is worth 3.4 times the hybrid. At equal serving cost the fit predicts 0.4%
to 1.0% fewer bits per byte on 1T tokens than the hybrid, as it ties the short ladders,
and 9.0% fewer than 2020 ([step 8](#8-todays-best-architecture)). The new layers of 2026
gain more at `d512`, and lose most of it by `d1024`. The largest, a copy pointer in the
head, falls from 4.1% to 1.4% against the candidate. With GatedNorm it is the contender,
`stack5`: ahead of the candidate at every rung so far, its lead shrinking by more than a
third a rung, and within the fit's spread of it at the target. One change improves the
candidate for free: its one dense layer moved from the first layer to the last gains
0.27% to 0.42% at every rung to `d1024`, in 7 of 7 seeds, at the same serving cost, and
0.5% at the target. Build the candidate that way.

:::{note}
Two sets of runs are still training, and their rows here are interim: `stack5` at
`d1280`, and the third seed at `d768` of index sharing with one indexer in two.
:::

Of the single changes, two kinds win:

- **Better per parameter.** Fine-grained mixtures of experts are worth 4.4 to 5.1
  times the parameters a token goes through: 4.2% to 5.7% fewer bits per byte on 1T
  tokens. Gated attention with QK-norm, KDA and the other linear-attention mixers,
  hyper-connections, the gated residual stream and the value residual follow, at 1.2
  to 2.2 times.
- **Cheaper to serve.** The changes that shrink the KV cache are no better per
  parameter, and some are worse. At equal serving cost their targets hold 3.7 to 9
  times the reference's parameters per token: on 1T tokens, multi-head latent
  attention predicts 4.1% fewer bits per byte, grouped-query attention 3.2%, sliding
  windows 2.8%.

The rest split by what they were claimed for:

- **Claimed for throughput or the KV cache.** RMSNorm, no biases, heads of 128, parallel
  blocks, MLA, the sparse indexers DSA, QSA and MSA, and windows of 128 or in every
  other layer match the reference's quality, as claimed. What they buy is speed, which
  the serving targets count. Every architecture this page recommends keeps RMSNorm:
  pre-norm, and QK-norm in its attention.
- **Claimed for quality or stability.** The gated MLPs, sandwich norms, soft caps,
  multi-token prediction, Engram and sinks show no measurable gain at these sizes.
  QK-norm alone gains about 1% at every rung, and keeps training stable (step 4).
  The gated residual stream gains on both claims: it is better per parameter, and
  flat to 8e-3.
- **Worse.** An untied head, post-branch norms, no positions, and DyT.

Quality is trained and measured at 16k tokens of context or less. Serving is
measured at 64k, 128k and 256k too, where some changes' costs move
([step 5](#serving-cost-by-context-length)), but their quality there is not.

Getting there takes nine steps:

1. Train on code and whole repositories, ending at 16k tokens with YaRN.
2. Size four rungs, scan each variant's learning rate, and seed it.
3. Sweep one change at a time, from the 2020 baseline or the 2021 reference.
4. Test the claims of stability.
5. Measure each family's serving cost in vLLM in FP8, and find its target of equal
   cost, from 16k to 256k tokens of context.
6. Fit the ladders and extrapolate, with unlimited and with limited data.
7. Add the changes up into each year's milestone.
8. Screen the 2026 hybrid's add-ons by a factorial, then put the new layers of 2026
   on its best.
9. Read the results and their caveats.

This tutorial builds on [Old against new at the same serving cost](old-vs-new.md):
its data, its agentic load and its fits. Do that one first. This one trains about
1,000 runs on nodes of 8 NVIDIA B300s ([Time and hardware](#time-and-hardware)). Every
measurement here was taken on them, from 2026-10-27 to 2026-10-29.

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
objective is 1.4% above seed 0's in every arm. `fit.py` treats the seeds as
blocks, and puts every point on seed 0's scale. A run's own noise is 0.33% of
the objective, and the seeds' shifts 0.69% (412 degrees of freedom). The repository
eval reads 128 rows of 16k, the fewest of the three, and is the noisiest: 0.64% own
and 2.06% by seed.

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
| `ideas.bux`, `ideas2.bux`, `canon.bux`, `stack3.bux` | the candidate (step 8) | settings of bux the study had not used: GatedNorm, the embedding's scale, Canon convs, shared keys and values, Engram; their winners stacked |
| `newlayers.bux`, `newlayers_cplm.bux`, `newlayers_doc.bux`, `newlayers_aux.bux` | the candidate (step 8) | the new layers of 2026: XSA, value embeddings, HyGA, K=V, differential attention, MSAR, null experts, paired heads, mimetic init; the copy pointer; the document-local copy; the prefix-token loss |
| `stack4.bux`, `stack5.bux`, `stack6.bux` | the candidate (step 8) | the new layers' winners stacked: value embeddings, learned XSA and differential attention in a factorial with GatedNorm; the copy pointer with GatedNorm; every change that held at d768 |
| `ideas_opt.bux`, `newlayers_opt.bux`, `muon_2020.bux`, `muon_2021.bux` | the candidate; the 2020 baseline; the 2021 reference | training options beside the AdamW recipe, not architecture: inits, Muon, NorMuon, the tail EMA, cautious weight decay |
| `sparse_native.bux` | the candidate (step 8) | DSA, QSA and MSA on its global layers, in the shapes vLLM serves on its own kernels (DeepSeek-V3.2's, Qwen3.8's, MiniMax-M3's) |
| `dense0.bux` | the candidate (step 8) | its first layer's FFN routed like every other's: no dense layer before the experts |
| `indexshare.bux` | `sparse_native.bux`'s `dsa_c` | index sharing (GLM-5.2's IndexShare): one global layer in four, or in two, runs the indexer, and the global layers up to the next reuse its picks; with IndexCache's cross-layer loss too |
| `indexshare_ref.bux` | `attention.bux`'s `dsa` | index sharing where every layer is sparse, as GLM-5.2's are: one layer in four runs the indexer, with IndexCache's cross-layer loss or without |
| `stack5_d0.bux` | `stack5.bux` | its first layer's FFN routed like every other's, as `dense0.bux` routes the candidate's |
| `dense_last.bux` | `dense0.bux` | the candidate's one dense layer moved from the first layer to the last |
| `stack5_dl.bux` | `stack5_d0.bux` | stack5's one dense layer moved from the first layer to the last |
| `mhc_init.bux`, `factorial_mhc.bux` | the 2021 reference; the 2026 hybrid | mHC's start, three ways to set its streams apart; mHC in place of `hc` in the candidate |
| `western.bux`, `gdn2_hybrid.bux` | the 2026 frontier; the 2026 hybrid | the frontier's global layers with partial RoPE or MLA; GDN-2 in place of KDA, alone and with the candidate's add-ons |
| `lfm2.bux` | the 2021 reference; the candidate (step 8) | LFM2's gated short conv in three layers of four, as `mixers.bux`'s arms; in place of the candidate's KDA, at its 3:1, at LFM2-1.2B's 3 global layers in 8, and with RoPE on the global layers as LFM2's |

Some changes are not arms. Zero-centered gains train as plain ones, since bux's
weight decay leaves every gain alone. ALiBi has no layer in bux.

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
scans at `d512` go on to 8e-3, four times the reference's best:

```{image} timeline/lr-scans.png
:alt: Each arm's objective at d512 against the 2021 reference's best, at each peak learning rate from 1e-3 to 8e-3, in four panels: norms, attention, residual streams, and mixers and milestones
```

Each point is a run of seed 0, against the reference's best. Off-scale points sit at
the top with their value, ringed where a loss spike passed 0.25 nats or did not
recover. Loss spikes do not separate the arms at the working rate (step 9), so the
claims show in these scans and in the observed metrics:

- **QK-norm holds.** It stays within 1.3% of its best from 1e-3 to 8e-3, without a
  spike. Its largest attention logit is 0.27 times the reference's, in 7 of 7 seed
  pairs, and its attention entropy never collapses. It does not remove massive
  activations: the stream's crest, its largest value over its rms, is 1.14 times the
  reference's. At its best rate it gains 1.1%, 0.9% and 0.9% at `d512`, `d768`
  and `d1024`.
- **Without QK-norm, attention sharpens too fast.** By step 500 the reference's
  largest logit (`attn/logit_max`) reaches 71 at 2e-3 and 351 at 4e-3, and the
  entropy of its middle layers (`attn/entropy`) falls to 0.03 at 4e-3; both recover
  by step 3000. With QK-norm, the logits stay at 12 to 25 and the entropy at 0.33 or
  more up to 8e-3 (Dehghani et al., 2023; Wortsman et al., 2023).
- **Gated attention removes massive activations, but is no stability fix alone.**
  Its crest is 0.24 times the reference's, yet it diverges at 8e-3. Gated attention
  with QK-norm gets both.
- **Post-branch norms fail without QK-norm**, at every rate from 2e-3 up. With it,
  OLMo 2's layout has the flattest scan of any arm.
- **Sandwich norms and soft caps do not widen the range**: they lose 5% to 12% at
  8e-3, against 1.3% for QK-norm.
- **The gated residual stream holds to 8e-3.** It scores 0.7273 at 4e-3 and 0.7274
  at 8e-3, 2.7% under the reference's best, without a spike.
- **mHC degrades less than hc at 8e-3**: +20% against +48%, each against its own
  best. mHC's loss does not spike there, but it clips 17.5% of its steps, against
  0.2% at 2e-3.
- **QK-norm needs its learned gains** (`interactions.bux`): without them, or with
  the attention's logits soft-capped at its range instead, its gain is gone. It adds
  to the gate and to the key-value conv.

## 5. Serving cost, measured

Each variant extrapolates to its **target**: the width at which it serves the
agentic load of [Old against new](old-vs-new.md#2-measure-the-serving-cost) as fast
as the budget. A change that shrinks the KV cache buys a wider target; one that
costs more per token, a narrower one.

### The method

Each family is exported without weights and timed in vLLM 0.32 with bux's plugin, in
FP8, at a few widths:

```sh
python scripts/timeline/serving.py export examples/experiments/timeline/milestones.bux \
  --arm m2024_llama3 --out exported/llama3-4608 'd_model = 4608'
CUDA_VISIBLE_DEVICES=0 python scripts/old_vs_new/agentic.py exported/llama3-4608 \
  --concurrency 8,16,32,64,128 \
  --engine 'quantization="fp8"' --engine 'kv_cache_dtype="fp8"'
```

- **The load** is Old against new's: sessions of 2k shared and 14k own tokens, 4
  turns of 512 output tokens with a 2k tool result between them, prefix caching,
  dummy weights, MoE on vLLM's Triton kernels.
- **A width's cost** is its best sessions per hour per GPU with a median time per
  output token of 30 ms or less.
- **The budget** is the same for GPT-3 6.7B's shape (`old_2022.bux` at
  `d_model = 4096`, 32 layers of multi-head attention): 1,924 sessions per hour per
  B300 in FP8, at 32 sessions at once. In bf16 it is 991, at 16 at once.
- **A family's target** is where its cost crosses the budget, interpolated in log
  sessions per hour against log active parameters between the two widths that
  bracket it. `scripts/timeline/sweep_fit.py` reads the timing lines, with each
  width's GPUs and parameters, and writes every target to `targets-measured.json`.

Arms that differ from a timed family only by elementwise operations or by training
(norms, activations, RoPE variants) are timed at one width and take that family's
curve, scaled by their measured ratio there.

### What sets a target

- **Tensor parallelism comes in steps.** A width runs on the fewest GPUs that hold
  its weights at a byte each and a session's cache at a byte an element, and vLLM
  needs the key and value heads to divide by them. So the 2026 families run
  `d_model = 3584` on one GPU and 4096 on two, and a family's cost per GPU falls in a
  step between them. The hybrid serves 3,683 sessions per hour at 2816 and 2,075 at
  3584, both on one GPU, and 985 at 4096 on two. Its target lands inside the step,
  at 3633.
- **The residual streams need fused kernels.** With them, the candidate of step 8
  (hyper-connections, the value residual and the conv) serves 16% fewer sessions
  per GPU than the hybrid at 3584, and AttnRes with the same two 7% fewer. The
  measured cost is an upper bound, since kernels can improve. Its lower bound prices
  the algebra as `serving.py`'s analytic model does: the candidate at 3391 (9.13B
  parameters per token) as measured, 3629 (11.3B) at the model's price.
- **MLA runs at the kernel's dims.** Trained with a latent of 256 and 32 rotated
  dims, it runs on vLLM's Triton MLA, about half as fast as at DeepSeek's dims (512
  and 64) on FlashInfer's. Its target is DeepSeek's dims' match, counted in the
  trained architecture's parameters.
- **Some families are kernel-limited.** The sparse arms of the 2021 reference
  (`attention.bux`'s `dsa`, `sparse.bux`'s `msa` and `qsa`) attend on the plugin's own
  kernel, since vLLM's are slower at multi-head attention or do not take its heads, and
  CSA runs as exact PyTorch. GDN-2 runs on fla's kernels at 0.42 times the KDA hybrid's
  speed. Their timings measure those kernels, not the architecture, so they take the
  analytic model's target. The sparse arms of the candidate and of the hybrid run on
  vLLM's native kernels, and are measured (step 8).

Across 424 timed widths, measured sessions per hour come at a median of 1.04 times
the analytic model's, from 0.07 to 2.22; the farthest are the kernel-limited ones.

### FP8

Every target is measured in FP8, as a model of 2026 serves:

- **What runs in FP8.** vLLM's online FP8, per tensor: every projection and the
  routed experts take FP8 weights, scaled per tensor, and FP8 activations, scaled per
  token in the projections and per tensor in the experts. Every attention cache vLLM
  keeps is FP8, its scales uncalibrated. The router, the embeddings, the head, the
  norms, the convs and the recurrent states stay as vLLM keeps them.
- **FP8 shrinks the targets.** The budget model serves 1.94 times as many sessions
  in FP8 as in bf16, more than the families gain, so their FP8 targets hold a median
  0.89 times their bf16 parameters per token.
- **Finer scales cost speed.** Weights scaled per channel serve 10% to 13% fewer
  sessions than per tensor on the 2026 hybrid and the candidate. DeepSeek's blocks of
  128 serve 17% to 20% fewer on CUTLASS. vLLM 0.32's DeepGEMM kernel for them does
  not load on a B300.
- **FP8 passes the quality gate.** A target counts only if FP8 costs less than 0.3%
  of the bits per byte. Each trained `d1024` checkpoint is exported and scored in
  vLLM on the eval rows training scored, in bf16 and in FP8. bf16 in vLLM matches
  each run's logged eval within 0.01%, and FP8 costs:

| Family at `d1024` | code | repositories |
|-------------------|------|--------------|
| the 2026 hybrid, seed 1 | +0.070% | +0.106% |
| hybrid + hc, value residual, key-value conv, seed 0 | +0.068% | +0.097% |
| the same, seed 1 | +0.066% | +0.096% |
| hybrid + AttnRes, value residual, key-value conv | +0.076% | +0.110% |
| frontier + hc, value residual in every layer, key-value conv | +0.116% | +0.153% |
| candidate + copy pointer, seed 0 | +0.073% | +0.109% |

Every family passes by twice or more. The frontier's row is the highest, from its FP8
KV cache: its weights alone cost what the others' do. Weights per channel or by block
cost the same within 0.03 points, so per tensor stays the measurement.

`scripts/timeline/fp8_quality.py` takes this measurement in three steps, since bux
and vLLM have environments of their own. `prepare` exports the run and writes its
eval rows, in bux's environment. `run` scores them in vLLM's, once per configuration.
`report` prints each configuration's change from bf16, and checks bf16 against the
run's logged eval:

```sh
JAX_PLATFORMS=cpu python scripts/timeline/fp8_quality.py prepare runs/<run> fp8q/<run>
python scripts/timeline/fp8_quality.py run fp8q/<run>
python scripts/timeline/fp8_quality.py run fp8q/<run> --quantization fp8 --kv-cache-dtype fp8
python scripts/timeline/fp8_quality.py report fp8q/<run>
```

### The targets

```{include} timeline/tables/serving-fp8.md
```

Each timed width gives its best sessions per hour per GPU, at how many sessions at
once, on how many GPUs, and the analytic model's prediction: the 2024 Llama 3 at 4608
serves 1,707 sessions per hour at 32 at once, against the model's 1,665.

### Serving cost by context length

A coding agent deep in a large repository reads far more than 16k tokens. So the
contenders are timed at 64k, 128k and 256k too, each against the budget model at the
same context:

- **The load** grows each session to its context T. A session starts at two thirds
  of T: the shared system prompt, one of 4 repositories' files, and its own files
  and task. The system prompt and a repository's files make a quarter of T, shared by
  the sessions on that repository, as a team's agents share a monorepo. Its 4 turns
  write 512 tokens each, and its tool results take it to T.
- **The budget** falls with context: 634, 290 and 121 sessions per hour per GPU at
  64k, 128k and 256k.

`agentic.py --long-context T` runs this load, on `--repos` repositories (4 by
default). Each line also has the first turns' time to the first token,
`ttft_first_p50` and `ttft_first_p90`: the prefill of a cold session. At 64k:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/old_vs_new/agentic.py exported/llama3-4608 \
  --long-context 65536 --concurrency 8,16,32,64,128 --slo 30 \
  --engine 'quantization="fp8"' --engine 'kv_cache_dtype="fp8"'
```

Each family's target at each context, in parameters per token and its width:

```{include} timeline/tables/long-targets.md
```

- **Linear attention and windows pull ahead with context.** At a fixed width against
  the budget, the lean frontier serves 2.08, 3.12, 3.17 and 2.36 times the budget's
  sessions at 16-24k, 64k, 128k and 256k, and the hybrid 1.08, 1.54, 1.60 and 1.45
  times.
- **Full attention falls behind.** Llama 3's grouped-query attention serves 0.89
  times the budget at 16-24k and 0.45 times at 256k, so its target shrinks from
  13.5B parameters per token to 5.9B. DeepSeek-V3's MLA holds to 64k, then falls to
  5.0B at 256k.

A change's cost against its parent's, at the same width and each context, shows
which costs depend on context. In brackets, the change from its ratio at 16-24k:

```{include} timeline/tables/long-changes.md
```

- **Differential attention gets costlier with context**: its global layers have
  twice the query heads, so twice the attention's work. It loses 6%, 17% and 24% of
  its ratio at 64k, 128k and 256k, and the stacks that carry it lose as much.
- **Grouped-query and multi-query attention get costlier too**: against the 2021
  reference's target, theirs lose 25% to 67% from 64k on.
- **Windows, KDA, Mamba-2 and heads of 128 get cheaper** against the 2021
  reference: by 42% to 56% at 128k, and by 21% to 38% at 256k.
- **Changes priced per token keep their ratio**: hyper-connections with 2 copies,
  one KV cache for every global layer, the copy pointer, and the dense layer's
  position (step 8) stay within 10% of their 16-24k ratio at every context. The copy
  pointer with a window of 4096 serves 0.98 to 0.99 times the candidate from 64k
  on.
- **The 128k column is the noisiest**, about ±10%: its widths sit near the SLO's
  edge.

**Quality is not measured past 16k.** Every run trains its last tenth at 16k, and
its evals read at most 16k tokens. So these targets price a 256k session, not how
well a model trained at 16k works there: a window's or a NoPE global layer's reach,
or the copy pointer's, past its training length. Quality at 256k, with a training
phase that extends each contender to 256k, is a study of its own.

## 6. Fit and extrapolate

### The law

`scripts/timeline/fit.py` reads the run directories under `runs/`, or files of their
`RESULT` lines. A family at a rung is one point, at its best learning rate, its seeds
averaged on seed 0's scale. Runs that stopped
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
  2026 hybrid that the file uses. `--parents` names others in a JSON file, so a
  change built on the candidate of step 8 has the candidate as its parent.
- **A family with fewer than three rungs takes its parent's exponent.** It fits no
  exponent of its own, but its points weigh on its parent's, since every family
  fits at once ([step 8](#the-candidate-at-equal-serving-cost)).
- **E is shared by every family**: the data's entropy. `--fix-e` holds it at the
  values Old against new's ladders fit on the same data: 0.3204 bits per byte for
  the objective. These rungs do not pin E down: fit free, it lands at 0.2849, and
  the points cannot tell it from lower values. So the main numbers hold E, and the
  candidates' ranges reach to E fit.

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
  moved by its noise and its seeds' shifts. The candidates' ranges at their targets
  span E and the serving cost instead: from E held to E fit, and from the measured
  cost to its lower bound.

```sh
python scripts/timeline/fit.py runs --targets targets-measured.json --serve fp8 \
  --targets-longctx targets-measured-longctx.json --match B1b --mfu mfu.json \
  --root positions/rope --parents parents.json --drop-scan baseline --share B,beta \
  --fix-e objective=0.3204,code_bpb=0.2808,repo_bpb=0.2448,web_bpb=0.3832 \
  --e-free-variant --candidates factorial/r1v1k1f0p0s0,stack5,dense_last --out fit
```

It writes `fit/report.md`, every number in `fit/fit.json`, and the plots below.
`--serve fp8` takes the FP8 targets, and `--targets-longctx` the long-context ones.
`parents.json` maps every change built on the candidate to
`factorial/r1v1k1f0p0s0`.
`--match B1b` sizes any family that `targets-measured.json` lacks with the analytic
model. `--candidates` lists the families the report compares at their targets: this
page's are in step 8. `--selftest` fits synthetic runs of known laws instead.

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

- **With unlimited data, MFU decides as much as bits do.** Llama 3 trains at 0.65
  MFU and the 2026 families at 0.16, so Llama 3's dense target reaches 2020's loss in
  1.9 times fewer GPU-hours, against their 1.28 to 1.37. Qwen3-Next at 0.11 needs 1.2
  times fewer. GPT-NeoX, whose parallel blocks cost what its RoPE gains, is about
  even with 2020, and PaLM's 36B target needs 1.8 times more GPU-hours.
- **The dense milestones' gains shrink with width**: the baseline's learned
  positions catch up (step 9). The mixtures of experts keep most of theirs.

### The sliding-window frontier against the KDA hybrid

The two 2026 families are the consensus of two camps, scaled down. They differ in
the layers that see a bounded context: windows of 512 keys in the frontier, KDA in
the hybrid.

```{include} timeline/tables/swa.md
```

- **At the same width, the frontier serves 1.6 to 1.7 times as many sessions per
  GPU**: 3,547 against 2,075 at 3584 on one GPU, 1,614 against 985 at 4096 on two.
  Its cache and kernels are its architecture's own.
- **The hybrid holds 1.31 times the parameters per token at a width.** At equal
  cost, the frontier gets 9% more width and 26% more parameters in all, and the same
  11.3B per token.
- **At the same width, the frontier trails the hybrid**: paired by seed, it scores
  3.1%, 2.6%, 2.8% and 2.1% more bits per byte from `d512` to `d1280`.
- **The hybrid's winners close most of the gap.** The frontier with
  hyper-connections, the value residual in every layer and the key-value conv
  (`swa_hv1k`) gains 4.2% on the frontier at `d512` and 2.2% at `d1280`. That leaves
  it 0.12% under the hybrid at `d1280`, and 1.8% behind the candidate of step 8.
- **MiMo's lean windows of 128 with sinks** lose 2.6% on the frontier at `d512`, and
  serve 13% more sessions at 3584.
- **The frontier's global layers gain nothing from partial RoPE, and lose with MLA**
  (`western.bux`): partial RoPE makes no measurable difference (-0.29% at `d512`),
  and MLA is worse, +0.68% with its decoupled RoPE and +1.29% without.

At their targets on 1T tokens, the frontier predicts 0.33% more bits per byte than
the hybrid, `swa_hv1k` 0.36% fewer, and the lean frontier 0.57% more. At 64k and
128k the frontier's targets hold more parameters per token than the hybrid's, 14.8B
and 14.1B against 12.5B and 13.0B, and still it trails, by 0.17% and 0.25%
([step 8](#the-candidate-at-equal-serving-cost)).

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

```{include} timeline/tables/factorial.md
```

Paired by seed, a run's noise is 0.33% of the objective, against 0.80% unpaired.

- **Hyper-connections are the largest add-on** (t = -18.5), then AttnRes, the conv
  and the value residual. The four add up: their interactions are within the noise.
- **ReLU² experts are worse on the hybrid** (+0.68%), unlike on the reference.
  Beside `hc`, an interaction of -1.02% cancels that.
- **Partial RoPE on the global layers is slightly worse** (+0.17%, t = 2.0), and the
  DSA indexer does nothing measurable.

The candidate, the hybrid with `hc`, the value residual and the conv, holds its gain
from `d512` to `d1280`. Paired by seed, `fit.py` measures it 2.61 ± 0.20% under the
hybrid at `d512` over three seeds, and 2.16 ± 0.22%, 1.88 ± 0.16% and 1.84 ± 0.02% at
`d768`, `d1024` and `d1280` over two, each ± its standard error over the seeds. These
are the numbers this page quotes. `factorial.py`'s fit of the whole screen puts its
`d512` cell at -2.64 ± 0.27%. Its learning rate holds: at `d512` it scores 0.6661,
0.6607 and 0.6654 at 1e-3, 2e-3 and 4e-3.

### The candidate at equal serving cost

```{include} timeline/tables/best.md
```

Per parameter the candidate is worth 3.43 times the hybrid. At equal serving cost in
FP8 it predicts 0.44% fewer bits per byte on 1T tokens than the hybrid, 0.44% to
0.57% over E and the serving cost, and 8.95% fewer than 2020. With unlimited data it
reaches the hybrid's 1T-token loss in 1.46 times fewer GPU-hours. The predicted loss
at the two targets, on 1T tokens, in bits per byte, shows why the gain at the target
is smaller than at the rungs:

| Target | per token | loss | E | parameter term, A N^-α | data term, B D^-β |
|--------|-----------|------|---|------------------------|-------------------|
| the 2026 hybrid | 11.3B | 0.3560 | 0.3204 (90.0%) | 0.0064 (1.8%) | 0.0291 (8.2%) |
| hybrid + hc, value residual, key-value conv | 9.13B | 0.3544 | 0.3204 (90.4%) | 0.0048 (1.4%) | 0.0291 (8.2%) |

- **About 90% of the predicted loss is the floor E**, and 8% the data term on 1T
  tokens. Only the parameter term depends on the parameters, and it is all that a
  change in efficiency per parameter can move.
- **The hybrid's fitted exponent is steep**: α = 0.44, against the reference's 0.25.
  So its parameter term is already small at its target: under 2% of the loss.
- **The candidate's exponent is the hybrid's**, 0.44. Its parameter term at its
  target is 25% smaller than the hybrid's, though it holds 20% fewer parameters per
  token.
- **`hc` costs serving**: the candidate's target holds 9.13B parameters per token as
  measured, against the hybrid's 11.3B, which takes back part of the gain.

**How the fit ties the short ladders moves the number at the target.** Most changes
built on the candidate have one or two rungs, so they share its exponent, and the
fit weighs their points in it. The changes whose gains shrink fastest pull it down,
to 0.44. With those changes on the hybrid's exponent instead, the candidate's is
0.50, and it gains 0.96% at its target, not 0.44%. So its gain at the target is 0.4%
to 1.0%. The candidate, the hybrid with `hc` and the value residual alone, and mHC
in place of `hc` are within that spread of each other, and so are the copy pointer
and `stack5` below. At every rung, measured, the candidate leads the first and ties
the second.

The exponents rest on four rungs whose widest has 514M parameters per token. The result
holds for this hybrid's shape: 256 experts with 7 routed, KDA in three layers of four,
and the learning rate of `d512` carried to every width. Two residual streams in place of
`hc` are co-candidates:

- **mHC ties the candidate at every rung**: -0.24% at `d512` over three seeds, then
  +0.06%, +0.09% and +0.14% at `d768`, `d1024` and `d1280`. But it serves fewer
  sessions: its target holds 7.38B parameters per token, against the candidate's
  9.13B. At the target the two are within the fit's spread of each other.
- **AttnRes gains less, and less with width**: 0.64%, 0.74%, 1.13% and 0.56% behind
  the candidate from `d512` to `d1280`. It costs half as much to serve: 7% of the
  hybrid's sessions at 3584, against the candidate's 16%.

At longer contexts, the same laws at each context's targets predict, against the
hybrid's target there, in percent on 1T tokens:

```{include} timeline/tables/by-context.md
```

- **The candidate keeps its gain at every context**: 0.44% to 0.50% fewer bits per
  byte than the hybrid. So do mHC and AttnRes in its place, the copy pointer, and
  the dense layer last, at 0.94% to 1.00%.
- **The frontier stays behind the hybrid**, by 0.17% to 0.45%, though it serves more
  sessions at 64k and 128k: its parameters are worth less.
- **Full attention loses ground with context**: Llama 3 goes from 4.2% behind the
  hybrid to 6.2% at 256k, and DeepSeek-V3 from even to 1.1% behind.

These predictions carry quality measured at 16k or less to a 256k target
([step 5](#serving-cost-by-context-length)).

### New layers on the candidate

The candidate is the base for the new layers of 2026, and for the settings of bux the
study had not yet used. Each arm is one change on it, paired with it by seed: three
seeds at `d512`, and two at `d768` for the arms that gained. Each trains at the
candidate's rate, 2e-3, without a scan of its own. Over three seeds, a
paired difference's noise is about 0.27%. The objective's change against the
candidate, in percent:

| Change | Arm | `d512` | `d768` | `d1024` |
|--------|-----|--------|--------|---------|
| a copy pointer in the head (CPLM) | `newlayers_cplm[nl_cplm]` | -4.14 | -2.45 | -1.44 |
| the same, its window 4096 tokens | `newlayers_cplm[nl_cplm_w4k]` | -4.02 | -2.18 | |
| the embedding times √d | `ideas2[emb_sqrt]` | -0.73 | +0.79 | |
| GatedNorm, a gate of rank 16 after every norm | `ideas[gnorm]` | -0.71 | -0.49 | |
| the embedding normed | `ideas[emb_norm]` | -0.53 | | |
| a document-local copy feature | `newlayers_doc` | -0.51 | -0.10 | |
| value embeddings before every global layer | `newlayers[nl_ve]` | -0.35 | +0.02 | |
| each global head's output normed | `ideas[onorm]` | -0.30 | | |
| mimetic init of the global layers | `newlayers[nl_mimetic]` | -0.28 | | |
| differential attention V2 | `newlayers[nl_diff]` | -0.25 | -0.12 | |
| DeepSeek-V4.1's single-pass hc | `ideas2[hc_single]` | -0.25 | | |
| exclusive self-attention, its share learned | `newlayers[nl_xsa_learned]` | -0.20 | -0.07 | |
| the value residual's weights learned | `ideas[vres_learned]` | -0.13 | | |
| global heads of 128 | `ideas[ghead128]` | -0.08 | | |
| a loss on each target's prefix token | `newlayers_aux` | -0.08 | | |
| HyGA's gates | `newlayers[nl_hyga]` | -0.02 | | |
| Engram | `ideas[engram]` | -0.00 | | |
| a gate on the shared expert | `ideas[seg]` | +0.04 | | |
| Canon-C, a conv before the MLP | `canon[canon_c]` | +0.07 | | |
| one KV cache for every global layer | `ideas[kvshare]` | +0.18 | | |
| zero-computation experts | `newlayers[nl_null]` | +0.18 | | |
| paired heads | `newlayers[nl_paired]` | +0.19 | | |
| exclusive self-attention, its share fixed | `newlayers[nl_xsa]` | +0.41 | | |
| Canon-A and -C | `canon[canon_ac]` | +0.44 | | |
| MSAR in place of `hc` | `newlayers[nl_msar]` | +0.53 | | |
| Canon-A, a conv before the mixer | `ideas2[canon_a]` | +0.78 | | |
| zero-computation experts, 5 FFN experts of 8 | `newlayers[nl_null_lean]` | +1.05 | | |
| K=V in the global layers | `newlayers[nl_kveq]` | +1.16 | | |
| the same, without the value residual | `newlayers[nl_kveq_plain]` | +1.28 | | |

- **A gain under 1% at `d512` does not predict one at `d768`.** The embedding
  scaled by √d goes from -0.73% to +0.79%. Value embeddings, learned XSA,
  differential attention and the document-local copy keep half of their gain or
  less. Only GatedNorm holds most of its own, -0.71% to -0.49%. So a gain at `d512`
  counts here only once it holds at `d768`.
- **The copy pointer gains the most, and shrinks the fastest.** CPLM (modded-nanogpt,
  2026) gives the head a `<copy>` row, whose mass goes to a pointer over the
  document's earlier final hidden states. On repositories past 8k it scores 7.4%,
  4.7% and 2.8% fewer bits per byte at `d512`, `d768` and `d1024`. Its gain loses
  about 40% at each rung, faster than the candidate's own gain over the hybrid.
- **The pointer's gain is not a leak.** It is causal, it stays inside its document,
  and on data without repetition it gains nothing. A float64 recompute of its bits
  per byte agrees within 2e-5.
- **The pointer is cheap to serve**: its target holds 8.49B parameters per token in
  FP8, against the candidate's 9.13B, and its window of 4096 tokens 8.67B. Its cost
  does not grow with context (step 5), and its FP8 quality cost passes the gate.
- **At its target the pointer is within the fit's spread of the candidate.** On three
  rungs it fits its own exponent, 0.39, below the candidate's, as its gain shrinks.
  In this fit it predicts 0.38% fewer bits per byte than the candidate on 1T tokens.
  With the short ladders on the hybrid's exponent, it predicts 0.18% more.
- **The others lose or make no measurable difference at `d512`**: Canon convs in
  every placement, K=V, MSAR (also 2.1 times the step time, and 58% fewer sessions),
  the zero-computation experts, a fixed XSA, paired heads, HyGA, one KV cache for
  every global layer, Engram on top, the shared expert's gate, and the single-pass
  hc.

### Stacking the winners

Each stack sets the winners on the candidate together, its arms paired with it by
seed. The objective's change against the candidate, in percent:

| Stack | Arm | `d512` | `d768` | `d1024` |
|-------|------|--------|--------|---------|
| √d embedding, GatedNorm, output norm, learned value residual | `stack3[s3_e1g1o1l1]` | -1.17 | -0.12 | |
| the same without GatedNorm | `stack3[s3_e1g0o1l1]` | -0.73 | +0.74 | |
| value embeddings, learned XSA, differential attention | `stack4[s4_v1x1d1]` | -0.37 | -0.53 | |
| the same + GatedNorm | `stack4[s4_gvxd]` | | -0.84 | |
| copy pointer + GatedNorm | `stack5` | -4.84 | -2.79 | -1.78 |
| `stack5` + value embeddings, learned XSA, differential attention | `stack6` | -4.78 | -3.07 | |

- **The copy pointer and GatedNorm add up**: `stack5`'s -4.84% at `d512` is the sum
  of the two alone, and its -2.79% at `d768` 95% of it. It serves 7.86B parameters
  per token in FP8.
- **Value embeddings, XSA and differential attention add nothing at `d512`**, alone,
  in pairs or as three, and 0.28% on top of `stack5` at `d768`. Differential
  attention costs more with context, so `stack6` gives part of that back at 256k.

At their targets, against the candidate's on 1T tokens:

```{include} timeline/tables/stacks.md
```

- **`stack5` on three rungs** predicts 0.40% fewer bits per byte than the candidate
  at its target, 0.18% more with the short ladders on the hybrid's exponent. Its own
  exponent, 0.37, follows its shrinking gain. Its `d1280` runs decide.
- **The stacks on two rungs** take the candidate's exponent, so they cannot follow a
  shrinking gain: `stack5`'s variants predict 0.71% each, though at the rungs
  they are within 0.15% of `stack5`. Read them as upper bounds.
- **The dense layer at the end gains at the target too.** On three rungs, with an
  exponent of its own, `dense_last` predicts 0.53% fewer bits per byte than the
  candidate, 0.25% to 0.64% over E and the serving cost. Its gain grows from `d768`
  to `d1024`, on two seeds each, so its own exponent, 0.53, is the steepest of any
  candidate. Routing the first layer adds nothing: `dense0` +0.03%.

### Training options

Muon and NorMuon change the optimizer, not the model, so no stack carries them. They
are options beside the AdamW recipe, measured on their own:

- **Muon gains on the candidate**: 2.18% fewer bits per byte at `d512` over three
  seeds, 1.77% at `d768` on seed 0. **NorMuon** gains 2.30% at `d512`.
- **Muon gains on the 2021 reference**: 3.35% at `d512`, 2.08% at `d768`. Unlike the
  small changes to the architecture, these gains survive the step to `d768`.
- **Muon breaks the 2020 baseline's learned positions**: +6.7% on the objective at
  `d512`, and +66% on repositories past 8k.
- **The other options make no measurable difference**: an average of the last
  quarter's weights (-0.16%, but +1.6% past 8k), cautious weight decay, outputs that
  start at zero, and a decay that ends at a tenth of the peak.

### Where the dense layer goes

The candidate's first layer has a dense SwiGLU and every later one the experts, as
every 2026 model of the study, after DeepSeekMoE. `dense0.bux` routes the first
layer too, and `dense_last.bux` moves the one dense layer to the end. Each keeps the
candidate's active parameters, and pairs with it by seed:

- **Routing the first layer leaves the loss as it is**: -0.03% at `d512` over three
  seeds, +0.14% at `d768` over two.
- **It calms the gradient.** The candidate clips 9.9%, 9.0% and 11.1% of its steps
  at `d512`, seeds 0 to 2, and 1.5% and 2.2% at `d768`. Routed, it clips 2.5%, 1.2%
  and 0.2%, and none at `d768`. Its largest gradient norm is 5.8 to 9.0 at `d512`,
  and 1.2 to 2.2 routed. The dense first layer's SwiGLU has the largest gradient of
  any module, so it is most of the candidate's spiky gradient, more than `hc`
  (step 9).
- **The routed first layer starts unbalanced and ends the best balanced.** At
  `d512` its busiest expert takes 7.8 times its share at a tenth of the run, and 1.6
  times by 88% of it, the lowest of any layer. No expert dies.
- **The dense layer at the end gains at every rung**: -0.42% at `d512` (-0.43,
  -0.66, -0.17), -0.27% at `d768` (-0.31, -0.22) and -0.33% at `d1024` (-0.55,
  -0.11). It is better in 7 of 7 seeds, and -0.51% on repositories past 8k at
  `d1024`. Its gradient is calm too, 0% to 0.2% of steps clipped: the spikes come
  from a dense layer at the start.
- **Neither costs serving.** In FP8, `dense0`'s target holds 8.99B parameters per
  token and `dense_last`'s 9.12B, against the candidate's 9.13B. Both serve within 1%
  of the candidate's sessions at every context.
- **On `stack5`, routing the first layer is neutral**: -0.14% at `d512`, -0.02% at
  `d768`. GatedNorm already keeps `stack5`'s gradient calm: it clips no steps. The
  dense layer at the end makes no measurable difference on it either: -0.06% at
  `d512`, -0.05% at `d768`.

### Sparse attention on the candidate

`sparse_native.bux` puts each sparse attention on the candidate's global layers, in
the shape vLLM serves on kernels of its own: DSA as DeepSeek-V3.2's, QSA as
Qwen3.8's, MSA as MiniMax-M3's. Each keeps 2048 keys a query: all of a row of 2k,
and an eighth of one of 16k.

- **Quality is neutral**: against the candidate, DSA -0.13% and +0.15% at `d512` and
  `d768`, QSA +0.11% and +0.05%, MSA +0.11% and +0.05%.
- **At 16-24k they serve fewer sessions.** One layer in four is global, and its dense
  FP8 attention is cheap at that length, so the indexer costs more than it saves. At
  equal cost DSA holds 7.45B parameters per token, QSA 6.55B and MSA 6.36B, against
  the candidate's 9.13B. The time to the first token is where they pay.
- **At 256k, QSA serves more**: its target holds 13.3B, against the candidate's
  12.7B. MSA's holds 12.3B and DSA's 10.5B.

**Index sharing** (GLM-5.2's IndexShare, after IndexCache) runs the indexer in one
global layer of four (`is4_c`) or of two (`is2_c`), and the global layers up to the
next reuse its picks, with no indexer of their own (`indexshare.bux`). `is4x_c` adds
IndexCache's cross-layer loss: each layer that reads the picks also distills the
indexer towards its own attention. `indexshare_ref.bux` shares among every layer of
the 2021 reference, all sparse, as GLM-5.2's are.

- **Sharing serves more from 64k on.** At the same width against the candidate, one
  indexer in four serves 0.88 to 0.92 times its sessions at 16-24k, 1.03 times at
  64k, 1.40 at 128k and 1.10 at 256k; DSA in every global layer 0.86 to 0.90, 0.96,
  0.99 and 0.77. At 256k its time to the first token is the candidate's. At equal
  cost its target holds 7.8B, 11.6B, 14.8B and 13.2B parameters per token, against the
  candidate's 9.1B, 11.5B, 10.8B and 12.7B. The 128k point sits at a step of tensor
  parallelism and the SLO's edge, and is the noisiest.
- **Sharing trains faster**: a step of 16k at `d768` takes 1.3 s with one indexer in
  four, against 2.3 s for DSA in every global layer. The indexer is most of DSA's
  training cost at 16k.
- **One indexer in two keeps quality**: +0.12% at `d768` over two seeds.
- **One in four with the plain indexer loss fails at long positions on some seeds.**
  At `d768` seed 0 it is 1.17% worse than the candidate, and 6.5% worse on
  repositories past 8k, while seeds 1 and 2 are +0.20% and -0.09%. The cross-layer
  loss removes the failure: +0.11% at `d768` over three seeds, +0.09% past 8k.
- **On the reference, every layer sparse, the same holds.** One indexer in four is
  0.61% worse than DSA in every layer at `d512`, its seed 2 10.0% worse past 8k. At
  `d768`, on seed 0, it is 0.32% worse, and 4.8% worse past 8k. With the
  cross-layer loss no seed fails: it costs 0.80% at `d512`, and nothing at `d768`
  (-0.04%, and +0.49% past 8k).

These quality results are interim: one indexer in two has two seeds at `d768`, and
the reference one at `d768`. At their targets of 16-24k, against the
candidate's on 1T tokens:

```{include} timeline/tables/sparse-candidate.md
```

Sparse attention costs the candidate 0.14% to 0.27% at 16-24k, where it serves fewer
sessions, and nothing at 256k. There QSA predicts what the candidate does, 0.47%
under the hybrid, and DSA 0.12% less. Index sharing matches the candidate from 64k
on, and leads it by 0.12% at 128k, the noisiest point.

## 9. Results and caveats

### Each change alone

The appendix lists every change against its parent. Beyond the summary at the top:

- **The positions lead the dense changes at the rungs, and fade.** Every RoPE arm
  with YaRN or a raised base is worth 2.0 to 2.4 times learned positions per
  parameter. The learned table catches up with width: the baseline is 7.5% worse
  than the reference at `d512`, 2.4% at `d1024`, and the fit puts it about even at
  the target.
- **Linear-attention mixers gain both ways**: 1.2 to 2.2 times the parameters, and
  targets of 6B to 15B parameters per token, against the reference's 3.7B.
- **The residual streams gain on the reference, and shrink with width.** Paired by
  seed, `hc` measures 2.44%, 2.20% and 1.27% fewer bits per byte at `d512`, `d768`
  and `d1024`, the gated residual stream 2.97% and 1.42% at `d512` and `d768`, and
  mHC 2.82% at `d512`. At the target, `hc` alone ends at +0.19% on 1T tokens: its
  own exponent, from three rungs, follows its shrinking gain. The gated residual
  stream (-1.38%) and mHC (-1.99%) have two rungs and one, so they take the
  reference's exponent and keep their gain.
- **RoPE at base 500k flips sign at the target** too, to +0.52% against the baseline
  on 1T tokens, and the value residual's gain on the reference fades to nothing
  (+0.03%): both shrink fast with width.

### Long context

The report's long-context table compares each family's bits per byte past 8k tokens
with those of a repository's first 2k, against its parent, averaged over the rungs:

```{include} timeline/tables/long.md
```

- **YaRN is what makes RoPE use 16k**: without it, RoPE gains on the baseline in a
  repository's first 2k (3.9%) but hardly past 8k (0.8%). At the switch to rows of
  16k, its loss jumps 0.10 nats against the reference's, and at `d512` it ends 10.7%
  worse past 8k.
- **Learned positions catch up with width**: 6 points behind the reference at
  `d512`, 1 at `d1280`.
- **A window as long as the training rows hurts.** `swa_2048` loses 11% past 8k:
  in the 16k phase its local layers still see 2048 keys.
- **CSA loses at long context.** Against the reference, it is 8.6% worse in a
  repository's first 2k and 16.3% past 8k. MSA matches the reference at both, +0.10%
  and +0.56%.
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
at its equal-cost width of 3633, the hybrid has about 57 layers. Deep and thin at equal
parameters (`deep_thin`) makes no measurable difference; shallow and wide is
untested.

### Short notes

- **DyT fails at this init.** With embeddings drawn at 0.02, `tanh(αx)` reads inputs
  far below 1: DyT is 74% worse at `d512`, and 3.1% worse with the embedding scaled
  by √d.
- **KDA in seven layers of eight is slightly worse than in three of four** (×1.70
  against ×2.15 per parameter), and KDA in every other layer, ×1.72.
- **The routing choices of the frontier's experts make no measurable difference**:
  softmax, no shared expert, SMEBU, or a square-root softplus (`zoo_moe.bux`).

### Stability, outliers and routing

Every run's observed metrics, paired by seed, say more than its loss.

- **Loss spikes do not separate the arms at the working rate.** 141 of 469 runs have
  one of 0.08 nats or more, and only 4 one of more than 0.25; all recover. Only the
  arms that fail outright (NoPE, DyT, post-branch norms without QK-norm) spike again
  and again.
- **The candidates are as stable in loss as the hybrid**, with two differences:
  - their gradient is spiky: at `d512`, 6% to 10% of steps clip at a norm of 1.0,
    against 0.08% for the hybrid, and 0.8% to 2.3% at `d768` and `d1024`. Most of it
    comes from the dense first layer under `hc`: with that layer routed, the
    candidate clips 0.2% to 2.5% at `d512` ([step 8](#where-the-dense-layer-goes));
  - `hc`'s stream has 1.5 to 3.5 times the hybrid's rms, the ratio shrinking with
    width.
- **Aux-free routing balances, but slowly.** Its worst layer takes 40% to 60% of the
  run to settle, and runs with a plain residual stream leave some experts dead on
  the way: all 5 of the hybrid's, up to 7.4% of a layer. With `hc`, none of 14
  candidate runs has a dead expert.

The late-2026 families have the smallest activation outliers, which matters for
FP8. The 2020 baseline, without QK-norm, has attention logits of 240 to 560. Each
cell is the stream's crest and the largest attention logit, before the long phase:

| Family | `d512` | `d1024` | `d1280` |
|--------|--------|---------|---------|
| 2020 baseline | 84 / 240 | 97 / 559 | 157 / 381 |
| 2021 reference | 131 / 94 | 228 / 171 | 353 / 188 |
| 2024 Llama 3 | 159 / 37 | 285 / 57 | 381 / 106 |
| 2024 DeepSeek-V3 | 49 / 67 | 99 / 54 | 125 / 936 |
| 2025 Qwen3-Next | 36 / 22 | 77 / 29 | |
| 2026 sliding-window frontier | 39 / 26 | 99 / 32 | 120 / 36 |
| 2026 linear-attention hybrid | 41 / 24 | 82 / 31 | 120 / 34 |
| hybrid + hc, value residual, key-value conv | 44 / 25 | 84 / 30 | |

DeepSeek-V3's 936 at `d1280` comes from one layer whose attention entropy collapsed
for the whole run. The crest grows about linearly with width in every family, so the
targets will have larger outliers than the rungs. Only the residual stream is
observed, not the quantized matmuls' inputs or the keys and values. FP8's cost in bits
per byte is measured on the trained checkpoints instead, and is under 0.16% for every
family measured ([step 5](#fp8)).

### Caveats

- **Learning rates transfer from `d512`.** Every family trains its wider rungs at its
  `d512` vertex. The most fragile, one rate step above their best, lose 3.7% to 5.8%
  (the baseline, sandwich norms, `hc`, GeGLU) against the reference's 1.3%, so
  they may sit off their optimum at wider rungs.
- **Seeds are shared shifts.** Paired by seed, a difference is 2 to 3 times sharper
  than unpaired. Most points past `d512` are one run, on seed 0.
- **Rungs differ by family.** 22 families have four rungs (the reference, the
  baseline, four milestones and 16 arms), 21 three, 75 two, and 73 one, most of the
  factorial among them. A family with fewer than three takes its parent's exponent,
  so its gain is one multiplier at every scale.
- **E is held at 0.3204.** Fit free, it lands at 0.2849, and moves the candidate's
  gain on the hybrid at 1T from 0.44% to 0.48%. The candidates' ranges carry it.
- **The targets are far.** A dense target at the reference's cost has 3.7B parameters
  per token, 43 times `d768`'s; the 2026 families' have 9.1B to 11.3B. The ranges
  are not confidence intervals.
- **Serving is measured on one load shape, with dummy weights.** A router's choices
  on random weights need not spread as trained ones do. Kernel-limited families take
  the analytic model's targets.
- **Quality is measured at 16k or less.** Serving is measured to 256k, but no run
  trains or evaluates past 16k ([step 5](#serving-cost-by-context-length)).
- **MFU is measured on fewer layers for the largest targets**, and the one-change MoE
  arms take the reference's dense MFU, so their unlimited-data speedups are upper
  bounds.
- **Some rungs do not fit or are too slow.**
  - At 16k rows, Qwen3-Next's attention heads of 256 need
    `methods = methods(remat=full, microbatches=2)` on one GPU at `d768`, and a mesh
    of 4 GPUs with `microbatches=3` at `d1024`.
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
- Extend the contenders to 256k with a training phase of their own, and measure their
  quality past 16k: this page prices 256k sessions but measures quality at 16k.
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
