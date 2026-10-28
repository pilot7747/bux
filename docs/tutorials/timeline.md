(tutorial-timeline)=
# 2020 to 2026, one change at a time

What has each change to the decoder since GPT-3 been worth? Papers claim gains for
RoPE, QK-norm, SwiGLU, grouped-query attention, mixtures of experts, linear-attention
hybrids and dozens more, each on its authors' own setup. This tutorial measures them
on one setup: agentic coding with long context, at the same serving cost. It starts
from GPT-3's decoder of 2020 and changes one thing at a time. Every variant trains on
a ladder of small widths, and a scaling law extrapolates it to the size that serves
agents as fast as a GPT-3 6.7B. Then the changes add up into the milestones of each
year, from GPT-NeoX to the hybrids of late 2026.

By the end you will have the answer. {TBD: the headline. For example: "The late-2026
hybrid reaches the 2020 baseline's bits per byte in X times fewer B300 GPU-hours,
and on the same 1T tokens it predicts Y% fewer. Of the 50 single changes, N are worth
more than the noise at scale: ..."}. Getting there takes seven steps:

1. Train on code and whole repositories, ending at 16k tokens with YaRN.
2. Size four rungs, and scan each variant's learning rate on the smallest.
3. Sweep one change at a time, from the 2020 baseline or the 2021 reference.
4. Test the claims of stability at high learning rates.
5. Find each variant's target: the size at which it serves as fast as the budget.
6. Fit the ladders and extrapolate, with unlimited data and with limited data.
7. Stack the changes into milestones, and read the results and their caveats.

This tutorial builds on [Old against new at the same serving cost](old-vs-new.md):
its data, its agentic load and its fits. Do that one first. This one trains a few
hundred runs, on nodes of 8 GPUs ([Time and hardware](#time-and-hardware)). Every
measurement here was taken on NVIDIA B300s, {TBD: dates}. The serving costs come from
a model fitted to measurements (step 5).

## 1. The recipe

Every run trains as Old against new's do: code files, whole Python repositories
and web text in rows of 2k, then the last tenth in rows of 16k. Get the data as
[step 3 of Old against new](old-vs-new.md#3-get-the-data) does, at the same paths.
`examples/experiments/recipes/timeline_code.bux` is the recipe, a module with no
model:

```bux
tokenizer  = tokenizer(name="EleutherAI/gpt-neox-20b")
web        = text(path="data/fineweb-edu/sample/10BT/*.parquet")
code       = text(path="data/stack-edu/files/train/*.parquet")
code_eval  = text(path="data/stack-edu/files/eval/*.parquet")
repos      = text(path="data/stack-edu/repos/train/*.parquet")
repos_eval = text(path="data/stack-edu/repos/eval/*.parquet")
packing    = best_fit

data    = 0.6 * code + 0.4 * web(split=[0, 0.999])
seq_len = 2048
batch   = 256
budget  = 10B                            # tokens
steps   = budget // (batch * seq_len)

long       = 0.5 * repos + 0.3 * code + 0.2 * web(split=[0, 0.999])
long_phase = phase(at=0.9, seq_len=16384, batch=batch // 8, data=long)
yarn_phase = long_phase(rope_scaling=yarn(factor=8, original_max_len=2048))
curriculum = yarn_phase

optimizer = adamw(lr=2e-3, weight_decay=0.1)
schedule  = wsd(warmup=0.05, decay=0.2, shape=sqrt)
loss      = loss(z=1e-4)

eval = [eval(name="code", data=code_eval, every=2k, batch=64, batches=40),
        eval(name="web", data=web(split=[0.999, 1]), every=2k, batch=64, batches=40),
        eval(name="repo", data=repos_eval, every=2k, seq_len=16384, batch=8, batches=16,
             positions=[2048, 8192])]
objective = 0.5 * code_bpb + 0.3 * repo_bpb + 0.2 * web_bpb

observe      = observe(every=500)
log.trackers = [wandb(project="bux-timeline"),]
```

It is `agentic_code.bux` with the three changes below and a W&B tracker. Its
learning rate, 2e-3, is only a default: every variant scans its own (step 2).

**The long phase switches RoPE to YaRN.** A model with RoPE that trains on rows of
2k has seen its slow channels turn only as far as 2048 positions take them. In rows
of 16k they turn eight times as far, to angles it has never seen. YaRN divides the
slowest frequencies by `factor`, so that 16k positions take the angles that 2k took.
It keeps the fastest ones, which carry local order, and ramps between the two
([Long-context extension](../training/curriculum.md#long-context-extension)). The
phase switches every full-attention layer that rotates; layers with a window, or
without RoPE, keep theirs. `positions.bux`'s `rope_noyarn` arm measures what YaRN
adds (step 3).

**The repository eval logs bits per byte by position.** `positions=[2048, 8192]`
cuts its targets into three buckets by how much of their document comes before them
([By position](../training/running.md#by-position)). `repo_bpb_0` is the targets in
the first 2k tokens of a repository, `repo_bpb_2k` those from 2k to 8k, and
`repo_bpb_8k` those past 8k. A model that uses long context predicts the later
buckets better than the first.

**It observes the model.** `observe(every=500)` logs, among others, each attention's
largest logit and its entropy ([Observing a run](../training/observing.md)). Step 4
reads them.

Every run also logs to the W&B project `bux-timeline`. Add `'log.trackers = []'` to a
command to turn that off.

## 2. Size the rungs and scan the learning rate

### Rungs at the same tokens

A rung is a width, and every variant trains at the same four.
`examples/experiments/timeline/rungs.bux` zips them by their labels:

```bux
d_model = d512: 512  | d768: 768  | d1024: 1024  | d1280: 1280
budget  = d512: 503M | d768: 1.7B | d1024: 4.03B | d1280: 7.86B
batch   = d512: 48   | d768: 96   | d1024: 192   | d1280: 288
mesh    = d512: mesh(data=1) | d768: mesh(data=1) | d1024: mesh(data=2) | d1280: mesh(data=4)
```

Each rung's budget is 20 tokens for every parameter of the 2020 baseline at that
width besides its embeddings: 25.2M at `d512`, 85.1M at `d768`, 201M at `d1024` and
394M at `d1280`. Every variant at a rung reads those same tokens, whatever its own
size. A variant that holds more parameters gets no more data, so a rung compares the
variants at equal data. Their sizes go into the fit (step 6). The batch follows
DeepSeek LLM's batch-size law, as in [Old against new](old-vs-new.md#5-train-the-ladders),
and the long phase reads an eighth as many rows of 16k.

`examples/experiments/timeline/baseline.bux` is the 2020 baseline at every rung:

```bux
use gpt3_2020
use timeline_code
use rungs
```

`rungs` comes after the recipe, so that its settings replace the recipe's. `bux check`
shows the first rung, the long phase and its YaRN:

```console
$ bux check examples/experiments/timeline/baseline.bux
d_model=512 vocab_size=50304 layers=35
layernorm×17  gqa×8  mlp×8  embed×1  head×1
5,116 steps = 503M tokens
from step 0: 48 × 2048 tokens
phase 1, from step 4,604: 6 × 16384 tokens, data=0.5 * text(path="data/stack-edu/repos/train/*.parquet") + 0.3 * text(path="data/stack-edu/files/train/*.parquet") + 0.2 * text(path="data/fineweb-edu/sample/10BT/*.parquet", split=[0.0, 0.999]), rope_scaling=yarn(factor=8.0, original_max_len=2048)
optimizer=adamw(lr=0.002) schedule=wsd(warmup=0.05, shape=sqrt)
data=0.6 * text(path="data/stack-edu/files/train/*.parquet") + 0.4 * text(path="data/fineweb-edu/sample/10BT/*.parquet", split=[0.0, 0.999])
tokenizer=tokenizer(name="EleutherAI/gpt-neox-20b") packing=best_fit
sweep: 4 runs over d_model (d512 | d768 | d1024 | d1280)
```

```console
$ bux plan examples/experiments/timeline/baseline.bux
4 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
234c002192  59.4M   503M    8.5        494M       1     226MiB      453MiB     0%      baseline[d512]
46726c506b  136M    1.7B    12.5       1.09B      1     520MiB      1.02GiB    0%      baseline[d768]
27cf3dbf3e  270M    4.03B   14.9       2.07B      2     1.01GiB     2.01GiB    0%      baseline[d1024]
9ce8da37d8  479M    7.86B   16.4       3.54B      4     1.78GiB     3.57GiB    0%      baseline[d1280]
```

`tok/param` counts every parameter, the embeddings and the baseline's table of 16k
learned positions too. `bux prepare` of any timeline file packs the datasets that
all of them read.

### A learning-rate scan at the smallest rung

Each variant trains at its own learning rate. One rate for all would favor the
variants it suits. Every variant scans three rates at `d512`, as one sweep from the
command line:

```sh
bux launch examples/experiments/timeline/norms.bux --only "[qknorm, d512" \
  'optimizer.lr = 1e-3 | 2e-3 | 4e-3'
```

`--only` keeps the runs whose name contains its text: here
`norms[qknorm, d512, optimizer.lr=1e-3]` and its two neighbors. The arms that claim
stability scan 8e-3 too (step 4), with `'optimizer.lr = 8e-3'`.

The wider rungs train at the vertex of a parabola in log learning rate, fitted by
least squares through the scan's points within 3% of its best. Where the parabola
has no minimum inside the scan, they take the best point. In
[Old against new](old-vs-new.md#4-tune-each-familys-learning-rate), the 2022
family's best rate did not move with width under bux's default init, a normal of
0.02 at every width ([Initialization](../architectures/initialization.md)). So a
scan at `d512` serves every rung. QK-norm's four points, from 1e-3 to 8e-3, put its
vertex at 2.26e-3:

```sh
bux launch examples/experiments/timeline/norms.bux --only "[qknorm, d768]" \
  'optimizer.lr = 2.26e-3'
```

A setting of one value adds no label, so this run is `norms[qknorm, d768]`.

% TBD: no script in the repository prints the vertex yet; fit.py or a helper could.

## 3. One change at a time

### The axis files

An axis file sweeps one part of the model: a labeled arm for each way to change it.
`examples/experiments/timeline/positions.bux` sweeps how attention sees order:

```bux
use gpt3_2020
use timeline_code
use rungs

layer(a) = [norm a] [norm ffn]
lm(a)    = embed layer(a)^layers norm head(tied=true)   # the baseline, no learned table
R = layer(attn(rope=rope()))
N = layer(attn)

model = base:             model
      # 2021, RoFormer, GPT-J, GPT-NeoX: relative positions by rotation; extrapolation
      | rope:             lm(attn(rope=rope()))
      # 2021 GPT-J/NeoX, 2025 Qwen3-Next: a quarter of the dims; long context
      | rope_partial:     lm(attn(rope=rope(fraction=0.25)))
      # 2024 Llama 3: base 500k; long context
      | rope_500k:        lm(attn(rope=rope(base=500000)))
      # 2025 Qwen3-Next, 2026 Qwen3.8: base 10M on a quarter of the dims; long context
      | rope_10m_partial: lm(attn(rope=rope(base=10000000, fraction=0.25)))
      # 2023 Kazemnejad et al., 2025-26 Kimi Linear, Nemotron 3: no positions; length
      # generalization
      | nope:             lm(attn)
      # 2025 SmolLM3, Llama 4's iRoPE: NoPE every 4th layer, RoPE on the rest; long context
      | irope:            embed "RRRN"^(layers // 4) norm head(tied=true)
      # 2026 Beam 1.0: NoPE, queries scaled by (1 + n / 2048) ** 0.1; sharp at long range
      | nope_power:       lm(attn(power_scale=0.1))
      # 2026 Inkling: a learned relative bias over 1024 offsets; extrapolation
      | relbias:          lm(attn(relative_bias=relbias(rank=16, extent=1024)))
      # RoPE as in `rope`, but the long phase keeps it as trained: what YaRN adds
      | rope_noyarn:      lm(attn(rope=rope()))
      # 2023 Xiong et al., 2024 Llama 3 (ABF): the long phase raises RoPE's base to 500k
      # instead of YaRN
      | abf:              lm(attn(rope=rope()))

curriculum = base: yarn_phase | rope: yarn_phase | rope_partial: yarn_phase
           | rope_500k: yarn_phase | rope_10m_partial: yarn_phase | nope: yarn_phase
           | irope: yarn_phase | nope_power: yarn_phase | relbias: yarn_phase
           | rope_noyarn: long_phase | abf: long_phase(rope_base=500000)
```

- **Each arm changes one thing** and keeps the rest of the baseline. `base` is the
  baseline itself.
- **The comment above an arm** says its era and source, what it changes, and what it
  was claimed for, after the `;`. `fit.py` reads it into its leaderboard's columns
  (step 6).
- **`curriculum` zips with `model` by label**, so the two arms that test the
  extension get a long phase of their own: no YaRN for `rope_noyarn`, Llama 3's
  raised base for `abf`.

Eleven arms at four rungs make 44 runs; `bux plan` lists them. Without YaRN, RoPE
loses at long context: at `d512`, `rope_noyarn` predicts the targets past 8k no
better than those from 2k to 8k (0.617 against 0.620 bits per byte). With YaRN,
`rope` predicts them at 0.563, below its 0.581 from 2k to 8k.

% Preliminary: single runs at d512, lr 2e-3.

Nine more files sweep the rest of the model, and one tests pairs. Every file besides
`positions.bux` starts from the 2021 reference, the baseline with RoPE, for the
reasons the next section gives:

| File | Starts from | Arms |
|------|-------------|------|
| `positions.bux` | the 2020 baseline | `rope`, `rope_partial`, `rope_500k`, `rope_10m_partial`, `nope`, `irope`, `nope_power`, `relbias`, `rope_noyarn`, `abf` |
| `norms.bux` | the 2021 reference | `rmsnorm`, `qknorm`, `nobias`, `sandwich`, `postnorm`, `sandwich_depth`, `softcap` |
| `attention.bux` | the 2021 reference | `head128`, `gqa4`, `mqa`, `mla`, `swa`, `gated`, `sinks`, `kvconv`, `dsa` |
| `ffn.bux` | the 2021 reference | `swiglu`, `geglu`, `relu2`, `situ` |
| `moe.bux` | the 2021 reference | `moe_8x2`, `moe_fine`, `moe_dsv3`, `moe_latent` |
| `mixers.bux` | the 2021 reference | `mamba2`, `gdn`, `kda`, `gdn2`, `rwkv7`, each in three layers of four |
| `embeddings.bux` | the 2021 reference | `untied`, `mtp`, `engram`, `value_res` |
| `residuals.bux` | the 2021 reference | `hc`, `mhc`, `attnres`, `gr`, `vpgate` |
| `structure.bux` | the 2021 reference | `parallel`, `deep_thin` |
| `interactions.bux` | the 2021 reference | `olmo2`, `nope_qk`, `nope_power_qk`, `qknorm_fixed`, `attn_softcap`, `gated_qk`, `kvconv_qk` (step 4) |

The mixtures of experts send every token through as much MLP as the reference's, so
they change how a token's MLP is chosen, not how much it computes. Some changes are
not arms. Zero-centered gains train as plain ones, since bux's weight decay leaves
every gain alone. ALiBi, differential attention and value embeddings have no layer
in bux.

### The 2020 baseline and the 2021 reference

`examples/experiments/timeline/gpt3_2020.bux` is GPT-3's decoder at any width:

```bux
d_model    = 768
vocab_size = 50304
layers     = d_model // 64

norm  = layernorm(eps=1e-5, bias=true)
attn  = gqa(num_heads=d_model // 64, num_kv_heads=d_model // 64, bias=true)
ffn   = mlp(intermediate_size=4 * d_model, activation=gelu, bias=true)
block = [norm attn] [norm ffn]
model = embed(max_positions=16384) block^layers norm head(tied=true)
```

It has LayerNorms and projections with biases, multi-head attention in heads of 64,
a GELU MLP four times as wide as the stream, and a head tied to the embedding. There
is a layer for every 64 of the width, so `d_model = 768` is GPT-3 small. Its
positions are a learned table of 16k, because every run ends with rows of 16k.

That table is the baseline's weak point at long context. Its entries past 2048 train
only in the last tenth of the run, and only where a document runs that long. What
they learn there depends on how those few steps go. The scan at `d512` shows it:

| Learning rate | baseline: objective | `repo_bpb` | `repo_bpb_8k` | reference: objective | `repo_bpb` | `repo_bpb_8k` |
|---------------|---------------------|------------|---------------|----------------------|------------|---------------|
| 1e-3 | 0.8963 | 0.7950 | 0.7879 | 0.7495 | 0.5902 | 0.5630 |
| 2e-3 | 0.8017 | 0.6527 | 0.6321 | 0.7476 | 0.5882 | 0.5625 |
| 4e-3 | 0.8478 | 0.7898 | 0.8229 | 0.7572 | 0.5971 | 0.5708 |
| 8e-3 | 0.8755 | 0.7835 | 0.7976 | | | |

The baseline's code files move smoothly with the rate (0.796, 0.716, 0.724, 0.769
bits per byte). Its repositories jump between 0.65 and 0.80. At 4e-3 it predicts the
targets past 8k worse than those in a repository's first 2k (0.823 against 0.765): the
longer context hurts it. Its token scan (step 6) breaks the same way: on 40 tokens
per parameter it scores worse than on 20 (0.8632 against 0.8017), and on 80 it
diverges.

The **2021 reference**, `gpt3_rope.bux`, is the baseline with RoPE in place of the
table, as GPT-J and GPT-NeoX had it. It is `positions.bux`'s `rope` arm:

```bux
use gpt3_2020
attn  = gqa(num_heads=d_model // 64, num_kv_heads=d_model // 64, bias=true, rope=rope())
model = embed block^layers norm head(tied=true)
```

Its repositories follow its code files, and its long context works at every rate.
So every axis besides positions starts from the reference: a change to the norms or
the MLP is then measured on a model whose long-context numbers mean something.
`positions.bux` alone starts from the baseline, and its `rope` arm links the two.

The `base` arm of every other axis file is the reference again. `fit.py` counts it
as the reference's run, so train each arm by name and skip it:

```sh
for arm in head128 gqa4 mqa mla swa gated sinks kvconv dsa; do
  bux launch examples/experiments/timeline/attention.bux --only "[$arm, d512" \
    'optimizer.lr = 1e-3 | 2e-3 | 4e-3'
done
```

Each launch takes the GPUs of its mesh, and the next waits until there is room.

### Noise, and the third rung

Two runs of one point differ by their seed alone, which sets the model's init and
the data's order. Three more seeds of the reference at `d512` measure that noise, σ:

```sh
bux launch examples/experiments/timeline/positions.bux --only "[rope, d512" \
  'optimizer.lr = 2e-3' 'seed = 1 | 2 | 3'
```

`fit.py` averages the runs of a point at one learning rate and reports their pooled
spread. Here σ is {TBD: σ, relative} of the objective.

Every arm trains `d512` (its scan) and `d768`. An arm trains `d1024` as well when
either holds:

- **its gain at `d768` exceeds 2σ**, so it is a real change whose trend with scale
  matters;
- **its target has at least 1.5 times the reference's active parameters** (step 5),
  so its extrapolation reaches far and needs an exponent of its own.

Three rungs give a family its own exponent in the fit (step 6). The baseline, the
reference and the milestones train all four rungs. {TBD: the arms that pass, and
how many runs that makes.}

## 4. Test the stability claims

Several arms were introduced for stable training at high learning rates: QK-norm,
sandwich norms, depth-scaled sandwich norms, post-branch norms, the head's soft cap,
gated attention, and the doubly stochastic hyper-connections (`mhc`, beside the plain
`hc`). Their scans at `d512` go on to 8e-3, four times the reference's best:

| Arm | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|-----|------|------|------|------|
| reference | 0.7495 | 0.7476 | 0.7572 | {TBD} |
| `qknorm` | 0.7453 | **0.7389** | 0.7449 | 0.7489 |
| `softcap` | 0.7526 | 0.7456 | 0.7524 | 0.7829 |
| `sandwich` | 0.7560 | 0.7511 | 0.7907 | 0.8396 |
| `sandwich_depth` | 0.7564 | 0.7518 | 0.7735 | 0.7941 |
| `postnorm` | 0.8905 | 1.4260 | 1.6492 | 2.9909 |
| `gated` | 0.7420 | 0.7318 | 0.7535 | 3.0718 |
| `hc` | 0.7360 | 0.7229 | 0.7546 | 1.0702 |
| `mhc` | 0.7488 | 0.7419 | 0.7763 | 0.9165 |

% Preliminary: single runs.

- **QK-norm is flat to 8e-3.** Its objective rises 1.4% from its best to 8e-3; the
  reference's rises 1.3% from 2e-3 to 4e-3 alone. At its own best rate it gains 1.2%
  over the reference at `d512` and 1.1% at `d768`.
- **The reference does not diverge; its attention sharpens too fast.** By step 500
  its largest attention logit (`attn/logit_max`) reaches 71 at 2e-3 and 351 at
  4e-3. The entropy of its middle layers (`attn/entropy`) falls to 0.22 and 0.03,
  near one key per query. It recovers within 2k to 3k steps, so it has no loss
  spikes for a norm to remove. With QK-norm, the logits at step 500 are 12 to 25 at
  every rate, and the entropy 0.40 or more. Its queries and keys stay unit-scaled,
  so AdamW's updates turn them but no longer grow the logits. Its gain is the steps
  the reference loses early, and a wider basin (Dehghani et al., 2023; Wortsman et
  al., 2023).
- **Sandwich norms and the soft cap do not help alone.** None of them touches the
  product of queries and keys: `sandwich`'s logits still reach 121 to 227, and the
  soft cap is on the head's logits. From 4e-3 up, the sandwich arms are worse
  than the reference.
- **Post-branch norms fail at every rate.** `x + norm(f(x))` scores 0.8905 at 1e-3,
  19% worse than the reference, and worse above. OLMo 2, which ships it, normalizes
  queries and keys as well.
- **Gated attention and hyper-connections gain at 2e-3, and lose it at 8e-3.** Their
  logits still grow: the gated arm's reach 501 at 4e-3.
- **NoPE diverges.** Without positions, layer 4's largest logit goes from 120 to 896
  to 3.1e5 between steps 2000 and 3000, and the loss spikes from 3.0 to 5.3. The arm
  ends above 1.6 bits per byte at every rate, against 0.75 for the reference. The
  models that ship NoPE also normalize queries and keys.

`examples/experiments/timeline/interactions.bux` takes QK-norm apart, and pairs it
with what fails without it:

```bux
model = base:          model
      # 2024 OLMo 2: the output alone normalized, x + norm(f(x)), with QK-norm; stability
      | olmo2:         lm([attn(qk_norm=rms) norm] [ffn norm])
      # 2025 Kimi Linear, 2026 Nemotron 3: no positions, with QK-norm; length generalization
      | nope_qk:       lm([norm attn(rope=none, qk_norm=rms)] [norm ffn])
      # 2026 Beam 1.0: NoPE, queries scaled by (1 + n / 2048) ** 0.1, with QK-norm
      | nope_power_qk: lm([norm attn(rope=none, power_scale=0.1, qk_norm=rms)] [norm ffn])
      # QK-norm without its learned gains: is a learned temperature what helps?
      | qknorm_fixed:  lm([norm attn(qk_norm=rmsnorm(eps=1e-5, affine=false))] [norm ffn])
      # 2024 Gemma 2: attention logits soft-capped at 20, QK-norm's range: is a bound enough?
      | attn_softcap:  lm([norm attn(softcap=20)] [norm ffn])
      # 2025 Qwen3-Next, 2026 Kimi K3: gated attention and QK-norm together
      | gated_qk:      lm([norm attn(gated=true, qk_norm=rms)] [norm ffn])
      # QK-norm and Inkling's key-value conv together
      | kvconv_qk:     lm([norm attn(kv_conv_size=4, qk_norm=rms)] [norm ffn])
```

If `attn_softcap` matches `qknorm`, bounded logits are the whole gain; if it falls
short, the scale-invariant queries and keys are. If `qknorm_fixed` loses, the learned
temperature matters. {TBD: what each pair found.}

## 5. Size the targets at equal serving cost

Each variant extrapolates to its **target**: the width at which it serves the
agentic load of [Old against new](old-vs-new.md#2-measure-the-serving-cost) as fast
as a budget. A change that shrinks the KV cache buys a wider target; a change that
costs more per token buys a narrower one.

### The serving model

Pricing every variant in vLLM would take an export and a benchmark at several widths
each. `scripts/timeline/serving.py` prices them without a GPU instead, by an analytic
model of vLLM on one B300:

- **What a model costs comes from bux's compiler**: its parameters by module, the
  experts a batch of tokens touches, the bytes each layer keeps per token or per
  sequence, and the FLOPs of a token and of its attention.
- **A simulation of vLLM's scheduler** turns them into each engine step's work:
  chunked prefill, prefix caching in blocks, eviction and preemption.
- **A step takes a fitted overhead** plus the larger of its bytes over the HBM's
  bandwidth and its FLOPs over the peak, each at a fitted efficiency.

The efficiencies are fitted to the 38 measurements of
[Old against new's step 2](old-vs-new.md#2-measure-the-serving-cost): sessions per
hour, times per output token and times to the first token, of six models from a 7B
dense to a 116B hybrid MoE. `fit` prints each one against its prediction, and
`--loo` refits without each model in turn and prices it:

```console
$ python scripts/timeline/serving.py fit --loo
...
model      what           c  predicted  measured   error
old-4096   sessions/h     8      879.6     892.0   -1.4%
old-4096   tpot_ms        8       15.2      15.2   -0.1%
old-4096   ttft_s         8        0.2       0.1  +69.5%
...
new-3072   sessions/h     8      935.3     894.0   +4.6%
new-3072   tpot_ms        8       13.7      14.0   -1.9%
...
mean |log error|: all 0.058, sessions/h 0.030
...
all: sessions/h mean |error| 3.6%
```

Sessions per hour come within 3% on average, and within 3.6% for a model the fit
did not see. The time to the first token is off by up to 70%; no budget uses it.

### The budget

The budget is the serving cost of a GPT-3 6.7B: `old_2022.bux` at `d_model = 4096`,
32 layers of multi-head attention 4096 wide. `serving.py` has three:

| Budget | sessions per hour per GPU | what it is |
|--------|---------------------------|------------|
| `B1a` | 880 | the 6.7B at 8 sessions at once |
| `B1b` | 985 | the 6.7B at its best, with a median time per output token of 30 ms or less |
| `B2b` | 246 | a quarter of `B1b`: sessions that cost four times as much |

This tutorial uses `B1b`. Each model is priced at its own best load within the
30 ms, rather than at one load for all. `match` finds the width at which a file serves
the budget:

```console
$ python scripts/timeline/serving.py match examples/experiments/timeline/gpt3_2020.bux --budget B1b
{
 "budget": "B1b",
 "target": 984.8312813423211,
 "d_model": 2854.2899304642156,
 "bracket": [
  2816,
  2944
 ],
 "tp": 1,
 "params": 4552249348.710004,
 "served": 4742735187.341839,
 "nonembed": 4361751656.0131445,
 "active": 4361751656.0131445,
 "kv_bytes_per_token": 509185.6879468383,
 "tried": {
  "2816": 1010.3,
  "2944": 928.9,
  "3072": 827.5
 }
}
```

The 2020 baseline at `d_model = 2854` serves 985 sessions per hour: a model of 4.55B
parameters, 4.36B of them besides the embeddings. The match interpolates between
the widths it prices. `--arm` picks one arm of a sweep, by part of its name.

### Every target

`table` prices every arm of every timeline file:

```sh
python scripts/timeline/serving.py table --out serving
```

It writes `serving/serving-table.md` and `serving/serving-table.json`, which `fit.py`
reads. The targets at `B1b`, and the KV cache each keeps per token at long context:

| Variant | `d_model` | parameters | besides embeddings, per token | KV per token |
|---------|-----------|------------|-------------------------------|--------------|
| 2020 baseline | 2854 | 4.55B | 4.36B | 484 KiB |
| 2021 reference | 2865 | 4.56B | 4.41B | 484 KiB |
| `attention`: `gqa4` | 4558 | 15.8B | 15.5B | 324 KiB |
| `attention`: `mqa` | 6386 | 41.1B | 40.8B | 25 KiB |
| `attention`: `mla` | 6063 | 37.3B | 37.0B | 52.9 KiB |
| `attention`: `swa` | 4556 | 17.8B | 17.5B | 324 KiB |
| `mixers`: `gdn` | 4167 | 16.9B | 16.7B | 264 KiB |
| `mixers`: `kda` | 4166 | 17.0B | 16.8B | 264 KiB |
| `moe`: `moe_8x2` | 2667 | 10.8B | 3.56B | 441 KiB |
| `moe`: `moe_dsv3` | 2500 | 32.6B | 2.95B | 400 KiB |
| `embeddings`: `engram` | 2860 | 22.1B | 4.39B | 484 KiB |
| `structure`: `deep_thin` | 1738 | 2.41B | 2.32B | 518 KiB |

- **A smaller KV cache buys width.** One key and value head (`mqa`) leaves room for
  a target with 9 times the reference's parameters per token.
- **Experts buy parameters, not FLOPs.** Reading the experts a batch touches costs
  bandwidth, so `moe_dsv3` serves at 32.6B parameters with 2.95B per token, fewer
  than the reference's.
- **Most arms keep the reference's target.** A change that moves neither the cache
  nor the FLOPs per token serves at about 2865.

The milestones' targets are in step 7. The largest need 2 GPUs, by tensor
parallelism, which the serving model prices from a prior, not from a measurement.

## 6. Fit and extrapolate

### The points

`fit.py` reads one line per finished run, its name and final metrics:

```sh
python scripts/timeline/collect.py runs > results.txt
```

% TBD: scripts/timeline/collect.py is not in the repository yet.

A run's name gives its family (the file and its arm), its rung (`d768`) and its
token budget. A family at a rung is one point, at its best learning rate. A point
worse by more than 0.5% than the same family's at a smaller rung, or on fewer
tokens, is dropped as diverged, and the report lists it.

### The law

Each metric of each family fits

$$L(N, D) = E + A N^{-\alpha} + B D^{-\beta},$$

where N is the parameters a token goes through besides the embeddings, the head,
Engram's tables and the MTP module, and D is the tokens. An MoE's routed experts
count `top_k` of `num_experts`.

- **E is shared by every family**: it is the data's entropy, which no architecture
  goes below.
- **The 2020 baseline, the root, has every term of its own.**
- **Every other family has its parent's terms, except those its points tell
  apart.** A is always its own. α is its own with three rungs, or with two when
  three or more families descend from it. B and β are its own with a token scan.
- **Parents** follow the axes: the reference for every axis besides positions, the
  baseline for positions and the milestones, the late-2026 hybrid for the stacks.
  `scripts/timeline/parents.json` names them.

% TBD: scripts/timeline/parents.json is not in the repository yet.

All families fit at once, by the Huber loss of log L, as Hoffmann et al. (2022) fit
theirs. A term that many families share then rests on all of their points.

The token scans train the `d512` rung of the baseline, the reference and every
milestone on 5, 10, 40 and 80 tokens per baseline parameter, at the scan's vertex:

```sh
bux launch examples/experiments/timeline/milestones.bux --only "[m2024_llama3, d512, t" \
  'budget = t5: 126M | t10: 252M | t40: 1006M | t80: 2012M' 'optimizer.lr = 1.67e-3'
```

`--share B,beta` pools B and β over every family instead, so that only the root has
them.

### What the fit says

- **The effective-parameter multiplier** of a family at a rung is the N at which its
  parent's law reaches the family's loss, over the family's own N. A multiplier of
  1.2 means the change is worth 20% more parameters. Its slope against N says
  whether the gain grows with scale (above 0) or fades.
- **Unlimited data** gives each target as many tokens as its GPU time buys. The fit
  finds the GPU-hours its target takes to reach the loss of its parent's target, on
  20 tokens per parameter and on 1T tokens. Its speedup is the parent's GPU-hours over
  its own. The fit also gives the loss it reaches in the parent's GPU-hours.
- **Limited data** gives every target the same tokens: 130B, 1T and 10T. The fit
  predicts each target's loss on them, against its parent's and the root's.
- **Every prediction has two ranges**: the spread of the fits that leave out one
  rung, and the 5th to 95th percentile of 100 refits with each point moved by a
  relative noise of 0.2% (`--noise`).

### Training throughput at target scale

GPU-hours are tokens times FLOPs per token, over the MFU times a B300's dense
bfloat16 peak, 2.25 PFLOP/s. The MFU is measured: each target trains for 20 steps at
its width, on synthetic rows of 4096, 4 rows per GPU in 2 microbatches, with MXFP8
matmuls, and with the long phase and the evals off. For the 2023 milestone:

```sh
lane=('budget = 1B' 'steps = 20' 'seq_len = 4096' 'batch = 16' 'data = synthetic'
      'curriculum = []' 'eval = []' 'objective = loss' 'observe = none'
      'log.trackers = []' 'mesh = mesh(data=1, fsdp=4, expert=1)'
      'methods = methods(microbatches=2, gather="layer")'
      'precision = precision(compute=bfloat16, matmuls=mxfp8)')
bux launch examples/experiments/timeline/milestones.bux --only "[m2023_llama2]" \
  'd_model = 4608' "${lane[@]}"
```

A target too large for one node trains at its width with about half its layers, in
whole periods of its pattern. A layer costs the same with fewer of them around it,
so the MFU of half the layers is that of all of them, to a few percent. The measured
targets, at the median of steps 10 to 18:

| Target | `d_model` | layers trained | GPUs | tokens/s per GPU | MFU |
|--------|-----------|----------------|------|------------------|-----|
| 2020 baseline, 2021 reference | 2816 | 44 of 44 | 2 | 39,386 | 0.518 |
| `attention`: `gqa4` | 4608 | 72 of 72 | 4 | 12,900 | 0.614 |
| `attention`: `swa` | 4608 | 72 of 72 | 4 | 12,396 | 0.639 |
| `attention`: `mla` | 6144 | 48 of 96 | 4 | 10,965 | 0.624 |
| `attention`: `mqa` | 6400 | 50 of 100 | 4 | 11,580 | 0.690 |
| `mixers`: `gdn` | 4224 | 66 of 66 | 4 | 9,947 | 0.482 |
| 2022 PaLM | 6400 | 50 of 100 | 4 | 10,171 | 0.606 |
| 2023 Llama 2, 2024 Llama 3 | 4608 | 72 of 72 | 4 | 13,611 | 0.647 |
| 2024 DeepSeek-V3, in bfloat16 | 3840 | 30 of 60 | 8 | 16,080 | 0.260 |
| 2025 Qwen3-Next | 3328 | 24 of 52 | 8 | 8,182 | 0.114 |
| 2026 sliding-window frontier | 3584 | 12 of 56 | 8 | 23,879 | 0.157 |
| 2026 linear-attention hybrid | 3328 | 24 of 52 | 8 | 12,228 | 0.160 |

MXFP8 trains the milestones 1.26 to 1.42 times as fast as bfloat16. The sparse
targets reach an MFU of 0.11 to 0.26, against 0.48 to 0.69 for the dense ones, so
they give back in GPU-hours part of what they gain in bits. Write the measurements
to `mfu.json`, by family, as `{"attention/gqa4": {"mfu": 0.6135}, ...}`.

### The fit

```sh
python scripts/timeline/fit.py results.txt --targets serving/serving-table.json \
  --mfu mfu.json --parents scripts/timeline/parents.json --share B,beta --out fit
```

It writes `fit/report.md`, every number in `fit/fit.json`, and plots of each
family's ladder and of both regimes; `--wandb` logs the tables to W&B as well.
`python scripts/timeline/fit.py --selftest` fits synthetic runs drawn from known
laws, to check that the fit finds them again.

{TBD: fit.py's report, trimmed: the leaderboard's first rows, and the laws of the
baseline and the reference.}

## 7. Milestones and stacks

### A family for each era

Each milestone is a file of its own, `m2020_gpt3.bux` to `m2026_hybrid.bux`, with
names of its own, so that `milestones.bux` sweeps them all over the rungs:

```bux
model = m2020_gpt3: gpt3 | m2021_neox: neox | m2022_palm: palm | m2023_llama2: llama2
      | m2024_llama3: llama3 | m2024_dsv3: dsv3 | m2025_qwen3next: qwen3next
      | m2026_frontier: frontier | m2026_hybrid: frontier_hybrid
```

They keep the timeline's layer for every 64 of the width, so that a rung means the
same depth in every family. Their mixtures of experts send a token through as much
MLP as the dense ones. Train one family alone with
`bux launch examples/experiments/timeline/milestones.bux --only "[m2022_palm"`.

| Milestone | what is new | target at `B1b` (per token) | MFU | objective, `d768` | unlimited data: speedup | 1T tokens: against 2020 |
|-----------|-------------|-----------------------------|-----|-------------------|-------------------------|-------------------------|
| 2020 GPT-3 | the baseline | 4.55B (4.36B) | 0.518 | {TBD} | 1 | {TBD} |
| 2021 GPT-J, GPT-NeoX | RoPE on a quarter of the dims, parallel blocks | 4.56B (4.41B) | 0.518 | {TBD} | {TBD} | {TBD} |
| 2022 PaLM | multi-query attention, SwiGLU, no biases | 41.1B (40.8B) | 0.606 | {TBD} | {TBD} | {TBD} |
| 2023 Llama 2 | RMSNorm, sequential blocks, GQA, an untied head | 16.6B (16.1B) | 0.647 | {TBD} | {TBD} | {TBD} |
| 2024 Llama 3 | RoPE at base 500k | 16.6B (16.1B) | 0.647 | {TBD} | {TBD} | {TBD} |
| 2024 DeepSeek-V3 | MLA, 128 experts by sigmoid, MTP | 112B (9.09B) | 0.260 | {TBD} | {TBD} | {TBD} |
| 2025 Qwen3-Next | Gated DeltaNet in 3 layers of 4, gated GQA, QK-norm, 256 experts | 163B (9.44B), 2 GPUs | 0.114 | {TBD} | {TBD} | {TBD} |
| 2026 sliding-window frontier | windows of 512 in 3 layers of 4, NoPE in the fourth, QK-norm, gates | 182B (8.08B), 2 GPUs | 0.157 | {TBD} | {TBD} | {TBD} |
| 2026 linear-attention hybrid | Kimi Delta Attention in place of the windows | 160B (9.09B), 2 GPUs | 0.160 | {TBD} | {TBD} | {TBD} |

The two 2026 families are the late-2026 consensus of two camps, scaled down. They
differ by one thing: the layers that see a bounded context are windows of 512 keys
in one and Kimi Delta Attention in the other.

### Stacks

`examples/experiments/timeline/stacks.bux` asks whether what the axes found adds up
past the consensus. It starts from the late-2026 hybrid and stacks the one-change
winners it lacks:

| Arm | what it adds to the hybrid |
|-----|----------------------------|
| `vres` | value residual in the global layers |
| `kvconv` | a causal conv of 4 taps on the global layers' keys and values |
| `rope` | partial RoPE at base 500k in the global layers, in place of NoPE |
| `best` | all three |
| `best_nope` | value residual and the conv, the global layers still NoPE |

{TBD: the milestones' and stacks' results, their plot, and what they say.}

## 8. Results and caveats

{TBD: the one-change leaderboard at the targets, under both regimes: the changes
worth most, those worth nothing, and those that cost.}

| Axis | best arm | multiplier at `d768` | unlimited data: speedup | 1T tokens: against the reference |
|------|----------|----------------------|-------------------------|-----------------------------------|
| positions | {TBD} | {TBD} | {TBD} | {TBD} |
| norms | {TBD} | {TBD} | {TBD} | {TBD} |
| attention | {TBD} | {TBD} | {TBD} | {TBD} |
| MLP | {TBD} | {TBD} | {TBD} | {TBD} |
| mixtures of experts | {TBD} | {TBD} | {TBD} | {TBD} |
| mixers | {TBD} | {TBD} | {TBD} | {TBD} |
| embeddings | {TBD} | {TBD} | {TBD} | {TBD} |
| residual stream | {TBD} | {TBD} | {TBD} | {TBD} |
| structure | {TBD} | {TBD} | {TBD} | {TBD} |

The caveats:

- **The rungs are small.** The widest is 394M parameters besides the embeddings,
  and most arms stop at 85M. Changes whose worth grows with scale, as mixtures of
  experts may, are the hardest to read from them.
- **The targets are far.** A dense target at the reference's cost is 4.4B, 50 times
  past `d768`. The KV-cache arms' targets are 15B to 41B, and the milestones' up to
  182B. The ranges show how much the fits move without each rung; they are not
  confidence intervals.
- **Most runs are single seeds.** Only the reference has replicates, which measure
  σ. A change smaller than 2σ at a rung is not told apart from the reference there.
- **The serving model is a model.** It prices sessions within 3% of vLLM on the six
  models it was fitted to, and within 3.6% for a model left out. Variants unlike
  those six, as MLA, Engram, sinks or a relative bias, it prices from their bytes
  and FLOPs alone. Tensor parallelism comes from a prior.
- **Some throughput is measured on fewer layers.** The largest targets train half
  their layers or fewer: the 2026 sliding-window frontier 12 of its 56. DeepSeek-V3's
  experts split over 8 GPUs by NCCL's expert parallelism, in bfloat16. The lanes of
  the 2025 and 2026 milestones make their experts 1152 wide, where the files make
  them 1088 (Qwen3-Next, the hybrid) and 1216 (the frontier) at those widths.
- **Some throughput is not measured.** A family without a lane takes a placeholder
  MFU, 0.35 dense or 0.25 sparse, and the report lists those families. {TBD: the
  lanes still missing.}
- **The learning-rate scan is coarse.** Three rates a factor of 2 apart place the
  bottom only as well as a parabola through them does. The report marks every best
  rate at the edge of its scan, which a wider scan should check. {TBD: how many.}

## Time and hardware

On nodes of 8 B300s, each step as written:

| Step | what | runs | GPU-hours | wall-clock |
|------|------|------|-----------|------------|
| 1 | the data, as in Old against new | | {TBD} | {TBD} |
| 2, 3 | the scans at `d512`, three or four rates per arm | {TBD} | {TBD} | {TBD} |
| 3 | `d768` for every arm | {TBD} | {TBD} | {TBD} |
| 3 | `d1024` for the arms that pass, `d1280` for the baseline, the reference and the milestones | {TBD} | {TBD} | {TBD} |
| 3 | the reference's replicates and the token scans | {TBD} | {TBD} | {TBD} |
| 4 | `interactions.bux` | {TBD} | {TBD} | {TBD} |
| 5 | the serving table, on a laptop | | | {TBD} |
| 6 | the throughput lanes | {TBD} | {TBD} | {TBD} |
| 6 | the fit, on a laptop | | | {TBD} |
| 7 | the milestones and the stacks | {TBD} | {TBD} | {TBD} |

{TBD: the total, and what to cut first on less hardware.}

## Next steps

- Add an arm: a labeled line in an axis file, with the comment `fit.py` reads above
  it ([Labels zip axes](../language/sweeps.md#labels-zip-axes)).
- Add a layer bux does not have, such as ALiBi
  ([Adding a layer](../architectures/adding-a-layer.md)).
- [Old against new at the same serving cost](old-vs-new.md): the measured vLLM
  benchmark that `serving.py` is fitted to.
- [Searching hyperparameters](../training/searching.md): a search in place of the
  fixed scan.

## References

- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
- Brown et al. (2020), *Language Models are Few-Shot Learners*,
  [arXiv:2005.14165](https://arxiv.org/abs/2005.14165): GPT-3, the baseline.
- Chowdhery et al. (2022), *PaLM: Scaling Language Modeling with Pathways*,
  [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).
- Dehghani et al. (2023), *Scaling Vision Transformers to 22 Billion Parameters*,
  ICML, [arXiv:2302.05442](https://arxiv.org/abs/2302.05442): QK-norm, and attention
  entropy collapse.
- DeepSeek-AI (2024), *DeepSeek LLM: Scaling Open-Source Language Models with
  Longtermism*, [arXiv:2401.02954](https://arxiv.org/abs/2401.02954): the batch-size
  law.
- DeepSeek-AI (2024), *DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model*,
  [arXiv:2405.04434](https://arxiv.org/abs/2405.04434): multi-head latent attention.
- DeepSeek-AI (2024), *DeepSeek-V3 Technical Report*,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437): sigmoid routing balanced by a
  bias, and multi-token prediction.
- Gemma Team (2024), *Gemma 2: Improving Open Language Models at a Practical Size*,
  [arXiv:2408.00118](https://arxiv.org/abs/2408.00118): sandwich norms and soft caps.
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*,
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556): 20 tokens per parameter, and
  the fit.
- Kandpal et al. (2025), *The Common Pile v0.1*,
  [arXiv:2506.05209](https://arxiv.org/abs/2506.05209): Stack v2 Edu.
- Kazemnejad et al. (2023), *The Impact of Positional Encoding on Length
  Generalization in Transformers*, [arXiv:2305.19466](https://arxiv.org/abs/2305.19466):
  NoPE.
- Meta (2024), *The Llama 3 Herd of Models*,
  [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).
- OLMo Team (2024), *2 OLMo 2 Furious*,
  [arXiv:2501.00656](https://arxiv.org/abs/2501.00656): post-branch norms with
  QK-norm.
- Peng et al. (2023), *YaRN: Efficient Context Window Extension of Large Language
  Models*, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).
- Penedo et al. (2024), *The FineWeb Datasets*,
  [arXiv:2406.17557](https://arxiv.org/abs/2406.17557): FineWeb-Edu.
- Qiu et al. (2025), *Gated Attention for Large Language Models*,
  [arXiv:2505.06708](https://arxiv.org/abs/2505.06708).
- Qwen Team (2025), *Qwen3-Next*: Gated DeltaNet in three layers of four, the 2025
  milestone.
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need*,
  [arXiv:1911.02150](https://arxiv.org/abs/1911.02150): multi-query attention.
- Shazeer (2020), *GLU Variants Improve Transformer*,
  [arXiv:2002.05202](https://arxiv.org/abs/2002.05202): SwiGLU and GeGLU.
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864): RoPE.
- Touvron et al. (2023), *Llama 2: Open Foundation and Fine-Tuned Chat Models*,
  [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).
- Wortsman et al. (2023), *Small-scale proxies for large-scale Transformer training
  instabilities*, [arXiv:2309.14322](https://arxiv.org/abs/2309.14322): attention
  logit growth.
- Xiong et al. (2023), *Effective Long-Context Scaling of Foundation Models*,
  [arXiv:2309.16039](https://arxiv.org/abs/2309.16039): raising RoPE's base (ABF).
- Yang, Kautz & Hatamizadeh (2025), *Gated Delta Networks*, ICLR,
  [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).

Every other arm's source is in the comment above it, in its axis file.
