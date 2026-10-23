# Runs and sweeps

A `.bux` file that also says how to train its model is a **run**. Training settings
are top-level definitions, like `d_model`, and their configs are called by name, like
layers:

```
d_model    = 768
vocab_size = 50304

norm  = rmsnorm
attn  = gqa(num_heads=12, num_kv_heads=4, rope=rope())
ffn   = gated_mlp(intermediate_size=2048, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^12 norm head(tied=true)

tokenizer = tokenizer(name="gpt2")
data      = 0.8 * text(path="data/web/*.parquet") + 0.2 * text(path="data/code/*.parquet")
packing   = best_fit
optimizer = muon(lr=0.02)
schedule  = wsd(warmup=1k, decay=0.2)
seq_len   = 2048
batch     = 256
steps     = 10B // (batch * seq_len)
```

There is no second config format and there are no paths to configs. A file builds on
other files by name (`use llama3_8b`), changes one setting of what it uses
(`optimizer.lr = 3e-3`), and turns into a sweep with alternatives
(`optimizer.lr = 1e-3 | 3e-3`). The command line takes the same statements. So the
ablation below is a whole experiment of six runs, with its architecture and its
recipe from other files:

```
use llama3_8b                      # examples/architectures/llama3_8b.bux
use fineweb_wsd                    # examples/experiments/recipes/fineweb_wsd.bux

d_model = 768
ffn.intermediate_size = 2048
model = embed block^12 norm head(tied=true)

attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3
budget            = 3B
```

```
$ bux plan examples/experiments/attention_ablation.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  ...  run
4217af5bcf  183M    3B      16.4       1.33B      1     ...  attention_ablation[mha, optimizer.lr=1e-3]
77044894a9  183M    3B      16.4       1.33B      1     ...  attention_ablation[mha, optimizer.lr=3e-3]
86d63e1072  174M    3B      17.2       1.27B      1     ...  attention_ablation[gqa, optimizer.lr=1e-3]
...
```

## Training settings

| setting      | value                                                  | default        |
|--------------|--------------------------------------------------------|----------------|
| `data`       | `text(...)`, `chat(...)`, `tokens(...)`, `synthetic`, or a mix: `0.7 * a + 0.3 * b` | required |
| `tokenizer`  | `tokenizer(name=...)`, for `text` and `chat` data      | `none`         |
| `packing`    | `concat(...)`, `greedy(...)` or `best_fit(...)`        | `best_fit`     |
| `optimizer`  | `adamw(...)` or `muon(...)`                            | required       |
| `schedule`   | `constant(...)`, `cosine(...)` or `wsd(...)`           | `constant`     |
| `seq_len`    | tokens a row                                           | required       |
| `batch`      | rows a step                                            | required       |
| `steps`      | optimizer steps                                        | required       |
| `seed`       | an integer                                             | `0`            |
| `clip`       | global gradient norm, or `none`                        | `1.0`          |
| `loss`       | `loss(z=..., mtp=...)`: z-loss and multi-token weights | `loss`         |
| `eval`       | `eval(data=..., every=..., batches=...)`, or `none`    | `none`         |
| `checkpoint` | `checkpoint(every=..., keep=...)`, or `none`           | `none`         |
| `log`        | `log(every=...)`                                       | `log`          |
| `profile`    | `profile(start=..., steps=...)`: a trace of `steps` steps after step `start`, or `none` | `none` |
| `precision`  | `precision(compute=bfloat16, matmuls=mxfp8, reduce=float32, ...)`: the dtype forward and backward compute in, parameters staying float32, the formats block projections multiply in ([Precision](#precision)), and the dtype gradients are summed over the devices in | `precision`: `bfloat16`, `compute`, `compute` |
| `mesh`       | `mesh(data=..., fsdp=..., tensor=...)`: the device grid | `mesh`: every device on `data` |
| `layout`     | `layout(...)`: which mesh axes each named dimension splits over | `layout` |
| `methods`    | `methods(attention=..., moe=..., microbatches=...)`: how what communicates is computed, and the microbatches a step takes its batch in | `methods` |
| `compiler`   | `compiler(preset=..., options=...)`: XLA's options for the training step ([below](#xlas-options-compiler)) | `compiler`: XLA's defaults |

The settings of each config are pydantic fields in `bux/train/config.py`, and the
editor shows them on hover and completes them, as it does for layers. In short:

- `adamw(lr, b1=0.9, b2=0.95, eps, weight_decay=0.1)` decays matrices only, not norm
  scales, biases or Engram's tables.
- `muon(lr, momentum=0.95, nesterov, ns_steps=5, ns_coefficients=jordan,
  ns_dtype=bfloat16, weight_decay=0.1, consistent_rms=0.2, adam=none)` trains the
  matrices inside the blocks with Muon, stacked expert weights one expert at a time,
  and everything else (the embedding, the head, vectors) with `adam`, which defaults to
  `adamw(lr=lr)`. With `consistent_rms=0.2` (Moonlight) one learning rate suits both;
  `none` scales updates as Keller Jordan's Muon does. `ns_coefficients` names the
  quintics of Newton-Schulz's iterations: `jordan`'s, the same at each, or
  `polar_express`'s (Amsel et al., 2025), one for each, closer to orthogonal in as
  many. Newton-Schulz runs in `ns_dtype`, bfloat16 as in Keller Jordan's Muon, and the
  updates move between devices in it (`methods.orthogonalize`). Its result is within
  a few percent of float32's; on a B300, 240 steps of the 1.3B dense model on packed
  documents lose the same with either, within the difference between two float32 runs.
- A schedule multiplies `lr`. `warmup` rises linearly over its steps. `cosine` decays to
  `final` (0.1) at the last step. `wsd` holds the peak, then cools down to `final` (0)
  over the last `decay` (0.2) of the steps, `linear`, `sqrt` (1 - sqrt(t)) or `cosine`.
- `loss(z=0, mtp=0.3)`: the cross-entropy of the first head, plus `z` times its
  log(Z)², plus `mtp` times the mean loss of the heads after it, each of which predicts
  one token further ahead for every `nextn` before it (see the language guide).

**MoE load balancing** is set on the layer, by a loss, a bias or both:
`moe(..., load_balancing=aux_free, aux_loss_coef=1e-4, aux_loss_scope=sequence)` is
DeepSeek-V3's. The loss is `aux_loss_coef` times E Σ_e f_e P_e, f_e the fraction of the
top-k picks that went to expert e and P_e its mean score normalized over the experts:
1 when both are uniform, over the whole batch (Switch, Qwen) or over each sequence and
then averaged (DeepSeek-V3). Its weight defaults to 1e-2, or 0 with `aux_free`; each
layer adds its own, and `balance_loss` logs their sum. (transformers'
`load_balancing_loss_func` is top_k times this, taken over all layers at once.)
`aux_free` adds a bias per expert to the scores that pick the top-k, and after every
training step moves it by `bias_update_rate` (1e-3) toward the experts picked less
than the mean over the batch. Eval neither adds the loss nor moves the bias, and
checkpoints keep the bias.

**The sparse attention indexer** learns from a loss of its own, as DeepSeek-V3.2's
does: `indexer(..., loss_weight=1)`. Each layer that runs one adds `loss_weight` times
KL(p ‖ softmax(I)), taken over the keys the indexer kept and averaged over the queries:
I its scores, and p the layer's attention to those keys, summed over heads and
normalized over them. p comes from the attention's logsumexp a chunk of queries at a
time, and the indexer scores each chunk again, so neither makes t × t floats. With
`block_size` (Qwen3.8 QSA) a block counts as its most
attended key, and only the complete blocks kept count. In CSA (DeepSeek-V4) the keys
are the compressed latents, and p keeps the share the window and sinks take. The
indexer's inputs and p are detached, so this loss trains the indexer alone and the
language-modeling loss trains the rest. The weight then only scales the indexer's
gradient, and its part of the norm `clip` bounds. A layer that reads `index` runs no
indexer and adds nothing, and `indexer_loss` logs the sum. DeepSeek's dense warm-up is
a stage with `top_k` at least `seq_len`, where attention is dense and the loss covers
every key the query sees; bux does not freeze the rest of the model for it.

Settings are definitions, so arithmetic, `name.field` reads and the number suffixes
`k`, `M`, `B` and `T` work on them: `steps = 10B // (batch * seq_len)`, `warmup=2k`.
A suffix is exact, so `1.1B` is the integer 1_100_000_000.

Before anything runs, the compiler checks each setting against its type, where it is
written, and that they fit together: the warmup and cooldown fit in the steps, and
sequences fit in the positions a learned positional embedding has.

```
$ bux check examples/experiments/attention_ablation.bux steps=50
error: the schedule takes 1010 steps (1000 of warmup and 10 of cooldown), but the run has 50
  --> examples/experiments/recipes/fineweb_wsd.bux:19:13
   |
19 | schedule  = wsd(warmup=1k, decay=0.2, shape=sqrt)
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

A file is a run once it sets any training setting; then it needs all the required
ones. A file with no model and no `d_model`, such as a recipe, is a module: it is
checked on its own, but a run needs a model.

## Precision

```
precision = precision(compute=bfloat16, matmuls=mxfp8,
                      roles=roles(kv=compute), first=2, last=2)
```

A run computes in `compute`, its parameters cast to it inside each step and kept in
float32, and the projections inside blocks multiply in `matmuls`, a format of
`bux.architecture.precision.matmuls`: `compute`, `mxfp8`, `mxfp8_2d` (W's scales per 32
x 32 tile, the same for y and dx) or `nvfp4` (docs/parallelism.md, "Matmul formats").
Four settings choose which projections, and when:

- `roles` gives a kind of projection a format of its own; `none` is `matmuls`. Its
  layer gives each projection its role (`sharding.linear(..., role="kv")`):

  | role | projections |
  |------|-------------|
  | `q` | attention's queries and output gate: gqa's, mla's, csa's, the indexer's |
  | `kv` | attention's keys and values |
  | `o` | attention's output projection |
  | `mlp` | the MLPs, the shared experts' too |
  | `latent` | LatentMoE's projections to and from its latent |
  | `recurrence` | the linear attentions', RWKV-7's and Mamba's projections, but the output |
  | `recurrence_out` | their output projection |
  | `mtp` | nextn's projection, and every branch after the first head |
  | `experts` | the routed experts, by a grouped GEMM: `compute`, `mxfp8` or `mxfp8_2d`; `none` is compute, not `matmuls`, as theirs is slower than bfloat16's (docs/parallelism.md) |

  The embedding, the head, the router and softmax attention's own matmuls always
  stay in compute.
- `first` and `last` keep that many residual branches outside any other, at the
  start and at the end of the model before its first head, in compute: a number, or
  a share of them, rounded up (`last=0.15`). A branch is a `[...]` inside no other,
  so a Llama-style block, `[norm attn] [norm ffn]`, is two. A scanned repeat scans apart
  the passes that multiply alike; a pipelined one must multiply alike throughout.
- `passes` are the GEMMs of y = x W that multiply in the format: `fprop`, y's, and
  the gradient's by x, `dgrad`, and by W, `wgrad`. The others multiply in compute,
  and the forward keeps their operands as they are.
- `switch=switch(at=0.82, passes=[dgrad, wgrad])` takes up other `passes` from that
  share of the steps on, and compiles the steps again: fprop in compute for the last
  18%, here, and all of it with `passes=[]`.

`eval=compute` evaluates in compute, whatever training multiplies in, and is not in
the run's id. `dispatch=mxfp8` has the ragged MoE methods send the tokens to their
experts in MXFP8, as DeepSeek-V3 does, and what comes back in compute
(docs/parallelism.md, "FP8 dispatch"). `moments=bfloat16` keeps the optimizer's
moments in bfloat16 between steps, as DeepSeek-V3's AdamW does, and updates them in float32. Rounded to nearest,
a moment ignores a change below 2^-9 of itself: at b2 = 0.95, a square gradient 4%
off the second moment moves it, at 0.999 only one 3x off.

A recipe is a module (`examples/experiments/recipes/`), which a run uses and sweeps:

```
use llama3_8b
use nvidia_nvfp4
precision.last = 4 | 8
```

| recipe | precision | left out |
|--------|-----------|----------|
| `nvidia_mxfp8` | Transformer Engine's MXFP8: E4M3 everywhere, gradients too | MXFP8 routed experts, which are slower in bux |
| `nvidia_nvfp4` | NVIDIA's NVFP4 pretraining: W in 16 x 16 tiles, dy rounded stochastically, a Hadamard transform for dW; `first=2, last=8` | NVFP4 routed experts |
| `nemotron3_nvfp4` | NVFP4, `recurrence_out` in MXFP8, and `q`, `kv`, `o`, `latent`, `mtp` and the last 15% in bfloat16 | NVFP4 routed experts, most of its matmuls |
| `deepseek_v3_fp8` | `mxfp8_2d`, the tokens sent to the routed experts in MXFP8, moments in bfloat16 | groups of 128 with float32 scales, the float32 promotion, FP8 routed experts, activations kept in FP8 |
| `torchtitan_mxfp8` | `mxfp8_2d` | MXFP8 routed experts, and saving a linear's input in bfloat16 |

On one B300 each, the 1.3B Llama-style model trained 3000 steps of 8 x 4096 tokens of
FineWeb-Edu (98M tokens), GEMM autotuning off. Its held-out loss at the end, in
bfloat16 (`eval=compute`), is this much above bfloat16's of the same seed, where two
seeds differ by 1.6%:

| precision | seed 0 | seed 1 | tok/s |
|-----------|--------|--------|-------|
| `mxfp8` | +0.12% | -0.02% | 126k |
| `deepseek_v3_fp8` | +0.15% | -0.03% | 127k |
| `moments=bfloat16` | +0.07% | -0.01% | 116k |
| `mxfp8, first=2, last=2` | +0.09% | | 127k |
| `nvfp4` | +0.75% | | 135k |
| `nvidia_nvfp4` | +0.48% | +0.56% | 129k |
| `nvidia_nvfp4` and its switch at 0.82 | +0.46% | +0.51% | 127k |
| `nvidia_nvfp4`, `last=4` † | +0.46% | | 44k |
| `nvfp4, last=0.15`, `q`, `kv` and `o` in compute † | +0.34% | | 56k |

NVIDIA measured 1.5% at the end of 10T tokens on a 12B, against FP8, and 0.5% with
the switch. NVFP4 quantizes by its kernels here, bfloat16 4.0383 and 3.9696 beside
them. The rows marked † quantized by XLA, which on a GPU folds the cast to E4M3 and
back: it scaled each block's values by its scale unrounded, and the GEMM by the
rounded one, up to 6% apart (`blockscaled.nv_quantize` keeps them alike now). That
moved nothing past the noise: NVIDIA's recipe quantized by XLA gives +0.44% and
+0.51%, with the fold +0.44% and +0.41%, and by the kernels with a float32 GEMM output
+0.50% and +0.53%. A change at the ulp moves a gap by up to 0.15 points.

The baseline's MoE (64 experts, top-8, 16 layers of `d_model` 2048) trained the same
3000 steps on two B300s a run, its experts split over them (`ragged_all_to_all`, no
capacity). With `roles=roles(experts=mxfp8)` and `dispatch=mxfp8` its held-out loss is
0.13% and 0.05% below bfloat16's at seeds 0 and 1, where the seeds differ by 1.65%;
it trains at 105k tokens a second against 127k, in 75 GiB against 89.

## Data

A batch is rows of `seq_len` tokens, each three arrays: `input_ids`, `position_ids`
(where each token is in its document, so one starts wherever they stop increasing)
and `loss_mask` (whether a token is a target). The model attends, rotates and
convolves within documents, and the loss is on targets of the document that predicts
them.

`text(path=...)` and `chat(path=...)` are documents in Parquet files: raw text, or
conversations in the OpenAI format. `bux prepare` tokenizes them with the run's
`tokenizer` and packs them into rows by its `packing`, once, as a MapReduce job; the
[data guide](data.md) has the settings, the packings and the pipeline.

`tokens(path="data/web/train_*.bin", dtype=uint16, header=0)` reads flat binary shards
of token ids through memory maps: a glob, relative to the working directory, of
`uint16` or `uint32` files, each after `header` bytes (1024 in modded-nanogpt's
shards). A row is a window of a shard; windows overlap by a token, so that every token
is a target once, and come in a new order every epoch. `synthetic` is uniformly random
tokens, for smoke tests and throughput.

A mixture is a weighted sum of sources. `+` mixes and `*` weights, so every batch of
`0.7 * web + 0.3 * code` has 70% of its rows from `web`, rounded; `web + code` is an
even mix, and nested mixtures multiply out. Every source is read in order, epoch after
epoch, so batch `i` of a run depends only on its seed and `i`, and a run that resumes
reads exactly what it would have read.

## Modules: `use`

`use llama3_8b` brings in the definitions of `llama3_8b.bux`. A module is found by its
name, never by a path: in the folder of the file that uses it, then anywhere under
the folders listed in `pyproject.toml`, then those in `BUX_PATH`:

```toml
[tool.bux]
path = ["examples"]
```

Folders organize files without becoming part of their names, so moving a file into a
subfolder breaks nothing, and a name used twice under one folder is an error. A name
that is not an identifier is written as a string: `use "qwen3.8_27b"`.

Definitions form layers. The modules a file uses come first, in the order it uses
them (with the modules they use below them, each module once), then the file. A
definition replaces the one below it **everywhere**, including inside the definitions
it did not write: after `use llama3_8b`, defining `attn` changes the attention in
llama3_8b's `block`. Inside its own definition, a name means the definition it
replaces, so `ffn = ffn(intermediate_size=2048)` changes one setting of the inherited
`ffn`; with nothing below it, a name means the builtin, as `mlp = mlp(...)` always did.

Errors point into the file they are in, and the editor shows errors in modules at the
`use` that brought them in. When a module is missing, the names it might have
defined are not reported as well.

## Amendments: `name.key = value`

`optimizer.lr = 3e-3` means `optimizer = optimizer(lr=3e-3)`: the optimizer as it was,
with one setting changed. Dotted keys reach into nested settings
(`attn.rope.base = 1e6`), several amendments of one name in a file act as one, and an
amendment applies over the file's own definition of the name, if it has one. A shared
layer stays shared. Only configs (layers and settings) have settings to amend; to
change a block, redefine it.

## The command line

Every command takes statements after the file, which redefine the file's as one more
layer over it would:

```
bux train examples/experiments/smoke.bux optimizer.lr=1e-2 'schedule = cosine(warmup=5)'
bux train examples/experiments/smoke.bux 'use fineweb_wsd'           # modules too
bux plan  examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3'  # and sweeps
```

So there are four ways to vary an experiment, `use`, an amendment, the command line and
an alternative, and they are one mechanism: each gives a name another definition, and
everything written in terms of that name follows.

## Sweeps: `a | b`

`|` separates alternatives, and a file with alternatives is several runs, one for
each combination. It is the loosest operator, below `+`, and works anywhere a value or
a block does:

```
optimizer.lr = 1e-3 | 3e-3 | 1e-2                       # a setting
attn = gqa(num_heads=8, num_kv_heads=8 | 2 | 1)         # inside a call
block = pre: [norm attn] [norm ffn] | par: [norm (attn + ffn)]   # the wiring itself
model = embed "DDDA"^12 norm head | embed "DDA"^16 norm head
```

Each `|` as written is an axis, and the axes form a grid, the first written varying
slowest. Runs are named after the file and what they took:
`ablation[attn.num_kv_heads=2, optimizer.lr=3e-3]`.

**Labels** name alternatives, `mqa: 1 | gqa: 4`, and the runs that take them. All
alternatives with the same labels are one axis, so they are picked together, which
zips them instead of crossing them. This is three runs, not nine:

```
d_model      = s: 512  | m: 768  | l: 1024
optimizer.lr = s: 4e-3 | m: 3e-3 | l: 2e-3
attn.num_heads = d_model // 64            # follows d_model in every run
```

Alternatives that share some labels but not all are an error. Only alternatives that
a run reaches make runs: one in a definition that nothing uses, or in a definition
that a later layer replaced, is not an axis. A macro takes the same alternative at
every call. `bux check` and the editor check every alternative once, with the others
at their first.

## Running

```
bux check examples/experiments/smoke.bux     # the model, the run, the sweep
bux plan  examples/experiments/smoke.bux     # every run: id, parameters, tokens, GPUs, bytes on each
bux prepare examples/experiments/sft.bux     # tokenize and pack its documents; --cache DIR, --workers N
bux train examples/experiments/smoke.bux     # train every run, one after another; --only TEXT, --out DIR
bux launch examples/experiments/smoke.bux    # train what is left of every run, a GPU each: see below
bux fmt   examples/experiments/smoke.bux     # every run as a file of its own
bux json  examples/experiments/smoke.bux     # every run's IR
```

`bux.lang.compile_runs(source)` returns a `Run` per combination: an `Architecture`, a
`TrainConfig` (`bux/train/config.py`), a name, and the alternatives it took, all plain
pydantic. `bux.train.trainer.train(run)` trains one with the reference Flax NNX
backend, on the devices its `mesh` takes (see the
[parallelism guide](parallelism.md)), into `runs/<name>-<id>/`:

```
run.bux        the run as one file: no modules, no alternatives; bux train takes it as is
run.json       the IR
metrics.jsonl  the log's metrics every log.every steps, and eval losses
checkpoints/   model and optimizer state, with a checkpoint setting (orbax)
profile/       a jax.profiler trace, with a profile setting
```

The id is a hash of what the run computes: renaming a definition, reaching the same
settings through another file or a sweep, logging, evaluating, checkpointing and
profiling at other steps, splitting it over devices another way (`mesh`, `layout`,
`methods`) or compiling it with other options (`compiler`) keeps it, and anything else
changes it. So a changed run starts in a directory of its own, and an unchanged one
started again resumes from its latest checkpoint, ending exactly where it would have.

The log has the step's metrics: `loss` and every term of the objective, `grad_norm`
before clipping, `lr`, `tokens` so far and `tokens_per_s` since the last log. On a GPU
whose peak it knows (B300, B200, H200, H100, A100: dense bf16, from NVIDIA's
datasheets), it also has `mfu`: the model's FLOPs a second over the peak of the
mesh's GPUs. A step's FLOPs are counted once, before training, from the program of
its loss's gradient on a batch, traced on shapes alone with remat and kernels off
(`trainer.step_flops`, `bux/train/flops.py`): 2·m·n·k for every matrix product and
convolution, times the length of every scan around it. So a new layer counts with
no formula of its own, and a rematerialized forward is not counted twice. Attention
counts the share of its t × t logits that its mask keeps, (t + 1) / 2t causal and
less with a window, as Megatron's MFU does and a flash kernel computes; not the
documents of a packed row, which change from batch to batch. (PaLM's MFU counts them
whole, which made bux's 10% higher at 4096 tokens a row.) `bux plan` shows the FLOPs a
token: 6.9G for a Llama-style 1.3B at 4096 tokens a row, 6 N of its matrices and 0.8G
of attention.

`profile = profile(start=20, steps=5)` traces the 5 steps after step 20 into
`profile/`, waiting for the step before it starts and stops, for `xprof` or
TensorBoard's profile plugin (`tensorboard --logdir runs/<run>/profile`). On GPUs the
log then has, at step 25, what a step of them spent in collectives, `comm_ms`, and in
collectives with no kernel computing beside them on any stream, `exposed_comm_ms`: the
communication the step waits for. Each is the mean of the GPUs the process traced
(`bux/train/trace.py`): a kernel communicates if NCCL runs it or XLA's op that
launched it is a collective, CUDA's copies neither communicate nor compute, and every
other kernel computes.

## Speed on GPUs

What XLA compiles slowly on a GPU runs by a kernel there, the same computation on a
CPU by XLA; none of it is in a run's id. Each is in `bux.architecture` and falls back
by itself where it cannot run, and all of them do inside `bux.kernels.off()` or with
`BUX_KERNELS=off`, to debug one:

| what | by | on a B300, against XLA |
|------|----|------------------------|
| softmax attention | `methods.attention = auto`, the default: cuDNN's fused attention in rows of 8192 or more and with a window, else tokamax's flash attention (sinks, a soft cap) | 7x, and no t × t logits |
| sparse attention (DSA, QSA, CSA) | `sparse`: tokamax's flash attention with the kept keys as a mask; the indexer's scores by a Pallas kernel (`scoring`), its top-k by counting | 2.6x a chunk of queries at a time, and no t × t floats |
| linear attention, RWKV-7 | `chunkwise`: chunks of 64 positions, the state across them by a Pallas kernel (`affine_scan`); `methods(states=recompute)` keeps a fourteenth of the memory for the backward | 14-17x a scan over positions |
| Mamba-2 | `chunkwise`, B and C a group's, the state across chunks by a matmul | 25x a scan over positions (2 × 4096 tokens, rematerialized: at 8 × 4096 the scan needs 199 GiB) |
| Gated DeltaNet, DeltaNet | cuDNN's fused GDN kernel (`fused_delta`), by cuDNN's graph from a `buffer_callback`: any gate | 5x `chunkwise` |
| KDA with `gate_lower_bound` | cuDNN's fused KDA kernel (`fused_delta`) | 10x `chunkwise` |
| MoE experts | `grouped.ragged_dot`: tokamax's Mosaic and Triton kernels | 27x `jax.lax.ragged_dot` |
| MoE dispatch | `pick_rows`, `permuted`: gathers where XLA would scatter | 4x |
| NVFP4's quantizers | `nvfp4`: Pallas kernels, a pass for the largest magnitudes and one that writes both copies, rounded by Blackwell's E2M1 conversions | 3-5x |

Tokens a second on B300s, 8 × 4096 tokens a GPU, `precision(compute=bfloat16)`, and
the log's `mfu` (of the attention a causal mask keeps):

| model (d = 2048, 16 layers) | 1 GPU | 8, `mesh(fsdp=8)` |
|-----------------------------|-------|-------------------|
| Llama-style, 1.3B           | 144k, 44% MFU | 1.12M, 43% |
| Gated DeltaNet hybrid, `"DDDA"` | 117k, 37% | 818k |
| Mamba-2 hybrid, `"MMMA"`, 1.5B (Nemotron-H's `mamba2`) | 97k | 741k |
| KDA hybrid, `"KKKA"`        | 59k   |                   |
| KDA with `gate_lower_bound=-5` | 127k, 40% | 926k     |
| MoE, 64 experts, top 8      | 102k, 31% | 834k          |

The GDN hybrid trained at 93k (715k on 8) by chunks before its kernel, in 121 GiB
where it now takes 71 (58 on 8, against 108), and so fits 24 rows (136k, 169 GiB); KDA
with `gate_lower_bound` at 110k in 173 GiB, now 91, and 16 rows (133k, 163 GiB). Their
losses over 30 steps, GEMM autotuning off, are within 1.7e-4 of the chunks' (GDN) and
the same (KDA). On 8 GPUs the GDN kernel's callbacks run one at a time, 87% of 8 GPUs
alone, where KDA's kernel, which cudnn-frontend calls from JAX itself, runs at 91%.

Sparse attention trains at long context. A DeepSeek-V3.2-style 1B (MLA of 16 heads,
DSA's indexer of 64 heads and top 2048 in all 16 layers, dense MLPs) on one B300, in
bfloat16, against the same model by all t × t logits, as bux attended before `sparse`:

| tokens a row × rows | t × t logits | `sparse` |
|---------------------|--------------|----------|
| 4096 × 2            | 27.0k tok/s, 85.6 GiB | 28.6k, 31.0 GiB |
| 4096 × 8            | out of memory | 34.9k, 73.9 GiB |
| 16384 × 1           | out of memory | 10.1k, 50.4 GiB |
| 32768 × 1           | out of memory | 7.1k, 84.7 GiB |

At 32k a layer's attention takes 1.4-1.9 times dense flash attention's time, and the
indexer most of the rest: its selection 61 ms, its loss 75 ms and the loss's target
33 ms, forward and backward. Under `methods(remat=save_attention)` the backward neither
selects the keys again nor runs the flash forward again
([Repeats and remat](parallelism.md#repeats-and-remat-methodsrepeat-methodsremat)):
at 32k, 7.27k tokens a second in 39.8 GiB, as fast as no remat (7.11k, 77.9 GiB),
where it ran at 5.85k before; at 16k, 10.2k against 8.65k. MLA keeps its latents for
`save_qkv_attention`, not the heads' projections up from them: at 8 rows of 4096,
34.1k in 41.2 GiB, against 30.1k in 51.5 before.

A thread loads the next two batches as a step runs (`prefetch`). A step's new state
takes the old one's memory (`methods.donate`): the 1.3B's step, its loss by logits,
peaks at 70.6 GiB instead of 81.3 on one GPU (51.1 by chunks, the default), and it
trains 36 rows a GPU, which run out of memory without
([Donation](parallelism.md#donation-methodsdonate)).

bux sets one XLA flag of its own (`bux/runtime.py`), for expert parallelism:
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, without which
NCCL's ragged all-to-all hangs. The flags that usually speed up collectives and
scheduling changed nothing past the noise (1%) on the 1.3B on 1 and 8 GPUs, nor on
the MoE on `mesh(fsdp=8)`: the latency-hiding scheduler, pipelined all-gathers,
all-reduces and reduce-scatters, combine thresholds of 128 MiB, double-buffered while
loops, command buffers off, Triton GEMMs off. FSDP over 8 GPUs gets 97% of 8 times
one GPU's tokens already. Other splits wait longer for their collectives, and the
`compiler` setting tries XLA's options on them (below).

On packed documents (`best_fit` of documents of 16 to 4000 words, GPT-2 tokens), on
one GPU, the 1.3B trains at 145k tok/s in rows of 4096 by tokamax's kernel, and at
139k in rows of 16384 by cuDNN's, which skips every block no query sees, where
tokamax's, whose backward pass does every block of a row, gives 109k (random tokens:
111k and 101k). With sinks, 142k in rows of 4096, where `auto` took xla, whose t × t
logits did not fit. Rows of 32k over 8 GPUs: docs/parallelism.md, "Context
parallelism".

Unlike these, a low-precision matmul format changes what a run computes, so it is a
setting, `precision(matmuls=...)` (docs/parallelism.md, "Matmul formats"). On a B300,
cuDNN's MXFP8 GEMM runs at 1.75x bfloat16's rate (2.7-3.0 PFLOP/s at the model's
shapes), and quantizing each operand twice, in one pass of a Pallas kernel, gives some
of it back: an MLP projection's forward and backward take 1.3-1.35x less time, q's and
o's 1.1x, and k's and v's, half as wide, 7% more. cuDNN's NVFP4 GEMM runs at 4.8-5.5
PFLOP/s at the MLP's shapes, 1.7-1.9x MXFP8's, with a bfloat16 output that the two
tensors' scales then rescale, in what reads it (cuDNN's own global scale is no faster,
and takes it in bfloat16). NVFP4's quantizers are two passes of Pallas kernels over
each operand (``bux.architecture.precision.nvfp4``): its largest magnitude and that of
its transpose through the Hadamard transform, then both copies at once, each value
rounded, saturated and packed by Blackwell's `cvt.rn` and `cvt.rs` (stochastic, 8
random bits a value). An MLP projection's forward and backward then take 1.07-1.1x
less time than MXFP8's, but q's and o's 1.1x more and k's and v's 1.25x: the second
pass, ~20 more small kernels a projection (scales for cuDNN's layout, the global
scales) and the transform's copy, at 2-2.5 TB/s, cost more than their GEMMs save. The
routed experts stay in bfloat16:

| `matmuls`, 1.3B Llama-style | 1 GPU | 8, `mesh(fsdp=8)` |
|-----------------------------|-------|-------------------|
| `compute` (bfloat16)        | 142k  | 1.12M             |
| `mxfp8`                     | 167k (1.18x) | 1.30M (1.16x) |
| `nvfp4`                     | 179k (1.22x), bfloat16 147k and `mxfp8` 169k beside it |   |

On 300 steps of packed documents, autotuning pinned, `mxfp8`'s loss is bfloat16's to
within the noise between seeds. The MoE trains as fast in `mxfp8` as in bfloat16,
105k: its experts are most of its matmuls.

Which projections multiply in `mxfp8`, on one B300 ([Precision](#precision)):

| `precision(matmuls=mxfp8, ...)`, 1.3B | tok/s | peak memory |
|---------------------------------------|-------|-------------|
| `compute` (bfloat16)                  | 144k  | 70.6 GiB    |
| `mxfp8`                               | 167k  | 66.3 GiB    |
| `mxfp8_2d`                            | 165k  | 66.3 GiB    |
| `roles=roles(kv=compute)`             | 164k  | 68.4 GiB    |
| `first=2, last=2`, of 32 branches     | 161k  | 66.9 GiB    |
| `passes=[fprop, dgrad]`               | 152k  | 69.9 GiB    |
| `moments=bfloat16`                    | 169k  | 61.6 GiB    |
| `nvfp4, first=2, last=8`              | 167k  | 46.1 GiB    |
| `nvfp4`                               | 179k  | 43.8 GiB    |

The other rows take the loss by logits; the `nvfp4` rows are of a later tree, which
takes it by chunks (`methods.loss`): bfloat16 there 147k in 51.1 GiB, `mxfp8` 169k in
46.9. Its first 2 and last 8 branches in bfloat16 leave NVIDIA's recipe as fast as
`mxfp8`; NVFP4 everywhere is 1.06x it.

k and v in bfloat16 are slower, not faster: q still quantizes x, the copy all three
shared, and k's and v's GEMMs then run at bfloat16's rate, x kept for them too. Over 8
GPUs, `mesh(fsdp=8)`, `mxfp8` and `mxfp8_2d` both train at 1.34M (bfloat16: 1.14M),
and scanned, `first=2, last=2` at 1.20M, where `mxfp8` alone gives 1.22M.

### XLA's options: `compiler`

```
compiler = compiler(preset=o1)
compiler = compiler(options="--xla_gpu_threshold_for_windowed_einsum_mib=0")
compiler.preset = default | o1 | lhs | maxtext       # a sweep
```

The training step compiles with a preset's options, and the setting's own `options`
over them, written as XLA_FLAGS writes them: XLA's debug options, and jax's
`optimization_level` and `memory_fitting_level`. They are the step's alone
(`bux/train/xla.py`), so a run's step compiles with its options in a process that
compiled another's with others, and they go over XLA_FLAGS. Like `mesh`, `layout` and
`methods`, `compiler` changes how a run computes, not what, and is not in its id.

| preset              | what it sets                                                     |
|---------------------|------------------------------------------------------------------|
| `default`           | nothing: XLA's defaults                                           |
| `o1`                | `optimization_level=O1`: collective pipelining, double-buffered loops, and the latency hiding scheduler for any collective |
| `lhs`               | the latency hiding scheduler, for any collective                  |
| `maxtext`           | MaxText's GPU recipe: the scheduler, combine thresholds of 128, 128 and 64 MiB, pipelined collectives, double-buffered loops, a high-priority stream for collectives, Triton GEMMs, command buffers and XLA's rematerialization off |
| `collective_matmul` | XLA's windowed einsum: tensor parallelism's all-gathers and reduce-scatters as loops of collective-permutes beside their shares of the matmul |

Another preset is an entry of `bux.train.xla.PRESETS`. XLA (in jax 0.12 and at its
head) runs the latency hiding scheduler on Hopper and Blackwell by itself, but only on
a step whose collectives its cost model knows: not one with `ragged_all_to_all`
(`methods(moe=ragged_all_to_all)`), which `lhs` and `o1` schedule. Host offload does
not stop it.

On one node of 8 B300s, bfloat16, rows of 4096 tokens but for the ring's, tokens a
second, and the exposed communication of a step in ms of its time (the log's
`exposed_comm_ms`, profiled):

| run | `default` | `o1` | `lhs` | `maxtext` |
|-----|-----------|------|-------|-----------|
| TP=8 and sequence parallelism: 4 of Llama 3 70B's layers, scanned, `remat=full`, 8 rows | **218k**, 34 of 150 | 211k, 31 | 218k, 34 | 216k, 35 |
| EP=8 by `all_to_all`: 8 layers of 64 experts, top 8, d = 2048, a row a GPU | 158k, 62 of 208 | 157k | 155k | 159k |
| EP=8 by `ragged_all_to_all`, as above | 45k, 618 of 733 | 45k | 45k | 45k |
| the 1.3B, `fsdp=8`, 8 rows a GPU | **1.14M**, 3.5 of 230 | 1.13M | | 1.09M, 9.0 |
| the same in 8 microbatches | **842k**, 57 of 311 | 830k, 64 | 836k | 789k, 74 |
| the same scanned (`repeat=scan`), a row a GPU | 662k, 11 of 50 | 682k, 8 | 661k | **718k**, 10 |
| `ring` over `context=8`, rows of 32k, one document, the 218M of [CP](parallelism.md#context-parallelism-methodsattention) | **183k**, 137 of 1436 | 167k | 182k | |
| the same on packed documents | 223k, 128 of 1177 | 210k | **225k**, 23 | |

`collective_matmul` takes TP to 93k: its 325 collective-permutes a step, 0.84 ms each,
take 274 ms (XLA's windowed einsum; over memcpy, `--xla_gpu_use_memcpy_local_p2p`, the
same). TP waits for its reduce-scatters (17 ms a step) and all-gathers (15 ms) at the
boundaries of the matmuls they feed, which no schedule moves. `methods(tensor=chunked)`
moves them beside the matmuls, a chunk at a time, and hides a quarter of the wait,
which the matmuls beside give back ([Tensor
parallelism](parallelism.md#tensor-parallelism-methodstensor)). `ragged_all_to_all`
waits 610 ms a step for its 96 exchanges (of a slice a peer, before they went by
expert and in pieces: [Expert parallelism](parallelism.md#expert-parallelism)); with 4
rows a GPU its step runs out of memory under `default` (a temporary of 97 GiB) and
fits under the others, at 55k against `all_to_all`'s 176k. FSDP's
collectives take 14% of the step and hide all but 1.5% of it. A collective's time
includes the wait for its slowest peer, so on unevenly loaded devices, such as the
ring's on packed rows, `exposed_comm_ms` swings between runs of one speed.

Only scanned layers gain, under FSDP: `maxtext` trains the scanned 1.3B at 719k
(three runs, against 656-662k), faster than the same unrolled (623k, and 572k under
`maxtext`). Of its options, the pipelining and double buffering of `o1` give 3%, and
cuBLAS's GEMMs where XLA would take Triton's inside the loop
(`--xla_gpu_enable_triton_gemm=false`) 5%; the combine thresholds, XLA's
rematerialization and command buffers nothing. With TP the scanned layers gain nothing.
So `default`, XLA's own options, stays the default, and a scanned FSDP run is worth
`compiler(preset=maxtext)`.

## Launching

`bux train` trains in its own process, which suits a laptop and a debugger. `bux launch`
is the entrypoint for production, and the Docker image's: it trains an experiment to its
end, however often it is stopped and started again.

```
bux launch examples/experiments/attention_ablation.bux             # all 6 runs, a GPU each
bux launch examples/experiments/attention_ablation.bux --index 3   # the fourth, in an array job
```

A run needs the GPUs of its `mesh`, the product of its axes' sizes, where `data` unset
is every visible GPU for a run alone and one in a sweep; `bux plan` lists them. A run
that needs more than there are fails at once. The launcher prepares the datasets the
others read, once. A run alone then trains in the launcher's process, on the first
GPUs it needs. A sweep trains in workers, a process a run, each on a group of free
GPUs (`CUDA_VISIBLE_DEVICES` says which), the first pending run that fits starting
first; or one at a time if there are no GPUs. A worker's output goes to the console
and to its run's `train.log`, and a run that fails or runs out of memory leaves the
others training.

A run's directory also keeps a `status.json`: whether the run is `running`, `done`,
`failed` or `stopped`, on what host and GPUs, since when, at what step, and its latest
metrics. A run that is done is not trained again, and one that is not resumes from its
latest checkpoint, so launching an experiment again, after a preemption, a crash or a
fix, trains what is left of it. A sweep ends with a table of its runs:

```
3 runs: 2 done, 1 stopped
state    step       loss    eval loss  run
done     3814/3814  3.2141  3.2577     width_sweep[s]
done     3814/3814  3.0836  3.1302     width_sweep[m]
stopped  2731/3814  3.0412  3.0815     width_sweep[l]
```

SIGTERM or Ctrl-C stops training after the step it is at, and saves that step if the
run has a `checkpoint` setting; a second signal, a second or more later, stops it at
once. The launcher exits with 0 when every run is done, 1 when one failed, and 128 + the
signal's number when a signal stopped it.

`bux train`, `bux launch` and a sweep's workers keep what they compile in JAX's
persistent cache, in `$XDG_CACHE_HOME/bux/jax` (`~/.cache/bux/jax`), with the picks
of XLA's GEMM autotuner next to it: a run launched again, or the next run of a sweep
with the same program, loads its step instead of compiling it (0.5 s instead of 29
for the MoE above on 8 B300s), and one that differs in a flag still finds the
autotuner's picks (17 s). `JAX_COMPILATION_CACHE_DIR` says where instead, and set
empty turns the cache off. They also put bux's XLA flags (`bux/runtime.py`) before
`XLA_FLAGS`, a flag that it sets already keeping its value. In a job of many hosts
only the first writes the cache, so put it on a filesystem they all mount: bux turns
off one that is each host's own ([Many nodes](parallelism.md#the-compile-cache)).

`--index N` trains only the run at N, from 0, in the order `bux plan` lists them, so an
array job trains a sweep a run a task: `--index $SLURM_ARRAY_TASK_ID` on Slurm, or
`--index $(JOB_COMPLETION_INDEX)` in an indexed Kubernetes Job. Tasks that read the
same documents each prepare them, which is safe but wasted work; run `bux prepare` once
beforehand.

A run alone trains on many hosts when the environment says how many processes there
are: under Slurm's `srun` or Open MPI's `mpirun`, or with `JAX_COORDINATOR_ADDRESS`
(process 0's `host:port`), `JAX_NUM_PROCESSES` and `JAX_PROCESS_ID` set on each. Every
process runs the same command, and they join through `jax.distributed`; under Slurm
and Open MPI, JAX gives each the GPU of its local rank, so start a task a GPU:

```
srun --nodes=4 --ntasks-per-node=8 bux launch speedrun.bux   # 32 GPUs, one mesh
```

The first process alone writes the console, `metrics.jsonl`, `status.json` and the
run's files, and every one writes its part of a checkpoint, so `--out` is on a
filesystem they all see. A signal to any of them stops them all after the same step.
A sweep does not train on many hosts: give each run a job of its own, with `--index`.

## Benchmarks

`examples/benchmarks/` has a suite for each scale, a sweep of named runs of the 1.3B
and its kin in `baseline.bux`: `one_gpu.bux`, the dense model, its GDN, KDA, bounded
KDA and Mamba-2 hybrids, its MoE, and the dense one in MXFP8 and NVFP4, a GPU each;
`one_node.bux`, FSDP at 1 and 8 microbatches, the MoE sorted and by EP, TP=8 in
chunks and a 1f1b pipeline, on 8 GPUs; and `two_nodes.bux`, FSDP, HSDP and the MoE
both ways, on 16. `scripts/benchmark.py` trains a suite and compares two results:

```
python scripts/benchmark.py run examples/benchmarks/one_gpu.bux --out new.json
python scripts/benchmark.py compare old.json new.json
```

`run` trains each run in a worker of its own on the GPUs its mesh needs, as many side
by side as fit, as a sweep's launch does; on many hosts each runs the same command,
and the runs train one at a time on all of them. Of each run it keeps the median
`tokens_per_s` and `mfu` of the logs after the first, whose time includes compiling,
the peak memory of the busiest GPU, and the loss at every log, in the JSON file; the
runs' directories and each worker's output (`<run>.txt`) go in the folder beside it.
Settings after the suite go to every run (`'steps = 60'`), and `--only TEXT` trains
only the runs whose name has TEXT.

Nodes differ by a few percent, so measure a change against the tree before it, on the
same node. `--tree PATH` trains the suite with another checkout's bux, `PATH/src`, in
this one's environment, so a one-GPU suite runs both side by side on half a node each,
and a node's suite runs one tree after the other:

```
git worktree add ../before main
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/benchmark.py run examples/benchmarks/one_gpu.bux \
    --tree ../before --out old.json &
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/benchmark.py run examples/benchmarks/one_gpu.bux \
    --out new.json
wait
python scripts/benchmark.py compare old.json new.json
```

`compare` prints a line a run, here of one tree against itself, side by side on a
node of 8 B300s:

```
run              old tok/s  new tok/s   diff  new MFU  old GiB  new GiB  loss diff
one_gpu[dense]     144,920    144,408  -0.4%    44.4%     51.1     51.1    -0.0001
one_gpu[gdn]       121,571    120,504  -0.9%    37.9%     70.6     70.6    -0.0009
one_gpu[kda]        60,217     59,961  -0.4%    18.9%    122.0    122.0    +0.0007
one_gpu[kdab]      127,947    129,974  +1.6%    41.0%     91.4     91.4    -0.0005
one_gpu[moe]       106,512    106,531  +0.0%    32.8%    134.0    134.0    -0.0407
one_gpu[mamba2]     97,456     96,862  -0.6%    33.0%    122.6    122.6    +0.0002
one_gpu[mxfp8]     169,221    167,646  -0.9%    51.5%     46.9     46.9    -0.0109
one_gpu[nvfp4]     182,654    183,500  +0.5%    56.4%     43.8     43.8    -0.0005
```

`diff` is the change of the tokens a second, and `loss diff` the change of the loss at
the last log. The node's suite, one tree after the other, moved by 0.3% at most: FSDP
1.13M, at 8 microbatches 923k, the MoE sorted 832k and by EP 540k, TP 227k and the
pipeline 683k. A run is `slower` or `faster` when its tokens a second move by more
than `--noise` (3%), `more memory` or `less memory` when its peak does, `loss` when
its last loss moves by more than `--loss` (0.05), and `failed`; `compare` exits with 1
if any run is flagged but `faster` or `less memory`.

XLA's GEMM autotuner picks kernels by timing them, so two runs that tune apart, as
side by side, round differently: on random tokens the MoE, whose top-k picks amplify
it, ended 0.04 apart, the rest 0.01 at most. One after the other, the second loads the
first's picks from the compile cache, and they ended within 0.009. Pin
`XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"` on both
before calling a smaller change of a loss a regression.

## The trainer's blocks

`train(run)` is a few lines that put blocks together, and each block works without
the others, so changing one part of training means replacing one block:

| block       | in                       | what it is                                        | to change it                     |
|-------------|--------------------------|---------------------------------------------------|----------------------------------|
| sources     | `bux/train/data.py`      | an endless stream of rows, per data config type    | register a builder               |
| schedules   | `bux/train/optim.py`     | the curve after the warmup, per schedule type      | register a builder               |
| optimizers  | `bux/train/optim.py`     | an optax transformation, per optimizer type        | register a builder               |
| objective   | `bux/train/objective.py` | the loss: named, weighted terms, each a metric     | add a `Term`                     |
| `Trainer`   | `bux/train/trainer.py`   | model, optimizer, objective; the compiled steps    | override `train_step`            |
| hooks       | `bux/train/hooks.py`     | what happens every `every` steps: log, eval, save  | `train(run, hooks=[...])`        |
| sinks       | `bux/train/sinks.py`     | where reports go: console, metrics.jsonl           | `train(run, sinks=[...])`        |

A config's `type` finds its builder in a registry, as a layer's does. Registering under
a taken name replaces the builder, so a variant of an optimizer is a function, not a
fork; a new name also needs its config in `bux/train/config.py`, where the compiler
and the editor find it:

```python
from bux.train.optim import lr_schedule, optimizers


@optimizers.register("adamw")
def _(c, multiplier, model):  # a config, the schedule's multiplier, the model
    return optax.adamw(
        lr_schedule(c.lr, multiplier), b1=c.b1, b2=c.b2, weight_decay=c.weight_decay
    )
```

A hook runs after every `every` steps and after the last, and the scalars it
returns, metrics by name, go to every sink. When training stops early, because
something set `trainer.stopping` as the launcher does at SIGTERM, every hook's `stop`
runs, and the checkpoint hook saves the step. A tracker is a sink:

```python
class WeightNorm(Hook):
    every = 100

    def __call__(self, trainer, metrics):
        return {
            "weight_norm": float(optax.tree.norm(nnx.state(trainer.model, nnx.Param)))
        }


class Wandb(Sink):
    def __init__(self, run):
        self.run = wandb.init(project="bux", name=run.name)

    def write(self, step, scalars):
        self.run.log(scalars, step=step)


train(run, hooks=[WeightNorm()], sinks=[Wandb(run)])
```

A `Trainer` takes a model, an optax transformation and an objective, so a script or a
notebook can train with blocks of its own and keep the rest:

```python
loss = objective(run.train.loss, run.arch)
loss.terms["logit_l2"] = Term(1e-4, lambda logits, batch: jnp.mean(logits[0] ** 2))
trainer = Trainer(Model(run.arch, rngs=nnx.Rngs(0)), optax.adamw(1e-3), loss)
trainer.fit(
    batches(run.train.data, run.train, vocab),
    steps=1000,
    hooks=[Log(every=10, lr=..., tokens=...)],
    sinks=[Console("mine", 1000)],
)
```

The model gives the objective its heads' outputs, not their logits: `Model.heads`
(`training_heads` in a training step) returns each head's input and its projection,
a `HeadOutput` (`bux/architecture/logits.py`). The default terms, `loss`, `z_loss`
and `mtp`, are means over the targets of the cross-entropy, log Z minus the target's
logit, and of log Z squared, which `heads.sums(i, offset, z)` gives for head i by the
run's `methods.loss`:

| method   | how                                                                 |
|----------|---------------------------------------------------------------------|
| `chunks` | the matmul, the logsumexp and the target's logit of at most 2 GiB of float32 logits at once, and their gradients by the kernel and the stream with them, the kernel's added up in float32: the backward only scales those; the default |
| `logits` | one matmul for every position, the logits whole in `compute`        |

At a vocabulary of 128K, the float32 logits of 8 rows of 4096 tokens take 16 GB, and
their gradient as much again. A sum over positions is linear in each position's term,
so its gradient is known in the forward but for one factor, its cotangent, which the
backward multiplies in: by chunks, a head multiplies by its vocabulary three times, as
by whole logits, and keeps the gradients from the forward to the backward in
`compute`, the kernel's and the stream's. Each sum keeps its own, so a z-loss costs a
second matmul by the kernel and one by the stream. Another method is a function of a
head's output, the targets, their weights and whether to sum log Z squared,
`@methods.loss.register("mine")`, that returns their `Sums`. A term of one's own gets
the same `logits`, a sequence that makes a head's logits once a term reads them, once
for every term that does: `logits[0] ** 2` above makes the first head's whole, by
either method. `Model.outputs` returns every head's logits at once.

On a B300, for the dense 1.3B of 16 Llama blocks of width 2048 (vocabulary 128256,
`bfloat16`, 8 rows of 4096 tokens), against the whole logits, and against the chunks
before they took their gradients in the forward, whose backward made each chunk's
logits again:

| head, `methods.loss`      | tok/s | peak memory | largest batch |
|---------------------------|-------|-------------|---------------|
| untied, `logits`          | 138k  | 70.6 GiB    |               |
| untied, chunks, before    | 125k  | 52.1 GiB    |               |
| untied, `chunks`          | 139k  | 51.1 GiB    | 56 rows, 146k |
| tied, `logits`            | 141k  | 67.5 GiB    |               |
| tied, `chunks`            | 139k  | 48.1 GiB    |               |

On 8 B300s under `fsdp=8`, 8 rows each, `chunks` trains at 1.071M tokens a second in
38.9 GiB, and `logits` at 1.064M in 58.5: each device adds its share of the head's
gradient up alone, and the mesh reduces it once. The GDN hybrid and the MoE of
[Repeats and remat](parallelism.md#repeats-and-remat-methodsrepeat-methodsremat) train
6% and 5% faster by `chunks` than they did by chunks before.

A layer adds a term of its own from its forward, in one line:
`report(taps, "my_loss", weight, value)` (`bux/architecture/registry.py`). The
objective adds weight × value to the training loss and logs the values, summed over
the layers that report the name, under that name; eval leaves them out.

`train_step` and `eval_step` are what the trainer compiles, once; a subclass that
overrides them changes what a step does (averaging the weights, skipping a spike) and
keeps the loop, the hooks and checkpoints. Gradient accumulation is built in:
`methods(microbatches=n)`, which on a mesh gathers the parameters once a step and, where
the rows are independent and span hosts, reduces the gradients once (docs/parallelism.md, "Gathered
once, reduced once"); `Trainer(..., gather=, accumulate=, reduce=)` takes the same
settings.

With `Trainer(..., donate=True)`, as a run's `methods.donate` makes it by default, a
training step donates the model's and the optimizer's arrays: an array read from
`trainer.model` before a step is gone after it, so a hook that keeps one keeps a
copy (`np.asarray`). `fit` reads `data` through `prefetch` (`bux/train/data.py`),
a thread that loads the next two batches on the host while a step runs; a run that
resumes still reads from its step on, and an error reading a batch is raised at that
batch. Each goes to the devices just before its step, from the training thread: one
put there from the thread, while a step's temporaries held most of a GPU, landed
below them and split what they left free, and the next step's no longer fit.

## Design notes

**Why extend the language** rather than add a YAML or TOML file that points at the
`.bux` architecture. Almost nothing had to be added. Layer settings were already typed
keyword configs, checked against pydantic with errors at the source and suggestions,
with arithmetic, `name.field` reads and nested overrides, and the editor already
completed and documented them from the schema. A training setting is one more such
config, so all of that applies to it unchanged, and a run is one file you can read
top to bottom. The new syntax is what configs lacked anywhere: a way to build on
another file without a path (`use`), to change one setting of it (`name.key =`), and
to say "each of these" (`|`). Each is also useful for architectures alone.

**Late binding** (a definition replaces the one below it everywhere) is what makes
`use` worth having: an ablation is the lines that differ. It comes from the language
as it was, since definitions were already looked up by name, in any order. Nix
overlays, Jsonnet and Pkl's `amends` work the same way.

**Alternatives considered.** Config groups selected by path (Hydra) tie experiments to
the layout of a folder. Configs written in Python (Fiddle, gin) can do anything, and so
can only be checked by running them. CUE and Pkl type and merge configs well, but they
would be a second language next to `.bux`, with the architecture still out of reach.
Sweep files that list values per key (W&B sweeps) cannot vary the wiring, and they zip
or cross axes by convention instead of by name.

The resolved run is plain data, so other front ends can make one: a `Run` built in
Python trains the same way. Possible extensions: per-parameter-group optimizer
settings selected by definition label; device meshes and sharding; learning-rate
schedules composed in sequence with the same algebra; and hooks, sinks and builders
of one's own named in a `.bux` file, as plugins found on the module path.
