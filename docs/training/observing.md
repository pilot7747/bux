# Observing a run

A run reports what it does to the console, to `metrics.jsonl`, and to any trackers
(W&B, TensorBoard). Every log has the loss of each head, the gradient's norm, the
throughput, where the step's time goes, and the device's memory. The `observe`
setting adds the model layer by layer every so many steps: the residual stream after
each branch, each attention's logits, each MoE's routing, and each layer's gradient,
weights and update. Those are the metrics that tell you what an architecture gets
wrong.

## The console

`bux train examples/experiments/smoke.bux 'observe = observe(every=20)' steps=40`:

```text
smoke: runs/smoke-f3237709a1, on 1 × cpu
smoke: 156k parameters, 40 steps × 8 × 64 tokens = 20.5k tokens, adamw(lr=0.003), wsd(warmup=10)
smoke: log every 10, observe every 20
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step 10/40  loss 6.2429  lr 3.00e-03  grad 0.607  7,894 tok/s  65 ms/step  data 0.1 ms  compiled in 0.5s  0:00:01 < 0:00:02
smoke step 20/40  loss 6.2618  lr 3.00e-03  grad 0.642  8,412 tok/s  61 ms/step  data 0.0 ms  compiled in 0.6s  0:00:01 < 0:00:01
smoke step 20/40  observed  rms 0.0266-0.0675 (ffn.01)  max|x| 0.179-0.444 (ffn.01)  kurtosis 1.03-1.44 (attn.01)  logit 0.154-0.176 (attn.00)  attn entropy 1.00-1.00 (attn.00)  grad 1.02e-02-5.44e-01 (embed.00)  update/param 1.61e-03-3.35e-02 (embed.00)
smoke step 30/40  loss 6.2731  lr 3.00e-03  grad 0.516  78,030 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00
```

- The first lines say where the run writes, what it trains on, its model and tokens,
  and what it logs.
- A log line ends with the time since the start and the time left at the latest
  pace. `compiled in` marks a log after a step compiled: the first step, the first that
  observes, the first after a precision `switch` or at a new batch shape.
- An `observed` line gives each metric's range over the layers, and its worst layer:
  the largest value, except for attention entropy, where it is the smallest.
- A line whose loss or gradient is not finite ends in `!! not finite: grad_norm`.

The lines are written for a container's log: one per event, with no colors or
carriage returns, each flushed as it is printed. So `kubectl logs -f` shows them as
they come, and `kubectl logs --timestamps` adds the wall clock. `grep observed` picks
out the layers, and `grep '!!'` the steps that went wrong. The Docker image sets
`PYTHONUNBUFFERED=1`, so a library's prints and a traceback come through at once too.

## What every log has

| Metric | What it is |
|--------|------------|
| `loss`, `grad_norm`, `lr`, `tokens`, `tokens_per_s`, `mfu` | as in [The log](running.md#the-log) |
| `lm_loss/t+1`, `lm_loss/t+2`, ... | with more than one head, each head's cross-entropy, by how far ahead it predicts; `lm_loss/t+1` is `loss` |
| `time/step_ms` | the wall time per step since the last log |
| `time/data_ms` | the host's time per step waiting for the next batch |
| `time/launch_ms` | the host's time per step launching it |
| `time/wait_ms` | the host's time per step waiting for the step before to finish, to read its metrics, and for a step that compiled |
| `time/hooks_ms` | the host's time per step in hooks: evals, checkpoints |
| `time/compile_s` | the seconds spent compiling since the last log, where a step compiled |
| `mem/gib`, `mem/peak_gib` | on GPUs, the memory in use on the process's fullest device, and the most it has held |

These cost nothing on the device: the per-head losses are what the `mtp` term
computes anyway, and the rest is read on the host.

The times say what a step waits for. When the devices stay busy, `wait_ms` is most
of `step_ms`. A large `data_ms` means the input pipeline is falling behind, and a
large `hooks_ms` means evals or saves take the time.

## Observing the model

```bux
observe = observe(every=100)
observe = observe(every=100, histograms=true)
```

Every `every` steps, the run takes a separately compiled step that makes the same
update and also measures the model layer by layer. That step compiles once more, and
takes a few percent longer ([What it costs](#what-it-costs)). The other steps compile
to the same program as without the setting, so they cost the same
(`tests/test_observe.py` checks this). Like `log`, `observe` is not in the run's id.

### Layers and their names

A layer's metric is named `group/what/layer`: `act/rms/attn.03`. A layer is a
residual branch, such as `[norm attn]`, or an item of the model that is not one: the
embedding, the final norm, the head, a `nextn`. Its name is the definition its main
layer came from, the first that is not a norm: `attn` for `[norm attn]`, `ffn` for
`[norm ffn]`, `gqa` for `[rmsnorm gqa(num_heads=4)]`. Its number counts the layers of
that name in the order they run: `attn.03` is the fourth attention branch. A scanned
repeat counts its passes one by one, so a model's layers have the same names whether
it is unrolled, scanned or pipelined. A branch's parameters are the whole branch's,
its norm's included.

Since the names come from definitions, a sweep over `attn = gqa(...) | mla(...)` logs
both under `attn.03`, and their curves line up in a tracker.

### The metrics

| Metric | Of | What it is |
|--------|----|------------|
| `act/rms` | branch | the root mean square of the residual stream after the branch |
| `act/update_rms` | branch | the same, of what the branch added |
| `act/max_abs` | branch | the stream's largest magnitude |
| `act/kurtosis` | branch | the kurtosis of the stream's features' mean squares: 1 when the features are alike, large when a few carry the stream |
| `act/hist` | branch | with `histograms`: the share of the stream's entries in each bin of log2 \|x\| |
| `attn/logit_max` | attention | the largest logit of a key a query sees, softcapped, over 16 queries per row |
| `attn/entropy` | attention | the mean entropy of those queries' softmax, over that of a uniform one: 1 uniform, 0 on one key |
| `router/max_load`, `router/min_load` | MoE | the most and the fewest picks an expert gets, over the mean: 1 when balanced |
| `router/dead` | MoE | the share of experts no token picked |
| `router/entropy` | MoE | a token's mean entropy over the experts' scores, over that of uniform scores |
| `router/z` | MoE | the mean log Z² of the router's logits (ST-MoE's z-loss) |
| `router/balance` | MoE | the balance loss, whatever its weight |
| `router/bias_std` | MoE | the spread of the router bias, with `router_bias` |
| `grad/norm` | layer | the gradient's norm, before clipping; together, the layers' make up `grad_norm` |
| `grad/hist` | layer | with `histograms`: the share of the gradient's entries in each bin of log2 \|g\| |
| `param/norm` | layer | the parameters' norm before the update |
| `update/norm`, `update/ratio` | layer | the update's norm, and its ratio to the parameters' |
| `z_loss/t+1`, ... | head | each head's mean log Z² |

A histogram's bins run from log2 \|x\| = -24, which also holds zeros, to 15 and above.
Each is of an even sample of the entries: about 4M of the stream's positions, and 1M
of each of a layer's arrays along its longest dimension. That shows the shape of the
distribution, though not a share below one in a million. Histograms go to
`histograms.jsonl` and the trackers, not to `metrics.jsonl`.

### What to look for

| What you see | What it suggests | What to try |
|--------------|------------------|-------------|
| `act/rms` growing steeply with depth | the residual stream grows faster than the norms can follow | a norm on each branch's output (`[norm attn norm]`), `residual = mhc(...)`, smaller output projections |
| `act/max_abs` far above `act/rms`, `act/kurtosis` large, in a few layers | massive activations, outlier features | `qk_norm`, attention `sinks`, a `gated` attention output, no biases; they also break low precision |
| `attn/logit_max` growing | attention logit growth, the instability QK-norm fixes | `qk_norm=rmsnorm`, `softcap` |
| `attn/entropy` falling toward 0 | attention entropy collapse | `qk_norm`, a lower learning rate |
| `router/max_load` well above 1, `router/dead` above 0 | experts out of balance | `aux_loss_coef`, `router_bias=true` with `bias_update_rate` |
| `router/z` growing | the router's logits run away | `router_logit_scale` below 1 |
| `update/ratio` spread over more than 10× across layers | layers learn at different rates | Muon for the matrices, a learning rate per layer |
| `grad/norm` much smaller in the first layers than the last | gradients vanish with depth | the norms, the residual scale, the init |
| `act/update_rms` near 0 next to `act/rms` | a branch that does nothing | its init, its gate, its learning rate |
| `z_loss/t+1` growing | the head's logits drift | `loss(z=1e-4)` |
| `grad/hist` mass in the lowest bins | gradients that bfloat16 or FP8 flush to zero | float32 reductions (`precision(reduce=float32)`), a loss scale |

### Under microbatches and pipelines

A step of many microbatches observes each one and logs their mean, as it does its
loss: `act/max_abs` is then the mean of the microbatches' maxima. A scheduled
pipeline (`1f1b`, `zbh1`) runs the model cut at its pipelined repeat, and the forward
of the layers before the repeat stops there. Their activations are not observed,
though they count in the numbering, and their parameters are. Under context
parallelism the attention sample is not taken, since it would gather the keys.

### What it costs

On B300s, 2026-10-26: the benchmarks' 1.3B (`examples/benchmarks/baseline.bux`, 16
layers) and its MoE (64 experts, top 8), 8 rows of 4096 tokens per GPU, in bfloat16.
The median of 15 steps, each observed except in the `plain` column:

| Run | `plain` | observed | with histograms |
|-----|---------|----------|-----------------|
| 1.3B, one GPU | 214 ms | 222 ms (+4%) | 228 ms (+7%) |
| 1.3B, 8 GPUs, `mesh(fsdp=8)` | 220 ms | 225 ms (+2%) | 228 ms (+3.5%) |
| MoE, one GPU | 289 ms | 307 ms (+6%) | 315 ms (+9%) |

Observed every 10 steps, a run's steps take as long as without it, within the noise
of 1%: 211 ms against `plain`'s 215 for the 1.3B, and 296 against 290 for the MoE. An
observed step takes at most 0.1 GiB more, and compiles separately: 15 to 20 s more for
the 1.3B. Of an observed step of the 1.3B, measured by switching each part off, the
stream's metrics take 4 ms, the attention sample 2 ms, the per-layer norms 1 to 3 ms,
the routers nothing measurable, and the histograms 6 ms.

## Trackers

```bux
log = log(every=10, trackers=[wandb(project="ablations"), tensorboard])
```

Each tracker gets every metric and histogram, from the job's first process.

**W&B** ({bux}`wandb`). Install it with `uv sync --extra wandb`, and log in
(`wandb login`, or `WANDB_API_KEY`). Each run becomes a W&B run with the same name, in
a group of its file's runs, with its settings and its sweep's `choices` as its config.
A run that resumes from a checkpoint goes on logging to the W&B run it started, whose
id it keeps in its directory's `wandb_id`. That W&B run keeps the steps a crash logged
past the checkpoint: W&B ignores a step below one it has, so the resumed run's own
steps reach it only from there on. `mode="offline"` writes into the run's directory,
for `wandb sync` later, and `mode="disabled"` logs nowhere. Without a `mode`, W&B
takes the one `WANDB_MODE` sets, or online if it is unset. So `WANDB_MODE=disabled`
turns W&B off for a run without editing its file. A `mode` in the file wins over
`WANDB_MODE`.

The config's `axes` has what the run took on each axis of its sweep, by the axis's
name: `{"variant": "rope", "d_model": "d768", "optimizer.lr": "2e-3"}`. An axis is
named for the setting its first alternative is written at, so a sweep's runs group and
filter by any one axis. The config also has `params`, and `active_params`: the
parameters a token goes through, without the routed experts it skips. A run with an
`objective` logs its value as `objective` at each eval that updates it, so W&B sorts
runs by it.

**TensorBoard** ({bux}`tensorboard`). Install it with `uv sync --extra tensorboard`.
Event files go to the run's `tensorboard/`, and `tensorboard --logdir runs` shows every
run. A run that resumes drops the events past its checkpoint.

The Docker image has both. Another tracker is a sink plus a builder registered for
its config's type ([Sinks](extending.md#sinks)).

## Reading the files

`metrics.jsonl` has one line of JSON per log, which pandas reads as a table:

```python
import pandas as pd

m = pd.read_json("runs/smoke-f3237709a1/metrics.jsonl", lines=True).set_index("step")
m["loss"].plot()
m.filter(like="act/rms/").dropna().T  # the stream's rms by layer, at each observed step
```

`histograms.jsonl` has a line for each step that took histograms: its `step`, the
bins' `edges` in log2 \|x\|, and the shares of each histogram by name.
