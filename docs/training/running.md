# Running

`bux train` trains a file's runs in its own process, one after another. This page
covers the command, what a run writes to its directory, when two runs share an id,
what the log measures and how it counts FLOPs, and the eval and checkpoint settings.
To train a sweep on a cluster, or to stop and resume, see [Launching](launching.md).

```console
$ bux train examples/experiments/smoke.bux
smoke: runs/smoke-5a4f0f7d85, on 1 × cpu
smoke: 156k parameters, 50 steps × 8 × 64 tokens = 25.6k tokens, adamw(lr=0.003), wsd(warmup=10)
smoke: log every 10
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step 10/50  loss 6.2429  lr 3.00e-03  grad 0.607  7,844 tok/s  65 ms/step  data 0.1 ms  compiled in 0.5s  0:00:01 < 0:00:03
smoke step 20/50  loss 6.2618  lr 3.00e-03  grad 0.642  77,133 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00
smoke step 30/50  loss 6.2731  lr 3.00e-03  grad 0.516  77,903 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00
smoke step 40/50  loss 6.2478  lr 3.00e-03  grad 0.440  77,802 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00
smoke step 50/50  loss 6.2633  lr 0.00e+00  grad 0.418  74,244 tok/s  7 ms/step  data 0.0 ms  0:00:01 < 0:00:00
```

The first lines say where the run writes, its model and its tokens, and what it
logs; then `methods.remat = auto` chooses what the backward keeps
([Remat](../parallelism/remat.md)). The first log's tokens a second include the
compile, and each log ends with the time since the start and the time left
([The console](observing.md#the-console)).

## `bux train`

```sh
bux train FILE [SETTING=VALUE ...] [--out DIR] [--cache DIR] [--only TEXT] [--index N]
```

| Option | Default | What it does |
|--------|---------|--------------|
| `SETTING=VALUE` | | statements after the file, which replace the file's own: `optimizer.lr=1e-2`, `'schedule = cosine(warmup=5)'`, `'use fineweb_wsd'` ([Modules](../language/modules.md)) |
| `--out DIR` | `runs` | where run directories go |
| `--cache DIR` | `data/packed` | where prepared datasets are, and go ([The cache](../data/pipeline.md#the-cache)) |
| `--only TEXT` | | train only the runs whose name contains TEXT |
| `--index N` | | train only the run at N, from 0, in the order `bux plan` lists them |

```sh
bux train examples/experiments/smoke.bux optimizer.lr=1e-2 'schedule = cosine(warmup=5)'
bux train examples/experiments/attention_ablation.bux --only mqa
```

Before it trains, `bux train` prepares the datasets a run reads that the cache lacks,
with a process a core ([The pipeline](../data/pipeline.md)). It trains on every
device its process sees, as the run's `mesh` splits them
([Parallelism](../parallelism/index.md)). Ctrl-C stops it at once, without saving the
step it is at; to stop after a step and save it, train with `bux launch`
([Signals](launching.md#signals)).

The same in Python is {py:func}`bux.train.trainer.train`, which returns the latest
value of every metric:

```python
from bux.lang import compile_runs
from bux.train.trainer import train

path = "examples/experiments/smoke.bux"
[run] = compile_runs(open(path).read(), filename=path)
metrics = train(run, "runs")  # {'loss': 6.2633, 'grad_norm': 0.4177, ...}
```

## The run directory

A run writes to `<out>/<name>-<id>/`: its name, with each stretch of characters other
than letters, digits and `_.=+-` made one `_` and any `_` at the ends dropped, then its
id. `smoke[optimizer.lr=1e-3]` goes to `runs/smoke_optimizer.lr=1e-3-bcdf2abce8/`.

| File | What it holds | Written by |
|------|---------------|------------|
| `run.bux` | the run as one file: no modules, no alternatives. `bux train` takes it as is | `train` |
| `run.json` | the run's IR, {py:class}`bux.lang.ir.Run` as JSON | `train` |
| `metrics.jsonl` | a line of metrics every `log.every` steps, at every eval, and at every step that observes the model | `train` |
| `histograms.jsonl` | with `observe(histograms=true)`, the histograms of each step that observes the model ([Observing a run](observing.md)) | `train` |
| `checkpoints/` | the model's and the optimizer's state, with a `checkpoint` setting (orbax) | `train` |
| `profile/` | a `jax.profiler` trace, or an nsys or ncu report, with a `profile` setting | `train`, or the tool |
| `tensorboard/`, `wandb/`, `wandb_id` | a tracker's files ([Trackers](observing.md#trackers)) | `train` |
| `status.json` | whether the run is running, done, failed or stopped, and its latest metrics | `bux launch` |
| `train.log` | a sweep's worker's output | `bux launch` |

The smoke run's `run.bux`, its settings as the compiler resolved them (`4 * d_model` is
256 here):

```bux
# smoke, run 5a4f0f7d85
d_model = 64
vocab_size = 512

norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=256, activation=swish)

block = [norm attn] [norm ffn]

model = embed block^2 norm head(tied=true)

data = synthetic()
optimizer = adamw(lr=0.003)
schedule = wsd(warmup=10)
seq_len = 64
batch = 8
steps = 50
```

`bux fmt FILE` prints the same for every run of a file without training it, and
`bux json FILE` every run's IR.

## Run ids

A run's id is a hash of what it computes, so a changed run starts in a directory of
its own, and an unchanged one started again finds its checkpoints and resumes from
the latest, ending exactly where it would have ([Resuming](launching.md#resuming)).

| Keeps the id | Changes it |
|--------------|------------|
| renaming a definition | the architecture: any layer, setting or wiring |
| reaching the same settings through another file, an amendment or a sweep | `data`, `tokenizer` and `packing` |
| `log`, `eval`, `checkpoint`, `profile` | `optimizer`, `schedule`, `clip`, `loss` |
| `mesh`, `layout`, `methods`, `compiler` | `seq_len`, `batch`, `steps`, `seed` |
| `precision.eval` | the rest of `precision` |

The first line of `bux fmt`'s output shows the id, so it shows what keeps it:

```console
$ bux fmt examples/experiments/smoke.bux | head -1
# smoke, run 5a4f0f7d85
$ bux fmt examples/experiments/smoke.bux 'mesh = mesh(fsdp=2)' 'methods.remat = full' 'compiler.preset = o1' | head -1
# smoke, run 5a4f0f7d85
$ bux fmt examples/experiments/smoke.bux 'checkpoint = checkpoint(every=10)' 'eval = eval(data=synthetic, every=10)' | head -1
# smoke, run 5a4f0f7d85
$ bux fmt examples/experiments/smoke.bux 'optimizer.lr = 1e-2' | head -1
# smoke, run 54d678be8d
$ bux fmt examples/experiments/smoke.bux seed=1 | head -1
# smoke, run 9d2a4a5bf8
```

The sweep `'optimizer.lr = 1e-3 | 3e-3'` of the smoke run makes `smoke[optimizer.lr=3e-3]`
with id `5a4f0f7d85`, the plain smoke run's: its learning rate is the file's own.

:::{note}
Since `mesh`, `layout` and `methods` are not in the id, a run resumes on another mesh
or another number of nodes. The few methods that change how a checkpoint lays out the
state, `methods.repeat` and a pipeline's `rounds` and `stages`, are checked when a run
resumes ([Resuming](launching.md#resuming)).
:::

A setting or a field added after run ids existed is left out of the hash where its
value computes as runs did before it, so that the runs of an older tree keep their
ids and resume.

## The log

`log = log(every=10)`, the default, reports the step's metrics every 10 steps and at
the last, to the console and to `metrics.jsonl`:

```text
{"step": 10, "grad_norm": 0.6074022650718689, "loss": 6.2429351806640625, "lr": 0.003000000026077032, "tokens": 5120, "tokens_per_s": 7844.047464570489, "time/step_ms": 65.27242502197623, "time/data_ms": 0.08244607597589493, "time/wait_ms": 0.00047925859689712524, "time/hooks_ms": 0.0015832018107175827, "time/launch_ms": 0.3369542770087719, "time/compile_s": 0.5486519997939467}
{"step": 20, "grad_norm": 0.6416390538215637, "loss": 6.261777877807617, "lr": 0.003000000026077032, "tokens": 10240, "tokens_per_s": 77132.63393675412, "time/step_ms": 6.637916713953018, "time/hooks_ms": 9.552074875682592, "time/data_ms": 0.006195856258273125, "time/launch_ms": 0.21903333254158497, "time/wait_ms": 0.03154585137963295}
```

| Metric | What it is |
|--------|------------|
| `loss` | the first head's cross-entropy on the step's batch |
| `z_loss`, `mtp` | the objective's other terms, with a z-loss or heads after the first ([The loss](optimizers.md#the-loss)) |
| `balance_loss`, `indexer_loss`, ... | what layers report, summed over the layers that report the name ([Losses from a layer](extending.md#losses-from-a-layer)) |
| `grad_norm` | the gradient's global norm, before clipping |
| `lr` | the learning rate of the step just taken |
| `tokens` | the tokens trained on so far: steps × `batch` × `seq_len`, each step at its own under a curriculum or a batch ramp ([Curriculum](curriculum.md)) |
| `tokens_per_s` | tokens a second since the last log |
| `mfu` | model FLOPs a second over the devices' peak, on GPUs whose peak bux knows ([MFU](#mfu)) |
| `eval_loss`, `eval_bpb` | the held-out loss and bits per byte, at an eval; `<name>_loss` and `<name>_bpb` for an eval of that name ([Eval](#eval)) |
| `comm_ms`, `exposed_comm_ms`, `idle_ms` | what a profiled step spent communicating and idle ([Profiling](../performance/profiling.md)) |

The console shows `loss`, `lr`, `grad_norm`, `tokens_per_s`, `mfu` and each eval's
loss and bits per byte; `metrics.jsonl` has every metric. Reading a step's loss waits for that step, so the
log reads it once the next step is launched, and the GPU never waits for the host to
log ([The host between steps](extending.md#the-host-between-steps)).

## MFU

On a GPU whose peak bux knows, the log has `mfu`: the model's FLOPs a second over the
peak of the mesh's GPUs. The peaks are dense bfloat16 tensor FLOPs a second, from
NVIDIA's datasheets for the SXM parts ({py:data}`bux.train.flops.PEAK`):

| GPU | Peak, dense bfloat16 |
|-----|----------------------|
| B300, B200 | 2.25 PFLOP/s |
| H200, H100 | 989 TFLOP/s |
| A100 | 312 TFLOP/s |

### How FLOPs are counted

The model's FLOPs are counted once, before training, from the program of its loss's
gradient on a batch, traced on shapes alone with remat and kernels off
({py:func}`bux.train.trainer.step_flops`, {py:mod}`bux.train.flops`):

- 2·m·n·k for every matrix product and convolution, times the length of every scan
  around it;
- the larger branch of a `cond`, and a while loop's body once;
- each product once: what a backward pass computes again (`jax.checkpoint`'s, the
  logits of a flash backward) counts none.

So a new layer counts with no formula of its own.

**Attention counts only the keys each query sees**, as Megatron's MFU does and a flash
kernel computes. Its products over the keys of a row are counted apart (inside
`flops.attention`). Each step then counts them from its batch's positions
({py:meth}`bux.train.flops.Work.of`, on the host), at the keys a query sees:

- keys of the query's own document, up to the query itself;
- of those, the ones within the window, for a sliding window;
- of those, no more than a sparse attention's `top_k`.

So:

- causal attention on rows of one document counts (t + 1) / 2t of its t × t logits;
- packed documents count each one's own triangle;
- DSA counts its `top_k` keys a query in the attention, the indexer's loss and its
  target, with the indexer's ranking at every key it may pick;
- CSA's latents count whole.

On many hosts a step counts the keys of the rows its host reads. PaLM's MFU counts
attention whole, which made bux's 10% higher at 4096 tokens a row.

`bux plan` shows the FLOPs a token on rows of one document: 6.91B for the Llama-style
1.3B of `examples/benchmarks/baseline.bux` at 4096 tokens a row, 6 N of its matrices
and 0.8B of attention.

## Eval

```bux
eval = eval(data=fineweb(split=[0.99, 1]), every=125, batches=20)
```

{bux}`eval` measures the model on held-out rows every `every` steps and at the last:
`eval_loss`, the first head's cross-entropy, and `eval_bpb`, its bits per byte, where
the run has a `tokenizer`.

| Field | Default | What it is |
|-------|---------|------------|
| `name` | `eval` | what its metrics' names start with |
| `data` | required | the held-out rows: any data a run takes, a mixture too |
| `every` | `1000` | steps between evals |
| `batches` | `10` | batches of `batch` rows each, the same ones every time |
| `seq_len` | `none` | tokens a row, at least 2; none: the run's `seq_len` |
| `batch` | `none` | rows a batch; none: the run's `batch` |

An eval is a training step's forward on packed rows, with no backward. Each batch
sums what it measures over its targets, and the eval adds the sums up:

- **The loss** is the cross-entropy's sum over all the batches' targets, over their
  number. Every target counts alike, as in training: a batch with fewer of them, with
  more padding or more of a chat's prompts, counts for less.
- **Bits per byte** are the same targets' nats over the bytes of text they stand for,
  in bits: `Σ nats / (ln 2 · Σ bytes)`. A token stands for its UTF-8 bytes in the
  tokenizer's vocabulary, so bits per byte compare across tokenizers, which the loss
  does not. A special token (`<|endoftext|>`, `<|im_end|>`) stands for no text, so
  bits per byte leave its loss out, as nanochat's do. `tokens` data has them where the
  run's `tokenizer` is the one its shards were made with.

The model is in eval mode: dropout off, an MoE's router bias left as it is, and the
losses layers report left out. It multiplies as training does, or in `compute` with
`precision(eval=compute)` ([Precision](precision.md)).

Held-out rows come from a stream of their own, drawn independently of training's. A
`split` of the same documents is the usual held-out set: `split=[0, 0.99]` and
`split=[0.99, 1]` are disjoint and cover everything ([Splits](../data/sources.md#splits)).

### Several held-out sets

`eval` also takes a list of evals, each a held-out set under a name of its own:

```bux
data = 0.8 * web(split=[0, 0.99]) + 0.2 * code(split=[0, 0.99])
eval = [eval(name="web", data=web(split=[0.99, 1]), every=500),
        eval(name="code", data=code(split=[0.99, 1]), every=500)]
```

This run logs `web_loss`, `web_bpb`, `code_loss` and `code_bpb`, and the console
shows each eval on a line of its own. An eval of a mixture measures the mixture as a
whole; to measure each source, give each an eval. `bux launch`'s summary shows the
first eval's loss.

### Long rows

An eval reads rows of the run's own `seq_len` and `batch` unless it sets its own, so
that a model trained on short rows is held to long ones too, as after a
long-context phase ([Curriculum](curriculum.md)):

```bux
seq_len    = 2048
curriculum = phase(at=0.9, seq_len=16384, batch=batch // 8, data=repos)
eval = [eval(name="code", data=code_eval),
        eval(name="repo", data=repos_eval, seq_len=16384, batch=8)]
```

`bux prepare` packs the held-out documents at the eval's `seq_len`, and the run checks
before it trains that the eval's rows split over its mesh, as a phase's must. The
eval's step compiles once more, for its shape.

## Checkpoints

```bux
checkpoint = checkpoint(every=500, keep=2)
```

{bux}`checkpoint` saves the model's state and the optimizer's under the run's
`checkpoints/`, every `every` steps (1000 by default) and at the last, keeping the
`keep` most recent (2). Under `bux launch` a run that a signal stops also saves the
step it stopped at; under `bux train`, Ctrl-C loses the steps since the latest save
([Signals](launching.md#signals)).

A checkpoint holds everything a run needs to go on as if it had not stopped: the
model's state (its parameters and buffers, random keys and an MoE's router bias with
them) and the optimizer's. A run that finds checkpoints in its directory resumes from
the latest one:

```text
smoke: restored in 0.1s, resuming at step 1500
```

[Resuming](launching.md#resuming) says what a resume keeps, how it was verified, and
how it fails when it cannot work.

## Profile

`profile = profile(start=20, steps=5)` traces the 5 steps after step 20 into the run's
`profile/`, for `xprof` or TensorBoard's profile plugin; its defaults are `start=10`
and `steps=5`. On GPUs the log then reports what a traced step spent communicating and
idle. With `tool=nsys` or `tool=ncu`, `bux launch` runs the run under Nsight Systems
or Nsight Compute, whose report of those steps goes there instead.
[Profiling](../performance/profiling.md) covers the traces and those metrics.
