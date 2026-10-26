# Training

A **run** is a `.bux` file that describes a model and also says how to train it: its
data, optimizer, schedule, and how many steps of how many rows. This section is the
guide to runs: how to write one, check it, train it on a laptop or a cluster, and stop
and resume it. Read it once you can write a model ([The language](../language/index.md))
and want to train one.

## A complete run

`examples/experiments/smoke.bux` is a whole run: a tiny Llama-style model trained for 50
steps on random tokens, a few seconds on a laptop's CPU.

```bux
d_model    = 64
vocab_size = 512

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=4 * d_model, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^2 norm head(tied=true)

data      = synthetic
optimizer = adamw(lr=3e-3)
schedule  = wsd(warmup=10)
seq_len   = 64
batch     = 8
steps     = 50
```

The first block is the architecture. The second is the run: these are top-level
**settings**, written like any other definition ([Settings](settings.md)).

Check it, see what it takes, and train it:

```console
$ bux check examples/experiments/smoke.bux
d_model=64 vocab_size=512 layers=11
rmsnorm×5  gqa×2  gated_mlp×2  embed×1  head×1
50 steps × 8 × 64 tokens = 25.6k tokens
optimizer=adamw(lr=0.003) schedule=wsd(warmup=10)
data=synthetic()
$ bux plan examples/experiments/smoke.bux
1 run
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
5a4f0f7d85  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke
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

The run's files are in `runs/smoke-5a4f0f7d85/`: the run as one file (`run.bux`), its
IR (`run.json`) and its metrics (`metrics.jsonl`) ([The run
directory](running.md#the-run-directory)).

:::{tip}
Every command takes settings after the file, which replace the file's own:
`bux train examples/experiments/smoke.bux 'optimizer = muon(lr=0.02)' steps=100`.
An alternative there, `'optimizer.lr = 1e-3 | 3e-3'`, makes a sweep
([Sweeps](../language/sweeps.md)).
:::

## A sweep

A run builds on other files by name, changes one setting of what it uses, and turns
into several runs with `|`. `examples/experiments/attention_ablation.bux` is six runs,
its architecture and its training recipe taken from other files:

```bux
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

```console
$ bux plan examples/experiments/attention_ablation.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
1678540f7c  183M    3B      16.4       1.61B      1     700MiB      1.37GiB    0%      attention_ablation[mha, optimizer.lr=1e-3]
e1f8a6438d  183M    3B      16.4       1.61B      1     700MiB      1.37GiB    0%      attention_ablation[mha, optimizer.lr=3e-3]
ab996b07f8  174M    3B      17.2       1.55B      1     664MiB      1.3GiB     0%      attention_ablation[gqa, optimizer.lr=1e-3]
9ff4aff263  174M    3B      17.2       1.55B      1     664MiB      1.3GiB     0%      attention_ablation[gqa, optimizer.lr=3e-3]
eb97f05b72  170M    3B      17.6       1.53B      1     650MiB      1.27GiB    0%      attention_ablation[mqa, optimizer.lr=1e-3]
96becee690  170M    3B      17.6       1.53B      1     650MiB      1.27GiB    0%      attention_ablation[mqa, optimizer.lr=3e-3]
```

`bux launch` trains them side by side, a GPU each ([Launching](launching.md)).
With the learning rate written as `log_uniform(1e-3, 1e-2)` instead, each attention is
a search of its own, and its best learning rate is found, not picked from two
([Searching hyperparameters](searching.md)).

## The life of a run

1. **Check.** `bux check FILE` compiles the file and checks every setting against its
   type, where it is written, and that the settings fit together, before anything
   runs ([Checks](settings.md#checks)). It imports no JAX and takes a fraction of a
   second.
2. **Plan.** `bux plan FILE` lists every run: its id, parameters, tokens, FLOPs a
   token, GPUs, and the bytes of its parameters and optimizer state on each GPU. It
   builds the model from shapes alone, so it needs no GPU.
3. **Prepare.** `bux prepare FILE` tokenizes and packs the documents the runs read,
   once, into a cache that every run shares ([Data](../data/index.md)). Training
   prepares what is missing by itself, so this step is optional: it does the work
   ahead of time, or on another machine.
4. **Train or launch.** `bux train FILE` trains every run in this process, one after
   another: the command for a laptop and a debugger ([Running](running.md)).
   `bux launch FILE` trains each run on the GPUs its mesh needs, keeps a status for
   each, and stops cleanly at a signal: the command for a cluster
   ([Launching](launching.md)).
5. **The run directory.** Each run writes to `runs/<name>-<id>/`: the run as one file,
   its IR, its metrics, and its checkpoints. The id is a hash of what the run
   computes ([Run ids](running.md#run-ids)).
6. **Resume.** A run started again in the same directory resumes from its latest
   checkpoint and ends exactly where it would have
   ([Resuming](launching.md#resuming)). A run whose settings changed gets a new id,
   and so a directory of its own.

## In this section

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Settings
:link: settings
:link-type: doc

Every training setting, what is required, and the checks before anything runs.
:::

:::{grid-item-card} Optimizers and schedules
:link: optimizers
:link-type: doc

AdamW and Muon, warmup, cosine and WSD, clipping, and the loss.
:::

:::{grid-item-card} Curriculum and batch ramps
:link: curriculum
:link-type: doc

Phases that change seq_len, batch or data from a step on, and a batch that ramps up.
:::

:::{grid-item-card} Precision
:link: precision
:link-type: doc

bfloat16, MXFP8 and NVFP4: which projections, which branches, which GEMMs.
:::

:::{grid-item-card} Running
:link: running
:link-type: doc

`bux train`, the run directory, run ids, the log, MFU, eval and checkpoints.
:::

:::{grid-item-card} Observing a run
:link: observing
:link-type: doc

The console and `kubectl logs`, timers, the model by layer, W&B and TensorBoard.
:::

:::{grid-item-card} Launching
:link: launching
:link-type: doc

`bux launch`, sweeps over GPUs, array jobs, many hosts, and resuming.
:::

:::{grid-item-card} Searching hyperparameters
:link: searching
:link-type: doc

Dimensions, objectives, GP, TPE and quasi-random search, ASHA and Hyperband.
:::

:::{grid-item-card} Extending the trainer
:link: extending
:link-type: doc

Hooks, sinks, registries, objective terms, and a `Trainer` in a script.
:::

:::{grid-item-card} Design notes
:link: design
:link-type: doc

Why a run is a `.bux` file, and the config systems considered instead.
:::
::::

```{toctree}
:hidden:

settings
optimizers
curriculum
precision
running
observing
launching
searching
extending
design
```
