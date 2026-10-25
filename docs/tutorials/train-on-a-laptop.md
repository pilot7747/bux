# Train on a laptop

In this tutorial you turn a model into a run, train it on your CPU, and read what it
reports: the console log, `metrics.jsonl`, held-out evaluation and checkpoints. Then
you stop a longer run in the middle and start it again, and it carries on from where
it stopped.

You need the Llama-style model of [Your first architecture](first-architecture.md),
made smaller so that a CPU trains it in seconds. The data is random tokens, so
nothing needs downloading; [Your own data](your-own-data.md) trains on real text.

## 1. Turn the model into a run

A run is a `.bux` file that also says how to train its model. Create `mine/tiny.bux`:

```bux
# A tiny Llama-style model, trained on random tokens: a run that fits a laptop.
d_model    = 128
vocab_size = 512

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=352, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^4 norm head(tied=true)

data      = synthetic
optimizer = adamw(lr=3e-3)
schedule  = wsd(warmup=20, decay=0.2)
seq_len   = 128
batch     = 8
steps     = 200

log        = log(every=20)
eval       = eval(data=synthetic, every=100, batches=4)
checkpoint = checkpoint(every=50)
```

The model is the one you wrote, at a width of 128 and a vocabulary of 512. Below it,
each training setting is a top-level definition:

| setting      | here                         | what it says                                         |
|--------------|------------------------------|------------------------------------------------------|
| `data`       | `synthetic`                  | uniformly random tokens, for smoke tests and speed   |
| `optimizer`  | `adamw(lr=3e-3)`             | AdamW at a peak learning rate of 3e-3                |
| `schedule`   | `wsd(warmup=20, decay=0.2)`  | warm up over 20 steps, hold, cool down to 0 over the last 20% |
| `seq_len`    | `128`                        | tokens a row                                         |
| `batch`      | `8`                          | rows a step                                          |
| `steps`      | `200`                        | optimizer steps                                      |
| `log`        | `log(every=20)`              | a line of metrics every 20 steps                     |
| `eval`       | `eval(data=..., every=100, batches=4)` | a held-out loss every 100 steps, over 4 batches |
| `checkpoint` | `checkpoint(every=50)`       | save the model and the optimizer every 50 steps, keeping the latest 2 |

`data`, `optimizer`, `seq_len`, `batch` and `steps` are required; the rest have
defaults. [Training settings](../training/settings.md) lists them all, and the
[reference](../reference/language/training.md) has every field of every config:
{bux}`synthetic`, {bux}`adamw`, {bux}`wsd`, {bux}`log`, {bux}`eval` and
{bux}`checkpoint` here.

## 2. Check it

```console
$ bux check mine/tiny.bux
d_model=128 vocab_size=512 layers=19
rmsnorm×9  gqa×4  gated_mlp×4  embed×1  head×1
200 steps × 8 × 128 tokens = 205k tokens
optimizer=adamw(lr=0.003) schedule=wsd(warmup=20)
data=synthetic()
```

The settings are checked against each other too. A run too short for its schedule is
an error at the schedule:

```console
$ bux check mine/tiny.bux steps=20
...
error: the schedule takes 24 steps (20 of warmup and 4 of cooldown), but the run has 20
  --> mine/tiny.bux:13:13
   |
13 | schedule  = wsd(warmup=20, decay=0.2)
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^
```

`bux plan` shows the run's size before it trains: its id, parameters, tokens, model
FLOPs a token, and the bytes of its parameters and its optimizer's state on each
device.

```console
$ bux plan mine/tiny.bux
1 run
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
d48dd7d011  804k    205k    0.3        5.21M      1     3.07MiB     6.13MiB    0%      tiny
```

## 3. Train it

```console
$ bux train mine/tiny.bux
remat=auto: none in 1s (none 0.1 GiB in 0s)
tiny step 20/200  loss 6.3815  lr 3.00e-03  grad 0.621  13,138 tok/s
tiny step 40/200  loss 6.3131  lr 3.00e-03  grad 0.418  28,251 tok/s
tiny step 60/200  loss 6.2725  lr 3.00e-03  grad 0.360  27,732 tok/s
tiny step 80/200  loss 6.2650  lr 3.00e-03  grad 0.347  28,332 tok/s
tiny step 100/200  loss 6.2796  lr 3.00e-03  grad 0.343  24,213 tok/s
tiny step 100/200  eval loss 6.2760
tiny step 120/200  loss 6.2559  lr 3.00e-03  grad 0.337  28,217 tok/s
tiny step 140/200  loss 6.2747  lr 3.00e-03  grad 0.322  27,784 tok/s
tiny step 160/200  loss 6.2723  lr 3.00e-03  grad 0.294  27,114 tok/s
tiny step 180/200  loss 6.2557  lr 1.50e-03  grad 0.571  27,366 tok/s
tiny step 200/200  loss 6.2419  lr 0.00e+00  grad 10.304  26,585 tok/s
tiny step 200/200  eval loss 6.2553
```

How to read it:

- The first line says which rematerialization policy the step chose as it first
  compiled (`methods.remat = auto`): `none`, as the step holds 0.1 GiB and a CPU has
  room for it.
- Each log line is the training loss, the learning rate, the global gradient norm
  before clipping (`clip`, 1.0 by default), and the tokens a second since the last
  line. The first line's speed includes compiling the step.
- `lr` holds at its peak after the warmup, then the cooldown takes it to 0 over the
  last 40 steps.
- The eval lines are the loss on held-out batches: the same 4 every time, drawn apart
  from the training batches.
- On a GPU whose peak bux knows, the line also has the model FLOPs utilization,
  `mfu` ([Running](../training/running.md)).

:::{note}
The tokens are uniformly random, so no model can predict them: the loss falls to
log 512 = 6.238 and stops there. That is all a synthetic run shows, that the model
trains; [Your own data](your-own-data.md) trains on text that can be learned.
:::

## 4. Read the run's directory

The run wrote everything into `runs/<name>-<id>/`:

```console
$ ls runs/tiny-d48dd7d011 runs/tiny-d48dd7d011/checkpoints
runs/tiny-d48dd7d011:
checkpoints  metrics.jsonl  run.bux  run.json

runs/tiny-d48dd7d011/checkpoints:
150  200
$ tail -n 3 runs/tiny-d48dd7d011/metrics.jsonl
{"step": 180, "grad_norm": 0.5706921815872192, "loss": 6.255720615386963, "lr": 0.001500000013038516, "tokens": 184320, "tokens_per_s": 27366.093675823613}
{"step": 200, "grad_norm": 10.303742408752441, "loss": 6.241860866546631, "lr": 0.0, "tokens": 204800, "tokens_per_s": 26584.77769143028}
{"step": 200, "eval_loss": 6.255273342132568}
```

`metrics.jsonl` has a line at every log, with every metric the console shows and more
(`tokens` so far), and a line at every eval. Read it with a few lines of Python:

```python
import json

with open("runs/tiny-d48dd7d011/metrics.jsonl") as f:
    lines = [json.loads(line) for line in f]

train = [(m["step"], round(m["loss"], 4)) for m in lines if "loss" in m]
evals = [(m["step"], round(m["eval_loss"], 4)) for m in lines if "eval_loss" in m]
print("train:", train[0], "...", train[-1])
print("eval: ", evals)
```

```text
train: (20, 6.3815) ... (200, 6.2419)
eval:  [(100, 6.276), (200, 6.2553)]
```

`run.bux` is the run as one self-contained file, with no modules and no
alternatives, which `bux train` takes as it is, and `run.json` is its IR. `checkpoints/` keeps the latest two saves
(`checkpoint(keep=2)`, the default) as orbax checkpoints of the model's and the
optimizer's state.

The id, `d48dd7d011`, is a hash of what the run computes. `log`, `eval` and
`checkpoint` are not part of it, so changing how often the run logs, evaluates or
saves keeps the id and the directory; changing `steps`, the model or the optimizer
makes a new one.

## 5. Train it again

Start the same run again. It finds its checkpoint at step 200, the last step, and has
nothing left to do:

```console
$ bux train mine/tiny.bux
tiny: restored in 0.2s, resuming at step 200
```

## 6. Stop a run and resume it

Make the run longer so that you have time to stop it. `steps=3000` on the command
line changes what the run computes, so it gets a new id and a directory of its own.
This time train it with `bux launch`, the launcher, which stops cleanly at a signal:

```console
$ bux launch mine/tiny.bux steps=3000
tiny: training on cpu of my-laptop with jax 0.12.0
remat=auto: none in 1s (none 0.1 GiB in 0s)
tiny step 20/3000  loss 6.3815  lr 3.00e-03  grad 0.621  13,556 tok/s
...
tiny step 200/3000  loss 6.2623  lr 3.00e-03  grad 0.266  26,937 tok/s
tiny step 200/3000  eval loss 6.2609
tiny: stopped at step 200 by SIGTERM; launching it again resumes it from its latest checkpoint
```

Here a `kill` from another terminal sent SIGTERM, as a cluster does when it preempts a
job. The launcher finished the step it was at, saved it, and exited with 143
(128 + 15, SIGTERM's number). Launch it again with the same command, and press
Ctrl-C a few seconds in:

```console
$ bux launch mine/tiny.bux steps=3000
tiny: training on cpu of my-laptop with jax 0.12.0
tiny: restored in 0.2s, resuming at step 200
remat=auto: none in 1s (none 0.1 GiB in 0s)
tiny step 220/3000  loss 6.2564  lr 3.00e-03  grad 0.274  13,897 tok/s
...
tiny step 380/3000  loss 6.2461  lr 3.00e-03  grad 0.553  28,166 tok/s
tiny: stopped at step 381 by SIGINT; launching it again resumes it from its latest checkpoint
```

Ctrl-C (SIGINT) stops it the same way, after step 381, which it saves although it is
no multiple of 50. Launch it once more, and it trains to the end:

```console
$ bux launch mine/tiny.bux steps=3000
tiny: training on cpu of my-laptop with jax 0.12.0
tiny: restored in 0.2s, resuming at step 381
remat=auto: none in 1s (none 0.1 GiB in 0s)
tiny step 400/3000  loss 6.2491  lr 3.00e-03  grad 0.196  12,293 tok/s
...
tiny step 3000/3000  loss 6.2384  lr 0.00e+00  grad 0.068  27,003 tok/s
tiny step 3000/3000  eval loss 6.2385
tiny: done
$ bux launch mine/tiny.bux steps=3000
1 of 1 run done already
```

The resumed run trains as if it had never stopped: the checkpoint holds the model's
and the optimizer's state with their random keys, and batch `i` depends only on the
seed and `i`, so it reads exactly what it would have read. `metrics.jsonl` keeps one
line per log across all three launches; it drops any lines past the checkpoint that
a crash left behind.

The launcher also writes a `status.json` into the run's directory:

```console
$ cat runs/tiny-002cbdb08a/status.json
{
  "state": "stopped",
  "host": "my-laptop",
  "pid": 8710,
  "gpus": null,
  "started": "2026-10-25T21:23:33+00:00",
  "ended": "2026-10-25T21:23:42+00:00",
  "error": null,
  "step": 200,
  "metrics": {
    "grad_norm": 0.26626190543174744,
    "loss": 6.2623291015625,
    "lr": 0.003000000026077032,
    "tokens": 204800,
    "tokens_per_s": 26937.06913439296,
    "eval_loss": 6.2608559131622314
  }
}
```

That is the one after the first stop; its `state` is `running`, `done`, `failed` or
`stopped`. A run that is `done` is not trained again, which is how the last launch
knew.

:::{warning}
`bux train` does not catch signals: Ctrl-C raises a `KeyboardInterrupt` and the step
it was at is lost. Started again, the run resumes from its latest periodic checkpoint
(here a multiple of 50). Use `bux launch` for anything you may need to stop, and give
every run you care about a `checkpoint` setting: without one, a stopped run starts
over.
:::

| signal                | `bux launch` does                                  |
|-----------------------|----------------------------------------------------|
| SIGTERM or Ctrl-C     | finishes the step, saves it (with a `checkpoint` setting), exits with 128 + the signal |
| a second one, a second or more later | stops at once                       |

## What you learned

- A run is a model plus a handful of training settings in one file, checked before it
  trains.
- `bux train` trains in this process; `bux launch` trains to the end however often it
  is stopped.
- Everything a run writes is in `runs/<name>-<id>/`: `metrics.jsonl` for its numbers,
  `checkpoints/` to resume, `run.bux` to reproduce it.

Next, [an ablation sweep](ablation-sweep.md) compares several runs at once. For every
setting and option, see [Running](../training/running.md) and
[Launching](../training/launching.md).
