# An ablation sweep

In this tutorial you run an experiment the way bux is meant for: an architecture and
a training recipe from other files, used by name, and a handful of lines that say
what to vary. You plan the sweep, launch it, read the table it ends with, and compare
the runs from their metrics. Then you tie two settings together with labels, and
sweep the wiring itself.

It builds on [Train on a laptop](train-on-a-laptop.md). The six runs train in about a
minute on a laptop's CPU.

## 1. Use an architecture by name

`examples/architectures/llama3_8b.bux` is Llama 3 8B:

```bux
# Llama 3 8B (Meta, 2024): the reference pre-norm decoder.
d_model    = 4096
vocab_size = 128256

norm = rmsnorm(eps=1e-5)
attn = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))
ffn  = gated_mlp(intermediate_size=14336, activation=swish)

block = [norm attn] [norm ffn]
model = embed block^32 norm head
```

`use llama3_8b` brings in its definitions. A module is found by its name, never by a
path: in the using file's folder first, then under the folders that `[tool.bux]
path` lists in `pyproject.toml`, which in bux's checkout is `examples/`.

Definitions after the `use` take the place of llama3_8b's **everywhere**, including
inside its own definitions. So redefining `d_model`, or amending `attn`, changes the
attention inside llama3_8b's `block`, which you never rewrite. An amendment,
`attn.num_heads = 4`, changes one setting and keeps the rest: it means
`attn = attn(num_heads=4)`.

## 2. Write a recipe

A recipe is a module of training settings with no model. Create
`mine/laptop_recipe.bux`:

```bux
# How to train on a laptop: random tokens, AdamW, a short warmup-stable-decay schedule.
data       = synthetic
optimizer  = adamw(lr=3e-3)
schedule   = wsd(warmup=20)
seq_len    = 128
batch      = 8
steps      = 300

log        = log(every=50)
eval       = eval(data=synthetic, every=150, batches=4)
checkpoint = checkpoint(every=100)
```

```console
$ bux check mine/laptop_recipe.bux
a module: it defines data, optimizer, schedule, seq_len, batch, steps, log, eval, checkpoint, and no model
```

`examples/experiments/recipes/fineweb_wsd.bux` is a real one: an 80/20 mix of
FineWeb-Edu and Stack-Edu in Llama 3's tokens, AdamW, a WSD schedule over a budget of
10B tokens, evaluation and checkpoints. Swap `use laptop_recipe` for
`use fineweb_wsd` once you have its data ([Your own data](your-own-data.md)).

## 3. Say what to vary

Create `mine/ablation.bux`:

```bux
# MHA, GQA or MQA, at which learning rate? Llama 3 8B's block, scaled down to a laptop.
use llama3_8b
use laptop_recipe

d_model    = 128
vocab_size = 512
ffn.intermediate_size = 352
model = embed block^4 norm head(tied=true)

attn.num_heads    = 4
attn.num_kv_heads = mha: 4 | gqa: 2 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3
```

The first block scales Llama 3 8B down: four blocks of width 128, a smaller MLP, a
tied head. The last three lines are the experiment:

- `|` separates **alternatives**, and each `|` as written is an axis of the sweep.
  The axes cross, so this is 3 × 2 = 6 runs.
- `mha:`, `gqa:` and `mqa:` are **labels**. They name the alternatives, and the runs
  that take them.

```console
$ bux check mine/ablation.bux
d_model=128 vocab_size=512 layers=19
rmsnorm×9  gqa×4  gated_mlp×4  embed×1  head×1
300 steps × 8 × 128 tokens = 307k tokens
optimizer=adamw(lr=0.001) schedule=wsd(warmup=20)
data=synthetic()
sweep: 6 runs over attn.num_kv_heads (mha | gqa | mqa) × optimizer.lr (1e-3 | 3e-3)
```

`bux check` checks every alternative once, with the others at their first, and
reports the first run. `bux fmt` prints every run as a file of its own, with nothing
used and nothing to vary, which shows what each one inherited:

```console
$ bux fmt mine/ablation.bux
# ablation[mha, optimizer.lr=1e-3], run f1814da7d1
d_model = 128
vocab_size = 512

norm = rmsnorm(eps=1e-05)
attn = gqa(num_heads=4, num_kv_heads=4, rope=rope(base=500000.0))
ffn = gated_mlp(intermediate_size=352, activation=swish)

block = [norm attn] [norm ffn]

model = embed block^4 norm head(tied=true)

data = synthetic()
optimizer = adamw(lr=0.001)
schedule = wsd(warmup=20)
seq_len = 128
batch = 8
steps = 300
eval = eval(data=synthetic(), every=150, batches=4)
checkpoint = checkpoint(every=100)
log = log(every=50)

# ablation[mha, optimizer.lr=3e-3], run 39f756c12c
...
```

The attention kept llama3_8b's RoPE base and the norm its epsilon; only what the file
says changed.

## 4. Plan it

```console
$ bux plan mine/ablation.bux
6 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
f1814da7d1  870k    307k    0.4        5.61M      1     3.32MiB     6.63MiB    0%      ablation[mha, optimizer.lr=1e-3]
39f756c12c  870k    307k    0.4        5.61M      1     3.32MiB     6.63MiB    0%      ablation[mha, optimizer.lr=3e-3]
b247174ad8  804k    307k    0.4        5.21M      1     3.07MiB     6.13MiB    0%      ablation[gqa, optimizer.lr=1e-3]
b42eb5ba19  804k    307k    0.4        5.21M      1     3.07MiB     6.13MiB    0%      ablation[gqa, optimizer.lr=3e-3]
0282510df4  771k    307k    0.4        5.02M      1     2.94MiB     5.88MiB    0%      ablation[mqa, optimizer.lr=1e-3]
f355634194  771k    307k    0.4        5.02M      1     2.94MiB     5.88MiB    0%      ablation[mqa, optimizer.lr=3e-3]
```

The runs are in grid order, the first axis written varying slowest. Each is named
for the file and what it took: a labeled alternative by its label, an unlabeled one
by its setting and value. Fewer key and value heads make fewer parameters and FLOPs.

## 5. Launch it

`bux launch` trains every run of the sweep that is not done yet. On a machine with
GPUs it trains them side by side, each on the GPUs its mesh needs; with none, one at
a time:

```console
$ bux launch mine/ablation.bux
training 6 runs, one at a time
ablation[mha, optimizer.lr=1e-3]: training on cpu of my-laptop with jax 0.12.0
remat=auto: none in 1s (none 0.1 GiB in 0s)
ablation[mha, optimizer.lr=1e-3] step  50/300  loss 6.2541  lr 1.00e-03  grad 0.413  18,054 tok/s  ...
...
ablation[mha, optimizer.lr=1e-3] step 300/300  eval loss 6.2444
ablation[mha, optimizer.lr=1e-3]: done
ablation[mha, optimizer.lr=3e-3]: training on cpu of my-laptop with jax 0.12.0
...
ablation[mqa, optimizer.lr=3e-3]: done
6 runs: 6 done
state  step     loss    eval loss  run
done   300/300  6.2362  6.2444     ablation[mha, optimizer.lr=1e-3]
done   300/300  6.2363  6.2416     ablation[mha, optimizer.lr=3e-3]
done   300/300  6.2367  6.2436     ablation[gqa, optimizer.lr=1e-3]
done   300/300  6.2358  6.2424     ablation[gqa, optimizer.lr=3e-3]
done   300/300  6.2361  6.2444     ablation[mqa, optimizer.lr=1e-3]
done   300/300  6.2363  6.2437     ablation[mqa, optimizer.lr=3e-3]
```

The table at the end has each run's state, its step, and its latest loss and eval
loss. Stop the launcher at any point (Ctrl-C) and launch it again: the runs that are
done stay done, a stopped one resumes from its checkpoint, and the rest start. A run
that fails leaves the others training.

:::{note}
On random tokens every run ends at log 512 ≈ 6.24, and the differences are noise.
The point here is the mechanics; with real data
([Your own data](your-own-data.md)) the same file compares the attentions.
:::

To train only some of the runs, `--only TEXT` keeps those whose name has `TEXT`, and
`--index N` the run at `N` in the plan's order, from 0, as an array job's task does:

```sh
bux launch mine/ablation.bux --only mqa      # the two MQA runs
bux launch mine/ablation.bux --index 3       # ablation[gqa, optimizer.lr=3e-3]
```

## 6. Compare the runs

Each run has its own directory, `runs/<name>-<id>/`, with its `run.json` and its
`metrics.jsonl`:

```console
$ ls -1d runs/ablation*
runs/ablation_gqa_optimizer.lr=1e-3-b247174ad8
runs/ablation_gqa_optimizer.lr=3e-3-b42eb5ba19
runs/ablation_mha_optimizer.lr=1e-3-f1814da7d1
runs/ablation_mha_optimizer.lr=3e-3-39f756c12c
runs/ablation_mqa_optimizer.lr=1e-3-0282510df4
runs/ablation_mqa_optimizer.lr=3e-3-f355634194
```

A few lines of Python collect the final eval loss of each:

```python
import json
from pathlib import Path

for folder in sorted(Path("runs").glob("ablation*")):
    name = json.loads((folder / "run.json").read_text())["name"]
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").open()]
    evals = [m["eval_loss"] for m in metrics if "eval_loss" in m]
    print(f"{name:34}  final eval loss {evals[-1]:.4f}")
```

```text
ablation[gqa, optimizer.lr=1e-3]    final eval loss 6.2436
ablation[gqa, optimizer.lr=3e-3]    final eval loss 6.2424
ablation[mha, optimizer.lr=1e-3]    final eval loss 6.2444
ablation[mha, optimizer.lr=3e-3]    final eval loss 6.2416
ablation[mqa, optimizer.lr=1e-3]    final eval loss 6.2444
ablation[mqa, optimizer.lr=3e-3]    final eval loss 6.2437
```

Every line also has `tokens` and `tokens_per_s`, so the same loop plots loss against
tokens or compares speeds.

## 7. Tie settings together with labels

Alternatives with the **same labels** are one axis: they are picked together, not
crossed. A width sweep with a learning rate for each width is three runs, not nine.
Create `mine/widths.bux`:

```bux
# Three widths, each with its own learning rate: three runs, not nine.
use llama3_8b
use laptop_recipe

vocab_size = 512
model = embed block^4 norm head(tied=true)

d_model      = s: 64   | m: 128  | l: 192
optimizer.lr = s: 4e-3 | m: 3e-3 | l: 2e-3

attn.num_heads        = d_model // 32
attn.num_kv_heads     = d_model // 64
ffn.intermediate_size = 8 * d_model // 3
```

Everything written in terms of `d_model` follows it in every run:

```console
$ bux check mine/widths.bux
...
sweep: 3 runs over d_model (s | m | l)
$ bux plan mine/widths.bux
3 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
20e418e39e  213k    307k    1.4        1.47M      1     832KiB      1.63MiB    0%      widths[s]
6264630896  787k    307k    0.4        5.11M      1     3MiB        6MiB       0%      widths[m]
46db229743  1.72M   307k    0.2        10.9M      1     6.57MiB     13.1MiB    0%      widths[l]
```

Alternatives that share some labels but not all are an error, as a forgotten one
would be:

```console
$ bux check mine/widths.bux 'optimizer.lr = s: 4e-3 | m: 3e-3'
error: these alternatives share some labels with others, but not all: m, s against l, m, s
 --> <command line>:1:19
  |
1 | optimizer.lr = s: 4e-3 | m: 3e-3
  |                   ^^^^^^^^^^^^^^
  = hint: alternatives with the same labels are picked together, so give them all the same labels, or none in common
```

:::{tip}
`name.field` reads a setting back, but as written in the definition the amendment
replaces. In `widths.bux`, `attn.num_kv_heads = attn.num_heads // 2` would read
llama3_8b's 32 heads, not `d_model // 32`; write both in terms of `d_model`, as above.
:::

## 8. Sweep the wiring

`|` works anywhere a value or a block does, so the architecture itself is a setting
like any other. Cross the ablation with two block shapes from the command line,
pre-norm and a parallel block:

```console
$ bux check mine/ablation.bux 'block = pre: [norm attn] [norm ffn] | par: [norm (attn + ffn)]'
...
sweep: 12 runs over attn.num_kv_heads (mha | gqa | mqa) × optimizer.lr (1e-3 | 3e-3) × block (pre | par)
```

The command line takes the same statements a file does, so `use`, amendments, the
command line and alternatives are one mechanism: each gives a name another
definition, and everything written in terms of that name follows.

## What you learned

| you wrote                        | it means                                          |
|----------------------------------|---------------------------------------------------|
| `use llama3_8b`                  | its definitions, found by name, under yours        |
| `attn.num_heads = 4`             | `attn = attn(num_heads=4)`, everywhere `attn` is used |
| `a \| b \| c`                    | an axis of three runs, crossed with the others     |
| `s: a \| m: b` twice             | one axis: the labels pick them together            |
| `bux plan`, `bux launch`         | the runs and their sizes; train what is left of them |

Next, [your own data](your-own-data.md) replaces the random tokens with text that can
be learned. See [Modules](../language/modules.md) and [Sweeps](../language/sweeps.md)
for the rules, and [Launching](../training/launching.md) for sweeps on GPUs, array
jobs and clusters.
