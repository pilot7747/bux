# Quickstart

In ten minutes on a laptop, with no GPU: check a model, draw it, train it, read what
the run wrote, change a setting from the command line, and plan a sweep. Every output
below is real, from a laptop's CPU.

Install bux first ([Installation](installation.md)):

```sh
git clone https://github.com/pilot7747/bux.git
cd bux
uv sync
source .venv/bin/activate      # or prefix every command with `uv run`
```

## 1. Read a run

`examples/experiments/smoke.bux` is a whole experiment: a tiny Llama-style model and
how to train it, on random tokens.

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

The first half is the model. `[f]` means `x + f(x)`, a residual branch, so
`[norm attn] [norm ffn]` is a pre-norm Transformer block and `block^2` is two of
them. The second half says how to train it: the data, the optimizer, the learning
rate schedule, and the shape of the batches.

## 2. Check it

`bux check` compiles the file and checks every setting, without importing JAX:

```console
$ bux check examples/experiments/smoke.bux
d_model=64 vocab_size=512 layers=11
rmsnorm×5  gqa×2  gated_mlp×2  embed×1  head×1
50 steps × 8 × 64 tokens = 25.6k tokens
optimizer=adamw(lr=0.003) schedule=wsd(warmup=10)
data=synthetic()
```

A mistake is reported at the line that made it, with a suggestion; see
[Reading errors](../tutorials/first-architecture.md#7-read-the-errors).

## 3. Draw it

`bux render` draws the model as a circuit: the line is the residual stream, and each
`[...]` is a branch that leaves it and adds back.

```console
$ bux render examples/experiments/smoke.bux
model
──embed──┤block├×2──norm──head──

block
──┬────────────⊕──┬───────────⊕──
  └─norm──attn─┘  └─norm──ffn─┘
```

`--expand` draws the blocks inline:

```console
$ bux render examples/experiments/smoke.bux --expand
model
         ╭─ ×2 ──────────────────────────╮
──embed──┤─┬────────────⊕──┬───────────⊕─├──norm──head──
         │ └─norm──attn─┘  └─norm──ffn─┘ │
         ╰───────────────────────────────╯
```

## 4. Train it

```console
$ bux train examples/experiments/smoke.bux
remat=auto: none in 0s (none 0.0 GiB in 0s)
smoke step 10/50  loss 6.6252  lr 3.00e-03  grad 1.614  9,176 tok/s
smoke step 20/50  loss 6.5475  lr 3.00e-03  grad 1.038  75,865 tok/s
smoke step 30/50  loss 6.3513  lr 3.00e-03  grad 0.554  73,936 tok/s
smoke step 40/50  loss 6.2974  lr 3.00e-03  grad 0.458  75,005 tok/s
smoke step 50/50  loss 6.2854  lr 0.00e+00  grad 0.395  76,099 tok/s
```

A line every 10 steps has the loss, the learning rate (the schedule's warmup, its
peak, then its cooldown to 0), the gradient's norm before clipping, and the tokens a
second. The first line includes the time to compile the step. The tokens are random,
so the loss falls to log 512 = 6.24 and no lower: there is nothing more to learn.

:::{note}
The first line, `remat=auto: none`, says which rematerialization policy the step
chose as it compiled: `none`, which keeps every activation, because the step fits in
memory ([Remat](../parallelism/remat.md)).
:::

## 5. Look at the run

The run wrote a directory under `runs/`, named for the run and its id:

```console
$ ls runs/smoke-f114ac17ae
metrics.jsonl  run.bux  run.json
$ head -2 runs/smoke-f114ac17ae/metrics.jsonl
{"step": 10, "grad_norm": 1.6137837171554565, "loss": 6.625185489654541, "lr": 0.003000000026077032, "tokens": 5120, "tokens_per_s": 9175.617634801478}
{"step": 20, "grad_norm": 1.0376627445220947, "loss": 6.547478199005127, "lr": 0.003000000026077032, "tokens": 10240, "tokens_per_s": 75865.15237532384}
```

| file            | what it is                                                     |
|-----------------|----------------------------------------------------------------|
| `run.bux`       | the run as one self-contained file: no modules, no sweeps; `bux train` takes it as is |
| `run.json`      | the run's IR: the model's graph and every training setting, as JSON |
| `metrics.jsonl` | a line of metrics at every log, and at every eval              |

The id, `f114ac17ae`, is a hash of what the run computes. Train the same file again
and it lands in the same directory; change what it computes and it gets a new one.
With a `checkpoint` setting, a run started again resumes from its latest checkpoint
([Train on a laptop](../tutorials/train-on-a-laptop.md)).

## 6. Change a setting from the command line

Statements after the file redefine the file's own, as if you wrote them at its end.
Train with Muon for 100 steps:

```console
$ bux train examples/experiments/smoke.bux 'optimizer = muon(lr=0.02)' steps=100
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step 10/100  loss 6.5168  lr 2.00e-02  grad 1.304  6,636 tok/s
smoke step 20/100  loss 6.3315  lr 2.00e-02  grad 0.491  58,525 tok/s
...
smoke step 90/100  loss 6.2401  lr 1.00e-02  grad 0.083  57,099 tok/s
smoke step 100/100  loss 6.2359  lr 0.00e+00  grad 0.070  58,095 tok/s
```

Quote a statement that has spaces or parentheses. `optimizer.lr=1e-2` changes one
setting of the optimizer and keeps the rest; `'optimizer = muon(lr=0.02)'` replaces
it.

## 7. Plan a sweep

`|` separates alternatives, and a file with alternatives is several runs. Try three
learning rates from the command line, and plan them with `bux plan`, which needs no
GPU:

```console
$ bux plan examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3 | 1e-2'
3 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
e3a8d2164a  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[optimizer.lr=1e-3]
f114ac17ae  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[optimizer.lr=3e-3]
f5babdd0aa  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[optimizer.lr=1e-2]
```

Each run is named for what it took, and its directory is its name and its id.
`smoke[optimizer.lr=3e-3]` computes what the file alone does, so it has the same id,
`f114ac17ae`. The columns are each run's parameters, tokens, tokens a
parameter, model FLOPs a token, GPUs, and the bytes of its parameters and of its
optimizer's state on each GPU.

`bux train` trains a sweep's runs one after another; `bux launch` trains them side by
side, a GPU each, and picks up where it stopped when launched again
([Launching](../training/launching.md)).

## Next steps

- [Concepts](concepts.md): the ideas behind what you just ran.
- [Your first architecture](../tutorials/first-architecture.md): write a model from
  scratch and evolve it.
- [An ablation sweep](../tutorials/ablation-sweep.md): compare variants the way bux is
  meant to be used.
- [The language](../language/index.md) and [the reference](../reference/index.md):
  every operator, layer and setting.
