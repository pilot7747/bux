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
smoke: runs/smoke-5a4f0f7d85, on 1 × cpu
smoke: 156k parameters, 50 steps × 8 × 64 tokens = 25.6k tokens, adamw(lr=0.003), wsd(warmup=10)
smoke: log every 10
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step 10/50  loss 6.2429  lr 3.00e-03  grad 0.607  8,097 tok/s  ...
smoke step 20/50  loss 6.2618  lr 3.00e-03  grad 0.642  77,084 tok/s  ...
smoke step 30/50  loss 6.2731  lr 3.00e-03  grad 0.516  75,977 tok/s  ...
smoke step 40/50  loss 6.2478  lr 3.00e-03  grad 0.440  76,711 tok/s  ...
smoke step 50/50  loss 6.2633  lr 0.00e+00  grad 0.418  75,557 tok/s  ...
```

The first lines say where the run goes, its size and how often it logs. Then a line
every 10 steps has the loss, the learning rate (the schedule's warmup, its peak, then
its cooldown to 0), the gradient's norm before clipping, and the tokens a second. The
first of them includes the time to compile the step. The tokens are random, so there
is nothing to learn: the loss stays near log 512 = 6.24, a uniform guess's, where the
model starts, its weights small ([Initialization](../architectures/initialization.md)).

:::{note}
The first line, `remat=auto: none`, says which rematerialization policy the step
chose as it compiled: `none`, which keeps every activation, because the step fits in
memory ([Remat](../parallelism/remat.md)).
:::

## 5. Look at the run

The run wrote a directory under `runs/`, named for the run and its id:

```console
$ ls runs/smoke-5a4f0f7d85
metrics.jsonl  run.bux  run.json
$ head -2 runs/smoke-5a4f0f7d85/metrics.jsonl
{"step": 10, "grad_norm": 0.6074022650718689, "loss": 6.2429351806640625, "lr": 0.003000000026077032, "tokens": 5120, "tokens_per_s": 8096.7231682951115, "time/step_ms": 63.23545826599002, "time/data_ms": 0.08332501165568829, "time/wait_ms": 0.00046673230826854706, "time/hooks_ms": 0.0016250181943178177, "time/launch_ms": 0.4079831298440695, "time/compile_s": 0.5281932908110321}
{"step": 20, "grad_norm": 0.6416390538215637, "loss": 6.261777877807617, "lr": 0.003000000026077032, "tokens": 10240, "tokens_per_s": 77083.76420570571, "time/step_ms": 6.642125034704804, "time/hooks_ms": 9.541625110432506, "time/data_ms": 0.0056749675422906876, "time/launch_ms": 0.25339601561427116, "time/wait_ms": 0.02698753960430622}
```

| file            | what it is                                                     |
|-----------------|----------------------------------------------------------------|
| `run.bux`       | the run as one self-contained file: no modules, no sweeps; `bux train` takes it as is |
| `run.json`      | the run's IR: the model's graph and every training setting, as JSON |
| `metrics.jsonl` | a line of metrics at every log, and at every eval              |

The id, `5a4f0f7d85`, is a hash of what the run computes. Train the same file again
and it lands in the same directory; change what it computes and it gets a new one.
With a `checkpoint` setting, a run started again resumes from its latest checkpoint
([Train on a laptop](../tutorials/train-on-a-laptop.md)).

## 6. Change a setting from the command line

Statements after the file redefine the file's own, as if you wrote them at its end.
Train with Muon for 100 steps:

```console
$ bux train examples/experiments/smoke.bux 'optimizer = muon(lr=0.02)' steps=100
smoke: runs/smoke-e432405460, on 1 × cpu
smoke: 156k parameters, 100 steps × 8 × 64 tokens = 51.2k tokens, muon(lr=0.02), wsd(warmup=10)
smoke: log every 10
remat=auto: none in 1s (none 0.0 GiB in 0s)
smoke step  10/100  loss 6.2745  lr 2.00e-02  grad 0.600  6,170 tok/s  ...
smoke step  20/100  loss 6.3796  lr 2.00e-02  grad 0.668  56,772 tok/s  ...
...
smoke step  90/100  loss 6.2389  lr 1.00e-02  grad 0.060  61,506 tok/s  ...
smoke step 100/100  loss 6.2338  lr 0.00e+00  grad 0.058  57,780 tok/s  ...
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
bcdf2abce8  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[optimizer.lr=1e-3]
5a4f0f7d85  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[optimizer.lr=3e-3]
54d678be8d  156k    25.6k   0.2        984k       1     609KiB      1.19MiB    0%      smoke[optimizer.lr=1e-2]
```

Each run is named for what it took, and its directory is its name and its id.
`smoke[optimizer.lr=3e-3]` computes what the file alone does, so it has the same id,
`5a4f0f7d85`. The columns are each run's parameters, tokens, tokens a
parameter, model FLOPs a token, GPUs, the bytes of its parameters and of its
optimizer's state on each GPU, and the share of a step a pipeline's stages idle
(`bubble`).

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
