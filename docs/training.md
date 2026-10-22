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
| `precision`  | `precision(compute=bfloat16)`: the dtype forward and backward compute in; parameters stay float32 | `precision`: `float32` |
| `mesh`       | `mesh(data=..., fsdp=..., tensor=...)`: the device grid | `mesh`: every device on `data` |
| `layout`     | `layout(...)`: which mesh axes each named dimension splits over | `layout` |
| `methods`    | `methods(attention=..., moe=..., microbatches=...)`: how what communicates is computed, and the microbatches a step takes its batch in | `methods` |

The settings of each config are pydantic fields in `bux/train/config.py`, and the
editor shows them on hover and completes them, as it does for layers. In short:

- `adamw(lr, b1=0.9, b2=0.95, eps, weight_decay=0.1)` decays matrices only, not norm
  scales, biases or Engram's tables.
- `muon(lr, momentum=0.95, nesterov, ns_steps=5, weight_decay=0.1, consistent_rms=0.2,
  adam=none)` trains the matrices inside the blocks with Muon, stacked expert weights
  one expert at a time, and everything else (the embedding, the head, vectors) with
  `adam`, which defaults to `adamw(lr=lr)`. With `consistent_rms=0.2` (Moonlight) one
  learning rate suits both; `none` scales updates as Keller Jordan's Muon does.
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
normalized over them. With `block_size` (Qwen3.8 QSA) a block counts as its most
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
profiling at other steps, or splitting it over devices another way (`mesh`, `layout`,
`methods`) keeps it, and anything else changes it. So a changed run starts in a
directory of its own, and an unchanged one started again resumes from its latest
checkpoint, ending exactly where it would have.

The log has the step's metrics: `loss` and every term of the objective, `grad_norm`
before clipping, `lr`, `tokens` so far and `tokens_per_s` since the last log. On a GPU
whose peak it knows (B300, B200, H200, H100, A100: dense bf16, from NVIDIA's
datasheets), it also has `mfu`: the model's FLOPs a second over the peak of the
mesh's GPUs. A step's FLOPs are counted once, before training, from the program of
its loss's gradient on a batch, traced on shapes alone with remat and kernels off
(`trainer.step_flops`, `bux/train/flops.py`): 2·m·n·k for every matrix product and
convolution, times the length of every scan around it. So a new layer counts with
no formula of its own, a rematerialized forward is not counted twice, and attention
counts its t × t scores whole, as PaLM's MFU does, though a causal kernel computes
half of them. `bux plan` shows the FLOPs a token: 7.7G for a Llama-style 1.3B at
4096 tokens a row, 6 N of its matrices and 1.6G of attention.

`profile = profile(start=20, steps=5)` traces the 5 steps after step 20 into
`profile/`, waiting for the step before it starts and stops, for `xprof` or
TensorBoard's profile plugin (`tensorboard --logdir runs/<run>/profile`).

## Speed on GPUs

What XLA compiles slowly on a GPU runs by a kernel there, the same computation on a
CPU by XLA; none of it is in a run's id. Each is in `bux.architecture` and falls back
by itself where it cannot run, and all of them do inside `bux.kernels.off()` or with
`BUX_KERNELS=off`, to debug one:

| what | by | on a B300, against XLA |
|------|----|------------------------|
| softmax attention | `methods.attention = auto`, the default: tokamax's flash attention | 7x, and no t × t logits |
| linear attention, RWKV-7 | `chunkwise`: chunks of 64 positions, the state across them by a Pallas kernel (`affine_scan`) | 14-17x a scan over positions |
| KDA with `gate_lower_bound`, DeltaNet | cuDNN's fused KDA kernel (`fused_delta`) | 13x `chunkwise` |
| MoE experts | `grouped.ragged_dot`: tokamax's Mosaic and Triton kernels | 27x `jax.lax.ragged_dot` |
| MoE dispatch | `pick_rows`, `permuted`: gathers where XLA would scatter | 4x |

Tokens a second on B300s, 8 × 4096 tokens a GPU, `precision(compute=bfloat16)`, and
the log's `mfu`:

| model (d = 2048, 16 layers) | 1 GPU | 8, `mesh(fsdp=8)` |
|-----------------------------|-------|-------------------|
| Llama-style, 1.3B           | 144k, 49% MFU | 1.12M, 48% |
| Gated DeltaNet hybrid, `"DDDA"` | 93k, 30% | 720k |
| KDA hybrid, `"KKKA"`        | 58k   |                   |
| KDA with `gate_lower_bound=-5` | 110k, 36% |            |
| MoE, 64 experts, top 8      | 102k, 35% | 834k          |

Batches go to the devices two ahead of the step (`prefetch`), and `methods(donate=true)`
lets a step's new state take the old one's memory: the 1.3B's step then peaks at 70.5
GiB instead of 81.1 on one GPU ([Donation](parallelism.md#donation-methodsdonate),
which says why it is off by default).

bux sets one XLA flag of its own (`bux/runtime.py`), for expert parallelism:
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, without which
NCCL's ragged all-to-all hangs. The flags that usually speed up collectives and
scheduling changed nothing past the noise (1%) on the 1.3B on 1 and 8 GPUs, nor on
the MoE on `mesh(fsdp=8)`: the latency-hiding scheduler, pipelined all-gathers,
all-reduces and reduce-scatters, combine thresholds of 128 MiB, double-buffered while
loops, command buffers off, Triton GEMMs off. FSDP over 8 GPUs gets 97% of 8 times
one GPU's tokens already.

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
`XLA_FLAGS`, a flag that it sets already keeping its value.

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

A layer adds a term of its own from its forward, in one line:
`report(taps, "my_loss", weight, value)` (`bux/architecture/registry.py`). The
objective adds weight × value to the training loss and logs the values, summed over
the layers that report the name, under that name; eval leaves them out.

`train_step` and `eval_step` are what the trainer compiles, once; a subclass that
overrides them changes what a step does (averaging the weights, skipping a spike) and
keeps the loop, the hooks and checkpoints. Gradient accumulation is built in:
`methods(microbatches=n)`.

With `Trainer(..., donate=True)`, as `methods(donate=true)` makes it, a training
step donates the model's and the optimizer's arrays: an array read from
`trainer.model` before a step is gone after it, so a hook that keeps one keeps a
copy (`np.asarray`). `fit` reads `data` through `prefetch` (`bux/train/data.py`),
a thread that puts the next two batches on the devices while a step runs; a run
that resumes still reads from its step on, and an error reading a batch is raised at
that batch.

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
