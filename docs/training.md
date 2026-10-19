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
use llama3_8b                      # examples/llama3_8b.bux
use fineweb_wsd                    # experiments/recipes/fineweb_wsd.bux

d_model = 768
ffn.intermediate_size = 2048
model = embed block^12 norm head(tied=true)

attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3
budget            = 3B
```

```
$ bux plan experiments/attention_ablation.bux
6 runs
id          params  tokens  tok/param  run
4c6613d749  183M    3B      16.4       attention_ablation[mha, optimizer.lr=1e-3]
fe14a5e39c  183M    3B      16.4       attention_ablation[mha, optimizer.lr=3e-3]
69cdd0b96d  174M    3B      17.2       attention_ablation[gqa, optimizer.lr=1e-3]
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

The settings of each config are pydantic fields in `bux/train/config.py`, and the
editor shows them on hover and completes them, as it does for layers. In short:

- `adamw(lr, b1=0.9, b2=0.95, eps, weight_decay=0.1)` decays matrices only, not norm
  scales or biases.
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

Settings are definitions, so arithmetic, `name.field` reads and the number suffixes
`k`, `M`, `B` and `T` work on them: `steps = 10B // (batch * seq_len)`, `warmup=2k`.
A suffix is exact, so `1.1B` is the integer 1_100_000_000.

Before anything runs, the compiler checks each setting against its type, where it is
written, and that they fit together: the warmup and cooldown fit in the steps, and
sequences fit in the positions a learned positional embedding has.

```
$ bux check experiments/attention_ablation.bux steps=50
error: the schedule takes 1010 steps (1000 of warmup and 10 of cooldown), but the run has 50
  --> experiments/recipes/fineweb_wsd.bux:19:13
   |
19 | schedule  = wsd(warmup=1k, decay=0.2, shape=sqrt)
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

A file is a run once it sets any training setting; then it needs all the required
ones. A file with no model and no `d_model`, such as a recipe, is a module: it is
checked on its own, but a run needs a model.

## Data

A batch is rows of `seq_len` tokens, each three arrays: `input_ids`, `position_ids`
(where each token is in its document, so a 0 starts one) and `loss_mask` (whether a
token is a target). The model attends, rotates and convolves within documents, and
the loss is on targets of the document that predicts them.

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
path = ["examples", "experiments"]
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
bux train experiments/smoke.bux optimizer.lr=1e-2 'schedule = cosine(warmup=5)'
bux train experiments/smoke.bux 'use fineweb_wsd'           # modules too
bux plan  experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3'  # and sweeps
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
bux check experiments/smoke.bux     # the model, the run, the sweep
bux plan  experiments/smoke.bux     # every run: id, parameters, tokens, tokens per parameter
bux prepare experiments/sft.bux     # tokenize and pack its documents; --cache DIR, --workers N
bux train experiments/smoke.bux     # train every run, one after another; --only TEXT, --out DIR
bux fmt   experiments/smoke.bux     # every run as a file of its own
bux json  experiments/smoke.bux     # every run's IR
```

`bux.lang.compile_runs(source)` returns a `Run` per combination: an `Architecture`, a
`TrainConfig` (`bux/train/config.py`), a name, and the alternatives it took, all plain
pydantic. `bux.train.trainer.train(run)` trains one with the reference Flax NNX
backend, on the default device, into `runs/<name>-<id>/`:

```
run.bux        the run as one file: no modules, no alternatives; bux train takes it as is
run.json       the IR
metrics.jsonl  loss, lr, gradient norm and throughput every log.every steps, and eval losses
checkpoints/   model and optimizer state, with a checkpoint setting (orbax)
```

The id is a hash of what the run computes: renaming a definition, reaching the same
settings through another file or a sweep, or logging, evaluating and checkpointing at
other intervals keeps it, and anything else changes it. So a changed run starts in a
directory of its own, and an unchanged one started again resumes from its latest
checkpoint, ending exactly where it would have.

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
schedules composed in sequence with the same algebra; and trackers next to
metrics.jsonl.
