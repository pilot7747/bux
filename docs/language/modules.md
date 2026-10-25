(lang-modules)=
# Modules and amendments

This page covers how a `.bux` file builds on others: `use` and how modules are found,
how definitions from several files layer over each other, amendments that change one
setting (`attn.num_kv_heads = 1`), statements on the command line, and when a file
is a run or a module. Read it to write an experiment as the lines that differ from a
baseline.

## Example: an ablation on top of two files

```bux
use llama3_8b                      # examples/architectures/llama3_8b.bux
use fineweb_wsd                    # examples/experiments/recipes/fineweb_wsd.bux

# Llama 3 8B, scaled down.
d_model = 768
ffn.intermediate_size = 2048
model = embed block^12 norm head(tied=true)

# Three attentions, crossed with two learning rates: 3 × 2 runs.
attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3

budget = 3B
```

This is `examples/experiments/attention_ablation.bux`. The architecture comes from
`llama3_8b.bux` and the data, optimizer and schedule from the recipe
`fineweb_wsd.bux`. The file then changes what it needs:

- `d_model = 768` replaces Llama's width, and everything written in terms of
  `d_model` follows.
- `attn.num_heads = 12` changes one setting of Llama's `attn`, and Llama's `block`,
  which uses `attn`, picks it up.
- `budget = 3B` replaces the recipe's budget, and the recipe's
  `steps = budget // (batch * seq_len)` follows.

```console
$ bux check examples/experiments/attention_ablation.bux
d_model=768 vocab_size=128256 layers=51
rmsnorm×25  gqa×12  gated_mlp×12  embed×1  head×1
5,722 steps × 256 × 2048 tokens = 3B tokens
optimizer=adamw(lr=0.001) schedule=wsd(warmup=1000, shape=sqrt)
data=0.8 * text(path="data/fineweb-edu/*.parquet", bos=true, split=[0.0, 0.999]) + 0.2 * text(path="data/stack-edu/*.parquet", column=content, bos=true)
tokenizer=tokenizer(name="unsloth/Meta-Llama-3.1-8B", bos="<|begin_of_text|>", eos="<|end_of_text|>") packing=best_fit
sweep: 6 runs over attn.num_kv_heads (mha | gqa | mqa) × optimizer.lr (1e-3 | 3e-3)
```

(lang-use)=
## `use`

`use llama3_8b` brings in the definitions of `llama3_8b.bux`. The rules:

- **By name, never by path.** A module is found by its file name without `.bux`. A
  path is an error:

  ```text
  error: modules are found by name, not by path: 'architectures/llama3_8b.bux'
   --> model.bux:1:5
    |
  1 | use "architectures/llama3_8b.bux"
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    = hint: use llama3_8b, with its folder on the search path ([tool.bux] path)
  ```

- **Names that are not identifiers** are written as strings: `use "qwen3.8_27b"`.
- **Anywhere in the file.** A module's definitions always sit below the file's own,
  wherever the `use` is written; by convention it goes at the top.
- **Each module once.** A module used twice, directly or through other modules, is
  loaded once. Modules that use each other are an error:
  `modules use each other: model -> cyc_a -> cyc_b -> cyc_a`.
- **A missing module** is an error at its `use`, with a suggestion:

  ```text
  error: no module named 'llama3_8bb'
   --> model.bux:2:5
    |
  2 | use llama3_8bb
    |     ^^^^^^^^^^
    = hint: did you mean 'llama3_8b'?
  ```

  The names it might have defined are not reported as errors too.

(lang-module-path)=
## The module path

A module is looked for, in order:

1. in the folder of the file that has the `use`, without its subfolders;
2. under every folder listed in `path` under `[tool.bux]` of the nearest
   `pyproject.toml`, looking up from that folder, subfolders included. The folders
   are relative to the `pyproject.toml`;
3. under every folder in the environment variable `BUX_PATH` (separated by `:`, or
   `;` on Windows), subfolders included.

```toml
[tool.bux]
path = ["examples"]
```

bux's own `pyproject.toml` has exactly this, so in a checkout every example file is
a module by its name.

- **Folders organize files without becoming part of their names**, so moving a file
  into a subfolder breaks nothing.
- **A name used twice under one folder of the path is an error**, since it would be
  ambiguous: `there are 2 modules named 'twice' under ...: a/twice.bux, b/twice.bux`.
  The first folder in the order above that has the name wins.
- Hidden folders (starting with `.`), `__pycache__` and `node_modules` are skipped.
- A file with no folder of its own, such as an editor's unsaved buffer, looks in the
  working directory.

(lang-layering)=
## Layering and late binding

The definitions of a program form layers, the lowest first:

1. the modules the file uses, in the order it uses them, each with the modules it
   uses below it, and each module once;
2. the file itself;
3. the command line: first the modules used there, then its statements.

Within each file, its amendments form a layer of their own just above its other
definitions. For the ablation above, from the top:

| Layer | Defines |
|-------|---------|
| the command line, if any | `optimizer.lr = 1e-2`, ... |
| `attention_ablation.bux`, its amendments | `ffn.intermediate_size`, `attn.num_heads`, `attn.num_kv_heads`, `optimizer.lr` |
| `attention_ablation.bux` | `d_model`, `model`, `budget` |
| `fineweb_wsd.bux` | `tokenizer`, `data`, `optimizer`, `schedule`, `budget`, `steps`, ... |
| `llama3_8b.bux` | `d_model`, `vocab_size`, `norm`, `attn`, `ffn`, `block`, `model` |

A definition replaces the one below it **everywhere**, including inside the
definitions it did not write. After `use llama3_8b`, defining `attn` changes the
attention in llama3_8b's `block`. This is late binding: names are looked up in the
program as a whole, after every layer is in place.

Inside its own definition, a name means the definition it replaces. So
`ffn = ffn(intermediate_size=2048)` changes one setting of the inherited `ffn`. With
nothing below it, the name means the builtin, as `mlp = mlp(...)` does.

Errors point into the file they are in, and the editor shows errors in modules at
the `use` that brought them in.

(lang-amendments)=
## Amendments

`name.key = value` changes one setting of a definition. `optimizer.lr = 3e-3` means
`optimizer = optimizer(lr=3e-3)`: the optimizer as it was, with one setting changed.

```bux
use llama3_8b

attn.num_kv_heads = 1          # multi-query attention
attn.rope.base    = 1e6        # dotted keys reach into nested settings
ffn = ffn(intermediate_size=2048)   # as ffn.intermediate_size = 2048 would
```

The rules:

- **Dotted keys** reach into nested settings: `attn.rope.base = 1e6`.
- **Several amendments of one name in a file act as one.** Setting the same key
  twice is an error: `'num_heads' is given twice`.
- **An amendment applies over the file's own definition** of the name, if it has
  one, since a file's amendments are a layer above its definitions.
- **A shared layer stays shared.** The amendment changes the one module. Writing
  `shared` in front of an amendment is an error.
- **Only configs have settings to amend.** Layers and settings do; to change a
  block, redefine it:

  ```text
  error: 'block' is a block, which has no settings to amend
   --> model.bux:4:1
    |
  4 | block.norm = rmsnorm
    | ^^^^^
    = hint: redefine it instead: block = ...
  ```

- **Something must define the name.** `nothing defines 'norm2' for this to amend`,
  unless the name is a builtin: `rmsnorm.eps = 1e-5` defines `rmsnorm` as
  `rmsnorm(eps=1e-5)`, so every `rmsnorm` written anywhere in the program takes it.
  A builtin with required settings cannot be amended this way.

(lang-command-line)=
## The command line

Every command takes statements after the file. They redefine the file's
definitions, as one more layer over it would:

```sh
bux train examples/experiments/smoke.bux optimizer.lr=1e-2 'schedule = cosine(warmup=5)'
bux train examples/experiments/smoke.bux 'use fineweb_wsd'             # modules too
bux plan  examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3'  # and sweeps
```

- Each statement is one argument: `name=value` needs no quotes, and anything with
  spaces does. A statement must contain `=` or start with `use `.
- A module used on the command line goes above the file, so it replaces the file's
  own definitions: `'use fineweb_wsd'` gives the smoke run the recipe's data,
  optimizer, schedule and steps.
- Errors in them point at `<command line>`:

  ```text
  error: unknown name 'stepz'
   --> <command line>:1:13
    |
  1 | steps = 5 * stepz
    |             ^^^^^
    = hint: did you mean 'steps'?
  ```

### Four ways, one mechanism

So there are four ways to vary an experiment: `use`, an amendment, the command line
and an alternative ([Sweeps](sweeps.md)). They are one mechanism: each gives a name
another definition, and everything written in terms of that name follows. In
`examples/experiments/width_sweep.bux`, the heads follow the width in every run:

```bux
use llama3_8b
use fineweb_wsd

model = embed block^12 norm head(tied=true)

d_model      = s: 512  | m: 768  | l: 1024
optimizer.lr = s: 4e-3 | m: 3e-3 | l: 2e-3

attn.num_heads        = d_model // 64
attn.num_kv_heads     = d_model // 256
ffn.intermediate_size = 8 * d_model // 3

budget = 2B
```

(lang-runs-and-modules)=
## Runs and modules

What a file is depends on what it defines:

| The file defines | It is | What works |
|------------------|-------|------------|
| a model and `d_model`, and no training setting | an architecture | `check`, `render`, `fmt`, `json` |
| a model and any training setting | a run, or a sweep of runs | every command, `plan`, `train` and `launch` too |
| no `model` and no `d_model`, such as a recipe | a module | `check`, and `use` from other files |

- **A file is a run once it sets any training setting**, and then it needs all the
  required ones: `data`, `optimizer`, `seq_len`, `batch` and `steps`. A missing one
  is an error that names it and suggests a value.
- **A module is checked on its own.** Its settings are checked against their types,
  but a run needs a model, so `bux check` only lists what it defines.
- **An architecture does not train**: `bux plan`, `bux train` and `bux launch` say
  `this file does not say how to train its model`. Add a recipe from the command
  line to train it: `bux train model.bux 'use fineweb_wsd'`.

[What is required](../training/settings.md#what-is-required) shows the error and what
`bux check` says of a module, and [Settings](../training/settings.md) lists every
training setting and its default.
