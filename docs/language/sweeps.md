(lang-sweeps)=
# Sweeps

One `|` turns a line into several runs. Alternatives, `a | b`, work anywhere a value
or a block does, from a learning rate to the wiring of a block. A file with
alternatives is a sweep: one run for each combination. This page covers how the axes
form a grid, how runs are named, how labels zip axes together, which alternatives
count, and how a sweep is checked. To train a sweep, see
[Running](../training/running.md) (one run after another) and
[Launching](../training/launching.md) (side by side on GPUs).

## Example: three attentions at two learning rates

```bux
use llama3_8b
use fineweb_wsd

d_model = 768
ffn.intermediate_size = 2048
model = embed block^12 norm head(tied=true)

attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3

budget = 3B
```

`examples/experiments/attention_ablation.bux` has two axes, one of them labeled, so
it is 3 × 2 = 6 runs:

```text
attention_ablation[mha, optimizer.lr=1e-3]
attention_ablation[mha, optimizer.lr=3e-3]
attention_ablation[gqa, optimizer.lr=1e-3]
attention_ablation[gqa, optimizer.lr=3e-3]
attention_ablation[mqa, optimizer.lr=1e-3]
attention_ablation[mqa, optimizer.lr=3e-3]
```

`bux check` sums up the sweep in its last line:

```console
$ bux check examples/experiments/attention_ablation.bux | tail -1
sweep: 6 runs over attn.num_kv_heads (mha | gqa | mqa) × optimizer.lr (1e-3 | 3e-3)
```

`bux plan` lists every run with its id, size and budget
([A sweep](../training/index.md#a-sweep) shows the plan of this file).

## Alternatives anywhere

`|` separates alternatives. It is the loosest operator, looser than `+`:

```bux
optimizer.lr = 1e-3 | 3e-3 | 1e-2                                # a setting
attn  = gqa(num_heads=8, num_kv_heads=8 | 2 | 1)                 # inside a call
block = pre: [norm attn] [norm ffn] | par: [norm (attn + ffn)]   # the wiring itself
model = embed "DDDA"^12 norm head | embed "DDA"^16 norm head     # the whole model
```

Because `|` binds loosest, each alternative is a whole expression:
`embed "DDDA"^12 norm head | embed "DDA"^16 norm head` is two models, not one model
with a choice in the middle. To choose a part, group it:
`embed ("DDDA"^12 | "DDA"^16) norm head`.

## Axes form a grid

Each `|` as written is an axis, and the axes form a grid: one run for each
combination, with the first axis varying slowest. The file's own axes come first,
then those of its modules, then those on the command line:

```console
$ bux check examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3' 'attn = gqa(num_heads=4, num_kv_heads=4 | 2 | 1)' | tail -1
sweep: 6 runs over optimizer.lr (1e-3 | 3e-3) × attn.num_kv_heads (4 | 2 | 1)
```

## Run names

A run is named after the file and the alternatives it took, in the order of the axes:
`smoke[optimizer.lr=1e-3, attn.num_kv_heads=4]`.

- An **unlabeled** alternative is named by where it is written (the definition and
  the setting keys) and by its text: `optimizer.lr=3e-3`. Inside a macro, the
  definition is the macro: `layer.init=0.5`. When the alternatives are the whole
  value of a definition, the place is the definition alone:
  `model=embed "DDA"^16 norm head`.
- A **labeled** alternative is named by its label: `mqa`.
- A file with no alternatives is one run, named after the file: `smoke`.

`--only` matches against these names: `bux train attention_ablation.bux --only mqa`
trains the two `mqa` runs.

(lang-labels)=
## Labels zip axes

Labels name alternatives, `mqa: 1 | gqa: 4`, and the runs that take them. All
alternatives with the same labels are **one axis**, so they are picked together:
labels zip axes instead of crossing them. This is three runs, not nine:

```bux
d_model      = s: 512  | m: 768  | l: 1024
optimizer.lr = s: 4e-3 | m: 3e-3 | l: 2e-3
attn.num_heads = d_model // 64            # follows d_model in every run
```

```text
width_sweep[s]
width_sweep[m]
width_sweep[l]
```

Labels work across files: alternatives with the same labels in a module and in the
file are one axis too. The rules:

- Label every alternative of a `|` or none of them.
- Two alternatives of one `|` cannot have the same label.
- A label needs alternatives: `a: 1` alone is an error.
- Alternatives that share some labels but not all are an error:

  ```text
  error: these alternatives share some labels with others, but not all: l, s against m, s
   --> model.bux:7:25
    |
  7 | optimizer = adamw(lr=s: 4e-3 | l: 2e-3)
    |                         ^^^^^^^^^^^^^^
    = hint: alternatives with the same labels are picked together, so give them all the same labels, or none in common
  ```

## Which alternatives count

Only alternatives that a run reaches make runs:

- **An alternative in a definition that nothing uses is not an axis.** A file can
  keep `block = pre: ... | par: ...` around without doubling its runs, as long as the
  model does not use `block`.
- **An alternative in a definition that a later layer replaced is not an axis.**
  `optimizer = adamw(lr=1e-3 | 3e-3)` in a recipe makes no runs once the file
  defines `optimizer = muon(lr=0.02)`.
- **A macro takes the same alternative at every call.**
  `layer(mixer) = [norm mixer scale(init=0.5 | 1.0)] [norm ffn]` is two runs, however
  often `layer` is called: every layer takes 0.5, or every layer takes 1.0.
- **An alternative whose choice changes no run is not an axis.** An amendment over a
  swept setting overrides it:
  `bux train attention_ablation.bux 'optimizer.lr = 1e-2'` is three runs at 1e-2,
  named for the attention alone. An axis is dropped when every run it would make is
  the same as its first.

## Dimensions: a search instead of a grid

A sweep trains every alternative. A search samples instead. A distribution such as
`log_uniform(1e-4, 1e-2)` is a dimension of the search, and the search method picks
the values of each trial from how the earlier trials did
([Searching hyperparameters](../training/searching.md)):

```bux
optimizer.lr           = log_uniform(1e-4, 1e-2)   # a number on a log scale
optimizer.weight_decay = uniform(0, 0.2)           # a number
schedule.warmup        = int_uniform(100, 2k)      # an integer
ffn.activation         = choice(swish, gelu)       # an option: values, settings or blocks
objective              = 0.8 * web_bpb + 0.2 * code_bpb   # what the search minimizes
```

A dimension is named by where it is written, as an unlabeled alternative is, and a
dimension inside a macro takes the same value at every call. A file with both
alternatives and dimensions runs one search for each run of its sweep:
`attn.num_kv_heads = mha: 12 | mqa: 1` with a dimension is two searches. Outside a
search, `bux check`, `bux plan` and `bux fmt` show a dimension at its middle.

## Checking a sweep

`bux check` and the editor check every alternative once, with the other axes at their
first alternative, and every dimension at both ends. So an error in any alternative is
reported without compiling every combination:

```text
error: gqa: num_kv_heads: Input should be a valid integer, got a number with a fractional part
 --> model.bux:4:26
  |
4 | attn  = gqa(num_heads=2, num_kv_heads=2 | 3 | 1.5)
  |                          ^^^^^^^^^^^^^^^^^^^^^^^^
```

`bux check` prints the summary of the first run and the axes of the sweep. `bux plan`
compiles every run, and `bux fmt` prints every run as a separate file, with nothing
left to choose ([Tools](tooling.md)).
