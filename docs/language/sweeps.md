(lang-sweeps)=
# Sweeps

This page covers alternatives, `a | b`, which turn a file into a sweep of several
runs: where they can go, how they form a grid, how runs are named, how labels zip
axes together, which alternatives count, and how `bux check` and the editor treat
them. For running a sweep, see [Running](../training/running.md).

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

`examples/experiments/attention_ablation.bux` has two axes, one labeled, so it is
3 × 2 = 6 runs:

```text
attention_ablation[mha, optimizer.lr=1e-3]
attention_ablation[mha, optimizer.lr=3e-3]
attention_ablation[gqa, optimizer.lr=1e-3]
attention_ablation[gqa, optimizer.lr=3e-3]
attention_ablation[mqa, optimizer.lr=1e-3]
attention_ablation[mqa, optimizer.lr=3e-3]
```

`bux check` sums the sweep up in its last line, and `bux plan` lists every run with
its id, size and budget:

```console
$ bux check examples/experiments/attention_ablation.bux | tail -1
sweep: 6 runs over attn.num_kv_heads (mha | gqa | mqa) × optimizer.lr (1e-3 | 3e-3)
```

## Alternatives anywhere

`|` separates alternatives. It is the loosest operator, below `+`, and works anywhere
a value or a block does:

```bux
optimizer.lr = 1e-3 | 3e-3 | 1e-2                                # a setting
attn  = gqa(num_heads=8, num_kv_heads=8 | 2 | 1)                 # inside a call
block = pre: [norm attn] [norm ffn] | par: [norm (attn + ffn)]   # the wiring itself
model = embed "DDDA"^12 norm head | embed "DDA"^16 norm head     # the whole model
```

Because `|` binds loosest, each alternative is a whole expression:
`embed "DDDA"^12 norm head | embed "DDA"^16 norm head` is two models, not one model
with a choice in the middle. Group to choose a part: `embed ("DDDA"^12 | "DDA"^16)
norm head`.

## Axes form a grid

Each `|` as written is an axis, and a file with axes is one run for each combination.
The axes form a grid, the first written varying slowest. The file's own axes come
first, then those of its modules, then those on the command line:

```console
$ bux check examples/experiments/smoke.bux 'optimizer.lr = 1e-3 | 3e-3' 'attn = gqa(num_heads=4, num_kv_heads=4 | 2 | 1)' | tail -1
sweep: 6 runs over optimizer.lr (1e-3 | 3e-3) × attn.num_kv_heads (4 | 2 | 1)
```

## Run names

A run is named after the file and what it took, in the order of the axes:
`smoke[optimizer.lr=1e-3, attn.num_kv_heads=4]`.

- An **unlabeled** alternative is named by where it is written, the definition and
  setting keys, and the alternative as written: `optimizer.lr=3e-3`. Inside a macro
  it is the macro's name: `layer.init=0.5`. A whole definition is its name:
  `model=embed "DDA"^16 norm head`.
- A **labeled** alternative is named by its label: `mqa`.
- A file with no alternatives is one run named after the file: `smoke`.

Name parts are what `--only` matches: `bux train attention_ablation.bux --only mqa`
trains the two `mqa` runs.

(lang-labels)=
## Labels zip axes

Labels name alternatives, `mqa: 1 | gqa: 4`, and the runs that take them. All
alternatives with the same labels are **one axis**, so they are picked together.
This zips them instead of crossing them. This is three runs, not nine:

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

Labels work across files: an alternative in a module and one in the file with the
same labels are one axis too. The rules:

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
  keep a `block = pre: ... | par: ...` around without doubling its runs while the
  model does not use `block`.
- **An alternative in a definition that a later layer replaced is not an axis.**
  `optimizer = adamw(lr=1e-3 | 3e-3)` in a recipe makes no runs once the file
  defines `optimizer = muon(lr=0.02)`.
- **A macro takes the same alternative at every call.** `layer(mixer) = [norm mixer
  scale(init=0.5 | 1.0)] [norm ffn]` is two runs however often `layer` is called:
  every layer takes 0.5, or every layer 1.0.

:::{warning}
An amendment builds on the definition below it, so it keeps that definition's
alternatives, even of the key it sets. `bux train attention_ablation.bux
'optimizer.lr = 1e-2'` is still six runs: each pair is the same run, at 1e-2, named
`optimizer.lr=1e-3` and `optimizer.lr=3e-3`. To pin a swept setting, redefine the
whole definition: `'optimizer = adamw(lr=1e-2, weight_decay=0.1)'` is three runs.
:::

## Checking a sweep

`bux check` and the editor check every alternative once, with the others at their
first. So an error in any alternative is reported, without compiling every
combination:

```text
error: gqa: num_kv_heads: Input should be a valid integer, got a number with a fractional part
 --> model.bux:4:26
  |
4 | attn  = gqa(num_heads=2, num_kv_heads=2 | 3 | 1.5)
  |                          ^^^^^^^^^^^^^^^^^^^^^^^^
```

`bux check` prints the first run's summary and the sweep's axes. `bux plan` compiles
every run, and `bux fmt` prints every run as a file of its own, with nothing left to
choose ([Tools](tooling.md)).
