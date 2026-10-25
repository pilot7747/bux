# Concepts

The few ideas the rest of bux builds on. Each has a short example here and a link to
the page that covers it in full. Read this after the [quickstart](quickstart.md), or
whenever a `.bux` file does something you did not expect.

(concepts-residual-algebra)=
## The residual algebra: `[f]` is `x + f(x)`

A model is an expression over the residual stream. Layers are functions of the
stream; writing two in a row composes them, and brackets draw a skip connection
around what they hold:

```bux
block = [norm attn] [norm ffn]          # x + attn(norm(x)), then the same with ffn
model = embed block^12 norm head        # 12 fresh copies of block in a row
```

| syntax           | meaning                                              |
|------------------|------------------------------------------------------|
| `f g`            | composition, left to right: `g(f(x))`                |
| `[f]`            | a residual branch: `x + f(x)`                        |
| `f + g`, `f * g` | parallel branches on the same input, summed or multiplied |
| `f^n`            | `n` fresh copies of `f` in a row                     |
| `@name`, `$name` | save the stream under a name, and read it back later |
| `f \| g`         | alternatives: a run for each                         |

So common shapes read as they are drawn: `[attn] norm [ffn] norm` is post-norm,
`[norm (attn + ffn)]` is a parallel block (GPT-J, PaLM), and
`(linear(features=h) swish * linear(features=h)) linear(features=d_model)` is SwiGLU,
written by hand. `bux render` draws any of them. A `residual` setting can also change
what `[...]` means, to hyper-connections and others, without touching the wiring.

See [Operators](../language/operators.md),
[Skip connections](../language/skip-connections.md) and
[Residual algebras](../language/residual-algebras.md).

(concepts-definitions)=
## Definitions and late binding

A file is a list of definitions, in any order. A definition of a layer is a template:
each use gets fresh parameters, unless the definition is `shared`. Calling a
definition overrides its settings, and a definition with parameters is a macro:

```bux
local  = attn(sliding_window_size=1024)    # attn, with one setting changed
layer(mixer, ffn=experts) = [norm mixer] [norm ffn]   # a macro, with a default
shared sa = [norm attn] [norm ffn]         # one module, reused wherever sa appears
```

Names bind **late**: a definition replaces the one it shadows *everywhere*, including
inside definitions it did not write. After `use llama3_8b`, defining `attn` changes
the attention inside llama3_8b's `block`. Inside its own definition, a name means the
definition it replaces, so `ffn = ffn(intermediate_size=2048)` changes one setting of
the inherited `ffn`. This is what makes an ablation the few lines that differ.

See [Definitions](../language/definitions.md) and [Modules](../language/modules.md).

(concepts-run-is-a-file)=
## A run is a file

The same file that describes a model says how to train it. Training settings are
top-level definitions next to `d_model`, and their configs are called by name, like
layers:

```bux
data      = synthetic
optimizer = adamw(lr=3e-3)
schedule  = wsd(warmup=10)
seq_len   = 64
batch     = 8
steps     = 50
```

A file is a **run** once it sets any training setting, and then it needs all the
required ones (`data`, `optimizer`, `seq_len`, `batch`, `steps`). A file with no
model, such as a recipe of training settings, is a **module** that runs use. Every
setting is checked against its type before anything runs, at the line that wrote it,
and settings are checked against each other: a schedule's warmup and cooldown must fit
in the steps.

See [Training settings](../training/settings.md) and the
[language reference](../reference/language/index.md).

(concepts-one-mechanism)=
## Modules, amendments, the command line and sweeps

There are four ways to vary an experiment, and they are one mechanism: each gives a
name another definition, and everything written in terms of that name follows.

```bux
use llama3_8b                              # a module, found by name, never by path
use fineweb_wsd

attn.num_kv_heads = mha: 32 | gqa: 8       # an amendment, with two alternatives
optimizer.lr      = 1e-3 | 3e-3            # 2 × 2 = 4 runs
```

```sh
bux plan ablation.bux 'optimizer.lr = 3e-3' steps=2k   # the command line, one more layer
```

- `use name` brings in `name.bux`, found in the file's folder, then under the folders
  `[tool.bux] path` lists in `pyproject.toml` (`examples/` in bux's own), then
  `BUX_PATH`.
- `name.key = value` is an **amendment**: `optimizer.lr = 3e-3` means
  `optimizer = optimizer(lr=3e-3)`.
- Statements after the file on the **command line** redefine the file's.
- `a | b` makes **alternatives**: each `|` is an axis, and the axes cross.
  Alternatives with the same **labels** (`s: 512 | m: 768`) are one axis, so they are
  picked together instead of crossed.

See [Modules](../language/modules.md) and [Sweeps](../language/sweeps.md).

(concepts-run-ids)=
## Run ids and run directories

Every run has a **name**, the file and the alternatives it took
(`ablation[gqa, optimizer.lr=3e-3]`), and an **id**, a hash of what it computes. A run
trains into `runs/<name>-<id>/`:

```text
run.bux        the run as one file: no modules, no alternatives
run.json       its IR
metrics.jsonl  a line of metrics at every log and every eval
checkpoints/   with a checkpoint setting
status.json    with bux launch: running, done, failed or stopped, and where
```

The id ignores how a run is written, watched or split: renaming a definition,
reaching the same settings another way, `log`, `eval`, `checkpoint`, `profile`,
`mesh`, `layout`, `methods` and `compiler` all keep it. Anything else changes it. So a
changed run starts fresh in a directory of its own, and an unchanged one started
again resumes from its latest checkpoint, even on another mesh.

See [Running](../training/running.md) and [Launching](../training/launching.md).

(concepts-ir)=
## The IR

The compiler turns a file into plain pydantic data, with no JAX in sight:
`bux.lang.compile` gives an `Architecture` (`d_model`, `vocab_size`, `residual` and a
graph of 11 node kinds), and `bux.lang.compile_runs` gives a `Run` per combination of
alternatives, with its `TrainConfig`. A backend turns the graph into code; bux's own,
in `bux.architecture`, is a Flax NNX module per node kind.

```python
from bux.lang import compile

arch = compile("""
d_model    = 64
vocab_size = 512
block = [rmsnorm gqa(num_heads=4, num_kv_heads=2)] [rmsnorm mlp(intermediate_size=256, activation=gelu)]
model = embed block^2 rmsnorm head
""")
print(arch.d_model, arch.vocab_size, arch.graph.kind)
print([item.kind for item in arch.graph.items])
```

```text
64 512 seq
['layer', 'repeat', 'layer', 'layer']
```

The IR serializes to JSON, goes into checkpoints as it is, and prints back to source
(`bux.lang.printer.to_source`), so programs can generate and mutate architectures.
`bux fmt` prints it, and `bux json` dumps it.

See [The IR and backends](../architectures/ir-and-backends.md) and the
[Python API](../reference/python/index.md).

(concepts-parallelism)=
## Mesh, layout and methods

A run splits over devices by three settings, none of which changes what it computes,
so none is in its id:

```bux
mesh    = mesh(fsdp=8, tensor=2)              # a size per axis; data takes the rest
layout  = layout(stream=[context, tensor])    # which axes each named dimension uses
methods = methods(attention=ring, moe=all_to_all)  # how what communicates is done
```

- `mesh` is one grid of six axes, `stage, data, fsdp, context, expert, tensor`,
  slowest links first. An axis not set has size 1.
- `layout` says which mesh axes each named dimension (`batch`, `heads`, `mlp`,
  `experts`, ...) splits over. Layers name their dimensions; they never say how
  anything splits.
- `methods` picks, by name, the function that computes each part that communicates:
  attention across `context`, an MoE's dispatch, a pipeline's schedule.

See [Parallelism](../parallelism/index.md) and
[Scale out](../tutorials/scale-out.md).

(concepts-registries)=
## Registries are the extension point

Configs are plain data, and the code that does the work is found by a config's `type`
in a registry. Registering under a taken name replaces bux's own; a new name also
needs its config in `bux/config.py` (layers) or `bux/train/config.py` (training), where
the compiler and the editor find it.

| to change            | register with                                    |
|----------------------|--------------------------------------------------|
| a layer              | `@bux.architecture.register("gqa")`              |
| an optimizer         | `@bux.train.optim.optimizers.register("adamw")`  |
| a schedule           | `@bux.train.optim.schedules.register("wsd")`     |
| a parallel method    | `@bux.parallel.methods.attention.register("mine")` |

What a training loop does between steps and where its metrics go are blocks too:
hooks and sinks, passed to `train(run, hooks=[...], sinks=[...])`.

```python
import optax

from bux.train.optim import lr_schedule, optimizers, weight_decay_mask


@optimizers.register("adamw")  # every run's adamw is now NAdamW, in this process
def nadamw(c, multiplier, model):
    return optax.adamw(
        lr_schedule(c.lr, multiplier),
        b1=c.b1,
        b2=c.b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        mask=weight_decay_mask(model),
        nesterov=True,
    )
```

See [Add a layer](../tutorials/add-a-layer.md),
[Custom training](../tutorials/custom-training.md) and
[Extending the trainer](../training/extending.md).
