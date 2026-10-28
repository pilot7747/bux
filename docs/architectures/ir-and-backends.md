# The IR and backends

A `.bux` file compiles to a typed intermediate representation (IR): plain pydantic
data, with no JAX in it. A backend turns the IR into code that runs. Read this page to
generate or mutate architectures from Python, to call a model outside the trainer, or
to port bux's models to another framework. It covers the IR (`bux.lang.ir`), printing
it back to source, how bux's Flax NNX backend (`bux.architecture`) builds a model
from it, and what another backend has to do.

```python
from pathlib import Path

from bux.lang import compile

arch = compile(Path("examples/architectures/llama3_8b.bux").read_text())
print(arch.d_model, arch.vocab_size, arch.residual)
print(type(arch.graph).__name__, [type(n).__name__ for n in arch.graph.items])
print(sum(1 for _ in arch.layers()))
```

```text
4096 128256 None
Seq ['Layer', 'Repeat', 'Layer', 'Layer']
131
```

## The IR

{py:func}`bux.lang.compile <bux.lang.compiler.compile>` returns an {py:class}`bux.lang.ir.Architecture`:

| Field | What it is |
|-------|-----------|
| `d_model` | the width of the residual stream |
| `vocab_size` | the vocabulary, or `None` |
| `residual` | the residual algebra (`hc`, `mhc`, `gr`, `attnres`, `vpgate`), or `None` for plain additive residuals |
| `init` | how the weight matrices start, by role ([Initialization](initialization.md)), or `None` for the default |
| `graph` | the model, a tree of nodes |

`arch.layers()` yields every layer in execution order, with repeats unrolled;
`ir.walk(node)` yields the nodes as written, a repeat's body once.

### Nodes

A graph is a tree of 12 kinds of node, each a function of the incoming stream $x$:

| Node | Computes | Written |
|------|----------|---------|
| `Layer` | `config(x)`, a leaf from `bux.config` | `gqa(...)`, `rmsnorm` |
| `Seq` | `items[-1](...items[0](x))`: composition, left to right | `f g` |
| `Residual` | `x + body(x)`: a residual branch | `[f]` |
| `Sum` | `sum(t(x) for t in terms)` | `f + g` |
| `Prod` | `prod(f(x) for f in factors)`: gating, or scaling by a `Const` | `f * g`, `f * 0.5` |
| `Neg` | `-body(x)` | `-f` |
| `Const` | a scalar, only inside `Sum` and `Prod` | `0.5` |
| `Identity` | `x` | `_` |
| `Repeat` | `body` applied `times` times, fresh parameters each time | `f^n` |
| `Tap` | `x`, and pushes `x` onto the tap | `@name` |
| `Ref` | the top of the tap, ignoring `x`; with `pop`, also pops it | `$name`, `$name!` |
| `Image` | `x`, each token `token` replaced by the next output of `body`, the encoder, on the row's images | `image(vision, token=248056)` |

Besides its own fields, every node has:

- `label`: the definition it came from, for printing and diagrams.
- `share`: a key; nodes with the same key are a single module (`shared`).

A `Layer` also has:

- `width`: the width of what it reads, an int, `"tokens"` for an embedding,
  `"pixels"` for an encoder's `patches`, or `"streams"` for a stream widened by the
  residual algebra.
- `ports`: the taps it writes or reads internal tensors through, each a
  {py:class}`bux.lang.ir.Port` of a `name` (the layer's port, as `v`), a `tap`, and a
  `mode` (`write`, `read` or `pop`).

The branches of `Sum` and `Prod` run left to right, which only matters for taps.
`Residual` is kept distinct from `_ + body` so that a backend can swap the residual
algebra in for `[f]` alone, while `_ + f` stays plain arithmetic. `Repeat` is kept,
not unrolled, so that a backend can scan identical blocks.

### Runs

A file that says how to train its model compiles to runs
({py:func}`bux.lang.compile_runs <bux.lang.compiler.compile_runs>`), one per
combination of alternatives. A {py:class}`bux.lang.ir.Run` holds:

- `name`: the file and the alternatives it took;
- `arch`: its `Architecture`;
- `train`: its training settings ({py:class}`bux.train.config.TrainConfig`);
- `choices`: what it took along each axis of the sweep;
- `id`: a hash of what the run computes, not of how it is written, watched or split
  over devices ([Running](../training/running.md)).

### JSON

The IR is plain pydantic, so it serializes to JSON and back unchanged. `bux json`
prints it. Every run directory keeps it beside its checkpoints as `run.json`, next to
`run.bux`, the same run as source:

```console
$ bux json examples/architectures/gpt2.bux
{
  "d_model": 768,
  "vocab_size": 50257,
  "residual": null,
  "init": null,
  "graph": {
    "label": "model",
    "share": null,
    "kind": "seq",
    "items": [
      {
        "label": null,
        "share": null,
        "kind": "layer",
        "config": {
          "type": "embed",
          "max_positions": 1024,
          "scale_by_sqrt_dim": false,
          "norm": null,
          "dropout": 0.0
        },
        "width": "tokens",
        "ports": []
      },
      ...
```

`ir.Architecture.model_validate_json(text)` reads it back, and `ir.Run.model_validate_json`
a run's.

## Printing IR back to source

`bux.lang.printer.to_source(arch)` prints any IR as readable source, including IR
that was generated or changed in Python, such as by an architecture search. Every labeled
node becomes a definition: leaf definitions, then blocks in dependency order, then the
model. Macro calls are inlined, since the IR keeps only what they produced.
`compile(to_source(arch))` equals `arch` up to labels, and `run_to_source(run)`
prints a whole run, which compiles back to a run with its id.

This gives every `gqa` of Llama 3 8B a single kv head, in Python:

```python
from bux.lang import compile, ir
from bux.lang.printer import to_source


def mqa(node: ir.Node) -> ir.Node:
    """Return `node` with every gqa layer in it given one kv head."""
    if isinstance(node, ir.Layer) and node.config.type == "gqa":
        config = node.config.model_copy(update={"num_kv_heads": 1})
        return node.model_copy(update={"config": config})
    return ir.replace_children(node, tuple(mqa(c) for c in ir.children(node)))


arch = arch.model_copy(update={"graph": mqa(arch.graph)})
print(to_source(arch))
```

```bux
d_model = 4096
vocab_size = 128256

norm = rmsnorm(eps=1e-05)
attn = gqa(num_heads=32, num_kv_heads=1, rope=rope(base=500000.0))
ffn = gated_mlp(intermediate_size=14336, activation=swish)

block = [norm attn] [norm ffn]

model = embed block^32 norm head
```

`bux fmt` prints the same: the IR, not your file, so it inlines macros and numeric
definitions, drops comments, and shows the `merge` layers the compiler inserted.
`ir.strip(node)` drops labels and inferred widths, to compare two graphs by structure.

:::{tip}
In `.bux`, the same change is one line, `attn.num_kv_heads = 1`. Reach for the IR
when the change depends on the structure, as in a search over architectures.
:::

## The Flax NNX backend

{py:class}`bux.architecture.Model <bux.architecture.model.Model>` builds an `Architecture` into one `nnx.Module`, from
tokens to logits:

```python
import jax
from flax import nnx

from bux.architecture import Model, num_params
from bux.lang import compile

arch = compile("""
d_model    = 64
vocab_size = 512

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=256, activation=swish)
block = [norm attn] [norm ffn]

shared out = head
mtp   = $h nextn block norm out
model = embed block^2 norm @h out mtp
""")
model = Model(arch, rngs=nnx.Rngs(0))
tokens = jax.random.randint(jax.random.key(0), (2, 16), 0, arch.vocab_size)

print(num_params(model))
print(model(tokens).shape)  # the first head's logits
print([y.shape for y in model.outputs(tokens)])  # every head's
```

```text
258688
(2, 16, 512)
[(2, 16, 512), (2, 16, 512)]
```

| Method | Returns |
|--------|---------|
| `model(tokens)` | the first head's logits: the next-token prediction |
| `model.outputs(tokens, positions=None)` | every head's logits, in execution order; head $j$ predicts the token $1 + k$ ahead, $k$ the `nextn` layers run before it |
| `model.heads(tokens, positions=None)` | every head's input and projection (`HeadOutput`), from which the loss makes logits a chunk at a time |
| `model.training_heads(tokens, positions=None)` | the heads, and the losses layers report: a training step's forward, in which layers also update their own state (an MoE's router bias) |

`positions`, each token's offset in its document, packs several documents into a
row: a document starts wherever the positions stop increasing, and each is attended
to, convolved and embedded on its own.

### How it builds the graph

`bux.architecture.graph.build_node` maps every node kind to one small module:

| Node | Module |
|------|--------|
| `Layer` | the builder registered for its config's `type` ([Adding a layer](adding-a-layer.md)) |
| `Seq` | `Sequence`, its items one after another |
| `Residual` | `ResidualBranch`, `x + body(x)`; outside any other branch, the residual algebra's branch |
| `Sum`, `Prod` | `Parallel`, its branches summed or multiplied |
| `Neg`, `Const`, `Identity` | `Negate`, `Constant` (a scalar in the stream's dtype), `Identity` |
| `Repeat` | the run's `methods.repeat` method |
| `Tap`, `Ref` | `TapWrite` pushes the stream; `TapRead` returns the top, or pops it |
| `Image` | `ImageSplice`: the encoder on the forward pass's images, its outputs over the image tokens ([Vision encoders](vision.md)) |

A node with a share key is built once, and every use returns the same module. A
builder gets a {py:class}`Context <bux.architecture.registry.Context>`, which counts the residual
branches around the layer (`depth`) and holds what earlier layers left (the first
embedding, which tied heads and `nextn` read).

### Residual algebras

A residual branch inside another is always the plain `x + f(x)`. One outside any
other is the architecture's residual algebra's, if it sets one
(`bux.architecture.residual`): `hc`, `mhc` and `gr` widen the stream into n copies,
an array `(b, t, n, d)`, and `attnres` keeps the embedding and every branch output so
far. `embed` widens its output, and `merge` reads the stream back into one
([Residual algebras](../language/residual-algebras.md)).

### Repeats: unrolled, scanned, pipelined

`methods.repeat` builds each `Repeat`:

- **`unroll`** (the default) builds one module per pass and runs them in a row.
- **`scan`** stacks the passes into one `Scanned` module, each variable gaining a
  leading `layers` dimension, and runs them by `jax.lax.scan`: a scan for each run of
  passes that trace alike. Passes stack only if they have one graph and the same
  variable shapes, share no module or variable (a `shared` definition, `nextn`'s
  embedding), and leave the stream as wide as they found it (which `attnres`'s
  outputs do not); otherwise they run in a row.
- With `mesh.stage` above 1, the longest scanned repeat outside any other is
  **pipelined** over the stages: it splits its stacked layers over `stage`, and the
  run's `methods.pipeline` runs them. A model with no scanned repeat raises: repeat
  its blocks (`block^8`) and scan them, `methods(repeat=scan)`
  ([Pipelines](../parallelism/pipelines.md)).

Each residual branch outside any other, and each scanned pass, is rematerialized as
a unit, as `methods.remat` says ([Remat](../parallelism/remat.md)). Those branches are
also numbered as they trace, so that `precision(first=..., last=...)` can keep the
model's first and last in the compute dtype ([Precision](../training/precision.md)).

### Taps

Modules are called as `module(x, taps)`. {py:class}`Taps <bux.architecture.registry.Taps>` is a dict
of one forward pass's tap stacks by name, plus the rest of what the pass carries:

| Attribute | What it is |
|-----------|-----------|
| `tokens` | the input tokens |
| `segments` | the documents packed into the rows; `None` when every row is one |
| `training` | whether this is a training step's forward |
| `heads` | the outputs of the heads run so far |
| `losses` | the losses layers report, collected in a training forward only |
| `nextn_runs` | how many `nextn` layers have run |
| `pending_read` | the read coefficients the last single-pass `hc` or `mhc` branch computed for the next |

Layers read them through functions, not the attributes:
[The call convention](adding-a-layer.md#the-call-convention).

## Writing another backend

A backend is one function per node kind, plus a registry of leaves keyed by
`config.type`. `bux/architecture/graph.py` is the reference. Its contract:

- **Leaves**: one implementation per layer type, built from the config, the input
  `width` and the `ports`. A type with no implementation should fail with a message,
  as the reference backend's does.
- **Composition**: `Seq` left to right; `Sum` and `Prod` evaluate their terms left to
  right on the same input; `Const` is a scalar.
- **Repeats** get fresh parameters for every pass; a node with a share key is one
  module wherever it appears, including inside a repeat.
- **Taps** are stacks scoped to one forward pass: `Tap` pushes and passes the stream
  on, `Ref` reads the top (or pops it) and ignores its input. Ports push and read
  internal tensors on the same stacks.
- **Residuals**: `Residual` is `x + body(x)`, or the residual algebra's for a branch
  outside any other, with `"streams"` widths between them.
- **Heads**: every `head` is an output; return them in execution order.

A skeleton, in the shape of the reference:

```python
from bux.lang import ir

LEAVES = {}  # config type -> builder(config, width, ports)


def build(node: ir.Node, shared: dict) -> object:
    if node.share is not None:
        if node.share not in shared:
            shared[node.share] = _build(node, shared)
        return shared[node.share]
    return _build(node, shared)


def _build(node: ir.Node, shared: dict) -> object:
    match node:
        case ir.Layer(config=config, width=width, ports=ports):
            return LEAVES[config.type](config, width, ports)
        case ir.Seq(items=items):
            return Sequence([build(i, shared) for i in items])
        case ir.Residual(body=body):
            return Residual(build(body, shared))
        case ir.Sum(terms=terms):
            return Sum([build(t, shared) for t in terms])
        case ir.Prod(factors=factors):
            return Product([build(f, shared) for f in factors])
        case ir.Repeat(body=body, times=times):
            return Sequence([build(body, shared) for _ in range(times)])
        case ir.Tap(name=name):
            return Push(name)
        case ir.Ref(name=name, pop=pop):
            return Read(name, pop)
        ...  # Neg, Const, Identity
```

The IR depends on nothing but pydantic and the configs, so a backend in another
framework imports `bux.lang` and nothing of JAX.
