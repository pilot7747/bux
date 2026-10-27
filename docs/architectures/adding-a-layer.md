# Adding a layer

This page is the reference for extending bux's model: everything a new layer can
declare and use, from its config to its kernels. A new layer is two pieces, a
**config** in `bux/config.py` and a **builder** registered for its type; then every
`.bux` file can call it. For a guided walk through one, see the tutorial
[Add a layer](../tutorials/add-a-layer.md). This page lists every hook it touches.

## The two pieces at a glance

A token shift (RWKV): each token moved toward the one before it in its document, by
a learned amount per channel, then projected. First its config, in `bux/config.py`,
added to the `LayerConfig` union:

```python
class ShiftConfig(Config):
    """Token shift (RWKV): each token moved toward the one before it in its document,
    by a learned amount per channel, then projected."""

    type: Literal["shift"] = "shift"
    bias: bool = False  # on the projection


LayerConfig = Annotated[
    NormalizationConfig | ... | ShiftConfig,
    Field(discriminator="type"),
]
```

Then its module and builder, in a module of `bux.architecture` (say
`bux/architecture/layers/shift.py`, imported by `bux/architecture/__init__.py`):

```python
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.common.sharding import projection
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir


class Shift(nnx.Module):
    def __init__(self, width: int, bias: bool, rngs: nnx.Rngs):
        self.mix = nnx.Param(jnp.full(width, 0.5, jnp.float32), names=("embed",))
        self.proj = projection(width, width, "mlp", use_bias=bias, rngs=rngs)

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        prev = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1]
        if (docs := segments(taps)) is not None:  # nothing before a document's start
            prev = jnp.where((docs.within > 0)[..., None], prev, 0)
        return self.proj(x + (prev - x) * self.mix[...])


@register("shift")
def _(
    c: cfg.ShiftConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Shift:
    return Shift(width, c.bias, ctx.rngs)
```

Now `shift` is a layer of the language, checked, completed and documented like any
other:

```bux
model = embed [norm shift] [norm ffn] norm head
```

## The config

A layer's config is a pydantic class in `bux/config.py`. It is the layer's whole
interface: the language, the IR, checkpoints, run ids, the printer, the editor and
these docs all read it.

### The type tag and the union

- Subclass {py:class}`bux.config.Config` (frozen, and extra fields are an error).
- Give it a `type: Literal["name"] = "name"` field. The tag is globally unique, and
  it is the layer's name in `.bux`. One class may carry several tags, as
  `NormalizationConfig` does for `layernorm` and `rmsnorm`
  (`type: Literal["layernorm", "rmsnorm"]`).
- Add the class to the `LayerConfig` union. It is a pydantic discriminated union
  that `ir.Layer` and the compiler fix when they load, so a layer config can only
  come from `bux/config.py`.
- A config that parametrizes layers but is not one, as `rope` or `indexer`, goes in
  the `SubConfig` union instead.

Calling a name that is not in the union tells you what to do:

```console
$ bux check model.bux
error: unknown layer or block 'shift'
 --> model.bux:3:24
  |
3 | model = embed [rmsnorm shift(bias=true)] rmsnorm head
  |                        ^^^^^
  = hint: did you mean 'situ'? Or: a new layer 'shift' needs a config in the LayerConfig union of bux/config.py and a builder under @bux.architecture.register('shift')
```

### Fields, docs and the editor

The config documents itself, once, for every place that shows it
(`bux.lang.reference`):

- Its **docstring** is its entry in the [language reference](../reference/language/layers.md)
  and the editor's hover. Say what it computes and where it comes from.
- The **comment above or beside each field** is that field's description, in the
  reference table and in the editor's completion.
- The `# --- Title ---` comment above a group of classes is the section of the
  reference page they appear in.
- Fields print, complete and list in order: the class's own first, then those it
  inherits.

Field types are what the compiler checks values against, where they are written:

| Field type | In `.bux` |
|------------|-----------|
| `int`, `float`, `bool` | `4`, `1e-5`, `true` |
| `Literal["half", "interleaved"]` | a bare name, `layout=interleaved` |
| `X \| None = None` | `none` turns it off |
| `Annotated[float, Field(ge=0)]` | a constraint, reported at the value |
| another config, `NormalizationConfig \| None` | `qk_norm=rmsnorm`, `qk_norm=norm`, and dotted keys `attn(rope.base=1e6)` |

A shared base class groups fields: `QuadraticAttentionConfig` holds what `gqa`,
`mla` and `csa` share, and `LinearAttentionConfig` what the linear attentions
share. Subclassing `AttentionConfig` marks a layer as a token mixer. Layers that start
their parameters by their depth among the model's token mixers (`lightning`,
`rwkv7`) count it.

Fields that must agree with each other are checked by a validator on the config,
and the compiler reports its error at the setting: `gqa`'s `num_kv_heads` must divide
its `num_heads`, for example.

### A new field on an existing layer

A run's id hashes its layers' settings, so a new field would change the id of every
run that uses the layer, and those runs could no longer resume. Give the field a
default that computes what the layer computed before, and list that default in
`bux.lang.ir._NEWER_LAYER_FIELDS`: a field at that value is left out of the id, as
`gated_mlp`'s `up_offset=0` is ([Run ids](../training/running.md#run-ids)). A training
setting's new field goes in `_NEWER_FIELDS` the same way.

### Ports

A layer that passes internal tensors to other layers lists its ports:

```python
class GQAConfig(QuadraticAttentionConfig):
    """...

    Ports: ``v`` exports the value heads, or imports them for value-residual mixing
    (ResFormer); ...
    """

    ports: ClassVar[frozenset[str]] = frozenset({"v", "kv", "index"})
```

`attn(v=@v1)` then writes the port's tap, and `attn(v=$v1)` reads it; any other port
name is a compile error that lists the layer's ports. Document what each port
carries in the docstring ([Ports](attention.md#sharing-between-layers-ports)).

### Widths

A new layer keeps the width of the stream it reads: the compiler infers every other
width from that, and checks that residual branches come back to the width they
started at. Only `embed`, `linear`, `head`, `nextn` and `merge` change the width
(`bux.lang.compiler`, `layer_output`). A layer reads one stream: under a residual
algebra that widens it, the compiler puts a `merge` before it, except for `engram`
and `nextn`, which act on every copy.

## The builder

### `@register`

{py:func}`bux.architecture.registry.register` registers a builder for one or more
config types: `@register("mlp", "gated_mlp")`. A layer module registers its builders
when it is imported, and `bux.architecture` imports all of bux's own; a test checks
that every type in `LayerConfig` has one once it is imported. A builder registered
in your own module works once that module is imported, before the model is built.

A builder is called as `builder(config, width, ports, ctx)`
({py:class}`bux.architecture.registry.Builder`) and returns an `nnx.Module`:

| Argument | What it is |
|----------|-----------|
| `config` | the layer's config |
| `width` | the width of the stream it reads: an int, or `"streams"` for a widened stream (`engram`, `nextn`) |
| `ports` | the taps it reads and writes, a tuple of {py:class}`bux.lang.ir.Port` |
| `ctx` | a {py:class}`bux.architecture.registry.Context` |

The `Context` holds:

| Attribute | What it is |
|-----------|-----------|
| `arch` | the whole `Architecture`: `d_model`, `vocab_size`, `residual`, `init` |
| `rngs` | the `nnx.Rngs` that draw the initial parameters |
| `depth` | the residual branches around the layer |
| `built` | what earlier layers left: `count` of layers built by type, the first `embedding` (which tied heads and `nextn` read), the `primes` Engram's tables took |

Read and write ports with `bux.architecture.common.ports`: `reads(ports, "v")` and
`writes(ports, "v")` say whether the layer does, `read(taps, ports, "v")` returns the
value (popping it for `$v!`), and `write(taps, ports, "v", value)` pushes it if the
layer writes the port.

### Replacing a builtin's builder

Registering a builder for a type that has one replaces it, so another implementation
of a builtin layer, a fused kernel say, plugs in without a fork:

```python
from bux import config as cfg
from bux.architecture import Context, register
from bux.lang import ir


@register("gdn")
def _(
    config: cfg.GatedDeltaNetConfig,
    width: int,
    ports: tuple[ir.Port, ...],
    ctx: Context,
):
    return MyGatedDeltaNet(config, width, rngs=ctx.rngs)
```

Import it after `bux.architecture`, so that yours is the last registered.

## The call convention

A module is called as `module(x, taps)` and returns the new stream, of the same
shape: `x` is `Float[Array, "b t d"]`, b rows of t positions, d wide. `taps` is the
forward pass's {py:class}`bux.architecture.registry.Taps`. Read it through these
functions of `bux.architecture`, which accept `taps=None` too:

| Function | Returns |
|----------|---------|
| `segments(taps)` | the documents packed into the rows ({py:class}`bux.architecture.common.segments.Segments`), or `None` when each row is one |
| `positions(taps)` | each token's position in its document, or `None` |
| `training(taps)` | whether this is a training step's forward |
| `report(taps, name, weight, value)` | adds `weight * value` to the training loss |
| `observe.stat(taps, name, value)` | logs `value` by layer on the steps a run observes ([Observing a run](../training/observing.md)) |

`taps.tokens` holds the input tokens, which `engram` and `nextn` read.

### Documents

A layer that looks across positions must not look across documents. `Segments`
holds, made once a forward pass and none of them t by t:

- `ids`: each token's document, the segment ids flash attention kernels take;
- `within`: how many tokens of its document come before each token, where causal
  convolutions, n-grams and targets stop looking back;
- `positions`: each token's position in its document, for RoPE.

### Losses

A layer adds a term to the training loss in one line:

```python
report(taps, "balance_loss", weight, value)  # value: a scalar array
```

The objective adds `weight * value` to the loss, and logs the values, summed over the
layers that report the name, as a metric of that name. Only a training step's forward
collects them; elsewhere, eval included, `report` does nothing. Check
`training(taps)` to skip a costly loss outside training.

### What a layer observes

A value worth watching, a gate's mean or a router's load, is one line too:

```python
from bux.architecture import observe

observe.stat(taps, "mine/gate_mean", gate.mean())
```

It is logged per layer, as `mine/gate_mean/attn.03`, on the steps a run's `observe`
setting picks, with no gradient; the other steps compile without it. Put work that
only an observation needs under `if observe.observing(taps):`.

### State that is not trained by gradients

Keep it in an `nnx.Variable`, not an `nnx.Param`: no optimizer trains it, and
checkpoints keep it. A module that updates it from the batch, as an MoE moves its
router bias, follows the `StepEndUpdate` protocol of `bux.train.trainer`:

- a training step sets its `update_at_step_end`;
- its forward passes only count what they see;
- the trainer calls its `step_end_update()` once, after all of a step's
  microbatches.

## Names for sharding

A layer names the dimensions of its parameters where it makes them, and of its
activations where it knows them; the run's `layout` says which mesh axes each name
splits over ([Layout](../parallelism/layout.md)):

```python
from bux import parallel

self.q = nnx.Linear(
    d, h * k, kernel_init=parallel.named("embed", ("heads", h)), rngs=rngs
)
self.sinks = nnx.Param(jnp.zeros(h), names=("heads",))

q = parallel.shard(q, "batch", "seq", "heads", None)
```

`parallel.named` starts a kernel as the model's `init` says for the role its names
give it: a kernel whose last name is `embed` writes back to the stream, so it is an
`out` ([Initialization](initialization.md)). Pass `init=` for the layer's own start,
such as zeros or a decay, which it keeps whatever the model's `init`. A weight matrix
made by hand starts by `bux.initializers.weight(role)`.
`bux.initializers.weight(role, own=...)` starts it the layer's own way unless the
file sets an `init`, as Gated DeltaNet-2 starts its projections from Xavier's.

The names are the fields of `layout`, a closed set; any other is an error:

| Name | What it is |
|------|-----------|
| `batch` | rows |
| `seq` | positions, inside a layer |
| `stream` | positions of the residual stream, between branches |
| `embed` | the model's width |
| `vocab` | the vocabulary |
| `heads`, `kv_heads` | query heads or a mixer's heads; key and value heads |
| `mlp` | hidden units |
| `experts` | experts |
| `latent` | low-rank bottlenecks: MLA, LatentMoE |
| `layers`, `stages` | a scanned repeat's stacked passes; a pipelined one's |

A dimension that flattens several of what it names, as a projection's `h * k`
columns flatten `h` heads, names them with their count, `("heads", h)`: it then
splits only between them. Names do nothing outside a mesh, and a layer with no names
still trains under FSDP: names matter for tensor and expert parallelism.

`bux.architecture.common.sharding` has the pieces most layers use:

| Function | What it does |
|----------|--------------|
| `linear(n_in, n_out, names, bias, rngs, in_compute=False, role=None)` | an `nnx.Linear` with named dimensions, in its role's format |
| `projection(n_in, n_out, role=None, **kwargs)` | an `nnx.Linear` in its role's format |
| `shard_layer_input(x)` | splits a layer's input as inside a layer, gathering it under sequence parallelism |
| `project_input(x, *linears)`, `project_output(h, linear)` | a layer's input and output projections by the run's `methods.tensor` ([Tensor parallelism](../parallelism/tensor.md)) |
| `shard_heads(x, name="heads")` | splits something of every head by head |

## Precision roles

A projection inside a block multiplies in the format the run's `precision` gives its
**role** ([Precision](../training/precision.md)). Pass the role where you make it:
`projection(n_in, n_out, "mlp", ...)` or `linear(..., role="mlp")`. The roles are
the fields of `roles`:

| Role | Projections |
|------|-------------|
| `q` | attention's queries and gate (`gqa`, `mla`, `csa`, the indexer) |
| `kv` | attention's keys and values |
| `o` | attention's output projection |
| `mlp` | MLPs, the shared experts too |
| `experts` | the routed experts |
| `latent` | LatentMoE's projections to and from its latent |
| `recurrence` | the recurrent mixers' projections, except the output |
| `recurrence_out` | their output projection |
| `mtp` | `nextn`'s projection, and every branch after the first head |

A role that is not one of these is an error. `role=None` multiplies in
`precision.matmuls`; `linear(..., in_compute=True)` always in the compute dtype, as
the router, the head and the relative bias's projection do. Statistics a layer
computes, as a norm's, are its own to take in float32.

## Remat names

A remat policy keeps what layers name and computes the rest again
([Remat](../parallelism/remat.md)). Name what is worth keeping:

- `checkpoint_name(x, "mlp")` (from `jax.ad_checkpoint`) names an intermediate.
- `remat.named("attention", f, *args)` (`bux.architecture.repeats.remat`) runs `f`,
  naming its inputs `attention.in` and its output, with what else its backward takes
  (a kernel's residuals), `attention.out`, so that a policy that keeps `attention`
  computes none of `f` again.
- `remat.label("attention.index", tree)` names every array of a tree.

The names bux's layers use:

| Name | What it covers |
|------|---------------|
| `attention.in` | softmax attention's q, k and v, as its kernel takes them |
| `attention.out` | its output, and what else its backward takes: the logsumexp |
| `attention.index` | the keys sparse attention's indexer kept for each query |
| `recurrence.in` | a recurrent mixer's inputs, as its kernel takes them |
| `recurrence.out` | its output, and what else its backward takes: the chunks' states |
| `qkv` | attention's projections, MLA's latents |
| `mlp` | an MLP's up and gate projections |
| `experts` | an MoE's grouped products |

A name covers those under it, so `attention` is all three of attention's. A new name
works in `remat(save=[...])` with no other change.

## Optimizer attributes

Modules tell the optimizers how to treat their parameters with class attributes
(`bux.train.optim`):

| Attribute | What it does | Used by |
|-----------|--------------|---------|
| `no_weight_decay = ("scale", "bias")` | weight decay skips these parameters, whatever their shape | norms, DyT, Engram's table, Mamba's `A_log`, RWKV-7's token shift, hc's static maps |
| `no_muon = True` | under `muon`, the module and everything in it train by its `adam` | the embedding, the head |
| `muon_axes = {"up": (1, 2)}` | Muon trains this parameter as matrices over these axes (reduction, output); the axes before are batch | the MoE's stacked experts |

By default AdamW decays matrices only (two or more dimensions past a scanned
repeat's stacked one), and Muon trains every `nnx.Linear`'s kernel outside a
`no_muon` module ([Optimizers](../training/optimizers.md)).

## Kernels with fallbacks

A layer that runs a kernel on a GPU falls back to XLA elsewhere, computing the same
values (`bux.kernels`):

- `kernels.gpu()` says whether kernels run here: on a GPU, outside `kernels.off()`,
  and without `BUX_KERNELS=off`. Ask it, and the shapes and dtypes the kernel takes,
  in one predicate, as `fused_delta.runs` and `pallas_delta.runs` do.
- `kernels.off()` turns every kernel off while a block traces, to count the FLOPs a
  kernel hides or to debug one.
- `kernels.interpreted()` says a Pallas kernel called here runs interpreted, off a
  GPU, as the tests call them.
- `kernels.per_device(kernel, ins, outs)` runs a kernel in a `shard_map` on each
  device's part of its inputs, since XLA cannot split a kernel over a mesh.

The kernels bux has, and what they gain: [GPU kernels](../performance/kernels.md).

## Exporting

A model exports to transformers ([Exporting models](../export/index.md)) once each
of its layer types has a class in `bux.export.hf`: the layer in PyTorch, which every
exported repo copies. Until then, exporting a model that uses it raises
`NotImplementedError` naming it. The contract, which `bux.export.hf.common` states
in full:

- **Built from its config.** `bux_types = ("shift",)` lists the types it implements;
  it is built as `Layer(c, width, site)`, `c` the config as JSON, and called as
  `layer(x, ctx)`, `ctx` the pass's positions, masks, cache and taps.
- **Named as in bux.** Its attributes mirror the NNX module's: `nn.Linear` where bux
  has `nnx.Linear`, `nn.Embedding` for `nnx.Embed`, `Norm` for bux's norms. Then
  `bux.export.weights` converts its weights with no code; a layer whose layout
  differs converts its own tensors in a `bux_weights` classmethod.
- **Every setting, or an error.** A setting it does not compute raises through
  `unsupported(c, ...)`; it is never ignored. Settings that only matter in training
  (dropout, a loss's weight) are, and its docstring says so.
- **State while decoding.** A layer that keeps state says its cache's kind in
  `cache_type(c, ports)` and takes `ctx.cache_slot()` once per call; `hf/conv.py` is the
  worked example, left padding included. A layer that reads attention masks lists
  them in `masks(c)`.
- **Runs where transformers does.** It imports only torch, transformers and its
  sibling modules (relatively), and parses on Python 3.10: transformers loads it as
  remote code.

`check.compare(arch).check()` tests it against bux on perturbed weights: the same
logits, the same greedy tokens with and without the cache, and left-padded batches
that match each prompt run alone. Each family of layers has its own
`tests/test_export_*.py`. `tests/test_export_zoo.py` keeps a table of the zoo's models
that do not export yet, by what they lack; remove a model from it once it exports.

## Testing

- **Shapes.** Annotate every array with its dtype and shape, as
  `Float[Array, "b t d"]`. The tests instrument `bux.architecture`, `bux.data`,
  `bux.parallel` and `bux.train` with jaxtyping and beartype, so every call checks
  the arrays into and out of every function (`tests/conftest.py`), and
  `tests/test_typing.py` checks that every array annotation has a dtype.
- **The layer runs.** `tests/test_backend.py` builds small models in the language
  and runs them: a new option belongs in `test_new_layer_options_run`.
  `test_importing_the_backend_registers_every_layer` checks your builder is
  registered.
- **Its parameters are float32.** Start each in a dtype of its own,
  `jnp.full(width, 0.5, jnp.float32)`: one weakly typed, as `jnp.full(width, 0.5)`
  and `jnp.asarray(0.5)` are, turns strong at its first update, and the training
  step compiles again. `tests/test_examples.py` checks every layer's, of the zoo's
  models and of `OTHERS`, its models of the layers the zoo lacks, which
  `test_every_layer_and_residual_algebra_is_in_an_example_or_another_model` makes
  you add yours to if no model of the zoo uses it.
- **It splits.** The tests run on 8 CPU devices: the `losses(source, mesh=...)`
  fixture trains a few steps split over them, to compare with one device
  ([Parallelism](../parallelism/index.md)).
- **A model of the zoo uses it.** Add the model's file and a `SHRUNK` entry
  ([Adding a model](model-zoo.md#adding-a-model)).

Run one file at a time: `uv run pytest tests/test_backend.py`.
