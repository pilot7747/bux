# Add a layer

In this tutorial you add a new layer to bux and use it in a `.bux` file: a **token
shift**, which mixes each token's stream with the token before it, as RWKV does. You
write its config, which the compiler, the editor and the docs read, and its builder,
which makes the Flax module that runs it. Then you train with it, sweep it on and
off, see its documentation on hover, and test it.

A layer is two pieces of Python in bux's source, so you edit your checkout:

| piece   | where                                   | what it does                                    |
|---------|-----------------------------------------|-------------------------------------------------|
| config  | `src/bux/config.py`, in the `LayerConfig` union | its name in the language and its settings, checked as pydantic fields |
| builder | a module under `src/bux/architecture/`, registered with `@register` | makes the module that computes it |

It builds on [Your own data](your-own-data.md), whose stories run it trains.
[Adding a layer](../architectures/adding-a-layer.md) is the reference for everything
a layer can do.

## 1. What the layer computes

With a learned share $\mu$ per channel, the token shift at position $t$ is

$$
y_t = (1 - \mu) \, x_t + \mu \, x_{t-1},
$$

and a document's first token has nothing before it, so there $x_{t-1}$ is 0. Packed
rows hold several documents, so "the token before" must stop at a document's start.
It keeps the stream's width, so it can go anywhere in a block:
`[norm token_shift attn]`.

Before you write anything, the compiler does not know the name, and says what a new
layer needs:

```console
$ bux check mine/tiny.bux 'shift = token_shift(init=0.5)'
error: unknown layer or block 'token_shift'
 --> <command line>:1:9
  |
1 | shift = token_shift(init=0.5)
  |         ^^^^^^^^^^^
  = hint: did you mean 'tokens'? Or: a new layer 'token_shift' needs a config in the LayerConfig union of bux/config.py and a builder under @bux.architecture.register('token_shift')
```

## 2. Write its config

A config is a pydantic class with a `type` tag, which is its name in the language. Its
docstring says what it is, and the comment beside or above each field says what the
field does. In `src/bux/config.py`, next to the other sequence mixers (after
`ShortConvConfig`), add:

```python
class TokenShiftConfig(Config):
    """Token shift (RWKV): each channel mixes in the token before, by a learned share
    mu, as (1 - mu) x[t] + mu x[t - 1]. A document's first token has none before it."""

    type: Literal["token_shift"] = "token_shift"
    init: float = 0.5  # the share mu that every channel starts at
```

Then add it to the `LayerConfig` union, further down the same file. The union is
where the compiler learns layer names, and what the IR, checkpoints, run ids, the
printer and the editor read configs by:

```python
LayerConfig = Annotated[
    NormalizationConfig
    | DyTConfig
    ...
    | ShortConvConfig
    | TokenShiftConfig
    | LinearConfig
    ...
    Field(discriminator="type"),
]
```

That is enough for the language. `bux check` imports no JAX and needs no builder:

```console
$ bux check mine/tiny.bux 'block = [norm token_shift attn] [norm ffn]'
d_model=128 vocab_size=512 layers=23
rmsnorm×9  token_shift×4  gqa×4  gated_mlp×4  embed×1  head×1
200 steps × 8 × 128 tokens = 205k tokens
optimizer=adamw(lr=0.003) schedule=wsd(warmup=20)
data=synthetic()
$ bux render mine/tiny.bux 'block = [norm token_shift attn] [norm ffn]'
model
──embed──┤block├×4──norm──head──

block
──┬─────────────────────────⊕──┬───────────⊕──
  └─norm──token_shift──attn─┘  └─norm──ffn─┘
```

Its settings are checked against the schema, with suggestions, like every layer's:

```console
$ bux check mine/tiny.bux 'shift = token_shift(int=0.9)'
error: token_shift has no setting 'int'
 --> <command line>:1:21
  |
1 | shift = token_shift(int=0.9)
  |                     ^^^^^^^
  = hint: did you mean 'init'?
```

Training is another matter. Without a builder, the backend cannot make the module, and
says so as it builds the model:

```console
$ bux train mine/shifted.bux
...
NotImplementedError: the reference backend has no 'token_shift' layer yet; add one with @bux.architecture.register('token_shift')
```

## 3. Write its builder

A builder takes the layer's config, the width of the stream it reads, the taps it
reads and writes (its ports), and a `Context` with the whole architecture and the
`rngs` for initial parameters. It returns a Flax NNX module, which the model calls as
`module(x, taps)`. Create `src/bux/architecture/layers/shift.py`:

```python
"""Token shift: each channel mixes in the token before, within its document."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from bux import config as cfg
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir


class TokenShift(nnx.Module):
    def __init__(self, c: cfg.TokenShiftConfig, width: int):
        self.mu = nnx.Param(jnp.full(width, c.init))

    def __call__(
        self, x: Float[Array, "b t d"], taps: Taps | None = None
    ) -> Float[Array, "b t d"]:
        before = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1]
        if (docs := segments(taps)) is not None:  # None: every row is one document
            before = jnp.where((docs.within > 0)[..., None], before, 0)
        return x + self.mu[...] * (before - x)


@register("token_shift")
def _(
    c: cfg.TokenShiftConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> TokenShift:
    return TokenShift(c, width)
```

- `segments(taps)` says where the documents of a packed batch are, made once a
  forward pass from the batch's positions. `within` counts the tokens of a token's
  document before it, so `within > 0` is false exactly at a document's start. It is
  `None` when every row is one document.
- `x + mu * (before - x)` is $(1 - \mu) x_t + \mu x_{t-1}$.
- The annotations say each array's dtype and shape (`"b t d"`: rows, positions,
  width); bux's tests check them on every call.

A builder registers when its module is imported, and `bux.architecture` imports bux's
own. In `src/bux/architecture/__init__.py`, import it next to the other layers, and
add `"TokenShift"` to `__all__`:

```python
from bux.architecture.layers.norm import DyT, Norm
from bux.architecture.layers.shift import TokenShift
```

:::{note}
Registering a builder for a type that has one already replaces bux's own, so another
implementation of a layer, a fused kernel say, plugs in the same way. A *new* type
always needs its config in `bux/config.py`: the union is a pydantic discriminated
union that the IR and the compiler fix as they load.
:::

## 4. Use it

Create `mine/shifted.bux`, the stories run of [Your own data](your-own-data.md) with
a token shift before every attention:

```bux
# The stories run, with a token shift before every attention.
use stories

shift = token_shift(init=0.5)
block = [norm shift attn] [norm ffn]
```

```console
$ bux train mine/shifted.bux
remat=auto: none in 1s (none 0.5 GiB in 0s)
shifted step 25/200  loss 2.2135  lr 3.00e-03  grad 3.427  5,649 tok/s
shifted step 50/200  loss 0.8024  lr 3.00e-03  grad 0.551  7,612 tok/s
shifted step 50/200  eval loss 0.7792
...
shifted step 200/200  loss 0.7181  lr 0.00e+00  grad 0.265  7,591 tok/s
shifted step 200/200  eval loss 0.7299
```

Nothing else had to know about the new layer: the trainer, the optimizer's weight
decay (which skips vectors such as `mu`), checkpoints, `bux plan`'s parameter and
FLOP counts, and the run's id all follow from the config and the module.

To ablate it, make it an alternative. `_` is the identity, so `off` is the block
without the shift:

```console
$ bux plan mine/shifted.bux 'shift = off: _ | on: token_shift'
2 runs
id          params  tokens  tok/param  flops/tok  gpus  params/gpu  optim/gpu  bubble  run
00edf52242  7.18M   205k    0.0        43.5M      1     27.4MiB     54.8MiB    0%      shifted[off]
9dae7a4e31  7.18M   205k    0.0        43.5M      1     27.4MiB     54.8MiB    0%      shifted[on]
```

`shifted[on]` has the id of the run you trained: `token_shift` with its default is
the same as `token_shift(init=0.5)`.

## 5. See it in the editor

`bux-lsp` documents a layer on hover from its config: the docstring, and each field
with its type, its default and its comment. Hover over `token_shift` in
`mine/shifted.bux` and it shows:

```text
**token_shift** — layer

Token shift (RWKV): each channel mixes in the token before, by a learned share
mu, as (1 - mu) x[t] + mu x[t - 1]. A document's first token has none before it.

- `init: float = 0.5` — the share mu that every channel starts at
```

Completion offers `init` inside `token_shift(`, and errors appear as you type
([Tooling](../language/tooling.md)). The [language reference](../reference/language/layers.md)
of these docs is generated from the same docstring and comments, so the layer is
documented there too.

The editor's features are plain functions, so you can call them yourself:

```python
from bux.lang.lsp.features import hover

path = "mine/shifted.bux"
source = open(path).read()
print(hover(source, source.index("token_shift") + 1, path).markdown)
```

## 6. Test it

A test of a layer checks what it computes on a small input. Create
`tests/test_token_shift.py`:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from bux import config as cfg
from bux.architecture import Model, Segments, Taps, TokenShift
from bux.lang import compile


def test_mixes_in_the_token_before_within_its_document():
    shift = TokenShift(cfg.TokenShiftConfig(init=0.25), width=3)
    x = jax.random.normal(jax.random.key(0), (1, 5, 3))
    positions = jnp.array([[0, 1, 2, 0, 1]])  # two documents in one row
    y = shift(x, Taps(segments=Segments.of(positions)))
    assert jnp.allclose(y[:, 1], 0.75 * x[:, 1] + 0.25 * x[:, 0])
    assert jnp.allclose(y[:, 3], 0.75 * x[:, 3])  # nothing before a document's start


def test_a_model_runs_it():
    arch = compile(
        "d_model = 16\nvocab_size = 32\n"
        "attn = gqa(num_heads=2, num_kv_heads=1)\n"
        "model = embed [rmsnorm token_shift attn] rmsnorm head"
    )
    logits = Model(arch, rngs=nnx.Rngs(0))(jnp.zeros((2, 8), jnp.int32))
    assert logits.shape == (2, 8, 32)
```

```console
$ uv run pytest -q tests/test_token_shift.py
..                                                                       [100%]
2 passed in 2.01s
```

The tests run on 8 CPU devices (`tests/conftest.py`) with every array's annotation
checked. Run `make lint typecheck` too: ruff and ty check the new code as they check
the rest.

## 7. Where to go from here

A layer can do more than this one, each through the same two pieces:

- **Settings that are other configs**, such as `rope=rope(...)` or a nested
  `expert=gated_mlp(...)`, are pydantic fields of those types.
- **Optimizer treatment**: class attributes on the module say how optimizers treat
  its parameters: `no_weight_decay` names parameters to skip in weight decay whatever
  their shape, and `no_muon` keeps a module's matrices out of Muon.
- **Parallelism**: a parameter's dimensions take names, `nnx.Param(..., names=("heads",))`,
  which the run's `layout` splits over the mesh. A layer with no names trains under
  FSDP; names matter for tensor and expert parallelism ([Layout](../parallelism/layout.md)).
- **A loss of its own**: `report(taps, "my_loss", weight, value)` in the forward adds
  `weight * value` to the training loss and logs it ([Custom training](custom-training.md)).
- **Ports**: a config's `ports` let it export an internal tensor to a tap, and read it
  back in another layer, as `attn(v=@v1)` does.

[Adding a layer](../architectures/adding-a-layer.md) covers each of these in detail.
