# Definitions, macros, patterns and sharing

Definitions let you write each piece of a model once: a layer with its settings, a
block, a value. On top of them:

- a **macro** takes parameters and builds a new block at every call;
- a **pattern** spells a sequence of one-letter definitions, `"DDDA"`;
- **`shared`** makes every use of a definition one module, with one set of weights.

Read this page to factor a model into reusable pieces. It ends with tied heads.

## Example: Qwen3-Next

```bux
d_model    = 2048
vocab_size = 151936

norm    = rmsnorm(zero_centered=true)
delta   = gdn(num_heads=16, num_v_heads=32, head_dim=128)
attn    = gqa(num_heads=16, num_kv_heads=2, head_dim=256, gated=true, qk_norm=norm,
              rope=rope(base=10000000, fraction=0.25))
experts = moe(num_experts=512, top_k=10, num_shared_experts=1, shared_expert_gate=true,
              aux_loss_coef=1e-3, expert=gated_mlp(intermediate_size=512, activation=swish))

layer(mixer) = [norm mixer] [norm experts]
D = layer(delta)
A = layer(attn)

shared out = head
mtp = $h nextn(norm=norm) A norm out

model = embed "DDDA"^12 norm @h out mtp
```

This is `examples/architectures/qwen3_next.bux`. `layer` is a macro that builds a
block around any mixer, and `D` and `A` are the two layers it makes. The pattern
`"DDDA"^12` lays out 12 groups of three Gated DeltaNet layers and one attention
layer. `out` is `shared`, so the main head and the multi-token prediction head are
one module.

## Definitions

```bux
norm   = rmsnorm                           # a layer: a template, fresh parameters per use
local  = attn(sliding_window_size=1024)    # call a layer to override settings
global = attn(rope.base=1000000)           # dotted keys reach into nested settings
mlp    = mlp(intermediate_size=3072, activation=gelu)  # self-reference means the builtin
block  = [norm attn] [norm mlp]            # a block
hidden = 4 * d_model // 3                  # a number
```

`name = expr` names a layer, a block, a setting or a value. The rules:

- **Order does not matter.** A definition can use one written below it. A cycle is
  an error, reported for each definition in it:

  ```text
  error: definition cycle: a -> b -> a
   --> model.bux:6:5
    |
  6 | b = a norm
    |     ^
  ```

- **A layer definition is a template.** Each use of `norm` is its own `rmsnorm`,
  with its own parameters, unless the definition is [shared](#sharing-shared).
- **Calling a definition overrides its settings.** `attn(sliding_window_size=1024)`
  is `attn` with one setting changed, and dotted keys reach into nested settings:
  `attn(rope.base=1000000)`. Only layers and settings can be called this way. For a
  block that takes arguments, write a [macro](#macros).
- **Inside its own definition, a name means what it replaces.** `mlp = mlp(...)`
  calls the builtin `mlp`. When a [module](modules.md#layering-and-late-binding)
  defines `ffn`, `ffn = ffn(intermediate_size=2048)` changes one setting of that
  `ffn`. With neither a builtin nor a module's definition to replace, a
  self-reference is a cycle.
- **One definition per name in a file.** Defining a name twice in one file is an
  error. Across files, and from the command line, a later definition replaces an
  earlier one ([Modules](modules.md)).
- **Names.** A definition's name is any identifier except the keywords `true`,
  `false`, `none`, `shared` and `_`. It may shadow a builtin, as `mlp = mlp(...)`
  does. Then every `mlp` written anywhere in the program, modules included, means
  the definition. A few names are settings with their own meaning: `model`,
  `d_model`, `vocab_size`, `residual`, `init` and the training settings
  ([Top-level settings](values.md#top-level-settings)).

A definition's name also labels what it builds. `bux render` draws a labeled block as
`┤block├` and draws the block itself below, and `bux fmt` prints it back as a
definition.

(lang-macros)=
## Macros

A macro is a definition with parameters. It builds a new block, layer or value at
every call:

```bux
layer(mixer, ffn=experts) = [norm mixer] [norm ffn]        # a default
attn_kv(n, kv=n // 4) = gqa(num_heads=n, num_kv_heads=kv)  # defaults see earlier parameters

D = layer(gdn(num_heads=16, head_dim=128))
F = layer(attn, ffn=dense)
K = layer(attn_kv(16))
```

The rules:

- **Parameters.** Parameters without a default come first. A default can use the
  parameters before it: `kv=n // 4`. A macro may have none: `pre() = [norm attn]`,
  called as `pre()`.
- **Arguments.** Pass arguments by position, by name, or both: `layer(attn, dense)`,
  `layer(mixer=attn, ffn=dense)`. Too many arguments, an unknown name, a parameter
  given twice and a missing argument are errors.
- **Fresh at every call.** Each call evaluates the body again, so each call has its
  own parameters.
- **Scope.** The body sees its parameters and the program's definitions. A parameter
  hides a definition of the same name.
- **Macros are values.** A macro can be passed to another macro:
  `twice(f, x) = f(x) f(x)`, then `twice(layer, attn)`.
- **A macro needs its arguments.** Using one bare is an error:

  ```text
  error: 'layer' is a macro and needs arguments: layer(mixer, ffn)
   --> model.bux:8:15
    |
  8 | model = embed layer norm head
    |               ^^^^^
  ```

- **Blocks take no arguments.** Calling a plain block is an error,
  `'block' is a block and takes no arguments`, with a hint to make it a macro.
- **No recursion.** A macro that calls itself has no way to stop, so more than 64
  nested calls is an error.
- **Reserved names.** `model`, `d_model`, `vocab_size`, `residual`, `init` and the
  training settings cannot be macros.

A macro call is labeled by its text. `bux render` draws
`layer(dense)^3 layer(experts)^58`, the model of DeepSeek-V3, as two blocks:

```text
model
──embed──┤layer(dense)├×3──┤layer(experts)├×58──norm──head──

layer(dense)
──┬────────────⊕──┬─────────────⊕──
  └─norm──attn─┘  └─norm──dense─┘

layer(experts)
──┬────────────⊕──┬───────────────⊕──
  └─norm──attn─┘  └─norm──experts─┘
```

In a sweep, a macro takes the same alternative at every call
([Sweeps](sweeps.md#which-alternatives-count)).

(lang-patterns)=
## Patterns

A pattern is a string of one-letter definitions, written where a block goes.
`"DDDA"` is `D D D A`:

```bux
m = layer(ssm, dense)
M = layer(ssm, experts)
a = layer(attn, dense)

model = embed "mMmM aMmM"^4 norm head      # Jamba: 1 attention in 8, MoE every other layer
```

- Each letter names a one-letter top-level definition, upper or lower case.
- Spaces inside a pattern are ignored; use them to group.
- A pattern is a block like any other, so it repeats: `"DDDA"^12`.
- As a setting's value, a string is a string, `tokenizer(name="gpt2")`, not a
  pattern.

A letter that is not a definition is an error at that letter:

```text
error: pattern letter 'M' is not a one-letter definition
 --> model.bux:6:18
  |
6 | model = embed "AAM" norm head
  |                  ^
```

`bux fmt` prints patterns spelled out: `"DDDA"^12` becomes `(D D D A)^12`.

(lang-shared)=
## Sharing: `shared`

`shared` makes a definition a single module. Every use of it, including each pass of
a `^n`, has the same parameters, where each use of a plain definition gets fresh ones.

```bux
m = [norm ssm]
shared sa = norm attn norm ffn
h = [(_ + sa linear(features=d_model)) norm ssm]

model = embed (m^5 h)^6 m^2 norm head(tied=true)
```

This is Zamba2 (`examples/architectures/zamba2.bux`). One transformer block, `sa`,
runs before every sixth Mamba-2 layer. Each time it has its own `linear`, but `sa`
keeps the same weights. ALBERT-style weight sharing is `shared block = ...` and
`block^12`.

The rules:

- **Only plain definitions.** A macro cannot be shared: share the blocks it builds,
  `shared A = layer(attn)`.
- **No changed copies.** Calling a shared layer with settings would make a second
  module, so it is an error:

  ```text
  error: 'attn' is shared, so it cannot be changed here
   --> model.bux:5:9
    |
  5 | local = attn(sliding_window_size=1024)
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    = hint: define a separate layer
  ```

  Calling it with only [ports](skip-connections.md#internal-tensors-ports) is fine,
  if every use takes the same ones, and an [amendment](modules.md#amendments)
  changes the shared module itself.
- **Same widths everywhere.** Every use of a shared block must see the same widths:
  `shared 'lin' is used where it would need different widths`. A merge that the
  compiler puts before a shared layer is not part of it.
- **Branches at one depth.** A residual branch outside any other is the residual
  algebra's, and one inside another is a plain `x + f(x)`, so a shared block with
  branches is used either outside branches or inside them, not both:
  `shared 'blk' has residual branches and is used both outside residual branches
  and inside one`.
- `shared` only starts a definition: `shared name = ...`.

[Multi-token prediction](outputs.md) shares the head, and sometimes the whole MTP
module.

## Tied heads

`head(tied=true)` reuses the embedding matrix as the output projection. A model with
a tied head needs exactly one `embed`:

```text
error: a tied head needs exactly one embedding, found 0
```

{bux}`nextn` also reads the embedding, so it has the same rule.
