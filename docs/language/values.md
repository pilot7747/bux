(lang-values)=
# Settings and values

This page covers how layers and settings are configured: calling a config with
keyword arguments, the values a setting can take (numbers, strings, lists, configs,
arithmetic, settings read off other layers), how bare names are read, the top-level
settings, and the width checks. Every layer's and setting's fields are listed in the
[language reference](../reference/language/index.md).

## Example

```bux
d_model    = 4096
vocab_size = 128k + 256                       # 128256: a count with a suffix

norm  = rmsnorm(eps=1e-5)
attn  = gqa(num_heads=32, num_kv_heads=8,
            qk_norm=norm,                     # a definition as a value
            rope=rope(base=5e5, layout=interleaved))  # a config; a bare name is a string
local = gqa(num_heads=attn.num_heads,         # a setting read off another layer
            num_kv_heads=attn.num_kv_heads // 2,
            sliding_window_size=4096)
ffn   = gated_mlp(intermediate_size=8 * d_model // 3, activation=swish)  # arithmetic

block = [norm attn] [norm ffn]
sw    = [norm local] [norm ffn]
model = embed (sw block)^16 norm head
```

## Calling a layer or a setting

A call configures a layer or a setting: `gqa(num_heads=32, num_kv_heads=8)`. The
rules:

- **Keyword arguments.** Each argument names a field of the config, the pydantic
  classes in `bux/config.py` and `bux/train/config.py`. Positional arguments are an
  error: `layer settings are keyword arguments: name=value`.
- **The name is the type.** Every config's `type` tag is its name in the language,
  so the type cannot be set as an argument.
- **Dotted keys** reach into a nested setting: `gqa(..., rope.base=1e6)` changes
  `base` of the attention's `rope`.
- **Each setting once.** Giving one twice in a call is an error.
- **No call means the defaults.** A bare config name is that config with its
  defaults: `qk_norm=rmsnorm`, `packing = best_fit`.
- **Calling a definition** overrides some of its settings:
  `local = attn(sliding_window_size=1024)` ([Definitions](definitions.md)).

The names the language knows:

| Kind | Names |
|------|-------|
| layers | `gqa`, `mla`, `csa`, `mamba`, `mamba2`, `gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`, `rwkv7`, `conv`, `short_conv`, `mlp`, `gated_mlp`, `moe`, `layernorm`, `rmsnorm`, `dyt`, `linear`, `act`, `scale`, `dropout`, `embed`, `head`, `nextn`, `engram`, `merge` |
| settings of a model | `rope`, `pi`, `yarn`, `llama3`, `relbias`, `indexer`; the residual algebras `hc`, `mhc`, `gr`, `attnres`, `vpgate` |
| training settings | `adamw`, `muon`, `wsd`, `text`, `tokenizer`, `mesh`, ... ([Training settings](../reference/language/training.md)) |
| activations | `swish`, `gelu`, `gelu_tanh`, `relu`, `relu2`, `sigmoid`, `tanh`, `softplus`, `situ` |

An activation is a layer that applies it elementwise, `linear(features=h) swish`, and
also a value for the settings that take one, `activation=swish`.

(lang-value-kinds)=
## Values

| Kind | Examples | Notes |
|------|----------|-------|
| integers | `4096`, `128_256`, `32k`, `1.5B` | `_` separates digits |
| floats | `1e-5`, `0.02`, `5e5` | a number with a `.` or an exponent: `1e3` is the float 1000.0 |
| strings | `"gpt2"`, `'data/*.parquet'` | either quote; no escapes; on one line |
| booleans and none | `true`, `false`, `none` | |
| lists | `[1, 2]`, `[0, 0.999]`, `[1,]`, `[]` | a list needs a comma or nothing: `[x]` is a [residual branch](skip-connections.md) |
| configs | `rope(base=1e6)`, `rmsnorm` | a bare name is the config with its defaults |
| definitions | `qk_norm=norm`, `expert=dense` | the value of another definition |
| arithmetic | `4 * d_model // 3`, `2^10` | over numbers and numeric definitions |
| bare names | `layout=interleaved`, `router=sigmoid` | a name nothing defines is a string |
| field reads | `attn.num_heads // 4`, `rope.base` | a setting of a layer or config |

### Numbers and suffixes

A count can end in a suffix: `k` or `K` (10^3), `M` (10^6), `B` (10^9) or `T`
(10^12). `128k` is 128000 and `1.5B` is 1500000000.

- **A suffix is exact.** `1.1B` is the integer 1_100_000_000, not a float that
  rounds near it. A suffixed number is an integer when it comes out whole:
  `1.5k` is the integer 1500, and `1.2345k` is the float 1234.5.
- **A suffix ends the number.** `4kv` is the number 4 followed by the name `kv`.

Suffixes are most useful in training settings: `steps = 10B // (batch * seq_len)`,
`warmup=2k`, `budget = 3B`.

(lang-arithmetic)=
### Arithmetic

`+`, `-`, `*`, `/`, `//` and `^` work on numbers and on definitions that are
numbers, with the [precedence](operators.md#precedence-and-associativity) of the
operators on blocks:

| Operator | On numbers |
|----------|------------|
| `a + b`, `a - b`, `-a` | sum, difference, negation |
| `a * b` | product |
| `a / b` | division, always a float: `7 / 2` is 3.5 |
| `a // b` | floor division: `7 // 2` is 3 |
| `a ^ b` | power: `2^10` is 1024 |

Dividing by zero is an error. Settings are definitions, so arithmetic works on them
too: `steps = budget // (batch * seq_len)`. Data sources are the one exception to
numbers-only arithmetic: `0.8 * web + 0.2 * code` is a weighted mix
([Data sources](../data/sources.md)).

(lang-bare-names)=
### Bare names are strings

A name that nothing defines, and that is not a builtin, is a string when it is the
value of a setting. So enumerated settings need no quotes: `layout=interleaved`,
`load_balancing=aux_free`. Activation names are strings in the same way:
`router=sigmoid`, `activation=swish`.

Where a setting wants something other than a string, or in arithmetic, an undefined
name is a typo, and the compiler reports it as an unknown name with a suggestion:

```text
error: unknown name 'dmodel'
 --> model.bux:5:41
  |
5 | ffn   = gated_mlp(intermediate_size=4 * dmodel, activation=swish)
  |                                         ^^^^^^
  = hint: did you mean 'd_model'?
```

:::{tip}
In the editor, hovering a bare name tells you which it is: the definition it refers
to, or that it is not defined and so is a string. Defining a name that you also use
as a string changes what the string means.
:::

The settings `save` and `offload` of {bux}`remat` are an exception: they read the
names of what layers make (`attention`, `attention.in`, `mlp`) as written, never as
definitions, and `[attention]` is a list of one.

(lang-field-reads)=
### Reading a setting: `name.field`

`name.field` reads a setting back from a layer or config:

```bux
kv      = gqa(num_heads=32, num_kv_heads=attn.num_heads // 4)
inner   = experts.expert                     # a layer: the expert's config
size    = experts.expert.intermediate_size   # reads chain
base    = rope.base                          # a builtin's default: 10000.0
```

- It reads the value **as written**. A default that depends on the width, such as
  `head_dim` of `gqa`, reads as `none`.
- The `.` must follow the name directly: `attn.num_heads`, not `attn .num_heads`.
- Only layers and settings have fields: reading one off a block is an error, and an
  unknown field gets a suggestion.

## Checking against the schema

Every setting is checked against its config's schema where it is written, before
anything runs. Unknown and missing settings, and values of the wrong type, are
errors, with suggestions where there is a close match. `bux check` reports them all
at once. For this file:

```bux
d_model    = 512
vocab_size = 32000
norm       = rmsnorm

ffn   = gated_mlp(intermediate_size=4 * dmodel, activation=swish)
attn  = gqa(num_heads=2, num_kv_head=2)
wide  = gqa(num_kv_heads=2)
pos   = rope(layout=interleave)
qk    = gqa(num_heads=8, num_kv_heads=2, qk_norm=nrm)
half  = gqa(num_heads=8.5, num_kv_heads=2)
block = [norm] [norm]
heads = block.num_heads
model = embed block norm head
```

it prints the `dmodel` error above, then:

```text
error: gqa has no setting 'num_kv_head'
 --> model.bux:6:26
  |
6 | attn  = gqa(num_heads=2, num_kv_head=2)
  |                          ^^^^^^^^^^^^^
  = hint: did you mean 'num_kv_heads'?

error: gqa needs 'num_heads'
 --> model.bux:7:9
  |
7 | wide  = gqa(num_kv_heads=2)
  |         ^^^^^^^^^^^^^^^^^^^
  = hint: gqa(num_heads=...)

error: rope: layout: Input should be 'half' or 'interleaved'
 --> model.bux:8:14
  |
8 | pos   = rope(layout=interleave)
  |              ^^^^^^^^^^^^^^^^^
  = hint: did you mean 'interleaved'?

error: unknown name 'nrm'
 --> model.bux:9:50
  |
9 | qk    = gqa(num_heads=8, num_kv_heads=2, qk_norm=nrm)
  |                                                  ^^^
  = hint: did you mean 'norm'?

error: gqa: num_heads: Input should be a valid integer, got a number with a fractional part
  --> model.bux:10:13
   |
10 | half  = gqa(num_heads=8.5, num_kv_heads=2)
   |             ^^^^^^^^^^^^^

error: only layers and settings have fields, but this is a block
  --> model.bux:12:9
   |
12 | heads = block.num_heads
   |         ^^^^^
```

(lang-top-level)=
## Top-level settings

A few top-level names are settings of the architecture or the run rather than
definitions you use elsewhere:

| Name | Value | Notes |
|------|-------|-------|
| `model` | the block to build, `embed ... head` | required; `--entry` builds another definition instead |
| `d_model` | an integer: the width of the residual stream | required |
| `vocab_size` | an integer: the vocabulary of `embed` and `head` | required when the model has an `embed` or a `head` |
| `residual` | `hc(...)`, `mhc(...)`, `gr(...)`, `attnres(...)` or `vpgate(...)` | the residual algebra ([Residual algebras](residual-algebras.md)) |
| `data`, `optimizer`, `seq_len`, `batch`, `steps`, ... | training settings | setting any of them makes the file a run ([Training settings](../training/settings.md)) |

Settings are definitions, so arithmetic, field reads and suffixes work on them, and
other definitions can read them: `intermediate_size=4 * d_model`. They
cannot be macros. A file that sets how to train is a run and needs every required
training setting; one without a model is a module
([Runs and modules](modules.md#runs-and-modules)). The full table is in
[Top-level settings](../reference/language/top-level.md).

The compiler checks their types too:

```text
error: 'd_model' must be an integer, found a number
 --> model.bux:1:14
  |
1 | d_model    = 512.0
  |              ^^^^^
```

(lang-widths)=
## Widths

Every leaf knows the width of its input, and what it returns:

| Layer | Takes | Returns |
|-------|-------|---------|
| `embed` | token ids | `d_model` (or the [widened stream](residual-algebras.md)) |
| `linear(features=n)` | any width | `n` |
| `head` | hidden states | logits |
| everything else | hidden states | the width it takes |

A model may also start without an `embed`; it then takes hidden states of width
`d_model`. The compiler checks that:

- a residual branch and a repeated block return to the width they started at;
- parallel branches agree on their width;
- `embed` comes first, and nothing comes after `head`;
- every use of a `shared` block has the same widths.

With `up = linear(features=2048)` and `d_model = 512`, a branch that widens the
stream and a sum of branches of two widths are errors:

```text
error: a residual branch must keep the width, but this one maps 512 to 2048
 --> model.bux:6:15
  |
6 | model = embed [norm up] norm head
  |               ^^^^^^^^^
```

```text
error: parallel branches disagree on width: 2048, 512
 --> model.bux:6:22
  |
6 | model = embed [norm (up + norm)] norm head
  |                      ^^^^^^^^^
```

Widths, like taps, are checked along the model that is built. A definition the model
does not use has its settings checked, but not its widths.
