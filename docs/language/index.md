# The `.bux` language

A `.bux` file describes a language model as an **algebra over the residual stream**.
Layers are functions of the stream. The language composes them, runs them side by
side and sums or multiplies what they return, and draws skip connections between
them. The same file can say how to train the model, and `|` turns any line of it into
a sweep.

This guide describes the whole language: what you can write, what it means, and what
the compiler checks. For the settings of each layer, see the
[language reference](../reference/language/index.md).

## Llama 3 8B in eight lines

```bux
d_model    = 4096
vocab_size = 128256

norm  = rmsnorm(eps=1e-5)
attn  = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))
ffn   = gated_mlp(intermediate_size=14336, activation=swish)

block = [norm attn] [norm ffn]
model = embed block^32 norm head
```

That is all of Llama 3 8B (`examples/architectures/llama3_8b.bux`). The core idea is
that **`[f]` is a skip arc**: it means `x + f(x)`. The bracket reads as the bypass
wire drawn over `f`, and `bux render` draws it that way:

```console
$ bux render examples/architectures/llama3_8b.bux
model
──embed──┤block├×32──norm──head──

block
──┬────────────⊕──┬───────────⊕──
  └─norm──attn─┘  └─norm──ffn─┘
```

Read the model left to right: `embed` turns tokens into the stream, `block^32` is 32
blocks in a row, each with its own parameters, and `norm head` turns the stream into
logits. Each block is two residual branches, so this is the pre-norm decoder.

## A tour of the language

Each section below shows one feature in a line or two, and links to its page.

### Operators

Writing layers next to each other composes them, `+` runs branches side by side and
sums them, `*` multiplies them (gating) or scales them by a number, and `^n` repeats
a block. Composition binds tighter than `+` and `*`, so SwiGLU reads as it is drawn:

```bux
swiglu = (linear(features=h) swish * linear(features=h)) linear(features=d_model)
```

See [Operators](operators.md) for every operator, how expressions parse, and a
catalogue of common block shapes.

### Skip connections

`[f]` adds the stream around `f`. For longer skips, `@name` saves the stream and
`$name` reads it back anywhere later, so a U-net is one line:

```bux
model = embed @x0 (block @skip)^6 ([$skip! scale] block)^6 norm head
```

Layers also export internal tensors through ports, `attn(v=@v1)`, for value
residuals and shared keys and values. See [Skip connections](skip-connections.md).

### Definitions, macros, patterns and sharing

A definition names a layer or block; a macro takes parameters; a pattern spells a
sequence of one-letter definitions; `shared` makes every use one module:

```bux
layer(mixer) = [norm mixer] [norm ffn]
D = layer(gdn(num_heads=16, head_dim=128))
A = layer(attn)
model = embed "DDDA"^12 norm head      # 3 Gated DeltaNet layers : 1 attention
```

See [Definitions](definitions.md).

### Several outputs

Every `head` is an output, and `nextn` mixes in the next token's embedding, so
multi-token prediction is a few more layers after the main head. See
[Outputs and multi-token prediction](outputs.md).

### Settings and values

A layer's settings are keyword arguments, checked against its schema where you write
them. Values are numbers (`128k`, `1.5B`), strings, lists, configs, arithmetic over
other definitions (`4 * d_model // 3`) and settings read off other layers
(`attn.num_heads // 4`). See [Settings and values](values.md).

### Residual algebras

`residual = mhc(expansion=4)` keeps the wiring and changes what `[f]` means: the
stream widens into 4 streams that every branch reads and writes through learned
connections. See [Residual algebras](residual-algebras.md).

### Modules and amendments

A file builds on others by name, and changes one setting of what it uses:

```bux
use llama3_8b
attn.num_kv_heads = 1      # every block of llama3_8b now has multi-query attention
```

The command line takes the same statements. See [Modules and amendments](modules.md).

### Sweeps

`|` separates alternatives anywhere, and a file with alternatives is a sweep of
runs, one for each combination:

```bux
attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3                 # 3 × 2 = 6 runs
```

See [Sweeps](sweeps.md).

### Tools

`bux check`, `bux render`, `bux fmt` and `bux json` compile a file and show what it
means. Errors point at the line that caused them, and every error in a file is
reported at once. `bux-lsp` brings the same checks, with completion and hover, into
your editor. See [Tools and editor support](tooling.md).

## Where to go next

| If you want to | Read |
|----------------|------|
| look up a layer's settings | [Language reference](../reference/language/index.md) |
| write a first model step by step | [Your first architecture](../tutorials/first-architecture.md) |
| train what a file describes | [Training](../training/index.md) |
| add a layer of your own | [Adding a layer](../architectures/adding-a-layer.md) |
| see the exact syntax | [Lexical rules and grammar](grammar.md) |
| know why the language looks this way | [Design notes](design.md) |

```{toctree}
:hidden:
:maxdepth: 2

operators
skip-connections
definitions
outputs
values
residual-algebras
modules
sweeps
tooling
grammar
design
```
