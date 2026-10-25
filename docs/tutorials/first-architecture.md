# Your first architecture

In this tutorial you write a small GPT-2 from scratch in the `.bux` language, then
evolve it step by step: a Llama-style block with RoPE, grouped-query attention and
SwiGLU, SwiGLU written by hand, and a hybrid of Gated DeltaNet and attention. Along
the way you use `bux check`, `bux render` and `bux fmt`, and learn to read the errors.

Nothing here trains, so nothing imports JAX: every command answers in about a second.
The next tutorial, [Train on a laptop](train-on-a-laptop.md), trains what you build.

## 1. The smallest model

Create `mine/gpt.bux`:

```bux
d_model    = 256
vocab_size = 50257

attn  = gqa(num_heads=4, num_kv_heads=4)
model = embed [attn] head
```

Every file sets `d_model`, the width of the residual stream, and defines `model`.
`attn = gqa(...)` defines a layer: grouped-query attention, with as many key and
value heads as query heads, which is plain multi-head attention. `model` reads left to
right: embed the tokens, then `[attn]`, then the head that turns the stream into
logits. The brackets are the residual connection: `[f]` means `x + f(x)`.

Check it, and draw it:

```console
$ bux check mine/gpt.bux
d_model=256 vocab_size=50257 layers=3
embed×1  gqa×1  head×1
$ bux render mine/gpt.bux
model
──embed──┬──────⊕──head──
         └─attn─┘
```

`bux check` compiles the file and counts its layers by type. `bux render` draws the
stream as a line: `[attn]` leaves it, runs `attn`, and adds back at `⊕`.

## 2. A GPT-2 block

GPT-2's block normalizes before each branch (pre-norm), with a LayerNorm that has a
bias, and follows attention with an MLP. Replace the file with the whole model:

```bux
# A small GPT-2: learned positions, LayerNorm with biases, a GELU MLP, a tied head.
d_model    = 256
vocab_size = 50257

norm = layernorm(eps=1e-5, bias=true)
attn = gqa(num_heads=4, num_kv_heads=4, bias=true)
ffn  = mlp(intermediate_size=4 * d_model, activation=gelu_tanh, bias=true)

block = [norm attn] [norm ffn]
model = embed(max_positions=1024) block^4 norm head(tied=true)
```

What is new:

- `norm`, `attn` and `ffn` are definitions. Each use of a definition is a fresh
  layer with parameters of its own, so the two `norm`s in `block` are two LayerNorms.
- Settings are arithmetic over definitions: `4 * d_model` is 1024.
- `block = [norm attn] [norm ffn]` is two residual branches in a row: a pre-norm
  Transformer block.
- `block^4` is four fresh copies of `block` in a row.
- `embed(max_positions=1024)` adds learned positions, as GPT-2 does; with none, the
  model has no positions but what its attention's `rope` gives.
- `head(tied=true)` reuses the embedding matrix as the output projection.

```console
$ bux check mine/gpt.bux
d_model=256 vocab_size=50257 layers=19
layernorm×9  gqa×4  mlp×4  embed×1  head×1
$ bux render mine/gpt.bux
model
──embed──┤block├×4──norm──head──

block
──┬────────────⊕──┬───────────⊕──
  └─norm──attn─┘  └─norm──ffn─┘
```

`bux render` draws each block once and refers to it by name. `--expand` draws it
inline, inside a box for the repeat:

```console
$ bux render mine/gpt.bux --expand
model
         ╭─ ×4 ──────────────────────────╮
──embed──┤─┬────────────⊕──┬───────────⊕─├──norm──head──
         │ └─norm──attn─┘  └─norm──ffn─┘ │
         ╰───────────────────────────────╯
```

Every layer's settings, with their types and defaults, are in the
[layer reference](../reference/language/layers.md): {bux}`layernorm`, {bux}`gqa`,
{bux}`mlp`, {bux}`embed` and {bux}`head` here.

:::{tip}
`examples/architectures/gpt2.bux` is GPT-2 small at its real size: 12 heads, 12
blocks, `d_model = 768`, and the queries, keys and values in one projection
(`fused_qkv=true`).
:::

## 3. Move the norms

Where the norm goes is the wiring, and the wiring is the expression. The original
Transformer normalizes after each residual add (post-norm). Try it from the command
line, without editing the file: statements after the file redefine the file's own.

```console
$ bux render mine/gpt.bux 'block = [attn] norm [ffn] norm'
model
──embed──┤block├×4──norm──head──

block
──┬──────⊕──norm──┬─────⊕──norm──
  └─attn─┘        └─ffn─┘
```

Other blocks are one line each, for example:

```bux
block = [norm attn norm] [norm ffn norm]      # sandwich norm (Gemma 2 and 3)
block = [norm (attn + ffn)]                   # a parallel block (GPT-J, PaLM)
block = [norm attn scale(init=0)] [norm ffn scale(init=0)]   # ReZero
```

[Operators](../language/operators.md) lists every operator and the shapes they make.

## 4. A Llama-style model

Llama keeps GPT-2's pre-norm wiring and changes the layers: RMSNorm, rotary position
embeddings, fewer key and value heads than query heads (GQA), and a gated MLP with a
SiLU (SwiGLU). Create `mine/llama.bux`:

```bux
# The same size, Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU.
d_model    = 256
vocab_size = 50257

norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn  = gated_mlp(intermediate_size=704, activation=swish)

block = [norm attn] [norm ffn]
model = embed block^4 norm head(tied=true)
```

- `rmsnorm` with no call takes every default.
- `rope=rope()` gives the attention rotary embeddings; `rope(base=500000)` would
  change their base. There is no `max_positions` on `embed` now.
- `num_kv_heads=2` shares each key and value head between two query heads.
- `gated_mlp(..., activation=swish)` is SwiGLU. Its hidden width, 704, is about
  8/3 of `d_model`, rounded up to a multiple of 64, so that its three projections have
  about as many parameters as the two of a 4x MLP.

```console
$ bux check mine/llama.bux
d_model=256 vocab_size=50257 layers=19
rmsnorm×9  gqa×4  gated_mlp×4  embed×1  head×1
```

## 5. SwiGLU by hand

`gated_mlp` is a layer, but you can also write it from smaller ones. SwiGLU is
`W2 (swish(W1 x) * W3 x)`: two projections side by side, one through a SiLU,
multiplied, then projected back. In the language, `*` multiplies two parallel
branches and composition binds tighter than `*`, so it reads as written. In
`mine/llama.bux`, replace `ffn`:

```bux
ffn = (linear(features=704) swish * linear(features=704)) linear(features=d_model)
```

```console
$ bux check mine/llama.bux
d_model=256 vocab_size=50257 layers=31
linear×12  rmsnorm×9  gqa×4  act×4  embed×1  head×1
$ bux render mine/llama.bux
model
──embed──┤block├×4──norm──head──

block
──┬────────────⊕──┬─────────────⊕──
  └─norm──attn─┘  └─norm──┤ffn├─┘

ffn
──┬─linear──swish─⊗──linear──
  └─linear────────┘
```

A `linear` maps to `features` wide, and the compiler follows every width: the two
branches of `*` must agree, and a residual branch must come back to `d_model`, which
the last `linear` does. Put `ffn = gated_mlp(...)` back before you go on.

## 6. A hybrid pattern

Recent models mix attention with linear-attention layers, which carry a state along
the sequence instead of attending to every token. Qwen3-Next, for example, has three
Gated DeltaNet layers to one attention layer. Create `mine/hybrid.bux`:

```bux
# Two Gated DeltaNet layers to one attention layer, as Qwen3-Next mixes them.
d_model    = 256
vocab_size = 50257

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
delta = gdn(num_heads=4, head_dim=64)
ffn   = gated_mlp(intermediate_size=704, activation=swish)

layer(mixer) = [norm mixer] [norm ffn]
D = layer(delta)
A = layer(attn)
model = embed "DDA"^2 norm head(tied=true)
```

- `layer(mixer) = ...` is a **macro**: a definition with a parameter. `layer(delta)`
  is the block with `delta` as its mixer.
- `"DDA"` is a **pattern**: the sequence `D D A` of one-letter definitions. `^2`
  repeats it, so the model has six blocks.

```console
$ bux check mine/hybrid.bux
d_model=256 vocab_size=50257 layers=27
rmsnorm×13  gated_mlp×6  gdn×4  gqa×2  embed×1  head×1
$ bux render mine/hybrid.bux
model
         ╭─ ×2 ──────────╮
──embed──┤─┤D├──┤D├──┤A├─├──norm──head──
         ╰───────────────╯

D
──┬─────────────⊕──┬───────────⊕──
  └─norm──delta─┘  └─norm──ffn─┘

A
──┬────────────⊕──┬───────────⊕──
  └─norm──attn─┘  └─norm──ffn─┘
```

`bux fmt` prints the canonical source of what compiled, not your file: macros and
numeric definitions inlined, patterns spelled out, comments dropped. It is a quick way
to see what a macro or a pattern expanded to:

```console
$ bux fmt mine/hybrid.bux
d_model = 256
vocab_size = 50257

norm = rmsnorm
delta = gdn(num_heads=4, head_dim=64)
ffn = gated_mlp(intermediate_size=704, activation=swish)
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())

D = [norm delta] [norm ffn]
A = [norm attn] [norm ffn]

model = embed (D D A)^2 norm head(tied=true)
```

Patterns repeat and nest like anything else: `"mMmM aMmM"^4` is Jamba's 32 layers,
spaces ignored. [Definitions](../language/definitions.md) covers macros, patterns and
`shared` blocks, whose every use has the same parameters.

## 7. Read the errors

bux checks everything before anything runs and reports every error it finds, each at
the line that made it. Create `mine/broken.bux` with three mistakes:

```bux
d_model    = 256
vocab_size = 50257

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_head=2, rope=rope())
ffn   = gated_mlp(intermediate_size=4 * dmodel, activation=swish)
up    = linear(features=512)

block = [norm attn] [norm ffn] [norm up]
model = embed block^4 norm head(tied=true)
```

```console
$ bux check mine/broken.bux
error: gqa has no setting 'num_kv_head'
 --> mine/broken.bux:5:26
  |
5 | attn  = gqa(num_heads=4, num_kv_head=2, rope=rope())
  |                          ^^^^^^^^^^^^^
  = hint: did you mean 'num_kv_heads'?

error: unknown name 'dmodel'
 --> mine/broken.bux:6:41
  |
6 | ffn   = gated_mlp(intermediate_size=4 * dmodel, activation=swish)
  |                                         ^^^^^^
  = hint: did you mean 'd_model'?
```

Each error names the file, line and column, underlines the span, and suggests the
nearest name it knows. The settings come from the layers' schemas, so a misspelled
setting is caught for every layer. Fix the two (`num_kv_heads`, `d_model`) and check
again; the third error was hidden behind the definitions that failed:

```console
$ bux check mine/broken.bux
error: a residual branch must keep the width, but this one maps 256 to 512
 --> mine/broken.bux:9:32
  |
9 | block = [norm attn] [norm ffn] [norm up]
  |                                ^^^^^^^^^
```

A few more you will meet:

```console
$ bux check mine/llama.bux 'block = [norm attn] * 3'
error: multiplying a block by the integer 3 is ambiguous
 --> <command line>:1:23
  |
1 | block = [norm attn] * 3
  |                       ^
  = hint: write blk^3 to repeat it, or 3.0 to scale its output
$ bux check mine/llama.bux 'model = embed block^4 norm head norm'
error: nothing can follow the LM head
 --> <command line>:1:33
  |
1 | model = embed block^4 norm head norm
  |                                 ^^^^
  = hint: to go on after a head, read a tap: @h norm head $h ...
```

:::{tip}
`bux-lsp` shows the same errors in your editor as you type, completes every layer's
settings and documents them on hover ([Tooling](../language/tooling.md)).
:::

## What you built

| file              | what it is                                              |
|-------------------|---------------------------------------------------------|
| `mine/gpt.bux`    | a small GPT-2: pre-norm, LayerNorm, learned positions, a GELU MLP |
| `mine/llama.bux`  | the same size, Llama-style: RMSNorm, RoPE, GQA, SwiGLU  |
| `mine/hybrid.bux` | two Gated DeltaNet layers to one attention, by a macro and a pattern |

Next, [train one on a laptop](train-on-a-laptop.md). To go further with the
language:

- [Skip connections](../language/skip-connections.md): taps for long-range skips,
  U-nets and value residuals.
- [Several outputs](../language/outputs.md): multi-token prediction heads.
- [Architectures](../architectures/index.md): every attention, recurrent and expert
  layer bux has, and a zoo of transcribed models.
