# Operators

Operators combine layers into blocks: composition, parallel branches that are summed
or multiplied, scaling, negation, repetition and the identity. With them, SwiGLU, a
gated attention or nGPT's update is one line. Read this page when a block is more
than `[norm attn] [norm ffn]`. It ends with how expressions parse and a catalogue of
common block shapes.

## Example: SwiGLU and a gated attention from primitives

```bux
d_model    = 512
vocab_size = 32000
hidden     = 1408

swiglu     = (linear(features=hidden) swish * linear(features=hidden)) linear(features=d_model)
gated_attn = gqa(num_heads=8, num_kv_heads=8, rope=rope()) * linear(features=d_model) sigmoid

block = [rmsnorm gated_attn] [rmsnorm swiglu]
model = embed block^8 rmsnorm head(tied=true)
```

This is `examples/architectures/primitives.bux`. `swiglu` runs two projections on the
same input, multiplies them (one through `swish`), and projects the product back to
`d_model`. Composition binds tighter than `*`, so `gated_attn` multiplies the
attention by `sigmoid(linear(x))`. `bux render` draws each `*` as a fork that merges
at `⊗`:

```text
swiglu
──┬─linear──swish─⊗──linear──
  └─linear────────┘

gated_attn
──┬─gqa─────────────⊗──
  └─linear──sigmoid─┘
```

## The operators

Loosest first:

| Syntax | Meaning |
|--------|---------|
| `f \| g`, `a: f \| b: g` | alternatives: a run for each, a sweep ([Sweeps](sweeps.md)) |
| `f + g`, `f - g` | parallel branches on the same input, summed: `f(x) + g(x)`, `f(x) - g(x)` |
| `f * g` | parallel branches multiplied elementwise (gating): `f(x) * g(x)` |
| `f * 0.5` | `f` scaled by a number |
| `f g`, `f -> g`, `f → g` | composition, left to right: `g(f(x))` |
| `-f` | negation: `-f(x)` |
| `f^n` | `n` fresh copies of `f` in a row (composition power) |
| `[f]` | residual branch: `x + f(x)` ([Skip connections](skip-connections.md)) |
| `_` | identity: `x`, so `[f]` behaves like `(_ + f)` |
| `(f)` | grouping |
| `@name` | tap: push the stream onto the stack `name` ([taps](skip-connections.md#long-range-taps)) |
| `$name`, `$name!` | the top of the tap `name`, ignoring the input; `!` pops it |
| `"MMMA"` | pattern: the sequence `M M M A` of one-letter definitions ([patterns](definitions.md#patterns)) |

On numbers, `+`, `-`, `*`, `/`, `//` and `^` are arithmetic
([Settings and values](values.md#arithmetic)).

## Composition: `f g`

Writing layers one after another applies them left to right: `norm attn` is
`attn(norm(x))`. The arrows `->` and `→` mean the same thing. They are optional; use
them where they read better: `norm -> attn`.

A composition may span lines. A statement continues while the next line is indented
or a bracket is still open:

```bux
model = embed norm @x0
        (block @skip)^6
        ([$skip! scale] block)^6
        norm head
```

## Parallel branches: `+`, `-` and `*`

`f + g` runs `f` and `g` on the same input and sums their outputs. `f - g` subtracts
the second, and `f * g` multiplies them elementwise: that is how you write a gate.
Combine as many branches as you like, `f + g + h`. They run left to right, which
matters only when they read or write taps.

Every branch must return the same width ([Widths](values.md#widths)). `bux render`
draws a sum as a fork merging at `⊕` and a product as one merging at `⊗`.

A number in a sum or a product is a constant:

- `f * 0.5` scales the output of `f`;
- `_ * 2.0 + attn` weights the skip around `attn` by 2 (DeepNorm-style).

For a learned weight instead of a fixed one, compose with {bux}`scale`, a learned
multiplier: `f scale`, or `f scale(per_channel=true)` for one weight per channel.

`/` and `//` only divide numbers. To scale a branch, multiply it:

```text
error: '/' only divides numbers
 --> model.bux:5:9
  |
5 | half  = attn / 2.0
  |         ^^^^^^^^^^
  = hint: scale a branch with `f * 0.5`
```

### `f * 3` is an error

Multiplying a block by an integer is ambiguous: you may have meant `f^3`, three
copies in a row, or `f * 3.0`, its output scaled by 3. The compiler asks:

```text
error: multiplying a block by the integer 3 is ambiguous
 --> model.bux:6:16
  |
6 | three = attn * 3
  |                ^
  = hint: write blk^3 to repeat it, or 3.0 to scale its output
```

The same holds for `3 * attn`. Write `3.0` (a float) to scale.

## Negation: `-f`

`-f` negates the output of `f`, and `f - g` is `f + (-g)`. nGPT moves the stream
toward the normalized output of each branch, `h <- norm(h + a * (norm(f(h)) - h))`.
In `.bux` that is a subtraction of the identity:

```bux
unit  = rmsnorm(affine=false)
step  = scale(init=0.05, per_channel=true)
block = [(attn unit - _) step] unit [(ffn unit - _) step] unit
```

:::{warning}
A `-` between two items is a subtraction, even without a space after it: `f -g` is
`f - g`. To compose with a negated layer, group it: `f (-g)`.
:::

## Repetition: `f^n`

`f^n` is `n` copies of `f` in a row, each with its own parameters (unless `f` is
[`shared`](definitions.md#sharing-shared)). `block^32` is 32 blocks. The count is a
whole number: `f^1` is `f`, and `f^0` is nothing, left out of the sequence it is in
(alone, it is `_`). So a count computed from the width may come out 0, as the readers
left over after whole groups do: `(F S^3)^(n // 4) S^(n - n // 4 * 4)`.

```bux
M = [norm ssm]
model = embed (M^5 block)^6 norm head      # 6 × (5 Mamba layers, then a block)
```

A repeated block must return the width it started at. The IR keeps a repeat as one
node rather than `n` copies, so a backend can scan over the identical blocks. The
compiler still checks every pass, which matters for [taps](skip-connections.md).

On numbers `^` is a power: `2^10` is 1024.

## Identity: `_`

`_` returns its input. Under the plain residual stream, `[f]` and `(_ + f)` compute
the same thing, but only `[f]` changes under a
[residual algebra](residual-algebras.md). Write `[f]` for a residual branch and
`_ + f` for the sum itself.

## Precedence and associativity

From loosest to tightest:

| Level | Operators | Groups |
|-------|-----------|--------|
| alternatives | `\|` | all alternatives of one `\|` chain are one axis |
| sum | `+`, `-` | left to right: `f - g - h` is `(f - g) - h` |
| product | `*`, `/`, `//` | left to right |
| composition | `f g`, `->`, `→` | one sequence: `f g h` |
| negation | `-f` | prefix |
| power | `^` | right to left: `2^3^2` is `2^(3^2)` |
| atoms | names, calls, numbers, strings, `_`, taps, `( )`, `[ ]` | |

Composition binds tighter than `*` and `+`, as concatenation does in regular
expressions. So `norm attn + ffn` means `(norm attn) + ffn`, and the gate `a b * c d`
multiplies two compositions. Some worked parses:

| Written | Parses as |
|---------|-----------|
| `norm attn + ffn` | `(norm attn) + ffn` |
| `norm (attn + ffn)` | `norm`, then the sum `attn + ffn` |
| `a b * c d` | `(a b) * (c d)` |
| `f * g h` | `f * (g h)` |
| `a + b * c` | `a + (b * c)` |
| `f * 0.5 + g` | `(f * 0.5) + g` |
| `-f g` | `(-f) g` |
| `f -g` | `f - g` |
| `f g^2` | `f (g^2)`: `f g g` |
| `(f g)^2` | `f g f g` |
| `-f^2` | `-(f^2)` |
| `f \| g h` | `f \| (g h)`: two alternatives |
| `a: f g \| b: h` | the label `a` names all of `f g` |
| `2^3^2` | `2^(3^2)`, which is 512 |

## Common shapes

```bux
[norm attn] [norm ffn]                       # pre-norm (Llama, GPT-2)
[attn] norm [ffn] norm                       # post-norm (original Transformer)
[norm attn norm] [norm ffn norm]             # sandwich norm (Gemma 2/3)
[attn norm] [ffn norm]                       # each branch's output normed (OLMo 2/3)
[norm (attn + ffn)]                          # parallel block (GPT-J, PaLM)
[norm (attn norm scale + ssm norm scale)]    # hybrid heads (Hymba)
[norm attn scale(init=0)]                    # ReZero; scale(per_channel=true) for LayerScale
(_ * 2.0 + attn) norm                        # DeepNorm-style weighted skip
[(attn unit - _) step] unit                  # nGPT: h <- norm(h + a * (norm(f(h)) - h))
gqa(gated=true)                              # gated attention, before o_proj (Qiu et al. G1)
gqa(...) * (linear(features=d_model) sigmoid) # gated after o_proj instead (their G5)
```

Here `attn`, `ffn` and `ssm` are definitions such as `gqa(...)`, `gated_mlp(...)`
and `mamba(...)`, and `unit` and `step` are the nGPT definitions above. Four of them,
as `bux render` draws them:

::::{tab-set}

:::{tab-item} Parallel block
```text
──┬────────────────⊕──
  └─norm──┬─attn─⊕─┘
          └─ffn──┘
```
:::

:::{tab-item} Hybrid heads
```text
──┬─────────────────────────────⊕──
  └─norm──┬─attn──norm──scale─⊕─┘
          └─ssm──norm──scale──┘
```
:::

:::{tab-item} DeepNorm
```text
──┬───×2─⊕──norm──
  └─attn─┘
```
:::

:::{tab-item} nGPT
```text
──┬──────────────────────⊕──unit──
  └─┬─attn──unit─⊕──step─┘
    └─−──────────┘
```
:::

::::
