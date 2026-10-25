(lang-outputs)=
# Outputs and multi-token prediction

This page covers models with more than one output: every `head` is one, and `nextn`
lets a later head predict a token further ahead. Read it to write multi-token
prediction (MTP) as DeepSeek-V3, GLM-5, Nemotron 3 or MiMo-V2 do, and to know what
a model with several heads returns.

## Example: DeepSeek-V3's MTP module

```bux
layer(ffn) = [norm attn] [norm ffn]

shared out = head
mtp = $h nextn layer(experts) norm out

model = embed layer(dense)^3 layer(experts)^58 norm @h out mtp
```

This is the end of `examples/architectures/deepseek/deepseek_r1.bux`. Reading
`model` left to right:

1. `norm @h out`: the main model's final norm, then `@h` saves its hidden state, and
   `out` is the first head. It predicts the next token.
2. `mtp` starts from the saved state, `$h`, since nothing can follow a head.
3. `nextn` mixes that state with the embedding of the next token, and one more layer
   and a norm follow.
4. `out` again: the same head, since it is `shared`. It predicts the token after
   next.

## Heads end the stream

A `head` maps the stream to logits, so nothing can follow it but a tap read. The
compiler says so:

```text
error: nothing can follow the LM head
 --> model.bux:5:37
  |
5 | model = embed [norm attn] norm head norm
  |                                     ^^^^
  = hint: to go on after a head, read a tap: @h norm head $h ...
```

Save the stream before the head with `@h`, and go on from `$h` after it. Every head
is an output, so `embed block^4 norm @h head $h norm head` is a model with two.

## `nextn`: looking one token further

{bux}`nextn` mixes the stream with the next token's embedding:
`W [norm(e); norm(h)]`. Here `e` is what `embed` makes of the token (scaled and
normalized, without positions), so every head after a `nextn` predicts one token
further ahead.

The k-th `nextn` to run embeds the token k positions ahead. So a head predicts the
token `1 + k` positions ahead, where k counts the `nextn` layers run before it.

`nextn` reads the embedding matrix, so the model needs exactly one `embed`, as a
[tied head](definitions.md#tied-heads) does. Its `norm` setting is the norm on both
inputs (`rmsnorm` by default).

## Chains of MTP modules

A chain of modules, each reading the previous one's hidden state, is a repeat. Each
module pops the state before it and saves its own:

```bux
shared out = head
mtp = $h! nextn layer(experts) norm @h out
model = embed block^40 norm @h out mtp^3       # 4 outputs: 1, 2, 3 and 4 tokens ahead
```

The three ways to write the depths:

| Written | Each depth | Used by |
|---------|------------|---------|
| `mtp = $h! nextn ... @h out`, `mtp^3` | its own module, reading the previous depth's state | Inkling's eight chained MTP layers (with a macro, `mtp(mixer)`) |
| `shared mtp = $h! nextn ... @h out`, `mtp^3` | one module run at every depth, still one token further each time | GLM-5, Nemotron 3 |
| `mtp = $h nextn ... out`, `mtp^3` | its own module, every one reading the main model's state | MiMo-V2 |

With `shared`, one module runs at every depth, and each run still looks one token
further, since the k-th `nextn` to run embeds the token k ahead. With `$h` instead of
`$h!`, nothing is popped and every depth reads what the main model saved.

```bux
shared out = head
shared mtp = $h! nextn(norm=norm) layer(attn) norm @h out

model = embed "AAA" "SSS" "FSSS"^18 norm @h out mtp^3
```

That is GLM-5.3 (`examples/architectures/glm/glm_5.3.bux`): one MTP layer, trained
as three chained steps that share it.

## What a model with several heads returns

A backend returns the heads' logits in execution order. In the Flax backend,
{py:meth}`bux.architecture.model.Model.outputs` returns that list (or the graph's
output, if the model has no head), and calling the model returns the first head's
logits, the next-token prediction.

Training uses every head: the loss is the cross-entropy of the first head, plus
`mtp` times the mean loss of the heads after it (`loss(mtp=0.3)` by default; see
{bux}`loss` and [Training settings](../training/settings.md)).

## Under a residual algebra

When the [residual algebra](residual-algebras.md) widens the stream, `nextn` mixes
the next token's embedding into every copy of the stream, and widens it when its
input is one stream. Qwen3.8-Flash-Next continues its MTP layer from the widened
stream saved before the main head's read:

```bux
residual = gr(expansion=4, rank=320)

shared out = head
mtp = $h nextn(norm=norm) A out

model = embed D ple "DDA" "DDDA"^11 @h out mtp
```

The compiler puts a `merge` before each `out`, the read that turns the copies back
into one stream ([Outside the brackets](residual-algebras.md#outside-the-brackets)).
