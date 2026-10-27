(lang-outputs)=
# Outputs and multi-token prediction

A model can have several outputs. Every `head` is one, and `nextn` lets a later head
predict a token further ahead. That is all you need to write multi-token prediction
(MTP) as DeepSeek-V3, GLM-5, Nemotron 3 or MiMo-V2 do. This page also covers what a
model with several heads returns, and how training weighs their losses.

## Example: DeepSeek-V3's MTP module

```bux
layer(ffn) = [norm attn] [norm ffn]

shared out = head
mtp = $h nextn layer(experts) norm out

model = embed layer(dense)^3 layer(experts)^58 norm @h out mtp
```

This is the end of `examples/architectures/deepseek/deepseek_r1.bux`. Reading
`model` left to right:

1. `norm @h out`: the final norm of the main model, then `@h` saves its hidden state,
   and `out` is the first head. It predicts the next token.
2. `mtp` starts from the saved state, `$h`, because nothing can follow a head.
3. `nextn` mixes that state with the embedding of the next token. One more layer and
   a norm follow.
4. `out` again: the same head, because it is `shared`. It predicts the token after
   next.

## Heads end the stream

A `head` maps the stream to logits, so only a tap read can follow it. The compiler
says so:

```text
error: nothing can follow the LM head
 --> model.bux:5:37
  |
5 | model = embed [norm attn] norm head norm
  |                                     ^^^^
  = hint: to go on after a head, read a tap: @h norm head $h ...
```

Save the stream before the head with `@h`, and continue from `$h` after it. Every
head is an output, so `embed block^4 norm @h head $h norm head` has two.

## `nextn`: looking one token further

{bux}`nextn` mixes the stream with the embedding of a later token:
`W [norm(e); norm(h)]`. Here `e` is what `embed` makes of the token: scaled and
normalized, without positions.

The k-th `nextn` to run embeds the token k positions ahead. So a head predicts the
token `1 + k` positions ahead, where k counts the `nextn` layers that ran before it.

`nextn` reads the embedding matrix, so the model needs exactly one `embed`, as a
[tied head](definitions.md#tied-heads) does. Its `norm` setting is the norm on both
inputs (`rmsnorm` by default).

## Chains of MTP modules

A chain of modules, each reading the hidden state of the one before it, is a repeat.
Each module pops the state it reads and saves its own:

```bux
shared out = head
mtp = $h! nextn layer(experts) norm @h out
model = embed block^40 norm @h out mtp^3       # 4 outputs: 1, 2, 3 and 4 tokens ahead
```

There are three ways to write the depths:

| Written | Each depth | Used by |
|---------|------------|---------|
| `mtp = $h! nextn ... @h out`, `mtp^3` | its own module, reading the state of the depth before it | Inkling, with eight chained MTP layers (as a macro, `mtp(mixer)`) |
| `shared mtp = $h! nextn ... @h out`, `mtp^3` | one module run at every depth | GLM-5, Nemotron 3 |
| `mtp = $h nextn ... out`, `mtp^3` | its own module, every one reading the state of the main model | MiMo-V2 |

A shared module still looks one token further at each depth, because every run is
another `nextn`. With `$h` instead of `$h!`, nothing is popped, so every depth reads
what the main model saved.

GLM-5.3 (`examples/architectures/glm/glm_5.3.bux`) has one MTP layer, trained as
three chained steps that share it:

```bux
shared out = head
shared mtp = $h! nextn(norm=norm) layer(attn) norm @h out

model = embed "AAA" "SSS" "FSSS"^18 norm @h out mtp^3
```

## What a model with several heads returns

A backend returns the logits of every head, in the order the heads run. In the Flax
backend, {py:meth}`bux.architecture.model.Model.outputs` returns that list (or the
output of the graph, if the model has no head). Calling the model returns the logits
of the first head: the next-token prediction.

Training uses every head. The loss is the cross-entropy of the first head, plus `mtp`
times the mean loss of the heads after it: `loss(mtp=0.3)` by default. See
{bux}`loss` and [Training settings](../training/settings.md).

## Under a residual algebra

When the [residual algebra](residual-algebras.md) widens the stream, `nextn` mixes
the embedding into every copy of the stream, and widens the stream when its input is
one. Save the widened stream with `@h` before the main head, and the MTP layer
continues from it. The compiler puts a `merge` before each head, the read that turns
the copies back into one stream. Qwen3.8-Flash-Next is the worked example, in
[`merge`](residual-algebras.md#merge).
