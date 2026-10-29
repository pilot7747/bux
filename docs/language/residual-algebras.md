(lang-residual)=
# Residual algebras

The top-level setting `residual` keeps a model's wiring and changes what every
residual branch `[f]` means. One line turns a pre-norm Transformer into one with
hyper-connections (`hc`, `mhc`), the gated residual (`gr`), attention residuals
(`attnres`, `msar`) or the variance-preserving gate (`vpgate`). This page lists the algebras
and their settings, then covers what happens outside the brackets when the stream is
widened, and the `merge` the compiler inserts there.

## Example: manifold-constrained hyper-connections

```bux
d_model    = 2048
vocab_size = 128256
residual   = mhc(expansion=4)

norm = rmsnorm
attn = gqa(num_heads=16, num_kv_heads=4, rope=rope())
ffn  = gated_mlp(intermediate_size=5632, activation=swish)

block = [norm attn] [norm ffn]
model = embed block^24 norm head
```

This is `examples/architectures/mhc.bux`: the usual pre-norm graph, plus one line.
`residual = mhc(expansion=4)` widens the stream into 4 streams. Every residual branch
reads them, writes them and mixes them through learned, doubly stochastic
connections. The wiring stays the same; only the meaning of `[...]` changes.

`bux fmt` shows what the compiler added, a `merge` before the final norm:

```console
$ bux fmt examples/architectures/mhc.bux | tail -1
model = embed block^24 merge norm head
```

## Why `[f]` and `_ + f` differ

Under the plain residual stream, `[f]` and `_ + f` compute the same thing, but the IR
keeps them apart. `[f]` is a `Residual` node, the residual stream's own connection,
so an algebra can replace it. `_ + f` is plain arithmetic that no algebra touches.
Write `[f]` for the residual branches of a model, and `_ + f` where you mean the sum
itself.

## The algebras

| Setting | `[f]` becomes |
|---------|---------------|
| none (the default) | `x + f(x)` |
| `hc(expansion=4)` | hyper-connections: `expansion` streams with learned read, write and mix connections around every branch (Zhu et al., 2025) |
| `mhc(expansion=4)` | the same, with the stream mixing projected onto doubly stochastic matrices by Sinkhorn (DeepSeek mHC, 2025) |
| `gr(rank=320)` | a gated read of 4 normalized streams, and a weighted write to each; branches need no pre-norm (Qwen3.8-Flash-Next) |
| `attnres(block_size=24)` | `f` reads a learned softmax over depth of the embedding and the earlier branch outputs, instead of their sum (Kimi K3) |
| `msar(block_size=8)` | `attnres` in 4 streams: each source as it is and its causal convs over the tokens of 4, 8 and 12 taps, a softmax over depth in each, mixed by sigmoid weights (Qin et al., 2026) |
| `vpgate(temperature=0.3)` | `a x + b f(x)` with `a^2 + b^2 = 1` per channel (Muse Spark) |

Their settings, with defaults:

| Setting | Of | Default | Meaning |
|---------|----|---------|---------|
| `expansion` | `hc`, `mhc`, `gr` | 4 | streams; 2 or more for `hc` and `mhc` |
| `dynamic` | `hc`, `mhc` | `true` | coefficients that depend on the stream, else static ones |
| `sinkhorn_iters` | `mhc` | 20 | Sinkhorn iterations |
| `single_pass` | `hc`, `mhc` | `false` | see below |
| `rank` | `gr` | required | rank of the read gate (Qwen3.8-Flash-Next: `d_model / 8`) |
| `block_size` | `attnres`, `msar` | `none` | branch outputs summed into one source; `none`: each is its own |
| `kernel_sizes` | `msar` | `[4, 8, 12]` | the taps of each stream's conv after the first: `msar` has one stream more |
| `norm_eps` | `hc`, `mhc`, `gr`, `attnres`, `msar` | 1e-6 | epsilon of the algebra's own norm |
| `temperature` | `vpgate` | 0.3 | `b = sigmoid(g / temperature)` |
| `eps` | `vpgate` | 1e-3 | `a = sqrt(max(1 - b^2, eps))`: the stream is never cut off |

Every algebra with its own norm takes `norm_eps`. See {bux}`hc`, {bux}`gr`,
{bux}`attnres`, {bux}`msar` and {bux}`vpgate` in the reference for the full
descriptions.

### Multi-stream attention residuals

`msar` (Qin et al., 2026) reads over depth as `attnres` does, in several streams. Each
source (the embedding, a finished block's sum of branch outputs, or the block under
way) is one stream as it is, and one more for each of its causal depthwise convs over
the tokens, of 4, 8 and 12 taps by default. Each branch reads a softmax over depth in
every stream, by a query of its own per stream. It then mixes the streams it read,
each by the weight $\sigma(a\,\mathrm{rmsnorm}(Z)\,W + b)$. The head reads the last
branch's input plus its output. On the factorial's candidate, in place of `hc`:

```bux
residual = msar(block_size=layers // 4)
```

The convs belong to the source: a finished block's are those of the branch that
read it first, and the block under way's those of the branch that reads it. They
start at 0, as do the queries and the mixer's projection, so a branch starts by
reading the mean of its sources, a quarter of it in each of the 4 streams.

It costs more than it gave at small scale. On a KDA hybrid of 680M parameters at a
width of 512, in place of `hc`, it raised the held-out bits per byte by 0.5% and took
2.1 times as long a step (503M tokens, 3 seeds, B300s, 2026-10-29). Its vLLM export
at a width of 3328 served 58% fewer agentic sessions an hour per GPU in FP8.

### Single-pass mHC

`mhc(single_pass=true)` is the variant from DeepSeek-V4.1. Each branch reads with the
coefficients that the branch before it computed (the first reads stream 0 alone),
and the read before the head has no parameters:

```bux
residual = mhc(expansion=4, single_pass=true, norm_eps=1e-20)
```

(lang-outside-brackets)=
## Outside the brackets

`hc`, `mhc`, `gr`, `attnres` and `msar` widen the stream. Between the branches, the
stream is several copies of itself (for `attnres` and `msar`, the outputs so far). Only a few things act
on it as a whole:

- `embed` widens it, and so does `nextn` when its input is one stream.
- Residual branches `[f]` read it, run `f` on one stream, and write back. Brackets
  inside `f` are plain skips.
- Taps `@h` and `$h` save and restore it as it is.
- `engram` adds its memory to every copy, gated by each copy's own key, and `nextn`
  mixes the embedding into every copy (Qwen3.8-Flash-Next). Under `attnres` and
  `msar`, whose state is the outputs so far rather than copies, both need one
  stream.

`vpgate` does not widen the stream, so none of this applies to it.

### `merge`

Any other layer outside a branch, such as the final norm or the head, needs one
stream. The compiler puts a {bux}`merge` before it: the read a branch does, without
the write. `bux fmt` prints it.

```bux
residual = gr(expansion=4, rank=320)

shared out = head
mtp = $h nextn(norm=norm) A out

model = embed D ple "DDA" "DDDA"^11 @h out mtp      # = ... @h merge out $h nextn ...
```

These lines are from Qwen3.8-Flash-Next
(`examples/architectures/qwen/qwen3.8_flash_next.bux`). `ple` is an `engram` that
adds to every copy, and `@h` saves the widened stream for the MTP layer. The compiler
puts a merge before each head:

```console
$ bux fmt examples/architectures/qwen/qwen3.8_flash_next.bux | tail -3
mtp = $h nextn(norm=rmsnorm(zero_centered=true)) A merge out

model = embed D ple D D A (D D D A)^11 @h merge out mtp
```

You can write `merge` yourself, to read the stream back into one where you choose.
Where the stream is not widened, a `merge` is an error.

`engram` is written bare, because it adds to the stream itself. `[engram]` is an
error.

## Errors

After a merge the copies are gone, so a residual branch after it is an error:

```text
error: a residual branch needs the widened residual stream, but it was read back into one before this
 --> model.bux:9:26
  |
9 | model = embed block norm [norm ffn] head
  |                          ^^^^^^^^^^
  = hint: only branches, taps, engram and nextn keep it; save it with @h and go on from $h
```

So is reading a tap that holds the widened stream inside a branch:

```text
error: tap 'h' holds the widened residual stream, which only exists outside residual branches
 --> model.bux:9:36
  |
9 | model = embed @h block [norm attn [$h scale]] norm head
  |                                    ^^
  = hint: read it back into one first: merge @h
```

The other errors:

| Error | Cause |
|-------|-------|
| `engram adds to the stream itself, so [engram] would add the stream twice` | `[engram]`; write it bare |
| `merge reads a widened residual stream back into one, but there is none here` | a `merge` without a widening `residual`, or after the stream was read back |
| `'residual' must be hc(...) or mhc(...) or gr(...) or attnres(...) or msar(...) or vpgate(...), found a rope config` | `residual` set to something else |

Under `attnres`, an `engram` between branches gets a merge before it, so the branch
after it is the first error above.
