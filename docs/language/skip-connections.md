# Skip connections

This page covers the three kinds of skip connection a `.bux` file draws: local
residual branches `[f]`, long-range taps that save the stream and read it back later
(`@name`, `$name`, `$name!`), and ports that pass a layer's internal tensors to
another layer (`attn(v=@v1)`). It ends with the rules the compiler checks for taps.

## Example: modded-nanogpt

```bux
d_model    = 768
vocab_size = 50304

norm = rmsnorm(affine=false)
attn = gqa(num_heads=6, num_kv_heads=6, head_dim=128, fused_qkv=true, qk_norm=norm,
           scale=0.12, rope=rope(base=1024, fraction=0.5))
ffn  = mlp(intermediate_size=4 * d_model, activation=relu2)

# x <- a*x + b*x0, then an ordinary pre-norm block
block = (scale + $x0 scale(init=0.0)) [norm attn] [norm ffn]

model = embed norm @x0
        (block @skip)^6              # first half: push every output onto `skip`
        ([$skip! scale] block)^6     # second half: pop them back in reverse, weighted
        norm head(softcap=15)
```

All three kinds of skip are here (`examples/architectures/modded_nanogpt.bux`). Each
`[...]` is a local residual. `@x0` saves the embedding, and every block mixes it back
in with `$x0`. `@skip` saves the output of each block of the first half, and `$skip!`
pops them in reverse order in the second half, a U-net:

```console
$ bux render examples/architectures/modded_nanogpt.bux
model
                    ╭─ ×6 ───────────╮  ╭─ ×6 ───────────────────────╮
──embed──norm──@x0──┤─┤block├──@skip─├──┤─┬───────────────⊕──┤block├─├──norm──head──
                    ╰────────────────╯  │ └─$skip!──scale─┘          │
                                        ╰────────────────────────────╯

block
────scale──────⊕──┬────────────⊕──┬───────────⊕──
    $x0──scale─┘  └─norm──attn─┘  └─norm──ffn─┘
```

A branch that starts from a tap is not connected to the fork: it ignores its input.

## Local: `[f]`

`[f]` is a residual branch: `x + f(x)`. Brackets nest, and a residual branch must
return the width it started at ([Widths](values.md#widths)).

| Written | Means |
|---------|-------|
| `[norm attn]` | `x + attn(norm(x))` |
| `[norm attn] [norm ffn]` | two branches in a row: a pre-norm block |
| `[norm (attn + ffn)]` | one branch around a parallel attention and MLP |
| `[norm attn scale(init=0)]` | ReZero: the branch starts switched off |

Under the plain residual stream `[f]` computes what `(_ + f)` does. The difference is
that `[f]` is the stream's own connection, so `residual = ...` can change what it
means ([Residual algebras](residual-algebras.md)) while `_ + f` stays a sum.

:::{note}
Square brackets with a comma make a list, not a branch: `[1, 2]`, and `[1,]` for one
element ([Lists](values.md#values)).
:::

(lang-taps)=
## Long-range: taps

| Syntax | What it does |
|--------|--------------|
| `@name` | pushes the stream onto the stack `name` and passes it on unchanged |
| `$name` | returns the top of the stack `name` and ignores its input |
| `$name!` | the same, and pops it off the stack |

Because `$name` ignores its input, it starts a branch, and the branch adds the saved
value back: `[$x0 scale]` is `x + w * x0`.

### x0 mixing

Every block mixes the stream with the embedding, `a * x + b * x0`, before its
residual branches:

```bux
blk = (scale + $x0 scale) [norm attn] [norm ffn]
model = embed @x0 blk^12 norm head
```

### U-net

Taps are stacks, and `$name!` pops. The mirror pairing of a U-net, the first block
with the last, falls out of last-in-first-out order, with no index arithmetic:

```bux
model = embed @x0 (block @skip)^6 ([$skip! scale] block)^6 norm head
```

The sixth `@skip` is popped by the first `$skip!`, and the first by the last.

### Going on after a head

A head ends the stream, so a model goes on to a second output by reading a tap:
`@h out $h nextn ...`. See [Outputs](outputs.md).

(lang-ports)=
## Internal tensors: ports

A port passes a tensor from inside one layer to another layer, through a tap. Give
the port a tap as its value:

- `attn(v=@v1)` exports the attention's values to the tap `v1`;
- `attn(v=$v1)` mixes the values on `v1` into its own (value residual, ResFormer);
- `attn(v=$v1!)` does the same and pops them.

```bux
d_model    = 768
vocab_size = 50277

norm = rmsnorm(eps=1e-5)
attn = gqa(num_heads=12, num_kv_heads=12, rope=rope())
ffn  = gated_mlp(intermediate_size=2688, activation=swish)

first = [norm attn(v=@v1)] [norm ffn]
rest  = [norm attn(v=$v1)] [norm ffn]
model = embed first rest^11 norm head
```

This is ResFormer (`examples/architectures/resformer.bux`): the first layer exports
its values and every later one mixes them in as `0.5 v + 0.5 v1`. The setting
`value_residual` of {bux}`gqa` chooses how the two are weighted.

| Port | Layers | Written | Read |
|------|--------|---------|------|
| `v` | {bux}`gqa`, {bux}`rwkv7` | the values | mixed into the layer's own values (ResFormer, RWKV-7's value residual) |
| `kv` | {bux}`gqa`, {bux}`csa` | the keys and values (for `csa`, the compressed latents with the indexer's keys) | used instead of the layer's own (YOCO, CLA) |
| `index` | {bux}`gqa`, {bux}`mla`, {bux}`csa` | the sparse indexer's top-k | attended instead of running an indexer of its own (GLM-5's IndexCache, DeepSeek-V4.1) |
| `candidates` | {bux}`csa` | coarse blocks of compressed positions | an indexer that reads them selects only inside them (DeepSeek-V4.1) |

Hymba shares keys and values between neighbouring sliding-window layers
(`examples/architectures/hymba.bux`): a layer that computes them exports them (`W`),
the layers after it read them (`S`), and the last of those pops them (`R`).

```bux
W = layer(local(kv=@kv))
S = layer(local(kv=$kv))
R = layer(local(kv=$kv!))
model = embed F "WR"^7 F "WSR" "WR"^6 F norm head(tied=true)
```

Rules for ports:

- A port's value must be a tap: `@t` writes it, `$t` reads it and `$t!` reads and
  pops it. A layer reads its ports before it writes any.
- Calling a definition with only ports keeps its name and its sharing: `attn(v=@v1)`
  is still `attn`, drawn as `attn(v=@v1)`, and a
  [shared](definitions.md#sharing-shared) layer can take ports, the same ones at
  every use, as it is one module.
- A config lists its ports in its class's `ports`; a new layer declares its own
  ([Adding a layer](../architectures/adding-a-layer.md)).

## How taps are checked

The compiler simulates every tap stack through the model, with repeats unrolled and
parallel branches run left to right. A mistake is an error at compile time, before
anything runs:

| Error | Cause |
|-------|-------|
| `tap 'x0' is read before anything writes it` | a `$x0` before any `@x0` |
| `tap 's' is empty: everything written to it was already popped` | more `$s!` than `@s` |
| `tap 'v1' holds the 'v' port of a layer` | a port's tap read as hidden states, `[$v1 scale]` |
| `port 'v' reads tap 'h', which holds hidden states` | a stream tap given to a port |
| `no port 'values' (ports: index, kv, v)` | a port the layer does not have |
| `tap 'h' holds the widened residual stream, ...` | see [Residual algebras](residual-algebras.md#outside-the-brackets) |

For example:

```text
error: tap 'x0' is read before anything writes it
 --> model.bux:5:16
  |
5 | model = embed [$x0 scale] @x0 norm head
  |                ^^^
  = hint: write it first with @x0
```

:::{note}
Taps and widths are checked along the model that is built: `model`, or the
definition `--entry` names. A definition that the model does not use is still
evaluated, so its settings are checked, but its taps are not simulated.
:::

### Tap names and scope

Tap names are global, so a block can leave a value behind that hides one its caller
wrote. The compiler catches this. A *block* here is a definition, a macro call, or
one pass of `^n`.

A value that a block both writes and reads is that block's own. If it is still on
the stack when the block ends, and something outside the block reads the tap while an
older value lies under it, that is an error:

```bux
mix(f) = @h [f] [$h scale]                # reads h but leaves it on the stack
model  = embed @h mix(attn) [$h! scale] norm head
```

```text
error: tap 'h' here reads the value that 'mix(attn)' wrote and read for itself, not the one written before it
 --> model.bux:7:30
  |
7 | model  = embed @h mix(attn) [$h! scale] norm head
  |                              ^^^
  = hint: pop it in 'mix(attn)' with $h! where it is read last, or give one of the two taps another name
```

Without the check, `[$h! scale]` would pop `mix`'s `h`, not the embedding it was
written for. There are two fixes:

1. Pop the value inside the block on its last read: `mix(f) = @h [f] [$h! scale]`.
2. Give one of the two taps another name.

Two common patterns are unaffected:

- **Exports**: a block writes a tap and leaves the reading to others, as `first`
  exports `v1` above, or `(block @skip)^6` its outputs.
- **Chains**: each block reads the previous block's value and writes its own, as a
  chain of multi-token prediction modules does (`$h! nextn ... @h out`).
