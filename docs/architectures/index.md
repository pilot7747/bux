# Architectures

This guide covers the building blocks bux implements: what each layer computes, the
paper or model it comes from, how to write it in a `.bux` file, and the options that
matter. Read it when you design a model, transcribe one from a paper or a
`config.json`, or want to know exactly what a setting does before you sweep it.

Every field of every layer is in the generated
[language reference](../reference/language/layers.md); this guide links to an entry
there with its name, as in {bux}`gqa`, rather than repeating the tables.

## How a model is assembled

A model is a line of the [`.bux` language](../language/index.md): an embedding, blocks
of residual branches, and a head. Here is all of Llama 3 8B, from
`examples/architectures/llama3_8b.bux`:

```bux
# Llama 3 8B (Meta, 2024): the reference pre-norm decoder.
d_model    = 4096
vocab_size = 128256

norm = rmsnorm(eps=1e-5)
attn = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))
ffn  = gated_mlp(intermediate_size=14336, activation=swish)

block = [norm attn] [norm ffn]
model = embed block^32 norm head
```

`bux render` draws it:

```console
$ bux render examples/architectures/llama3_8b.bux
model
──embed──┤block├×32──norm──head──

block
──┬────────────⊕──┬───────────⊕──
  └─norm──attn─┘  └─norm──ffn─┘
```

Read the model left to right:

1. **`embed`** maps token ids to the residual stream, `d_model` wide
   ({bux}`embed`).
2. **Blocks** of residual branches change the stream. `[f]` is `x + f(x)`, so
   `[norm attn]` adds a normalized attention to the stream. A block usually has one
   **token mixer**, which moves information between positions, and one **channel
   mixer**, which transforms each position on its own.
3. **`head`** maps the stream to logits over the vocabulary ({bux}`head`). A model
   can have several heads, one for each output (multi-token prediction, see
   [Outputs](../language/outputs.md)).

| Role | What it does | Layers |
|------|--------------|--------|
| Token mixer | mixes positions, causally | [`gqa`, `mla`, `csa`](attention.md); [`gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`, `rwkv7`](recurrent.md); [`mamba`, `mamba2`, `conv`, `short_conv`](recurrent.md#state-space-models-and-convolutions) |
| Channel mixer | transforms each position | [`mlp`, `gated_mlp`](layers.md#mlps); [`moe`](experts.md) |
| Normalization | rescales each position | [`rmsnorm`, `layernorm`, `dyt`](layers.md#normalization) |
| Input and output | tokens in, logits out | [`embed`, `head`, `nextn`, `engram`](layers.md#embeddings-and-heads) |
| Plumbing | elementwise and structural | [`linear`, `act`, `scale`, `dropout`, `merge`](layers.md#elementwise-layers) |

Every layer keeps the width of the stream it reads, except three: `embed` maps token
ids to `d_model`, `linear` maps to its `features`, and `head` maps to logits. The
compiler checks that every residual branch comes back to the width it started at.

A setting that is not a layer parametrizes one: {bux}`rope` and its scalings go in an
attention's `rope=`, {bux}`indexer` in its `indexer=`, and the residual algebras
(`hc`, `mhc`, `gr`, `attnres`, `vpgate`) in the top-level `residual =`, which changes
what every `[f]` means ([Residual algebras](../language/residual-algebras.md)). The
top-level `init =` says how every weight matrix starts
([Initialization](layers.md#initialization)).

:::{note}
Rows can hold several packed documents. Given the positions of the tokens in their
documents, every layer sees each document on its own: attention masks the others,
convolutions and recurrent states start again at a document's start, and `nextn`
and `engram` never look across one. A document starts wherever the positions stop
increasing ({py:class}`bux.architecture.common.segments.Segments`).
:::

## Families of layers

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Softmax attention
:link: attention
:link-type: doc

`gqa` (MHA, GQA, MQA), `mla` and `csa`: windows, sinks, gates, QK-norm, RoPE and its
scalings, shared keys and values, and sparse attention by an indexer.
:::

:::{grid-item-card} Linear attention and state-space models
:link: recurrent
:link-type: doc

`gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`, `rwkv7`, `mamba`, `mamba2`, `conv`
and `short_conv`: the recurrence each computes, and hybrids with attention.
:::

:::{grid-item-card} Mixture of experts
:link: experts
:link-type: doc

`moe`: routers, top-k, shared experts, LatentMoE, and load balancing by a loss or a
bias.
:::

:::{grid-item-card} Norms, MLPs and plumbing
:link: layers
:link-type: doc

`rmsnorm`, `layernorm`, `dyt`, `mlp`, `gated_mlp`, `embed`, `head`, `nextn`,
`engram`, and the elementwise layers.
:::

:::{grid-item-card} Model zoo
:link: model-zoo
:link-type: doc

62 example architectures, from GPT-2 to 2026's releases, each a file to `use`
and shrink.
:::

:::{grid-item-card} The IR and backends
:link: ir-and-backends
:link-type: doc

What a `.bux` file compiles to, how the Flax NNX backend builds it, and how to write
another backend.
:::

:::{grid-item-card} Adding a layer
:link: adding-a-layer
:link-type: doc

The complete reference for extending the model: configs, builders, sharding names,
precision roles, remat names and kernels.
:::
::::

## Every layer, by page

| Layer | Page |
|-------|------|
| `gqa`, `mla`, `csa` | [Softmax attention](attention.md) |
| `gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`, `rwkv7` | [Linear attention](recurrent.md#linear-attention) |
| `mamba`, `mamba2`, `conv`, `short_conv` | [State-space models and convolutions](recurrent.md#state-space-models-and-convolutions) |
| `moe` | [Mixture of experts](experts.md) |
| `layernorm`, `rmsnorm`, `dyt` | [Normalization](layers.md#normalization) |
| `mlp`, `gated_mlp` | [MLPs](layers.md#mlps) |
| `embed`, `head`, `nextn` | [Embeddings and heads](layers.md#embeddings-and-heads) |
| `engram` | [Engram](layers.md#engram-hashed-n-gram-memory) |
| `linear`, `act`, `scale`, `dropout` | [Elementwise layers](layers.md#elementwise-layers) |
| `merge` | [Merge](layers.md#merge) |

```{toctree}
:hidden:
:maxdepth: 2

attention
recurrent
experts
layers
model-zoo
ir-and-backends
adding-a-layer
```
