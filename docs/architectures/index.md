# Architectures

This section shows how to build a model in bux: what each layer computes, the paper or
model it comes from, how to write it in a `.bux` file, and the settings that matter.
Come here to design a model, to transcribe one from a paper or a `config.json`, or to
learn exactly what a setting does before you sweep it.

To reproduce a published model, start from the [model zoo](model-zoo.md): 62 of them,
each a file you can `use` and change. Every field of every layer is in the generated
[language reference](../reference/language/layers.md); these pages link to its entries
by name, as in {bux}`gqa`, instead of repeating its tables.

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

Every layer bux has, by role and page:

| Role | What it does | Layers |
|------|--------------|--------|
| Token mixer | mixes positions, causally | [`gqa`, `mla`, `csa`](attention.md); [`gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`, `rwkv7`](recurrent.md#linear-attention); [`mamba`, `mamba2`, `conv`, `short_conv`](recurrent.md#state-space-models-and-convolutions) |
| Channel mixer | transforms each position | [`mlp`, `gated_mlp`](layers.md#mlps); [`moe`](experts.md) |
| Normalization | rescales each position | [`rmsnorm`, `layernorm`, `dyt`](layers.md#normalization) |
| Input and output | tokens in, logits out | [`embed`, `head`, `nextn`](layers.md#embeddings-and-heads); [`engram`](layers.md#engram-hashed-n-gram-memory) |
| Plumbing | elementwise and structural | [`linear`, `act`, `scale`, `dropout`](layers.md#elementwise-layers); [`merge`](layers.md#merge) |

Every layer keeps the width of the stream it reads, except three: `embed` maps token
ids to `d_model`, `linear` maps to its `features`, and `head` maps to logits. (Under a
residual algebra that widens the stream, `merge` and `nextn` also move between one
stream and several.) The compiler checks that every residual branch comes back to the
width it started at.

A setting that is not a layer parametrizes one: {bux}`rope` and its scalings go in an
attention's `rope=`, and {bux}`indexer` in its `indexer=`. The top-level
`residual =` picks a residual algebra (`hc`, `mhc`, `gr`, `attnres`, `vpgate`), which
changes what every `[f]` means ([Residual algebras](../language/residual-algebras.md)).
The top-level `init =` says how the weight matrices start
([Initialization](initialization.md)).

:::{note}
A row can hold several packed documents, and every layer sees each one on its own:
attention masks the others, convolutions and recurrent states restart at each
document's start, and `nextn` and `engram` never look across a boundary. A document
starts wherever the token positions stop increasing
({py:class}`bux.architecture.common.segments.Segments`).
:::

## The pages

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

`gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`, `rwkv7`, `mamba`, `mamba2`,
`conv` and `short_conv`: the recurrence each computes, and hybrids with attention.
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
`engram`, `merge` and the elementwise layers.
:::

:::{grid-item-card} Initialization
:link: initialization
:link-type: doc

The default normal of 0.02, the top-level `init` that changes it by role, recipes
from GPT-2 to Mamba, and what the papers found.
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

What a `.bux` file compiles to, how to generate and print architectures from Python,
how the Flax NNX backend builds a model, and how to write another backend.
:::

:::{grid-item-card} Adding a layer
:link: adding-a-layer
:link-type: doc

The reference for extending the model: configs, builders, sharding names, precision
roles, remat names, kernels, export and tests.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

attention
recurrent
experts
layers
initialization
model-zoo
ir-and-backends
adding-a-layer
```
