# Norms, MLPs and plumbing

This page covers everything in a block besides its token mixer: norms, MLPs, the
embedding and the heads, multi-token prediction (`nextn`), Engram's n-gram memory, and
the elementwise layers you build other layers from. GPT-2 small uses most of them
(`examples/architectures/gpt2.bux`):

```bux
# GPT-2 small (OpenAI, 2019): LayerNorm with biases, learned positions, tied head.
d_model    = 768
vocab_size = 50257

norm = layernorm(eps=1e-5, bias=true)
attn = gqa(num_heads=12, num_kv_heads=12, bias=true, fused_qkv=true)
ffn  = mlp(intermediate_size=4 * d_model, activation=gelu_tanh, bias=true)

block = [norm attn] [norm ffn]
model = embed(max_positions=1024) block^12 norm head(tied=true)
```

## Normalization

### `rmsnorm` and `layernorm`

`rmsnorm` divides each position by its root mean square; `layernorm` subtracts the
mean first. Both multiply by a learned gain per channel ({bux}`rmsnorm`,
{bux}`layernorm`):

$$
\mathrm{rmsnorm}(x) = g \odot \frac{x}{\sqrt{\overline{x^2} + \epsilon}}, \qquad
\mathrm{layernorm}(x) = g \odot \frac{x - \bar x}{\sqrt{\overline{(x - \bar x)^2} + \epsilon}} + b.
$$

The statistics are in float32, whatever dtype the run computes in.

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `eps` | `1e-6` | $\epsilon$ | |
| `affine` | `true` | the gain $g$; `false` for none | nGPT, modded-nanogpt |
| `bias` | `false` | the bias $b$ (LayerNorm) | GPT-2 |
| `zero_centered` | `false` | the gain as $1 + w$, $w$ starting at 0 | Gemma, Qwen3-Next |
| `scale_input` | `false` | scale before normalizing, $\mathrm{norm}(g \odot x)$ | Muse Spark |
| `gain_init` | `1` | where the gain $g$ starts | Trinity |

Define a norm once and use it everywhere; each use gets its own parameters. Gemma 3
and Qwen3-Next use zero-centered gains:

```bux
norm = rmsnorm(zero_centered=true)
```

Muse Spark uses four norms (`examples/architectures/muse/muse_spark_open_816b_a42b.bux`):

```bux
norm    = rmsnorm(eps=1e-5, zero_centered=true)
bare    = rmsnorm(eps=1e-5, affine=false)
post    = rmsnorm(eps=1e-8, affine=false)
rescale = rmsnorm(eps=1e-8, zero_centered=true, scale_input=true)
```

A norm also goes in other layers' settings: `qk_norm=norm` in attention,
`latent_input_norm=norm` in `moe`, `norm=norm` in `nextn` and `engram`.

The algebra decides where the norms go ([Operators](../language/operators.md)):

| Placement | Written | Models |
|-----------|---------|--------|
| pre-norm | `[norm attn]` | Llama, most models |
| post-norm | `[attn] norm` | the original Transformer |
| sandwich | `[norm attn norm]` | Gemma 2 and 3 |
| output only | `[attn norm]` | OLMo 2 and 3 |

Trinity's sandwich norms are depth-scaled: each branch's output norm starts with a
gain of $1 / \sqrt{L}$ for $L$ layers, so a deep stack starts close to the embedding
(`examples/architectures/arcee/trinity_large.bux`):

```bux
layers = 60
norm   = rmsnorm(eps=1e-5)
post   = norm(gain_init=layers ^ -0.5)

layer(mixer, ffn=experts) = [norm mixer post] [norm ffn post]
```

### `dyt`: Dynamic Tanh

DyT (Zhu et al., 2025) replaces a norm with $\gamma \odot \tanh(\alpha x) + \beta$: a
learned scalar $\alpha$ (starting at `alpha_init`, 0.5), and a gain and a bias per
channel ({bux}`dyt`):

```bux
block = [dyt attn] [dyt ffn]
model = embed block^12 dyt head
```

Weight decay skips the norms' gains and biases, and DyT's parameters.

## MLPs

`mlp` and `gated_mlp` transform each position on its own ({bux}`mlp`,
{bux}`gated_mlp`). Llama 3's is a SwiGLU:

```bux
ffn = gated_mlp(intermediate_size=14336, activation=swish)
```

They compute

$$
\mathrm{mlp}(x) = W_d\, \phi(W_u x), \qquad
\mathrm{gated\_mlp}(x) = W_d \big(\phi(W_g x) \odot W_u x\big),
$$

with $\phi$ the `activation`. A `gated_mlp` with `swish` is SwiGLU (Shazeer, 2020);
with `gelu_tanh`, GeGLU.

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `intermediate_size` | required | the hidden units | |
| `activation` | required | $\phi$ | |
| `bias` | `false` | biases on every projection | GPT-2 |
| `dropout` | `0` | dropout on the output | |
| `limit` | `none` | clamped SwiGLU: the gate input capped at `limit`, the up branch at $\pm$`limit` | gpt-oss, DeepSeek-V4 |
| `up_softcap` | `none` | $c \tanh(u / c)$ on the up branch | Kimi K3 |
| `up_offset` | `0` | added to a gated MLP's up branch after it is clipped | gpt-oss ([its experts](experts.md#what-it-computes)) |

DeepSeek-V4.1-Flash clamps its experts at 10
(`examples/architectures/deepseek/deepseek_v4.1_flash.bux`):

```bux
expert = gated_mlp(intermediate_size=2304, activation=swish, limit=10)
```

Kimi K3's SiTU-GLU soft-caps the up branch (`examples/architectures/kimi/kimi_k3.bux`):

```bux
dense = gated_mlp(intermediate_size=33792, activation=situ, up_softcap=25)
```

The algebra can also spell an MLP out from primitives
(`examples/architectures/primitives.bux`):

```bux
swiglu = (linear(features=hidden) swish * linear(features=hidden)) linear(features=d_model)
```

### Activations

An activation is a value of `activation=`, and also a layer on its own:
`linear(features=h) swish` is `linear(features=h) act(fn=swish)`
([Activations](../reference/language/activations.md)).

| Name | Function |
|------|----------|
| `relu` | $\max(x, 0)$ |
| `relu2` | $\max(x, 0)^2$ (Primer, modded-nanogpt, Nemotron 3's experts) |
| `gelu` | GELU, exact |
| `gelu_tanh` | GELU, tanh approximation (GPT-2, Gemma) |
| `swish` | SiLU, $x \sigma(x)$ |
| `sigmoid` | $\sigma(x)$ |
| `tanh` | $\tanh(x)$ |
| `softplus` | $\ln(1 + e^x)$ |
| `situ` | $4 \tanh(x / 4)\, \sigma(x)$: SiLU with its linear factor soft-capped (Kimi K3) |
| `quick_gelu` | $x\, \sigma(1.702 x)$: GELU's sigmoid approximation (gpt-oss's gate) |

## Embeddings and heads

### `embed`

`embed` maps token ids to the stream, `d_model` wide, by a table of `vocab_size` rows
({bux}`embed`). It must come first.

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `max_positions` | `none` | adds learned absolute positions, this many | GPT-2 |
| `scale_by_sqrt_dim` | `false` | multiplies by $\sqrt{d}$ | Gemma |
| `norm` | `none` | a norm after the scale | Inkling |
| `dropout` | `0` | dropout on the output | |

Gemma 3 scales its embedding, and Inkling normalizes it:

```bux
model = embed(scale_by_sqrt_dim=true) "LLLLLG"^5 "LLLL" norm head(tied=true)
```

```bux
model = embed(norm=norm) "DD" "LLLG" "LLLLLG"^10 norm @h logits ...
```

With `max_positions`, the compiler checks that the run's `seq_len` fits, and those
of its curriculum's phases and of its evals. Under a
residual algebra that widens the stream (`hc`, `mhc`, `gr`, `attnres`), `embed` widens
its output ([Residual algebras](../language/residual-algebras.md)).

### `head`

`head` maps the stream to logits over the vocabulary ({bux}`head`).

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `tied` | `false` | reuse the first embedding's table: $x E^\top$ | GPT-2, Gemma |
| `bias` | `false` | a bias per token | GPT-J |
| `softcap` | `none` | $c \tanh(\ell / c)$ on the logits | Gemma 2, modded-nanogpt |

The logits are computed in the stream's dtype with float32 accumulation, then the
bias and the soft-cap are applied. By default the loss takes the logits a chunk of
positions at a time, so the logits of every position never exist at once
(`methods.loss`).

Every head is an output, and nothing can follow a head except a tap read. To scale
the logits, scale the stream before the head. modded-nanogpt soft-caps at 15, and
Inkling scales the stream by 1/24 (`examples/architectures/inkling/inkling.bux`):

```bux
norm head(softcap=15)        # 15 tanh(x / 15): the record's 30 sigmoid(x / 7.5) - 15
```

```bux
shared out = head
logits = (_ * logit_scale) out
```

Under `muon`, the embedding and the head train by its `adam`, never orthogonalized
([Optimizers](../training/optimizers.md)).

### `nextn`: multi-token prediction

`nextn` is DeepSeek-V3's multi-token prediction input: it mixes the stream with the
embedding of a token further ahead ({bux}`nextn`),

$$
h \mapsto W \big[\mathrm{norm}(e_{i+k});\ \mathrm{norm}(h)\big].
$$

For the $k$-th `nextn` to run in a forward pass, $e_{i+k}$ is what `embed` makes of
the token $k$ positions ahead (scaled and normalized, without positions), so every
head after it predicts one token further ahead. Its one setting is `norm`, an
`rmsnorm` by default, applied to both inputs.

An MTP module is `$h! nextn block norm @h head`, run after the main head. DeepSeek-R1's
shares the main head (`examples/architectures/deepseek/deepseek_r1.bux`):

```bux
shared out = head
mtp = $h nextn layer(experts) norm out

model = embed layer(dense)^3 layer(experts)^58 norm @h out mtp
```

A shared module run twice looks 1, then 2 tokens ahead (Nemotron 3, GLM-5). How heads,
taps and `nextn` make several outputs is in [Outputs](../language/outputs.md); the
loss weighs them by `loss(mtp=...)` ([Settings](../training/settings.md)).

On a widened stream, `nextn` mixes the embedding into each copy,
$W_e\,\mathrm{norm}(e) + W_h\,\mathrm{norm}(R)_j$, normalizing the copies together
(Qwen3.8-Flash-Next). On a single stream under a widening algebra, it widens its
output, as `embed` does.

## `engram`: hashed n-gram memory

Engram (DeepSeek, 2026) is conditional memory: hashed n-gram embeddings of the input
tokens, gated by the stream ({bux}`engram`). Qwen3.8-Flash-Next uses it as per-layer
embeddings (`examples/architectures/qwen/qwen3.8_flash_next.bux`):

```bux
ple = engram(head_dim=160, table_size=20000000, max_ngram_size=3, norm=norm,
             boundary_token=248044)

model = embed D ple "DDA" "DDDA"^11 @h out mtp
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `head_dim` | required | the width of a table row |
| `table_size` | required | rows per head: the next prime at or above it that no earlier head took |
| `max_ngram_size` | `3` | n-grams of 2 to this many tokens |
| `num_heads` | `8` | hash heads per n-gram size |
| `norm` | `rmsnorm` | of the query, the key and the conv's input |
| `conv_size` | `4` | the conv, zero-initialized and dilated by `max_ngram_size`; `none` turns it off |
| `vocab_classes` | `none` | hash normalized token classes, not raw ids |
| `boundary_token` | `none` | n-grams stop at this token, as at a document's end (Qwen) |

For each position:

1. Every n-gram of 2 to `max_ngram_size` tokens ending there is hashed by `num_heads`
   heads, each into its own table, and the rows are concatenated into $e$.
2. The stream $x$ gates a projection of $e$: with
   $a = \langle \mathrm{norm}(x), \mathrm{norm}(W_k e)\rangle / \sqrt{d}$,
   $u = \sigma\big(\mathrm{sign}(a) \sqrt{|a|}\big)\, W_v e$.
3. The layer adds $u + \mathrm{SiLU}(\mathrm{conv}(\mathrm{norm}(u)))$ to $x$.

Write `engram` bare, not in `[...]`: it adds to the stream itself, and `[engram]` is
an error. It reads the model's input tokens itself, as RoPE reads positions, and its
n-grams never cross a document. On a widened stream it adds to each copy, each gated
by its own key (DeepSeek-V4.1, Qwen3.8-Flash-Next).

:::{note}
`vocab_classes` maps token ids to classes by a stand-in, the id modulo the number of
classes; a checkpoint of the real model replaces it with the tokenizer's map. The
tables' rows split over devices as the vocabulary does, and weight decay skips them.
:::

## Elementwise layers

| Layer | What it does | Settings |
|-------|--------------|----------|
| {bux}`linear` | $W x$ (+ $b$), to `features` wide | `features`, `bias=false` |
| {bux}`act` | an activation, `act(fn=swish)`; or the activation's bare name | `fn` |
| {bux}`scale` | a learned multiplier: a scalar, or a vector with `per_channel=true` | `init=1`, `per_channel=false` |
| {bux}`dropout` | dropout | `rate` |

A number in a product scales by a constant: `f * 0.5`, `_ * 2.0`. With these, the
algebra writes the classic residual variants:

```bux
rezero     = [norm attn scale(init=0)]            # ReZero
layerscale = [norm attn scale(per_channel=true)]  # LayerScale
block = (scale + $x0 scale(init=0.0)) [norm attn] [norm ffn]  # x0 mixing (modded-nanogpt)
model = embed norm @x0 block^12 norm head
```

nGPT moves the stream toward each branch's normalized output by a learned step per
channel (`examples/architectures/ngpt.bux`):

```bux
unit = rmsnorm(affine=false)
step = scale(init=0.05, per_channel=true)

block = [(attn unit - _) step] unit [(ffn unit - _) step] unit
```

## `merge`

`merge` reads a widened residual stream (`hc`, `mhc`, `gr`, `attnres`) back into one:
the read the algebra does for a branch, without the write ({bux}`merge`). You rarely
write it: the compiler puts one before the first layer outside a branch that needs
a single stream, such as the final norm, and `bux fmt` prints it. See
[Residual algebras](../language/residual-algebras.md).

## Initialization

Every weight matrix starts from a normal of 0.02 unless the top-level `init` says
otherwise; the embedding tables, an untied head and the residual branches' outputs
can each start their own way. Norms' gains (`gain_init`), biases, short convolutions,
Mamba's decays and the residual algebras' connections keep their own start. See
[Initialization](initialization.md).
