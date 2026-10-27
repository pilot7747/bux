# Softmax attention

This page covers bux's softmax attention layers, `gqa`, `mla` and `csa`, the options
they share (windows, sinks, gates, norms, scales), positional encodings, sharing keys
and values between layers, and sparse attention by an indexer. Read it to configure
the attention of a model, or to find which setting reproduces a paper's variant.

```bux
# Qwen3-30B-A3B: 32 query heads over 4 kv heads of 128, QK-norm, RoPE at a base of 1M
norm = rmsnorm(eps=1e-6)
attn = gqa(num_heads=32, num_kv_heads=4, head_dim=128, qk_norm=norm,
           rope=rope(base=1000000))
```

All three layers are causal: a query attends to its own position and the ones before
it, within its document. Each has an entry in the reference with every field:
{bux}`gqa`, {bux}`mla`, {bux}`csa`.

| Layer | What it is | From |
|-------|-----------|------|
| [`gqa`](#gqa-multi-head-grouped-query-and-multi-query-attention) | multi-head attention with fewer key and value heads than query heads | MHA (Vaswani et al., 2017), GQA (Ainslie et al., 2023), MQA (Shazeer, 2019) |
| [`mla`](#mla-multi-head-latent-attention) | keys and values projected up from one low-rank latent a token | DeepSeek-V2/V3, Kimi K2, Kimi Linear |
| [`csa`](#csa-compressed-sparse-attention) | a sliding window of per-token latents plus a sparse pick of compressed ones | DeepSeek-V4 |

## `gqa`: multi-head, grouped-query and multi-query attention

`gqa(num_heads=h, num_kv_heads=g)` projects the input to `h` query heads and `g` key
and value heads, and each kv head serves `h / g` query heads:

- **MHA**: `num_kv_heads == num_heads`.
- **GQA**: `1 < num_kv_heads < num_heads`.
- **MQA**: `num_kv_heads = 1`.

One sweep compares all three (abridged from
`examples/experiments/attention_ablation.bux`):

```bux
use llama3_8b

d_model = 768
attn.num_heads    = 12
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
model = embed block^12 norm head(tied=true)
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `head_dim` | `width // num_heads` | the query and key head width |
| `v_head_dim` | `head_dim` | the value head width (MiMo-V2: 128 against 192) |
| `bias` | `false` | biases on the input projections: q, k, v and the gate (GPT-2, Qwen2) |
| `output_bias` | `bias` | a bias on the output projection (GPT-2 has one, Qwen2 has none) |
| `fused_qkv` | `false` | make q, k and v with one projection and split it: the same function and init, held as one matrix (GPT-2, MiMo-V2.6-Pro) |
| `qk_norm_all_heads` | `false` | `qk_norm` of all the heads together, not of each ([below](#qk-norm-and-output-norms)) (MiniMax-M2, OLMo 2) |
| `rope` | `none` | rotary position embeddings ([below](#rotary-position-embeddings-rope)) |
| `power_scale`, `power_scale_length` | `none`, `2048` | queries scaled up with a power of the position ([below](#power-scaling-at-long-range)) (Beam) |
| `kv_conv_size` | `none` | a causal depthwise conv on keys and values, added to them: `k + conv(k)` (Inkling) |
| `value_residual` | `fixed` | how a layer that reads the `v` port mixes in its values ([Ports](#sharing-between-layers-ports)) |

GPT-2 small is MHA with biases and one fused projection
(`examples/architectures/gpt2.bux`):

```bux
attn = gqa(num_heads=12, num_kv_heads=12, bias=true, fused_qkv=true)
```

MiMo-V2.6-Pro has wider query and key heads than value heads
(`examples/architectures/mimo/mimo_v2.6_pro.bux`):

```bux
attn = gqa(num_heads=128, num_kv_heads=8, head_dim=192, v_head_dim=128,
           fused_qkv=true, rope=rope(base=10000000, fraction=0.334))
```

:::{warning}
`gqa` has **no positional encoding by default**: `rope` is `none` unless you set it.
That is what Jamba and Nemotron 3 want, and what an unwary transcription of Llama
does not. `mla` and `csa` default to `rope()`.
:::

:::{note}
A fused projection's split points between q, k and v do not fall on head
boundaries, so under tensor parallelism it stays whole, and every device computes
it ([Tensor parallelism](../parallelism/tensor.md)).
:::

## `mla`: multi-head latent attention

Multi-head latent attention (DeepSeek-V2/V3) compresses keys and values into one
`kv_lora_rank` wide latent per token, and projects every head's keys and values up
from it. Queries come from a `q_lora_rank` wide latent, or straight from the input at
full rank. Both latents are RMS-normalized.

Each query and key head has `qk_nope_head_dim` dims of its own and
`qk_rope_head_dim` more. Only those are rotated by RoPE; for keys they are one vector
that every head shares, projected from the input directly. With `rope=none` no dims
are rotated (Kimi Linear's NoPE layers). The default scale is
`(qk_nope_head_dim + qk_rope_head_dim) ** -0.5`. bux's MLA follows transformers'
`DeepseekV3Attention`.

DeepSeek-V3 (`examples/architectures/deepseek_v3.bux`):

```bux
attn = mla(num_heads=128, q_lora_rank=1536, kv_lora_rank=512,
           rope=rope(layout=interleaved,
                     scaling=yarn(factor=40, original_max_len=4096,
                                  attention_factor_on=softmax)))
```

GLM-5.3 widens the heads (`examples/architectures/glm/glm_5.3.bux`):

```bux
attn = mla(num_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
           qk_rope_head_dim=64, v_head_dim=256,
           rope=rope(base=8000000, layout=interleaved), ...)
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `q_lora_rank` | `none` | the query latent's width; `none` projects queries at full rank |
| `kv_lora_rank` | `512` | the key and value latent's width |
| `qk_nope_head_dim` | `128` | each head's own query and key dims |
| `qk_rope_head_dim` | `64` | the rotated dims, shared by all heads for keys |
| `v_head_dim` | `128` | the value head width |
| `rope` | `rope()` | on the `qk_rope_head_dim` part; `none` for NoPE |

Every head has keys and values of its own, so under tensor parallelism they split
over `heads`, and the latents are named `latent`
([Names](../parallelism/layout.md)). A remat policy that keeps `qkv` keeps the
latents and the shared key dims, and computes the heads' projections up from them
again, as DeepSeek's and Megatron's MLA do.

## `csa`: compressed sparse attention

Compressed sparse attention (DeepSeek-V4) replaces keys and values with **latents**:
a latent is a `head_dim` wide RMS-normalized projection of a token, the key and the
value at once, a single kv head that every query head shares. Each query attends,
in one softmax, to:

1. the latents of a sliding window of tokens (`sliding_window_size`), and
2. **compressed latents**: every `compress_ratio` tokens of a document pooled into
   one, channel by channel under a softmax gate learned from the tokens, then
   RMS-normalized. An indexer picks the `top_k` of them for each query; without one,
   a query attends to all of them.

A compressed latent is rotated to the position of its group's first token, and a
query may attend to it from its group's last token on. The first `rope.fraction` of
each query's and latent's dims are rotated, and since the values were rotated with
the keys, the head outputs are rotated back by their query's position.
`o_lora_rank` makes the output projection grouped and low-rank: `o_groups` groups of
heads, each projected to `o_lora_rank`, then mixed back to the width.

DeepSeek-V4.1-Flash (`examples/architectures/deepseek/deepseek_v4.1_flash.bux`) has
window-only layers, layers that pool pairs of tokens, and layers of one latent a
token:

```bux
window = csa(num_heads=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8,
             sliding_window_size=128, sinks=true, latent_norm=norm,
             rope=rope(fraction=0.125, layout=interleaved))
far    = rope(base=160000, fraction=0.125, layout=interleaved,
              scaling=yarn(factor=16, original_max_len=65536, attention_factor=1))
pairs  = window(compress_ratio=2, rope=far, indexer=indexer(num_heads=32, top_k=512))
single = pairs(compress_ratio=1)
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `head_dim` | `512` | the latent's width, which is every head's key and value width |
| `q_lora_rank` | `none` | the query latent's width; `none` for full rank |
| `o_lora_rank`, `o_groups` | `none`, `1` | the grouped low-rank output projection; `none` for one dense projection |
| `compress_ratio` | `none` | tokens pooled into one compressed latent; `none` for the window only |
| `rope` | `rope()` | on the first `rope.fraction` of the dims |
| `latent_norm` | `rmsnorm` | the norm of the query latent, the latents, the compressed ones and the indexer's keys |

Its ports share the compressed latents, the indexer's picks and a candidate pool
between layers ([Ports](#sharing-between-layers-ports)). An indexer, and any port,
need `compress_ratio`; `o_groups` must divide `num_heads`.

## Options every softmax attention takes

`gqa`, `mla` and `csa` share these settings (`QuadraticAttentionConfig`):

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `sliding_window_size` | `none` | each query sees its own and the `n - 1` keys before it | Mistral, Gemma |
| `sinks` | `false` | a learned logit per head that takes a share of the softmax and attends to nothing | gpt-oss |
| `gated` | `false` | an elementwise gate on the head outputs, before the output projection | Qwen3-Next (Qiu et al.'s G1) |
| `gate_per_head` | `false` | one gate value per head instead | Laguna |
| `gate_activation` | `sigmoid` | the gate's activation | Laguna: `softplus` |
| `gate_head_bias` | `false` | a learned bias per head on the gate, before its activation | Beam |
| `output_norm` | `none` | a norm of each head's output, before the gate | Muse Spark |
| `qk_norm` | `none` | a norm of each query and key head, before RoPE | Qwen3, Gemma 3 |
| `softcap` | `none` | `c * tanh(logits / c)` on the logits | Gemma 2 |
| `scale` | `head_dim ** -0.5` | the logits' scale | |
| `log_scale`, `log_scale_floor` | `none`, `8192` | logits scaled up with the log of the position | Inkling |
| `relative_bias` | `none` | a data-dependent relative position bias, `relbias(...)` | Inkling |
| `attention_dropout`, `output_dropout` | `0` | dropout on the probabilities, and on the output | |
| `indexer` | `none` | sparse attention: an `indexer(...)` picks the keys each query sees | DeepSeek-V3.2 |

:::{warning}
`relative_bias`, `log_scale` and `attention_dropout` need all t by t logits, which
only the `xla` method makes. Under `methods.attention = auto`, the default, a layer
with one of them runs by `xla`; under any other method it raises
`NotImplementedError` ([Context parallelism](../parallelism/attention.md)).
:::

### Sliding windows

`sliding_window_size=n` lets a query see its own key and the `n - 1` before it, and
no key `n` or more positions behind. Gemma 3 runs five local layers for every global
one (`examples/architectures/gemma3_4b.bux`):

```bux
attn   = gqa(num_heads=8, num_kv_heads=4, head_dim=256, qk_norm=norm, rope=rope(base=10000))
local  = attn(sliding_window_size=1024)
global = attn(rope=rope(base=1000000, scaling=pi(factor=8)))

layer(mixer) = [norm mixer norm] [norm ffn norm]
L = layer(local)
G = layer(global)

model = embed(scale_by_sqrt_dim=true) "LLLLLG"^5 "LLLL" norm head(tied=true)
```

:::{tip}
A window counts the query itself. Hymba's window of 1024 does not, so
`examples/architectures/hymba.bux` writes `sliding_window_size=1025`.
:::

### Attention sinks

`sinks=true` gives each head a learned logit, starting at 0, that sits in the
softmax beside the keys' and attends to no value: a query can put probability on it
instead of on any key. MiMo-V2.6-Pro's sliding-window layers have one
(`examples/architectures/mimo/mimo_v2.6_pro.bux`):

```bux
local = attn(sliding_window_size=128, sinks=true, rope.base=10000)
```

### Gated attention

Qiu et al. (2025) compare gates at several places in attention. bux writes their two
main ones:

- **G1**, after the attention and before the output projection: `gated=true`. A
  projection of the layer's input, through `gate_activation`, multiplies every head's
  output elementwise (one value per head with `gate_per_head=true`). Qwen3-Next and
  Qwen3.8 use it:

  ```bux
  attn = gqa(num_heads=16, num_kv_heads=2, head_dim=256, gated=true, qk_norm=norm,
             rope=rope(base=10000000, fraction=0.25))
  ```

  Laguna XS 2.1 scales each head by a softplus of its own
  (`examples/architectures/laguna/laguna_xs_2.1.bux`):

  ```bux
  attn = gqa(num_heads=64, num_kv_heads=8, head_dim=128, qk_norm=norm,
             gated=true, gate_per_head=true, gate_activation=softplus)
  ```

  Beam 1.0 adds a learned bias per head, from 0, to its elementwise sigmoid gate
  (`examples/architectures/reflection/beam_1.0_501b_a23b.bux`):

  ```bux
  attn = gqa(num_heads=48, num_kv_heads=8, head_dim=128, qk_norm=norm,
             gated=true, gate_head_bias=true)
  ```

- **G5**, after the output projection: write the gate with the algebra, as
  `examples/architectures/primitives.bux` does:

  ```bux
  gated_attn = gqa(num_heads=8, num_kv_heads=8, rope=rope()) * linear(features=d_model) sigmoid
  ```

### QK-norm and output norms

`qk_norm=rmsnorm` normalizes every query head and key head on its own, before RoPE,
with a gain per channel of the head. Qwen3-30B-A3B's is above. nGPT normalizes them
with gains, so that the logits are a scaled cosine
(`examples/architectures/ngpt.bux`):

```bux
attn = gqa(num_heads=16, num_kv_heads=16, qk_norm=rmsnorm, rope=rope(layout=interleaved))
```

`qk_norm_all_heads=true`, a setting of `gqa`, makes it one norm of all the heads
together instead: of the queries' `num_heads * head_dim` channels, and one of the
keys', each with a gain per channel, before they split into heads and RoPE turns them.
MiniMax-M2 normalizes so (`examples/architectures/minimax/minimax_m2.bux`), as do
OLMo 2 and 3 (`examples/architectures/olmo/`):

```bux
norm = rmsnorm(eps=1e-6)
attn = gqa(num_heads=48, num_kv_heads=8, head_dim=128, qk_norm=norm,
           qk_norm_all_heads=true, rope=rope(base=5000000, fraction=0.5))
```

Under tensor parallelism each device has some of the heads, and the norm's mean
square sums over all of them. A layer that reads its keys from the `kv` port
normalizes its queries alone ([Ports](#sharing-between-layers-ports)).

`output_norm` normalizes every head's output before the gate. Muse Spark uses both,
without gains (`examples/architectures/muse/muse_spark_open_816b_a42b.bux`):

```bux
bare = rmsnorm(eps=1e-5, affine=false)
attn = gqa(num_heads=128, num_kv_heads=16, head_dim=64, qk_norm=bare,
           scale=5.4730064863875 * 64^-0.5, gated=true, output_norm=bare)
```

:::{note}
On a GPU, an RMSNorm of each head with a gain and no bias followed by RoPE in the
`half` layout runs as one Pallas kernel a pass, forward and backward
([GPU kernels](../performance/kernels.md)).
:::

### Scale and soft-capping

The logits are `scale * q . k`, with `scale` at `head_dim ** -0.5` by default (MLA:
of `qk_nope_head_dim + qk_rope_head_dim`). YaRN with `attention_factor_on=softmax`
multiplies it by the attention factor squared ([YaRN](#yarn)). modded-nanogpt sets
its own (`examples/architectures/modded_nanogpt.bux`):

```bux
attn = gqa(num_heads=6, num_kv_heads=6, head_dim=128, fused_qkv=true, qk_norm=norm,
           scale=0.12, rope=rope(base=1024, fraction=0.5))
```

`softcap=c` caps the logits at `c * tanh(logits / c)`, after the scale, as Gemma 2
does, at 50 (no file of the zoo uses it):

```bux
attn = gqa(num_heads=8, num_kv_heads=4, head_dim=256, softcap=50, rope=rope())
```

### Log scaling at long range

`log_scale=s` multiplies the logits of the query at position $n$ (counted from 1) by

$$
1 + s \ln \max\!\left(\frac{n}{\text{log\_scale\_floor}},\; 1\right),
$$

so that attention stays sharp past `log_scale_floor` tokens. Inkling's global layers
use it past 128K tokens (`examples/architectures/inkling/inkling.bux`):

```bux
attn  = gqa(num_heads=64, num_kv_heads=8, head_dim=128, qk_norm=norm, scale=1 / 128,
            kv_conv_size=4, relative_bias=relbias(rank=16, extent=1024),
            log_scale=0.1, log_scale_floor=128000)
local = attn(num_kv_heads=16, sliding_window_size=512, relative_bias.extent=512,
             log_scale=none)
```

This is not Llama 4's temperature tuning, $1 + s \ln(1 + \lfloor n / f \rfloor)$, a
step function on its NoPE layers.

### Power scaling at long range

`power_scale=p`, a setting of `gqa`, multiplies the query at position $n$ (counted
from 1), and so its logits, by

$$
\left(1 + \frac{n}{\text{power\_scale\_length}}\right)^{p}.
$$

Beam 1.0's global layers have no RoPE and use it so that attention stays sharp as
the context grows: by 1.07 at 2048 tokens and 1.87 at 1M
(`examples/architectures/reflection/beam_1.0_501b_a23b.bux`):

```bux
local  = attn(sliding_window_size=512, rope=rope(base=10000))
global = attn(power_scale=0.1, power_scale_length=2048)
```

It scales the queries, not the logits, so unlike `log_scale` it runs under every
attention method.

## Positions

### Rotary position embeddings: `rope`

`rope(...)` rotates pairs of each query and key head's dims by angles proportional
to the token's position in its document ({bux}`rope`). It goes in an attention's
`rope=`; `rope=none` turns it off.

| Setting | Default | What it does |
|---------|---------|--------------|
| `base` | `10000` | the base of the frequencies: pair $i$ of $d$ rotated dims turns at $\text{base}^{-2i/d}$ |
| `fraction` | `1.0` | the share of each head's dims that rotate, in whole pairs; the rest do not (partial rotary: GPT-J, GLM, Qwen3-Next) |
| `layout` | `half` | `half` pairs dim $i$ with $i + d/2$ (GPT-NeoX, Llama); `interleaved` pairs $2i$ with $2i + 1$ (GPT-J, Cohere) |
| `scaling` | `none` | a context extension: `pi`, `yarn` or `llama3` |

`gqa` and `csa` rotate the first `fraction` of each head's dims; `mla` rotates its
`qk_rope_head_dim` dims. GPT-J rotates a quarter of each head, in interleaved pairs
(`examples/architectures/gpt_j.bux`):

```bux
attn = gqa(num_heads=16, num_kv_heads=16, rope=rope(fraction=0.25, layout=interleaved))
```

**No positional encoding (NoPE).** Leave `rope` out of `gqa`, or set `rope=none` on
any layer. Command A+ alternates three local layers with RoPE and one global layer
without (`examples/architectures/cohere/command_a_plus_05_2026.bux`):

```bux
attn   = gqa(num_heads=128, num_kv_heads=8, head_dim=128,
             rope=rope(base=50000, layout=interleaved))
local  = attn(sliding_window_size=4096)
global = attn(rope=none)
```

SmolLM3 drops RoPE from every fourth layer, and none of its layers has a window
(`examples/architectures/smollm/smollm3_3b.bux`):

```bux
attn = gqa(num_heads=16, num_kv_heads=4, rope=rope(base=5000000))
nope = attn(rope=none)
```

### Context extensions: `pi`, `yarn`, `llama3`

Each scaling divides some of the frequencies by `factor`, so that a model trained on
`original_max_len` positions reaches further. YaRN's and Llama 3's match
transformers' (the tests compare them).

| Scaling | What it does | Settings |
|---------|--------------|----------|
| {bux}`pi` | position interpolation (Chen et al., 2023): every frequency divided by `factor` | `factor` |
| {bux}`yarn` | YaRN (Peng et al., 2023): fast pairs keep their frequency, slow ones are divided by `factor`, a ramp between; and an attention factor | `factor`, `original_max_len`, `beta_fast=32`, `beta_slow=1`, `attention_factor`, `attention_factor_on=rotary` |
| {bux}`llama3` | Llama 3.1's: slow pairs divided by `factor`, fast ones kept, a ramp between | `factor`, `original_max_len=8192`, `low_freq_factor=1`, `high_freq_factor=4` |

Gemma 3 interpolates its global layers' positions by 8 (above). Llama 3.1 extends
Llama 3's RoPE by 8 (no file of the zoo uses it):

```bux
attn = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000, scaling=llama3(factor=8)))
```

#### YaRN

YaRN counts the turns each rotary pair makes over `original_max_len` positions.
Pairs that make more than `beta_fast` turns keep their frequency, those that make
fewer than `beta_slow` have it divided by `factor`, and a linear ramp blends the
ones between. q . k is scaled by `attention_factor ** 2`, which defaults to
$0.1 \ln(\text{factor}) + 1$; set `attention_factor=1` to turn it off.
`attention_factor_on` says where the factor goes:

- `rotary` (transformers): cos and sin are multiplied by it, so only the rotated dims
  scale.
- `softmax` (DeepSeek's MLA): the softmax scale is multiplied by its square, so all
  of q . k scales.

Laguna XS 2.1's global layers widen the fast band (`beta_fast=64`),
Mellum2's stretch RoPE 16 times, and Olmo 3's 8 times while its sliding-window
layers keep theirs as they were (`examples/architectures/olmo/olmo_3_7b.bux`):

```bux
global = attn(num_heads=48, rope=rope(base=500000, fraction=0.5,
              scaling=yarn(factor=32, original_max_len=8192, beta_fast=64)))
```

```bux
global = attn(rope.scaling=yarn(factor=16, original_max_len=8192))
```

```bux
local  = attn(sliding_window_size=4096, rope=rope(base=500000))
global = attn(rope=rope(base=500000, scaling=yarn(factor=8, original_max_len=8192)))
```

### Relative position bias: `relbias`

`relative_bias=relbias(rank=r, extent=e)` adds a data-dependent bias to every logit
(Inkling): head $h$ adds $\langle W_h x_i, P[i - j]\rangle$ to its logit for key $j$,
where $P$ is a learned table of `extent` offsets, `rank` wide, starting at zero. Keys
`extent` or more positions behind get no bias. Inkling has no RoPE: its positions
come from this bias (see [Log scaling](#log-scaling-at-long-range) for its file).

{bux}`relbias` needs all t by t logits: under `methods.attention = auto` the layer
runs by `xla`.

### Learned absolute positions

`embed(max_positions=n)` adds a learned position embedding, as GPT-2 does: see
[Embeddings](layers.md#embed). The compiler checks that `seq_len` fits in it.

## Sharing between layers: ports

A layer can export an internal tensor to a tap, `attn(v=@v1)`, and a later layer can
read it, `attn(v=$v1)`, or pop it, `attn(v=$v1!)`. The taps are the same stacks the
stream's `@name` and `$name` use ([Skip connections](../language/skip-connections.md)).
A config lists its ports; using one it does not have is a compile error.

| Layer | Port | Writing it (`@t`) | Reading it (`$t`) |
|-------|------|-------------------|-------------------|
| `gqa` | `v` | pushes the value heads | mixes the tap's values into its own (value residual) |
| `gqa` | `kv` | pushes the keys and values | uses them instead of projecting its own (cross-layer KV sharing, YOCO, CLA) |
| `gqa`, `mla`, `csa` | `index` | pushes the keys the indexer kept for each query | attends to them, and runs no indexer of its own (IndexCache) |
| `csa` | `kv` | pushes the compressed latents, with the indexer's keys | uses them instead of compressing its own; an indexer it runs scores those keys |
| `csa` | `candidates` | pushes a pool of coarse blocks of compressed positions | its indexer keeps keys only inside the pool |

`rwkv7` has a `v` port too ([RWKV-7](recurrent.md#rwkv7-rwkv-7-goose)).

### Value residuals

ResFormer (Zhou et al., 2024) mixes the first layer's values into every later
layer's (`examples/architectures/resformer.bux`):

```bux
first = [norm attn(v=@v1)] [norm ffn]
rest  = [norm attn(v=$v1)] [norm ffn]
model = embed first rest^11 norm head
```

A reading layer computes $a v + b v_1$, with $a = b = 0.5$ to start.
`value_residual` says what it learns:

| `value_residual` | Mix | Paper |
|------------------|-----|-------|
| `fixed` | $a = b = 0.5$, not learned | ResFormer's code (Identity-ResFormer) |
| `learned` | $a$ and $b$ learned | Learnable-ResFormer |
| `convex` | $a$ learned, $b = 1 - a$ | |

### Shared keys and values

Hymba's sliding-window layers share their keys and values in pairs and triples: the
layer that writes `kv` projects them, and the layers that read it project queries
only (`examples/architectures/hymba.bux`):

```bux
W = layer(local(kv=@kv))
S = layer(local(kv=$kv))
R = layer(local(kv=$kv!))

model = embed F "WR"^7 F "WSR" "WR"^6 F norm head(tied=true)
```

A layer that reads `kv` has no key and value projections, no key norm and no
`kv_conv_size` conv of its own.

## Sparse attention: `indexer`

DeepSeek Sparse Attention (DSA, DeepSeek-V3.2) adds a **lightning indexer** to a
layer: a small attention that scores every key a query may attend to and keeps the
`top_k` best, to which the layer then attends alone. Set `indexer=indexer(...)` on
`gqa`, `mla` or `csa` ({bux}`indexer`). DeepSeek-V3.2
(`examples/architectures/deepseek/deepseek_v3.2.bux`):

```bux
attn = mla(num_heads=128, q_lora_rank=1536, kv_lora_rank=512, indexer=indexer(top_k=2048),
           rope=rope(layout=interleaved,
                     scaling=yarn(factor=40, original_max_len=4096,
                                  attention_factor_on=softmax)))
```

The indexer has `num_heads` query heads of `head_dim` and one key a token. It scores
key $s$ for the query at $t$ by

$$
I_{t,s} = \sum_j w_{t,j}\,\mathrm{ReLU}(q_{t,j} \cdot k_s),
$$

with weights $w_{t,j}$ projected from the input when `weighted=true` (DSA), or the
same for every head otherwise. With `relu=false` the dot products are plain.

- Its queries come from the layer's query latent (`mla`, `csa`), or its input
  (`gqa`, and `mla` at full rank).
- Its keys, one a token, are projected from the layer's input and layer-normalized
  (DSA) in `gqa` and `mla`; in `csa`, from the compressed latents, normalized by
  `latent_norm`. `norm`, if set, replaces either norm.
- Both rotate as many dims as the layer does. Where its keys are the tokens, it pairs
  them as `rope_layout` says, whatever the layer's layout.

| Setting | Default | What it does |
|---------|---------|--------------|
| `num_heads`, `head_dim` | `64`, `128` | the indexer's query heads |
| `top_k` | `2048` | keys each query keeps |
| `block_size` | `1` | keep whole blocks of this many keys, scored as `pool` says |
| `pool` | `mean` | how a block scores: the mean of its keys (QSA), or the best score of those a query sees (MSA) |
| `per_group` | `false` | a pick for each kv head's group of query heads, by a head of its own (MSA) |
| `relu` | `true` | ReLU on the dot products (DSA); else plain ones (MSA) |
| `weighted` | `true` | learned per-head weights (DSA); else a plain sum |
| `norm` | `none` | the norm of each query head and of the keys; `none`: no query norm, and the layer's own key norm (DSA: a LayerNorm) |
| `rope_layout` | `half` | how it pairs RoPE dims when its keys are the tokens, whatever the layer's layout (DeepSeek-V3.2: `half`, GLM-5.3: `interleaved`) |
| `candidate_block_size`, `candidate_top_k` | `8`, `2048` | what a `candidates` port exports |
| `loss_weight` | `1.0` | the weight of [its loss](#the-indexers-loss), its only training signal |

The indexer's inputs are detached and its pick passes no gradient, so only its loss
trains it, and that loss trains nothing else.

### Blocks: QSA

With `block_size=n` (and `pool=mean`), the indexer mean-pools the keys in blocks of
`n`, normalizes each block's key and rotates it to the position of the block's first
key, and keeps the `top_k // n` best whole blocks among those the query may attend
to all of. It also keeps the keys of the query's own unfinished block. Blocks count
from a document's first key, so none spans two documents. Qwen3.8-Flash-Next's QSA
keeps the best 512 blocks of 4 (`examples/architectures/qwen/qwen3.8_flash_next.bux`):

```bux
attn = gqa(num_heads=24, num_kv_heads=2, head_dim=256, gated=true, qk_norm=norm,
           rope=rope(base=10000000, fraction=0.25),
           indexer=indexer(num_heads=4, head_dim=128, top_k=2048, block_size=4,
                           weighted=false, norm=norm))
```

(indexer-msa)=
### Blocks for each group: MSA

MiniMax Sparse Attention (MSA, MiniMax-M3) keeps whole blocks too, but scores them
by their keys, and picks them for each kv head's group of query heads apart:

- `pool=max` scores a block by the best score of its keys the query sees, so a block
  that the query sees part of counts too. The query always keeps its own block,
  among the `top_k // n` it keeps.
- `per_group=true` gives the indexer a head for each of `gqa`'s kv heads
  (`num_heads` is the layer's `num_kv_heads`), and each head picks for its group by
  its own scores, with `weighted=false`. The query heads of a group then attend,
  exactly, to the keys of their group's blocks alone.
- `relu=false` scores a key by the plain dot product, $q_r \cdot k_s / \sqrt{d}$ for
  group $r$'s head.

MiniMax-M3 keeps 16 blocks of 128 keys for each of its 4 groups of 16 query heads
(`examples/architectures/minimax/minimax_m3.bux`):

```bux
norm = rmsnorm(zero_centered=true)
attn = gqa(num_heads=64, num_kv_heads=4, head_dim=128, qk_norm=norm,
           rope=rope(base=5000000, fraction=0.5))
msa  = attn(indexer=indexer(num_heads=4, head_dim=128, top_k=2048, block_size=128,
                            pool=max, per_group=true, relu=false, weighted=false,
                            norm=norm))
```

`pool=max` alone keeps one pick for all heads, of their summed scores; `per_group`
without blocks picks keys for each group. A layer that reads a per-group `index`
needs as many kv heads as it has groups. Under tensor parallelism the indexer's
heads and their picks split with the kv heads. On a GPU tokamax's kernel attends a
group at a time, each with its group's mask; the indexer's scores, top-k and loss run
by XLA, as its kernels (`scoring`'s, and cuDNN's on a B200 or B300) sum the heads'
ReLU'd dot products and pick once for all heads. Each of `pool=max`, `per_group` and
`relu=false` keeps an indexer off them.

:::{warning}
`per_group` is `gqa`'s alone, and with blocks it takes `pool=max`. `pool=max` on
`csa`'s compressed latents is not implemented.
:::

### Sharing an index: IndexCache

A layer that writes the `index` port exports its indexer's picks, and a layer that
reads it attends to the same keys without an indexer of its own (and adds no loss).
GLM-5.3 runs an indexer in layers 0, 1, 2, 6, 10, ..., 74 only
(`examples/architectures/glm/glm_5.3.bux`):

```bux
A = layer(attn(index=@topk), dense)
F = layer(attn(index=@topk))
S = layer(attn(index=$topk))

model = embed "AAA" "SSS" "FSSS"^18 norm @h out mtp^3
```

### Candidate pools

In `csa`, a layer that writes `candidates` exports the `candidate_top_k` best blocks
of `candidate_block_size` compressed positions, each scored by its best key, and
always the block of the newest key a query may attend to. A layer whose indexer
reads `candidates` keeps keys only inside that pool. DeepSeek-V4.1-Flash computes
the latents and the pool once at layer 20, and every fourth layer from 24 indexes
again inside the pool:

```bux
S = layer(single(kv=@kv, index=@topk, candidates=@pool))
s = layer(single(kv=$kv, index=$topk))
I = layer(single(kv=$kv, index=@topk, candidates=$pool))  # re-indexes within the pool
```

:::{warning}
Writing `index`, and either side of `candidates`, need an indexer the layer runs:
set `indexer=indexer(...)` and do not read `index`. A candidate pool with
`block_size > 1` is not implemented.
:::

(indexer-loss)=
### The indexer's loss

The indexer learns from a loss of its own, as DeepSeek-V3.2's does:
`indexer(..., loss_weight=1)`. Each layer that runs one adds `loss_weight` times

$$
\mathrm{KL}\big(p \,\|\, \mathrm{softmax}(I)\big),
$$

taken over the keys the indexer kept and averaged over the queries. $I$ is its
scores, and $p$ the layer's attention to those keys, summed over heads and
normalized over them.

- $p$ comes from the attention's logsumexp a chunk of queries at a time, and the
  indexer scores each chunk again, so neither makes t × t floats.
- With `block_size` (Qwen3.8 QSA), a block counts as its most attended key, and only
  the complete blocks kept count.
- With `pool=max` or `per_group` (MiniMax MSA), it is over the tokens of the kept
  blocks that the query sees, each key its own, and for each group apart: $p$ is
  the attention of the group's heads, the mean of their probabilities once
  normalized, and $I$ the group's head's scores. It is averaged over the queries and
  the groups. A chunk of queries scores every key and takes $p$, and does both again
  in its backward pass.
- In CSA (DeepSeek-V4), the keys are the compressed latents, and $p$ comes from the
  layer's one softmax over the window, the latents and the sinks: it keeps the share
  the window and sinks take before it is normalized over the kept latents.
- The indexer's inputs and $p$ are detached, so this loss trains the indexer alone,
  and the language-modeling loss trains the rest. The weight only scales the
  indexer's gradient, and its part of the norm that `clip` bounds.
- A layer that reads `index` runs no indexer and adds nothing.
- The log's `indexer_loss` is the sum over layers. Only a training step adds it: eval
  does not.

DeepSeek's dense warm-up is a stage with `top_k` at least `seq_len`, where attention
is dense and the loss covers every key the query sees. bux does not freeze the rest
of the model for it. MSA's is the same stage with MSA's settings: every block kept,
and the loss of each group over every key.

## How attention runs

The layers make q, k and v; the run's `methods.attention` computes the softmax
(`auto` by default: cuDNN's kernels, FlashAttention-4's and tokamax's on a GPU, `xla`
elsewhere). Sparse attention attends by `bux.architecture.attention.sparse`: on a
GPU, tokamax's flash kernel with the kept keys as a mask, for heads of up to 192
dims; elsewhere and for wider heads, XLA a chunk of queries at a time; neither makes
t × t floats. The indexer ranks a chunk of queries
at a time too, and picks its top-k by counting the bits of the k-th highest score
rather than sorting, which on a GPU is 6 times faster at 32k keys.

`csa`'s latents are 512 wide, more than tokamax's kernels take. On a B200 or B300,
cuDNN's DSA kernels (`bux.architecture.attention.sparse_cudnn`) attend each query to
its window and the latents it keeps, and its sinks, in one softmax, forward and
backward; the window-only layers too. Each query's list of slots holds its window's
and its picks, so the cost is in the slots a query sees, not in the row's length. The
indexer's loss then takes its target at the picks alone, by cuDNN's kernel too.

- Methods, context parallelism and how attention splits over devices:
  [Attention](../parallelism/attention.md).
- The kernels and their speed: [GPU kernels](../performance/kernels.md).
- Precision: the `q`, `kv` and `o` projections take `precision.roles`; softmax
  attention's own matmuls stay in the compute dtype
  ([Precision](../training/precision.md)).
- Remat: a policy can keep `attention.in` (q, k and v as the kernel takes them),
  `attention.out` (the output and its logsumexp), `attention.index` (the indexer's
  picks), or `qkv` (the projections, MLA's latents)
  ([Remat](../parallelism/remat.md)).
