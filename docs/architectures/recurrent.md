# Linear attention and state-space models

This page covers bux's recurrent token mixers: linear attention (`gla`, `deltanet`,
`gdn`, `kda`, `lightning`), RWKV-7 (`rwkv7`), the state-space models `mamba` and
`mamba2`, and the short convolutions `conv` and `short_conv`. Each keeps a state
that it updates token by token, so its cost grows linearly with the sequence. Read it
to choose a mixer, to see the recurrence a setting changes, or to build a hybrid with
softmax attention.

```bux
# Qwen3-Next: three Gated DeltaNet layers for every gated attention layer
delta = gdn(num_heads=16, num_v_heads=32, head_dim=128)
attn  = gqa(num_heads=16, num_kv_heads=2, head_dim=256, gated=true, qk_norm=norm,
            rope=rope(base=10000000, fraction=0.25))

layer(mixer) = [norm mixer] [norm experts]
D = layer(delta)
A = layer(attn)

model = embed "DDDA"^12 norm @h out mtp
```

| Layer | State update | Decay | From |
|-------|--------------|-------|------|
| [`gla`](#gla-gated-linear-attention) | add $k v^\top$ | per key channel, data-dependent | Yang et al., 2024 |
| [`deltanet`](#deltanet) | delta rule | none | Schlag et al., 2021; Yang et al., 2024 |
| [`gdn`](#gdn-gated-deltanet) | delta rule | per head, data-dependent | Yang et al., 2025; Qwen3-Next |
| [`kda`](#kda-kimi-delta-attention) | delta rule | per key channel, data-dependent | Kimi Linear, 2025 |
| [`lightning`](#lightning-lightning-attention) | add $k v^\top$ | per head, fixed | TransNormerLLM; MiniMax-01 |
| [`rwkv7`](#rwkv7-rwkv-7-goose) | generalized delta rule | per key channel, data-dependent | Peng et al., 2025 |
| [`mamba`](#mamba-selective-ssm) | selective scan | per channel and state dim | Gu and Dao, 2023; Jamba |
| [`mamba2`](#mamba2-mamba-2-ssd) | selective scan | per head | Dao and Gu, 2024; Zamba2, Nemotron-H |
| [`conv`](#conv-causal-depthwise-convolution) | a causal depthwise convolution | | Canon layers; Inkling |
| [`short_conv`](#short_conv-gated-short-convolution) | a double-gated short convolution | | LFM2 |

Every state is zero at the start of every document, so a row packed with several
documents gives each what it gives alone.

## Linear attention

### One recurrence for all of them

Every linear attention layer, and RWKV-7, runs one recurrence over a state $S$ of
$d_k \times d_v$ per value head:

$$
S_t = \mathrm{diag}(w_t)\, S_{t-1} + b_t \left(a_t^\top S_{t-1}\right) + k_t v_t^\top,
\qquad o_t = S_t^\top q_t,
$$

with a decay $w_t \in (0, 1]$ per key channel or one per head. Without $a$ and $b$ it
is decayed linear attention. The $b\, a^\top$ term is how the delta rule writes over
what $S$ holds at a key, and RWKV-7 generalizes it. The layers differ in how they make
$q$, $k$, $v$, $w$, $a$ and $b$ from the input.

### What they share

`gla`, `deltanet`, `gdn`, `kda`, `lightning` and `rwkv7` take these settings
(`LinearAttentionConfig`):

| Setting | Default | What it does |
|---------|---------|--------------|
| `num_heads` | required | q and k heads |
| `head_dim` | required | the q and k head width, $d_k$ |
| `num_v_heads` | `num_heads` | value heads; with more value heads than q and k heads, each q and k head serves a group of them (Qwen3-Next) |
| `v_head_dim` | `head_dim` | the value head width, $d_v$ |
| `conv_size` | `4` | a causal depthwise conv on q, k and v; `none` turns it off |
| `output_gate` | `swish` | the gate on the output norm before the output projection, `swish` or `sigmoid`; `none` turns it off |
| `norm_eps` | `1e-6` | the epsilon of the output norm (Qwen3-Next: 1e-6; Kimi Linear and fla: 1e-5) |

A layer projects q, k and v from its input, convolves them (with `conv_size`) and
applies SiLU, runs the recurrence, then RMS-normalizes each head's output, multiplies
it by `output_gate`'s activation of a projection of the input (the gated norm of
Qwen3-Next and fla), and projects back to the width. The defaults some layers change
are in their sections.

### `gla`: gated linear attention

Gated linear attention (Yang et al., 2024), as fla has it: linear attention with a
decay per key channel and no delta rule,

$$
S_t = \mathrm{diag}(w_t)\, S_{t-1} + k_t v_t^\top, \qquad
\log w_t = \tfrac{1}{16} \log \sigma(f_t),
$$

where $f_t$ is a low-rank projection of the input with a bias (`gate_rank`, 16), and
$q$ is scaled by $d_k^{-1/2}$. Its `conv_size` defaults to `none`, and without a conv
q, k and v do not go through SiLU. See {bux}`gla`.

```bux
mixer = gla(num_heads=4, head_dim=64)
```

### `deltanet`

DeltaNet (Schlag et al., 2021; Yang et al., 2024), as fla has it. Each token moves
what the state holds at its key a step $\beta_t$ toward its value:

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) S_{t-1} + \beta_t k_t v_t^\top,
\qquad \beta_t = \sigma(W_\beta x_t),
$$

on L2-normalized $q$ and $k$, with $q$ scaled by $d_k^{-1/2}$ and one $\beta$ per value
head. See {bux}`deltanet`.

```bux
mixer = deltanet(num_heads=4, head_dim=64)
```

### `gdn`: Gated DeltaNet

Gated DeltaNet (Yang et al., 2025), as transformers' Qwen3-Next and fla have it:
DeltaNet with a decay per value head,

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) \alpha_t S_{t-1} + \beta_t k_t v_t^\top,
\qquad
\alpha_t = \exp\!\big(-e^{A_{\log}}\, \mathrm{softplus}(W_a x_t + \text{dt\_bias})\big),
$$

with `A_log` and `dt_bias` learned per value head. `A_log` starts at the logs of
draws uniform in [0.01, 16] (transformers), and `dt_bias` at Mamba's time steps (fla
and the paper's code; transformers starts it at ones). See {bux}`gdn`.

Qwen3.8-27B's, with 48 value heads over 16 q and k heads
(`examples/architectures/qwen/qwen3.8_27b.bux`):

```bux
delta = gdn(num_heads=16, num_v_heads=48, head_dim=128)
```

### `kda`: Kimi Delta Attention

Kimi Delta Attention (Kimi Linear, 2025), as transformers and fla have it: Gated
DeltaNet with a decay per key channel,

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) \mathrm{diag}(w_t)\, S_{t-1} + \beta_t k_t v_t^\top,
\qquad
\log w_t = -e^{A_{\log}}\, \mathrm{softplus}(f_t + \text{dt\_bias}),
$$

where $f_t = W_{f2} W_{f1} x_t$ is a low-rank projection (`gate_rank`, by default
`head_dim`) and `A_log` is per head. Its output gate is a `sigmoid` by default, and
low-rank with `output_gate_rank`. See {bux}`kda`.

With `gate_lower_bound=L` (Kimi K3, GLM-5-Next), the decay is bounded:

$$
\log w_t = L\, \sigma\!\big(e^{A_{\log}} (f_t + \text{dt\_bias})\big) \in (L, 0),
$$

and `A_log` starts at 0, as in fla's safe gate (K3's code starts it at
$\log U(1, 16)$ either way).

Kimi Linear's KDA, and Kimi K3's bounded one
(`examples/architectures/kimi_linear.bux`, `examples/architectures/kimi/kimi_k3.bux`):

```bux
kda = kda(num_heads=32, head_dim=128, output_gate_rank=128, norm_eps=1e-5)
```

```bux
kda = kda(num_heads=96, head_dim=128, gate_lower_bound=-5, norm_eps=1e-5)
```

:::{tip}
`gate_lower_bound` also decides which kernel runs on a GPU: cuDNN's fused KDA kernel
takes only a decay bounded at -5 or above, as `gate_lower_bound=-5` bounds it. On a
B300 it runs 10 times as fast as the chunks, where bux's own Pallas kernels, which take
any gate, run 2.6 times as fast ([How they run](#how-they-run)).
:::

### `lightning`: lightning attention

Lightning attention (TransNormerLLM; MiniMax-01, as transformers has it): linear
attention on SiLU'd, unscaled q, k and v, with a fixed decay $e^{-s_h}$ per head
that weakens with depth,

$$
S_t = e^{-s_h} S_{t-1} + k_t v_t^\top, \qquad
s_h = 2^{-8h/H} \left(1 - \frac{\ell}{L - 1}\right),
$$

for heads $h = 1, \dots, H$, where $\ell$ is the layer's index among the model's $L$
token mixers (plus $10^{-5}$, as MiniMax has it). The last layer sums the past
plainly. One RMSNorm covers all the heads together before the gate, which is a
`sigmoid` by default; `conv_size` defaults to `none`. See {bux}`lightning`.

```bux
mixer = lightning(num_heads=4, head_dim=64)
```

MiniMax-Text-01 runs seven for every softmax attention layer, 80 layers in all, so
its decays weaken over all 80 (`examples/architectures/minimax/minimax_text_01.bux`):

```bux
lightning = lightning(num_heads=64, head_dim=128)
model = embed "LLLLLLLA"^10 norm head
```

### `rwkv7`: RWKV-7 "Goose"

RWKV-7 time mixing (Peng et al., 2025), as the RWKV-LM reference has it. Its state
follows the generalized delta rule,

$$
S_t = \left(\mathrm{diag}(w_t) - (a_t \odot \kappa_t)\, \kappa_t^\top\right) S_{t-1}
      + k_t v_t^\top,
\qquad o_t = S_t^\top r_t,
$$

with $\kappa$ the normalized key it removes from, $a$ the in-context learning rate, and
$\log w = -e^{-0.5}\, \sigma(\cdot)$ per key channel. Every projection reads the token
moved toward the one before it by a learned amount per channel (token shift).
Low-rank projections make $w$, $a$, the output gate $g$ and the value residual's mix.
The output is a LayerNorm of each head, plus a bonus $(r \odot k \odot r_k)\, v$ per
head, times $g$. Parameters start as RWKV-LM starts them, by the layer's depth among
the model's token mixers. See {bux}`rwkv7`.

| Setting | Default | What it does |
|---------|---------|--------------|
| `decay_rank`, `a_rank`, `v_rank`, `gate_rank` | `64`, `64`, `32`, `128` | the ranks of the low-rank projections of $w$, $a$, the value mix and $g$ |
| `output_gate` | `sigmoid` | the gate is $\mathrm{act}(x G_1) G_2$, a low-rank projection with this activation between |
| `norm_eps` | `1e-5` | times the value head dim, as in fla (RWKV-LM: 64e-5) |
| `conv_size` | `none` | must stay `none`: RWKV-7 shifts tokens instead |

RWKV-7's **value residual** goes through its `v` port: the first layer exports its
values, and every later one moves its own toward them by a learned amount per
channel, from a projection of rank `v_rank`:

```bux
first = [norm rwkv7(num_heads=4, head_dim=64, v=@v0)] [norm ffn]
rest  = [norm rwkv7(num_heads=4, head_dim=64, v=$v0)] [norm ffn]
model = embed first rest^11 norm head
```

## State-space models and convolutions

### `mamba`: selective SSM

Mamba's selective state-space model (Gu and Dao, 2023) as Jamba has it, with the time
steps, B and C RMS-normalized. The input projects to $u$ and a gate $z$, both
`expand` times the width. $u$ is convolved over positions and SiLU'd, and projects to
$B$ and $C$, which all its channels share, and to a time step $\Delta$ of rank
`dt_rank`, projected up to one per channel. Every channel keeps a state of
`state_size`, which decays at its own rate in each state dim:

$$
h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t\, u_t, \qquad y_t = C_t^\top h_t,
$$

with $A = -\exp(A_{\log})$, starting at $-(1, 2, \dots, n)$ in every channel
(S4D-real). With $D u$ added, the output is gated by $\mathrm{SiLU}(z)$ and projected
back. See {bux}`mamba`.

| Setting | Default | What it does |
|---------|---------|--------------|
| `state_size` | `16` | the state dims of each channel |
| `expand` | `2` | the inner width, as a multiple of the width |
| `conv_size` | `4` | the causal conv on $u$ |
| `dt_rank` | `ceil(width / 16)` | the rank of the time step's projection |
| `dt_min`, `dt_max` | `1e-3`, `1e-1` | the range the time steps start in, log-uniform |

Jamba runs Mamba in seven layers of eight (`examples/architectures/jamba.bux`):

```bux
ssm = mamba(state_size=16)
```

### `mamba2`: Mamba-2 (SSD)

Mamba-2 (Dao and Gu, 2024), as transformers' Mamba-2, Zamba2 and Nemotron-H mixers
compute it. The input projects to a gate $z$, to $u$, $B$ and $C$ (convolved over
positions together and SiLU'd), and to one time step per head. $u$ splits into heads
of `head_dim`, whose states decay by one rate per head, $A$ uniform in $-[1, 16]$ to
start; $B$ and $C$ split into `num_groups` groups that the heads share, as kv heads
are shared in GQA. With $D u$ added, the output is gated by $\mathrm{SiLU}(z)$,
RMS-normalized one group of heads at a time, and projected back. See {bux}`mamba2`.

| Setting | Default | What it does |
|---------|---------|--------------|
| `state_size` | `128` | the state dims of each head |
| `head_dim` | `64` | the channels of a head |
| `num_heads` | `none` | heads; `none`: `expand * width // head_dim` |
| `num_groups` | `1` | groups of heads that share $B$ and $C$ |
| `expand`, `conv_size`, `dt_min`, `dt_max` | `2`, `4`, `1e-3`, `1e-1` | as `mamba`'s |

`expand * width` must split into heads of `head_dim`, and the heads into
`num_groups`. Where no whole `expand` gives a model's heads, set `num_heads`:
Nemotron 3 Nano has 64 heads of 64 at a width of 2688. Nemotron 3's (`examples/architectures/nemotron/nemotron_3_ultra_550b_a55b.bux`):

```bux
ssm = mamba2(state_size=128, head_dim=64, num_groups=8)
```

### `conv`: causal depthwise convolution

`conv(size=4)` convolves each channel over the last `size` positions, with no gate or
activation ({bux}`conv`). As a residual branch, `[conv]` is a short-conv shortcut,
$y + \mathrm{conv}(y)$ (Canon layers). Inkling puts one on every branch's output
(`examples/architectures/inkling/inkling.bux`):

```bux
layer(mixer, ffn) = [norm mixer [conv]] [norm ffn scale [conv]]
```

### `short_conv`: gated short convolution

LFM2's double-gated short convolution (cf. H3, Hyena): the input projects to $B$,
$C$ and $u$, and the output is $C \odot \mathrm{conv}(B \odot u)$, projected back.
`conv_size` defaults to 3. See {bux}`short_conv`.

```bux
mixer = short_conv(conv_size=3)
```

## Hybrids

A hybrid is a pattern of one-letter blocks ([Definitions](../language/definitions.md)).
The zoo's:

| Model | Pattern | File |
|-------|---------|------|
| Qwen3-Next, Qwen3.8 | three `gdn` for every gated `gqa`, `"DDDA"^12` | `qwen3_next.bux`, `qwen/qwen3.8_27b.bux` |
| Kimi Linear | three `kda` for every NoPE `mla`, `"FKKA" "KKKA"^5 "KKA"` | `kimi_linear.bux` |
| Kimi K3 | the same at 93 layers, under attention residuals | `kimi/kimi_k3.bux` |
| Jamba | `mamba` with one `gqa` in eight, MoE every other layer, `"mMmM aMmM"^4` | `jamba.bux` |
| Nemotron 3 | `mamba2` and MoE layers alternating, attention every 7 to 9 | `nemotron/nemotron_3_ultra_550b_a55b.bux` |
| Zamba2 | `mamba2`, and one `shared` transformer block before every sixth | `zamba2.bux` |
| Hymba | attention and Mamba heads side by side in every layer | `hymba.bux` |

Jamba (`examples/architectures/jamba.bux`):

```bux
layer(mixer, ffn) = [norm mixer] [norm ffn]
m = layer(ssm, dense)
M = layer(ssm, experts)
a = layer(attn, dense)

model = embed "mMmM aMmM"^4 norm head
```

Zamba2 calls one shared block, whose output, through a linear layer of each call's
own, adds to the input of the Mamba-2 layer after it (`examples/architectures/zamba2.bux`):

```bux
m = [norm ssm]
shared sa = norm attn norm ffn
h = [(_ + sa linear(features=d_model)) norm ssm]

model = embed (m^5 h)^6 m^2 norm head(tied=true)
```

Hymba runs attention and Mamba heads in parallel on one input, each normalized and
scaled (`examples/architectures/hymba.bux`):

```bux
layer(attn) = [norm (attn norm scale + ssm norm scale)] [norm ffn]
```

:::{note}
Attention in a hybrid often has no positional encoding, since the recurrent layers
carry position: Kimi Linear's MLA layers set `rope=none`, and Jamba's and Nemotron's
`gqa` leave `rope` out.
:::

## How they run

The recurrence runs by **chunks** of 64 positions, as fla's chunked kernels run it
(`bux.architecture.recurrent.chunkwise`): matmuls within a chunk, and a scan of the
state across chunks. Every chunk's maps come from matmuls over all chunks at once
(the WY form), and the state across them by a Pallas kernel on a GPU
(`affine_scan`), by `lax.scan` elsewhere; for one decay a head, by a matmul. No
exponent in it is positive, whatever the gate, but for RWKV-7, whose decay is
bounded: it takes a chunk's pairs through its middle position, by one matmul each,
in a third of the time. Recurrent states are float32, whatever the run computes in.

On a GPU, kernels run the delta rule, GLA and Lightning with q, k and v of 16 bits
and heads of 64 or 128, under `methods.recurrence = scan`:

| Layer | Kernel | Condition |
|-------|--------|-----------|
| `gdn`, `deltanet` | cuDNN's fused GDN kernel (`fused_delta`) | a B200 or B300; any gate |
| `kda` with a `gate_lower_bound` of -5 or above | cuDNN's fused KDA kernel (`fused_delta`) | a B200 or B300 |
| `kda` otherwise | bux's Pallas kernels (`pallas_delta`) | any GPU; any gate |
| `gla`, `lightning` | cuDNN's fused GDN-2 kernel (`fused_delta`), its delta rule at a step of 0 | a B200 or B300; bfloat16 |
| `rwkv7` | chunks, a pair product one matmul (its decay is bounded), the state across them by `affine_scan` | |
| `mamba2` | chunks, the state across them by a matmul (one decay a head) | |
| `mamba` | a scan over positions (one decay per channel and state dim) | |

cuDNN's KDA kernel factors each decay through the start of a tile of 16 positions,
so it loses precision where a tile decays a channel by $e^{-86}$, and overflows at
$e^{-90}$. Within 12 steps of bux's start, KDA's unbounded gates are that steep in 30
to 90% of the (row, head) pairs of most layers, so those, and gates bounded below -5,
run by bux's kernels, which are exact at any gate.

Settings that change how they run (none changes what they compute):

- `methods.states`: what the backward pass reads of the chunks' states. `keep`, the
  default, keeps them from the forward pass; `recompute` computes them again from the
  inputs, in a fourteenth of the memory (a sixteenth for Mamba-2) and a fifth more
  time. cuDNN's kernels keep only their inputs either way.
- `methods.recurrence`: `scan`, the default, or `pass_state`, which splits the
  positions over the `context` axis and passes the state along
  ([Recurrence](../parallelism/recurrence.md)).
- Remat: a policy can keep `recurrence.in` (the mixer's inputs, as its kernel takes
  them) and `recurrence.out` (its output and the chunks' states)
  ([Remat](../parallelism/remat.md)).
- Precision: the projections take `precision.roles.recurrence`, the output projection
  `roles.recurrence_out` ([Precision](../training/precision.md)).
- Tensor parallelism splits every projection and per-head parameter over `heads`
  (Mamba's and `short_conv`'s channels over `mlp`), and the recurrence runs on each
  device's own heads with no communication.

The kernels' speed, and what each model trains at: [GPU kernels](../performance/kernels.md).
