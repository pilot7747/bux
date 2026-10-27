# Linear attention and state-space models

Recurrent token mixers keep a fixed-size state and update it token by token, so their
cost grows linearly with the sequence instead of quadratically. bux has linear
attention (`gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning`), RWKV-7 (`rwkv7`),
the state-space models `mamba` and `mamba2`, and the short convolutions `conv` and
`short_conv`. Read this page to choose a mixer, to see the recurrence a setting
changes, or to build a hybrid with softmax attention. Qwen3-Next runs three Gated
DeltaNet layers for every gated attention layer (abridged from
`examples/architectures/qwen3_next.bux`):

```bux
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
| [`gdn2`](#gdn2-gated-deltanet-2) | delta rule, erasing and writing by gates per channel | per key channel, data-dependent | Hatamizadeh et al., 2026 |
| [`lightning`](#lightning-lightning-attention) | add $k v^\top$ | per head, fixed | TransNormerLLM; MiniMax-01 |
| [`rwkv7`](#rwkv7-rwkv-7-goose) | generalized delta rule | per key channel, data-dependent | Peng et al., 2025 |
| [`mamba`](#mamba-selective-ssm) | selective scan | per channel and state dim | Gu and Dao, 2023; Jamba |
| [`mamba2`](#mamba2-mamba-2-ssd) | selective scan | per head | Dao and Gu, 2024; Zamba2, Nemotron-H |
| [`conv`](#conv-causal-depthwise-convolution) | a causal depthwise convolution | | Canon layers; Inkling |
| [`short_conv`](#short_conv-gated-short-convolution) | a double-gated short convolution | | LFM2 |

Every state is zero at the start of every document, so each document in a packed row
gets the same output it would get alone.

## Linear attention

### One recurrence for all of them

Every linear attention layer, and RWKV-7, runs one recurrence over a state $S$ of
$d_k \times d_v$ per value head:

$$
S_t = \mathrm{diag}(w_t)\, S_{t-1} + b_t \left(a_t^\top S_{t-1}\right) + k_t v_t^\top,
\qquad o_t = S_t^\top q_t,
$$

with a decay $w_t \in (0, 1]$ per key channel or one per head. Without $a$ and $b$ it
is decayed linear attention. The $b\, a^\top$ term is how the delta rule overwrites
what $S$ holds at a key, and RWKV-7 generalizes it. The layers differ in how they make
$q$, $k$, $v$, $w$, $a$ and $b$ from the input.

### What they share

`gla`, `deltanet`, `gdn`, `kda`, `gdn2`, `lightning` and `rwkv7` take these settings
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

A layer runs in five steps:

1. Project q, k and v from the input, convolve them (with `conv_size`), and apply
   SiLU.
2. Run the recurrence.
3. RMS-normalize each head's output.
4. Multiply it by `output_gate`'s activation of a projection of the input (the gated
   norm of Qwen3-Next and fla).
5. Project back to the width.

Some layers change these defaults; their sections say how.

### `gla`: gated linear attention

Gated linear attention (Yang et al., 2024), as fla has it: linear attention with a
decay per key channel and no delta rule ({bux}`gla`).

```bux
mixer = gla(num_heads=4, head_dim=64)
```

$$
S_t = \mathrm{diag}(w_t)\, S_{t-1} + k_t v_t^\top, \qquad
\log w_t = \tfrac{1}{16} \log \sigma(f_t),
$$

where $f_t$ is a low-rank projection of the input with a bias (`gate_rank`, 16), and
$q$ is scaled by $d_k^{-1/2}$. Its `conv_size` defaults to `none`, and without a conv
q, k and v skip the SiLU.

### `deltanet`

DeltaNet (Schlag et al., 2021; Yang et al., 2024), as fla has it ({bux}`deltanet`).
Each token moves what the state holds at its key a step $\beta_t$ toward its value:

```bux
mixer = deltanet(num_heads=4, head_dim=64)
```

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) S_{t-1} + \beta_t k_t v_t^\top,
\qquad \beta_t = \sigma(W_\beta x_t),
$$

on L2-normalized $q$ and $k$, with $q$ scaled by $d_k^{-1/2}$ and one $\beta$ per value
head.

### `gdn`: Gated DeltaNet

Gated DeltaNet (Yang et al., 2025), as transformers' Qwen3-Next and fla have it:
DeltaNet with a decay per value head ({bux}`gdn`). Qwen3.8-27B's has 48 value heads
over 16 q and k heads (`examples/architectures/qwen/qwen3.8_27b.bux`):

```bux
delta = gdn(num_heads=16, num_v_heads=48, head_dim=128)
```

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) \alpha_t S_{t-1} + \beta_t k_t v_t^\top,
\qquad
\alpha_t = \exp\!\big(-e^{A_{\log}}\, \mathrm{softplus}(W_a x_t + \text{dt\_bias})\big),
$$

with `A_log` and `dt_bias` learned per value head. `A_log` starts at the logs of
draws uniform in [0.01, 16] (transformers), and `dt_bias` at Mamba's time steps (fla
and the paper's code; transformers starts it at ones).

### `kda`: Kimi Delta Attention

Kimi Delta Attention (Kimi Linear, 2025), as transformers and fla have it: Gated
DeltaNet with a decay per key channel ({bux}`kda`). Kimi Linear's KDA, and Kimi K3's
bounded one (`examples/architectures/kimi_linear.bux`,
`examples/architectures/kimi/kimi_k3.bux`):

```bux
kda = kda(num_heads=32, head_dim=128, output_gate_rank=128, norm_eps=1e-5)
```

```bux
kda = kda(num_heads=96, head_dim=128, gate_lower_bound=-5, norm_eps=1e-5)
```

$$
S_t = \left(I - \beta_t k_t k_t^\top\right) \mathrm{diag}(w_t)\, S_{t-1} + \beta_t k_t v_t^\top,
\qquad
\log w_t = -e^{A_{\log}}\, \mathrm{softplus}(f_t + \text{dt\_bias}),
$$

where $f_t = W_{f2} W_{f1} x_t$ is a low-rank projection (`gate_rank`, by default
`head_dim`) and `A_log` is per head. Its output gate is a `sigmoid` by default, and
low-rank with `output_gate_rank`.

With `gate_lower_bound=L` (Kimi K3, GLM-5-Next), the decay is bounded:

$$
\log w_t = L\, \sigma\!\big(e^{A_{\log}} (f_t + \text{dt\_bias})\big) \in (L, 0),
$$

and `A_log` starts at 0, as in fla's safe gate (K3's code starts it at
$\log U(1, 16)$ either way).

:::{tip}
`gate_lower_bound` also decides which kernel runs on a GPU. cuDNN's fused KDA kernel,
the fastest, takes only a decay bounded at -5 or above, as `gate_lower_bound=-5`
bounds it; any other gate runs by bux's Pallas kernels
([How they run](#how-they-run)).
:::

### `gdn2`: Gated DeltaNet-2

Gated DeltaNet-2 (Hatamizadeh et al., 2026), as the paper's code and fla have it
({bux}`gdn2`). It is KDA with the step $\beta_t$, which both erases and writes, split
into two gates per channel: an erase gate $b_t$ over the key channels and a write
gate $w_t$ over the value channels. The paper's hybrid puts a `gdn2` layer before
each sliding-window attention layer (`examples/architectures/gated_deltanet_2.bux`):

```bux
gdn2 = gdn2(num_heads=16, head_dim=128)
```

$$
S_t = \left(I - k_t (b_t \odot k_t)^\top\right) \mathrm{diag}(e^{g_t})\, S_{t-1}
  + k_t (w_t \odot v_t)^\top,
\qquad b_t = \sigma(W_b x_t),\quad w_t = \sigma(W_w x_t),
$$

with KDA's decay per key channel, $g_t = -e^{A_{\log}}\, \mathrm{softplus}(f_t +
\text{dt\_bias})$ of a low-rank $f_t$ (`gate_rank`, by default `v_head_dim`), and the
same `gate_lower_bound`. At $b_t = w_t = \beta_t$ it is KDA. $g$, $b$ and `A_log`
belong to a q and k head and are shared out to its value heads, as q and k are; $w$
belongs to a value head. `negative_eigenvalues=true` doubles the erase gate, into
$(0, 2)$.

It starts as the paper's code does:

- every projection from Xavier's uniform at a gain of $2^{-2.5}$, unless the file
  sets an `init` ([Initialization](initialization.md));
- the output gate low-rank (`output_gate_rank`, by default `v_head_dim`), with a bias
  starting at 0;
- `A_log` at $\log U(1, 16)$, and `dt_bias` at Mamba's time steps.

Its output norm's gate is a `swish`, the paper's SiLU (fla's port has a `sigmoid`), and
`norm_eps` is `1e-5`.

:::{tip}
`gate_lower_bound=-5` lets cuDNN's GDN-2 kernel run the layer on a B200 or B300,
several times faster than the chunks. The paper's unbounded gate runs by the chunks
([How they run](#how-they-run)).
:::

### `lightning`: lightning attention

Lightning attention (TransNormerLLM; MiniMax-01, as transformers has it): linear
attention on SiLU'd, unscaled q, k and v, with a fixed decay $e^{-s_h}$ per head
that weakens with depth ({bux}`lightning`):

```bux
mixer = lightning(num_heads=4, head_dim=64)
```

$$
S_t = e^{-s_h} S_{t-1} + k_t v_t^\top, \qquad
s_h = 2^{-8h/H} \left(1 - \frac{\ell}{L - 1}\right),
$$

for heads $h = 1, \dots, H$, where $\ell$ is the layer's index among the model's $L$
token mixers (plus $10^{-5}$, as MiniMax has it). The last layer sums the past
without decay. One RMSNorm covers all the heads together before the gate, which is a
`sigmoid` by default; `conv_size` defaults to `none`.

MiniMax-Text-01 runs seven for every softmax attention layer, 80 layers in all, so
its decays weaken over all 80 (`examples/architectures/minimax/minimax_text_01.bux`):

```bux
lightning = lightning(num_heads=64, head_dim=128)
model = embed "LLLLLLLA"^10 norm head
```

### `rwkv7`: RWKV-7 "Goose"

RWKV-7 time mixing (Peng et al., 2025), as the RWKV-LM reference has it
({bux}`rwkv7`):

```bux
mixer = rwkv7(num_heads=4, head_dim=64)
```

Its state follows the generalized delta rule,

$$
S_t = \left(\mathrm{diag}(w_t) - (a_t \odot \kappa_t)\, \kappa_t^\top\right) S_{t-1}
      + k_t v_t^\top,
\qquad o_t = S_t^\top r_t,
$$

with $\kappa$ the normalized key it removes from, $a$ the in-context learning rate, and
$\log w = -e^{-0.5}\, \sigma(\cdot)$ per key channel.

- **Token shift**: every projection reads the token moved toward the one before it,
  by a learned amount per channel.
- **Low-rank projections** make $w$, $a$, the output gate $g$ and the value
  residual's mix.
- **Output**: a LayerNorm of each head, plus a bonus $(r \odot k \odot r_k)\, v$ per
  head, times $g$.
- **Init**: parameters start as RWKV-LM starts them, by the layer's depth among the
  model's token mixers.

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

Mamba's selective state-space model (Gu and Dao, 2023), as Jamba has it, with the
time steps, B and C RMS-normalized ({bux}`mamba`). Jamba runs Mamba in seven layers of
eight (`examples/architectures/jamba.bux`):

```bux
ssm = mamba(state_size=16)
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `state_size` | `16` | the state dims of each channel |
| `expand` | `2` | the inner width, as a multiple of the width |
| `conv_size` | `4` | the causal conv on $u$ |
| `dt_rank` | `ceil(width / 16)` | the rank of the time step's projection |
| `dt_min`, `dt_max` | `1e-3`, `1e-1` | the range the time steps start in, log-uniform |

The input projects to $u$ and a gate $z$, both `expand` times the width. $u$ is
convolved over positions and SiLU'd. It then projects to $B$ and $C$, which all its
channels share, and to a time step $\Delta$ of rank `dt_rank`, projected up to one
per channel. Every channel keeps a state of `state_size`, which decays at its own
rate in each state dim:

$$
h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t\, u_t, \qquad y_t = C_t^\top h_t,
$$

with $A = -\exp(A_{\log})$, starting at $-(1, 2, \dots, n)$ in every channel
(S4D-real). With $D u$ added, the output is gated by $\mathrm{SiLU}(z)$ and projected
back.

### `mamba2`: Mamba-2 (SSD)

Mamba-2 (Dao and Gu, 2024), as transformers' Mamba-2, Zamba2 and Nemotron-H mixers
compute it ({bux}`mamba2`). Nemotron 3's
(`examples/architectures/nemotron/nemotron_3_ultra_550b_a55b.bux`):

```bux
ssm = mamba2(state_size=128, head_dim=64, num_groups=8)
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `state_size` | `128` | the state dims of each head |
| `head_dim` | `64` | the channels of a head |
| `num_heads` | `none` | heads; `none`: `expand * width // head_dim` |
| `num_groups` | `1` | groups of heads that share $B$ and $C$ |
| `expand`, `conv_size`, `dt_min`, `dt_max` | `2`, `4`, `1e-3`, `1e-1` | as `mamba`'s |

The input projects to a gate $z$, to $u$, $B$ and $C$ (convolved over positions
together and SiLU'd), and to one time step per head. $u$ splits into heads of
`head_dim`, whose states decay at one rate per head, with $A$ starting uniform in
$-[1, 16]$. $B$ and $C$ split into `num_groups` groups that the heads share, as GQA's
query heads share kv heads. With $D u$ added, the output is gated by
$\mathrm{SiLU}(z)$, RMS-normalized one group of heads at a time, and projected back.

`expand * width` must split into heads of `head_dim`, and the heads into
`num_groups`. Where no whole `expand` gives a model's heads, set `num_heads`:
Nemotron 3 Nano has 64 heads of 64 at a width of 2688.

### `conv`: causal depthwise convolution

`conv(size=4)` convolves each channel over the last `size` positions, with no gate or
activation ({bux}`conv`). As a residual branch, `[conv]` is a short-conv shortcut,
$y + \mathrm{conv}(y)$ (Canon layers). Inkling puts one on every branch's output
(`examples/architectures/inkling/inkling.bux`):

```bux
layer(mixer, ffn) = [norm mixer [conv]] [norm ffn scale [conv]]
```

### `short_conv`: gated short convolution

LFM2's double-gated short convolution (cf. H3, Hyena) ({bux}`short_conv`). The input
projects to $B$, $C$ and $u$, and the output is $C \odot \mathrm{conv}(B \odot u)$,
projected back. `conv_size` defaults to 3.

```bux
mixer = short_conv(conv_size=3)
```

## Hybrids

A hybrid is a pattern of one-letter blocks ([Definitions](../language/definitions.md)).
The zoo has these:

| Model | Pattern | File |
|-------|---------|------|
| Qwen3-Next, Qwen3.8 | three `gdn` for every gated `gqa`, `"DDDA"^12` | `qwen3_next.bux`, `qwen/qwen3.8_27b.bux` |
| Kimi Linear | three `kda` for every NoPE `mla`, `"FKKA" "KKKA"^5 "KKA"` | `kimi_linear.bux` |
| Kimi K3 | the same at 93 layers, under attention residuals | `kimi/kimi_k3.bux` |
| Gated DeltaNet-2 | `gdn2` and sliding-window `gqa` alternating, `"NS"^9` | `gated_deltanet_2.bux` |
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

Zamba2 calls one shared block. Each call passes the block's output through its own
linear layer and adds it to the input of the Mamba-2 layer after it
(`examples/architectures/zamba2.bux`):

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
(the WY form). The state passes across chunks by a Pallas kernel on a GPU
(`affine_scan`), by `lax.scan` elsewhere, and by a matmul when there is one decay per
head. No exponent in it is positive, whatever the gate; RWKV-7, whose decay is
bounded, takes each pair through the chunk's middle position instead, in one matmul.
Recurrent states are float32, whatever dtype the run computes in.

On a GPU, under `methods.recurrence = scan`, kernels run the delta rule, GLA and
Lightning with q, k and v of 16 bits and heads of 64 or 128:

| Layer | Kernel | Condition |
|-------|--------|-----------|
| `gdn`, `deltanet` | cuDNN's fused GDN kernel (`fused_delta`) | a B200 or B300; any gate |
| `kda` with a `gate_lower_bound` of -5 or above | cuDNN's fused KDA kernel (`fused_delta`) | a B200 or B300 |
| `kda` otherwise | bux's Pallas kernels (`pallas_delta`) | any GPU; any gate |
| `gdn2` with a `gate_lower_bound` of -5 or above | cuDNN's fused GDN-2 kernel (`fused_delta`) | a B200 or B300; bfloat16 |
| `gdn2` otherwise | chunks | |
| `gla`, `lightning` | cuDNN's fused GDN-2 kernel (`fused_delta`), its delta rule at a step of 0 | a B200 or B300; bfloat16 |
| `rwkv7` | chunks, each pair product one matmul (its decay is bounded), the state across them by `affine_scan` | |
| `mamba2` | chunks, the state across them by a matmul (one decay per head) | |
| `mamba` | a scan over positions (one decay per channel and state dim) | |

cuDNN's KDA and GDN-2 kernels factor each decay through a tile of 16 positions. The
KDA kernel loses precision where a tile decays a channel by $e^{-86}$ and overflows
at $e^{-90}$; the GDN-2 kernel stays exact, but past $e^{-83}$ it reruns the whole
call by a slower path. Unbounded gates get that steep within the first few training
steps, so both kernels run only with a `gate_lower_bound` of -5 or above, which keeps
a tile within $e^{-80}$. bux's Pallas kernels for KDA are exact at any gate
([Recurrence](../parallelism/recurrence.md)).

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
