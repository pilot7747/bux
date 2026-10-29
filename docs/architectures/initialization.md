# Initialization

Every weight matrix in bux starts from a normal of 0.02 unless you say otherwise. The
top-level setting `init` changes that for all matrices at once, or by role (the
embeddings, the head, the residual branches' outputs), so you can reproduce a paper's
init or ablate one in a line. GPT-2's:

```bux
# GPT-2: every weight matrix from a normal of 0.02, and the residual branches' outputs
# scaled down by the square root of their number
layers = 12
init   = normal(out=normal(std=0.02 / (2 * layers) ^ 0.5))
block  = [norm attn] [norm ffn]
model  = embed block^layers norm head
```

## The default

Every weight matrix starts from a normal with a standard deviation of 0.02:
projections, experts, routers, embedding tables and the head. The one exception is
Gated DeltaNet-2, whose projections start from Xavier's uniform at a gain of
$2^{-2.5}$, as its paper's code does, unless the file sets an `init`.

0.02 has a long history. GPT-1 and BERT drew every weight this way, it is the
`initializer_range` of nearly every open model's config, and OLMo 2 draws every
parameter so, truncated at 3 standard deviations. It does not follow the width: it
equals LeCun's $1 / \sqrt{d}$ for a projection from a stream of $d = 2500$. Narrower
models start smaller than LeCun's, wider ones larger.

The run's `seed` draws every weight.

## `init`

The top-level setting `init` says how the weight matrices start. Its own settings
start them all; `embed`, `head` and `out` each take a separate init for one role:

| Role | The weight matrices | Default |
|------|---------------------|---------|
| `embed` | the embedding tables: the tokens' (which a tied head reads) and the learned positions' | as the rest |
| `head` | an untied head | as the rest |
| `out` | the projections back to the stream's width, which end a residual branch: attention's and the mixers' output, MLPs' and experts' down | as the rest |
| the rest | every other: q, k and v, up and gate, routers, low-rank projections, `linear`, `nextn` | the init's own settings |

A layer gives a matrix its role by the names of its dimensions: a matrix whose output
is named `embed`, the stream's width, is an `out` ([Adding a layer](adding-a-layer.md)).

The inits ({bux}`normal`, {bux}`fan_in`, {bux}`zeros`):

| Init | Draws | Settings |
|------|-------|----------|
| `normal` | from $\mathcal{N}(0, \sigma^2)$ | `std` ($\sigma$, 0.02), `truncate` (draw again past this many $\sigma$, without correcting the std; none) |
| `fan_in`, `fan_out`, `fan_avg` | at a variance of `scale` over the matrix's fan in, out, or their mean: LeCun's, He's at `scale=2`, Glorot's | `scale` (1), `distribution`: `truncated` at 2 $\sigma$ with the std corrected, as Flax's, `gaussian` or `uniform` |
| `zeros` | 0 | |

A fan init treats each matrix of a stack separately: an expert's fan in is its own
input width. An embedding table's fans are both its width $d$, as for
`flax.nnx.Embed`.

## Recipes

`layers` is the file's own definition, the depth it repeats its block by. `2 * layers`
counts the residual branches of a block of attention and an MLP; a hybrid with one
branch per layer counts `layers` (Megatron-LM's hybrids, Mamba).

| From | `init` |
|------|--------|
| GPT-2, nanoGPT, Megatron-LM, SmolLM3 | `normal(out=normal(std=0.02 / (2 * layers) ^ 0.5))` |
| OPT | `normal(std=0.006, out=normal(std=0.006 / (2 * layers) ^ 0.5))` |
| OLMo 2 and 3 | `normal(truncate=3)` |
| DeepSeek-V2 and V3 | `normal(std=0.006)` |
| Trinity | `normal(std=0.5 * d_model ^ -0.5, truncate=3)` |
| GPT-NeoX, Pythia: small init (Nguyen & Salazar, 2019), Wang's for the outputs | `normal(std=(2 / (5 * d_model)) ^ 0.5, out=normal(std=2 / (layers * d_model ^ 0.5)))` |
| PaLM | `fan_in(distribution=gaussian, embed=normal(std=1))` |
| Mamba | `fan_in(scale=1 / 3, distribution=uniform, embed=normal(std=0.02), out=fan_in(scale=1 / (3 * layers), distribution=uniform))` |
| Zero residual outputs (Fixup's last layers) | `normal(out=zeros)` |
| Zero outputs and head (modded-nanogpt, January 2025) | `normal(out=zeros, head=zeros)` |
| PyTorch's defaults, `nn.Linear` and `nn.Embedding` | `fan_in(scale=1 / 3, distribution=uniform, embed=normal(std=1))` |
| Flax NNX's defaults, `nnx.Linear` and `nnx.Embed` | `fan_in(embed=fan_in(distribution=gaussian))` |

A config's `initializer_range` says little about how a model was pretrained:
DeepSeek-V3's says 0.02, its paper 0.006.

A sweep compares them ([Sweeps](../language/sweeps.md));
`examples/experiments/init_ablation.bux` trains five:

```bux
init = normal: normal
     | olmo2: normal(truncate=3)
     | gpt2: normal(out=normal(std=0.02 / (2 * layers) ^ 0.5))
     | lecun: fan_in
     | zero_out: normal(out=zeros)
```

`init` alone can't express:

- a std that depends on the layer's index (torchtitan's Llama, OLMo 1);
- one projection started apart from the others of its role (T5's queries,
  modded-nanogpt's MLP down alone), but attention's values and output together,
  which `gqa`'s `mimetic_vo` starts ([below](#mimetic-values-and-output));
- μP and its successors, which scale the forward pass and each role's learning rate
  as well as the init.

(mimetic-values-and-output)=
## Mimetic values and output

`gqa(mimetic_vo=beta)` starts a layer's value and output projections so that its
value-output map is $-\beta I$ (modded-nanogpt's mimetic V-O init, after Trockman &
Kolter, 2023, who find trained attention has that structure). Attention then
subtracts a little of what each query attends to from its first step, where it
otherwise adds noise. modded-nanogpt takes $\beta = 0.05$ on every attention layer
and found $0.1$ or more unstable:

```bux
attn = gqa(num_heads=8, num_kv_heads=2, qk_norm=norm, gated=true, mimetic_vo=0.05)
```

- **Values:** value head $j$ reads the stream's channels $j\,d_v$ to $(j + 1)\,d_v$,
  times $\sqrt\beta$.
- **Output:** each query head writes its output back onto its value head's
  channels, times $-\sqrt\beta / n$, where $n$ is the number of query heads in a kv
  head's group. A head attending to itself alone subtracts $\beta / n$ of those
  channels, and the group $\beta$.
- **Noise:** both projections then take $\mathcal{N}(0, \sigma^2)$ in place of
  the init's draw, `mimetic_noise` $\sigma$ (0.01, the source's).

With as many kv heads as heads tiling the width, this is the source's init exactly:
each head's product is $-\beta I$ and so is the layer's. With fewer kv heads the
layer subtracts $\beta$ of the channels its value heads read, `num_kv_heads *
v_head_dim` of them, and nothing of the rest. The source starts its queries and keys
its own way too; bux leaves them to `init`. The draws come after the layer's others,
from the same seed ({py:func}`bux.initializers.mimetic_vo`).

## What the papers found

No study settles which init is best. The differences are small next to the learning
rate's, and depth scaling's are the smallest of all.

- **OLMo 2** (OLMo et al., 2025, §3.2) replaced OLMo-0424's init, a std scaled by
  the width and each layer's depth, with $\mathcal{N}(0, 0.02)$ truncated at 3. The
  norms of activations and gradients grew less from the first layer to the last,
  across widths; in a run made to spike, the gradient's spike score fell from 0.40
  to 0.03. It converged slightly slower.
- **The std** (Owen et al., 2025; a 1B Llama on 100B tokens): from 0.03 down to 0.006,
  the mean of 7 tasks went 48.5, 49.8, 50.3, 50.5 (0.008), 50.1. GPT-2's depth
  scaling added 0.1. At 210M, 0.006 beat 0.02 only with a long warmup (Semenov et
  al., 2025).
- **Depth scaling** of the outputs, by the total depth or by each layer's, changed the
  loss by 0.02 at most, from 2 to 64 layers trained with Scion (Filatov et al., 2025).
- **Zero outputs**: zeroing attention's and the MLP's output projections lowered the
  perplexity of a 2.9B model from 14.09 to 13.86 with MHA and from 13.93 to 13.73 with
  MLA, and changed little with GQA and MQA (MLRA, 2026).
- **Embeddings**: small sublayers need embeddings with a std near 1 to keep the
  gradient from spiking, by scaling them by $\sqrt{d}$
  (`embed(scale_by_sqrt_dim=true)`) or by normalizing them (Takase et al., *Spike No
  More*, 2025). At 1.7B, either fix took the WikiText perplexity from 22.58 to 21.29,
  and only those runs had no spikes. Megatron-LM suggests an embedding std of 1 for
  the same reason.

## What starts its own way

A parameter that is not a plain weight matrix keeps the start its layer gives it,
whatever `init` says:

- norms' gains (`gain_init`) and biases, at 1 and 0; every bias at 0;
- short convolutions' kernels (`conv`, and those inside Mamba, the linear attentions
  and attention's `kv_conv_size`), LeCun normal over their taps, or 0 where a layer
  says so (Engram);
- Mamba's and Mamba-2's `A_log`, `D`, time-step bias and projection, and the linear
  attentions' decays and time steps, as their papers start them;
- RWKV-7's projections, uniform; its low-rank ones, orthogonal; and its output, at 0;
- Engram's tables, at unit variance;
- the residual algebras' connections, which start as the plain stream, and gates and
  scales (`scale(init=...)`), attention sinks and value mixes.

## Adding an init

An init is a config in `bux/config.py`, added to `Init`, and a builder registered for
its tag in `bux.initializers.inits`. The builder returns the
[Flax initializer](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html)
that draws a weight matrix. It is told the matrix's role and the axes that stack
matrices ({py:class}`bux.initializers.Matrix`):

```python
from flax import nnx

from bux import initializers


@initializers.inits.register("orthogonal")
def _(c: OrthogonalInitConfig, matrix: initializers.Matrix) -> nnx.Initializer:
    """Start orthogonal, scaled by ``gain`` (Saxe et al., 2014)."""
    return nnx.initializers.orthogonal(c.gain)
```

The [registries](../reference/language/registries.md) list the inits there are.
