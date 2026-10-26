# Initialization

This page covers how a model's weights start: the default, the top-level setting
`init` that changes it, the roles a weight matrix can start apart by, what the
papers found, and the parameters that always start their own way. Read it to
reproduce a paper's init or to ablate one.

```bux
# GPT-2: every weight matrix from a normal of 0.02, the residual branches' outputs
# scaled down by the square root of their number
layers = 12
init   = normal(out=normal(std=0.02 / (2 * layers) ^ 0.5))
model  = embed [norm attn] [norm ffn]^layers norm head
```

## The default

Every weight matrix starts from a normal of standard deviation 0.02: projections,
experts, routers, embedding tables and the head; but Gated DeltaNet-2's, which start
from Xavier's uniform at a gain of $2^{-2.5}$ as its paper's code does, unless the
file sets an `init`. GPT-1 and BERT drew every weight so,
it is the `initializer_range` of nearly every open model's config, and OLMo 2 draws
every parameter so, truncated at 3 standard deviations. It does not follow the
width: it is LeCun's $1 / \sqrt{d}$ for a projection from a stream of $d = 2500$;
narrower models start smaller than LeCun's, wider ones larger.

The run's `seed` draws every weight.

## `init`

The top-level setting `init` says how the weight matrices start. Its own settings
start them all; `embed`, `head` and `out` each take an init of their own for one
role:

| Role | The weight matrices | Default |
|------|---------------------|---------|
| `embed` | the embedding tables: the tokens', which a tied head reads, and the learned positions' | as the rest |
| `head` | an untied head | as the rest |
| `out` | the projections back to the stream's width, which end a residual branch: attention's and the mixers' output, MLPs' and experts' down | as the rest |
| the rest | every other: q, k and v, up and gate, routers, low-rank projections, `linear`, `nextn` | the init's own settings |

A layer gives a matrix its role by the names of its dimensions: one whose output is
named `embed`, the stream's width, is an `out` ([Adding a layer](adding-a-layer.md)).

The inits ({bux}`normal`, {bux}`fan_in`, {bux}`zeros`):

| Init | Draws | Settings |
|------|-------|----------|
| `normal` | from $\mathcal{N}(0, \sigma^2)$ | `std` ($\sigma$, 0.02), `truncate` (draw again past this many $\sigma$, without correcting the std; none) |
| `fan_in`, `fan_out`, `fan_avg` | at a variance of `scale` over the matrix's fan in, out, or their mean: LeCun's, He's at `scale=2`, Glorot's | `scale` (1), `distribution`: `truncated` at 2 $\sigma$ with the std corrected, as Flax's, `gaussian` or `uniform` |
| `zeros` | 0 | |

A fan init takes each matrix of a stack apart: an expert's fan in is its own input
width. An embedding table's fans are both its width $d$, as `flax.nnx.Embed`'s.

## Recipes

`layers` is the file's own definition, the depth it repeats its block by; `2 * layers`
counts the residual branches of a block of attention and an MLP. A hybrid of one
branch a layer counts `layers` (Megatron-LM's hybrids, Mamba).

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
| bux before `normal` was the default | `fan_in(embed=fan_in(distribution=gaussian))` |

Runs that set bux's former init keep the ids they had, but an MoE's: its routed
experts started $\sqrt{E}$ times smaller then. A config's `initializer_range` says
little of how a model was pretrained: DeepSeek-V3's says 0.02, its paper 0.006.

A sweep compares them ([Sweeps](../language/sweeps.md)); the file
`examples/experiments/init_ablation.bux` trains five:

```bux
init = normal: normal
     | olmo2: normal(truncate=3)
     | gpt2: normal(out=normal(std=0.02 / (2 * layers) ^ 0.5))
     | lecun: fan_in
     | zero_out: normal(out=zeros)
```

Out of reach of `init` alone: a std by the index of a layer (torchtitan's Llama now,
OLMo 1's), one projection's apart from the others of its role (T5's queries,
modded-nanogpt's MLP down alone), and μP and its successors, which scale the forward
pass and each role's learning rate besides the init.

## What the papers found

No study settles which init is best; the differences are small next to a learning
rate's, and depth scaling's smallest of all.

- **OLMo 2** (OLMo et al., 2025, §3.2) replaced OLMo-0424's init, of a std scaled by
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
- **Embeddings**: small sublayers need embeddings of a std near 1 to keep the gradient
  from spiking, by scaling them by $\sqrt{d}$ (`embed(scale_by_sqrt_dim=true)`) or by
  normalizing them (Takase et al., *Spike No More*, 2025). At 1.7B, either fix took the
  WikiText perplexity from 22.58 to 21.29, and only they ran without spikes.
  Megatron-LM suggests an embedding std of 1 for the same reason.

## What starts its own way

A parameter that is not a plain weight matrix keeps the start its layer gives it,
whatever `init` says:

- norms' gains (`gain_init`) and biases, at 1 and 0; every bias at 0;
- short convolutions' kernels (`conv`, and those inside Mamba, the linear attentions
  and attention's `kv_conv_size`), LeCun normal over their taps, or 0 where a layer
  says so (Engram);
- Mamba's and Mamba-2's `A_log`, `D`, time-step bias and projection, and the linear
  attentions' decays and time steps, as their papers start them;
- RWKV-7's projections, uniform, its low-rank ones, orthogonal, and its zero output;
- Engram's tables, of unit variance;
- the residual algebras' connections, which start as the plain stream, and gates and
  scales (`scale(init=...)`), attention sinks and value mixes.

## Adding an init

An init is a config in `bux/config.py`, added to `Init`, and a builder registered for
its tag in `bux.initializers.inits`, which returns the
[Flax initializer](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/initializers.html)
a weight matrix is drawn by. The builder is told the matrix's role and the axes that
stack matrices ({py:class}`bux.initializers.Matrix`):

```python
from flax import nnx

from bux import initializers


@initializers.inits.register("orthogonal")
def _(c: OrthogonalInitConfig, matrix: initializers.Matrix) -> nnx.Initializer:
    """Start orthogonal, scaled by ``gain`` (Saxe et al., 2014)."""
    return nnx.initializers.orthogonal(c.gain)
```

The [registries](../reference/language/registries.md) list the inits there are.
