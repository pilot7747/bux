# Mixture of experts

`moe` is bux's token-choice mixture of experts. This page covers how its router picks
experts, shared experts, LatentMoE, and how to balance the experts' load by a loss or
a bias. Read it to configure an MoE layer, or to reproduce a model's routing. How the
experts run on many GPUs, and in which kernels, is on the parallelism page
[Experts](../parallelism/experts.md).

`moe` is a channel mixer: write it where an MLP would go. Qwen3-30B-A3B routes each
token to 8 of 128 experts (`examples/architectures/qwen/qwen3_30b_a3b.bux`):

```bux
experts = moe(num_experts=128, top_k=8, aux_loss_coef=1e-3,
              expert=gated_mlp(intermediate_size=768, activation=swish))

block = [norm attn] [norm experts]
model = embed block^48 norm head
```

Every field is in its reference entry, {bux}`moe`.

## What it computes

For each token $x$, the router scores every expert, keeps the `top_k` best, and sums
their outputs by their weights, plus the shared experts if any:

$$
y = \sum_{e \in \text{top-}k(x)} g_e(x)\, E_e(x) \;+\; \text{shared}(x).
$$

Each expert $E_e$ is an MLP of the shape `expert` gives: `mlp(...)` or
`gated_mlp(...)`, with any activation, clamp or soft-cap
([MLPs](layers.md#mlps)). DeepSeek-V4.1-Flash's experts are clamped SwiGLUs
(`examples/architectures/deepseek/deepseek_v4.1_flash.bux`):

```bux
experts = moe(num_experts=384, top_k=6, num_shared_experts=1, router=sqrtsoftplus,
              routed_scaling_factor=1.5, load_balancing=aux_free, aux_loss_coef=1e-4,
              aux_loss_scope=sequence,
              expert=gated_mlp(intermediate_size=2304, activation=swish, limit=10))
```

With `bias=true` in `expert`, every routed expert has its own biases on its up, gate
and down projections, and weight decay skips them. gpt-oss-120b's experts have them,
and use their own SwiGLU variant: a `quick_gelu` gate, $g\,\sigma(1.702\, g)$, capped
at 7, times `up` clipped to $\pm 7$ plus 1 (`up_offset`). Its router has a bias on its
logits too (`examples/architectures/gpt_oss_120b.bux`):

```bux
experts = moe(num_experts=128, top_k=4, router_bias=true,
              expert=gated_mlp(intermediate_size=2880, activation=quick_gelu,
                               bias=true, limit=7.0, up_offset=1.0))
```

:::{note}
cuDNN's fused kernels take gpt-oss's clamp and offset as a GeGLU, but no biases, so
these experts run by bux's grouped GEMMs ([Activations fused, by
cuDNN](../parallelism/experts.md#activations-fused-by-cudnn)). Every `methods.moe`
adds the experts' biases, `nccl_ep` included.
:::

## Routing

The router is a linear layer from the width to one logit per expert. With
`router_bias=true` (gpt-oss) it also learns a bias on the logits, which changes both
which experts a token picks and their weights. `aux_free`'s bias
([below](#the-aux-free-bias)) changes only which experts a token picks.

For each token, in order:

1. **Logits**: $x W_r$, in float32, times `router_logit_scale` (Muse Spark scales them
   by $\sqrt{256 / d}$).
2. **Scores**, by `router`:

   | `router` | Score of an expert | Models |
   |----------|-------------------|--------|
   | `softmax` (default) | $\mathrm{softmax}(\ell)_e$ | Mixtral, Qwen3-MoE, Jamba |
   | `sigmoid` | $\sigma(\ell_e)$ | DeepSeek-V3, Kimi K2, Nemotron 3 |
   | `sqrtsoftplus` | $\sqrt{\mathrm{softplus}(\ell_e)}$ | DeepSeek-V4 |

3. **Choice**: with `load_balancing=aux_free` or `smebu`, a bias per expert is added
   to the scores that pick the top-k, and not to the weights
   ([Load balancing](#load-balancing)).
4. **Groups** (DeepSeek-V3): with `num_groups > 1`, the experts split into
   `num_groups` groups, each scored by the sum of its best two experts, and only the
   best `top_k_groups` groups may be picked from.
5. **Top-k**: each token keeps its `top_k` experts by the choice scores.
6. **Weights**: the picked experts' scores, divided by their sum when
   `normalize_top_k=true` (the default), then multiplied by `routed_scaling_factor`.

Scores and the top-k are in float32, whatever dtype the run computes in.

DeepSeek-V3 routes by sigmoid scores within 4 of 8 groups, and scales the weights by
2.5 (`examples/architectures/deepseek_v3.bux`):

```bux
experts = moe(num_experts=256, top_k=8, num_shared_experts=1, router=sigmoid,
              routed_scaling_factor=2.5, num_groups=8, top_k_groups=4, load_balancing=aux_free,
              aux_loss_coef=1e-4, aux_loss_scope=sequence,
              expert=gated_mlp(intermediate_size=2048, activation=swish))
```

Jamba keeps the top two of its softmax as they are, without renormalizing them
(`examples/architectures/jamba.bux`):

```bux
experts = moe(num_experts=16, top_k=2, normalize_top_k=false, aux_loss_coef=1e-3,
              expert=dense)
```

:::{note}
`num_groups` must divide `num_experts`, `top_k_groups` be at most `num_groups`, and
the `top_k_groups` groups must hold at least `top_k` experts; `bux check` says so
otherwise.
:::

## Shared experts

`num_shared_experts=n` adds `n` experts that every token goes through, with weight 1
(DeepSeekMoE). They run as one MLP with `n` times the expert's hidden units, in the
shape of `shared_expert`, or of `expert` when `shared_expert` is `none`. The settings
after `num_shared_experts` need some: with none, setting them is an error.

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `num_shared_experts` | `0` | experts every token goes through | DeepSeekMoE |
| `shared_expert` | `none` | their MLP; `none` is `expert`'s shape | Nemotron 3 |
| `shared_expert_gate` | `false` | a sigmoid gate on their output, projected from the token | Qwen2-MoE, Qwen3-Next |
| `shared_expert_sink` | `false` | the router scores them too, and normalizes their weights with the top-k | Inkling |

Qwen3-Next gates its shared expert (`examples/architectures/qwen3_next.bux`):

```bux
experts = moe(num_experts=512, top_k=10, num_shared_experts=1, shared_expert_gate=true,
              aux_loss_coef=1e-3, expert=gated_mlp(intermediate_size=512, activation=swish))
```

With `shared_expert_sink=true`, the router has a column for each shared expert. Each
shared expert's share of the hidden units is weighted by its score, normalized
together with the top-k's (`examples/architectures/inkling/inkling.bux`):

```bux
experts = moe(num_experts=256, top_k=6, num_shared_experts=2, shared_expert_sink=true,
              router=sigmoid, routed_scaling_factor=8, load_balancing=aux_free,
              expert=gated_mlp(intermediate_size=3072, activation=swish))
```

Command A+ runs 4 shared experts and averages them with the routed ones by scaling
the whole layer with the algebra
(`examples/architectures/cohere/command_a_plus_05_2026.bux`):

```bux
experts = moe(num_experts=128, top_k=8, num_shared_experts=4, router=sigmoid,
              load_balancing=aux_free,
              expert=gated_mlp(intermediate_size=4096, activation=swish)) * 0.5
```

## LatentMoE

`latent_size=l` runs the routed experts in a narrower space (Nemotron 3): the tokens
are projected down to `l` before the experts, and their sum back up to the width
after. The router and the shared experts see the full width.

| Setting | What it normalizes | From |
|---------|-------------------|------|
| `latent_input_norm` | the tokens, after the down projection | Muse Spark |
| `latent_output_norm` | the routed sum, before the up projection | Kimi K3 |
| `expert_norm` | each routed expert's output | Muse Spark |

`expert_norm` works without `latent_size` too. Nemotron 3 Ultra's 512 ReLU² experts
work in a 2048-wide latent, and its shared expert at the full 8192
(`examples/architectures/nemotron/nemotron_3_ultra_550b_a55b.bux`):

```bux
experts = moe(num_experts=512, top_k=22, num_shared_experts=1, router=sigmoid,
              routed_scaling_factor=5.0, load_balancing=aux_free, latent_size=2048,
              expert=mlp(intermediate_size=5120, activation=relu2),
              shared_expert=mlp(intermediate_size=10240, activation=relu2))
```

Muse Spark normalizes on the way in and after every expert
(`examples/architectures/muse/muse_spark_open_816b_a42b.bux`):

```bux
experts = moe(num_experts=256, top_k=8, load_balancing=aux_free,
              router=sigmoid, router_logit_scale=logit_scale,
              latent_size=4096, latent_input_norm=norm, expert_norm=rescale,
              expert=gated_mlp(intermediate_size=4096, activation=swish))
```

:::{tip}
A macro sizes each layer on its own. Nemotron 3.5 Bolt resizes every MoE layer
(`examples/architectures/nemotron/nemotron_3.5_bolt_74b_a9b.bux`):

```bux
E(n, width) = [norm experts(num_experts=n, expert.intermediate_size=width)]

model = embed
        M E(240, 2048) M E(240, 1536) M E(240, 2048) M A
        ...
```
:::

(moe-load-balancing)=
## Load balancing

You balance the experts' load on the layer itself, by a loss, a bias or both.
DeepSeek-V3's, as amendments of an `experts` layer:

```bux
experts.load_balancing = aux_free
experts.aux_loss_coef  = 1e-4
experts.aux_loss_scope = sequence
```

| Setting | Default | What it does |
|---------|---------|--------------|
| `load_balancing` | `aux_loss` | `aux_loss`: the balance loss alone; `aux_free`: also a bias per expert; `smebu`: the bias moved by SMEBU |
| `aux_loss_coef` | `1e-2`, or `0` with a bias | the weight of each layer's balance loss |
| `aux_loss_scope` | `batch` | `batch`: over the whole batch (Switch, Qwen); `sequence`: per sequence, then averaged (DeepSeek-V3) |
| `bias_update_rate` | `1e-3` | how far `aux_free`'s bias moves per step; SMEBU's largest move |
| `center_bias` | `false` | `aux_free`: subtract the biases' mean after every move (Trinity Nano and Mini) |
| `bias_softclamp` | `2` | SMEBU's $\kappa$, how soon its moves saturate |
| `bias_momentum` | `0.5` | SMEBU's $\beta$, the decay of its moves' running average |

### The balance loss

Each layer adds `aux_loss_coef` times

$$
\mathcal{L}_\text{balance} = E \sum_{e=1}^{E} f_e P_e,
$$

where $f_e$ is the fraction of the top-k picks that went to expert $e$, and $P_e$ its
mean score, normalized over the experts (Switch; DeepSeek-V3, eqs. 17 to 20). It is 1
when both are uniform. With `aux_loss_scope=sequence` it is taken over each sequence
and then averaged: the complementary sequence-wise loss of DeepSeek-V3.

Every layer adds its own loss, as in Megatron-LM, and the log's `balance_loss` is
their sum.

:::{note}
transformers' `load_balancing_loss_func` is `top_k` times this, taken over all layers
at once, and it counts tokens, not picks. So its coefficient $c$ is about
$c \cdot \text{top\_k} / \text{layers}$ in bux.
:::

### The aux-free bias

`load_balancing=aux_free` adds a bias per expert to the scores that pick the top-k,
not to the weights (DeepSeek-V3). After every training step, each expert's bias moves
by `bias_update_rate`: up for the experts picked less than the mean over the step's
batch, down for those picked more. The bias counts the picks of all of a step's
microbatches and moves once, after them. It is state, not a parameter: no optimizer
trains it, and checkpoints keep it.

`aux_free` sets the loss's weight to 0 unless you set `aux_loss_coef`, so
DeepSeek-V3's file sets both. With `center_bias=true`, the biases' mean is subtracted
after every move, so that they stay centered on 0, as in Trinity Nano and Mini
(`examples/architectures/arcee/trinity_mini.bux`).

### SMEBU

`aux_free` moves every bias by the whole rate, by the sign of its expert's imbalance,
so a bias near its balance point keeps stepping over it. `load_balancing=smebu`,
Trinity Large's soft-clamped momentum expert bias update, moves the same bias by a
step that shrinks as an expert's load $n_e$ nears the mean $\bar n$, smoothed over
steps:

$$
\begin{aligned}
\Delta_e &= \lambda \tanh\!\left(\kappa\, \frac{\bar n - n_e}{\bar n}\right),
  \qquad \Delta_e \leftarrow \Delta_e - \tfrac{1}{E} \textstyle\sum_j \Delta_j, \\
m_e &\leftarrow \beta\, m_e + (1 - \beta)\, \Delta_e, \qquad b_e \leftarrow b_e + m_e,
\end{aligned}
$$

with $\lambda$ the `bias_update_rate`, $\kappa$ the `bias_softclamp` and $\beta$ the
`bias_momentum` (Trinity's report, eqs. 23 to 28). The momentum $m$ is state beside
the bias, and checkpoints keep it. Trinity Large balances by SMEBU and by a
sequence-wise loss of $10^{-4}$ (`examples/architectures/arcee/trinity_large.bux`):

```bux
experts = moe(num_experts=256, top_k=4, num_shared_experts=1, router=sigmoid,
              routed_scaling_factor=2.448, load_balancing=smebu,
              bias_update_rate=5e-4, bias_softclamp=2, bias_momentum=0.5,
              aux_loss_coef=1e-4, aux_loss_scope=sequence,
              expert=gated_mlp(intermediate_size=3072, activation=swish))
```

Eval neither adds the loss nor moves the bias.

(random-logits)=
## Random logits for benchmarks

For benchmarks, `moe(..., random_logits=true)` routes each training forward by
logits drawn from $N(0, 1)$, from its own random stream like dropout's, instead of
the router's. Megatron-Core's `moe_router_force_load_balancing` does the same for all
its MoE benchmarks: every expert takes an even share of the picks in expectation, so
a capacity drops few picks, whatever the data. On random tokens, a router's own
routing drifts uneven (a balance loss of 1.4 to 1.8 per layer on Qwen3-30B-A3B), and
a benchmark would then measure the drift instead of the method.

- The router still runs, and its logits take the drawn ones' gradient (a
  straight-through estimator).
- Eval routes by the router.
- It changes what a run computes, so it is part of the run's id.

`examples/benchmarks/megatron.bux` sets it:

```bux
use qwen3_30b_a3b

experts.random_logits = true
```

## How experts run

The router and the balancing see the whole batch. A method of `methods.moe` then
runs the experts on their picks: `sorted`, the default, sorts each device's picks by
expert and runs them by grouped GEMMs, dropping none; `dense` runs every expert on
every token, as a reference; the others split the experts over the `expert` axis by
expert parallelism. The methods, capacities and their trade-offs are in
[Experts](../parallelism/experts.md); the grouped GEMMs' kernels, and the experts in
MXFP8, in [GPU kernels](../performance/kernels.md).

- Precision: the routed experts multiply in `precision.roles.experts`, LatentMoE's
  projections in `roles.latent`, the shared experts in `roles.mlp`; the router stays
  in the compute dtype ([Precision](../training/precision.md)).
- Muon orthogonalizes the stacked experts one expert at a time, and an expert's up and
  gate separately ([Optimizers](../training/optimizers.md)).
- Remat: a policy can keep `experts`, the experts' grouped products
  ([Remat](../parallelism/remat.md)).
