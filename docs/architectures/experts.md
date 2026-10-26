# Mixture of experts

This page covers `moe`, bux's token-choice mixture of experts: how its router picks
experts, shared experts, LatentMoE, and how to balance the experts' load, by a loss
or a bias. Read it to configure an MoE layer, or to reproduce a model's routing. How
the experts run on devices, and in which kernels, is parallelism: see
[Experts](../parallelism/experts.md).

```bux
# Qwen3-30B-A3B: 128 experts of 768 hidden units, the top 8 a token
experts = moe(num_experts=128, top_k=8, aux_loss_coef=1e-3,
              expert=gated_mlp(intermediate_size=768, activation=swish))

block = [norm attn] [norm experts]
model = embed block^48 norm head
```

`moe` is a channel mixer: write it where an MLP would go. Every field is in its
reference entry, {bux}`moe`.

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

With `bias=true` in `expert`, every routed expert has biases of its own on its up,
gate and down projections. They are not weight-decayed. gpt-oss-120b's experts have
them, and are SwiGLUs of its own: a `quick_gelu` gate, $g\,\sigma(1.702\, g)$, capped
at 7, times `up` clipped to $\pm 7$ plus 1 (`up_offset`), with a bias on the router's
logits too (`examples/architectures/gpt_oss_120b.bux`):

```bux
experts = moe(num_experts=128, top_k=4, router_bias=true,
              expert=gated_mlp(intermediate_size=2880, activation=quick_gelu,
                               bias=true, limit=7.0, up_offset=1.0))
```

:::{note}
cuDNN's fused SwiGLU kernels take no biases, offset, clamp or soft-cap, so experts
with any of them run by bux's grouped GEMMs ([SwiGLU fused, by
cuDNN](../parallelism/experts.md#swiglu-fused-by-cudnn)). Every `methods.moe` adds
an expert's biases where it runs its kernels, `nccl_ep` too: on 8 B300s, gpt-oss's
experts by each method compute what `sorted` does on one GPU, the layer within 7.7e-7
and 12 steps' losses within 1.2e-4.
:::

## Routing

The router is a linear layer of the width to the experts, with a bias, learned, only
under `router_bias=true` (gpt-oss). It differs from `aux_free`'s bias (below), which
moves only which experts a token picks, not their weights. In order, for each token:

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
   `normalize_top_k=true` (the default), then times `routed_scaling_factor`.

Scores and the top-k are in float32, whatever the run computes in.

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
`num_groups` must divide `num_experts`, and the `top_k_groups` groups must hold at
least `top_k` experts; the layer raises otherwise.
:::

## Shared experts

`num_shared_experts=n` adds `n` experts that every token goes through, with weight 1
(DeepSeekMoE). They run as one MLP of `n` times the expert's hidden units, of
`shared_expert`'s shape, or `expert`'s when it is `none`.

| Setting | Default | What it does | From |
|---------|---------|--------------|------|
| `num_shared_experts` | `0` | experts every token goes through | DeepSeekMoE |
| `shared_expert` | `none` | their MLP; `none` is `expert`'s shape | Nemotron 3 |
| `shared_expert_gate` | `false` | a sigmoid gate, a projection of the token, on their output | Qwen2-MoE, Qwen3-Next |
| `shared_expert_sink` | `false` | the router scores them too, and normalizes their weights with the top-k | Inkling |

Qwen3-Next gates its shared expert (`examples/architectures/qwen3_next.bux`):

```bux
experts = moe(num_experts=512, top_k=10, num_shared_experts=1, shared_expert_gate=true,
              aux_loss_coef=1e-3, expert=gated_mlp(intermediate_size=512, activation=swish))
```

With `shared_expert_sink=true`, the router has a column for each shared expert, and
each one's share of the hidden units is weighted by its score, normalized together
with the top-k's (`examples/architectures/inkling/inkling.bux`):

```bux
experts = moe(num_experts=256, top_k=6, num_shared_experts=2, shared_expert_sink=true,
              router=sigmoid, routed_scaling_factor=8, load_balancing=aux_free,
              expert=gated_mlp(intermediate_size=3072, activation=swish))
```

Command A+ runs 4 shared experts and averages them with the routed ones by scaling
the whole layer, with the algebra
(`examples/architectures/cohere/command_a_plus_05_2026.bux`):

```bux
experts = moe(num_experts=128, top_k=8, num_shared_experts=4, router=sigmoid,
              load_balancing=aux_free,
              expert=gated_mlp(intermediate_size=4096, activation=swish)) * 0.5
```

## LatentMoE

`latent_size=l` runs the routed experts in a narrower space (Nemotron 3): the tokens
are projected down to `l` before the experts and their sum back up to the width
after. The router and the shared experts see the full width.

| Setting | What it normalizes | From |
|---------|-------------------|------|
| `latent_input_norm` | the tokens, after the down projection | Muse Spark |
| `latent_output_norm` | the routed sum, before the up projection | Kimi K3 |
| `expert_norm` | each routed expert's output | Muse Spark |

`expert_norm` works without `latent_size` too. Nemotron 3 Ultra's 512 ReLU² experts
work in a 2048-wide latent, its shared expert at the full 8192
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

MoE load balancing is set on the layer, by a loss, a bias or both. DeepSeek-V3's,
as amendments of an `experts` layer:

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
| `bias_update_rate` | `1e-3` | how far `aux_free`'s bias moves a step; SMEBU's largest move |
| `center_bias` | `false` | `aux_free`: take the biases' mean off them after every move (Trinity Nano and Mini) |
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

Every layer adds its own, as in Megatron-LM, and the log's `balance_loss` is their
sum.

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
microbatches and moves once, after them. The bias is state, not a parameter: no
optimizer trains it, and checkpoints keep it.

`aux_free` sets the loss's weight to 0 unless you set `aux_loss_coef`, so
DeepSeek-V3's file sets both. With `center_bias=true`, the biases' mean is taken off
them after every move, so that they stay centered on 0: Trinity Nano's and Mini's
(`examples/architectures/arcee/trinity_mini.bux`).

### SMEBU

`aux_free`'s sign moves every expert by the whole rate, so a bias near its balance
keeps stepping over it. `load_balancing=smebu`, the soft-clamped momentum expert bias
update of Trinity Large, moves the same bias by a step that shrinks as an expert's
load $n_e$ nears the mean $\bar n$, smoothed over steps:

$$
\begin{aligned}
\Delta_e &= \lambda \tanh\!\left(\kappa\, \frac{\bar n - n_e}{\bar n}\right),
  \qquad \Delta_e \leftarrow \Delta_e - \tfrac{1}{E} \textstyle\sum_j \Delta_j, \\
m_e &\leftarrow \beta\, m_e + (1 - \beta)\, \Delta_e, \qquad b_e \leftarrow b_e + m_e,
\end{aligned}
$$

with $\lambda$ the `bias_update_rate`, $\kappa$ the `bias_softclamp` and $\beta$ the
`bias_momentum` (Trinity's report, eqs. 23 to 28). The momentum $m$ is state beside
the bias, which checkpoints keep. Trinity Large balances by it and by a sequence-wise
loss of $10^{-4}$ (`examples/architectures/arcee/trinity_large.bux`):

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
logits drawn from $N(0, 1)$, a random stream of its own as dropout's, in place of the
router's. Megatron-Core's `moe_router_force_load_balancing` does the same for all its
MoE benchmarks: every expert takes an even share of the picks in expectation, so a
capacity drops few, whatever the data.

- The router still runs, and its logits take the drawn ones' gradient (a
  straight-through estimator).
- Eval routes by the router.
- It changes what a run computes, so it is in the run's id.

`examples/benchmarks/megatron.bux` sets it:

```bux
use qwen3_30b_a3b

experts.random_logits = true
```

:::{dropdown} Measurements: Qwen3-30B-A3B on 8 B300s
On its random tokens, the router's own routing drifts uneven: a balance loss of 1.4
to 1.8 a layer, and `nccl_ep` at a capacity of 1.25 dropped 10 to 15% of the picks.
By random logits the balance loss is 1.00 and it drops none, and it runs 7% faster on
8 B300s (29.6k tokens a second a GPU against 27.7k); `sorted` 1.4% faster (31.8k
against 31.4k).
:::

## How experts run

The router and the balancing run in the global view of the batch; a method of
`methods.moe` runs the experts on their picks:

| Method | What it does |
|--------|--------------|
| `sorted` (default) | each device sorts its own tokens' picks by expert and runs every expert on its own rows by grouped GEMMs, dropping none |
| `dense` | every expert on every token, weighted by the router: the reference |
| `all_gather`, `all_to_all`, `ragged_all_to_all`, `ragged_tokens`, `nccl_ep` | expert parallelism over the `expert` axis |

Expert parallelism, capacities and the methods' trade-offs:
[Experts](../parallelism/experts.md). The grouped GEMMs' kernels, and the experts in
MXFP8: [GPU kernels](../performance/kernels.md).

- Precision: the routed experts multiply in `precision.roles.experts`, LatentMoE's
  projections in `roles.latent`, the shared experts in `roles.mlp`; the router stays
  in the compute dtype ([Precision](../training/precision.md)).
- Muon orthogonalizes the stacked experts one expert at a time, and an expert's up and
  gate apart ([Optimizers](../training/optimizers.md)).
- Remat: a policy can keep `experts`, the experts' grouped products
  ([Remat](../parallelism/remat.md)).
