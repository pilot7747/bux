---
html_theme.sidebar_secondary.remove: true
---

# bux

**bux** is a JAX framework for architecture and data ablations. You write a language
model as an algebra over the residual stream in a short `.bux` file, say in the same
file how to train it, and turn any line into a sweep with `|`. bux checks every
setting before anything runs, then trains with Flax NNX on anything from a laptop CPU
to multi-node GPU clusters, at Megatron-class speed.

```bux
use llama3_8b                      # an architecture, found by name
use fineweb_wsd                    # a recipe: data mix, optimizer, schedule

d_model = 768
model   = embed block^12 norm head(tied=true)

attn.num_heads    = 12                          # llama3_8b's block picks these up
attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1
optimizer.lr      = 1e-3 | 3e-3                 # 3 × 2 = 6 runs
```

That is most of `examples/experiments/attention_ablation.bux`: three attentions
crossed with two learning rates. Plan the six runs, then train them:

```console
$ bux plan examples/experiments/attention_ablation.bux     # the 6 runs, their sizes and budgets
$ bux launch examples/experiments/attention_ablation.bux   # train them, one GPU each
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Get started
:link: get-started/index
:link-type: doc

Install bux, train a model on your laptop in a minute, and learn the handful of
ideas everything else builds on.
:::

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: tutorials/index
:link-type: doc

Step by step: your first architecture, a sweep, your own data, a new layer, a custom
training loop, scaling out, and a full 2022-vs-2026 architecture study.
:::

:::{grid-item-card} {octicon}`code` The `.bux` language
:link: language/index
:link-type: doc

Operators, skip connections, macros, sharing, modules, amendments and sweeps.
:::

:::{grid-item-card} {octicon}`stack` Architectures
:link: architectures/index
:link-type: doc

Attention, linear attention, state-space models, mixtures of experts, and a zoo of
transcribed models from GPT-2 to 2026's releases.
:::

:::{grid-item-card} {octicon}`play` Training
:link: training/index
:link-type: doc

Runs, optimizers, precision, launching on a cluster, observing, hyperparameter
search, and the trainer's replaceable parts.
:::

:::{grid-item-card} {octicon}`database` Data
:link: data/index
:link-type: doc

From Parquet files of text or chat to packed rows: tokenized and packed once, on
your cores or on Spark.
:::

:::{grid-item-card} {octicon}`server` Parallelism
:link: parallelism/index
:link-type: doc

One mesh of six axes: FSDP, tensor, expert, context and pipeline parallelism, and
any mix of them, from one GPU to many nodes.
:::

:::{grid-item-card} {octicon}`zap` Performance
:link: performance/index
:link-type: doc

The tuning guide, GPU kernels, XLA's options, profiling, and benchmarks against
Megatron-Bridge.
:::

:::{grid-item-card} {octicon}`package` Exporting
:link: export/index
:link-type: doc

Trained models as transformers code with their weights, checked against bux, and
served by vLLM.
:::

:::{grid-item-card} {octicon}`book` Reference
:link: reference/index
:link-type: doc

Every layer and setting of the language, the command line, and the Python API.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

get-started/index
tutorials/index
language/index
architectures/index
training/index
data/index
parallelism/index
performance/index
export/index
reference/index
development/index
```
