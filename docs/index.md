---
html_theme.sidebar_secondary.remove: true
---

# bux

**bux** is a lightweight JAX framework for running architecture and data ablations
fast. You describe a language model as an algebra over the residual stream in a
compact `.bux` file, say in the same file how to train it, and turn any line of it
into a sweep with `|`. bux checks every setting before anything runs, at the line that
wrote it, and trains the result with Flax NNX on anything from a laptop's CPU to many
nodes of GPUs.

```bux
use llama3_8b                      # an architecture, found by name
use fineweb_wsd                    # a recipe: data mix, optimizer, schedule

d_model = 768
model   = embed block^12 norm head(tied=true)

attn.num_kv_heads = mha: 12 | gqa: 4 | mqa: 1   # llama3_8b's block picks this up
optimizer.lr      = 1e-3 | 3e-3                 # 3 × 2 = 6 runs
```

```console
$ bux plan examples/experiments/attention_ablation.bux     # the 6 runs, their sizes and budgets
$ bux launch examples/experiments/attention_ablation.bux   # train them, a GPU each
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Get started
:link: get-started/index
:link-type: doc

Install bux, train a model on your laptop in a minute, and learn the four ideas the
rest builds on.
:::

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: tutorials/index
:link-type: doc

Step by step: your first architecture, a sweep, your own data, a new layer, a custom
training loop, and scaling out.
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

Runs, optimizers, precision, launching on a cluster, resuming, and the trainer's
replaceable blocks.
:::

:::{grid-item-card} {octicon}`database` Data
:link: data/index
:link-type: doc

Documents to packed rows: text and chat in Parquet, tokenized and packed once, on
your cores or on Spark.
:::

:::{grid-item-card} {octicon}`server` Parallelism
:link: parallelism/index
:link-type: doc

One mesh of six axes: FSDP, tensor, expert, context and pipeline parallelism, and any
mix of them.
:::

:::{grid-item-card} {octicon}`zap` Performance
:link: performance/index
:link-type: doc

The tuning guide, GPU kernels, XLA's options, profiling, and benchmarks against
Megatron-Bridge.
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
reference/index
development/index
```
