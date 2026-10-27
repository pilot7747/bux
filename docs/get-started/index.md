# Get started

bux is a JAX framework for architecture and data ablations. You write a language
model as an algebra over the residual stream in a short `.bux` file, say in the same
file how to train it, and turn any line into a sweep with `|`. bux checks every
setting before anything runs, then trains with Flax NNX on a laptop's CPU or on many
nodes of GPUs. Here is a complete experiment, two runs of a tiny Llama-style model:

```bux
d_model    = 64
vocab_size = 512

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=4 * d_model, activation=swish)
block = [norm attn] [norm ffn]              # [f] is x + f(x): a pre-norm block
model = embed block^2 norm head(tied=true)

data      = synthetic
optimizer = adamw(lr=3e-3 | 1e-2)           # two runs
schedule  = wsd(warmup=10)
seq_len   = 64
batch     = 8
steps     = 50
```

## Who it is for

bux is built for research: the shortest path from an idea to a trained, comparable
run. Reach for it when you want to:

- compare architectures: attention variants, linear attention and state-space
  models, mixtures of experts, residual wiring, normalization placement;
- compare training recipes: optimizers, schedules, precision, data mixtures and
  packings;
- keep each comparison to the few lines that differ, not a fork of a training script;
- take the winner from a laptop to a multi-node GPU cluster without rewriting it.

bux trains models; it does not serve them. Export a trained model to transformers
or vLLM for inference ([Exporting](../export/index.md)). Every model trains from its
own initialization: bux does not load weights trained by other frameworks.

## In this section

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

Python 3.14 and uv, the optional extras, GPUs, the Docker image, and a check that
everything works.
:::

:::{grid-item-card} {octicon}`rocket` Quickstart
:link: quickstart
:link-type: doc

Ten minutes from a fresh clone to a trained model on a laptop, and a sweep planned.
:::

:::{grid-item-card} {octicon}`light-bulb` Concepts
:link: concepts
:link-type: doc

The handful of ideas everything else builds on, each with a short example and a
link to its full page.
:::
::::

Once you have the basics, the [tutorials](../tutorials/index.md) walk through
complete tasks step by step.

```{toctree}
:hidden:
:maxdepth: 1

installation
quickstart
concepts
```
