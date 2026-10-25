# Get started

bux is a lightweight JAX framework for running architecture and data ablations fast.
You write a language model as an algebra over the residual stream in a short `.bux`
file, say in the same file how to train it, and turn any line into a sweep with `|`.
bux checks every setting before anything runs and trains the result with Flax NNX, on
a laptop's CPU or on many nodes of GPUs.

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

bux is a research tool, not a production system. It suits you if you:

- compare architectures: attention variants, linear attention and state-space
  models, mixtures of experts, residual wiring, normalization placement;
- compare training recipes: optimizers, schedules, precision, data mixtures and
  packings;
- want the comparison to be a few lines that differ, not a fork of a training script.

It is not a serving stack, and it does not load weights trained by other frameworks:
every model trains from its own initialization.

## In this section

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

Python 3.14 and uv, the optional extras, GPUs, the Docker image, and checking that
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

The few ideas the rest of bux builds on, each with a short example and a link to its
full page.
:::
::::

When you have the basics, the [tutorials](../tutorials/index.md) walk through
complete tasks step by step.

```{toctree}
:hidden:
:maxdepth: 1

installation
quickstart
concepts
```
