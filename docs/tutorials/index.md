# Tutorials

Hands-on walkthroughs, from a first `.bux` file to a scaling study on a node of GPUs.
Each is a short sequence of numbered steps. Every command was run as written, and its
output is pasted from that run, trimmed where it repeats.

The first eight run on a laptop's CPU, each in a few minutes. The last needs a node
of 8 GPUs. Do them in order the first time: later tutorials reuse the files that
earlier ones create.

| tutorial | what you do |
|----------|-------------|
| [Your first architecture](first-architecture.md) | Write GPT-2 in `.bux`, turn it into a Llama-style model and a hybrid, and read the errors. |
| [Train on a laptop](train-on-a-laptop.md) | Train a tiny model on your CPU, read its log and metrics, then stop it and resume it. |
| [An ablation sweep](ablation-sweep.md) | Borrow Llama 3's block and a recipe by name, sweep attention and learning rate, and compare six runs. |
| [Your own data](your-own-data.md) | Train on Parquet text of your own: tokenize, pack, hold out an eval split, and mix sources. |
| [Fine-tuning on chat](fine-tuning-chat.md) | Train on OpenAI-format conversations, with the loss on the assistant's replies only. |
| [Add a layer](add-a-layer.md) | Add a token-shift layer to bux: its config, its builder, hover docs and a test. |
| [Custom training](custom-training.md) | Swap an optimizer, add a hook, a sink and a loss term, and drive a `Trainer` from Python. |
| [Scale out](scale-out.md) | Split one run six ways over fake CPU devices, then see what changes on GPU nodes. |
| [Old against new at the same serving cost](old-vs-new.md) | Compare a 2022 7B with a 2026 hybrid MoE that serves as fast, by tuned scaling ladders. Needs 8 GPUs. |

## Before you start

Install bux ([Installation](../get-started/installation.md)), then work from the root
of your checkout with its environment active:

```sh
cd bux
source .venv/bin/activate      # or prefix every bux command with `uv run`
mkdir -p mine                  # the tutorials' files go here
```

The tutorials keep their `.bux` files in `mine/`, inside the checkout. From there,
`use llama3_8b` finds bux's own examples, because the checkout's `pyproject.toml`
lists `examples/` as a module path ([Modules](../language/modules.md)). Runs go to
`runs/` and prepared data to `data/packed/`, both relative to where you run `bux`.
Git ignores both.

:::{tip}
The laptop runs train in seconds to a minute or two. On a machine with GPUs, set
`JAX_PLATFORMS=cpu` to reproduce the outputs shown.
:::

```{toctree}
:hidden:
:maxdepth: 1

first-architecture
train-on-a-laptop
ablation-sweep
your-own-data
fine-tuning-chat
add-a-layer
custom-training
scale-out
old-vs-new
```
