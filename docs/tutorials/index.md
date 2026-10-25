# Tutorials

Complete walkthroughs, each a numbered list of steps you can follow on a laptop's
CPU in a few minutes. Every command was run as written, and its output is pasted from
that run, trimmed where it repeats. Do them in order the first time: each builds on
the files of the one before.

| tutorial | what you do |
|----------|-------------|
| [Your first architecture](first-architecture.md) | Write GPT-2 from scratch, evolve it into a Llama-style model and a hybrid, and read the errors. |
| [Train on a laptop](train-on-a-laptop.md) | Turn a model into a run, train it, read its log and metrics, stop it and resume it, and evaluate it. |
| [An ablation sweep](ablation-sweep.md) | Use an architecture and a recipe by name, sweep with alternatives and labels, launch it, and compare the runs. |
| [Your own data](your-own-data.md) | Make a Parquet file of text, tokenize and pack it, hold out a split for eval, and mix sources. |
| [Fine-tuning on chat](fine-tuning-chat.md) | Train on conversations in the OpenAI format, with the loss on the assistant's replies. |
| [Add a layer](add-a-layer.md) | Add a new layer to bux end to end: its config, its builder, its use in a `.bux` file, hover docs and a test. |
| [Custom training](custom-training.md) | Replace an optimizer by registry, add a hook, a sink and a loss term, and drive a `Trainer` from a script. |
| [Scale out](scale-out.md) | Split one run six ways over a CPU's fake devices, then see what changes on a GPU node and on many nodes. |

## Before you start

Install bux ([Installation](../get-started/installation.md)) and work from the root
of your checkout, with the environment active:

```sh
cd bux
source .venv/bin/activate      # or prefix every bux command with `uv run`
mkdir -p mine                  # the tutorials' files go here
```

The tutorials keep their `.bux` files in `mine/`, inside the checkout, so that
`use llama3_8b` finds bux's examples: modules are found in the using file's folder,
then under the folders listed in `[tool.bux] path` of the nearest `pyproject.toml`,
which is `examples/` in bux's own. Runs go to `runs/` and prepared data to
`data/packed/`, both relative to where you run `bux`; git ignores `runs/` and `data/`.

:::{tip}
The runs here are small enough that a laptop trains them in seconds to a minute or
two. On a machine with GPUs, `JAX_PLATFORMS=cpu` reproduces the outputs shown.
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
```
