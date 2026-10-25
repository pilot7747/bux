# Reference

The pages here are generated from the code as the documentation builds, so they are
always the code's: the language reference from the configs' docstrings and field
comments (the same text `bux-lsp` shows on hover), the registries from what each one
holds, and the Python API from the docstrings.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Language reference
:link: language/index
:link-type: doc

Every layer and setting a `.bux` file can call, with every field's type, default and
meaning.
:::

:::{grid-item-card} Command line
:link: cli
:link-type: doc

`bux check`, `plan`, `prepare`, `train`, `launch`, `render`, `fmt` and `json`.
:::

:::{grid-item-card} Environment variables
:link: environment
:link-type: doc

What bux, JAX and XLA read from the environment.
:::

:::{grid-item-card} Python API
:link: python/index
:link-type: doc

The compiler, the Flax NNX backend, the trainer, parallelism and the data pipeline.
:::
::::

```{toctree}
:hidden:

language/index
cli
environment
python/index
```
