# Reference

Look up any layer, setting, command or function here. Most of these pages are
generated from the code each time the docs build, so they can't drift from it:

- the language reference comes from the configs' docstrings and field comments (the
  same text `bux-lsp` shows on hover);
- the registries list what each registry holds;
- the command line comes from `bux --help`;
- the Python API comes from the docstrings.

The environment variables are listed by hand.

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

`bux check`, `plan`, `prepare`, `train`, `launch`, `results`, `render`, `fmt`,
`json`, `export` and `studio`.
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
