# Contributing to bux

Thanks for helping. bux is a research tool built to be changed: new layers,
optimizers, kernels, data sources and zoo architectures are all welcome, and so are
bug reports and fixes to the docs.

## Reporting a bug

Open an [issue](https://github.com/pilot7747/bux/issues/new/choose) with the smallest
`.bux` file that shows the bug, the command you ran, and its full output. Say where it
ran: the output of `uv run python -c "import jax; jax.print_environment_info()"` covers
the versions and devices.

Report a security vulnerability privately instead: see the
[security policy](SECURITY.md).

## Proposing a change

For anything bigger than a fix, open an issue first, so we can agree on where the
change goes before you write it. bux has a few rules of thumb:

- **The DSL is the interface.** If an experiment can be written in `.bux`, extend the
  language or a registry rather than write Python for one experiment.
- **Register, don't fork.** A new layer is a function under `@register("name")`, and a
  variant of an optimizer is `@optimizers.register("adamw")`. If something is hard to
  replace, that is a bug worth fixing first.
- **Small pieces.** Short functions with clear names. Comment only what the code
  can't say.
- **Typed arrays.** Every function says what it takes and returns, and arrays say
  their dtype and shape: `Float[Array, "b t d"]`, not `jax.Array`.

[Adding a layer](https://bux.readthedocs.io/en/latest/architectures/adding-a-layer.html)
and [Extending the trainer](https://bux.readthedocs.io/en/latest/training/extending.html)
show where each kind of change goes.

## Setting up

```sh
git clone https://github.com/pilot7747/bux.git
cd bux
uv sync          # Python 3.14, the dev tools and the docs' Sphinx
```

The [development guide](https://bux.readthedocs.io/en/latest/development/index.html)
maps the code, lists the checks, and explains how the docs are built.

## Before you open a pull request

- `make lint typecheck` passes: ruff, including the Python blocks in the docs, and ty.
- The tests that cover your change pass, such as
  `uv run pytest tests/test_lang.py -q`. CI runs the whole suite over six runners.
- The docs say what your change does, in the same pull request.
  [Keeping the docs in sync](https://bux.readthedocs.io/en/latest/development/index.html#keeping-docs-in-sync)
  lists the pages for each part of the code.
- `make docs` passes: any warning fails it.
- A speed claim comes with its numbers: before and after, the hardware, the setup and
  the date.

Keep a pull request to one change, and say in its description what it does and why.
Coding agents find the same rules in [`AGENTS.md`](AGENTS.md).

## Code of conduct and license

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md). bux is
licensed under the [Apache License 2.0](LICENSE). By contributing, you agree that your
contributions are licensed under it too.
