# bux

bux is a lightweight JAX framework for running architecture and data ablations fast.
It is a research tool, not a production system: the goal is to get from an idea to a
trained, comparable run in as few lines as possible. Experiments are written in the bux
DSL. A `.bux` file describes an architecture as an algebra over the residual stream
(`[f]` is `x + f(x)`, so `[norm attn] [norm ffn]` is a pre-norm block), and the same
file says how to train it: data, optimizer, schedule, and `|` sweeps over any of it,
architecture included. `src/bux` compiles these files to a typed IR and trains them
with Flax NNX.

## Philosophy

- **Research, not production.** We use the newest, fanciest libraries and paradigms
  whenever they make the code shorter or safer: Python 3.14, JAX, Flax NNX, optax,
  orbax, pydantic configs, jaxtyping shapes checked by beartype, uv, ruff, ty. There is
  no backwards compatibility to keep and no reason to play it safe on dependencies.
- **Hackable and moddable.** It should be quick to swap, add, or change any piece to
  test an idea. A new layer is a function under `@register("name")`, and a variant of
  an optimizer is `@optimizers.register("adamw")`, not a fork. If something is hard to
  replace, treat that as a bug.
- **Small pieces.** No long functions and no walls of text, in code, comments, or docs.
  Write short functions with clear names, and comment only what the code can't say.
- **The DSL is the interface.** If an experiment can be written in `.bux`, write it
  there. Extend the language or a registry instead of writing Python for one
  experiment. See `docs/language/`, `docs/training/`, `docs/data/`, and
  `docs/parallelism/`.

## Documentation

The docs are a Sphinx site in `docs/` (MyST Markdown, published on Read the Docs);
`docs/development/index.md` is the guide to writing them. A change is not done until
its documentation is, in the same commit:

- **Find the page.** The table in `docs/development/index.md` ("Keeping the docs in
  sync") maps each part of the code to its pages: `docs/language/` for the language,
  `docs/architectures/` for layers, `docs/training/`, `docs/data/`,
  `docs/parallelism/` (a page per method), `docs/performance/` (kernels, tuning,
  benchmarks), `docs/reference/` (CLI, environment variables), `docs/tutorials/`.
  Update what your change makes wrong, and add what it makes new.
- **Configs document themselves.** Every config in `bux/config.py` and
  `bux/train/config.py` has a docstring, and every field a comment above or beside
  it: they generate the language reference and the editor's hover
  (`bux.lang.reference`). A new field without a comment is undocumented everywhere.
- **Docstrings are the API reference.** Google style (`Args:`, `Returns:`,
  `Raises:`) for what users call or extend, one accurate line for helpers; valid
  reStructuredText (``code``, `name` links, `::` before a code block). A function
  registered under a name says what that entry does in its first paragraph: it is
  its row in the registries' reference.
- **Never edit generated output** (`docs/_generated/`,
  `docs/reference/python/generated/`): change the code it comes from.
- **Measurements** go to `docs/performance/` or the page of what was measured, with
  the hardware, setup and date.
- **`make docs` passes**: it fails on any warning, such as a link to a heading that
  was renamed or a docstring that no longer renders. CI runs it.

## Documentation

The docs are a Sphinx site in `docs/` (MyST Markdown, published on Read the Docs);
`docs/development/index.md` is the guide to writing them. A change is not done until
its documentation is, in the same commit:

- **Find the page.** The table in `docs/development/index.md` ("Keeping the docs in
  sync") maps each part of the code to its pages: `docs/language/` for the language,
  `docs/architectures/` for layers, `docs/training/`, `docs/data/`,
  `docs/parallelism/` (a page per method), `docs/performance/` (kernels, tuning,
  benchmarks), `docs/reference/` (CLI, environment variables), `docs/tutorials/`.
  Update what your change makes wrong, and add what it makes new.
- **Configs document themselves.** Every config in `bux/config.py` and
  `bux/train/config.py` has a docstring, and every field a comment above or beside
  it: they generate the language reference and the editor's hover
  (`bux.lang.reference`). A new field without a comment is undocumented everywhere.
- **Docstrings are the API reference.** Google style (`Args:`, `Returns:`,
  `Raises:`) for what users call or extend, one accurate line for helpers; valid
  reStructuredText (``code``, `name` links, `::` before a code block). A function
  registered under a name says what that entry does in its first paragraph: it is
  its row in the registries' reference.
- **Never edit generated output** (`docs/_generated/`,
  `docs/reference/python/generated/`): change the code it comes from.
- **Measurements** go to `docs/performance/` or the page of what was measured, with
  the hardware, setup and date.
- **`make docs` passes**: it fails on any warning, such as a link to a heading that
  was renamed or a docstring that no longer renders. CI runs it.
