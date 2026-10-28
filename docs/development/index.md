# Development

How to work on bux itself: set it up, run the checks CI runs, find your way around the
code, and keep the documentation in step with it. bux is meant to be changed: if
something is hard to replace, that is a bug. To report a bug or send a pull request,
see [CONTRIBUTING.md](https://github.com/pilot7747/bux/blob/main/CONTRIBUTING.md).

## Setting up

```sh
git clone https://github.com/pilot7747/bux.git
cd bux
uv sync          # Python 3.14, the dev tools and the docs' Sphinx
```

`uv sync` installs the `dev` and `docs` groups by default. On Linux with GPUs, add
`--extra cuda` for JAX on CUDA 13 and tokamax's kernels.

## The checks

`make` runs everything CI runs; each check also runs alone:

| Command | What it runs |
|---------|--------------|
| `make lint` | `ruff check .` and `ruff format --check .`, which also formats the ```` ```python ```` blocks of the docs |
| `make typecheck` | `ty check`, warnings as errors |
| `make test` | `pytest -q`, on 8 fake CPU devices |
| `make docs` | the documentation, as Read the Docs builds it: any warning fails it |
| `make` | all four |

The tests check every call's arrays against their annotations (`tests/conftest.py`
installs jaxtyping's import hook with beartype), so an array of the wrong dtype or
shape fails where it goes in. `tests/test_typing.py` checks that
every array is annotated: write `Float[Array, "b t d"]`, not `jax.Array`. Every
function says what it takes and returns (ruff's `ANN` rules).

The whole suite takes a while on a laptop; while you work, run the files that cover
what you changed, such as `uv run pytest tests/test_lang.py -q`. CI splits the suite
over six runners by a hash of each test's id: `BUX_SHARD=2/6 make test` runs the
third runner's share ([environment variables](../reference/environment.md#bux)).

## The code

| Path | What it is |
|------|------------|
| `src/bux/lang/` | the `.bux` language: syntax, compiler, IR, printer, renderer, modules, the language server |
| `src/bux/config.py` | every layer's config: the schema of the language's layers |
| `src/bux/architecture/` | the Flax NNX backend: a module per IR node and per layer, kernels, residual algebras, repeats |
| `src/bux/train/` | training configs (`config.py`), the trainer, optimizers, objective, hooks, sinks, data loading |
| `src/bux/parallel/` | the mesh, the layout, and a registry of methods per kind of parallelism |
| `src/bux/data/` | tokenizing, shuffling and packing documents into rows, locally or on Spark |
| `src/bux/launch.py`, `__main__.py` | the launcher and the command line |
| `src/bux/studio/` | `bux studio`: its server, view model, registries of how layers are drawn, and the page in `static/` |
| `examples/architectures/` | the model zoo; `tests/test_examples.py` trains a shrunk copy of each |
| `examples/experiments/` | runs, sweeps and recipes; `examples/benchmarks/` the benchmark suites |
| `scripts/benchmark.py` | trains a suite and compares two results ([Benchmarks](../performance/benchmarks.md)) |
| `docs/` | this documentation |

Extending bux is registering a function: a layer under `bux.architecture.register`, a
method under `bux.parallel.methods.<kind>.register`, an optimizer, schedule or data
source under the registries of `bux.train`. See
[Adding a layer](../architectures/adding-a-layer.md),
[Extending the trainer](../training/extending.md) and the
[registries](../reference/language/registries.md).

## The documentation

The docs are a Sphinx site in `docs/`, written in Markdown (MyST) and published on
Read the Docs (`.readthedocs.yaml`).

```sh
make docs                                  # build, warnings as errors
open docs/_build/html/index.html
```

`make docs` rebuilds only the pages that changed since the last build; add `-E` to
the Makefile's `sphinx-build` command to rebuild everything.

### What is generated

Much of the reference is generated from the code before every build, by
`docs/_ext/bux_sphinx`, so it cannot fall out of date. Never edit the output; change
its source instead:

| Page | Generated from |
|------|----------------|
| [Layers](../reference/language/layers.md), [Training settings](../reference/language/training.md) | the docstrings and field comments of `bux/config.py` and `bux/train/config.py`, through `bux.lang.reference`, which the editor's hover reads too |
| [Top-level settings](../reference/language/top-level.md) | `TrainConfig`'s fields and their comments |
| [Registries](../reference/language/registries.md) | every registry's entries, and the first paragraph of each one's docstring |
| [Model zoo](../architectures/model-zoo.md) | every file in `examples/architectures/`: its header comment, `bux check`'s summary, its source |
| [Command line](../reference/cli.md) | `bux --help` (`bux.__main__.parser`) |
| [Python API](../reference/python/index.md) | every module's docstrings, by autodoc and autosummary |

### Docstrings

Google style, read by napoleon, in reStructuredText:

```python
def lr_schedule(lr: float, multiplier: Schedule) -> Schedule:
    """Return the learning rate at each step: the peak `lr`, scaled by `multiplier`.

    Args:
        lr: the peak learning rate.
        multiplier: the schedule's multiplier at each step, from 0 to 1.

    Returns:
        A function from a step to its learning rate, traceable by JAX.

    Example::

        rate = lr_schedule(3e-3, lr_multiplier(run.train))
    """
```

- ``double backticks`` are literal code; single backticks name a Python object and
  link to it when it is documented (`Trainer`, `bux.train.trainer.train`).
- A code block follows a paragraph that ends with `::`.
- Public API that users call or extend gets `Args:`, `Returns:` and `Raises:`; a
  helper gets a line that says what it does. Signatures show the annotations, so
  `Args:` does not repeat types.
- A config's docstring and its fields' comments are its documentation in the
  language reference and in the editor: every field has a comment, above it or
  beside it. Write them as plain prose with backticks, which reads well both as
  Markdown and as reStructuredText.
- A function registered under a name has a docstring whose first paragraph says what
  that entry does: it is the entry's row in the registries' reference.

### Writing pages

- Label every code fence: ```` ```bux ````, ```` ```console ```` (a session with `$ `
  prompts), ```` ```sh ````, ```` ```python ````, ```` ```text ````.
- Link to a layer or setting's reference entry with ``{bux}`gqa` ``, to the API with
  ``{py:class}`bux.train.trainer.Trainer` ``, and to another page by its relative
  path, `[Resuming](../training/launching.md#resuming)`.
- A new page goes in its section's `index.md` toctree.
- Lead with what a page is for and an example; put measurements after the usage,
  with the hardware, setup and date they were taken on.

(docs-voice)=
### Voice

Write like the docs of a mature project: confident, concrete, and quick to the
point. Second person, present tense, active voice.

- **Lead with the payoff.** A page opens with what it lets the reader do and when
  they need it, then an example, then details, reference tables, and measurements
  last.
- **One idea per sentence.** Split comma-chains and nested "which" clauses into
  sentences, lists or tables.
- **Plain English over shorthand.** "8 rows per GPU", not "8 rows a GPU"; "40,769
  for Megatron-Bridge", not "Megatron-Bridge's 40,769"; "every suite except X", not
  "every suite but X".
- **Say it once.** A concept, caveat or table lives on the page that owns it; other
  pages link to it.
- **No lab notebook.** Guides carry what a user needs: what a feature does, how to
  use it, its defaults, constraints and errors, and the measurements behind a
  recommendation. Development history (branches, commit hashes, before/after of an
  internal change) belongs in git, and investigation logs in a short
  "What doesn't help" list at most.
- **Precise and real.** Show real commands with their real output, state every
  default, and never write a number you didn't measure.

(keeping-docs-in-sync)=
### Keeping the docs in sync

A change to bux is done when its documentation is. Update, in the same change:

| When you change | Update |
|-----------------|--------|
| a layer's or setting's config | its docstring and field comments (the reference regenerates), and the guide page that explains it |
| the language: syntax, semantics, errors | `docs/language/`, and `grammar.md` for syntax |
| a command or flag | `docs/reference/cli.md`'s table, and the guide that uses it (`training/running.md`, `training/launching.md`) |
| the studio: its server, view model, palette or internals (`bux.studio`) | `docs/language/studio.md`, and the plan's JSON contract in `plans/studio.md` that the page is coded against |
| training: trainer, hooks, sinks, objective, optimizers, checkpoints | `docs/training/`, and the docstrings of what changed |
| search: dimensions, objectives, methods, halving (`bux.search`) | `docs/training/searching.md`, and `docs/language/sweeps.md` for the syntax |
| data: sources, tokenizer, packing, the pipeline | `docs/data/` |
| parallelism: mesh, layout, a method | the page in `docs/parallelism/` for it |
| a kernel, a default that changes speed, new measurements | `docs/performance/` (kernels, tuning, benchmarks), with hardware and date |
| an exported layer, or the exporter | `docs/export/`, its table of what exports, and `tests/test_export*.py` |
| a layer of the vLLM plugin (`bux_vllm`) | `docs/export/vllm.md`'s table of what runs, and its FP8 weights if the layer adds a projection, and `scripts/vllm_parity.py`'s cases |
| an environment variable | `docs/reference/environment.md` |
| an example architecture | its header comment (the zoo regenerates) and `tests/test_examples.py` |
| an example run or recipe | the tutorial or guide that uses it |
| the checks, or what a pull request needs | `CONTRIBUTING.md` and `.github/pull_request_template.md` |

Then run `make docs`: a link to a heading you renamed, a reference to a setting you
removed, or a docstring that no longer renders fails it.
