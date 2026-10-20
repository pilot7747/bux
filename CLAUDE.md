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
  experiment. See `docs/language.md`, `docs/training.md`, `docs/data.md`, and
  `docs/parallelism.md`.
