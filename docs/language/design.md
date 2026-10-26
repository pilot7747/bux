# Design notes

This page records why the language looks as it does: the alternatives considered for
writing architectures, and the extensions that may come. The design of the training
side, why runs are written in the same language and how late binding came about, is
in the [training design notes](../training/design.md).

## The syntax looks like the drawing

The core idea is that `[f]` is a skip arc, the bypass wire drawn over `f`. The 1D
syntax is chosen so that it already looks like the circuit diagram: composition runs
left to right, as the stream does, and brackets sit where the arcs are.
`bux render` draws the diagram from the source, so the drawing is the output, never
the input.

## Alternatives considered

- **Arrows with labels everywhere**, such as `@a ln1 attn +a @b ln2 mlp +b`. This is
  fully general, but every residual needs a name. Brackets cover the common case, and
  taps remain for the rest.
- **Drawn 2D ASCII graphs** as the input. This is the most visual option, but it is
  painful to edit and to diff. Instead the drawing is the *output* (`bux render`),
  and the 1D syntax is chosen so that it already looks like the drawing.
- **Edge lists**, such as NAS-Bench-201 cell strings, or HF-style `layer_types`
  arrays. They are precise but unreadable, and per-layer lists say nothing about
  wiring.

## Possible extensions

- a `cat` merge: Zamba concatenates x0 into its shared block, which
  `examples/architectures/zamba2.bux` leaves out;
- recursive macros, for nested structures;
- parsing the rendered diagrams back, so that a drawing can be edited directly.

## The example architectures

The example files at the top of `examples/architectures/` transcribe public configs
from memory. Ones marked "-style" or "approximate" simplify details, so verify the
numbers before relying on them. The folders, one per model family (`cohere/`,
`deepseek/`, `glm/`, `inkling/`, `kimi/`, `laguna/`, `mellum/`, `mimo/`, `minimax/`,
`muse/`, `nemotron/`, `qwen/`, `reflection/`), transcribe recent releases from their
published `config.json` and modeling code. Their comments name what is left out (vision
encoders, speculative drafters) or approximated. The
[model zoo](../architectures/model-zoo.md) lists them all.
