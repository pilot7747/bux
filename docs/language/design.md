# Design notes

Why the language looks the way it does: the idea it is built on, the choices in its
syntax, and what was considered instead. Nothing here is needed to write a `.bux`
file. The design of the training side, including why runs are written in the same
language and why names bind late, is in the
[training design notes](../training/design.md).

## The syntax looks like the drawing

The core idea is that `[f]` is a skip arc, the bypass wire drawn over `f`. The
one-line syntax is chosen so that it already looks like the circuit diagram:
composition runs left to right, as the stream does, and brackets sit where the arcs
are. `bux render` draws the diagram from the source, so the drawing is always the
output, never the input.

## Choices in the syntax

Each of these keeps the common case short and the rest possible:

- **Composition binds tighter than `+` and `*`**, as concatenation does in regular
  expressions. `norm attn + ffn` is `(norm attn) + ffn`, so gates and parallel
  branches need few brackets, and SwiGLU reads as it is drawn
  ([Precedence](operators.md#precedence-and-associativity)).
- **`[f]` is not `_ + f`.** Both compute `x + f(x)` on the plain residual stream, but
  only `[f]` is the stream's own connection. That is what lets one `residual = ...`
  line change every residual branch of a model while leaving explicit sums alone
  ([Residual algebras](residual-algebras.md)).
- **Taps are stacks.** A U-net pairs the first block with the last, and
  last-in-first-out order gives that pairing with no index arithmetic
  ([U-net](skip-connections.md#u-net)).
- **A bare name nothing defines is a string**, so enumerated settings need no quotes:
  `layout=interleaved` ([Bare names are strings](values.md#bare-names-are-strings)).
- **Modules are found by name, never by path.** Folders organize files without
  becoming part of their names, so moving a file breaks nothing
  ([The module path](modules.md#the-module-path)).

## Alternatives considered

- **Arrows with labels everywhere**, such as `@a ln1 attn +a @b ln2 mlp +b`. This is
  fully general, but every residual needs a name. Brackets cover the common case, and
  taps remain for the rest.
- **Drawn 2D ASCII graphs** as the input. This is the most visual option, but it is
  painful to edit and to diff. Instead the drawing is the output (`bux render`).
- **Edge lists**, such as NAS-Bench-201 cell strings, or HF-style `layer_types`
  arrays. They are precise but unreadable, and per-layer lists say nothing about
  wiring.

## Not in the language yet

- A `cat` merge. Zamba concatenates x0 into its shared block, which
  `examples/architectures/zamba2.bux` leaves out.
- Recursive macros, for nested structures. A macro that calls itself is an error
  today ([Macros](definitions.md#macros)).
- Parsing rendered diagrams back, so that a drawing can be edited directly.
  [bux studio](studio.md) edits the diagram it draws instead.

The example architectures, and how closely each one follows its source, are
described in the [model zoo](../architectures/model-zoo.md).
