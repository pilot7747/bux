"""bux studio: a visual editor of ``.bux`` files, ``bux studio FILE``.

It serves a local page that shows a file three ways at once: its source, its model
as an architecture diagram in the style of Sebastian Raschka's figures, and, for a
run, its training (budget, learning-rate schedule, curriculum, data, evals) and its
sweep (axes, runs, search dimensions). Every one of them is editable, and every edit
is a splice of the file's text, so its comments, macros, patterns and modules stay
as they were. The page draws what the compiler says the file means, the IR of the run
it picked, never a model of its own.

The pieces, each a pure function of the source but the server:

- `bux.studio.server`: the stdlib HTTP server, its JSON API, and `serve`, the
  command; it routes, and calls the rest.
- `bux.studio.view`: the view model of a file, the plan's ``View``: its model's
  graph as a tree of nodes, each with where it is written, its facts, its
  definitions and its sweep.
- `bux.studio.training`: the training view of a run, and its learning rate and
  parameters, which need JAX.
- `bux.studio.picks`: the run of a sweep the page picked, by axis ids that survive
  edits elsewhere in the file.
- `bux.studio.palette` and `bux.studio.internals`: registries, by layer type, of
  how a layer is named and drawn, and of what is inside it; a new layer registers
  its own in a few lines.
- ``bux.studio.forms`` and ``bux.studio.edits``: the inspector's forms, and the
  edits that turn the page's changes into splices of the text.
- ``static/``: the page, ES modules and CSS with no build step.

The compiler's `bux.lang.compiler.trace` says where every node and config of a file
is written, which is what lets an edit land where the file says what it changes.
"""
