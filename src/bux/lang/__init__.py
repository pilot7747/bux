"""The .bux language in Python: compile source into a typed IR, check it, print it.

A ``.bux`` file describes an architecture as an algebra over the residual stream,
and can say how to train it as well:

.. code-block:: bux

    d_model    = 64
    vocab_size = 512

    norm  = rmsnorm
    attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
    ffn   = gated_mlp(intermediate_size=4 * d_model, activation=swish)
    block = [norm attn] [norm ffn]          # [f] is a residual branch: x + f(x)
    model = embed block^2 norm head(tied=true)

This package turns that source into plain data. A model compiles to an
`~bux.lang.ir.Architecture`: a frozen pydantic tree of layer configs, with every
layer's input width inferred and checked, which a backend builds
(`~bux.architecture.model.Model`). A file that also sets ``data``, ``optimizer``,
``seq_len``, ``batch`` and ``steps`` is a *run*, and each combination of its
alternatives (``optimizer.lr = 1e-3 | 3e-3``) compiles to a `~bux.lang.ir.Run` of
its own, a sweep. Nothing here imports JAX, so the compiler is quick to import and
runs anywhere: in a script, a notebook, or an editor.

The language itself is described in :doc:`/language/index`, training in
:doc:`/training/index`, and every layer and setting a file can call in
:doc:`/reference/language/index`. This page is the language's Python API.

.. rubric:: Compile a model

`compile` returns the architecture of the definition named ``model`` (or `entry`).
It raises a `BuxError` at the first mistake::

    from bux.lang import compile
    from bux.lang.render import render

    arch = compile(open("tiny.bux").read(), filename="tiny.bux")
    arch.d_model, arch.vocab_size  # (64, 512)
    len(list(arch.layers()))  # 11, with repeats unrolled
    print(render(arch))

`~bux.lang.render.render` draws the model as a circuit of the residual stream, one
drawing per labeled block (what ``bux render`` prints)::

    model
    ──embed──┤block├×2──norm──head──

    block
    ──┬────────────⊕──┬───────────⊕──
      └─norm──attn─┘  └─norm──ffn─┘

.. rubric:: Compile a run, or a sweep

`compile_runs` returns every run of a file, one per combination of the alternatives
the run reaches. Each `~bux.lang.ir.Run` holds its architecture, its validated
training settings (`bux.train.config.TrainConfig`), a name that says which
alternatives it took, and an `~bux.lang.ir.Run.id` that hashes what it computes::

    from bux.lang import compile_runs
    from bux.train.trainer import train

    source = open("examples/experiments/smoke.bux").read()
    for run in compile_runs(source, filename="smoke.bux"):
        print(run.name, run.id)  # smoke 5a4f0f7d85
        train(run)  # into runs/smoke-5a4f0f7d85/

With ``optimizer = adamw(lr=3e-3 | 1e-2)`` in the file, the same loop prints two
runs, ``smoke[optimizer.lr=3e-3]`` and ``smoke[optimizer.lr=1e-2]``.

.. rubric:: Compile a search

A setting written as a distribution, ``optimizer.lr = log_uniform(1e-4, 1e-2)``, is a
dimension of a search (`bux.search`). `compile_spaces` returns a `~compiler.Space`
for each run, with the dimensions it reaches, and its `~compiler.Space.trial` is the
run at a point of them::

    from bux.lang import compile_spaces

    [space] = compile_spaces(source, filename="tune.bux")
    space.trial({"optimizer.lr": 3e-4}).name  # tune[optimizer.lr=3e-4]

.. rubric:: Check a file, as an editor does

`check` compiles like `compile`, but recovers from errors and collects all of them
in a `Report`, with what did evaluate: the definitions' values, the sweep's axes,
the first run. ``bux check`` and the language server (`bux.lang.lsp`) use it::

    from bux.lang import check

    report = check(source, filename="smoke.bux")
    for error in report.errors:
        print(error)
    if report.arch is not None:  # no errors
        print(len(report.axes), "axes")

.. rubric:: Errors

Every mistake in source is a `BuxError`, with a message, a span into the file, and
often a hint. Printed, it points at the source as a compiler does::

    error: unknown layer or block 'nrom'
     --> tiny.bux:2:15
      |
    2 | model = embed nrom head
      |               ^^^^
      = hint: did you mean 'engram'?

.. rubric:: Settings from the command line, and modules

All four entry points take the same keywords. `overrides` are statements that
redefine the file's own, as if written at its end (``bux train FILE
optimizer.lr=1e-3``); alternatives there make a sweep too. ``use name`` brings in the
definitions of another file, found by name on the search path
(`bux.lang.modules.find`: the file's folder, ``[tool.bux] path`` in pyproject.toml,
then ``BUX_PATH``), or in the `modules` mapping, which holds sources in memory::

    lib = "norm = rmsnorm\\nblock = [norm gqa(num_heads=4, num_kv_heads=2)]\\n"
    arch = compile(
        "use lib\\nd_model = 64\\nvocab_size = 512\\nmodel = embed block^2 norm head",
        modules={"lib": lib},
    )

.. rubric:: Back to source

`~bux.lang.printer.to_source` prints an architecture as ``.bux`` source and
`~bux.lang.printer.run_to_source` a whole run as one file, with no modules and no
alternatives (``bux fmt``). Either compiles back to what it printed:
``compile(to_source(arch))`` equals ``arch`` up to labels, and a run's source keeps
its id.

.. rubric:: How the pieces fit

Source flows through the modules in this order:

======================  ===========================================================
`bux.lang.syntax`       tokens, the AST and the parser; `BuxError` and its spans
`bux.lang.modules`      finds what ``use`` names and joins the files into a program
`bux.lang.compiler`     evaluates names, macros and sweeps, then lowers to the IR
`bux.lang.ir`           the IR: `~bux.lang.ir.Architecture`, `~bux.lang.ir.Run`
`bux.lang.printer`      the IR back to source
`bux.lang.render`       the IR as a circuit diagram
`bux.lang.reference`    the configs' own documentation, for hovers and these docs
`bux.lang.lsp`          diagnostics, completion and hover for editors (``bux-lsp``)
======================  ===========================================================

.. rubric:: Exported here

.. autosummary::
    :nosignatures:

    ~compiler.compile
    ~compiler.compile_runs
    ~compiler.compile_spaces
    ~compiler.compile_file
    ~compiler.check
    ~compiler.Report
    ~compiler.Axis
    ~compiler.Space
    ~compiler.Dimension
    ~ir.Architecture
    ~ir.Run
    ~syntax.parse
    ~syntax.BuxError
"""

from bux.lang.compiler import (
    Axis,
    Dimension,
    Report,
    Space,
    check,
    compile,
    compile_file,
    compile_runs,
    compile_spaces,
)
from bux.lang.ir import Architecture, Run
from bux.lang.syntax import BuxError, parse

__all__ = [
    "Architecture",
    "Axis",
    "BuxError",
    "Dimension",
    "Report",
    "Run",
    "Space",
    "check",
    "compile",
    "compile_file",
    "compile_runs",
    "compile_spaces",
    "parse",
]
