"""Editor support for .bux files: diagnostics, completion and hover.

`bux.lang.lsp.features` computes them from the source text alone, as plain
functions that work without an editor. `bux.lang.lsp.server` serves them over the
Language Server Protocol: ``bux-lsp``, a language server over stdio, which needs the
``lsp`` extra (``uv sync --extra lsp`` puts it at ``.venv/bin/bux-lsp``).
``src/bux/lang/lsp/pycharm`` holds a TextMate grammar and an LSP4IJ template for
PyCharm. :doc:`/language/tooling` says how to set up PyCharm and Neovim.
"""
