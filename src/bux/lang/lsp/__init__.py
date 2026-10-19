"""Editor support for .bux files.

``features`` computes diagnostics, completions and hovers from the source text alone;
``server`` serves them over the Language Server Protocol (``bux-lsp``, which needs
the ``lsp`` extra: ``uv sync --extra lsp``). ``pycharm/`` configures PyCharm to use
it; see docs/language.md.
"""
