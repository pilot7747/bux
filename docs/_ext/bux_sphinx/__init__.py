"""bux's Sphinx extension: a lexer for .bux, a role that links to the language
reference, and the pages generated from the code before every build.

Generated pages go to ``docs/_generated`` (not in git), and the written pages
include them, so the reference follows the code with nothing to keep in sync by hand:

    configs     every layer and setting of the language, from bux/config.py and
                bux/train/config.py: their docstrings and field comments
    registries  the names each registry knows (methods, optimizers, formats, ...)
    models      the example architectures, from their files' header comments
    cli         ``bux --help``

API signatures show their annotations and defaults as the source writes them
(``signatures``), and no page may show the build machine's environment (``leaks``).
"""

from pathlib import Path

from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.application import Sphinx

from bux_sphinx import cli, configs, leaks, models, registries, signatures, submodules
from bux_sphinx.lexer import BuxLexer


def generate(app: Sphinx) -> None:
    """Write the generated pages, before Sphinx reads any source."""
    out = Path(app.srcdir) / "_generated"
    out.mkdir(exist_ok=True)
    pages = {
        **configs.pages(),
        **registries.pages(),
        **models.pages(Path(app.srcdir).parent / "examples" / "architectures"),
        **cli.pages(),
    }
    for name, text in pages.items():
        path = out / name
        if not path.exists() or path.read_text() != text:  # unchanged: no rebuild
            path.write_text(text)


def bux_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: dict | None = None,
    content: list[str] | None = None,
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``{bux}`gqa``` links to the reference entry of the layer or setting gqa."""
    ref = addnodes.pending_xref(
        rawtext, refdomain="std", reftype="ref", reftarget=configs.label(text)
    )
    ref["refexplicit"] = True
    ref["refwarn"] = True
    ref += nodes.literal(rawtext, text)
    return [ref], []


def setup(app: Sphinx) -> dict[str, object]:
    app.add_lexer("bux", BuxLexer)
    app.add_role("bux", bux_role)
    app.connect("builder-inited", generate)
    signatures.setup(app)
    leaks.setup(app)
    submodules.setup(app)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
