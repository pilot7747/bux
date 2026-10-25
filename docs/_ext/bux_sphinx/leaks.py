"""A check that no page shows what the build machine holds: its environment or its
home directory. An object's repr can carry them into a page (``os.environ`` as a
default, a path made at import time), and Read the Docs would publish them."""

import re
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def check(app: Sphinx, error: Exception | None) -> None:
    """Warn of every page that shows the build's environment or home directory."""
    if error is not None or app.builder.format != "html":
        return
    leak = re.compile(re.escape("environ({") + "|" + re.escape(str(Path.home())))
    for page in Path(app.outdir).rglob("*.html"):
        if leak.search(page.read_text(errors="replace")):
            logger.warning("%s shows the build machine's environment or home", page)


def setup(app: Sphinx) -> None:
    app.connect("build-finished", check)
