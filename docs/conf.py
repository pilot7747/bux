"""Sphinx's configuration for bux's documentation: ``make docs`` builds it."""

import sys
from importlib.metadata import version as _version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "bux"
author = "Nikita Pavlichenko"
copyright = "2026, Nikita Pavlichenko"
release = _version("bux")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "bux_sphinx",
]

exclude_patterns = [
    "_build",
    "_generated",
    "tutorials/timeline/tables",
    "Thumbs.db",
    ".DS_Store",
]
templates_path = ["_templates"]
default_role = "py:obj"  # `name` is code, and a link where it names something
highlight_language = "text"
pygments_style = "friendly"
pygments_dark_style = "monokai"

# --- Markdown -------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 4  # [text](page.md#a-heading) links, as on GitHub

# --- The API reference ----------------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
# As the source writes them (bux_sphinx.signatures), a class by its name alone.
autodoc_typehints = "signature"
python_use_unqualified_type_names = True
# A package documents what it defines; what it re-exports has its page where defined.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}
# Optional extras that some modules import where they are used; GPUs only.
autodoc_mock_imports = [
    "pyspark",
    "tokamax",
    "cudnn",
    "cutlass",
    "triton",
    "jax_triton",
    "transformer_engine",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_attr_annotations = True
napoleon_include_special_with_doc = True  # __call__ says what a layer takes

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "flax": ("https://flax.readthedocs.io/en/latest", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# --- HTML -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "bux"
html_static_path = ["_static"]
html_css_files = ["bux.css"]
html_show_sourcelink = False
html_context = {
    "github_user": "pilot7747",
    "github_repo": "bux",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme_options = {
    "github_url": "https://github.com/pilot7747/bux",
    "use_edit_page_button": True,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 7,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
