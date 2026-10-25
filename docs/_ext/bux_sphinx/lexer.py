"""A Pygments lexer for the .bux language, for ```bux blocks."""

from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    Whitespace,
)

from bux.lang.compiler import ACTIVATIONS, TYPES

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


class BuxLexer(RegexLexer):
    """Highlights definitions, builtin layers and settings, taps and numbers."""

    name = "bux"
    aliases = ("bux",)
    filenames = ("*.bux",)

    tokens = {  # noqa: RUF012 (Pygments reads it as is)
        "root": [
            (r"#.*?$", Comment.Single),
            (r"\s+", Whitespace),
            (r"\"[^\"\n]*\"|'[^'\n]*'", String),
            (words(("use", "shared"), suffix=r"\b"), Keyword.Namespace),
            (words(("true", "false", "none"), suffix=r"\b"), Keyword.Constant),
            (rf"[@$]{_NAME}!?", Name.Variable),
            (r"\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?[kKMBT]?\b", Number),
            (rf"^{_NAME}(?:\.{_NAME})*(?=\s*(?:\([^)]*\))?\s*=(?!=))", Name.Function),
            (rf"{_NAME}(?=\s*:(?!=))", Name.Label),
            (words(sorted({*TYPES, *ACTIVATIONS}), suffix=r"\b"), Name.Builtin),
            (r"\b_\b", Name.Builtin.Pseudo),
            (_NAME, Name),
            (r"->|→|//|[-+*/^|]", Operator),
            (r"[\[\](),.;=:!]", Punctuation),
            (r".", Text),  # never an error token, so a block always highlights
        ]
    }
