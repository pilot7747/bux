"""Chat templates: a Hugging Face tokenizer's Jinja template, rendered as transformers
renders it, so that a conversation reads as the tokenizer's model reads it.

A template comes from a Hub repo or a folder: its ``chat_template.jinja``, or else the
``chat_template`` of its ``tokenizer_config.json``, which also names the special
tokens that templates write by name (``bos_token``, ``eos_token``)::

    from bux.data.templates import load

    template = load("HuggingFaceTB/SmolLM-135M-Instruct")
    template.render([{"role": "user", "content": "hi"}], prompt=True)
    # '<|im_start|>user\\nhi<|im_end|>\\n<|im_start|>assistant\\n'
"""

import datetime
import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import jinja2
import jinja2.ext
import jinja2.nodes
import jinja2.parser

# Fetches a file of a Hub repo through the Hugging Face cache, as gigatoken fetches
# the tokenizer.json; it is not exported either.
from gigatoken._load.hub import hub_file
from jinja2.sandbox import ImmutableSandboxedEnvironment

#: The special tokens a tokenizer_config.json names, which templates write by name.
NAMED = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)


@dataclass(frozen=True, eq=False)
class ChatTemplate:
    """A chat template, compiled.

    Attributes:
        jinja: The template.
        tokens: The special tokens it writes by name, as ``bos_token``.
        specials: The special tokens of its tokenizer, which it may write.
    """

    jinja: jinja2.Template
    tokens: dict[str, str]
    specials: frozenset[str]

    def render(self, messages: list[dict], prompt: bool = False) -> str:
        """Render `messages`; with `prompt`, followed by the header of the reply that
        comes next (transformers' ``add_generation_prompt``)."""
        return self.jinja.render(
            messages=messages, add_generation_prompt=prompt, **self.tokens
        )


@functools.cache
def load(source: str) -> ChatTemplate:
    """Load the chat template of `source`, once per process.

    Args:
        source: A Hub repo, a folder, or a file in one, as a tokenizer.json.

    Raises:
        ValueError: If `source` has no chat template.
    """
    config = json.loads(_read(source, "tokenizer_config.json") or "{}")
    text = _read(source, "chat_template.jinja") or config.get("chat_template")
    if isinstance(text, list):  # templates by name, as [{name, template}]
        text = {t["name"]: t["template"] for t in text}.get("default")
    if not text:
        raise ValueError(
            f"{source!r} has no chat template; render with another's, as "
            'chat(template="HuggingFaceTB/SmolLM-135M-Instruct")'
        )
    tokens = {name: _text(config[name]) for name in NAMED if config.get(name)}
    added = (config.get("added_tokens_decoder") or {}).values()
    specials = frozenset(t["content"] for t in added if t.get("special"))
    jinja = _ENVIRONMENT.from_string(text, globals=_GLOBALS)
    return ChatTemplate(jinja, tokens, specials)


def _read(source: str, filename: str) -> str | None:
    """Return a file of a Hub repo or a folder (a file's), or None if it has none."""
    path = Path(source)
    folder = path.parent if path.is_file() else path
    if folder.is_dir():
        file = folder / filename
        return file.read_text() if file.is_file() else None
    try:
        return hub_file(source, filename).read_text()
    except FileNotFoundError:
        return None


def _text(token: str | dict) -> str:
    return token["content"] if isinstance(token, dict) else token  # an AddedToken's


# transformers' environment for chat templates (its _compile_jinja_template).


class _Generation(jinja2.ext.Extension):
    """``{% generation %}``, with which templates mark replies for transformers'
    assistant mask, rendered as what it holds."""

    tags = {"generation"}  # noqa: RUF012 (jinja's Extension declares a set)

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_body")
        return jinja2.nodes.CallBlock(call, [], [], body).set_lineno(lineno)

    def _body(self, caller: Callable[[], str]) -> str:
        return caller()


def _raise(message: str) -> NoReturn:
    raise jinja2.TemplateError(message)


def _tojson(
    x: object,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        x,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _strftime_now(format: str) -> str:
    return datetime.datetime.now().strftime(format)


_ENVIRONMENT = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[_Generation, jinja2.ext.loopcontrols],
)
_ENVIRONMENT.filters["tojson"] = _tojson
_GLOBALS = {"raise_exception": _raise, "strftime_now": _strftime_now}
