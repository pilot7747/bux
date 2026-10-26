"""The map: turn records into documents, each with token ids, a loss mask and a key.

A document's key is a hash of its content, not of where it is or how it tokenizes.
So a split by key holds out the same documents whatever the tokenizer, and the
documents a reduce gets, sorted by key, do not depend on the engine.

`text` maps a batch of raw text and `chat` a batch of conversations, each to a
record batch of documents in `SCHEMA`. `~bux.data.pipeline.Job.map` picks the one
its source needs.
"""

import bisect
import hashlib
import itertools
import json
import re
from collections.abc import Buffer, Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int32, Int64, Integer

from bux.data import templates
from bux.data.tokenizer import Tokenizer, utf8
from bux.train import config as tcfg

#: The documents the map makes, a row each.
#:
#: ``key`` is 63 random bits, a hash of the content; ``input_ids`` the tokens, and
#: ``loss_mask`` whether each is a target.
#:
#: :meta hide-value:
SCHEMA = pa.schema(
    [
        ("key", pa.int64()),  # 63 random bits
        ("input_ids", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.bool_())),
    ]
)


def _keys(contents: Iterable[Buffer]) -> Int64[np.ndarray, " docs"]:
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(c, digest_size=8).digest()) >> 1
            for c in contents
        ),
        np.int64,
    )


def _in_split(
    keys: Int64[np.ndarray, " docs"], split: tuple[float, float]
) -> Bool[np.ndarray, " docs"]:
    at = keys * 2.0**-63
    return (split[0] <= at) & ((at < split[1]) | (split[1] == 1))


def record_batch(
    keys: Int64[np.ndarray, " docs"],
    ids: Integer[np.ndarray, " tokens"],
    offsets: Integer[np.ndarray, " docs+1"],
    mask: Bool[np.ndarray, " tokens"],
) -> pa.RecordBatch:
    """Return documents as a record batch in `SCHEMA`: their keys, and their token ids
    and loss masks end to end, the i-th document at ``offsets[i]:offsets[i + 1]``."""
    offsets = pa.array(offsets.astype(np.int32))
    columns = [
        pa.array(keys, pa.int64()),
        pa.ListArray.from_arrays(offsets, pa.array(ids, pa.int32())),
        pa.ListArray.from_arrays(offsets, pa.array(mask, pa.bool_())),
    ]
    return pa.RecordBatch.from_arrays(columns, schema=SCHEMA)


def text(
    batch: pa.RecordBatch, config: tcfg.TextConfig, tok: Tokenizer
) -> pa.RecordBatch:
    """Map a batch of raw text to documents, every token a target.

    A row's text is a document, keyed by a hash of its bytes. Documents outside the
    source's `split` are left out, and so are those that tokenize to nothing; the
    others get the tokenizer's bos first and eos last where the source asks.

    Args:
        batch: Records with the source's text column.
        config: The ``text(...)`` source.
        tok: The run's tokenizer.

    Returns:
        The documents, in `SCHEMA`.
    """
    texts = batch.column(config.column)
    start, chars = utf8(texts)
    keys = _keys(chars[a:b] for a, b in itertools.pairwise(start.tolist()))
    keep = _in_split(keys, config.split)
    ids, offsets = tok.encode(texts.filter(keep))
    lengths = np.diff(offsets)
    keys = keys[keep][lengths > 0]
    offsets = np.concatenate([[0], np.cumsum(lengths[lengths > 0])])
    # Inserting one token per document shifts the i-th document's offsets by i.
    shift = np.arange(len(offsets))
    if config.bos:
        ids, offsets = np.insert(ids, offsets[:-1], tok.bos), offsets + shift
    if config.eos:
        ids, offsets = np.insert(ids, offsets[1:], tok.eos), offsets + shift
    return record_batch(keys, ids, offsets, np.ones(len(ids), bool))


def chat(
    batch: pa.RecordBatch, config: tcfg.ChatConfig, tok: Tokenizer
) -> pa.RecordBatch:
    """Map a batch of conversations to documents, rendered with a chat template.

    A row is a conversation in the OpenAI format, a list of ``{role, content}``
    messages, ``content`` a string or a list of ``{type, text}`` parts; it is keyed
    by a hash of its messages. The template is the source's `template`, or else the
    tokenizer's own (`bux.data.templates`), and a special token written in a message
    stays text. The loss mask follows `train_on`: every assistant reply
    (``"assistant"``), only the last (``"last"``), or every token (``"all"``). A
    reply is learned from where the template's prompt for it ends (its
    ``add_generation_prompt``) through the first special token after that, which
    ends it. A conversation with nothing to learn is left out, as are those outside
    the source's `split`.

    Args:
        batch: Records with the source's column of messages.
        config: The ``chat(...)`` source.
        tok: The run's tokenizer, which must have the template's special tokens.

    Returns:
        The documents, in `SCHEMA`.

    Raises:
        ValueError: If the tokenizer lacks a special token that the template writes,
            or the template renders a reply's prompt otherwise than the conversation.
        jinja2.TemplateError: If the template refuses a conversation, as one whose
            roles do not alternate.
    """
    conversations = batch.column(config.column).to_pylist()
    keys = _keys(json.dumps(m, sort_keys=True).encode() for m in conversations)
    keep = _in_split(keys, config.split)
    template = templates.load(config.template or tok.config.name)
    specials = _Specials.of(tok, template)
    docs, kept = [], []
    for key, conversation in zip(keys[keep], itertools.compress(conversations, keep)):
        text, starts = _render(template, specials.hide(conversation), config.train_on)
        doc = _cut(text, starts, specials.find, config.train_on == "all")
        if any(learned for _, _, learned in doc):  # without a reply, nothing to learn
            docs.append(doc)
            kept.append(key)
    return record_batch(np.array(kept, np.int64), *_encode(docs, tok, specials.show))


def _content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content)  # parts of a message


# A special token written in a message stays text: while the template renders, it is
# this private-use character and one of its own after it, and the character itself is
# written twice.
_ESCAPE = "\U000f0000"
_ESCAPED = re.compile(f"{_ESCAPE}(.)", re.DOTALL)


@dataclass(frozen=True)
class _Specials:
    """The special tokens of a tokenizer and of a template: a pattern that finds them
    in rendered text, and their escapes in messages (`_ESCAPE`)."""

    find: re.Pattern
    escapes: re.Pattern  # the special tokens and the escape character
    codes: dict[str, str]  # the escape of each
    texts: dict[str, str]  # what each character after the escape stands for

    @classmethod
    def of(cls, tok: Tokenizer, template: templates.ChatTemplate) -> "_Specials":
        tokens = sorted(tok.specials.keys() | template.specials, key=len, reverse=True)
        codes = {
            t: _ESCAPE + chr(ord(_ESCAPE) + i) for i, t in enumerate([_ESCAPE, *tokens])
        }
        find = re.compile("|".join(map(re.escape, tokens)))  # the longest first
        escapes = re.compile(f"{re.escape(_ESCAPE)}|{find.pattern}")
        return cls(find, escapes, codes, {c[1]: t for t, c in codes.items()})

    def hide(self, conversation: list[dict]) -> list[dict]:
        """Return the messages, their contents joined and special tokens escaped."""
        return [
            {**m, "content": self.escapes.sub(lambda t: self.codes[t[0]], _content(m))}
            for m in conversation
        ]

    def show(self, text: str) -> str:
        """Return rendered text whose escaped special tokens are text again."""
        return _ESCAPED.sub(lambda c: self.texts[c[1]], text)


def _render(
    template: templates.ChatTemplate,
    messages: list[dict],
    train_on: Literal["assistant", "last", "all"],
) -> tuple[str, list[int]]:
    """Render a conversation, and find where each reply that `train_on` learns starts:
    where the template's prompt for it ends."""
    text = template.render(messages)
    replies = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    starts = []
    for i in {"assistant": replies, "last": replies[-1:], "all": []}[train_on]:
        prompt = template.render(messages[:i], prompt=True)
        if not text.startswith(prompt):
            raise ValueError(
                f"the chat template renders the prompt for message {i}, a reply, "
                "otherwise than the conversation, so where the reply starts is "
                'unknown; render with another template, as chat(template="...")'
            )
        starts.append(len(prompt))
    return text, starts


def _cut(
    text: str, starts: list[int], find: re.Pattern, learn_all: bool
) -> list[tuple[str, bool, bool]]:
    """Cut a rendered conversation at its special tokens and where replies start.

    Returns:
        Each piece, whether it is a special token, and whether it is learned: all
        with `learn_all`, else those from a reply's start through the special token
        that ends it, the first after it.
    """
    specials = [m.span() for m in find.finditer(text)]
    firsts = [a for a, _ in specials]
    ends = [
        specials[k][1]
        if (k := bisect.bisect_left(firsts, s)) < len(specials)
        else len(text)
        for s in starts
    ]
    cuts = sorted({0, len(text), *starts, *itertools.chain.from_iterable(specials)})
    special = set(specials)
    return [
        (
            text[a:b],
            (a, b) in special,
            learn_all or any(s <= a < e for s, e in zip(starts, ends)),
        )
        for a, b in itertools.pairwise(cuts)
    ]


def _encode(
    docs: list[list[tuple[str, bool, bool]]],
    tok: Tokenizer,
    show: Callable[[str], str],
) -> tuple[
    Int32[np.ndarray, " tokens"],
    Int64[np.ndarray, " _"],
    Bool[np.ndarray, " tokens"],
]:
    """Encode the pieces of rendered conversations, text as text and special tokens as
    their ids: return their ids end to end, n + 1 offsets into them, and the mask."""
    plain = [show(p) for doc in docs for p, special, _ in doc if not special]
    ids, offsets = tok.encode(pa.array(plain, pa.large_string()))
    spans = itertools.pairwise(offsets.tolist())
    pieces = [piece for doc in docs for piece in doc]
    tokens = [
        np.array([tok.special(p)], np.int32) if special else ids[slice(*next(spans))]
        for p, special, _ in pieces
    ]
    sizes = np.array([0, *map(len, tokens)], np.int64).cumsum()
    mask = np.repeat([learned for _, _, learned in pieces], np.diff(sizes))
    ends = np.array([0, *map(len, docs)], np.int64).cumsum()
    return (
        np.concatenate([*tokens, np.zeros(0, np.int32)]),
        sizes[ends],
        mask.astype(bool),
    )
