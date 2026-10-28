"""The map: turn records into documents, each with token ids, a loss mask and a key.

A document's key is a hash of its content, not of where it is or how it tokenizes.
So a split by key holds out the same documents whatever the tokenizer, and the
documents a reduce gets, sorted by key, do not depend on the engine.

`text` maps a batch of raw text and `chat` a batch of conversations, each to a
record batch of documents in `SCHEMA`; a chat source with images, in `IMAGE_SCHEMA`.
`~bux.data.pipeline.Job.map` picks the one its source needs.
"""

import base64
import bisect
import hashlib
import itertools
import json
import re
from collections.abc import Buffer, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int32, Int64, Integer

from bux.data import images, templates
from bux.data.tokenizer import Tokenizer, utf8
from bux.export.hf import pixels
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

#: The documents of a chat source with images: `SCHEMA`, and each token's image.
#:
#: ``image`` is the key of the image each token stands for, 0 for a token of text;
#: ``images`` the document's images in order, each its key, its bytes, and its rows
#: and columns of patches.
#:
#: :meta hide-value:
IMAGE_SCHEMA = SCHEMA.append(pa.field("image", pa.list_(pa.int64()))).append(
    pa.field(
        "images",
        pa.list_(
            pa.struct(
                [
                    ("key", pa.int64()),
                    ("data", pa.binary()),
                    ("rows", pa.int32()),
                    ("cols", pa.int32()),
                ]
            )
        ),
    )
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
    pictures = [_images(c, config.images) for c in conversations]
    keys = _keys(
        json.dumps(_keyed(m, p), sort_keys=True).encode()
        for m, p in zip(conversations, pictures, strict=True)
    )
    keep = _in_split(keys, config.split)
    template = templates.load(config.template or tok.config.name)
    specials = _Specials.of(tok, template, config.images)
    docs, kept, shown = [], [], []
    for key, conversation, mine in zip(
        keys[keep],
        itertools.compress(conversations, keep),
        itertools.compress(pictures, keep),
        strict=True,
    ):
        text, starts = _render(template, specials.hide(conversation), config.train_on)
        doc = _cut(text, starts, specials.find, config.train_on == "all")
        if any(learned for _, _, learned in doc):  # without a reply, nothing to learn
            docs.append(doc)
            kept.append(key)
            shown.append(mine)
    ids, offsets, mask = _encode(docs, tok, specials.show)
    out = record_batch(np.array(kept, np.int64), ids, offsets, mask)
    if config.images is None:
        return out
    return _with_images(out, ids, offsets, shown, tok.special(config.images.token))


def _with_images(
    docs: pa.RecordBatch,
    ids: Int32[np.ndarray, " tokens"],
    offsets: Int64[np.ndarray, " _"],
    pictures: list[list["Picture"]],
    token: int,
) -> pa.RecordBatch:
    """Return `docs` in `IMAGE_SCHEMA`: each image's token repeated once for each of
    its blocks, those tokens no targets, and each token's image."""
    parts, masks, keys, ends = [], [], [], [0]
    mask = docs.column("loss_mask").flatten().to_numpy(zero_copy_only=False)
    for d, mine in enumerate(pictures):
        doc, learned = (
            ids[offsets[d] : offsets[d + 1]],
            mask[offsets[d] : offsets[d + 1]],
        )
        at = np.flatnonzero(doc == token)
        if len(at) != len(mine):
            raise ValueError(
                f"a conversation has {len(mine)} images, but its template wrote "
                f"{len(at)} image tokens"
            )
        counts = np.ones(len(doc), np.int64)
        counts[at] = [p.tokens for p in mine]
        key = np.zeros(len(doc), np.int64)
        key[at] = [p.key for p in mine]
        parts.append(np.repeat(doc, counts))
        masks.append(np.repeat(learned & (doc != token), counts))
        keys.append(np.repeat(key, counts))
        ends.append(ends[-1] + int(counts.sum()))
    bounds = pa.array(np.array(ends, np.int32))
    images_type = IMAGE_SCHEMA.field("images").type
    columns = [
        docs.column("key"),
        *(
            pa.ListArray.from_arrays(
                bounds, pa.array(np.concatenate([np.zeros(0, t), *xs]), a)
            )
            for xs, t, a in (
                (parts, np.int32, pa.int32()),
                (masks, bool, pa.bool_()),
                (keys, np.int64, pa.int64()),
            )
        ),
        pa.array(
            [
                [
                    {"key": p.key, "data": p.data, "rows": p.rows, "cols": p.cols}
                    for p in ps
                ]
                for ps in pictures
            ],
            images_type,
        ),
    ]
    return pa.RecordBatch.from_arrays(columns, schema=IMAGE_SCHEMA)


class Picture(NamedTuple):
    """An image of a conversation: its key, a hash of its bytes, its bytes, its rows
    and columns of patches, and the tokens it is written with."""

    key: int
    data: bytes
    rows: int
    cols: int
    tokens: int


def _images(
    conversation: list[dict], config: tcfg.ImagesConfig | None
) -> list[Picture]:
    """Return the images of a conversation, in order, sized as `config` says.

    Raises:
        ValueError: The conversation has images, and the source no `images`.
    """
    found = [_image_bytes(p) for m in conversation for p in _parts(m) if _is_image(p)]
    if found and config is None:
        raise ValueError(
            "a conversation has images, but its source reads none: "
            "chat(..., images=images())"
        )
    out = []
    for data in found:
        assert config is not None
        h, w = images.size(data)
        hh, ww = pixels.resized(
            h,
            w,
            size=config.size,
            merge=config.merge,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        rows, cols = hh // config.size, ww // config.size
        key = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest()) >> 1
        out.append(Picture(key or 1, data, rows, cols, rows * cols // config.merge**2))
    return out


def _parts(message: dict) -> list:
    content = message.get("content") or ""
    return [] if isinstance(content, str) else content


def _is_image(part: dict) -> bool:
    return part.get("type") in ("image", "image_url")


def _image_bytes(part: dict) -> bytes:
    """Return an image part's bytes: from its path, its bytes, Hugging Face datasets'
    ``{bytes, path}``, or a base64 ``data:`` URL."""
    value = part.get("image") if part.get("type") == "image" else part["image_url"]
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("url") or value.get("path")
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f"an image part holds no image: {part!r}")
    if value.startswith("data:"):
        return base64.b64decode(value.split(",", 1)[1])
    return Path(value).read_bytes()


def _keyed(conversation: list[dict], pictures: list[Picture]) -> list[dict]:
    """Return a conversation with its images as their keys, which a hash takes."""
    if not pictures:
        return conversation
    keys = iter(p.key for p in pictures)
    return [
        {
            **m,
            "content": [
                {"type": "image", "image": next(keys)} if _is_image(p) else p
                for p in _parts(m)
            ]
            or m.get("content"),
        }
        for m in conversation
    ]


def _content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    # The parts of a message, an image as `_IMAGE`, which `_Specials.hide` writes out.
    return "".join(_IMAGE if _is_image(p) else p.get("text") or "" for p in content)


# A special token written in a message stays text: while the template renders, it is
# this private-use character and one of its own after it, and the character itself is
# written twice.
_ESCAPE = "\U000f0000"
_ESCAPED = re.compile(f"{_ESCAPE}(.)", re.DOTALL)
# Where an image goes in a message's text, until it is written as its special tokens.
_IMAGE = "\U000ffffd"


@dataclass(frozen=True)
class _Specials:
    """The special tokens of a tokenizer and of a template: a pattern that finds them
    in rendered text, and their escapes in messages (`_ESCAPE`)."""

    find: re.Pattern
    escapes: re.Pattern  # the special tokens and the escape character
    codes: dict[str, str]  # the escape of each
    texts: dict[str, str]  # what each character after the escape stands for
    image: str  # what an image is written as, special tokens

    @classmethod
    def of(
        cls,
        tok: Tokenizer,
        template: templates.ChatTemplate,
        images: tcfg.ImagesConfig | None = None,
    ) -> "_Specials":
        tokens = sorted(tok.specials.keys() | template.specials, key=len, reverse=True)
        codes = {
            t: _ESCAPE + chr(ord(_ESCAPE) + i) for i, t in enumerate([_ESCAPE, *tokens])
        }
        find = re.compile("|".join(map(re.escape, tokens)))  # the longest first
        escapes = re.compile(f"{re.escape(_ESCAPE)}|{find.pattern}")
        texts = {c[1]: t for t, c in codes.items()}
        return cls(find, escapes, codes, texts, images.text if images else "")

    def hide(self, conversation: list[dict]) -> list[dict]:
        """Return the messages, their contents joined, special tokens escaped, and
        images written as their special tokens."""
        return [
            {**m, "content": self.escape(_content(m)).replace(_IMAGE, self.image)}
            for m in conversation
        ]

    def escape(self, text: str) -> str:
        """Return `text` with its special tokens escaped."""
        return self.escapes.sub(lambda t: self.codes[t[0]], text)

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
