"""The map: turn records into documents, each with token ids, a loss mask and a key.

A document's key is a hash of its content, not of where it is or how it tokenizes.
So a split by key holds out the same documents whatever the tokenizer, and the
documents a reduce gets, sorted by key, do not depend on the engine.

`text` maps a batch of raw text and `chat` a batch of conversations, each to a
record batch of documents in `SCHEMA`. `~bux.data.pipeline.Job.map` picks the one
its source needs.
"""

import hashlib
import itertools
import json
from collections.abc import Buffer, Iterable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int64, Integer

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


@dataclass(frozen=True)
class Template:
    """A chat template: the text around each message, special tokens as ``<|name|>``.

    Attributes:
        start: Before the first message.
        header: Before each message, with its ``{role}``.
        end: After each message: the token that ends a reply, learned with it.
        sep: After `end`.
    """

    start: str
    header: str
    end: str
    sep: str = ""


#: The chat templates that ``chat(template=...)`` takes, by name.
TEMPLATES = {
    "chatml": Template("", "<|im_start|>{role}\n", "<|im_end|>", "\n"),
    "llama3": Template(
        "<|begin_of_text|>",
        "<|start_header_id|>{role}<|end_header_id|>\n\n",
        "<|eot_id|>",
    ),
}


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


def _content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content)  # parts of a message


def chat(
    batch: pa.RecordBatch, config: tcfg.ChatConfig, tok: Tokenizer
) -> pa.RecordBatch:
    """Map a batch of conversations to documents, rendered with a chat template.

    A row is a conversation in the OpenAI format, a list of ``{role, content}``
    messages, ``content`` a string or a list of ``{type, text}`` parts; it is keyed
    by a hash of its messages. Each message is its template's header, its content,
    the template's end and its separator. The loss mask follows `train_on`: every
    assistant reply with the end that closes it (``"assistant"``), only the last
    reply (``"last"``), or every token (``"all"``). A conversation with nothing to
    learn is left out, as are those outside the source's `split`.

    Args:
        batch: Records with the source's column of messages.
        config: The ``chat(...)`` source.
        tok: The run's tokenizer, which must have the template's special tokens.

    Returns:
        The documents, in `SCHEMA`.

    Raises:
        ValueError: If the tokenizer lacks a special token of the template.
    """
    conversations = batch.column(config.column).to_pylist()
    keys = _keys(json.dumps(m, sort_keys=True).encode() for m in conversations)
    keep = _in_split(keys, config.split)
    conversations = list(itertools.compress(conversations, keep))
    messages = [m for messages in conversations for m in messages]
    contents = pa.array([_content(m) for m in messages], pa.large_string())
    ids, offsets = tok.encode(contents)
    template, train_all = TEMPLATES[config.template], config.train_on == "all"
    docs, masks, kept_keys, base = [], [], [], 0
    for key, conversation in zip(keys[keep], conversations, strict=True):
        roles = [m["role"] for m in conversation]
        last = max((i for i, r in enumerate(roles) if r == "assistant"), default=-1)
        pieces = [(tok.markup(template.start), train_all)]
        for i, role in enumerate(roles):
            learn = train_all or (
                role == "assistant" and (config.train_on == "assistant" or i == last)
            )
            body = ids[offsets[base + i] : offsets[base + i + 1]]
            pieces += [
                (tok.markup(template.header.format(role=role)), train_all),
                (body, learn),
                (tok.markup(template.end), learn),
                (tok.markup(template.sep), train_all),
            ]
        base += len(roles)
        mask = np.concatenate([np.full(len(p), learn) for p, learn in pieces])
        if mask.any():  # a conversation without a reply teaches nothing
            docs.append(np.concatenate([p for p, _ in pieces]))
            masks.append(mask)
            kept_keys.append(key)
    lengths = np.array([len(d) for d in docs], np.int64)
    return record_batch(
        np.array(kept_keys, np.int64),
        np.concatenate(docs or [np.zeros(0, np.int32)]),
        np.concatenate([[0], np.cumsum(lengths)]),
        np.concatenate(masks or [np.zeros(0, bool)]),
    )
