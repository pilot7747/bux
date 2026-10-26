"""Tokenizers: strings to ids, in bulk, by gigatoken (Rust, SIMD pretokenization), or
by Hugging Face's tokenizers for the tokenizers gigatoken cannot run.

The encoder never emits special tokens, like tiktoken's ``encode_ordinary``: a web
page that contains ``"<|endoftext|>"`` is text, not the end of a document. Only
`Tokenizer.special` gives their ids, for bos, eos and chat templates::

    import pyarrow as pa

    from bux.data.tokenizer import load
    from bux.train.config import TokenizerConfig

    tok = load(TokenizerConfig(name="gpt2"))
    ids, offsets = tok.encode(pa.array(["Hello world", "<|endoftext|>"]))
    # ids: [15496 995 27 91 437 1659 5239 91 29], offsets: [0 2 9]
    tok.eos  # 50256
    tok.decode([15496, 995, tok.eos])  # 'Hello world<|endoftext|>'
"""

import functools
import itertools
import json
from collections.abc import Sequence

import awkward as ak
import gigatoken
import numpy as np
import pyarrow as pa
import tokenizers

# Resolves a Hub repo id or a path to the tokenizer.json contents. gigatoken does not
# export it, hence the pinned minor version.
from gigatoken._load.hf import to_tokenizer_json
from jaxtyping import Int32, Int64, Integer, UInt8

from bux.train import config as tcfg


class Tokenizer:
    """A tokenizer of a Hugging Face Hub repo or a tokenizer.json.

    The encoder writes the special tokens of the tokenizer.json in text as text, so
    that only `special` makes them, and the tokens of ``config.added`` take the ids
    after the vocabulary. Use `load`, which makes one a config per process.

    Args:
        config: The run's ``tokenizer(...)`` setting.

    Attributes:
        config: That setting.
        specials: The special tokens' ids, by their text.
        encoder: What encodes text, special tokens as text: `Gigatoken`, or
            `HFTokenizers` for a tokenizer gigatoken cannot run (SmolLM2's).
        vocab_size: The ids the tokenizer makes, special and added ones included;
            the model's ``vocab_size`` must be at least this.
        bos: The id of ``config.bos``.
        eos: The id of ``config.eos``.

    Raises:
        ValueError: If `config`'s bos or eos is not a special token.
    """

    def __init__(self, config: tcfg.TokenizerConfig):
        spec = json.loads(to_tokenizer_json(config.name))
        added = spec.get("added_tokens") or []
        self.config = config
        self.specials = {t["content"]: t["id"] for t in added if t.get("special")}
        self.encoder = _encoder(spec)
        ends = (i + 1 for i in self.specials.values())
        self.vocab_size: int = max(self.encoder.vocab_size, *ends)
        for token in config.added:
            if token not in self.specials:
                self.specials[token] = self.vocab_size
                self.vocab_size += 1
        self.bos, self.eos = self.special(config.bos), self.special(config.eos)

    def special(self, token: str) -> int:
        """Return the id of the special token `token`, such as ``"<|im_end|>"``.

        Raises:
            ValueError: If the tokenizer has no such special token; the message says
                to add it with ``tokenizer(added=[...])``.
        """
        if token not in self.specials:
            raise ValueError(
                f"the tokenizer has no special token {token!r}; add it with "
                f"tokenizer(added=[{token!r}]) if the model has room for it"
            )
        return self.specials[token]

    def encode(
        self, texts: pa.Array
    ) -> tuple[Int32[np.ndarray, " ids"], Int64[np.ndarray, " _"]]:
        """Encode every string of `texts`, as text only: no special tokens.

        Args:
            texts: An Arrow array of n strings; nulls encode to nothing.

        Returns:
            The ids of all of them end to end, and n + 1 offsets into them. The
            i-th string's ids are ``ids[offsets[i]:offsets[i + 1]]``.
        """
        return self.encoder.encode(texts)

    def decode(self, ids: Integer[np.ndarray, " ids"] | Sequence[int]) -> str:
        """Return the text of `ids`, each special token as its own text."""
        names = {i: s for s, i in self.specials.items()}
        flat: list[int] = np.asarray(ids).tolist()
        out = []
        for name, run in itertools.groupby(flat, names.get):
            run = list(run)
            out.append(self.encoder.decode(run) if name is None else name * len(run))
        return "".join(out)

    def token_bytes(self) -> Int32[np.ndarray, " vocab"]:
        """Return the bytes of text each id stands for, its UTF-8 bytes; 0 for a
        special token, which stands for none. Bits per byte divide by them."""
        vocab = self.encoder.vocab
        lengths = np.zeros(self.vocab_size, np.int32)
        lengths[list(vocab)] = [len(piece) for piece in vocab.values()]
        lengths[list(self.specials.values())] = 0
        return lengths


class Gigatoken:
    """gigatoken's BPE, at hundreds of megabytes a second.

    Raises:
        RuntimeError: If gigatoken cannot run the tokenizer: one that is not BPE, or
            whose byte-level vocabulary lacks some bytes, as SmolLM2's.
    """

    def __init__(self, spec: dict):
        added = spec.get("added_tokens") or []
        text = [t for t in added if not t.get("special")]  # the special ones stay text
        self.tokenizer = gigatoken.Tokenizer.from_json(
            json.dumps(spec | {"added_tokens": text})
        )
        self.vocab_size: int = self.tokenizer.vocab_size

    def encode(
        self, texts: pa.Array
    ) -> tuple[Int32[np.ndarray, " ids"], Int64[np.ndarray, " _"]]:
        """Return the ids of n strings end to end, and n + 1 offsets into them."""
        start, chars = utf8(texts)
        strings = ak.contents.ListOffsetArray(
            ak.index.Index64(start),
            # awkward's ArrayLike protocol does not admit numpy arrays as typed.
            ak.contents.NumpyArray(chars, parameters={"__array__": "char"}),  # ty: ignore[invalid-argument-type]
            parameters={"__array__": "string"},
        )
        out = self.tokenizer.encode_batch(ak.Array(strings)).layout
        ids = np.asarray(out.content.data).view(np.int32)
        bounds = np.asarray(out.offsets, np.int64)
        return ids[bounds[0] : bounds[-1]], bounds - bounds[0]

    def decode(self, ids: list[int]) -> str:
        """Return the text of `ids`, none of them special."""
        return self.tokenizer.decode(ids).decode(errors="replace")

    @functools.cached_property
    def vocab(self) -> dict[int, bytes]:
        """Each id's bytes of text."""
        return self.tokenizer.vocab


class HFTokenizers:
    """Hugging Face's tokenizers, for what gigatoken cannot run: any tokenizer.json
    (BPE, WordPiece, Unigram), an order of magnitude slower."""

    def __init__(self, spec: dict):
        # Every added token stays, as it numbers them in order, whatever their ids.
        self.tokenizer = tokenizers.Tokenizer.from_str(json.dumps(spec))
        self.tokenizer.encode_special_tokens = True  # special tokens in text are text
        self.vocab_size: int = max(self.tokenizer.get_vocab().values()) + 1

    def encode(
        self, texts: pa.Array
    ) -> tuple[Int32[np.ndarray, " ids"], Int64[np.ndarray, " _"]]:
        """Return the ids of n strings end to end, and n + 1 offsets into them."""
        strings = texts.cast(pa.large_string()).fill_null("").to_pylist()
        out = self.tokenizer.encode_batch_fast(strings, add_special_tokens=False)
        rows = [e.ids for e in out]
        lengths = np.array([0, *map(len, rows)], np.int64)
        ids = np.fromiter(itertools.chain.from_iterable(rows), np.int32, lengths.sum())
        return ids, np.cumsum(lengths)

    def decode(self, ids: list[int]) -> str:
        """Return the text of `ids`, none of them special."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @functools.cached_property
    def vocab(self) -> dict[int, bytes]:
        """Each id's bytes of text: exactly for a byte-level vocabulary, by GPT-2's
        alphabet, and for any other as its decoder writes the token alone."""
        added = self.tokenizer.get_added_tokens_decoder()  # text, as written
        pieces = self.tokenizer.get_vocab(with_added_tokens=False)
        if isinstance(self.tokenizer.decoder, tokenizers.decoders.ByteLevel):
            byte = {c: b for b, c in _byte_alphabet().items()}
            vocab = {i: bytes(byte[c] for c in p) for p, i in pieces.items()}
        else:
            vocab = {i: self.tokenizer.decode([i]).encode() for i in pieces.values()}
        return vocab | {i: t.content.encode() for i, t in added.items()}


#: What encodes text with a tokenizer.json, its special tokens as text.
Encoder = Gigatoken | HFTokenizers


def _encoder(spec: dict) -> Encoder:
    """Return gigatoken's encoder of a tokenizer.json, or Hugging Face's if gigatoken
    cannot run it."""
    try:
        return Gigatoken(spec)
    except RuntimeError:
        return HFTokenizers(spec)


def _byte_alphabet() -> dict[int, str]:
    """GPT-2's byte-level alphabet: a printable character for every byte."""
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rest = [b for b in range(256) if b not in printable]
    chars = printable + [256 + i for i in range(len(rest))]
    return {b: chr(c) for b, c in zip(printable + rest, chars, strict=True)}


def utf8(
    texts: pa.Array,
) -> tuple[Int64[np.ndarray, " _"], UInt8[np.ndarray, " bytes"]]:
    """Return the bytes of strings without a copy: n + 1 offsets into one array."""
    texts = texts.cast(pa.large_string())
    _, offsets, data = texts.buffers()
    start = np.frombuffer(offsets, np.int64, len(texts) + 1, texts.offset * 8)
    return start, np.frombuffer(data or b"", np.uint8)


@functools.cache
def load(config: tcfg.TokenizerConfig) -> Tokenizer:
    """Load the tokenizer that `config` describes, once per process."""
    return Tokenizer(config)
