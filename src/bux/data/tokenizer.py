"""Tokenizers run by gigatoken (Rust, SIMD pretokenization): strings to ids, in bulk.

The encoder never emits special tokens, like tiktoken's encode_ordinary: a web page
that contains "<|endoftext|>" is text, not the end of a document. Only `special` and
`markup` insert special tokens.
"""

import functools
import json
import re

import awkward as ak
import gigatoken
import numpy as np
import pyarrow as pa

# Resolves a Hub repo id or a path to the tokenizer.json contents. gigatoken does not
# export it, hence the pinned minor version.
from gigatoken._load.hf import to_tokenizer_json
from jaxtyping import Int32, Int64, UInt8

from bux.train import config as tcfg

_SPECIAL = re.compile(r"(<\|[^|<>]+\|>)")


class Tokenizer:
    def __init__(self, config: tcfg.TokenizerConfig):
        spec = json.loads(to_tokenizer_json(config.name))
        added = spec.get("added_tokens") or []
        self.specials = {t["content"]: t["id"] for t in added if t.get("special")}
        spec["added_tokens"] = [t for t in added if not t.get("special")]
        self.encoder = gigatoken.Tokenizer.from_json(json.dumps(spec))
        ends = (i + 1 for i in self.specials.values())
        self.vocab_size: int = max(self.encoder.vocab_size, *ends)
        for token in config.added:
            if token not in self.specials:
                self.specials[token] = self.vocab_size
                self.vocab_size += 1
        self.bos, self.eos = self.special(config.bos), self.special(config.eos)
        self._markup: dict[str, Int32[np.ndarray, " ids"]] = {}

    def special(self, token: str) -> int:
        if token not in self.specials:
            raise ValueError(
                f"the tokenizer has no special token {token!r}; add it with "
                f"tokenizer(added=[{token!r}]) if the model has room for it"
            )
        return self.specials[token]

    def encode(
        self, texts: pa.Array
    ) -> tuple[Int32[np.ndarray, " ids"], Int64[np.ndarray, " _"]]:
        """Encode every string; return the ids end to end, and n + 1 offsets where each
        string's ids start. Nulls encode to nothing."""
        start, chars = utf8(texts)
        strings = ak.contents.ListOffsetArray(
            ak.index.Index64(start),
            # awkward's ArrayLike protocol does not admit numpy arrays as typed.
            ak.contents.NumpyArray(chars, parameters={"__array__": "char"}),  # ty: ignore[invalid-argument-type]
            parameters={"__array__": "string"},
        )
        out = self.encoder.encode_batch(ak.Array(strings)).layout
        ids = np.asarray(out.content.data).view(np.int32)
        bounds = np.asarray(out.offsets, np.int64)
        return ids[bounds[0] : bounds[-1]], bounds - bounds[0]

    def markup(self, text: str) -> Int32[np.ndarray, " ids"]:
        """Encode template text, in which each <|name|> is that special token."""
        if text not in self._markup:
            self._markup[text] = self._encode_markup(text)
        return self._markup[text]

    def _encode_markup(self, text: str) -> Int32[np.ndarray, " ids"]:
        pieces = _SPECIAL.split(text)
        plain = pa.array(pieces[::2], pa.large_string())
        ids, offsets = self.encode(plain)
        out = []
        for i, piece in enumerate(pieces):
            if i % 2:
                out.append(np.array([self.special(piece)], np.int32))
            else:
                out.append(ids[offsets[i // 2] : offsets[i // 2 + 1]])
        return np.concatenate(out)


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
