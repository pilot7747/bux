"""Tokenizers run by gigatoken (Rust, SIMD pretokenization): strings to ids, in bulk.

The encoder never emits special tokens, like tiktoken's ``encode_ordinary``: a web
page that contains ``"<|endoftext|>"`` is text, not the end of a document. Only
`Tokenizer.special` and `Tokenizer.markup` insert special tokens::

    import pyarrow as pa

    from bux.data.tokenizer import load
    from bux.train.config import TokenizerConfig

    tok = load(TokenizerConfig(name="gpt2"))
    ids, offsets = tok.encode(pa.array(["Hello world", "<|endoftext|>"]))
    # ids: [15496 995 27 91 437 1659 5239 91 29], offsets: [0 2 9]
    tok.eos  # 50256
    tok.markup("<|endoftext|>x")  # [50256 87]
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
    """A BPE tokenizer of a Hugging Face Hub repo or a tokenizer.json, by gigatoken.

    The special tokens of the tokenizer.json are taken out of the encoder, so that
    only `special` and `markup` make them, and the tokens of ``config.added`` take
    the ids after the vocabulary. Use `load`, which makes one a config per process.

    Args:
        config: The run's ``tokenizer(...)`` setting.

    Attributes:
        specials: The special tokens' ids, by their text.
        encoder: gigatoken's tokenizer, without the special tokens.
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
        """Encode template text, in which each ``<|name|>`` is that special token.

        Each text is encoded once and kept, as templates repeat in every document.

        Raises:
            ValueError: If the text names a special token the tokenizer lacks.
        """
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
