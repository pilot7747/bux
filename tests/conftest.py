"""Fixtures for the data tests: a tokenizer and a corpus that need no network."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bux.train import config as tcfg

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def _byte_alphabet() -> dict[int, str]:
    """GPT-2's byte-level alphabet: a printable character for every byte."""
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rest = [b for b in range(256) if b not in printable]
    return {
        b: chr(c)
        for b, c in zip(
            printable + rest, printable + [256 + i for i in range(len(rest))]
        )
    }


@pytest.fixture(scope="session")
def byte_tokenizer(tmp_path_factory) -> tcfg.TokenizerConfig:
    """A byte-level BPE without merges: every byte is the token of its value, and the
    special tokens come after them."""
    spec = {
        "version": "1.0",
        "added_tokens": [
            {"id": 256 + i, "content": s, "special": True, "normalized": False}
            | {"single_word": False, "lstrip": False, "rstrip": False}
            for i, s in enumerate(SPECIALS)
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {c: b for b, c in _byte_alphabet().items()},
            "merges": [],
        },
    }
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_text(json.dumps(spec))
    return tcfg.TokenizerConfig(name=str(path))


def corpus(folder, n=600) -> str:
    """A glob of two Parquet files of `n` short texts, in row groups of 50."""
    rng = np.random.default_rng(0)
    words = ["data", "pack", "row", "token", "fits", "best", "chunk", "shuffle"]
    texts = [" ".join(rng.choice(words, rng.integers(1, 30))) for _ in range(n)]
    folder.mkdir()
    pq.write_table(
        pa.table({"text": texts[: n // 2]}), folder / "a.parquet", row_group_size=50
    )
    pq.write_table(
        pa.table({"text": texts[n // 2 :]}), folder / "b.parquet", row_group_size=50
    )
    return str(folder / "*.parquet")
