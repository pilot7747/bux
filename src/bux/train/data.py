"""Batches of rows, from prepared datasets, token shards, random tokens, or mixtures.

A row is seq_len tokens: input_ids, position_ids (where each token is in its
document, so a 0 starts one) and loss_mask (whether each token is a target). Every
source is an endless stream of rows, each epoch in a new order. Batch i holds rows
i * n to (i + 1) * n of each source, n of them in proportion to its weight, so it
depends only on the seed and i: a run that resumes at step i reads what it would
have read had it never stopped.
"""

import functools
import glob
import itertools
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pyarrow.parquet as pq

from bux.data.pipeline import DEFAULT_CACHE, Job
from bux.train import config as tcfg


class Batch(NamedTuple):
    input_ids: np.ndarray  # (rows, seq_len) int32
    position_ids: np.ndarray  # int32
    loss_mask: np.ndarray  # bool

    @classmethod
    def concat(cls, parts: Sequence[Batch]) -> Batch:
        return cls(*(np.concatenate(c) for c in zip(*parts, strict=True)))


def plain(ids: np.ndarray) -> Batch:
    """Rows that each read as one text, every token a target."""
    positions = np.broadcast_to(np.arange(ids.shape[1], dtype=np.int32), ids.shape)
    return Batch(ids.astype(np.int32), positions, np.ones(ids.shape, bool))


class Source(Protocol):
    def rows(self, start: int, n: int) -> Batch:
        """Rows `start` to `start + n` of the source's endless stream."""
        ...


@functools.lru_cache(maxsize=8)
def _order(n: int, seed: tuple[int, ...], epoch: int) -> np.ndarray:
    """A permutation of range(n), a new one every epoch."""
    return np.random.default_rng((*seed, epoch)).permutation(n)


class Prepared:
    """A dataset bux.data prepared: its row groups, each epoch in a new order."""

    def __init__(
        self, folder: Path, vocab_size: int, seq_len: int, seed: tuple[int, ...]
    ):
        if not (folder / "dataset.json").exists():
            raise FileNotFoundError(f"{folder} is not prepared: bux prepare does it")
        meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        if meta["vocab_size"] > vocab_size:
            raise ValueError(
                f"{folder} is in {meta['vocab_size']} tokens, more than the model's "
                f"vocab_size of {vocab_size}"
            )
        self.files = [pq.ParquetFile(folder / part) for part in meta["parts"]]
        self.groups = [
            (f, g)
            for f, file in enumerate(self.files)
            for g in range(file.num_row_groups)
        ]
        self.sizes = np.array(
            [self.files[f].metadata.row_group(g).num_rows for f, g in self.groups],
            np.int64,
        )
        self.total = int(self.sizes.sum())
        if not self.total:
            raise ValueError(f"{folder} has no rows: its documents fill none")
        self.seq_len, self.seed = seq_len, seed
        self.last: tuple[int, Batch] | None = None  # rows are read in order

    def _group(self, i: int) -> Batch:
        if self.last is None or self.last[0] != i:
            f, g = self.groups[i]
            table = self.files[f].read_row_group(g)
            columns = (
                table[name].combine_chunks().flatten().to_numpy(zero_copy_only=False)
                for name in Batch._fields
            )
            self.last = i, Batch(*(c.reshape(-1, self.seq_len) for c in columns))
        return self.last[1]

    def rows(self, start: int, n: int) -> Batch:
        parts = []
        while n:
            epoch, at = divmod(start, self.total)
            order = _order(len(self.groups), self.seed, epoch)
            ends = np.cumsum(self.sizes[order])
            k = int(np.searchsorted(ends, at, side="right"))
            first = at - int(ends[k] - self.sizes[order[k]])
            take = min(n, int(ends[k]) - at)
            group = self._group(int(order[k]))
            parts.append(Batch(*(c[first : first + take] for c in group)))
            start, n = start + take, n - take
        return Batch.concat(parts)


class Shards:
    """Windows of memory-mapped shards of tokens, each epoch in a new order. They
    overlap by a token, so that every token is a target once."""

    def __init__(self, c: tcfg.TokensConfig, seq_len: int, seed: tuple[int, ...]):
        paths = sorted(glob.glob(c.path))
        if not paths:
            raise FileNotFoundError(f"no shards match {c.path!r}")
        self.shards = [
            np.memmap(p, dtype=c.dtype, mode="r", offset=c.header) for p in paths
        ]
        self.counts = np.array(
            [max(0, (len(s) - 1) // (seq_len - 1)) for s in self.shards]
        )
        self.ends = np.cumsum(self.counts)
        if not self.ends[-1]:
            raise ValueError(
                f"every shard of {c.path!r} is shorter than {seq_len} tokens"
            )
        self.seq_len, self.seed = seq_len, seed

    def rows(self, start: int, n: int) -> Batch:
        total = int(self.ends[-1])
        epoch, at = np.divmod(np.arange(start, start + n), total)
        windows = np.concatenate(
            [_order(total, self.seed, int(e))[at[epoch == e]] for e in np.unique(epoch)]
        )
        shard = np.searchsorted(self.ends, windows, side="right")
        offset = (windows - self.ends[shard] + self.counts[shard]) * (self.seq_len - 1)
        return plain(
            np.stack(
                [self.shards[s][o : o + self.seq_len] for s, o in zip(shard, offset)]
            )
        )


class Synthetic:
    def __init__(self, vocab_size: int, seq_len: int, seed: tuple[int, ...]):
        self.vocab_size, self.seq_len, self.seed = vocab_size, seq_len, seed

    def rows(self, start: int, n: int) -> Batch:
        rng = np.random.default_rng((*self.seed, start))
        return plain(rng.integers(0, self.vocab_size, (n, self.seq_len), np.int32))


def source(
    c: tcfg.Source,
    t: tcfg.TrainConfig,
    vocab_size: int,
    cache: str | Path,
    seed: tuple[int, ...],
) -> Source:
    match c:
        case tcfg.TextConfig() | tcfg.ChatConfig():
            folder = Path(cache) / Job.of(c, t).name
            return Prepared(folder, vocab_size, t.seq_len, seed)
        case tcfg.TokensConfig():
            return Shards(c, t.seq_len, seed)
        case tcfg.SyntheticConfig():
            return Synthetic(vocab_size, t.seq_len, seed)
    raise AssertionError(c)


def apportion(weights: Sequence[float], n: int) -> list[int]:
    """`n` in proportion to `weights`, to the largest remainders first."""
    quota = np.asarray(weights) / sum(weights) * n
    counts = np.floor(quota).astype(int)
    counts[np.argsort(counts - quota, kind="stable")[: n - counts.sum()]] += 1
    return counts.tolist()


def batches(
    c: tcfg.DataConfig,
    t: tcfg.TrainConfig,
    vocab_size: int,
    *,
    cache: str | Path = DEFAULT_CACHE,
    start: int = 0,
    stream: int = 0,
) -> Iterator[Batch]:
    """Batches of t.batch rows of t.seq_len tokens, from batch `start` on. Streams
    with different numbers draw independently: training is 0, evaluation 1."""
    parts = (
        c.sources
        if isinstance(c, tcfg.MixtureConfig)
        else (tcfg.Weighted(weight=1, source=c),)
    )
    counts = apportion([p.weight for p in parts], t.batch)
    for p, n in zip(parts, counts, strict=True):
        if not n:
            raise ValueError(
                f"a batch of {t.batch} rows has no room for a source of weight "
                f"{p.weight / sum(q.weight for q in parts):.3g}"
            )
    sources = [
        source(p.source, t, vocab_size, cache, (t.seed, stream, k))
        for k, p in enumerate(parts)
    ]
    for i in itertools.count(start):
        batch = Batch.concat([s.rows(i * n, n) for s, n in zip(sources, counts)])
        if (top := int(batch.input_ids.max())) >= vocab_size:
            raise ValueError(
                f"token id {top} is outside the vocabulary of {vocab_size}"
            )
        yield batch
