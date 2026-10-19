"""The reduce: a chunk of documents to rows of `length` tokens.

A plan lays the rows out as pieces of documents end to end, `length` tokens a row,
padding included as pieces of a pad region after the documents. Any range of rows
is then one gather, so that a chunk is written a row group at a time.
"""

import itertools
from bisect import bisect_left, insort
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from bux.train import config as tcfg

SCHEMA = pa.schema(
    [
        ("input_ids", pa.list_(pa.int32())),
        ("position_ids", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.bool_())),
    ]
)


def flat(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """A list column's values end to end, and n + 1 offsets into them."""
    lists = column.cast(pa.large_list(column.type.value_type)).combine_chunks()
    offsets = lists.offsets.to_numpy()
    return lists.flatten().to_numpy(zero_copy_only=False), offsets - offsets[0]


@dataclass(frozen=True)
class Docs:
    """Documents in key order, end to end, then a row of padding."""

    ids: np.ndarray
    mask: np.ndarray
    positions: np.ndarray  # of every token in its document, and in the padding
    offsets: np.ndarray  # n + 1, where the documents start; the last is the padding's

    @classmethod
    def of(cls, table: pa.Table, length: int) -> Docs:
        table = table.sort_by("key")  # whatever order an engine delivered them in
        ids, offsets = flat(table["input_ids"])
        mask, _ = flat(table["loss_mask"])
        positions = np.arange(len(ids)) - np.repeat(offsets[:-1], np.diff(offsets))
        return cls(
            np.concatenate([ids, np.zeros(length, np.int32)]),
            np.concatenate([mask, np.zeros(length, bool)]),
            np.concatenate([positions, np.arange(length)]).astype(np.int32),
            offsets,
        )


@dataclass(frozen=True)
class Plan:
    starts: np.ndarray  # of the pieces, in row order
    lengths: np.ndarray
    rows: np.ndarray  # r + 1, where every row's pieces start
    stats: Counter


def _pieces(docs: Docs, long: str, length: int):
    """Pieces of the documents: whole when they fit, else dropped, truncated, or split
    into pieces of `length` that overlap by a token, so that no target is lost."""
    sizes = np.diff(docs.offsets)
    over = sizes > length
    if long == "split":
        count = np.where(over, (sizes - 2) // (length - 1) + 1, 1)
    else:
        count = np.where(over & (long == "drop"), 0, 1)
    doc = np.repeat(np.arange(len(sizes)), count)
    j = np.arange(len(doc)) - np.repeat(np.cumsum(count) - count, count)
    starts = docs.offsets[:-1][doc] + j * (length - 1)
    lengths = np.minimum(sizes[doc] - j * (length - 1), length)
    cut = {"drop": sizes[over].sum(), "truncate": (sizes[over] - length).sum()}
    what = {"drop": "dropped", "truncate": "truncated", "split": "split"}[long]
    return (
        starts,
        lengths,
        Counter({what: int(over.sum()), "cut": int(cut.get(long, 0))}),
    )


def _next_fit(sizes: np.ndarray, length: int, max_docs: int | None) -> list[list[int]]:
    """Pieces in order, a new row whenever the next does not fit in the last."""
    rows: list[list[int]] = []
    room = 0
    for piece, n in enumerate(sizes.tolist()):
        if n > room or len(rows[-1]) == max_docs:
            rows.append([])
            room = length
        rows[-1].append(piece)
        room -= n
    return rows


def _best_fit(
    sizes: np.ndarray, length: int, max_docs: int | None, sort: bool
) -> list[list[int]]:
    """Each piece, the longest first, into the open row with the least room for it."""
    order = np.argsort(-sizes, kind="stable") if sort else np.arange(len(sizes))
    rows: list[list[int]] = []
    rooms: list[int] = []  # the distinct rooms of open rows, ascending
    open_rows: dict[int, list[int]] = {}  # by their room
    for piece, n in zip(order.tolist(), sizes[order].tolist(), strict=True):
        i = bisect_left(rooms, n)
        if i < len(rooms):
            room = rooms[i]
            row = open_rows[room].pop()
            if not open_rows[room]:
                del open_rows[room], rooms[i]
        else:
            room, row = length, len(rows)
            rows.append([])
        rows[row].append(piece)
        room -= n
        if room and len(rows[row]) != max_docs:
            if room not in open_rows:
                insort(rooms, room)
                open_rows[room] = []
            open_rows[room].append(row)
    return rows


def _layout(rows, starts, lengths, length: int, pad: int, stats: Counter) -> Plan:
    """The rows' pieces end to end, each row's padded to `length`."""
    counts = np.array([len(r) for r in rows], np.int64)
    flat = np.fromiter(itertools.chain.from_iterable(rows), np.int64, counts.sum())
    ends = np.cumsum(counts)
    used = np.add.reduceat(lengths[flat], ends - counts) if rows else counts
    stats.update(rows=len(rows), padding=int((length - used).sum()))
    return Plan(
        np.insert(starts[flat], ends, pad),
        np.insert(lengths[flat], ends, length - used),
        np.concatenate([[0], ends + np.arange(1, len(rows) + 1)]),
        stats,
    )


def plan(docs: Docs, c: tcfg.PackingConfig, length: int, seed: int) -> Plan:
    """Rows of the documents in `docs`, in a random order."""
    rng = np.random.default_rng(seed)
    tokens = int(docs.offsets[-1])
    seen = Counter(docs=len(docs.offsets) - 1, tokens=tokens)
    if isinstance(c, tcfg.ConcatPacking):
        # Windows [i (length - 1), i (length - 1) + length): what does not fill one is cut.
        n = max(0, (tokens - 1) // (length - 1))
        seen.update(rows=n, cut=tokens - (n * (length - 1) + 1 if n else 0))
        starts = rng.permutation(n) * (length - 1)
        return Plan(starts, np.full(n, length), np.arange(n + 1), seen)
    starts, lengths, stats = _pieces(docs, c.long, length)
    if isinstance(c, tcfg.BestFitPacking):
        rows = _best_fit(lengths, length, c.max_docs, c.sort)
    else:
        rows = _next_fit(lengths, length, c.max_docs)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    seen.update(stats)
    return _layout(rows, starts, lengths, length, tokens, seen)


def materialize(
    docs: Docs, plan: Plan, length: int, doc_mask: bool, size: int
) -> Iterator[pa.RecordBatch]:
    """The planned rows, `size` at a time. Counts the targets in plan.stats as it goes."""
    for a in range(0, len(plan.rows) - 1, size):
        lo, hi = plan.rows[a], plan.rows[min(a + size, len(plan.rows) - 1)]
        starts, lengths = plan.starts[lo:hi], plan.lengths[lo:hi]
        first = np.cumsum(lengths) - lengths
        at = np.arange(lengths.sum())
        index = np.repeat(starts - first, lengths) + at
        # A piece that continues a document starts at position 0, as rows do.
        within = at - np.repeat(first, lengths)
        positions = (
            np.minimum(docs.positions[index], within) if doc_mask else at % length
        )
        mask = docs.mask[index]
        plan.stats["targets"] += int((mask & (positions > 0)).sum())
        bounds = pa.array(np.arange(0, len(at) + 1, length, dtype=np.int32))
        columns = [docs.ids[index], positions.astype(np.int32), mask]
        yield pa.RecordBatch.from_arrays(
            [pa.ListArray.from_arrays(bounds, pa.array(c)) for c in columns],
            schema=SCHEMA,
        )
