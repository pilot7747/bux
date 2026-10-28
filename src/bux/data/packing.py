"""The reduce: pack a chunk of documents into rows of `length` tokens.

A plan lays out the rows as pieces of documents end to end, `length` tokens per row,
with padding as pieces of a pad region after the documents. Any range of rows is then
one gather, so a chunk is written one row group at a time.

`Docs.of` reads a chunk's documents, `plan` packs them as the run's packing says
(``concat``, ``greedy`` or ``best_fit``), and `materialize` writes the planned rows
in `SCHEMA`.

Documents with images (`bux.data.documents.IMAGE_SCHEMA`) are packed whole: a row
also holds no more patches than its room for them, and a document that fits in no
row is dropped. Their rows, in `IMAGE_SCHEMA`, say which image each token stands for.
"""

import itertools
from bisect import bisect_left, insort
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int, Int32, Int64, Shaped

from bux.train import config as tcfg

#: The rows a prepared dataset holds, each three lists of `length` values.
#:
#: ``input_ids`` the tokens, ``position_ids`` where each is in its document (or in
#: the row, without a document mask), and ``loss_mask`` whether each is a target.
#:
#: :meta hide-value:
SCHEMA = pa.schema(
    [
        ("input_ids", pa.list_(pa.int32())),
        ("position_ids", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.bool_())),
    ]
)

#: The rows of a dataset with images: `SCHEMA`, and ``image``, the key of the image
#: each token stands for, 0 for a token of text (`bux.data.images.Store`).
#:
#: :meta hide-value:
IMAGE_SCHEMA = SCHEMA.append(pa.field("image", pa.list_(pa.int64())))


def flat(
    column: pa.ChunkedArray,
) -> tuple[Shaped[np.ndarray, " values"], Int64[np.ndarray, " _"]]:
    """Return a list column's values end to end, and n + 1 offsets into them."""
    lists = column.cast(pa.large_list(column.type.value_type)).combine_chunks()
    offsets = lists.offsets.to_numpy()
    return lists.flatten().to_numpy(zero_copy_only=False), offsets - offsets[0]


@dataclass(frozen=True)
class Docs:
    """Documents in key order, end to end, then a row of padding.

    Attributes:
        ids: Every token, then `length` zeros of padding.
        mask: Whether each is a target; the padding is not.
        positions: Each token's position in its document, or in the padding.
        offsets: n + 1 offsets: where each document starts, and then where the
            padding does.
        image: Each token's image's key, 0 for text, then the padding's 0; None for
            documents without images.
        patches: Each document's patches; None without images.
    """

    ids: Int32[np.ndarray, " tokens"]
    mask: Bool[np.ndarray, " tokens"]
    positions: Int32[np.ndarray, " tokens"]
    offsets: Int64[np.ndarray, " _"]
    image: Int64[np.ndarray, " tokens"] | None = None
    patches: Int64[np.ndarray, " docs"] | None = None

    @classmethod
    def of(cls, table: pa.Table, length: int, merge: int | None = None) -> "Docs":
        """Return the documents of `table` (in the map's schema, in any order),
        sorted by key, with a row of `length` tokens of padding after them; with
        `merge`, the patches on a side of a block an image token stands for, their
        images too."""
        table = table.sort_by("key")  # whatever order an engine delivered them in
        ids, offsets = flat(table["input_ids"])
        mask, _ = flat(table["loss_mask"])
        positions = np.arange(len(ids)) - np.repeat(offsets[:-1], np.diff(offsets))
        image = patches = None
        if merge is not None:
            image = flat(table["image"])[0].astype(np.int64)
            tokens = np.add.reduceat(image != 0, offsets[:-1]) if len(ids) else []
            patches = np.asarray(tokens, np.int64)[: len(offsets) - 1] * merge * merge
            image = np.concatenate([image, np.zeros(length, np.int64)])
        return cls(
            np.concatenate([ids, np.zeros(length, np.int32)]),
            np.concatenate([mask, np.zeros(length, bool)]),
            np.concatenate([positions, np.arange(length)]).astype(np.int32),
            offsets,
            image,
            patches,
        )


@dataclass(frozen=True)
class Plan:
    """The rows of a chunk, as pieces of `Docs` laid end to end.

    Row ``r`` is the pieces ``rows[r]:rows[r + 1]``, which add up to `length`
    tokens; a piece is ``starts[i]:starts[i] + lengths[i]`` of `Docs`, padding
    included.

    Attributes:
        starts: Where each piece starts in `Docs`, in row order.
        lengths: Each piece's tokens.
        rows: r + 1 offsets: where each row's pieces start.
        stats: Counts of documents, tokens, rows, padding, tokens cut and
            documents dropped, truncated or split; `materialize` adds targets.
    """

    starts: Int[np.ndarray, " pieces"]
    lengths: Int[np.ndarray, " pieces"]
    rows: Int[np.ndarray, " _"]
    stats: Counter


def _pieces(
    docs: Docs, long: str, length: int
) -> tuple[Int[np.ndarray, " pieces"], Int[np.ndarray, " pieces"], Counter]:
    """Cut the documents into pieces: whole when they fit; else dropped, truncated, or
    split into pieces of `length` that overlap by one token, so that no target is lost.
    """
    sizes = np.diff(docs.offsets)
    too_long = sizes > length
    if long == "split":
        count = np.where(too_long, (sizes - 2) // (length - 1) + 1, 1)
    else:
        count = np.where(too_long & (long == "drop"), 0, 1)
    doc = np.repeat(np.arange(len(sizes)), count)
    j = np.arange(len(doc)) - np.repeat(np.cumsum(count) - count, count)
    starts = docs.offsets[:-1][doc] + j * (length - 1)
    lengths = np.minimum(sizes[doc] - j * (length - 1), length)
    cut = {"drop": sizes[too_long].sum(), "truncate": (sizes[too_long] - length).sum()}
    stat = {"drop": "dropped", "truncate": "truncated", "split": "split"}[long]
    return (
        starts,
        lengths,
        Counter({stat: int(too_long.sum()), "cut": int(cut.get(long, 0))}),
    )


def _next_fit(
    sizes: Int[np.ndarray, " pieces"], length: int, max_docs: int | None
) -> list[list[int]]:
    """Pack pieces in order, starting a new row whenever the next one does not fit."""
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
    sizes: Int[np.ndarray, " pieces"], length: int, max_docs: int | None, sort: bool
) -> list[list[int]]:
    """Put each piece, longest first, into the open row with the least room for it."""
    order = np.argsort(-sizes, kind="stable") if sort else np.arange(len(sizes))
    rows: list[list[int]] = []
    rooms: list[int] = []  # the open rows' distinct free space, ascending
    open_rows: dict[int, list[int]] = {}  # open rows by their free space
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


def _fit_both(
    sizes: Int[np.ndarray, " pieces"],
    patches: Int[np.ndarray, " pieces"],
    length: int,
    room: int,
    max_docs: int | None,
    sort: bool,
) -> list[list[int]]:
    """Put each piece, longest first unless not `sort`, into the open row with the
    least room for its tokens that also has room for its patches."""
    order = np.argsort(-sizes, kind="stable") if sort else np.arange(len(sizes))
    rows: list[list[int]] = []
    open_rows: list[tuple[int, int]] = []  # (tokens of room, row), ascending
    left: dict[int, int] = {}  # each open row's patches of room
    for piece in order.tolist():
        n, m = int(sizes[piece]), int(patches[piece])
        i = bisect_left(open_rows, (n, -1))
        while i < len(open_rows) and left[open_rows[i][1]] < m:
            i += 1  # rows with room for the tokens, but not the patches
        if i < len(open_rows):
            tokens, row = open_rows.pop(i)
        else:
            tokens, row = length, len(rows)
            rows.append([])
            left[row] = room
        rows[row].append(piece)
        tokens, left[row] = tokens - n, left[row] - m
        if tokens and len(rows[row]) != max_docs:
            insort(open_rows, (tokens, row))
    return rows


def _layout(
    rows: list[list[int]],
    starts: Int[np.ndarray, " pieces"],
    lengths: Int[np.ndarray, " pieces"],
    length: int,
    pad: int,
    stats: Counter,
) -> Plan:
    """Lay the rows' pieces end to end, padding each row to `length`."""
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


def _plan_with_images(
    docs: Docs,
    config: tcfg.PackingConfig,
    length: int,
    room: int | None,
    rng: np.random.Generator,
    stats: Counter,
) -> Plan:
    """Plan rows of whole documents with images, each row within `length` tokens and
    `room` patches; a document that fits in no row is dropped."""
    # The compiler checks both (`_check_images`).
    assert docs.patches is not None and room is not None
    assert not isinstance(config, tcfg.ConcatPacking)
    sizes = np.diff(docs.offsets)
    fit = (sizes <= length) & (docs.patches <= room)
    stats.update(dropped=int((~fit).sum()), cut=int(sizes[~fit].sum()))
    starts, lengths, patches = docs.offsets[:-1][fit], sizes[fit], docs.patches[fit]
    sort = isinstance(config, tcfg.BestFitPacking) and config.sort
    rows = _fit_both(lengths, patches, length, room, config.max_docs, sort)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    return _layout(rows, starts, lengths, length, int(docs.offsets[-1]), stats)


def plan(
    docs: Docs,
    config: tcfg.PackingConfig,
    length: int,
    seed: int,
    patches: int | None = None,
) -> Plan:
    """Plan the rows of the documents in `docs`, in a random order.

    ``concat`` cuts the documents, end to end, into windows of `length` that
    overlap by a token, so that every token is a target once; what does not fill a
    window is cut. ``greedy`` and ``best_fit`` first cut documents longer than a row
    as their `long` says (drop, truncate, or split into pieces that overlap by a
    token), then pack the pieces into rows padded to `length`: in key order, which is
    random, a new row whenever the next piece does not fit (``greedy``), or each into
    the open row it leaves least room in, longest first (``best_fit``). Either closes
    a row once it holds `max_docs` pieces.

    Args:
        docs: A chunk's documents.
        config: The run's packing.
        length: The tokens of a row: the run's ``seq_len``.
        seed: Seeds the order of the rows; the pipeline passes the chunk's index.
        patches: The patches of images a row has room for, for documents with
            images.

    Returns:
        The plan, with its statistics so far.
    """
    rng = np.random.default_rng(seed)
    tokens = int(docs.offsets[-1])
    stats = Counter(docs=len(docs.offsets) - 1, tokens=tokens)
    if isinstance(config, tcfg.ConcatPacking):
        # Windows [i (length - 1), i (length - 1) + length): what does not fill one is cut.
        n = max(0, (tokens - 1) // (length - 1))
        stats.update(rows=n, cut=tokens - (n * (length - 1) + 1 if n else 0))
        starts = rng.permutation(n) * (length - 1)
        return Plan(starts, np.full(n, length), np.arange(n + 1), stats)
    if docs.patches is not None:  # whole documents, in rows of room for both
        return _plan_with_images(docs, config, length, patches, rng, stats)
    starts, lengths, cuts = _pieces(docs, config.long, length)
    if isinstance(config, tcfg.BestFitPacking):
        rows = _best_fit(lengths, length, config.max_docs, config.sort)
    else:
        rows = _next_fit(lengths, length, config.max_docs)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    stats.update(cuts)
    return _layout(rows, starts, lengths, length, tokens, stats)


def materialize(
    docs: Docs, plan: Plan, length: int, doc_mask: bool, size: int
) -> Iterator[pa.RecordBatch]:
    """Yield the planned rows, `size` at a time, counting targets in `plan.stats`.

    With `doc_mask`, positions count from each document's start, and a piece that
    continues a document starts at 0, as rows do; without it, they count from each
    row's start. A target is a token in `mask` that is not at position 0.

    Args:
        docs: The documents that `plan` was made of.
        plan: Their rows.
        length: The tokens of a row.
        doc_mask: Whether positions restart at every document (the packing's
            ``doc_mask``).
        size: Rows a batch: the dataset's row group.

    Yields:
        Record batches of rows in `SCHEMA`.
    """
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
        if docs.image is not None:
            columns.append(docs.image[index])
        yield pa.RecordBatch.from_arrays(
            [pa.ListArray.from_arrays(bounds, pa.array(c)) for c in columns],
            schema=SCHEMA if docs.image is None else IMAGE_SCHEMA,
        )
