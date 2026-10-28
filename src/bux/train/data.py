"""Batches of rows, from prepared datasets, token shards, random tokens, or mixtures.

A row is ``seq_len`` tokens: ``input_ids``, ``position_ids`` (where each token is in
its document, so one starts wherever they stop increasing) and ``loss_mask`` (whether
each token is a target). A run with ``patches_per_row`` also reads each row's images
(`Batch.images`), decoded and cut into patches as batches load. Every source is an endless stream of rows, each epoch in a
new order. Batch i holds rows ``i * n`` to ``(i + 1) * n`` of each source, n of them
in proportion to its weight, so it depends only on the seed and i: a run that resumes
at step i reads what it would have read had it never stopped. Where the run's
curriculum or batch ramp change its batches, each stretch of steps reads on from
where the stretches before it left each source (`training_batches`). In a job of
many processes, each reads only the rows of a batch that its devices hold.

A source is found by its config's type in `sources`. The trainer reads batches
through `prefetch`, which loads the next ones as a step runs::

    for batch in prefetch(training_batches(run.train, run.arch.vocab_size)):
        ...
"""

import bisect
import concurrent.futures
import contextlib
import functools
import glob
import itertools
import json
import queue
import threading
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import pyarrow.parquet as pq
from jaxtyping import Array, Bool, Int, Int32, Integer, UInt64

from bux import parallel
from bux.data import images as pictures
from bux.data.pipeline import DEFAULT_CACHE, Job
from bux.train import config as tcfg
from bux.train import curriculum
from bux.train.curriculum import Stretch
from bux.train.registry import Registry

#: The threads that decode a batch's images, a row each.
_DECODERS = concurrent.futures.ThreadPoolExecutor(8)


class Batch(NamedTuple):
    """Rows of tokens, numpy on the host and jax on a device, each ``b t``: rows,
    positions. Position t predicts ``input_ids[t + 1]``, a target where
    ``loss_mask[t + 1]`` is and token t + 1 is in t's document
    (`~bux.train.objective.target_mask`)."""

    input_ids: Int32[np.ndarray | Array, "b t"]  #: The tokens.
    #: Where each token is in its document. A document starts at a row's first token
    #: and wherever they stop increasing.
    position_ids: Int32[np.ndarray | Array, "b t"]
    loss_mask: Bool[np.ndarray | Array, "b t"]  #: Whether each token is a target.
    #: The rows' images, which the model's image(...) reads; None for rows of text.
    images: pictures.Rows | None = None

    @classmethod
    def concat(cls, parts: Sequence["Batch"]) -> "Batch":
        """Return the rows of `parts`, one after another, on the host."""
        ids, positions, mask = (
            np.concatenate(c) for c in zip(*(p[:3] for p in parts), strict=True)
        )
        if parts[0].images is None:
            return cls(ids, positions, mask)
        rows = [p.images for p in parts]
        assert all(r is not None for r in rows), "every part has images or none"
        joined = (np.concatenate(c) for c in zip(*rows, strict=True))
        return cls(ids, positions, mask, pictures.Rows(*joined))


def plain(ids: Integer[np.ndarray, "b t"]) -> Batch:
    """Rows that each read as one text, every token a target."""
    positions = np.broadcast_to(np.arange(ids.shape[1], dtype=np.int32), ids.shape)
    return Batch(ids.astype(np.int32), positions, np.ones(ids.shape, bool))


@runtime_checkable
class Source(Protocol):
    """An endless stream of rows, which a builder in `sources` makes from a config.
    A row depends only on its index and the stream's seed, so a source reads any
    rows in any order: those of a resumed run's batches, or a process's own."""

    def rows(self, start: int, n: int) -> Batch:
        """Return rows `start` to ``start + n`` of the source's endless stream."""
        ...


@functools.lru_cache(maxsize=8)
def _order(n: int, seed: tuple[int, ...], epoch: int) -> Int[np.ndarray, " n"]:
    """A permutation of range(n), a new one every epoch."""
    return np.random.default_rng((*seed, epoch)).permutation(n)


ROUNDS = 4  # a Feistel network's: Luby and Rackoff's bound for a random permutation


def permuted(
    i: Integer[np.ndarray, " k"], n: int, seed: tuple[int, ...]
) -> Int[np.ndarray, " k"]:
    """Where a random permutation of range(n), drawn from `seed`, sends each of `i`,
    without making the permutation: a corpus of 10T tokens has 2.4G windows of 4096.

    A Feistel network permutes the numbers of 2h bits, 4^h >= n, and what it sends past
    n goes through it again until it lands below n (cycle walking), as grain's
    index_shuffle does."""
    half = max(1, ((n - 1).bit_length() + 1) // 2)
    keys = np.random.SeedSequence(seed).generate_state(ROUNDS, np.uint64)
    out = _feistel(i.astype(np.uint64), half, keys)
    while (past := out >= n).any():
        out[past] = _feistel(out[past], half, keys)
    return out.astype(np.int64)


def _feistel(
    x: UInt64[np.ndarray, " k"], half: int, keys: UInt64[np.ndarray, " r"]
) -> UInt64[np.ndarray, " k"]:
    """A bijection of the numbers of 2 * `half` bits: a round per key, each of which
    swaps the halves and mixes a hash of one into the other."""
    bits, mask = np.uint64(half), np.uint64((1 << half) - 1)
    left, right = x >> bits, x & mask
    for key in keys:
        left, right = right, left ^ (_hash(right ^ key) & mask)
    return (left << bits) | right


def _hash(z: UInt64[np.ndarray, " k"]) -> UInt64[np.ndarray, " k"]:
    """SplitMix64's finalizer: every bit of `z` moves every bit of the hash."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class Prepared:
    """A dataset bux.data prepared: its row groups, each epoch in a new order; a
    dataset with images, their images too (`bux.data.images.row`)."""

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
        job = meta["job"]
        self.images = job["source"].get("images")  # how they are cut, or None
        self.room = job.get("patches_per_row")
        self.store = pictures.Store(folder) if self.images else None
        self.last: tuple[int, tuple] | None = None  # rows are read in order

    def _group(self, i: int) -> tuple:
        """Return row group i's columns, each a row of seq_len values a row."""
        if self.last is None or self.last[0] != i:
            f, g = self.groups[i]
            table = self.files[f].read_row_group(g)
            names = [*Batch._fields[:3], *(["image"] if self.store else [])]
            columns = (
                table[name].combine_chunks().flatten().to_numpy(zero_copy_only=False)
                for name in names
            )
            self.last = i, tuple(c.reshape(-1, self.seq_len) for c in columns)
        return self.last[1]

    def _batch(self, columns: tuple) -> Batch:
        """Return rows of the columns `_group` reads, their images decoded."""
        if self.store is None:
            return Batch(*columns)
        ids, positions, mask, keys = columns
        cut = functools.partial(
            pictures.row,
            store=self.store,
            size=self.images["size"],
            merge=self.images["merge"],
            patches_per_row=self.room,
        )
        rows = list(_DECODERS.map(cut, keys, positions))
        return Batch(
            ids, positions, mask, pictures.Rows(*(np.stack(c) for c in zip(*rows)))
        )

    def rows(self, start: int, n: int) -> Batch:
        """Return rows `start` to ``start + n``, from the row groups in their order
        of each epoch."""
        parts = []
        while n:
            epoch, at = divmod(start, self.total)
            order = _order(len(self.groups), self.seed, epoch)
            ends = np.cumsum(self.sizes[order])
            k = int(np.searchsorted(ends, at, side="right"))
            first = at - int(ends[k] - self.sizes[order[k]])
            take = min(n, int(ends[k]) - at)
            group = self._group(int(order[k]))
            parts.append(self._batch(tuple(c[first : first + take] for c in group)))
            start, n = start + take, n - take
        return Batch.concat(parts)


class Shards:
    """Windows of memory-mapped shards of tokens, each epoch in a new order. They
    overlap by a token, so that every token is a target once."""

    def __init__(self, config: tcfg.TokensConfig, seq_len: int, seed: tuple[int, ...]):
        paths = sorted(glob.glob(config.path))
        if not paths:
            raise FileNotFoundError(f"no shards match {config.path!r}")
        self.shards = [
            np.memmap(p, dtype=config.dtype, mode="r", offset=config.header)
            for p in paths
        ]
        self.counts = np.array(
            [max(0, (len(s) - 1) // (seq_len - 1)) for s in self.shards]
        )
        self.ends = np.cumsum(self.counts)
        if not self.ends[-1]:
            raise ValueError(
                f"every shard of {config.path!r} is shorter than {seq_len} tokens"
            )
        self.seq_len, self.seed = seq_len, seed

    def rows(self, start: int, n: int) -> Batch:
        """Return rows `start` to ``start + n``: windows of each epoch's permutation
        of them (`permuted`)."""
        total = int(self.ends[-1])
        epoch, at = np.divmod(np.arange(start, start + n), total)
        windows = np.concatenate(
            [
                permuted(at[epoch == e], total, (*self.seed, int(e)))
                for e in np.unique(epoch)
            ]
        )
        shard = np.searchsorted(self.ends, windows, side="right")
        offset = (windows - self.ends[shard] + self.counts[shard]) * (self.seq_len - 1)
        return plain(
            np.stack(
                [self.shards[s][o : o + self.seq_len] for s, o in zip(shard, offset)]
            )
        )


class Synthetic:
    """Uniformly random tokens, each row of the stream drawn from its own seed, so that
    a process draws the rows it holds alone."""

    def __init__(self, vocab_size: int, seq_len: int, seed: tuple[int, ...]):
        self.vocab_size, self.seq_len, self.seed = vocab_size, seq_len, seed

    def rows(self, start: int, n: int) -> Batch:
        """Return rows `start` to ``start + n``, each drawn from its own seed."""
        return plain(np.stack([self._row(r) for r in range(start, start + n)]))

    def _row(self, r: int) -> Int32[np.ndarray, " t"]:
        rng = np.random.default_rng((*self.seed, r))
        return rng.integers(0, self.vocab_size, self.seq_len, np.int32)


#: The data sources, by the ``type`` of their config. A builder gets its config, the
#: run's settings, the model's vocabulary size, where prepared datasets are, and the
#: seed of its stream, and returns a `Source`.
sources = Registry[Source]("data source", "bux.train.data.sources")


@sources.register("text", "chat")
def _(
    c: tcfg.TextConfig | tcfg.ChatConfig,
    t: tcfg.TrainConfig,
    vocab_size: int,
    cache: str | Path,
    seed: tuple[int, ...],
) -> Source:
    """The rows bux.data prepared from the documents under the cache (`Prepared`),
    their row groups in a new order every epoch."""
    return Prepared(Path(cache) / Job.of(c, t).name, vocab_size, t.seq_len, seed)


@sources.register("tokens")
def _(
    c: tcfg.TokensConfig,
    t: tcfg.TrainConfig,
    vocab_size: int,
    cache: str | Path,
    seed: tuple[int, ...],
) -> Source:
    """Windows of memory-mapped shards of token ids (`Shards`), overlapping by a
    token, in a new order every epoch."""
    return Shards(c, t.seq_len, seed)


@sources.register("synthetic")
def _(
    c: tcfg.SyntheticConfig,
    t: tcfg.TrainConfig,
    vocab_size: int,
    cache: str | Path,
    seed: tuple[int, ...],
) -> Source:
    """Uniformly random tokens over the vocabulary (`Synthetic`), a seed each row."""
    return Synthetic(vocab_size, t.seq_len, seed)


def apportion(weights: Sequence[float], n: int) -> list[int]:
    """`n` in proportion to `weights`, to the largest remainders first."""
    quota = np.asarray(weights) / sum(weights) * n
    counts = np.floor(quota).astype(int)
    counts[np.argsort(counts - quota, kind="stable")[: n - counts.sum()]] += 1
    return counts.tolist()


class _End(NamedTuple):
    error: Exception | None  # what ended the items, if not their end


def prefetch[T](items: Iterator[T], ahead: int = 2) -> Generator[T]:
    """Yield `items`, which a thread reads `ahead` early: so the next batches load as a
    step runs. It stops when the generator closes, and what it raises is raised here,
    in order.

    It leaves the devices alone: a batch put there from another thread, as a step's
    temporaries hold most of the memory, lands below them, and the next step's no
    longer fit in what is left above it.

    Args:
        items: What to read, as batches from `batches`.
        ahead: How many items the thread reads ahead of the caller.

    Yields:
        The items, in order.

    Example::

        with contextlib.closing(prefetch(batches(data, config, vocab))) as loaded:
            for step in range(steps):
                trainer.train_on_batch(next(loaded))
    """
    ready: queue.Queue[T | _End] = queue.Queue(ahead)
    closed = threading.Event()

    def fill() -> None:
        try:
            for item in items:
                ready.put(item)
                if closed.is_set():
                    return
            ready.put(_End(None))
        except Exception as err:  # noqa: BLE001
            ready.put(_End(err))

    thread = threading.Thread(target=fill, daemon=True)
    thread.start()
    try:
        while not isinstance(item := ready.get(), _End):
            yield item
        if item.error is not None:
            raise item.error
    finally:
        closed.set()
        while thread.is_alive():  # free its place in the queue, so that it sees
            with contextlib.suppress(queue.Empty):
                ready.get(timeout=0.1)


def batches(
    data: tcfg.DataConfig,
    config: tcfg.TrainConfig,
    vocab_size: int,
    *,
    cache: str | Path = DEFAULT_CACHE,
    start: int = 0,
    held_out: bool = False,
    rows: Sequence[int] | Int[np.ndarray, " n"] | None = None,
) -> Iterator[Batch]:
    """Yield batches of ``config.batch`` rows of ``config.seq_len`` tokens from `data`,
    from batch `start` on. Held-out batches are drawn independently of training's.

    Of each batch only `rows` are read, by default those that this process's devices
    hold (`bux.parallel.local_rows`), every row on one process; the others stay 0.
    The run's ``curriculum`` and ``batch_ramp`` are `training_batches`' to follow.

    Args:
        data: The source or mixture to read: the run's ``data``, or its eval's.
        config: The run's settings: its ``batch``, ``seq_len``, ``seed``,
            ``tokenizer`` and ``packing``.
        vocab_size: The model's vocabulary, which every token id must be under.
        cache: Where bux.data prepared `text` and `chat` datasets
            (`bux.data.pipeline.prepare`).
        start: The first batch: the step a run resumes at.
        held_out: Draw the batches from a stream of their own seed, as eval does.
        rows: The rows of each batch to read; None: this process's.

    Yields:
        Batch after batch, endlessly, on the host.

    Raises:
        ValueError: If a mixture leaves a source no row of a batch, or a token id is
            outside the vocabulary.
        FileNotFoundError: If a dataset is not prepared, or no shards match.

    Example::

        it = batches(run.train.data, run.train, run.arch.vocab_size, start=1000)
        batch = next(it)  # the batch of step 1000
    """
    whole = Stretch(0, config.steps, config.model_copy(update={"data": data}))
    plans = _plans([whole])
    if rows is None:
        rows = parallel.local_rows(config.batch, config.seq_len)
    own = [np.asarray(rows, np.int64)]
    return _batches(plans, own, vocab_size, cache, start, held_out)


def training_batches(
    config: tcfg.TrainConfig,
    vocab_size: int,
    *,
    cache: str | Path = DEFAULT_CACHE,
    start: int = 0,
) -> Iterator[Batch]:
    """Yield a run's training batches from step `start` on, each of the seq_len, batch
    and data of its stretch of the run (`bux.train.curriculum.stretches`): those of
    `batches` of ``config.data``, where the run has no curriculum and no batch ramp.

    A source read at the same seq_len in two stretches goes on in the second from
    the row it stopped at in the first, so that a phase that only changes the batch
    or adds a source repeats no row. Each process reads its own rows of each batch.

    Args:
        config: The run's settings, ``run.train``.
        vocab_size: The model's vocabulary, which every token id must be under.
        cache: Where bux.data prepared `text` and `chat` datasets.
        start: The step a run resumes at.

    Yields:
        Batch after batch, on the host: past the run's last step, the last
        stretch's, endlessly.

    Raises:
        ValueError: As `batches`, for a batch of any stretch.
        FileNotFoundError: If a dataset of any stretch is not prepared.
    """
    plans = _plans(curriculum.stretches(config))
    own = [parallel.local_rows(p.config.batch, p.config.seq_len) for p in plans]
    return _batches(plans, own, vocab_size, cache, start, held_out=False)


class _Plan(NamedTuple):
    """How the batches of a stretch, from step `start` under `config`, read their
    sources: by the number of each part's stream, its rows a batch, and the rows it
    gave the stretches before."""

    start: int  # the stretch's first step
    config: tcfg.TrainConfig  # its settings
    streams: list[int]  # the number of each part's stream, which seeds it
    counts: list[int]  # the rows a batch takes from each
    before: list[int]  # the rows each gave the stretches before this one


def _parts(data: tcfg.DataConfig) -> tuple[tcfg.Weighted, ...]:
    if isinstance(data, tcfg.MixtureConfig):
        return data.sources
    return (tcfg.Weighted(weight=1, source=data),)


def _plans(stretches: Sequence[Stretch]) -> list[_Plan]:
    """Return how each stretch reads its sources. A stream is a source read at a
    seq_len, one for each time a mixture names it, numbered in the order the
    stretches first read them: a run of one stretch numbers them as its mixture
    does."""
    numbers: dict[tuple[str, int, int], int] = {}
    given: dict[int, int] = {}
    plans = []
    for s in stretches:
        parts, c = _parts(s.config.data), s.config
        counts = apportion([p.weight for p in parts], c.batch)
        for p, n in zip(parts, counts, strict=True):
            if not n:
                share = p.weight / sum(q.weight for q in parts)
                where = f" (from step {s.start})" if s.start else ""
                raise ValueError(
                    f"a batch of {c.batch} rows{where} has no room for a source of "
                    f"weight {share:.3g}"
                )
        keys = [p.source.model_dump_json() for p in parts]
        ids = [
            numbers.setdefault((key, c.seq_len, keys[:k].count(key)), len(numbers))
            for k, key in enumerate(keys)
        ]
        before = [given.get(n, 0) for n in ids]
        plans.append(_Plan(s.start, c, ids, counts, before))
        for n, count in zip(ids, counts, strict=True):
            given[n] = given.get(n, 0) + count * (s.stop - s.start)
    return plans


def _batches(
    plans: Sequence[_Plan],
    own: Sequence[Int[np.ndarray, " _"]],  # each of its own length
    vocab_size: int,
    cache: str | Path,
    start: int,
    held_out: bool,
) -> Iterator[Batch]:
    """Yield batch after batch from `start` on, by `plans`, of which this process reads
    `own` rows of each plan's: found by the caller, under the mesh in force, which the
    thread that reads the batches (`prefetch`) is not under."""
    seed = (plans[0].config.seed, int(held_out))
    streams: dict[int, Source] = {}
    for plan in plans:
        for n, p in zip(plan.streams, _parts(plan.config.data), strict=True):
            if n not in streams:
                streams[n] = sources.build(
                    p.source, plan.config, vocab_size, cache, (*seed, n)
                )
    starts = [plan.start for plan in plans]
    for i in itertools.count(start):
        k = bisect.bisect_right(starts, i) - 1
        plan = plans[k]
        at = i - plan.start
        firsts = [b + at * n for b, n in zip(plan.before, plan.counts, strict=True)]
        parts = [streams[n] for n in plan.streams]
        room = image_room(plan.config)
        batch = _read(parts, plan.counts, firsts, own[k], plan.config.seq_len, room)
        if (top := int(batch.input_ids.max())) >= vocab_size:
            raise ValueError(
                f"token id {top} is outside the vocabulary of {vocab_size}"
            )
        yield batch


def _read(
    streams: Sequence[Source],
    counts: Sequence[int],
    firsts: Sequence[int],
    rows: Int[np.ndarray, " n"],
    seq_len: int,
    room: tuple[int, int] | None = None,
) -> Batch:
    """A batch of `counts[k]` rows of each stream in turn, from its row `firsts[k]`, of
    which only `rows` are read; the others stay 0. With `room` for n patches of p
    bytes a row, its images too."""
    shape = (sum(counts), seq_len)
    batch = Batch(
        np.zeros(shape, np.int32), np.zeros(shape, np.int32), np.zeros(shape, bool)
    )
    if room is not None:
        batch = batch._replace(images=_no_images(shape, room))
    seen = np.zeros(shape[0], bool)  # the rows whose images a stream gave
    first = 0  # the stream's first row in the batch
    for stream, n, start in zip(streams, counts, firsts, strict=True):
        own = rows[(first <= rows) & (rows < first + n)] - first
        for at, m in _runs(own):
            got = stream.rows(start + at, m)
            for whole, part in zip(batch[:3], got[:3], strict=True):
                whole[first + at : first + at + m] = part
            if got.images is not None and batch.images is not None:
                for whole, part in zip(batch.images, got.images, strict=True):
                    whole[first + at : first + at + m] = part
                seen[first + at : first + at + m] = True
        first += n
    if batch.images is not None:  # a row of text is at its position on every axis
        rope = np.repeat(batch.position_ids[~seen][..., None], 3, -1)
        batch.images.rope[~seen] = rope
    return batch


def _no_images(shape: tuple[int, int], room: tuple[int, int]) -> pictures.Rows:
    """Return the images of `shape`'s rows of text, with `room` for n patches of p
    bytes: no patch, each its own image, and no image tokens."""
    b, t = shape
    n, p = room
    return pictures.Rows(
        np.zeros((b, n, p), np.uint8),
        np.zeros((b, n), np.int32),
        np.zeros((b, n, 2), np.int32),
        np.ones((b, n, 2), np.int32),
        np.zeros((b, t, 3), np.int32),
    )


def image_room(config: tcfg.TrainConfig) -> tuple[int, int] | None:
    """Return the patches of a row of a run with images, and the bytes of a patch;
    None for a run of text."""
    if config.patches_per_row is None:
        return None
    sources = [*tcfg.leaves(config.data)]
    sources += [s for e in tcfg.evals(config) for s in tcfg.leaves(e.data)]
    sources += [s for p in config.curriculum if p.data for s in tcfg.leaves(p.data)]
    # The compiler checks that some source has images, each cut as the model's are.
    [size, *_] = [
        s.images.size
        for s in sources
        if isinstance(s, tcfg.ChatConfig) and s.images is not None
    ]
    return config.patches_per_row, 3 * size**2


def _runs(rows: Int[np.ndarray, " n"]) -> list[tuple[int, int]]:
    """The runs of consecutive numbers in sorted `rows`: (first, length) of each."""
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    return [(int(run[0]), len(run)) for run in np.split(rows, breaks) if len(run)]
