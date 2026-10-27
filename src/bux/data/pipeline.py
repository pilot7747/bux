"""The pipeline: a map and a reduce over Arrow record batches, which any MapReduce
engine can run, and an engine that runs them on this machine's cores::

    records ─map─▶ documents under random keys ─shuffle─▶ chunks ─reduce─▶ rows

The map (`Job.map`) tokenizes records into documents (`bux.data.documents`). The
shuffle groups them by the top bits of their keys into a power of two of chunks
(`Job.chunks`), each about ``packing.chunk`` tokens and a random sample. The reduce
(`Job.reduce`) packs a chunk into rows (`bux.data.packing`). `bux.data.spark` runs
the same two functions on Spark.

`prepare` runs them here, in one process per core. The map spills each task's
documents, sorted by key, as raw arrays, about 5 bytes per token of scratch space
next to the cache; each reduce finds its keys in every spill by bisection and maps
in only its own documents, so that the shuffle reads every byte once.

A prepared dataset is a folder under a cache, named for a hash of the job::

    data/packed/text-0c8c782640/
        part-00000.parquet ...   rows: input_ids, position_ids, loss_mask, seq_len each
        dataset.json             the job, its inputs, the parts and statistics; last

The input files are not in the hash, so that a cache outlives them: delete a folder
made from other files under the same glob by hand.
"""

import contextlib
import glob
import hashlib
import itertools
import json
import math
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bux.data import documents, packing
from bux.data.tokenizer import load
from bux.train import config as tcfg
from bux.train import curriculum

#: The version of what a job makes from its documents, in every dataset's hash.
#:
#: A new one prepares every dataset again.
VERSION = 1
#: Where prepared datasets go by default (``--cache``), relative to the working
#: directory.
DEFAULT_CACHE = Path("data/packed")


@dataclass(frozen=True)
class Job:
    """What makes a dataset: documents, a tokenizer, a packing and a row length.

    A run reads one dataset a ``text`` or ``chat`` source (`jobs` lists them), and a
    job's `name` hashes all of it, so two runs that read the same documents the
    same way share one dataset, and a changed setting makes another. `map` and
    `reduce` are the two halves of the MapReduce, for any engine to run; `prepare`
    runs them here.

    Attributes:
        source: The documents, a ``text(...)`` or ``chat(...)`` source.
        tokenizer: The run's tokenizer.
        packing: The run's packing.
        seq_len: The tokens of a row.

    Example::

        job = jobs(run.train)[0]
        job.name  # 'text-0c8c782640'
        folder = prepare(job, "data/packed")
    """

    source: tcfg.TextConfig | tcfg.ChatConfig
    tokenizer: tcfg.TokenizerConfig
    packing: tcfg.PackingConfig
    seq_len: int

    @classmethod
    def of(
        cls, source: tcfg.TextConfig | tcfg.ChatConfig, config: tcfg.TrainConfig
    ) -> "Job":
        """Return the job that makes `source`'s dataset for a run of `config`."""
        assert config.tokenizer is not None, "the compiler requires one for documents"
        return cls(source, config.tokenizer, config.packing, config.seq_len)

    def describe(self) -> dict:
        """Return the job as JSON data, which `name` hashes and dataset.json keeps."""
        return {
            "version": VERSION,
            "source": self.source.model_dump(mode="json"),
            "tokenizer": self.tokenizer.model_dump(mode="json"),
            "packing": self.packing.model_dump(mode="json"),
            "seq_len": self.seq_len,
        }

    @property
    def name(self) -> str:
        """The name of the dataset's folder, the source's type and a hash of
        `describe`, such as ``text-0c8c782640``."""
        blob = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return f"{self.source.type}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"

    @property
    def group_rows(self) -> int:
        """The rows a row group of the output holds, about a million tokens."""
        return max(1, 2**20 // self.seq_len)

    def map(self, records: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Turn records into documents, one batch at a time, as Spark's mapInArrow.

        Args:
            records: Record batches with the source's column.

        Returns:
            The documents of each batch, in `bux.data.documents.SCHEMA`.
        """
        tok = load(self.tokenizer)
        if isinstance(self.source, tcfg.TextConfig):
            return (documents.text(b, self.source, tok) for b in records)
        return (documents.chat(b, self.source, tok) for b in records)

    def chunks(self, tokens: int) -> int:
        """Return the number of chunks for `tokens` tokens: a power of two, so that a
        document's chunk is exactly the top bits of its key, in any engine."""
        return 1 << max(0, round(math.log2(max(tokens, 1) / self.packing.chunk)))

    def reduce(
        self, chunk: int, docs: pa.Table
    ) -> tuple[Iterator[pa.RecordBatch], Counter]:
        """Pack a chunk's documents, in any order, into rows, one row group at a time.

        Args:
            chunk: The chunk's index, which seeds the order of its rows.
            docs: Its documents, in `bux.data.documents.SCHEMA`.

        Returns:
            The rows, in record batches of `group_rows` in
            `bux.data.packing.SCHEMA`, and the statistics, complete once every
            batch is read.
        """
        d = packing.Docs.of(docs, self.seq_len)
        plan = packing.plan(d, self.packing, self.seq_len, seed=chunk)
        rows = packing.materialize(
            d, plan, self.seq_len, self.packing.doc_mask, self.group_rows
        )
        return rows, plan.stats


def shift(chunks: int) -> int:
    """Return how far to shift a key right to get its chunk, out of `chunks`."""
    return 63 - (chunks.bit_length() - 1)


def jobs(config: tcfg.TrainConfig) -> list[Job]:
    """Return the datasets a run reads, in training and evaluation, once each.

    Args:
        config: The run's training settings, ``run.train``.

    Returns:
        A job for each ``text`` and ``chat`` source of ``data``, of each phase of the
        ``curriculum`` at its seq_len, and of every ``eval`` at its own, mixes
        included, in the order written; none for ``tokens`` or ``synthetic``.
    """
    read = [(s, c) for _, c in curriculum.phases(config) for s in tcfg.leaves(c.data)]
    read += [
        (s, tcfg.evaluated(config, e))
        for e in tcfg.evals(config)
        for s in tcfg.leaves(e.data)
    ]
    found = (Job.of(s, c) for s, c in read if isinstance(s, tcfg.DOCUMENTS))
    return list(dict.fromkeys(found))


# --- On this machine ---------------------------------------------------------------------


def _splits(path: str, size: int = 1 << 28) -> list[list[int]]:
    """Group a Parquet file's row groups into runs of about `size` bytes: map tasks."""
    meta = pq.read_metadata(path)
    runs, total = [[]], 0
    for g in range(meta.num_row_groups):
        bytes_ = meta.row_group(g).total_byte_size
        if runs[-1] and total + bytes_ > size:
            runs.append([])
            total = 0
        runs[-1].append(g)
        total += bytes_
    return runs


# A spill holds one map task's documents sorted by key, as raw arrays, so that a
# reduce finds its keys by bisection and memory-maps only its own documents.
_SPILLED = ("key", "offsets", "ids", "mask")


def _map(job: Job, task: tuple[str, list[int]], spill: Path) -> int:
    """Write the documents of some row groups to a spill; return their token count."""
    path, groups = task
    records = pq.ParquetFile(path).iter_batches(
        batch_size=4096, row_groups=groups, columns=[job.source.column]
    )
    docs = pa.Table.from_batches(job.map(records), documents.SCHEMA).sort_by("key")
    ids, offsets = packing.flat(docs["input_ids"])
    mask, _ = packing.flat(docs["loss_mask"])
    spill.mkdir()
    for name, a in zip(_SPILLED, (docs["key"].to_numpy(), offsets, ids, mask)):
        np.save(spill / f"{name}.npy", a)
    return len(ids)


def _take(spill: Path, chunk: int, chunks: int) -> pa.RecordBatch:
    """Read a chunk's documents from a spill."""
    key, offsets, ids, mask = (
        np.load(spill / f"{name}.npy", mmap_mode="r") for name in _SPILLED
    )
    a = np.searchsorted(key, chunk << shift(chunks))
    b = (
        np.searchsorted(key, (chunk + 1) << shift(chunks))
        if chunk + 1 < chunks
        else len(key)
    )
    lo, hi = offsets[a], offsets[b]
    return documents.record_batch(
        np.array(key[a:b]),
        np.array(ids[lo:hi]),
        offsets[a : b + 1] - lo,
        np.array(mask[lo:hi]),
    )


def _reduce(
    job: Job, chunk: int, chunks: int, spills: list[Path], part: Path
) -> Counter:
    """Pack a chunk's documents, from every spill, into a Parquet file of rows."""
    pieces = [_take(spill, chunk, chunks) for spill in spills]
    rows, stats = job.reduce(chunk, pa.Table.from_batches(pieces, documents.SCHEMA))
    with pq.ParquetWriter(part, packing.SCHEMA, compression="zstd") as out:
        for batch in rows:
            out.write_batch(batch)
    return stats


def summary(name: str, stats: Counter, seq_len: int, seconds: float) -> str:
    """Return the line that `prepare` prints for a dataset: its documents, tokens and
    rows, how full the rows are, the share of targets and the tokens cut."""
    rows = stats["rows"]
    cells = max(rows * seq_len, 1)
    return (
        f"{name}: {stats['docs']:,} documents, {stats['tokens']:,} tokens -> "
        f"{rows:,} rows of {seq_len}, {1 - stats['padding'] / cells:.1%} filled, "
        f"{stats['targets'] / cells:.1%} targets, {stats['cut']:,} tokens cut "
        f"({seconds:.1f}s)"
    )


def prepare(
    job: Job,
    cache: str | Path = DEFAULT_CACHE,
    *,
    workers: int | None = None,
    echo: Callable[[str], None] = print,
) -> Path:
    """Return the folder of `job`'s dataset under `cache`, preparing it if missing.

    This is ``bux prepare``, and what ``bux train`` and ``bux launch`` do first. A
    folder with a dataset.json is done, and returned at once. Otherwise the map runs
    a task about every 256 MiB of row groups of the files that the source's glob
    matches, and the reduce a task a chunk, in a scratch folder under `cache`, which
    becomes the dataset once dataset.json is written. Processes that prepare the
    same job at once, as the tasks of an array job may, are safe: one wins, and the
    others use its folder.

    Args:
        job: What to prepare: one of `jobs`.
        cache: The folder of prepared datasets.
        workers: Processes to run the tasks in: None for one per core, 0 for this
            one alone.
        echo: Where the summary line goes once the dataset is made.

    Returns:
        The dataset's folder, ``cache / job.name``.

    Raises:
        FileNotFoundError: If no file matches the source's path.

    Example::

        prepare(job)
        # text-0c8c782640: 14,720,729 documents, 10,248,316,202 tokens -> 10,022,812
        # rows of 1024, 99.9% filled, 99.7% targets, 0 tokens cut (175.2s)
    """
    cache = Path(cache)
    folder = cache / job.name
    if (folder / "dataset.json").exists():
        return folder
    files = sorted(glob.glob(job.source.path))
    if not files:
        raise FileNotFoundError(f"no files match {job.source.path!r}")
    cache.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    with contextlib.ExitStack() as stack:
        run = map
        if workers != 0:
            run = stack.enter_context(ProcessPoolExecutor(workers)).map
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=cache)))
        tasks = [(f, groups) for f in files for groups in _splits(f)]
        spills = [tmp / f"map-{i:05d}" for i in range(len(tasks))]
        tokens = sum(run(_map, itertools.repeat(job), tasks, spills))
        chunks = job.chunks(tokens)
        out = tmp / job.name
        out.mkdir()
        parts = [f"part-{g:05d}.parquet" for g in range(chunks)]
        stats = sum(
            run(
                _reduce,
                itertools.repeat(job),
                range(chunks),
                itertools.repeat(chunks),
                itertools.repeat(spills),
                [out / p for p in parts],
            ),
            Counter(),
        )
        meta = {"job": job.describe(), "files": files, "parts": parts} | {
            "vocab_size": load(job.tokenizer).vocab_size,
            "stats": stats,
        }
        (out / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            out.rename(folder)
        except OSError:  # another process, such as a task of the same array job, won
            if not (folder / "dataset.json").exists():
                raise
            return folder
    echo(summary(job.name, stats, job.seq_len, time.perf_counter() - clock))
    return folder
