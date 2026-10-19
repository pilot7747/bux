"""The pipeline: a map and a reduce over Arrow record batches, for any MapReduce
engine to run, and an engine that runs them on this machine's cores.

    records ─map─▶ documents under random keys ─shuffle─▶ chunks ─reduce─▶ rows

The map tokenizes records into documents (bux.data.documents). The shuffle groups
them by the top bits of their keys, into a power of two of chunks of about
`packing.chunk` tokens, each a random sample. The reduce packs a chunk into rows
(bux.data.packing). bux.data.spark runs the same two functions on Spark.

A prepared dataset is a folder named for a hash of the job, under a cache:

    data/packed/text-3f9a0c2e1b/
        part-00000.parquet ...   rows: input_ids, position_ids, loss_mask, seq_len each
        dataset.json             the job, its inputs, the parts and statistics; last
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

VERSION = 1  # of what a job makes from its documents; a new one prepares them again
DEFAULT_CACHE = Path("data/packed")


@dataclass(frozen=True)
class Job:
    """What makes a dataset: documents, a tokenizer, a packing and a row length."""

    source: tcfg.TextConfig | tcfg.ChatConfig
    tokenizer: tcfg.TokenizerConfig
    packing: tcfg.PackingConfig
    seq_len: int

    @classmethod
    def of(cls, source: tcfg.TextConfig | tcfg.ChatConfig, t: tcfg.TrainConfig):
        assert t.tokenizer is not None, "the compiler requires one for documents"
        return cls(source, t.tokenizer, t.packing, t.seq_len)

    def describe(self) -> dict:
        return {
            "version": VERSION,
            "source": self.source.model_dump(mode="json"),
            "tokenizer": self.tokenizer.model_dump(mode="json"),
            "packing": self.packing.model_dump(mode="json"),
            "seq_len": self.seq_len,
        }

    @property
    def name(self) -> str:
        blob = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return f"{self.source.type}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"

    @property
    def group_rows(self) -> int:
        """Rows in a row group of the output: about a million tokens."""
        return max(1, 2**20 // self.seq_len)

    def map(self, records: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Records to documents, a batch at a time: Spark's mapInArrow."""
        tok = load(self.tokenizer)
        if isinstance(self.source, tcfg.TextConfig):
            return (documents.text(b, self.source, tok) for b in records)
        return (documents.chat(b, self.source, tok) for b in records)

    def chunks(self, tokens: int) -> int:
        """The chunks for `tokens` tokens of documents: a power of two, so that a
        document's chunk is the top bits of its key, exactly, in any engine."""
        return 1 << max(0, round(math.log2(max(tokens, 1) / self.packing.chunk)))

    def reduce(
        self, chunk: int, docs: pa.Table
    ) -> tuple[Iterator[pa.RecordBatch], Counter]:
        """A chunk's documents, in any order, to its rows, a row group at a time, and
        statistics, which are complete once the rows are."""
        d = packing.Docs.of(docs, self.seq_len)
        plan = packing.plan(d, self.packing, self.seq_len, seed=chunk)
        rows = packing.materialize(
            d, plan, self.seq_len, self.packing.doc_mask, self.group_rows
        )
        return rows, plan.stats


def shift(chunks: int) -> int:
    """How far to shift a key right for its chunk, of `chunks`."""
    return 63 - (chunks.bit_length() - 1)


def jobs(t: tcfg.TrainConfig) -> list[Job]:
    """The datasets a run reads, in training and in evaluation, each once."""
    sources = tcfg.leaves(t.data) + (tcfg.leaves(t.eval.data) if t.eval else ())
    return list(
        dict.fromkeys(Job.of(s, t) for s in sources if isinstance(s, tcfg.DOCUMENTS))
    )


# --- On this machine ---------------------------------------------------------------------


def _splits(path: str, size: int = 1 << 28) -> list[list[int]]:
    """A Parquet file's row groups, in runs of about `size` bytes: the map's tasks."""
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


def _map(job: Job, task: tuple[str, list[int]], spill: Path) -> int:
    """Documents of some row groups, each batch sorted by key, to an Arrow file; the
    tokens in them."""
    path, groups = task
    records = pq.ParquetFile(path).iter_batches(
        batch_size=4096, row_groups=groups, columns=[job.source.column]
    )
    tokens = 0
    with pa.ipc.new_file(str(spill), documents.SCHEMA) as out:
        for docs in job.map(records):
            out.write_batch(docs.sort_by("key"))
            tokens += len(docs["input_ids"].values)
    return tokens


def _reduce(
    job: Job, chunk: int, chunks: int, spills: list[Path], part: Path
) -> Counter:
    """The documents of a chunk, from every spill, to a Parquet file of rows."""
    pieces = []
    for spill in spills:
        reader = pa.ipc.open_file(pa.memory_map(str(spill)))
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            keys = batch["key"].to_numpy() >> shift(chunks)
            a, b = np.searchsorted(keys, [chunk, chunk + 1])
            pieces.append(batch.slice(a, b - a))
    rows, stats = job.reduce(chunk, pa.Table.from_batches(pieces, documents.SCHEMA))
    with pq.ParquetWriter(part, packing.SCHEMA, compression="zstd") as out:
        for batch in rows:
            out.write_batch(batch)
    return stats


def summary(name: str, stats: Counter, seq_len: int, seconds: float) -> str:
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
    """The folder of `job`'s dataset under `cache`, made unless it is there already,
    by `workers` processes (None: one a core; 0: this one)."""
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
        spills = [tmp / f"map-{i:05d}.arrow" for i in range(len(tasks))]
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
        out.rename(folder)
    echo(summary(job.name, stats, job.seq_len, time.perf_counter() - clock))
    return folder
