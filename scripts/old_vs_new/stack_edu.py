"""Download Common Pile's Stack v2 Edu, openly licensed code, as Parquet files for bux.

The tutorial "Old against new at the same serving cost" (docs/tutorials/old-vs-new.md)
trains on it. It writes, under OUT:

- ``files/{train,eval}/LANG-NNNN.parquet``: a source file a row, in ``text``;
- ``repos/{train,eval}/python-NN.parquet``: a Python repository a row, its files in
  path order, each after a line ``--- path ---``; only repositories of at least
  ``--min-chars`` characters, the long documents of the long-context phase and eval.

A repository is held out, all of its files and its document, when the hash of its name
falls in the last ``--held-out`` of [0, 1), so that no eval file or repository is
trained on. The shards hold files in no order, so a repository is put together from
all twelve Python shards.

    python scripts/old_vs_new/stack_edu.py data/stack-edu
"""

import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/common-pile/stackv2_edu_filtered/resolve/main"
#: Shard numbers by language: the files of the training mix, a shard or a few a language.
FILES = {
    "markdown": [0], "c": [18], "cpp": [20], "csharp": [28], "go": [34], "java": [38],
    "javascript": [50, 51], "php": [65], "python": [73, 74, 75, 76], "ruby": [85],
    "rust": [86], "shell": [87], "swift": [89], "typescript": [91],
}  # fmt: skip
#: Every Python shard, which the repositories are put together from.
PYTHON = list(range(73, 85))
BUCKETS = 16


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--held-out", type=float, default=0.01)
    p.add_argument("--min-chars", type=int, default=64_000)
    p.add_argument("--max-chars", type=int, default=400_000)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    shards = sorted({s for ss in FILES.values() for s in ss} | set(PYTHON))
    t = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for line in pool.map(
            _shard, shards, [a.out] * len(shards), [a.held_out] * len(shards)
        ):
            print(line, f"({time.time() - t:.0f}s)", flush=True)
        jobs = [(b, a.out, a.min_chars, a.max_chars) for b in range(BUCKETS)]
        for line in pool.map(_repos, *zip(*jobs, strict=True)):
            print(line, f"({time.time() - t:.0f}s)", flush=True)
    shutil.rmtree(a.out / "buckets")


def _held_out(repo: str, share: float) -> bool:
    """Whether `repo` is held out: its name's hash in the last `share` of [0, 1)."""
    h = int.from_bytes(hashlib.blake2b(repo.encode(), digest_size=8).digest(), "big")
    return h / 2**64 >= 1 - share


def _shard(n: int, out: Path, share: float) -> str:
    """Download shard `n`, write its files for training if its language takes it, and
    a Python shard's files into the buckets of their repositories."""
    raw = out / "raw" / f"stack-edu-{n:04d}.json.gz"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{URL}/{raw.name}", raw.with_suffix(".part"))
        raw.with_suffix(".part").rename(raw)
    lang = next((k for k, v in FILES.items() if n in v), None)
    files = {"train": [], "eval": []}
    buckets = defaultdict(list)
    with gzip.open(raw, "rt") as f:
        for line in f:
            doc = json.loads(line)
            meta, text = doc["metadata"], doc["text"]
            if not text:
                continue
            split = "eval" if _held_out(meta["repo_name"], share) else "train"
            files[split].append(text)
            if n in PYTHON:
                row = (meta["repo_name"], meta["path"], text, split)
                buckets[_bucket(meta["repo_name"])].append(row)
    if lang is not None:
        for split, texts in files.items():
            _write(texts, out / "files" / split / f"{lang}-{n:04d}.parquet")
    for b, rows in buckets.items():
        repo, path, text, split = map(list, zip(*rows, strict=True))
        table = pa.table({"repo": repo, "path": path, "text": text, "split": split})
        dest = out / "buckets" / f"{b:02d}-{n:04d}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
    count = sum(map(len, files.values()))
    return f"shard {n} ({lang or 'python repos only'}): {count} files"


def _bucket(repo: str) -> int:
    """The bucket of `repo`'s files, the same in every process."""
    return hashlib.blake2b(repo.encode(), digest_size=2).digest()[0] % BUCKETS


def _repos(b: int, out: Path, min_chars: int, max_chars: int) -> str:
    """Put bucket `b`'s repositories together, a document each, its files in path
    order, and write those of at least `min_chars` characters, cut at a file's end
    past `max_chars`."""
    repos = defaultdict(list)
    for f in sorted((out / "buckets").glob(f"{b:02d}-*.parquet")):
        t = pq.read_table(f).to_pydict()
        for repo, path, text, split in zip(
            t["repo"], t["path"], t["text"], t["split"], strict=True
        ):
            repos[repo, split].append((path, text))
    docs = {"train": [], "eval": []}
    for (_, split), files in repos.items():
        parts, size = [], 0
        for path, text in sorted(files):
            if size > max_chars:
                break
            parts.append(f"--- {path} ---\n{text}\n")
            size += len(parts[-1])
        if size >= min_chars:
            docs[split].append("".join(parts))
    for split, texts in docs.items():
        _write(texts, out / "repos" / split / f"python-{b:02d}.parquet")
    chars = sum(len(d) for d in docs["train"] + docs["eval"])
    return f"bucket {b}: {len(repos)} repos, {len(docs['train'])} + {len(docs['eval'])} long ({chars / 1e6:.0f}M chars)"


def _write(texts: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts}), dest, row_group_size=10_000)


if __name__ == "__main__":
    main()
