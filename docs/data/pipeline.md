# The pipeline

`bux prepare` turns a run's documents into rows once, as a MapReduce job, on this
machine's cores or on Spark. This page covers the job, the cache it writes, running
it on Spark, and how training reads the result.

```sh
bux prepare examples/experiments/sft.bux                      # every core of this machine
bux prepare examples/experiments/sft.bux --workers 8          # 8 processes
bux prepare examples/experiments/sft.bux --cache /shared/packed
```

| Option | Default | What it does |
|--------|---------|--------------|
| `--cache DIR` | `data/packed` | where prepared datasets go, and where training looks for them |
| `--workers N` | one per core | the processes the map and the reduce run in; `0` runs them in this process |

`bux prepare` prepares every dataset that the file's runs read, for training and for
eval, once each, and prints each dataset's folder. A dataset already in the cache is
not prepared again. `bux train` and `bux launch` prepare whatever is missing before
they train, so you only need `bux prepare` to do it ahead of time or on another
machine. A file that reads no documents, only `tokens` or `synthetic`, has nothing to
prepare:

```console
$ bux prepare examples/experiments/smoke.bux
this file reads no documents to prepare
```

## The job

```text
Parquet ─map─▶ documents under random keys ─shuffle─▶ chunks ─reduce─▶ rows ─▶ Parquet
```

1. **Map** ({py:meth}`~bux.data.pipeline.Job.map`) turns record batches into
   documents ({py:mod}`bux.data.documents`). Each document gets a 63-bit key, the
   hash of its content, plus `input_ids` and `loss_mask` as Arrow lists.
2. **Shuffle** groups the documents by the top bits of their keys into a power of two
   of chunks, each about `packing.chunk` tokens. Because the groups are defined by
   bits, every engine forms exactly the same ones, and each is a random sample of
   the documents.
3. **Reduce** ({py:meth}`~bux.data.pipeline.Job.reduce`) sorts a chunk by key, so
   the order in which an engine delivered it does not matter. It then plans the
   chunk's rows by the packing ({py:mod}`bux.data.packing`) and writes them one row
   group at a time.

Map and reduce are functions over Arrow record batches and share nothing else, so
any engine that speaks Arrow can run them.

### On this machine

{py:func}`bux.data.pipeline.prepare` runs the job with one process per core:

- Each map task takes about 256 MiB of a Parquet file's row groups. It spills its
  documents, sorted by key, as raw arrays. That needs about 5 bytes of scratch space
  per token, next to the cache, and is deleted when the job ends.
- Each reduce task finds its key range in every spill by bisection and maps in only
  its own documents, so the shuffle reads every byte once.
- The dataset is written to a temporary folder and moved into place last. Two jobs
  that prepare the same dataset at once, as the tasks of an array job may, both
  succeed: the second uses the first one's folder.

On a 16-core laptop, this turns the speedrun's FineWeb (29 GB of Parquet) into 17 GB
of rows in under three minutes ([Quick start](index.md#quick-start)).

## The cache

A prepared dataset is a folder named for a hash of what made it: the source setting,
the tokenizer, the packing and `seq_len`. So a run finds the dataset it needs, a
changed setting makes a new one, and a cache can be shared between runs and copied
between machines. A curriculum phase that reads documents at another `seq_len` reads
a separate dataset, which `bux prepare` makes along with the rest
([What each phase reads](../training/curriculum.md#what-each-phase-reads)).

```text
data/packed/text-0c8c782640/
    part-00000.parquet ...    rows: a row group of about a million tokens
    dataset.json              the job, its input files, the parts, statistics; written last
```

A part holds one chunk's rows: `input_ids`, `position_ids` and `loss_mask`, each of
`seq_len`, zstd-compressed, in row groups of about 2^20 tokens (256 rows of 4096).
Here is `text-17a28baeab/dataset.json`, the held-out split of the speedrun's
FineWeb:

```text
{
  "job": {
    "version": 1,
    "source": {"type": "text", "path": "data/fineweb/sample/10BT/*.parquet",
               "column": "text", "bos": true, "eos": false, "split": [0.99, 1.0]},
    "tokenizer": {"type": "tokenizer", "name": "gpt2", "bos": "<|endoftext|>",
                  "eos": "<|endoftext|>", "added": []},
    "packing": {"type": "best_fit", "doc_mask": true, "long": "split",
                "max_docs": null, "sort": true, "chunk": 16000000},
    "seq_len": 1024
  },
  "files": ["data/fineweb/sample/10BT/000_00000.parquet", ...],
  "parts": ["part-00000.parquet", ..., "part-00007.parquet"],
  "vocab_size": 50257,
  "stats": {"docs": 148133, "tokens": 102868057, "split": 24280, "rows": 100602,
            "padding": 104387, "targets": 102719924}
}
```

`vocab_size` is the tokenizer's. Training checks it against the model's.

:::{warning}
The input files are not part of the hash, so that a cache outlives them. If the files
under a glob change, bux does not prepare the dataset again: delete its folder by
hand.
:::

In a job that spans many hosts, `--cache` must be a folder that every host sees
([Shared filesystems](../training/launching.md#shared-filesystems)).

## On Spark

{py:func}`bux.data.spark.prepare` runs the same map and reduce on Spark, as
`mapInArrow` and `applyInArrow`. It needs pyspark 4, which bux does not depend on by
default: `uv sync --extra spark`.

```python
from pyspark.sql import SparkSession

from bux.data import spark
from bux.data.pipeline import jobs
from bux.lang import compile_runs

path = "examples/experiments/speedrun.bux"
[run] = compile_runs(open(path).read(), filename=path)
session = SparkSession.builder.getOrCreate()
for job in jobs(run.train):
    spark.prepare(job, session, "s3://bucket/packed")
```

The cache can be any URI that Spark and Arrow can write to. The dataset has the same
layout as a local one, `dataset.json` included, and Spark makes the same rows as the
local engine; the test suite checks this. Only the assignment of chunks to part files
is up to Spark.

## Reading prepared data

A run reads each source epoch after epoch:

- a prepared dataset by row groups: each epoch visits them in a new order drawn from
  the run's seed, and reads the rows of a row group in order;
- a `tokens` source by windows: each epoch visits all its windows in a new order;
- `synthetic` by rows, each drawn from its own seed.

A batch takes its share of rows from each source's stream in turn, so batch `i`
depends only on the seed and `i` ([Determinism](sources.md#determinism)).

### The windows' order

The windows' order is a Feistel network over them ({py:func}`bux.train.data.permuted`).
It computes where any window goes without building the whole permutation. At 10T
tokens, which is 2.4G windows of 4096, that permutation would take 19 GB of memory in
every process and a minute per epoch. So a source can read any of its rows on its
own.

### Many processes

In a job of many processes, each process reads only the rows of a batch that its
devices hold, and leaves the rest as zeros
([Each process reads its own rows](../parallelism/multi-node.md#each-process-reads-its-own-rows)).
A prepared dataset is read one row group at a time (about a million tokens, 256 rows
of 4096), and a `tokens` source one window at a time.

On four nodes of 8 B300s, at `fsdp=32` and 256 rows per batch, each process read its
64 rows of every batch on the background thread that loads batches ahead
(2026-10-24):

| Source | Rows per process | Time per batch |
|--------|------------------|----------------|
| FineWeb-Edu's first sample shard, packed | 64 of 256, on four nodes | 11-19 ms |
| | 256 of 256, on one node | 15 ms |
| | 64 of 64, on one node | 1.6 ms |
| `tokens`, 4 shards of 1 TiB (1.1T tokens) | 64 of 256, on four nodes | 5-6 ms |
| | 64 of 64, on one node | 2.6-4.4 ms |

- A process reads the one or two row groups that hold its rows, so its 64 rows of
  256 cost as much as the whole batch does on one node. At 64 rows per batch on one
  node, a row group serves four batches.
- That cost does not grow with the number of nodes, and it is a few percent of a
  step (0.24 s).
- Reading added 0.1-0.3 GB to each process's memory. The windows' permutation alone
  would have taken 2 GiB.
- The batches on the devices were bit for bit the ones a single process reads whole,
  and the first step's loss on four nodes matched one node's exactly.
