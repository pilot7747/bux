# The pipeline

`bux prepare` turns a run's documents into rows, once, as a MapReduce job: on this
machine's cores, or on Spark. This page covers the job, the cache it writes to, how to
run it on Spark, and how training reads what it made.

```sh
bux prepare examples/experiments/sft.bux                      # every core of this machine
bux prepare examples/experiments/sft.bux --workers 8          # 8 processes
bux prepare examples/experiments/sft.bux --cache /shared/packed
```

| Option | Default | What it does |
|--------|---------|--------------|
| `--cache DIR` | `data/packed` | where prepared datasets go, and where training looks for them |
| `--workers N` | one a core | the processes the map and the reduce run in; `0` runs them in this one |

`bux prepare` prepares every dataset the file's runs read, in training and in eval,
once each, and prints the folder of each. A dataset already in the cache is not
prepared again. `bux train` and `bux launch` prepare what is missing before they
train, so `bux prepare` is a way to do it ahead of time, or on another machine. A file
that reads no documents, only `tokens` or `synthetic`, has nothing to prepare:

```console
$ bux prepare examples/experiments/smoke.bux
this file reads no documents to prepare
```

## The job

```text
Parquet ─map─▶ documents under random keys ─shuffle─▶ chunks ─reduce─▶ rows ─▶ Parquet
```

1. **The map** ({py:meth}`~bux.data.pipeline.Job.map`) turns record batches into
   documents: a 63-bit key, the hash of the content, and `input_ids` and `loss_mask`
   as Arrow lists ({py:mod}`bux.data.documents`).
2. **The shuffle** groups documents by the top bits of their keys into a power of two
   of chunks, of about `packing.chunk` tokens each. Grouped by bits, the groups are
   exact in any engine, and each is a random sample of the documents.
3. **The reduce** ({py:meth}`~bux.data.pipeline.Job.reduce`) sorts a chunk by key,
   whatever order an engine delivered it in, plans its rows by the packing
   ({py:mod}`bux.data.packing`), and writes them a row group at a time.

Both are functions over Arrow record batches, with nothing else to share, so any
engine that speaks Arrow runs them.

### On this machine

{py:func}`bux.data.pipeline.prepare` runs the job in one process per core:

- Each map task takes about 256 MiB of a Parquet file's row groups. It spills its
  documents, sorted by key, as raw arrays: about 5 bytes per token of scratch space
  next to the cache, deleted when the job ends.
- Each reduce finds its key range in every spill by bisection, and maps in its
  documents and nothing else, so that the shuffle reads every byte once.
- The dataset is written to a temporary folder and moved into place last. Two jobs
  that prepare the same dataset at once, as the tasks of an array job may, both
  succeed, and the second uses the first's.

That is how the speedrun's FineWeb, 29 GB of Parquet, becomes 17 GB of rows in under
three minutes on 16 cores of a laptop ([Data](index.md#a-quick-example)).

## The cache

A prepared dataset is a folder named for a hash of what made it: the documents'
source setting, the tokenizer, the packing and `seq_len`. So a run finds the one it
needs, a changed setting makes another, and a cache can be shared by runs and copied
between machines. A curriculum's phase that reads documents at another `seq_len`
reads a dataset of its own, which `bux prepare` makes with the rest
([What each phase reads](../training/curriculum.md#what-each-phase-reads)):

```text
data/packed/text-0c8c782640/
    part-00000.parquet ...    rows: a row group of about a million tokens
    dataset.json              the job, its input files, the parts, statistics; written last
```

A part is a chunk's rows, `input_ids`, `position_ids` and `loss_mask` of `seq_len`
each, zstd-compressed, in row groups of about 2^20 tokens (256 rows of 4096). The
held-out split of the speedrun's FineWeb, `text-17a28baeab/dataset.json`:

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

`vocab_size` is the tokenizer's, which training checks against the model's.

:::{warning}
The input files are not in the hash, so that a cache outlives them. A folder made
from other files under the same glob is not made again: delete it by hand.
:::

In a job of many hosts, `--cache` is a folder every host sees
([Shared filesystems](../training/launching.md#shared-filesystems)).

## On Spark

{py:func}`bux.data.spark.prepare` runs the same map and reduce on Spark, as
`mapInArrow` and `applyInArrow`. It needs pyspark 4, which bux does not depend on:
`uv sync --extra spark`.

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

The cache is any URI Spark and Arrow can write to, and the dataset is laid out as the
local engine lays it out, with a `dataset.json` of its own. Spark makes the same rows
as this machine does; the test suite checks it. Which file holds a chunk's rows is up
to Spark.

## Reading prepared data

A run reads every source epoch after epoch:

- a prepared dataset by row groups, each epoch in a new order of them drawn from the
  run's seed, each row group's rows in order;
- a `tokens` source by windows, each epoch in a new order of all its windows;
- `synthetic` by rows, each drawn from a seed of its own.

A batch takes its share of rows from each source's stream in turn, so batch `i`
depends only on the seed and `i` ([Determinism](sources.md#determinism)).

### The windows' order

The windows' order is a Feistel network over them ({py:func}`bux.train.data.permuted`),
which says where any window goes without making the permutation of them all: at 10T
tokens, 2.4G windows of 4096, the permutation would take 19 GB of every process's
memory and a minute an epoch. So a source reads any of its rows alone.

### Many processes

In a job of many processes each reads only the rows of a batch that its devices hold,
and leaves the others 0 ([Each process reads its own rows](../parallelism/multi-node.md#each-process-reads-its-own-rows)). A prepared
dataset is read a row group at a time, about a million tokens, 256 rows of 4096, and a
`tokens` source a window at a time.

On four nodes of 8 B300s, `fsdp=32` at 256 rows a batch, each process read its 64 rows
of a batch, on the thread that loads batches ahead, in:

| Source | Rows a process | A batch |
|--------|----------------|---------|
| FineWeb-Edu's first sample shard, packed | 64 of 256, on four nodes | 11-19 ms |
| | 256 of 256, on one | 15 ms |
| | 64 of 64, on one | 1.6 ms |
| `tokens`, 4 shards of 1 TiB: 1.1T tokens | 64 of 256, on four nodes | 5-6 ms |
| | 64 of 64, on one | 2.6-4.4 ms |

- A process reads the one or two row groups that hold its rows, so its 64 rows of 256
  cost what the whole batch does on one node. At 64 rows a batch on one node, a row
  group serves four batches.
- That cost does not grow with the nodes, and it is a few percent of a step (0.24 s).
- Reading added 0.1-0.3 GB to a process's memory, where the windows' permutation
  alone would take 2 GiB.
- The batches on the devices were, bit for bit, those one process reads whole, and
  the first step's loss on four nodes was one node's to the bit.
