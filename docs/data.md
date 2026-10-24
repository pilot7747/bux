# Data: documents to rows

A run trains on rows of `seq_len` tokens. Documents, raw text or conversations, become
rows once, before training: a MapReduce job tokenizes them, shuffles them, and packs
them, on this machine's cores or on Spark. What it does is three settings of the run:

```
tokenizer = tokenizer(name="gpt2")
fineweb   = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
packing   = best_fit

data = fineweb(split=[0, 0.99])
eval = eval(data=fineweb(split=[0.99, 1]))
```

```
$ bux prepare examples/experiments/speedrun.bux
text-0c8c782640: 14,720,729 documents, 10,248,316,202 tokens -> 10,022,812 rows of 1024, 99.9% filled, 99.7% targets, 0 tokens cut (175.2s)
text-17a28baeab: 148,133 documents, 102,868,057 tokens -> 100,602 rows of 1024, 99.9% filled, 99.7% targets, 0 tokens cut (16.5s)
```

That is the speedrun's FineWeb, 29 GB of Parquet, in under three minutes on 16 cores
of a laptop, to 17 GB of rows; the 1% held out comes from the same files. `bux train`
prepares what is missing first, so `bux prepare` is only a way to do it ahead, or
elsewhere.

## Rows

A row is three arrays of `seq_len`:

| array          | what                                                                 |
|----------------|----------------------------------------------------------------------|
| `input_ids`    | tokens                                                               |
| `position_ids` | where each token is in its document, until they stop increasing      |
| `loss_mask`    | whether a token is a target: false for prompts, padding, templates   |

Positions say everything about the packing. A document starts at a row's first token
and wherever positions stop increasing, as at the 0 after a document. The model
rotates and embeds by them, and attends, convolves and hashes n-grams only within a
document. The loss of a head that predicts `k` tokens ahead is on token `t + k` from
position `t` when `loss_mask[t + k]` and token `t + k` is in the document of token
`t`: a target of the document that predicts it, never the first token of the next
one. It is the mean over every target of the batch, so that a row of many short
replies weighs no more than a row of one long one; the z-loss is the mean over the
same positions of the first head.

## Documents

`text(...)` is raw text, one document per row of Parquet files:

| setting  | what                                                        | default    |
|----------|-------------------------------------------------------------|------------|
| `path`   | a glob of Parquet files, relative to the working directory | required   |
| `column` | the column of text                                          | `text`     |
| `bos`    | start every document with the tokenizer's `bos`             | `false`    |
| `eos`    | end every document with its `eos`                           | `true`     |
| `split`  | the documents whose hash falls in `[start, end)`            | `[0, 1]`   |

`chat(...)` is conversations in the OpenAI format, a list of `{role, content}`
messages per row, `content` a string or a list of `{type: text, text}` parts:

| setting    | what                                                        | default    |
|------------|-------------------------------------------------------------|------------|
| `path`     | a glob of Parquet files                                     | required   |
| `column`   | the column of messages                                      | `messages` |
| `template` | `chatml` (Qwen, SmolLM) or `llama3`                         | `chatml`   |
| `train_on` | `assistant` replies, the `last` one, or `all` of it         | `assistant`|
| `split`    | as for `text`                                               | `[0, 1]`   |

A reply is learned with the token that ends it (`<|im_end|>`, `<|eot_id|>`), so that
the model learns to stop; a conversation without one is dropped.

A document's hash is of its content, so `split=[0, 0.99]` and `split=[0.99, 1]` are
disjoint and cover everything, whatever the tokenizer or the packing: a held-out set
needs no file of its own. A split is also a deterministic sample, `split=[0, 0.1]`.

`tokenizer(name=...)` is a BPE tokenizer run by
[gigatoken](https://github.com/marcelroed/gigatoken), at gigabytes a second: a
Hugging Face Hub repo (`gpt2`, `Qwen/Qwen3-0.6B`) or a path to a `tokenizer.json`.
`bos` and `eos` name its tokens (`<|endoftext|>` by default; Llama 3's are
`<|begin_of_text|>` and `<|end_of_text|>`), and `added` appends special tokens after
its vocabulary, for a template it lacks: GPT-2's 50257 tokens, padded to 50304, have
room for `["<|im_start|>", "<|im_end|>"]`. Special tokens written in documents stay
text, as tiktoken's `encode_ordinary` keeps them: a web page that mentions
`<|endoftext|>` does not end there. Only `bos`, `eos` and templates put them in.

## Packing

Documents are shuffled by their hashes and packed `chunk` tokens at a time, each chunk
a random sample of them. All three packings take `doc_mask`: with it, positions start
again at every document and attention stays within it; without it, a row reads as
one text, as GPT-2's did.

| packing        | rows                                                          | `doc_mask` |
|----------------|---------------------------------------------------------------|------------|
| `concat`       | documents end to end, cut every `seq_len - 1` tokens: no padding, and documents are whole only by chance (GPT-2, the speedrun's shards) | `false` |
| `greedy`       | whole documents in their shuffled order, a row padded and closed when the next does not fit (classic SFT); `max_docs=1` is one per row | `true` |
| `best_fit`     | best-fit decreasing (Ding et al., 2024): the longest first, each into the open row it leaves least room in; nearly no padding and nearly no cuts | `true` |

`greedy` and `best_fit` keep documents whole. `long` says what happens to one longer
than a row: `split` into pieces that fill gaps (`best_fit`'s default), `truncate` it,
or `drop` it (`greedy`'s default, as a cut conversation may lose its answer).
`max_docs` caps the documents and pieces in a row, for attention kernels that take a
fixed number of them. `best_fit(sort=false)` is online best fit, in the shuffled order:
rows of mixed lengths rather than long documents together and short ones together.

Cuts lose no targets: rows of `concat`, and pieces of a split document, overlap by a
token, whose first copy is a target and whose second, at position 0, is not. So
every token is a target once, but the first of every document.

On FineWeb-Edu (53,248 documents, 55M GPT-2 tokens, `seq_len = 1024`):

| packing                  | rows    | filled | tokens cut   |
|--------------------------|---------|--------|--------------|
| `concat`                 | 54,080  | 100%   | 2,161 (the ends of chunks) |
| `best_fit`               | 54,098  | 99.9%  | 0            |
| `greedy`                 | 25,016  | 74.7%  | 36M (the documents over 1024) |

## The pipeline

```
Parquet ─map─▶ documents under random keys ─shuffle─▶ chunks ─reduce─▶ rows ─▶ Parquet
```

The **map** (`Job.map`) turns record batches into documents: a 63-bit key, the hash
of the content, and `input_ids` and `loss_mask` as Arrow lists. The **shuffle**
groups documents by the top bits of their keys into a power of two of chunks of
about `chunk` tokens, so that the groups are exact in any engine, and each is a
random sample. The **reduce** (`Job.reduce`) sorts a chunk by key, whatever order an
engine delivered it in, plans its rows, and writes them a row group at a time.

Both are functions over Arrow record batches, with nothing else to share, so any
engine that speaks Arrow runs them. `bux.data.pipeline.prepare` runs them on this
machine, in one process per core: the map spills each task's documents, sorted by key,
as raw arrays, about 5 bytes per token of scratch space next to the cache, and each
reduce finds its key range in every spill by bisection and maps in its documents and
nothing else, so that the shuffle reads every byte once.
`bux.data.spark.prepare` runs them on Spark, as `mapInArrow` and `applyInArrow`
(`uv sync --extra spark`):

```python
from pyspark.sql import SparkSession

from bux.data import spark
from bux.data.pipeline import jobs
from bux.lang import compile_runs

[run] = compile_runs(open("examples/experiments/speedrun.bux").read())
for job in jobs(run.train):
    spark.prepare(job, SparkSession.builder.getOrCreate(), "s3://bucket/packed")
```

Spark makes the same rows as this machine does; the test suite checks it.

A prepared dataset is a folder named for a hash of what made it: the documents, the
tokenizer, the packing and `seq_len`. So a run finds the one it needs, a changed
setting makes another, and a cache can be shared by runs and copied between machines:

```
data/packed/text-0c8c782640/
    part-00000.parquet ...    rows; a row group of about a million tokens
    dataset.json              the job, its input files, the parts, statistics; written last
```

The input files are not in the hash, so that a cache outlives them: a folder made
from other files under the same glob is deleted by hand.

## Reading

A run reads every source epoch after epoch: a prepared dataset's row groups, or a
`tokens` source's windows, each epoch in a new order drawn from the run's seed. A
mixture has `round(weight * batch)` rows of each source in every batch, by largest
remainder. So batch `i` depends only on the seed and `i`, and a run that resumes at
step `i` reads what it would have read.

The windows' order is a Feistel network over them (`data.permuted`), which says where
any window goes without making the permutation of them all: at 10T tokens, 2.4G
windows of 4096, the permutation would take 19 GB of every process's memory and a
minute an epoch. `synthetic` draws each row of its stream from a seed of its own. So
a source reads any rows alone, and in a job of many processes each reads only the rows
of a batch that its devices hold
([Many nodes](parallelism.md#many-nodes)).

A prepared dataset is read a row group at a time, about a million tokens, 256 rows of
4096, and a `tokens` source a window at a time. On four nodes of 8 B300s, `fsdp=32` at
256 rows a batch, each process read its 64 rows of a batch, on the thread that loads
batches ahead, in:

| source                                    | rows a process           | a batch    |
|-------------------------------------------|--------------------------|------------|
| FineWeb-Edu's first sample shard, packed  | 64 of 256, on four nodes | 11-19 ms   |
|                                           | 256 of 256, on one       | 15 ms      |
|                                           | 64 of 64, on one         | 1.6 ms     |
| `tokens`, 4 shards of 1 TiB: 1.1T tokens  | 64 of 256, on four nodes | 5-6 ms     |
|                                           | 64 of 64, on one         | 2.6-4.4 ms |

A process reads the one or two row groups that hold its rows, so its 64 rows of 256
cost what the whole batch does on one node, where at 64 rows a batch a group serves
four batches. That does not grow with the nodes, and it is a few percent of a step,
0.24 s. Reading added 0.1-0.3 GB to a process's memory, where the windows'
permutation alone would take 2 GiB. The batches on the devices were, bit for bit,
those one process reads whole, and the first step's loss on four nodes was one node's
to the bit.

## Design notes

**Offline, once.** Packing well needs to see many documents at once, which is cheap
in a MapReduce job and awkward in a training loop that must resume exactly. Done
once, a dataset is reused by every run of a sweep, and reading it costs nothing.

**Keys are hashes of content**, not of where a document is: the same documents land
in the same chunks in the same order whatever the files, the tokenizer or the
engine, splits need no bookkeeping, and the reduce's output depends on nothing but
its chunk.

**Positions, not segment ids.** Resetting positions per document carries the
segments and the positions RoPE needs in one array. The model makes the rest from
them once per forward pass (`bux.architecture.Segments`): a document id per token, the
segment ids that splash and Pallas flash attention take in place of a t by t mask,
and how far each token is into its document, where convolutions and the loss stop,
and where each query's first key is for kernels that take that (tokamax's). The
`cu_seqlens` of varlen kernels are where the ids change.

**Packings considered.** First-fit decreasing (Axolotl's multipack, NeMo) and
shortest-pack-first histogram packing (Krell et al., 2021) pack about as well as best
fit, which is at least as good and faster (Ding et al., 2024), and order-preserving
first fit is `best_fit(sort=false)`. Balancing rows across data-parallel ranks by
attention cost (WLB-LLM, 2025) matters at long context, once there are ranks.
Dataset decomposition (Pouransari et al., 2024) packs documents into rows of several
lengths, for a curriculum over them; it needs a trainer that takes several shapes.
Packing similar documents together (In-Context Pretraining, Shi et al., 2023; BM25
chunks, Zhao et al., 2024) needs attention across documents, and a retrieval step
before the reduce.
