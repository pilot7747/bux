# Your own data

In this tutorial you train on text of your own. You write a small Parquet file of
documents with pyarrow, point a run at it with `text(...)` and a tokenizer, and let
`bux prepare` tokenize the documents and pack them into rows. You hold out a split
of the documents for evaluation, compare the three packings, and train on a mixture
of two sources.

It builds on [Train on a laptop](train-on-a-laptop.md). The only download is GPT-2's
tokenizer, a few megabytes from the Hugging Face Hub.

## 1. Write some documents

bux reads documents from Parquet files: one document a row, its text in a column.
Any tool that writes Parquet will do. This script makes 4000 tiny stories, in two
files; save it as `make_stories.py` and run it from the checkout's root:

```python
"""Write a few thousand tiny stories to data/stories/, as Parquet files of one column."""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ANIMALS = ["cat", "dog", "fox", "owl", "rabbit", "bear", "duck", "mouse"]
PLACES = ["forest", "village", "garden", "meadow", "valley", "barn"]
ADJECTIVES = ["little", "brave", "sleepy", "curious", "kind", "clever", "tiny"]
DEEDS = [
    "looked for berries",
    "sang to the moon",
    "built a small house",
    "watched the clouds",
    "counted the stars",
    "played in the rain",
]
ENDINGS = [
    "became the best of friends",
    "shared a warm dinner",
    "went home happy",
    "laughed until the sun went down",
]


def sentence(rng: random.Random) -> str:
    a, b = rng.sample(ANIMALS, 2)
    return rng.choice(
        [
            f"The {rng.choice(ADJECTIVES)} {a} lived in a {rng.choice(PLACES)}.",
            f"Every day the {a} {rng.choice(DEEDS)}.",
            f"One day the {a} met a {rng.choice(ADJECTIVES)} {b}.",
            f"The {a} and the {b} {rng.choice(ENDINGS)}.",
        ]
    )


def story(rng: random.Random) -> str:
    return "Once upon a time. " + " ".join(
        sentence(rng) for _ in range(rng.randint(1, 12))
    )


rng = random.Random(0)
out = Path("data/stories")
out.mkdir(parents=True, exist_ok=True)
for part in range(2):
    texts = [story(rng) for _ in range(2000)]
    pq.write_table(pa.table({"text": texts}), out / f"part-{part}.parquet")
print(pq.read_table(out).num_rows, "stories, for example:")
print(pq.read_table(out)["text"][0])
```

```console
$ python make_stories.py
4000 stories, for example:
Once upon a time. The duck and the cat went home happy. The brave bear lived in a valley. The clever rabbit lived in a valley. Every day the bear built a small house. The tiny mouse lived in a valley. One day the duck met a little bear. Every day the owl looked for berries.
```

The stories are 1 to 12 sentences long, so they pack into rows unevenly, as real
documents do, and their grammar is simple enough that a tiny model learns it in a
couple of hundred steps.

## 2. Point a run at them

Create `mine/stories.bux`:

```bux
# A tiny Llama-style model on tiny stories of our own, in GPT-2's tokens.
d_model    = 128
vocab_size = 50304                     # GPT-2's 50257 tokens, padded to a multiple of 64

norm  = rmsnorm
attn  = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn   = gated_mlp(intermediate_size=352, activation=swish)
block = [norm attn] [norm ffn]
model = embed block^4 norm head(tied=true)

tokenizer = tokenizer(name="gpt2")
stories   = text(path="data/stories/*.parquet")
packing   = best_fit

data      = stories(split=[0, 0.95])   # 95% of the stories to train on
eval      = eval(data=stories(split=[0.95, 1]), every=50, batches=4)

optimizer = adamw(lr=3e-3)
schedule  = wsd(warmup=20)
seq_len   = 128
batch     = 8
steps     = 200
log       = log(every=25)
```

What is new:

- `tokenizer(name="gpt2")` is a BPE tokenizer, run by
  [gigatoken](https://github.com/marcelroed/gigatoken): a Hugging Face Hub repo, as
  here, or a path to a `tokenizer.json`. `vocab_size` must hold every id it makes.
- `text(path=...)` is raw text, one document a row of the Parquet files the glob
  matches, relative to where you run `bux`. Its column is `text` unless you say
  `column="..."`, and it ends every document with the tokenizer's end-of-text token
  (`eos=true`, the default; `bos=true` would also start each with one).
- `stories` is a definition like any other, so `stories(split=...)` calls it with one
  setting changed.
- `split=[0, 0.95]` is the documents whose hash falls in the first 95% of [0, 1], and
  `split=[0.95, 1]` the rest. They are disjoint and cover everything, whatever the
  tokenizer or the packing, so a held-out set needs no file of its own.
- `packing = best_fit` packs whole documents into rows, best fit decreasing.

The reference has every setting of {bux}`tokenizer`, {bux}`text` and
{bux}`best_fit`.

```console
$ bux check mine/stories.bux
d_model=128 vocab_size=50304 layers=19
rmsnorm×9  gqa×4  gated_mlp×4  embed×1  head×1
200 steps × 8 × 128 tokens = 205k tokens
optimizer=adamw(lr=0.003) schedule=wsd(warmup=20)
data=text(path="data/stories/*.parquet", split=[0.0, 0.95])
tokenizer=tokenizer(name=gpt2) packing=best_fit
```

## 3. Prepare the data

Documents become rows once, before training: a MapReduce job tokenizes them, shuffles
them by their hashes and packs them, in a process per core. `bux prepare` runs it for
every set of documents the run reads, here the training split and the eval split:

```console
$ bux prepare mine/stories.bux
text-2f218c2bc9: 3,814 documents, 249,643 tokens -> 1,981 rows of 128, 98.5% filled, 96.9% targets, 0 tokens cut (0.6s)
data/packed/text-2f218c2bc9
text-f70d7968d0: 186 documents, 12,259 tokens -> 101 rows of 128, 94.8% filled, 93.4% targets, 0 tokens cut (0.4s)
data/packed/text-f70d7968d0
```

Each line says how many documents and tokens went in, how many rows came out, how
full the rows are (the rest is padding), the share of tokens that are targets of the
loss, and how many tokens were lost to cuts. The 4000 stories split 3814 to 186, about
95 to 5.

`bux train` prepares what is missing before it trains, so `bux prepare` is only a way
to do it ahead of time, or on another machine. Options: `--workers N` processes
(default: one a core) and `--cache DIR` (default: `data/packed`).

A prepared dataset is a folder named for a hash of what made it: the documents'
settings, the tokenizer, the packing and `seq_len`. So a run finds the one it needs,
a changed setting makes another, and runs of a sweep share one:

```console
$ ls data/packed/text-2f218c2bc9
dataset.json  part-00000.parquet
```

`dataset.json`, written last, records the job, the input files, the parts and their
statistics. The input files are not in the hash, so a cache outlives them; if you
change the files under the same glob, delete the folder by hand.

## 4. Look at a row

A row is three arrays of `seq_len`: `input_ids`, `position_ids` (where each token is
in its document) and `loss_mask` (whether a token is a target). Read the end of the
first row:

```python
import pyarrow.parquet as pq

rows = pq.read_table("data/packed/text-2f218c2bc9/part-00000.parquet")
print(rows.num_rows, "rows:", ", ".join(rows.column_names))
row = rows.slice(0, 1).to_pylist()[0]
for name in rows.column_names:
    print(f"{name:12}", row[name][100:128])
```

```text
1981 rows: input_ids, position_ids, loss_mask
input_ids    [3797, 25889, 284, 262, 8824, 13, 50256, 7454, 2402, 257, 640, 13, 3887, 1110, 262, 39610, 14789, 262, 5788, 13, 50256, 0, 0, 0, 0, 0, 0, 0]
position_ids [100, 101, 102, 103, 104, 105, 106, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6]
loss_mask    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False]
```

A 107-token story ends with `50256`, GPT-2's `<|endoftext|>`; the positions start
again at 0 where the next, 14-token story begins; and 7 tokens of padding close the
row, which are not targets. Positions say everything about the packing: the model
attends, convolves and computes the loss only within each document, so a story never
sees the one packed before it.

## 5. Train

```console
$ bux train mine/stories.bux
remat=auto: none in 1s (none 0.5 GiB in 0s)
stories step  25/200  loss 1.6177  lr 3.00e-03  grad 1.168  6,423 tok/s  ...
stories step  50/200  loss 0.8029  lr 3.00e-03  grad 0.606  7,615 tok/s  ...
stories step  50/200  eval loss 0.7767  eval bpb 0.2621
stories step  75/200  loss 0.7538  lr 3.00e-03  grad 0.349  8,194 tok/s  ...
stories step 100/200  loss 0.7697  lr 3.00e-03  grad 0.516  7,836 tok/s  ...
stories step 100/200  eval loss 0.7449  eval bpb 0.2507
stories step 125/200  loss 0.7407  lr 3.00e-03  grad 0.367  8,417 tok/s  ...
stories step 150/200  loss 0.7477  lr 3.00e-03  grad 0.317  7,944 tok/s  ...
stories step 150/200  eval loss 0.7409  eval bpb 0.2469
stories step 175/200  loss 0.7432  lr 1.88e-03  grad 0.365  8,305 tok/s  ...
stories step 200/200  loss 0.7189  lr 0.00e+00  grad 0.261  7,976 tok/s  ...
stories step 200/200  eval loss 0.7293  eval bpb 0.2447
```

This time the loss falls far below the random-token runs': the stories are mostly
predictable, and the held-out stories, which the model never trained on, score as
well as the training ones. The run is slower than the 512-token one because the head
now projects to 50,304 logits a token.

The eval also reports bits per byte, `eval bpb`: the held-out stories' nats over their
bytes of text, in bits. A GPT-2 token here is about four bytes, so 0.73 nats a token
are 0.24 bits a byte; unlike the loss, the number compares with a run in another
tokenizer's tokens ([Eval](../training/running.md#eval)).

## 6. Choose a packing

How documents become rows is the `packing` setting. All three shuffle the documents
by their hashes and keep each document seeing only itself (`doc_mask`), except
`concat`, which by default reads a row as one text, as GPT-2 did:

| packing    | rows                                                                   |
|------------|------------------------------------------------------------------------|
| `concat`   | documents end to end, cut every `seq_len - 1` tokens: no padding, documents whole only by chance |
| `greedy`   | whole documents in their shuffled order, a row padded and closed when the next does not fit; longer ones dropped (`long=drop`) |
| `best_fit` | the longest first, each into the open row it leaves least room in: nearly no padding, and documents longer than a row split into pieces that fill gaps (`long=split`) |

Compare them from the command line on the training split. At 128 tokens a row:

```console
$ bux prepare mine/stories.bux packing=concat
text-7cd2acd810: 3,814 documents, 249,643 tokens -> 1,965 rows of 128, 100.0% filled, 99.2% targets, 87 tokens cut (0.7s)
...
$ bux prepare mine/stories.bux packing=greedy
text-650817e8ce: 3,814 documents, 249,643 tokens -> 2,615 rows of 128, 74.5% filled, 73.4% targets, 130 tokens cut (0.5s)
...
```

and at 64, where most stories are longer than a row:

```console
$ bux prepare mine/stories.bux seq_len=64
text-9ed51fc74f: 3,814 documents, 249,643 tokens -> 3,985 rows of 64, 98.6% filled, 96.4% targets, 0 tokens cut (0.5s)
...
$ bux prepare mine/stories.bux seq_len=64 packing=greedy
text-e8d97c415b: 3,814 documents, 249,643 tokens -> 1,470 rows of 64, 74.4% filled, 72.4% targets, 179,690 tokens cut (0.5s)
...
```

| packing, `seq_len` | rows  | filled | tokens cut |
|--------------------|-------|--------|------------|
| `best_fit`, 128    | 1,981 | 98.5%  | 0          |
| `concat`, 128      | 1,965 | 100.0% | 87         |
| `greedy`, 128      | 2,615 | 74.5%  | 130        |
| `best_fit`, 64     | 3,985 | 98.6%  | 0          |
| `greedy`, 64       | 1,470 | 74.4%  | 179,690    |

`best_fit` fills rows nearly as densely as `concat` and keeps documents whole;
`greedy` pads a quarter of every row and drops what does not fit, which suits chat
data, where a cut conversation may lose its answer. Rows of `concat`, and pieces of a
split document, overlap by a token, so that every token is a target once.

[Packing](../data/packing.md) has every option (`long`, `max_docs`, `sort`,
`doc_mask`, `chunk`).

## 7. Mix two sources

A run can read several sources at once. `+` mixes them and `*` weights them, so every
batch of `0.75 * a + 0.25 * b` has three quarters of its rows from `a`. Write a second
source, sums written out as text, in a column named `content`. Save this as
`make_facts.py` and run it:

```python
"""Write sums as text to data/facts/, in a column named content."""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

rng = random.Random(1)
sums = [(rng.randint(0, 99), rng.randint(0, 99)) for _ in range(3000)]
texts = [f"{a} plus {b} is {a + b}." for a, b in sums]
Path("data/facts").mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({"content": texts}), "data/facts/sums.parquet")
print(len(texts), "facts, for example:", texts[0])
```

```console
$ python make_facts.py
3000 facts, for example: 17 plus 72 is 89.
```

Then create `mine/mixed.bux`, which uses the stories run and changes only its data:

```bux
# The stories run, on a mixture: three rows of stories to one row of sums.
use stories

facts = text(path="data/facts/*.parquet", column="content")
data  = 0.75 * stories(split=[0, 0.95]) + 0.25 * facts
eval  = eval(data=stories(split=[0.95, 1]), every=50, batches=4)
```

```console
$ bux train mine/mixed.bux
text-9fab7efe63: 3,000 documents, 21,000 tokens -> 167 rows of 128, 98.2% filled, 84.2% targets, 0 tokens cut (0.5s)
remat=auto: none in 1s (none 0.5 GiB in 0s)
mixed step  25/200  loss 2.0186  lr 3.00e-03  grad 1.066  6,204 tok/s  ...
mixed step  50/200  loss 1.0552  lr 3.00e-03  grad 0.479  7,192 tok/s  ...
mixed step  50/200  eval loss 0.7869
...
mixed step 200/200  loss 0.9300  lr 0.00e+00  grad 0.254  7,976 tok/s  ...
mixed step 200/200  eval loss 0.7337
```

`bux train` prepared the one dataset that was missing, the sums, and reused the
stories'. The training loss is now the mean over both sources; the eval, on held-out
stories alone, lands close to the stories-only run. Every source is read in order,
epoch after epoch, so batch `i` depends only on the seed and `i`: a run that resumes
reads exactly what it would have read.

## 8. Going bigger

The same settings scale to real corpora:

- `examples/experiments/speedrun.bux` trains on FineWeb's 10B-token sample, and
  `bux prepare` packs its 29 GB of Parquet in under three minutes on 16 cores
  ([Data](../data/index.md)).
- `uv sync --extra spark` and `bux.data.spark.prepare` run the same job on Spark,
  and make the same rows ([Pipeline](../data/pipeline.md)).
- `tokens(path=..., dtype=uint16)` reads shards of token ids that are tokenized
  already, through memory maps, with no preparation
  ([Sources](../data/sources.md)).

## What you learned

| you wrote                                 | it does                                      |
|-------------------------------------------|----------------------------------------------|
| `tokenizer(name="gpt2")`                  | tokenizes with a Hub repo's BPE              |
| `text(path="...*.parquet", column=...)`   | a document a row of Parquet                  |
| `split=[0, 0.95]` and `[0.95, 1]`         | a training set and a held-out set, by hash   |
| `packing = best_fit`                      | whole documents, nearly no padding           |
| `0.75 * a + 0.25 * b`                     | a mixture, by rows of every batch            |
| `bux prepare`                             | tokenize and pack once, into `data/packed/`  |

Next, [fine-tune on chat](fine-tuning-chat.md) trains on conversations, with the loss
on the replies alone.
