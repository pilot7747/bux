# Data

A run trains on **rows** of `seq_len` tokens. Documents, raw text or conversations,
become rows once, before training: a MapReduce job tokenizes them, shuffles them and
packs them, on this machine's cores or on Spark. This section covers the sources a
run reads, how documents are packed into rows, and the pipeline that does it.

## A quick example

Three settings of the run say what the job does: the tokenizer, the documents, and
the packing.

```bux
tokenizer = tokenizer(name="gpt2")
fineweb   = text(path="data/fineweb/sample/10BT/*.parquet", bos=true, eos=false)
packing   = best_fit

data = fineweb(split=[0, 0.99])
eval = eval(data=fineweb(split=[0.99, 1]))
```

```console
$ bux prepare examples/experiments/speedrun.bux
text-0c8c782640: 14,720,729 documents, 10,248,316,202 tokens -> 10,022,812 rows of 1024, 99.9% filled, 99.7% targets, 0 tokens cut (175.2s)
text-17a28baeab: 148,133 documents, 102,868,057 tokens -> 100,602 rows of 1024, 99.9% filled, 99.7% targets, 0 tokens cut (16.5s)
```

That is the speedrun's FineWeb, 29 GB of Parquet, in under three minutes on 16 cores
of a laptop, to 17 GB of rows. The 1% held out comes from the same files, by a hash of
each document ([Splits](sources.md#splits)). `bux train` prepares what is missing
first, so `bux prepare` is only a way to do it ahead of time, or elsewhere.

## Rows

A row is three arrays of `seq_len`:

| Array | What it holds |
|-------|---------------|
| `input_ids` | the tokens |
| `position_ids` | where each token is in its document: a document starts wherever they stop increasing |
| `loss_mask` | whether a token is a target: false for prompts, padding and templates |

Positions say everything about the packing. A document starts at a row's first token
and wherever positions stop increasing, as at the 0 after a document. The model
rotates and embeds by them, and attends, convolves and hashes n-grams only within a
document ([Packing](packing.md#documents-stay-apart)).

The loss of a head that predicts `k` tokens ahead is on token `t + k` from position
`t` when `loss_mask[t + k]` is true and token `t + k` is in the document of token
`t`: a target of the document that predicts it, never the first token of the next
one. It is the mean over every target of the batch, so that a row of many short
replies weighs no more than a row of one long one; the z-loss is the mean over the
same positions of the first head.

## Documents to rows

Four documents, A of 5 tokens, B of 3, C of 10 and D of 2, packed into rows of 8 by
`best_fit`, the default (`bux.data.packing` made these rows):

```text
documents      A A A A A   B B B   C C C C C C C C C C   D D

               row 0             row 1             row 2
input_ids      C C C D D · · ·   C C C C C C C C   A A A A A B B B
position_ids   0 1 2 0 1 0 1 2   0 1 2 3 4 5 6 7   0 1 2 3 4 0 1 2
loss_mask      1 1 1 1 1 0 0 0   1 1 1 1 1 1 1 1   1 1 1 1 1 1 1 1
```

- C is longer than a row, so it is split into pieces of 8 that overlap by a token:
  its second piece, in row 0, starts with C's eighth token again, at position 0. Its
  `loss_mask` is set, but nothing predicts a position 0, so only the first copy, in
  row 1, is a target.
- D fills the gap after C's second piece, and the rest of row 0 is padding (`·`),
  which is no target.
- The rows come out in a random order.

So every token is a target exactly once, but the first of each document.
[Packing](packing.md) compares `best_fit` with `concat` and `greedy` on the same four
documents.

## In this section

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Sources
:link: sources
:link-type: doc

Text and chat in Parquet, token shards, random tokens, the tokenizer, splits and
mixtures.
:::

:::{grid-item-card} Packing
:link: packing
:link-type: doc

`concat`, `greedy` and `best_fit`, and how documents stay apart in attention.
:::

:::{grid-item-card} The pipeline
:link: pipeline
:link-type: doc

`bux prepare` as a MapReduce job, the cache, Spark, and reading prepared data.
:::

:::{grid-item-card} Design notes
:link: design
:link-type: doc

Why offline, why hashes of content, why positions, and the packings considered.
:::
::::

```{toctree}
:hidden:

sources
packing
pipeline
design
```
