# Data

This section is about training on your own documents. Training data is Parquet:
plain text, or chat conversations in the OpenAI format. bux turns it into packed rows
of tokens once, before training, with a MapReduce job that runs on this machine's
cores or on Spark. Every run of a sweep then reads the same rows. A run on random
tokens or on pre-tokenized shards skips all of this and needs only
[Sources](sources.md).

## Quick start

Three settings say what the job does: the tokenizer, the documents, and the packing.
These are the lines of `examples/experiments/speedrun.bux` that train on FineWeb:

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

That is 29 GB of Parquet turned into 17 GB of rows in under three minutes, on a
16-core laptop. The two lines are the two datasets the run reads: the training split
and the 1% held out for eval. Both come from the same files, split by a hash of each
document's content ([Splits](sources.md#splits)).

You rarely need to call `bux prepare` yourself: `bux train` prepares whatever is
missing before it starts. Run `bux prepare` to do the work ahead of time, or on
another machine.

## Rows

A row is three arrays of `seq_len` values:

| Array | What it holds |
|-------|---------------|
| `input_ids` | the tokens |
| `position_ids` | each token's position in its document |
| `loss_mask` | whether the token is a target: false for prompts, padding and templates |

The positions encode the packing. A document starts at the first token of a row, and
wherever the positions stop increasing. The model rotates and embeds tokens by these
positions, and it attends, convolves and hashes n-grams only within a document
([Documents stay apart](packing.md#documents-stay-apart)).

A head that predicts `k` tokens ahead is scored at position `t` on token `t + k`
when both of these hold:

- `loss_mask[t + k]` is true;
- token `t + k` is in the same document as token `t`.

So a document's tokens are only ever predicted from inside that document, never from
the end of the one before it. The loss is the mean over every target in the batch,
so a row of many short replies weighs no more than a row of one long reply. The
z-loss is the mean over the same positions, for the first head.

## Documents to rows

Here are four documents, A of 5 tokens, B of 3, C of 10 and D of 2, packed into rows
of 8 by `best_fit`, the default. `bux.data.packing` made these rows:

```text
documents      A A A A A   B B B   C C C C C C C C C C   D D

               row 0             row 1             row 2
input_ids      C C C D D · · ·   C C C C C C C C   A A A A A B B B
position_ids   0 1 2 0 1 0 1 2   0 1 2 3 4 5 6 7   0 1 2 3 4 0 1 2
loss_mask      1 1 1 1 1 0 0 0   1 1 1 1 1 1 1 1   1 1 1 1 1 1 1 1
```

- C is longer than a row, so it is split into pieces of 8 that overlap by one token.
  Its second piece, in row 0, starts again with C's eighth token, at position 0.
  Nothing predicts position 0, so only the first copy of that token, in row 1, is a
  target.
- D fills the gap after C's second piece. The rest of row 0 is padding (`·`), which
  is never a target.
- The rows come out in a random order.

Every token is a target exactly once, except the first token of each document.
[Packing](packing.md) packs the same four documents with `concat` and `greedy` for
comparison.

## In this section

Read the pages in this order. The tutorials [Your own data](../tutorials/your-own-data.md)
and [Fine-tuning on chat](../tutorials/fine-tuning-chat.md) walk through the same
material step by step.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Sources
:link: sources
:link-type: doc

Text and chat in Parquet, token shards, random tokens, the tokenizer, held-out splits
and mixtures.
:::

:::{grid-item-card} Packing
:link: packing
:link-type: doc

`concat`, `greedy` and `best_fit` compared, and how documents in one row stay apart.
:::

:::{grid-item-card} The pipeline
:link: pipeline
:link-type: doc

`bux prepare` as a MapReduce job, the cache, Spark, and how training reads the rows.
:::

:::{grid-item-card} Design notes
:link: design
:link-type: doc

Why data is prepared offline, keyed by content, and packed by positions.
:::
::::

```{toctree}
:hidden:

sources
packing
pipeline
design
```
