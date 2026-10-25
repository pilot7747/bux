# Packing

The `packing` setting says how `text` and `chat` documents become rows of `seq_len`
tokens: end to end and cut, or whole and padded, or whole and fitted together. This
page compares the three packings, lists their settings, and explains how documents
packed into one row stay apart.

```bux
packing = best_fit                     # the default
packing = greedy(long=drop)            # classic SFT: whole documents, a row padded
packing = concat                       # GPT-2: end to end, cut every row
```

Documents are shuffled by their hashes and packed `chunk` tokens at a time, each
chunk a random sample of them, and each chunk's rows come out in a random order
([The pipeline](pipeline.md)).

## The three packings

| Packing | Rows | `doc_mask` by default |
|---------|------|-----------------------|
| {bux}`concat` | documents end to end, cut every `seq_len - 1` tokens: no padding, and documents are whole only by chance (GPT-2, the speedrun's shards) | `false` |
| {bux}`greedy` | whole documents in their shuffled order, a row padded and closed when the next does not fit (classic SFT); `max_docs=1` is one a row | `true` |
| {bux}`best_fit` | best-fit decreasing (Ding et al., 2024): the longest first, each into the open row it leaves least room in; nearly no padding and nearly no cuts | `true` |

The four documents of [Documents to rows](index.md#documents-to-rows), A of 5 tokens,
B of 3, C of 10 and D of 2, in rows of 8 (`·` is padding):

```text
concat                     AAAAABBB  BCCCCCCC              C's last 3 tokens and D cut
greedy                     AAAAABBB  DD······              C, longer than a row, dropped
best_fit                   CCCDD···  CCCCCCCC  AAAAABBB    C split in two pieces, nothing cut
best_fit(long=truncate)    DD······  CCCCCCCC  AAAAABBB    C's last 2 tokens cut
```

`concat`'s rows read as one text each, positions 0 to 7; the others' positions start
again at every document.

## Settings

| Field | `concat` | `greedy` | `best_fit` | What it is |
|-------|----------|----------|------------|------------|
| `doc_mask` | `false` | `true` | `true` | positions start again at every document, and attention stays within it ([below](#documents-stay-apart)) |
| `long` | | `drop` | `split` | a document longer than a row: `split` into pieces that fill gaps, `truncate` it, or `drop` it |
| `max_docs` | | `none` | `none` | the most documents, and pieces of them, in a row |
| `sort` | | | `true` | longest first; `false` keeps the shuffled order (online best fit) |
| `chunk` | `16_000_000` | `16_000_000` | `16_000_000` | tokens of documents packed together |

- `greedy` and `best_fit` keep documents whole. `greedy` drops a long one by default,
  as a cut conversation may lose its answer; `best_fit` splits it.
- `max_docs` caps a row's documents and pieces, for attention kernels that take a
  fixed number of them.
- `best_fit(sort=false)` is online best fit, in the shuffled order: rows of mixed
  lengths, rather than long documents together and short ones together.

## Cuts lose no targets

Rows of `concat`, and the pieces of a split document, overlap by a token. Its first
copy is a target, and its second, at position 0, is not: nothing predicts the first
token of a row or a document. So every token is a target once, but the first of every
document.

## Documents stay apart

With `doc_mask`, positions start again at every document, and a document starts
wherever positions stop increasing. RoPE and a learned positional embedding take the
positions as they are. The model makes the rest from them once a forward pass
({py:class}`~bux.architecture.common.segments.Segments`), none of it t × t:

- a document id per token: the segment ids that splash and Pallas flash attention take
  in place of a t × t mask, so attention stays within a document and skips the blocks
  no query sees;
- how far each token is into its document: where causal convolutions, n-grams and
  targets stop looking back;
- where each query's first key is, for kernels that take that (tokamax's); the
  `cu_seqlens` of varlen kernels are where the ids change.

The loss is only on targets of the document that predicts them ([Rows](index.md#rows)).
Without `doc_mask`, a row reads as one text, as GPT-2's did: positions run from 0 to
`seq_len - 1`, and every token attends to every one before it in the row.

## Measurements

On FineWeb-Edu (53,248 documents, 55M GPT-2 tokens, `seq_len = 1024`):

| Packing | Rows | Filled | Tokens cut |
|---------|------|--------|------------|
| `concat` | 54,080 | 100% | 2,161 (the ends of chunks) |
| `best_fit` | 54,098 | 99.9% | 0 |
| `greedy` | 25,016 | 74.7% | 36M (the documents over 1024) |

On a laptop, 20,000 FineWeb documents (13.9M GPT-2 tokens) in rows of 1024, a
hundredth held out, prepared by a sweep over the packings,
`packing = concat | greedy | best_fit`:

```console
$ bux prepare packings.bux --workers 4
text-d56ac6f021: 19,791 documents, 13,910,395 tokens -> 13,597 rows of 1024, 100.0% filled, 99.9% targets, 663 tokens cut (2.1s)
data/packed/text-d56ac6f021
text-a83ecd6142: 209 documents, 141,267 tokens -> 138 rows of 1024, 100.0% filled, 99.9% targets, 92 tokens cut (0.6s)
data/packed/text-a83ecd6142
text-dd2cb5346c: 19,791 documents, 13,910,395 tokens -> 8,158 rows of 1024, 77.0% filled, 76.8% targets, 7,481,586 tokens cut (1.5s)
data/packed/text-dd2cb5346c
text-b52216a7fa: 209 documents, 141,267 tokens -> 86 rows of 1024, 77.0% filled, 76.8% targets, 73,498 tokens cut (0.6s)
data/packed/text-b52216a7fa
text-be90cc916b: 19,791 documents, 13,910,395 tokens -> 13,604 rows of 1024, 99.9% filled, 99.7% targets, 0 tokens cut (1.8s)
data/packed/text-be90cc916b
text-c1beb61b0e: 209 documents, 141,267 tokens -> 139 rows of 1024, 99.3% filled, 99.1% targets, 0 tokens cut (0.6s)
data/packed/text-c1beb61b0e
```

Each packing makes a dataset of its own, for training and for the held-out split
alike. A line's figures:

| Figure | What it counts |
|--------|----------------|
| filled | the share of the rows' tokens that are not padding |
| targets | the share that are targets: in the loss mask, and not a document's first |
| tokens cut | tokens of documents that no row holds: dropped, truncated, or left at the end of a chunk by `concat` |
