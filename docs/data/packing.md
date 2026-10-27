# Packing

The `packing` setting says how `text` and `chat` documents become rows of `seq_len`
tokens. There are three choices: documents end to end and cut, whole documents
padded, or whole documents fitted together. This page compares them, lists their
settings, and explains how documents packed into one row stay apart.

```bux
packing = best_fit                     # the default
packing = greedy(long=drop)            # classic SFT: whole documents, each row padded
packing = concat                       # GPT-2: end to end, cut at every row
```

Documents are shuffled by their hashes and packed `chunk` tokens at a time. Each
chunk is a random sample of the documents, and each chunk's rows come out in a random
order ([The pipeline](pipeline.md)).

## The three packings

| Packing | Rows | `doc_mask` by default |
|---------|------|-----------------------|
| {bux}`concat` | documents end to end, cut every `seq_len - 1` tokens: no padding, and a document is whole only by chance (GPT-2, the speedrun's shards) | `false` |
| {bux}`greedy` | whole documents in their shuffled order; a row is padded and closed when the next document does not fit (classic SFT). `max_docs=1` puts one document in each row | `true` |
| {bux}`best_fit` | best-fit decreasing (Ding et al., 2024): longest documents first, each into the open row where it leaves the least room. Almost no padding and almost no cuts | `true` |

The four documents from [Documents to rows](index.md#documents-to-rows), A of 5
tokens, B of 3, C of 10 and D of 2, in rows of 8 (`·` is padding):

```text
concat                     AAAAABBB  BCCCCCCC              C's last 3 tokens and D cut
greedy                     AAAAABBB  DD······              C, longer than a row, dropped
best_fit                   CCCDD···  CCCCCCCC  AAAAABBB    C split in two pieces, nothing cut
best_fit(long=truncate)    DD······  CCCCCCCC  AAAAABBB    C's last 2 tokens cut
```

Each `concat` row reads as one text, with positions 0 to 7. In the other packings,
positions start again at every document.

## Settings

| Field | `concat` | `greedy` | `best_fit` | What it is |
|-------|----------|----------|------------|------------|
| `doc_mask` | `false` | `true` | `true` | positions start again at every document, and attention stays within it ([below](#documents-stay-apart)) |
| `long` | | `drop` | `split` | what to do with a document longer than a row: `split` it into pieces that fill gaps, `truncate` it, or `drop` it |
| `max_docs` | | `none` | `none` | the most documents, or pieces of them, in one row |
| `sort` | | | `true` | longest first; `false` keeps the shuffled order (online best fit) |
| `chunk` | `16_000_000` | `16_000_000` | `16_000_000` | how many tokens of documents are packed together |

- `greedy` and `best_fit` keep documents whole. By default `greedy` drops a document
  longer than a row, because a cut conversation may lose its answer; `best_fit`
  splits it.
- `max_docs` caps the documents and pieces in a row, for attention kernels that take
  a fixed number of them.
- `best_fit(sort=false)` is online best fit, in the shuffled order. Its rows mix long
  and short documents, rather than grouping long documents together and short ones
  together.

## Cuts lose no targets

The rows of `concat`, and the pieces of a split document, overlap by one token. The
first copy of that token is a target. The second, at position 0, is not, because
nothing predicts the first token of a row or of a document. So every token is a
target exactly once, except the first token of each document.

## Documents stay apart

With `doc_mask`, positions start again at every document, and a new document starts
wherever positions stop increasing. RoPE and learned positional embeddings take the
positions as they are. From the positions, the model builds everything else once per
forward pass ({py:class}`~bux.architecture.common.segments.Segments`), and none of
it is t × t:

- **A document id per token.** These are the segment ids that splash attention and
  Pallas flash attention take instead of a t × t mask. Attention stays within a
  document, and the kernels skip blocks that no query sees.
- **How far each token is into its document.** Causal convolutions, n-grams and
  targets stop looking back there.
- **Where each query's first key is**, for kernels that take it (tokamax's). The
  `cu_seqlens` of varlen kernels are the points where the ids change.

The loss covers only targets in the same document as the position that predicts them
([Rows](index.md#rows)). Without `doc_mask`, a row reads as one text, as in GPT-2:
positions run from 0 to `seq_len - 1`, and every token attends to every token before
it in the row.

## Measurements

FineWeb-Edu, 53,248 documents (55M GPT-2 tokens), at `seq_len = 1024`:

| Packing | Rows | Filled | Tokens cut |
|---------|------|--------|------------|
| `concat` | 54,080 | 100% | 2,161 (the ends of chunks) |
| `best_fit` | 54,098 | 99.9% | 0 |
| `greedy` | 25,016 | 74.7% | 36M (the documents over 1024 tokens) |

`best_fit` fills rows as well as `concat` without cutting a single document.
`greedy` drops every document longer than a row, which here is most of the tokens.

You can compare the packings on your own data with a sweep. Here is 20,000 FineWeb
documents (13.9M GPT-2 tokens) in rows of 1024, with a hundredth held out, prepared
on a laptop by `packing = concat | greedy | best_fit`:

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

Each packing makes its own dataset, for the training split and the held-out split
alike. The figures on each line:

| Figure | What it counts |
|--------|----------------|
| filled | the share of the rows' tokens that are not padding |
| targets | the share that are targets: in the loss mask, and not the first token of a document |
| tokens cut | tokens of documents that no row holds: dropped, truncated, or left at the end of a chunk by `concat` |
