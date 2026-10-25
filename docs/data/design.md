# Design notes

Why the data pipeline works the way it does. Nothing here is needed to prepare or
read data; it explains the choices behind [The pipeline](pipeline.md) and
[Packing](packing.md).

## Offline, once

Packing well needs to see many documents at once. That is cheap in a MapReduce job,
and awkward in a training loop that must resume exactly. Done once, a dataset is
reused by every run of a sweep, and reading it costs nothing.

## Keys are hashes of content

A document's key is a hash of its content, not of where it is. So:

- the same documents land in the same chunks in the same order, whatever the files,
  the tokenizer or the engine;
- splits need no bookkeeping;
- the reduce's output depends on nothing but its chunk.

## Positions, not segment ids

Resetting positions at every document carries the segments and the positions RoPE
needs in one array. Everything else a kernel or a layer needs, document ids, how far
each token is into its document, where each query's first key is, the model makes
from them once per forward pass, none of it t × t
([Documents stay apart](packing.md#documents-stay-apart)).

## Packings considered

- **First-fit decreasing** (Axolotl's multipack, NeMo) and **shortest-pack-first
  histogram packing** (Krell et al., 2021) pack about as well as best fit, which is at
  least as good and faster (Ding et al., 2024). Order-preserving first fit is
  `best_fit(sort=false)`.
- **Balancing rows across data-parallel ranks by attention cost** (WLB-LLM, 2025)
  matters at long context, once there are ranks.
- **Dataset decomposition** (Pouransari et al., 2024) packs documents into rows of
  several lengths, for a curriculum over them; it needs a trainer that takes several
  shapes.
- **Packing similar documents together** (In-Context Pretraining, Shi et al., 2023;
  BM25 chunks, Zhao et al., 2024) needs attention across documents, and a retrieval
  step before the reduce.
