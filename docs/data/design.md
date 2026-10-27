# Design notes

Why the data pipeline works the way it does. You don't need any of this to prepare or
read data; it explains the choices behind [The pipeline](pipeline.md) and
[Packing](packing.md).

## Offline, once

Packing well means seeing many documents at once. That is cheap in a MapReduce job,
and awkward inside a training loop that must resume exactly where it stopped.
Prepared once, a dataset serves every run of a sweep, and reading it costs almost
nothing.

## Keys are hashes of content

A document's key is a hash of its content, not of its location. As a result:

- the same documents land in the same chunks, in the same order, whatever the files,
  the tokenizer or the engine;
- splits need no bookkeeping: a document's hash says which split it is in;
- each reduce's output depends only on its own chunk, so engines agree row for row.

## Positions, not segment ids

Restarting positions at every document puts both the segments and the positions RoPE
needs into one array. Everything else a kernel or a layer needs, the model derives
from them once per forward pass: document ids, how far each token is into its
document, and where each query's first key is. None of it is t × t
([Documents stay apart](packing.md#documents-stay-apart)).

## Packings considered

- **First-fit decreasing** (Axolotl's multipack, NeMo) and **shortest-pack-first
  histogram packing** (Krell et al., 2021) pack about as well as best fit. Best fit
  is at least as good and faster (Ding et al., 2024). For an order-preserving
  packing, bux offers online best fit, `best_fit(sort=false)`.
- **Balancing rows across data-parallel ranks by attention cost** (WLB-LLM, 2025)
  matters at long context, and only once there are several ranks.
- **Dataset decomposition** (Pouransari et al., 2024) packs documents into rows of
  several lengths, for a curriculum over them. It needs a trainer that takes several
  shapes.
- **Packing similar documents together** (In-Context Pretraining, Shi et al., 2023;
  BM25 chunks, Zhao et al., 2024) needs attention across documents, and a retrieval
  step before the reduce.
