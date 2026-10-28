"""Sparse attention over vLLM's paged caches in plain PyTorch: the keys each query of
a ``gqa`` layer with an indexer keeps, and its softmax over exactly those keys.

A cache is vLLM's logical view of a layer's pages, ``[blocks, heads, block_size,
channels]``: a request's token at position p is in block ``rows[p // block_size]``
of its block table, at ``p % block_size``. A sparse call keeps its indexer's keys in
the same rows as its keys and values (`Cache`). A query at position p sees the keys at 0
to p, the last ``sliding_window_size`` of them with a window. What it keeps is
bux's (`bux.export.hf.sparse`): the top-k of the keys it sees by its indexer's
scores, ranked by the floats' bits, the first of equal ones; with ``block_size``,
whole blocks counted from the request's first token, mean-pooled (QSA) or each
scored by its best key (MSA); with ``per_group``, a pick for each kv head's group
of query heads. Picks are key positions with a mask of the valid ones, ``(Q, [G,]
K)``, so that attending gathers those keys alone. A layer that reads another's picks
(the ``index`` port) attends to those it sees itself, by its own window, over its
own keys (`attend_picked`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

#: The most elements a chunk of queries makes in its largest intermediate.
BUDGET = 1 << 26
MIN = torch.finfo(torch.float32).min  # the logit of a masked key, as bux's
LOW = (1 << 31) - 1  # the bits a ranking key keeps for the key's index


def ordered(x: torch.Tensor) -> torch.Tensor:
    """Integers in [0, 2^32) in the order of the float32s `x` (so -0 ranks below
    +0), as bux's ``_ordered``."""
    u = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(u >> 31 == 1, ~u & 0xFFFFFFFF, u | (1 << 31))


def top(
    scores: torch.Tensor, valid: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The indices of each row's `k` highest `scores` among the `valid` ones, the
    first of equal ones (as bux's ``_top``), and which of them are valid: a row
    with fewer valid entries keeps them all."""
    index = torch.arange(scores.shape[-1], device=scores.device)
    return best_of(torch.where(valid, (ordered(scores) << 31) | (LOW - index), -1), k)


def best_of(key: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The indices of each row's `k` highest ranking keys, a score's `ordered` bits
    above its index's complement (`top`'s), -1 for an invalid entry; and which of
    them are valid."""
    best = key.topk(min(k, key.shape[-1]), dim=-1).values
    return LOW - (best & LOW), best >= 0


def seen(positions: torch.Tensor, n: int, window: int | None) -> torch.Tensor:
    """Which of the keys at positions 0 to n - 1 each query, at `positions` (Q,),
    sees: causal, and the last `window` of them with one. (Q, n)."""
    keys = torch.arange(n, device=positions.device)
    out = keys <= positions[:, None]
    if window is not None:
        out &= positions[:, None] - keys < window
    return out


def scores(
    q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor, relu: bool, per_group: bool
) -> torch.Tensor:
    """Each query's score of each key, ``sum_j w_j ReLU(q_j . k_s)`` in float32
    (the plain dot products without `relu`), (Q, S); each head's ``w_j ReLU(q_j .
    k_s)`` with `per_group`, (Q, G, S).

    Args:
        q: The indexer's queries, (Q, H, i).
        w: Their heads' weights, (Q, H).
        keys: The keys, (S, i) for all queries or (Q, S, i) for each.
        relu: Whether to ReLU the dot products.
        per_group: Whether each head scores for a group of its own.
    """
    spec = "qhi,si->qhs" if keys.ndim == 2 else "qhi,qsi->qhs"
    dots = torch.einsum(spec, q.float(), keys.float())
    if relu:
        dots = dots.relu()
    if per_group:
        return dots * w.float()[..., None]
    return torch.einsum("qhs,qh->qs", dots, w.float())


def tokens(
    s: torch.Tensor, see: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSA's picks: the `top_k` best-scored keys a query sees, (Q, [G,] K)."""
    return top(s, see if s.ndim == see.ndim else see[:, None], top_k)


def block_keys(
    blocks: torch.Tensor, ok: torch.Tensor, size: int, see: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys of the picked `blocks` of `size` keys, (Q, [G,] kb size), valid
    where their block is (`ok`) and the query sees them (`see`, (Q, S))."""
    keys = blocks[..., None] * size + torch.arange(size, device=blocks.device)
    keys, ok = keys.flatten(-2), ok[..., None].expand(*ok.shape, size).flatten(-2)
    n = see.shape[-1]
    inside = keys < n
    keys = keys.clamp(max=n - 1)
    view = see if keys.ndim == 2 else see[:, None].expand(-1, keys.shape[1], -1)
    return keys, ok & inside & view.gather(-1, keys)


def best_blocks(
    s: torch.Tensor, see: torch.Tensor, positions: torch.Tensor, size: int, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSA's picks: the ``top_k // size`` blocks of `size` keys that score best by
    the best of their keys a query sees, its own block always among them; their
    keys, (Q, [G,] K)."""
    n = s.shape[-1]
    count = -(-n // size)
    view = see if s.ndim == see.ndim else see[:, None]
    masked = s.masked_fill(~view, -torch.inf)
    masked = F.pad(masked, (0, count * size - n), value=-torch.inf)
    best = masked.unflatten(-1, (count, size)).amax(-1)
    return blocks_of_best(best, see, positions, size, top_k)


def blocks_of_best(
    best: torch.Tensor,
    see: torch.Tensor,
    positions: torch.Tensor,
    size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`best_blocks`' picks from each block's best score a query sees, (Q, [G,]
    blocks), -inf for a block it sees none of."""
    own = F.one_hot(positions // size, best.shape[-1]).bool()
    best = best.masked_fill(own if best.ndim == 2 else own[:, None], torch.inf)
    blocks, ok = top(best, best > -torch.inf, top_k // size)
    return block_keys(blocks, ok, size, see)


def full_blocks(
    positions: torch.Tensor, count: int, size: int, window: int | None
) -> torch.Tensor:
    """Which of the first `count` blocks of `size` keys each query sees all of."""
    first = torch.arange(count, device=positions.device) * size
    out = first + size - 1 <= positions[:, None]
    if window is not None:
        out &= positions[:, None] - first < window
    return out


def mean_blocks(
    s: torch.Tensor,
    see: torch.Tensor,
    positions: torch.Tensor,
    window: int | None,
    size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QSA's picks from the scores `s` of the complete blocks' pooled keys, (Q,
    blocks): the ``top_k // size`` best of the blocks a query sees all of, and the
    query's own block while it is unfinished; their keys, (Q, K)."""
    full = full_blocks(positions, s.shape[-1], size, window)
    blocks, ok = top(s, full, top_k // size)
    own = (positions // size)[:, None]
    finished = (own + 1) * size - 1 <= positions[:, None]
    if window is not None:
        finished &= positions[:, None] - own * size < window
    blocks, ok = torch.cat([blocks, own], -1), torch.cat([ok, ~finished], -1)
    return block_keys(blocks, ok, size, see)


def pooled(raw: torch.Tensor, size: int) -> torch.Tensor:
    """The mean of each complete block of `size` raw keys, (..., S // size, i),
    from (..., S, i), as bux pools them (before their norm and RoPE)."""
    count = raw.shape[-2] // size
    blocks = raw[..., : count * size, :].unflatten(-2, (count, size))
    return blocks.sum(-2) / size


def write(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Write `rows`, (T, heads, channels), into the cache at vLLM's `slots`, (T,),
    skipping padding (slot -1)."""
    keep = slots >= 0
    slots, rows = slots[keep], rows[keep]
    size = cache.shape[2]
    cache[slots // size, :, slots % size] = rows.to(cache.dtype)


def merged(kv: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The rows of tokens in a `Cache`, (T, g, C + e), from their keys and values,
    (T, g, C), and their indexer keys, (T, i)."""
    t, g, _ = kv.shape
    e = -(-index.shape[-1] // g)
    spread = F.pad(index, (0, g * e - index.shape[-1])).view(t, g, e)
    return torch.cat([kv, spread.to(kv.dtype)], -1)


@dataclass(frozen=True)
class Cache:
    """A sparse call's paged cache, ``[blocks, g, block_size, C + e]``: each kv
    head's key and value (or MLA's latent), C channels, then its share of the
    indexer key, e = ceil(i / g) channels, head j's the key's channels j e to
    (j + 1) e. The indexer's keys so page with the attention's, in blocks of the
    same tokens, as vLLM's prefix caching needs of every cache of a model.

    Attributes:
        cache: The pages.
        channels: C.
        dims: i; 0 for a call that keeps no indexer keys.
    """

    cache: torch.Tensor
    channels: int
    dims: int = 0

    @property
    def kv(self) -> torch.Tensor:
        """The keys and values, ``[blocks, g, block_size, C]``."""
        return self.cache[..., : self.channels]

    def keys(self, rows: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """The indexer keys at the positions `keys`, (Q, K), of the requests whose
        block tables are `rows`, (Q, blocks): (Q, K, i)."""
        parts = gather(self.cache[..., self.channels :], rows, keys)
        return parts.transpose(1, 2).flatten(2)[..., : self.dims]


def gather(cache: torch.Tensor, rows: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """The cache's rows at the key positions `keys`, of the requests whose block
    tables are `rows`, (Q, blocks): (Q, heads, K, channels) for keys (Q, K) of all
    heads, or each head's for keys (Q, heads, K)."""
    size = cache.shape[2]
    flat = keys.flatten(1)
    block = rows.gather(1, flat // size).view_as(keys).long()
    offset = keys % size
    if keys.ndim == 2:
        return cache[block, :, offset].transpose(1, 2)
    head = torch.arange(keys.shape[1], device=keys.device)[None, :, None]
    return cache[block, head, offset]


def attend(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid: torch.Tensor,
    dk: int,
    scale: float,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
    latent: int = 0,
    logits_of: Any = None,
) -> torch.Tensor:
    """Softmax attention of each query over its picked keys alone, in float32 as
    bux's ``softmax_attention``; zeros for a query that keeps none.

    Args:
        q: The queries, (Q, h, dk), each kv head's group of query heads together.
        kv: The picked keys and values, (Q, g, K, dk + dv), or (g, K, dk + dv)
            that all queries share.
        valid: Which picks are keys, (Q, g, K), or (Q, 1, K) for all heads.
        dk: The keys' dims, before the values' in `kv`.
        scale: What every logit is multiplied by.
        softcap: Logits go to ``softcap * tanh(logits / softcap)``; None for none.
        sinks: A logit per head beside the keys', on no value, (h,); None for none.
        latent: For MLA's latents, which are keys and values at once: the values
            are the keys' first `latent` dims, and `kv` holds the keys alone.
        logits_of: Changes the scaled logits, (Q, g, h / g, K), before soft-capping
            (a relative bias); None for none.

    Returns:
        The heads' outputs, (Q, h, dv), in the values' dtype.
    """
    n, h, _ = q.shape
    g, shared = kv.shape[-3], kv.ndim == 3
    k, v = kv[..., :dk], (kv[..., :latent] if latent else kv[..., dk:])
    groups = q.view(n, g, h // g, dk).float()
    spec = "qgrd,gkd->qgrk" if shared else "qgrd,qgkd->qgrk"
    logits = torch.einsum(spec, groups, k.float()) * scale
    if logits_of is not None:
        logits = logits_of(logits)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    keep = valid[:, :, None, :]
    logits = logits.masked_fill(~keep, MIN)
    if sinks is None:
        probs = logits.softmax(-1)
    else:
        sink = sinks.float().view(1, g, h // g, 1).expand(n, g, h // g, 1)
        probs = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
    probs = probs.masked_fill(~keep, 0)
    spec = "qgrk,gkd->qgrd" if shared else "qgrk,qgkd->qgrd"
    out = torch.einsum(spec, probs.to(v.dtype), v)
    return out.reshape(n, h, -1)


@dataclass
class Sparse:
    """What a sparse layer's call attends with, besides its tensors.

    Attributes:
        top_k: The keys each query keeps (in whole blocks, with blocks).
        block_size: The keys of a block; 1 for single keys.
        mean: Whether blocks score by their pooled key (QSA), else by their best
            key (MSA), with blocks.
        per_group: Whether each kv head's group of query heads picks apart.
        relu: Whether the indexer ReLUs its dot products.
        window: The layer's sliding window; None for none.
        scale: What every attention logit is multiplied by.
        softcap: The attention logits' soft cap; None for none.
        dk: The dims of a key head, before the values' in the cache.
        groups: The first of this rank's kv heads among the layer's, and how many.
        sinks: The rank's heads' sink logits; None for none.
        indexer: The indexer, which normalizes and rotates pooled block keys.
        latent: For MLA's latent cache, the dims of a latent, which are the values
            and the keys' first dims at once (`attend`); 0 for keys and values.
    """

    top_k: int
    block_size: int
    mean: bool
    per_group: bool
    relu: bool
    window: int | None
    scale: float
    softcap: float | None
    dk: int
    groups: tuple[int, int]
    sinks: torch.Tensor | None = None
    indexer: Any = None
    latent: int = 0


def absorbed(q: torch.Tensor, kv_b: torch.Tensor, n: int) -> torch.Tensor:
    """MLA's queries in the latent space, (T, h, kv_lora_rank + r) in float32, from
    its queries (T, h, n + r): each head's first `n` dims through its key
    projection, the rows of ``kv_b`` (h (n + dv), kv_lora_rank) that make its keys,
    then its rotated dims as they are."""
    w = _heads_of(kv_b, q.shape[1])
    nope = torch.einsum("thn,hnr->thr", q[..., :n].float(), w[:, :n])
    return torch.cat([nope, q[..., n:].float()], -1)


def projected(out: torch.Tensor, kv_b: torch.Tensor, n: int) -> torch.Tensor:
    """MLA's heads' outputs, (T, h, dv) in float32, from the latents they weigh,
    (T, h, kv_lora_rank): each head's value projection, the rows of ``kv_b`` after
    its first `n`."""
    w = _heads_of(kv_b, out.shape[1])
    return torch.einsum("thr,hvr->thv", out.float(), w[:, n:])


def _heads_of(kv_b: torch.Tensor, heads: int) -> torch.Tensor:
    """``kv_b``'s weight by head, (h, n + dv, kv_lora_rank), in float32."""
    return kv_b.float().view(heads, -1, kv_b.shape[-1])


def layer_spec(
    ix: dict | None,
    indexer: Any,
    attention: tuple[int | None, float, float | None, torch.Tensor | None],
    dk: int,
    groups: tuple[int, int],
    latent: int = 0,
) -> Sparse:
    """A sparse layer's `Sparse` settings: its indexer's config `ix` and module
    (None for a layer that reads its picks from its ``index`` port), and its
    `attention`'s window, scale, softcap and sinks."""
    window, scale, softcap, sinks = attention
    return Sparse(
        ix["top_k"] if ix else 0,
        ix["block_size"] if ix else 1,
        bool(indexer is not None and indexer.mean),
        bool(ix and ix["per_group"]),
        bool(ix and ix["relu"]),
        window,
        scale,
        softcap,
        dk,
        groups,
        sinks,
        indexer,
        latent,
    )


def choose(
    spec: Sparse,
    q: torch.Tensor,
    w: torch.Tensor,
    keys: torch.Tensor,
    see: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys the queries keep, (Q, [G,] K), and which picks are valid, from the
    indexer's queries `q`, weights `w` and cached `keys`, (S, i) or (Q, S, i)."""
    size = spec.block_size
    if spec.mean:
        # In float32 from the cache's keys on, as bux's reference pools them.
        pooled_keys = spec.indexer.block_keys(pooled(keys.float(), size), size)
        s = scores(q, w, pooled_keys, spec.relu, per_group=False)
        return mean_blocks(s, see, positions, spec.window, size, spec.top_k)
    s = scores(q, w, keys, spec.relu, spec.per_group)
    if spec.per_group:
        first, n = spec.groups
        s = s[:, first : first + n]
    if size > 1:
        return best_blocks(s, see, positions, size, spec.top_k)
    return tokens(s, see, spec.top_k)


def relative(
    r: torch.Tensor,
    table: torch.Tensor,
    gain: torch.Tensor | None,
    positions: torch.Tensor,
) -> Any:
    """A relative bias on the logits of queries at `positions`, (Q,), over keys at
    0 to K - 1, as `attend` takes it: the bias of a query for the key j positions
    behind it, ``<r_h, table[:, j]>`` for j below the table's extent, none farther,
    then the logits times `gain`, (Q,), as bux's length scale; both in float32.

    Args:
        r: Each query head's projection, (Q, h, rank).
        table: The bias table, (rank, extent).
        gain: Each query's length scale; None for none.
        positions: The queries' positions.
    """
    each = torch.einsum("qhr,re->qhe", r.float(), table.float())

    def change(logits: torch.Tensor) -> torch.Tensor:
        n, g, per, k = logits.shape
        offsets = positions[:, None] - torch.arange(k, device=positions.device)
        near = (offsets >= 0) & (offsets < table.shape[1])
        index = offsets.clamp(0, table.shape[1] - 1)[:, None].expand(-1, g * per, -1)
        bias = each.gather(-1, index) * near[:, None]
        out = logits + bias.view(n, g, per, k)
        return out if gain is None else out * gain.float()[:, None, None, None]

    return change


def attend_all(
    spec: Sparse,
    q: torch.Tensor,
    bias: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    positions: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """A segment's outputs, (Q, h, dv), over every key its queries see, with a
    relative bias and a length scale (`relative`), a chunk of queries at a time.

    Args:
        spec: The layer's settings; its window, scale, softcap, dims and sinks.
        q: The queries, (Q, h, dk).
        bias: Each query head's projection, (Q, h, rank), the bias table, and each
            query's length scale, (Q,), or None.
        positions: The queries' positions, (Q,).
        cache: The layer's cache of keys and values.
        rows: Its block tables' rows of the queries' requests, (Q, blocks), or
            (1, blocks) for one request's.
        n: How many key positions the requests have, at least.
    """
    keys = torch.arange(n, device=q.device)
    shared = rows.shape[0] == 1
    kv = gather(cache, rows, keys[None])[0] if shared else None
    per = q.shape[1] * n * (1 if shared else cache.shape[3])
    chunk, (r, table, gain), parts = max(1, BUDGET // per), bias, []
    for a in range(0, q.shape[0], chunk):
        cut = slice(a, a + chunk)
        pos = positions[cut]
        mine = (
            kv if kv is not None else gather(cache, rows[cut], keys.expand(len(pos), n))
        )
        change = relative(r[cut], table, None if gain is None else gain[cut], pos)
        see = seen(pos, n, spec.window)[:, None]
        args = (spec.dk, spec.scale, spec.softcap, spec.sinks)
        parts.append(attend(q[cut], mine, see, *args, logits_of=change))
    return torch.cat(parts)


@dataclass
class Segment:
    """Queries that attend together: one request's, which share its keys, or
    single queries of several requests.

    Attributes:
        tokens: The queries' indices in the step's tokens, (Q,).
        requests: Their requests' indices in the batch, (Q,) or (1,) if shared.
        keys: How many key positions of its requests to read, at least.
    """

    tokens: torch.Tensor
    requests: torch.Tensor
    keys: int


def segments(starts: list[int], bounds: list[int], device: Any) -> list[Segment]:
    """The step's queries by `Segment`: those of the requests with one query
    together, then each other request's; `starts` are the requests' first queries
    and the end, and `bounds` the most keys each request has."""
    lens = [b - a for a, b in itertools.pairwise(starts)]
    single = [r for r, n in enumerate(lens) if n == 1]
    out = []
    if single:
        at = torch.tensor([starts[r] for r in single], device=device)
        ids = torch.tensor(single, device=device)
        out.append(Segment(at, ids, max(bounds[r] for r in single)))
    for r, n in enumerate(lens):
        if n > 1:
            at = torch.arange(starts[r], starts[r] + n, device=device)
            out.append(Segment(at, torch.tensor([r], device=device), bounds[r]))
    return out


#: The keys each query keeps, (Q, [G,] K), and which of them are valid.
Picked = tuple[torch.Tensor, torch.Tensor]


def most(spec: Sparse) -> int:
    """The most keys a query keeps: the top-k, in whole blocks with blocks, and its
    own block beside them in QSA's."""
    size = spec.block_size
    return spec.top_k if size == 1 else (spec.top_k // size + spec.mean) * size


def padded(picked: Picked, k: int) -> Picked:
    """`picked` padded to `k` keys a query, the padding invalid."""
    keys, valid = picked
    pad = keys.new_zeros((*keys.shape[:-1], k - keys.shape[-1]))
    return torch.cat([keys, pad], -1), torch.cat([valid, pad.bool()], -1)


def attend_segment(
    spec: Sparse,
    q: torch.Tensor,
    index: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    cache: Cache,
    rows: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, Picked]:
    """A segment's outputs, (Q, h, dv), a chunk of queries at a time, and its picks
    padded to `most` keys a query, which the ``index`` port passes on.

    Args:
        spec: The layer's settings.
        q: The queries, (Q, h, dk).
        index: The indexer's queries and weights, (Q, H, i) and (Q, H).
        positions: The queries' positions, (Q,).
        cache: The call's cache of keys, values and indexer keys.
        rows: Its block table's rows of the queries' requests, (Q, blocks), or
            (1, blocks) for one request's.
        n: How many key positions to read.
    """
    shared = rows.shape[0] == 1
    chunk = max(1, BUDGET // _per_query(spec, q, index[0], cache.kv, n, shared))
    parts, picks = [], []
    for a in range(0, q.shape[0], chunk):
        cut = slice(a, a + chunk)
        part = rows if shared else rows[cut]
        tensors = (q[cut], index[0][cut], index[1][cut], positions[cut])
        out, picked = _attend_chunk(spec, *tensors, cache, part, n)
        parts.append(out)
        picks.append(padded(picked, most(spec)))
    keys, valid = zip(*picks, strict=True)
    return torch.cat(parts), (torch.cat(keys), torch.cat(valid))


def attend_picked(
    spec: Sparse,
    q: torch.Tensor,
    picked: Picked,
    positions: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """A segment's outputs, (Q, h, dv), over the keys another call's indexer
    `picked` for its queries (the ``index`` port), of those they see by the layer's
    own window, a chunk of queries at a time.

    Args:
        spec: The layer's settings; its window, scale, softcap, dims and sinks.
        q: The queries, (Q, h, dk).
        picked: The keys the other call kept for each query, (Q, [G,] K).
        positions: The queries' positions, (Q,).
        cache: The layer's own cache of keys and values.
        rows: Its block tables' rows of the queries' requests, (Q, blocks), or
            (1, blocks) for one request's.
        n: How many key positions the requests have, at least.
    """
    keys, valid = picked
    per = cache.shape[1] * keys.shape[-1] * max(cache.shape[3], q.shape[1])
    chunk = max(1, BUDGET // per)
    parts = []
    for a in range(0, q.shape[0], chunk):
        cut = slice(a, a + chunk)
        part = rows if rows.shape[0] == 1 else rows[cut]
        mine = (keys[cut], valid[cut])
        parts.append(_attend_picked(spec, q[cut], mine, positions[cut], cache, part, n))
    return torch.cat(parts)


def _attend_picked(
    spec: Sparse,
    q: torch.Tensor,
    picked: Picked,
    positions: torch.Tensor,
    cache: torch.Tensor,
    rows: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """A chunk's outputs over the picked keys it sees."""
    keys, valid = picked[0].clamp(0, n - 1), picked[1]
    see = seen(positions, n, spec.window)
    view = see if keys.ndim == 2 else see[:, None].expand(-1, keys.shape[1], -1)
    valid = valid & view.gather(-1, keys)
    kv = gather(cache, rows.expand(q.shape[0], -1), keys)
    valid = valid[:, None] if valid.ndim == 2 else valid
    return attend(
        q, kv, valid, spec.dk, spec.scale, spec.softcap, spec.sinks, spec.latent
    )


def _per_query(
    spec: Sparse,
    q: torch.Tensor,
    iq: torch.Tensor,
    cache: torch.Tensor,
    n: int,
    shared: bool,
) -> int:
    """The elements of a query's largest intermediate: its scores, its keys if not
    shared, or its picked keys and values."""
    picks = spec.top_k + spec.block_size
    attention = cache.shape[1] * picks * max(cache.shape[3], q.shape[1])
    return max(iq.shape[1] * n, 0 if shared else n * iq.shape[2], attention)


def _attend_chunk(
    spec: Sparse,
    q: torch.Tensor,
    iq: torch.Tensor,
    iw: torch.Tensor,
    positions: torch.Tensor,
    cache: Cache,
    rows: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, Picked]:
    """A chunk's outputs: its picks, then its attention over them; and the picks."""
    span = torch.arange(n, device=q.device)
    if rows.shape[0] == 1:
        keys = cache.keys(rows, span[None])[0]
    else:
        keys = cache.keys(rows, span.expand(q.shape[0], n))
    see = seen(positions, n, spec.window)
    picks, valid = choose(spec, iq, iw, keys, see, positions)
    kv = gather(cache.kv, rows.expand(q.shape[0], -1), picks)
    each = valid[:, None] if valid.ndim == 2 else valid
    out = attend(
        q, kv, each, spec.dk, spec.scale, spec.softcap, spec.sinks, spec.latent
    )
    return out, (picks, valid)
