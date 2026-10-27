"""DeepSeek sparse attention's lightning indexer, exported: bux's ``Indexer`` as
``from_stream`` makes it, for keys that are the tokens.

It picks the keys each query of a sparse attention layer attends to: its top-k by
``sum_j w_j ReLU(q_j . k_s)``, of the keys its mask lets it see, ranked as bux ranks
them (by the floats' bits, the first of equal ones). Its keys are kept in the layer's
cache, an ``indexed_attention`` layer (``cache.update_indexer``), so that a decoding
step ranks every key before it. With ``block_size > 1`` it keeps whole blocks of keys:
mean-pooled (Qwen3.8 QSA), from the keys as projected, which it pools again at each
step; or each scored by its best key (MiniMax MSA), from the keys as for single ones.
What it keeps is a mask, b t s, or b t g s for each kv head's group of query heads
with ``per_group``, over the keys the layer's cache returns: what the ``index`` port
carries. Its loss is training's alone, and is left out.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import Ctx, Norm, norm_config
from .rope import Rope

#: DeepSeek-V3.2's indexer keys, projected from the layer's input, are layer-normalized.
STREAM_KEY_NORM = norm_config("layernorm", bias=True)


def ordered(x: torch.Tensor) -> torch.Tensor:
    """Integers in the order of the float32s `x`, as bux's ``_ordered`` (so -0 ranks
    below +0)."""
    u = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(u >> 31 == 1, ~u & 0xFFFFFFFF, u | (1 << 31))


def top(scores: torch.Tensor, seen: torch.Tensor, k: int) -> torch.Tensor:
    """Return which of the last axis' entries each row keeps: its k highest scores of
    those `seen` (which broadcasts to the scores), the first of equal ones, as bux's
    ``_top``."""
    k = min(k, scores.shape[-1])
    key = torch.where(seen, ordered(scores), -1)
    best = key.sort(dim=-1, descending=True, stable=True).indices[..., :k]
    return torch.zeros_like(key, dtype=torch.bool).scatter(-1, best, True) & seen


def along(block: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return `block`, b s, which block each slot is in, expanded to x, b t s or
    b t g s."""
    b, s = block.shape
    return block.view(b, *[1] * (x.ndim - 2), s).expand_as(x)


def per_block(x: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Return the sums of x, b t s, over each block's slots, at the block's index;
    `block`, b s, says which block each slot is in."""
    return torch.zeros_like(x).scatter_add(-1, along(block, x), x)


def key_positions(ctx: Ctx, s: int) -> torch.Tensor:
    """Return each of the s keys' position, b s, -1 for padding: the keys of a cache
    layer that keeps all of them, the row's tokens."""
    pad = ctx.attention_mask
    if pad is None or pad.ndim != 2:
        b = ctx.tokens.shape[0]
        return torch.arange(s, device=ctx.tokens.device).expand(b, s)
    pad = pad[:, :s].long()
    return torch.where(pad == 1, pad.cumsum(-1) - 1, -1)


def groups(order: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return which block of `size` keys each slot is in, b s, counted along the row
    from a document's first key, and which slots start one; `order` is which key of
    its document each slot holds, -1 for none (bux's ``_groups``)."""
    starts = (order >= 0) & (order % size == 0)
    return (starts.long().cumsum(-1) - 1).clamp(min=0), starts


def per_block_max(x: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Return the maxima of x, b t s or b t g s, over each block's slots, at the
    block's index; -inf for a block with none."""
    out = torch.full_like(x, -torch.inf)
    return out.scatter_reduce(-1, along(block, x), x, reduce="amax", include_self=True)


def by_slot(blocks: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Return, for each slot, b t s or b t g s, whether its block is among `blocks`,
    by the blocks' index."""
    return blocks.gather(-1, along(block, blocks))


def newest(seen: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Return, by the blocks' index, b t s, the block of the newest key each query
    sees: none for a query that sees none (bux's ``_newest``)."""
    s = seen.shape[-1]
    last = torch.where(seen, torch.arange(s, device=seen.device), -1).argmax(-1)
    own = torch.zeros_like(seen).scatter(-1, block.gather(1, last)[..., None], True)
    return own & seen.any(-1, keepdim=True)


class Indexer(nn.Module):
    """bux's ``Indexer``: for keys that are the tokens (``from_stream``, the default),
    or keys the layer gives it (CSA's compressed latents).

    Without ``relu`` its dot products are plain. With ``per_group`` each of its heads
    picks for its kv head's group of the layer's query heads, by that head's scores
    alone (MiniMax MSA): its picks have a group axis, b t g s.

    Args:
        c: The indexer's config (``SparseIndexerConfig``), as JSON.
        width: The width of the layer's input, which the head weights come from.
        query_width: The width of what the queries come from: the input, or a query
            latent.
        rope: The layer's RoPE config, or None.
        rope_dims: How many of the layer's rotated dims; the indexer rotates as many
            of its own first dims, at most its head dim.
        key_width: The width of what its keys are projected from; None when they
            come projected (another layer's).
        key_norm: The keys' norm, unless ``c["norm"]`` sets one.
        layout: How it pairs the rotated dims; None for ``c["rope_layout"]``, as for
            keys that are the tokens.

    Raises:
        NotImplementedError: Max-pooled blocks of keys another layer gives it, which
            bux does not build either.
    """

    def __init__(
        self,
        c: dict,
        width: int,
        query_width: int,
        rope: dict | None,
        rope_dims: int,
        key_width: int | None = -1,
        key_norm: dict = STREAM_KEY_NORM,
        layout: str | None = None,
    ):
        super().__init__()
        h, i = c["num_heads"], c["head_dim"]
        self.heads, self.head_dim = h, i
        self.top_k, self.block_size = c["top_k"], c["block_size"]
        self.max_pool = self.block_size > 1 and c["pool"] == "max"
        self.per_group, self.relu = c["per_group"], c["relu"]
        if self.max_pool and key_width != -1:
            raise NotImplementedError(
                "max-pooled blocks (indexer(pool=max)) of another layer's keys"
            )
        self.pool_size, self.pool_top_k = (
            c["candidate_block_size"],
            c["candidate_top_k"],
        )
        key_width = width if key_width == -1 else key_width
        self.q = nn.Linear(query_width, h * i, bias=False)
        self.k = nn.Linear(key_width, i, bias=False) if key_width else None
        self.weights = nn.Linear(width, h, bias=False) if c["weighted"] else None
        self.q_norm = Norm(c["norm"], i) if c["norm"] else None
        has_norm = key_width or self._mean_blocks()
        self.k_norm = Norm(c["norm"] or key_norm, i) if has_norm else None
        n = min(rope_dims, i) // 2 * 2
        own = None
        if rope is not None:
            own = {**rope, "fraction": 1.0, "layout": layout or c["rope_layout"]}
        self.rope = Rope(own, n) if own is not None and n else None

    def forward(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        positions: torch.Tensor,
        keys: torch.Tensor,
        seen: torch.Tensor,
        order: torch.Tensor,
        inside: torch.Tensor | None = None,
        pool: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return which keys each query keeps, b t s (b t g s with ``per_group``),
        and its candidate pool if asked.

        Args:
            x: The layer's input, b t d.
            query: What the queries come from.
            positions: The queries' positions, b t.
            keys: The keys, b s i, from `keys` (all of them the cache keeps).
            seen: Which keys each query may attend to, b t s.
            order: Which key of its document each slot holds, b s; -1 for none.
            inside: A candidate pool, b t s, which the kept keys must lie in; None
                for none.
            pool: Whether to return a candidate pool of its own: the best
                candidate_top_k blocks of candidate_block_size keys, each scored by
                its best key, and the block of the newest key a query sees.
        """
        q, w = self._queries(query, positions), self._head_weights(x)
        if self.max_pool:
            return self._max_blocks(q, w, keys, seen, order), None
        if self.block_size > 1:
            return self._blocks(q, w, keys, seen, order), None
        seen = seen if inside is None else seen & inside
        scores = self._scores(q, w, keys)
        kept = top(scores, self._like(seen, scores), self.top_k)
        if not pool:
            return kept, None
        return kept, self._pool(scores.masked_fill(~seen, -torch.inf), seen, order)

    def keys(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Project its keys, b s i, from x: as projected if it mean-pools blocks of
        them, else normalized and rotated by the positions."""
        assert self.k is not None and self.k_norm is not None
        k = self.k(x)
        if self._mean_blocks():
            return k
        return self._rotate(self.k_norm(k)[:, :, None], positions)[:, :, 0]

    def _mean_blocks(self) -> bool:
        """Whether it pools blocks of keys by their mean (Qwen3.8 QSA)."""
        return self.block_size > 1 and not self.max_pool

    def _like(self, seen: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """`seen`, b t s, with a group axis of 1 if `scores` have one."""
        return seen if scores.ndim == seen.ndim else seen[:, :, None]

    def _pool(
        self, ranked: torch.Tensor, seen: torch.Tensor, order: torch.Tensor
    ) -> torch.Tensor:
        """The candidate pool, b t s: bux's ``_candidates``, as a mask of slots."""
        block, _ = groups(order, self.pool_size)
        best = per_block_max(ranked, block).masked_fill(newest(seen, block), torch.inf)
        return by_slot(top(best, best > -torch.inf, self.pool_top_k), block)

    def _queries(self, query: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Project, normalize and rotate the query heads, b t h i."""
        b, t, _ = query.shape
        q = self.q(query).view(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        return self._rotate(q, positions)

    def _head_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Each head's weight in a query's scores, with the head dims' scale, b t h."""
        scale = self.head_dim**-0.5
        if self.weights is None:
            return torch.full(
                (*x.shape[:2], self.heads), scale, dtype=x.dtype, device=x.device
            )
        return self.weights(x) * (scale * self.heads**-0.5)

    def _scores(
        self, q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """Every query's score of every key, ``sum_j w_j ReLU(q_j . k_s)``, b t s;
        without ``relu`` of the plain dot products; with ``per_group`` each head's
        ``w_j ReLU(q_j . k_s)``, b t g s."""
        dots = torch.einsum("bthi,bsi->bths", q, keys)
        if self.relu:
            dots = dots.relu()
        if self.per_group:
            return dots * w[..., None]
        return torch.einsum("bths,bth->bts", dots, w)

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the first dims of x, b t h i, by the positions."""
        return x if self.rope is None else self.rope(x, positions)

    def _blocks(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        keys: torch.Tensor,
        seen: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        """Keep whole blocks of block_size keys: the top_k // block_size best of those
        a query sees all of, and the keys of its own block until it is complete.
        `order` is each key's position, -1 for none."""
        size = self.block_size
        block, starts = groups(order, size)
        scores = self._scores(q, w, self._block_keys(keys, order, block, starts))
        full = per_block(seen.long(), block) == size
        picked = top(scores, full, self.top_k // size)
        return by_slot(picked | (newest(seen, block) & ~full), block)

    def _max_blocks(
        self,
        q: torch.Tensor,
        w: torch.Tensor,
        keys: torch.Tensor,
        seen: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        """Keep whole blocks of block_size keys, each scored by the best of its keys
        a query sees: its own block and the best others, top_k // block_size in all,
        for each group with ``per_group`` (MiniMax MSA)."""
        block, _ = groups(order, self.block_size)
        scores = self._scores(q, w, keys)
        unseen = ~self._like(seen, scores)
        best = per_block_max(scores.masked_fill(unseen, -torch.inf), block)
        best = best.masked_fill(self._like(newest(seen, block), best), torch.inf)
        kept = top(best, best > -torch.inf, self.top_k // self.block_size)
        return by_slot(kept, block)

    def _block_keys(
        self, keys: torch.Tensor, order: torch.Tensor, block: torch.Tensor, starts: Any
    ) -> torch.Tensor:
        """Each block's key, at its index: the mean of its block_size keys, normalized,
        then rotated to the position of its first key."""
        assert self.k_norm is not None
        b, s, i = keys.shape
        index = block[..., None].expand(b, s, i)
        kept = keys * (order >= 0)[..., None].to(keys.dtype)
        sums = torch.zeros_like(keys).scatter_add(1, index, kept)
        means = self.k_norm(sums / self.block_size)
        first = torch.zeros_like(order).scatter_add(
            1, block, torch.where(starts, order, 0)
        )
        return self._rotate(means[:, :, None], first)[:, :, 0]
