"""DeepSeek sparse attention's lightning indexer, exported: bux's ``Indexer`` as
``from_stream`` makes it, for keys that are the tokens.

It picks the keys each query of a sparse attention layer attends to: its top-k by
``sum_j w_j ReLU(q_j . k_s)``, of the keys its mask lets it see, ranked as bux ranks
them (by the floats' bits, the first of equal ones). Its keys are kept in the layer's
cache, an ``indexed_attention`` layer (``cache.update_indexer``), so that a decoding
step ranks every key before it. With ``block_size > 1`` it keeps whole blocks of keys
(Qwen3.8 QSA), from the keys as projected, which it pools again at each step. What it
keeps is a mask, b t s, over the keys the layer's cache returns: what the ``index``
port carries. Its loss is training's alone, and is left out.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import Ctx, Norm
from .rope import Rope

#: DeepSeek-V3.2's indexer keys, projected from the layer's input, are layer-normalized.
STREAM_KEY_NORM = {
    "type": "layernorm",
    "eps": 1e-06,
    "affine": True,
    "bias": True,
    "zero_centered": False,
    "scale_input": False,
}


def ordered(x: torch.Tensor) -> torch.Tensor:
    """Integers in the order of the float32s `x`, as bux's ``_ordered`` (so -0 ranks
    below +0)."""
    u = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(u >> 31 == 1, ~u & 0xFFFFFFFF, u | (1 << 31))


def top(scores: torch.Tensor, seen: torch.Tensor, k: int) -> torch.Tensor:
    """Return which of the last axis' entries each row keeps: its k highest scores of
    those `seen`, the first of equal ones, as bux's ``_top``."""
    k = min(k, scores.shape[-1])
    key = torch.where(seen, ordered(scores), -1)
    best = key.sort(dim=-1, descending=True, stable=True).indices[..., :k]
    return torch.zeros_like(seen).scatter(-1, best, True) & seen


def per_block(x: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Return the sums of x, b t s, over each block's slots, at the block's index;
    `block`, b s, says which block each slot is in."""
    index = block[:, None].expand_as(x)
    return torch.zeros_like(x).scatter_add(-1, index, x)


def key_positions(ctx: Ctx, s: int) -> torch.Tensor:
    """Return each of the s keys' position, b s, -1 for padding: the keys of a cache
    layer that keeps all of them, the row's tokens."""
    pad = ctx.attention_mask
    if pad is None or pad.ndim != 2:
        b = ctx.tokens.shape[0]
        return torch.arange(s, device=ctx.tokens.device).expand(b, s)
    pad = pad[:, :s].long()
    return torch.where(pad == 1, pad.cumsum(-1) - 1, -1)


class Indexer(nn.Module):
    """bux's ``Indexer`` of a layer whose keys are the tokens (``from_stream``).

    Args:
        c: The indexer's config (``SparseIndexerConfig``), as JSON.
        width: The width of the layer's input, which the keys and the head weights
            come from.
        query_width: The width of what the queries come from: the input, or MLA's
            query latent.
        rope: The layer's RoPE config, or None.
        rope_dims: How many of the layer's rotated dims; the indexer rotates as many
            of its own first dims, at most its head dim, in its ``rope_layout``.
    """

    def __init__(
        self, c: dict, width: int, query_width: int, rope: dict | None, rope_dims: int
    ):
        super().__init__()
        h, i = c["num_heads"], c["head_dim"]
        self.heads, self.head_dim = h, i
        self.top_k, self.block_size = c["top_k"], c["block_size"]
        self.q = nn.Linear(query_width, h * i, bias=False)
        self.k = nn.Linear(width, i, bias=False)
        self.weights = nn.Linear(width, h, bias=False) if c["weighted"] else None
        self.q_norm = Norm(c["norm"], i) if c["norm"] else None
        self.k_norm = Norm(c["norm"] or STREAM_KEY_NORM, i)
        n = min(rope_dims, i) // 2 * 2
        own = (
            None
            if rope is None
            else {**rope, "fraction": 1.0, "layout": c["rope_layout"]}
        )
        self.rope = Rope(own, n) if own is not None and n else None

    def forward(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        ctx: Ctx,
        slot: int | None,
        seen: torch.Tensor,
    ) -> torch.Tensor:
        """Return which keys each query keeps, b t s.

        Args:
            x: The layer's input, b t d.
            query: What the queries come from.
            ctx: The pass, whose cache keeps the indexer's keys at `slot`.
            slot: The layer's cache layer; None without a cache.
            seen: Which keys each query may attend to, b t s.
        """
        keys = self.keys(x, ctx.positions)
        if ctx.cache is not None and slot is not None:
            keys = ctx.cache.update_indexer(keys, slot)
        q, w = self._queries(query, ctx.positions), self._head_weights(x)
        if self.block_size > 1:
            return self._blocks(q, w, keys, seen, key_positions(ctx, keys.shape[1]))
        return top(self._scores(q, w, keys), seen, self.top_k)

    def keys(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys, b s i: as projected if it keeps blocks, else normalized and
        rotated."""
        k = self.k(x)
        if self.block_size > 1:
            return k
        return self._rotate(self.k_norm(k)[:, :, None], positions)[:, :, 0]

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
        """Every query's score of every key, ``sum_j w_j ReLU(q_j . k_s)``, b t s."""
        dots = torch.einsum("bthi,bsi->bths", q, keys)
        return torch.einsum("bths,bth->bts", dots.relu(), w)

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
        size, exists = self.block_size, order >= 0
        starts = exists & (order % size == 0)
        block = (starts.long().cumsum(-1) - 1).clamp(min=0)
        scores = self._scores(q, w, self._block_keys(keys, order, block, starts))
        full = per_block(seen.long(), block) == size
        picked = top(scores, full, self.top_k // size)
        last = torch.where(seen, torch.arange(seen.shape[-1], device=seen.device), -1)
        own = block.gather(1, last.argmax(-1).clamp(min=0))  # b t
        unfinished = ~full.gather(-1, own[..., None]) & seen.any(-1, keepdim=True)
        picked = picked | (
            torch.zeros_like(picked).scatter(-1, own[..., None], True) & unfinished
        )
        return picked.gather(-1, block[:, None].expand_as(picked))

    def _block_keys(
        self, keys: torch.Tensor, order: torch.Tensor, block: torch.Tensor, starts: Any
    ) -> torch.Tensor:
        """Each block's key, at its index: the mean of its block_size keys, normalized,
        then rotated to the position of its first key."""
        b, s, i = keys.shape
        index = block[..., None].expand(b, s, i)
        kept = keys * (order >= 0)[..., None].to(keys.dtype)
        sums = torch.zeros_like(keys).scatter_add(1, index, kept)
        means = self.k_norm(sums / self.block_size)
        first = torch.zeros_like(order).scatter_add(
            1, block, torch.where(starts, order, 0)
        )
        return self._rotate(means[:, :, None], first)[:, :, 0]
