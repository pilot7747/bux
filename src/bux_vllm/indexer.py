"""The lightning indexer of a ``gqa`` or ``mla`` layer in plain PyTorch, as bux's
``Indexer`` makes it from the layer's input (``from_stream``): its queries (from an
``mla`` layer's query latent where it has one, as DeepSeek-V3.2's), their heads'
weights, and the keys it caches, one per token.

The keys are normalized and rotated as they are cached, but where the indexer
mean-pools blocks of them (Qwen3.8 QSA): then the cache keeps them as projected, and
each step pools a block's, normalizes the mean and rotates it to the block's first
position (`block_keys`). The indexer's weights are the same on every rank under
tensor parallelism, so every rank picks the same keys.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import Norm, Rope

#: DeepSeek-V3.2's indexer keys, projected from the layer's input, are
#: layer-normalized (bux's ``_STREAM_KEY_NORM``).
STREAM_KEY_NORM = {
    "type": "layernorm",
    "eps": 1e-6,
    "affine": True,
    "bias": True,
    "zero_centered": False,
    "scale_input": False,
}


def rotated(head_dim: int, rope: dict | None) -> int:
    """How many of a head's dims the layer's `rope` rotates, in whole pairs."""
    return 0 if rope is None else int(head_dim * rope["fraction"]) // 2 * 2


class Indexer(nn.Module):
    """bux's indexer of a ``gqa`` or ``mla`` layer: `num_heads` query heads and one
    key of `head_dim` per token, from the layer's input.

    Args:
        c: The indexer's config (``SparseIndexerConfig``), as JSON.
        width: The width of the layer's input.
        rope: The layer's RoPE config, or None.
        rope_dims: How many of the layer's head dims it rotates; the indexer rotates
            as many of its own first dims, at most its head dim.
        query_width: The width of what its queries come from: an ``mla`` layer's
            query latent; None for the input.
    """

    def __init__(
        self,
        c: dict,
        width: int,
        rope: dict | None,
        rope_dims: int,
        query_width: int | None = None,
    ):
        super().__init__()
        h, i = c["num_heads"], c["head_dim"]
        self.heads, self.head_dim = h, i
        self.mean = c["block_size"] > 1 and c["pool"] == "mean"
        self.q = nn.Linear(query_width or width, h * i, bias=False)
        self.k = nn.Linear(width, i, bias=False)
        self.weights = nn.Linear(width, h, bias=False) if c["weighted"] else None
        self.q_norm = Norm(c["norm"], i) if c["norm"] else None
        self.k_norm = Norm(c["norm"] or STREAM_KEY_NORM, i)
        n = min(rope_dims, i) // 2 * 2
        own = {**rope, "fraction": 1.0, "layout": c["rope_layout"]} if rope else None
        self.rope = Rope(own, n) if own is not None and n else None

    def queries(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The query heads of the tokens x (or their query latents), (T, H, i):
        projected, normalized and rotated by their positions."""
        q = self.q(x).view(x.shape[0], self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        return self._rotate(q, positions)

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Each head's weight in a query's scores, with the head dims' scale, (T, H)."""
        scale = self.head_dim**-0.5
        if self.weights is None:
            return x.new_full((x.shape[0], self.heads), scale)
        return self.weights(x) * (scale * self.heads**-0.5)

    def keys(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys the cache keeps of the tokens x, (T, i): as projected if the
        indexer mean-pools blocks of them, else normalized and rotated."""
        k = self.k(x)
        if self.mean:
            return k
        return self._rotate(self.k_norm(k)[:, None], positions)[:, 0]

    def block_keys(self, means: torch.Tensor, size: int) -> torch.Tensor:
        """The keys of blocks of `size` keys from their means, (..., blocks, i):
        normalized, then rotated to the position of each block's first key."""
        first = torch.arange(means.shape[-2], device=means.device) * size
        flat = self.k_norm(means).reshape(-1, 1, self.head_dim)
        at = first.expand(*means.shape[:-1]).reshape(-1)
        return self._rotate(flat, at).view_as(means)

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the first dims of x, (T, heads, i), by the positions, (T,)."""
        return x if self.rope is None else self.rope(x, positions)
