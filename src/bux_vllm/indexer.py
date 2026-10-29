"""The lightning indexer of a ``gqa`` or ``mla`` layer in plain PyTorch, as bux's
``Indexer`` makes it from the layer's input (``from_stream``): its queries (from an
``mla`` layer's query latent where it has one, as DeepSeek-V3.2's), their heads'
weights, and the keys it caches, one per token.

The keys are normalized and rotated as they are cached, but where the indexer
mean-pools blocks of them (Qwen3.8 QSA): then the cache keeps them as projected, and
each step pools a block's, normalizes the mean and rotates it to the block's first
position (`block_keys`). The indexer's weights are the same on every rank under
tensor parallelism, so every rank picks the same keys.

A ``gqa`` layer's indexer projects its queries, keys and head weights from the
layer's input by one GEMM, a vLLM linear layer the layer makes (`Indexer.merged`,
`project`), which takes the engine's quantization config as the layer's other
projections do; on its own the indexer keeps three PyTorch ones.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import Norm, Rope, load_into

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
        merged: One linear layer of its queries', keys' and head weights'
            projections of the input, side by side (`sizes`), in place of its own
            three; None for those.
    """

    def __init__(
        self,
        c: dict,
        width: int,
        rope: dict | None,
        rope_dims: int,
        query_width: int | None = None,
        merged: Any = None,
    ):
        super().__init__()
        h, i = c["num_heads"], c["head_dim"]
        self.heads, self.head_dim, self.weighted = h, i, c["weighted"]
        self.mean = c["block_size"] > 1 and c["pool"] == "mean"
        self.merged = merged
        self.q: nn.Linear | None = None
        self.k: nn.Linear | None = None
        self.weights: nn.Linear | None = None
        if merged is None:
            self.q = nn.Linear(query_width or width, h * i, bias=False)
            self.k = nn.Linear(width, i, bias=False)
            self.weights = nn.Linear(width, h, bias=False) if self.weighted else None
        self.q_norm = Norm(c["norm"], i) if c["norm"] else None
        self.k_norm = Norm(c["norm"] or STREAM_KEY_NORM, i)
        n = min(rope_dims, i) // 2 * 2
        own = {**rope, "fraction": 1.0, "layout": c["rope_layout"]} if rope else None
        self.rope = Rope(own, n) if own is not None and n else None

    @staticmethod
    def sizes(c: dict) -> list[int]:
        """The outputs of a `merged` projection of the indexer of config `c`: its
        queries', its keys', and its head weights' (if weighted), padded to 16, as
        vLLM's FP8 GEMMs take them."""
        h, i = c["num_heads"], c["head_dim"]
        return [h * i, i] + ([-(-h // 16) * 16] if c["weighted"] else [])

    def project(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The queries, (T, H, i), head weights, (T, H), and keys, (T, i), of the
        tokens x: by the `merged` projection where it has one."""
        if self.merged is None:
            q, w = self.queries(x, positions), self.head_weights(x)
            return q, w, self.keys(x, positions)
        h, i = self.heads, self.head_dim
        out = self.merged(x)
        q, k, w = out[:, : h * i], out[:, h * i : (h + 1) * i], out[:, (h + 1) * i :]
        q = self._queried(q.view(-1, h, i), positions)
        w = self._weighed(w[:, :h] if self.weighted else None, x)
        return q, w, self._keyed(k, positions)

    def queries(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The query heads of the tokens x (or their query latents), (T, H, i):
        projected, normalized and rotated by their positions."""
        assert self.q is not None, "a merged indexer projects in `project`"
        q = self.q(x).view(x.shape[0], self.heads, self.head_dim)
        return self._queried(q, positions)

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Each head's weight in a query's scores, with the head dims' scale, (T, H)."""
        return self._weighed(None if self.weights is None else self.weights(x), x)

    def keys(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The keys the cache keeps of the tokens x, (T, i): as projected if the
        indexer mean-pools blocks of them, else normalized and rotated."""
        assert self.k is not None, "a merged indexer projects in `project`"
        return self._keyed(self.k(x), positions)

    def _queried(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The projected query heads, (T, H, i), normalized and rotated."""
        if self.q_norm is not None:
            q = self.q_norm(q)
        return self._rotate(q, positions)

    def _weighed(self, w: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        """The projected head weights, (T, H), or None for unweighted heads, with
        the scales of the head dims and the heads."""
        scale = self.head_dim**-0.5
        if w is None:
            return x.new_full((x.shape[0], self.heads), scale)
        return w * (scale * self.heads**-0.5)

    def _keyed(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The projected keys, (T, i), as the cache keeps them (`keys`)."""
        if self.mean:
            return k
        return self._rotate(self.k_norm(k)[:, None], positions)[:, 0]

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        """Load the checkpoint's indexer weight `name` (``q.weight``) into its
        parameter: a `merged` projection's shard, the head weights padded with
        rows of zeros; return the parameters filled.

        Raises:
            KeyError: The indexer has no such weight.
        """
        shards = {"q.weight": 0, "k.weight": 1, "weights.weight": 2}
        if self.merged is not None and name in shards:
            if name == "weights.weight":
                rows = self.merged.output_sizes[2] - tensor.shape[0]
                tensor = F.pad(tensor, (0, 0, 0, rows))
            load_into(self.merged.weight, tensor, shards[name])
            return [self.merged.weight]
        params = dict(self.named_parameters())
        if name not in params or name.startswith("merged."):
            raise KeyError(f"no weight {name!r} in the indexer")
        load_into(params[name], tensor)
        return [params[name]]

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
