"""Compressed sparse attention, exported: bux's ``csa`` (DeepSeek-V4), its ``kv``,
``index`` and ``candidates`` ports.

Every query head attends to a sliding window of per-token latents and to compressed
latents, each the key and the value of one head that all query heads share, in one
softmax with the sinks. While decoding, a layer's one cache layer keeps the tokens'
latents as its keys and, as its values, the compressed latent each token ends (with
the indexer's key of it), so that both are every token's, and the window a mask over
them; with ``compress_ratio > 1`` its conv state keeps the last tokens' projections
that the next compressed latent pools (a ``hybrid`` layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .attention import SoftmaxAttention, reads, softmax_attention
from .common import Ctx, Norm, Site
from .conv import history
from .rope import Rope, rotated_dims
from .sparse import Indexer, key_positions


@dataclass
class Latents:
    """The compressed latents a layer attends to, a slot per token: what its ``kv``
    port carries.

    Attributes:
        kv: The latents, rotated, b 1 s d: the latent of the group a slot's token
            ends, where it ends one.
        index: The indexer's keys of them, b s i, if the layer that made them runs an
            indexer; else None.
        order: Which latent of its document each slot holds, b s; -1 for none.
        slot: The cache layer keeping them; None without a cache.
    """

    kv: torch.Tensor
    index: torch.Tensor | None
    order: torch.Tensor
    slot: int | None


class CompressedSparseAttention(SoftmaxAttention):
    """bux's ``csa``: queries from a normalized latent (or the input), a latent per
    token for the window, compressed latents of every ``compress_ratio`` tokens
    pooled under a softmax gate, the indexer's top-k of those, and the head outputs
    rotated back by their query's position before a grouped low-rank projection
    (``o_lora_rank``, ``o_groups``)."""

    bux_types = ("csa",)

    @staticmethod
    def cache_type(c: dict, ports: list) -> tuple[str, dict] | None:
        """The cache layer it keeps: its window's (``compress_ratio=None``), or every
        token's latents, compressed latents and the pooling's state (`Latents`)."""
        window, ratio = c["sliding_window_size"], c["compress_ratio"]
        if ratio is None and window is not None:
            return "sliding_attention", {"sliding_window": window}
        compresses = ratio is not None and ratio > 1 and not reads(ports, "kv")
        return ("hybrid" if compresses else "full_attention"), {}

    @staticmethod
    def masks(c: dict) -> list[tuple[str, Any]]:
        """The masks it reads, as ``ctx.mask`` takes them."""
        return [("sizes", None)]

    def __init__(self, c: dict, width: int, site: Site):
        h, d, r = c["num_heads"], c["head_dim"], c["compress_ratio"]
        super().__init__(c, width, site, h, d, d, False)
        if c["indexer"] and c["indexer"]["block_size"] > 1 and _has(site, "candidates"):
            raise NotImplementedError(
                "candidate pools of an indexer with block_size > 1"
            )
        self.d, self.ratio = d, r
        self.computes_logits = True
        compresses = r is not None and not reads(site.ports, "kv")
        latent, norm = c["q_lora_rank"], c["latent_norm"]
        self.q_a = nn.Linear(width, latent, bias=False) if latent else None
        self.q = nn.Linear(latent or width, h * d, bias=False)
        self.kv = nn.Linear(width, d, bias=False)
        self.compress_kv = nn.Linear(width, d, bias=False) if compresses else None
        pooled = compresses and r is not None and r > 1
        self.compress_gate = nn.Linear(width, d, bias=False) if pooled else None
        rank, groups = c["o_lora_rank"], c["o_groups"]
        self.o_a = (
            nn.Parameter(torch.zeros(groups, h * d // groups, rank)) if rank else None
        )
        self.o = nn.Linear(groups * rank if rank else h * d, width, bias=False)
        self.q_a_norm = Norm(norm, latent) if latent else None
        self.q_norm = Norm(c["qk_norm"], d) if c["qk_norm"] else None
        self.kv_norm = Norm(norm, d)
        self.compress_norm = Norm(norm, d) if compresses else None
        rope = c["rope"]
        self.rope = Rope(rope, d) if rope else None
        index = c["indexer"] and not reads(site.ports, "index")
        rotated, keys = rotated_dims(d, rope), d if compresses else None
        self.indexer = (
            Indexer(
                c["indexer"],
                width,
                latent or width,
                rope,
                rotated,
                keys,
                norm,
                rope and rope["layout"],
            )
            if index
            else None
        )

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        b, t, _ = x.shape
        query = (
            x
            if self.q_a is None or self.q_a_norm is None
            else self.q_a_norm(self.q_a(x))
        )
        q = self.q(query).view(b, t, self.h, self.d)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = self._rotate(q, ctx.positions)
        latent = self._rotate(self.kv_norm(self.kv(x))[:, :, None], ctx.positions)
        slot = ctx.cache_slot()
        if self.ratio is None:
            out = self.attend(x, q, self.cached(latent, latent, ctx, slot), ctx)
        else:
            keys, latents = self._latents(x, latent, ctx, slot)
            self.write(ctx, "kv", latents)
            kept = self._kept(x, query, ctx, latents)
            out = self._attend_parts(x, q, keys, latents, kept, ctx, slot)
        return self.output(x, self._unrotate(out, ctx.positions))

    # --- latents ---------------------------------------------------------------------

    def _latents(
        self, x: torch.Tensor, latent: torch.Tensor, ctx: Ctx, slot: int
    ) -> tuple[torch.Tensor, Latents]:
        """Return every token's latent the cache keeps, b 1 s d, and the compressed
        ones: this layer's, or its kv port's."""
        if reads(self.ports, "kv"):
            return self.cached(latent, latent, ctx, slot).k, self.read(ctx, "kv")
        compressed, index = self._compress(x, ctx, slot)
        values = compressed if index is None else torch.cat([compressed, index], -1)
        keys, values = latent.transpose(1, 2), values[:, None]
        if ctx.cache is not None:
            keys, values = ctx.cache.update(keys, values, slot)
        order = self._order(key_positions(ctx, keys.shape[2]))
        own = values[:, 0, :, self.d :] if index is not None else None
        held = slot if ctx.cache is not None else None
        return keys, Latents(values[:, :, :, : self.d], own, order, held)

    def _compress(
        self, x: torch.Tensor, ctx: Ctx, slot: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the compressed latent each token ends, rotated to its group's first
        token, b t d, and the indexer's key of it, b t i, if it runs one."""
        assert self.compress_kv is not None and self.compress_norm is not None
        r = self.ratio
        assert r is not None
        kv = self.compress_kv(x)
        if self.compress_gate is not None:
            kv = self._pool(torch.cat([kv, self.compress_gate(x)], -1), ctx, slot)
        latent = self.compress_norm(kv)
        first = (ctx.positions - (r - 1)).clamp(min=0)
        index = None if self.indexer is None else self.indexer.keys(latent, first)
        return self._rotate(latent[:, :, None], first)[:, :, 0], index

    def _pool(self, both: torch.Tensor, ctx: Ctx, slot: int) -> torch.Tensor:
        """Pool each token's group, itself and the r - 1 tokens before it, of the
        projections and their gates, b t 2d, under a softmax of the gates per
        channel; the r - 1 last tokens stay in the cache layer's conv state."""
        r, t = self.ratio, both.shape[1]
        assert r is not None
        xs = history(both, ctx, slot, 0, r - 1)
        group = torch.stack([xs[:, i : i + t] for i in range(r)], 2)
        kv, gate = group.split(self.d, -1)
        return (kv * gate.softmax(2)).sum(2)

    def _order(self, positions: torch.Tensor) -> torch.Tensor:
        """Which latent of its document each slot holds: a token that ends a group of
        compress_ratio holds one; -1 for the others and padding."""
        r = self.ratio
        assert r is not None
        ends = (positions >= 0) & ((positions + 1) % r == 0)
        return torch.where(ends, positions // r, -1)

    # --- attention -------------------------------------------------------------------

    def _kept(
        self, x: torch.Tensor, query: torch.Tensor, ctx: Ctx, latents: Latents
    ) -> torch.Tensor | None:
        """Return which compressed latents each query keeps, b t s: the ``index``
        port's, or the indexer's top-k of those it sees (inside the ``candidates``
        port's pool, if it reads one); None for all."""
        if reads(self.ports, "index"):
            kept = self.read(ctx, "index")
        elif self.indexer is not None:
            if latents.index is None:
                raise ValueError(
                    "the indexer has no keys: the kv port's layer has none"
                )
            seen = self._compressed_mask(ctx, latents)
            inside = (
                self.read(ctx, "candidates")
                if reads(self.ports, "candidates")
                else None
            )
            pool = _has(self, "candidates", "write")
            kept, candidates = self.indexer(
                x,
                query,
                ctx.positions,
                latents.index,
                seen,
                latents.order,
                inside,
                pool,
            )
            self.write(ctx, "candidates", candidates)
        else:
            return None
        self.write(ctx, "index", kept)
        return kept

    def _compressed_mask(self, ctx: Ctx, latents: Latents) -> torch.Tensor:
        """Which compressed latents each query sees, b t s: those of groups ended by
        it or before."""
        keep = ctx.mask("dense", (latents.slot, None)).keep[:, 0]
        return keep & (latents.order >= 0)[:, None]

    def _attend_parts(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        keys: torch.Tensor,
        latents: Latents,
        kept: torch.Tensor | None,
        ctx: Ctx,
        slot: int,
    ) -> torch.Tensor:
        """Attend to the window of token latents and the compressed latents kept, in
        one softmax with the sinks; b t h d."""
        own = slot if ctx.cache is not None else None
        window = ctx.mask("dense", (own, self.window))
        compressed = ctx.mask("dense", (latents.slot, None))
        seen = self._compressed_mask(ctx, latents)
        seen = seen if kept is None else seen & kept
        keep = torch.cat([window.keep, seen[:, None]], -1)
        offsets = torch.cat([window.offsets, compressed.offsets], -1)
        k = torch.cat([keys, latents.kv], 2)
        bias, gain = self._relative_bias(x, offsets), self._length_scale(ctx)
        return softmax_attention(
            q.transpose(1, 2),
            k,
            k,
            keep,
            self.scale,
            self.softcap,
            self.sinks,
            bias,
            gain,
        )

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the first rope.fraction dims of queries or latents."""
        return x if self.rope is None else self.rope(x, positions)

    def _unrotate(self, out: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate the head outputs back by their query's position, as bux does: by
        -positions, over the gain squared."""
        if self.rope is None:
            return out
        back, n = self.rope(out, -positions), self.rope.dims
        return torch.cat([back[..., :n] / self.rope.gain**2, back[..., n:]], -1)

    def project(self, out: torch.Tensor) -> torch.Tensor:
        """Project the heads back to the width, through o_lora_rank for each group of
        heads when the projection is grouped."""
        if self.o_a is None:
            return self.o(out.flatten(2))
        b, t = out.shape[:2]
        grouped = out.reshape(b, t, self.o_a.shape[0], -1)
        return self.o(torch.einsum("btgi,gir->btgr", grouped, self.o_a).flatten(2))


def _has(layer: Any, name: str, mode: str | None = None) -> bool:
    """Whether the layer (or site) lists the port `name`, in `mode` if given."""
    return any(n == name and mode in (None, m) for n, _, m in layer.ports)
