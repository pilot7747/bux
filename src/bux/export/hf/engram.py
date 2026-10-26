"""Engram, exported: memory looked up by hashed n-grams of the input tokens, bux's
``engram``.

Decoding, each step sees its new tokens alone, and its n-grams reach back
``max_ngram_size - 1`` tokens before them: the layer's cache slot (a
``LinearAttentionLayer``) keeps those token ids as its recurrent state, and the
conv's inputs' tail as its conv state (`bux.export.hf.conv`).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Norm, Site
from .conv import CausalConv


class Engram(nn.Module):
    """bux's ``engram`` (DeepSeek Engram): every n-gram of 2 to ``max_ngram_size``
    tokens ending at a position, hashed by ``num_heads`` heads into tables of prime
    sizes (``site.extra``, `bux.export.plan.engram_hashes`), its rows concatenated
    into e. With ``a = <norm(x), norm(W_k e)> / sqrt(d)``, it adds
    ``u = sigmoid(sign(a) sqrt(abs(a))) W_v e`` to the stream, and
    ``silu(conv(norm(u)))`` with a conv. N-grams do not reach back past the boundary
    token, the sequence's start or a left-padded batch's padding. On a widened
    stream every copy has a key of its own. A token's class is its id mod
    ``vocab_classes``, as bux makes them.
    """

    bux_types = ("engram",)

    @staticmethod
    def cache_type(c: dict) -> str:
        """The cache layer it keeps, as ``config.layer_types`` names it."""
        return "conv"

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        d, n = site.config.hidden_size, c["max_ngram_size"]
        residual = site.config.bux["residual"] or {}
        copies = residual.get("expansion", 1) if width == "streams" else 1
        self.n, self.widened = n, width == "streams"
        self.primes: list[list[int]] = site.extra["primes"]
        self.multipliers: list[int] = site.extra["multipliers"]
        flat = [p for row in self.primes for p in row]
        offsets = [0, *itertools.accumulate(flat[:-1])]
        heads = c["num_heads"]
        self.offsets = [offsets[i : i + heads] for i in range(0, len(flat), heads)]
        self.classes, self.boundary = c["vocab_classes"], c["boundary_token"]
        self.table = nn.Parameter(torch.zeros(sum(flat), c["head_dim"]))
        self.kv = nn.Linear(len(flat) * c["head_dim"], (copies + 1) * d, bias=False)
        norm, size = c["norm"], c["conv_size"]
        self.q_norm, self.k_norm = Norm(norm, (copies, d)), Norm(norm, (copies, d))
        self.conv = CausalConv(copies * d, size, False, dilation=n) if size else None
        self.conv_norm = Norm(norm, (copies, d)) if size else None

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        slot = ctx.cache_slot()
        rows = self.rows(self.tokens(ctx, slot))
        memory = F.embedding(rows, self.table).flatten(2)
        h = x if self.widened else x[..., None, :]
        b, t, n, d = h.shape
        kv = self.kv(memory)
        key, value = kv[..., : n * d].reshape(b, t, n, d), kv[..., n * d :]
        a = (self.q_norm(h) * self.k_norm(key)).sum(-1) / math.sqrt(d)
        gate = torch.sigmoid(torch.sign(a) * torch.sqrt(a.abs().clamp(min=1e-6)))
        u = gate[..., None] * value[..., None, :]
        if self.conv is not None and self.conv_norm is not None:
            conv = self.conv(self.conv_norm(u).reshape(b, t, n * d), ctx, slot)
            u = u + F.silu(conv).reshape(u.shape)
        y = h + u
        return y if self.widened else y[..., 0, :]

    def tokens(self, ctx: Ctx, slot: int) -> torch.Tensor:
        """Return each row's token ids plus 1, after the ``max_ngram_size - 1`` before
        them: 0 where there is none, before the sequence or in padding. Decoding,
        those are the ones the cache slot kept, and it keeps the last of these."""
        ids = ctx.tokens + 1
        mask = ctx.attention_mask
        if mask is not None and mask.ndim == 2:
            ids = ids * mask[:, -ids.shape[1] :].to(ids.dtype)
        before = ids.new_zeros(ids.shape[0], self.n - 1)
        cache = ctx.cache
        if cache is not None and cache.layers[slot].is_recurrent_states_initialized[0]:
            before = cache.layers[slot].recurrent_states[0]
        ids = torch.cat([before, ids], 1)  # before the cache overwrites what it kept
        if cache is not None:
            cache.update_recurrent_state(ids[:, 1 - self.n :], slot)
        return ids

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the table rows, b t (n-1)*heads, of the n-grams ending at each of
        the last t of `ids` (from `tokens`), as bux's ``_ngram_rows``: the n-gram of
        the last j + 1 ids hashes to ``XOR_{i <= j} ids[t - i] * m_i`` in wrapping
        uint32 arithmetic, an id at or before the last boundary or missing one before
        t counting as the boundary's (or 0's)."""
        ids = ids - 1
        pos = torch.arange(ids.shape[1], device=ids.device)
        stop = ids < 0
        if self.boundary is not None:
            stop = stop | (ids == self.boundary)
        after = torch.where(stop, pos + 1, 0)
        start = F.pad(after.cummax(1).values[:, :-1], (1, 0))
        pad, ids = self.boundary or 0, ids.clamp(min=0)
        if self.classes:
            ids, pad = ids % self.classes, pad % self.classes
        primes = torch.tensor(self.primes, device=ids.device)
        offsets = torch.tensor(self.offsets, device=ids.device)
        h, rows = torch.zeros_like(ids), []
        for i, m in enumerate(self.multipliers):
            back = pos - i
            prev = torch.where(back >= start, ids[:, back.clamp(min=0)], pad)
            h = h ^ ((prev * m) & 0xFFFFFFFF)
            if i:
                rows.append(h[..., None] % primes[i - 1] + offsets[i - 1])
        return torch.cat(rows, -1)[:, self.n - 1 :]
