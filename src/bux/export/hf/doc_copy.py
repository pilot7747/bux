"""The document-local copy feature, exported: bux's ``doc_copy`` (modded-nanogpt PR
378), each position's candidate next token from the latest earlier occurrence of
its context in its sequence.

Decoding, each step sees its new tokens alone, and its contexts match against every
token before them: the layer's cache slot (a full-attention cache layer) keeps the
sequence's token ids, and which of them are real, as its keys.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn

from .common import Ctx, Site
from .conv import real_tokens

#: Two polynomial hashes of a window, mod 2^64 together, and the constants that tell
#: lengths apart, as bux's matcher keys them (`candidates`).
BASES = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F)
LENGTH_MIX = (0x2545F4914F6CDD1D, 0x5851F42D4C957F2D)
MASK = 2**64 - 1


def _signed(n: int) -> int:
    """n mod 2^64 as an int64."""
    n &= MASK
    return n - 2**64 if n >= 2**63 else n


@functools.cache
def _power_lists(base: int, t: int) -> tuple[list[int], list[int]]:
    """``base^i`` and ``base^-i`` mod 2^64, as int64 values, for i < t."""
    up, down = [1], [1]
    inverse = pow(base, -1, 2**64)
    for _ in range(t - 1):
        up.append(up[-1] * base & MASK)
        down.append(down[-1] * inverse & MASK)
    return [_signed(p) for p in up], [_signed(p) for p in down]


def _powers(base: int, t: int, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``base^i`` and ``base^-i`` mod 2^64, int64 tensors, for i < t."""
    up, down = _power_lists(base, t)
    as_tensor = lambda p: torch.tensor(p, dtype=torch.int64, device=device)
    return as_tensor(up), as_tensor(down)


def candidates(
    tokens: torch.Tensor,
    within: torch.Tensor,
    real: torch.Tensor,
    lengths: list[int],
    counts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each position's candidate next token (0 for none) and its bucket (0 for
    none), as bux's ``candidates``: the longest length whose context, ending at the
    position, ended earlier in the row, the token after its latest occurrence, and
    the length and how often it occurred (1 to `counts`).

    Args:
        tokens: The token ids, b rows of t, each row one sequence.
        within: How many real tokens of the row come before each.
        real: Which tokens are real, not a left-padded batch's padding.
        lengths: The context lengths, shortest first.
        counts: How many occurrences the buckets tell apart.
    """
    b, t = tokens.shape
    n, dev = len(lengths), tokens.device
    size = torch.tensor(lengths, device=dev)
    valid = real[:, None] & (within[:, None] >= size[:, None] - 1)  # b n t
    pos = torch.arange(t, device=dev)
    keys = []
    for base, mix in zip(BASES, LENGTH_MIX, strict=True):
        up, down = _powers(base, t, dev)
        prefix = torch.cumsum((tokens.long() + 1) * down, 1)
        padded = torch.cat([prefix.new_zeros(b, 1), prefix], 1)
        start = (pos - size[:, None] + 1).clamp(min=0)  # n t
        window = up * (padded[:, None, 1:] - padded[:, start])
        keys.append(window ^ (size[:, None] * _signed(mix)))
    alone = torch.arange(n * t, device=dev).view(n, t).expand(b, n, t)
    first, second = (torch.where(valid, k, alone) for k in keys)
    flag = (~valid).long()
    # A stable sort by the three keys: lexicographic, the last key first.
    index = torch.arange(n * t, device=dev).expand(b, -1)
    order = index
    for k in (second, first, flag):
        k = k.reshape(b, -1).gather(1, order)
        order = order.gather(1, torch.sort(k, dim=1, stable=True).indices)
    sk = [k.reshape(b, -1).gather(1, order) for k in (flag, first, second)]
    same = torch.stack([k[:, 1:] == k[:, :-1] for k in sk]).all(0)
    at = order % t
    prev = torch.full((b, n * t), -1, dtype=torch.long, device=dev)
    prev.scatter_(1, order[:, 1:], torch.where(same, at[:, :-1], -1))
    prev = prev.view(b, n, t)
    new_run = torch.cat([same.new_ones(b, 1), ~same], 1)
    run_first = torch.cummax(torch.where(new_run, index, 0), 1).values
    earlier = torch.zeros_like(prev.view(b, -1)).scatter_(1, order, index - run_first)
    earlier = earlier.view(b, n, t)
    found = (prev >= 0) & (prev < pos) & valid
    rank = torch.where(found, torch.arange(1, n + 1, device=dev)[:, None], 0).amax(1)
    pick = (rank - 1).clamp(min=0)[:, None]
    best = prev.gather(1, pick)[:, 0]
    seen = earlier.gather(1, pick)[:, 0]
    after = tokens.gather(1, (best + 1).clamp(max=t - 1))
    candidate = torch.where(rank > 0, after, 0)
    bucket = (rank - 1) * counts + seen.clamp(1, counts)
    return candidate, torch.where(rank > 0, bucket, 0)


class DocCopy(nn.Module):
    """bux's ``doc_copy``: adds ``gain (E[c] s[b] + e[b])`` of each position's
    candidate c and bucket b (`candidates`), E the model's first embedding's table,
    to the stream, every copy of a widened one. With its ``copy`` port it pushes the
    vector, or reads it instead and adds it by its own gain."""

    bux_types = ("doc_copy",)

    @staticmethod
    def cache_type(c: dict, ports: list) -> str | None:
        """The cache layer it keeps, as ``config.layer_types`` names it: the token
        ids so far, as an attention layer's keys; none for a reader of the port."""
        return None if _reads(ports) else "full_attention"

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        self.lengths, self.counts = list(c["lengths"]), c["counts"]
        self.ports = site.ports
        self.gain = nn.Parameter(torch.tensor(float(c["gain"])))
        self.reads = _reads(site.ports)
        if self.reads:
            return
        buckets = 1 + len(self.lengths) * self.counts
        self.scale = nn.Parameter(torch.zeros(buckets))
        self.bucket_embed = nn.Parameter(torch.zeros(buckets, site.config.hidden_size))

    def _history(self, ctx: Ctx) -> tuple[torch.Tensor, torch.Tensor]:
        """The row's token ids so far and which are real, b rows of s, the step's
        last: kept in the layer's cache slot between steps."""
        tokens, t = ctx.tokens, ctx.tokens.shape[1]
        keep = real_tokens(ctx, t)
        real = torch.ones_like(tokens, dtype=torch.bool) if keep is None else keep > 0
        keys = torch.stack([tokens.float(), real.float()], -1)[:, None]  # b 1 t 2
        slot = ctx.cache_slot()
        if ctx.cache is not None:
            keys, _ = ctx.cache.update(keys, keys[..., :1], slot)
        return keys[:, 0, :, 0].long(), keys[:, 0, :, 1] > 0

    def vector(self, ctx: Ctx) -> torch.Tensor:
        """The copy vector of the step's tokens, b t d."""
        tokens, real = self._history(ctx)
        within = real.long().cumsum(1) - 1
        found = candidates(tokens, within, real, self.lengths, self.counts)
        t = ctx.tokens.shape[1]
        candidate, bucket = (f[:, -t:] for f in found)
        live = (torch.arange(self.scale.shape[0], device=bucket.device) > 0).float()
        scale = (self.scale * live)[bucket][..., None]
        embed = (self.bucket_embed * live[:, None])[bucket]
        rows = ctx.embedding.tokens(candidate)
        return rows * scale.to(rows.dtype) + embed.to(rows.dtype)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        if self.reads:
            tap, pop = next((t, m == "pop") for n, t, m in self.ports if n == "copy")
            v = ctx.read(tap, pop)
        else:
            v = self.vector(ctx)
        for name, tap, mode in self.ports:
            if name == "copy" and mode == "write":
                ctx.push(tap, v)
        v = v if x.ndim == 3 else v[:, :, None]
        return x + (self.gain * v).to(x.dtype)


def _reads(ports: list) -> bool:
    """Whether the layer reads its ``copy`` port."""
    return any(n == "copy" and m != "write" for n, _, m in ports)
