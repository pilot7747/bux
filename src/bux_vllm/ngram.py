"""bux's ``engram`` on the batch's tokens side by side, in plain PyTorch: each token's
n-gram, its table rows, the gate, and the dilated causal conv, as
`bux.export.hf.engram` computes them; `bux_vllm.engram` keeps the conv's state in
vLLM's cache and takes the tokens before a step from vLLM's model runner.

A step holds several sequences' tokens one after another, each sequence's from
``query_start_loc[r]`` on. A token's n-gram reaches back into the tokens of its
sequence before the step (``context``), and the conv into its last inputs before the
step (``past``): everything here is a gather over the step's tokens, with no loop
over them.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Layer, Norm


def requests(query_start_loc: torch.Tensor, tokens: int) -> torch.Tensor:
    """The sequence of each of the step's `tokens`, by where each starts
    (``query_start_loc``, R + 1 of them); the last for tokens past the end."""
    at = torch.arange(tokens, device=query_start_loc.device)
    r = torch.searchsorted(query_start_loc, at.to(query_start_loc.dtype), right=True)
    return (r - 1).clamp(0, query_start_loc.numel() - 2)


def windows(
    ids: torch.Tensor, query_start_loc: torch.Tensor, context: torch.Tensor, n: int
) -> torch.Tensor:
    """Each token's last n ids, (T, n), newest first: column k is the id k tokens
    back in its sequence, from the step's `ids` (T,) or else from `context`, (R, C)
    for C >= n - 1, each sequence's last C ids before the step; -1 where there is
    none."""
    t, c = ids.shape[0], context.shape[1]
    req = requests(query_start_loc, t)
    at = torch.arange(t, device=ids.device)
    pos = at - query_start_loc[req]
    back = torch.arange(n, device=ids.device)
    in_step = pos[:, None] >= back
    step = ids.long()[(at[:, None] - back).clamp(min=0)]
    column = (c - (back - pos[:, None])).clamp(0, c - 1)
    before = context.long()[req[:, None], column]
    return torch.where(in_step, step, before)


def ngram_rows(
    window: torch.Tensor, hashes: Hashes, classes: int | None, boundary: int | None
) -> torch.Tensor:
    """The table rows, (T, (n-1) heads), of each token's n-grams from its `window`
    (`windows`), as bux's ``_ngram_rows``: the n-gram of the last j + 1 ids hashes to
    ``XOR_{i <= j} id_i * m_i`` in wrapping uint32 arithmetic, an id at or before the
    last boundary or missing one counting as the boundary's (or 0's).

    Args:
        window: Each token's ids, newest first, -1 for none.
        hashes: What the layer hashes by.
        classes: Hash each id mod this many classes; None: the ids.
        boundary: The token n-grams do not reach back past; None: none.
    """
    stop = window[:, 1:] < 0
    if boundary is not None:
        stop = stop | (window[:, 1:] == boundary)
    valid = F.pad(stop.cumsum(1) == 0, (1, 0), value=True)
    pad, ids = boundary or 0, window.clamp(min=0)
    if classes:
        ids, pad = ids % classes, pad % classes
    ids = torch.where(valid, ids, pad)
    h, rows = torch.zeros_like(ids[:, 0]), []
    for i, m in enumerate(hashes.multipliers):
        h = h ^ ((ids[:, i] * m) & 0xFFFFFFFF)
        if i:
            rows.append(h[:, None] % hashes.primes[i - 1] + hashes.offsets[i - 1])
    return torch.cat(rows, -1)


class Hashes(nn.Module):
    """What an engram hashes by, from its site (`bux.export.plan.engram_hashes`):
    the odd ``multipliers`` of its n ids, and the prime size of each n-gram size's
    and head's table, (n - 1, heads), with where each table starts among the rows."""

    primes: torch.Tensor
    offsets: torch.Tensor

    def __init__(self, extra: dict):
        super().__init__()
        self.multipliers: list[int] = extra["multipliers"]
        primes = torch.tensor(extra["primes"])
        offsets = F.pad(primes.flatten().cumsum(0)[:-1], (1, 0)).view(primes.shape)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)


def dilated_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
    query_start_loc: torch.Tensor,
    past: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bux's causal depthwise conv of a dilation over the step's sequences:
    ``y_t = sum_i w[i] x[t - (size - 1 - i) dilation]``, reading before each
    sequence's first token of the step its `past`.

    Args:
        x: The step's inputs, (T, C).
        weight: The taps, (size, C).
        dilation: The distance between two taps.
        query_start_loc: Where each sequence's tokens start, (R + 1,).
        past: Each sequence's last ``(size - 1) dilation`` inputs before the step,
            (R, keep, C): zeros before its first.

    Returns:
        The conv, (T, C), and each sequence's last inputs after the step, as `past`.
    """
    t, r, keep = x.shape[0], past.shape[0], past.shape[1]
    size = weight.shape[0]
    req = requests(query_start_loc, t)
    at = torch.arange(t, device=x.device)
    pos = at - query_start_loc[req]
    inputs = torch.cat([past.reshape(r * keep, -1), x])
    y: Any = 0
    for i in range(size):
        back = (size - 1 - i) * dilation
        early = pos - back
        index = torch.where(early >= 0, r * keep + at - back, req * keep + keep + early)
        y = y + weight[i] * inputs[index]
    return y, inputs[_last(query_start_loc, r, keep)]


def _last(query_start_loc: torch.Tensor, r: int, keep: int) -> torch.Tensor:
    """Where each sequence's last `keep` inputs after the step are among the past
    inputs and then the step's, (R, keep)."""
    k = torch.arange(keep, device=query_start_loc.device)
    lengths = (query_start_loc[1:] - query_start_loc[:-1]).long()
    end = lengths[:, None] + k
    seqs = torch.arange(r, device=k.device)[:, None]
    from_step = r * keep + query_start_loc[1:, None].long() + k - keep
    return torch.where(end < keep, seqs * keep + end, from_step)


class EngramCore(Layer):
    """bux's ``engram`` (DeepSeek Engram) but where its conv keeps its state: every
    n-gram of 2 to ``max_ngram_size`` tokens ending at a token hashed by
    ``num_heads`` heads into tables of prime sizes, its rows concatenated into e;
    with ``a = <norm(x), norm(W_k e)> / sqrt(d)``, it adds
    ``u = sigmoid(sign(a) sqrt(abs(a))) W_v e`` to the stream, and
    ``silu(conv(norm(u)))`` with a conv (`convolve`). On a widened stream every copy
    has a key of its own.

    Args:
        c: Its config, as JSON.
        d: The model's width.
        copies: The copies of the stream it adds to: 1, or a widened stream's n.
        hashes: Its site's hashes (`Hashes`).
    """

    def __init__(self, c: dict, d: int, copies: int, hashes: dict):
        super().__init__()
        self.n, self.copies, self.d = c["max_ngram_size"], copies, d
        self.hashes = Hashes(hashes)
        self.classes, self.boundary = c["vocab_classes"], c["boundary_token"]
        rows = sum(p for row in hashes["primes"] for p in row)
        tables = (self.n - 1) * c["num_heads"]
        self.table = nn.Parameter(torch.empty(rows, c["head_dim"]), requires_grad=False)
        self.kv = nn.Linear(tables * c["head_dim"], (copies + 1) * d, bias=False)
        self.kv.requires_grad_(False)
        norm, size = c["norm"], c["conv_size"]
        self.q_norm, self.k_norm = Norm(norm, (copies, d)), Norm(norm, (copies, d))
        self.conv_norm = Norm(norm, (copies, d)) if size else None

    def memory(self, ctx: Ctx) -> torch.Tensor:
        """The concatenated table rows e of the step's tokens, (T, rows' width).

        Raises:
            NotImplementedError: vLLM gave no token ids or no tokens before the step.
        """
        if ctx.input_ids is None or ctx.ngram_context is None:
            raise NotImplementedError(
                "engram reads the token ids and those before each step, which "
                "vLLM's model runner V2 (its default) gives"
            )
        assert ctx.query_start_loc is not None
        window = windows(ctx.input_ids, ctx.query_start_loc, ctx.ngram_context, self.n)
        rows = ngram_rows(window, self.hashes, self.classes, self.boundary)
        return F.embedding(rows, self.table).flatten(1)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        h = x if x.ndim == 3 else x[:, None, :]
        t, n, d = h.shape
        kv = self.kv(self.memory(ctx)[:t])
        key, value = kv[..., : n * d].reshape(t, n, d), kv[..., n * d :]
        a = (self.q_norm(h) * self.k_norm(key)).sum(-1) / math.sqrt(d)
        gate = torch.sigmoid(torch.sign(a) * torch.sqrt(a.abs().clamp(min=1e-6)))
        u = gate[..., None] * value[..., None, :]
        if self.conv_norm is not None:
            conv = self.convolve(self.conv_norm(u).reshape(t, n * d), ctx)
            u = u + F.silu(conv).reshape(u.shape)
        y = h + u
        return y if x.ndim == 3 else y[:, 0, :]

    def convolve(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """The dilated causal conv of u's norm, (T, copies d), by its state."""
        raise NotImplementedError
