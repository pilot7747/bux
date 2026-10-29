"""CPLM's copy pointer as vLLM samples it: plain PyTorch, which `bux_vllm.pointer`
runs, and the tests check against bux (`bux.architecture.copy`).

A head with a copy pointer keeps, for each token of a request, the pointer's key
and the token in a paged cache of its own, one row a token (`encode`): the key's
channels, in the cache's dtype (FP8 bytes under an FP8 KV cache), then the token as
`DIGITS` base-256 digits, which every dtype of a cache holds exactly. For each
request's last token of a step, the token vLLM samples from, the pointer attends
to the request's tokens before it and a sink (`pointed`), and `mixed` turns the
head's logits into the log of the mixture, which vLLM then samples:

    p(y) = p_lm(y) (1 - a (1 - a_sink)) + a p_copy(y)
"""

from __future__ import annotations

import torch

#: The base-256 digits of a token in a row of the cache: ids below 2^24.
DIGITS = 3


def row_width(dim: int) -> int:
    """The channels of a token's row: its key's `dim`, then its token's digits, to an
    even count, as vLLM splits a row into a key's and a value's halves."""
    width = dim + DIGITS
    return width + width % 2


def encode(
    keys: torch.Tensor,
    tokens: torch.Tensor,
    dtype: torch.dtype,
    fp8: torch.dtype | None,
) -> torch.Tensor:
    """The rows of tokens, (T, 1, `row_width`), in the cache's `dtype`: their keys,
    (T, c), in it, or as the bytes of `fp8` in a cache of bytes; then each token's
    digits, low first."""
    t, dim = keys.shape
    digits = torch.stack([(tokens.long() >> (8 * i)) & 255 for i in range(DIGITS)], -1)
    pad = row_width(dim) - dim - DIGITS
    if fp8 is not None:
        most = torch.finfo(fp8).max
        k = keys.float().clamp(-most, most).to(fp8).view(torch.uint8)
        tail = torch.cat([digits.to(torch.uint8), k.new_zeros(t, pad)], -1)
        return torch.cat([k, tail], -1)[:, None]
    tail = torch.cat([digits.to(dtype), keys.new_zeros(t, pad, dtype=dtype)], -1)
    return torch.cat([keys.to(dtype), tail], -1)[:, None]


def decode(
    rows: torch.Tensor, dim: int, fp8: torch.dtype | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys, (..., dim) in float32, and the tokens, (...,) int64, of `rows`,
    (..., `row_width`), as `encode` wrote them."""
    keys = rows[..., :dim]
    keys = keys.view(fp8).float() if fp8 is not None else keys.float()
    digits = rows[..., dim : dim + DIGITS].long()
    tokens = sum(digits[..., i] << (8 * i) for i in range(DIGITS))
    return keys, torch.as_tensor(tokens)


def gathered(cache: torch.Tensor, table: torch.Tensor, length: int) -> torch.Tensor:
    """The rows of each request's first `length` positions, (R, length, row), from
    the paged `cache`, ``[blocks, 1, block_size, row]``, by its block `table`."""
    size = cache.shape[2]
    at = torch.arange(length, device=cache.device)
    blocks = table[:, at // size].long()
    return cache[blocks, 0, at % size]


def pointed(
    q: torch.Tensor,
    sink: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    position: torch.Tensor,
    vocab: int,
    fp8: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pointer's distribution over the vocabulary, and its sink's share, for
    each request's query: its attention to the request's positions before the
    query's, and to the sink, in float32.

    Args:
        q: The queries, (R, c): RMS-normed, times the gain.
        sink: Each query's sink logit, (R,), scaled as its scores are.
        cache: The head's paged cache, ``[blocks, 1, block_size, row]``.
        table: Each request's blocks, (R, blocks).
        position: Each query's position; its keys are those before it, (R,).
        vocab: The vocabulary's size.
        fp8: The FP8 dtype of an FP8 cache's keys; None for the model's dtype.

    Returns:
        p_copy, (R, vocab), and a_sink, (R,).
    """
    r, dim = q.shape
    length = max(int(position.max()), 1) if r else 1
    keys, tokens = decode(gathered(cache, table, length), dim, fp8)
    scores = torch.einsum("rc,rlc->rl", q.float(), keys) * dim**-0.5
    seen = torch.arange(length, device=q.device) < position[:, None]
    scores = scores.masked_fill(~seen, -torch.inf)
    every = torch.cat([scores, sink.float()[:, None]], -1).softmax(-1)
    copy = torch.zeros(r, vocab, dtype=torch.float32, device=q.device)
    copy.scatter_add_(-1, tokens.clamp(0, vocab - 1), every[:, :-1])
    return copy, every[:, -1]


def mixed(
    logits: torch.Tensor, p_copy: torch.Tensor, a_sink: torch.Tensor, copy: int
) -> torch.Tensor:
    """The log of the mixture, (R, vocab), from the head's logits, (R, vocab), and
    the pointer's p_copy and a_sink, in float32; -inf at <copy>, which is no token.

    With a the softmax's mass on `copy` and p_lm the softmax over the other rows:
    ``log(p_lm (1 - a + a a_sink) + a p_copy)``, the mixture bux's loss scores."""
    z = logits.float()
    rows = torch.arange(z.shape[-1], device=z.device)
    others = z.masked_fill(rows == copy, -torch.inf)
    lse_lm = others.logsumexp(-1, keepdim=True)
    gate = z[:, copy : copy + 1]
    lse = torch.logaddexp(lse_lm, gate)
    log_gate = gate - lse
    kept = torch.logaddexp(lse_lm - lse, log_gate + a_sink.float()[:, None].log())
    out = torch.logaddexp(others - lse_lm + kept, log_gate + p_copy.float().log())
    return out.masked_fill(rows == copy, -torch.inf)
