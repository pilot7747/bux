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

from dataclasses import dataclass

import torch

from . import picks

#: The base-256 digits of a token in a row of the cache: ids below 2^24.
DIGITS = 3


@dataclass
class Steps:
    """A step's tokens, as the pointer's op reads them: views of vLLM's buffers,
    which every replay of a CUDA graph of decoded tokens reads in place, and the
    counts the graph was captured with. The op works each request's last token and
    its position out of them (`stepped`): a tensor made with the metadata would be
    the capture's at every replay.

    Attributes:
        slot_mapping: Each token's slot, (T,); -1 for padding.
        block_table_tensor: Each request's blocks, (R, blocks).
        num_actual_tokens: The step's tokens, padding left out.
        query_start_loc: Each request's first token in the step, then the step's
            end, (R + 1,).
        seq_lens: Each request's length with the step's tokens, (R,).
    """

    slot_mapping: torch.Tensor
    block_table_tensor: torch.Tensor
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor

    @property
    def requests(self) -> int:
        """The step's requests, padding in them."""
        return self.seq_lens.shape[0]


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
    # A row no step wrote, as the null block's that padded requests point at,
    # gives bytes: its token is then some token, never an index out of range.
    digits = rows[..., dim : dim + DIGITS].float()
    digits = torch.where((digits >= 0) & (digits <= 255), digits, 0).long()
    tokens = sum(digits[..., i] << (8 * i) for i in range(DIGITS))
    return keys, torch.as_tensor(tokens)


def gathered(cache: torch.Tensor, table: torch.Tensor, length: int) -> torch.Tensor:
    """The rows of each request's first `length` positions, (R, length, row), from
    the paged `cache`, ``[blocks, 1, block_size, row]``, by its block `table`."""
    size = cache.shape[2]
    at = torch.arange(length, device=cache.device)
    columns = (at // size).clamp(max=table.shape[1] - 1)
    blocks = table[:, columns].long().clamp(0, cache.shape[0] - 1)
    return cache[blocks, 0, at % size]


def stepped(
    cache: torch.Tensor,
    steps: Steps,
    q: torch.Tensor,
    k: torch.Tensor,
    tokens: torch.Tensor,
    sink: torch.Tensor,
    rows: int,
    fp8: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write the step's keys `k`, (T, c), and `tokens`, (T,), into the paged
    `cache`; return the query, (rows, c), and sink, (rows,), of each of its first
    `rows` requests' last token, the one vLLM samples, with the request's blocks
    and that token's position, before which its keys lie."""
    n = steps.num_actual_tokens
    picks.write(
        cache, steps.slot_mapping[:n], encode(k[:n], tokens[:n], cache.dtype, fp8)
    )
    last = sampled(steps.query_start_loc[1 : rows + 1] - 1, q.shape[0])
    position = steps.seq_lens[:rows] - 1
    return q[last], sink[last], steps.block_table_tensor[:rows], position


def sampled(last: torch.Tensor, tokens: int) -> torch.Tensor:
    """The step's rows the pointer reads for each request's last token, `last`, of a
    step of `tokens` rows, int64: clamped into them, as vLLM's dummy and capture runs
    give padded requests rows past the step's, whose results nothing reads."""
    return last.long().clamp(0, max(tokens - 1, 0))


def pointed(
    q: torch.Tensor,
    sink: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    position: torch.Tensor,
    vocab: int,
    fp8: torch.dtype | None,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pointer's distribution over the vocabulary, and its sink's share, for
    each request's query: its attention to the request's positions before the
    query's, and to the sink, in float32. It reads the first `length` positions of
    every request, a count fixed on the host, so that a CUDA graph captures it.

    Args:
        q: The queries, (R, c): RMS-normed, times the gain.
        sink: Each query's sink logit, (R,), scaled as its scores are.
        cache: The head's paged cache, ``[blocks, 1, block_size, row]``.
        table: Each request's blocks, (R, blocks).
        position: Each query's position; its keys are those before it, (R,).
        vocab: The vocabulary's size.
        fp8: The FP8 dtype of an FP8 cache's keys; None for the model's dtype.
        length: The positions it reads of each request, `max_model_len`: as many
            as any query has before it.

    Returns:
        p_copy, (R, vocab), and a_sink, (R,).
    """
    r, dim = q.shape
    keys, tokens = decode(gathered(cache, table, length), dim, fp8)
    scores = torch.einsum("rc,rlc->rl", q.float(), keys) * dim**-0.5
    seen = torch.arange(length, device=q.device) < position[:, None]
    scores = scores.masked_fill(~seen, -torch.inf)
    every = torch.cat([scores, sink.float()[:, None]], -1).softmax(-1)
    copy = torch.zeros(r, vocab, dtype=torch.float32, device=q.device)
    copy.scatter_add_(-1, tokens.clamp(0, vocab - 1), every[:, :-1])
    return copy, every[:, -1]


def rows_of(
    p_copy: torch.Tensor, a_sink: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first `n` rows of the pointer's buffers, (max_rows, vocab) and
    (max_rows,): past the buffers' rows, as vLLM's dummy sampler run asks for more
    rows than requests, rows of no pointer, p_copy 0 and a_sink 1, which leave the
    head's softmax over the vocabulary but <copy>."""
    have = p_copy.shape[0]
    if n <= have:
        return p_copy[:n], a_sink[:n]
    more = n - have
    p_copy = torch.cat([p_copy, p_copy.new_zeros(more, p_copy.shape[1])])
    return p_copy, torch.cat([a_sink, a_sink.new_ones(more)])


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
