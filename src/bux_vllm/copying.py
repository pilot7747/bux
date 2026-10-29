"""bux's ``doc_copy`` (modded-nanogpt PR 378) in vLLM, its plain PyTorch: each step's
tokens' candidate next tokens and buckets from their sequences' tokens so far, and
the copy vector of them, as `bux.export.hf.doc_copy` computes them.

A sequence is one document. For a token at position p, every earlier position j
ends a window of each length L that matches the token's own when the last L tokens
up to j are those up to p (j >= L - 1, p >= L - 1): their common suffix is L long
or longer. The longest L with such a j picks the latest one; the candidate is the
token after it, and its bucket counts the matching j (1 to `counts`). The plugin
scans each token's sequence on the GPU (`bux_vllm.doc_copy`); `scan` is the same
scan in PyTorch, which a CPU test checks against bux's matcher.
"""

from __future__ import annotations

import torch


def scan(
    history: torch.Tensor,
    rows: torch.Tensor,
    pos: torch.Tensor,
    lengths: list[int],
    counts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token's candidate (0 for none) and bucket (0 for none).

    Args:
        history: Every sequence's tokens so far, (R, S), the step's among them.
        rows: Each token's sequence's row of `history`, (T,).
        pos: Each token's position in its sequence, (T,).
        lengths: The context lengths, shortest first.
        counts: How many earlier occurrences the buckets tell apart.
    """
    t, s, most = pos.shape[0], history.shape[1], max(lengths)
    h = history[rows].long()  # T S
    j = torch.arange(s, device=h.device)
    run = j[None] < pos[:, None]  # the earlier positions
    common = torch.zeros(t, s, dtype=torch.long, device=h.device)
    for k in range(most):
        at, mine = j[None] - k, pos[:, None] - k
        theirs = h.gather(1, at.clamp(min=0).expand(t, -1))
        ours = h.gather(1, mine.clamp(min=0))
        run = run & (at >= 0) & (mine >= 0) & (theirs == ours)
        common += run
    candidate = torch.zeros(t, dtype=torch.long, device=h.device)
    bucket = torch.zeros(t, dtype=torch.long, device=h.device)
    for i, n in enumerate(lengths):  # the longest last
        match = (common >= n) & (j[None] >= n - 1) & (pos[:, None] >= n - 1)
        seen = match.sum(1)
        last = torch.where(match, j[None], -1).amax(1)
        after = h.gather(1, (last + 1).clamp(0, s - 1)[:, None])[:, 0]
        found = seen > 0
        candidate = torch.where(found, after, candidate)
        bucket = torch.where(found, i * counts + seen.clamp(max=counts), bucket)
    return candidate, bucket


def vector(
    rows: torch.Tensor,
    bucket: torch.Tensor,
    scale: torch.Tensor,
    bucket_embed: torch.Tensor,
) -> torch.Tensor:
    """The copy vector, (T, d), of the candidates' embedding rows, (T, d), and their
    buckets: ``E[c] s[b] + e[b]``, nothing for bucket 0."""
    live = (torch.arange(scale.shape[0], device=bucket.device) > 0).to(rows.dtype)
    s = (scale.to(rows.dtype) * live)[bucket][:, None]
    e = (bucket_embed.to(rows.dtype) * live[:, None])[bucket]
    return rows * s + e
