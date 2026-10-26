"""Multi-token prediction's input, exported: bux's ``nextn``."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Norm, Site
from .residual import widen, widening


class NextN(nn.Module):
    """bux's ``nextn`` (DeepSeek-V3): the stream mixed with the embedding of the token
    k positions ahead, ``W [norm(e); norm(h)]``, for the k-th nextn the pass runs;
    the last k positions, which have no token that far ahead, embed token 0, as in
    bux. On the n copies of a widened stream, it mixes the embedding into each,
    normalizing the copies together; on one stream under a widening algebra, it
    widens its output.

    It reads tokens after the ones it predicts from, so it runs on whole sequences,
    without a cache: the model runs the multi-token prediction after its first head
    only when asked for every head (``all_heads``), never as it generates.
    """

    bux_types = ("nextn",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        d = site.config.hidden_size
        copies = widening(site.config.bux["residual"])
        if width == "streams":  # n copies in, n copies out
            assert copies is not None
            self.hidden_norm, self.copies = Norm(c["norm"], copies * d), None
            width = d
        else:  # one stream in, widened outside any branch
            self.hidden_norm = Norm(c["norm"], width)
            self.copies = None if site.depth else copies
        self.embed_norm = Norm(c["norm"], d)
        self.proj = nn.Linear(d + width, width, bias=False)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        if ctx.cache is not None:
            raise ValueError("nextn reads tokens ahead: run it without a cache")
        ctx.nextn_runs += 1
        k, tokens = ctx.nextn_runs, ctx.tokens
        ahead = F.pad(tokens[:, k:], (0, min(k, tokens.shape[1])))
        e = self.embed_norm(ctx.embedding.embed_tokens(ahead))
        if x.ndim == 4:  # the copies of a widened stream
            h = self.hidden_norm(x.reshape(*x.shape[:2], -1)).reshape(x.shape)
            e = e[:, :, None].expand(x.shape)
        else:
            h = self.hidden_norm(x)
        y = self.proj(torch.cat([e, h], -1))
        return y if self.copies is None else widen(y, self.copies)
