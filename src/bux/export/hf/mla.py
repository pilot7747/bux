"""Multi-head latent attention, exported: bux's ``mla`` (DeepSeek-V2/V3, Kimi K2,
Kimi Linear), with the settings it shares with ``gqa`` and DeepSeek-V3.2's indexer."""

from __future__ import annotations

import torch
from torch import nn

from .attention import SoftmaxAttention, reads
from .common import Ctx, Norm, Site
from .rope import Rope, rotated_dims
from .sparse import Indexer

#: The norm of MLA's query and key-value latents.
LATENT_NORM = {
    "type": "rmsnorm",
    "eps": 1e-06,
    "affine": True,
    "bias": False,
    "zero_centered": False,
    "scale_input": False,
}


class LatentAttention(SoftmaxAttention):
    """bux's ``mla``: every head's keys and values projected up from one normalized
    latent per token, its queries from another (or from the input, at full rank), and
    ``qk_rope_head_dim`` more key dims that all heads share, the only ones rotated
    (none with ``rope=None``).

    It caches every head's keys and values as projected up, b h s (n + r) and b h s
    dv, not the latents: what transformers' cache layers hold. With an indexer, its
    queries come from the query latent and its keys from the input (DeepSeek-V3.2).
    """

    bux_types = ("mla",)

    def __init__(self, c: dict, width: int, site: Site):
        h, n = c["num_heads"], c["qk_nope_head_dim"]
        r, dv = c["qk_rope_head_dim"], c["v_head_dim"]
        super().__init__(c, width, site, h, n + r, dv, False)
        self.n, self.r, self.rank = n, r, c["kv_lora_rank"]
        latent = c["q_lora_rank"]
        self.q_a = nn.Linear(width, latent, bias=False) if latent else None
        self.q = nn.Linear(latent or width, h * (n + r), bias=False)
        self.kv_a = nn.Linear(width, self.rank + r, bias=False)
        self.kv_b = nn.Linear(self.rank, h * (n + dv), bias=False)
        self.o = nn.Linear(h * dv, width, bias=False)
        self.q_a_norm = Norm(LATENT_NORM, latent) if latent else None
        self.kv_a_norm = Norm(LATENT_NORM, self.rank)
        qk = c["qk_norm"]
        self.q_norm = Norm(qk, n + r) if qk else None
        self.k_norm = Norm(qk, n + r) if qk else None
        rope = c["rope"]
        self.rope = Rope(rope, r) if rope else None
        index = c["indexer"] and not reads(site.ports, "index")
        rotated = rotated_dims(r, rope)
        self.indexer = (
            Indexer(c["indexer"], width, latent or width, rope, rotated)
            if index
            else None
        )

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        query = (
            x
            if self.q_a is None or self.q_a_norm is None
            else self.q_a_norm(self.q_a(x))
        )
        q, k, v = self._heads(x, query, ctx)
        kv = self.cached(k, v, ctx, ctx.cache_slot())
        kept = self.kept(x, query, ctx, kv)
        return self.output(x, self.attend(x, q, kv, ctx, kept))

    def _heads(
        self, x: torch.Tensor, query: torch.Tensor, ctx: Ctx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every head's q, k and v, b t h _, the keys' last r dims shared by all heads."""
        b, t, _ = x.shape
        h, n, r = self.h, self.n, self.r
        q = self.q(query).view(b, t, h, n + r)
        kv_latent, k_rope = self.kv_a(x).split([self.rank, r], -1)
        kv = self.kv_b(self.kv_a_norm(kv_latent)).view(b, t, h, n + self.dv)
        k_nope, v = kv.split([n, self.dv], -1)
        k = torch.cat([k_nope, k_rope[:, :, None].expand(b, t, h, r)], -1)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return self._rotate(q, ctx), self._rotate(k, ctx), v

    def _rotate(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """Rotate the last r dims of queries or keys by the positions."""
        if self.rope is None:
            return x
        n = self.n
        return torch.cat([x[..., :n], self.rope(x[..., n:], ctx.positions)], -1)
