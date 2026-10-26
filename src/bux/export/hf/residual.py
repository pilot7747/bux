"""The residual algebras, exported: what a residual branch outside any other does
under ``residual = ...`` instead of ``x + f(x)`` (bux's
`bux.architecture.residual`), and ``merge``, which reads a widened stream back into
one.

The generated code calls a branch's module around the branch's body:
``h, kept = branch.read(x, ctx)`` gives the body its input, and
``branch.write_back(kept, f(h))`` the stream after the branch. hc, mhc and gr widen
the stream into n copies, ``b t n d``; attnres holds the embedding and every branch
output so far, ``b t 1+L d``; vpgate keeps one stream. All of it is per token, so
decoding keeps no state of its own.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Norm, Site

#: In Sinkhorn's sums and added to mhc's read: DeepSeek-V4's hc_eps.
EPS = 1e-6


def widening(residual: dict | None) -> int | None:
    """Return how many copies of the stream the algebra's widened stream starts as:
    n for hc, mhc and gr, 1 for attnres (the embedding, its first source); None when
    it does not widen the stream."""
    if residual is None:
        return None
    if residual["type"] in ("hc", "mhc", "gr"):
        return residual["expansion"]
    return 1 if residual["type"] == "attnres" else None


def widen(x: torch.Tensor, n: int) -> torch.Tensor:
    """Return n copies of x, b t d: the widened stream that starts from it."""
    return x[:, :, None].expand(-1, -1, n, -1)


def _norm(kind: str, eps: float, affine: bool = True) -> dict:
    """Return the config of an algebra's own norm, as bux's ``NormalizationConfig``."""
    return {
        "type": kind,
        "eps": eps,
        "affine": affine,
        "bias": False,
        "zero_centered": False,
        "scale_input": False,
    }


def mix(read: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return ``sum_j read_j x_j`` of the n streams x, b t n d, by read, b t n."""
    return torch.einsum("btn,btnd->btd", read, x)


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Return the doubly stochastic matrix Sinkhorn-Knopp makes of ``exp(logits)``,
    as bux's ``sinkhorn``: a softmax over every row, the columns divided by their
    sums, then ``iters - 1`` more rounds of rows and columns."""
    m = logits.softmax(-1) + EPS
    m = m / (m.sum(-2, keepdim=True) + EPS)
    for _ in range(iters - 1):
        m = m / (m.sum(-1, keepdim=True) + EPS)
        m = m / (m.sum(-2, keepdim=True) + EPS)
    return m


def pending(ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
    """Return the coefficients a single-pass read uses: those the last branch
    computed, or stream 0 alone before any branch."""
    if ctx.pending_read is not None:
        return ctx.pending_read
    first = torch.zeros(x.shape[2], dtype=x.dtype, device=x.device)
    first[0] = 1
    return first.expand(x.shape[:3])


def _single_pass_read(computed: torch.Tensor, x: torch.Tensor, ctx: Ctx) -> Any:
    """Return the `pending` coefficients, and leave the ones this branch computed for
    the next read."""
    read = pending(ctx, x)
    ctx.pending_read = computed
    return read


class Dynamic(nn.Module):
    """bux's ``_Dynamic``, the term hc's weights add with `dynamic`:
    ``s tanh(norm(H) W)`` at every stream of H, the norm a layernorm without
    parameters."""

    def __init__(self, d: int, columns: int, eps: float):
        super().__init__()
        self.norm = Norm(_norm("layernorm", eps, affine=False), d)
        self.proj = nn.Parameter(torch.zeros(d, columns))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.norm(x) @ self.proj)


class HyperConnections(nn.Module):
    """bux's ``hc``: f reads ``A_m^T H`` of the n streams H, and the branch returns
    ``A_r^T H + B^T f(h)``, ``A = [A_m A_r]`` and B static, plus a term of the stream
    with `dynamic`. Single-pass, f reads with the A_m the branch before computed.
    Without a body, it is the read alone, merge's."""

    bux_residuals = ("hc",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        n, eps = c["expansion"], c["norm_eps"]
        columns = n + 1 if body else 1
        self.single_pass = c["single_pass"]
        self.static_alpha = nn.Parameter(torch.zeros(n, columns))
        self.static_beta = nn.Parameter(torch.ones(n)) if body else None
        dynamic = c["dynamic"]
        self.dynamic_alpha = Dynamic(d, columns, eps) if dynamic else None
        self.dynamic_beta = Dynamic(d, 1, eps) if dynamic and body else None

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        """Return A at every position, b t n k: ``[A_m A_r]``, or A_m alone."""
        a = self.static_alpha.expand(*x.shape[:3], -1)
        return a if self.dynamic_alpha is None else a + self.dynamic_alpha(x)

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the n streams x, and what `write_back` needs."""
        alpha = self._alpha(x)
        read = alpha[..., 0]
        if self.single_pass:
            read = _single_pass_read(read, x, ctx)
        return mix(read, x), (x, alpha)

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the n streams after the branch, of f's output y."""
        assert self.static_beta is not None
        x, alpha = kept
        beta = self.static_beta.expand(x.shape[:3])
        if self.dynamic_beta is not None:
            beta = beta + self.dynamic_beta(x)[..., 0]
        streams = torch.einsum("btjk,btjd->btkd", alpha[..., 1:], x)
        return streams + beta[..., None] * y[:, :, None]


class ManifoldHyperConnections(nn.Module):
    """bux's ``mhc``: with r the n streams flattened and RMS-normalized, f reads
    ``sum_j pre_j x_j`` and stream k becomes ``sum_j res_jk x_j + post_k f(h)``, the
    coefficients ``a r W + b`` (b alone without `dynamic`) through a sigmoid, twice
    a sigmoid, and `sinkhorn`. Single-pass, f reads with the pre the branch before
    computed. Without a body, it is the read alone, merge's."""

    bux_residuals = ("mhc",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        n = c["expansion"]
        self.n, self.single_pass = n, c["single_pass"]
        self.iters, self.norm_eps = c["sinkhorn_iters"], c["norm_eps"]
        columns = 2 * n + n * n if body else n
        dynamic = c["dynamic"]
        self.proj = nn.Parameter(torch.zeros(n * d, columns)) if dynamic else None
        self.scale = nn.Parameter(torch.zeros(3 if body else 1)) if dynamic else None
        self.bias = nn.Parameter(torch.zeros(columns))

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``a r W + b``, b t k: the columns of pre, then of post, then of res
        row by row."""
        b, t, n, d = x.shape
        if self.proj is None or self.scale is None:
            return self.bias.expand(b, t, -1)
        flat = x.reshape(b, t, n * d)
        rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        widths = (n, n, n * n)
        a = torch.cat([s.expand(w) for s, w in zip(self.scale, widths)])
        return (flat @ self.proj) * rms * a + self.bias

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the n streams x, and what `write_back` needs."""
        logits = self._logits(x)
        read = torch.sigmoid(logits[..., : self.n]) + EPS
        if self.single_pass:
            read = _single_pass_read(read, x, ctx)
        return mix(read, x), (x, logits)

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the n streams after the branch, of f's output y."""
        x, logits = kept
        b, t, n, _ = x.shape
        post = 2 * torch.sigmoid(logits[..., n : 2 * n])
        res = sinkhorn(logits[..., 2 * n :].reshape(b, t, n, n), self.iters)
        streams = torch.einsum("btjk,btjd->btkd", res, x)
        return streams + post[..., None] * y[:, :, None]


class GatedResidual(nn.Module):
    """bux's ``gr``: of the n streams, each RMS-normalized with a gain of its own into
    r_j, f reads ``mean_j g_j r_j`` for the gate ``g = sigmoid(W_u silu(W_d r / n))``,
    and the branch adds f to stream j by ``2 sigmoid(W_w r / n)_j``. Without a body,
    it is the read alone, merge's."""

    bux_residuals = ("gr",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        n, rank = c["expansion"], c["rank"]
        self.norm = Norm(_norm("rmsnorm", c["norm_eps"]), (n, d))
        self.down = nn.Linear(n * d, rank, bias=False)
        self.up = nn.Linear(rank, n * d, bias=False)
        self.write = nn.Linear(n * d, n, bias=False) if body else None

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the n streams x, and what `write_back` needs."""
        r = self.norm(x)
        b, t, n, d = r.shape
        gate = self.up(F.silu(self.down(r.reshape(b, t, n * d)) / n))
        return (torch.sigmoid(gate).reshape(r.shape) * r).mean(2), (x, r)

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the n streams after the branch, of f's output y."""
        assert self.write is not None
        x, r = kept
        b, t, n, d = x.shape
        weight = 2 * torch.sigmoid(self.write(r.reshape(b, t, n * d)) / n)
        return x + weight[..., None] * y[:, :, None]


def sources(state: torch.Tensor, block_size: int | None) -> torch.Tensor:
    """Return what attnres reads from, as bux's ``sources``: the embedding, then every
    branch output, or with `block_size` the outputs summed in blocks of that many,
    the block under way included once it has one."""
    if block_size is None:
        return state
    b, t, s, d = state.shape
    outputs = state[:, :, 1:]
    full = (s - 1) // block_size
    blocks = outputs[:, :, : full * block_size].reshape(b, t, full, block_size, d)
    parts = [state[:, :, :1], blocks.sum(3)]
    if full * block_size < s - 1:
        parts.append(outputs[:, :, full * block_size :].sum(2, keepdim=True))
    return torch.cat(parts, 2)


class AttentionResidual(nn.Module):
    """bux's ``attnres``: f reads the softmax over the `sources` v of their scores
    ``dot(w, rmsnorm(v))``, for a query w of its own, and its output joins the
    state. Without a body, it is the read alone, merge's."""

    bux_residuals = ("attnres",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d))
        self.key_norm = Norm(_norm("rmsnorm", c["norm_eps"]), d)
        self.block_size = c["block_size"]

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the state x, and what `write_back` needs."""
        v = sources(x, self.block_size)
        scores = torch.einsum("btmd,d->btm", self.key_norm(v), self.query)
        return torch.einsum("btm,btmd->btd", scores.softmax(-1), v), x

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the state after the branch: the state, and f's output y after it."""
        return torch.cat([kept, y[:, :, None]], 2)


class ResidualGate(nn.Module):
    """bux's ``vpgate``: x becomes ``a x + b f(x)`` per channel, for
    ``b = sigmoid(g / temperature)`` and ``a = sqrt(max(1 - b^2, eps))``."""

    bux_residuals = ("vpgate",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d))
        self.temperature, self.eps = c["temperature"], c["eps"]

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads, the stream, and what `write_back` needs, the same."""
        return x, x

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the stream after the branch, ``a x + b y``, ``1 - b^2`` computed as
        ``(1 - b)(1 + b)``."""
        w = self.gate / self.temperature
        b = torch.sigmoid(w)
        a = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1 + b), min=self.eps))
        return a * kept + b * y


#: The algebras' branches, by the residual setting's type.
ALGEBRAS: dict[str, type] = {
    kind: cls
    for cls in (
        HyperConnections,
        ManifoldHyperConnections,
        GatedResidual,
        AttentionResidual,
        ResidualGate,
    )
    for kind in cls.bux_residuals
}


class Merge(nn.Module):
    """bux's ``merge``: the widened stream read back into one, by the read a branch
    of the algebra does, with parameters of its own; single-pass hc and mhc read
    with the coefficients the last branch computed, and have none."""

    bux_types = ("merge",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        residual = site.config.bux["residual"]
        single = residual["type"] in ("hc", "mhc") and residual["single_pass"]
        algebra = ALGEBRAS[residual["type"]]
        d = site.config.hidden_size
        self.reader = None if single else algebra(residual, d, body=False)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        if self.reader is None:
            return mix(pending(ctx, x), x)
        return self.reader.read(x, ctx)[0]
