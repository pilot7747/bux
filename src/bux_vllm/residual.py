"""The residual algebras in vLLM: what a residual branch outside any other does under
``residual = ...`` instead of ``x + f(x)``, and ``merge``, which reads a widened
stream back into one, as `bux.export.hf.residual` computes them, on the batch's
tokens side by side.

A branch's module runs around the branch's body: ``h, kept = branch.read(x, ctx)``
gives the body its input, and ``branch.write_back(kept, f(h))`` the stream after the
branch. hc, mhc and gr widen the stream into n copies, (T, n, d); attnres holds the
embedding and the branch outputs so far as `Sources`; vpgate keeps one stream. All of
it is per token, so decoding keeps no state of its own. Plain PyTorch, which
`bux_vllm.model` runs and a CPU test checks against the export; in a 16-bit model
hc's, mhc's and attnres's reads and writes run `bux_vllm.streams`' fused kernels
instead (`fused`), which read and write a branch's streams once each.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .common import Ctx, Layer, Norm, Site

#: In Sinkhorn's sums and added to mhc's read: DeepSeek-V4's hc_eps.
EPS = 1e-6
#: Which path hc's, mhc's and attnres's reads and writes take: ``triton``
#: (`bux_vllm.streams`), ``torch``, or unset for the kernels in a 16-bit model on a
#: GPU; read once.
CHOICE = os.environ.get("BUX_VLLM_RESIDUAL_KERNELS")
#: How a fused read weighs the streams (`bux_vllm.streams`): by its own
#: coefficients, by those the branch before computed, or stream 0 alone.
OWN, PENDING, FIRST = 0, 1, 2


def fused() -> bool:
    """Whether a branch built now runs `bux_vllm.streams`' kernels, by `CHOICE` and
    the dtype vLLM builds the model in, torch's default then."""
    if CHOICE is None:
        sixteen = torch.get_default_dtype() in (torch.bfloat16, torch.float16)
        on = sixteen and torch.cuda.is_available()
    else:
        on = CHOICE == "triton"
    if on:
        importlib.import_module(f"{__package__}.streams")  # its ops
    return on


def _pad(n: int) -> int:
    """The power of two a kernel holds `n` in, 2 at least (`bux_vllm.streams.pad`)."""
    return max(2, 1 << (n - 1).bit_length())


def _how(single_pass: bool, ctx: Ctx) -> tuple[int, torch.Tensor | None]:
    """How a fused read weighs the streams, and the coefficients it reads by."""
    if not single_pass:
        return OWN, None
    return (FIRST, None) if ctx.pending_read is None else (PENDING, ctx.pending_read)


def _param(*shape: int, fill: float = 0.0) -> nn.Parameter:
    """A parameter of `shape` that the checkpoint fills."""
    return nn.Parameter(torch.full(shape, fill), requires_grad=False)


def _norm(kind: str, eps: float, width: Any, affine: bool = True) -> Norm:
    """A norm of the algebra's own, as `bux.export.hf.common.norm_config` makes it."""
    c = {"type": kind, "eps": eps, "affine": affine, "bias": False}
    return Norm(c | {"zero_centered": False, "scale_input": False}, width)


def mix(read: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return ``sum_j read_j x_j`` of the n streams x, (T, n, d), by read, (T, n)."""
    return torch.einsum("...n,...nd->...d", read, x)


def mixed(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the n streams of x mixed by m, (T, n, n): stream k is
    ``sum_j m_jk x_j``."""
    return torch.einsum("...jk,...jd->...kd", m, x)


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Return the doubly stochastic matrix Sinkhorn-Knopp makes of ``exp(logits)``:
    a softmax over every row, the columns divided by their sums, then ``iters - 1``
    more rounds of rows and columns."""
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
    # stream 0's one-hot, made on the device: a CPU scalar write would copy in a graph
    first = (torch.arange(x.shape[-2], device=x.device) == 0).to(x.dtype)
    return first.expand(x.shape[:-1])


def _single_pass_read(computed: torch.Tensor, x: torch.Tensor, ctx: Ctx) -> Any:
    """Return the `pending` coefficients, and leave the ones this branch computed for
    the next read."""
    read = pending(ctx, x)
    ctx.pending_read = computed
    return read


@dataclass
class Sources:
    """attnres's state: the embedding and each finished block's sum of branch
    outputs (each output alone without blocks), side by side, (T, m, d), and the sum
    of the block under way, in float32, of its `count` outputs.

    It is what the export's state, the embedding and every output so far, reads
    as (`bux.export.hf.residual.sources`), kept as it grows: a branch adds its output
    to the block under way rather than to a list of every output.
    """

    done: torch.Tensor
    partial: torch.Tensor | None = None
    count: int = 0

    def add(self, y: torch.Tensor, block_size: int | None) -> Sources:
        """Return the state after a branch's output y."""
        if block_size is None:
            return Sources(torch.cat([self.done, y[:, None]], 1))
        partial = y.float() if self.partial is None else self.partial + y.float()
        if self.count + 1 == block_size:
            return Sources(torch.cat([self.done, partial.to(y.dtype)[:, None]], 1))
        return Sources(self.done, partial, self.count + 1)

    def values(self) -> torch.Tensor:
        """Return the sources, (T, m, d): the done ones, then the block under way."""
        if self.partial is None:
            return self.done
        under_way = self.partial.to(self.done.dtype)[:, None]
        return torch.cat([self.done, under_way], 1)

    def __add__(self, other: Sources) -> Sources:
        """Two states of as many outputs, summed source by source."""
        if not isinstance(other, Sources):
            return NotImplemented
        if self.partial is None or other.partial is None:
            return Sources(self.done + other.done)
        return Sources(self.done + other.done, self.partial + other.partial, self.count)

    def __mul__(self, other: float) -> Sources:
        """The state times a constant."""
        if not isinstance(other, (int, float)):
            raise NotImplementedError("attnres: a product of its states in vLLM")
        partial = None if self.partial is None else self.partial * other
        return Sources(self.done * other, partial, self.count)

    __rmul__ = __mul__

    def __neg__(self) -> Sources:
        return self * -1


class Branch(Layer):
    """A residual algebra's branch around a body f: `read` gives f its input, and
    `write_back` the stream after the branch. Built without a body, it is the read
    alone, merge's.

    Args:
        c: The algebra's settings, as JSON.
        d: The model's width.
        body: Whether it has a body to write back.
    """

    bux_residuals: tuple[str, ...] = ()  # the algebras it is a branch of

    def read(self, x: Any, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the stream x, and what `write_back` needs."""
        raise NotImplementedError

    def write_back(self, kept: Any, y: torch.Tensor) -> Any:
        """Return the stream after the branch, of f's output y."""
        raise NotImplementedError


class HyperConnections(Branch):
    """bux's ``hc``: f reads ``A_m^T H`` of the n streams H, and the branch returns
    ``A_r^T H + B^T f(h)``, ``A = [A_m A_r]`` and B static, plus a term of the stream
    with `dynamic`. Single-pass, f reads with the A_m the branch before computed.
    Without a body, it is the read alone, merge's."""

    bux_residuals = ("hc",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        n, eps = c["expansion"], c["norm_eps"]
        columns = n + 1 if body else 1
        self.single_pass, self.eps, self.kernels = c["single_pass"], eps, fused()
        self.static_alpha = _param(n, columns)
        self.static_beta = _param(n, fill=1.0) if body else None
        dynamic = c["dynamic"]
        self.dynamic_alpha = Dynamic(d, columns, eps) if dynamic else None
        self.dynamic_beta = Dynamic(d, 1, eps) if dynamic and body else None

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        """Return A at every token, (T, n, k): ``[A_m A_r]``, or A_m alone."""
        a = self.static_alpha.expand(*x.shape[:-1], -1)
        return a if self.dynamic_alpha is None else a + self.dynamic_alpha(x)

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the n streams x, and what `write_back` needs."""
        if self.kernels:
            return self._fused_read(x, ctx)
        alpha = self._alpha(x)
        read = alpha[..., 0]
        if self.single_pass:
            read = _single_pass_read(read, x, ctx)
        return mix(read, x), (x, alpha)

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the n streams after the branch, of f's output y."""
        assert self.static_beta is not None
        x, alpha = kept
        if self.kernels:  # alpha holds A, then B in column n + 1
            n = x.shape[1]
            return _write(x, y, alpha[:, :n, 1 : n + 1], alpha[:, :n, n + 1], 0)
        beta = self.static_beta.expand(x.shape[:-1])
        if self.dynamic_beta is not None:
            beta = beta + self.dynamic_beta(x)[..., 0]
        return mixed(alpha[..., 1:], x) + beta[..., None] * y[..., None, :]

    def _fused_read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """`read` by `bux_vllm.streams.hc_read`: the coefficients, (T, pad(n),
        pad(k + 1)) in float32, A's and then B's, and the read, in one kernel."""
        t, n, d = x.shape
        k = self.static_alpha.shape[1]
        coef = x.new_empty((t, _pad(n), _pad(k + 1)), dtype=torch.float32)
        h, (how, before) = x.new_empty((t, d)), _how(self.single_pass, ctx)
        a, b = self.dynamic_alpha, self.dynamic_beta
        torch.ops.vllm.bux_hc_read(
            x,
            self.static_alpha,
            self.static_beta,
            None if a is None else a.proj,
            None if b is None else b.proj,
            None if a is None else a.scale,
            None if b is None else b.scale,
            before,
            h,
            coef,
            self.eps,
            how,
        )
        if self.single_pass:
            ctx.pending_read = coef[:, :n, 0]
        return h, (x, coef)


def _write(
    x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, b: torch.Tensor, iters: int
) -> torch.Tensor:
    """The streams after a branch by `bux_vllm.streams.mix_write`, of the streams x
    and the body's output y."""
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    torch.ops.vllm.bux_mix_write(x, y, m, b, out, iters)
    return out


class Dynamic(nn.Module):
    """The term hc's weights add with `dynamic`: ``s tanh(norm(H) W)`` at every
    stream of H, the norm a layernorm without parameters."""

    def __init__(self, d: int, columns: int, eps: float):
        super().__init__()
        self.norm = _norm("layernorm", eps, d, affine=False)
        self.proj = _param(d, columns)
        self.scale = _param(fill=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.norm(x) @ self.proj)


class ManifoldHyperConnections(Branch):
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
        dynamic, self.kernels = c["dynamic"], fused()
        self.proj = _param(n * d, columns) if dynamic else None
        self.scale = _param(3 if body else 1) if dynamic else None
        self.bias = _param(columns)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``a r W + b``, (T, k): the columns of pre, then of post, then of
        res row by row."""
        if self.proj is None or self.scale is None:
            return self.bias.expand(*x.shape[:-2], -1)
        n, d = x.shape[-2:]
        flat = x.reshape(*x.shape[:-2], n * d)
        rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        widths = (n, n, n * n)
        a = torch.cat([s.expand(w) for s, w in zip(self.scale, widths)])
        return (flat @ self.proj) * rms * a + self.bias

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the n streams x, and what `write_back` needs."""
        if self.kernels:
            return self._fused_read(x, ctx)
        logits = self._logits(x)
        read = torch.sigmoid(logits[..., : self.n]) + EPS
        if self.single_pass:
            read = _single_pass_read(read, x, ctx)
        return mix(read, x), (x, logits)

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the n streams after the branch, of f's output y."""
        x, logits = kept
        n = self.n
        if self.kernels:  # Sinkhorn's rounds in the kernel
            res = logits[:, 2 * n : 2 * n + n * n].view(-1, n, n)
            return _write(x, y, res, logits[:, n : 2 * n], self.iters)
        post = 2 * torch.sigmoid(logits[..., n : 2 * n])
        res = logits[..., 2 * n :].reshape(*logits.shape[:-1], n, n)
        return mixed(sinkhorn(res, self.iters), x) + post[..., None] * y[..., None, :]

    def _fused_read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """`read` by `bux_vllm.streams.mhc_read`: the logits, (T, pad(k)) in float32,
        the coefficients the read computed, and the read, in one kernel."""
        t, n, d = x.shape
        logits = x.new_empty((t, _pad(self.bias.shape[0])), dtype=torch.float32)
        computed = x.new_empty((t, _pad(n)), dtype=torch.float32)
        h, (how, before) = x.new_empty((t, d)), _how(self.single_pass, ctx)
        torch.ops.vllm.bux_mhc_read(
            x,
            self.proj,
            self.scale,
            self.bias,
            before,
            h,
            logits,
            computed,
            self.norm_eps,
            how,
        )
        if self.single_pass:
            ctx.pending_read = computed[:, :n]
        return h, (x, logits)


class GatedResidual(Branch):
    """bux's ``gr``: of the n streams, each RMS-normalized with a gain of its own into
    r_j, f reads ``mean_j g_j r_j`` for the gate ``g = sigmoid(W_u silu(W_d r / n))``,
    and the branch adds f to stream j by ``2 sigmoid(W_w r / n)_j``. Without a body,
    it is the read alone, merge's."""

    bux_residuals = ("gr",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        n, rank = c["expansion"], c["rank"]
        self.norm = _norm("rmsnorm", c["norm_eps"], (n, d))
        self.down = _linear(n * d, rank)
        self.up = _linear(rank, n * d)
        self.write = _linear(n * d, n) if body else None

    def read(self, x: torch.Tensor, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the n streams x, and what `write_back` needs."""
        r = self.norm(x)
        n, d = r.shape[-2:]
        gate = self.up(F.silu(self.down(r.reshape(*r.shape[:-2], n * d)) / n))
        return (torch.sigmoid(gate).reshape(r.shape) * r).mean(-2), (x, r)

    def write_back(self, kept: Any, y: torch.Tensor) -> torch.Tensor:
        """Return the n streams after the branch, of f's output y."""
        assert self.write is not None
        x, r = kept
        n, d = x.shape[-2:]
        weight = 2 * torch.sigmoid(self.write(r.reshape(*r.shape[:-2], n * d)) / n)
        return x + weight[..., None] * y[..., None, :]


def _linear(features: int, out: int) -> nn.Linear:
    """A projection without a bias, whole on every rank, that the checkpoint fills."""
    return nn.Linear(features, out, bias=False).requires_grad_(False)


class AttentionResidual(Branch):
    """bux's ``attnres``: f reads the softmax over the sources v (`Sources`) of their
    scores ``dot(w, rmsnorm(v))``, for a query w of its own, and its output joins
    them. Without a body, it is the read alone, merge's."""

    bux_residuals = ("attnres",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        self.query = _param(d)
        self.key_norm = _norm("rmsnorm", c["norm_eps"], d)
        self.block_size, self.kernels = c["block_size"], fused()

    def read(self, x: Sources, ctx: Ctx) -> tuple[torch.Tensor, Any]:
        """Return what f reads of the state x, and what `write_back` needs; by
        `bux_vllm.streams.attnres_read` with the kernels."""
        if self.kernels:
            t, _, d = x.done.shape
            h = x.done.new_empty((t, d))
            gain, eps = self.key_norm.weight, self.key_norm.eps
            torch.ops.vllm.bux_attnres_read(x.done, x.partial, gain, self.query, h, eps)
            return h, x
        v = x.values()
        scores = torch.einsum("...md,d->...m", self.key_norm(v), self.query)
        return torch.einsum("...m,...md->...d", scores.softmax(-1), v), x

    def write_back(self, kept: Sources, y: torch.Tensor) -> Sources:
        """Return the state after the branch, with f's output y."""
        return kept.add(y, self.block_size)


class ResidualGate(Branch):
    """bux's ``vpgate``: x becomes ``a x + b f(x)`` per channel, for
    ``b = sigmoid(g / temperature)`` and ``a = sqrt(max(1 - b^2, eps))``."""

    bux_residuals = ("vpgate",)

    def __init__(self, c: dict, d: int, body: bool = True):
        super().__init__()
        self.gate = _param(d)
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
ALGEBRAS: dict[str, type[Branch]] = {
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


def algebra(residual: dict) -> type[Branch]:
    """Return the class of the branches of the algebra `residual`.

    Raises:
        NotImplementedError: The algebra is not in vLLM yet.
    """
    kind = residual["type"]
    if kind not in ALGEBRAS:
        raise NotImplementedError(f"the residual algebra {kind!r} not in vLLM yet")
    return ALGEBRAS[kind]


def widen(residual: dict | None, x: torch.Tensor) -> Any:
    """Return the stream an embedding's output x, (T, d), outside any branch, starts
    as under the algebra `residual`: n copies of x, (T, n, d), for hc, mhc and gr;
    attnres's `Sources` of x alone; else x."""
    kind = None if residual is None else residual["type"]
    if residual is not None and kind in ("hc", "mhc", "gr"):
        return x[..., None, :].expand(*x.shape[:-1], residual["expansion"], -1)
    return Sources(x[:, None]) if kind == "attnres" else x


class Merge(Layer):
    """bux's ``merge``: the widened stream read back into one, by the read a branch
    of the algebra does, with parameters of its own; single-pass hc and mhc read
    with the coefficients the last branch computed, and have none."""

    bux_types = ("merge",)

    def __init__(self, c: dict, width: Any, site: Site):
        super().__init__()
        residual = site.config.bux["residual"]
        single = residual["type"] in ("hc", "mhc") and residual["single_pass"]
        cls, d = algebra(residual), site.config.bux["d_model"]
        self.reader: Branch | None = None if single else cls(residual, d, body=False)

    def forward(self, x: Any, ctx: Ctx) -> torch.Tensor:
        if self.reader is None:  # the kernels' coefficients are float32
            return mix(pending(ctx, x).to(x.dtype), x)
        return self.reader.read(x, ctx)[0]
