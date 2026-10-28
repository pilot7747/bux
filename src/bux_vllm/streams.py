"""Fused Triton kernels of the residual algebras' reads and writes
(`bux_vllm.residual`), each a token's streams read once and written once.

PyTorch's path materializes a branch's widened stream, (T, n, d), several times
over: a norm, a projection, a batched matmul of n streams for the read, the same
norm again for the write's weights, another for the mix, and the add. Here a
branch reads it in one kernel and writes it in another:

- `hc_read`: hc's coefficients, static plus ``s tanh(norm(x_j) W)`` with
  ``dynamic`` (``norm(x) W`` as ``rstd (x W - mean sum(W))``, its statistics and
  dot products in one pass), and the read ``sum_j r_j x_j``;
- `mhc_read`: mhc's ``a r W + b`` of the flattened streams, and its read;
- `mix_write`: ``x'_k = sum_j M_jk x_j + b_k y``, M and b hc's (``A_r``, B) or
  mhc's (Sinkhorn's matrix of its logits and twice their sigmoid, in the kernel);
- `attnres_read`: attnres's softmax over its sources of
  ``rmsnorm(v) . w``, and their mix.

They compute in float32 from the stream's dtype; ``scripts/vllm_residual.py``
checks them against PyTorch's path on a GPU. They run in a 16-bit model by
default, and under ``BUX_VLLM_RESIDUAL_KERNELS=triton`` in any (``torch`` keeps
PyTorch's path: `bux_vllm.residual.fused`). Each is one of vLLM's custom ops, which
torch.compile keeps whole and CUDA graphs capture.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op

BLOCK_D = 512  # the channels a kernel takes at once, of each stream
MHC_BLOCK = 128  # those mhc's read takes, beside a tile of its projection
SOURCES_BLOCK = 256  # those attnres's read takes, of every source
EPS = 1e-6  # in Sinkhorn's sums and added to mhc's read: bux_vllm.residual.EPS
#: How a read weighs the streams: by its own coefficients, by those the branch
#: before computed (single-pass), or stream 0 alone (single-pass, the first).
OWN, PENDING, FIRST = 0, 1, 2


def pad(n: int) -> int:
    """The power of two that holds `n` in a kernel, 2 at least."""
    return max(2, triton.next_power_of_2(n))


@triton.jit
def _read_weights(own, pend_ptr, sp_t, sp_n, t, j, on_j, MODE: tl.constexpr):
    """The weights of the streams j in a read: `own`, the pending ones, or 1 for
    stream 0."""
    if MODE == 0:
        r = own
    elif MODE == 1:
        r = tl.load(pend_ptr + t * sp_t + j * sp_n, mask=on_j, other=0.0).to(tl.float32)
    else:
        r = tl.where(j == 0, 1.0, 0.0)
    return tl.where(on_j, r, 0.0)


@triton.jit
def _mix_read(x_ptr, sx_t, sx_n, h_ptr, sh_t, r, t, j, on_j, d, BLOCK: tl.constexpr):
    """Store ``sum_j r_j x_j``, a token's read of its streams."""
    for start in range(0, d, BLOCK):
        i = start + tl.arange(0, BLOCK)
        on_i = i < d
        x = tl.load(
            x_ptr + t * sx_t + j[:, None] * sx_n + i[None, :],
            mask=on_j[:, None] & on_i[None, :],
            other=0.0,
        ).to(tl.float32)
        h = tl.sum(r[:, None] * x, 0)
        tl.store(h_ptr + t * sh_t + i, h.to(h_ptr.dtype.element_ty), mask=on_i)


@triton.jit
def _hc_read_kernel(
    x_ptr,
    sx_t,
    sx_n,
    sa_ptr,
    sb_ptr,
    wa_ptr,
    wb_ptr,
    scale_a_ptr,
    scale_b_ptr,
    pend_ptr,
    sp_t,
    sp_n,
    h_ptr,
    sh_t,
    coef_ptr,
    d,
    eps,
    N: tl.constexpr,
    NP: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BETA: tl.constexpr,
    DYNAMIC: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A token's hc coefficients, (NP, KP): A's K columns, then B in column K with
    `BETA`; and its read."""
    t = tl.program_id(0).to(tl.int64)
    j, c = tl.arange(0, NP), tl.arange(0, KP)
    on_j, alpha, beta = j < N, c < K, c == K  # B's column, with BETA
    coef = tl.load(
        sa_ptr + j[:, None] * K + c[None, :],
        mask=on_j[:, None] & alpha[None, :],
        other=0.0,
    ).to(tl.float32)
    if BETA:
        b = tl.load(sb_ptr + j, mask=on_j, other=0.0).to(tl.float32)
        coef += tl.where(beta[None, :], b[:, None], 0.0)
    if DYNAMIC:
        s1, s2 = tl.zeros((NP,), tl.float32), tl.zeros((NP,), tl.float32)
        dots, wsum = tl.zeros((NP, KP), tl.float32), tl.zeros((KP,), tl.float32)
        for start in range(0, d, BLOCK):
            i = start + tl.arange(0, BLOCK)
            on_i = i < d
            x = tl.load(
                x_ptr + t * sx_t + j[:, None] * sx_n + i[None, :],
                mask=on_j[:, None] & on_i[None, :],
                other=0.0,
            ).to(tl.float32)
            s1 += tl.sum(x, 1)
            s2 += tl.sum(x * x, 1)
            for col in tl.static_range(K + BETA):  # A's columns, then B's
                if col < K:
                    w = tl.load(wa_ptr + i * K + col, mask=on_i, other=0.0)
                else:
                    w = tl.load(wb_ptr + i, mask=on_i, other=0.0)
                w = w.to(tl.float32)
                dots += tl.where(
                    c[None, :] == col, tl.sum(x * w[None, :], 1)[:, None], 0.0
                )
                wsum += tl.where(c == col, tl.sum(w, 0), 0.0)
        mean = s1 / d
        rstd = tl.rsqrt(tl.maximum(s2 / d - mean * mean, 0.0) + eps)
        z = rstd[:, None] * (dots - mean[:, None] * wsum[None, :])
        scale = tl.load(scale_a_ptr).to(tl.float32) + tl.zeros((KP,), tl.float32)
        if BETA:
            scale = tl.where(beta, tl.load(scale_b_ptr).to(tl.float32), scale)
        coef += scale[None, :] * (2.0 * tl.sigmoid(2.0 * z) - 1.0)  # s tanh(z)
    tl.store(coef_ptr + t * NP * KP + j[:, None] * KP + c[None, :], coef)
    own = tl.sum(tl.where(c[None, :] == 0, coef, 0.0), 1)
    r = _read_weights(own, pend_ptr, sp_t, sp_n, t, j, on_j, MODE)
    _mix_read(x_ptr, sx_t, sx_n, h_ptr, sh_t, r, t, j, on_j, d, BLOCK)


def hc_read(
    x: torch.Tensor,
    static_alpha: torch.Tensor,
    static_beta: torch.Tensor | None,
    wa: torch.Tensor | None,
    wb: torch.Tensor | None,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    pending: torch.Tensor | None,
    h: torch.Tensor,
    coef: torch.Tensor,
    eps: float,
    mode: int,
) -> None:
    """hc's read of the streams x, (T, n, d), into `h`, (T, d), and its coefficients
    into `coef`, (T, pad(n), pad(k + 1)) in float32: A's k columns (static, plus
    the dynamic term of `wa` and `scale_a` where given), then B (of `static_beta`,
    `wb`, `scale_b`) where given. `mode` weighs the read (`OWN`, `PENDING`: by
    `pending`, (T, n), or `FIRST`)."""
    t, n, d = x.shape
    k = static_alpha.shape[1]
    sp = (0, 0) if pending is None else pending.stride()
    _hc_read_kernel[(t,)](
        x,
        x.stride(0),
        x.stride(1),
        static_alpha.contiguous(),
        x if static_beta is None else static_beta,
        x if wa is None else wa.contiguous(),
        x if wb is None else wb.contiguous(),
        x if scale_a is None else scale_a,
        x if scale_b is None else scale_b,
        coef if pending is None else pending,
        *sp,
        h,
        h.stride(0),
        coef,
        d,
        eps,
        N=n,
        NP=coef.shape[1],
        K=k,
        KP=coef.shape[2],
        BETA=static_beta is not None,
        DYNAMIC=wa is not None,
        MODE=mode,
        BLOCK=BLOCK_D,
        num_warps=4,
    )


@triton.jit
def _mhc_read_kernel(
    x_ptr,
    sx_t,
    sx_n,
    p_ptr,
    scale_ptr,
    bias_ptr,
    pend_ptr,
    sp_t,
    sp_n,
    h_ptr,
    sh_t,
    logits_ptr,
    read_ptr,
    d,
    eps,
    N: tl.constexpr,
    NP: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    DYNAMIC: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A token's mhc logits, (KP,), its read's coefficients ``sigmoid(pre) + EPS``,
    (NP,), and its read."""
    t = tl.program_id(0).to(tl.int64)
    j, c = tl.arange(0, NP), tl.arange(0, KP)
    on_j, on_c = j < N, c < K
    logits = tl.load(bias_ptr + c, mask=on_c, other=0.0).to(tl.float32)
    if DYNAMIC:
        squares, dots = tl.zeros((BLOCK,), tl.float32), tl.zeros((KP,), tl.float32)
        for stream in tl.static_range(N):
            for start in range(0, d, BLOCK):
                i = start + tl.arange(0, BLOCK)
                on_i = i < d
                x = tl.load(x_ptr + t * sx_t + stream * sx_n + i, mask=on_i, other=0.0)
                x = x.to(tl.float32)
                rows = (stream * d + i)[:, None] * K + c[None, :]
                p = tl.load(p_ptr + rows, mask=on_i[:, None] & on_c[None, :], other=0.0)
                squares += x * x
                dots += tl.sum(x[:, None] * p.to(tl.float32), 0)
        rms = tl.rsqrt(tl.sum(squares, 0) / (N * d) + eps)
        group = tl.where(c < N, 0, tl.where(c < 2 * N, 1, 2))
        a = tl.load(scale_ptr + group, mask=on_c, other=0.0).to(tl.float32)
        logits += dots * rms * a
    tl.store(logits_ptr + t * KP + c, logits)
    pre = tl.sum(tl.where(c[None, :] == j[:, None], logits[None, :], 0.0), 1)
    own = tl.where(on_j, tl.sigmoid(pre) + 1e-6, 0.0)
    tl.store(read_ptr + t * NP + j, own)
    r = _read_weights(own, pend_ptr, sp_t, sp_n, t, j, on_j, MODE)
    _mix_read(x_ptr, sx_t, sx_n, h_ptr, sh_t, r, t, j, on_j, d, BLOCK)


def mhc_read(
    x: torch.Tensor,
    proj: torch.Tensor | None,
    scale: torch.Tensor | None,
    bias: torch.Tensor,
    pending: torch.Tensor | None,
    h: torch.Tensor,
    logits: torch.Tensor,
    read: torch.Tensor,
    eps: float,
    mode: int,
) -> None:
    """mhc's read of the streams x, (T, n, d), into `h`, its logits ``a r W + b``
    (``b`` alone without `proj`) into `logits`, (T, pad(k)), and the coefficients
    its read computed, ``sigmoid(pre) + EPS``, into `read`, (T, pad(n)), in float32;
    `mode` weighs the read as `hc_read`'s does."""
    t, n, d = x.shape
    sp = (0, 0) if pending is None else pending.stride()
    _mhc_read_kernel[(t,)](
        x,
        x.stride(0),
        x.stride(1),
        x if proj is None else proj.contiguous(),
        x if scale is None else scale,
        bias,
        read if pending is None else pending,
        *sp,
        h,
        h.stride(0),
        logits,
        read,
        d,
        eps,
        N=n,
        NP=read.shape[1],
        K=bias.shape[0],
        KP=logits.shape[1],
        DYNAMIC=proj is not None,
        MODE=mode,
        BLOCK=MHC_BLOCK,
        num_warps=4,
    )


@triton.jit
def _sinkhorn(lg, mask, ITERS: tl.constexpr, eps: tl.constexpr):
    """The doubly stochastic matrix of ``exp(lg)``, as bux's ``sinkhorn``: a softmax
    over every row, the columns divided by their sums, then ``ITERS - 1`` more
    rounds of rows and columns; zeros outside `mask`."""
    e = tl.exp(lg - tl.max(lg, 1)[:, None])
    e = tl.where(mask, e, 0.0)
    m = tl.where(mask, e / tl.sum(e, 1)[:, None] + eps, 0.0)
    m = m / (tl.sum(m, 0)[None, :] + eps)
    for _ in tl.static_range(ITERS - 1):
        m = m / (tl.sum(m, 1)[:, None] + eps)
        m = m / (tl.sum(m, 0)[None, :] + eps)
    return m


@triton.jit
def _mix_write_kernel(
    x_ptr,
    sx_t,
    sx_n,
    y_ptr,
    sy_t,
    m_ptr,
    sm_t,
    sm_j,
    sm_k,
    b_ptr,
    sb_t,
    sb_k,
    out_ptr,
    so_t,
    so_n,
    d,
    N: tl.constexpr,
    NP: tl.constexpr,
    SINKHORN: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A token's streams after a branch: ``x'_k = sum_j M_jk x_j + b_k y``; with
    `SINKHORN`, M and b are mhc's of its logits there."""
    t = tl.program_id(0).to(tl.int64)
    j, k = tl.arange(0, NP), tl.arange(0, NP)
    on_k = k < N
    mask = (j[:, None] < N) & on_k[None, :]
    at = m_ptr + t * sm_t + j[:, None] * sm_j + k[None, :] * sm_k
    b = tl.load(b_ptr + t * sb_t + k * sb_k, mask=on_k, other=0.0).to(tl.float32)
    if SINKHORN:
        m = _sinkhorn(tl.load(at, mask=mask, other=float("-inf")), mask, ITERS, 1e-6)
        b = 2.0 * tl.sigmoid(b)
    else:
        m = tl.load(at, mask=mask, other=0.0).to(tl.float32)
    for start in range(0, d, BLOCK):
        i = start + tl.arange(0, BLOCK)
        on_i = i < d
        y = tl.load(y_ptr + t * sy_t + i, mask=on_i, other=0.0).to(tl.float32)
        out = b[:, None] * y[None, :]
        for stream in tl.static_range(N):  # stream j's share of each stream k
            x = tl.load(x_ptr + t * sx_t + stream * sx_n + i, mask=on_i, other=0.0)
            row = tl.sum(tl.where(j[:, None] == stream, m, 0.0), 0)
            out += row[:, None] * x.to(tl.float32)[None, :]
        tl.store(
            out_ptr + t * so_t + k[:, None] * so_n + i[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=on_k[:, None] & on_i[None, :],
        )


def mix_write(
    x: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    iters: int,
) -> None:
    """The streams after a branch into `out`, (T, n, d), of the streams x and the
    body's output y, (T, d): ``sum_j m_jk x_j + b_k y`` for m, (T, n, n), and b,
    (T, n); with `iters`, m and b are mhc's logits, of Sinkhorn's matrix of that
    many rounds and twice the sigmoid."""
    t, n, d = x.shape
    y = y.contiguous()
    _mix_write_kernel[(t,)](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        m,
        *m.stride(),
        b,
        *b.stride(),
        out,
        out.stride(0),
        out.stride(1),
        d,
        N=n,
        NP=pad(n),
        SINKHORN=iters > 0,
        ITERS=max(iters, 1),
        BLOCK=BLOCK_D,
        num_warps=4,
    )


@triton.jit
def _attnres_read_kernel(
    src_ptr,
    ss_t,
    ss_m,
    part_ptr,
    sp_t,
    g_ptr,
    q_ptr,
    h_ptr,
    sh_t,
    m,
    d,
    eps,
    MP: tl.constexpr,
    PARTIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A token's attnres read: the softmax over its m sources, and the block under
    way with `PARTIAL`, of ``rmsnorm(v) . w``, and their mix."""
    t = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, MP)
    on_s = s < m
    s2, dot = tl.zeros((MP,), tl.float32), tl.zeros((MP,), tl.float32)
    p2, pdot = tl.zeros((BLOCK,), tl.float32), tl.zeros((BLOCK,), tl.float32)
    for start in range(0, d, BLOCK):
        i = start + tl.arange(0, BLOCK)
        on_i = i < d
        gw = tl.load(g_ptr + i, mask=on_i, other=0.0).to(tl.float32)
        gw *= tl.load(q_ptr + i, mask=on_i, other=0.0).to(tl.float32)
        v = tl.load(
            src_ptr + t * ss_t + s[:, None] * ss_m + i[None, :],
            mask=on_s[:, None] & on_i[None, :],
            other=0.0,
        ).to(tl.float32)
        s2 += tl.sum(v * v, 1)
        dot += tl.sum(v * gw[None, :], 1)
        if PARTIAL:
            pv = tl.load(part_ptr + t * sp_t + i, mask=on_i, other=0.0)
            pv = pv.to(h_ptr.dtype.element_ty).to(tl.float32)  # the export's sum
            p2 += pv * pv
            pdot += pv * gw
    score = tl.where(on_s, tl.rsqrt(s2 / d + eps) * dot, float("-inf"))
    top = tl.max(score, 0)
    if PARTIAL:
        under_way = tl.rsqrt(tl.sum(p2, 0) / d + eps) * tl.sum(pdot, 0)
        top = tl.maximum(top, under_way)
    e = tl.where(on_s, tl.exp(score - top), 0.0)
    total = tl.sum(e, 0)
    if PARTIAL:
        e_part = tl.exp(under_way - top)
        total += e_part
    for start in range(0, d, BLOCK):
        i = start + tl.arange(0, BLOCK)
        on_i = i < d
        v = tl.load(
            src_ptr + t * ss_t + s[:, None] * ss_m + i[None, :],
            mask=on_s[:, None] & on_i[None, :],
            other=0.0,
        ).to(tl.float32)
        h = tl.sum(e[:, None] * v, 0)
        if PARTIAL:
            pv = tl.load(part_ptr + t * sp_t + i, mask=on_i, other=0.0)
            h += e_part * pv.to(h_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            h_ptr + t * sh_t + i, (h / total).to(h_ptr.dtype.element_ty), mask=on_i
        )


def attnres_read(
    done: torch.Tensor,
    partial: torch.Tensor | None,
    gain: torch.Tensor,
    query: torch.Tensor,
    h: torch.Tensor,
    eps: float,
) -> None:
    """attnres's read into `h`, (T, d), of its sources `done`, (T, m, d), and the
    sum of the block under way, `partial`, (T, d) in float32, where there is one:
    the softmax of ``rmsnorm(v) . query``, the norm's gain `gain`, over them."""
    t, m, d = done.shape
    _attnres_read_kernel[(t,)](
        done,
        done.stride(0),
        done.stride(1),
        h if partial is None else partial,
        0 if partial is None else partial.stride(0),
        gain,
        query,
        h,
        h.stride(0),
        m,
        d,
        eps,
        MP=pad(m),
        PARTIAL=partial is not None,
        BLOCK=SOURCES_BLOCK,
        num_warps=4,
    )


direct_register_custom_op("bux_hc_read", hc_read, mutates_args=["h", "coef"])
direct_register_custom_op(
    "bux_mhc_read", mhc_read, mutates_args=["h", "logits", "read"]
)
direct_register_custom_op("bux_mix_write", mix_write, mutates_args=["out"])
direct_register_custom_op("bux_attnres_read", attnres_read, mutates_args=["h"])
