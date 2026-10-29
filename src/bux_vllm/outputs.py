"""What a ``gqa`` layer does to its heads' outputs between attending and its output
projection, in plain PyTorch: differential attention's pairs of query heads
(DIFF V2), exclusive self-attention (XSA), and the logits of HyGA's gates. The model's forward runs them under torch.compile, which fuses them
into the kernels around them."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def exclusive(
    out: torch.Tensor, v: torch.Tensor, strength: torch.Tensor | None = None
) -> torch.Tensor:
    """Each head's output y less its part along its query's own value v, as bux's
    ``exclusive`` (XSA): ``y - s (y . u) u`` with ``u = v / max(norm(v), 1e-12)``, in
    float32, s = 1 or `strength` for each head.

    Args:
        out: The heads' outputs, (T, h, dv).
        v: The values of the same tokens, (T, g, dv), a kv head for each group of
            ``h / g`` query heads.
        strength: s for each of the h heads, (h,); None for 1.

    Returns:
        The outputs, (T, h, dv), in their dtype.
    """
    t, h, dv = out.shape
    g = v.shape[1]
    y = out.float().view(t, g, h // g, dv)
    u = v.float().view(t, g, 1, dv)
    u = u * u.square().sum(-1, keepdim=True).clamp(min=1e-24).rsqrt()
    along = (y * u).sum(-1, keepdim=True) * u
    if strength is not None:
        along = along * strength.float().view(g, h // g, 1)
    return (y - along).view(t, h, dv).to(out.dtype)


def by_head(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Each head's x, (T, h, i), times its own matrix of w, (h, i, o)."""
    return torch.einsum("thi,hio->tho", x, w.to(x.dtype))


def hybrid_logits(
    logits: torch.Tensor,
    out: torch.Tensor,
    up: torch.Tensor | None = None,
    down: torch.Tensor | None = None,
    own_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """The gate's logits with HyGA's X- and H-gates, as bux's ``_gate`` makes them.

    Args:
        logits: The input's projection by the gate, (T, h, _): x W, or x W_d under
            a low-rank input gate.
        out: The heads' outputs the gate multiplies, (T, h, dv).
        up: Each head's W_u of a low-rank input gate, (h, r, dv), which makes the
            logits SiLU(x W_d) W_u; None for none.
        down: Each head's W_d of its own output, (h, dv, r'), for the H-gate.
        own_up: Each head's W_u of it, (h, r', dv); the H-gate adds
            SiLU(o W_d) W_u to the logits.

    Returns:
        The logits, (T, h, _), before the gate's head bias and activation.
    """
    if up is not None:
        logits = by_head(F.silu(logits), up)
    if down is not None and own_up is not None:
        logits = logits + by_head(F.silu(by_head(out, down)), own_up)
    return logits


def differential(out: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Each output head's ``a1 - sigmoid(lam) a2`` of its pair of query heads, as
    bux's ``differential`` (DIFF V2), in the outputs' dtype.

    Args:
        out: The query heads' outputs, (T, 2h, dv), each pair side by side.
        lam: The input's projection to a weight a head, (T, h).

    Returns:
        The output heads, (T, h, dv).
    """
    t, hq, dv = out.shape
    pairs = out.view(t, hq // 2, 2, dv)
    return pairs[:, :, 0] - torch.sigmoid(lam).to(out.dtype)[..., None] * pairs[:, :, 1]
