"""What a ``gqa`` layer does to its heads' outputs between attending and its output
projection, in plain PyTorch: exclusive self-attention (XSA). The model's forward
runs it under torch.compile, which fuses it into the kernels around it."""

from __future__ import annotations

import torch


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
