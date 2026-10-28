"""What a ``gqa`` layer does to its keys and values besides attending, in plain
PyTorch: its ports, the value residual of its ``v`` port, and the kv heads a rank
keeps under tensor parallelism.

A port is ``[name, tap, mode]``, as ``bux.export.plan`` writes it: ``attn(v=@v1)``
writes the layer's values to the tap ``v1`` (``write``), ``attn(v=$v1)`` reads them
(``read``; ``pop`` takes them off with ``$v1!``).
"""

from __future__ import annotations

import torch

#: The ports a ``gqa`` layer runs in vLLM.
PORTS = ("v",)


def reads(ports: list, name: str) -> tuple[str, bool] | None:
    """The tap the layer reads its port `name` from, and whether it pops it; None
    if it does not read that port."""
    return next(
        (
            (tap, mode == "pop")
            for n, tap, mode in ports
            if n == name and mode != "write"
        ),
        None,
    )


def writes(ports: list, name: str) -> list[str]:
    """The taps the layer pushes its port `name` to."""
    return [tap for n, tap, mode in ports if n == name and mode == "write"]


def check_ports(ports: list) -> None:
    """Raise for the ports of a ``gqa`` layer that vLLM does not run.

    Raises:
        NotImplementedError: It has a ``kv`` (cross-layer KV sharing) or ``index``
            port.
    """
    others = sorted({name for name, _, _ in ports} - set(PORTS))
    if others:
        raise NotImplementedError(f"gqa: the {', '.join(others)} port not in vLLM yet")


def mix_values(
    kind: str, w: torch.Tensor | None, v: torch.Tensor, v1: torch.Tensor
) -> torch.Tensor:
    """bux's value residual ``a v + b v1`` of the layer's values `v` and those `v1`
    it reads from its ``v`` port, as ``value_residual`` says.

    Args:
        kind: ``fixed`` (a = b = 0.5), ``learned`` (a, b = w), or ``convex``
            (a = w, b = 1 - w).
        w: The learned weights, float32: two, one, or None for ``fixed``.
        v: The layer's values.
        v1: The values it reads, of v's shape.

    Returns:
        The mixed values in v's dtype; learned weights mix in float32, as bux's
        float32 weights do.
    """
    if w is None:
        return 0.5 * v + 0.5 * v1
    a, b = (w[0], w[1]) if kind == "learned" else (w, 1 - w)
    return (a * v.float() + b * v1.float()).to(v.dtype)


def kv_heads(kv: int, tp: int, rank: int) -> tuple[int, int]:
    """The first kv head of `kv` a rank of `tp` keeps, and how many: a share each,
    or one head copied on each of ``tp // kv`` ranks where there are fewer heads
    than ranks (as vLLM's ``QKVParallelLinear`` splits them)."""
    if kv >= tp:
        return rank * (kv // tp), kv // tp
    return rank // (tp // kv), 1
