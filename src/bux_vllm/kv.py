"""What a ``gqa`` layer does to its queries, keys and values besides attending, in
plain PyTorch: its ports, the value embeddings of its ``ve`` port, the value residual
of its ``v`` port, the kv heads a rank keeps under tensor parallelism, its conv's
taps (``kv_conv_size``) as vLLM's conv kernels take them, and the queries' scale by
their position.

A port is ``[name, tap, mode]``, as ``bux.export.plan`` writes it: ``attn(v=@v1)``
writes the layer's values to the tap ``v1`` (``write``), ``attn(v=$v1)`` reads them
(``read``; ``pop`` takes them off with ``$v1!``). The ``kv`` port shares a layer's
KV cache with the layers that read it, which attend to the writer's keys and values
and keep none of their own: which cache each call reads is known before the model
runs (`sources`), as vLLM's cross-layer KV sharing needs it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: The ports a ``gqa`` layer runs in vLLM.
PORTS = ("v", "kv", "index", "ve")


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


def check_ports(c: dict, ports: list) -> None:
    """Raise for the ports of the ``gqa`` layer `c` that vLLM does not run.

    Raises:
        NotImplementedError: It reads ``kv`` and has a ``v`` port or an indexer, as
            the export does not take either, or reads ``index`` too, whose sparse
            attention over another's keys vLLM does not share.
    """
    others = sorted({name for name, _, _ in ports} - set(PORTS))
    if others:
        raise NotImplementedError(f"gqa: the {', '.join(others)} port not in vLLM yet")
    if not reads(ports, "kv"):
        return
    if (
        any(n in ("v", "ve") for n, _, _ in ports)
        or c["indexer"]
        or reads(ports, "index")
    ):
        raise NotImplementedError("gqa: a v port or an indexer beside reading kv")


def sources(ports: list[list]) -> list[int | None]:
    """The call each call attends to the keys and values of through its ``kv``
    port, by its index, from each call's ports in the order the calls run: the one
    whose push the tap it reads holds, as bux's taps run (``$kv`` the last push,
    ``$kv!`` taken off); None for a call that reads none. A reader that writes the
    port too pushes its writer's keys and values on."""
    stacks: dict[str, list[int]] = {}
    out: list[int | None] = []
    for i, call in enumerate(ports):
        source, read = None, reads(call, "kv")
        if read is not None:
            stack = stacks[read[0]]
            source = stack.pop() if read[1] else stack[-1]
        for tap in writes(call, "kv"):
            stacks.setdefault(tap, []).append(i if source is None else source)
        out.append(source)
    return out


def check_shared(reader: dict, writer: dict, width: int) -> None:
    """Raise for keys and values of the ``writer`` layer's call that vLLM cannot
    share with the ``reader``'s (their configs, on a stream `width` wide).

    Raises:
        NotImplementedError: The writer is no dense ``gqa`` (vLLM shares the KV
            caches of its own attention layers alone), their kv heads or dims
            differ, or the writer keeps fewer keys than the reader's window.
    """
    if writer["type"] != "gqa" or writer["indexer"]:
        raise NotImplementedError("gqa: a kv port from other than a dense gqa layer")
    heads = [(c["num_kv_heads"], *head_dims(c, width)) for c in (reader, writer)]
    if heads[0] != heads[1]:
        raise NotImplementedError(f"gqa: a kv port of {heads[1]} to {heads[0]} heads")
    mine, kept = reader["sliding_window_size"], writer["sliding_window_size"]
    if kept is not None and (mine is None or mine > kept):
        raise NotImplementedError(f"gqa: a window of {mine} over {kept} shared keys")


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


def add_value_embedding(
    x: torch.Tensor,
    v: torch.Tensor,
    e: torch.Tensor,
    w: torch.Tensor | None,
    heads: tuple[int, int],
) -> torch.Tensor:
    """bux's value embeddings: the layer's values `v` plus the ``ve`` port's rows,
    by a gate per kv head of the first channels of its input and of the rows,
    ``v + 2 sigmoid(W [x[:n], e[:n]]) e``; ``v + e`` without one.

    Args:
        x: The layer's input, (T, d).
        v: This rank's values, (T, g dv): its kv heads'.
        e: The rows of every kv head, (T, kv dv), in the model's dtype.
        w: The gate's weight of every kv head, (kv, 2n); None for no gate.
        heads: This rank's first kv head and how many it keeps, g.

    Returns:
        The values, as v.
    """
    first, g = heads
    dv = v.shape[-1] // g
    rows = e[:, first * dv : (first + g) * dv]
    if w is None:
        return v + rows
    n = w.shape[1] // 2
    gate = F.linear(torch.cat([x[:, :n], e[:, :n]], -1), w[first : first + g])
    return v + (2 * torch.sigmoid(gate)).repeat_interleave(dv, -1) * rows


def kv_heads(kv: int, tp: int, rank: int) -> tuple[int, int]:
    """The first kv head of `kv` a rank of `tp` keeps, and how many: a share each,
    or one head copied on each of ``tp // kv`` ranks where there are fewer heads
    than ranks (as vLLM's ``QKVParallelLinear`` splits them)."""
    if kv >= tp:
        return rank * (kv // tp), kv // tp
    return rank // (tp // kv), 1


def head_dims(c: dict, width: int) -> tuple[int, int]:
    """The dims of a key head and of a value head of the ``gqa`` layer `c`."""
    dk = c["head_dim"] or width // c["num_heads"]
    return dk, c["v_head_dim"] or dk


def conv_channels(c: dict, width: int, tp: int) -> int:
    """The channels of the conv on keys and values a rank of `tp` keeps: its kv
    heads' keys, then their values."""
    dk, dv = head_dims(c, width)
    return kv_heads(c["num_kv_heads"], tp, 0)[1] * (dk + dv)


def conv_rows(taps: torch.Tensor, dims: int, first: int, n: int) -> torch.Tensor:
    """bux's taps of a conv on keys or values, (k, g dims), cut to the `n` kv heads
    from `first` on, as rows of vLLM's conv weight, (n dims, k)."""
    return taps[:, first * dims : (first + n) * dims].T.contiguous()


def query_scale(c: dict, positions: torch.Tensor) -> torch.Tensor | None:
    """The factor on each query, (T,) in float32: `power_scale`'s times
    `length_scale`'s, which bux puts on the logits before soft-capping and so
    scales them as on the queries; None for neither. With a relative bias, which
    the length scale multiplies too, the attention takes that one itself."""
    power, length = power_scale(c, positions), length_scale(c, positions)
    if power is None or length is None:
        return power if length is None else length
    return power * length


def power_scale(c: dict, positions: torch.Tensor) -> torch.Tensor | None:
    """Beam's factor on each query, ``(1 + n / power_scale_length) ** power_scale``
    at its position n counted from 1, (T,) in float32; None without one."""
    if c["power_scale"] is None:
        return None
    n = positions.float() + 1
    return torch.exp(c["power_scale"] * torch.log1p(n / c["power_scale_length"]))


def length_scale(c: dict, positions: torch.Tensor) -> torch.Tensor | None:
    """Inkling's factor on each query's logits, ``1 + log_scale ln(max(n /
    log_scale_floor, 1))`` at its position n counted from 1, (T,) in float32;
    None without one."""
    if c["log_scale"] is None:
        return None
    ahead = ((positions.float() + 1) / c["log_scale_floor"]).clamp(min=1.0)
    return 1 + c["log_scale"] * torch.log(ahead)
