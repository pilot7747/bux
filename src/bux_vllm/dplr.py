"""bux's delta rule (`bux_vllm.delta`) on flash-linear-attention's DPLR kernels, where
the package is installed (``pip install flash-linear-attention``).

fla's rule, ``S_t = diag(exp(gk_t)) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T`` read
as ``o_t = scale S_t^T q_t``, is bux's with bux's a, b and log decay as fla's a, b
and gk, and a scale of 1, since bux's layers scale their queries themselves. Its
state per head is (dk, dv), as bux's. Its kernels take q, k, v, a and b in one dtype,
the model's; the decay and the state stay in float32.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch

#: The dtypes fla's kernels run the rule in.
HALF = (torch.bfloat16, torch.float16)


@cache
def kernels() -> tuple[Callable, Callable] | None:
    """fla's DPLR kernels, by chunks and recurrent; None where fla is not installed."""
    try:
        from fla.ops.generalized_delta_rule import (
            chunk_dplr_delta_rule,
            fused_recurrent_dplr_delta_rule,
        )
    except ImportError:
        return None
    return chunk_dplr_delta_rule, fused_recurrent_dplr_delta_rule


def runs(dtype: torch.dtype) -> bool:
    """Whether fla's kernels run the rule of a model in `dtype`: installed, and a
    dtype they take."""
    return dtype in HALF and kernels() is not None


def inputs(xs: list[torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """bux's q, k, v, log decay, a and b as fla's arguments: in `dtype`, the decay in
    float32."""
    q, k, v, g, a, b = xs
    out = {"q": q, "k": k, "v": v, "a": a, "b": b}
    return {n: x.to(dtype) for n, x in out.items()} | {"gk": g.float()}


def step(
    xs: list[torch.Tensor], state: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """One position of each sequence, ``xs`` (B, h, d), from its state (B, h, dk,
    dv), by fla's recurrent kernel: the outputs, (B, h, dv), and the new states."""
    found = kernels()
    assert found is not None
    o, after = found[1](
        **inputs([x[:, None] for x in xs], dtype),
        scale=1.0,
        initial_state=state,
        output_final_state=True,
    )
    return o[:, 0], after


def prompts(
    xs: list[torch.Tensor],
    starts: torch.Tensor,
    state: torch.Tensor,
    dtype: torch.dtype,
    floor: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rule over prompts side by side, ``xs`` (N, h, d) with sequence i from
    ``starts[i]``, from their states (B, h, dk, dv), by fla's chunked kernel; with a
    `floor` on the log decay above 0, its bound licenses fla's tensor-core scheme
    (fla takes no bound of 0, which DeltaNet's, with no decay, would be). Returns
    the outputs, (N, h, dv), and the last states."""
    found = kernels()
    assert found is not None
    o, after = found[0](
        **inputs([x[None] for x in xs], dtype),
        scale=1.0,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=starts,
        lower_bound=-floor if floor else None,
    )
    return o[0], after
