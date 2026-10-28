"""Linear attention's heads, weights and state as vLLM keeps them: plain PyTorch,
which `bux_vllm.ssm` and `bux_vllm.kda` hand vLLM.

bux's linear-attention layers project q and k to their own heads, each serving a
group of value heads next to each other (``num_v_heads``), and convolve q, k and v
side by side: a conv's taps are (k, C), over the channels of q, k and v in turn.
Under tensor parallelism a rank keeps its share of each: its q heads, its k heads and
its value heads, in that order.
"""

from __future__ import annotations

import torch


def linear_dims(c: dict) -> tuple[int, int, int, int]:
    """A linear attention's q/k heads, value heads, and their dims."""
    h, dk = c["num_heads"], c["head_dim"]
    return h, c["num_v_heads"] or h, dk, c["v_head_dim"] or dk


def conv_channels(c: dict) -> list[int]:
    """The channels of q, k and v that the conv runs over, side by side."""
    h, hv, dk, dv = linear_dims(c)
    return [h * dk, h * dk, hv * dv]


def conv_weight(tensor: torch.Tensor) -> torch.Tensor:
    """bux's conv taps, (k, C), as vLLM's conv weight, (C, 1, k)."""
    return tensor.T.unsqueeze(1).contiguous()


def conv_shard(
    tensor: torch.Tensor, sizes: list[int], rank: int, ranks: int
) -> torch.Tensor:
    """This rank's channels of bux's conv taps, (k, C) over `sizes` side by side, as
    vLLM's conv weight, (C / ranks, 1, k): its share of each part in turn."""
    mine = [part.chunk(ranks, -1)[rank] for part in tensor.split(sizes, -1)]
    return conv_weight(torch.cat(mine, -1))


def to_value_heads(x: torch.Tensor, heads: int, group: int) -> torch.Tensor:
    """q or k, (T, heads * d), as (1, T, heads * group, d): each head repeated for
    the group of value heads it serves, as the kernels take them."""
    y = x.reshape(1, x.shape[0], heads, -1)
    return y.repeat_interleave(group, 2) if group > 1 else y.contiguous()
