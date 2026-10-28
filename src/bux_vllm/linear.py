"""Linear attention's heads, weights and state as vLLM keeps them: plain PyTorch,
which `bux_vllm.ssm` and `bux_vllm.kda` hand vLLM.

bux's linear-attention layers project q and k to their own heads, each serving a
group of value heads next to each other (``num_v_heads``), and convolve q, k and v
side by side: a conv's taps are (k, C), over the channels of q, k and v in turn.
Under tensor parallelism a rank keeps its share of each: its q heads, its k heads and
its value heads, in that order.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

#: What a rank's output of a fused projection is padded to a multiple of. Inductor
#: pads a matmul's output to a multiple of 8 and slices it back, which leaves its
#: parts strided otherwise than vLLM traced them: one that crosses a split of the
#: graph then fails the next piece's checks of its inputs. 64 is also its
#: comprehensive padding's alignment (128 bytes of bfloat16).
ALIGN = 64


def linear_dims(c: dict) -> tuple[int, int, int, int]:
    """A linear attention's q/k heads, value heads, and their dims."""
    h, dk = c["num_heads"], c["head_dim"]
    return h, c["num_v_heads"] or h, dk, c["v_head_dim"] or dk


def conv_channels(c: dict) -> list[int]:
    """The channels of q, k and v that the conv runs over, side by side."""
    h, hv, dk, dv = linear_dims(c)
    return [h * dk, h * dk, hv * dv]


def conv_kernel(c: dict) -> int:
    """The taps of a linear attention's conv, as vLLM sizes its state by them: 2 for
    no conv, a state of one position that nothing reads, as vLLM shapes no state of
    none."""
    return c["conv_size"] or 2


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


class Projection(NamedTuple):
    """A fused projection, as vLLM's merged column-parallel layer makes it.

    Attributes:
        sizes: Its outputs' sizes over all ranks, by the weights that load them.
        split: A rank's parts of its output, the padding (`aligned`) last.
        parts: How many of them the layer reads.
        whole: The output that every rank makes whole, by its index in `sizes`;
            None for none.
    """

    sizes: list[int]
    split: list[int]
    parts: int
    whole: int | None = None

    def width(self, ranks: int) -> int:
        """A rank's width, as vLLM's layer lays it out: the whole output's size
        times `ranks` (its replicated shard), every size over the ranks."""
        return sum(
            n if i == self.whole else n // ranks for i, n in enumerate(self.sizes)
        )


def aligned(sizes: list[int], split: list[int], ranks: int) -> tuple[list, list]:
    """A fused projection's output sizes, over all `ranks`, and a rank's parts of
    its output, with the columns that make a rank's width a multiple of `ALIGN`
    added last: unused, their weights zero."""
    pad = -sum(split) % ALIGN
    return (sizes + [pad * ranks], split + [pad]) if pad else (sizes, split)


def kda_projection(c: dict, ranks: int) -> Projection:
    """KDA's fused input projection over `ranks`: q, k, v, a dense output gate, the
    decay's down projection, which every rank makes whole, and beta; a rank's
    parts: q, k and v together, the gate, the down projection, beta."""
    _, hv, dk, dv = linear_dims(c)
    gate = [hv * dv] if c["output_gate_rank"] is None else []
    rank = c["gate_rank"] or dk
    sizes = [*conv_channels(c), *gate, rank, hv]
    split = [sum(conv_channels(c)) // ranks, *(n // ranks for n in gate)]
    split += [rank, hv // ranks]
    return Projection(*aligned(sizes, split, ranks), len(split), len(sizes) - 2)


def gdn2_projection(c: dict, ranks: int) -> Projection:
    """GDN-2's fused input projection over `ranks`: q, k, v, the erase gate and the
    write gate; a rank's parts: q, k and v together, the two gates."""
    h, hv, dk, dv = linear_dims(c)
    sizes = [*conv_channels(c), h * dk, hv * dv]
    split = [sum(conv_channels(c)) // ranks, h * dk // ranks, hv * dv // ranks]
    return Projection(*aligned(sizes, split, ranks), len(split))


def deltanet_projection(c: dict, ranks: int) -> Projection:
    """DeltaNet's fused input projection over `ranks`: q, k, v and the step; a
    rank's parts: q, k and v together, the step."""
    _, hv, _, _ = linear_dims(c)
    sizes = [*conv_channels(c), hv]
    split = [sum(conv_channels(c)) // ranks, hv // ranks]
    return Projection(*aligned(sizes, split, ranks), len(split))


def gla_projection(c: dict, ranks: int) -> Projection:
    """GLA's fused input projection over `ranks`: q, k and v; a rank's part: q, k
    and v together."""
    return Projection(
        *aligned(conv_channels(c), [sum(conv_channels(c)) // ranks], ranks), 1
    )


def zero_padding(layer: nn.Module, columns: int) -> None:
    """Zero the weights of a fused projection's last `columns` outputs, its padding
    (`bux_vllm.linear.aligned`), which no checkpoint holds."""
    weight = layer.weight
    assert isinstance(weight, torch.Tensor)
    if columns:
        weight.data[-columns:].zero_()


def slopes(heads: int, layer: int, layers: int) -> torch.Tensor:
    """Lightning attention's decay rates, MiniMax's ``2 ** (-8 i / heads)`` for heads
    i = 1 to heads, scaled by ``1 - layer / (layers - 1)`` (and 1e-5 more), for the
    `layer`-th of a model's `layers` token mixers, as bux's ``slopes``."""
    factor = 1 - layer / (layers - 1 + 1e-5) + 1e-5
    return 2 ** (-8 / heads * torch.arange(1, heads + 1, dtype=torch.float32)) * factor
