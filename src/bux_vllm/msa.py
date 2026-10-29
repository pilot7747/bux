"""MSA's indexer and attention on vLLM's own kernels, the Triton ones vLLM runs
MiniMax-M3's sparse layers on: each index head's block scores and their top-k
(``minimax_m3_index_score`` and ``minimax_m3_index_topk`` for a prompt chunk,
``minimax_m3_index_decode`` for decoded tokens), and each kv head's group's
attention over its blocks (``minimax_m3_sparse_attn`` and its decoding twin).

bux's MSA indexer is MiniMax-M3's: an index head for each kv head scores every key
it sees by a plain dot product, a block of 128 keys by its best, and the kv head's
group of query heads attends to the group's best ``top_k / 128`` blocks, its own
block among them. The kernels take it where its blocks are 128 keys and its heads
unweighted, with no window, sinks or softcap, index and attention heads of a power
of two of dims, and fewer than 64 blocks a query (`takes`). Their index kernels
span a rank's index heads in one range, which must be a power of two: other counts
are padded with heads of no query (`padded`), whose picks go unread. They rank the blocks as
bux does, but break ties their own way.

The index keys, normalized and rotated, are cached in bfloat16 in head slots after
the call's kv heads (`bux_vllm.dsa.Raw`), and the cache in blocks of 128 tokens, a
block of MSA's to a page, as the kernels read them.
"""

from __future__ import annotations

import functools
import os

import torch

from .picks import Sparse

#: Which path the sparse ops take (`bux_vllm.kernels.CHOICE`).
CHOICE = os.environ.get("BUX_VLLM_SPARSE_KERNELS")
BLOCK = 128  # MiniMax-M3's blocks of keys, one a page of the cache
MOST_BLOCKS = 64  # the blocks a query keeps, fewer than this


def takes(spec: Sparse, ix: dict, kv_heads: int, dims: int, dtype: torch.dtype) -> bool:
    """Whether the kernels pick and attend for a layer of `spec`, its indexer's
    config `ix`, `kv_heads` kv heads of `dims` a key and a value, in a model of
    `dtype` (by `CHOICE`: in a bfloat16 one unless it names a path)."""
    on = dtype == torch.bfloat16 if CHOICE is None else CHOICE == "native"
    blocks = spec.block_size == BLOCK and not spec.mean and spec.per_group
    plain = not spec.relu and not ix["weighted"] and spec.window is None
    plain = plain and spec.sinks is None and spec.softcap is None and not spec.latent
    heads = ix["num_heads"] == kv_heads and _power(ix["head_dim"]) and _power(dims)
    count = 0 < spec.top_k // BLOCK < MOST_BLOCKS
    return on and blocks and plain and heads and count and available()


def _power(n: int, least: int = 16) -> bool:
    """Whether `n` is a power of two of `least` or more."""
    return n >= least and n & (n - 1) == 0


@functools.cache
def available() -> bool:
    """Whether the kernels run here: a CUDA GPU, with vLLM's MiniMax-M3 model."""
    if not torch.cuda.is_available():
        return False
    import importlib.util

    return importlib.util.find_spec("vllm.models.minimax_m3") is not None


def padded(iq: torch.Tensor) -> torch.Tensor:
    """Index queries, (T, g, i), padded with zeros to a power of two of heads, as
    the index kernels take them; their picks for the padding are not read."""
    g = iq.shape[1]
    pad = (1 << (g - 1).bit_length()) - g
    return iq if not pad else torch.nn.functional.pad(iq, (0, 0, 0, pad))


def _scales(kv: torch.Tensor, scales: torch.Tensor) -> tuple:
    """An FP8 cache's keys' and values' `scales`, (2,) float32, as the kernels take
    them: views of one each, which a CUDA graph's capture may read; none else."""
    if kv.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return None, None
    return scales[0], scales[1]


def prompt(
    keys: torch.Tensor,
    iq: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    rows: torch.Tensor,
    n: int,
    attending: tuple[int, float, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """A prompt chunk's outputs, (Q, h, d), and its picks, (g, Q, k) block ids with
    -1 for none: the chunk is the last Q of its request's `n` tokens, its index
    queries `iq`, (Q, g, i), over the request's index `keys`, (blocks, 128, i), and
    its queries `q` over the paged `kv`, (blocks, g, 128, 2 d), by the request's
    block table `rows`, (1, blocks). `attending` is the blocks a query keeps, the
    attention's scale and an FP8 cache's scales (`_scales`)."""
    from vllm.models.minimax_m3.common.ops import (
        minimax_m3_index_score,
        minimax_m3_index_topk,
        minimax_m3_sparse_attn,
    )

    top, scale, scales = attending
    count, g = q.shape[0], kv.shape[1]
    device = q.device
    cu = torch.tensor([0, count], dtype=torch.int32, device=device)
    lens = torch.tensor([n], dtype=torch.int32, device=device)
    before = lens - count
    iq = padded(iq)
    heads = iq.shape[1]
    score = minimax_m3_index_score(iq, keys, rows, cu, lens, before, count, n, heads)
    picked = minimax_m3_index_topk(score, cu, before, count, top, 0, 1)[:g]
    out = torch.empty_like(q)
    args = (picked, rows, cu, lens, before, count, g, scale, out)
    minimax_m3_sparse_attn(q, kv, *args, *_scales(kv, scales))
    return out, picked


def decode(
    keys: torch.Tensor,
    iq: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    table: torch.Tensor,
    lens: torch.Tensor,
    longest: int,
    attending: tuple[int, float, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoded tokens' outputs, (B, h, d), and picks, (g, B, k): each one's index
    query over its request's keys, of `lens` (B,) int32, and its queries over the
    paged `kv` by the block table `table`, (B, blocks), as `prompt` does; `longest`
    is the most keys a request may have (``max_model_len``)."""
    from vllm.models.minimax_m3.common.ops import (
        minimax_m3_index_decode,
        minimax_m3_sparse_attn_decode,
    )

    top, scale, scales = attending
    g, iq = kv.shape[1], padded(iq)
    args = (keys, table, lens, longest, top, 0, 1, iq.shape[1], 1, 1)
    picked = minimax_m3_index_decode(iq, *args)[:g]
    out = torch.empty_like(q)
    args = (picked, table, lens, g, scale, out, 1)
    minimax_m3_sparse_attn_decode(q, kv, *args, *_scales(kv, scales))
    return out, picked


def reserve(q: torch.Tensor, groups: int, longest: int) -> None:
    """Claim, in vLLM's profiling run of `q`'s tokens, the block scores a step's
    prompt chunks make at their most, which the profile counts as they are freed."""
    blocks, heads = -(-longest // BLOCK) + 16, 1 << (groups - 1).bit_length()
    torch.empty(heads * q.shape[0] * blocks, dtype=torch.float32, device=q.device)


def as_picks(
    blocks: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernels' picks, (g, Q, k) block ids with -1 for none, as
    `bux_vllm.picks` keeps them, (Q, g, 128 k): each block's keys, those the query
    at `positions`, (Q,), sees valid."""
    blocks = blocks.transpose(0, 1).long()
    keys = blocks[..., None] * BLOCK + torch.arange(BLOCK, device=blocks.device)
    seen = (blocks >= 0)[..., None] & (keys <= positions[:, None, None, None])
    return keys.flatten(2).clamp(min=0), seen.flatten(2)
