"""MLA's latent cache in plain PyTorch: the rows of the requests' cached context, as
vLLM's ``gather_and_maybe_dequant_cache`` gathers them for sizes its kernel lacks.

vLLM's MLA attends to a prompt's context in the cache (its earlier chunks, a cached
prefix) by gathering the context's latents into a workspace, then projecting them
up. Its CUDA kernel takes caches of 320, 512 or 576 values a token (`KERNEL_SIZES`)
alone; `gather` does the same for any size, in the same layout, so `bux_vllm.mla`
calls it for the others.
"""

from __future__ import annotations

import torch

#: The cache's values a token that vLLM 0.32's gather kernel takes
#: (``csrc/cache_kernels.cu``).
KERNEL_SIZES = (320, 512, 576)

#: The cache dtypes kept as they are, and those dequantized by the scale.
PLAIN = ("auto", "bfloat16", "float16")
FP8 = ("fp8", "fp8_e4m3", "fp8_e5m2")


def gather(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather the cached context of a chunk of requests into `dst`, as vLLM's
    ``gather_and_maybe_dequant_cache`` does.

    Row i of `dst` is token i of the chunk: token ``i - cu_seq_lens[s]`` of its
    request ``s = token_to_seq[i]``, at ``seq_starts[s]`` on in the request's
    context, in the cache's block ``block_table[s, p // block_size]``.

    Args:
        src_cache: The paged cache, (blocks, block_size, size), any strides.
        dst: The workspace, (rows, size), of which the first `num_tokens` rows
            are written.
        block_table: Each request's cache blocks, (requests, max_blocks).
        cu_seq_lens: Where each request's tokens start in `dst`, (requests + 1,).
        token_to_seq: The request of each row of `dst`, (num_tokens,) or more.
        num_tokens: The rows to gather.
        kv_cache_dtype: The cache's dtype, as vLLM names it.
        scale: The scale of an fp8 cache's values.
        seq_starts: Where in its context each request's tokens start; None: at 0.

    Raises:
        NotImplementedError: The cache's dtype is neither plain nor fp8.
    """
    if num_tokens == 0:
        return
    seq = token_to_seq[:num_tokens].long()
    rows = torch.arange(num_tokens, device=dst.device)
    at = rows - cu_seq_lens.long()[seq]
    if seq_starts is not None:
        at = at + seq_starts.long()[seq]
    size = src_cache.shape[1]
    blocks = block_table.long()[seq, torch.div(at, size, rounding_mode="floor")]
    dst[:num_tokens] = _values(src_cache[blocks, at % size], kv_cache_dtype, scale)


def _values(
    rows: torch.Tensor, kv_cache_dtype: str, scale: torch.Tensor
) -> torch.Tensor:
    """The cache's `rows` as values: as they are, or an fp8 cache's times its
    scale, in float32.

    Raises:
        NotImplementedError: The cache's dtype is neither plain nor fp8.
    """
    if kv_cache_dtype in PLAIN:
        return rows
    if kv_cache_dtype in FP8:
        return rows.float() * scale.float()
    raise NotImplementedError(f"an MLA cache of {kv_cache_dtype} not in vLLM yet")
