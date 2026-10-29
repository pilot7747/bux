"""A KV cache of one vector per token, read as its key and as its value: ``gqa``'s
values from the keys (``k_eq_v``) where every key is its value times the key norm's
gain, which then goes on the queries instead (`bux_vllm.kv.keys_from_values`).

`KeyedAttention` is vLLM's attention layer with a cache spec of keys alone, half
the page of keys and values, so that vLLM fits twice the tokens in the layer's
memory. Its backend, `KeyedTritonBackend`, is vLLM's Triton attention: its kernels,
metadata and CUDA graphs, with the one paged slab passed as both the key cache and
the value cache, written once a token. An FP8 cache (``--kv-cache-dtype fp8``)
quantizes the slab as vLLM's backend quantizes its keys.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention as PagedAttention
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


class KeyedAttention(PagedAttention):
    """vLLM's attention layer whose cache keeps a token's key alone: its value is
    the same vector. Built with `KeyedTritonBackend`, which reads it as both."""

    def get_kv_cache_spec(self, vllm_config: Any) -> Any:
        spec = super().get_kv_cache_spec(vllm_config)
        return None if spec is None else replace(spec, head_size_v=0)


class KeyedTritonImpl(TritonAttentionImpl):
    """vLLM's Triton attention over a slab of one vector per token and kv head,
    ``[blocks, kv heads, block, head]``, the key and the value both.

    Raises:
        NotImplementedError: A KV cache quantized per token and head, whose scales
            the slab has no room for.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                f"gqa(k_eq_v=true): kv_cache_dtype={self.kv_cache_dtype} with keys "
                "for values (quantized per token and head)"
            )

    def _slab(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """The cache as the kernels take keys: ``[blocks, block, kv heads, head]``,
        in FP8 under an FP8 cache."""
        slab = kv_cache.transpose(1, 2)
        quantized = is_quantized_kv_cache(self.kv_cache_dtype)
        return slab.view(self.fp8_dtype) if quantized else slab

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vLLM's Triton attention, `TritonAttentionImpl.forward`, with the slab as
        the key and the value cache."""
        if output_block_scale is not None:
            raise NotImplementedError("fused block-scale output quantization")
        if attn_metadata is None:  # a profiling run
            return output.fill_(0)
        n, slab = attn_metadata.num_actual_tokens, self._slab(kv_cache)
        per_tensor = self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
        fp8_query = per_tensor and query.dtype == self.fp8_dtype
        shape = (attn_metadata.query_start_loc.shape[0] - 1, slab.shape[2])
        m = attn_metadata
        unified_attention(
            q=query[:n],
            k=slab,
            v=slab,
            out=output[:n],
            cu_seqlens_q=m.query_start_loc,
            max_seqlen_q=m.max_query_len,
            seqused_k=m.seq_lens,
            max_seqlen_k=m.max_seq_len,
            softmax_scale=self.scale,
            causal=m.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=m.block_table,
            softcap=self.logits_soft_cap,
            q_descale=layer._q_scale if fp8_query else None,
            k_descale=layer._k_scale.expand(shape),
            v_descale=layer._k_scale.expand(shape),
            seq_threshold_3D=m.seq_threshold_3D,
            num_par_softmax_segments=m.num_par_softmax_segments,
            softmax_segm_output=m.softmax_segm_output,
            softmax_segm_max=m.softmax_segm_max,
            softmax_segm_expsum=m.softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=m.mm_prefix_range_tensor,
            rswa_prefix_lens=m.rswa_prefix_lens,
            rswa_window=m.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
        )
        return output

    def do_kv_cache_update(
        self,
        layer: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write each token's key, which is its value, into the slab."""
        slab = self._slab(kv_cache)
        triton_reshape_and_cache_flash(
            key,
            key,
            slab,
            slab,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._k_scale,
        )


class KeyedTritonBackend(TritonAttentionBackend):
    """vLLM's Triton attention backend for `KeyedAttention`'s cache."""

    @staticmethod
    def get_impl_cls() -> type[KeyedTritonImpl]:
        return KeyedTritonImpl
