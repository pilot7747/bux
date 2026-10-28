"""What vLLM makes of a bux model's config: its ``ModelArchitectureConfig``, by a
convertor registered for ``model_type`` ``bux``.

vLLM decides that a model runs multi-head latent attention by its model type, from a
list of its own (DeepSeek's, Kimi's and others', ``is_deepseek_mla``), and goes on
from there: ``ModelConfig.use_mla``, the latent KV cache's size per token
(``kv_lora_rank + qk_rope_head_dim``, one head), the MLA backends, and how hybrid
models and KV transfer lay the cache out. The convertor says a bux model is MLA
where its config carries an ``mla`` layer's dims (`bux_vllm.config.latent`), and
leaves the rest to vLLM's own. It changes nothing for any other model type.
"""

from __future__ import annotations

from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from .config import BuxConfig, latent


class BuxArchConvertor(ModelArchConfigConvertorBase):
    """vLLM's reading of a `BuxConfig`: MLA where its attention is ``mla``."""

    def is_deepseek_mla(self) -> bool:
        return latent(self.hf_text_config)


def register() -> None:
    """Make vLLM read every config of ``model_type`` ``bux`` by `BuxArchConvertor`."""
    MODEL_ARCH_CONFIG_CONVERTORS[BuxConfig.model_type] = BuxArchConvertor
