"""bux's exports in vLLM: a plugin that serves the checkpoints `bux.export.huggingface`
writes as native vLLM models, with vLLM's paged attention, fused MoE kernels, Mamba
cache and tensor parallelism.

vLLM cannot run a checkpoint's own model code, so the plugin brings it: a model,
``BuxForCausalLM``, that builds any exported architecture from its ``config.json``
(`bux.export.vllm.model`), from layers of vLLM's (`bux.export.vllm.basic`,
``attention``, ``moe``, ``ssm``) placed where `bux.export.plan` named their weights
(`bux.export.vllm.plan`). Like `bux.export.hf`, its files import nothing of bux, only
torch, transformers and vLLM: it installs alone, as the package ``bux_vllm``, from
this folder (``pip install src/bux/export/vllm``), and vLLM loads it by its entry
point (`register`).
"""


def register() -> None:
    """vLLM's entry point (``vllm.general_plugins``): register the config of
    ``model_type`` ``bux`` with transformers, and ``BuxForCausalLM`` with vLLM."""
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .config import BuxConfig

    AutoConfig.register("bux", BuxConfig, exist_ok=True)
    ModelRegistry.register_model("BuxForCausalLM", f"{__name__}.model:BuxForCausalLM")
