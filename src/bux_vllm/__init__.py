"""bux's exports in vLLM: a plugin that serves the checkpoints `bux.export.huggingface`
writes as native vLLM models, with vLLM's paged attention, fused MoE kernels, Mamba
cache and tensor parallelism.

vLLM cannot run a checkpoint's own model code, so the plugin brings it: a model,
``BuxForCausalLM``, that builds any exported architecture from its ``config.json``
(``bux_vllm.model``), from layers of vLLM's (``basic``, ``attention``, ``moe``,
``ssm``) placed where ``bux.export.plan`` named their weights (``bux_vllm.plan``).
It is a separate package next to bux, not part of it: like ``bux.export.hf``, its
files import nothing from bux, only torch, transformers and vLLM, because it runs in
vLLM's environment. It installs from this folder (``pip install src/bux_vllm``),
and vLLM loads it by its entry point (`register`).
"""


def register() -> None:
    """vLLM's entry point (``vllm.general_plugins``): register the config of
    ``model_type`` ``bux`` with transformers, and ``BuxForCausalLM`` with vLLM."""
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .config import BuxConfig

    AutoConfig.register("bux", BuxConfig, exist_ok=True)
    ModelRegistry.register_model("BuxForCausalLM", f"{__name__}.model:BuxForCausalLM")
