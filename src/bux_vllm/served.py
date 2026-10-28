"""What vLLM serves a model with, layer by layer: each layer's quantization method
and its FP8 GEMM, each attention layer's KV cache dtype and backend, and the routed
experts' backends, counted by kind. It runs in vLLM's workers,
``LLM.apply_model(bux_vllm.served.survey)``, which pickle it by reference: a
function of an importable module, as a script's ``__main__`` is not."""

from __future__ import annotations

from collections import Counter
from typing import Any


def survey(model: Any) -> dict[str, dict[str, int]]:
    """The layers of `model` by kind (its vLLM class and the last part of its name)
    and what serves them, counted: ``methods``, the quantization method and its FP8
    GEMM; ``caches``, an attention layer's KV cache dtype and backend; ``backends``,
    the routed experts' backend and its experts' class."""
    methods: Counter = Counter()
    caches: Counter = Counter()
    backends: Counter = Counter()
    for name, module in model.named_modules():
        kind = f"{type(module).__name__} {name.rsplit('.', 1)[-1]}"
        if (method := getattr(module, "quant_method", None)) is not None:
            methods[f"{kind}: {_method(method)}"] += 1
            if (backend := _experts(method)) is not None:
                backends[backend] += 1
        if (kv := getattr(module, "kv_cache_dtype", None)) is not None:
            backend = getattr(module, "attn_backend", None)
            picked = backend.get_name() if backend is not None else None
            caches[f"{type(module).__name__}: kv {kv} ({picked})"] += 1
    return {
        "methods": dict(methods),
        "caches": dict(caches),
        "backends": dict(backends),
    }


def _method(method: Any) -> str:
    """A quantization method's name, with its FP8 GEMM's if it has one."""
    gemm = getattr(method, "fp8_linear", None)
    return type(method).__name__ + (f" ({type(gemm).__name__})" if gemm else "")


def _experts(method: Any) -> str | None:
    """The backend of a routed experts' method, and its experts' class; None for
    another method."""
    backend = getattr(method, "fp8_backend", None)
    backend = backend or getattr(method, "unquantized_backend", None)
    if backend is None:
        return None
    experts = getattr(getattr(method, "experts_cls", None), "__name__", None)
    return f"{getattr(backend, 'value', backend)} ({experts})"
