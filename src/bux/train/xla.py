"""XLA's options for a training step: the `compiler` setting.

A preset is a name for options, and the setting's own `options`, written as in
XLA_FLAGS, go over its:

    compiler = compiler(preset=o1)
    compiler = compiler(options="--xla_gpu_threshold_for_windowed_einsum_mib=0")

Options are XLA's debug options and jax's `optimization_level` and
`memory_fitting_level`. They compile the step that takes them, over XLA_FLAGS, so a
run's step can differ from the rest of the process's. Another preset is an entry of
PRESETS:

    xla.PRESETS["mine"] = {"xla_gpu_enable_latency_hiding_scheduler": "true"}
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax

from bux.train.config import CompilerConfig

Options = Mapping[str, str | bool]

# Combine collectives up to this many bytes into one (MaxText's GPU recipe).
_COMBINE = {
    "all_reduce": 128 << 20,
    "all_gather": 128 << 20,
    "reduce_scatter": 64 << 20,
}
_ON = "COLLECTIVE_PIPELINING_MODE_ON"

PRESETS: dict[str, Options] = {
    # XLA's defaults. On Hopper and Blackwell it runs the latency hiding scheduler
    # (LHS) when every collective is one its cost model knows: ragged_all_to_all is not.
    "default": {},
    # Collective pipelining, double-buffered loops, and the LHS for any collective.
    "o1": {"optimization_level": "O1"},
    # The LHS for any collective, alone.
    "lhs": {"xla_gpu_enable_latency_hiding_scheduler": "true"},
    # MaxText's GPU recipe (src/maxtext/configs/gpu/a3/llama_2_7b/1vm.sh).
    "maxtext": {
        "xla_gpu_enable_latency_hiding_scheduler": "true",
        "xla_gpu_enable_triton_gemm": "false",
        "xla_gpu_enable_command_buffer": "",
        "xla_gpu_enable_highest_priority_async_stream": "true",
        **{f"xla_gpu_{c}_combine_threshold_bytes": str(n) for c, n in _COMBINE.items()},
        **{f"xla_gpu_pipeline_{c}": _ON for c in _COMBINE},
        "xla_gpu_enable_while_loop_double_buffering": "true",
        "xla_gpu_enable_all_gather_combine_by_dim": "false",
        "xla_gpu_enable_reduce_scatter_combine_by_dim": "false",
        "xla_disable_hlo_passes": "rematerialization",
    },
    # Tensor parallelism's all-gathers and reduce-scatters as loops of
    # collective-permutes, each beside its share of the matmul (windowed einsum).
    "collective_matmul": {"xla_gpu_threshold_for_windowed_einsum_mib": "0"},
}


def options(config: CompilerConfig) -> dict[str, str | bool]:
    """Return the options `config` compiles a step with: its preset's, then its own."""
    if config.preset not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(f"compiler.preset = {config.preset} is none of {known}")
    flags = (flag.lstrip("-").split("=", 1) for flag in config.options.split())
    given = {**PRESETS[config.preset], **dict(flags)}
    return {name: _value(value) for name, value in given.items()}


def _value(value: str | bool) -> str | bool:
    """A bool for true or false, which XLA reads from a string only as True or False;
    anything else as it is: a number, or an enum's name."""
    return {"true": True, "false": False}.get(str(value).lower(), value)


def jit(fun: Callable[..., Any], options: Options, **kwargs: Any) -> Any:
    """Return ``jax.jit(fun, **kwargs)``, which compiles with XLA's `options`, a call of
    it as its `lower(...).compile()`: the one finds the other's compile in JAX's
    caches."""
    return jax.jit(fun, compiler_options=dict(options) or None, **kwargs)
