"""A causal depthwise conv over the sequence whose state, each sequence's last
inputs, vLLM keeps in its Mamba cache: the conv of ``gqa``'s ``kv_conv_size`` on its
keys and values, bux's ``conv`` layer (`Conv`) and LFM2's ``short_conv``
(`ShortConv`).

It is one of vLLM's short convs: a layer of its own, with a conv state in each
sequence's cache slot (``MambaBase``), which makes the model a hybrid for vLLM. A
prefill convolves its tokens after the state by ``causal_conv1d_fn``, and a decoding
step by ``causal_conv1d_update``, both vLLM's Triton kernels, which leave the last
inputs in the state. It runs inside vLLM's ``short_conv`` op, which calls the
layer's `CausalConv.forward_cuda` and which vLLM splits its CUDA graphs around, as
around attention.
"""

from __future__ import annotations

from typing import Any

import torch
import vllm.model_executor.layers.mamba.short_conv  # noqa: F401  (its op)
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from . import kv
from .common import Ctx, Layer, Site
from .config import states
from .plan import live, plan, ports


def state_shapes(vllm_config: Any) -> list[tuple]:
    """The shapes of the states of the model's convs on keys and values, on one
    rank: one of each."""
    config = vllm_config.model_config.hf_config
    tp = vllm_config.parallel_config.tensor_parallel_size
    calls = live(plan(config.bux, config.bux_reserved).calls)
    layers = [c.layer for c in calls if c.layer["config"]["type"] == "gqa"]
    convs = {
        (kv.conv_channels(n["config"], n["width"], tp), n["config"]["kv_conv_size"])
        for n in layers
        if "conv" in states(n["config"], ports(n))
    }
    for call in calls:  # the conv layers', on the whole stream, and short convs'
        c, width = call.layer["config"], call.layer["width"]
        if c["type"] == "conv":
            convs.add((width, c["size"]))
        if c["type"] == "short_conv":
            convs.add((width // tp, c["conv_size"]))
    return [state_shape(vllm_config, *conv) for conv in sorted(convs)]


def state_shape(vllm_config: Any, channels: int, size: int) -> tuple:
    """The shape of a conv state of `channels` on one rank, for `size` taps: the last
    ``size - 1`` inputs, and room for the speculative tokens."""
    spec = vllm_config.num_speculative_tokens
    return MambaStateShapeCalculator.short_conv_state_shape(1, channels, size, spec)


def state_dtype(vllm_config: Any) -> tuple:
    """The dtype of a conv state: the model's, or ``--mamba-cache-dtype``."""
    return MambaStateDtypeCalculator.short_conv_state_dtype(
        vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
    )


class CausalConv(nn.Module, MambaBase):
    """bux's ``CausalConv`` without a bias: each channel c of a token at t is
    ``sum_i w[i, c] x[t - (size - 1 - i), c]`` over the sequence's tokens, zeros
    before its first.

    Args:
        channels: The channels convolved, each with its taps, on this rank.
        size: The taps of each channel.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of its state, unique in the model.
    """

    def __init__(self, channels: int, size: int, vllm_config: Any, prefix: str):
        super().__init__()
        self.channels, self.size, self.prefix = channels, size, prefix
        self.vllm_config = vllm_config
        self.weight = nn.Parameter(torch.empty(channels, size), requires_grad=False)
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_state_shape(self) -> tuple:
        return state_shape(self.vllm_config, self.channels, self.size)

    def get_state_dtype(self) -> tuple:
        return state_dtype(self.vllm_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve x, (T, channels), the batch's tokens side by side."""
        out = torch.zeros_like(x)
        torch.ops.vllm.short_conv(x.contiguous(), out, self.prefix)
        return out

    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        """Write the conv of the step's tokens, decodes first, into `output`; nothing
        in vLLM's profiling run, which has no metadata."""
        meta = get_forward_context().attn_metadata
        meta = meta.get(self.prefix) if isinstance(meta, dict) else None
        if meta is None:
            return
        state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            state = state.transpose(-1, -2)  # (slots, channels, state_len)
        d, p = meta.num_decode_tokens, meta.num_prefill_tokens
        x_d, x_p = hidden_states[: d + p].split([d, p])
        if meta.num_prefills > 0:
            output[d : d + p] = self._prefill(x_p, state, meta)
        if d > 0:
            self._decode(x_d, state, meta, output[:d])

    def _prefill(self, x: torch.Tensor, state: torch.Tensor, meta: Any) -> torch.Tensor:
        """The conv of the prefills' tokens, (P, channels), after their states."""
        y = causal_conv1d_fn(
            x.transpose(0, 1),
            self.weight,
            None,
            activation=None,
            conv_states=state,
            has_initial_state=meta.has_initial_states_p,
            cache_indices=meta.state_indices_tensor_p,
            metadata=meta,
            query_start_loc=meta.query_start_loc_p,
        )
        return y.transpose(0, 1)[: x.shape[0]]

    def _decode(
        self, x: torch.Tensor, state: torch.Tensor, meta: Any, out: torch.Tensor
    ) -> None:
        """Write the conv of the decoding tokens into `out`, a token a sequence, or
        a sequence's speculative tokens as vLLM's short conv takes them."""
        spec: dict[str, Any] = {}
        if meta.num_accepted_tokens is not None:
            spec = {
                "num_accepted_tokens": meta.num_accepted_tokens,
                "query_start_loc": meta.query_start_loc_d,
                "max_query_len": meta.state_indices_tensor_d.size(-1),
            }
        causal_conv1d_update(
            x,
            state,
            self.weight,
            None,
            activation=None,
            conv_state_indices=meta.state_indices_tensor_d,
            out=out,
            **spec,
        )

    def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        """vLLM's CPU path, which the plugin does not run.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("bux's kv conv runs on GPUs only in vLLM")


class Conv(Layer):
    """bux's ``conv``: a causal depthwise conv over positions as a layer, a bias per
    channel after it, each call's last inputs in vLLM's Mamba cache by its short
    conv (`CausalConv`); the whole stream, on every rank."""

    bux_types = ("conv",)
    stateful = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.path = site.path
        convs = [CausalConv(width, c["size"], site.vllm_config, n) for n in site.caches]
        self.weight = convs[0].weight
        for other in convs[1:]:
            other.weight = self.weight
        self.convs = nn.ModuleList(convs)
        bias = torch.zeros(width) if c["bias"] else None
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name == "weight":  # bux's taps, (size, channels), as vLLM's rows
            self.weight.data.copy_(tensor.T)
            return [self.weight]
        return super().load(name, tensor)

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        y = self.convs[ctx.call(self.path)](x)
        return y if self.bias is None else y + self.bias.to(y.dtype)


class ShortConv(Layer):
    """bux's ``short_conv``, LFM2's: B, C and u by one column-parallel projection,
    ``C * conv(B u)`` by vLLM's short-conv kernels (`CausalConv`) with a bias per
    channel, each call's last inputs in vLLM's Mamba cache, and the row-parallel
    output projection; the channels split over ranks."""

    bux_types = ("short_conv",)
    stateful = True

    def __init__(self, c: dict, width: int, site: Site):
        super().__init__()
        self.path, bias = site.path, c["bias"]
        self.channels = width // get_tensor_model_parallel_world_size()
        self.in_proj = MergedColumnParallelLinear(
            width, [width] * 3, bias=bias, return_bias=False, prefix=f"{self.path}.in"
        )
        size, vllm = c["conv_size"], site.vllm_config
        convs = [CausalConv(self.channels, size, vllm, n) for n in site.caches]
        self.weight = convs[0].weight
        for other in convs[1:]:
            other.weight = self.weight
        self.convs = nn.ModuleList(convs)
        b = torch.zeros(self.channels) if bias else None
        self.conv_bias = None if b is None else nn.Parameter(b, requires_grad=False)
        self.out_proj = RowParallelLinear(
            width, width, bias=bias, return_bias=False, prefix=f"{self.path}.out"
        )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out: dict[str, tuple[nn.Parameter, tuple]] = {}
        for name in ("in_proj", "out_proj"):
            for kind in ("weight", "bias"):
                if (p := getattr(getattr(self, name), kind)) is not None:
                    out[f"{name}.{kind}"] = (p, ())
        return out

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        if name not in ("conv.weight", "conv.bias"):
            return super().load(name, tensor)
        first = get_tensor_model_parallel_rank() * self.channels
        mine = tensor[..., first : first + self.channels]  # this rank's channels
        param = self.weight if name == "conv.weight" else self.conv_bias
        assert param is not None
        param.data.copy_(mine.T if name == "conv.weight" else mine)
        return [param]

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        B, C, u = self.in_proj(x).chunk(3, -1)
        y = self.convs[ctx.call(self.path)]((B * u).contiguous())
        if self.conv_bias is not None:
            y = y + self.conv_bias.to(y.dtype)
        return self.out_proj(C * y)
