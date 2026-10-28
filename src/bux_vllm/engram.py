"""bux's ``engram`` in vLLM: its n-grams hashed over the step's tokens, and its
dilated conv with its state in vLLM's Mamba cache.

An n-gram reaches back ``max_ngram_size - 1`` tokens, before the step's first: vLLM's
model runner (V2, its default) gives each sequence's last ids before the step, from
the tokens it holds, through the model's own runner state (`model_state`), as it does
for Qwen4Exp's n-gram embeddings; nothing of them is cached. The conv is a short conv
of vLLM's (`DilatedConv`): a layer with its last ``(conv_size - 1) max_ngram_size``
inputs as its state, built by vLLM's short-conv metadata of Qwen4Exp's dilated conv,
which says which decoding sequences have a state yet. It runs inside vLLM's
``short_conv`` op, in PyTorch over the step's tokens (`bux_vllm.ngram`).
"""

from __future__ import annotations

from typing import Any

import torch
import vllm.model_executor.layers.mamba.short_conv  # noqa: F401  (its op)
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend

from . import conv
from .common import Ctx, Site
from .ngram import EngramCore, dilated_conv
from .plan import Node, live, plan


def copies(residual: dict | None, width: Any) -> int:
    """The copies of the stream an engram adds to: a widened stream's n, else 1."""
    return (residual or {}).get("expansion", 1) if width == "streams" else 1


def _engrams(config: Any) -> list[Node]:
    """The engram layers the model runs before its first head."""
    calls = live(plan(config.bux, config.bux_reserved).calls)
    return [c.layer for c in calls if c.layer["config"]["type"] == "engram"]


def state_shapes(vllm_config: Any) -> list[tuple]:
    """The shapes of the states of the model's engrams' convs, one of each."""
    config = vllm_config.model_config.hf_config
    d, residual = config.bux["d_model"], config.bux["residual"]
    convs = {
        (copies(residual, n["width"]) * d, _keep(n["config"]))
        for n in _engrams(config)
        if n["config"]["conv_size"]
    }
    return [state_shape(*conv) for conv in sorted(convs)]


def state_shape(channels: int, keep: int) -> tuple:
    """The shape of the state of a conv of `channels` that reads `keep` inputs back."""
    return MambaStateShapeCalculator.short_conv_state_shape(1, channels, keep + 1)


def _keep(c: dict) -> int:
    """How many inputs back an engram's conv reads: ``(size - 1)`` dilations."""
    return (c["conv_size"] - 1) * c["max_ngram_size"]


def context(config: Any) -> int:
    """How many ids before a step the model's engrams read: 0 without one."""
    return max((n["config"]["max_ngram_size"] - 1 for n in _engrams(config)), default=0)


class Engram(EngramCore):
    """bux's ``engram`` (`bux_vllm.ngram.EngramCore`), whole on every rank, with a
    conv of its state (`DilatedConv`) for each call.

    Raises:
        NotImplementedError: Speculative decoding, or a conv of a single tap.
    """

    bux_types = ("engram",)
    stateful = True

    def __init__(self, c: dict, width: Any, site: Site):
        bux = site.config.bux
        hashes = site.config.bux_sites[site.path]["extra"]
        n = copies(bux["residual"], width)
        super().__init__(c, bux["d_model"], n, hashes)
        self.path, self.conv_taps, self.convs = site.path, None, None
        if size := c["conv_size"]:
            if size == 1 or site.vllm_config.num_speculative_tokens:
                raise NotImplementedError("engram: one tap or drafts not in vLLM yet")
            taps = torch.empty(size, n * bux["d_model"])
            self.conv_taps = nn.Parameter(taps, requires_grad=False)
            self.convs = nn.ModuleList(
                DilatedConv(self.conv_taps, self.n, site.vllm_config, name)
                for name in site.caches
            )

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        out = {k: v for k, v in super().targets().items() if k != "conv_taps"}
        if self.conv_taps is not None:
            out["conv.weight"] = (self.conv_taps, ())
        return out

    def convolve(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        assert self.convs is not None
        return self.convs[ctx.call(self.path)](x)


class DilatedConv(nn.Module, MambaBase):
    """bux's ``CausalConv`` of an engram: each channel c of a token at t is
    ``sum_i w[i, c] x[t - (size - 1 - i) r, c]`` over its sequence, for the dilation
    r, zeros before its first; its state the last ``(size - 1) r`` inputs.

    Args:
        weight: The taps, (size, channels), which every call's conv shares.
        dilation: r.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of its state, unique in the model.
    """

    def __init__(
        self, weight: nn.Parameter, dilation: int, vllm_config: Any, prefix: str
    ):
        super().__init__()
        self.weight, self.dilation, self.prefix = weight, dilation, prefix
        self.vllm_config = vllm_config
        self.channels, self.keep = weight.shape[1], (weight.shape[0] - 1) * dilation
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        """vLLM's short convs of a dilation (Qwen4Exp's): their metadata says which
        decoding sequences have a state yet."""
        return PleShortConvAttentionBackend

    def get_state_shape(self) -> tuple:
        return state_shape(self.channels, self.keep)

    def get_state_dtype(self) -> tuple:
        return conv.state_dtype(self.vllm_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve x, (T, channels), the batch's tokens side by side."""
        out = torch.zeros_like(x)
        torch.ops.vllm.short_conv(x.contiguous(), out, self.prefix)
        return out

    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        """Write the conv of the step's tokens, decodes first, into `output`, and
        keep each sequence's last inputs; nothing in vLLM's profiling run, which has
        no metadata."""
        meta = get_forward_context().attn_metadata
        meta = meta.get(self.prefix) if isinstance(meta, dict) else None
        if meta is None:
            return
        state = self.kv_cache[0]
        if is_conv_state_dim_first():
            state = state.transpose(-1, -2)  # (slots, keep, channels)
        if meta.num_decodes > 0:
            slots = meta.state_indices_tensor_d[:, 0]  # graph-padded rows included
            rows = slots.shape[0]
            starts = torch.arange(rows + 1, device=slots.device, dtype=torch.int32)
            x, out = hidden_states[:rows], output[:rows]
            self._convolve(x, out, state, slots, meta.has_initial_states_d, starts)
        if meta.num_prefills > 0:
            d, p = meta.num_decode_tokens, meta.num_prefill_tokens
            slots = meta.state_indices_tensor_p[: meta.num_prefills]
            x, out = hidden_states[d : d + p], output[d : d + p]
            first = meta.has_initial_states_p
            self._convolve(x, out, state, slots, first, meta.query_start_loc_p)

    def _convolve(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        started: torch.Tensor,
        starts: torch.Tensor,
    ) -> None:
        """Write the conv of sequences' tokens x into `out`, after the states at
        their `slots` where they have `started`, and keep their last inputs there."""
        kept = state[slots]
        past = torch.where(started[:, None, None], kept, torch.zeros_like(kept))
        y, last = dilated_conv(x, self.weight, self.dilation, starts, past)
        out.copy_(y)
        state[slots] = last.to(state.dtype)

    def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        """vLLM's CPU path, which the plugin does not run.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("bux's engram conv runs on GPUs only in vLLM")


class NGramInputs:
    """What a model runner state adds to the inputs of a model with engrams: each
    sequence's last `context` ids before the step (-1 for none), and where each
    sequence's tokens start, both in buffers of fixed shapes that CUDA graphs
    replay, as vLLM's Qwen4Exp model state gives its n-gram embedding them."""

    context: int = 0

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        reqs, device = self.max_num_reqs, self.device  # ty: ignore[unresolved-attribute]
        self.ngram_context = torch.full(
            (reqs, self.context), -1, dtype=torch.int64, device=device
        )
        self.ngram_offsets = torch.arange(-self.context, 0, device=device)
        self.ngram_starts = torch.zeros(reqs + 1, dtype=torch.int32, device=device)

    def prepare_inputs(self, input_batch: Any, req_states: Any) -> dict[str, Any]:
        out = super().prepare_inputs(input_batch, req_states)  # ty: ignore[unresolved-attribute]
        padded = input_batch.num_reqs_after_padding
        self.ngram_starts[: padded + 1].copy_(input_batch.query_start_loc)
        self.ngram_starts[padded + 1 :].copy_(input_batch.query_start_loc[-1])
        return out | {
            "query_start_loc": self.ngram_starts,
            "ngram_context": self._context(input_batch, req_states),
        }

    def _context(self, input_batch: Any, req_states: Any) -> torch.Tensor:
        """Each sequence's last ids before the step, from the tokens vLLM holds."""
        n, out = input_batch.num_reqs, self.ngram_context
        out.fill_(-1)
        if n == 0:
            return out
        reqs = input_batch.idx_mapping[:n].long()
        end = req_states.num_computed_tokens.gpu[reqs].long()
        at = end[:, None] + self.ngram_offsets
        ids = req_states.all_token_ids.gpu[reqs[:, None], at.clamp(min=0)].long()
        out[:n].copy_(torch.where(at >= 0, ids, -1))
        return out

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        out = super().prepare_dummy_inputs(num_reqs, num_tokens)  # ty: ignore[unresolved-attribute]
        each, extra = divmod(num_tokens, num_reqs)
        lengths = torch.full((num_reqs,), each, dtype=torch.int32, device=self.device)  # ty: ignore[unresolved-attribute]
        if extra:
            lengths[-extra:] += 1
        self.ngram_starts[0] = 0
        torch.cumsum(lengths, 0, out=self.ngram_starts[1 : num_reqs + 1])
        self.ngram_starts[num_reqs + 1 :].fill_(num_tokens)
        self.ngram_context.fill_(-1)
        return out | {
            "query_start_loc": self.ngram_starts,
            "ngram_context": self.ngram_context,
        }


def model_state(vllm_config: Any) -> type:
    """The class of vLLM's model runner state for the model: vLLM's own for a
    hybrid model, or another, with `NGramInputs` where it has engrams."""
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    hybrid = vllm_config.model_config.is_hybrid
    base = MambaHybridModelState if hybrid else DefaultModelState
    n = context(vllm_config.model_config.hf_config)
    if not n:
        return base
    return type("BuxNGramModelState", (NGramInputs, base), {"context": n})
