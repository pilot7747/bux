"""msar's convs over the tokens in vLLM: each call of an msar branch expands the
newest source by causal depthwise convs (`bux_vllm.residual.MultiStreamAttentionResidual`),
which read each sequence's last ``max(kernel_sizes) - 1`` values of that source
before the step. vLLM keeps them in its Mamba cache, as a short conv's state.

The calls' states are packed: one vLLM layer (`Histories`) keeps the states of up to
`quota` calls side by side in its channels, so that a pack fills about one page of
the model's largest recurrent state, the page every Mamba state takes, rather than a
page a call. Each call convolves its channels of a pack (`History`) in one op,
``vllm::bux_msar_conv``, which the graph splits at, as at vLLM's attention: the
decoded tokens of a step one a sequence, its prompts' chunks by their sequences.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from . import conv
from .recurrent import split_op
from .residual import decode_convs, prefill_convs, stacked_taps

#: The op a call's convs run in, out of the compiled graph.
OP = "bux_msar_conv"


class Histories(nn.Module, MambaBase):
    """vLLM's layer of a pack of msar calls' states: each sequence's last ``size -
    1`` values of each call's newest source, side by side in `channels`.

    Args:
        channels: The calls' channels, d each.
        size: The most taps of a conv.
        vllm_config: vLLM's config of the engine.
        prefix: vLLM's name of its state, unique in the model.
    """

    def __init__(self, channels: int, size: int, vllm_config: Any, prefix: str):
        super().__init__()
        self.channels, self.size, self.prefix = channels, size, prefix
        self.vllm_config = vllm_config
        compilation = vllm_config.compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self
        split_op(compilation, OP)
        self.kv_cache = (torch.tensor([]),)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_state_shape(self) -> tuple:
        return conv.state_shape(self.vllm_config, self.channels, self.size)

    def get_state_dtype(self) -> tuple:
        return conv.state_dtype(self.vllm_config)

    def states(self) -> torch.Tensor:
        """The pack's states, (slots, channels, size - 1)."""
        state = self.kv_cache[0]
        return state if is_conv_state_dim_first() else state.transpose(-1, -2)


class History(nn.Module):
    """One msar call's channels of a pack: where its convs read and keep the newest
    source's last values."""

    def __init__(self, pack: Histories, first: int):
        super().__init__()
        self.name, self.first = pack.prefix, first
        self.size = pack.size

    def convolve(self, v: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        """The convs of v, (T, d), by each of `taps`, (k, d) the token's own last,
        over each sequence's earlier values: (T, len(taps), d)."""
        weight = stacked_taps(taps, self.size)

        out = v.new_zeros(v.shape[0], len(taps), v.shape[1])
        torch.ops.vllm.bux_msar_conv(v.contiguous(), weight, out, self.name, self.first)
        return out


def msar_conv(
    v: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, name: str, first: int
) -> None:
    """`OP`: the convs of the step's tokens' source v, (T, d), by `weight`, (r, k,
    d), into `out`, (T, r, d), reading and keeping each sequence's last k - 1
    values in its pack's channels from `first`; nothing in vLLM's profiling run."""
    meta = get_forward_context().attn_metadata
    meta = meta.get(name) if isinstance(meta, dict) else None
    if meta is None:
        return
    pack = get_forward_context().no_compile_layers[name]
    states = pack.states()[:, first : first + v.shape[1]]  # slots, d, k - 1
    sixteen = v.dtype in (torch.bfloat16, torch.float16)
    if CHOICE == "triton" or (CHOICE is None and sixteen):
        _convolve(v, weight, states, meta, out)
        return
    d, p = meta.num_decode_tokens, meta.num_prefill_tokens
    if d:
        decode_convs(v[:d], weight, states, meta.state_indices_tensor_d, out[:d])
    if meta.num_prefills:
        prefill_convs(v[d : d + p], weight, states, meta, out[d : d + p])


direct_register_custom_op(op_name=OP, op_func=msar_conv, mutates_args=["out"])


#: Which path the convs take, as the residual algebras' reads and writes do: the
#: Triton kernels below (``triton``), PyTorch's (``torch``), or unset for the
#: kernels in a 16-bit model.
CHOICE = os.environ.get("BUX_VLLM_RESIDUAL_KERNELS")
BLOCK = 128  # channels a program takes


def _steps(meta: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The step's sequences side by side, decodes (a token each) first: where each
    one's tokens start, (n + 1,), its state's slot (vLLM's null block, 0, for a
    padded row), and whether it has kept values; and each token's sequence."""
    d, n_p = meta.num_decode_tokens, meta.num_prefills
    starts = [torch.arange(d + 1, device=device, dtype=torch.int32)]
    slots = [meta.state_indices_tensor_d.reshape(-1)[:d]] if d else []
    init = [torch.ones(d, device=device, dtype=torch.bool)] if d else []
    if n_p:
        starts.append((meta.query_start_loc_p[1:] + d).to(torch.int32))
        slots.append(meta.state_indices_tensor_p.reshape(-1)[:n_p])
        init.append(meta.has_initial_states_p.bool())
    start = torch.cat(starts)
    tokens = torch.arange(d + meta.num_prefill_tokens, device=device, dtype=torch.int32)
    seq = torch.searchsorted(start[1:], tokens, right=True).to(torch.int32)
    slot = torch.cat(slots).clamp(min=0).to(torch.int32)
    return start, slot, torch.cat(init), seq


def _convolve(
    v: torch.Tensor,
    weight: torch.Tensor,
    states: torch.Tensor,
    meta: Any,
    out: torch.Tensor,
) -> None:
    """The convs by Triton: every token's outputs, then each sequence's last values
    kept, reading the kept ones before they change."""
    start, slot, init, seq = _steps(meta, v.device)
    t, dims = seq.shape[0], v.shape[1]
    assert v.is_contiguous() and weight.is_contiguous() and out.is_contiguous()
    assert states.shape[1] == dims and weight.shape[2] == dims, (states.shape, dims)
    r, k = weight.shape[:2]
    s0, s1, s2 = states.stride()
    if t:
        grid = (t, triton.cdiv(dims, BLOCK))
        _conv_kernel[grid](
            v, weight, out, states, start, slot, init, seq,
            dims, s0, s1, s2, R=r, K=k, BLOCK=BLOCK,
        )  # fmt: skip
    n = start.shape[0] - 1
    if n:
        _keep_kernel[(n, triton.cdiv(dims, BLOCK))](
            v, states, start, slot, init, dims, s0, s1, s2, K=k, BLOCK=BLOCK
        )


@triton.jit
def _kept_or_token(v_ptr, st_ptr, row, src, slot, init, ch, mask, dims, s0, s1, s2, K):
    """The value `src` places from the start of a token's chunk: the token's own at
    `row` (src >= 0), else the sequence's kept value, if it has any."""
    own = src >= 0
    x = tl.load(v_ptr + row * dims + ch, mask=mask & own, other=0.0).to(tl.float32)
    # A pack's slot is pages apart: past 2^31 elements at a width of thousands.
    at = st_ptr + slot * s0 + ch * s1 + (K - 1 + src).to(tl.int64) * s2
    kept = tl.load(at, mask=mask & (src < 0) & init, other=0.0).to(tl.float32)
    return tl.where(own, x, kept)


@triton.jit
def _conv_kernel(
    v_ptr, w_ptr, out_ptr, st_ptr, start_ptr, slot_ptr, init_ptr, seq_ptr,
    dims, s0, s1, s2, R: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """A token's R convs of K taps, a block of its channels."""
    tok, blk = tl.program_id(0), tl.program_id(1)
    ch = blk * BLOCK + tl.arange(0, BLOCK)
    mask = ch < dims
    tok = tok.to(tl.int64)
    s = tl.load(seq_ptr + tok)
    pos = tok - tl.load(start_ptr + s)
    slot, init = tl.load(slot_ptr + s).to(tl.int64), tl.load(init_ptr + s)
    for r in tl.static_range(R):
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for j in tl.static_range(K):  # tap j reads the value K - 1 - j before
            lag = K - 1 - j
            x = _kept_or_token(
                v_ptr, st_ptr, tok - lag, pos - lag, slot, init, ch, mask,
                dims, s0, s1, s2, K,
            )  # fmt: skip
            w = tl.load(w_ptr + (r * K + j) * dims + ch, mask=mask, other=0.0)
            acc += w.to(tl.float32) * x
        tl.store(out_ptr + (tok * R + r) * dims + ch, acc, mask=mask)


@triton.jit
def _keep_kernel(
    v_ptr, st_ptr, start_ptr, slot_ptr, init_ptr,
    dims, s0, s1, s2, K: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """A sequence's last K - 1 values, of its kept ones and its chunk, kept: in
    place, each read before an earlier place overwrites it."""
    s, blk = tl.program_id(0), tl.program_id(1)
    ch = blk * BLOCK + tl.arange(0, BLOCK)
    mask = ch < dims
    first = tl.load(start_ptr + s).to(tl.int64)
    end = tl.load(start_ptr + s + 1).to(tl.int64)
    slot, init = tl.load(slot_ptr + s).to(tl.int64), tl.load(init_ptr + s)
    n = end - first
    for j in tl.static_range(K - 1):
        c = n - (K - 1) + j
        x = _kept_or_token(
            v_ptr, st_ptr, first + c, c, slot, init, ch, mask, dims, s0, s1, s2, K
        )  # fmt: skip
        at = st_ptr + slot * s0 + ch * s1 + j * s2
        tl.store(at, x.to(st_ptr.dtype.element_ty), mask=mask)
