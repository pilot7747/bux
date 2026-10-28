"""What the plugin's own recurrent layers share (``kda``, ``gdn2``, ``rwkv7``): a
vLLM layer per call, its state in vLLM's Mamba cache as Gated DeltaNet's is (the
conv's last inputs, and a (dv, dk) state per value head), its steps that read or
write the state run outside the compiled graph, and bux's low-rank projections.

A step is one op, ``vllm::bux_mix`` (`MIX`), which calls a method of the layer by
name (`Recurrent.convolve`, `Recurrent.recur` or a layer's own) on the batch's tokens:
the graph splits at it, as at vLLM's attention. vLLM pads a batch past its tokens;
a step leaves the padding's outputs at zero.
"""

from typing import Any

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from . import delta
from .common import ACTIVATIONS, sharded_loader
from .linear import conv_channels, conv_shard

#: The op a recurrent layer's steps run in, out of the compiled graph.
MIX = "bux_mix"


def mix_op(
    xs: list[torch.Tensor], out: torch.Tensor, layer_name: str, what: str
) -> None:
    """`MIX`: the method `what` of the layer `layer_name`, on `xs`, into `out`."""
    layer = get_forward_context().no_compile_layers[layer_name]
    getattr(layer, what)(xs, out)


direct_register_custom_op(op_name=MIX, op_func=mix_op, mutates_args=["out"])


def state_dtype(model_config: Any, cache_config: Any) -> tuple:
    """The dtypes of the state: the conv's the model's (or ``mamba_cache_dtype``),
    the recurrence's float32, as bux's (or ``mamba_ssm_cache_dtype``)."""
    return MambaStateDtypeCalculator.kda_state_dtype(
        model_config.dtype,
        cache_config.mamba_cache_dtype,
        cache_config.mamba_ssm_cache_dtype,
    )


def split_op(compilation: Any, op: str) -> None:
    """Make vLLM's op `op` a place the compiled graph splits at."""
    ops = compilation.splitting_ops
    if ops is not None and f"vllm::{op}" not in ops:
        ops.append(f"vllm::{op}")


def float32(*shape: int) -> nn.Parameter:
    """A float32 parameter of this rank's share of the first dim."""
    p = nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)
    p.weight_loader = sharded_loader(0)  # ty: ignore[unresolved-attribute]
    return p


class Recurrent(nn.Module, MambaBase):
    """vLLM's layer of one call of a recurrent layer: registered under `prefix`, its
    state in the Mamba cache, its conv's taps if it has one.

    Raises:
        NotImplementedError: Speculative decoding, which no layer here keeps the
            states of.
    """

    what = ""  # the bux layer type, for errors
    floor: float | None = None  # how far a position's log decay may fall, if bounded

    def __init__(self, vllm_config: Any, prefix: str):
        super().__init__()
        if vllm_config.num_speculative_tokens:
            raise NotImplementedError(
                f"{self.what}: speculative decoding not in vLLM yet"
            )
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.tp = get_tensor_model_parallel_world_size()
        self.conv: nn.Parameter | None = None
        compilation = vllm_config.compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self
        split_op(compilation, MIX)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple:
        return state_dtype(self.model_config, self.cache_config)

    def conv_taps(self, c: dict) -> None:
        """Make the conv's taps over q, k and v: this rank's channels, (C, 1, k)."""
        channels = sum(conv_channels(c)) // self.tp
        self.c_conv = c
        self.conv = nn.Parameter(
            torch.empty(channels, 1, c["conv_size"]), requires_grad=False
        )

    def load_conv(self, tensor: torch.Tensor) -> nn.Parameter:
        """Load bux's conv taps, (k, C), as this rank's; return the parameter."""
        assert self.conv is not None
        rank = get_tensor_model_parallel_rank()
        sizes = conv_channels(self.c_conv)
        self.conv.data.copy_(conv_shard(tensor, sizes, rank, self.tp))
        return self.conv

    def run(self, what: str, xs: list[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        """Run the step `what` (a method's name) out of the compiled graph; return
        `out`, which it fills."""
        torch.ops.vllm.bux_mix(xs, out, self.prefix, what)
        return out

    def batch(self) -> Any:
        """This step's metadata of the layer (vLLM's GDN metadata); None in a run
        without (vLLM's profiling) or with no tokens."""
        meta = get_forward_context().attn_metadata
        m = meta.get(self.prefix) if isinstance(meta, dict) else None
        if m is None or m.num_prefills + m.num_decodes == 0:
            return None
        return m

    @staticmethod
    def tokens(m: Any) -> int:
        """The batch's tokens: all of a prefill's, one a sequence decoding."""
        return m.num_actual_tokens if m.num_prefills else m.num_decodes

    def conv_state(self) -> torch.Tensor:
        """The conv's state of every slot, (slots, C, taps), as the kernels take it."""
        state = self.kv_cache[0]
        return state if is_conv_state_dim_first() else state.transpose(-1, -2)

    def convolve(self, xs: list[torch.Tensor], out: torch.Tensor) -> None:
        """A step: the causal conv of ``xs[0]``, (T, C), SiLU'd, from each sequence's
        last inputs (none for a new one), keeping its own; into `out`."""
        m = self.batch()
        if m is None or self.conv is None:
            return
        n = self.tokens(m)
        out[:n] = self.conv_tokens(m, xs[0][:n])

    def conv_tokens(self, m: Any, x: torch.Tensor) -> torch.Tensor:
        """The conv of the batch's tokens, (n, C), SiLU'd, keeping the state."""
        assert self.conv is not None
        weight, state = self.conv.flatten(1), self.conv_state()
        idx = m.non_spec_state_indices_tensor
        if m.num_prefills:
            return causal_conv1d_fn(
                x.transpose(0, 1),
                weight,
                None,
                conv_states=state,
                query_start_loc=m.non_spec_query_start_loc,
                cache_indices=idx,
                has_initial_state=m.has_initial_state,
                activation="silu",
                metadata=m,
            ).transpose(0, 1)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        return causal_conv1d_update(
            x,
            state,
            weight,
            None,
            activation="silu",
            conv_state_indices=idx[: x.shape[0]],
            validate_data=True,
            out=out,
        )

    def recur(self, xs: list[torch.Tensor], out: torch.Tensor) -> None:
        """A step: bux's delta rule (`bux_vllm.delta`) of ``xs``, its q, k, v, log
        decay, a and b at the value heads, (T, hv, d), from each sequence's state,
        keeping the last; its outputs, (T, hv, dv), into `out`."""
        m = self.batch()
        if m is None:
            return
        n = self.tokens(m)
        out[:n] = self.rule(m, *(x[:n] for x in xs))

    def rule(self, m: Any, *xs: torch.Tensor) -> torch.Tensor:
        """The delta rule over the batch's tokens, from each sequence's state, which
        it keeps: a step of each decoding sequence (vLLM puts them first), and the
        prompts by chunks (`prompts`)."""
        state = self.states()
        idx = m.non_spec_state_indices_tensor[: m.num_prefills + m.num_decodes]
        before = state[idx]
        if m.num_prefills:  # vLLM's empty requests, if any, come last
            has = m.has_initial_state[: idx.shape[0], None, None, None]
            before = torch.where(has, before, torch.zeros_like(before))
        steps = m.num_decodes
        outs, afters = [], []
        if steps:
            o, after = delta.step(*(x[:steps] for x in xs), before[:steps])
            outs.append(o)
            afters.append(after)
        if m.num_prefills:
            o, after = self.prompts(m, [x[steps:] for x in xs], before[steps:])
            outs.append(o)
            afters.append(after)
        after = torch.cat(afters).to(state.dtype)
        kept = (idx > 0)[:, None, None, None]  # padding's slots stay as they were
        state[idx] = torch.where(kept, after, before)
        return torch.cat(outs)

    def states(self) -> torch.Tensor:
        """Every slot's state, (slots, hv, dk, dv): vLLM's GDN shape, (hv, dv, dk),
        holds it in bux's layout, which only this layer reads."""
        state = self.kv_cache[1]
        return state.view(*state.shape[:2], state.shape[3], state.shape[2])

    def prompts(
        self, m: Any, xs: list[torch.Tensor], before: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rule over the batch's prompts, after its decoding sequences: by
        chunks, each padded to the longest; their outputs and last states."""
        starts = m.non_spec_query_start_loc[m.num_decodes :].long()
        starts = starts[: before.shape[0] + 1] - starts[0]
        longest = int((starts[1:] - starts[:-1]).max())
        ys = [delta.padded(x, starts, longest) for x in xs]
        o, after = delta.rank_one(*ys, before, floor=self.floor)
        return delta.unpadded(o, starts, xs[0].shape[0]), after


class LowRank(nn.Module):
    """bux's ``LowRank``, ``act(x A) B + c``: A on every rank, B and c split by
    their outputs.

    Args:
        n_in: The width of the input.
        rank: The dims between A and B.
        n_out: The width of the output.
        prefix: vLLM's name of the module.
        activation: The activation between, by name; None for none.
        bias: Whether to add a bias c.
    """

    def __init__(
        self,
        n_in: int,
        rank: int,
        n_out: int,
        prefix: str,
        activation: str | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.down = ReplicatedLinear(
            n_in, rank, bias=False, return_bias=False, prefix=f"{prefix}.down"
        )
        self.up = ColumnParallelLinear(
            rank, n_out, bias=bias, return_bias=False, prefix=f"{prefix}.up"
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        if self.activation is not None:
            h = ACTIVATIONS[self.activation](h)
        return self.up(h)

    def targets(self, name: str) -> dict[str, tuple[nn.Parameter, tuple]]:
        """Its parameters by bux's names under `name`: ``down``, ``up``, ``bias``."""
        out = {
            f"{name}.down": (self.down.weight, ()),
            f"{name}.up": (self.up.weight, ()),
        }
        if self.up.bias is not None:
            out[f"{name}.bias"] = (self.up.bias, ())
        return out


def low_rank_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """bux's tensor `name` as vLLM's layers hold it: a low-rank projection's A and B
    transposed (bux's x A B, vLLM's linear layers' weights A^T and B^T)."""
    return tensor.T if name.endswith((".down", ".up")) else tensor
