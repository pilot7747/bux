"""What every layer of the vLLM model shares: the forward pass's context, a layer's
site, the activations, the norms and RoPE.

A layer is a class with ``bux_types``, the bux layer types it implements, built as
``Layer(c, width, site)``: `c` is the layer's bux config as JSON, `width` what it
reads, and `site` its `Site`. It is called as ``layer(x, ctx)`` on the batch's tokens
side by side, (T, width), as vLLM schedules them, and loads its own weights,
``load(name, tensor)``, by their names in the checkpoint under its path
(``q.weight``), returning the parameters it filled. A layer that keeps state between
steps (``stateful``) makes one of vLLM's attention or Mamba layers for each of its
calls, named as `Site.caches` says.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

#: bux's activations by name (`bux.config.Activation`).
ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "relu2": lambda x: F.relu(x).square(),
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    "swish": F.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softplus": F.softplus,
    "situ": lambda x: 4 * torch.tanh(x / 4) * torch.sigmoid(x),
}


def unsupported(c: dict, *names: str) -> None:
    """Raise if the config `c` sets any of `names` away from its default (None,
    False or 0), which the vLLM model does not implement yet."""
    on = [n for n in names if c.get(n) not in (None, False, 0, 0.0)]
    if on:
        raise NotImplementedError(f"{c['type']}: {', '.join(on)} not in vLLM yet")


@dataclass
class Site:
    """Where a layer is in the model, and what it builds with.

    Attributes:
        path: The module's path, the prefix of its weights: ``model.layers.3.attn``.
        vllm_config: vLLM's config of the engine.
        config: The model's config, with the bux IR in ``config.bux``.
        caches: The name of the state of each of its calls, for a layer that keeps
            state: vLLM's layer name, with one integer, the call's place among those
            that keep state.
        embedding: The model's first embedding, which a tied head reads.
        ports: The taps it writes or reads internal tensors through.
        shares: The KV cache each of its calls attends to instead of its own, by
            vLLM's layer name: the writer's, for a call that reads the ``kv`` port
            (`bux_vllm.kv.sources`); None for its own.
    """

    path: str
    vllm_config: Any
    config: Any
    caches: list[str] = field(default_factory=list)
    embedding: Any = None
    ports: list = field(default_factory=list)
    shares: list[str | None] = field(default_factory=list)

    @property
    def quant(self) -> Any:
        """The engine's quantization config (`quantization`)."""
        return quantization(self.vllm_config)


def quantization(vllm_config: Any) -> Any:
    """The engine's quantization config, which every vLLM linear layer the plugin
    runs as a module takes, and the routed experts: vLLM's online FP8 weights under
    ``quantization="fp8"``, None for none. The embeddings, the heads, the routers and
    every weight the plugin reads itself take none, and stay in the model's dtype."""
    return vllm_config.quant_config


class Ctx:
    """One forward pass: the tokens' positions, bux's taps, the shared modules, and
    the input of the first head, which the model returns.

    Attributes:
        positions: Each token's position in its sequence, (T,).
        shared: The modules of shared definitions, by their key.
        inputs_embeds: The embedded tokens vLLM passes instead of ids, if any.
        taps: The tap stacks, by name.
        embedding: The model's first embedding, once it ran.
        hidden: The first head's input, once it ran: the hidden states vLLM takes
            logits of.
        pending_read: The read coefficients, (T, n), the last single-pass hc or mhc
            branch computed for the next one; None before the first.
        input_ids: The step's token ids, (T,), which engram hashes.
        query_start_loc: Where each sequence's tokens start in the step, (R + 1,),
            as vLLM's model runner gives them to a model that asks (engram's).
        ngram_context: Each sequence's last ids before the step, (R, C), -1 for
            none, likewise.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        shared: nn.ModuleDict,
        inputs_embeds: torch.Tensor | None,
        plan: Any,
    ):
        self.positions, self.shared, self.plan = positions, shared, plan
        self.inputs_embeds = inputs_embeds
        self.taps: dict[str, list] = {}
        self.embedding: Any = None
        self.hidden: torch.Tensor | None = None
        self.pending_read: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self.query_start_loc: torch.Tensor | None = None
        self.ngram_context: torch.Tensor | None = None
        self._calls: dict[str, int] = {}

    def push(self, name: str, x: Any) -> Any:
        """``@name``: push `x` onto the tap `name`, and return it."""
        self.taps.setdefault(name, []).append(x)
        return x

    def read(self, name: str, pop: bool = False) -> Any:
        """``$name``: the top of the tap `name`; with `pop`, taken off it."""
        stack = self.taps[name]
        return stack.pop() if pop else stack[-1]

    def call(self, path: str) -> int:
        """Return how many times the module at `path` ran before in this pass: which
        of its states this call keeps."""
        n = self._calls.get(path, 0)
        self._calls[path] = n + 1
        return n


def load_into(param: nn.Parameter, tensor: torch.Tensor, *args: Any) -> None:
    """Load `tensor` into `param` by its vLLM weight loader (which takes its shard
    under tensor parallelism), or as it is."""
    loader = getattr(param, "weight_loader", None)
    if loader is None:
        assert param.shape == tensor.shape, (param.shape, tensor.shape)
        param.data.copy_(tensor)
    else:
        loader(param, tensor, *args)


def sharded_loader(dim: int) -> Callable[[nn.Parameter, torch.Tensor], None]:
    """Return a weight loader that takes this rank's slice of a tensor along `dim`."""

    def load(param: nn.Parameter, tensor: torch.Tensor) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        size = param.shape[dim]
        start = get_tensor_model_parallel_rank() * size
        param.data.copy_(tensor.narrow(dim, start, size))

    return load


class Layer(nn.Module):
    """What the layers share: `load` by a table of the checkpoint's names."""

    bux_types: tuple[str, ...] = ()
    stateful = False  # whether it keeps state between steps
    # Whether a module can only run once a pass: vLLM's MoE and recurrent layers.
    once = False

    def targets(self) -> dict[str, tuple[nn.Parameter, tuple]]:
        """The checkpoint's names under the layer, each with the parameter it loads
        into and the weight loader's extra arguments (a shard id)."""
        return {name: (p, ()) for name, p in self.named_parameters()}

    def load(self, name: str, tensor: torch.Tensor) -> list[nn.Parameter]:
        """Load the checkpoint's tensor `name`, under the layer's path, into its
        parameter; return the parameters filled.

        Raises:
            KeyError: The layer has no such weight.
        """
        targets = self.targets()
        if name not in targets:
            raise KeyError(f"no weight {name!r} in {type(self).__name__}")
        param, args = targets[name]
        load_into(param, tensor, *args)
        return [param]


class Norm(Layer):
    """bux's ``layernorm`` and ``rmsnorm``: statistics in float32, the result in x's
    dtype, a gain (1 + w when zero-centered; before normalizing with scale_input)
    and a bias."""

    bux_types = ("layernorm", "rmsnorm")

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        self.kind, self.eps = c["type"], c["eps"]
        self.zero_centered, self.scale_input = c["zero_centered"], c["scale_input"]
        init = torch.zeros if self.zero_centered else torch.ones
        affine, bias = c["affine"], c["bias"]
        self.weight = nn.Parameter(init(width), requires_grad=False) if affine else None
        self.bias = (
            nn.Parameter(torch.zeros(width), requires_grad=False) if bias else None
        )

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        gain = None
        if self.weight is not None:
            gain = 1 + self.weight if self.zero_centered else self.weight
        if gain is not None and self.scale_input:
            x = x * gain
        x = self.normalize(x)
        if gain is not None and not self.scale_input:
            x = x * gain
        return x if self.bias is None else x + self.bias

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x over its RMS, centered first for a layernorm, computed in float32."""
        y = x.float()
        if self.kind == "layernorm":
            y = y - y.mean(-1, keepdim=True)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype)


class DyT(Layer):
    """bux's ``dyt``: ``weight * tanh(alpha * x) + bias``, a norm without statistics,
    in float32, the result in x's dtype."""

    bux_types = ("dyt",)

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        alpha = torch.tensor(float(c["alpha_init"]))
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self.weight = nn.Parameter(torch.ones(width), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(width), requires_grad=False)

    def forward(self, x: torch.Tensor, ctx: Ctx | None = None) -> torch.Tensor:
        y = torch.tanh(self.alpha.float() * x.float())
        return (y * self.weight.float() + self.bias.float()).to(x.dtype)


def _yarn_factor(s: dict) -> float:
    """YaRN's attention factor: as given, else 0.1 ln(factor) + 1."""
    if s["attention_factor"] is not None:
        return s["attention_factor"]
    return 0.1 * math.log(s["factor"]) + 1 if s["factor"] > 1 else 1.0


def rope_frequencies(rope: dict, dims: int) -> tuple[torch.Tensor, float]:
    """Return the inverse frequencies of `dims` rotated dims' pairs, and the gain on
    cos and sin, as `bux.architecture.attention.rope.frequencies` makes them."""
    inv = 1.0 / rope["base"] ** (torch.arange(0, dims, 2, dtype=torch.float64) / dims)
    s = rope["scaling"]
    if s is None:
        return inv, 1.0
    if s["type"] == "pi":
        return inv / s["factor"], 1.0
    if s["type"] == "llama3":
        turns = s["original_max_len"] * inv / (2 * math.pi)
        span = s["high_freq_factor"] - s["low_freq_factor"]
        ramp = ((turns - s["low_freq_factor"]) / span).clamp(0, 1)
        return inv / s["factor"] * (1 - ramp) + inv * ramp, 1.0
    if s["type"] == "yarn":
        return _yarn(rope, s, inv, dims)
    raise NotImplementedError(f"rope scaling {s['type']} not in vLLM yet")


def _yarn(
    rope: dict, s: dict, inv: torch.Tensor, dims: int
) -> tuple[torch.Tensor, float]:
    """YaRN's frequencies, blended between the original and the interpolated ones."""

    def pair(turns: float) -> float:
        span = s["original_max_len"] / (turns * 2 * math.pi)
        return dims * math.log(span) / (2 * math.log(rope["base"]))

    low = max(math.floor(pair(s["beta_fast"])), 0)
    high = min(math.ceil(pair(s["beta_slow"])), dims - 1)
    steps = torch.arange(dims // 2, dtype=torch.float64)
    ramp = ((steps - low) / ((high - low) or 1e-3)).clamp(0, 1)
    inv = inv / s["factor"] * ramp + inv * (1 - ramp)
    return inv, _yarn_factor(s) if s["attention_factor_on"] == "rotary" else 1.0


def logit_gain(rope: dict | None) -> float:
    """YaRN's ``attention_factor ** 2`` where it goes on the softmax scale, else 1."""
    s = rope["scaling"] if rope else None
    if s is not None and s["type"] == "yarn" and s["attention_factor_on"] == "softmax":
        return _yarn_factor(s) ** 2
    return 1.0


class Rope(nn.Module):
    """Rotates the first ``fraction`` of each head's dims by the tokens' positions,
    paired ``half`` (NeoX) or ``interleaved`` (GPT-J), in float32 as bux does."""

    inv_freq: torch.Tensor

    def __init__(self, rope: dict, head_dim: int):
        super().__init__()
        self.layout = rope["layout"]
        self.dims = int(head_dim * rope["fraction"]) // 2 * 2
        inv, self.gain = rope_frequencies(rope, self.dims)
        self.register_buffer("inv_freq", inv.float(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate x, (T, h, k), at positions, (T,)."""
        if self.dims == 0:
            return x
        angles = positions[:, None].float() * self.inv_freq
        sin = (self.gain * angles.sin())[:, None, :]
        cos = (self.gain * angles.cos())[:, None, :]
        y, rest = x[..., : self.dims].float(), x[..., self.dims :]
        if self.layout == "interleaved":
            y1, y2 = y[..., 0::2], y[..., 1::2]
            y = torch.stack([y1 * cos - y2 * sin, y2 * cos + y1 * sin], -1).flatten(-2)
        else:
            y1, y2 = y.chunk(2, -1)
            y = torch.cat([y1 * cos - y2 * sin, y2 * cos + y1 * sin], -1)
        return torch.cat([y.to(x.dtype), rest], -1)
