"""What every exported layer shares: the forward pass's context, a layer's site, the
activations and the norms.

The files of this folder are copied into every exported model's repo, next to the
generated ``modeling_bux.py``, and transformers loads them as remote code. So they
import nothing of bux, only torch and transformers, import each other relatively, and
parse on Python 3.10.

A layer is a class with ``bux_types``, the bux layer types it implements, built as
``Layer(c, width, site)``: `c` is the layer's bux config as JSON (``model_dump``),
`width` what it reads (an int, ``"tokens"`` or ``"streams"``), and `site` its `Site`.
It is called as ``layer(x, ctx)``, with the pass's `Ctx`. Its attributes mirror the
NNX module's, so weights convert by one rule (`bux.export.weights`); a layer whose
torch layout differs converts its own tensors in a ``bux_weights`` classmethod.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

#: The state that is no parameter and that bux keeps in float32 whatever it computes
#: in, by name: a bfloat16 export keeps it in float32, in its checkpoint and as it
#: loads (``_keep_in_fp32_modules_strict``), where a rounded router bias would flip
#: near-tied picks of experts.
KEEP_FLOAT32 = ["router_bias"]

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


def norm_config(
    kind: str, eps: float = 1e-6, *, affine: bool = True, bias: bool = False
) -> dict:
    """Return the config of a norm a layer makes of its own, as bux's
    ``NormalizationConfig`` as JSON: a ``layernorm`` or ``rmsnorm`` of `eps`, its
    gain (`affine`) and its `bias` if asked."""
    return {
        "type": kind,
        "eps": eps,
        "affine": affine,
        "bias": bias,
        "zero_centered": False,
        "scale_input": False,
    }


def unsupported(c: dict, *names: str) -> None:
    """Raise if the config `c` sets any of `names` away from its default (None,
    False or 0), which this export does not implement yet."""
    on = [n for n in names if c.get(n) not in (None, False, 0, 0.0)]
    if on:
        raise NotImplementedError(f"{c['type']}: {', '.join(on)} not exported yet")


@dataclass
class Site:
    """Where a layer is in the model, as `bux.export.plan` places it.

    Attributes:
        config: The model's config (``BuxConfig``), with the bux IR in ``config.bux``.
        path: The module's path, the prefix of its weights: ``model.layers.3.attn``.
        depth: How many residual branches are around the layer.
        ports: The taps it writes or reads internal tensors through, as
            ``(name, tap, mode)``.
        extra: What else its exporter computed for it (`bux.export.plan`).
    """

    config: Any
    path: str
    depth: int = 0
    ports: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def port(self, name: str, mode: str) -> str | None:
        """Return the tap the layer's port `name` goes through in `mode` (``write``,
        or ``read`` / ``pop``), or None."""
        modes = ("read", "pop") if mode == "read" else (mode,)
        return next((t for n, t, m in self.ports if n == name and m in modes), None)


def site(config: Any, path: str) -> Site:
    """Return the `Site` of the layer at `path`, from ``config.bux_sites``."""
    return Site(config, path, **config.bux_sites[path])


class Ctx:
    """One forward pass: its positions, masks and cache, and bux's taps.

    Attributes:
        tokens: The input ids, b rows of t.
        positions: Each token's position, b rows of t.
        attention_mask: transformers' 2D mask of the tokens to attend to, or None.
        cache: The transformers cache, or None without one.
        taps: The tap stacks, by name, as bux's ``Taps``.
        heads: The logits of every head run so far.
        embedding: The model's first embedding, which tied heads and nextn read.
        shared: The modules of shared definitions, by their key.
        nextn_runs: How many nextn layers have run: the next looks one token further
            ahead.
        pending_read: The read coefficients, b t n, the last single-pass hc or mhc
            branch computed for the next one; None before the first.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: Any,
        shared: nn.ModuleDict,
        masks: Callable[[Ctx, str, Any], Any],
    ):
        self.tokens, self.positions = tokens, positions
        self.attention_mask, self.cache, self.shared = attention_mask, cache, shared
        self.taps: dict[str, list] = {}
        self.heads: list[torch.Tensor] = []
        self.embedding: Any = None
        self.dtype: torch.dtype = torch.float32  # the model's, which sets them
        self.config: Any = None
        self.nextn_runs = 0
        self.pending_read: torch.Tensor | None = None
        self._masks, self._made, self._slots = masks, {}, 0

    def push(self, name: str, x: Any) -> Any:
        """``@name``: push `x` onto the tap `name`, and return it."""
        self.taps.setdefault(name, []).append(x)
        return x

    def read(self, name: str, pop: bool = False) -> Any:
        """``$name``: the top of the tap `name`; with `pop`, taken off it."""
        stack = self.taps[name]
        return stack.pop() if pop else stack[-1]

    def cache_slot(self) -> int:
        """Return the cache layer of the next call that keeps state: calls take them
        in the order they run, as ``config.layer_types`` lists them."""
        self._slots += 1
        return self._slots - 1

    def mask(self, kind: str = "full_attention", arg: Any = None) -> Any:
        """Return the attention mask of `kind` (with `arg`, a sliding window say),
        made once per pass by the model's mask maker."""
        key = (kind, arg)
        if key not in self._made:
            self._made[key] = self._masks(self, kind, arg)
        return self._made[key]


class Norm(nn.Module):
    """bux's ``layernorm`` and ``rmsnorm``: statistics in float32, the result in x's
    dtype, a gain (1 + w when zero-centered; before normalizing with scale_input)
    and a bias. ``gain_init`` only starts training."""

    bux_types = ("layernorm", "rmsnorm")

    def __init__(self, c: dict, width: Any, site: Site | None = None):
        super().__init__()
        self.kind, self.eps = c["type"], c["eps"]
        self.zero_centered, self.scale_input = c["zero_centered"], c["scale_input"]
        init = torch.zeros if self.zero_centered else torch.ones
        self.weight = nn.Parameter(init(width)) if c["affine"] else None
        self.bias = nn.Parameter(torch.zeros(width)) if c["bias"] else None

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
