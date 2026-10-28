"""Softmax attention, exported: the masks, what every softmax attention layer shares,
and ``gqa`` with its ports (RoPE is `rope`'s, the indexer `sparse`'s, MLA `mla`'s).

A layer attends by transformers' attention interface (eager, sdpa, flash, flex) and
keeps its keys and values in transformers' cache. Where a setting needs the logits
(sinks, a softcap, a relative bias, a length scale, an indexer's kept keys) it
computes the softmax itself, `softmax_attention`, under every implementation: sdpa
and flash attention would drop sinks with no word, and sdpa has no softcap.

The masks are made before any layer runs, since the cache grows as layers run
(`make_mask`): transformers' own for the interface, and the sizes of every cache
layer, from which a layer that computes its softmax makes its mask (`dense_mask`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import (
    CacheLayerMixin,
    DynamicIndexedLayer,
    LinearAttentionAndFullAttentionLayer,
)
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
    sliding_window_overlay,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .common import ACTIVATIONS, Ctx, Norm, Site
from .conv import CausalConv
from .rope import Rope, logit_gain, rotated_dims
from .sparse import Indexer, key_positions

__all__ = [
    "Attention",
    "IndexedHybridLayer",
    "Rope",
    "SoftmaxAttention",
    "make_mask",
    "softmax_attention",
]

#: The cache layers whose keys a mask of each kind is sized by.
FAMILIES = {
    "full_attention": (
        "full_attention",
        "hybrid",
        "indexed_attention",
        "indexed_hybrid",
    ),
    "sliding_attention": ("sliding_attention", "hybrid_sliding"),
}
MIN = torch.finfo(torch.float32).min  # the logit of a masked key, as bux's


# --- cache layers ---------------------------------------------------------------------


class IndexedHybridLayer(LinearAttentionAndFullAttentionLayer):
    """transformers' cache layer of a sparse layer with a conv on its keys and values,
    ``indexed_hybrid``: its keys and values, the conv's two states (as a ``hybrid``
    layer keeps them) and the indexer's keys (as an ``indexed_attention`` layer)."""

    _layer_type = "indexed_hybrid"  # registers it with transformers' caches
    update_indexer = DynamicIndexedLayer.update_indexer
    lazy_initialization_indexer = DynamicIndexedLayer.lazy_initialization_indexer

    def __init__(self, number_of_states: int = 2, **kwargs: Any):
        super().__init__(number_of_states=number_of_states)
        self.indexer_keys: torch.Tensor | None = None
        self.is_indexer_initialized = False

    def _indexer(self, change: Any) -> None:
        """Replace the indexer's keys by `change` of them, once there are some."""
        if self.indexer_keys is not None and self.indexer_keys.numel() > 0:
            self.indexer_keys = change(self.indexer_keys)

    def offload(self) -> None:
        super().offload()
        self._indexer(lambda k: k.to("cpu", non_blocking=True))

    def prefetch(self) -> None:
        super().prefetch()
        self._indexer(lambda k: k.to(self.device, non_blocking=True))

    def reset(self) -> None:
        super().reset()
        self.indexer_keys, self.is_indexer_initialized = None, False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        self._indexer(lambda k: k.index_select(0, beam_idx.to(k.device)))

    def crop(self, tokens_to_remove: int) -> None:
        super().crop(tokens_to_remove)
        if tokens_to_remove:
            self._indexer(lambda k: k[:, : -abs(tokens_to_remove)])

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        self._indexer(lambda k: k.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        self._indexer(lambda k: k[indices])


# --- masks ----------------------------------------------------------------------------


@dataclass
class Dense:
    """A mask as a layer that computes its softmax takes it.

    Attributes:
        keep: Which keys each query attends to, b 1 t s.
        offsets: How far each key is behind each query, t s, in the row's tokens.
    """

    keep: torch.Tensor
    offsets: torch.Tensor


def make_mask(ctx: Ctx, kind: str, arg: Any) -> Any:
    """Return the pass's mask of `kind`, made before any layer runs.

    - ``full_attention``, ``sliding_attention``: transformers' mask for the run's
      attention implementation, sized by the first cache layer of the kind
      (`FAMILIES`), causal and `arg` keys long at most (a window) if not None. None
      when no cache layer is of the kind: no layer reads it.
    - ``sizes``: each cache layer's sizes before the pass, `cache_sizes`.
    - ``dense``: `dense_mask` of ``arg``, a cache layer and a window.

    Raises:
        NotImplementedError: generate made the masks itself, by layer type, as it does
            for a compileable cache (a static one), which is not exported yet.
    """
    if isinstance(ctx.attention_mask, dict):
        raise NotImplementedError(
            "compileable caches (a static cache) are not exported yet"
        )
    if kind == "sizes":
        return cache_sizes(ctx)
    if kind == "dense":
        return dense_mask(ctx, *arg)
    layer = _first_layer(ctx.config, kind, arg)
    if layer is None and ctx.cache is not None:
        return None
    return interface_mask(ctx, kind, arg, layer)


def interface_mask(ctx: Ctx, kind: str, window: int | None, layer: int | None) -> Any:
    """Return transformers' mask of `kind` for the run's attention implementation,
    sized by the cache layer `layer`, as its modeling files make them."""
    b, t = ctx.tokens.shape
    like = torch.empty((b, t, 1), dtype=ctx.dtype, device=ctx.tokens.device)
    if kind == "sliding_attention" and layer is not None:
        return create_sliding_window_causal_mask(
            config=ctx.config.per_layer_config[layer],
            inputs_embeds=like,
            attention_mask=ctx.attention_mask,
            past_key_values=ctx.cache,
            position_ids=ctx.positions,
            layer_idx=layer,
        )
    return create_causal_mask(
        config=ctx.config,
        inputs_embeds=like,
        attention_mask=ctx.attention_mask,
        past_key_values=ctx.cache,
        position_ids=ctx.positions,
        and_mask_function=None if window is None else sliding_window_overlay(window),
        layer_idx=layer,
    )


def _first_layer(config: Any, kind: str, window: int | None) -> int | None:
    """Return the first cache layer of `kind`, of the `window` if sliding."""
    for i, layer in enumerate(config.layer_types or []):
        if layer not in FAMILIES[kind]:
            continue
        if (
            kind == "full_attention"
            or config.per_layer_config[i].sliding_window == window
        ):
            return i
    return None


def cache_sizes(ctx: Ctx) -> dict[int, tuple[int, int, int]] | None:
    """Return the sizes of every cache layer of keys before the pass adds to it, by its
    index: the tokens before the queries, and the keys it will return and the first
    one's index in the row, as transformers' masks take them. None without a cache."""
    if ctx.cache is None:
        return None
    t, out = ctx.tokens.shape[1], {}
    for i, layer in enumerate(ctx.cache.layers):
        if isinstance(layer, CacheLayerMixin):
            out[i] = (layer.get_seq_length(), *layer.get_mask_sizes(t))
    return out


def dense_mask(ctx: Ctx, slot: int | None, window: int | None) -> Dense:
    """Return the mask of the keys the cache layer `slot` returns (the pass's own
    without one): causal, `window` long at most if not None, and no padding."""
    b, t = ctx.tokens.shape
    sizes = ctx.mask("sizes") or {}
    seen, s, first = sizes.get(slot, (0, t, 0))
    device = ctx.tokens.device
    queries = torch.arange(seen, seen + t, device=device)
    offsets = queries[:, None] - torch.arange(first, first + s, device=device)
    keep = offsets >= 0
    if window is not None:
        keep &= offsets < window
    keep = keep.expand(b, 1, t, s)
    pad = ctx.attention_mask
    if pad is not None and pad.ndim == 2:
        keep = keep & pad[:, None, None, first : first + s].bool()
    return Dense(keep, offsets)


# --- attention ------------------------------------------------------------------------


def eager_attention(
    module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention of q, b h t k, over k and v, b g s k: the eager fallback of
    transformers' attention interface. Returns b t h k, and the probabilities."""
    groups = module.num_key_value_groups
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    w = q @ k.transpose(2, 3) * scaling
    if mask is not None:
        w = w + mask
    w = w.softmax(-1, dtype=torch.float32).to(q.dtype)
    return (w @ v).transpose(1, 2).contiguous(), w


def softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    scale: float,
    softcap: float | None = None,
    sinks: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax attention computed here, as bux's ``xla`` method and its logit features
    compute it, in float32.

    Args:
        q: The queries, b h t dk.
        k: The keys, b g s dk, each of g heads shared by h / g query heads.
        v: The values, b g s dv.
        keep: Which keys each query attends to, b 1 t s or b h t s.
        scale: What every logit is multiplied by.
        softcap: Logits go to ``softcap * tanh(logits / softcap)``; None for none.
        sinks: A logit per head beside the keys', on no value; None for none.
        bias: Added to the scaled logits, b h t s (a relative bias); None for none.
        gain: The logits are multiplied by it after the bias, b 1 t 1 (a length
            scale); None for none.

    Returns:
        Every query's values weighted by its probabilities, b t h dv; zeros for a query
        that keeps no key.
    """
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    logits = (q.float() @ k.float().transpose(2, 3)) * scale
    if bias is not None:
        logits = logits + bias
    if gain is not None:
        logits = logits * gain
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits.masked_fill(~keep, MIN)
    if sinks is None:
        probs = logits.softmax(-1)
    else:
        sink = sinks.float()[None, :, None, None].expand(*logits.shape[:-1], 1)
        probs = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
    probs = probs.masked_fill(~keep, 0)
    return (probs.to(v.dtype) @ v).transpose(1, 2)


@dataclass
class KV:
    """The keys and values a layer attends to, and the cache layer that keeps them:
    what its ``kv`` port carries.

    Attributes:
        k: The keys, b g s dk, as the cache returns them.
        v: The values, b g s dv.
        slot: The cache layer keeping them; None without a cache.
        kind: The kind of mask that cache layer sizes, a key of `FAMILIES`.
        window: The window that cache layer keeps; None for every key.
    """

    k: torch.Tensor
    v: torch.Tensor
    slot: int | None
    kind: str
    window: int | None


def reads(ports: list, name: str) -> bool:
    """Whether the layer reads (or pops) its port `name`."""
    return any(n == name and m != "write" for n, _, m in ports)


def sparse(c: dict, ports: list) -> bool:
    """Whether the layer attends to kept keys: it runs an indexer, or reads the keys
    another's kept (bux's ``gqa.pick``)."""
    return c.get("indexer") is not None or any(n == "index" for n, _, _ in ports)


def cache_layer(c: dict, ports: list) -> tuple[str, dict] | None:
    """Return the cache layer a softmax attention layer keeps, and its settings, or
    None for a layer that reads its keys and values from its ``kv`` port.

    A sparse layer keeps every key, which its kept keys index (an indexer's in an
    ``indexed_attention`` layer, which keeps the indexer's keys too); a layer with a
    sliding window keeps that window (``sliding_attention``); one with a conv on its
    keys and values keeps the conv's two states beside them (``hybrid``, and
    ``indexed_hybrid`` with an indexer of its own, `IndexedHybridLayer`).
    """
    if reads(ports, "kv"):
        return None
    conv = {"number_of_conv_states": 2} if c.get("kv_conv_size") else {}
    window = c["sliding_window_size"]
    if sparse(c, ports):
        if reads(ports, "index"):
            return ("hybrid" if conv else "full_attention"), conv
        return ("indexed_hybrid", conv) if conv else ("indexed_attention", {})
    if window is None:
        return ("hybrid" if conv else "full_attention"), conv
    kind = "hybrid_sliding" if conv else "sliding_attention"
    return kind, {"sliding_window": window, **conv}


def per_head(kept: torch.Tensor, heads: int) -> torch.Tensor:
    """Return which keys each query keeps, b 1 t s for all heads, from b t s; or
    b h t s for each of the `heads`, its kv head's group's, from b t g s (bux's
    ``_per_head``)."""
    if kept.ndim == 3:
        return kept[:, None]
    return kept.transpose(1, 2).repeat_interleave(heads // kept.shape[2], 1)


def computes_logits(c: dict) -> bool:
    """Whether the layer computes its softmax (`softmax_attention`): a setting needs
    the logits that transformers' attention implementations do not all take."""
    logit_features = (c["relative_bias"], c["log_scale"] is not None, c["indexer"])
    return any((c["sinks"], c["softcap"] is not None, *logit_features))


class SoftmaxAttention(nn.Module):
    """What the softmax attention layers share once they have q, k and v: bux's
    ``SoftmaxAttention`` and ``FullLogits``, and ``Sparse``'s choice of keys.

    It keeps the keys and values in the cache, picks the keys each query keeps (an
    indexer's, or the ``index`` port's), attends to them and makes the output: its
    norm, gate and projection. Dropout is training's alone, and is left out.
    """

    o: nn.Linear  # the output projection, which the layer makes
    indexer: Indexer | None

    @staticmethod
    def cache_type(c: dict, ports: list) -> tuple[str, dict] | None:
        """The cache layer it keeps, as ``config.layer_types`` names it, and its
        settings (`cache_layer`)."""
        return cache_layer(c, ports)

    @staticmethod
    def masks(c: dict) -> list[tuple[str, Any]]:
        """The masks it reads, as ``ctx.mask`` takes them."""
        if computes_logits(c):
            return [("sizes", None)]
        w = c["sliding_window_size"]
        return [
            ("sizes", None),
            ("full_attention", None) if w is None else ("sliding_attention", w),
        ]

    def __init__(
        self, c: dict, width: int, site: Site, heads: int, dk: int, dv: int, bias: bool
    ):
        super().__init__()
        self.h, self.dv, self.ports = heads, dv, site.ports
        gate_width = heads if c["gate_per_head"] else heads * dv
        self.gate = nn.Linear(width, gate_width, bias=bias) if c["gated"] else None
        self.gate_activation = ACTIVATIONS[c["gate_activation"]]
        head_bias = c["gated"] and c["gate_head_bias"]
        self.gate_bias = nn.Parameter(torch.zeros(heads)) if head_bias else None
        rel = c["relative_bias"]
        self.rel_proj = (
            nn.Linear(width, heads * rel["rank"], bias=bias) if rel else None
        )
        self.rel_table = (
            nn.Parameter(torch.zeros(rel["rank"], rel["extent"])) if rel else None
        )
        self.log_scale, self.log_scale_floor = c["log_scale"], c["log_scale_floor"]
        self.scale = (c["scale"] or dk**-0.5) * logit_gain(c["rope"])
        self.softcap, self.window = c["softcap"], c["sliding_window_size"]
        self.sinks = nn.Parameter(torch.zeros(heads)) if c["sinks"] else None
        out_norm = c["output_norm"]
        self.out_norm = Norm(out_norm, dv) if out_norm else None
        self.computes_logits = computes_logits(c)
        # The cache layer it keeps, or would keep if it did not read the kv port's.
        own = type(self).cache_type(c, [p for p in site.ports if p[0] != "kv"])
        sliding = own is not None and own[0] in FAMILIES["sliding_attention"]
        self.kind = "sliding_attention" if sliding else "full_attention"
        self.cache_window = self.window if self.kind == "sliding_attention" else None
        self.config, self.is_causal = site.config, True
        self.num_key_value_groups = 1  # the interface's; set at each call

    # --- ports -----------------------------------------------------------------------

    def read(self, ctx: Ctx, name: str) -> Any:
        """Return what the layer reads (or pops) from its port `name`."""
        tap, mode = next((t, m) for n, t, m in self.ports if n == name and m != "write")
        return ctx.read(tap, mode == "pop")

    def write(self, ctx: Ctx, name: str, value: Any) -> None:
        """Push `value` to the tap of port `name`, if the layer writes that port."""
        for n, tap, mode in self.ports:
            if n == name and mode == "write":
                ctx.push(tap, value)

    # --- keys, values and kept keys ---------------------------------------------------

    def cached(self, k: torch.Tensor, v: torch.Tensor, ctx: Ctx, slot: int) -> KV:
        """Return the keys and values, b t g d, with those the cache layer `slot`
        keeps before them, as `KV`."""
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if ctx.cache is None:
            return KV(k, v, None, self.kind, self.cache_window)
        k, v = ctx.cache.update(k, v, slot)
        return KV(k, v, slot, self.kind, self.cache_window)

    def kept(
        self, x: torch.Tensor, query: torch.Tensor, ctx: Ctx, kv: KV
    ) -> torch.Tensor | None:
        """Return which keys of `kv` each query keeps, b t s, or b t g s for each kv
        head's group of query heads: the ``index`` port's, or the indexer's top-k
        (from `query` and the input `x`); None for all."""
        if reads(self.ports, "index"):
            kept = self.read(ctx, "index")
        elif self.indexer is not None:
            seen = ctx.mask("dense", (kv.slot, self.window)).keep[:, 0]
            keys = self.indexer.keys(x, ctx.positions)
            if ctx.cache is not None and kv.slot is not None:
                keys = ctx.cache.update_indexer(keys, kv.slot)
            order = key_positions(ctx, keys.shape[1])
            kept, _ = self.indexer(x, query, ctx.positions, keys, seen, order)
        else:
            return None
        self.write(ctx, "index", kept)
        return kept

    # --- attention -------------------------------------------------------------------

    def attend(
        self, x: torch.Tensor, q: torch.Tensor, kv: KV, ctx: Ctx, kept: Any = None
    ) -> torch.Tensor:
        """Return every query's softmax-weighted values, b t h dv, of the queries q,
        b t h dk, over `kv`: by transformers' interface, or by `softmax_attention`
        where the layer computes its logits, keeps some keys, or reads keys a cache
        layer of another window keeps."""
        q = q.transpose(1, 2)
        if self.computes_logits or kept is not None or self._foreign(kv, ctx):
            mask = ctx.mask("dense", (kv.slot, self.window))
            keep = mask.keep if kept is None else mask.keep & per_head(kept, self.h)
            bias, gain = self._relative_bias(x, mask.offsets), self._length_scale(ctx)
            return softmax_attention(
                q, kv.k, kv.v, keep, self.scale, self.softcap, self.sinks, bias, gain
            )
        self.num_key_value_groups = self.h // kv.k.shape[1]
        attend = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention
        )
        mask = ctx.mask(kv.kind, kv.window)
        out, _ = attend(
            self,
            q,
            kv.k,
            kv.v,
            mask,
            dropout=0.0,
            scaling=self.scale,
            sliding_window=self.window,
        )
        return out

    def _foreign(self, kv: KV, ctx: Ctx) -> bool:
        """Whether `kv` comes from a cache layer of another kind or window than the
        layer's own (a ``kv`` port's).

        Raises:
            NotImplementedError: It keeps a window shorter than the layer's, while
                decoding.
        """
        if (kv.kind, kv.window) == (self.kind, self.cache_window):
            return False
        short = kv.window is not None and (
            self.window is None or self.window > kv.window
        )
        if ctx.cache is not None and short:
            raise NotImplementedError(
                f"a layer with sliding_window_size={self.window} reads keys and values "
                f"of a layer that caches {kv.window} of them"
            )
        return True

    def _relative_bias(self, x: torch.Tensor, offsets: torch.Tensor) -> Any:
        """The bias <r[b, t, h], table[:, t - s]> of each key, b h t s; keys ahead of
        the query, or extent or more behind, get none. None without one."""
        if self.rel_proj is None or self.rel_table is None:
            return None
        b, t, _ = x.shape
        r = self.rel_proj(x).view(b, t, self.h, -1)
        extent = self.rel_table.shape[-1]
        rows = self.rel_table[:, offsets.clamp(0, extent - 1)]
        near = (offsets >= 0) & (offsets < extent)
        return torch.einsum("bthr,rts->bhts", r, rows) * near

    def _length_scale(self, ctx: Ctx) -> Any:
        """The factor on each query's logits for its position n, counted from 1:
        1 + log_scale * ln(max(n / floor, 1)), b 1 t 1. None without one."""
        if self.log_scale is None:
            return None
        ahead = ((ctx.positions + 1) / self.log_scale_floor).clamp(min=1.0)
        return (1 + self.log_scale * torch.log(ahead))[:, None, :, None]

    def output(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Normalize and gate the heads' outputs, b t h dv, then project them back to
        the width."""
        b, t, _ = x.shape
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is not None:
            gate = self.gate(x).view(b, t, self.h, -1)
            if self.gate_bias is not None:
                gate = gate + self.gate_bias.to(gate.dtype)[:, None]
            out = out * self.gate_activation(gate)
        return self.project(out)

    def project(self, out: torch.Tensor) -> torch.Tensor:
        """Project the heads' outputs, b t h dv, back to the width."""
        return self.o(out.flatten(2))


def check_reader(c: dict, ports: list) -> None:
    """Raise for what a layer that reads kv does not export: a v port (its values are
    the writer's, all of them while decoding) or an indexer of its own (whose keys it
    would have no cache layer for)."""
    if any(n == "v" for n, _, _ in ports):
        raise NotImplementedError("gqa: the v port of a layer that reads kv")
    if c["indexer"] and not reads(ports, "index"):
        raise NotImplementedError("gqa: an indexer in a layer that reads kv")


class Attention(SoftmaxAttention):
    """bux's ``gqa``, every setting: q, k and v projections (one each, or fused), a
    causal conv on k and v, QK-norm (of each head, or of all heads together under
    ``qk_norm_all_heads``) then RoPE, and its ports: ``v`` (value residual), ``kv``
    (cross-layer KV sharing) and ``index`` (an indexer's kept keys).

    A layer that reads ``kv`` attends to the writer's keys and values as the writer's
    cache layer keeps them, all of them while decoding, and keeps none of its own; it
    takes the writer's mask, so its window may be no longer than the writer's. A
    layer that reads ``kv`` writes or reads no ``v``.
    """

    bux_types = ("gqa",)

    def __init__(self, c: dict, width: int, site: Site):
        h, g = c["num_heads"], c["num_kv_heads"]
        dk = c["head_dim"] or width // h
        dv = c["v_head_dim"] or dk
        super().__init__(c, width, site, h, dk, dv, c["bias"])
        self.g, self.dk = g, dk
        self.reads_kv = reads(site.ports, "kv")
        if self.reads_kv:
            check_reader(c, site.ports)
        self._projections(c, width, h, g, dk, dv)
        qk, self.all_heads = c["qk_norm"], c["qk_norm_all_heads"]
        q_width, k_width = (h * dk, g * dk) if self.all_heads else (dk, dk)
        self.q_norm = Norm(qk, q_width) if qk else None
        self.k_norm = Norm(qk, k_width) if qk and not self.reads_kv else None
        self.rope = Rope(c["rope"], dk) if c["rope"] else None
        self.power_scale = c["power_scale"], c["power_scale_length"]
        self.value_residual = c["value_residual"]
        trained = reads(site.ports, "v") and self.value_residual != "fixed"
        shape = (2,) if self.value_residual == "learned" else ()
        self.v_mix = nn.Parameter(torch.full(shape, 0.5)) if trained else None
        index = c["indexer"] and not reads(site.ports, "index")
        rotated = rotated_dims(dk, c["rope"])
        self.indexer = (
            Indexer(c["indexer"], width, width, c["rope"], rotated) if index else None
        )

    def _projections(
        self, c: dict, width: int, h: int, g: int, dk: int, dv: int
    ) -> None:
        """Make q, k and v's projections (fused or not; q alone if the layer reads
        kv), the output's, and the conv on k and v."""
        bias = c["bias"]
        out_bias = bias if c["output_bias"] is None else c["output_bias"]
        fused = c["fused_qkv"] and not self.reads_kv
        split = not (fused or self.reads_kv)
        self.qkv = (
            nn.Linear(width, h * dk + g * (dk + dv), bias=bias) if fused else None
        )
        self.q = None if fused else nn.Linear(width, h * dk, bias=bias)
        self.k = nn.Linear(width, g * dk, bias=bias) if split else None
        self.v = nn.Linear(width, g * dv, bias=bias) if split else None
        self.o = nn.Linear(h * dv, width, bias=out_bias)
        size = None if self.reads_kv else c["kv_conv_size"]
        self.k_conv = CausalConv(g * dk, size, False) if size else None
        self.v_conv = CausalConv(g * dv, size, False) if size else None

    def forward(self, x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        q, k, v = self._project(x)
        q = self._queries(q, ctx)
        if self.reads_kv:
            kv = self.read(ctx, "kv")
        else:
            slot = ctx.cache_slot()
            k, v = self._keys_and_values(k, v, ctx, slot)
            if reads(self.ports, "v"):
                v = self._mix_values(v, self.read(ctx, "v"))
            self.write(ctx, "v", v)
            kv = self.cached(k, v, ctx, slot)
        self.write(ctx, "kv", kv)
        kept = self.kept(x, x, ctx, kv)
        return self.output(x, self.attend(x, q, kv, ctx, kept))

    def _project(self, x: torch.Tensor) -> list[Any]:
        """Project x to q, k and v, b t _; k and v are None if the layer reads kv."""
        if self.qkv is not None:
            q, k = self.h * self.dk, self.g * self.dk
            return list(self.qkv(x).split([q, k, self.g * self.dv], -1))
        assert self.q is not None
        kv = (
            [None, None] if self.k is None or self.v is None else [self.k(x), self.v(x)]
        )
        return [self.q(x), *kv]

    def _queries(self, q: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        """Split the projected queries into heads with QK-norm (`_split_normed`),
        then apply RoPE and the power scale."""
        q = self._split_normed(q, self.h, self.q_norm)
        if self.rope is not None:
            q = self.rope(q, ctx.positions)
        power, length = self.power_scale
        if power is None:
            return q
        # (1 + n / length) ** power at the query's position n, counted from 1.
        scale = torch.exp(power * torch.log1p((ctx.positions + 1) / length))
        return (q * scale[..., None, None]).to(q.dtype)

    def _keys_and_values(
        self, k: torch.Tensor, v: torch.Tensor, ctx: Ctx, slot: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the causal conv to the projected keys and values (its states 0 and 1 in
        the cache layer `slot`), split them into heads, the keys with QK-norm
        (`_split_normed`), then apply RoPE to the keys."""
        b, t, _ = k.shape
        if self.k_conv is not None and self.v_conv is not None:
            k, v = k + self.k_conv(k, ctx, slot, 0), v + self.v_conv(v, ctx, slot, 1)
        k, v = self._split_normed(k, self.g, self.k_norm), v.view(b, t, self.g, self.dv)
        return (k if self.rope is None else self.rope(k, ctx.positions)), v

    def _split_normed(
        self, x: torch.Tensor, heads: int, norm: Norm | None
    ) -> torch.Tensor:
        """Split projected queries or keys, b t _, into `heads` heads, normalized by
        `norm`: each head, or all of them together before the split under
        ``qk_norm_all_heads``."""
        b, t, _ = x.shape
        if self.all_heads and norm is not None:
            x, norm = norm(x), None
        x = x.view(b, t, heads, self.dk)
        return x if norm is None else norm(x)

    def _mix_values(self, v: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
        """The value residual a v + b v1: a = b = 0.5 fixed, both learned, or a learned
        and b = 1 - a."""
        if self.v_mix is None:
            return 0.5 * v + 0.5 * v1
        w = self.v_mix
        if self.value_residual == "learned":
            return w[0] * v + w[1] * v1
        return w * v + (1 - w) * v1
