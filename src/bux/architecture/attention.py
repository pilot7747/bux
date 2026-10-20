"""Softmax attention: the core that its layers share, and causal grouped-query attention."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int

from bux import config as cfg
from bux.architecture.conv import CausalConv
from bux.architecture.functional import activation, softcap
from bux.architecture.indexer import Indexer, from_stream
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.rope import logit_gain, rotate
from bux.architecture.segments import Segments
from bux.lang import ir


def _read(taps: Taps, ports: tuple[ir.Port, ...], name: str) -> Any:
    """What the layer reads (or pops) from its port `name`."""
    port = next(p for p in ports if p.name == name and p.mode != "write")
    stack = taps[port.tap]
    return stack.pop() if port.mode == "pop" else stack[-1]


def _write(taps: Taps, ports: tuple[ir.Port, ...], name: str, value: object) -> None:
    """Push `value` to the tap of port `name`, if the layer writes that port."""
    for port in ports:
        if port.name == name and port.mode == "write":
            taps.setdefault(port.tap, []).append(value)


def _reads(ports: Sequence[ir.Port], name: str) -> bool:
    """Whether the layer reads (or pops) its port `name`."""
    return any(p.name == name and p.mode != "write" for p in ports)


def _writes(ports: Sequence[ir.Port], name: str) -> bool:
    """Whether the layer writes its port `name`."""
    return any(p.name == name and p.mode == "write" for p in ports)


def _own_indexer(
    c: cfg.GQAConfig | cfg.MLAConfig | cfg.CSAConfig, ports: Sequence[ir.Port]
) -> cfg.SparseIndexerConfig | None:
    """The indexer the layer runs: none when it reads its selection from the index port
    instead, so a layer that does needs none in its config either."""
    own = None if _reads(ports, "index") else c.indexer
    needs = [
        p.name
        for p in ports
        if p.name == "candidates" or (p.name == "index" and p.mode == "write")
    ]
    if needs and own is None:
        raise ValueError(
            f"{c.type}: the '{needs[0]}' port needs an indexer that the layer runs: "
            "give it indexer=indexer(...), and do not read index"
        )
    if own is not None and own.block_size > 1 and "candidates" in needs:
        raise NotImplementedError("candidate pools of an indexer with block_size > 1")
    return own


def _rows(docs: Segments | None, b: int, t: int) -> Segments:
    """The documents of the rows, or one a row."""
    if docs is not None:
        return docs
    return Segments.of(jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t)))


def _causal(
    t: int, window: int | None, docs: Segments | None
) -> Bool[Array, "*rows t s"]:
    """Which of t keys at the same positions each of t queries sees: none ahead of it,
    none `window` or more behind, and none of another document."""
    index = jnp.arange(t)
    offset = index[:, None] - index[None, :]
    visible = offset >= 0
    if window is not None:
        visible &= offset < window
    if docs is not None:
        visible = visible & (docs.ids[:, :, None] == docs.ids[:, None, :])
    return visible


def _relative_bias(
    r: Float[Array, "b t h rank"],
    table: Float[Array, "rank extent"],
    offset: Int[Array, "t s"],
) -> Float[Array, "b h t s"]:
    """The bias <r[b, t, h], table[:, t - s]>; keys ahead of the query, or extent or
    more behind, get none."""
    extent = table.shape[-1]
    rows = table[:, jnp.clip(offset, 0, extent - 1)]
    near = (offset >= 0) & (offset < extent)
    return jnp.einsum("bthr,rts->bhts", r, rows) * near


def _length_scale(
    pos: Int[Array, "*lead t"], log_scale: float, floor: int
) -> Float[Array, "*lead t 1"]:
    """The factor on each query's logits, for its position: 1 + log_scale *
    ln(max(n / floor, 1)), at position n counted from 1."""
    ahead = jnp.maximum((pos + 1) / floor, 1.0)
    return (1 + log_scale * jnp.log(ahead))[..., None]


def _softmax_with_sinks(
    logits: Float[Array, "b h t s"], sinks: Float[Array, " h"]
) -> Float[Array, "b h t s"]:
    """Softmax over the keys, next to a per-head sink logit that takes a share of the
    probability and attends to nothing."""
    b, h, t, _ = logits.shape
    sink = jnp.broadcast_to(sinks[None, :, None, None], (b, h, t, 1))
    return jax.nn.softmax(jnp.concatenate([logits, sink], -1))[..., :-1]


class SoftmaxAttention(nnx.Module):
    """What the softmax attention layers share, once they have their queries, keys and
    values: the logits, with their scale, relative bias, length scale and softcap; the
    causal, window, document and top-k masks; sinks and dropout; the output's norm,
    gate and projection; and the sparse attention indexer, with its ports. A layer
    makes its own projections, then calls this, then builds its indexer.

    Ports: ``index`` written pushes the indexer's top-k, a mask of the keys each query
    attends to; read uses the tap's instead of running an indexer, so a layer that reads
    it builds none, whatever its config says. ``candidates`` written pushes the
    indexer's candidate pool, and read keeps its top-k inside the tap's.
    """

    o: nnx.Linear  # the output projection, which the layer makes
    indexer: Indexer | None  # which the layer builds, if it runs one

    def __init__(
        self,
        c: cfg.QuadraticAttentionConfig,
        width: int,
        heads: int,
        kv_heads: int,
        v_head_dim: int,
        scale: float,
        bias: bool,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        self.heads, self.kv_heads = heads, kv_heads
        self.ports = tuple(ports)
        dv, rel = v_head_dim, c.relative_bias

        def linear(n_in: int, n_out: int) -> nnx.Linear:
            return nnx.Linear(n_in, n_out, use_bias=bias, rngs=rngs)

        # The randomly initialized layers, after the layer's own projections.
        self.gate = (
            linear(width, heads if c.gate_per_head else heads * dv) if c.gated else None
        )
        self.rel_proj = linear(width, heads * rel.rank) if rel else None

        # Logits and probabilities
        self.scale = scale
        self.rel_table = nnx.Param(jnp.zeros((rel.rank, rel.extent))) if rel else None
        self.log_scale, self.log_scale_floor = c.log_scale, c.log_scale_floor
        self.softcap = c.softcap
        self.window = c.sliding_window_size
        self.sinks = nnx.Param(jnp.zeros(heads)) if c.sinks else None
        self.attention_dropout = (
            nnx.Dropout(c.attention_dropout, rngs=rngs) if c.attention_dropout else None
        )

        # Output
        self.out_norm = Norm(c.output_norm, dv) if c.output_norm else None
        self.gate_activation = c.gate_activation
        self.output_dropout = (
            nnx.Dropout(c.output_dropout, rngs=rngs) if c.output_dropout else None
        )

    def _attend(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h k"],
        k: Float[Array, "b s g k"],
        v: Float[Array, "b s g v"],
        docs: Segments | None,
        keep: Bool[Array, "b t s"] | None = None,
    ) -> Float[Array, "b t h v"]:
        """Every query's average of the values of the keys it sees, of those it `keep`s
        if given, by the softmax of their logits. This makes all t by t logits, and the
        mask with them; a flash kernel skips the blocks no query sees, by the documents'
        ids (and the top-k, by the indexer's selection)."""
        visible = self._visible(docs, x.shape[1])
        if keep is not None:
            visible = visible & keep
        visible = visible[..., None, :, :]
        logits = self._logits(x, q, k, docs)
        logits = jnp.where(visible, logits, jnp.finfo(logits.dtype).min)
        if self.sinks is not None:
            probs = _softmax_with_sinks(logits, self.sinks[...])
        else:
            probs = jax.nn.softmax(logits)
        probs = jnp.where(visible, probs, 0)  # a query that sees no key gets nothing
        if self.attention_dropout is not None:
            probs = self.attention_dropout(probs)
        return jnp.einsum("bhts,bshd->bthd", probs, self._per_query_head(v))

    def _logits(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h k"],
        k: Float[Array, "b s g k"],
        docs: Segments | None,
    ) -> Float[Array, "b h t s"]:
        """Every query's logits for every key: scaled, with the relative bias, scaled
        for the query's position and soft-capped."""
        b, t, _ = x.shape
        logits = jnp.einsum("bthd,bshd->bhts", q, self._per_query_head(k)) * self.scale
        if self.rel_proj is not None and self.rel_table is not None:
            r = self.rel_proj(x).reshape(b, t, self.heads, -1)
            logits = logits + _relative_bias(r, self.rel_table[...], self._offsets(t))
        if self.log_scale is not None:
            at = jnp.arange(t) if docs is None else docs.positions[:, None]
            logits *= _length_scale(at, self.log_scale, self.log_scale_floor)
        return softcap(logits, self.softcap)

    def _offsets(self, t: int) -> Int[Array, "t s"]:
        """How far each key is behind each query."""
        index = jnp.arange(t)
        return index[:, None] - index[None, :]

    def _visible(self, docs: Segments | None, t: int) -> Bool[Array, "*rows t s"]:
        """Which keys each query sees: none ahead of it or a window or more behind, and
        none of another document."""
        return _causal(t, self.window, docs)

    def _per_query_head(self, kv: Float[Array, "b s g e"]) -> Float[Array, "b s h e"]:
        """Keys or values repeated for every query head of their group."""
        return jnp.repeat(kv, self.heads // self.kv_heads, axis=2)

    def _select(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"] | None,
        allowed: Bool[Array, "b t s"],
        order: Int[Array, "b s"],
        taps: Taps,
    ) -> Bool[Array, "b t s"] | None:
        """The keys each query attends to, of those it is `allowed`: what the index port
        reads, or the indexer's top-k, from its `query` input and `keys`; None for all.
        `order` says which key of its document each slot holds, -1 for none."""
        if _reads(self.ports, "index"):
            keep = _read(taps, self.ports, "index")
        elif self.indexer is not None:
            if keys is None:
                raise ValueError(
                    "the indexer has no keys: the kv port's layer has none"
                )
            pool = None
            if _reads(self.ports, "candidates"):
                pool = _read(taps, self.ports, "candidates")
            export = _writes(self.ports, "candidates")
            keep, pool = self.indexer(x, query, pos, keys, allowed, order, pool, export)
            _write(taps, self.ports, "candidates", pool)
        else:
            return None
        _write(taps, self.ports, "index", keep)
        return keep

    def _select_tokens(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        docs: Segments | None,
        taps: Taps,
    ) -> Bool[Array, "b t t"] | None:
        """`_select` where the keys are the tokens, and the indexer's keys come from them."""
        if self.indexer is None and not _reads(self.ports, "index"):
            return None
        b, t, _ = x.shape
        pos = None if docs is None else docs.positions
        keys = None if self.indexer is None else self.indexer.keys(x, pos)
        allowed = jnp.broadcast_to(self._visible(docs, t), (b, t, t))
        return self._select(
            x, query, pos, keys, allowed, _rows(docs, b, t).within, taps
        )

    def _output(
        self, x: Float[Array, "b t d"], out: Float[Array, "b t h v"]
    ) -> Float[Array, "b t d"]:
        """The head outputs, normalized, gated and projected back to the width."""
        b, t, _ = x.shape
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is not None:
            gate = activation(self.gate_activation, self.gate(x))
            out = out * gate.reshape(b, t, self.heads, -1)
        y = self._project(out)
        return y if self.output_dropout is None else self.output_dropout(y)

    def _project(self, out: Float[Array, "b t h v"]) -> Float[Array, "b t d"]:
        """The heads projected back to the width."""
        b, t, _, _ = out.shape
        return self.o(out.reshape(b, t, -1))


class Attention(SoftmaxAttention):
    """Causal grouped-query attention.

    Ports: `v` written pushes the values, read mixes the tap's values in with a learned
    weight (value residual); `kv` written pushes (k, v), read uses them instead of
    projecting its own (cross-layer KV sharing); `index` and `candidates` as for
    `SoftmaxAttention`, with the indexer's keys projected from the layer's input.

    Shapes: b rows, t queries and s keys (the same positions), d the width, h heads and
    g kv heads, k and v the head dims of keys and values.

    `_attend` is where a fused kernel goes: what it needs of packed rows is
    `Segments.ids`, a document id a token, and no t by t mask.
    """

    def __init__(
        self, c: cfg.GQAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
    ):
        h, kv = c.num_heads, c.num_kv_heads
        d = c.head_dim or width // h
        dv = c.v_head_dim or d
        self.head_dim, self.v_head_dim = d, dv
        reads = {p.name for p in ports if p.mode != "write"}
        own_kv = "kv" not in reads  # else the kv tap's keys and values replace its own
        indexer = _own_indexer(c, ports)

        def linear(n_in: int, n_out: int, bias: bool | None = None) -> nnx.Linear:
            use_bias = c.bias if bias is None else bias
            return nnx.Linear(n_in, n_out, use_bias=use_bias, rngs=rngs)

        # The randomly initialized layers; reordering them changes a seed's init.
        fused, split = own_kv and c.fused_qkv, own_kv and not c.fused_qkv
        self.qkv = linear(width, h * d + kv * (d + dv)) if fused else None
        self.q = None if fused else linear(width, h * d)
        self.k = linear(width, kv * d) if split else None
        self.v = linear(width, kv * dv) if split else None
        self.o = linear(h * dv, width, c.output_bias)
        conv = c.kv_conv_size if own_kv else None
        self.k_conv = CausalConv(kv * d, conv, False, rngs) if conv else None
        self.v_conv = CausalConv(kv * dv, conv, False, rngs) if conv else None
        scale = (c.scale or d**-0.5) * logit_gain(c.rope)
        super().__init__(c, width, h, kv, dv, scale, c.bias, ports, rngs)
        rotated = int(d * c.rope.fraction) // 2 * 2 if c.rope else 0
        self.indexer = (
            from_stream(indexer, width, width, c.rope, rotated, rngs)
            if indexer
            else None
        )

        # Queries, keys and values
        self.q_norm = Norm(c.qk_norm, d) if c.qk_norm else None
        self.k_norm = Norm(c.qk_norm, d) if c.qk_norm and own_kv else None
        self.rope = c.rope
        self.v_mix = nnx.Param(jnp.asarray(0.5)) if "v" in reads else None

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        docs = segments(taps)
        pos = None if docs is None else docs.positions
        q, *kv = self._projections(x)
        q = self._queries(q, pos)
        k, v = self._keys_and_values(*kv, taps) if kv else _read(taps, self.ports, "kv")
        if self.v_mix is not None:  # value residual
            v = self.v_mix * v + (1 - self.v_mix) * _read(taps, self.ports, "v")
        _write(taps, self.ports, "v", v)
        _write(taps, self.ports, "kv", (k, v))
        keep = self._select_tokens(x, x, docs, taps)
        if keep is None:
            return self._output(x, self._attend(x, q, k, v, docs))
        return self._output(x, self._attend(x, q, k, v, docs, keep))

    def _projections(self, x: Float[Array, "b t d"]) -> list[Float[Array, "b t _"]]:
        """The projected queries, then keys and values unless the layer reads them from
        its kv port: slices of one projection when fused, else one each."""
        if self.qkv is not None:
            q, k = self.heads * self.head_dim, self.kv_heads * self.head_dim
            return jnp.split(self.qkv(x), [q, q + k], axis=-1)
        return [p(x) for p in (self.q, self.k, self.v) if p is not None]

    def _queries(
        self, q: Float[Array, "b t _"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h k"]:
        """The queries, from their projection."""
        b, t, _ = q.shape
        q = q.reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.rope is not None:
            q = rotate(q, self.rope, pos)
        return q

    def _keys_and_values(
        self, k: Float[Array, "b s _"], v: Float[Array, "b s _"], taps: Taps
    ) -> tuple[Float[Array, "b s g k"], Float[Array, "b s g v"]]:
        """The keys and values, from their projections."""
        b, t, _ = k.shape
        if self.k_conv is not None and self.v_conv is not None:
            k, v = k + self.k_conv(k, taps), v + self.v_conv(v, taps)
        k = k.reshape(b, t, self.kv_heads, self.head_dim)
        v = v.reshape(b, t, self.kv_heads, self.v_head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.rope is not None:
            docs = segments(taps)
            k = rotate(k, self.rope, None if docs is None else docs.positions)
        return k, v


@register("gqa")
def _(
    c: cfg.GQAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Attention:
    return Attention(c, width, ports, ctx.rngs)
