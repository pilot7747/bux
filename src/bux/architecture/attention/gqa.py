"""Causal grouped-query attention, and what it shares with MLA and CSA.

`Attention` is the GPT-2 and Llama layer: projections, QK-norm, RoPE, the run's
`methods.attention` (``bux.parallel.context``), and the output projection. Two mixins
add to it: `FullLogits`, what needs all t by t logits, which only `xla`'s math makes
(relative bias, length scale, attention dropout), and `Sparse`, the indexer of sparse
attention, which attends by ``bux.architecture.attention.sparse``. `build_gqa` picks the
class the config needs.
"""

from collections.abc import Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.attention import sparse
from bux.architecture.attention.indexer import Indexer, Scored, Target, from_stream
from bux.architecture.attention.rope import logit_gain, rotate, rotated_dims
from bux.architecture.common.functional import activation
from bux.architecture.common.ports import read, reads, write, writes
from bux.architecture.common.segments import Segments
from bux.architecture.common.sharding import (
    linear,
    project_input,
    project_output,
    shard_heads,
)
from bux.architecture.layers.conv import CausalConv
from bux.architecture.layers.norm import Norm
from bux.architecture.registry import (
    Context,
    Taps,
    register,
    report,
    segments,
    training,
)
from bux.architecture.repeats import remat
from bux.lang import ir
from bux.parallel.context import (
    AttentionArgs,
    causal_mask,
    masked_softmax,
    scaled_logits,
    softcap,
    weighted_values,
)
from bux.train import flops

Names = tuple[parallel.Name, parallel.Name]  # a projection's input and output
SoftmaxConfig = cfg.GQAConfig | cfg.MLAConfig | cfg.CSAConfig


class SoftmaxAttention(nnx.Module):
    """What the softmax attention layers share once they have q, k and v: the scale,
    softcap, window and sinks of the attention method, and the output's norm, gate,
    projection and dropout. A layer makes its projections, `o` among them, first.

    Under tensor parallelism a device has whole heads: their columns of the input
    projections, their q, k, v and outputs, and their rows of `o`.
    """

    o: nnx.Linear  # the output projection, which the layer makes

    def __init__(
        self,
        config: SoftmaxConfig,
        width: int,
        heads: int,
        kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        bias: bool,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        self.heads, self.kv_heads = heads, kv_heads
        self.ports = tuple(ports)
        gate_width = heads if config.gate_per_head else heads * v_head_dim
        names = ("embed", ("heads", heads))

        # The randomly initialized layers, after the layer's own projections.
        self.gate = (
            linear(width, gate_width, names, bias, rngs, role="q")
            if config.gated
            else None
        )
        self._init_logit_layers(config, width, bias, rngs)
        self.output_dropout = (
            nnx.Dropout(config.output_dropout, rngs=rngs)
            if config.output_dropout
            else None
        )

        self.scale = (config.scale or qk_head_dim**-0.5) * logit_gain(config.rope)
        self.softcap = config.softcap
        self.window = config.sliding_window_size
        self.sinks = (
            nnx.Param(jnp.zeros(heads), names=("heads",)) if config.sinks else None
        )
        self.out_norm = (
            Norm(config.output_norm, v_head_dim) if config.output_norm else None
        )
        self.gate_activation = config.gate_activation

    def _init_logit_layers(
        self, config: SoftmaxConfig, width: int, bias: bool, rngs: nnx.Rngs
    ) -> None:
        """Make the layers of `FullLogits`' features."""

    def _attend(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b s g dk"],
        v: Float[Array, "b s g dv"],
        docs: Segments | None,
    ) -> Float[Array, "b t h dv"]:
        """Return every query's softmax-weighted values by the run's
        `methods.attention`, named `attention` with what its backward takes
        (``remat.named``); `x` is for `FullLogits`."""
        ids = None if docs is None else docs.ids
        args = AttentionArgs(ids, self.window, self.scale, self.softcap, self._sinks())
        method = parallel.method("attention")

        def attend(
            q: Float[Array, "b t h dk"],
            k: Float[Array, "b s g dk"],
            v: Float[Array, "b s g dv"],
            ids: Int[Array, "b t"] | None,
            sinks: Float[Array, " h"] | None,
        ) -> Float[Array, "b t h dv"]:
            return method(q, k, v, args._replace(ids=ids, sinks=sinks))

        return remat.named("attention", attend, q, k, v, ids, args.sinks)

    def _sinks(self) -> Float[Array, " h"] | None:
        return None if self.sinks is None else self.sinks[...]

    def _output(
        self, x: Float[Array, "b t d"], out: Float[Array, "b t h dv"]
    ) -> Float[Array, "b t d"]:
        """Normalize and gate the head outputs, then project them back to the width."""
        b, t, _ = x.shape
        out = checkpoint_name(out, "attention.out")
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is not None:
            gate = activation(self.gate_activation, self.gate(x))
            out = out * gate.reshape(b, t, self.heads, -1)
        y = self._project(shard_heads(out))
        return y if self.output_dropout is None else self.output_dropout(y)

    def _project(self, out: Float[Array, "b t h dv"]) -> Float[Array, "b t d"]:
        """Project the heads back to the width."""
        b, t, _, _ = out.shape
        return project_output(out.reshape(b, t, -1), self.o)


class Attention(SoftmaxAttention):
    """Causal grouped-query attention: MHA when num_kv_heads == num_heads, MQA when it
    is 1.

    Ports: `v` written pushes the values; read mixes the tap's values into the layer's,
    as `value_residual` says (ResFormer). `kv` written pushes (k, v); read uses them
    instead of projecting its own (cross-layer KV sharing).

    Shapes: b rows, t queries and s keys (the same positions), d the width, h heads and
    g kv heads, dk and dv the head dims of keys and values.
    """

    def __init__(
        self,
        config: cfg.GQAConfig,
        width: int,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        h, g = config.num_heads, config.num_kv_heads
        dk = config.head_dim or width // h
        dv = config.v_head_dim or dk
        self.head_dim, self.v_head_dim = dk, dv
        projects_kv = not reads(ports, "kv")  # else it takes the kv tap's
        heads, kv_heads = ("heads", h), ("kv_heads", g)  # flattened with their dims

        def proj(
            n_in: int, n_out: int, names: Names, role: str, bias: bool | None = None
        ) -> nnx.Linear:
            bias = config.bias if bias is None else bias
            return linear(n_in, n_out, names, bias, rngs, role=role)

        def conv(channels: int, size: int) -> CausalConv:
            return CausalConv(channels, size, False, rngs, channel=kv_heads)

        # The randomly initialized layers; reordering them changes a seed's init.
        fused = projects_kv and config.fused_qkv
        split = projects_kv and not config.fused_qkv
        # Fused, the split points of q, k and v do not fall on shard boundaries.
        self.qkv = (
            proj(width, h * dk + g * (dk + dv), ("embed", None), "q") if fused else None
        )
        self.q = None if fused else proj(width, h * dk, ("embed", heads), "q")
        self.k = proj(width, g * dk, ("embed", kv_heads), "kv") if split else None
        self.v = proj(width, g * dv, ("embed", kv_heads), "kv") if split else None
        self.o = proj(h * dv, width, (heads, "embed"), "o", config.output_bias)
        size = config.kv_conv_size if projects_kv else None
        self.k_conv = conv(g * dk, size) if size else None
        self.v_conv = conv(g * dv, size) if size else None
        super().__init__(config, width, h, g, dk, dv, config.bias, ports, rngs)

        qk_norm = config.qk_norm
        self.q_norm = Norm(qk_norm, dk) if qk_norm else None
        self.k_norm = Norm(qk_norm, dk) if qk_norm and projects_kv else None
        self.rope = config.rope
        self.value_residual = config.value_residual
        shape = (2,) if config.value_residual == "learned" else ()
        trained = reads(ports, "v") and config.value_residual != "fixed"
        self.v_mix = nnx.Param(jnp.full(shape, 0.5)) if trained else None

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        docs = segments(taps)
        x, projected = self._projections(x)
        q, k, v = self._heads(projected, docs, taps)
        return self._output(x, self._attend(x, q, k, v, docs))

    def _heads(
        self, projected: list[Float[Array, "b t _"]], docs: Segments | None, taps: Taps
    ) -> tuple[
        Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
    ]:
        """Return the queries, keys and values of the `projected` input, with the v
        port's values mixed in; the keys and values from the kv port if the layer
        reads it."""
        pos = None if docs is None else docs.positions
        q, *kv = projected
        q = self._queries(q, pos)
        if kv:
            k, v = self._keys_and_values(*kv, pos, taps)
        else:
            k, v = read(taps, self.ports, "kv")
        if reads(self.ports, "v"):
            v = self._mix_values(v, read(taps, self.ports, "v"))
        write(taps, self.ports, "v", v)
        write(taps, self.ports, "kv", (k, v))
        return q, k, v

    def _projections(
        self, x: Float[Array, "b t d"]
    ) -> tuple[Float[Array, "b t d"], list[Float[Array, "b t _"]]]:
        """Return the input as the layer takes it (`project_input`), and its
        projections: the queries, then the keys and values unless the layer reads them
        from its kv port, as slices of one projection when fused, else one each."""
        if self.qkv is not None:
            q, k = self.heads * self.head_dim, self.kv_heads * self.head_dim
            x, [qkv] = project_input(x, self.qkv)
            qkv = checkpoint_name(qkv, "qkv")
            return x, jnp.split(qkv, [q, q + k], axis=-1)
        projections = [p for p in (self.q, self.k, self.v) if p is not None]
        x, projected = project_input(x, *projections)
        return x, [checkpoint_name(p, "qkv") for p in projected]

    def _queries(
        self, q: Float[Array, "b t _"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h dk"]:
        """Reshape the projected queries into heads, then apply QK-norm and RoPE."""
        b, t, _ = q.shape
        q = q.reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.rope is not None:
            q = rotate(q, self.rope, pos)
        return shard_heads(q)

    def _keys_and_values(
        self,
        k: Float[Array, "b s _"],
        v: Float[Array, "b s _"],
        pos: Int[Array, "b s"] | None,
        taps: Taps,
    ) -> tuple[Float[Array, "b s g dk"], Float[Array, "b s g dv"]]:
        """Add the causal conv to the projected keys and values, reshape them into
        heads, then apply QK-norm and RoPE to the keys."""
        b, s, _ = k.shape
        if self.k_conv is not None and self.v_conv is not None:
            k, v = k + self.k_conv(k, taps), v + self.v_conv(v, taps)
        k = k.reshape(b, s, self.kv_heads, self.head_dim)
        v = v.reshape(b, s, self.kv_heads, self.v_head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.rope is not None:
            k = rotate(k, self.rope, pos)
        return shard_heads(k, "kv_heads"), shard_heads(v, "kv_heads")

    def _mix_values(
        self, v: Float[Array, "b s g dv"], v1: Float[Array, "b s g dv"]
    ) -> Float[Array, "b s g dv"]:
        """Return the value residual a v + b v1: a = b = 0.5 fixed, both learned, or a
        learned and b = 1 - a."""
        if self.v_mix is None:
            return 0.5 * v + 0.5 * v1
        w = self.v_mix[...]
        if self.value_residual == "learned":
            return w[0] * v + w[1] * v1
        return w * v + (1 - w) * v1


def token_mask(
    t: int, window: int | None, docs: Segments | None
) -> Bool[Array, "*b t s"]:
    """Return the causal mask of t queries over the t keys at their positions, within
    `window` and each query's document."""
    return causal_mask(t, window, None if docs is None else docs.ids)


def require_xla(feature: str) -> None:
    """Raise unless the run's attention is `xla`, the only method with `feature`, or
    `auto`, which takes xla for it."""
    on = parallel.active()
    name = on.methods.attention if on else "auto"
    if name not in ("xla", "auto"):
        raise NotImplementedError(
            f"methods.attention = {name} has no {feature}; only xla has"
        )


def _relative_bias(
    r: Float[Array, "b t h rank"],
    table: Float[Array, "rank extent"],
    offset: Int[Array, "t s"],
) -> Float[Array, "b h t s"]:
    """Return the bias <r[b, t, h], table[:, t - s]>; keys ahead of the query, or
    extent or more behind, get none."""
    extent = table.shape[-1]
    rows = table[:, jnp.clip(offset, 0, extent - 1)]
    near = (offset >= 0) & (offset < extent)
    return jnp.einsum("bthr,rts->bhts", r, rows) * near


def _length_scale(
    pos: Int[Array, "*lead t"], log_scale: float, floor: int
) -> Float[Array, "*lead t 1"]:
    """Return the factor on each query's logits for its position n, counted from 1:
    1 + log_scale * ln(max(n / floor, 1))."""
    ahead = jnp.maximum((pos + 1) / floor, 1.0)
    return (1 + log_scale * jnp.log(ahead))[..., None]


class FullLogits(SoftmaxAttention):
    """Softmax attention with features that need all t by t logits: relative bias,
    length scale and attention dropout. It runs `xla`'s masked softmax itself, so it
    raises under a method other than `xla` or `auto`."""

    def _init_logit_layers(
        self, config: SoftmaxConfig, width: int, bias: bool, rngs: nnx.Rngs
    ) -> None:
        rel, names = config.relative_bias, ("embed", ("heads", self.heads))
        self.rel_proj = (
            linear(width, self.heads * rel.rank, names, bias, rngs, in_compute=True)
            if rel
            else None
        )
        self.rel_table = nnx.Param(jnp.zeros((rel.rank, rel.extent))) if rel else None
        self.log_scale, self.log_scale_floor = config.log_scale, config.log_scale_floor
        rate = config.attention_dropout
        self.attention_dropout = nnx.Dropout(rate, rngs=rngs) if rate else None

    def _attend(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b s g dk"],
        v: Float[Array, "b s g dv"],
        docs: Segments | None,
    ) -> Float[Array, "b t h dv"]:
        feature = self._feature()
        if feature is None:
            return super()._attend(x, q, k, v, docs)
        require_xla(feature)
        return self._values(self._probabilities(x, q, k, docs), v)

    def _feature(self) -> str | None:
        """Return the first of the logit features that the layer uses, if any."""
        uses = {
            "relative_bias": self.rel_proj,
            "log_scale": self.log_scale,
            "attention_dropout": self.attention_dropout,
        }
        return next((name for name, used in uses.items() if used is not None), None)

    def _probabilities(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b s g dk"],
        docs: Segments | None,
        mask: Bool[Array, "*b t s"] | None = None,
    ) -> Float[Array, "b h t s"]:
        """Return the softmax over the keys `mask` keeps, the layer's own if none."""
        if mask is None:
            mask = self._mask(docs, x.shape[1])
        logits = self._logits(x, q, k, docs)
        return masked_softmax(logits, mask[..., None, :, :], self._sinks())

    def _values(
        self, probs: Float[Array, "b h t s"], v: Float[Array, "b s g dv"]
    ) -> Float[Array, "b t h dv"]:
        """Return every query's values weighted by its probabilities, after dropout."""
        if self.attention_dropout is not None:
            probs = self.attention_dropout(probs)
        return weighted_values(probs, v)

    def _logits(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b s g dk"],
        docs: Segments | None,
    ) -> Float[Array, "b h t s"]:
        """Return every query's logits for every key: scaled, plus the relative bias,
        scaled for the query's position, and soft-capped."""
        b, t, _ = x.shape
        logits = scaled_logits(q, k, self.scale)
        if self.rel_proj is not None and self.rel_table is not None:
            r = self.rel_proj(x).reshape(b, t, self.heads, -1)
            logits = logits + _relative_bias(r, self.rel_table[...], self._offsets(t))
        if self.log_scale is not None:
            at = jnp.arange(t) if docs is None else docs.positions[:, None]
            logits *= _length_scale(at, self.log_scale, self.log_scale_floor)
        return softcap(logits, self.softcap)

    def _offsets(self, t: int) -> Int[Array, "t s"]:
        """Return how far each key is behind each query."""
        index = jnp.arange(t)
        return index[:, None] - index[None, :]

    def _mask(self, docs: Segments | None, t: int) -> Bool[Array, "*b t s"]:
        return token_mask(t, self.window, docs)


def indexer_config(
    config: SoftmaxConfig, ports: Sequence[ir.Port]
) -> cfg.SparseIndexerConfig | None:
    """Return the config of the indexer the layer runs: None when it reads its keep
    mask from the index port instead, whatever its config says."""
    indexer = None if reads(ports, "index") else config.indexer
    needs = [
        p.name
        for p in ports
        if p.name == "candidates" or (p.name == "index" and p.mode == "write")
    ]
    if needs and indexer is None:
        raise ValueError(
            f"{config.type}: the '{needs[0]}' port needs an indexer that the layer "
            "runs: give it indexer=indexer(...), and do not read index"
        )
    if indexer is not None and indexer.block_size > 1 and "candidates" in needs:
        raise NotImplementedError("candidate pools of an indexer with block_size > 1")
    return indexer


def documents_or_rows(docs: Segments | None, b: int, t: int) -> Segments:
    """Return the packed documents, or one document per row."""
    if docs is not None:
        return docs
    return Segments.of(jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t)))


class Sparse(FullLogits):
    """Sparse attention (DeepSeek-V3.2): each query attends only to the keys an
    indexer keeps. Its loss alone trains it, against the attention's probabilities.

    The attention is ``sparse.attend``'s, with no t by t floats: tokamax's flash kernel
    with the kept keys as a mask on a GPU, else XLA a chunk of queries at a time.
    `FullLogits`' features take all t by t logits, as `xla` does.

    Ports: `index` written pushes the kept keys (``sparse.Kept``); read uses the tap's
    instead of running an indexer, so the layer builds none. `candidates` written
    pushes the candidate pool; read keeps the indexer's choice inside the tap's.
    """

    indexer: Indexer | None  # which the layer builds, if it runs one

    def _attend_kept(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        parts: Sequence[sparse.Part],
        docs: Segments | None,
        scored: Scored | None,
        taps: Taps,
    ) -> Float[Array, "b t h dv"]:
        """Return every query's softmax-weighted values over what it attends to of
        each part, all in one softmax, named `attention` (``remat.named``); in
        training, also report the indexer's loss from what it `scored`, against the
        attention to the last part."""
        if self._feature() is not None:
            return self._attend_logits(x, q, parts, docs, scored, taps)
        # The indexer's loss reads them too, so that a policy keeps them for both.
        q, parts = remat.label("attention.in", (q, tuple(parts)))
        out, lse = remat.named("attention", self._sparse, q, parts, self._sinks())
        if self._trains(scored, taps):
            assert scored is not None
            last = parts[-1]
            xs = (q, lse, *sparse.rows(last))
            width = q.shape[2] * last.k.shape[1]
            self._train_indexer(scored, partial(self._target, last), xs, width, taps)
        return out

    def _sparse(
        self,
        q: Float[Array, "b t h dk"],
        parts: tuple[sparse.Part, ...],
        sinks: Float[Array, " h"] | None,
    ) -> tuple[Float[Array, "b t h dv"], Float[Array, "b t h"]]:
        """Return ``sparse.attend``'s output and logsumexp, by the layer's scale and
        softcap."""
        return sparse.attend(q, parts, sinks, self.scale, self.softcap)

    def _target(self, part: sparse.Part, xs: tuple) -> Float[Array, "b c s"]:
        """Return a chunk of queries' attention to the keys of `part`, given their
        queries, logsumexp and `sparse.rows`."""
        q, lse, span, index = jax.lax.stop_gradient(xs)
        part = sparse.at(part, span, index)
        return sparse.probabilities(q, part, lse, self.scale, self.softcap)

    def _attend_logits(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        parts: Sequence[sparse.Part],
        docs: Segments | None,
        scored: Scored | None,
        taps: Taps,
    ) -> Float[Array, "b t h dv"]:
        """Return `_attend_kept` from all t by t logits, for `FullLogits`' features."""
        feature = self._feature()
        assert feature is not None
        require_xla(feature)
        k, v = (jnp.concatenate(kv, axis=1) for kv in zip(*((p.k, p.v) for p in parts)))
        mask = jnp.concatenate([sparse.visible(p) for p in parts], -1)
        probs = self._probabilities(x, q, k, docs, mask)
        if self._trains(scored, taps):
            assert scored is not None
            n = parts[-1].k.shape[1]
            attention = probs[..., -n:].astype(jnp.float32).sum(1)
            self._train_indexer(scored, lambda a: a, attention, n, taps)
        return self._values(probs, v)

    def _trains(self, scored: Scored | None, taps: Taps) -> bool:
        """Return whether the indexer learns in this forward: in training, if it ran."""
        trained = self.indexer is not None and self.indexer.loss_weight
        return scored is not None and bool(trained) and training(taps)

    def _train_indexer(
        self, scored: Scored, target: Target, xs: Any, width: int, taps: Taps
    ) -> None:
        """Report the indexer's loss against the attention `target` gives."""
        assert self.indexer is not None
        loss = self.indexer.loss(scored, target, xs, width)
        report(taps, "indexer_loss", self.indexer.loss_weight, loss)

    def _select(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        pos: Int[Array, "b t"] | None,
        keys: Float[Array, "b s i"] | None,
        span: sparse.Span,
        order: Int[Array, "b s"],
        taps: Taps,
    ) -> tuple[sparse.Kept | None, Scored | None]:
        """Return the keys each query keeps of those its `span` has (the index
        port's, or the indexer's top-k from `query` and `keys`; None for all), and
        what the indexer scored if it ran. `order` says which key of its document each
        slot holds, -1 for none."""
        scored = None
        if reads(self.ports, "index"):
            kept = read(taps, self.ports, "index")
        elif self.indexer is not None:
            if keys is None:
                raise ValueError(
                    "the indexer has no keys: the kv port's layer has none"
                )
            candidates = None
            if reads(self.ports, "candidates"):
                candidates = read(taps, self.ports, "candidates")
            export = writes(self.ports, "candidates")
            kept, candidates, scored = self.indexer(
                x, query, pos, keys, span, order, candidates, export
            )
            write(taps, self.ports, "candidates", candidates)
        else:
            return None, None
        write(taps, self.ports, "index", kept)
        return kept, scored

    def _select_tokens(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        rows: Segments,
        span: sparse.Span,
        taps: Taps,
    ) -> tuple[sparse.Kept | None, Scored | None]:
        """Return `_select` for keys that are the tokens, the indexer's keys projected
        from them."""
        if self.indexer is None and not reads(self.ports, "index"):
            return None, None
        keys = None if self.indexer is None else self.indexer.keys(x, rows.positions)
        return self._select(x, query, rows.positions, keys, span, rows.within, taps)

    def _attend_tokens(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        heads: tuple[
            Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
        ],
        docs: Segments | None,
        taps: Taps,
    ) -> Float[Array, "b t h dv"]:
        """Return `_attend_kept` over the tokens the queries keep of their window and
        document. For MFU, the indexer ranks every key a query may attend to, and the
        attention and the indexer's loss see the ones it keeps."""
        q, k, v = heads
        rows = documents_or_rows(docs, *x.shape[:2])
        span = sparse.token_span(rows, self.window)
        with flops.attention(self.window):
            kept, scored = self._select_tokens(x, query, rows, span, taps)
        part = sparse.Part(k, v, span, None, kept)
        with flops.attention(self.window, self._picks(kept)):
            return self._attend_kept(x, q, [part], docs, scored, taps)

    def _picks(self, kept: sparse.Kept | None) -> int | None:
        """Return how many keys a query keeps at most: the indexer's top_k, or as many
        as the index port's; None for every one."""
        if kept is None:
            return None
        return kept.index.shape[-1] if self.indexer is None else self.indexer.top_k


class FullLogitsAttention(FullLogits, Attention):
    pass


class SparseAttention(Sparse, Attention):
    """GQA with an indexer whose keys are projected from the layer's input."""

    def __init__(
        self,
        config: cfg.GQAConfig,
        width: int,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        indexer = indexer_config(config, ports)
        super().__init__(config, width, ports, rngs)
        rope = config.rope
        rotated = rotated_dims(self.head_dim, rope)
        self.indexer = (
            from_stream(indexer, width, width, rope, rotated, rngs) if indexer else None
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        docs = segments(taps)
        x, projected = self._projections(x)
        heads = self._heads(projected, docs, taps)
        return self._output(x, self._attend_tokens(x, x, heads, docs, taps))


def pick[L: SoftmaxAttention](
    config: SoftmaxConfig,
    ports: Sequence[ir.Port],
    plain: type[L],
    full: type[L],
    sparse: type[L],
) -> type[L]:
    """Return `sparse` for an indexer or its ports, `full` for another feature that
    needs all t by t logits, else `plain`."""
    sparse_ports = any(p.name in ("index", "candidates") for p in ports)
    if config.indexer is not None or sparse_ports:
        return sparse
    if config.relative_bias or config.log_scale is not None or config.attention_dropout:
        return full
    return plain


def build_gqa(
    config: cfg.GQAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
) -> Attention:
    """Make GQA with the features that `config` and `ports` ask for."""
    layer = pick(config, ports, Attention, FullLogitsAttention, SparseAttention)
    return layer(config, width, ports, rngs)


@register("gqa")
def _(
    c: cfg.GQAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Attention:
    return build_gqa(c, width, ports, ctx.rngs)
