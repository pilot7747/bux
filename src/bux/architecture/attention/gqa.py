"""Causal grouped-query attention, and what it shares with MLA and CSA.

`Attention` is the GPT-2 and Llama layer: projections, QK-norm, RoPE, the run's
``methods.attention`` (`bux.parallel.context`), and the output projection. Its base,
`SoftmaxAttention`, is what every softmax attention layer shares once it has q, k and
v. Two mixins add to them:

- `FullLogits`: the features that need all t by t logits, which only ``xla``'s math
  makes: relative bias, length scale and attention dropout.
- `Sparse`: sparse attention's indexer, and the attention over the keys it keeps
  (`bux.architecture.attention.sparse`).

`build_gqa` picks the class the config needs: `Attention`, `FullLogitsAttention` or
`SparseAttention`. MLA (`bux.architecture.attention.mla`) and CSA
(`bux.architecture.attention.csa`) build on the same pieces.
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
from bux import initializers, parallel
from bux.architecture import observe
from bux.architecture.attention import sparse
from bux.architecture.attention.indexer import Indexer, Scored, Target, from_stream
from bux.architecture.attention.norm_rope import norm_and_rotate
from bux.architecture.attention.rope import logit_gain, rotated_dims
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
    causal_mask_at,
    masked_softmax,
    scaled_logits,
    softcap,
    weighted_values,
)
from bux.parallel.flash import per_query_head
from bux.train import flops

Names = tuple[parallel.Name, parallel.Name]  # a projection's input and output
SoftmaxConfig = cfg.GQAConfig | cfg.MLAConfig | cfg.CSAConfig
SAMPLED = 16  # the queries of a row whose logits a step that observes looks at


class SoftmaxAttention(nnx.Module):
    """What the softmax attention layers share once they have q, k and v.

    It holds the attention method's settings (the scale, softcap, sliding window and
    sinks) and the output's norm, gate (with HyGA's, `_gate`), projection and dropout.
    A layer makes its own
    projections, `o` among them, before it calls this constructor, so that a seed
    draws them first. Its forward then makes q, k and v and calls `_attend` and
    `_output`.

    Under tensor parallelism a device has whole heads: their columns of the input
    projections, their q, k, v and outputs, and their rows of `o`.

    Args:
        config: the layer's settings (`~bux.config.GQAConfig`,
            `~bux.config.MLAConfig` or `~bux.config.CSAConfig`).
        width: the width d of the stream the layer reads and writes.
        heads: the query heads h.
        kv_heads: the key and value heads g; each serves h / g query heads.
        qk_head_dim: the head dim of queries and keys, dk. The softmax scale is
            ``dk ** -0.5`` unless ``config.scale`` sets one.
        v_head_dim: the head dim of values, dv.
        bias: whether the projections made here, the gate's and the relative
            bias's, have a bias.
        ports: the ports the layer lists in the source.
        rngs: draws the initial parameters.
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
        gate_dims = config.gate_input_rank or v_head_dim
        gate_width = heads if config.gate_per_head else heads * gate_dims
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
        gate_bias = config.gated and config.gate_head_bias
        self.gate_bias = (
            nnx.Param(jnp.zeros(heads), names=("heads",)) if gate_bias else None
        )
        self.out_norm = (
            Norm(config.output_norm, v_head_dim) if config.output_norm else None
        )
        self.gate_activation = config.gate_activation
        self._init_hybrid_gate(config, v_head_dim, bias, rngs)

    def _init_hybrid_gate(
        self, config: SoftmaxConfig, dv: int, bias: bool, rngs: nnx.Rngs
    ) -> None:
        """Make HyGA's parts of the gate (`_gate`): each head's up projection of a
        low-rank input gate, its down and up projections of its own output, and the
        gate of each head from all of them."""
        h, init = self.heads, initializers.weight(batch_axis=(0,))

        def per_head(n_in: int, n_out: int) -> nnx.Param:
            shape = (h, n_in, n_out)
            return nnx.Param(init(rngs.params(), shape), names=("heads", None, None))

        rank, own = config.gate_input_rank, config.gate_output_rank
        self.gate_up = per_head(rank, dv) if rank else None
        self.output_gate_down = per_head(dv, own) if own else None
        self.output_gate_up = per_head(own, dv) if own else None
        names = (("heads", h), None)
        cross = config.gate_cross_head
        self.cross_gate = linear(h * dv, h, names, bias, rngs) if cross else None

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

    def _observe(
        self,
        q: Float[Array, "b t h dk"],
        k: Float[Array, "b s g dk"],
        docs: Segments | None,
        taps: Taps,
    ) -> None:
        """Observe the logits of `SAMPLED` queries a row, the last of each stretch
        of t / SAMPLED, on the steps a run observes (`bux.architecture.observe`):

        - ``attn/logit_max``: the largest logit of a key a query sees, softcapped;
        - ``attn/entropy``: the mean entropy of a query's softmax over that of a
          uniform one over the keys it sees: 1 uniform, 0 on one key (entropy
          collapse).

        They leave out a relative bias and a length scale, and are not taken under
        context parallelism, whose keys the sample would gather.
        """
        on = parallel.active()
        if not observe.observing(taps) or (on is not None and on.size("context") > 1):
            return
        t, s = q.shape[1], k.shape[1]
        n = min(SAMPLED, t)
        at = jnp.arange(n) * (t // n) + t // n - 1
        ids = None if docs is None else (docs.ids[:, at], docs.ids)
        mask = causal_mask_at(self.window, at, jnp.arange(s), ids)
        mask = mask if mask.ndim == 2 else mask[:, None]  # b 1 n s, as the logits
        logits = softcap(scaled_logits(q[:, at], k, self.scale), self.softcap)
        probs = masked_softmax(logits, mask, self._sinks())
        seen = jnp.broadcast_to(mask, logits.shape).sum(-1)
        entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-30)), -1)
        many = seen > 1
        uniform = jnp.log(jnp.maximum(seen, 2).astype(jnp.float32))
        share = jnp.where(many, entropy / uniform, 0).sum() / jnp.maximum(many.sum(), 1)
        observe.stat(taps, "attn/logit_max", jnp.where(mask, logits, -jnp.inf).max())
        observe.stat(taps, "attn/entropy", share)

    def _output(
        self, x: Float[Array, "b t d"], out: Float[Array, "b t h dv"]
    ) -> Float[Array, "b t d"]:
        """Normalize and gate the head outputs (`_gate`), then project them back to
        the width."""
        out = checkpoint_name(out, "attention.out")
        if self.out_norm is not None:
            out = self.out_norm(out)
        if self.gate is not None:
            out = out * self._gate(x, out)
        y = self._project(shard_heads(out))
        return y if self.output_dropout is None else self.output_dropout(y)

    def _gate(
        self, x: Float[Array, "b t d"], out: Float[Array, "b t h dv"]
    ) -> Float[Array, "b t h _"]:
        """Return the gate on the heads' outputs `out`: ``act(x W + b_h)``, `x` the
        layer's input and b_h a bias per head, if any.

        HyGA (Zhou et al., 2026, arXiv 2608.11805) adds to it, each part a setting:
        the input's projection low-rank, ``SiLU(x W_d) W_u`` (its X-gate); the head's
        own output in the logits, ``+ SiLU(o W_d) W_u`` (its H-gate, added before the
        activation, the paper's fused form); and a gate of each head from all heads'
        outputs, ``sigmoid(o W_c)``, which multiplies the first (its C-gate). Each
        head has its own W_u, and its own W_d of its output. The paper's gate reads
        the heads' outputs as attention made them; here they are what the gate
        multiplies, after ``output_norm``. It sets no init: its matrices start as the
        model's others.
        """
        assert self.gate is not None
        b, t, _ = x.shape
        logits = self.gate(x).reshape(b, t, self.heads, -1)
        if self.gate_up is not None:
            logits = _by_head(jax.nn.silu(logits), self.gate_up)
        if self.output_gate_down is not None and self.output_gate_up is not None:
            low = jax.nn.silu(_by_head(out, self.output_gate_down))
            logits = logits + _by_head(low, self.output_gate_up)
        if self.gate_bias is not None:
            logits = logits + self.gate_bias[...].astype(logits.dtype)[:, None]
        gate = activation(self.gate_activation, logits)
        if self.cross_gate is None:
            return gate
        heads = jax.nn.sigmoid(self.cross_gate(out.reshape(b, t, -1)))
        return gate * heads[..., None]

    def _project(self, out: Float[Array, "b t h dv"]) -> Float[Array, "b t d"]:
        """Project the heads back to the width."""
        b, t, _, _ = out.shape
        return project_output(out.reshape(b, t, -1), self.o)


class Attention(SoftmaxAttention):
    """Causal grouped-query attention (GQA), the layer ``gqa``: GPT-2's and Llama's.

    The query heads share the key and value heads in groups: MHA when
    ``num_kv_heads == num_heads``, MQA when it is 1. The layer projects the input to
    q, k and v (one matrix each, or one fused under ``fused_qkv``), adds a causal conv
    to k and v under ``kv_conv_size`` (Inkling), applies QK-norm (of each head, or of
    all heads together under ``qk_norm_all_heads``: MiniMax-M2) and then RoPE, scales
    the queries by their position under ``power_scale`` (Beam), and attends by the
    run's ``methods.attention`` (`bux.parallel.context`). Under ``exclusive`` (XSA)
    each head's output loses its part along the query's own value (`exclusive`).
    The heads' outputs then go through `SoftmaxAttention`'s norm, gate, output
    projection and dropout.
    `build_gqa` makes this class, or `FullLogitsAttention` or `SparseAttention` where
    the config needs them.

    Ports, where the source lists them:

    - ``v``: written, it pushes the values; read, it mixes the tap's values into the
      layer's as ``value_residual`` says (ResFormer).
    - ``kv``: written, it pushes (k, v); read, the layer uses them instead of
      projecting its own (cross-layer KV sharing: YOCO, CLA).

    Shapes: b rows, t queries and s keys (the same positions), d the width, h heads and
    g kv heads, dk and dv the head dims of keys and values.

    Example:
        Llama 3 8B's attention, 32 query heads over 8 kv heads:

        .. code-block:: bux

            attn = gqa(num_heads=32, num_kv_heads=8, rope=rope(base=500000))

        The layer alone, in Python::

            import jax.numpy as jnp
            from flax import nnx

            from bux import config as cfg
            from bux.architecture.attention.gqa import Attention
            from bux.architecture.registry import Taps

            config = cfg.GQAConfig(num_heads=4, num_kv_heads=2, rope=cfg.RoPEConfig())
            attn = Attention(config, width=64, ports=(), rngs=nnx.Rngs(0))
            y = attn(jnp.ones((2, 16, 64)), Taps())  # (2, 16, 64)

    Args:
        config: the layer's settings.
        width: the width d of the stream it reads and writes.
        ports: the ports the source lists for it, as ``gqa(kv=@kv)``.
        rngs: draws the initial parameters.
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
        self.k_eq_v = config.k_eq_v  # values from the key projection
        own_v = g * dv if not config.k_eq_v else 0
        # Fused, the split points of q, k and v do not fall on shard boundaries.
        self.qkv = (
            proj(width, h * dk + g * dk + own_v, ("embed", None), "q")
            if fused
            else None
        )
        self.q = None if fused else proj(width, h * dk, ("embed", heads), "q")
        self.k = proj(width, g * dk, ("embed", kv_heads), "kv") if split else None
        self.v = (
            proj(width, own_v, ("embed", kv_heads), "kv") if split and own_v else None
        )
        self.o = proj(h * dv, width, (heads, "embed"), "o", config.output_bias)
        size = config.kv_conv_size if projects_kv else None
        self.k_conv = conv(g * dk, size) if size else None
        self.v_conv = conv(own_v, size) if size and own_v else None
        super().__init__(config, width, h, g, dk, dv, config.bias, ports, rngs)

        qk_norm, whole = config.qk_norm, config.qk_norm_all_heads
        self.qk_norm_all_heads = whole
        self.q_norm = Norm(qk_norm, h * dk if whole else dk) if qk_norm else None
        self.k_norm = (
            Norm(qk_norm, g * dk if whole else dk) if qk_norm and projects_kv else None
        )
        self.v_norm = _value_norm(config) if config.k_eq_v and projects_kv else None
        self.rope = config.rope
        self.power_scale = config.power_scale
        self.power_scale_length = config.power_scale_length
        self.value_residual = config.value_residual
        shape = (2,) if config.value_residual == "learned" else ()
        trained = reads(ports, "v") and config.value_residual != "fixed"
        self.v_mix = nnx.Param(jnp.full(shape, 0.5, jnp.float32)) if trained else None
        self.ve_gate = _value_embed_gate(config, ports, projects_kv, rngs)
        self._exclusion(config, projects_kv)

    def _exclusion(self, config: cfg.GQAConfig, projects_kv: bool) -> None:
        """Keep XSA's setting, and its strength for each head where it learns them.

        Raises:
            ValueError: XSA in a layer that reads its kv port, whose values are
                another layer's.
        """
        self.exclusive = config.exclusive
        if self.exclusive and not projects_kv:
            raise ValueError(
                "gqa: exclusive (XSA) takes the layer's own values, which a layer that "
                "reads the kv port does not make"
            )
        learned = config.exclusive == "learned"
        zeros = jnp.zeros(self.heads)
        self.xsa_alpha = nnx.Param(zeros, names=("heads",)) if learned else None

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Attend each position to those before it in its document, and to itself.

        Args:
            x: the stream: b rows of t positions, d wide.
            taps: the forward pass's taps: the documents packed into the rows, which
                no query looks across, and the stacks the layer's ports use.

        Returns:
            The layer's output, the shape of `x`.
        """
        docs = segments(taps)
        x, projected = self._projections(x)
        q, k, v = self._heads(x, projected, docs, taps)
        self._observe(q, k, docs, taps)
        return self._output(x, self._exclude(self._attend(x, q, k, v, docs), v))

    def _exclude(
        self, out: Float[Array, "b t h dv"], v: Float[Array, "b t g dv"]
    ) -> Float[Array, "b t h dv"]:
        """Return the heads' outputs by XSA (`exclusive`) under ``exclusive``, else
        as they are."""
        if not self.exclusive:
            return out
        strength = None if self.xsa_alpha is None else jnp.tanh(self.xsa_alpha[...])
        return exclusive(out, v, strength)

    def _heads(
        self,
        x: Float[Array, "b t d"],
        projected: list[Float[Array, "b t _"]],
        docs: Segments | None,
        taps: Taps,
    ) -> tuple[
        Float[Array, "b t h dk"], Float[Array, "b s g dk"], Float[Array, "b s g dv"]
    ]:
        """Return the queries, keys and values of the `projected` input `x`, with the
        ve port's rows added to the values and then the v port's values mixed in; the
        keys and values from the kv port if the layer reads it."""
        pos = None if docs is None else docs.positions
        q, *kv = projected
        q = self._queries(q, pos)
        if self.v_norm is not None:
            k, v = self._keys_as_values(kv[0], pos, taps)
        elif kv:
            k, v = self._keys_and_values(*kv, pos, taps)
        else:
            k, v = read(taps, self.ports, "kv")
        if reads(self.ports, "ve"):
            v = self._add_value_embedding(x, v, read(taps, self.ports, "ve"))
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
            return x, jnp.split(qkv, [q] if self.k_eq_v else [q, q + k], axis=-1)
        projections = [p for p in (self.q, self.k, self.v) if p is not None]
        x, projected = project_input(x, *projections)
        return x, [checkpoint_name(p, "qkv") for p in projected]

    def _queries(
        self, q: Float[Array, "b t _"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h dk"]:
        """Split the projected queries into heads with QK-norm and RoPE
        (`_split_normed`), then apply the power scale."""
        t = q.shape[1]
        q = self._split_normed(q, self.heads, self.q_norm, pos, "heads")
        if self.power_scale is not None:
            at = jnp.arange(t) if pos is None else pos
            scale = _power_scale(at, self.power_scale, self.power_scale_length)
            q = (q * scale).astype(q.dtype)
        return shard_heads(q)

    def _keys_and_values(
        self,
        k: Float[Array, "b s _"],
        v: Float[Array, "b s _"],
        pos: Int[Array, "b s"] | None,
        taps: Taps,
    ) -> tuple[Float[Array, "b s g dk"], Float[Array, "b s g dv"]]:
        """Add the causal conv to the projected keys and values, and split them into
        heads, the keys with QK-norm and RoPE (`_split_normed`)."""
        b, s, _ = k.shape
        if self.k_conv is not None and self.v_conv is not None:
            k, v = k + self.k_conv(k, taps), v + self.v_conv(v, taps)
        k = self._split_normed(k, self.kv_heads, self.k_norm, pos, "kv_heads")
        v = v.reshape(b, s, self.kv_heads, self.v_head_dim)
        return shard_heads(k, "kv_heads"), shard_heads(v, "kv_heads")

    def _keys_as_values(
        self, k: Float[Array, "b s _"], pos: Int[Array, "b s"] | None, taps: Taps
    ) -> tuple[Float[Array, "b s g dk"], Float[Array, "b s g dk"]]:
        """Return keys, with QK-norm and RoPE, and values, each head's projected key
        RMS-normalized without a gain, both after the conv (``k_eq_v``, Gemma 4)."""
        assert self.v_norm is not None
        b, s, _ = k.shape
        if self.k_conv is not None:
            k = k + self.k_conv(k, taps)
        v = self.v_norm(k.reshape(b, s, self.kv_heads, self.head_dim))
        k = self._split_normed(k, self.kv_heads, self.k_norm, pos, "kv_heads")
        return shard_heads(k, "kv_heads"), shard_heads(v, "kv_heads")

    def _split_normed(
        self,
        x: Float[Array, "b t _"],
        heads: int,
        norm: Norm | None,
        pos: Int[Array, "b t"] | None,
        name: parallel.Name,
    ) -> Float[Array, "b t n dk"]:
        """Split projected queries or keys into `heads` heads, then apply `norm` and
        RoPE (`norm_and_rotate`). Under ``qk_norm_all_heads`` the norm is of all the
        heads together, before the split: under tensor parallelism its mean square
        sums over the devices' heads, and the fused kernel, which normalizes each
        head, does not take it."""
        b, t, _ = x.shape
        if self.qk_norm_all_heads and norm is not None:
            x, norm = norm(x), None
        x = x.reshape(b, t, heads, self.head_dim)
        return norm_and_rotate(x, norm, self.rope, pos, name)

    def _add_value_embedding(
        self,
        x: Float[Array, "b t d"],
        v: Float[Array, "b s g dv"],
        e: Float[Array, "b s _"],
    ) -> Float[Array, "b s g dv"]:
        """Return the values plus the ve port's rows `e`, by a gate per kv head of the
        first channels of `x` and `e`: v + 2 sigmoid(W [x[:n], e[:n]]) e."""
        b, s, g, dv = v.shape
        if e.shape[-1] != g * dv:
            raise ValueError(
                f"gqa: a value embedding of {e.shape[-1]} features into {g} kv heads "
                f"of {dv}: give value_embed features={g * dv}"
            )
        e = e.astype(v.dtype)
        rows = e.reshape(b, s, g, dv)
        if self.ve_gate is None:
            return v + rows
        n = self.ve_gate.in_features // 2
        gate = self.ve_gate(jnp.concatenate([x[..., :n], e[..., :n]], -1))
        return v + 2 * jax.nn.sigmoid(gate)[..., None].astype(v.dtype) * rows

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


def exclusive(
    out: Float[Array, "b t h dv"],
    v: Float[Array, "b t g dv"],
    strength: Float[Array, " h"] | None = None,
) -> Float[Array, "b t h dv"]:
    """Return each head's output y less its part along its query's own value v:
    exclusive self-attention (XSA; Zhai, 2026, arXiv 2603.09078).

    ``y - s (y . u) u`` with ``u = v / max(norm(v), 1e-12)``, as ``F.normalize`` makes it,
    computed in float32. A query head takes the value of its group's kv head at its
    own position. The paper removes the whole part, s = 1; modded-nanogpt's learned
    XSA scales it by ``s = tanh(a)`` for each head (`strength`).

    Args:
        out: The heads' outputs, of t queries.
        v: The values the heads attended to at the same t positions.
        strength: The share s of the part removed for each head; None for all of it.

    Returns:
        The outputs, in their dtype.
    """
    y = out.astype(jnp.float32)
    u = per_query_head(v, out.shape[2]).astype(jnp.float32)
    u = u * jax.lax.rsqrt(jnp.maximum(jnp.sum(u * u, -1, keepdims=True), 1e-24))
    along = jnp.sum(y * u, -1, keepdims=True) * u
    if strength is not None:
        along = along * strength.astype(jnp.float32)[:, None]
    return (y - along).astype(out.dtype)


def _by_head(x: Float[Array, "b t h i"], w: nnx.Param) -> Float[Array, "b t h o"]:
    """Return each head's `x` times its own matrix of `w`, h by i by o."""
    return jnp.einsum("bthi,hio->btho", x, w[...].astype(x.dtype))


def _value_embed_gate(
    config: cfg.GQAConfig, ports: Sequence[ir.Port], projects_kv: bool, rngs: nnx.Rngs
) -> nnx.Linear | None:
    """Make the gate of the ve port's rows, from 2n channels to a value per kv head
    and starting at 0, for a layer that reads the port; None without the port or the
    gate.

    Raises:
        ValueError: The layer reads the ve port and the kv port: it has no values of
            its own to add the rows to.
    """
    if not reads(ports, "ve"):
        return None
    if not projects_kv:
        raise ValueError(
            "gqa: the ve port adds to the layer's own values, which a layer that "
            "reads kv does not have"
        )
    n = config.value_embed_gate
    if n is None:
        return None
    names = (None, ("kv_heads", config.num_kv_heads))
    zeros = nnx.initializers.zeros
    return linear(2 * n, config.num_kv_heads, names, False, rngs, init=zeros)


def _value_norm(config: cfg.GQAConfig) -> Norm:
    """Return the norm of values made from keys (``k_eq_v``): Gemma 4's RMSNorm
    without a gain, at the eps of the layer's QK-norm, or the norms' default."""
    eps = config.qk_norm.eps if config.qk_norm is not None else None
    norm = cfg.NormalizationConfig(type="rmsnorm", affine=False)
    return Norm(norm if eps is None else norm.model_copy(update={"eps": eps}), 1)


def require_xla(feature: str) -> None:
    """Raise unless the run's ``methods.attention`` is ``xla`` or ``auto``.

    ``xla`` is the only method with the logit features of `FullLogits`, and ``auto``
    takes ``xla`` for them.

    Args:
        feature: the feature's setting, which the error names.

    Raises:
        NotImplementedError: under any other method.
    """
    name = parallel.choice("attention")
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


def _power_scale(
    pos: Int[Array, "*lead t"], power: float, length: int
) -> Float[Array, "*lead t 1 1"]:
    """Return the factor on each query for its position n, counted from 1:
    (1 + n / length) ** power, in float32."""
    return jnp.exp(power * jnp.log1p((pos + 1) / length))[..., None, None]


def _length_scale(
    pos: Int[Array, "*lead t"], log_scale: float, floor: int
) -> Float[Array, "*lead t 1"]:
    """Return the factor on each query's logits for its position n, counted from 1:
    1 + log_scale * ln(max(n / floor, 1))."""
    ahead = jnp.maximum((pos + 1) / floor, 1.0)
    return (1 + log_scale * jnp.log(ahead))[..., None]


class FullLogits(SoftmaxAttention):
    """Softmax attention with features that need all t by t logits.

    The features: a data-dependent relative bias (``relative_bias``, Inkling), a
    length scale on the logits (``log_scale``, Inkling) and attention dropout. With
    any of them the layer runs ``xla``'s masked softmax itself, so it raises under a
    method other than ``xla`` or ``auto`` (`require_xla`); without them it attends as
    `SoftmaxAttention` does. A mixin: `FullLogitsAttention` and
    `~bux.architecture.attention.mla.FullLogitsMLA` put it before their layer.
    """

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
        mask: Bool[Array, "*b #h t s"] | None = None,
    ) -> Float[Array, "b h t s"]:
        """Return the softmax over the keys `mask` keeps of each head, the layer's
        own if none."""
        if mask is None:
            mask = self._mask(docs, x.shape[1])[..., None, :, :]
        logits = self._logits(x, q, k, docs)
        return masked_softmax(logits, mask, self._sinks())

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
        return causal_mask(t, self.window, None if docs is None else docs.ids)


def indexer_config(
    config: SoftmaxConfig, ports: Sequence[ir.Port]
) -> cfg.SparseIndexerConfig | None:
    """Return the config of the indexer the layer runs, if it runs one.

    Args:
        config: the layer's settings.
        ports: the ports the source lists for it.

    Returns:
        ``config.indexer``; None when the layer reads its kept keys from the ``index``
        port instead, whatever its config says.

    Raises:
        ValueError: a ``candidates`` port, or an ``index`` port it writes, without an
            indexer that the layer runs.
        NotImplementedError: candidate pools of an indexer with ``block_size > 1``;
            max-pooled blocks of CSA's compressed latents.
    """
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
    if indexer is None or indexer.block_size == 1:
        return indexer
    if "candidates" in needs:
        raise NotImplementedError("candidate pools of an indexer with block_size > 1")
    if indexer.pool == "max" and config.type == "csa":
        raise NotImplementedError(
            "csa: max-pooled blocks (indexer(pool=max)) of compressed latents"
        )
    return indexer


def _per_head(seen: Bool[Array, "b t *g s"], heads: int) -> Bool[Array, "b _ t s"]:
    """Return which keys each query sees, for all heads or for each of the `heads`,
    each its group's."""
    if seen.ndim == 3:
        return seen[:, None]
    return jnp.repeat(jnp.moveaxis(seen, 2, 1), heads // seen.shape[2], axis=1)


def _summed(
    probs: Float[Array, "b h t s"], groups: int | None
) -> Float[Array, "b t *g s"]:
    """Return the heads' probabilities summed, or summed in each of `groups` groups."""
    if groups is None:
        return probs.sum(1)
    b, h, t, s = probs.shape
    return jnp.moveaxis(probs.reshape(b, groups, h // groups, t, s).sum(2), 1, 2)


def documents_or_rows(docs: Segments | None, b: int, t: int) -> Segments:
    """Return the documents packed into b rows of t, or one document a row if None."""
    if docs is not None:
        return docs
    return Segments.of(jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t)))


class Sparse(FullLogits):
    """Sparse attention (DeepSeek-V3.2's DSA): each query attends only to the keys an
    indexer keeps.

    The indexer (`~bux.architecture.attention.indexer.Indexer`) ranks the keys each
    query may attend to and keeps its top-k. Its loss alone trains it, against the
    attention's probabilities, which it reports as ``indexer_loss`` in training. The
    attention is `bux.architecture.attention.sparse.attend`'s, with no t by t floats:
    tokamax's flash kernel with the kept keys as a mask on a GPU, else XLA a chunk of
    queries at a time. `FullLogits`' features take all t by t logits, as ``xla`` does.
    A mixin: `SparseAttention`, `~bux.architecture.attention.mla.SparseMLA` and
    `~bux.architecture.attention.csa.CSA` build on it.

    Ports, where the source lists them:

    - ``index``: written, it pushes the kept keys
      (`~bux.architecture.attention.sparse.Kept`); read, the layer uses the tap's
      instead of running an indexer, and builds none (GLM-5's IndexCache).
    - ``candidates``: written, it pushes the indexer's candidate pool; read, the
      indexer keeps keys only inside the tap's pool (DeepSeek-V4.1).
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

    def _target(self, part: sparse.Part, xs: tuple) -> Float[Array, "b c *g s"]:
        """Return a chunk of queries' attention to the keys of `part`, given their
        queries, logsumexp and `sparse.rows`: for each group if it keeps keys for
        each."""
        q, lse, span, index = jax.lax.stop_gradient(xs)
        part = sparse.at(part, span, index)
        groups = sparse.per_group(part)
        return sparse.probabilities(q, part, lse, self.scale, self.softcap, groups)

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
        probs = self._probabilities(x, q, k, docs, _per_head(mask, self.heads))
        if self._trains(scored, taps):
            assert scored is not None
            n = parts[-1].k.shape[1]
            groups = mask.shape[2] if mask.ndim == 4 else None
            attention = _summed(probs[..., -n:].astype(jnp.float32), groups)
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
            self._check_groups(kept)
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

    def _check_groups(self, kept: sparse.Kept) -> None:
        """Raise unless an index read for each group has one for each kv head."""
        groups = kept.index.shape[2] if kept.index.ndim == 4 else None
        if groups not in (None, self.kv_heads):
            raise ValueError(
                f"{self.kv_heads} kv heads cannot read an index of picks for each of "
                f"{groups} groups (indexer(per_group=true)): give them as many"
            )

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
    """GQA with the logit features of `FullLogits`: relative bias, length scale or
    attention dropout. It takes `Attention`'s arguments."""


class SparseAttention(Sparse, Attention):
    """GQA with sparse attention's indexer (`Sparse`), whose queries and keys are
    projected from the layer's input, as DeepSeek-V3.2's are. It takes `Attention`'s
    arguments; ``config.indexer`` sets the indexer.

    Example:
        Layers that run the indexer and export its top-k, each followed by three
        that reuse it:

        .. code-block:: bux

            attn  = gqa(num_heads=16, num_kv_heads=4, indexer=indexer(top_k=1024))
            first = [norm attn(index=@topk)] [norm ffn]  # runs the indexer
            later = [norm attn(index=$topk)] [norm ffn]  # attends to its picks
            model = embed (first later^3)^4 norm head
    """

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
        """Attend each position to the keys it keeps, as `Attention` does to all.

        Args:
            x: the stream: b rows of t positions, d wide.
            taps: the forward pass's taps, as `Attention` takes them; a training
                forward collects the indexer's loss in them.

        Returns:
            The layer's output, the shape of `x`.
        """
        docs = segments(taps)
        x, projected = self._projections(x)
        heads = self._heads(x, projected, docs, taps)
        out = self._attend_tokens(x, x, heads, docs, taps)
        return self._output(x, self._exclude(out, heads[2]))


def pick[L: SoftmaxAttention](
    config: SoftmaxConfig,
    ports: Sequence[ir.Port],
    plain: type[L],
    full: type[L],
    sparse: type[L],
) -> type[L]:
    """Return the class of a softmax attention layer that `config` and `ports` need.

    Args:
        config: the layer's settings.
        ports: the ports the source lists for it.
        plain: the layer.
        full: the layer with `FullLogits`.
        sparse: the layer with `Sparse`.

    Returns:
        `sparse` for an indexer or its ports, `full` for another feature that needs
        all t by t logits, else `plain`.
    """
    sparse_ports = any(p.name in ("index", "candidates") for p in ports)
    if config.indexer is not None or sparse_ports:
        return sparse
    if config.relative_bias or config.log_scale is not None or config.attention_dropout:
        return full
    return plain


def build_gqa(
    config: cfg.GQAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
) -> Attention:
    """Make GQA with the features that `config` and `ports` ask for.

    Args:
        config: the layer's settings.
        width: the width d of the stream it reads and writes.
        ports: the ports the source lists for it.
        rngs: draws the initial parameters.

    Returns:
        An `Attention`, `FullLogitsAttention` or `SparseAttention`, as `pick` picks.
    """
    layer = pick(config, ports, Attention, FullLogitsAttention, SparseAttention)
    return layer(config, width, ports, rngs)


@register("gqa")
def _(
    c: cfg.GQAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context
) -> Attention:
    return build_gqa(c, width, ports, ctx.rngs)
