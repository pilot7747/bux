"""Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear)."""

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux.architecture.attention import (
    Names,
    SoftmaxAttention,
    _by_head,
    _input,
    _linear,
    _own_indexer,
)
from bux.architecture.indexer import from_stream
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.rope import logit_gain, rotate
from bux.lang import ir

_LATENT_NORM = cfg.NormalizationConfig(type="rmsnorm")


class MLA(SoftmaxAttention):
    """Multi-head latent attention, as transformers' DeepseekV3Attention: every head's
    keys and values are projected up from one kv_lora_rank wide latent a token, and
    its queries from one q_lora_rank wide latent (or from the input, at full rank),
    each latent RMS-normalized. A head's query and key are qk_nope_head_dim dims of its
    own and qk_rope_head_dim more, which for keys are shared by all heads and projected
    from the input directly; only those are rotated, and with rope=None none are
    (Kimi Linear's NoPE).

    Ports: `index` and `candidates`, as for `SoftmaxAttention`; the indexer's queries
    come from the query latent (or the input) and its keys from the input, as in
    DeepSeek-V3.2.

    Every head has keys and values of its own, so they split over `heads`; the down
    projections make the `latent`s, which the up projections take.
    """

    def __init__(
        self, c: cfg.MLAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
    ):
        h, n, r, dv = c.num_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        self.nope_head_dim, self.rope_head_dim, self.v_head_dim = n, r, dv
        self.kv_lora_rank = c.kv_lora_rank
        indexer = _own_indexer(c, ports)

        def linear(n_in: int, n_out: int, names: Names) -> nnx.Linear:
            return _linear(n_in, n_out, names, False, rngs)

        # The randomly initialized layers; reordering them changes a seed's init.
        latent = c.q_lora_rank
        heads = ("heads", h)  # flattened with their dims
        query = "latent" if latent else "embed"  # what the queries are projected from
        self.q_a = linear(width, latent, ("embed", "latent")) if latent else None
        self.q = linear(latent or width, h * (n + r), (query, heads))
        self.kv_a = linear(width, c.kv_lora_rank + r, ("embed", "latent"))
        self.kv_b = linear(c.kv_lora_rank, h * (n + dv), ("latent", heads))
        self.o = linear(h * dv, width, (heads, "embed"))
        scale = (c.scale or (n + r) ** -0.5) * logit_gain(c.rope)
        super().__init__(c, width, h, h, dv, scale, False, ports, rngs)
        rotated = int(r * c.rope.fraction) // 2 * 2 if c.rope else 0
        self.indexer = (
            from_stream(indexer, width, latent or width, c.rope, rotated, rngs, query)
            if indexer
            else None
        )

        # Queries, keys and values
        self.q_a_norm = Norm(_LATENT_NORM, latent) if latent else None
        self.kv_a_norm = Norm(_LATENT_NORM, c.kv_lora_rank)
        self.q_norm = Norm(c.qk_norm, n + r) if c.qk_norm else None
        self.k_norm = Norm(c.qk_norm, n + r) if c.qk_norm else None
        self.rope = c.rope

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = _input(x)
        b, t, _ = x.shape
        h, n, r = self.heads, self.nope_head_dim, self.rope_head_dim
        docs = segments(taps)
        pos = None if docs is None else docs.positions
        query = x
        if self.q_a is not None and self.q_a_norm is not None:
            query = self.q_a_norm(self.q_a(x))
        q = self.q(query).reshape(b, t, h, n + r)
        latent, k_rope = jnp.split(self.kv_a(x), [self.kv_lora_rank], axis=-1)
        kv = self.kv_b(self.kv_a_norm(latent)).reshape(b, t, h, n + self.v_head_dim)
        k_nope, v = jnp.split(kv, [n], axis=-1)
        k = jnp.concatenate(
            [k_nope, jnp.broadcast_to(k_rope[:, :, None], (b, t, h, r))], -1
        )
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self._rotate(q, pos), self._rotate(k, pos)
        q, k, v = (_by_head(a, "heads") for a in (q, k, v))
        keep, scores = self._select_tokens(x, query, docs, taps)
        if keep is None:
            return self._output(x, self._attend(x, q, k, v, docs))
        return self._output(x, self._attend(x, q, k, v, docs, keep, scores, taps))

    def _rotate(
        self, x: Float[Array, "b t h k"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h k"]:
        """Queries or keys with their last qk_rope_head_dim dims turned by the positions."""
        if self.rope is None:
            return x
        n = self.nope_head_dim
        return jnp.concatenate([x[..., :n], rotate(x[..., n:], self.rope, pos)], -1)


@register("mla")
def _(c: cfg.MLAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLA:
    return MLA(c, width, ports, ctx.rngs)
