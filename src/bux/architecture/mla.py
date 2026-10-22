"""Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear)."""

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux.architecture.attention import (
    FullLogits,
    Names,
    SoftmaxAttention,
    Sparse,
    indexer_config,
    pick,
)
from bux.architecture.indexer import from_stream
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.rope import rotate
from bux.architecture.segments import Segments
from bux.architecture.sharding import linear, shard_heads, shard_layer_input
from bux.lang import ir

_LATENT_NORM = cfg.NormalizationConfig(type="rmsnorm")


class MLA(SoftmaxAttention):
    """Multi-head latent attention, as transformers' DeepseekV3Attention. Every head's
    keys and values are projected up from one kv_lora_rank wide latent per token, and
    its queries from one q_lora_rank wide latent (or from the input, at full rank);
    each latent is RMS-normalized. A head's query and key have qk_nope_head_dim dims
    of their own and qk_rope_head_dim more, which for keys all heads share and are
    projected from the input directly. Only those are rotated, and with rope=None
    none are (Kimi Linear's NoPE).

    Every head has keys and values of its own, so they split over `heads`; the down
    projections make the `latent`s, which the up projections take.
    """

    def __init__(
        self,
        config: cfg.MLAConfig,
        width: int,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        h, n = config.num_heads, config.qk_nope_head_dim
        r, dv = config.qk_rope_head_dim, config.v_head_dim
        self.nope_head_dim, self.rope_head_dim, self.v_head_dim = n, r, dv
        self.kv_lora_rank = config.kv_lora_rank

        def proj(n_in: int, n_out: int, names: Names) -> nnx.Linear:
            return linear(n_in, n_out, names, False, rngs)

        # The randomly initialized layers; reordering them changes a seed's init.
        latent = config.q_lora_rank
        heads = ("heads", h)  # flattened with their dims
        query = "latent" if latent else "embed"  # what the queries are projected from
        self.q_a = proj(width, latent, ("embed", "latent")) if latent else None
        self.q = proj(latent or width, h * (n + r), (query, heads))
        self.kv_a = proj(width, config.kv_lora_rank + r, ("embed", "latent"))
        self.kv_b = proj(config.kv_lora_rank, h * (n + dv), ("latent", heads))
        self.o = proj(h * dv, width, (heads, "embed"))
        super().__init__(config, width, h, h, n + r, dv, False, ports, rngs)

        self.q_a_norm = Norm(_LATENT_NORM, latent) if latent else None
        self.kv_a_norm = Norm(_LATENT_NORM, config.kv_lora_rank)
        self.q_norm = Norm(config.qk_norm, n + r) if config.qk_norm else None
        self.k_norm = Norm(config.qk_norm, n + r) if config.qk_norm else None
        self.rope = config.rope

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = shard_layer_input(x)
        docs = segments(taps)
        q, k, v = self._heads(x, self._query_input(x), docs)
        return self._output(x, self._attend(x, q, k, v, docs))

    def _query_input(self, x: Float[Array, "b t d"]) -> Float[Array, "b t e"]:
        """Return the normalized query latent, or the input at full rank."""
        if self.q_a is None or self.q_a_norm is None:
            return x
        return self.q_a_norm(self.q_a(x))

    def _heads(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        docs: Segments | None,
    ) -> tuple[
        Float[Array, "b t h dk"], Float[Array, "b t h dk"], Float[Array, "b t h dv"]
    ]:
        """Return every head's q, k and v, the keys' RoPE dims shared by all heads."""
        b, t, _ = x.shape
        h, n, r = self.heads, self.nope_head_dim, self.rope_head_dim
        pos = None if docs is None else docs.positions
        q = self.q(query).reshape(b, t, h, n + r)
        kv_latent, k_rope = jnp.split(self.kv_a(x), [self.kv_lora_rank], axis=-1)
        kv = self.kv_b(self.kv_a_norm(kv_latent)).reshape(b, t, h, n + self.v_head_dim)
        k_nope, v = jnp.split(kv, [n], axis=-1)
        k = jnp.concatenate(
            [k_nope, jnp.broadcast_to(k_rope[:, :, None], (b, t, h, r))], -1
        )
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self._rotate(q, pos), self._rotate(k, pos)
        return shard_heads(q), shard_heads(k), shard_heads(v)

    def _rotate(
        self, x: Float[Array, "b t h dk"], pos: Int[Array, "b t"] | None
    ) -> Float[Array, "b t h dk"]:
        """Rotate the last qk_rope_head_dim dims of queries or keys by the positions."""
        if self.rope is None:
            return x
        n = self.nope_head_dim
        return jnp.concatenate([x[..., :n], rotate(x[..., n:], self.rope, pos)], -1)


class FullLogitsMLA(FullLogits, MLA):
    pass


class SparseMLA(Sparse, MLA):
    """MLA with an indexer whose queries come from the query latent (or the input) and
    keys from the input, as in DeepSeek-V3.2."""

    def __init__(
        self,
        config: cfg.MLAConfig,
        width: int,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        indexer = indexer_config(config, ports)
        super().__init__(config, width, ports, rngs)
        latent, rope = config.q_lora_rank, config.rope
        query = "latent" if latent else "embed"
        rotated = int(self.rope_head_dim * rope.fraction) // 2 * 2 if rope else 0
        self.indexer = (
            from_stream(indexer, width, latent or width, rope, rotated, rngs, query)
            if indexer
            else None
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = shard_layer_input(x)
        docs = segments(taps)
        query = self._query_input(x)
        q, k, v = self._heads(x, query, docs)
        keep_mask, scores = self._select_tokens(x, query, docs, taps)
        out = self._attend_kept(x, q, k, v, docs, keep_mask, scores, taps)
        return self._output(x, out)


def build_mla(
    config: cfg.MLAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
) -> MLA:
    """Make MLA with the features that `config` and `ports` ask for."""
    layer = pick(config, ports, MLA, FullLogitsMLA, SparseMLA)
    return layer(config, width, ports, rngs)


@register("mla")
def _(c: cfg.MLAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLA:
    return build_mla(c, width, ports, ctx.rngs)
