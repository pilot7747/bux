"""Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear).

`MLA` is the layer ``mla``. `FullLogitsMLA` and `SparseMLA` add the features of
`~bux.architecture.attention.gqa.FullLogits` and
`~bux.architecture.attention.gqa.Sparse` to it, and `build_mla` picks the class a
config needs.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux.architecture.attention.gqa import (
    FullLogits,
    Names,
    SoftmaxAttention,
    Sparse,
    indexer_config,
    pick,
)
from bux.architecture.attention.indexer import from_stream
from bux.architecture.attention.rope import rotate, rotated_dims
from bux.architecture.common.segments import Segments
from bux.architecture.common.sharding import linear, shard_heads, shard_layer_input
from bux.architecture.layers.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir

_LATENT_NORM = cfg.NormalizationConfig(type="rmsnorm")


class MLA(SoftmaxAttention):
    """Multi-head latent attention (MLA), the layer ``mla``: DeepSeek-V2's and V3's,
    Kimi K2's and Kimi Linear's, as transformers' ``DeepseekV3Attention``.

    Every head's keys and values are projected up from one ``kv_lora_rank`` wide
    latent per token, and its queries from one ``q_lora_rank`` wide latent (or from
    the input, at full rank); each latent is RMS-normalized. A head's query and key
    have ``qk_nope_head_dim`` dims of their own and ``qk_rope_head_dim`` more, which
    for keys all heads share and are projected from the input directly. Only those
    are rotated, and with ``rope=None`` none are (Kimi Linear's NoPE). The softmax
    scale is ``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5`` unless ``scale`` sets
    one. `build_mla` makes this class, or `FullLogitsMLA` or `SparseMLA` where the
    config needs them.

    Every head has keys and values of its own, so they split over ``heads``; the down
    projections make the latents, split over ``latent``, which the up projections
    take.

    Example:
        DeepSeek-V3's attention:

        .. code-block:: bux

            attn = mla(num_heads=128, q_lora_rank=1536, kv_lora_rank=512,
                       rope=rope(layout=interleaved,
                                 scaling=yarn(factor=40, original_max_len=4096,
                                              attention_factor_on=softmax)))

    Args:
        config: the layer's settings.
        width: the width d of the stream it reads and writes.
        ports: the ports the source lists for it.
        rngs: draws the initial parameters.
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

        def proj(n_in: int, n_out: int, names: Names, role: str) -> nnx.Linear:
            return linear(n_in, n_out, names, False, rngs, role=role)

        # The randomly initialized layers; reordering them changes a seed's init.
        latent = config.q_lora_rank
        heads = ("heads", h)  # flattened with their dims
        query = "latent" if latent else "embed"  # what the queries are projected from
        self.q_a = proj(width, latent, ("embed", "latent"), "q") if latent else None
        self.q = proj(latent or width, h * (n + r), (query, heads), "q")
        self.kv_a = proj(width, config.kv_lora_rank + r, ("embed", "latent"), "kv")
        self.kv_b = proj(config.kv_lora_rank, h * (n + dv), ("latent", heads), "kv")
        self.o = proj(h * dv, width, (heads, "embed"), "o")
        super().__init__(config, width, h, h, n + r, dv, False, ports, rngs)

        self.q_a_norm = Norm(_LATENT_NORM, latent) if latent else None
        self.kv_a_norm = Norm(_LATENT_NORM, config.kv_lora_rank)
        self.q_norm = Norm(config.qk_norm, n + r) if config.qk_norm else None
        self.k_norm = Norm(config.qk_norm, n + r) if config.qk_norm else None
        self.rope = config.rope

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Attend each position to those before it in its document, and to itself.

        Args:
            x: the stream: b rows of t positions, d wide.
            taps: the forward pass's taps: the documents packed into the rows, which
                no query looks across.

        Returns:
            The layer's output, the shape of `x`.
        """
        x = shard_layer_input(x)
        docs = segments(taps)
        q, k, v = self._heads(x, self._query_input(x), docs)
        return self._output(x, self._attend(x, q, k, v, docs))

    def _query_input(self, x: Float[Array, "b t d"]) -> Float[Array, "b t e"]:
        """Return the normalized query latent, named `qkv`, or the input at full
        rank."""
        if self.q_a is None or self.q_a_norm is None:
            return x
        return self.q_a_norm(checkpoint_name(self.q_a(x), "qkv"))

    def _heads(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        docs: Segments | None,
    ) -> tuple[
        Float[Array, "b t h dk"], Float[Array, "b t h dk"], Float[Array, "b t h dv"]
    ]:
        """Return every head's q, k and v, the keys' RoPE dims shared by all heads.
        The kv latent and the shared key dims are named `qkv`, as the query latent is:
        a policy that keeps them computes the heads' projections up from them again,
        as DeepSeek's and Megatron's MLA do."""
        b, t, _ = x.shape
        h, n, r = self.heads, self.nope_head_dim, self.rope_head_dim
        pos = None if docs is None else docs.positions
        q = self.q(query).reshape(b, t, h, n + r)
        latent = checkpoint_name(self.kv_a(x), "qkv")
        kv_latent, k_rope = jnp.split(latent, [self.kv_lora_rank], axis=-1)
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
    """MLA with the logit features of `~bux.architecture.attention.gqa.FullLogits`:
    relative bias, length scale or attention dropout. It takes `MLA`'s arguments."""


class SparseMLA(Sparse, MLA):
    """MLA with sparse attention's indexer (`~bux.architecture.attention.gqa.Sparse`),
    as DeepSeek-V3.2's: its queries come from the query latent (or the input, at full
    rank) and its keys from the input. It takes `MLA`'s arguments; ``config.indexer``
    sets the indexer."""

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
        rotated = rotated_dims(self.rope_head_dim, rope)
        self.indexer = (
            from_stream(indexer, width, latent or width, rope, rotated, rngs, query)
            if indexer
            else None
        )

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        """Attend each position to the keys it keeps, as `MLA` does to all.

        Args:
            x: the stream: b rows of t positions, d wide.
            taps: the forward pass's taps, as `MLA` takes them; a training forward
                collects the indexer's loss in them.

        Returns:
            The layer's output, the shape of `x`.
        """
        x = shard_layer_input(x)
        docs = segments(taps)
        query = self._query_input(x)
        heads = self._heads(x, query, docs)
        return self._output(x, self._attend_tokens(x, query, heads, docs, taps))


def build_mla(
    config: cfg.MLAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
) -> MLA:
    """Make MLA with the features that `config` and `ports` ask for.

    Args:
        config: the layer's settings.
        width: the width d of the stream it reads and writes.
        ports: the ports the source lists for it.
        rngs: draws the initial parameters.

    Returns:
        An `MLA`, `FullLogitsMLA` or `SparseMLA`, as
        `~bux.architecture.attention.gqa.pick` picks.
    """
    layer = pick(config, ports, MLA, FullLogitsMLA, SparseMLA)
    return layer(config, width, ports, rngs)


@register("mla")
def _(c: cfg.MLAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> MLA:
    return build_mla(c, width, ports, ctx.rngs)
