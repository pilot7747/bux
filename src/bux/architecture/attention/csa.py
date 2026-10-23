"""Compressed sparse attention (DeepSeek-V4)."""

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import parallel
from bux.architecture.attention import sparse
from bux.architecture.attention.gqa import (
    Names,
    Sparse,
    documents_or_rows,
    indexer_config,
)
from bux.architecture.attention.indexer import Indexer
from bux.architecture.attention.rope import frequencies, rotate, rotated_dims
from bux.architecture.common.ports import read, reads, write
from bux.architecture.common.segments import Segments
from bux.architecture.common.sharding import linear, shard_heads, shard_layer_input
from bux.architecture.layers.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.lang import ir


class Latents(NamedTuple):
    """Compressed latents, one slot per token: the latent of the group of
    compress_ratio tokens that ends at a token, where one does. The kv port carries
    them."""

    kv: Float[Array, "b s dk"]  # rotated: the key and the value at once
    index: Float[Array, "b s i"] | None  # the indexer's keys, if the layer has one
    order: Int[Array, "b s"]  # which latent of its document a slot holds; -1 for none


class CSA(Sparse):
    """Compressed sparse attention, as DeepSeek-V4.1-Flash's inference code. A latent is
    a head_dim wide RMS-normalized projection of a token, the key and the value of a
    single KV head that every query head shares. Each query attends to the latents of
    a sliding window of tokens, and to compressed latents: every compress_ratio tokens
    of a document pooled into one, per channel under a softmax gate learned from the
    tokens, and RMS-normalized. A compressed latent is rotated to the position of its
    group's first token, and a query may attend to it from its group's last token on.
    An indexer picks the top_k of them for each query; without one a query attends to
    them all.

    Queries come from a q_lora_rank wide RMS-normalized latent, or from the input at
    full rank. The first `rope.fraction` of each query's and latent's dims are rotated,
    and the head outputs are rotated back by their query's position, since the values
    were rotated with the keys. o_lora_rank makes the output projection grouped and
    low-rank: o_groups groups of heads each projected to o_lora_rank, then mixed to the
    width.

    Ports: `kv` written pushes the `Latents`, with the indexer's keys; read uses them
    instead of compressing its own, and an indexer that it runs scores those keys.
    `index` and `candidates` as for `Sparse`, over the compressed latents.

    The window and the compressed latents are two ``sparse.Part``s of one softmax.

    The query heads split over `heads`, and the groups of the output projection with
    them; the latents, a single head, are every device's.
    """

    def __init__(
        self,
        config: cfg.CSAConfig,
        width: int,
        ports: Sequence[ir.Port],
        rngs: nnx.Rngs,
    ):
        h, d, r = config.num_heads, config.head_dim, config.compress_ratio
        if r is None and (config.indexer is not None or ports):
            raise ValueError(
                "csa: an indexer, and the kv, index and candidates ports, are for "
                "compressed latents; set compress_ratio"
            )
        rank, groups = config.o_lora_rank, config.o_groups
        if rank and h % groups:
            raise ValueError(f"csa: {h} heads do not split into {groups} o_groups")
        indexer = indexer_config(config, ports)
        compresses = r is not None and not reads(ports, "kv")
        self.head_dim, self.compress_ratio = d, r

        def proj(
            n_in: int, n_out: int, role: str, names: Names = (None, None)
        ) -> nnx.Linear:
            return linear(n_in, n_out, names, False, rngs, role=role)

        # The randomly initialized layers; reordering them changes a seed's init.
        latent = config.q_lora_rank
        query = "latent" if latent else "embed"  # what the queries are projected from
        self.q_a = proj(width, latent, "q", ("embed", "latent")) if latent else None
        self.q = proj(latent or width, h * d, "q", (query, ("heads", h)))
        self.kv = proj(width, d, "kv")
        self.compress_kv = proj(width, d, "kv") if compresses else None
        pooled = compresses and r is not None and r > 1
        self.compress_gate = proj(width, d, "kv") if pooled else None
        init = nnx.initializers.lecun_normal(batch_axis=(0,))
        self.o_a = (
            nnx.Param(
                init(rngs.params(), (groups, h * d // groups, rank)),
                names=("heads", None, None),
            )
            if rank
            else None
        )
        # What o takes: the groups' ranks, or the heads' dims, flattened
        heads = ("heads", groups) if rank else ("heads", h)
        self.o = proj(groups * rank if rank else h * d, width, "o", (heads, "embed"))
        super().__init__(config, width, h, 1, d, d, False, ports, rngs)
        rotated = rotated_dims(d, config.rope)
        self.indexer = (
            Indexer(
                indexer,
                width,
                latent or width,
                d if compresses else None,
                config.latent_norm,
                config.rope,
                rotated,
                rngs,
                query,
            )
            if indexer
            else None
        )

        # Queries, latents and compression
        norm = config.latent_norm
        self.q_a_norm = Norm(norm, latent) if latent else None
        self.q_norm = Norm(config.qk_norm, d) if config.qk_norm else None
        self.kv_norm = Norm(norm, d)
        self.compress_norm = Norm(norm, d) if compresses else None
        self.rope = config.rope

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        x = shard_layer_input(x)
        b, t, _ = x.shape
        docs = segments(taps)
        rows = documents_or_rows(docs, b, t)
        query = x
        if self.q_a is not None and self.q_a_norm is not None:
            query = self.q_a_norm(self.q_a(x))
        q = self.q(query).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = shard_heads(self._rotate(q, rows.positions))
        kv = self._rotate(self.kv_norm(self.kv(x))[:, :, None], rows.positions)
        if self.compress_ratio is None:
            out = self._attend(x, q, kv, kv, docs)
        else:
            if reads(self.ports, "kv"):
                latents = read(taps, self.ports, "kv")
            else:
                latents = self._compress(x, rows)
            write(taps, self.ports, "kv", latents)
            window = sparse.Part(kv, kv, sparse.token_span(rows, self.window))
            span = sparse.token_span(rows, None)
            kept, scored = self._select(
                x, query, rows.positions, latents.index, span, latents.order, taps
            )
            latent = latents.kv[:, :, None]
            compressed = sparse.Part(latent, latent, span, latents.order >= 0, kept)
            out = self._attend_kept(x, q, [window, compressed], docs, scored, taps)
        return self._output(x, self._unrotate(out, rows.positions))

    def _compress(self, x: Float[Array, "b t d"], rows: Segments) -> Latents:
        """Return this layer's compressed latents, with its indexer's keys if it runs
        one."""
        r = self.compress_ratio
        assert r is not None and self.compress_kv is not None
        assert self.compress_norm is not None
        t = x.shape[1]
        kv = self.compress_kv(x)
        if self.compress_gate is not None:  # pool each slot's r tokens, up to its own
            pad = ((0, 0), (r - 1, 0), (0, 0))
            gate = jnp.pad(self.compress_gate(x), pad, constant_values=-jnp.inf)
            kv = jnp.pad(kv, pad)
            kv = jnp.stack([kv[:, i : i + t] for i in range(r)], axis=2)
            gate = jnp.stack([gate[:, i : i + t] for i in range(r)], axis=2)
            kv = (kv * jax.nn.softmax(gate, axis=2)).sum(2)
        # Split as positions, not as the norm's scale: FSDP may split that by `context`.
        latent = parallel.shard(self.compress_norm(kv), "batch", "seq", None)
        first = rows.positions[:, jnp.maximum(jnp.arange(t) - (r - 1), 0)]
        order = jnp.where((rows.within + 1) % r == 0, rows.within // r, -1)
        index = None
        if self.indexer is not None:
            index = self.indexer.keys(latent, first)
        return Latents(self._rotate(latent[:, :, None], first)[:, :, 0], index, order)

    def _offsets(self, t: int) -> Int[Array, "t s"]:
        """Return how far each latent's token is behind each query; a compressed
        latent's token is the last of its group."""
        offset = super()._offsets(t)
        if self.compress_ratio is None:
            return offset
        return jnp.concatenate([offset, offset], -1)

    def _rotate(
        self, x: Float[Array, "b t h dk"], pos: Int[Array, "b t"]
    ) -> Float[Array, "b t h dk"]:
        """Rotate the first rope.fraction dims of queries or latents by `pos`."""
        return x if self.rope is None else rotate(x, self.rope, pos)

    def _unrotate(
        self, out: Float[Array, "b t h dk"], pos: Int[Array, "b t"]
    ) -> Float[Array, "b t h dk"]:
        """Rotate the head outputs back by their query's position, the inverse of
        `_rotate`, so that what a latent adds depends on how far behind it is."""
        if self.rope is None:
            return out
        dims = rotated_dims(out.shape[-1], self.rope)
        back = rotate(out, self.rope, -pos)  # by -pos, but times the gain again
        if dims == 0:
            return back
        gain = frequencies(self.rope, dims)[1]
        return jnp.concatenate([back[..., :dims] / gain**2, back[..., dims:]], -1)

    def _project(self, out: Float[Array, "b t h dv"]) -> Float[Array, "b t d"]:
        """Project the heads back to the width, through o_lora_rank for each group of
        heads when the projection is grouped."""
        b, t, _, _ = out.shape
        if self.o_a is None:
            return self.o(out.reshape(b, t, -1))
        groups = self.o_a.shape[0]
        # Split by group: groups that do not split stay whole, rather than each split
        # over part of an axis, which XLA can only reach by replicating them.
        grouped = shard_heads(out.reshape(b, t, groups, -1))
        low = jnp.einsum("btgi,gir->btgr", grouped, self.o_a[...])
        return self.o(low.reshape(b, t, -1))


@register("csa")
def _(c: cfg.CSAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> CSA:
    return CSA(c, width, ports, ctx.rngs)
