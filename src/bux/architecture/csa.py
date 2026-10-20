"""Compressed sparse attention (DeepSeek-V4)."""

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int

from bux import config as cfg
from bux.architecture.attention import (
    SoftmaxAttention,
    _causal,
    _own_indexer,
    _read,
    _reads,
    _rows,
    _write,
)
from bux.architecture.indexer import Indexer, Scores
from bux.architecture.norm import Norm
from bux.architecture.registry import Context, Taps, register, segments
from bux.architecture.rope import frequencies, logit_gain, rotate
from bux.architecture.segments import Segments
from bux.lang import ir


class Latents(NamedTuple):
    """Compressed latents, in a slot a token: the latent of the group of compress_ratio
    tokens that ends at a token, where one does. What the kv port carries."""

    kv: Float[Array, "b s k"]  # rotated: the key and the value at once
    index: Float[Array, "b s i"] | None  # the indexer's keys, if the layer has one
    order: Int[Array, "b s"]  # which latent of its document a slot holds; -1 for none


class CSA(SoftmaxAttention):
    """Compressed sparse attention, as DeepSeek-V4.1-Flash's inference code. A latent is
    a head_dim wide RMS-normalized projection of a token, the key and the value of a
    single KV head that every query head shares. Each query attends to the latents of
    a sliding window of tokens, and to compressed latents: every compress_ratio tokens
    of a document pooled into one, per channel under a softmax gate learned from the
    tokens, and RMS-normalized. A compressed latent is rotated to the position of its
    group's first token, and a query sees it from its group's last token on. An indexer
    picks the top_k of them for each query; without one a query sees them all.

    Queries come from a q_lora_rank wide RMS-normalized latent, or from the input at
    full rank. The first `rope.fraction` of each query's and latent's dims are rotated,
    and the head outputs are turned back by their query's position, since the values
    were rotated with the keys. o_lora_rank makes the output projection grouped and low-rank:
    o_groups groups of heads each projected to o_lora_rank, then mixed to the width.

    Ports: `kv` written pushes the `Latents`, with the indexer's keys; read uses them
    instead of compressing its own, and an indexer that it runs scores those keys.
    `index` and `candidates` as for `SoftmaxAttention`, over the compressed latents.
    """

    def __init__(
        self, c: cfg.CSAConfig, width: int, ports: Sequence[ir.Port], rngs: nnx.Rngs
    ):
        h, d, r = c.num_heads, c.head_dim, c.compress_ratio
        if r is None and (c.indexer is not None or ports):
            raise ValueError(
                "csa: an indexer, and the kv, index and candidates ports, are for "
                "compressed latents; set compress_ratio"
            )
        rank, groups = c.o_lora_rank, c.o_groups
        if rank and h % groups:
            raise ValueError(f"csa: {h} heads do not split into {groups} o_groups")
        indexer = _own_indexer(c, ports)
        own_kv = r is not None and not _reads(ports, "kv")
        self.head_dim, self.compress_ratio = d, r

        def linear(n_in: int, n_out: int) -> nnx.Linear:
            return nnx.Linear(n_in, n_out, use_bias=False, rngs=rngs)

        # The randomly initialized layers; reordering them changes a seed's init.
        latent = c.q_lora_rank
        self.q_a = linear(width, latent) if latent else None
        self.q = linear(latent or width, h * d)
        self.kv = linear(width, d)
        self.compress_kv = linear(width, d) if own_kv else None
        pooled = own_kv and r is not None and r > 1
        self.compress_gate = linear(width, d) if pooled else None
        init = nnx.initializers.lecun_normal(batch_axis=(0,))
        self.o_a = (
            nnx.Param(init(rngs.params(), (groups, h * d // groups, rank)))
            if rank
            else None
        )
        self.o = linear(groups * rank if rank else h * d, width)
        scale = (c.scale or d**-0.5) * logit_gain(c.rope)
        super().__init__(c, width, h, 1, d, scale, False, ports, rngs)
        rotated = int(d * c.rope.fraction) // 2 * 2 if c.rope else 0
        self.indexer = (
            Indexer(
                indexer,
                width,
                latent or width,
                d if own_kv else None,
                c.latent_norm,
                c.rope,
                rotated,
                rngs,
            )
            if indexer
            else None
        )

        # Queries, latents and compression
        self.q_a_norm = Norm(c.latent_norm, latent) if latent else None
        self.q_norm = Norm(c.qk_norm, d) if c.qk_norm else None
        self.kv_norm = Norm(c.latent_norm, d)
        self.compress_norm = Norm(c.latent_norm, d) if own_kv else None
        self.rope = c.rope

    def __call__(self, x: Float[Array, "b t d"], taps: Taps) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        docs = segments(taps)
        rows = _rows(docs, b, t)
        query = x
        if self.q_a is not None and self.q_a_norm is not None:
            query = self.q_a_norm(self.q_a(x))
        q = self.q(query).reshape(b, t, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = self._rotate(q, rows.positions)
        kv = self._rotate(self.kv_norm(self.kv(x))[:, :, None], rows.positions)
        if self.compress_ratio is None:
            out = self._attend(x, q, kv, kv, docs)
        else:
            if _reads(self.ports, "kv"):
                latents = _read(taps, self.ports, "kv")
            else:
                latents = self._compress(x, rows)
            _write(taps, self.ports, "kv", latents)
            keep, scores = self._compressed(x, query, rows, docs, latents, taps)
            kv = jnp.concatenate([kv, latents.kv[:, :, None]], axis=1)
            keep = jnp.concatenate([jnp.ones((b, t, t), bool), keep], -1)
            out = self._attend(x, q, kv, kv, docs, keep, scores, taps)
        return self._output(x, self._unrotate(out, rows.positions))

    def _compress(self, x: Float[Array, "b t d"], rows: Segments) -> Latents:
        """This layer's compressed latents, with its indexer's keys if it runs one."""
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
        latent = self.compress_norm(kv)
        first = rows.positions[:, jnp.maximum(jnp.arange(t) - (r - 1), 0)]
        order = jnp.where((rows.within + 1) % r == 0, rows.within // r, -1)
        index = None
        if self.indexer is not None:
            index = self.indexer.keys(latent, first)
        return Latents(self._rotate(latent[:, :, None], first)[:, :, 0], index, order)

    def _compressed(
        self,
        x: Float[Array, "b t d"],
        query: Float[Array, "b t e"],
        rows: Segments,
        docs: Segments | None,
        latents: Latents,
        taps: Taps,
    ) -> tuple[Bool[Array, "b t s"], Scores | None]:
        """Which compressed latents each query attends to: complete ones of its document,
        from its group's last token on, of those the indexer keeps; and the indexer's
        scores of them, if it runs one."""
        b, t, _ = x.shape
        allowed = jnp.broadcast_to(_causal(t, None, docs), (b, t, t))
        allowed = allowed & (latents.order >= 0)[:, None, :]
        keep, scores = self._select(
            x, query, rows.positions, latents.index, allowed, latents.order, taps
        )
        return allowed if keep is None else allowed & keep, scores

    def _visible(self, docs: Segments | None, t: int) -> Bool[Array, "*rows t s"]:
        """The window's latents, then the compressed ones, which no window limits."""
        window = _causal(t, self.window, docs)
        if self.compress_ratio is None:
            return window
        return jnp.concatenate([window, _causal(t, None, docs)], -1)

    def _offsets(self, t: int) -> Int[Array, "t s"]:
        """How far each latent's token is behind each query; a compressed latent's token
        is the last of its group."""
        offset = super()._offsets(t)
        if self.compress_ratio is None:
            return offset
        return jnp.concatenate([offset, offset], -1)

    def _rotate(
        self, x: Float[Array, "b t h k"], pos: Int[Array, "b t"]
    ) -> Float[Array, "b t h k"]:
        """Queries or latents with their first rope.fraction dims turned by the positions."""
        return x if self.rope is None else rotate(x, self.rope, pos)

    def _unrotate(
        self, out: Float[Array, "b t h k"], pos: Int[Array, "b t"]
    ) -> Float[Array, "b t h k"]:
        """The head outputs turned back by their query's position: the inverse of
        `_rotate`, so that what a latent adds depends on how far behind it is."""
        if self.rope is None:
            return out
        dims = int(out.shape[-1] * self.rope.fraction) // 2 * 2
        back = rotate(out, self.rope, -pos)  # by -pos, but times the gain again
        if dims == 0:
            return back
        gain = frequencies(self.rope, dims)[1]
        return jnp.concatenate([back[..., :dims] / gain**2, back[..., dims:]], -1)

    def _project(self, out: Float[Array, "b t h v"]) -> Float[Array, "b t d"]:
        """The heads projected back to the width, through o_lora_rank for each group of
        heads when the projection is grouped."""
        b, t, _, _ = out.shape
        if self.o_a is None:
            return self.o(out.reshape(b, t, -1))
        groups = self.o_a.shape[0]
        low = jnp.einsum("btgi,gir->btgr", out.reshape(b, t, groups, -1), self.o_a[...])
        return self.o(low.reshape(b, t, -1))


@register("csa")
def _(c: cfg.CSAConfig, width: int, ports: tuple[ir.Port, ...], ctx: Context) -> CSA:
    return CSA(c, width, ports, ctx.rngs)
