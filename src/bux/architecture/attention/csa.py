"""Compressed sparse attention (DeepSeek-V4): `CSA`, the layer ``csa``, and the
compressed `Latents` its ``kv`` port carries."""

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from bux import config as cfg
from bux import initializers, kernels, parallel
from bux.architecture.attention import sparse, sparse_cudnn
from bux.architecture.attention.gqa import (
    Names,
    Sparse,
    documents_or_rows,
    indexer_config,
)
from bux.architecture.attention.indexer import Indexer, Scored, Target
from bux.architecture.attention.rope import frequencies, rotate, rotated_dims
from bux.architecture.common.ports import read, reads, write
from bux.architecture.common.segments import Segments
from bux.architecture.common.sharding import linear, shard_heads, shard_layer_input
from bux.architecture.layers.norm import Norm
from bux.architecture.registry import Context, Taps, register, report, segments
from bux.architecture.repeats import remat
from bux.lang import ir
from bux.softmax import softmax


class Latents(NamedTuple):
    """Compressed latents, one slot per token: the latent of the group of
    ``compress_ratio`` tokens that ends at a token, where one does. The ``kv`` port
    carries them."""

    #: The latents, rotated, each the key and the value at once.
    kv: Float[Array, "b s dk"]
    #: The indexer's keys, if the layer has an indexer.
    index: Float[Array, "b s i"] | None
    #: Which latent of its document a slot holds; -1 for none.
    order: Int[Array, "b s"]


class CSA(Sparse):
    """Compressed sparse attention (CSA), the layer ``csa``: DeepSeek-V4's, as
    DeepSeek-V4.1-Flash's inference code computes it.

    A latent is a ``head_dim`` wide RMS-normalized projection of a token, the key and
    the value of a single KV head that every query head shares. Each query attends to
    the latents of a sliding window of tokens, and to compressed latents: every
    ``compress_ratio`` tokens of a document pooled into one, per channel under a
    softmax gate learned from the tokens, and RMS-normalized. A compressed latent is
    rotated to the position of its group's first token, and a query may attend to it
    from its group's last token on. An indexer picks the ``top_k`` of them for each
    query; without one a query attends to them all. With ``compress_ratio=None`` the
    layer attends to the window alone.

    Queries come from a ``q_lora_rank`` wide RMS-normalized latent, or from the input
    at full rank. The first ``rope.fraction`` of each query's and latent's dims are
    rotated, and the head outputs are rotated back by their query's position, since
    the values were rotated with the keys. ``o_lora_rank`` makes the output projection
    grouped and low-rank: ``o_groups`` groups of heads each projected to
    ``o_lora_rank``, then mixed to the width.

    Ports, where the source lists them:

    - ``kv``: written, it pushes the `Latents`, with the indexer's keys; read, the
      layer uses them instead of compressing its own, and an indexer that it runs
      scores those keys.
    - ``index`` and ``candidates``: as for `~bux.architecture.attention.gqa.Sparse`,
      over the compressed latents.

    The window and the compressed latents are two parts
    (`~bux.architecture.attention.sparse.Part`) of one softmax. On a B200 or B300,
    cuDNN's DSA kernels run them, and the window-only layers, where they take them
    (`~bux.architecture.attention.sparse_cudnn.takes`): 512 dims, in 16 bits. They
    score the indexer's target at its picks too.

    The query heads split over ``heads``, and the groups of the output projection with
    them; the latents, a single head, are every device's.

    Example:
        A layer that compresses every 2 tokens and runs the indexer, and one that
        reuses both by its ports, as DeepSeek-V4.1-Flash's do
        (``examples/architectures/deepseek/deepseek_v4.1_flash.bux``):

        .. code-block:: bux

            window = csa(num_heads=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8,
                         sliding_window_size=128, sinks=true,
                         rope=rope(fraction=0.125, layout=interleaved))
            pairs  = window(compress_ratio=2, indexer=indexer(num_heads=32, top_k=512))
            P = [norm pairs(kv=@kv, index=@topk)] [norm experts]
            p = [norm pairs(kv=$kv, index=$topk)] [norm experts]

    Args:
        config: the layer's settings.
        width: the width d of the stream it reads and writes.
        ports: the ports the source lists for it.
        rngs: draws the initial parameters.

    Raises:
        ValueError: an indexer or ports without ``compress_ratio``; ``num_heads``
            that do not split into ``o_groups`` under ``o_lora_rank``; or ports that
            need an indexer the layer does not run
            (`~bux.architecture.attention.gqa.indexer_config`).
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
        init = initializers.weight(batch_axis=(0,))
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
        """Attend each position to its window of latents and the compressed ones it
        keeps.

        Args:
            x: the stream: b rows of t positions, d wide.
            taps: the forward pass's taps: the documents packed into the rows, which
                no query looks across, and the stacks the layer's ports use; a
                training forward collects the indexer's loss in them.

        Returns:
            The layer's output, the shape of `x`.
        """
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
            out = self._attend_window(x, q, kv, rows, docs)
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

    def _attend_window(
        self,
        x: Float[Array, "b t d"],
        q: Float[Array, "b t h dk"],
        kv: Float[Array, "b t 1 dk"],
        rows: Segments,
        docs: Segments | None,
    ) -> Float[Array, "b t h dk"]:
        """Return each query's attention to its window of latents alone: by
        `sparse.attend` (cuDNN's DSA kernels where they take it, else XLA's chunks)
        where `_wide` says the run's dense kernels would not take it, else by the
        run's attention method."""
        window = (sparse.Part(kv, kv, sparse.token_span(rows, self.window)),)
        if self._feature() is not None or not self._wide(q):
            return self._attend(x, q, kv, kv, docs)
        return remat.named("attention", self._sparse, q, window, self._sinks())[0]

    def _wide(self, q: Float[Array, "b t h dk"]) -> bool:
        """Return whether the heads are wider than the dense kernels of
        ``methods.attention = auto`` take on a GPU (256 dims, as CSA's 512 are)."""
        on = parallel.active()
        auto = (on.methods.attention if on else "auto") == "auto"
        return auto and kernels.gpu() and q.shape[-1] > sparse.HEAD_DIMS

    def _sparse(
        self,
        q: Float[Array, "b t h dk"],
        parts: tuple[sparse.Part, ...],
        sinks: Float[Array, " h"] | None,
    ) -> tuple[Float[Array, "b t h dk"], Float[Array, "b t h"]]:
        """Return `sparse.attend`'s output and logsumexp: by cuDNN's DSA kernels
        where they take the parts."""
        if not self._kernels_take(q, parts):
            return super()._sparse(q, parts, sinks)
        windows = self._windows(parts)
        return sparse_cudnn.attend(q, parts, sinks, self.scale, windows)

    def _kernels_take(
        self, q: Float[Array, "b t h dk"], parts: Sequence[sparse.Part]
    ) -> bool:
        """Return whether cuDNN's DSA kernels take the parts (`sparse_cudnn.takes`)."""
        windows = self._windows(parts)
        return sparse_cudnn.takes(q, parts, windows, self.softcap)

    def _windows(self, parts: Sequence[sparse.Part]) -> tuple[int | None, ...]:
        """Return each part's window: the layer's for the first, the latents of the
        tokens; none for the compressed ones, kept by index."""
        return (self.window, None)[: len(parts)]

    def _target(self, part: sparse.Part, xs: tuple) -> Float[Array, "b c k"]:
        """Return a chunk of queries' attention to the latents of `part`: to those
        they keep, by cuDNN's kernel, where it scores them (`_scores_picks`); else to
        every one."""
        if not self._scores_picks(xs[0]):
            return super()._target(part, xs)
        q, lse, span, index = jax.lax.stop_gradient(xs)
        kept = sparse.at(part, span, index)
        return sparse_cudnn.probabilities(q, kept, lse, self.scale)

    def _train_indexer(
        self, scored: Scored, target: Target, xs: Any, width: int, taps: Taps
    ) -> None:
        """Report the indexer's loss against the attention `target` gives: to the
        latents each query keeps where cuDNN's kernel scores them. `FullLogits`'
        features give it all t by t probabilities."""
        if self._feature() is not None or not self._scores_picks(xs[0]):
            return super()._train_indexer(scored, target, xs, width, taps)
        assert self.indexer is not None
        picks = scored.kept.shape[-1]
        loss = self.indexer.loss(scored, target, xs, picks, picks=True)
        report(taps, "indexer_loss", self.indexer.loss_weight, loss)

    def _scores_picks(self, q: Float[Array, "b t h dk"]) -> bool:
        """Return whether cuDNN's kernel scores the indexer's target at its picks
        (`sparse_cudnn.probabilities_take`): latents it keeps one by one, not in
        blocks."""
        by_one = self.indexer is not None and self.indexer.block_size == 1
        return by_one and sparse_cudnn.probabilities_take(q, self.softcap)

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
            kv = (kv * softmax(gate, axis=2)).sum(2)
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
