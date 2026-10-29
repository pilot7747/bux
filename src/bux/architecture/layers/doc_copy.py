"""The document-local copy feature (modded-nanogpt PR 378): each position's candidate
next token, from the latest earlier occurrence of its context in its document."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, UInt

from bux import config as cfg
from bux.architecture.common.ports import read, reads, write
from bux.architecture.layers.embedding import Embedding, lookup
from bux.architecture.registry import Context, Stream, Taps, register, segments
from bux.lang import ir

# Two polynomial hashes of a window, mod 2^32 each, with constants to tell lengths
# and documents apart: a context matches where both agree.
_BASES = (0x9E3779B1, 0x85EBCA77)
_LENGTH_MIX = (0x2545F491, 0x4F6CDD1D)
_DOC_MIX = (0x5851F42D, 0x4C957F2D)


def _powers(base: int, t: int) -> tuple[UInt[Array, " t"], UInt[Array, " t"]]:
    """Return ``base^i`` and ``base^-i`` mod 2^32 for i < t: base is odd."""
    inverse = pow(base, -1, 2**32)
    up = jnp.cumprod(jnp.full(t, base, jnp.uint32).at[0].set(1), dtype=jnp.uint32)
    down = jnp.cumprod(jnp.full(t, inverse, jnp.uint32).at[0].set(1), dtype=jnp.uint32)
    return up, down


def _window_hash(
    y: UInt[Array, "b t"], lengths: UInt[Array, " n"], base: int
) -> UInt[Array, "b n t"]:
    """Return ``sum_i y[i] base^(t - i)`` over each window of each length ending at
    each position, mod 2^32: ``base^t (S[t] - S[t - L])`` of the prefix sums
    ``S[t] = sum_{i <= t} y[i] base^-i``."""
    t = y.shape[1]
    up, down = _powers(base, t)
    prefix = jnp.cumsum(y * down, axis=1, dtype=jnp.uint32)
    padded = jnp.pad(prefix, ((0, 0), (1, 0)))  # padded[:, i + 1] = prefix[:, i]
    start = jnp.maximum(jnp.arange(t) - lengths[:, None] + 1, 0)  # n t
    window = padded[:, None, 1:] - padded[:, start]
    return up * window


def _keys(
    tokens: Int[Array, "b t"],
    doc: Int[Array, "b t"],
    within: Int[Array, "b t"],
    lengths: tuple[int, ...],
) -> tuple[list[UInt[Array, "b nt"]], Bool[Array, "b n t"]]:
    """Return the sort keys of every (length, position)'s window, flattened length by
    length, and which windows lie inside their position's document: a window that
    does not gets a key no other window has."""
    n = jnp.asarray(lengths, jnp.uint32)
    y = tokens.astype(jnp.uint32) + 1
    valid = within[:, None] >= n[:, None] - 1  # b n t
    keys = []
    for base, length_mix, doc_mix in zip(_BASES, _LENGTH_MIX, _DOC_MIX, strict=True):
        h = _window_hash(y, n, base) ^ (n[:, None] * length_mix)
        keys.append(h ^ (doc.astype(jnp.uint32)[:, None] * doc_mix))
    b, t = tokens.shape
    flat = jnp.arange(len(lengths) * t, dtype=jnp.uint32).reshape(len(lengths), t)
    alone = jnp.broadcast_to(flat, (b, *flat.shape))
    keys = [jnp.where(valid, k, alone) for k in keys]
    keys.insert(0, (~valid).astype(jnp.uint32))  # invalid ones sort apart
    return [k.reshape(b, -1) for k in keys], valid


def _rows(x: Int[Array, "b m"], at: Int[Array, "b m"]) -> Int[Array, "b m"]:
    """Return each row's x at its indices `at`."""
    return jnp.take_along_axis(x, at, axis=1)


def candidates(
    tokens: Int[Array, "b t"],
    doc: Int[Array, "b t"],
    within: Int[Array, "b t"],
    lengths: tuple[int, ...],
    counts: int,
) -> tuple[Int[Array, "b t"], Int[Array, "b t"]]:
    """Return each position's candidate next token (0 for none) and its bucket (0
    for none), as PR 378's matcher finds them.

    One stable sort of every window's keys puts each window right after the latest
    earlier window of the same tokens, length and document. The longest length with
    one picks it: the candidate is the token after that window. The bucket is
    ``1 + counts l + min(occurrences, counts) - 1`` for the l-th length, of how many
    earlier windows match.

    Args:
        tokens: The input tokens, b rows of t.
        doc: Which document of its row each token is in.
        within: How many tokens of its document come before each token.
        lengths: The context lengths, shortest first.
        counts: How many occurrences the buckets tell apart.
    """
    b, t = tokens.shape
    keys, valid = _keys(tokens, doc, within, lengths)
    index = jnp.broadcast_to(jnp.arange(len(lengths) * t), (b, len(lengths) * t))
    *sorted_keys, order = jax.lax.sort((*keys, index), num_keys=len(keys) + 1)
    same = jnp.stack([k[:, 1:] == k[:, :-1] for k in sorted_keys]).all(0)
    pos = order % t
    before = jnp.where(same, pos[:, :-1], -1)
    rows = jnp.arange(b)[:, None]
    prev = jnp.full((b, len(lengths) * t), -1).at[rows, order[:, 1:]].set(before)
    prev = prev.reshape(b, len(lengths), t)
    # How many matching windows come before each: its place in its run of equals.
    new_run = jnp.concatenate([jnp.ones((b, 1), bool), ~same], 1)
    first = jax.lax.cummax(jnp.where(new_run, index, 0), axis=1)
    earlier = jnp.zeros_like(index).at[rows, order].set(index - first)
    earlier = earlier.reshape(b, len(lengths), t)
    found = (prev >= 0) & (prev < jnp.arange(t)) & valid
    rank = jnp.where(found, jnp.arange(1, len(lengths) + 1)[:, None], 0).max(1)
    pick = jnp.maximum(rank - 1, 0)[:, None]
    best = jnp.take_along_axis(prev, pick, axis=1)[:, 0]
    seen = jnp.take_along_axis(earlier, pick, axis=1)[:, 0]
    candidate = jnp.where(rank > 0, _rows(tokens, jnp.minimum(best + 1, t - 1)), 0)
    count = jnp.clip(seen, 1, counts) - 1
    bucket = jnp.where(rank > 0, (rank - 1) * counts + count + 1, 0)
    return candidate, bucket


class DocCopy(nnx.Module):
    """The document-local copy feature (modded-nanogpt PR 378), ``doc_copy``: adds
    ``gain (E[c] s[b] + e[b])`` to the stream, of each position's candidate c and
    bucket b (`candidates`), E the model's first embedding's table, and s and e a
    scale and an embedding of each bucket, from 0. Bucket 0, no match, adds nothing.

    The matcher hashes the tokens themselves, where PR 378 hashes PR 375's
    normalized classes of the GPT-2 tokenizer's tokens, so that a context matches
    across case and spacing; bux's tokenizers have no such classes.

    Ports, where the source lists them: ``copy``, written, pushes the vector; read,
    the layer adds the tap's vector, by a gain of its own, and has no tables.

    Args:
        config: Its settings (`bux.config.DocCopyConfig`).
        d: The model's width.
        ports: The ports the source lists for it.
        embedding: The model's first embedding, whose rows the candidates take.
    """

    no_weight_decay = ("gain", "scale", "bucket_embed")
    no_muon = True

    def __init__(
        self,
        config: cfg.DocCopyConfig,
        d: int,
        ports: tuple[ir.Port, ...],
        embedding: Embedding | None,
    ):
        self.ports, self.lengths, self.counts = ports, config.lengths, config.counts
        self.gain = nnx.Param(jnp.asarray(config.gain, jnp.float32))
        self.reads = reads(ports, "copy")
        if self.reads:
            return
        if embedding is None:
            raise ValueError("doc_copy takes its rows from an embedding before it")
        self.embedding = embedding
        buckets = 1 + len(config.lengths) * config.counts
        self.scale = nnx.Param(jnp.zeros(buckets))
        self.bucket_embed = nnx.Param(jnp.zeros((buckets, d)))

    def vector(self, taps: Taps) -> Float[Array, "b t d"]:
        """Return the copy vector of the pass's tokens: ``E[c] s[b] + e[b]``."""
        tokens = taps.tokens
        assert tokens is not None
        docs = segments(taps)
        b, t = tokens.shape
        doc = jnp.ones((b, t), jnp.int32) if docs is None else docs.ids
        within = (
            jnp.broadcast_to(jnp.arange(t), (b, t)) if docs is None else docs.within
        )
        found = candidates(tokens, doc, within, self.lengths, self.counts)
        candidate, bucket = jax.lax.stop_gradient(found)
        live = jnp.arange(self.scale.shape[0]) > 0  # bucket 0: no match, nothing
        scale = jnp.where(live, self.scale[...], 0)[bucket][..., None]
        embed = jnp.where(live[:, None], self.bucket_embed[...], 0)[bucket]
        rows = lookup(self.embedding.tokens.embedding[...], candidate, "vocab")
        return rows * scale.astype(rows.dtype) + embed.astype(rows.dtype)

    def __call__(self, x: Stream, taps: Taps) -> Stream:
        """Return the stream, ``b t d`` or its copies ``b t n d``, with the copy
        vector added by the gain, to every copy."""
        v = read(taps, self.ports, "copy") if self.reads else self.vector(taps)
        write(taps, self.ports, "copy", v)
        v = v if x.ndim == 3 else v[:, :, None]
        return x + (self.gain[...] * v).astype(x.dtype)


@register("doc_copy")
def _(
    config: cfg.DocCopyConfig,
    width: ir.Width,
    ports: tuple[ir.Port, ...],
    ctx: Context,
) -> DocCopy:
    return DocCopy(config, ctx.arch.d_model, ports, ctx.built.embedding)
