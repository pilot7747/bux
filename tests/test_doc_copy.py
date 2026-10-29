"""The document-local copy feature (modded-nanogpt PR 378), ``doc_copy``: its
matcher finds what a brute-force search of each document finds, it starts as
nothing, and it shares its vector between two sites through its port."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bux.architecture import Embedding, Model
from bux.architecture.common.segments import Segments
from bux.architecture.layers.doc_copy import DocCopy, candidates
from bux.lang import compile

LENGTHS = (1, 2, 3, 4, 6)
PRELUDE = """
d_model = 32
vocab_size = 16
norm = rmsnorm
attn = gqa(num_heads=4, num_kv_heads=2, rope=rope())
ffn = gated_mlp(intermediate_size=48, activation=swish)
"""


def brute(tokens, doc, within, lengths, counts):
    """Each position's candidate and bucket by searching its document's earlier
    windows of the longest length that matches."""
    b, t = tokens.shape
    candidate, bucket = np.zeros((b, t), int), np.zeros((b, t), int)
    for r in range(b):
        for p in range(t):
            for i in reversed(range(len(lengths))):
                n = lengths[i]
                if within[r, p] < n - 1:
                    continue
                context = tokens[r, p - n + 1 : p + 1]
                hits = [
                    j
                    for j in range(p)
                    if within[r, j] >= n - 1
                    and doc[r, j] == doc[r, p]
                    and (tokens[r, j - n + 1 : j + 1] == context).all()
                ]
                if hits:
                    candidate[r, p] = tokens[r, hits[-1] + 1]
                    bucket[r, p] = i * counts + min(len(hits), counts)
                    break
    return candidate, bucket


@pytest.mark.parametrize("vocab", [3, 6])
def test_the_matcher_finds_the_latest_earlier_occurrence_in_the_document(vocab):
    tokens = jax.random.randint(jax.random.key(vocab), (2, 40), 0, vocab)
    positions = jnp.stack([jnp.arange(40), jnp.r_[jnp.arange(17), jnp.arange(23)]])
    docs = Segments.of(positions)
    got = candidates(tokens, docs.ids, docs.within, LENGTHS, 3)
    want = brute(*map(np.asarray, (tokens, docs.ids, docs.within)), LENGTHS, 3)
    np.testing.assert_array_equal(np.asarray(got[0]), want[0])
    np.testing.assert_array_equal(np.asarray(got[1]), want[1])
    assert (np.asarray(got[1])[1, 17:] > 0).any()  # the second document matches too


def test_the_matcher_reads_no_later_token():
    tokens = jax.random.randint(jax.random.key(0), (1, 30), 0, 4)
    docs = Segments.of(jnp.arange(30)[None])
    changed = tokens.at[:, 20:].set((tokens[:, 20:] + 1) % 4)
    a = candidates(tokens, docs.ids, docs.within, LENGTHS, 3)
    b = candidates(changed, docs.ids, docs.within, LENGTHS, 3)
    for x, y in zip(a, b, strict=True):
        assert jnp.array_equal(x[:, :20], y[:, :20])


def model(body: str, seed: int = 0) -> Model:
    return Model(compile(PRELUDE + body), rngs=nnx.Rngs(seed))


TWO_SITES = (
    "model = embed doc_copy(copy=@c) [norm attn] [norm ffn] [norm attn] [norm ffn]"
    " doc_copy(gain=1, copy=$c!) norm head"
)


@pytest.mark.parametrize("residual", ["", "residual = hc(expansion=2)\n"])
def test_the_feature_starts_as_nothing_and_learns(residual):
    plain = model(residual + "model = embed ([norm attn] [norm ffn])^2 norm head")
    copying = model(residual + TWO_SITES)
    tokens = jax.random.randint(jax.random.key(1), (2, 24), 0, 4)
    assert jnp.array_equal(plain(tokens), copying(tokens))
    first, second = (x for _, x in nnx.iter_graph(copying) if isinstance(x, DocCopy))
    assert second.reads and not hasattr(second, "scale")
    embed = next(x for _, x in nnx.iter_graph(copying) if isinstance(x, Embedding))
    assert first.embedding is embed
    grads = nnx.grad(lambda m: m(tokens).mean())(copying)
    tables = [g for g in jax.tree.leaves(nnx.state(grads)) if g.shape == (31, 32)]
    assert len(tables) == 1 and bool(jnp.abs(tables[0]).sum() > 0)  # e, from 0


def test_the_vector_is_the_candidates_rows_scaled_plus_the_buckets():
    m = model(TWO_SITES)
    first = next(x for _, x in nnx.iter_graph(m) if isinstance(x, DocCopy))
    first.scale[...] = jax.random.normal(jax.random.key(2), first.scale.shape)
    first.bucket_embed[...] = jax.random.normal(
        jax.random.key(3), first.bucket_embed.shape
    )
    tokens = jax.random.randint(jax.random.key(4), (1, 20), 0, 3)
    from bux.architecture.registry import Taps

    v = first.vector(Taps(tokens=tokens))
    docs = Segments.of(jnp.arange(20)[None])
    c, b = candidates(tokens, docs.ids, docs.within, first.lengths, first.counts)
    table = first.embedding.tokens.embedding[...]
    live = (b > 0)[..., None]
    want = live * (
        table[c] * first.scale[...][b][..., None] + first.bucket_embed[...][b]
    )
    assert jnp.allclose(v, want, atol=1e-6)


def test_the_exports_matcher_is_buxs():
    """The export's matcher, of 64-bit hashes over a row's real tokens, after a
    left-padded batch's padding, finds what bux's does."""
    import torch

    from bux.export.hf import doc_copy as hf

    tokens = jax.random.randint(jax.random.key(5), (2, 36), 0, 3)
    want = candidates(
        tokens,
        jnp.ones_like(tokens),
        jnp.broadcast_to(jnp.arange(36), (2, 36)),
        LENGTHS,
        3,
    )
    pad = torch.zeros(2, 5, dtype=torch.long)
    ids = torch.cat([pad, torch.from_numpy(np.asarray(tokens)).long()], 1)
    real = torch.cat([pad.bool(), torch.ones(2, 36, dtype=torch.bool)], 1)
    got = hf.candidates(ids, real.long().cumsum(1) - 1, real, list(LENGTHS), 3)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g[:, 5:].numpy(), np.asarray(w))
